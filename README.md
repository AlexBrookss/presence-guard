# presence-guard：在座检测 + 工牌提醒

一个运行在 UOS/DDE 桌面上的小工具：

- **每 1 分钟**检查一次键鼠空闲状态；只有空闲达到 10 秒才短暂唤醒摄像头做“是否在座”检测；
- 键鼠活跃时**完全不碰摄像头**；连续确认无人 **10 秒**后，调用 DDE 的锁屏接口自动锁屏；
- **工作日 09:30–10:00、13:00–13:30** 每 **3 分钟**检查一次是否佩戴**深蓝色工牌**
  （通过 chinesecalendar 识别法定节假日与调休，不是死板的周一至周五）；
- 每个工牌检查时间窗最多提醒 **3 次**，未识别到就发桌面通知（3 条独立保留在通知中心）；
- 检测间隔内完全关闭摄像头、进程休眠，不做持续录像。

> 关键取舍：为了低占用，摄像头不是常开的；键鼠活跃时也不打开。离开后多久锁屏取决于
> 1 分钟的采样周期，最坏情况约 **70 秒**（下一次采样最多等 60 秒 + 10 秒确认）。

---

## 1. 运行策略

```text
每 60 秒：
    查询 xprintidle（约 1 ms，不打开摄像头）
    ├─ 键鼠活跃（idle < 10 s）：视为在座，跳过摄像头，60 秒后再查
    └─ 空闲 >= 10 s：打开摄像头 320x240 -> 丢 2 帧预热 -> 抓 1 帧 -> 关摄像头
         首次快速扫描只用 YuNet 人脸（约 20 ms/帧）
         ├─ 看到脸：在座，重置“无人计时”，60 秒后再查
         └─ 没看到脸：先记为“疑似无人”，10 秒后确认
                   ├─ 确认阶段才启用 NanoDet 人体检测（约 0.8 s CPU）
                   ├─ 期间键鼠有动作：取消
                   └─ 仍然无人 -> com.deepin.SessionManager.RequestLock

工作日 09:30-10:00、13:00-13:30（节假日自动跳过，每 180 秒一个检查点）：
    即使键鼠活跃也打开摄像头（否则永远查不到工牌）
    640x480 -> 抓 5 帧 -> 关摄像头
    YuNet 找脸 -> 以脸为尺度取“脖子到胸口”ROI（不需要 NanoDet）
    HSV 深蓝色掩码 -> 多帧投票
    ├─ 至少 2 帧看到蓝色：视为佩戴，不打扰
    ├─ 0 帧看到蓝色且至少 3 帧有效：发通知
    │    ├─ 同一窗口最多 3 次
    │    └─ 两次提醒至少间隔 600 秒
    └─ 画面太暗/脸太小/看不到脸：unknown，不发通知（宽松）

锁屏期间：
    完全不打开摄像头，只每 60 秒查询一次 Locked 属性
```

### 为什么这样省资源

| 项目 | 估算 |
|---|---|
| 键鼠活跃时 | 每分钟只查一次空闲时间，摄像头 **0 次** |
| 空闲时首次扫描 | 320x240，YuNet 人脸，实测 **约 170 ms CPU** |
| 疑似无人时的确认 | 才启用 NanoDet，实测 **约 830 ms CPU**，确认一次后即锁屏 |
| 工牌检查 | YuNet + HSV，640x480，5 帧，约 200–400 ms CPU；每 3 分钟一次 |
| 空闲但非检查时 | 进程阻塞在定时器上，CPU 接近 0 |
| 锁屏时 | 摄像头 **0 次**，只每 60 秒查一次 `Locked` |
| CPU 策略 | OpenCV 单线程、进程 `nice=19`、IO idle，避免抢占前台 |
| 常驻内存 | Python + OpenCV + 两个小模型，约 150–250 MB |

NanoDet 模型只有 3.7 MB，YuNet 只有 228 KB，CPU 推理即可，不需要 GPU。

---

## 2. 使用的现成框架与 D-Bus 接口

### 视觉

- **OpenCV DNN**：模型推理、图像处理、摄像头采集。
- **NanoDet-m-plus-1.5x-416**（Apache-2.0，COCO 预训练）：检测 `person` 类。
- **YuNet**（MIT）：轻量人脸检测，用来：
  - 坐得很近、只露出头肩时，作为“人在”的兜底；
  - 给工牌 ROI 提供稳定的人脸尺度（下巴以下到胸口）。

工牌本身没有用目标检测模型（没有公开的“深蓝色工牌”数据集，且工牌样式会变），
而是用**颜色 + 解剖位置 + 多帧投票**的宽松启发式：

- 以人脸框为基准，取 `下巴 - 0.1×脸高` 到 `下巴 + 2.6×脸高`、宽度 `±1.15×脸宽` 的 ROI；
- 该 ROI 覆盖挂绳和胸前的工牌；
- HSV 深蓝色范围默认 `H 95–140, S≥40, V 12–220`；
- 只统计**靠近下巴中线**的蓝色块：偏离中心超过 0.5 倍脸宽的直接忽略
  （桌面风扇、杯子、椅子等边缘物体）；
- 只认**深藏青**：蓝色块 V 中位数超过 150 时忽略（亮蓝色塑料/天空反光）；
- **条状判定（核心）**：中线上的深蓝像素必须形成“条状物”才算挂绳——
  同一连通块的最小外接矩形长 ≥ 0.45 倍脸高、宽 ≤ 0.28 倍脸宽才算；
  衣服上的蓝色小点（太短）和整片蓝色衣服（太宽）都不会触发；
- 整块 ROI 蓝色占比 > 55% 时认为是蓝色衣服，也按“已佩戴”处理，避免误报；
- 至少 3 帧有效、2 帧看到条状才判“佩戴”；只有 0 帧看到才判“未佩戴”。

这套规则偏“宽松”：宁可漏提醒，也尽量不误报。

### 两种判定模式

| `badge.mode` | 判定为“已佩戴”的条件 | 适用场景 |
|---|---|---|
| `strap`（默认） | 检测到**完整条状**深蓝挂绳（长条，非点状/碎片） | 宽松：挂绳完整可见就不提醒 |
| `card` | 必须看到**卡片形状**的深蓝色块 | 严格：挂绳还在但卡片摘了，也会提醒 |

实测三种状态（真实摄像头画面，已加入回归测试）：

| 画面状态 | `strap` | `card` |
|---|---|---|
| 挂绳 + 卡片 | present | present |
| 只有挂绳（卡片已摘，且仅剩碎片/横条） | **absent** | **absent** |
| 完全摘下（旁边有蓝色风扇） | absent | absent |

需要“摘掉卡片也算没戴”时：

```toml
[badge]
mode = "card"
```

### D-Bus

| 用途 | 服务名 / 对象路径 | 接口 / 方法 |
|---|---|---|
| 查询是否已锁屏 | `com.deepin.SessionManager` `/com/deepin/SessionManager` | `org.freedesktop.DBus.Properties.Get` → `Locked` |
| 请求锁屏 | 同上 | `com.deepin.SessionManager.RequestLock()` |
| 锁屏兜底（旧版 DDE） | `com.deepin.dde.lockFront` `/com/deepin/dde/lockFront` | `com.deepin.dde.lockFront.Show()` |
| 桌面通知 | `org.freedesktop.Notifications` `/org/freedesktop/Notifications` | `Notify(...)` |

本机已确认：

```text
com.deepin.SessionManager 提供 RequestLock / Locked
com.deepin.dde.lockFront  提供 Show
org.freedesktop.Notifications 由 DDE 通知服务提供
```

> 工具不会直接调用 `loginctl`、`xset` 或模拟按键，只走 DDE 自己的 D-Bus 接口，
> 因此锁屏行为和系统自带锁屏一致。

---

## 3. 安装

### 依赖

- UOS 20 / DDE（已在 UOS 20.1070、Python 3.7.3 和 3.10 上验证）
- Python ≥ 3.7
- `python3-venv`、`python3-gi`、`python3-pyqt5`（面板用，deb 已声明依赖）
- 可选 `xprintidle`（deb 的 Recommends）：键鼠活跃时跳过摄像头，
  是低性能机器最有效的省资源手段；没有它会退化为每 60 秒都打开摄像头。
- `chinesecalendar`（pip）：法定节假日与调休识别（离线 deb 内置 wheel）。
- 摄像头 `/dev/video0`（或其它 V4L2 设备）
- 当前用户在 `video` 组，或设备 ACL 已授予当前用户读写权限：

```bash
sudo usermod -aG video "$USER"
# 重新登录后生效
```

> OpenCV 使用 `opencv-python-headless==4.10.0.84` 的 `cp37-abi3` wheel，
> 在 UOS 20 的 Python 3.7.3 上可以直接安装；虚拟环境必须带
> `--system-site-packages`，否则看不到系统的 `python3-gi`。

### deb 安装（推荐）

仓库自带打包脚本，两种产物：

```bash
bash scripts/build-deb.sh          # 精简版（默认，约 4 MB）：安装时联网装依赖
FULL=1 bash scripts/build-deb.sh   # 离线版（约 85 MB）：wheel 内置，适合内网
sudo dpkg -i dist/presence-guard_0.1.0_amd64.deb
presence-guard --setup            # 生成配置、应用菜单/登录自启动并启用服务
```

精简版安装时需要 pip 网络，默认用清华镜像；可写 `/etc/default/presence-guard`
里的 `PIP_INDEX=...` 换成内网镜像。离线版默认只带 UOS 20 用到的
Python 3.7/3.10 的 numpy wheel（`PY_ABIS="37 38 39 310"` 可全量），
比早期 117 MB 的全家桶小很多。

### 一键安装（源码，开发者用）

```bash
cd presence-guard
bash scripts/install.sh
```

脚本会：

1. 复制程序到 `~/.local/share/presence-guard/app`；
2. 创建虚拟环境并安装 `opencv-python-headless`；
3. 首次安装时写入 `~/.config/presence-guard/config.toml`；
4. 安装并启动 systemd user service `presence-guard.service`；
5. 安装应用菜单项 **在岗守护** 和登录自启动项（需要 `python3-pyqt5`）。

查看日志：

```bash
journalctl --user -u presence-guard -f
```

### 从源码直接运行（调试）

```bash
cd presence-guard
bash scripts/run.sh --once --check-badge --save-debug
```

首次运行会在项目下创建 `.venv` 并安装依赖。

---

## 4. 先做一次“不动手”的验证

### 4.1 检查环境

```bash
~/.local/share/presence-guard/venv/bin/python -m presence_guard --status
```

输出包括：配置文件、摄像头设备、模型是否存在、D-Bus 服务是否在线、当前是否锁屏，
以及当前的 `badge_mode` 和下一次工牌检查时间 `next_badge_check`
（便于确认时间窗/模式配置是否生效）。

### 4.2 自检（不需要摄像头）

```bash
~/.local/share/presence-guard/venv/bin/python -m presence_guard --self-test
```

会验证：

- 两个模型能加载；
- 空白画面推理不崩溃；
- 合成画面中的深蓝工牌能被识别；
- 红色块不会被误判；
- 有效帧不足时返回 `unknown`；
- 主目标选择逻辑正确。

### 4.3 用一张照片验证工牌 ROI

```bash
~/.local/share/presence-guard/venv/bin/python -m presence_guard \
  --image /path/to/photo.jpg --check-badge --save-debug
```

会在 `~/.cache/presence-guard/debug/` 生成带标注的图片：

- 绿框：人体；
- 黄框：人脸；
- 蓝/红框：工牌 ROI；
- **紫红色高亮：深蓝候选像素（仅调试标注，不是判定结果）**。

> 深蓝候选阈值很宽松（H 95–140、S≥40、V 12–220），深色衣服、阴影褶皱、
> 相机色噪都可能被标出来；真正影响判定的是“是否形成长条形挂绳”。

JSON 输出中的 `blue_ratio`、`largest_area`、`mean_v`、`roi` 可用于调参。
如果照片里工牌没有被识别，优先看：

1. `face_width` 是否太小（离镜头太远）；
2. `mean_v` 是否太低（背光/太暗）；
3. `blue_ratio` 是否接近阈值，需要放宽 HSV 或 `min_blue_ratio`。

### 4.4 摄像头单次检测

```bash
~/.local/share/presence-guard/venv/bin/python -m presence_guard \
  --once --check-badge --save-debug
```

不会锁屏、不会发通知，只输出结果并保存标注图。

### 4.5 测试通知通道

```bash
~/.local/share/presence-guard/venv/bin/python -m presence_guard --notify-test
```

如果桌面出现“presence-guard 测试”，说明通知接口正常。

### 4.6 干跑（真实调度，但不锁屏/不通知）

```bash
~/.local/share/presence-guard/venv/bin/python -m presence_guard --dry-run
```

日志里会看到 `DRY-RUN: would lock the screen now` / `DRY-RUN: notification ...`。

---

## 5. 状态面板（可视化 UI）

安装后应用菜单里会出现 **在岗守护**，登录时也会自动启动。也可以手动运行：

```bash
presence-guard --gui          # deb 安装后
~/.local/share/presence-guard/venv/bin/python -m presence_guard --gui   # 源码安装
```

面板显示：

- **当前状态**：在工位 / 在工位（键鼠活跃）/ 已离座 / 锁屏中 / 已暂停 / 摄像头异常
- **工牌状态**：已佩戴 / 未佩戴 / 无法判断 / 未到检查时间 / 等待检查
- 下次在座检测、下次工牌检查、判定模式、摄像头状态
- **摄像头下拉框**：列出系统里的 `/dev/video*` 设备，切换立即生效并写入配置文件
- 最近事件（检测到在座/离座、工牌结果、提醒发送、锁屏等）
- 本轮 CPU 耗时和空闲秒数

按钮：

| 按钮 | 作用 |
|---|---|
| 立即检测 | 让后台马上做一次在座检测 |
| 检查工牌 | **立即做一次工牌检查**，即使不在 09:30–10:00 / 13:00–13:30 时段（不发通知；结果更新在“工牌状态”卡片） |
| 暂停 30 分钟 / 恢复 | 临时停止/恢复摄像头检测 |
| 日志 | 打开 `journalctl` 实时日志 |
| 帮助 | 打开内置帮助手册（含状态含义、节假日规则、暂存文件说明） |
| 启动/停止/重启服务 | 直接控制后台 systemd 服务；停止后不检测、不锁屏、不提醒 |

托盘菜单里也有“立即检查工牌”和“清理缓存图片”。关闭窗口会最小化到托盘，后台检测不受影响。

命令行等价操作：

```bash
... --check-now        # 立即在座检测
... --check-badge-now  # 立即检查工牌（不受时间窗限制）
... --set-camera /dev/video1   # 切换摄像头（同时写入配置）
... --retry-camera     # 摄像头恢复后立即重试
... --start-service / --stop-service / --restart-service
... --pause 1800       # 暂停 30 分钟
... --resume
```

> 手动工牌检查只更新面板状态和事件，**不会发通知**，避免调试时打扰。
> 立即检测/检查工牌时，面板会先显示“正在检测…”，完成后自动刷新最新状态。

暂存文件说明（面板“帮助”里也有）：

- 正常运行**不保存任何画面**；只有 `--save-debug` 才会写标注图到
  `~/.cache/presence-guard/debug/`，且自动保留最新 30 张、最多 7 天；
- 托盘菜单“清理缓存图片”或 `--clean-cache` 可立即清空；
- 精准度测试样本在 `~/.cache/presence-guard/accuracy/`，用其 `clean` 子命令清理。

---

## 6. 配置

配置文件：`~/.config/presence-guard/config.toml`
完整示例见仓库根目录的 `config.example.toml`。

最常改的几项：

```toml
[scheduler]
presence_interval_seconds = 60    # 每 60 秒检查一次空闲/在座
absence_confirm_seconds = 10      # 无人确认多久后锁屏
idle_gate_seconds = 10            # 键鼠活跃时跳过摄像头
idle_provider = "auto"            # auto / xprintidle / disabled

[badge]
holiday_mode = "chinese"           # 节假日识别（chinesecalendar）
# lock_on_holidays = false         # 非工作日不自动锁屏（默认）
workdays = [0, 1, 2, 3, 4]        # 回退用：周一至周五
windows = ["09:30-10:00", "13:00-13:30"]
interval_seconds = 180            # 窗口内每 3 分钟检查一次
max_reminders_per_window = 3      # 每个窗口最多提醒 3 次
notify_cooldown_seconds = 600     # 两次提醒至少间隔 10 分钟
min_blue_ratio = 0.006            # 调低更宽松
hue_min = 95
hue_max = 140

[camera]
presence_width = 320              # 在座检测低分辨率
presence_height = 240
badge_width = 640                 # 工牌检测高分辨率
badge_height = 480

[detector]
cv_threads = 1                    # 单线程，避免抢占 CPU

[general]
notify_on_camera_error = false    # 摄像头被占用时是否提醒
nice = 19                         # 进程优先级（最低）
```

修改后重启服务：

```bash
systemctl --user restart presence-guard
```

### 临时关闭某项功能

```bash
# 只做在座检测/锁屏，不检查工牌
... --no-badge

# 只检查工牌，不锁屏
... --no-lock

# 忽略键鼠活跃状态，每次都打开摄像头（调试/对比用）
... --no-idle-gate

# 全部只记录，不执行
... --dry-run
```

---

## 7. 隐私与安全

- 默认**不保存任何画面**：帧只在内存中处理，检查完立即释放。
- 正常检测**从不落盘**；只有显式加 `--save-debug` / `--debug-dir` 时才会保存带标注的图片，目录权限 `0700`。
- 调试图片会自动清理：每次写入后以及服务启动时，只保留最新 30 张、最多 7 天
  （`debug_keep_files` / `debug_max_age_days` 可调）。也可以手动执行 `--clean-cache`。
- 不录音、不联网、不上传任何数据。
- 摄像头打不开、模型报错、帧数不足时，**不会锁屏**（fail-safe），只记录日志。
- 锁屏后不再访问摄像头；解锁后下一次检查可能立即执行。
- 自动锁屏只调用 DDE 的 `RequestLock`，不涉及密码、不模拟输入。
- 锁屏期间完全不碰摄像头、不发通知；解锁后需**检测到在座**才会再次启用
  自动锁屏（避免解锁瞬间被重复锁屏）。

---

## 8. 已知限制

1. **采样延迟**：每 60 秒采样一次，离开后最坏约 70 秒锁屏。要更快就把
   `presence_interval_seconds` 调到 10–30 秒，但摄像头唤醒会更频繁。
2. **节假日**：默认非工作日（周末/法定节假日/调休放假）**只监测、不自动锁屏**，
   由 `[scheduler] lock_on_holidays = true` 可恢复节假日锁屏；chinesecalendar
   超出数据范围或缺失时回退到周一至周五。
3. **工牌颜色启发式**：深蓝色在暖光/低光下可能偏色；蓝色衣服可能被当成“已佩戴”。
   这是“宽松”策略的有意取舍。建议先用 `--image` / `--save-debug` 在真实环境校准。
4. **只看主目标**：画面里有多人时，只检查最大、最居中的人；路过的人一般不会被检查。
5. **看不到脸时**：工牌检查返回 `unknown`，不发通知，避免误报；3 分钟后的下一个检查点再试。
6. **没有 xprintidle 时**：无法判断键鼠活跃，会退化为每 60 秒打开一次摄像头；
   低性能机器建议安装 `xprintidle`（或设置 `idle_provider = "disabled"` 了解代价）。
7. **摄像头被占用**：例如正在视频会议，检查会失败并暂停锁屏；默认只写日志，不弹通知。
8. **摄像头 LED**：空闲时每次检查会短暂亮起（约 0.5–1 秒）；键鼠活跃时不亮。
9. **锁屏期间的解锁**：通过每 60 秒查询 `Locked` 感知，不会监听解锁信号；解锁后最多 60 秒内恢复检测。

---

## 9. 代码结构

```text
presence_guard/
  engine.py       # 调度状态机：60 秒采样、10 秒确认、工作日工牌窗口
  schedule.py     # 工作日 09:30-10:00 / 13:00-13:30 的 3 分钟检查点
  camera.py       # 短时打开/关闭 V4L2 摄像头（在座 320x240 / 工牌 640x480）
  idle.py         # xprintidle 空闲查询，键鼠活跃时跳过摄像头
  status.py       # status.json 发布 + 控制 socket（GUI 用）
  gui.py          # PyQt5 状态面板与托盘图标
  cache.py        # 调试图片自动清理
packaging/       # deb 打包（DEBIAN/control、postinst、systemd unit、desktop）
scripts/build-deb.sh  # 构建 deb：默认精简版（~4MB），FULL=1 离线版（~85MB）
  vision.py       # YuNet 优先；疑似无人确认时才用 NanoDet；ROI 标注
  nanodet.py      # NanoDet ONNX 后处理（OpenCV DNN）
  badge.py        # 胸口 ROI + HSV 深蓝色启发式 + 多帧投票
  dbus_api.py     # RequestLock / Locked / Notify
  config.py       # TOML 配置与校验
  state_store.py  # 通知冷却、上次检查时间
  selftest.py     # 离线自检
tests/            # 单元测试（不需要摄像头）
systemd/          # user service
scripts/          # install / run / uninstall
models/           # nanodet.onnx, yunet.onnx
```

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

当前 69 个单元测试覆盖：工牌 ROI/HSV 投票、提醒次数上限与窗口重置、
10 秒无人确认、空闲门控、摄像头异常不锁屏、锁屏期间不碰摄像头，
以及四张真实摄像头画面（挂绳+卡片 / 只有挂绳碎片 / 完全摘下且有蓝色风扇 /
深蓝衣服带蓝色斑点）
在两种模式下的回归测试，以及节假日/状态文件/控制 socket/缓存清理/手动工牌
检查/解锁后不重复锁屏/节假日不锁屏/摄像头暂停恢复/摄像头切换持久化。

---

## 10. 常见问题

**Q：日志里一直是 `camera device not found`？**
检查 `ls -l /dev/video*`；确认摄像头没有被 BIOS/硬件开关禁用；确认用户在 `video` 组。

**Q：摄像头被其他程序占用怎么办？**
工具会把它当作“不可用”，暂停自动锁屏并 5 分钟后重试，不会误锁屏。
需要提醒时把 `general.notify_on_camera_error` 设为 `true`。

**Q：为什么工牌检查不是每 3 分钟准点？**
在座检测每 60 秒一次，工牌检查点到达后，会在下一次在座检测时顺带执行，最多晚 60 秒。
如果检查点到达时座位上没人，这个检查点直接跳过，3 分钟后再查，避免对着空椅子分析。

**Q：为什么一个窗口最多只收到 3 次工牌提醒？**
默认 `max_reminders_per_window = 3`，并且两次提醒至少间隔 `notify_cooldown_seconds = 600` 秒，
所以 30 分钟窗口内最多 3 次、且分散在后半段。需要更多/更少可自行调整。

**Q：低性能机器会不会卡顿？**
键鼠活跃时完全不打开摄像头，也不做推理；只有空闲 10 秒后才做一次快速扫描。
快速扫描只用 YuNet（约 170 ms CPU），只有疑似无人、进入 10 秒确认时才运行 NanoDet
（约 830 ms CPU，只跑一次）；进程已设为 `nice=19`、单线程、IO idle，前台程序会优先获得 CPU。

**Q：工牌检查会打扰工作吗？**
只有 09:30–10:00、13:00–13:30 两个窗口会检查，窗口内每 3 分钟一次；
其它时间完全不检查工牌。`--no-badge` 可临时关闭。

**Q：能不能不用 systemd，登录后手动启动？**
可以：

```bash
~/.local/share/presence-guard/venv/bin/python -m presence_guard
```

---

## 11. 许可证

- 本项目代码：Apache-2.0，见 `LICENSE`。
- NanoDet 模型/代码：Apache-2.0，见 `models/NOTICE.md`。
- YuNet 模型：MIT，见 `models/NOTICE.md`。
- OpenCV：Apache-2.0。
