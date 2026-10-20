#!/usr/bin/env bash
# 安装到当前用户的 ~/.local/share/presence-guard，并注册 systemd user service。
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/presence-guard"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/presence-guard"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
PYTHON="${PYTHON:-python3}"

echo "==> 源码目录: $SRC_DIR"
echo "==> 安装目录: $APP_DIR"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "错误: 找不到 $PYTHON" >&2
    exit 1
fi
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 7):
    raise SystemExit("需要 Python 3.7 或更高版本")
PY

if [[ ! -f "$SRC_DIR/models/nanodet.onnx" || ! -f "$SRC_DIR/models/yunet.onnx" ]]; then
    echo "错误: models/ 下缺少 nanodet.onnx 或 yunet.onnx" >&2
    exit 1
fi

mkdir -p "$APP_DIR/app" "$CONFIG_DIR" "$SERVICE_DIR"

echo "==> 复制程序文件"
tar -C "$SRC_DIR" \
    --exclude='.venv' --exclude='.build-venv' --exclude='.git' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='tests' \
    --exclude='*.pt' --exclude='*.log' \
    -cf - . | tar -C "$APP_DIR/app" -xf -

if [[ ! -x "$APP_DIR/venv/bin/python" ]] \
        || ! "$APP_DIR/venv/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "==> 创建虚拟环境（复用系统 python3-gi，用于 D-Bus）"
    rm -rf "$APP_DIR/venv"
    "$PYTHON" -m venv --system-site-packages "$APP_DIR/venv"
fi

# Make `python -m presence_guard` work from any working directory.
SITE_PACKAGES="$(ls -d "$APP_DIR"/venv/lib/python*/site-packages 2>/dev/null | head -1)"
if [[ -n "$SITE_PACKAGES" ]]; then
    echo "$APP_DIR/app" > "$SITE_PACKAGES/presence-guard.pth"
fi

# pip 18 (UOS 20 default) cannot install modern wheels reliably.
"$APP_DIR/venv/bin/python" -m pip install --disable-pip-version-check -q \
    ${PRESENCE_GUARD_PIP_INDEX:+-i "$PRESENCE_GUARD_PIP_INDEX"} \
    -U "pip>=21,<24" >/dev/null 2>&1 || true

echo "==> 安装 Python 依赖（首次约 60 MB）"
PIP_INDEX_ARGS=()
if [[ -n "${PRESENCE_GUARD_PIP_INDEX:-}" ]]; then
    PIP_INDEX_ARGS=(-i "$PRESENCE_GUARD_PIP_INDEX")
fi
if ! "$APP_DIR/venv/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "错误: 虚拟环境中的 pip 不可用（UOS 20 需要 python3.7-venv/python3-pip）" >&2
    echo "      请执行: sudo apt install python3-venv python3-pip" >&2
    exit 1
fi
if ! "$APP_DIR/venv/bin/python" -m pip install --disable-pip-version-check -q \
        "${PIP_INDEX_ARGS[@]}" -r "$APP_DIR/app/requirements.txt"; then
    if [[ -z "${PRESENCE_GUARD_PIP_INDEX:-}" ]]; then
        echo "==> 默认源安装失败，改用清华镜像重试"
        "$APP_DIR/venv/bin/python" -m pip install --disable-pip-version-check -q \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            -r "$APP_DIR/app/requirements.txt"
    else
        echo "错误: 依赖安装失败" >&2
        exit 1
    fi
fi

if ! "$APP_DIR/venv/bin/python" -c "import gi" >/dev/null 2>&1; then
    echo "警告: 无法导入 python3-gi，锁屏和通知功能将不可用。"
    echo "      请安装: sudo apt install python3-gi"
fi

if ! command -v xprintidle >/dev/null 2>&1; then
    echo "提示: 安装 xprintidle 后，键鼠活跃时可跳过摄像头，进一步降低占用："
    echo "      sudo apt install xprintidle"
fi

if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
    echo "==> 写入默认配置: $CONFIG_DIR/config.toml"
    cp "$SRC_DIR/config.example.toml" "$CONFIG_DIR/config.toml"
else
    echo "==> 保留已有配置: $CONFIG_DIR/config.toml"
fi

echo "==> 安装状态面板（若系统有 python3-pyqt5）"
UI_OK=0
if "$APP_DIR/venv/bin/python" -c "import PyQt5" >/dev/null 2>&1; then
    UI_OK=1
    ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
    APP_DIR_DESKTOP="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    AUTOSTART_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
    mkdir -p "$ICON_DIR" "$APP_DIR_DESKTOP" "$AUTOSTART_DIR"
    cp "$SRC_DIR/assets/presence-guard.svg" "$ICON_DIR/presence-guard.svg"
    sed "s|@APP_DIR@|$APP_DIR|g" "$SRC_DIR/assets/presence-guard-ui.desktop" \
        > "$APP_DIR_DESKTOP/presence-guard-ui.desktop"
    cp "$APP_DIR_DESKTOP/presence-guard-ui.desktop" \
        "$AUTOSTART_DIR/presence-guard-ui.desktop"
    chmod 644 "$APP_DIR_DESKTOP/presence-guard-ui.desktop" \
        "$AUTOSTART_DIR/presence-guard-ui.desktop"
    echo "    已安装应用菜单项和登录自启动项"
else
    echo "    跳过：未找到 python3-pyqt5（sudo apt install python3-pyqt5）"
fi

echo "==> 安装 systemd user service"
sed "s|@APP_DIR@|$APP_DIR|g" \
    "$SRC_DIR/systemd/presence-guard.service" \
    > "$SERVICE_DIR/presence-guard.service"

START_SERVICE=1
if [[ "${PRESENCE_GUARD_NO_START:-0}" == "1" ]]; then
    START_SERVICE=0
fi

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    systemctl --user daemon-reload
    if [[ "$START_SERVICE" == "1" ]]; then
        systemctl --user enable --now presence-guard.service
        echo
        echo "已启动。查看日志："
        echo "  journalctl --user -u presence-guard -f"
    else
        echo "已安装但未启动（PRESENCE_GUARD_NO_START=1）。手动启动："
        echo "  systemctl --user enable --now presence-guard"
    fi
else
    echo "未检测到可用的 systemd user session，请手动启动："
    echo "  $APP_DIR/venv/bin/python -m presence_guard"
fi

if ! id -nG "${USER:-$(id -un)}" | grep -qw video; then
    echo
    echo "警告: 当前用户不在 video 组，可能无法打开摄像头。"
    echo "      请执行: sudo usermod -aG video \$USER  然后重新登录。"
fi

echo
echo "==> 状态检查"
"$APP_DIR/venv/bin/python" -m presence_guard --status || true
echo
echo "常用命令："
echo "  $APP_DIR/venv/bin/python -m presence_guard --gui            # 状态面板"
echo "  $APP_DIR/venv/bin/python -m presence_guard --once --check-badge --save-debug"
echo "  $APP_DIR/venv/bin/python -m presence_guard --check-badge-now  # 立即查工牌"
echo "  $APP_DIR/venv/bin/python -m presence_guard --clean-cache      # 清理旧图片"
echo "  $APP_DIR/venv/bin/python -m presence_guard --status"
echo "  systemctl --user restart presence-guard"
