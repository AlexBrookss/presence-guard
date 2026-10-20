"""PyQt5 status panel for presence-guard.

The GUI is a separate, read-mostly process:

* it reads ``status.json`` written by the daemon (no IPC needed for display);
* it sends commands (check now / pause / resume) over the Unix control socket;
* if it crashes or is closed, detection and locking are unaffected.

Run with ``presence-guard --gui``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    from PyQt5.QtCore import QSize, Qt, QTimer
    from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
    from PyQt5.QtWidgets import (
        QAction, QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout,
        QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox,
        QPlainTextEdit, QPushButton, QSystemTrayIcon, QVBoxLayout, QWidget,
    )
except ImportError as exc:  # pragma: no cover - depends on desktop packages
    raise SystemExit(
        "需要 python3-pyqt5 才能打开状态面板：sudo apt install python3-pyqt5"
    ) from exc

from .config import Config
from .status import ControlClient, StatusStore


HELP_TEXT = (
    "按钮\n"
    " 立即检测     让后台马上做一次在座检测\n"
    " 检查工牌     立即检查工牌，即使不在 09:30-10:00 / 13:00-13:30；\n"
    "              手动检查不会发通知\n"
    " 暂停/恢复    临时停止摄像头检测（30 分钟）\n"
    " 日志         打开 journalctl 实时日志\n"
    " 启动/停止/重启服务  控制后台 systemd 服务；停止后不检测、不锁屏、\n"
    "              不提醒，适合临时完全关闭\n"
    " 清理缓存图片 删除旧的调试截图\n"
    "\n状态含义\n"
    " 在工位 / 在工位（键鼠活跃） / 已离座 / 锁屏中 / 已暂停 / 摄像头异常\n"
    " 工牌：已佩戴 / 未佩戴 / 无法判断 / 未到检查时间 / 等待检查\n"
    "\n自动锁屏\n"
    " 离座约 10 秒后自动锁屏；锁屏期间不碰摄像头、不发通知。\n"
    " 解锁后需检测到在座才会再次启用自动锁屏（避免重复锁屏）。\n"
    " 非工作日（周末/法定节假日）默认不自动锁屏，配置项：\n"
    "   [scheduler] lock_on_holidays = false\n"
    "\n摄像头异常\n"
    " 摄像头缺失/被占用时检测自动暂停，恢复后自动继续（每 15-300 秒探测）；\n"
    " 画面过暗或被遮挡时只暂停检测，绝不会因此锁屏。\n"
    "\n工牌检查时段与节假日\n"
    " 默认工作日 09:30-10:00、13:00-13:30，每 3 分钟一次；\n"
    " 通过 chinesecalendar 识别法定节假日与调休，不是死板的周一至周五。\n"
    " 每个时段最多提醒 3 次。\n"
    "\n暂存文件说明（正常检测不保存任何画面）\n"
    " 调试截图目录：~/.cache/presence-guard/debug\n"
    "   仅在使用 --save-debug 时生成；自动保留最新 30 张、最多 7 天，\n"
    "   也可用「清理缓存图片」或 --clean-cache 立即清空。\n"
    " 精准度测试样本：~/.cache/presence-guard/accuracy\n"
    "   由 tools/accuracy 工具生成，请用其 clean 命令清理。\n"
    " 状态文件：~/.local/state/presence-guard/status.json（供面板读取）\n"
    " 配置：~/.config/presence-guard/config.toml\n"
    " 日志：journalctl --user -u presence-guard -f\n"
)

STATE_COLORS = {
    "present": "#1f9d55",
    "active": "#1f9d55",
    "absent": "#d98200",
    "locked": "#2f6fd0",
    "paused": "#8a8f99",
    "error": "#d9534f",
    "starting": "#8a8f99",
    "stopped": "#8a8f99",
}

BADGE_COLORS = {
    "present": "#1f9d55",
    "absent": "#d9534f",
    "unknown": "#d98200",
    "not_checked": "#8a8f99",
}

STYLE = """
QWidget#root { background: #f2f3f5; }
QFrame#card { background: #ffffff; border-radius: 14px; }
QLabel#title { font-size: 16px; font-weight: 600; color: #20242a; }
QLabel#section { font-size: 12px; color: #8a8f99; }
QLabel#big { font-size: 26px; font-weight: 600; }
QLabel#meta { font-size: 12px; color: #8a8f99; }
QLabel#value { font-size: 13px; color: #20242a; font-weight: 500; }
QPushButton {
    background: #ffffff; color: #20242a; border: 1px solid #dcdfe4;
    border-radius: 9px; padding: 8px 12px; font-size: 13px;
}
QPushButton:hover { background: #f6f7f9; }
QPushButton:pressed { background: #eceef2; }
QPushButton#primary {
    background: #2f6fd0; color: #ffffff; border: none; font-weight: 600;
}
QPushButton#primary:hover { background: #3b7ee0; }
QPushButton#primary:pressed { background: #2861b8; }
QPushButton:disabled { color: #b6bac2; border-color: #e6e8ec; }
QPushButton#danger { color: #c0392b; }
QPushButton#danger:hover { background: #fdf0ee; }
QListWidget {
    background: #ffffff; border: none; border-radius: 14px;
    padding: 6px; font-size: 12px;
}
"""


def _dot(color: str, size: int = 14) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return pm


def _app_icon(color: str) -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#2f6fd0"))
    painter.drawRoundedRect(4, 4, 56, 56, 16, 16)
    painter.setBrush(QColor(color))
    painter.drawEllipse(22, 22, 20, 20)
    painter.end()
    return QIcon(pm)


class StatusWindow(QWidget):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_dir = cfg.general.state_dir
        self.client = ControlClient(self.state_dir)
        self.last_events: tuple = ()
        self.paused = False
        self.service_active: Optional[bool] = None
        self._service_busy = False

        self.setObjectName("root")
        self.setWindowTitle("状态面板")
        self.setWindowIcon(_app_icon("#1f9d55"))
        self.setMinimumWidth(420)
        self.resize(430, 740)
        self.setStyleSheet(STYLE)

        self._build_ui()
        self._build_tray()

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

        self.service_timer = QTimer(self)
        self.service_timer.setInterval(5000)
        self.service_timer.timeout.connect(self._refresh_service)
        self.service_timer.start()
        self._refresh_service()

    # ------------------------------------------------------------------ UI
    def _card(self) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(4)
        return frame, layout

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # header
        header = QHBoxLayout()
        title = QLabel("在岗守护")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        self.pill = QLabel("检查服务状态…")
        self.pill.setObjectName("meta")
        header.addWidget(self.pill)
        root.addLayout(header)

        # presence card
        card, layout = self._card()
        section = QLabel("当前状态")
        section.setObjectName("section")
        layout.addWidget(section)
        row = QHBoxLayout()
        self.state_dot = QLabel()
        self.state_dot.setFixedSize(16, 16)
        row.addWidget(self.state_dot, 0, Qt.AlignVCenter)
        self.state_label = QLabel("正在读取…")
        self.state_label.setObjectName("big")
        row.addWidget(self.state_label, 1)
        layout.addLayout(row)
        self.state_meta = QLabel("—")
        self.state_meta.setObjectName("meta")
        layout.addWidget(self.state_meta)
        root.addWidget(card)

        # badge card
        card, layout = self._card()
        section = QLabel("工牌状态")
        section.setObjectName("section")
        layout.addWidget(section)
        row = QHBoxLayout()
        self.badge_dot = QLabel()
        self.badge_dot.setFixedSize(16, 16)
        row.addWidget(self.badge_dot, 0, Qt.AlignVCenter)
        self.badge_label = QLabel("—")
        self.badge_label.setObjectName("big")
        row.addWidget(self.badge_label, 1)
        layout.addLayout(row)
        self.badge_meta = QLabel("—")
        self.badge_meta.setObjectName("meta")
        self.badge_meta.setWordWrap(True)
        layout.addWidget(self.badge_meta)
        root.addWidget(card)

        # info grid
        card, layout = self._card()
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        self.info_labels: dict[str, QLabel] = {}
        entries = [
            ("next_presence", "下次在座检测"),
            ("next_badge", "下次工牌检查"),
            ("mode", "工牌判定模式"),
            ("camera", "摄像头"),
        ]
        for index, (key, caption) in enumerate(entries):
            row, col = divmod(index, 2)
            box = QVBoxLayout()
            box.setSpacing(1)
            cap = QLabel(caption)
            cap.setObjectName("section")
            val = QLabel("—")
            val.setObjectName("value")
            box.addWidget(cap)
            box.addWidget(val)
            grid.addLayout(box, row, col)
            self.info_labels[key] = val
        layout.addLayout(grid)
        root.addWidget(card)

        # buttons: two rows so the labels stay readable at 430 px
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.check_button = QPushButton("立即检测")
        self.check_button.setObjectName("primary")
        self.check_button.clicked.connect(self._check_now)
        buttons.addWidget(self.check_button, 1)

        self.badge_button = QPushButton("检查工牌")
        self.badge_button.clicked.connect(self._check_badge)
        buttons.addWidget(self.badge_button, 1)

        self.help_button = QPushButton("帮助")
        self.help_button.clicked.connect(self._show_help)
        buttons.addWidget(self.help_button, 1)
        root.addLayout(buttons)

        buttons2 = QHBoxLayout()
        buttons2.setSpacing(8)
        self.pause_button = QPushButton("暂停 30 分钟")
        self.pause_button.clicked.connect(self._toggle_pause)
        buttons2.addWidget(self.pause_button, 1)

        self.log_button = QPushButton("日志")
        self.log_button.clicked.connect(self._open_logs)
        buttons2.addWidget(self.log_button, 0)
        root.addLayout(buttons2)

        # service control: start / stop / restart the systemd user service
        service_row = QHBoxLayout()
        service_row.setSpacing(8)
        self.service_label = QLabel("后台服务：检查中…")
        self.service_label.setObjectName("section")
        service_row.addWidget(self.service_label, 1)
        self.start_button = QPushButton("启动服务")
        self.start_button.clicked.connect(lambda: self._service_action("start"))
        service_row.addWidget(self.start_button)
        self.stop_button = QPushButton("停止服务")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(lambda: self._service_action("stop"))
        service_row.addWidget(self.stop_button)
        self.restart_button = QPushButton("重启服务")
        self.restart_button.clicked.connect(lambda: self._service_action("restart"))
        service_row.addWidget(self.restart_button)
        root.addLayout(service_row)

        # events
        section = QLabel("最近事件")
        section.setObjectName("section")
        root.addWidget(section)
        self.events = QListWidget()
        self.events.setSelectionMode(QListWidget.NoSelection)
        self.events.setFocusPolicy(Qt.NoFocus)
        self.events.setMinimumHeight(110)
        root.addWidget(self.events, 1)

        footer = QLabel("关闭窗口会最小化到托盘；后台检测继续运行")
        footer.setObjectName("meta")
        footer.setAlignment(Qt.AlignCenter)
        root.addWidget(footer)

    def _build_tray(self) -> None:
        self.tray: Optional[QSystemTrayIcon] = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(_app_icon("#8a8f99"), self)
        self.tray.setToolTip("在岗守护")
        menu = QMenu()
        show_action = QAction("显示状态窗口", self)
        show_action.triggered.connect(self._show_window)
        menu.addAction(show_action)
        check_action = QAction("立即检测", self)
        check_action.triggered.connect(self._check_now)
        menu.addAction(check_action)
        badge_action = QAction("立即检查工牌", self)
        badge_action.triggered.connect(self._check_badge)
        menu.addAction(badge_action)
        self.tray_pause_action = QAction("暂停 30 分钟", self)
        self.tray_pause_action.triggered.connect(self._toggle_pause)
        menu.addAction(self.tray_pause_action)
        menu.addSeparator()
        service_start = QAction("启动服务", self)
        service_start.triggered.connect(lambda: self._service_action("start"))
        menu.addAction(service_start)
        service_stop = QAction("停止服务", self)
        service_stop.triggered.connect(lambda: self._service_action("stop"))
        menu.addAction(service_stop)
        service_restart = QAction("重启服务", self)
        service_restart.triggered.connect(lambda: self._service_action("restart"))
        menu.addAction(service_restart)
        menu.addSeparator()
        help_action = QAction("帮助", self)
        help_action.triggered.connect(self._show_help)
        menu.addAction(help_action)
        clean_action = QAction("清理缓存图片", self)
        clean_action.triggered.connect(self._clean_cache)
        menu.addAction(clean_action)
        menu.addSeparator()
        quit_action = QAction("退出面板", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    # -------------------------------------------------------------- update
    def refresh(self) -> None:
        if self.service_active is False:
            self._render_stopped()
            return
        data = StatusStore.read(self.state_dir)
        if data is None:
            self._render_missing()
            return

        state = str(data.get("state") or "unknown")
        busy = str(data.get("busy") or "")
        color = STATE_COLORS.get(state, "#8a8f99")
        self._set_dot(self.state_dot, color)
        self.state_label.setText(str(data.get("state_text") or state))
        self.state_label.setStyleSheet(f"color: {color};")

        updated = data.get("updated_at_text") or "—"
        idle = data.get("idle_seconds")
        cpu = data.get("cpu_ms")
        bits = [f"更新于 {updated}"]
        if idle is not None:
            bits.append(f"空闲 {idle:.0f} 秒" if idle >= 1 else "键鼠活跃")
        if cpu is not None:
            bits.append(f"本轮 CPU {cpu:.0f} ms")
        if busy == "presence":
            bits.append("正在检测…")
        if data.get("lock_allowed_today") is False:
            bits.append("非工作日：不自动锁屏")
        self.state_meta.setText(" · ".join(bits))

        badge = str(data.get("badge") or "not_checked")
        badge_color = BADGE_COLORS.get(badge, "#8a8f99")
        self._set_dot(self.badge_dot, badge_color)
        active = bool(data.get("badge_window_active"))
        if busy == "badge":
            self.badge_label.setText("正在检查工牌…")
            badge_color = "#2f6fd0"
            self._set_dot(self.badge_dot, badge_color)
            self.badge_label.setStyleSheet(f"color: {badge_color};")
        elif not active:
            self.badge_label.setText("未到检查时间")
            badge_color = "#8a8f99"
            self._set_dot(self.badge_dot, badge_color)
        elif badge == "not_checked":
            self.badge_label.setText("等待检查")
        else:
            self.badge_label.setText(str(data.get("badge_text") or "尚未检查"))
        self.badge_label.setStyleSheet(f"color: {badge_color};")
        mode = str(data.get("badge_mode") or "strap")
        mode_text = "挂绳即算（宽松）" if mode == "strap" else "必须看到卡片（严格）"
        holiday = str(data.get("badge_holiday_mode") or "")
        if holiday == "chinese":
            mode_text += " · 节假日自动识别"
        checked = data.get("badge_checked_at") or "尚未检查"
        votes = data.get("badge_votes") or ""
        reason = data.get("badge_reason") or ""
        windows = data.get("badge_windows") or []
        window_text = " / ".join(windows) if windows else "未配置"
        if busy == "badge":
            meta = "请面向摄像头，稍候…"
            self.badge_meta.setText(meta)
            self._update_buttons()
            return
        if not active:
            meta = f"检查时段：{window_text}"
            if badge != "not_checked":
                meta += f"\n上次检查：{checked} · {data.get('badge_text', '')}"
                if votes:
                    meta += f"（{votes} 帧）"
            meta += "\n模式：" + mode_text
        elif badge == "not_checked":
            meta = f"检查时段：{window_text}\n模式：{mode_text}"
        else:
            meta = f"模式：{mode_text} · 上次检查：{checked}"
            if votes:
                meta += f" · {votes} 帧"
            if badge == "absent":
                count = data.get("badge_reminder_count") or 0
                maximum = data.get("badge_reminder_max") or 0
                if count:
                    meta += f"\n本时段已提醒：{count}/{maximum}"
            if reason:
                meta += f"\n{reason}"
        self.badge_meta.setText(meta)

        self.info_labels["next_presence"].setText(
            str(data.get("next_presence_at") or "—"))
        self.info_labels["next_badge"].setText(
            str(data.get("next_badge_at") or "未启用"))
        self.info_labels["mode"].setText(mode_text)
        camera = str(data.get("camera") or "ok")
        device = str(data.get("camera_device") or "")
        self.info_labels["camera"].setText(
            (device if device else "摄像头") + ("（正常）" if camera == "ok" else "（异常）"))

        paused_until = data.get("paused_until")
        self.paused = bool(paused_until)
        self.pause_button.setText("恢复检测" if self.paused else "暂停 30 分钟")
        if self.tray is not None:
            self.tray_pause_action.setText(
                "恢复检测" if self.paused else "暂停 30 分钟")

        if self.service_active is True:
            self.service_label.setText("后台服务：运行中")
            self.service_label.setStyleSheet("color: #1f9d55; font-size: 12px;")
        interval = data.get("presence_interval") or 60
        stale = time.time() - float(data.get("updated_at") or 0) > max(180, interval * 4)
        if stale:
            self.pill.setText("状态可能已过期")
            self.pill.setStyleSheet("color: #d98200; font-size: 12px;")
        else:
            self.pill.setText("后台服务运行中")
            self.pill.setStyleSheet("color: #1f9d55; font-size: 12px;")

        self._render_events(data.get("events") or [])
        self._update_buttons()

        if self.tray is not None:
            self.tray.setIcon(_app_icon(color))
            self.tray.setToolTip(
                f"在岗守护：{data.get('state_text', '')} / "
                f"{data.get('badge_text', '')}")

    def _render_stopped(self) -> None:
        self._set_dot(self.state_dot, "#8a8f99")
        self.state_label.setText("服务已停止")
        self.state_label.setStyleSheet("color: #8a8f99;")
        self.state_meta.setText("点击「启动服务」恢复检测与自动锁屏")
        self._set_dot(self.badge_dot, "#8a8f99")
        self.badge_label.setText("—")
        self.badge_label.setStyleSheet("color: #8a8f99;")
        self.badge_meta.setText("服务停止期间不会检测、不会锁屏、不会提醒")
        for label in self.info_labels.values():
            label.setText("—")
        self.pill.setText("服务已停止")
        self.pill.setStyleSheet("color: #d9534f; font-size: 12px;")
        self.service_label.setText("后台服务：已停止")
        self.service_label.setStyleSheet("color: #d9534f; font-size: 12px;")
        self._update_buttons()

    def _render_missing(self) -> None:
        self._set_dot(self.state_dot, "#8a8f99")
        self.state_label.setText("后台服务未运行")
        self.state_label.setStyleSheet("color: #8a8f99;")
        self.state_meta.setText("启动服务后这里会显示实时状态")
        self._set_dot(self.badge_dot, "#8a8f99")
        self.badge_label.setText("—")
        self.badge_label.setStyleSheet("color: #8a8f99;")
        self.badge_meta.setText("—")
        for label in self.info_labels.values():
            label.setText("—")
        self.pill.setText("未连接")
        self.pill.setStyleSheet("color: #d9534f; font-size: 12px;")
        self.events.clear()
        self.paused = False
        self.pause_button.setText("暂停 30 分钟")
        self.service_label.setText("后台服务：未连接")
        self.service_label.setStyleSheet("color: #d9534f; font-size: 12px;")
        self._update_buttons()

    def _render_events(self, events: list[dict[str, Any]]) -> None:
        signature = tuple((e.get("time"), e.get("text"), e.get("level"))
                          for e in events)
        if signature == self.last_events:
            return
        self.last_events = signature
        self.events.clear()
        colors = {"info": "#4a4f57", "warn": "#c47f00", "error": "#d9534f"}
        for event in events:
            item = QListWidgetItem(
                f"{event.get('time', '')}  {event.get('text', '')}")
            item.setForeground(QColor(colors.get(event.get("level", "info"),
                                                "#4a4f57")))
            self.events.addItem(item)

    @staticmethod
    def _set_dot(label: QLabel, color: str) -> None:
        label.setPixmap(_dot(color))

    # ------------------------------------------------------------- actions
    def _send(self, cmd: str, **payload: Any) -> bool:
        result = self.client.send(cmd, **payload)
        if not result.get("ok"):
            QMessageBox.warning(self, "操作失败",
                                str(result.get("error") or "后台服务未响应"))
            return False
        QTimer.singleShot(300, self.refresh)
        return True

    def _update_buttons(self) -> None:
        running = self.service_active is not False
        self.check_button.setEnabled(running)
        self.badge_button.setEnabled(running)
        self.pause_button.setEnabled(running)
        self.start_button.setEnabled(self.service_active is False)
        self.stop_button.setEnabled(self.service_active is True)
        self.restart_button.setEnabled(self.service_active is not False)

    def _refresh_service(self) -> None:
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", "presence-guard"],
                capture_output=True, text=True, timeout=4, check=False)
            self.service_active = result.stdout.strip() == "active"
        except Exception:
            self.service_active = None
        self.refresh()

    def _service_action(self, action: str) -> None:
        if action == "stop":
            answer = QMessageBox.question(
                self, "停止后台服务",
                "停止后不会检测在座、不会自动锁屏、不会提醒工牌。\n确定停止吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        if self._service_busy:
            return
        self._service_busy = True
        try:
            result = subprocess.run(
                ["systemctl", "--user", action, "presence-guard"],
                capture_output=True, text=True, timeout=20, check=False)
            if result.returncode != 0:
                QMessageBox.warning(
                    self, "服务操作失败",
                    (result.stderr or result.stdout or "systemctl 执行失败").strip())
        except Exception as exc:
            QMessageBox.warning(self, "服务操作失败", str(exc))
        finally:
            self._service_busy = False
        QTimer.singleShot(1200, self._refresh_service)

    def _check_now(self) -> None:
        self._send("check_now")

    def _check_badge(self) -> None:
        if self._send("check_badge"):
            if self.tray is not None:
                self.tray.showMessage("在岗守护", "正在检查工牌…",
                                      QSystemTrayIcon.Information, 1500)

    def _show_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("在岗守护 · 帮助")
        dialog.resize(560, 520)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(HELP_TEXT)
        text.setStyleSheet("font-size: 13px; color: #20242a;")
        layout.addWidget(text)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec_()

    def _clean_cache(self) -> None:
        from .cache import cleanup_dir
        removed, freed = cleanup_dir(
            self.cfg.general.debug_dir,
            self.cfg.general.debug_keep_files,
            self.cfg.general.debug_max_age_days,
        )
        QMessageBox.information(
            self, "清理缓存",
            f"已删除 {removed} 张旧图片，释放 {freed // 1024} KB。")

    def _toggle_pause(self) -> None:
        if self.paused:
            self._send("resume")
        else:
            self._send("pause", seconds=1800)

    def _open_logs(self) -> None:
        command = "journalctl --user -u presence-guard -f"
        terminal = None
        for candidate in ("deepin-terminal", "x-terminal-emulator",
                          "gnome-terminal", "konsole", "xterm"):
            if shutil.which(candidate):
                terminal = candidate
                break
        if terminal is None:
            QMessageBox.information(self, "查看日志",
                                    f"请在终端执行：\n\n{command}")
            return
        try:
            if terminal == "deepin-terminal":
                subprocess.Popen([terminal, "-e", command])
            elif terminal == "gnome-terminal":
                subprocess.Popen([terminal, "--", "bash", "-lc", command])
            else:
                subprocess.Popen([terminal, "-e", command])
        except Exception as exc:
            QMessageBox.information(self, "查看日志",
                                    f"无法打开终端（{exc}），请手动执行：\n\n{command}")

    def _show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_window()

    def _quit(self) -> None:
        self.timer.stop()
        if self.tray is not None:
            self.tray.hide()
        QApplication.instance().quit()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.tray is not None:
            event.ignore()
            self.hide()
            self.tray.showMessage("在岗守护", "已最小化到托盘，后台检测继续运行",
                                  QSystemTrayIcon.Information, 2000)
        else:
            self._quit()


def run_gui(cfg: Config) -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("presence-guard")
    app.setApplicationDisplayName("在岗守护")
    app.setQuitOnLastWindowClosed(False)

    pid_path = Path(cfg.general.state_dir).expanduser() / "gui.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if pid_path.exists():
        try:
            other = int(pid_path.read_text().strip())
            os.kill(other, 0)
            QMessageBox.information(None, "在岗守护", "状态面板已经在运行。")
            return 0
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass
    pid_path.write_text(str(os.getpid()))

    def _cleanup_pid():
        try:
            if pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()
        except OSError:
            pass

    import atexit
    atexit.register(_cleanup_pid)

    window = StatusWindow(cfg)
    window.show()
    if QSystemTrayIcon.isSystemTrayAvailable():
        window.tray.showMessage(
            "在岗守护", "状态面板已启动，关闭窗口会最小化到托盘。",
            QSystemTrayIcon.Information, 2500)
    return app.exec_()
