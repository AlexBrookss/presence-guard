"""Command-line entry point."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .camera import CameraCapture
from .config import ConfigError, load_config, validate_config
from .dbus_api import DesktopActions
from .engine import Engine
from .state_store import StateStore
from .status import ControlClient, StatusStore
from .vision import VisionPipeline

log = logging.getLogger("presence_guard")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="presence-guard",
        description="定时检测在座状态，长时间无人自动锁屏；定期检查深蓝色工牌。",
    )
    p.add_argument("--config", help="TOML 配置文件路径")
    p.add_argument("--camera", help="摄像头设备，例如 /dev/video0 或 0")
    p.add_argument("--once", action="store_true", help="只执行一次检测并输出 JSON")
    p.add_argument("--image", help="分析一张静态图片（不访问摄像头）")
    p.add_argument("--check-badge", action="store_true",
                   help="单次检测时强制检查工牌")
    p.add_argument("--debug-dir", help="保存带标注的调试图片到该目录（仅单次检测）")
    p.add_argument("--save-debug", action="store_true",
                   help="使用默认调试目录 ~/.cache/presence-guard/debug")
    p.add_argument("--dry-run", action="store_true",
                   help="不实际锁屏/发通知，只记录将要执行的动作")
    p.add_argument("--no-lock", action="store_true", help="禁用自动锁屏")
    p.add_argument("--no-badge", action="store_true", help="禁用定期工牌检查")
    p.add_argument("--no-idle-gate", action="store_true",
                   help="忽略键鼠空闲状态，每次都打开摄像头")
    p.add_argument("--badge-mode", choices=["strap", "card"],
                   help="工牌判定模式：strap=有挂绳即算，card=必须看到卡片")
    p.add_argument("--presence-interval", type=int,
                   help="在座检测间隔（秒），默认 60")
    p.add_argument("--badge-interval", type=int,
                   help="工牌检查间隔（秒），默认 180")
    p.add_argument("--absence-seconds", type=int,
                   help="确认无人多少秒后锁屏，默认 10")
    p.add_argument("--notify-test", action="store_true", help="发送一条测试通知")
    p.add_argument("--gui", action="store_true", help="打开状态面板（PyQt5）")
    p.add_argument("--check-now", action="store_true",
                   help="请求正在运行的服务立即检测一次")
    p.add_argument("--check-badge-now", action="store_true",
                   help="请求立即检查一次工牌（即使不在检查时段）")
    p.add_argument("--retry-camera", action="store_true",
                   help="摄像头恢复后立即重试（暂停状态）")
    p.add_argument("--set-camera", metavar="DEVICE",
                   help="切换摄像头设备，例如 /dev/video1 或 0")
    p.add_argument("--start-service", action="store_true",
                   help="启动后台服务")
    p.add_argument("--stop-service", action="store_true",
                   help="停止后台服务（不会检测/锁屏/提醒）")
    p.add_argument("--restart-service", action="store_true",
                   help="重启后台服务")
    p.add_argument("--clean-cache", action="store_true",
                   help="清理旧的调试图片并退出")
    p.add_argument("--setup", action="store_true",
                   help="为当前用户生成配置/desktop/自启动并启用服务（deb 安装后执行一次）")
    p.add_argument("--no-start", action="store_true",
                   help="配合 --setup：只生成文件，不启动服务")
    p.add_argument("--pause", type=int, metavar="SECONDS",
                   help="暂停检测（例如 1800 = 30 分钟）")
    p.add_argument("--resume", action="store_true", help="恢复检测")
    p.add_argument("--status", action="store_true", help="显示 D-Bus/模型/配置状态")
    p.add_argument("--self-test", action="store_true",
                   help="不访问摄像头，运行内置自检")
    p.add_argument("--print-config", action="store_true", help="打印最终配置后退出")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--version", action="version", version=f"presence-guard {__version__}")
    return p


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _apply_nice(level: int) -> None:
    """Lower CPU scheduling priority without requiring privileges."""
    try:
        current = os.nice(0)
        if current < level:
            os.nice(level - current)
    except (AttributeError, OSError):
        pass


def _apply_nice(level: int) -> None:
    """Lower CPU scheduling priority without requiring privileges."""
    try:
        current = os.nice(0)
        if current < level:
            os.nice(level - current)
    except (AttributeError, OSError):
        pass


def _apply_overrides(cfg, args) -> None:
    if args.camera:
        cfg.camera.device = args.camera
    if args.no_badge:
        cfg.badge.enabled = False
    if args.no_idle_gate:
        cfg.scheduler.idle_gate_seconds = 0
    if args.badge_mode:
        cfg.badge.mode = args.badge_mode
    if args.presence_interval is not None:
        cfg.scheduler.presence_interval_seconds = args.presence_interval
    if args.badge_interval is not None:
        cfg.badge.interval_seconds = args.badge_interval
    if args.absence_seconds is not None:
        cfg.scheduler.absence_confirm_seconds = args.absence_seconds
    if args.log_level:
        cfg.general.log_level = args.log_level
    validate_config(cfg)


def _acquire_pid_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = state_dir / "guard.lock"
    fh = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        pid = fh.read().strip() or "?"
        fh.close()
        log.error("another presence-guard is already running (pid %s)", pid)
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _run_setup(start: bool) -> int:
    """Create per-user config / desktop entry / autostart and enable the service."""

    base = Path("/usr/share/presence-guard")
    unit_src = Path("/usr/lib/systemd/user/presence-guard.service")
    desktop_src = Path("/usr/share/applications/presence-guard-ui.desktop")
    if not (base / "app").is_dir():
        print("--setup 仅适用于通过 deb 安装；源码安装请运行 scripts/install.sh",
              file=sys.stderr)
        return 2

    config_dir = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    data_dir = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()

    cfg_dir = config_dir / "presence-guard"
    cfg_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    cfg_file = cfg_dir / "config.toml"
    if not cfg_file.exists() and (base / "app" / "config.example.toml").exists():
        shutil.copy(base / "app" / "config.example.toml", cfg_file)
        print(f"已生成配置: {cfg_file}")
    else:
        print(f"配置: {cfg_file} {'（已存在）' if cfg_file.exists() else '（未生成）'}")

    if unit_src.exists():
        unit_dir = config_dir / "systemd" / "user"
        unit_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(unit_src, unit_dir / "presence-guard.service")
        print(f"已安装服务单元: {unit_dir / 'presence-guard.service'}")

    if desktop_src.exists():
        apps = data_dir / "applications"
        apps.mkdir(parents=True, exist_ok=True)
        target = apps / "presence-guard-ui.desktop"
        shutil.copy(desktop_src, target)
        os.chmod(target, 0o644)
        autostart = config_dir / "autostart"
        autostart.mkdir(parents=True, exist_ok=True)
        shutil.copy(desktop_src, autostart / "presence-guard-ui.desktop")
        os.chmod(autostart / "presence-guard-ui.desktop", 0o644)
        print("已生成应用菜单项与登录自启动: 在岗守护")

    if not start:
        print("\n未启动服务（--no-start）。手动启动：")
        print("  systemctl --user daemon-reload")
        print("  systemctl --user enable --now presence-guard")
        return 0

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"],
                       capture_output=True, text=True, timeout=30, check=False)
        result = subprocess.run(
            ["systemctl", "--user", "enable", "--now", "presence-guard"],
            capture_output=True, text=True, timeout=30, check=False)
        if result.returncode == 0:
            print("服务已启动: systemctl --user status presence-guard")
        else:
            print(f"服务启动失败: {result.stderr.strip() or result.stdout.strip()}",
                  file=sys.stderr)
            return 3
    except Exception as exc:
        print(f"无法操作 systemd: {exc}", file=sys.stderr)
        return 3
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    try:
        _apply_overrides(cfg, args)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    _setup_logging(cfg.general.log_level)
    _apply_nice(cfg.general.nice)

    if args.setup:
        return _run_setup(start=not args.no_start)

    if args.print_config:
        _print_json(cfg.to_dict())
        return 0

    actions = DesktopActions(
        dry_run=args.dry_run,
        lock_enabled=(not args.no_lock) and cfg.scheduler.lock_enabled,
    )

    if args.status:
        next_badge = None
        badge_windows = []
        if cfg.badge.enabled:
            from .schedule import BadgeSchedule
            try:
                schedule = BadgeSchedule(cfg.badge)
                now_dt = __import__("datetime").datetime.now()
                next_badge = schedule.next_check_at(now_dt).strftime(
                    "%Y-%m-%d %H:%M:%S")
                badge_windows = cfg.badge.windows
            except ValueError as exc:
                next_badge = f"invalid schedule: {exc}"
        model_paths = {
            "nanodet": str(cfg.resolve_model_path(cfg.detector.nanodet_model)),
            "yunet": str(cfg.resolve_model_path(cfg.detector.yunet_model)),
        }
        data = {
            "version": __version__,
            "config": cfg.config_path,
            "state_dir": cfg.general.state_dir,
            "camera": cfg.camera.device,
            "intervals": {
                "presence_seconds": cfg.scheduler.presence_interval_seconds,
                "absence_confirm_seconds": cfg.scheduler.absence_confirm_seconds,
                "badge_seconds": cfg.badge.interval_seconds,
            },
            "badge_mode": cfg.badge.mode,
            "badge_windows": badge_windows,
            "badge_workdays": cfg.badge.workdays,
            "next_badge_check": next_badge,
            "models": {k: {"path": v, "exists": Path(v).exists()}
                       for k, v in model_paths.items()},
            "dbus": actions.status(),
            "control": ControlClient(cfg.general.state_dir).send("ping"),
            "live": StatusStore.read(cfg.general.state_dir),
        }
        _print_json(data)
        return 0

    if args.gui:
        from .gui import run_gui
        return run_gui(cfg)

    if args.clean_cache:
        from .cache import cleanup_dir
        removed, freed = cleanup_dir(
            cfg.general.debug_dir,
            cfg.general.debug_keep_files,
            cfg.general.debug_max_age_days,
        )
        _print_json({"removed": removed, "freed_bytes": freed})
        return 0

    if args.start_service or args.stop_service or args.restart_service:
        action = ("start" if args.start_service
                  else "stop" if args.stop_service else "restart")
        try:
            result = subprocess.run(
                ["systemctl", "--user", action, "presence-guard"],
                capture_output=True, text=True, timeout=30, check=False)
            _print_json({
                "ok": result.returncode == 0,
                "action": action,
                "detail": (result.stdout or result.stderr).strip(),
            })
            return 0 if result.returncode == 0 else 3
        except Exception as exc:
            _print_json({"ok": False, "action": action, "error": str(exc)})
            return 3

    if (args.check_now or args.check_badge_now or args.retry_camera
            or args.set_camera or args.pause or args.resume):
        client = ControlClient(cfg.general.state_dir)
        if args.check_now:
            result = client.send("check_now")
        elif args.set_camera:
            result = client.send("set_camera", device=args.set_camera)
        elif args.check_badge_now:
            result = client.send("check_badge")
        elif args.retry_camera:
            result = client.send("retry_camera")
        elif args.pause:
            result = client.send("pause", seconds=args.pause)
        else:
            result = client.send("resume")
        _print_json(result)
        return 0 if result.get("ok") else 3

    if args.notify_test:
        notification_id = actions.notify(
            "presence-guard 测试",
            "通知通道正常。正式运行时，工牌检查会使用该通道提醒。",
        )
        _print_json({"notification_id": notification_id})
        return 0 if notification_id is not None else 3

    if args.self_test:
        from .selftest import run_self_test
        return run_self_test(cfg)

    try:
        vision = VisionPipeline(cfg)
    except Exception as exc:
        log.error("模型加载失败: %s", exc)
        return 2

    state = StateStore(cfg.general.state_dir)

    if args.once or args.image:
        debug_dir = None
        if args.debug_dir:
            debug_dir = Path(args.debug_dir).expanduser()
        elif args.save_debug:
            debug_dir = Path(cfg.general.debug_dir).expanduser()
        camera = CameraCapture(cfg.camera)
        engine = Engine(cfg, camera, vision, actions, state)
        report, _ = engine.inspect(
            image_path=args.image,
            force_badge=args.check_badge,
            debug_dir=debug_dir,
        )
        _print_json(report.to_dict())
        return 0 if report.error is None else 2

    lock_fh = _acquire_pid_lock(Path(cfg.general.state_dir))
    if lock_fh is None:
        return 4

    camera = CameraCapture(cfg.camera)
    engine = Engine(cfg, camera, vision, actions, state)

    def _stop(_signum, _frame):
        log.info("signal received; stopping")
        engine.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    try:
        engine.run_forever()
    finally:
        # Keep the lock file: flock is released automatically on exit, and the
        # stale PID is useful in logs.
        try:
            lock_fh.close()
        except Exception:
            pass
    return 0
