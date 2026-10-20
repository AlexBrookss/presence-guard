#!/usr/bin/env bash
# 停止并移除当前用户的 service（程序和配置保留，除非加 --purge）。
set -euo pipefail
PURGE=0
if [[ "${1:-}" == "--purge" ]]; then PURGE=1; fi

if command -v systemctl >/dev/null 2>&1; then
    systemctl --user disable --now presence-guard.service 2>/dev/null || true
fi
rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/presence-guard.service"
rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/applications/presence-guard-ui.desktop"
rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/autostart/presence-guard-ui.desktop"
rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps/presence-guard.svg"
if command -v systemctl >/dev/null 2>&1; then
    systemctl --user daemon-reload 2>/dev/null || true
fi

APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/presence-guard"
rm -rf "$APP_DIR"
rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/presence-guard/config.toml"
if [[ "$PURGE" == "1" ]]; then
    rm -rf "${XDG_CONFIG_HOME:-$HOME/.config}/presence-guard"
    rm -rf "${XDG_STATE_HOME:-$HOME/.local/state}/presence-guard"
    rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/presence-guard"
fi
echo "已卸载。"
