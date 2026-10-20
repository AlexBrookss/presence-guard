#!/usr/bin/env bash
# 从源码目录直接运行（开发/调试用）。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python -m presence_guard "$@"
