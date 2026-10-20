#!/usr/bin/env bash
# Build presence-guard_<VERSION>_amd64.deb.
# The Python wheels are downloaded at build time so the deb installs offline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-0.1.0}"
BUILD="$ROOT/build/deb"
DIST="$ROOT/dist"
PIP_INDEX="${PRESENCE_GUARD_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

command -v dpkg-deb >/dev/null || { echo "需要 dpkg-deb"; exit 1; }
command -v python3 >/dev/null || { echo "需要 python3"; exit 1; }

echo "==> 清理构建目录"
rm -rf "$BUILD"
mkdir -p "$BUILD/root/usr/share/presence-guard/app" "$DIST"

echo "==> 复制应用文件"
tar -C "$ROOT" \
    --exclude='.venv' --exclude='.build-venv' --exclude='.git' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='tests' \
    --exclude='packaging' --exclude='build' --exclude='dist' \
    -cf - . | tar -C "$BUILD/root/usr/share/presence-guard/app" -xf -

echo "==> 下载依赖 wheel（离线安装用）"
WHEELS="$BUILD/root/usr/share/presence-guard/wheels"
mkdir -p "$WHEELS"
python3 -m pip download --disable-pip-version-check --no-deps --only-binary=:all: \
    -i "$PIP_INDEX" -d "$WHEELS" \
    opencv-python-headless==4.10.0.84 tomli==2.0.1 chinesecalendar==1.11.0 \
    pip==23.3.2
# numpy is ABI-specific: bundle wheels for the Python versions UOS may ship.
python3 -m pip download --disable-pip-version-check --no-deps --only-binary=:all: \
    --python-version 37 -i "$PIP_INDEX" -d "$WHEELS" numpy==1.21.6
python3 -m pip download --disable-pip-version-check --no-deps --only-binary=:all: \
    --python-version 38 -i "$PIP_INDEX" -d "$WHEELS" numpy==1.24.4
python3 -m pip download --disable-pip-version-check --no-deps --only-binary=:all: \
    --python-version 39 -i "$PIP_INDEX" -d "$WHEELS" numpy==1.24.4
python3 -m pip download --disable-pip-version-check --no-deps --only-binary=:all: \
    --python-version 310 -i "$PIP_INDEX" -d "$WHEELS" numpy==1.26.4
ls -lh "$WHEELS"

echo "==> 组装 systemd / desktop / icon"
mkdir -p "$BUILD/root/usr/lib/systemd/user" "$BUILD/root/usr/share/applications" "$BUILD/root/usr/bin"
cp "$ROOT/packaging/root/usr/lib/systemd/user/presence-guard.service" \
   "$BUILD/root/usr/lib/systemd/user/presence-guard.service"
cp "$ROOT/packaging/root/usr/share/applications/presence-guard-ui.desktop" \
   "$BUILD/root/usr/share/applications/presence-guard-ui.desktop"
mkdir -p "$BUILD/root/usr/share/icons/hicolor/scalable/apps"
cp "$ROOT/assets/presence-guard.svg" \
   "$BUILD/root/usr/share/icons/hicolor/scalable/apps/presence-guard.svg"
cp "$ROOT/packaging/root/usr/bin/presence-guard" "$BUILD/root/usr/bin/presence-guard"

mkdir -p "$BUILD/root/DEBIAN"
cp "$ROOT/packaging/DEBIAN/control" "$BUILD/root/DEBIAN/control"
cp "$ROOT/packaging/DEBIAN/postinst" "$BUILD/root/DEBIAN/postinst"
cp "$ROOT/packaging/DEBIAN/prerm" "$BUILD/root/DEBIAN/prerm"
chmod 0755 "$BUILD/root/DEBIAN/postinst" "$BUILD/root/DEBIAN/prerm"
# The control file is copied; keep its version in sync with VERSION.
sed -i "s/^Version: .*/Version: ${VERSION}/" "$BUILD/root/DEBIAN/control"

echo "==> 构建 deb"
dpkg-deb --build --root-owner-group "$BUILD/root" \
    "$DIST/presence-guard_${VERSION}_amd64.deb"
ls -lh "$DIST"
echo "完成: $DIST/presence-guard_${VERSION}_amd64.deb"
