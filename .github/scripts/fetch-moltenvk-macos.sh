#!/usr/bin/env bash
# The output path MUST match TORCHVULKAN_MOLTENVK in [tool.cibuildwheel.macos]
# environment in pyproject.toml. The dylib ships inside the macOS wheel, so the
# download is pinned by SHA-256; update the hash with the version.
set -euo pipefail

MOLTENVK_VERSION="1.4.2"
MOLTENVK_SHA256="f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e"
OUT_DIR="/tmp/moltenvk"

mkdir -p "$OUT_DIR"
curl -sSLf -o "$OUT_DIR/MoltenVK-macos.tar" \
  "https://github.com/KhronosGroup/MoltenVK/releases/download/v${MOLTENVK_VERSION}/MoltenVK-macos.tar"
echo "${MOLTENVK_SHA256}  $OUT_DIR/MoltenVK-macos.tar" | shasum -a 256 -c -
tar -xf "$OUT_DIR/MoltenVK-macos.tar" -C "$OUT_DIR"
lipo -thin "$(uname -m)" "$OUT_DIR/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib" -output "$OUT_DIR/libMoltenVK.dylib"

test -f "$OUT_DIR/libMoltenVK.dylib"
