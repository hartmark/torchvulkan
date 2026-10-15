#!/usr/bin/env bash
# Installs the torchvulkan build toolchain inside the cibuildwheel manylinux
# container (AlmaLinux 9 / manylinux_2_34): the Vulkan SDK (headers + loader)
# and the Slang compiler (slangc). Both downloads are pinned by SHA-256; update
# the hashes with the versions.
#
# The install paths here MUST match the VULKAN_SDK / SLANG_DIR / PATH entries in
# [tool.cibuildwheel.linux].environment in pyproject.toml.
set -euo pipefail

SLANG_VERSION="2026.8.1"
SLANG_SHA256="0d018272522f6d2a75e2f92a30e8eddef18b92e80ab3d29575ac3b0136838aac"
VULKAN_SDK_VERSION="1.3.296.0"
VULKAN_SDK_SHA256="79b0a1593dadc46180526250836f3e53688a9a5fb42a0e5859eb72316dc4d53e"

dnf install -y wget tar xz unzip

wget -q -O /tmp/vulkansdk.tar.xz \
  "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
echo "${VULKAN_SDK_SHA256}  /tmp/vulkansdk.tar.xz" | sha256sum -c -
mkdir -p /opt/vulkan
tar -xf /tmp/vulkansdk.tar.xz -C /opt/vulkan --strip-components=1

wget -q -O /tmp/slang.zip \
  "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.zip"
echo "${SLANG_SHA256}  /tmp/slang.zip" | sha256sum -c -
mkdir -p /opt/slang
unzip -q /tmp/slang.zip -d /opt/slang

test -f /opt/vulkan/x86_64/include/vulkan/vulkan.h
test -x /opt/slang/bin/slangc
