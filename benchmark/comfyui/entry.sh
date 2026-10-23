#!/bin/bash
# entry.sh setup  -> create /venv (bind mount) with torch 2.13.0 for $BACKEND + ComfyUI requirements
# entry.sh ARGS   -> run_workflow.py ARGS with /venv's python
set -euo pipefail
TORCH_VERSION=${TORCH_VERSION:-2.13.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.28.0}
TORCHVULKAN_VERSION=${TORCHVULKAN_VERSION:-0.2.0}
ROCM_INDEX=${ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.1}
CPU_INDEX=${CPU_INDEX:-https://download.pytorch.org/whl/cpu}

if [ "${1:-}" = setup ]; then
  python -m venv --clear /venv
  export PATH=/venv/bin:$PATH
  if [ "$BACKEND" = rocm ]; then
    pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url $ROCM_INDEX
  else
    pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url $CPU_INDEX
    pip install torchvulkan==$TORCHVULKAN_VERSION
  fi
  grep -vE '^(torch|torchvision|torchaudio)([<>=~ ].*)?$' /ComfyUI/requirements.txt > /tmp/req.txt
  pip install -r /tmp/req.txt gguf
  python -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION'), torch.__version__"
  { cat /bench/image-versions.txt; pip list 2>/dev/null | grep -iE '^(torch|torchvision|torchvulkan|triton|comfy)'; } \
    > /venv/versions.txt
  cat /venv/versions.txt
  exit 0
fi

[ -f /venv/versions.txt ] || { echo "/venv is not set up; run: entry.sh setup" >&2; exit 1; }
exec /venv/bin/python /bench/run_workflow.py "$@"
