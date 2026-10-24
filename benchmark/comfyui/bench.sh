#!/usr/bin/env bash
# torchvulkan vs ROCm in ComfyUI: download the models, build one ROCm and one Vulkan container
# (same torch 2.13.0, same ComfyUI + gguf node, same bind mounts) and generate N images on each
# while sampling GPU/CPU/memory/temperature on the host.
set -euo pipefail

usage() {
  cat <<EOF
usage: $0 [-n RUNS] [-b rocm,vulkan] [-m MODELS_DIR] [-V VENVS_DIR] [-o OUT_DIR] [-r RES] [-s STEPS] [-a "COMFY ARGS"] [--no-build] [--new-venv]
  -n RUNS        images per backend (default 10; the first one is the cold start)
  -b BACKENDS    comma-separated, default rocm,vulkan
  -m MODELS_DIR  where models are downloaded/verified (default ./models)
  -V VENVS_DIR   bind-mounted Python venvs with torch, one per backend (default ./venvs;
                 ROCm needs ~15 GB, Vulkan ~2 GB; created on first use)
  -o OUT_DIR     results (default ./results/<hostname>-<timestamp>)
  -r RES         image width = height (default 1024)
  -s STEPS       sampler steps (default 8)
  -a ARGS        extra ComfyUI args for both backends
                 (default "--disable-dynamic-vram --disable-pinned-memory")
  --no-build     reuse existing localhost/tvbench-<backend> images
  --new-venv     recreate the venvs even if they exist
  --stats        debug builds: sample torchvulkan.dispatch_stats() inside ComfyUI (<backend>/tvstats.jsonl)
  --warn-always  log every torchvulkan CPU fallback (torch.set_warn_always(True)) and count them
Stop other GPU users first; the script warns if VRAM is already in use.
EOF
}

HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=10 BACKENDS=rocm,vulkan MODELS=$HERE/models VENVS=$HERE/venvs OUT= RES=1024 STEPS=8 BUILD=1 NEWVENV=0 WARNALWAYS= STATS=
COMFY_ARGS="--disable-dynamic-vram --disable-pinned-memory"
while [ $# -gt 0 ]; do
  case $1 in
    -n) RUNS=$2; shift 2 ;;
    -b) BACKENDS=$2; shift 2 ;;
    -m) MODELS=$2; shift 2 ;;
    -V) VENVS=$2; shift 2 ;;
    --new-venv) NEWVENV=1; shift ;;
    --warn-always) WARNALWAYS=--warn-always; shift ;;
    --stats) STATS=--stats; shift ;;
    -o) OUT=$2; shift 2 ;;
    -r) RES=$2; shift 2 ;;
    -s) STEPS=$2; shift 2 ;;
    -a) COMFY_ARGS=$2; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
HOST=$(cat /etc/hostname 2>/dev/null || uname -n)
OUT=${OUT:-$HERE/results/$HOST-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$MODELS" "$VENVS" "$OUT"
MODELS=$(cd "$MODELS" && pwd) VENVS=$(cd "$VENVS" && pwd) OUT=$(cd "$OUT" && pwd)

# name | url | sha256
MODEL_LIST="
z_image_turbo-Q6_K.gguf|https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q6_K.gguf|fc137d87b49e06fdd5230d67d6c8cfa42a9e1fd38b65ccd355882450c3eb1c82
Qwen3-4B-Instruct-2507-Q8_0.gguf|https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf|391c1e410fd9f4cf2de2b510273b56a84c19ce18f4fa3bfb3774031dac4ef068
ae.safetensors|https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/vae/ae.safetensors|afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38
"

echo "== models in $MODELS"
for line in $MODEL_LIST; do
  IFS='|' read -r name url sha <<<"$line"
  f=$MODELS/$name ok=$MODELS/.$name.verified
  if [ -f "$f" ] && [ -f "$ok" ] && [ "$(cat "$ok")" = "$(stat -c '%s %Y' "$f")" ]; then
    echo "   $name: verified earlier"; continue
  fi
  if [ ! -f "$f" ] || ! echo "$sha  $f" | sha256sum -c --status; then
    echo "   $name: downloading"
    curl -fL --retry 5 -C - -o "$f.part" "$url"
    mv "$f.part" "$f"
    echo "$sha  $f" | sha256sum -c --status || { echo "   $name: sha256 mismatch" >&2; exit 1; }
  fi
  stat -c '%s %Y' "$f" > "$ok"
  echo "   $name: ok"
done

IFS=, read -ra BL <<<"$BACKENDS"
if [ "$BUILD" = 1 ]; then
  for b in "${BL[@]}"; do
    echo "== building localhost/tvbench-$b"
    podman build --build-arg BACKEND="$b" -t "localhost/tvbench-$b" -f "$HERE/Containerfile" "$HERE" \
      > "$OUT/build-$b.log" 2>&1 || { tail -30 "$OUT/build-$b.log"; exit 1; }
  done
fi

for b in "${BL[@]}"; do
  mkdir -p "$VENVS/$b"
  if [ "$NEWVENV" = 1 ] || [ ! -f "$VENVS/$b/versions.txt" ]; then
    echo "== setting up $VENVS/$b (torch for $b)"
    podman run --rm --security-opt label=disable -v "$VENVS/$b:/venv" "localhost/tvbench-$b" setup \
      > "$OUT/venv-$b.log" 2>&1 || { tail -30 "$OUT/venv-$b.log"; exit 1; }
  fi
done

python3 - "$OUT/host.json" "$RUNS" "$RES" "$STEPS" "$COMFY_ARGS" "$(podman --version | awk '{print $3}')" <<'EOF'
import glob, json, os, platform, subprocess, sys
out, runs, res, steps, args, podman = sys.argv[1:7]
best = None
for d in glob.glob('/sys/class/drm/card[0-9]*/device'):
    try:
        t = int(open(d + '/mem_info_vram_total').read())
    except OSError:
        continue
    if best is None or t > best[0]:
        best = (t, d)
slot = os.path.basename(os.path.realpath(best[1]))
gpu = subprocess.run(['lspci', '-s', slot], capture_output=True, text=True).stdout.split(': ', 1)[-1].strip()
cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor())
ram = round(int(open('/proc/meminfo').readline().split()[1]) / 1024 / 1024)
json.dump({'hostname': platform.node(), 'gpu': gpu or slot, 'vram_mb': best[0] >> 20, 'cpu': cpu, 'ram_gb': ram,
           'kernel': platform.release(), 'podman': podman, 'runs': int(runs), 'res': int(res),
           'steps': int(steps), 'comfy_args': args}, open(out, 'w'), indent=1)
EOF

CARD=$(for d in /sys/class/drm/card[0-9]*/device; do
         if [ -f "$d/mem_info_vram_total" ]; then echo "$(cat "$d/mem_info_vram_total") $d"; fi
       done | sort -n | tail -1 | cut -d' ' -f2)
for b in "${BL[@]}"; do
  used=$(( $(cat "$CARD/mem_info_vram_used") >> 20 ))
  if [ "$used" -gt 1024 ]; then echo "!! $used MiB VRAM already in use by something else; results will be skewed" >&2; fi
  mkdir -p "$OUT/$b"
  dev=(--device /dev/dri)
  if [ "$b" = rocm ]; then dev+=(--device /dev/kfd); fi
  echo "== $b: $RUNS images"
  podman run -d --replace --name "tvbench-$b" "${dev[@]}" --group-add keep-groups \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    ${TVBENCH_VULKAN_FREE:+-e TVBENCH_VULKAN_FREE=$TVBENCH_VULKAN_FREE} \
    ${TORCHVULKAN_GPU_PROFILE:+-e TORCHVULKAN_GPU_PROFILE=$TORCHVULKAN_GPU_PROFILE} \
    ${TORCHVULKAN_STRICT:+-e TORCHVULKAN_STRICT=$TORCHVULKAN_STRICT} \
    -v "$MODELS:/models:ro" -v "$VENVS/$b:/venv:ro" -v "$OUT/$b:/out" \
    "localhost/tvbench-$b" --backend "$b" --runs "$RUNS" --res "$RES" --steps "$STEPS" $WARNALWAYS $STATS -- $COMFY_ARGS >/dev/null
  pid=$(podman inspect -f '{{.State.Pid}}' "tvbench-$b")
  python3 "$HERE/monitor.py" "$pid" "$OUT/$b/monitor.csv" &
  mon=$!
  podman logs -f "tvbench-$b" 2>&1 | grep --line-buffered '^\[bench\]' || true
  rc=$(podman wait "tvbench-$b")
  wait "$mon" || true
  podman logs "tvbench-$b" > "$OUT/$b/container.log" 2>&1
  podman rm -f "tvbench-$b" >/dev/null
  echo "   $b exit $rc"
done

python3 "$HERE/summarize.py" "$OUT"
echo "== results: $OUT/results.md"
