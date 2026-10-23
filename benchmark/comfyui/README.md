# ComfyUI benchmark: torchvulkan vs ROCm

Reproducible comparison of [torchvulkan](https://github.com/zainsharief/torchvulkan) (PyTorch
Vulkan backend) and PyTorch ROCm on AMD GPUs, running the same ComfyUI text-to-image workflow
(Z-Image Turbo) in two containers that differ only in the torch backend.

> **Disclaimer:** I used AI to generate the tests and scripts, but the interpretation of the
> results is my own.

## What it does

`bench.sh`:

1. Downloads and sha256-verifies the models into `./models` (≈10.5 GB, skipped if present):
   - `z_image_turbo-Q6_K.gguf` — [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF)
   - `Qwen3-4B-Instruct-2507-Q8_0.gguf` (text encoder) — [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF)
   - `ae.safetensors` (VAE) — [Comfy-Org/Z-Image-Turbo](https://huggingface.co/Comfy-Org/Z-Image-Turbo)
2. Builds two small images from one `Containerfile` (Arch Linux, Python 3.14, same ComfyUI and
   gguf node commits; the vulkan one adds Mesa RADV from Arch).
3. Creates one Python venv per backend in `./venvs/<backend>` (bind-mounted at `/venv`, created by
   the image's `entry.sh setup` on first use, ROCm ≈15 GB, Vulkan ≈2.5 GB):
   - rocm: `torch==2.13.0+rocm7.1` from download.pytorch.org
   - vulkan: `torch==2.13.0+cpu` + `torchvulkan==0.2.0`

   Keeping torch out of the image avoids committing a 15 GB layer, which needs about twice that
   much free disk during `podman build`.
4. Runs each container with the same bind mounts (`models` and the venv read-only, results dir) and
   generates N images (1024², 8 steps, `res_multistep`, one prompt, seeds 1000…). The first
   image is the cold start (model loading); the rest are "warm".
5. Samples the host once per second: GPU busy %, VRAM, GTT, GPU edge/junction/memory
   temperature, GPU power and clock, host CPU %, CPU % of the ComfyUI process, CPU temperature,
   process RSS, MemAvailable, swap.
6. Writes `results/<host>-<time>/results.md` (+ `results.json`, per-backend `monitor.csv`,
   `run.json`, ComfyUI log and the first image of each backend).

```bash
./bench.sh -n 10               # 10 images per backend
./bench.sh -n 2 -b vulkan      # quick check, one backend
./bench.sh -h
```

Requirements: Linux with an amdgpu GPU, rootless or rootful podman, `/dev/kfd` + `/dev/dri`
access (user in the `render`/`video` groups), python3, curl, lspci, ~30 GB of disk (models +
venvs). Stop other GPU users first; the script warns when VRAM is already in use.

On a 16 GB card that also drives a desktop (~2 GB VRAM), the Vulkan run went out of memory on the
4th image at 1024² (torchvulkan's allocator hit the Vulkan memory budget while ComfyUI, which only
sees amdgpu's VRAM counters, still thought it had room). `-a "--disable-dynamic-vram
--disable-pinned-memory --reserve-vram 1.5"` fixes that; use the same flags for both backends.

## Results (2026-10-23)

1024², 8 steps, 10 images per backend, `--disable-dynamic-vram --disable-pinned-memory` (+ `--reserve-vram`
where noted). Both hosts have a 256 MB BAR (no Resizable BAR). Full tables, per-second
`monitor.csv` and ComfyUI logs are in [`results/`](results/).

| | RX 9060 XT (gfx1200), headless | | RX 7800 XT (gfx1101), KDE desktop running (~2.1 GB VRAM) | |
|---|---|---|---|---|
| | **ROCm 7.1** | **Vulkan (torchvulkan 0.2.0)** | **ROCm 7.1** | **Vulkan (torchvulkan 0.2.0)** |
| images OK | 10/10 | 10/10 | 10/10 | 2/10, then OOM ¹ |
| first image (cold) | 45.4 s | 195.2 s | 33.8 s | 101.0 s |
| warm image (median) | 18.1 s | 131.5 s | 21.0 s | 65.6 s (1 image) |
| sampling | 1.75 s/it | 14.75 s/it | 2.25 s/it | 5.12 s/it |
| GPU busy (avg) | 81 % | 89 % | 90 % | 54 % |
| GPU power avg / max | 119 / 161 W | 98 / 118 W | 195 / 259 W | 131 / 299 W |
| GPU junction temp max | 81 °C | 72 °C | 89 °C | 87 °C |
| VRAM peak (whole GPU) | 12.7 GB | 13.9 GB | 14.1 GB | 15.4 GB |
| ComfyUI CPU (avg, 100 % = 1 core) | 196 % | 15 % | 201 % | 59 % |
| process RSS peak | 13.5 GB | 10.4 GB | 17.1 GB | 13.6 GB |

RX 9060 XT: `results/kozel-rx9060xt`. RX 7800 XT: `results/bernard-rx7800xt-reserve7` (`--reserve-vram 7`).

¹ On the RX 7800 XT the Vulkan run went out of memory in all three attempts (no reserve: image 4,
`--reserve-vram 1.5`: image 2, `--reserve-vram 7`: image 3), always with torchvulkan reporting
~7.0–7.3 GiB in use against a ~7.2–7.3 GiB limit (`Tried to allocate … with 7516192768 of 7635114188
bytes in use`) while amdgpu reported ~15.4 GB VRAM in use on the whole GPU. With nothing else
running, the same container could allocate 12.75 GiB in 256 MiB blocks before hitting its limit.
The ROCm runs on the same machine never ran out of memory. The only CPU fallback logged was
`aten::add.out`.

### The RX 7800 XT out-of-memory errors: stale Vulkan memory budget

`repro_oom.py` and `repro_budget_lag.py` reproduce it without ComfyUI. torchvulkan refuses a
device allocation when `usage + size > 0.95 × budget`, with both values from VMA's
`vmaGetHeapBudgets` (`VK_EXT_memory_budget`). VMA only re-reads the driver's budget after ~30
`vkAllocateMemory`/`vkFreeMemory` calls, and refused allocations never reach VMA, so a budget sampled
at a bad moment (e.g. right after torchvulkan released its buffer cache, while the kernel still
counted the freed memory) stays in effect:

```
resident 4.00 GiB; usage/budget (4.125, 12.99)
cache filled with 31 buffers (6.2 GiB); usage/budget (4.125, 9.25)
t+ 4.00s after empty_cache: usage  4.12  budget(x0.95)  9.25  gpu_vram  6.20   <- memory is free, budget isn't
after 32 extra alloc/free ops: usage  4.12  budget(x0.95) 12.99                <- VMA refetched
```

The fix (branch [`fix/refresh-memory-budget`](https://github.com/hartmark/torchvulkan/tree/fix/refresh-memory-budget),
22 lines in `src/vulkan/`) calls `vmaSetCurrentFrameIndex()` to refresh the budget before refusing
an allocation, and after `flushAndReleaseBuffers()` retries with a short backoff (≤ ~255 ms, only on
the out-of-memory path). Same RX 7800 XT with the KDE desktop running, 10 images, default flags:

| run | torchvulkan | images OK | warm image | sampling |
|---|---|---|---|---|
| `bernard-rx7800xt-noreserve-oom` | 0.2.0 | 3/10, OOM | 107 s | 10.75 s/it |
| `bernard-rx7800xt-budgetfix` | 0.2.0 + fix | **10/10** | 54.7 s | 5.0 s/it |
| `bernard-rx7800xt-budgetfix-ramload` (RAM hog, MemAvailable down to 6.5 GB) | 0.2.0 + fix | 10/10 | 57.2 s | 5.0 s/it |
| `bernard-rx7800xt-budgetfix-vramload` (2nd process holding 3 GiB VRAM) | 0.2.0 + fix | 10/10 | 108.9 s | 11.1 s/it |
| `bernard-rx7800xt-rocm-ramload` (RAM hog ²) | ROCm 7.1 | 10/10 | 20.2 s | 2.25 s/it |

² the RAM hog only touches one page per MiB, so the kernel had swapped most of it out by the ROCm
run (MemAvailable min 26 GB); treat that row as "without RAM pressure".

With the fix the budget follows other processes: a 4 GiB VRAM hog lowers it from 12.99 to 9.19 GiB
and it returns to 12.99 when the hog exits (`repro_oom.py --step-sleep`). Without it, the same
reproducer fails at step 2 (budget stuck at 7.7 GiB). A related observation: torchvulkan's buffer
cache keeps every freed buffer until an allocation fails, so a running torchvulkan process holds
VRAM up to its budget and a second GPU process that starts later can run out of memory.

All four first images (same seed) are visually identical:
![first images](results/first-images-kozel-rocm-vulkan-bernard-rocm-vulkan.png)
(RX 9060 XT ROCm, RX 9060 XT Vulkan, RX 7800 XT ROCm, RX 7800 XT Vulkan)

## ComfyUI patch

ComfyUI has no notion of a `vulkan` device, so `comfyui-vulkan.patch` adds a `--vulkan` flag
(~50 lines in `comfy/model_management.py` + `cli_args.py`): device `vulkan:0`, total/free VRAM
from amdgpu sysfs (torchvulkan only reports its own allocations and has no `mem_get_info`),
fp16 on, bf16 if `torchvulkan.is_bf16_supported()`, synchronize/empty_cache. The patch is applied
in both images; without `--vulkan` it changes nothing, so the ROCm run uses stock ComfyUI code paths.

Both backends run with `--disable-dynamic-vram --disable-pinned-memory` by default:

- dynamic VRAM (comfy-aimdo) is not available for the Vulkan device, so it is off for ROCm too;
- with pinned host memory, ROCm 7.x/TheRock crashed on gfx1200 with an SDMA page fault while
  ComfyUI offloaded weights (`Memory access fault by GPU ... Page not present`), see
  ROCm/rocm-systems#12528 for the pinned-transfer issue family.

## Files

| file | |
|---|---|
| `bench.sh` | entry point |
| `entry.sh` | container entry point: `setup` creates the venv, otherwise runs `run_workflow.py` |
| `Containerfile` | `--build-arg BACKEND=rocm\|vulkan` |
| `comfyui-vulkan.patch` | ComfyUI `--vulkan` device support |
| `run_workflow.py` | runs inside the container: starts ComfyUI, queues the prompts, times them |
| `monitor.py` | host-side sampler |
| `summarize.py` | builds `results.md` / `results.json` |

`comfyui-vulkan.patch` is a patch against ComfyUI and is GPL-3.0 like ComfyUI; everything else follows this repository's license.
