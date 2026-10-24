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

### Testing #34 (bypass the budget after releasing the cache), 2026-10-24

Built from #34 (6288f04, includes `main` 189ac37 with the new `.out` operators) and, for
comparison, `main` without a fix and #33 rebased onto the same `main`. RX 7800 XT, KDE desktop running.

Reproducers (same as above):

| | `main` (no fix) | #34 | #33 |
|---|---|---|---|
| budget 4 s after `empty_cache()` (×0.95) | 9.20 GiB (stale) | 9.21 GiB (stale) | 12.95 GiB |
| `repro_oom.py --weights-gb 7 --act-mb 64,512 --upload` | OOM at step 10 | 30/30 | 30/30 |
| same with a 4 GiB VRAM hog started first (80 steps) | OOM at step 2 | 80/80 | 80/80 |

ComfyUI, 10 images at 1024², default flags, `--warn-always` (`torch.set_warn_always(True)`, so every
torchvulkan CPU fallback is logged instead of only the first):

| run | torchvulkan | images OK | warm image (median) | sampling | CPU fallbacks |
|---|---|---|---|---|---|
| `bernard-rx7800xt-pr34` | `main` + #34 | 10/10 | 83.2 s | 9.5 s/it | none |
| `bernard-rx7800xt-pr34-vramload` (2nd process holding 3 GiB VRAM) | `main` + #34 | 10/10 | 95.8 s | 10.9 s/it | none |
| `bernard-rx7800xt-fix2-on-main` | `main` + #33 | 10/10 | 94.7 s | 10.9 s/it | none |
| `abab-r1-fix` (5 images) | 0.2.0 + #33 | 5/5 | 96.6 s | 9.5 s/it | `aten::silu.out` × 140, `aten::add.out` × 36 |

- Both fixes avoid the out-of-memory errors in all runs, also with another process holding VRAM.
- No operation fell back to the CPU on `main`; with 0.2.0 the same workflow logged `aten::silu.out`
  and `aten::add.out` every step (only the first one is shown without `--warn-always`).
- All runs from this night were much slower than the afternoon run of the same 0.2.0 + #33 build
  (`bernard-rx7800xt-budgetfix`: 54.7 s per image, 5.0 s/it), including that build itself
  (`abab-r1-fix`: 96.6 s, 9.5 s/it). So the night numbers do not show a speed difference between #34,
  #33 and `main`; the cause of the slower night runs is not known yet. An interleaved A/B/C run was
  stopped after its first part, to be repeated.

### Resizable BAR on the RX 7800 XT, 2026-10-24

After enabling Above 4G decoding + Resizable BAR in the BIOS: BAR 0 = 16 GB (`BAR=16384M`, all VRAM
CPU-visible, was 256 MB). Same builds and flags as `bernard-rx7800xt-budgetfix` (ROCm 7.1 and
torchvulkan 0.2.0 + #33), 10 images, KDE desktop and Steam running:

| | 256 MB BAR | Resizable BAR (16 GB) |
|---|---|---|
| ROCm warm image (median) | 21.0–22.1 s ¹ | **19.5 s** |
| ROCm first image | 33.8–41.9 s | 34.4 s |
| ROCm sampling | 2.25 s/it | 2.25 s/it |
| Vulkan warm image (median) | 54.7 s (afternoon) / 96.6 s (night) | 91.2 s |
| Vulkan sampling | 5.0 s/it (afternoon) / 9.5 s/it (night) | 9.6 s/it |
| Vulkan + `--reserve-vram 2.5` (5 images) | – | 70.7–79.7 s (image 2: 61.6 s) |

¹ `bernard-rx7800xt-reserve7` and `bernard-rx7800xt-noreserve-oom`.

- ROCm got ~7 % faster per warm image; sampling itself did not change, so the gain is outside it.
- Vulkan: the slowdown seen in the night runs is still there after the reboot, with identical
  software (kernel 7.2.11, Mesa 26.2.5, same image, same torchvulkan build and flags), so this run
  cannot show a Resizable BAR effect for Vulkan. In the slow runs the GPU draws less power (110–180 W
  instead of ~220 W) at a higher shader clock while ~2.6 GB more GTT is in use during sampling; with
  `--reserve-vram 2.5` the 2nd image was fast again (GTT 0.8 GB, 46 s sampling) but images 3–5 were
  not. The cause is not known yet. `monitor.py` now also logs the memory and fabric clocks
  (`gpu_mclk_mhz`, `gpu_fclk_mhz`) for the next runs.

### Minimal reproducer and per-kernel GPU profile, 2026-10-24

[`debug/profile-0.2.1`](https://github.com/hartmark/torchvulkan/tree/debug/profile-0.2.1) is 0.2.1 plus
the counters and a per-op GPU profiler: with `TORCHVULKAN_GPU_PROFILE=1` every op gets a timestamp
before and after it (with a barrier, so ops are serialised) and `torchvulkan.gpu_profile()` returns
`{"kernel:dtype": (count, total_ms)}`.

ComfyUI (`results/prof-gpu`, 6 images): the op mix per sampling step is the same in fast and slow
images, but some kernels get much slower — `reduce_subgroup:Float` 547 → 9541 µs, `cast:Half`
455 → 4792 µs, `binary:Float` 153 → 986 µs, `integer_binary:Byte` 275 → 753 µs, matmul 7.2 → 9.7 ms —
while others (`binary:Half`, `unary:Half`, `reduce_subgroup:Half`, `copy:UInt16`) don't change.

`repro_evict.py` reproduces it in ~30 s without ComfyUI: 5 GiB that stay allocated (a text
encoder), 5.5 GiB of uint8 "weights" read every step through a cast to fp16 (like GGUF
dequantization) and reloaded from the CPU every 4 steps (like ComfyUI reloading the model per
image), plus differently sized temporaries:

```
python repro_evict.py --resident-gb 5 --weights-gb 5.5 --steps 16 --reload-every 4 --temp-max-mb 800
```

On the RX 7800 XT with the desktop holding ~2.4 GB, 0.2.1 gives (same every run):

```
step  6:  0.14 s (weight reads  0.09 s) | binary:Half  38 ms/88 ( 437 us)  fill:Half  29 ms/7 (4120 us)
step  7:  0.48 s (weight reads  0.47 s) | binary:Half 229 ms/88 (2607 us)  copy     212 ms/95 (2227 us)
...steps 7-14 stay slow...
step 15:  0.11 s (weight reads  0.07 s) | binary:Half  37 ms/88 ( 425 us)  copy      19 ms/95 ( 200 us)
```

Every step releases the whole buffer cache 1–5 times because the budget is reached, and after each
release the allocation bypasses the budget (#34). Once the device is physically overcommitted the
kernel evicts buffers to system memory, and every kernel that touches one of them — the weights,
but also recycled cache buffers such as the accumulator — runs 6–11× slower. With #33 (budget
respected, no bypass) the same script raises `OutOfMemoryError` at step 0 instead.

### torchvulkan 0.2.1, 2026-10-24

0.2.1 (= `main` 078dd8a: #34 merged, `mem_get_info()` refreshes the budget, the new `.out` ops).
RX 7800 XT with Resizable BAR, KDE desktop running, 10 images, `--warn-always`. The ComfyUI patch
now sizes its offloading from `torchvulkan.mem_get_info()` when available
(`TVBENCH_VULKAN_FREE=sysfs` restores the old amdgpu-sysfs estimate):

| | ROCm 7.1 | Vulkan 0.2.1, `mem_get_info` | Vulkan 0.2.1, sysfs |
|---|---|---|---|
| images OK | 10/10 | 10/10 | 10/10 |
| first image | 30.7 s | 83.4 s | 78.9 s |
| warm image median | 19.7 s | 80.7 s | 77.2 s |
| warm image min / max | 19.6 / 20.5 s | **48.1** / 81.7 s | 64.1 / 82.7 s |
| sampling (median) | 2.25 s/it | 9.25 s/it | 8.88 s/it |
| CPU fallbacks (every occurrence logged) | – | none | none |
| GTT peak | 0.27 GB | 4.7 GB | 5.0 GB |
| GPU power avg | 206 W | 162 W | 160 W |

- No out-of-memory errors and no CPU fallbacks with 0.2.1.
- Per image the Vulkan times are bimodal: 48–52 s (41 s of sampling, as fast as the best 0.2.0
  run) or ~81 s (74 s of sampling): 83, 81, 82, 82, 70, **49**, 81, **52**, **48**, 81 s with
  `mem_get_info`. That matches the eviction state described in "Profiling" above.
- The memory clock stayed at 1218 MHz throughout (new `gpu_mclk_mhz` column), so the slow images are
  not a memory downclock.

### Profiling the slow Vulkan runs, 2026-10-24

Tools: [`debug/dispatch-stats`](https://github.com/hartmark/torchvulkan/tree/debug/dispatch-stats)
adds `torchvulkan.dispatch_stats()` (submits by reason, GPU wait vs. record time, host↔device
transfers, VMA allocations, buffer-cache hits and pooled bytes, budget). `bench.sh --stats` samples
it every 2 s inside ComfyUI (`vulkan/tvstats.jsonl`); `repro_perf.py` is a ~1-minute stand-in for a
sampling step (16 DiT-style fp16 blocks, 5.3 GiB of weights, 4096 tokens) with `--hold-gb`,
`--fill-after`, `--empty-cache` and per-process VRAM/GTT from fdinfo; `repro_ops.py` times single ops.

Findings (RX 7800 XT, Resizable BAR, desktop holding ~2.1 GB VRAM):

- No regression: the slow `bernard-rx7800xt-rebar` run used the same `_C` binary (sha256 4e1729947f7c…)
  as the fast `bernard-rx7800xt-budgetfix` run.
- ComfyUI (`results/prof-dbgfix`, `main` + #33 + counters): the GPU is busy all the time (≈1.9 s of
  every 2 s waiting on fences, recording ~0), but in the slow phase each op takes about twice as
  long (~1500 ops per 2 s instead of ~3200). The first sampling step of every image after the first
  is fast (4.9 s/it), the rest slow (~10 s/it). The switch coincides with the budget jumping from
  13.24 to 15.14 GiB, i.e. 0.95 × the whole 15.98 GiB heap although the desktop holds 2.1 GB. That
  is what RADV reports when the process's own VRAM accounting exceeds the device-wide usage, i.e.
  when some of its "VRAM" buffers have already been evicted to GTT. With that budget torchvulkan
  keeps allocating into an overcommitted heap, buffers stay in GTT and the GPU reads them over PCIe.
- `repro_perf.py`: VRAM pressure inside the process alone does not slow it down (2.03 s/step flat up
  to 5 GiB extra; beyond that torchvulkan refuses, budget stays 13.26 GiB). With a second process
  taking 2.5 GiB plus `empty_cache()` after every step (as ComfyUI does), #34 overcommits:
  0.78 GiB of the process's VRAM is evicted, steps get 5 % slower and **stay slower after the other
  process exits**; #33 raises `OutOfMemoryError` instead. This shows the mechanism, not yet the full
  2× of the ComfyUI case.
- On this GPU `is_bf16_supported()` is False and bf16 ops go through the CPU (e.g. a 4096×3840×11520
  linear 2.8 s instead of 15.8 ms in fp16); ComfyUI picks fp16 here, so it is not affected.

Code notes (torchvulkan `main` 189ac37) relevant to this:

- Every `submit()` waits for its fence (`vkQueueSubmit` + `vkWaitForFences` in `DAGDispatcher::dispatch`),
  so recording and GPU execution never overlap; a batch is submitted at 512 ops, when the touched
  buffers exceed `memoryBudget() / 4` (and 256 MiB), when the metadata buffer is full, on every
  device→host copy and on cache release. A wrong budget therefore also changes the batch size.
- The device buffer cache keeps every freed buffer (keyed by rounded size) until an allocation
  fails; in ComfyUI it held up to 7 GiB, so the process sits at its budget and any other VRAM user
  forces evictions.
- Host→device staging buffers (`HOST_TO_DEVICE`, GTT) are cached the same way and never trimmed:
  1.7–3.9 GiB of GTT stayed allocated after model loads.
- The budget check uses VMA's `vmaGetHeapBudgets` × 0.95 and has no notion of the process's own
  buffers having been evicted.

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
| `repro_perf.py`, `repro_ops.py` | fast sampling-step stand-in and per-op timing (see Profiling) |
| `repro_oom.py`, `repro_budget_lag.py` | out-of-memory reproducers |
| `repro_evict.py` | minimal reproducer of the slow, overcommitted state |

`comfyui-vulkan.patch` is a patch against ComfyUI and is GPL-3.0 like ComfyUI; everything else follows this repository's license.
