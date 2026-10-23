# Shows that the Vulkan memory budget lags behind freed memory: fill torchvulkan's buffer cache with
# differently sized buffers, release it with empty_cache(), and sample the budget afterwards.
# usage: python repro_budget_lag.py [--fill-gb 6]
import argparse, re, time
import torch
import torchvulkan

ap = argparse.ArgumentParser()
ap.add_argument('--fill-gb', type=float, default=6.0)
ap.add_argument('--keep-gb', type=float, default=4.0, help='resident tensors kept the whole time')
a = ap.parse_args()
dev = torch.device('vulkan')
GB = 1 << 30


def gpu_vram():
    import glob
    best = max(glob.glob('/sys/class/drm/card[0-9]*/device/mem_info_vram_total'),
               key=lambda p: int(open(p).read()))
    return int(open(best.replace('_total', '_used')).read()) / GB


def budget():
    """(usage, 0.95*budget) as torchvulkan sees it, without triggering a cache release first."""
    try:
        torch.empty(1 << 46, dtype=torch.uint8, device=dev)
    except Exception as e:
        m = re.search(r'with (\d+) of (\d+) bytes in use', str(e))
        return int(m.group(1)) / GB, int(m.group(2)) / GB


keep = [torch.ones((64 << 18,), device=dev) for _ in range(int(a.keep_gb * 1024 / 64))]
print(f'resident {torchvulkan.memory_allocated() / GB:.2f} GiB; usage/budget {budget()}; gpu_vram {gpu_vram():.2f}')
# distinct sizes so nothing is reused from the cache: the cache keeps all of them after del
sizes, total, mb = [], 0, 100
while total < a.fill_gb * 1024:
    sizes.append(mb); total += mb; mb += 7
tmp = [torch.empty((s << 18,), device=dev).fill_(1.0) for s in sizes]
torchvulkan.synchronize()
del tmp
print(f'cache filled with {len(sizes)} buffers ({total / 1024:.1f} GiB); usage/budget {budget()}; gpu_vram {gpu_vram():.2f}')
torchvulkan.empty_cache()
t0 = time.time()
for wait in (0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4):
    while time.time() - t0 < wait:
        time.sleep(0.002)
    u, b = budget()
    print(f't+{time.time() - t0:5.2f}s after empty_cache: usage {u:5.2f}  budget(x0.95) {b:5.2f}  '
          f'free-in-budget {b - u:5.2f}  gpu_vram {gpu_vram():5.2f}', flush=True)

# Force VMA to refetch the driver budget: it does so after ~30 vkAllocateMemory/vkFreeMemory calls.
# Distinct sizes > 128 MiB get their own VkDeviceMemory, and empty_cache() frees them again.
for round_ in range(3):
    bufs = [torch.empty(((130 + 3 * i + 50 * round_) << 18,), device=dev) for i in range(16)]
    del bufs
    torchvulkan.empty_cache()
    u, b = budget()
    print(f'after {32 * (round_ + 1)} extra alloc/free ops: usage {u:5.2f}  budget(x0.95) {b:5.2f}  gpu_vram {gpu_vram():5.2f}', flush=True)
