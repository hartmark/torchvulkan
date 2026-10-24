# Minimal reproducer for the slow state: resident weights get evicted from VRAM once the process
# overcommits the device, and every kernel that reads them slows down.
#   weights: uint8 buffers (like GGUF-quantized weights) read every step by a cast to fp16 (like dequantization)
#   temporaries: differently sized fp16 buffers per step, so torchvulkan's buffer cache grows to the budget
# Per step it prints the step time, the time of the weight-reading part alone, and where the kernel
# placed this process's memory (amdgpu fdinfo: VRAM / GTT / CPU).
# usage: python repro_evict.py [--weights-gb 7] [--steps 16] [--temps 6] [--temp-max-mb 1200]
import argparse, glob, random, time
import torch
import torchvulkan

ap = argparse.ArgumentParser()
ap.add_argument('--weights-gb', type=float, default=7.0)
ap.add_argument('--chunk-mb', type=int, default=64)
ap.add_argument('--steps', type=int, default=16)
ap.add_argument('--temps', type=int, default=6)
ap.add_argument('--temp-max-mb', type=int, default=1200)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--resident-gb', type=float, default=0.0, help='extra buffers that stay allocated and are never read (a text encoder ComfyUI keeps loaded)')
ap.add_argument('--reload-every', type=int, default=0, help='every N steps drop the weights and upload them again from the CPU (ComfyUI unloading/reloading the model between images)')
a = ap.parse_args()
dev, GB, MB = torch.device('vulkan'), 1 << 30, 1 << 20
random.seed(a.seed)


def fdinfo():
    mem = {}
    for f in glob.glob('/proc/self/fdinfo/*'):
        try:
            with open(f) as fh:
                t = fh.read()
        except OSError:
            continue
        if 'drm-driver:\tamdgpu' not in t:
            continue
        for line in t.splitlines():
            for k in ('vram', 'gtt', 'cpu'):
                if line.startswith(f'drm-memory-{k}:'):
                    mem[k] = max(mem.get(k, 0), int(line.split()[1]) * 1024)
    return {k: v / GB for k, v in mem.items()}


def budget():
    if hasattr(torchvulkan, 'mem_get_info'):
        free, total = torchvulkan.mem_get_info()
        return free / GB
    return float('nan')


n = int(a.weights_gb * 1024 / a.chunk_mb)
resident = [torch.ones((256 * MB,), dtype=torch.uint8, device=dev) for _ in range(int(a.resident_gb * 4))]
host_weights = [torch.randint(0, 255, (a.chunk_mb * MB,), dtype=torch.uint8) for _ in range(n)]
weights = [w.to(dev) for w in host_weights]
torchvulkan.synchronize()
m = fdinfo()
print(f'{n} x {a.chunk_mb} MiB uint8 weights ({n * a.chunk_mb / 1024:.1f} GiB); proc vram {m.get("vram", 0):.2f} '
      f'gtt {m.get("gtt", 0):.2f} cpu {m.get("cpu", 0):.2f} GiB; mem_get_info free {budget():.2f} GiB; '
      f'torchvulkan {torchvulkan.__version__}', flush=True)

last_prof = torchvulkan.gpu_profile() if hasattr(torchvulkan, 'gpu_profile') else {}
last_st = torchvulkan.dispatch_stats() if hasattr(torchvulkan, 'dispatch_stats') else {}
for step in range(a.steps):
    reloaded = ''
    if a.reload_every and step and step % a.reload_every == 0:
        del weights
        weights = [w.to(dev) for w in host_weights]
        torchvulkan.synchronize()
        reloaded = ' (weights reloaded)'
    t0 = time.time()
    acc = torch.zeros(a.chunk_mb * MB // 2, dtype=torch.float16, device=dev)
    for w in weights:                       # "dequantize" every weight: reads the resident buffer
        acc += w[: a.chunk_mb * MB // 2].to(torch.float16)
    torchvulkan.synchronize()
    t_read = time.time() - t0
    temps = []
    for _ in range(a.temps):                # activations of varying size -> cache misses, cache growth
        mb = random.randint(64, a.temp_max_mb)
        temps.append(torch.empty((mb * MB // 2,), dtype=torch.float16, device=dev).fill_(1.0))
    float(acc[0]) + sum(float(t[0]) for t in temps)
    del temps
    dt = time.time() - t0
    m = fdinfo()
    prof = ''
    if hasattr(torchvulkan, 'dispatch_stats'):
        st = torchvulkan.dispatch_stats()
        prof += (f' | releases {st["cache_releases"] - last_st.get("cache_releases", 0)} '
                 f'new allocs {st["device_allocs"] - last_st.get("device_allocs", 0)} '
                 f'pool {st["pooled_device_bytes"] / GB:.2f} GiB')
        last_st = st
    if hasattr(torchvulkan, 'gpu_profile') and torchvulkan.gpu_profile():
        cur = torchvulkan.gpu_profile()
        d = {k: (v[0] - last_prof.get(k, (0, 0))[0], v[1] - last_prof.get(k, (0, 0))[1]) for k, v in cur.items()}
        last_prof = cur
        top = sorted(((k, v) for k, v in d.items() if v[0]), key=lambda kv: -kv[1][1])[:2]
        prof += ' | ' + '  '.join(f'{k} {v[1]:.0f} ms/{v[0]} ({1000 * v[1] / v[0]:.0f} us)' for k, v in top)
    print(f'step {step:2}: {dt:5.2f} s (weight reads {t_read:5.2f} s) | proc vram {m.get("vram", 0):5.2f} '
          f'gtt {m.get("gtt", 0):4.2f} cpu {m.get("cpu", 0):4.2f} GiB | mem_get_info free {budget():5.2f} GiB{reloaded}{prof}', flush=True)
