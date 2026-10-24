# Fast stand-in for a diffusion sampling step: N DiT-style blocks (qkv, attention, out proj,
# SiLU-gated FFN) over ~4096 tokens with ~5-6 GiB of resident weights. Prints the time per step
# together with torchvulkan's dispatch counters (needs a build with dispatch_stats(); degrades
# gracefully without) and what the GPU did meanwhile (busy, power, clocks, VRAM/GTT).
# usage: python repro_perf.py [--blocks 16] [--steps 6] [--stream-blocks 0] [--upload]
import argparse, glob, os, threading, time
import torch
import torch.nn.functional as F
import torchvulkan

ap = argparse.ArgumentParser()
ap.add_argument('--blocks', type=int, default=16)
ap.add_argument('--dim', type=int, default=3840)
ap.add_argument('--ffn', type=int, default=10240)
ap.add_argument('--heads', type=int, default=30)
ap.add_argument('--seq', type=int, default=4096)
ap.add_argument('--steps', type=int, default=6)
ap.add_argument('--dtype', default='float16', choices=['bfloat16', 'float16', 'float32'])
ap.add_argument('--upload', action='store_true', help='create weights on the CPU and copy them over')
ap.add_argument('--stream-blocks', type=int, default=0,
                help='keep this many blocks on the CPU and upload them every step (like partial offload)')
ap.add_argument('--device', default='vulkan')
ap.add_argument('--empty-cache', action='store_true', help='torchvulkan.empty_cache() after every step (ComfyUI soft_empty_cache)')
ap.add_argument('--fill-after', type=int, default=-1, help='after this step, allocate 256 MiB blocks until torchvulkan refuses (fill to the budget)')
ap.add_argument('--hold-gb', type=float, default=0.0, help='extra resident VRAM in this process (e.g. a text encoder ComfyUI keeps loaded)')
ap.add_argument('--attn-chunk', type=int, default=5, help='heads per attention chunk (ComfyUI splits attention too)')
a = ap.parse_args()
dev = torch.device(a.device)
dt = getattr(torch, a.dtype)
GB = 1 << 30
HAS_STATS = hasattr(torchvulkan, 'dispatch_stats') and a.device == 'vulkan'


def rd(p):
    try:
        with open(p) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def card():
    return max(glob.glob('/sys/class/drm/card[0-9]*/device/mem_info_vram_total'), key=rd).rsplit('/', 1)[0]


CARD = card()
HWMON = (glob.glob(CARD + '/hwmon/hwmon*') or [None])[0]


def proc_mem():
    # where the kernel placed this process's buffers (amdgpu fdinfo), GiB
    vram = gtt = 0
    for f in glob.glob('/proc/self/fdinfo/*'):
        try:
            with open(f) as fh:
                t = fh.read()
        except OSError:
            continue
        if 'drm-driver:\tamdgpu' not in t:
            continue
        for line in t.splitlines():
            if line.startswith('drm-memory-vram:'):
                vram = max(vram, int(line.split()[1]))
            elif line.startswith('drm-memory-gtt:'):
                gtt = max(gtt, int(line.split()[1]))
    return vram / (1 << 20), gtt / (1 << 20)


def dpm(name):
    try:
        with open(f'{CARD}/{name}') as f:
            for line in f:
                if line.rstrip().endswith('*'):
                    return int(line.split(':')[1].strip().split('M')[0])
    except OSError:
        pass
    return None


class Sampler(threading.Thread):
    """Samples GPU state every 50 ms while a step runs."""
    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.on = [], True
        self.start()

    def run(self):
        while self.on:
            pw = rd(f'{HWMON}/power1_average') if HWMON else None
            self.rows.append((rd(CARD + '/gpu_busy_percent'), None if pw is None else pw / 1e6,
                              dpm('pp_dpm_mclk'), dpm('pp_dpm_sclk'),
                              (rd(CARD + '/mem_info_vram_used') or 0) / GB, (rd(CARD + '/mem_info_gtt_used') or 0) / GB))
            time.sleep(0.05)

    def stop(self):
        self.on = False
        self.join()
        cols = list(zip(*self.rows)) or [[]] * 6
        avg = lambda c: sum(x for x in c if x is not None) / max(1, sum(1 for x in c if x is not None))
        mx = lambda c: max((x for x in c if x is not None), default=0)
        return dict(busy=avg(cols[0]), power=avg(cols[1]), mclk=avg(cols[2]), sclk=avg(cols[3]),
                    vram=mx(cols[4]), gtt=mx(cols[5]))


def make_block(where):
    def w(o, i):
        t = torch.randn(o, i, dtype=dt) * (i ** -0.5)
        return t.to(dev) if (where == 'gpu' and a.upload) else (t if where == 'cpu' else t.to(dev))
    if where == 'gpu' and not a.upload:
        mk = lambda o, i: (torch.randn(o, i, dtype=dt, device=dev) * (i ** -0.5))
        return dict(qkv=mk(3 * a.dim, a.dim), out=mk(a.dim, a.dim), w1=mk(a.ffn, a.dim), w3=mk(a.ffn, a.dim), w2=mk(a.dim, a.ffn))
    return dict(qkv=w(3 * a.dim, a.dim), out=w(a.dim, a.dim), w1=w(a.ffn, a.dim), w3=w(a.ffn, a.dim), w2=w(a.dim, a.ffn))


def block_fwd(x, p):
    h = F.layer_norm(x, (a.dim,))
    q, k, v = F.linear(h, p['qkv']).view(1, a.seq, 3, a.heads, a.dim // a.heads).unbind(2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scale = (a.dim // a.heads) ** -0.5
    att = torch.cat([torch.softmax((q[:, h:h + a.attn_chunk] @ k[:, h:h + a.attn_chunk].transpose(-1, -2)) * scale, dim=-1)
                     @ v[:, h:h + a.attn_chunk] for h in range(0, a.heads, a.attn_chunk)], dim=1)
    x = x + F.linear(att.transpose(1, 2).reshape(1, a.seq, a.dim), p['out'])
    h = F.layer_norm(x, (a.dim,))
    return x + F.linear(F.silu(F.linear(h, p['w1'])) * F.linear(h, p['w3']), p['w2'])


t0 = time.time()
resident = [make_block('gpu') for _ in range(a.blocks - a.stream_blocks)]
streamed = [make_block('cpu') for _ in range(a.stream_blocks)]
if a.device == 'vulkan':
    torchvulkan.synchronize()
per_block = sum(t.numel() for t in (resident or streamed)[0].values()) * torch.finfo(dt).bits / 8 / GB
print(f'{len(resident)} resident + {len(streamed)} streamed blocks, {per_block:.2f} GiB each, '
      f'{per_block * len(resident):.1f} GiB on the GPU; setup {time.time() - t0:.1f} s; '
      f'torchvulkan {torchvulkan.__version__}', flush=True)

hold = [torch.ones((256 << 18,), dtype=torch.float32, device=dev) for _ in range(int(a.hold_gb * 4))]
x0 = torch.randn(1, a.seq, a.dim, dtype=dt, device=dev)
for step in range(a.steps):
    if HAS_STATS:
        torchvulkan.reset_dispatch_stats()
    smp = Sampler()
    t = time.time()
    x = x0
    for p in resident:
        x = block_fwd(x, p)
    for p in streamed:
        x = block_fwd(x, {k: v.to(dev) for k, v in p.items()})
    float(x[0, 0, 0])  # forces the queue to finish
    dt_step = time.time() - t
    g = smp.stop()
    pv, pg = proc_mem()
    line = (f'step {step}: {dt_step:6.2f} s | proc vram {pv:5.2f} gtt {pg:4.2f} GiB | gpu busy {g["busy"]:3.0f}% {g["power"]:4.0f} W sclk {g["sclk"]:4.0f} '
            f'mclk {g["mclk"]:4.0f} | vram {g["vram"]:5.2f} gtt {g["gtt"]:4.2f} GiB')
    if HAS_STATS:
        s = torchvulkan.dispatch_stats()
        line += (f' | submits {s["submits"]} (ops {s["submit_ops_limit"]} bytes {s["submit_bytes_limit"]} '
                 f'flush {s["submit_flush"]} meta {s["submit_metadata"]} rel {s["submit_release"]}) ops {s["ops"]} '
                 f'wait {s["gpu_wait_s"]:.2f}s rec {s["record_s"]:.2f}s h2d {s["h2d_bytes"] / GB:.2f}GiB '
                 f'allocs {s["device_allocs"]} hits {s["cache_hits"]} budget {s["budget_now"] / GB:.2f} '
                 f'pool dev {s["pooled_device_bytes"] / GB:.2f} stg {s["pooled_staging_bytes"] / GB:.2f}')
    print(line, flush=True)
    if a.empty_cache and a.device == 'vulkan':
        torchvulkan.empty_cache()
    if step == a.fill_after:
        n = 0
        try:
            while True:
                hold.append(torch.ones((256 << 18,), dtype=torch.float32, device=dev)); n += 1
                if a.device == 'vulkan':
                    torchvulkan.synchronize()
        except torch.OutOfMemoryError as e:
            print(f'  filled {n * 0.25:.2f} GiB more until torchvulkan refused: {str(e)[-90:]}', flush=True)
