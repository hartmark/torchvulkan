# Small reproducer for torchvulkan running out of memory well below the card's VRAM.
# Mimics ComfyUI: keeps "weights" resident (many medium tensors), then runs steps that allocate
# and free activation-sized buffers. After each phase it prints
#   tv_used/tv_budget  - torchvulkan's own view (parsed from a deliberately impossible allocation)
#   proc_vram/proc_gtt - where the kernel actually placed this process's buffers (/proc/self/fdinfo)
#   gpu_vram/gpu_gtt   - whole-GPU usage from amdgpu sysfs
# usage: python repro_oom.py [--weights-gb 10] [--steps 40] [--act-mb 64,1024]
import argparse, glob, os, random, re, time
import torch
import torchvulkan

ap = argparse.ArgumentParser()
ap.add_argument('--weights-gb', type=float, default=10.0)
ap.add_argument('--weight-mb', type=int, default=40, help='size of each resident tensor')
ap.add_argument('--steps', type=int, default=40)
ap.add_argument('--act-mb', default='64,1024', help='min,max activation buffer size')
ap.add_argument('--acts-per-step', type=int, default=6)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--upload', action='store_true', help='create weights on the CPU and copy them (like ComfyUI)')
ap.add_argument('--stream-mb', type=int, default=0, help='per step, upload this much from the CPU and drop it '
                '(like ComfyUI streaming offloaded weights)')
ap.add_argument('--stream-chunk-mb', type=int, default=40)
ap.add_argument('--step-sleep', type=float, default=0.0)
ap.add_argument('--report-every', type=int, default=10)
a = ap.parse_args()
dev = torch.device('vulkan')
random.seed(a.seed)
GB = 1 << 30


def card():
    best = None
    for d in glob.glob('/sys/class/drm/card[0-9]*/device'):
        try:
            t = int(open(d + '/mem_info_vram_total').read())
        except OSError:
            continue
        if best is None or t > best[0]:
            best = (t, d)
    return best[1]


CARD = card()


def fdinfo():
    vram = gtt = 0
    for f in glob.glob('/proc/self/fdinfo/*'):
        try:
            txt = open(f).read()
        except OSError:
            continue
        if 'drm-driver:\tamdgpu' not in txt:
            continue
        v = re.search(r'drm-memory-vram:\s+(\d+) KiB', txt)
        g = re.search(r'drm-memory-gtt:\s+(\d+) KiB', txt)
        vram = max(vram, int(v.group(1)) << 10 if v else 0)
        gtt = max(gtt, int(g.group(1)) << 10 if g else 0)
    return vram, gtt


def tv_budget():
    try:
        torch.empty(1 << 46, dtype=torch.uint8, device=dev)
    except Exception as e:
        m = re.search(r'with (\d+) of (\d+) bytes in use', str(e))
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def report(tag):
    torchvulkan.synchronize()
    used, budget = tv_budget()
    pv, pg = fdinfo()
    gv = int(open(CARD + '/mem_info_vram_used').read())
    gg = int(open(CARD + '/mem_info_gtt_used').read())
    alloc = torchvulkan.memory_allocated()
    f = lambda x: '   -  ' if x is None else f'{x / GB:6.2f}'
    print(f'{time.strftime("%H:%M:%S")} {tag:<16} tv_alloc {f(alloc)}  tv_used {f(used)}  tv_budget {f(budget)}  '
          f'proc_vram {f(pv)}  proc_gtt {f(pg)}  gpu_vram {f(gv)}  gpu_gtt {f(gg)}', flush=True)


report('start')
weights = []
n = int(a.weights_gb * 1024 / a.weight_mb)
t0 = time.time()
try:
    for i in range(n):
        if a.upload:
            weights.append(torch.full((a.weight_mb << 18,), 1.0, dtype=torch.float32).to(dev))
        else:
            weights.append(torch.full((a.weight_mb << 18,), 1.0, dtype=torch.float32, device=dev))
        if (i + 1) % max(1, n // 4) == 0:
            report(f'weights {len(weights) * a.weight_mb / 1024:.1f}G')
except torch.OutOfMemoryError as e:
    print('OOM while loading weights:', str(e)[-160:])
    report('weights OOM')
print(f'loaded {len(weights) * a.weight_mb / 1024:.1f} GiB of weights in {time.time() - t0:.1f} s', flush=True)

lo, hi = (int(x) for x in a.act_mb.split(','))
host_chunk = torch.full((a.stream_chunk_mb << 18,), 2.0, dtype=torch.float32)
ok = 0
t_steps = time.time()
for step in range(a.steps):
    try:
        for _ in range(a.stream_mb // a.stream_chunk_mb):
            w = host_chunk.to(dev)
            w.add_(1.0)
            del w
        acts = []
        for _ in range(a.acts_per_step):
            mb = random.randint(lo, hi)
            x = torch.empty((mb << 18,), dtype=torch.float32, device=dev)
            x.fill_(0.5)
            acts.append(x)
        s = sum(float(x[:16].sum()) for x in acts)
        del acts, x
        ok += 1
    except torch.OutOfMemoryError as e:
        print(f'step {step}: OOM:', str(e)[-160:], flush=True)
        report(f'step {step} OOM')
        break
    time.sleep(a.step_sleep)
    if step % a.report_every == a.report_every - 1:
        report(f'step {step + 1}')
report('end')
print(f'RESULT steps_ok={ok}/{a.steps} weights_gib={len(weights) * a.weight_mb / 1024:.1f} '
      f'steps_s={time.time() - t_steps:.1f}', flush=True)
