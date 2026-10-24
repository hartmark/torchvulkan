# Times each op of the repro_perf.py block on its own and shows torchvulkan's transfer/submit counters,
# to find ops that silently round-trip through the CPU.
import time
import torch
import torch.nn.functional as F
import torchvulkan

import sys
dev, dt = torch.device("vulkan"), getattr(torch, sys.argv[1] if len(sys.argv) > 1 else "float16")
L, D, H, FF, C = 4096, 3840, 30, 10240, 5
hd = D // H
GB = 1 << 30
x = torch.randn(1, L, D, dtype=dt, device=dev)
w_qkv = torch.randn(3 * D, D, dtype=dt, device=dev)
w1 = torch.randn(FF, D, dtype=dt, device=dev)
qkv = F.linear(x, w_qkv)
q, k, v = (t.transpose(1, 2) for t in qkv.view(1, L, 3, H, hd).unbind(2))
torchvulkan.synchronize()

cases = {
    'layer_norm': lambda: F.layer_norm(x, (D,)),
    'linear 4096x3840 -> 11520': lambda: F.linear(x, w_qkv),
    'linear 4096x3840 -> 10240': lambda: F.linear(x, w1),
    'view+unbind+transpose': lambda: [t.transpose(1, 2) for t in qkv.view(1, L, 3, H, hd).unbind(2)],
    'contiguous(q chunk)': lambda: q[:, :C].contiguous(),
    'q@k^T (chunk of 5 heads)': lambda: q[:, :C] @ k[:, :C].transpose(-1, -2),
    'q@k^T contiguous inputs': lambda: q[:, :C].contiguous() @ k[:, :C].contiguous().transpose(-1, -2),
    'softmax 5x4096x4096': lambda: torch.softmax(torch.empty(1, C, L, L, dtype=dt, device=dev), dim=-1),
    'probs@v': lambda: torch.empty(1, C, L, L, dtype=dt, device=dev) @ v[:, :C],
    'cat 6 chunks dim=1': lambda: torch.cat([v[:, i:i + C] for i in range(0, H, C)], dim=1),
    'transpose+reshape': lambda: v.transpose(1, 2).reshape(1, L, D),
    'silu': lambda: F.silu(torch.empty(1, L, FF, dtype=dt, device=dev)),
    'mul': lambda: torch.empty(1, L, FF, dtype=dt, device=dev) * torch.empty(1, L, FF, dtype=dt, device=dev),
    'add': lambda: x + x,
    'sdpa 5 heads': lambda: F.scaled_dot_product_attention(q[:, :C], k[:, :C], v[:, :C]),
}
for name, fn in cases.items():
    fn(); torchvulkan.synchronize()
    torchvulkan.reset_dispatch_stats()
    t = time.time()
    for _ in range(3):
        out = fn()
    torchvulkan.synchronize()
    el = (time.time() - t) / 3
    s = torchvulkan.dispatch_stats()
    flag = '  <-- CPU round trip' if s['h2d_bytes'] or s['d2h_bytes'] else ''
    print(f'{name:28} {el * 1000:9.1f} ms  submits {s["submits"]:3} (flush {s["submit_flush"]:3}) ops {s["ops"]:4} '
          f'gpu_wait {s["gpu_wait_s"] * 1000 / 3:8.1f} ms  h2d {s["h2d_bytes"] / GB / 3:6.3f} GiB  '
          f'd2h {s["d2h_bytes"] / GB / 3:6.3f} GiB{flag}', flush=True)
