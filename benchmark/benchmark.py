"""Benchmark a model's forward + backward pass across every available backend.

Detects the compute backends on this machine (CPU, CUDA, MPS, and the
torchvulkan `vulkan` device), then times a combined forward + backward pass
(``float16`` on accelerators, ``float32`` on CPU) at a range of model sizes, and plots the
per-backend timings against parameter count so you can see how each
implementation scales as the model grows.

    python benchmark/benchmark.py all                   # every model, one panel each
    python benchmark/benchmark.py llm --iters 50        # one model, more samples per point
    python benchmark/benchmark.py mlp --out bench.png   # where to save the figure

The model is selected by subcommand (``mlp``, ``llm``, ``resnet``, or ``all``); everything
else -- backend detection, subprocess isolation, timing, plotting -- is shared.
Adding a new model to benchmark means writing one ``Task`` subclass, not a
second copy of this whole file.

Each backend is benchmarked in its own subprocess. This isolates them (one
backend crashing can't take down the others) and, importantly, keeps
``torchvulkan`` imported *only* in the vulkan worker: registering its
PrivateUse1 backend otherwise breaks MPS ``backward()`` in the same process.

Data is synthetic (random tensors), so no dataset or tokenizer is needed.
"""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.mlp import MLP
from examples.llm import GPT
from examples.conv import ResNet

ALL_BACKENDS = ["cpu", "cuda", "mps", "vulkan"]
RESULT_PREFIX = "RESULT_JSON "
DTYPE = torch.float16
# CPU half-precision kernels are far slower than float32 ones, so fp16 would misrepresent the CPU baseline
BACKEND_DTYPES = {"cpu": torch.float32}


class Task:
    """One model to benchmark: how to size it, build it, and score it.

    The harness below (backend detection, subprocess isolation, timing,
    plotting) is the same for every task; only these five methods vary.
    """

    name = "override-me"

    def add_args(self, parser):
        """Add this task's CLI arguments (at minimum: --sizes, --batch-size)."""
        raise NotImplementedError

    def cli_args(self, args):
        """This task's arguments, re-serialized to forward to a worker subprocess."""
        raise NotImplementedError

    def param_count(self, size, args) -> int:
        raise NotImplementedError

    def build(self, size, args, device, dtype):
        """Return (model, data, targets) for one benchmark point."""
        raise NotImplementedError

    def loss(self, model, data, targets, criterion):
        """Run the forward pass and return the loss tensor."""
        raise NotImplementedError


def meta_param_count(build):
    with torch.device("meta"):
        return sum(p.numel() for p in build().parameters())


class MLPTask(Task):
    name = "mlp"

    def add_args(self, parser):
        parser.add_argument(
            "--sizes", type=int, nargs="+",
            default=[64, 256, 1024, 2048, 4096, 8192],
            help="hidden widths to sweep (three hidden layers; the default top reaches ~141M params)",
        )
        parser.add_argument("--batch-size", type=int, default=128)

    def cli_args(self, args):
        return ["--batch-size", str(args.batch_size)]

    def param_count(self, size, args):
        return meta_param_count(lambda: MLP(size))

    def build(self, size, args, device, dtype):
        model = MLP(size).to(device).to(dtype)
        data = torch.randn(args.batch_size, 1, 28, 28, device=device, dtype=dtype)
        targets = torch.randint(0, 10, (args.batch_size,), device=device)
        return model, data, targets

    def loss(self, model, data, targets, criterion):
        return criterion(model(data), targets)


class LLMTask(Task):
    name = "llm"
    HEAD_DIM = 64

    def add_args(self, parser):
        parser.add_argument(
            "--sizes", type=int, nargs="+",
            default=[64, 128, 256, 384, 512, 768],
            help="n_embd (model width) values to sweep, each a multiple of the 64-wide head size; "
                 "the default top reaches ~26M params",
        )
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--seq-len", type=int, default=128, help="context length")
        parser.add_argument("--n-layer", type=int, default=4, help="transformer blocks, held fixed while n_embd is swept")

    def cli_args(self, args):
        return ["--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len), "--n-layer", str(args.n_layer)]

    def model(self, size, args):
        return GPT(block_size=args.seq_len, n_layer=args.n_layer, n_embd=size, head_dim=self.HEAD_DIM)

    def param_count(self, size, args):
        return meta_param_count(lambda: self.model(size, args))

    def build(self, size, args, device, dtype):
        model = self.model(size, args).to(device).to(dtype)
        data = torch.randint(0, 256, (args.batch_size, args.seq_len), device=device)
        targets = torch.randint(0, 256, (args.batch_size * args.seq_len,), device=device)
        return model, data, targets

    def loss(self, model, data, targets, criterion):
        logits = model(data)
        return criterion(logits.view(-1, logits.size(-1)), targets)


class ResNetTask(Task):
    name = "resnet"

    def add_args(self, parser):
        parser.add_argument(
            "--sizes", type=int, nargs="+",
            default=[8, 16, 32, 48, 64, 96],
            help="base channel widths to sweep (stages run at 1x, 2x and 4x the width; "
                 "the default top reaches ~6.2M params)",
        )
        parser.add_argument("--batch-size", type=int, default=32)

    def cli_args(self, args):
        return ["--batch-size", str(args.batch_size)]

    def param_count(self, size, args):
        return meta_param_count(lambda: ResNet(size))

    def build(self, size, args, device, dtype):
        model = ResNet(size).to(device).to(dtype)
        data = torch.randn(args.batch_size, 3, 32, 32, device=device, dtype=dtype)
        targets = torch.randint(0, 10, (args.batch_size,), device=device)
        return model, data, targets

    def loss(self, model, data, targets, criterion):
        return criterion(model(data), targets)


TASKS = {"mlp": MLPTask(), "llm": LLMTask(), "resnet": ResNetTask()}


def device_available(backend: str) -> bool:
    if backend == "cpu":
        return True
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "mps":
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_available()
    if backend == "vulkan":
        try:
            import torchvulkan

            return torchvulkan.is_available()
        except Exception:  # noqa: BLE001 - torchvulkan is optional
            return False
    return False


def sync_fn(backend: str):
    if backend == "cuda":
        return torch.cuda.synchronize
    if backend == "mps":
        return torch.mps.synchronize
    if backend == "vulkan":
        import torchvulkan

        return torchvulkan.synchronize
    return lambda: None


def time_pass(model, loss_fn, sync, iters, warmup):
    """Median combined forward + backward time (ms) over `iters` runs."""
    times = []
    for i in range(warmup + iters):
        model.zero_grad(set_to_none=True)

        sync()
        t0 = time.perf_counter()
        loss_fn().backward()
        sync()
        t1 = time.perf_counter()

        if i >= warmup:  # discard warmup iterations
            times.append((t1 - t0) * 1e3)

    return statistics.median(times)


def empty_cache(backend):
    if backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps":
        torch.mps.empty_cache()
    elif backend == "vulkan":
        import torchvulkan

        torchvulkan.empty_cache()


def worker(task, backend, args):
    """Benchmark one backend on `task`; emit rows as JSON on stdout, progress on stderr."""
    warnings.filterwarnings("ignore")  # hide one-time CPU-fallback notices
    result = {"backend": backend, "available": False, "rows": []}

    if not device_available(backend):
        print(f"[{backend}] not available on this machine", file=sys.stderr)
        print(RESULT_PREFIX + json.dumps(result))
        return

    result["available"] = True
    device = torch.device(backend)
    dtype = BACKEND_DTYPES.get(backend, DTYPE)
    sync = sync_fn(backend)
    criterion = nn.CrossEntropyLoss()

    for size in args.sizes:
        params = task.param_count(size, args)
        model = data = targets = None
        try:
            torch.manual_seed(0)
            model, data, targets = task.build(size, args, device, dtype)

            total = time_pass(model, lambda: task.loss(model, data, targets, criterion), sync, args.iters, args.warmup)
            result["rows"].append([params, total])
            print(
                f"[{backend}] size={size:<7} ~{params:>12,} params | "
                f"{total:9.3f} ms",
                file=sys.stderr,
            )
        except Exception as e:
            result["rows"].append([params, None])
            print(f"[{backend}] size={size:<7} failed: "
                  f"{str(e)[:80]}", file=sys.stderr)
        finally:  # free the tensors before the next size
            del model, data, targets
            gc.collect()
            empty_cache(backend)

    print(RESULT_PREFIX + json.dumps(result))


def spawn_worker(task, backend, args):
    """Run one backend in a fresh process; return its rows (or None)."""
    cmd = [
        sys.executable, os.path.abspath(__file__), task.name,
        "--worker", backend,
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        *task.cli_args(args),
        "--sizes", *[str(s) for s in args.sizes],
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
            if payload["available"] and any(r[1] is not None for r in payload["rows"]):
                return payload["rows"]
            return None
    print(f"[{backend}] worker produced no result (exit {proc.returncode})",
          file=sys.stderr)
    return None


def candidate_backends():
    """Backends worth probing on this machine (vulkan decided by its worker)."""
    cands = ["cpu"]
    if torch.cuda.is_available():
        cands.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        cands.append("mps")
    cands.append("vulkan")
    return cands


PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"


TITLES = {"mlp": "MLP", "llm": "LLM", "resnet": "ResNet"}


def draw(ax, results, title, legend):
    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(ALL_BACKENDS)}
    ax.set_facecolor(SURFACE)
    for name, rows in results.items():
        rows = [r for r in rows if r[1] is not None]
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        label = f"{name} ({str(BACKEND_DTYPES.get(name, DTYPE)).removeprefix('torch.')})"
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=colors[name], label=label)
        ax.annotate(
            name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=11, color=colors[name], fontweight="bold",
        )

    ax.set_title(title, fontsize=13, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model parameters", fontsize=11, color=MUTED)
    ax.set_ylabel("time per fwd + bwd pass (ms)", fontsize=11, color=MUTED)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    if legend:
        ax.legend(frameon=False, fontsize=11, labelcolor=INK)


def plot(panels, out_path):
    """Plot {model name: {backend: rows}}, one panel per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.5), facecolor=SURFACE, squeeze=False)
    for i, (model, results) in enumerate(panels.items()):
        draw(axes[0][i], results, TITLES.get(model, model), legend=i == 0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    print(f"\nSaved plot -> {out_path}")


def platform_name():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "vulkan"


def build_parser():
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--iters", type=int, default=10, help="timed iters per point")
    shared.add_argument("--warmup", type=int, default=3, help="untimed warmup iters")
    shared.add_argument("--out", default=None, help="plot path (default: media/benchmark-<platform>[-<model>].png)")
    shared.add_argument(
        "--worker", choices=ALL_BACKENDS,
        help="internal: benchmark a single backend and print JSON",
    )

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="model", required=True, help="which model to benchmark")
    for name, task in TASKS.items():
        task_parser = sub.add_parser(name, parents=[shared], help=f"benchmark {task.name}")
        task.add_args(task_parser)
    sub.add_parser("all", parents=[shared], help="benchmark every model with its default sizes, one panel each")
    return p


def run(task, args):
    print(f"\n== {task.name}: iters={args.iters}, warmup={args.warmup}", flush=True)
    results = {}
    for backend in candidate_backends():
        rows = spawn_worker(task, backend, args)
        if rows is not None:
            results[backend] = rows
    return results


def main():
    args = build_parser().parse_args()

    if args.worker:
        worker(TASKS[args.model], args.worker, args)
        return

    print("Probing backends:", ", ".join(candidate_backends()), flush=True)
    names = list(TASKS) if args.model == "all" else [args.model]
    panels = {}
    for name in names:
        # each model runs with its own default sizes; only the shared timing options carry over
        task_args = args if args.model != "all" else build_parser().parse_args([name, "--iters", str(args.iters), "--warmup", str(args.warmup)])
        results = run(TASKS[name], task_args)
        if results:
            panels[name] = results

    if not panels:
        print("No backend produced results.")
        return

    suffix = "" if args.model == "all" else f"-{args.model}"
    media = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
    plot(panels, args.out or os.path.join(media, f"benchmark-{platform_name()}{suffix}.png"))


if __name__ == "__main__":
    main()
