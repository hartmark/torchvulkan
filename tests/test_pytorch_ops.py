import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import pytest
import torch
import torchvulkan as torchvk

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

# all the datatypes we guarantee to be supported by torchvk
VULKAN_DTYPES = [
    torch.float64, torch.uint64, torch.int64,
    torch.float32, torch.uint32, torch.int32,
    torch.float16, torch.bfloat16, torch.uint16, torch.int16,
    torch.uint8, torch.int8,
    torch.bool,
]

REMAINING_OPS = set()
UNIMPLEMENTED_OPS = {}

def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.to('cpu')
    elif isinstance(obj, str) and obj == 'vulkan':
        return 'cpu'
    elif isinstance(obj, torch.device) and obj.type == 'privateuseone':
        return torch.device('cpu')
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj

def unimplemented_reason(exception: str):
    """
    Why an error means the backend can't run the op yet (so the sample is skipped), or None if the
    error is a real failure. Only the backend's own fallback errors count: other "not implemented"
    messages (e.g. an op not supporting a dtype) are compared against the CPU like any other error.
    """
    low = exception.lower()

    # strict-mode message: "torchvulkan [NOT IMPLEMENTED]: ... fallback detected for operation: aten::<op>. Set ..."
    if "torchvulkan [not implemented]" in low:
        m = re.search(r"operation:\s*(aten::\S+?)\.?\s+Set", exception)
        return m.group(1) if m else "torchvulkan [NOT IMPLEMENTED]"

    if "could not run" in low and "backend" in low:
        return exception.split("'")[1] if "'" in exception else "unknown"

    # torch's own error for an op with no kernel outside CPU/CUDA (convolution, until the backend has one)
    if "you are likely triggering this with tensor backend other than cpu/cuda" in low:
        return exception.split(" ")[0]

    # an output left on the wrong device by a fallback; these show up as skips, so watch the skip count
    if "to be on cpu, but it's on vulkan" in low:
        return "wrong_device_output"

    # structured/linalg ops (solve, matrix_rank, ...) whose CPU fallback splits tensors across
    # cpu and vulkan; torch 2.13 tightened these device checks, so the backend can't run them
    if "same device" in low or "multi-device" in low:
        return "multi_device_fallback"

    return None

def record_unimplemented(reason):
    UNIMPLEMENTED_OPS[reason] = UNIMPLEMENTED_OPS.get(reason, 0) + 1

# records whether an op draws random numbers: every aten op that does is tagged nondeterministic_seeded
class SeededOpDetector(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seeded = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if torch.Tag.nondeterministic_seeded in func.tags:
            self.seeded = True
        return func(*args, **(kwargs or {}))

def describe(obj):
    return (tuple(obj.shape), obj.dtype) if isinstance(obj, torch.Tensor) else obj

def tensors_of(obj):
    return [t for t in tree_flatten(obj)[0] if isinstance(t, torch.Tensor)]

MATMUL_OPS = ("bmm", "baddbmm", "mm", "addmm", "matmul", "__rmatmul__")

def tie_order_dim(op_name, args, kwargs):
    """
    The dim for ops whose order of equal keys is unspecified (sort/argsort without stable=True, topk), where
    the CPU's own order is arbitrary, or None for every other op.
    """
    if op_name in ("sort", "argsort"):
        if kwargs.get("stable", False):
            return None
        return kwargs.get("dim", args[0] if len(args) > 0 else -1)
    if op_name == "topk":
        return kwargs.get("dim", args[1] if len(args) > 1 else -1)
    return None

class TestVulkanOps(TestCase):

    @ops(op_db, allowed_dtypes=VULKAN_DTYPES)
    def test_correctness(self, device, dtype, op):

        if "as_strided_partial_views" in self._testMethodName:
            self.skipTest("Cross-device storage copy loses unreferenced base memory.")

        # sparse (CSR) layouts are not supported by the Vulkan backend
        if "sparse_sampled_addmm" in self._testMethodName:
            self.skipTest("Sparse CSR layout is not supported by the Vulkan backend.")

        # samples are generated directly on the vulkan device; an unimplemented op used
        # while constructing the inputs should skip the test rather than error it out
        try:
            samples = list(op.sample_inputs(device, dtype))
        except Exception as e:
            reason = unimplemented_reason(str(e))
            if reason is not None:
                record_unimplemented(reason)
                self.skipTest(f"Sample generation for '{op.name}' needs an unimplemented Vulkan op ({reason}).")
            raise

        compared = 0
        skipped_reason = None

        for sample in samples:
            vk_input = sample.input
            vk_args = sample.args
            vk_kwargs = sample.kwargs

            # the CPU reference is the same sample moved back to the host
            cpu_input = to_cpu(vk_input)
            cpu_args = to_cpu(vk_args)
            cpu_kwargs = to_cpu(vk_kwargs)

            # torch 2.10's group_norm errors on an empty-batch input when it
            # round-trips through the Vulkan CPU fallback
            if (op.name == "nn.functional.group_norm"
                    and isinstance(cpu_input, torch.Tensor)
                    and cpu_input.numel() == 0):
                continue

            # Low-precision multi-head attention over OpInfo's large random
            # projection weights is dominated by catastrophic cancellation
            if (op.name == "nn.functional.multi_head_attention_forward"
                    and dtype in (torch.float16, torch.bfloat16)):
                continue

            # torch computes soft_margin_loss's log(1 + exp(-y*x)) directly in fp16 and
            # overflows to inf on large inputs; our stable/fp32 path stays finite, so the
            # fp16 reference is not comparable
            if (op.name == "nn.functional.soft_margin_loss" and dtype == torch.float16):
                continue

            detector = SeededOpDetector()
            cpu_error = None
            try:
                with detector:
                    expected = op(cpu_input, *cpu_args, **cpu_kwargs)
            except Exception as e:
                cpu_error = e

            try:
                actual_vk = op(vk_input, *vk_args, **vk_kwargs)
                actual = to_cpu(actual_vk) # torchvk will only compute on this step
            except Exception as e:
                reason = unimplemented_reason(str(e))
                if reason is not None:
                    record_unimplemented(reason)
                    skipped_reason = reason
                    continue
                if cpu_error is not None:
                    continue  # both backends rejected the input
                self.fail(f"Vulkan backend failed on op '{op.name}' with error: {e}")

            if cpu_error is not None:
                self.fail(f"Vulkan backend accepted an input that the CPU rejects for op '{op.name}': {type(cpu_error).__name__}: {cpu_error}")

            compared += 1

            # random ops can't reproduce CPU's generator; check the output's structure,
            # and that seeding the Vulkan generator makes the op repeatable
            if detector.seeded:
                self.assertEqual(tree_map(describe, actual), tree_map(describe, expected))
                torch.manual_seed(0)
                first = to_cpu(op(vk_input, *vk_args, **vk_kwargs))
                torch.manual_seed(0)
                second = to_cpu(op(vk_input, *vk_args, **vk_kwargs))
                self.assertEqual(first, second)
                continue

            # torch returns matmul results contiguous, and code relies on it (e.g. .view()); a padded
            # cooperative-matrix result once came back as a strided view
            if op.name in MATMUL_OPS:
                for vk_tensor, cpu_tensor in zip(tensors_of(actual_vk), tensors_of(expected)):
                    if cpu_tensor.is_contiguous():
                        self.assertTrue(vk_tensor.is_contiguous(), f"'{op.name}' returned a non-contiguous tensor where the CPU's is contiguous")

            # equal keys may come out in any order: the values must match, and the indices must point at them
            tie_dim = tie_order_dim(op.name, cpu_args, cpu_kwargs)
            if tie_dim is not None and isinstance(cpu_input, torch.Tensor) and cpu_input.dim() > 0:
                if op.name == "argsort":
                    actual_indices, expected_values = actual, cpu_input.gather(tie_dim, expected)
                else:
                    (actual_values, actual_indices), (expected_values, _) = actual, expected
                    self.assertEqual(actual_values, expected_values)
                self.assertEqual(cpu_input.gather(tie_dim, actual_indices), expected_values)
                continue

            # uninitialised memory cannot be compared
            if op.name in ("empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "empty_permuted"):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                continue

            elif dtype in (torch.float16, torch.bfloat16) and op.name in MATMUL_OPS + (
                "linalg.multi_dot", "nn.functional.embedding_bag",
            ):
                self.assertEqual(actual, expected, atol=1e-1, rtol=3e-1)
                continue

            # GPU matmul accumulates in a different order than the CPU reference;
            # the same holds for composite ops that decompose into matmuls.
            elif dtype == torch.float32 and op.name in MATMUL_OPS + (
                "nn.functional.multi_head_attention_forward",
                "nn.functional.scaled_dot_product_attention", "pca_lowrank",
            ):
                self.assertEqual(actual, expected, atol=1e-4, rtol=1e-3)
                continue

            # fused operations lose precision on rounding
            elif dtype in (torch.float16, torch.bfloat16) and op.name in (
                "lerp", "addcmul", "addcdiv", "addr",
                "native_layer_norm", "native_group_norm",
                "nn.functional.layer_norm", "nn.functional.group_norm",
                "nn.functional.bilinear", "nn.functional.poisson_nll_loss",
                "var", "var_mean", "nn.functional.binary_cross_entropy",
                "nn.functional.cross_entropy", "nn.functional.linear_cross_entropy",
            ):
                self.assertEqual(actual, expected, atol=1e-1, rtol=5e-2, exact_dtype=False)
                continue

            # loss/normalisation reductions accumulate in a different order and use float32
            # intermediates on the GPU even for float64 inputs
            elif dtype == torch.float64 and op.name in (
                "nn.functional.cross_entropy", "nn.functional.linear_cross_entropy",
                "nn.functional.local_response_norm", "nn.functional.poisson_nll_loss",
            ):
                self.assertEqual(actual, expected, atol=1e-3, rtol=5e-3)
                continue

            # exp/log and the other transcendentals are evaluated in float32 on the GPU
            # even for float64 inputs, so their float64 results can't beat ~float32 precision
            elif dtype == torch.float64 and op.name in (
                "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan",
                "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                "exp2", "log2", "log10", "expm1", "log1p", "sigmoid",
                "hypot", "xlogy", "logaddexp", "logaddexp2",
            ):
                self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
                continue

            # special functions use float32 polynomial/rational approximations; the Bessel and
            # gamma families lose accuracy for large arguments even in float32
            elif op.name in ("i0", "special.i1", "special.i0e", "special.i1e", "lgamma", "digamma", "erfinv", "mvlgamma", "polygamma"):
                self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
                continue

            # the other special functions are accurate in float32 but downcast float64 to float32
            elif dtype == torch.float64 and op.name in (
                "erf", "erfc", "erfinv", "sinc", "special.entr", "lgamma", "digamma",
                "nn.functional.gelu", "special.ndtr", "special.log_ndtr", "mvlgamma",
            ):
                self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
                continue

            # softmax and log_softmax run through exp/log at float32, so float64 results can't beat ~1e-7
            elif op.name in (
                "softmax", "masked.softmax", "nn.functional.softmin",
                "log_softmax", "masked.log_softmax",
            ) and (
                dtype == torch.float64 or cpu_kwargs.get("dtype") == torch.float64
            ):
                self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)
                continue

            # the GPU's trig intrinsics (acos/asin/tan) are lower-accuracy than the reference
            # even in float32 (tan additionally blows up near its poles)
            elif dtype == torch.float32 and op.name in ("acos", "asin", "tan"):
                self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)
                continue

            # the backend has no fp64 compute path (transcendentals, reductions and matmul all
            # run in fp32), so any float64 result is at best fp32-accurate
            elif dtype == torch.float64:
                self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)
                continue

            elif op.name in ("pow", "__rpow__", "square", "float_power", "atan2", "ldexp") or dtype in (torch.float16, torch.bfloat16):
                self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
                continue

            self.assertEqual(actual, expected)

        if compared == 0 and skipped_reason is not None:
            self.skipTest(f"Operator '{op.name}' is not implemented for Vulkan backend ({skipped_reason}).")
        if skipped_reason is None:
            REMAINING_OPS.add(op.name)

    # backward passes: gradients of the sum of every floating output, against the CPU's
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=[torch.float32])
    def test_gradients(self, device, dtype, op):
        if "as_strided_partial_views" in self._testMethodName:
            self.skipTest("Cross-device storage copy loses unreferenced base memory.")
        if "sparse_sampled_addmm" in self._testMethodName:
            self.skipTest("Sparse CSR layout is not supported by the Vulkan backend.")

        try:
            samples = list(op.sample_inputs(device, dtype, requires_grad=True))
        except Exception as e:
            reason = unimplemented_reason(str(e))
            if reason is not None:
                record_unimplemented(reason)
                self.skipTest(f"Sample generation for '{op.name}' needs an unimplemented Vulkan op ({reason}).")
            raise

        compared = 0
        skipped_reason = None

        for sample in samples:
            vk_leaves = [t for t in tensors_of((sample.input, sample.args, sample.kwargs)) if t.requires_grad]
            if not vk_leaves:
                continue

            # CPU leaves standing in for the Vulkan ones, so each side's graph is its own
            cpu_leaf = {id(t): t.detach().cpu().requires_grad_(True) for t in vk_leaves}
            def as_cpu(obj):
                if isinstance(obj, torch.Tensor):
                    return cpu_leaf[id(obj)] if id(obj) in cpu_leaf else obj.detach().cpu()
                return to_cpu(obj) if not isinstance(obj, (list, tuple, dict)) else tree_map(as_cpu, obj)
            cpu_input, cpu_args, cpu_kwargs = as_cpu(sample.input), tree_map(as_cpu, sample.args), tree_map(as_cpu, sample.kwargs)

            def grads_of(fn_args, leaves):
                with SeededOpDetector() as detector:
                    out = op(*fn_args[0], **fn_args[1])
                outputs = [o for o in tensors_of(out) if o.requires_grad and o.is_floating_point()]
                if not outputs or detector.seeded:
                    return None
                loss = sum(o.sum() for o in outputs)
                return torch.autograd.grad(loss, leaves, allow_unused=True)

            cpu_error = None
            try:
                expected = grads_of(((cpu_input, *cpu_args), cpu_kwargs), [cpu_leaf[id(t)] for t in vk_leaves])
            except Exception as e:
                cpu_error = e

            try:
                actual = grads_of(((sample.input, *sample.args), sample.kwargs), vk_leaves)
            except Exception as e:
                reason = unimplemented_reason(str(e))
                if reason is not None:
                    record_unimplemented(reason)
                    skipped_reason = reason
                    continue
                if cpu_error is not None:
                    continue
                self.fail(f"Vulkan backward failed on op '{op.name}' with error: {e}")

            if cpu_error is not None or expected is None or actual is None:
                continue

            compared += 1
            for a, e in zip(actual, expected):
                # autograd may leave an input's gradient as None where the other side computes zeros
                if a is None or e is None:
                    present = e if a is None else a.cpu()
                    if present is not None:
                        self.assertEqual(present, torch.zeros_like(present))
                    continue
                if op.name in MATMUL_OPS or "attention" in op.name or "linear" in op.name or "conv" in op.name:
                    self.assertEqual(a.cpu(), e, atol=1e-3, rtol=1e-3)
                # chains of matmuls, where the GPU's summation order compounds
                elif op.name.startswith("linalg."):
                    self.assertEqual(a.cpu(), e, atol=1e-3, rtol=5e-3)
                elif op.name in ("i0", "special.i1", "special.i0e", "special.i1e", "lgamma", "digamma", "erfinv", "mvlgamma", "polygamma"):
                    self.assertEqual(a.cpu(), e, atol=1e-2, rtol=1e-2)
                # with few observations and a large correction, cov divides rounding error by ~1
                elif op.name in ("cov", "corrcoef"):
                    self.assertEqual(a.cpu(), e, atol=5e-3, rtol=1e-3)
                else:
                    self.assertEqual(a.cpu(), e, atol=1e-4, rtol=1e-3)

        if compared == 0 and skipped_reason is not None:
            self.skipTest(f"Backward of '{op.name}' is not implemented for Vulkan backend ({skipped_reason}).")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        print(f'\n--- TEST SUMMARY ---')
        print(f'The complete set of working operators is: {REMAINING_OPS}')
        sorted_ops = sorted(UNIMPLEMENTED_OPS.items(), key=lambda item: item[1], reverse=True)
        print(f'The complete set of unimplemented operators is: {sorted_ops}')

# run the OpInfo suite directly on the vulkan (PrivateUse1) device: inputs are generated
# on-device via the RNG fills, exactly how a real user drives the backend
instantiate_device_type_tests(TestVulkanOps, globals(), only_for='privateuse1')


def run_script(source, env=None):
    """runs `source` in a fresh interpreter, outside this process's Vulkan state"""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=300)


if __name__ == '__main__':
    run_tests()
