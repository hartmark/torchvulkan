import atexit
import sys

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required to use torchvulkan. Please install PyTorch and try again.") from e

from torch.torch_version import TorchVersion

if TorchVersion(torch.__version__) < TorchVersion("2.10.0"):
    raise RuntimeError(f"torchvulkan [ERROR]: torchvulkan requires PyTorch 2.10.0 or higher, but found {torch.__version__}.")

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import _C

def is_available() -> bool:
    return _C.is_available()

def device_count() -> int:
    return _C.device_count()

def current_device() -> int:
    return _C.current_device()

def device(dev):
    index = dev if isinstance(dev, int) or dev is None else torch.device(dev).index
    return torch.accelerator.device_index(index)

def synchronize() -> None:
    _C.synchronize()

def empty_cache() -> None:
    _C.empty_cache()

def _device_index(dev) -> int:
    if isinstance(dev, int):
        return dev
    index = None if dev is None else torch.device(dev).index
    return current_device() if index is None else index

def _default_generator(dev=None) -> torch.Generator:
    return _C._default_generator(_device_index(dev))

def _uint64_seed(seed) -> int:
    seed = int(seed)
    return seed + (1 << 64) if seed < 0 else seed

def manual_seed(seed) -> None:
    _C._manual_seed(current_device(), _uint64_seed(seed))

def manual_seed_all(seed) -> None:
    _C._manual_seed_all(_uint64_seed(seed))

def seed() -> None:
    _default_generator().seed()

def seed_all() -> None:
    random_seed = _default_generator(0).seed()
    for index in range(1, device_count()):
        _default_generator(index).manual_seed(random_seed)

def initial_seed() -> int:
    return _default_generator().initial_seed()

def get_rng_state(device="vulkan") -> torch.Tensor:
    return _default_generator(device).get_state()

def set_rng_state(new_state: torch.Tensor, device="vulkan") -> None:
    _default_generator(device).set_state(new_state.clone(memory_format=torch.contiguous_format))

def get_rng_state_all() -> list:
    return [get_rng_state(index) for index in range(device_count())]

def set_rng_state_all(new_states) -> None:
    for index, state in enumerate(new_states):
        set_rng_state(state, index)

def _is_in_bad_fork() -> bool:
    return _C._is_in_bad_fork()

atexit.register(_C._save_pipeline_caches)

torch.utils.rename_privateuse1_backend('vulkan')
torch._register_device_module('vulkan', sys.modules[__name__])