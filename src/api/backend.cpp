#include <torch/extension.h>
#include <torch/csrc/Generator.h>
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "vulkan/generator.h"
#include "vulkan/shader.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchvulkan backend";

    m.def("is_available", []() -> bool {
        try {
            return VulkanContext::Instance().getDeviceCount() > 0;
        } catch (...) {
            return false;
        }
    });

    m.def("device_count", []() -> int64_t {
        try {
            return static_cast<int64_t>(VulkanContext::Instance().getDeviceCount());
        } catch (...) {
            return 0;
        }
    });

    m.def("current_device", []() -> int64_t {
        return static_cast<int64_t>(VulkanContext::CurrentDevice());
    });

    m.def("synchronize", [](int64_t device_index) {
        VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index))->flush();
    }, py::call_guard<py::gil_scoped_release>());

    m.def("_default_generator", [](int64_t device_index) {
        return py::reinterpret_steal<py::object>(THPGenerator_initDefaultGenerator(VulkanGeneratorImpl::default_generator(device_index)));
    });

    m.def("_manual_seed", [](int64_t device_index, uint64_t seed) {
        VulkanGeneratorImpl::manual_seed(static_cast<c10::DeviceIndex>(device_index), seed);
    });

    m.def("_manual_seed_all", [](uint64_t seed) {
        VulkanGeneratorImpl::manual_seed_all(seed);
    });

    m.def("_is_in_bad_fork", []() -> bool {
        return VulkanContext::IsInBadFork();
    });

    m.def("empty_cache", []() {
        for (DeviceContext* device : VulkanContext::Instance().devices) device->shader_manager->flushAndReleaseBuffers();
    }, py::call_guard<py::gil_scoped_release>());

    m.def("_save_pipeline_caches", []() {
        if (!VulkanContext::IsInitialized() || VulkanContext::IsInBadFork()) return;
        for (DeviceContext* device : VulkanContext::Instance().devices) device->shader_manager->savePipelineCache();
    });

    m.def("_memory_usage", [](int64_t device_index) -> uint64_t {
        return VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index))->memoryUsage();
    });

    m.def("_validation_error_count", []() -> uint64_t {
        return VulkanContext::ValidationErrorCount();
    });
}
