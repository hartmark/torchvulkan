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

    m.def("_amp_supported_dtypes", [](int64_t device_index) {
        DeviceContext* device = VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index));
        std::vector<py::object> dtypes;
        if (device->support_float16) dtypes.push_back(py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(torch::getTHPDtype(at::kHalf))));
        if (device->support_bfloat16) dtypes.push_back(py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(torch::getTHPDtype(at::kBFloat16))));
        return dtypes;
    });

    m.def("_memory_usage", [](int64_t device_index) -> uint64_t {
        return VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index))->memoryUsage();
    });

    m.def("_dispatch_stats", [](int64_t device_index) {
        DeviceContext* d = VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index));
        DispatchStats& s = d->stats;
        py::dict r;
        r["submits"] = s.submits.load(); r["submit_ops_limit"] = s.submit_ops_limit.load();
        r["submit_bytes_limit"] = s.submit_bytes_limit.load(); r["submit_metadata"] = s.submit_metadata.load();
        r["submit_flush"] = s.submit_flush.load(); r["submit_release"] = s.submit_release.load();
        r["ops"] = s.ops.load(); r["waves"] = s.waves.load();
        r["record_s"] = s.record_ns.load() / 1e9; r["gpu_wait_s"] = s.wait_ns.load() / 1e9;
        r["h2d_copies"] = s.h2d_copies.load(); r["h2d_bytes"] = s.h2d_bytes.load();
        r["d2h_copies"] = s.d2h_copies.load(); r["d2h_bytes"] = s.d2h_bytes.load();
        r["device_allocs"] = s.device_allocs.load(); r["device_alloc_bytes"] = s.device_alloc_bytes.load();
        r["staging_allocs"] = s.staging_allocs.load(); r["staging_alloc_bytes"] = s.staging_alloc_bytes.load();
        r["cache_hits"] = s.cache_hits.load(); r["cache_releases"] = s.cache_releases.load();
        r["cache_released_bytes"] = s.cache_released_bytes.load();
        uint64_t bmin = s.bytes_flush_budget_min.load();
        r["bytes_flush_budget_min"] = bmin == UINT64_MAX ? 0 : bmin; r["bytes_flush_budget_max"] = s.bytes_flush_budget_max.load();
        r["budget_now"] = d->memoryBudget(); r["usage_now"] = d->memoryUsage();
        r["pooled_device_bytes"] = d->cache.pooledBytes(true); r["pooled_staging_bytes"] = d->cache.pooledBytes(false);
        return r;
    });

    m.def("_reset_dispatch_stats", [](int64_t device_index) {
        VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index))->stats.reset();
    });

    m.def("_memory_allocated", [](int64_t device_index) -> uint64_t {
        return VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index))->allocated_bytes.load();
    });

    m.def("_max_memory_allocated", [](int64_t device_index) -> uint64_t {
        return VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index))->peak_allocated_bytes.load();
    });

    m.def("_reset_peak_memory_stats", [](int64_t device_index) {
        DeviceContext* device = VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index));
        device->peak_allocated_bytes.store(device->allocated_bytes.load());
    });

    m.def("_is_initialized", []() -> bool {
        return VulkanContext::IsInitialized();
    });

    m.def("_lazy_init", []() {
        VulkanContext::Instance();
    });

    m.def("_device_properties", [](int64_t device_index) {
        DeviceContext* device = VulkanContext::Instance().GetDeviceContext(static_cast<c10::DeviceIndex>(device_index));
        const VkPhysicalDeviceProperties& props = device->properties;
        py::dict info;
        info["name"] = std::string(props.deviceName);
        info["total_memory"] = device->memoryTotal();
        info["vendor_id"] = props.vendorID;
        info["device_id"] = props.deviceID;
        info["api_version"] = std::to_string(VK_API_VERSION_MAJOR(props.apiVersion)) + "." + std::to_string(VK_API_VERSION_MINOR(props.apiVersion)) + "." + std::to_string(VK_API_VERSION_PATCH(props.apiVersion));
        info["driver_version"] = props.driverVersion;
        info["subgroup_size"] = device->subgroup_size;
        return info;
    });

    m.def("_validation_error_count", []() -> uint64_t {
        return VulkanContext::ValidationErrorCount();
    });
}
