#pragma once
#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <volk.h>

#include <atomic>

#include "vk_mem_alloc.h"
#include "cache.h"

class VulkanShaderManager;

// Debug counters (torchvulkan.dispatch_stats()); relaxed atomics, cheap enough to leave on.
struct DispatchStats {
    std::atomic<uint64_t> submits{0}, submit_ops_limit{0}, submit_bytes_limit{0}, submit_metadata{0},
        submit_flush{0}, submit_release{0};
    std::atomic<uint64_t> ops{0}, waves{0}, record_ns{0}, wait_ns{0};
    std::atomic<uint64_t> h2d_copies{0}, h2d_bytes{0}, d2h_copies{0}, d2h_bytes{0};
    std::atomic<uint64_t> device_allocs{0}, device_alloc_bytes{0}, staging_allocs{0}, staging_alloc_bytes{0};
    std::atomic<uint64_t> cache_hits{0}, cache_releases{0}, cache_released_bytes{0};
    std::atomic<uint64_t> bytes_flush_budget_min{UINT64_MAX}, bytes_flush_budget_max{0};
    void reset() {
        for (auto* a : {&submits, &submit_ops_limit, &submit_bytes_limit, &submit_metadata, &submit_flush, &submit_release,
                        &ops, &waves, &record_ns, &wait_ns, &h2d_copies, &h2d_bytes, &d2h_copies, &d2h_bytes,
                        &device_allocs, &device_alloc_bytes, &staging_allocs, &staging_alloc_bytes,
                        &cache_hits, &cache_releases, &cache_released_bytes, &bytes_flush_budget_max}) a->store(0);
        bytes_flush_budget_min.store(UINT64_MAX);
    }
};

struct DeviceContext {
    c10::DeviceIndex index = -1;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = UINT32_MAX;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    uint32_t vram_heap_index = 0;
    VolkDeviceTable device_table{};
    VkPhysicalDeviceProperties properties{};
    VkDriverId driver_id = static_cast<VkDriverId>(0);
    VulkanShaderManager* shader_manager = nullptr;
    VulkanCache cache;
    at::Generator generator;
    bool valid = true;

    bool support_float64 = false;
    bool support_int64 = false;
    bool support_float32 = false;
    bool support_int32 = false;
    bool support_float16 = false;
    bool support_bfloat16 = false;
    bool support_int16 = false;
    bool support_int8 = false;

    bool support_coopmat = false;
    bool support_bfloat16_coopmat = false;
    bool support_subgroup_control = false;
    bool support_subgroup_extended_types = false;
    bool support_subgroup_arithmetic = false;
    bool support_pipeline_statistics = false;
    bool support_memory_budget = false;
    uint32_t subgroup_size = 0;

    std::atomic<size_t> allocated_bytes{0};
    std::atomic<size_t> peak_allocated_bytes{0};
    DispatchStats stats;

    DeviceContext() { cache.setOwner(this); }
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    ~DeviceContext();

    size_t memoryBudget();
    size_t memoryUsage();
    size_t memoryTotal();
    size_t memoryFree();
    void recordAllocation(size_t bytes);
    void recordFree(size_t bytes);
    void flush();
};