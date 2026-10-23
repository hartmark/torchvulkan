#include "device_context.h"
#include "shader.h"

static constexpr double VRAM_USAGE_LIMIT = 0.95;

void DeviceContext::flush()
{
    shader_manager->flush();
}

size_t DeviceContext::memoryBudget()
{
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator, budgets);
    return static_cast<size_t>(budgets[vram_heap_index].budget * VRAM_USAGE_LIMIT);
}

// VMA only re-reads VK_EXT_memory_budget after ~30 vkAllocateMemory/vkFreeMemory calls (or on
// vmaSetCurrentFrameIndex). Allocations that createBuffer() refuses never reach VMA, so without an
// explicit refresh a budget sampled while the GPU was full stays in effect after memory was freed.
void DeviceContext::refreshMemoryBudget()
{
    vmaSetCurrentFrameIndex(allocator, budget_frame.fetch_add(1) + 1);
}

size_t DeviceContext::memoryUsage()
{
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator, budgets);
    return budgets[vram_heap_index].usage;
}

size_t DeviceContext::memoryTotal()
{
    const VkPhysicalDeviceMemoryProperties* props = nullptr;
    vmaGetMemoryProperties(allocator, &props);
    return props->memoryHeaps[vram_heap_index].size;
}

void DeviceContext::recordAllocation(size_t bytes)
{
    size_t now = allocated_bytes.fetch_add(bytes) + bytes;
    size_t peak = peak_allocated_bytes.load();
    while (now > peak && !peak_allocated_bytes.compare_exchange_weak(peak, now)) {}
}

void DeviceContext::recordFree(size_t bytes)
{
    allocated_bytes.fetch_sub(bytes);
}

DeviceContext::~DeviceContext()
{
    delete shader_manager;
    shader_manager = nullptr;

    cache.clear();

    if (commandPool != VK_NULL_HANDLE) device_table.vkDestroyCommandPool(device, commandPool, nullptr);
    if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
    if (device != VK_NULL_HANDLE) device_table.vkDestroyDevice(device, nullptr);
}
