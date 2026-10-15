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

size_t DeviceContext::memoryUsage()
{
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator, budgets);
    return budgets[vram_heap_index].usage;
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
