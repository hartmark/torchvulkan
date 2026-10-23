#include "allocator.h"
#include "shader.h"

#include <chrono>
#include <cstring>
#include <thread>

VulkanAllocator globalVulkanAllocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &globalVulkanAllocator);

c10::DataPtr VulkanAllocator::allocate(size_t nbytes)
{
    c10::Device device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice());
    if (nbytes == 0) return {nullptr, nullptr, &VulkanAllocator::deleter, device};

    DeviceContext* context = VulkanContext::Instance().CurrentDeviceContext();
    VulkanBuffer* buffer = allocateBuffer(context, nbytes, MemoryUsage::DEVICE_ONLY);
    context->recordAllocation(buffer->size());
    return {buffer, buffer, &VulkanAllocator::deleter, device};
}

void VulkanAllocator::deleter(void* ptr)
{
    if (ptr == nullptr || VulkanContext::IsInBadFork()) return;
    VulkanBuffer* buffer = static_cast<VulkanBuffer*>(ptr);
    buffer->owner()->recordFree(buffer->size());
    buffer->owner()->cache.deleteBuffer(buffer);
}

void VulkanAllocator::copy_data(void* dest, const void* src, std::size_t count) const
{
    copy_device_to_device(dest, /* dest_offset = */ 0, src, /* src_offset = */ 0, count);
}

void VulkanAllocator::copy_host_to_device(void* dest, uint64_t dest_offset, const void* src, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.");
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.");
    VulkanBuffer* dstBuffer = static_cast<VulkanBuffer*>(dest);
    DeviceContext* device = dstBuffer->owner();

    device->stats.h2d_copies++;
    device->stats.h2d_bytes += count;
    VulkanBuffer* stagingBuffer = allocateBuffer(device, count, MemoryUsage::HOST_TO_DEVICE);
    std::memcpy(stagingBuffer->data(), src, count);
    stagingBuffer->flush();

    device->shader_manager->dispatchCopy(stagingBuffer, /* srcOffset = */ 0, dstBuffer, dest_offset, count, /* release_after = */ stagingBuffer);
}

void VulkanAllocator::copy_device_to_host(void* dest, const void* src, uint64_t src_offset, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.");
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.");
    VulkanBuffer* srcBuffer = static_cast<VulkanBuffer*>(const_cast<void*>(src));
    DeviceContext* device = srcBuffer->owner();

    device->stats.d2h_copies++;
    device->stats.d2h_bytes += count;
    VulkanBuffer* stagingBuffer = allocateBuffer(device, count, MemoryUsage::DEVICE_TO_HOST);
    device->shader_manager->dispatchCopy(srcBuffer, src_offset, stagingBuffer, /* dstOffset = */ 0, count);

    try {
        device->flush();
    } catch (...) {
        device->cache.deleteBuffer(stagingBuffer);
        throw;
    }

    stagingBuffer->invalidate();
    std::memcpy(dest, stagingBuffer->data(), count);
    device->cache.deleteBuffer(stagingBuffer);
}

void VulkanAllocator::copy_device_to_device(void* dest, uint64_t dest_offset, const void* src, uint64_t src_offset, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.");
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.");
    VulkanBuffer* dstBuffer = static_cast<VulkanBuffer*>(dest);
    VulkanBuffer* srcBuffer = static_cast<VulkanBuffer*>(const_cast<void*>(src));
    TORCH_CHECK(srcBuffer->owner() == dstBuffer->owner(), "torchvulkan [ERROR]: copy_device_to_device needs both buffers on the same device.");

    dstBuffer->owner()->shader_manager->dispatchCopy(srcBuffer, src_offset, dstBuffer, dest_offset, count);
}

VulkanBuffer* VulkanAllocator::createBuffer(DeviceContext* device, size_t size, MemoryUsage usage)
{
    size_t rounded = VulkanCache::roundBufferSize(size);
    if (usage == MemoryUsage::DEVICE_ONLY && device->memoryUsage() + rounded > device->memoryBudget()) {
        device->refreshMemoryBudget();
        if (device->memoryUsage() + rounded > device->memoryBudget()) return nullptr;
    }

    auto buffer = std::make_unique<VulkanBuffer>(device);
    if (buffer->createBuffer(rounded, usage) != VK_SUCCESS) return nullptr;
    if (usage == MemoryUsage::DEVICE_ONLY) { device->stats.device_allocs++; device->stats.device_alloc_bytes += rounded; }
    else { device->stats.staging_allocs++; device->stats.staging_alloc_bytes += rounded; }
    return buffer.release();
}

VulkanBuffer* VulkanAllocator::allocateBuffer(DeviceContext* device, size_t size, MemoryUsage usage)
{
    if (VulkanBuffer* buffer = device->cache.allocateBuffer(size, usage)) { device->stats.cache_hits++; return buffer; }
    if (VulkanBuffer* buffer = createBuffer(device, size, usage)) return buffer;

    device->shader_manager->flushAndReleaseBuffers();
    // The kernel releases the freed memory asynchronously, and the driver's budget counts it until it
    // has; retry with a short backoff (at most ~255 ms, only on this out-of-memory path).
    for (int attempt = 0; attempt < 8; ++attempt) {
        if (VulkanBuffer* buffer = createBuffer(device, size, usage)) return buffer;
        std::this_thread::sleep_for(std::chrono::milliseconds(1 << attempt));
    }
    if (VulkanBuffer* buffer = createBuffer(device, size, usage)) return buffer;

    size_t budget = device->memoryBudget();
    size_t used = device->memoryUsage();
    TORCH_CHECK_WITH(OutOfMemoryError, false, "torchvulkan [ERROR]: Out of memory on ", device->properties.deviceName, ". Tried to allocate ",
        VulkanCache::roundBufferSize(size), " bytes with ", used, " of ", budget, " bytes in use."
    );
}