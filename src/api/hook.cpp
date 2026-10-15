#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Device.h>

#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h" 
#include "vulkan/generator.h"
#include "api/ops/internal.h"

struct VulkanHooksInterface : public at::PrivateUse1HooksInterface 
{
    bool isBuilt() const override { return true; }
    bool isAvailable() const override { return VulkanContext::Instance().getDeviceCount() > 0; }
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return device_index < VulkanContext::Instance().getDeviceCount(); }
    at::Device getDeviceFromPtr(void* data) const override {
        if (data == nullptr) return c10::Device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice());
        return c10::Device(c10::DeviceType::PrivateUse1, static_cast<VulkanBuffer*>(data)->owner()->index);
    }

    c10::Allocator* getPinnedMemoryAllocator() const override {
        return c10::GetAllocator(c10::DeviceType::CPU);
    }

    void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {
        torchvulkan::resize_bytes_vulkan(storage.unsafeGetStorageImpl(), newsize);
    }

    const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override {
        return VulkanGeneratorImpl::default_generator(device_index);
    }

    at::Generator getNewGenerator(c10::DeviceIndex device_index = -1) const override {
        return VulkanGeneratorImpl::create(device_index);
    }
};

static VulkanHooksInterface vulkan_hooks;
static bool is_hooks_registered = []() {
    at::RegisterPrivateUse1HooksInterface(&vulkan_hooks);
    return true;
}();