#include <torch/extension.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "vulkan/vulkan_context.h"
#include "vulkan/shader.h"

namespace c10 {
namespace impl {

class VulkanGuardImpl : public DeviceGuardImplInterface {
public:
    VulkanGuardImpl() {}
    DeviceType type() const override { return DeviceType::PrivateUse1; }
    
    // we only have 1 compute queue per GPU right now, possibly update in future
    Stream getStream(Device d) const override { return Stream(Stream::UNSAFE, d, 0); }
    Stream exchangeStream(Stream s) const override { return Stream(Stream::UNSAFE, s.device(), 0); }

    Device exchangeDevice(Device d) const override {  
        Device oldDevice = getDevice();
        setDevice(d);
        return oldDevice;
    }

    void setDevice(c10::Device d) const override { 
        VulkanContext::SetCurrentDevice(VulkanContext::Instance().GetDeviceContext(d.index())->index);
    }
    void uncheckedSetDevice(Device d) const noexcept override { VulkanContext::SetCurrentDevice(d.index()); }
    
    c10::Device getDevice() const override { return c10::Device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice()); }
    DeviceIndex deviceCount() const noexcept override {
        try {
            return static_cast<DeviceIndex>(VulkanContext::Instance().getDeviceCount());
        } catch (...) {
            return 0;
        }
    }

    void synchronizeDevice(const DeviceIndex device_index) const override {
        VulkanContext::Instance().GetDeviceContext(device_index)->flush();
    }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, VulkanGuardImpl);

}
}