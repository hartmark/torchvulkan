#pragma once
#include "vulkan/vulkan_context.h"
#include "vulkan/builders.h"
#include <c10/core/Device.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/strides.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

inline uint64_t get_tensor_address(const at::Tensor& t) 
{
    VulkanBuffer* buffer = static_cast<VulkanBuffer*>(t.storage().data_ptr().get_context());
    return buffer->bufferAddress() + t.storage_offset() * t.element_size();
}

inline bool is_dtype_supported(at::ScalarType dtype) 
{
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    switch (dtype) 
    {
        case at::kDouble: return device->support_float64;
        case at::kLong: return device->support_int64;
        case at::kUInt64: return device->support_int64;
        
        case at::kFloat: return device->support_float32;
        case at::kInt: return device->support_int32;
        case at::kUInt32: return device->support_int32;
        
        case at::kHalf: return device->support_float16;
        case at::kBFloat16: return device->support_bfloat16;
        case at::kShort: return device->support_int16;
        case at::kUInt16: return device->support_int16;
        
        case at::kChar: return device->support_int8; 
        case at::kByte: return device->support_int8; 
        case at::kBool: return device->support_int8; 
        
        default: return false;
    }
}

inline int get_dtype_vec_size(at::ScalarType dtype) 
{
    size_t bytes = c10::elementSize(dtype);
    if (bytes == 0) return 1;
    int size = (int)(16 / bytes);
    size = size <= 4 ? size : 4; // max vector size is 4
    return size; 
}

inline int get_dtype_workgroup_size(at::ScalarType dtype, uint32_t vecSize) 
{
    size_t bytes = c10::elementSize(dtype);
    if (bytes == 0 || vecSize == 0) return 1;
    int size = (int)(1024 / (bytes * vecSize)); // max out memory bandwidth
    return size;
}

inline int get_dtype_digits(at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return std::numeric_limits<double>::digits;
        case at::kFloat: return std::numeric_limits<float>::digits;
        case at::kHalf: return std::numeric_limits<c10::Half>::digits;
        case at::kBFloat16: return std::numeric_limits<c10::BFloat16>::digits;

        case at::kBool: return std::numeric_limits<bool>::digits;
        case at::kByte: return std::numeric_limits<uint8_t>::digits;
        case at::kChar: return std::numeric_limits<int8_t>::digits;
        case at::kShort: return std::numeric_limits<int16_t>::digits;
        case at::kUInt16: return std::numeric_limits<uint16_t>::digits;
        case at::kInt: return std::numeric_limits<int32_t>::digits;
        case at::kUInt32: return std::numeric_limits<uint32_t>::digits;
        case at::kLong: return std::numeric_limits<int64_t>::digits;
        case at::kUInt64: return std::numeric_limits<uint64_t>::digits;

        default: TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported dtype ", dtype, ".");
    }
}

inline std::pair<double, double> get_dtype_limits(at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
        case at::kFloat: return {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max()};
        case at::kHalf: return {static_cast<double>(std::numeric_limits<c10::Half>::lowest()), static_cast<double>(std::numeric_limits<c10::Half>::max())};
        case at::kBFloat16: return {static_cast<double>(std::numeric_limits<c10::BFloat16>::lowest()), static_cast<double>(std::numeric_limits<c10::BFloat16>::max())};

        case at::kBool: return {0, 1};
        case at::kByte: return {std::numeric_limits<uint8_t>::lowest(), std::numeric_limits<uint8_t>::max()};
        case at::kChar: return {std::numeric_limits<int8_t>::lowest(), std::numeric_limits<int8_t>::max()};
        case at::kShort: return {std::numeric_limits<int16_t>::lowest(), std::numeric_limits<int16_t>::max()};
        case at::kUInt16: return {std::numeric_limits<uint16_t>::lowest(), std::numeric_limits<uint16_t>::max()};
        case at::kInt: return {std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max()};
        case at::kUInt32: return {std::numeric_limits<uint32_t>::lowest(), std::numeric_limits<uint32_t>::max()};
        case at::kLong: return {static_cast<double>(std::numeric_limits<int64_t>::lowest()), static_cast<double>(std::numeric_limits<int64_t>::max())};
        case at::kUInt64: return {0, static_cast<double>(std::numeric_limits<uint64_t>::max())};

        default: TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported dtype ", dtype, ".");
    }
}

inline double get_dtype_epsilon(at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return std::numeric_limits<double>::epsilon();
        case at::kHalf: return static_cast<double>(std::numeric_limits<c10::Half>::epsilon());
        case at::kBFloat16: return static_cast<double>(std::numeric_limits<c10::BFloat16>::epsilon());
        default: return std::numeric_limits<float>::epsilon();
    }
}

inline double get_dtype_smallest_normal(at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return std::numeric_limits<double>::min();
        case at::kHalf: return static_cast<double>(std::numeric_limits<c10::Half>::min());
        case at::kBFloat16: return static_cast<double>(std::numeric_limits<c10::BFloat16>::min());
        default: return std::numeric_limits<float>::min();
    }
}

inline int64_t round_to_dtype(int64_t value, at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return static_cast<int64_t>(static_cast<double>(value));
        case at::kHalf: return static_cast<int64_t>(static_cast<float>(c10::Half(static_cast<float>(value))));
        case at::kBFloat16: return static_cast<int64_t>(static_cast<float>(c10::BFloat16(static_cast<float>(value))));
        default: return static_cast<int64_t>(static_cast<float>(value));
    }
}

inline void check_floating(const at::Tensor& self, const char* name)
{
    TORCH_CHECK_NOT_IMPLEMENTED(at::isFloatingType(self.scalar_type()), name, " is not implemented for '", self.scalar_type(), "'.");
}

inline std::vector<int64_t> compute_strides(
    c10::IntArrayRef sizes, 
    at::MemoryFormat format) 
{
    if (format == at::MemoryFormat::Contiguous) {
        auto strides = c10::contiguous_strides(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    if (format == at::MemoryFormat::ChannelsLast && sizes.size() == 4) {
        auto strides = c10::get_channels_last_strides_2d(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    if (format == at::MemoryFormat::ChannelsLast3d && sizes.size() == 5) {
        auto strides = c10::get_channels_last_strides_3d(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported memory format for sizes.");
}

inline at::Tensor promote_to_float(const at::Tensor& self)
{
    if (self.is_floating_point() || self.is_complex()) return self;
    return self.to(c10::typeMetaToScalarType(at::get_default_dtype()));
}

// copies an op result into an out= tensor, resizing it first when the shapes differ
inline at::Tensor& fill_out(at::Tensor& out, const at::Tensor& res)
{
    if (out.sizes() != res.sizes()) out.resize_(res.sizes());
    out.copy_(res);
    return out;
}

// integer reductions accumulate in int64 to avoid overflow
inline c10::ScalarType reduce_compute_dtype(const at::Tensor& self, c10::optional<at::ScalarType> dtype)
{
    if (dtype.has_value()) return *dtype;
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true) && self.scalar_type() != at::kLong) {
        return at::kLong;
    }
    return self.scalar_type();
}

// the dtype a reduction accumulates in is not always the dtype it returns
inline c10::ScalarType reduce_accumulate_dtype(c10::ScalarType dtype)
{
    return (dtype == at::kHalf || dtype == at::kBFloat16) ? at::kFloat : dtype;
}

// the dtype a random kernel samples in: float64 for float64 outputs, float32 otherwise (T.Real in philox.slang)
inline c10::ScalarType random_real_dtype(c10::ScalarType dtype)
{
    return dtype == at::kDouble ? at::kDouble : at::kFloat;
}

// the dtype a random kernel generates in when the device cannot write `dtype`
inline c10::ScalarType random_proxy_dtype(c10::ScalarType dtype)
{
    if (at::isFloatingType(dtype)) return at::kFloat;
    return is_dtype_supported(at::kLong) ? at::kLong : at::kInt;
}

// a parameter packed for IRandomReal.fromBits: a float in the low 32 bits, or a whole double
inline uint64_t random_real_bits(double value, c10::ScalarType real)
{
    if (real == at::kDouble) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(double));
        return bits;
    }

    float narrowed = static_cast<float>(value);
    uint32_t bits = 0;
    std::memcpy(&bits, &narrowed, sizeof(float));
    return bits;
}
