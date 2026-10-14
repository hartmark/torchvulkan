#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <tuple>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

enum class SortMode : uint32_t {
    LOAD = 0,
    STEP = 1
};

std::optional<std::tuple<at::Tensor, at::Tensor>> dispatch_sort_shader(const at::Tensor& lines, bool descending)
{
    DeviceContext* device = get_device_context(lines);
    int64_t L = lines.size(0);
    int64_t n = lines.size(1);
    int64_t padded = 1;
    while (padded < n) padded <<= 1;

    if (!is_dtype_supported(device, lines.scalar_type()) || L * padded > std::numeric_limits<int32_t>::max()) return std::nullopt;

    // two copies of the keys and positions: each step reads one and writes the other
    at::Tensor keys[2] = {at::empty({L, padded}, lines.options()), at::empty({L, padded}, lines.options())};
    at::Tensor positions[2] = {at::empty({L, padded}, lines.options().dtype(at::kLong)), at::empty({L, padded}, lines.options().dtype(at::kLong))};
    int current = 0;
    uint32_t workgroupSizeX = get_dtype_workgroup_size(lines.scalar_type(), 1, device);

    auto dispatch = [&](SortMode mode, uint64_t threads, uint32_t block, uint32_t distance) {
        int next = mode == SortMode::LOAD ? current : 1 - current;

        SpecializationBuilder spd{};
        spd.push(static_cast<uint32_t>(mode))
           .push(static_cast<uint32_t>(descending))
           .push(workgroupSizeX);

        PushConstantBuilder pcs{};
        pcs.push(get_tensor_address(keys[current]))
           .push(get_tensor_address(positions[current]))
           .push(get_tensor_address(keys[next]))
           .push(get_tensor_address(positions[next]))
           .push(get_tensor_address(lines))
           .push(threads)
           .push(static_cast<uint32_t>(n))
           .push(static_cast<uint32_t>(padded))
           .push(block)
           .push(distance);

        std::vector<at::Tensor> reads = {keys[current], positions[current]};
        if (mode == SortMode::LOAD) reads = {lines};

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::SORT, lines.scalar_type()},
            spd.build(),
            pcs.build(),
            /* read = */ reads,
            /* write = */ {keys[next], positions[next]},
            get_group_count(threads, workgroupSizeX), 1, 1
        );
        current = next;
    };

    dispatch(SortMode::LOAD, static_cast<uint64_t>(L * padded), 0, 0);
    for (int64_t block = 2; block <= padded; block <<= 1) {
        for (int64_t distance = block >> 1; distance > 0; distance >>= 1) {
            dispatch(SortMode::STEP, static_cast<uint64_t>(L * padded / 2), static_cast<uint32_t>(block), static_cast<uint32_t>(distance));
        }
    }

    return std::make_tuple(keys[current].narrow(1, 0, n), positions[current].narrow(1, 0, n));
}

std::tuple<at::Tensor, at::Tensor> sort_along(const at::Tensor& self, int64_t dim, bool descending)
{
    if (self.dim() == 0) return std::make_tuple(self.clone(), at::zeros({}, self.options().dtype(at::kLong)));

    dim = c10::maybe_wrap_dim(dim, self.dim());
    int64_t n = self.size(dim);
    if (self.numel() == 0 || n == 1) {
        return std::make_tuple(self.clone(), at::zeros(self.sizes(), self.options().dtype(at::kLong)));
    }

    at::Tensor moved = self.movedim(dim, -1).contiguous();
    at::Tensor lines = moved.view({-1, n});

    auto sorted = dispatch_sort_shader(lines, descending);
    if (!sorted.has_value()) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: A ", self.scalar_type(), " sort of this size can't run on the Vulkan device. Falling back to CPU.");
        auto [values, indices] = at::sort(self.cpu(), /*stable=*/true, dim, descending);
        return std::make_tuple(values.to(self.device()), indices.to(self.device()));
    }

    auto [values, indices] = *sorted;
    return std::make_tuple(values.reshape(moved.sizes()).movedim(-1, dim), indices.reshape(moved.sizes()).movedim(-1, dim));
}

std::tuple<at::Tensor&, at::Tensor&> sort_stable_out_vulkan(
    const at::Tensor& self,
    std::optional<bool> /* stable */,
    int64_t dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices)
{
    auto [sorted_values, sorted_indices] = sort_along(self, dim, descending);
    fill_out(values, sorted_values);
    fill_out(indices, sorted_indices);
    return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> topk_out_vulkan(
    const at::Tensor& self,
    c10::SymInt k_,
    int64_t dim,
    bool largest,
    bool /* sorted */,
    at::Tensor& values,
    at::Tensor& indices)
{
    int64_t k = k_.guard_int(__FILE__, __LINE__);
    int64_t d = c10::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
    int64_t size = self.dim() == 0 ? 1 : self.size(d);
    TORCH_CHECK(k >= 0 && k <= size, "selected index k out of range");

    auto [sorted_values, sorted_indices] = sort_along(self, d, /*descending=*/largest);
    if (self.dim() > 0) {
        sorted_values = sorted_values.narrow(d, 0, k);
        sorted_indices = sorted_indices.narrow(d, 0, k);
    }
    fill_out(values, sorted_values);
    fill_out(indices, sorted_indices);
    return std::forward_as_tuple(values, indices);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("sort.values_stable", &sort_stable_out_vulkan);
    m.impl("topk.values", &topk_out_vulkan);
}
