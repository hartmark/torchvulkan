#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <cmath>
#include <vector>
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

at::Tensor& fill_complex(at::Tensor& out, const at::Tensor& real, const at::Tensor& imag)
{
    std::vector<int64_t> shape = at::infer_size(real.sizes(), imag.sizes());
    if (out.sizes() != shape) out.resize_(shape);
    at::Tensor pairs = at::view_as_real(out);
    pairs.select(-1, 0).copy_(real.expand(shape));
    pairs.select(-1, 1).copy_(imag.expand(shape));
    return out;
}

at::Tensor& complex_out_vulkan(const at::Tensor& real, const at::Tensor& imag, at::Tensor& out)
{
    TORCH_CHECK(real.scalar_type() == imag.scalar_type(), "Expected object of scalar type ", real.scalar_type(), " but got scalar type ", imag.scalar_type(),
        " for second argument"
    );
    TORCH_CHECK(real.scalar_type() == at::kFloat || real.scalar_type() == at::kDouble || real.scalar_type() == at::kHalf,
        "Expected both inputs to be Half, Float or Double tensors but got ", real.scalar_type(), " and ", imag.scalar_type()
    );
    return fill_complex(out, real, imag);
}

at::Tensor& polar_out_vulkan(const at::Tensor& abs, const at::Tensor& angle, at::Tensor& out)
{
    TORCH_CHECK(abs.scalar_type() == angle.scalar_type(), "Expected object of scalar type ", abs.scalar_type(), " but got scalar type ", angle.scalar_type(),
        " for second argument"
    );
    return fill_complex(out, at::mul(abs, at::cos(angle)), at::mul(abs, at::sin(angle)));
}

at::Tensor angle_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) {
        at::Tensor pairs = at::view_as_real(self.resolve_conj());
        return at::atan2(pairs.select(-1, 1), pairs.select(-1, 0));
    }
    at::Tensor x = promote_to_float(self);
    at::Tensor result = at::where(at::lt(x, 0), at::full_like(x, M_PI), at::zeros_like(x));
    return at::where(at::isnan(x), x, result);
}

at::Tensor& angle_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, angle_vulkan(self));
}

at::Tensor& conj_physical_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    if (!self.is_complex()) return fill_out(out, self);
    at::Tensor result = self.resolve_conj().clone();
    at::view_as_real(result).select(-1, 1).neg_();
    return fill_out(out, result);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("view_as_real", &at::native::view_as_real);
    m.impl("view_as_complex", &at::native::view_as_complex);
    m.impl("complex.out", &complex_out_vulkan);
    m.impl("polar.out", &polar_out_vulkan);
    m.impl("angle", &angle_vulkan);
    m.impl("angle.out", &angle_out_vulkan);
    m.impl("conj_physical.out", &conj_physical_out_vulkan);
}
