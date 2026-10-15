#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

at::Tensor defined_or_empty(const c10::optional<at::Tensor>& t)
{
    return t.has_value() ? *t : at::Tensor();
}

std::vector<int64_t> batch_norm_dims(const at::Tensor& input)
{
    std::vector<int64_t> dims = {0};
    for (int64_t d = 2; d < input.dim(); d++) dims.push_back(d);
    return dims;
}

std::vector<int64_t> channel_view(const at::Tensor& input)
{
    std::vector<int64_t> shape(input.dim(), 1);
    shape[1] = input.size(1);
    return shape;
}

at::Tensor batch_norm_apply(const at::Tensor& x, const at::Tensor& mean, const at::Tensor& invstd, const at::Tensor& weight, const at::Tensor& bias)
{
    std::vector<int64_t> view = channel_view(x);
    at::Tensor y = at::mul(at::sub(x, mean.view(view)), invstd.view(view));
    if (weight.defined()) y = at::mul(y, weight.to(x.scalar_type()).view(view));
    if (bias.defined()) y = at::add(y, bias.to(x.scalar_type()).view(view));
    return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double momentum,
    double eps)
{
    TORCH_CHECK(input.dim() >= 2, "batch_norm: expected input to have at least 2 dimensions, got ", input.dim());
    at::ScalarType dtype = input.scalar_type();
    at::Tensor x = is_low_prec(dtype) ? input.to(at::kFloat) : input;
    std::vector<int64_t> dims = batch_norm_dims(input);

    if (!training) {
        TORCH_CHECK(running_mean.defined() && running_var.defined(), "running_mean and running_var must be defined in evaluation mode");
        at::Tensor invstd = at::rsqrt(at::add(running_var.to(x.scalar_type()), eps));
        at::Tensor y = batch_norm_apply(x, running_mean.to(x.scalar_type()), invstd, weight, bias);
        at::Tensor empty = at::empty({0}, input.options());
        return std::make_tuple(y.to(dtype), empty, empty.clone());
    }

    int64_t count = input.numel() / std::max<int64_t>(input.size(1), 1);
    TORCH_CHECK(count > 1 || input.numel() == 0, "Expected more than 1 value per channel when training, got input size ", input.sizes());
    at::Tensor mean = at::mean(x, dims);
    at::Tensor var = at::mean(at::square(at::sub(x, mean.view(channel_view(x)))), dims);
    at::Tensor invstd = at::rsqrt(at::add(var, eps));
    at::Tensor y = batch_norm_apply(x, mean, invstd, weight, bias);

    if (running_mean.defined()) {
        at::Tensor updated = at::add(at::mul(running_mean, 1.0 - momentum), at::mul(mean.to(running_mean.scalar_type()), momentum));
        running_mean.copy_(updated);
    }
    if (running_var.defined()) {
        double unbias = count > 1 ? static_cast<double>(count) / static_cast<double>(count - 1) : 1.0;
        at::Tensor updated = at::add(at::mul(running_var, 1.0 - momentum), at::mul(at::mul(var, unbias).to(running_var.scalar_type()), momentum));
        running_var.copy_(updated);
    }
    return std::make_tuple(y.to(dtype), mean.to(dtype), invstd.to(dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps)
{
    return batch_norm_forward(input, defined_or_empty(weight), defined_or_empty(bias), defined_or_empty(running_mean), defined_or_empty(running_var),
        training, momentum, eps);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    at::Tensor& out,
    at::Tensor& save_mean,
    at::Tensor& save_invstd)
{
    auto [y, mean, invstd] = native_batch_norm_vulkan(input, weight, bias, running_mean, running_var, training, momentum, eps);
    fill_out(out, y);
    fill_out(save_mean, mean);
    fill_out(save_invstd, invstd);
    return std::forward_as_tuple(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_legit_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    bool training,
    double momentum,
    double eps)
{
    return batch_norm_forward(input, defined_or_empty(weight), defined_or_empty(bias), running_mean, running_var, training, momentum, eps);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_legit_out_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    bool training,
    double momentum,
    double eps,
    at::Tensor& out,
    at::Tensor& save_mean,
    at::Tensor& save_invstd)
{
    auto [y, mean, invstd] = native_batch_norm_legit_vulkan(input, weight, bias, running_mean, running_var, training, momentum, eps);
    fill_out(out, y);
    fill_out(save_mean, mean);
    fill_out(save_invstd, invstd);
    return std::forward_as_tuple(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_legit_no_stats_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    bool training,
    double momentum,
    double eps)
{
    return batch_norm_forward(input, defined_or_empty(weight), defined_or_empty(bias), at::Tensor(), at::Tensor(), training, momentum, eps);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_legit_no_stats_out_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    bool training,
    double momentum,
    double eps,
    at::Tensor& out,
    at::Tensor& save_mean,
    at::Tensor& save_invstd)
{
    auto [y, mean, invstd] = native_batch_norm_legit_no_stats_vulkan(input, weight, bias, training, momentum, eps);
    fill_out(out, y);
    fill_out(save_mean, mean);
    fill_out(save_invstd, invstd);
    return std::forward_as_tuple(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_norm_with_update_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    double momentum,
    double eps)
{
    auto [y, mean, invstd] = batch_norm_forward(input, defined_or_empty(weight), defined_or_empty(bias), running_mean, running_var, true, momentum, eps);
    return std::make_tuple(y, mean, invstd, at::empty({0}, input.options().dtype(at::kByte)));
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> batch_norm_with_update_out_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    double momentum,
    double eps,
    at::Tensor& out,
    at::Tensor& save_mean,
    at::Tensor& save_invstd,
    at::Tensor& reserve)
{
    auto [y, mean, invstd, r] = batch_norm_with_update_vulkan(input, weight, bias, running_mean, running_var, momentum, eps);
    fill_out(out, y);
    fill_out(save_mean, mean);
    fill_out(save_invstd, invstd);
    fill_out(reserve, r);
    return std::forward_as_tuple(out, save_mean, save_invstd, reserve);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward_impl(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask)
{
    at::ScalarType dtype = input.scalar_type();
    at::ScalarType compute = is_low_prec(dtype) ? at::kFloat : dtype;
    at::Tensor x = input.to(compute);
    at::Tensor dy = grad_out.to(compute);
    std::vector<int64_t> dims = batch_norm_dims(input);
    std::vector<int64_t> view = channel_view(input);

    at::Tensor mean = train ? save_mean.to(compute) : running_mean.to(compute);
    at::Tensor invstd = train ? save_invstd.to(compute) : at::rsqrt(at::add(running_var.to(compute), eps));
    at::Tensor w = weight.defined() ? weight.to(compute) : at::ones({input.size(1)}, x.options());
    at::Tensor xhat = at::mul(at::sub(x, mean.view(view)), invstd.view(view));

    at::Tensor grad_input, grad_weight, grad_bias;
    if (output_mask[0]) {
        at::Tensor scale = at::mul(w, invstd).view(view);
        if (train) {
            at::Tensor mean_dy = at::mean(dy, dims, /*keepdim=*/true);
            at::Tensor mean_dy_xhat = at::mean(at::mul(dy, xhat), dims, /*keepdim=*/true);
            grad_input = at::mul(scale, at::sub(at::sub(dy, mean_dy), at::mul(xhat, mean_dy_xhat))).to(dtype);
        } else {
            grad_input = at::mul(dy, scale).to(dtype);
        }
    }
    if (output_mask[1]) grad_weight = at::sum(at::mul(dy, xhat), dims).to(weight.defined() ? weight.scalar_type() : dtype);
    if (output_mask[2]) grad_bias = at::sum(dy, dims).to(weight.defined() ? weight.scalar_type() : dtype);
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward_vulkan(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask)
{
    return batch_norm_backward_impl(grad_out, input, defined_or_empty(weight), defined_or_empty(running_mean), defined_or_empty(running_var),
        defined_or_empty(save_mean), defined_or_empty(save_invstd), train, eps, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward_vulkan(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_var,
    bool update,
    double eps,
    std::array<bool, 3> output_mask,
    const at::Tensor& /* reserve */)
{
    return batch_norm_backward_impl(grad_out, input, weight, defined_or_empty(running_mean), defined_or_empty(running_var),
        defined_or_empty(save_mean), defined_or_empty(save_var), update, eps, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward_vulkan(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    c10::SymInt N_,
    c10::SymInt C_,
    c10::SymInt HxW_,
    int64_t group,
    std::array<bool, 3> output_mask)
{
    int64_t n = N_.guard_int(__FILE__, __LINE__);
    int64_t c = C_.guard_int(__FILE__, __LINE__);
    int64_t hw = HxW_.guard_int(__FILE__, __LINE__);
    at::ScalarType dtype = input.scalar_type();
    at::ScalarType compute = is_low_prec(dtype) ? at::kFloat : dtype;

    at::Tensor x = input.to(compute).reshape({n, group, c / group, hw});
    at::Tensor dy = grad_out.to(compute).reshape({n, group, c / group, hw});
    at::Tensor m = mean.to(compute).reshape({n, group, 1, 1});
    at::Tensor r = rstd.to(compute).reshape({n, group, 1, 1});
    at::Tensor xhat = at::mul(at::sub(x, m), r);
    bool has_weight = weight.has_value() && weight->defined();
    at::Tensor w = has_weight ? weight->to(compute).reshape({1, group, c / group, 1}) : at::Tensor();

    at::Tensor grad_input, grad_weight, grad_bias;
    if (output_mask[0]) {
        at::Tensor dyw = has_weight ? at::mul(dy, w) : dy;
        at::Tensor mean_dyw = at::mean(dyw, at::IntArrayRef{2, 3}, /*keepdim=*/true);
        at::Tensor mean_dyw_xhat = at::mean(at::mul(dyw, xhat), at::IntArrayRef{2, 3}, /*keepdim=*/true);
        grad_input = at::mul(r, at::sub(at::sub(dyw, mean_dyw), at::mul(xhat, mean_dyw_xhat))).reshape(input.sizes()).to(dtype);
    }
    at::ScalarType param_dtype = has_weight ? weight->scalar_type() : dtype;
    if (output_mask[1]) grad_weight = at::sum(at::mul(dy, xhat), at::IntArrayRef{0, 3}).reshape({c}).to(param_dtype);
    if (output_mask[2]) grad_bias = at::sum(dy, at::IntArrayRef{0, 3}).reshape({c}).to(param_dtype);
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor pairwise_p_norm(const at::Tensor& x1, const at::Tensor& x2, double p)
{
    at::Tensor diff = at::abs(at::sub(x1.unsqueeze(-2), x2.unsqueeze(-3)));
    if (p == 0.0) return at::sum(at::ne(diff, 0).to(diff.scalar_type()), -1);
    if (p == 1.0) return at::sum(diff, -1);
    if (p == 2.0) return at::sqrt(at::sum(at::mul(diff, diff), -1));
    if (std::isinf(p)) return at::amax(diff, -1);
    return at::pow(at::sum(at::pow(diff, p), -1), 1.0 / p);
}

// the CPU's matrix-multiply form: |a|^2 - 2ab + |b|^2, clamped at 0
at::Tensor euclidean_dist(const at::Tensor& x1, const at::Tensor& x2)
{
    at::Tensor x1_norm = at::sum(at::mul(x1, x1), -1, /*keepdim=*/true);
    at::Tensor x2_norm = at::sum(at::mul(x2, x2), -1, /*keepdim=*/true);
    at::Tensor x1_ = at::cat({at::mul(x1, -2), x1_norm, at::ones_like(x1_norm)}, -1);
    at::Tensor x2_ = at::cat({x2, at::ones_like(x2_norm), x2_norm}, -1);
    return at::sqrt(at::clamp_min(at::matmul(x1_, x2_.transpose(-1, -2)), 1e-30));
}

at::Tensor cdist_forward_vulkan(const at::Tensor& x1, const at::Tensor& x2, double p, c10::optional<int64_t> compute_mode)
{
    TORCH_CHECK(x1.dim() >= 2 && x2.dim() >= 2, "cdist only supports at least 2D tensors");
    TORCH_CHECK(x1.size(-1) == x2.size(-1), "X1 and X2 must have the same number of columns. X1: ", x1.size(-1), " X2: ", x2.size(-1));
    TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
    int64_t r1 = x1.size(-2), r2 = x2.size(-2);
    int64_t mode = compute_mode.value_or(0);
    bool use_mm = p == 2.0 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)));

    std::vector<int64_t> batch = at::infer_size(x1.sizes().slice(0, x1.dim() - 2), x2.sizes().slice(0, x2.dim() - 2));
    std::vector<int64_t> s1 = batch, s2 = batch;
    s1.insert(s1.end(), {r1, x1.size(-1)});
    s2.insert(s2.end(), {r2, x2.size(-1)});
    at::Tensor a = x1.expand(s1), b = x2.expand(s2);
    if (x1.size(-1) == 0) {
        std::vector<int64_t> out_shape = batch;
        out_shape.insert(out_shape.end(), {r1, r2});
        return at::zeros(out_shape, x1.options());
    }
    return use_mm ? euclidean_dist(a, b) : pairwise_p_norm(a, b, p);
}

at::Tensor cdist_backward_vulkan(const at::Tensor& grad, const at::Tensor& x1, const at::Tensor& x2, double p, const at::Tensor& cdist)
{
    if (p == 0.0 || grad.numel() == 0 || x1.numel() == 0) return at::zeros_like(x1);
    at::Tensor diff = at::sub(x1.unsqueeze(-2), x2.unsqueeze(-3));
    at::Tensor dist = cdist.unsqueeze(-1);
    at::Tensor deriv;
    if (p == 1.0) {
        deriv = at::sign(diff);
    } else if (p == 2.0) {
        deriv = at::where(at::eq(dist, 0), at::zeros_like(diff), at::div(diff, dist));
    } else if (std::isinf(p)) {
        deriv = at::mul(at::sign(diff), at::eq(at::abs(diff), dist).to(diff.scalar_type()));
    } else if (p < 1.0) {
        deriv = at::where(at::logical_or(at::eq(dist, 0), at::eq(diff, 0)), at::zeros_like(diff),
            at::div(at::mul(at::sign(diff), at::pow(at::abs(diff), p - 1)), at::pow(dist, p - 1)));
    } else {
        deriv = at::where(at::eq(dist, 0), at::zeros_like(diff), at::div(at::mul(diff, at::pow(at::abs(diff), p - 2)), at::pow(dist, p - 1)));
    }
    return at::sum_to(at::sum(at::mul(deriv, grad.unsqueeze(-1)), -2), x1.sizes());
}

at::Tensor pdist_forward_vulkan(const at::Tensor& self, double p)
{
    TORCH_CHECK(self.dim() == 2, "pdist only supports 2D tensors, got: ", self.dim(), "D");
    TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");
    int64_t n = self.size(0);
    if (n <= 1) return at::empty({0}, self.options());
    at::Tensor full = pairwise_p_norm(self, self, p);
    at::Tensor pairs = at::triu_indices(n, n, 1, self.options().dtype(at::kLong));
    return at::index(full, {pairs[0], pairs[1]});
}

at::Tensor pdist_backward_vulkan(const at::Tensor& grad, const at::Tensor& self, double p, const at::Tensor& pdist)
{
    int64_t n = self.size(0);
    if (n <= 1 || grad.numel() == 0) return at::zeros_like(self);
    at::Tensor pairs = at::triu_indices(n, n, 1, self.options().dtype(at::kLong));
    c10::List<std::optional<at::Tensor>> upper({pairs[0], pairs[1]});
    c10::List<std::optional<at::Tensor>> lower({pairs[1], pairs[0]});
    at::Tensor g = at::zeros({n, n}, grad.options()).index_put_(upper, grad).index_put_(lower, grad);
    at::Tensor d = at::zeros({n, n}, pdist.options()).index_put_(upper, pdist).index_put_(lower, pdist);
    return cdist_backward_vulkan(g, self, self, p, d);
}

std::tuple<at::Tensor, at::Tensor> nll_loss2d_forward_vulkan(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index)
{
    TORCH_CHECK(self.dim() == 4, "only batches of spatial inputs supported (4D tensors), but got input of dimension: ", self.dim());
    TORCH_CHECK(target.dim() == 3, "only batches of spatial targets supported (3D tensors) but got targets of dimension: ", target.dim());
    TORCH_CHECK(self.size(0) == target.size(0) && self.size(2) == target.size(1) && self.size(3) == target.size(2),
        "size mismatch (got input: ", self.sizes(), " , target: ", target.sizes()
    );
    int64_t c = self.size(1);
    at::Tensor flat = self.permute({0, 2, 3, 1}).reshape({-1, c});
    auto [output, total_weight] = at::nll_loss_forward(flat, target.reshape({-1}), weight, reduction, ignore_index);
    if (reduction == 0) output = output.view(target.sizes());
    return std::make_tuple(output, total_weight);
}

std::tuple<at::Tensor&, at::Tensor&> nll_loss2d_forward_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& output,
    at::Tensor& total_weight)
{
    auto [o, w] = nll_loss2d_forward_vulkan(self, target, weight, reduction, ignore_index);
    fill_out(output, o);
    fill_out(total_weight, w);
    return std::forward_as_tuple(output, total_weight);
}

at::Tensor nll_loss2d_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight)
{
    int64_t c = self.size(1);
    at::Tensor flat = self.permute({0, 2, 3, 1}).reshape({-1, c});
    at::Tensor grad = reduction == 0 ? grad_output.reshape({-1}) : grad_output;
    at::Tensor flat_grad = at::nll_loss_backward(grad, flat, target.reshape({-1}), weight, reduction, ignore_index, total_weight);
    return flat_grad.view({self.size(0), self.size(2), self.size(3), c}).permute({0, 3, 1, 2}).contiguous();
}

at::Tensor& nll_loss2d_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, nll_loss2d_backward_vulkan(grad_output, self, target, weight, reduction, ignore_index, total_weight));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_margin_terms(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Scalar& p,
    const at::Scalar& margin,
    const c10::optional<at::Tensor>& weight)
{
    at::Tensor x = self.dim() <= 1 ? self.reshape({1, -1}) : self;
    at::Tensor y = target.reshape({-1, 1}).to(at::kLong);
    int64_t c = x.size(1);
    TORCH_CHECK(p.toLong() == 1 || p.toLong() == 2, "only p == 1 and p == 2 supported");
    TORCH_CHECK(x.size(0) == y.size(0), "inconsistent target size, expected ", x.size(0), " but got ", target.sizes());
    if (y.numel() > 0) {
        bool invalid = at::logical_or(at::lt(y, 0), at::ge(y, c)).any().item<bool>();
        TORCH_CHECK(!invalid, "target out of range");
    }
    at::Tensor x_target = at::gather(x, 1, y);
    at::Tensor z = at::add(at::sub(x, x_target), margin);
    at::Tensor not_target = at::ne(at::arange(c, y.options()).view({1, c}), y);
    at::Tensor active = at::logical_and(at::gt(z, 0), not_target);
    at::Tensor scale = at::ones({x.size(0), 1}, x.options());
    if (weight.has_value() && weight->defined()) scale = at::index_select(weight->to(x.scalar_type()), 0, y.reshape({-1})).view({-1, 1});
    return std::make_tuple(z, active, scale);
}

at::Tensor multi_margin_loss_vulkan(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Scalar& p,
    const at::Scalar& margin,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction)
{
    auto [z, active, scale] = multi_margin_terms(self, target, p, margin, weight);
    int64_t c = z.size(1);
    at::Tensor h = p.toLong() == 1 ? z : at::mul(z, z);
    h = at::where(active, at::mul(h, scale), at::zeros_like(h));
    at::Tensor loss = at::div(at::sum(h, 1), static_cast<double>(c));
    if (self.dim() <= 1 && reduction == 0) loss = loss.reshape({});
    return reduce_loss(loss, reduction);
}

at::Tensor& multi_margin_loss_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Scalar& p,
    const at::Scalar& margin,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    at::Tensor& out)
{
    return fill_out(out, multi_margin_loss_vulkan(self, target, p, margin, weight, reduction));
}

at::Tensor multi_margin_loss_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Scalar& p,
    const at::Scalar& margin,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction)
{
    auto [z, active, scale] = multi_margin_terms(self, target, p, margin, weight);
    int64_t n = z.size(0), c = z.size(1);
    double norm = reduction == 1 ? 1.0 / (static_cast<double>(c) * std::max<int64_t>(n, 1)) : 1.0 / static_cast<double>(c);
    at::Tensor g = reduction == 0 ? grad_output.reshape({-1, 1}) : grad_output.reshape({1, 1});
    at::Tensor h = p.toLong() == 1 ? at::ones_like(z) : at::mul(z, 2);
    h = at::where(active, at::mul(at::mul(at::mul(h, scale), g), norm), at::zeros_like(h));
    at::Tensor grad = h.scatter_add(1, target.reshape({-1, 1}).to(at::kLong), at::neg(at::sum(h, 1, /*keepdim=*/true)));
    return grad.reshape(self.sizes());
}

at::Tensor& multi_margin_loss_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Scalar& p,
    const at::Scalar& margin,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, multi_margin_loss_backward_vulkan(grad_output, self, target, p, margin, weight, reduction));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> multilabel_terms(const at::Tensor& self, const at::Tensor& target)
{
    at::Tensor x = self.dim() <= 1 ? self.reshape({1, -1}) : self;
    at::Tensor t = target.dim() <= 1 ? target.reshape({1, -1}) : target;
    int64_t c = x.size(1);
    TORCH_CHECK(t.sizes() == x.sizes(), "inconsistent target size: ", target.sizes(), " for input of size: ", self.sizes());
    at::Tensor valid = at::cumprod(at::ne(t, -1).to(at::kLong), 1).to(at::kBool);
    at::Tensor labels = at::where(valid, t, at::zeros_like(t));
    if (labels.numel() > 0) {
        bool invalid = at::logical_and(valid, at::logical_or(at::lt(t, 0), at::ge(t, c))).any().item<bool>();
        TORCH_CHECK(!invalid, "target out of range");
    }

    at::Tensor classes = at::arange(c, t.options()).view({1, 1, c});
    at::Tensor is_target = at::logical_and(at::eq(labels.unsqueeze(-1), classes), valid.unsqueeze(-1)).any(1);
    at::Tensor x_label = at::gather(x, 1, labels);
    at::Tensor hinge = at::add(at::sub(x.unsqueeze(1), x_label.unsqueeze(-1)), 1);
    at::Tensor active = at::logical_and(at::logical_and(at::gt(hinge, 0), valid.unsqueeze(-1)), at::logical_not(is_target).unsqueeze(1));
    return std::make_tuple(hinge, active, is_target);
}

std::tuple<at::Tensor, at::Tensor> multilabel_margin_loss_forward_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction)
{
    auto [hinge, active, is_target] = multilabel_terms(self, target);
    int64_t c = hinge.size(2);
    at::Tensor loss = at::div(at::sum(at::where(active, hinge, at::zeros_like(hinge)), at::IntArrayRef{1, 2}), static_cast<double>(c));
    if (self.dim() <= 1) loss = loss.reshape({});
    return std::make_tuple(reduce_loss(loss, reduction), is_target.to(self.scalar_type()).view(self.sizes()));
}

std::tuple<at::Tensor&, at::Tensor&> multilabel_margin_loss_forward_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& output,
    at::Tensor& is_target)
{
    auto [o, t] = multilabel_margin_loss_forward_vulkan(self, target, reduction);
    fill_out(output, o);
    fill_out(is_target, t);
    return std::forward_as_tuple(output, is_target);
}

at::Tensor multilabel_margin_loss_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    const at::Tensor& /* is_target */)
{
    auto [hinge, active, is_target] = multilabel_terms(self, target);
    int64_t n = hinge.size(0), c = hinge.size(2);
    double norm = reduction == 1 ? 1.0 / (static_cast<double>(c) * std::max<int64_t>(n, 1)) : 1.0 / static_cast<double>(c);
    at::Tensor g = reduction == 0 ? grad_output.reshape({-1, 1, 1}) : grad_output.reshape({1, 1, 1});
    at::Tensor h = at::where(active, at::mul(g, norm).expand(hinge.sizes()), at::zeros_like(hinge));
    at::Tensor t = target.dim() <= 1 ? target.reshape({1, -1}) : target;
    at::Tensor labels = at::where(at::cumprod(at::ne(t, -1).to(at::kLong), 1).to(at::kBool), t, at::zeros_like(t));
    at::Tensor grad = at::sum(h, 1).scatter_add(1, labels.to(at::kLong), at::neg(at::sum(h, 2)));
    return grad.reshape(self.sizes());
}

at::Tensor& multilabel_margin_loss_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    const at::Tensor& is_target,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, multilabel_margin_loss_backward_vulkan(grad_output, self, target, reduction, is_target));
}

at::Tensor offsets_to_bags(const at::Tensor& offsets, int64_t count, bool include_last_offset)
{
    at::Tensor starts = include_last_offset ? offsets.narrow(0, 0, offsets.size(0) - 1) : offsets;
    at::Tensor positions = at::arange(count, offsets.options().dtype(at::kLong));
    return at::sub(at::searchsorted(starts.to(at::kLong).contiguous(), positions, /*out_int32=*/false, /*right=*/true), 1);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> embedding_bag_impl(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    int64_t mode,
    const c10::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx)
{
    TORCH_CHECK(indices.dim() == 1, "input has to be a 1D or 2D Tensor, but got Tensor of dimension ", indices.dim());
    TORCH_CHECK(offsets.dim() == 1, "offsets has to be a 1D Tensor, but got Tensor of dimension ", offsets.dim());
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode has to be one of sum, mean or max");
    bool has_psw = per_sample_weights.has_value() && per_sample_weights->defined();
    TORCH_CHECK(!has_psw || mode == 0, "embedding_bag: per_sample_weights was not None. per_sample_weights is only supported for mode='sum' (got mode='",
        mode == 1 ? "mean" : "max", "'). Please open a feature request on GitHub."
    );

    int64_t count = indices.numel();
    int64_t bags = offsets.size(0) - (include_last_offset ? 1 : 0);
    int64_t dim = weight.size(1);
    at::Tensor idx = indices.to(at::kLong);
    at::Tensor bag = offsets_to_bags(offsets, count, include_last_offset);
    if (include_last_offset && count > 0) bag = at::clamp_max(bag, bags - 1);
    at::Tensor valid = padding_idx >= 0 ? at::ne(idx, padding_idx) : at::ones({count}, idx.options().dtype(at::kBool));
    at::Tensor target_bag = at::where(valid, bag, at::full_like(bag, bags));

    at::Tensor rows = at::index_select(weight, 0, idx);
    if (has_psw) rows = at::mul(rows, per_sample_weights->to(weight.scalar_type()).unsqueeze(1));
    at::Tensor bag_size = at::zeros({bags + 1}, idx.options()).index_add_(0, target_bag, valid.to(at::kLong)).narrow(0, 0, bags);

    at::Tensor output, max_indices;
    at::Tensor expanded_bags = target_bag.unsqueeze(1).expand({count, dim});
    if (mode == 2) {
        output = at::zeros({bags + 1, dim}, weight.options()).scatter_reduce(0, expanded_bags, rows, "amax", /*include_self=*/false);
        at::Tensor best = at::index_select(output, 0, target_bag);
        at::Tensor positions = at::arange(count, idx.options()).unsqueeze(1).expand({count, dim});
        at::Tensor candidate = at::where(at::eq(rows, best), positions, at::full_like(positions, std::numeric_limits<int64_t>::max()));
        at::Tensor first = at::full({bags + 1, dim}, std::numeric_limits<int64_t>::max(), idx.options()).scatter_reduce(0, expanded_bags, candidate, "amin");
        first = first.narrow(0, 0, bags);
        at::Tensor empty_bag = at::eq(first, std::numeric_limits<int64_t>::max());
        max_indices = at::where(empty_bag, at::zeros_like(first), at::index_select(idx, 0, at::where(empty_bag, at::zeros_like(first), first).reshape({-1})).view(first.sizes()));
        output = output.narrow(0, 0, bags);
    } else {
        output = at::zeros({bags + 1, dim}, weight.options()).index_add_(0, target_bag, rows).narrow(0, 0, bags);
        if (mode == 1) output = at::div(output, at::clamp_min(bag_size, 1).unsqueeze(1).to(weight.scalar_type()));
        max_indices = at::zeros({bags}, idx.options());
    }
    return std::make_tuple(output.contiguous(), bag, bag_size, max_indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> embedding_bag_vulkan(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool /* scale_grad_by_freq */,
    int64_t mode,
    bool /* sparse */,
    const c10::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx)
{
    return embedding_bag_impl(weight, indices, offsets, mode, per_sample_weights, include_last_offset, padding_idx);
}

at::Tensor embedding_bag_dense_backward_vulkan(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offset2bag,
    const at::Tensor& bag_size,
    const at::Tensor& maximum_indices,
    c10::SymInt num_weights_,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<at::Tensor>& per_sample_weights,
    int64_t padding_idx)
{
    int64_t num_weights = num_weights_.guard_int(__FILE__, __LINE__);
    int64_t dim = grad.size(1);
    at::Tensor idx = indices.to(at::kLong);
    at::Tensor grad_weight = at::zeros({num_weights, dim}, grad.options());
    if (idx.numel() == 0) return grad_weight;

    if (mode == 2) {
        at::Tensor bags_with_rows = at::gt(bag_size, 0).unsqueeze(1).expand(maximum_indices.sizes());
        at::Tensor target = at::where(bags_with_rows, maximum_indices, at::full_like(maximum_indices, num_weights));
        at::Tensor extended = at::zeros({num_weights + 1, dim}, grad.options());
        return extended.scatter_add_(0, target, grad).narrow(0, 0, num_weights).contiguous();
    }

    at::Tensor rows = at::index_select(grad, 0, offset2bag);
    if (mode == 1) rows = at::div(rows, at::clamp_min(at::index_select(bag_size, 0, offset2bag), 1).unsqueeze(1).to(grad.scalar_type()));
    if (per_sample_weights.has_value() && per_sample_weights->defined()) rows = at::mul(rows, per_sample_weights->to(grad.scalar_type()).unsqueeze(1));
    if (scale_grad_by_freq) {
        at::Tensor frequency = at::zeros({num_weights}, idx.options()).index_add_(0, idx, at::ones_like(idx));
        rows = at::div(rows, at::index_select(frequency, 0, idx).unsqueeze(1).to(grad.scalar_type()));
    }
    at::Tensor target = padding_idx >= 0 ? at::where(at::eq(idx, padding_idx), at::full_like(idx, num_weights), idx) : idx;
    at::Tensor extended = at::zeros({num_weights + 1, dim}, grad.options());
    return extended.index_add_(0, target, rows).narrow(0, 0, num_weights).contiguous();
}

at::Tensor embedding_bag_backward_vulkan(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& /* offsets */,
    const at::Tensor& offset2bag,
    const at::Tensor& bag_size,
    const at::Tensor& maximum_indices,
    c10::SymInt num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights,
    int64_t padding_idx)
{
    TORCH_CHECK(!sparse, "torchvulkan [NOT IMPLEMENTED]: sparse embedding_bag gradients are not supported.");
    return embedding_bag_dense_backward_vulkan(grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode,
        per_sample_weights, padding_idx);
}

at::Tensor embedding_bag_per_sample_weights_backward_vulkan(
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& /* offsets */,
    const at::Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx)
{
    TORCH_CHECK(mode == 0, "embedding_bag_backward: per_sample_weights only supported for mode='sum'");
    at::Tensor idx = indices.to(at::kLong);
    at::Tensor dots = at::sum(at::mul(at::index_select(grad, 0, offset2bag), at::index_select(weight, 0, idx)), 1);
    if (padding_idx >= 0) dots = at::where(at::eq(idx, padding_idx), at::zeros_like(dots), dots);
    return dots;
}

struct SegmentLayout {
    at::Tensor ids;
    // the dropped segment has length 1, so a mean over it never divides by 0
    at::Tensor lengths;
    std::vector<int64_t> out_shape;
    int64_t rows;
    int64_t positions;
    int64_t rest;
    int64_t segments;
};

SegmentLayout segment_layout(const at::Tensor& data, const at::Tensor& lengths_in, int64_t axis, bool unsafe)
{
    SegmentLayout s;
    s.rows = 1;
    for (int64_t d = 0; d < axis; d++) s.rows *= data.size(d);
    s.positions = data.size(axis);
    s.rest = 1;
    for (int64_t d = axis + 1; d < data.dim(); d++) s.rest *= data.size(d);
    s.segments = lengths_in.size(-1);
    TORCH_CHECK(lengths_in.numel() == s.rows * s.segments, "segment_reduce(): Expected lengths to have the shape of data up to axis, then the segments");

    at::Tensor lengths = lengths_in.to(at::kLong).reshape({s.rows, s.segments});
    at::Tensor ends = at::cumsum(lengths, 1);
    if (!unsafe && s.segments > 0 && s.rows > 0) {
        bool consistent = at::eq(ends.narrow(1, s.segments - 1, 1), s.positions).all().item<bool>();
        TORCH_CHECK(consistent, "segment_reduce(): Expected all rows of lengths along axis to sum to data.size(lengths.dim()-1) when !unsafe.");
    }
    at::Tensor positions = at::arange(s.positions, lengths.options()).unsqueeze(0).expand({s.rows, s.positions}).contiguous();
    at::Tensor ids = s.segments == 0 ? at::zeros_like(positions) : at::searchsorted(ends.contiguous(), positions, /*out_int32=*/false, /*right=*/true);
    s.ids = ids.unsqueeze(-1).expand({s.rows, s.positions, s.rest});
    s.lengths = at::cat({lengths, at::ones({s.rows, 1}, lengths.options())}, 1).unsqueeze(-1);
    s.out_shape = std::vector<int64_t>(data.sizes().begin(), data.sizes().begin() + axis);
    s.out_shape.push_back(s.segments);
    s.out_shape.insert(s.out_shape.end(), data.sizes().begin() + axis + 1, data.sizes().end());
    return s;
}

at::Tensor lengths_from(const c10::optional<at::Tensor>& lengths, const c10::optional<at::Tensor>& offsets)
{
    if (lengths.has_value() && lengths->defined()) return *lengths;
    TORCH_CHECK(offsets.has_value() && offsets->defined(), "segment_reduce(): Either lengths or offsets must be defined");
    int64_t n = offsets->size(-1);
    return at::sub(offsets->narrow(-1, 1, n - 1), offsets->narrow(-1, 0, n - 1));
}

at::Tensor segment_reduce_vulkan(
    const at::Tensor& data,
    c10::string_view reduce,
    const c10::optional<at::Tensor>& lengths_opt,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& offsets,
    int64_t axis,
    bool unsafe,
    const c10::optional<at::Scalar>& initial)
{
    TORCH_CHECK(!indices.has_value() || !indices->defined(), "segment_reduce(): indices based reduction is not supported yet.");
    axis = c10::maybe_wrap_dim(axis, data.dim());
    SegmentLayout s = segment_layout(data, lengths_from(lengths_opt, offsets), axis, unsafe);
    at::Tensor values = data.reshape({s.rows, s.positions, s.rest});

    double identity = 0.0;
    std::string mode;
    if (reduce == "sum") { mode = "sum"; identity = 0.0; }
    else if (reduce == "mean") { mode = "mean"; identity = std::numeric_limits<double>::quiet_NaN(); }
    else if (reduce == "max") { mode = "amax"; identity = -std::numeric_limits<double>::infinity(); }
    else if (reduce == "min") { mode = "amin"; identity = std::numeric_limits<double>::infinity(); }
    else if (reduce == "prod") { mode = "prod"; identity = 1.0; }
    else TORCH_CHECK(false, "segment_reduce(): unsupported reduce ", reduce);

    std::vector<int64_t> grid = {s.rows, s.segments + 1, s.rest};
    at::Tensor out;
    if (initial.has_value()) {
        at::Tensor start = at::full(grid, *initial, data.options());
        if (reduce == "mean") {
            at::Tensor sums = start.scatter_reduce(1, s.ids, values, "sum", /*include_self=*/true);
            at::Tensor counts = at::clamp_min(s.lengths.to(data.scalar_type()), 1);
            out = at::where(at::eq(s.lengths, 0), start, at::div(sums, counts));
        } else {
            out = start.scatter_reduce(1, s.ids, values, mode, /*include_self=*/true);
        }
    } else {
        out = at::full(grid, identity, data.options()).scatter_reduce(1, s.ids, values, mode, /*include_self=*/false);
    }
    return out.narrow(1, 0, s.segments).reshape(s.out_shape);
}

at::Tensor segment_reduce_backward_vulkan(
    const at::Tensor& grad,
    const at::Tensor& output,
    const at::Tensor& data,
    c10::string_view reduce,
    const c10::optional<at::Tensor>& lengths_opt,
    const c10::optional<at::Tensor>& offsets,
    int64_t axis,
    const c10::optional<at::Scalar>& /* initial */)
{
    axis = c10::maybe_wrap_dim(axis, data.dim());
    SegmentLayout s = segment_layout(data, lengths_from(lengths_opt, offsets), axis, /*unsafe=*/true);
    at::Tensor x = data.reshape({s.rows, s.positions, s.rest});
    at::Tensor dropped = at::zeros({s.rows, 1, s.rest}, grad.options());
    at::Tensor g = at::gather(at::cat({grad.reshape({s.rows, s.segments, s.rest}), dropped}, 1), 1, s.ids);
    at::Tensor o = at::gather(at::cat({output.reshape({s.rows, s.segments, s.rest}), dropped}, 1), 1, s.ids);
    std::vector<int64_t> grid = {s.rows, s.segments + 1, s.rest};

    at::Tensor result;
    if (reduce == "sum") {
        result = g;
    } else if (reduce == "mean") {
        result = at::div(g, at::gather(s.lengths.expand(grid).to(data.scalar_type()), 1, s.ids));
    } else if (reduce == "max" || reduce == "min") {
        // tied extremes split the gradient evenly
        at::Tensor hit = at::eq(x, o);
        at::Tensor ties = at::zeros(grid, data.options()).scatter_add(1, s.ids, hit.to(data.scalar_type()));
        result = at::where(hit, at::div(g, at::gather(ties, 1, s.ids)), at::zeros_like(g));
    } else {
        // with zeros, only a lone zero gets the product of the others
        at::Tensor zero = at::eq(x, 0);
        at::Tensor zeros_in_segment = at::gather(at::zeros(grid, data.options()).scatter_add(1, s.ids, zero.to(data.scalar_type())), 1, s.ids);
        at::Tensor safe = at::where(zero, at::ones_like(x), x);
        at::Tensor others = at::gather(at::ones(grid, data.options()).scatter_reduce(1, s.ids, safe, "prod"), 1, s.ids);
        at::Tensor nonzero_grad = at::div(at::mul(g, o), safe);
        at::Tensor zero_grad = at::where(at::eq(zeros_in_segment, 1), at::mul(g, others), at::zeros_like(g));
        result = at::where(zero, zero_grad, nonzero_grad);
    }
    return result.reshape(data.sizes()).contiguous();
}

at::Tensor rrelu_with_noise_vulkan(
    const at::Tensor& self,
    at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator)
{
    if (!training) {
        double slope = (lower.toDouble() + upper.toDouble()) / 2.0;
        return at::where(at::ge(self, 0), self, at::mul(self, slope));
    }
    at::Tensor sample = at::empty_like(self).uniform_(lower.toDouble(), upper.toDouble(), generator);
    at::Tensor positive = at::gt(self, 0);
    noise.copy_(at::where(positive, at::ones_like(sample), sample));
    return at::mul(self, noise);
}

at::Tensor& rrelu_with_noise_out_vulkan(
    const at::Tensor& self,
    at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator,
    at::Tensor& out)
{
    return fill_out(out, rrelu_with_noise_vulkan(self, noise, lower, upper, training, generator));
}

at::Tensor efficientzerotensor_vulkan(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> /* pin_memory */)
{
    std::vector<int64_t> shape;
    for (const c10::SymInt& s : size) shape.push_back(s.guard_int(__FILE__, __LINE__));
    at::TensorOptions options = at::TensorOptions().dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
        .layout(layout.value_or(at::kStrided)).device(device.value_or(at::Device(at::DeviceType::PrivateUse1, VulkanContext::CurrentDevice())));
    return at::zeros(shape, options);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("native_batch_norm", &native_batch_norm_vulkan);
    m.impl("native_batch_norm.out", &native_batch_norm_out_vulkan);
    m.impl("_native_batch_norm_legit", &native_batch_norm_legit_vulkan);
    m.impl("_native_batch_norm_legit.out", &native_batch_norm_legit_out_vulkan);
    m.impl("_native_batch_norm_legit.no_stats", &native_batch_norm_legit_no_stats_vulkan);
    m.impl("_native_batch_norm_legit.no_stats_out", &native_batch_norm_legit_no_stats_out_vulkan);
    m.impl("_batch_norm_with_update", &batch_norm_with_update_vulkan);
    m.impl("_batch_norm_with_update.out", &batch_norm_with_update_out_vulkan);
    m.impl("native_batch_norm_backward", &native_batch_norm_backward_vulkan);
    m.impl("batch_norm_backward", &batch_norm_backward_vulkan);
    m.impl("native_group_norm_backward", &native_group_norm_backward_vulkan);
    m.impl("_cdist_forward", &cdist_forward_vulkan);
    m.impl("_cdist_backward", &cdist_backward_vulkan);
    m.impl("_pdist_forward", &pdist_forward_vulkan);
    m.impl("_pdist_backward", &pdist_backward_vulkan);
    m.impl("nll_loss2d_forward", &nll_loss2d_forward_vulkan);
    m.impl("nll_loss2d_forward.output", &nll_loss2d_forward_out_vulkan);
    m.impl("nll_loss2d_backward", &nll_loss2d_backward_vulkan);
    m.impl("nll_loss2d_backward.grad_input", &nll_loss2d_backward_out_vulkan);
    m.impl("multi_margin_loss", &multi_margin_loss_vulkan);
    m.impl("multi_margin_loss.out", &multi_margin_loss_out_vulkan);
    m.impl("multi_margin_loss_backward", &multi_margin_loss_backward_vulkan);
    m.impl("multi_margin_loss_backward.grad_input", &multi_margin_loss_backward_out_vulkan);
    m.impl("multilabel_margin_loss_forward", &multilabel_margin_loss_forward_vulkan);
    m.impl("multilabel_margin_loss_forward.output", &multilabel_margin_loss_forward_out_vulkan);
    m.impl("multilabel_margin_loss_backward", &multilabel_margin_loss_backward_vulkan);
    m.impl("multilabel_margin_loss_backward.grad_input", &multilabel_margin_loss_backward_out_vulkan);
    m.impl("_embedding_bag", &embedding_bag_vulkan);
    m.impl("_embedding_bag_forward_only", &embedding_bag_vulkan);
    m.impl("_embedding_bag_backward", &embedding_bag_backward_vulkan);
    m.impl("_embedding_bag_dense_backward", &embedding_bag_dense_backward_vulkan);
    m.impl("_embedding_bag_per_sample_weights_backward", &embedding_bag_per_sample_weights_backward_vulkan);
    m.impl("segment_reduce", &segment_reduce_vulkan);
    m.impl("_segment_reduce_backward", &segment_reduce_backward_vulkan);
    m.impl("rrelu_with_noise", &rrelu_with_noise_vulkan);
    m.impl("rrelu_with_noise.out", &rrelu_with_noise_out_vulkan);
    m.impl("_efficientzerotensor", &efficientzerotensor_vulkan);
}
