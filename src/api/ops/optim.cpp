#include <torch/extension.h>
#include <cmath>
#include <vector>
#include "api/ops/helpers.h"

namespace {

enum class AdamMode
{
    ORIGINAL, 
    ADAMW
};

std::vector<double> read_host_values(const at::Device& device, const c10::optional<at::Tensor>& found_inf, at::TensorList steps)
{
    std::vector<at::Tensor> parts;
    if (found_inf.has_value() && found_inf->defined()) parts.push_back(found_inf->reshape({1}).to(device, at::kFloat));
    for (const at::Tensor& step : steps) parts.push_back(step.reshape({1}).to(device, at::kFloat));
    if (parts.empty()) return {};

    at::Tensor host = at::cat(parts).cpu();
    const float* data = host.const_data_ptr<float>();
    return std::vector<double>(data, data + host.numel());
}

bool has_found_inf(const c10::optional<at::Tensor>& found_inf)
{
    return found_inf.has_value() && found_inf->defined();
}

at::Tensor prepare_grad(const at::Tensor& grad, at::ScalarType compute, const c10::optional<at::Tensor>& grad_scale, bool maximize)
{
    at::Tensor g = grad.to(compute);
    if (grad_scale.has_value() && grad_scale->defined()) {
        g = at::div(g, grad_scale->reshape({}).to(compute));
        grad.copy_(g);
    }
    return maximize ? at::neg(g) : g;
}

at::ScalarType compute_dtype(const at::Tensor& param)
{
    return is_low_prec(param.scalar_type()) ? at::kFloat : param.scalar_type();
}

void adam_step(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& max_exp_avg_sq,
    double step,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps,
    bool amsgrad,
    bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    AdamMode mode)
{
    at::ScalarType compute = compute_dtype(param);
    at::Tensor g = prepare_grad(grad, compute, grad_scale, maximize);
    at::Tensor p = param.to(compute);
    if (weight_decay != 0.0) {
        if (mode == AdamMode::ORIGINAL) g = at::add(g, p, weight_decay);
        else p = at::mul(p, 1.0 - lr * weight_decay);
    }

    at::Tensor m = at::lerp(exp_avg.to(compute), g, 1.0 - beta1);
    at::Tensor v = at::add(at::mul(exp_avg_sq.to(compute), beta2), at::mul(at::mul(g, 1.0 - beta2), g));
    exp_avg.copy_(m);
    exp_avg_sq.copy_(v);

    at::Tensor second = v;
    if (amsgrad) {
        second = at::maximum(max_exp_avg_sq.to(compute), v);
        max_exp_avg_sq.copy_(second);
    }

    const double bias_correction1 = 1.0 - std::pow(beta1, step);
    const double bias_correction2_sqrt = std::sqrt(1.0 - std::pow(beta2, step));
    at::Tensor denom = at::add(at::div(at::sqrt(second), bias_correction2_sqrt), eps);
    param.copy_(at::sub(p, at::div(at::mul(m, lr / bias_correction1), denom)));
}

void fused_adam(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps,
    bool amsgrad,
    bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf,
    AdamMode mode)
{
    const size_t n = params.size();
    TORCH_CHECK(grads.size() == n);
    TORCH_CHECK(exp_avgs.size() == n);
    TORCH_CHECK(exp_avg_sqs.size() == n);
    TORCH_CHECK(amsgrad ? max_exp_avg_sqs.size() == n : max_exp_avg_sqs.empty());
    TORCH_CHECK(state_steps.size() == n);
    if (n == 0) return;

    std::vector<double> host = read_host_values(params[0].device(), found_inf, state_steps);
    size_t offset = 0;
    if (has_found_inf(found_inf)) {
        if (host[0] == 1.0) return;
        offset = 1;
    }

    for (size_t i = 0; i < n; ++i) {
        adam_step(params[i], grads[i], exp_avgs[i], exp_avg_sqs[i], amsgrad ? max_exp_avg_sqs[i] : at::Tensor(),
            host[offset + i], lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, mode
        );
    }
}

void fused_adam_vulkan_(
    at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2,
    double weight_decay, double eps, bool amsgrad, bool maximize,
    const c10::optional<at::Tensor>& grad_scale, const c10::optional<at::Tensor>& found_inf)
{
    fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2,
        weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, AdamMode::ORIGINAL
    );
}

void fused_adam_tensor_lr_vulkan_(
    at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor& lr, double beta1, double beta2,
    double weight_decay, double eps, bool amsgrad, bool maximize,
    const c10::optional<at::Tensor>& grad_scale, const c10::optional<at::Tensor>& found_inf)
{
    fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr.item<double>(), beta1, beta2,
        weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, AdamMode::ORIGINAL
    );
}

void fused_adamw_vulkan_(
    at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2,
    double weight_decay, double eps, bool amsgrad, bool maximize,
    const c10::optional<at::Tensor>& grad_scale, const c10::optional<at::Tensor>& found_inf)
{
    fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2,
        weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, AdamMode::ADAMW
    );
}

void fused_adamw_tensor_lr_vulkan_(
    at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor& lr, double beta1, double beta2,
    double weight_decay, double eps, bool amsgrad, bool maximize,
    const c10::optional<at::Tensor>& grad_scale, const c10::optional<at::Tensor>& found_inf)
{
    fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr.item<double>(), beta1, beta2,
        weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, AdamMode::ADAMW
    );
}

void fused_sgd_vulkan_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    double weight_decay,
    double momentum,
    double lr,
    double dampening,
    bool nesterov,
    bool maximize,
    bool is_first_step,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf)
{
    const size_t n = params.size();
    TORCH_CHECK(grads.size() == n);
    const bool has_momentum = momentum != 0.0;
    TORCH_CHECK(has_momentum ? momentum_buffer_list.size() == n : momentum_buffer_list.empty());
    if (n == 0) return;

    if (has_found_inf(found_inf) && read_host_values(params[0].device(), found_inf, {})[0] == 1.0) return;

    for (size_t i = 0; i < n; ++i) {
        at::ScalarType compute = compute_dtype(params[i]);
        at::Tensor g = prepare_grad(grads[i], compute, grad_scale, maximize);
        at::Tensor p = params[i].to(compute);
        if (weight_decay != 0.0) g = at::add(g, p, weight_decay);
        if (has_momentum) {
            at::Tensor buf = is_first_step ? g : at::add(at::mul(momentum_buffer_list[i].to(compute), momentum), at::mul(g, 1.0 - dampening));
            momentum_buffer_list[i].copy_(buf);
            g = nesterov ? at::add(g, buf, momentum) : buf;
        }
        params[i].copy_(at::sub(p, at::mul(g, lr)));
    }
}

void fused_sgd_tensor_lr_vulkan_(
    at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay,
    double momentum, const at::Tensor& lr, double dampening, bool nesterov, bool maximize, bool is_first_step,
    const c10::optional<at::Tensor>& grad_scale, const c10::optional<at::Tensor>& found_inf)
{
    fused_sgd_vulkan_(params, grads, momentum_buffer_list, weight_decay, momentum, lr.item<double>(), dampening,
        nesterov, maximize, is_first_step, grad_scale, found_inf
    );
}

void fused_adagrad_vulkan_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    double lr,
    double lr_decay,
    double weight_decay,
    double eps,
    bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf)
{
    const size_t n = params.size();
    TORCH_CHECK(grads.size() == n);
    TORCH_CHECK(state_sums.size() == n);
    TORCH_CHECK(state_steps.size() == n);
    if (n == 0) return;

    std::vector<double> host = read_host_values(params[0].device(), found_inf, state_steps);
    size_t offset = 0;
    if (has_found_inf(found_inf)) {
        if (host[0] == 1.0) return;
        offset = 1;
    }

    for (size_t i = 0; i < n; ++i) {
        at::ScalarType compute = compute_dtype(params[i]);
        at::Tensor g = prepare_grad(grads[i], compute, grad_scale, maximize);
        at::Tensor p = params[i].to(compute);
        if (weight_decay != 0.0) g = at::add(g, p, weight_decay);
        at::Tensor sum = at::add(state_sums[i].to(compute), at::mul(g, g));
        state_sums[i].copy_(sum);
        const double clr = lr / (1.0 + (host[offset + i] - 1.0) * lr_decay);
        params[i].copy_(at::sub(p, at::div(at::mul(g, clr), at::add(at::sqrt(sum), eps))));
    }
}

void fused_adagrad_tensor_lr_vulkan_(
    at::TensorList params, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps,
    const at::Tensor& lr, double lr_decay, double weight_decay, double eps, bool maximize,
    const c10::optional<at::Tensor>& grad_scale, const c10::optional<at::Tensor>& found_inf)
{
    fused_adagrad_vulkan_(params, grads, state_sums, state_steps, lr.item<double>(), lr_decay, weight_decay, eps,
        maximize, grad_scale, found_inf
    );
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_fused_adam_", &fused_adam_vulkan_);
    m.impl("_fused_adam_.tensor_lr", &fused_adam_tensor_lr_vulkan_);
    m.impl("_fused_adamw_", &fused_adamw_vulkan_);
    m.impl("_fused_adamw_.tensor_lr", &fused_adamw_tensor_lr_vulkan_);
    m.impl("_fused_sgd_", &fused_sgd_vulkan_);
    m.impl("_fused_sgd_.tensor_lr", &fused_sgd_tensor_lr_vulkan_);
    m.impl("_fused_adagrad_", &fused_adagrad_vulkan_);
    m.impl("_fused_adagrad_.tensor_lr", &fused_adagrad_tensor_lr_vulkan_);
}
