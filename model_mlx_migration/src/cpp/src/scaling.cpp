// Copyright 2024-2025 Andrew Yates
// Scaling modules implementation for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/scaling.hpp"
#include <stdexcept>
#include <cmath>

namespace zipformer {

// ============================================================================
// BiasNorm
// ============================================================================

BiasNorm::BiasNorm(int num_channels, int channel_dim, float log_scale)
    : num_channels_(num_channels)
    , channel_dim_(channel_dim)
    , log_scale_(array(log_scale))
    , bias_(zeros({num_channels}))
{
}

void BiasNorm::load_weights(const WeightMap& weights, const std::string& prefix) {
    if (zipformer::has_weight(weights, prefix + ".log_scale")) {
        log_scale_ = zipformer::get_weight(weights, prefix + ".log_scale");
    }
    if (zipformer::has_weight(weights, prefix + ".bias")) {
        bias_ = zipformer::get_weight(weights, prefix + ".bias");
    }
}

array BiasNorm::forward(const array& x) const {
    // Resolve negative channel_dim
    int channel_dim = channel_dim_;
    if (channel_dim < 0) {
        channel_dim = x.ndim() + channel_dim;
    }

    // Expand bias to match input dimensions
    array bias = bias_;
    for (int i = channel_dim + 1; i < static_cast<int>(x.ndim()); ++i) {
        bias = expand_dims(bias, -1);
    }

    // Clamp log_scale to valid range [-1.5, 1.5]
    array log_scale = clip(log_scale_, array(-1.5f), array(1.5f));

    // Compute scale: (mean((x - bias)^2))^(-0.5) * exp(log_scale)
    array centered = subtract(x, bias);
    array variance = mean(multiply(centered, centered), channel_dim, true);

    // Add small epsilon to avoid division by zero
    array scale = multiply(
        rsqrt(add(variance, array(1e-8f))),
        exp(log_scale)
    );

    // Return original x times scale (not centered!)
    return multiply(x, scale);
}

// ============================================================================
// ScaledLinear
// ============================================================================

ScaledLinear::ScaledLinear(int in_features, int out_features, bool has_bias)
    : in_features_(in_features)
    , out_features_(out_features)
    , has_bias_(has_bias)
    , weight_(zeros({out_features, in_features}))
    , bias_(zeros({has_bias ? out_features : 1}))  // Always init; use has_bias_ in forward
{
}

void ScaledLinear::load_weights(const WeightMap& weights, const std::string& prefix) {
    if (zipformer::has_weight(weights, prefix + ".weight")) {
        weight_ = zipformer::get_weight(weights, prefix + ".weight");
    }
    if (has_bias_ && zipformer::has_weight(weights, prefix + ".bias")) {
        bias_ = zipformer::get_weight(weights, prefix + ".bias");
    }
}

array ScaledLinear::forward(const array& x) const {
    // Linear: x @ weight.T + bias
    // x: (..., in_features) -> (..., out_features)
    array out = matmul(x, transpose(weight_));
    if (has_bias_) {
        out = add(out, bias_);
    }
    return out;
}

// ============================================================================
// ActivationDropoutAndLinear
// ============================================================================

ActivationDropoutAndLinear::ActivationDropoutAndLinear(
    int in_features,
    int out_features,
    const std::string& activation
)
    : linear_(in_features, out_features, true)
    , activation_type_(activation)
{
}

void ActivationDropoutAndLinear::load_weights(
    const WeightMap& weights,
    const std::string& prefix
) {
    linear_.load_weights(weights, prefix + ".linear");
}

array ActivationDropoutAndLinear::apply_activation(const array& x) const {
    if (activation_type_ == "swish" || activation_type_ == "silu") {
        // swish(x) = x * sigmoid(x)
        return multiply(x, sigmoid(x));
    } else if (activation_type_ == "relu") {
        return maximum(x, array(0.0f));
    } else if (activation_type_ == "gelu") {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        auto x_cubed = multiply(multiply(x, x), x);
        auto inner = multiply(
            array(sqrt_2_over_pi),
            add(x, multiply(array(0.044715f), x_cubed))
        );
        return multiply(
            multiply(array(0.5f), x),
            add(array(1.0f), tanh(inner))
        );
    } else if (activation_type_ == "identity" || activation_type_ == "none") {
        return x;
    } else {
        throw std::runtime_error("Unknown activation: " + activation_type_);
    }
}

array ActivationDropoutAndLinear::forward(const array& x) const {
    // In inference: activation -> linear (no dropout)
    array activated = apply_activation(x);
    return linear_.forward(activated);
}

} // namespace zipformer
