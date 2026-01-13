// Copyright 2024-2025 Andrew Yates
// Scaling modules for Zipformer MLX C++ implementation
//
// Ported from icefall/zipformer scaling.py
// Licensed under the Apache License, Version 2.0

#pragma once

#include <mlx/mlx.h>
#include "zipformer/array_utils.hpp"

namespace zipformer {

using namespace mlx::core;

/**
 * BiasNorm: A simpler replacement for LayerNorm from icefall.
 *
 * Instead of learned weight and bias applied after normalization,
 * BiasNorm uses:
 * - A trainable bias SUBTRACTED for computing the scale (but output uses original x)
 * - A trainable log_scale on the output
 *
 * Formula:
 *   centered = x - bias
 *   scales = (mean(centered^2, dim=channel_dim) + eps)^(-0.5) * exp(log_scale)
 *   output = x * scales
 */
class BiasNorm {
public:
    BiasNorm(int num_channels, int channel_dim = -1, float log_scale = 1.0f);

    // Load weights from state dict
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward pass
    array forward(const array& x) const;

    // Accessors for parameters (for debugging/validation)
    const array& get_log_scale() const { return log_scale_; }
    const array& get_bias() const { return bias_; }

private:
    int num_channels_;
    int channel_dim_;
    array log_scale_;
    array bias_;
};

/**
 * ScaledLinear: Linear layer with scaled initialization
 *
 * A wrapper that performs: output = x @ weight.T + bias
 */
class ScaledLinear {
public:
    ScaledLinear(int in_features, int out_features, bool has_bias = true);

    // Load weights from state dict
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward pass
    array forward(const array& x) const;

    // Accessors
    const array& get_weight() const { return weight_; }
    const array& get_bias() const { return bias_; }
    bool has_bias() const { return has_bias_; }

private:
    int in_features_;
    int out_features_;
    bool has_bias_;
    array weight_;
    array bias_;
};

/**
 * Identity: Pass-through module (used in residual connections)
 */
class Identity {
public:
    array forward(const array& x) const { return x; }
};

/**
 * Balancer: Gradient balancing module from icefall
 *
 * In inference mode, this is a no-op (identity).
 * Training functionality not implemented for inference-only C++.
 */
class Balancer {
public:
    explicit Balancer(int num_channels) : num_channels_(num_channels) {}
    array forward(const array& x) const { return x; }

private:
    int num_channels_;
};

/**
 * Whiten: Whitening module from icefall
 *
 * In inference mode, this is a no-op (identity).
 */
class Whiten {
public:
    array forward(const array& x) const { return x; }
};

/**
 * ActivationDropoutAndLinear: Combined activation, dropout, and linear
 *
 * In inference mode: applies activation and linear (no dropout).
 *
 * Formula: linear(activation(x))
 */
class ActivationDropoutAndLinear {
public:
    ActivationDropoutAndLinear(
        int in_features,
        int out_features,
        const std::string& activation = "swish"
    );

    // Load weights from state dict
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward pass
    array forward(const array& x) const;

private:
    ScaledLinear linear_;
    std::string activation_type_;

    array apply_activation(const array& x) const;
};

} // namespace zipformer
