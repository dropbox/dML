// Copyright 2024-2025 Andrew Yates
// Joiner network for RNN-T in Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#pragma once

#include <mlx/mlx.h>
#include <optional>

#include "zipformer/array_utils.hpp"
#include "zipformer/scaling.hpp"

namespace zipformer {

using namespace mlx::core;

/**
 * Configuration for RNN-T Joiner network.
 */
struct JoinerConfig {
    int encoder_dim = 512;
    int decoder_dim = 512;
    int joiner_dim = 512;
    int vocab_size = 500;
};

/**
 * RNN-T Joiner network.
 *
 * Combines encoder and decoder representations to produce
 * output logits for the vocabulary.
 *
 * Formula:
 *   joint = tanh(encoder_proj(encoder) + decoder_proj(decoder))
 *   logits = output_proj(joint)
 */
class Joiner {
public:
    explicit Joiner(const JoinerConfig& config);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward pass
    // encoder_out: (batch, time, encoder_dim)
    // decoder_out: (batch, decoder_dim)
    // Returns: (batch, time, vocab_size)
    array forward(const array& encoder_out, const array& decoder_out) const;

    // Single frame forward for streaming
    // encoder_frame: (batch, encoder_dim)
    // decoder_rep: (batch, decoder_dim)
    // Returns: (batch, vocab_size)
    array forward_single(const array& encoder_frame, const array& decoder_rep) const;

    const JoinerConfig& config() const { return config_; }

private:
    JoinerConfig config_;

    ScaledLinear encoder_proj_;
    ScaledLinear decoder_proj_;
    ScaledLinear output_proj_;
};

} // namespace zipformer
