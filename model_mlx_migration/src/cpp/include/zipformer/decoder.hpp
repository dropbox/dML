// Copyright 2024-2025 Andrew Yates
// Decoder (Predictor) for RNN-T in Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <optional>

#include "zipformer/array_utils.hpp"
#include "zipformer/scaling.hpp"

namespace zipformer {

using namespace mlx::core;

/**
 * Configuration for RNN-T Decoder (Predictor network).
 */
struct DecoderConfig {
    int vocab_size = 500;
    int decoder_dim = 512;
    int blank_id = 0;
    int context_size = 2;  // Number of previous tokens to consider
};

/**
 * RNN-T Decoder (Predictor) network.
 *
 * Takes previous tokens and predicts the decoder representation.
 * Uses a simple embedding + linear architecture for streaming efficiency.
 */
class Decoder {
public:
    explicit Decoder(const DecoderConfig& config);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward pass given previous token IDs
    // tokens: (batch, context_size) - previous token IDs
    // Returns: (batch, decoder_dim)
    array forward(const array& tokens) const;

    // Get blank token representation (for initialization)
    array get_blank_representation() const;

    const DecoderConfig& config() const { return config_; }

private:
    DecoderConfig config_;

    // Embedding layer
    array embedding_weight_{zeros({1})};  // (vocab_size, decoder_dim)

    // Context projection (combines multiple embedded tokens)
    ScaledLinear conv_proj_;

    // Output projection
    ScaledLinear out_proj_;
};

/**
 * Decoder state for streaming inference.
 */
struct DecoderState {
    // Previous token IDs for context
    std::vector<int> context;

    // Pre-computed decoder representation
    array decoder_rep{zeros({1})};

    // Initialize with blank tokens
    void init(int context_size, int blank_id);

    // Update with new token
    void update(int new_token, const Decoder& decoder);
};

} // namespace zipformer
