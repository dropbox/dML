// Copyright 2024-2025 Andrew Yates
// Array utilities for Zipformer MLX C++ implementation
//
// Licensed under the Apache License, Version 2.0

#pragma once

#include <mlx/mlx.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace zipformer {

using namespace mlx::core;

// Type alias for model weights
using WeightMap = std::unordered_map<std::string, array>;

// Load safetensors model weights
WeightMap load_weights(const std::string& path);

// Save weights to safetensors format
void save_weights(const std::string& path, const WeightMap& weights);

// Get a weight or throw if not found
const array& get_weight(const WeightMap& weights, const std::string& key);

// Check if weight exists
bool has_weight(const WeightMap& weights, const std::string& key);

// Tensor shape utilities
std::vector<int> get_shape(const array& x);
int get_dim(const array& x, int axis);
int numel(const array& x);

// Transpose for attention: (B, T, H, D) -> (B, H, T, D)
array transpose_for_attention(const array& x);

// Inverse transpose: (B, H, T, D) -> (B, T, H, D)
array transpose_from_attention(const array& x);

// Chunk tensor along time axis
std::vector<array> chunk_along_time(const array& x, int chunk_size);

// Pad tensor to multiple of chunk_size
array pad_to_multiple(const array& x, int axis, int multiple);

// Mask attention scores for causal masking
array causal_mask(int seq_len, Dtype dtype = float32);

// Streaming cache utilities
struct CacheState {
    std::vector<array> encoder_states;
    std::vector<array> conv_caches;
    array predictor_state{zeros({1})};  // Initialize with sentinel
    int processed_frames{0};

    void reset();
    bool empty() const;
};

// Numerical validation
struct ValidationResult {
    float max_diff;
    float mean_diff;
    float rms_diff;
    bool passed;
    std::string message;
};

ValidationResult compare_arrays(
    const array& actual,
    const array& expected,
    float rtol = 1e-5f,
    float atol = 1e-5f
);

} // namespace zipformer
