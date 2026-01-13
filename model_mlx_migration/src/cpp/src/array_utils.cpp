// Copyright 2024-2025 Andrew Yates
// Array utilities implementation for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/array_utils.hpp"
#include <stdexcept>
#include <cmath>

namespace zipformer {

WeightMap load_weights(const std::string& path) {
    auto result = load_safetensors(path);
    return result.first;
}

void save_weights(const std::string& path, const WeightMap& weights) {
    save_safetensors(path, weights);
}

const array& get_weight(const WeightMap& weights, const std::string& key) {
    auto it = weights.find(key);
    if (it == weights.end()) {
        throw std::runtime_error("Weight not found: " + key);
    }
    return it->second;
}

bool has_weight(const WeightMap& weights, const std::string& key) {
    return weights.find(key) != weights.end();
}

std::vector<int> get_shape(const array& x) {
    auto shape = x.shape();
    return std::vector<int>(shape.begin(), shape.end());
}

int get_dim(const array& x, int axis) {
    if (axis < 0) {
        axis = x.ndim() + axis;
    }
    return x.shape()[axis];
}

int numel(const array& x) {
    return x.size();
}

array transpose_for_attention(const array& x) {
    // (B, T, H, D) -> (B, H, T, D)
    return transpose(x, {0, 2, 1, 3});
}

array transpose_from_attention(const array& x) {
    // (B, H, T, D) -> (B, T, H, D)
    return transpose(x, {0, 2, 1, 3});
}

std::vector<array> chunk_along_time(const array& x, int chunk_size) {
    int time_dim = x.shape()[1];
    int num_chunks = (time_dim + chunk_size - 1) / chunk_size;

    std::vector<array> chunks;
    chunks.reserve(num_chunks);

    for (int i = 0; i < num_chunks; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, time_dim);
        chunks.push_back(slice(x, {0, start}, {x.shape()[0], end}));
    }

    return chunks;
}

array pad_to_multiple(const array& x, int axis, int multiple) {
    int current_size = x.shape()[axis];
    int remainder = current_size % multiple;
    if (remainder == 0) {
        return x;
    }

    int pad_amount = multiple - remainder;

    // Create padding specification as pairs (low, high) for each axis
    std::vector<std::pair<int, int>> pad_widths(x.ndim(), {0, 0});
    pad_widths[axis].second = pad_amount;  // Pad at end of axis

    return pad(x, pad_widths);
}

array causal_mask(int seq_len, Dtype dtype) {
    // Create lower triangular mask
    auto mask = tril(ones({seq_len, seq_len}, dtype));
    // Convert to attention mask: 0 where attending, -inf where masked
    return where(mask, zeros({seq_len, seq_len}, dtype),
                 full({seq_len, seq_len}, -1e9f, dtype));
}

void CacheState::reset() {
    encoder_states.clear();
    conv_caches.clear();
    predictor_state = zeros({1});  // Empty sentinel - size 1 marks as uninitialized
    processed_frames = 0;
}

bool CacheState::empty() const {
    return encoder_states.empty() && conv_caches.empty();
}

ValidationResult compare_arrays(
    const array& actual,
    const array& expected,
    float rtol,
    float atol
) {
    ValidationResult result;

    // Compute absolute difference
    auto diff = abs(subtract(actual, expected));
    eval(diff);

    // Max difference
    auto max_diff_arr = max(diff);
    eval(max_diff_arr);
    result.max_diff = max_diff_arr.item<float>();

    // Mean difference
    auto mean_diff_arr = mean(diff);
    eval(mean_diff_arr);
    result.mean_diff = mean_diff_arr.item<float>();

    // RMS difference
    auto squared_diff = multiply(diff, diff);
    auto rms_diff_arr = sqrt(mean(squared_diff));
    eval(rms_diff_arr);
    result.rms_diff = rms_diff_arr.item<float>();

    // Check if passed: |actual - expected| <= atol + rtol * |expected|
    auto tolerance = add(full(expected.shape(), atol),
                         multiply(full(expected.shape(), rtol), abs(expected)));
    auto passed_arr = all(less_equal(diff, tolerance));
    eval(passed_arr);
    result.passed = passed_arr.item<bool>();

    if (result.passed) {
        result.message = "Validation passed";
    } else {
        result.message = "Validation failed: max_diff=" +
                         std::to_string(result.max_diff) +
                         " > tolerance";
    }

    return result;
}

} // namespace zipformer
