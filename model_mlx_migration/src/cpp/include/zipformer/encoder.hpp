// Copyright 2024-2025 Andrew Yates
// Zipformer encoder components for MLX C++ implementation
//
// Ported from icefall/zipformer
// Licensed under the Apache License, Version 2.0

#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <tuple>
#include <optional>
#include <memory>

#include "zipformer/array_utils.hpp"
#include "zipformer/scaling.hpp"

namespace zipformer {

using namespace mlx::core;

/**
 * Configuration for Zipformer encoder.
 *
 * Default values match icefall streaming checkpoint.
 */
struct ZipformerConfig {
    // Input features
    int num_features = 80;

    // Encoder stages configuration
    // Default values match wenetspeech-streaming-small checkpoint
    std::vector<int> num_encoder_layers = {2, 2, 2, 2, 2, 2};
    std::vector<int> encoder_dims = {192, 256, 256, 256, 256, 256};
    std::vector<int> attention_dims = {128, 128, 128, 256, 128, 128};
    std::vector<int> num_heads = {4, 4, 4, 8, 4, 4};
    std::vector<int> downsampling_factors = {1, 2, 4, 8, 4, 2};
    std::vector<int> cnn_module_kernels = {31, 31, 15, 15, 15, 31};

    // Per-stage feedforward dimensions (ff1, ff2, ff3 for each stage)
    // Pattern: ff1=2*d, ff2=3*d (rounded), ff3=3.75*d (rounded)
    std::vector<int> ff1_dims = {384, 576, 576, 576, 576, 576};
    std::vector<int> ff2_dims = {512, 768, 768, 768, 768, 768};
    std::vector<int> ff3_dims = {640, 960, 960, 960, 960, 960};

    // Positional encoding
    int pos_dim = 48;
    int pos_head_dim = 4;
    int value_head_dim = 12;

    // Encoder embed output dimension
    int encoder_embed_dim = 192;  // Must match encoder_dims[0]

    // Streaming mode
    bool causal = true;

    // Output dimension (max of encoder_dims)
    int output_dim() const {
        int max_dim = 0;
        for (int d : encoder_dims) {
            if (d > max_dim) max_dim = d;
        }
        return max_dim;
    }
};

/**
 * Convolution module for Zipformer encoder layer.
 *
 * Uses depthwise separable convolution with gating and SwooshR activation.
 */
class ConvolutionModule {
public:
    ConvolutionModule(int d_model, int kernel_size = 31, bool causal = true);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    array forward(
        const array& x,
        const std::optional<array>& padding_mask = std::nullopt
    ) const;

    // Streaming forward with cache
    array streaming_forward(
        const array& x,
        array& conv_cache,
        const std::optional<array>& padding_mask = std::nullopt
    ) const;

private:
    int d_model_;
    int kernel_size_;
    bool causal_;

    // Layers
    ScaledLinear in_proj_;
    ScaledLinear out_proj_;

    // Depthwise conv weights - initialize to sentinel (1x1 zeros)
    array causal_conv_weight_{zeros({1})};
    array causal_conv_bias_{zeros({1})};
    array chunkwise_conv_weight_{zeros({1})};
    array chunkwise_conv_bias_{zeros({1})};
    array chunkwise_conv_scale_{zeros({1})};

    // Non-causal depthwise conv
    array depthwise_conv_weight_{zeros({1})};
    array depthwise_conv_bias_{zeros({1})};

    array get_chunk_scale(int chunk_size) const;
    array causal_depthwise_conv(const array& x) const;
};

/**
 * Feedforward module (FFN) with SwooshL/SwooshR activations.
 *
 * Supports two checkpoint formats:
 * - Original: linear1/linear2 with norm
 * - Icefall: in_proj/out_proj without norm
 */
class FeedforwardModule {
public:
    FeedforwardModule(int d_model, int d_ff, int dropout = 0);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    array forward(const array& x) const;

private:
    int d_model_;
    int d_ff_;
    mutable bool has_norm_{true};  // Set during load_weights

    BiasNorm norm_;
    ScaledLinear linear1_;
    ScaledLinear linear2_;
};

/**
 * Relative positional encoding for Zipformer.
 */
class RelPositionalEncoding {
public:
    explicit RelPositionalEncoding(int d_model, int max_len = 5000);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    array forward(int seq_len) const;

private:
    int d_model_;
    int max_len_;
    array pe_{zeros({1})};  // Pre-computed positional encodings
};

/**
 * Self-attention module with relative positional encoding.
 */
class SelfAttention {
public:
    SelfAttention(
        int d_model,
        int attention_dim,
        int num_heads,
        int pos_dim,
        int pos_head_dim,
        int value_head_dim
    );

    void load_weights(const WeightMap& weights, const std::string& prefix);

    array forward(
        const array& x,
        const array& pos_emb,
        const std::optional<array>& attn_mask = std::nullopt
    ) const;

    // Streaming forward with key-value cache
    array streaming_forward(
        const array& x,
        const array& pos_emb,
        array& kv_cache,
        const std::optional<array>& attn_mask = std::nullopt
    ) const;

private:
    int d_model_;
    int attention_dim_;
    int num_heads_;
    int pos_dim_;
    int pos_head_dim_;
    int value_head_dim_;

    BiasNorm norm_;
    ScaledLinear in_proj_;
    ScaledLinear pos_proj_;
    ScaledLinear out_proj_;

    array compute_attention(
        const array& query,
        const array& key,
        const array& value,
        const array& pos_key,
        const std::optional<array>& attn_mask
    ) const;
};

/**
 * Single Zipformer encoder layer.
 */
class ZipformerEncoderLayer {
public:
    ZipformerEncoderLayer(
        int d_model,
        int attention_dim,
        int feedforward_dim,
        int num_heads,
        int kernel_size,
        int pos_dim,
        int pos_head_dim,
        int value_head_dim,
        bool causal
    );

    void load_weights(const WeightMap& weights, const std::string& prefix);

    array forward(
        const array& x,
        const array& pos_emb,
        const std::optional<array>& attn_mask = std::nullopt
    ) const;

    // Streaming forward with caches
    array streaming_forward(
        const array& x,
        const array& pos_emb,
        array& attn_cache,
        array& conv_cache,
        const std::optional<array>& attn_mask = std::nullopt
    ) const;

private:
    int d_model_;

    SelfAttention self_attn_;
    FeedforwardModule ff1_;
    FeedforwardModule ff2_;
    ConvolutionModule conv_;
    BiasNorm norm_;
};

/**
 * Conv2d subsampling for initial feature processing.
 *
 * Reduces temporal dimension by roughly 2x (T' = (T-7)/2).
 * Architecture matches icefall Conv2dSubsampling:
 * - conv.0: 3x3 conv, 1->8 ch, pad=(0,1), stride=1
 * - conv.4: 3x3 conv, 8->32 ch, pad=0, stride=2
 * - conv.7: 3x3 conv, 32->128 ch, pad=0, stride=(1,2)
 * - ConvNeXt: depthwise 7x7 + pointwise
 * - out: Linear(width*128, out_channels)
 * - out_norm: BiasNorm
 */
class Conv2dSubsampling {
public:
    explicit Conv2dSubsampling(int in_channels, int out_channels);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    array forward(const array& x) const;

    // Streaming forward with cache
    array streaming_forward(const array& x, array& cache) const;

private:
    int in_channels_;
    int out_channels_;
    int layer3_channels_ = 128;  // Fixed per icefall architecture

    // Conv layer 0: 3x3, 1->8, pad=(0,1), stride=1
    array conv0_weight_{zeros({8, 3, 3, 1})};  // (out_ch, H, W, in_ch)
    array conv0_bias_{zeros({8})};

    // Conv layer 4: 3x3, 8->32, pad=0, stride=2
    array conv4_weight_{zeros({32, 3, 3, 8})};
    array conv4_bias_{zeros({32})};

    // Conv layer 7: 3x3, 32->128, pad=0, stride=(1,2)
    array conv7_weight_{zeros({128, 3, 3, 32})};
    array conv7_bias_{zeros({128})};

    // ConvNeXt depthwise conv: 7x7 groups=128
    array convnext_dw_weight_{zeros({128, 7, 7, 1})};
    array convnext_dw_bias_{zeros({128})};

    // ConvNeXt pointwise conv1: 128->384
    array convnext_pw1_weight_{zeros({384, 1, 1, 128})};
    array convnext_pw1_bias_{zeros({384})};

    // ConvNeXt pointwise conv2: 384->128
    array convnext_pw2_weight_{zeros({128, 1, 1, 384})};
    array convnext_pw2_bias_{zeros({128})};

    // Output linear
    array out_weight_{zeros({1})};  // Size depends on input features
    array out_bias_{zeros({1})};

    // BiasNorm
    array out_norm_log_scale_{zeros({})};  // scalar
    array out_norm_bias_{zeros({1})};
};

/**
 * Downsampling module for multi-scale encoder.
 */
class SimpleDownsample {
public:
    SimpleDownsample(int in_dim, int out_dim, int factor);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    array forward(const array& x) const;

private:
    int in_dim_;
    int out_dim_;
    int factor_;

    ScaledLinear linear_;
};

/**
 * Upsampling module for multi-scale encoder.
 */
class SimpleUpsample {
public:
    SimpleUpsample(int in_dim, int out_dim, int factor);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    array forward(const array& x) const;

private:
    int in_dim_;
    int out_dim_;
    int factor_;

    ScaledLinear linear_;
    BiasNorm norm_;
};

/**
 * Learnable skip connection with per-channel scale.
 * Used by Zipformer2EncoderLayer and Zipformer2EncoderStage.
 */
class BypassModule {
public:
    explicit BypassModule(int d_model);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Output = src_orig + scale * (src - src_orig)
    array forward(const array& src_orig, const array& src) const;

private:
    int d_model_;
    array bypass_scale_;
};

/**
 * Compact relative positional encoding for Zipformer2.
 * Generates position embeddings on-the-fly based on sequence length.
 */
class CompactRelPositionalEncoding {
public:
    explicit CompactRelPositionalEncoding(int pos_dim);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Generate positional embeddings for sequence length
    // Returns: (batch, 2*seq-1, pos_dim)
    array forward(const array& x) const;

    // Streaming version with left context
    array forward(const array& x, int left_context_len) const;

private:
    int pos_dim_;
};

// Zipformer2EncoderStage and ZipformerEncoder are declared in zipformer2.hpp
// to avoid circular dependency with Zipformer2EncoderLayer

// Activation functions used in Zipformer
array swoosh_l(const array& x);
array swoosh_r(const array& x);

} // namespace zipformer
