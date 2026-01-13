// Copyright 2024-2025 Andrew Yates
// Zipformer2 encoder components for MLX C++ implementation
//
// Ported from icefall/zipformer
// Licensed under the Apache License, Version 2.0
//
// This file implements the full Zipformer2EncoderLayer structure with:
// - RelPositionMultiheadAttentionWeights (shared attention weight computation)
// - SelfAttention2 (uses pre-computed attention weights)
// - NonlinAttention (non-linear attention with tanh gating)
// - BypassModule (learned skip connections)
// - Zipformer2EncoderLayer (full layer matching checkpoint structure)

#pragma once

#include <mlx/mlx.h>
#include <tuple>
#include <optional>
#include <memory>
#include <vector>

#include "zipformer/array_utils.hpp"
#include "zipformer/scaling.hpp"
#include "zipformer/encoder.hpp"  // For FeedforwardModule, ConvolutionModule, BiasNorm, etc.

namespace zipformer {

using namespace mlx::core;

/**
 * Computes attention weights with relative positional encoding.
 *
 * This is shared across the two SelfAttention modules and NonlinAttention
 * within a Zipformer2EncoderLayer.
 */
class RelPositionMultiheadAttentionWeights {
public:
    RelPositionMultiheadAttentionWeights(
        int d_model,
        int num_heads,
        int query_head_dim,
        int pos_head_dim = 4,
        int pos_emb_dim = 48
    );

    void load_weights(const WeightMap& weights, const std::string& prefix);

    /**
     * Compute attention weights.
     *
     * Args:
     *   x: (seq_len, batch_size, d_model)
     *   pos_emb: (batch, 2*seq-1, pos_emb_dim)
     *   attn_mask: Optional attention mask
     *   key_padding_mask: Optional key padding mask
     *
     * Returns:
     *   Attention weights of shape (batch * heads, seq, seq)
     */
    array forward(
        const array& x,
        const array& pos_emb,
        const std::optional<array>& attn_mask = std::nullopt,
        const std::optional<array>& key_padding_mask = std::nullopt
    ) const;

    /**
     * Streaming forward pass.
     *
     * Args:
     *   x: (seq_len, batch_size, d_model)
     *   pos_emb: (batch, seq_len+kv_len-1, pos_emb_dim)
     *   cached_key: (left_ctx, batch_size, num_heads*query_head_dim)
     *   left_context_len: Number of left context frames
     *   valid_cache_len: Number of valid cached frames
     *
     * Returns:
     *   Tuple of (attn_weights, new_cached_key)
     */
    std::tuple<array, array> streaming_forward(
        const array& x,
        const array& pos_emb,
        const array& cached_key,
        int left_context_len,
        int valid_cache_len = 0
    ) const;

private:
    int d_model_;
    int num_heads_;
    int query_head_dim_;
    int pos_head_dim_;
    int pos_emb_dim_;

    // Q, K, P projection
    // in_proj projects to 2 * num_heads * query_head_dim + num_heads * pos_head_dim
    ScaledLinear in_proj_;

    // Position projection: pos_emb_dim -> num_heads * pos_head_dim
    array linear_pos_weight_{zeros({1})};

    // Helper for relative position shift
    array rel_shift(const array& x, int seq_len) const;
    array rel_shift_streaming(const array& x, int seq_len, int kv_len) const;
};

/**
 * Self-attention module that uses pre-computed attention weights.
 *
 * Unlike the original SelfAttention, this only projects values and
 * applies the attention weights computed by RelPositionMultiheadAttentionWeights.
 */
class SelfAttention2 {
public:
    SelfAttention2(int d_model, int num_heads, int value_head_dim);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    /**
     * Forward pass with pre-computed attention weights.
     *
     * Args:
     *   x: (seq_len, batch_size, d_model)
     *   attn_weights: (batch * heads, seq, seq)
     *
     * Returns:
     *   Output of shape (seq_len, batch_size, d_model)
     */
    array forward(const array& x, const array& attn_weights) const;

    /**
     * Streaming forward pass.
     *
     * Args:
     *   x: (seq_len, batch_size, d_model)
     *   attn_weights: (batch * heads, seq_len, kv_len)
     *   cached_val: (left_ctx, batch_size, attention_dim)
     *   left_context_len: Number of left context frames
     *
     * Returns:
     *   Tuple of (output, new_cached_val)
     */
    std::tuple<array, array> streaming_forward(
        const array& x,
        const array& attn_weights,
        const array& cached_val,
        int left_context_len
    ) const;

private:
    int d_model_;
    int num_heads_;
    int value_head_dim_;
    int attention_dim_;

    ScaledLinear in_proj_;   // Projects to attention_dim
    ScaledLinear out_proj_;  // Projects back to d_model
};

/**
 * Non-linear attention module.
 *
 * Uses sigmoid gating with attention-weighted values.
 * Projects input to (s, v, y), applies tanh(s) * v gating,
 * then applies attention weights and output gate y.
 */
class NonlinAttention {
public:
    NonlinAttention(int d_model, int hidden_channels);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    /**
     * Forward pass.
     *
     * Args:
     *   x: (seq_len, batch_size, d_model)
     *   attn_weights: (batch * heads, seq, seq)
     *
     * Returns:
     *   Output of shape (seq_len, batch_size, d_model)
     */
    array forward(const array& x, const array& attn_weights) const;

    /**
     * Streaming forward pass.
     *
     * Returns:
     *   Tuple of (output, new_cached_v)
     */
    std::tuple<array, array> streaming_forward(
        const array& x,
        const array& attn_weights,
        const array& cached_v,
        int left_context_len
    ) const;

private:
    int d_model_;
    int hidden_channels_;

    ScaledLinear in_proj_;   // Projects to 3 * hidden_channels
    ScaledLinear out_proj_;  // Projects back to d_model
};

// BypassModule is now defined in encoder.hpp

/**
 * Full Zipformer2 encoder layer matching checkpoint structure.
 *
 * Contains:
 * - RelPositionMultiheadAttentionWeights (self_attn_weights)
 * - 2 SelfAttention2 modules (self_attn1, self_attn2)
 * - 3 FeedforwardModules (feed_forward1, feed_forward2, feed_forward3)
 * - 2 ConvolutionModules (conv_module1, conv_module2)
 * - NonlinAttention (nonlin_attention)
 * - 2 BypassModules (bypass, bypass_mid)
 * - bypass_scale vector
 * - BiasNorm (norm)
 *
 * Note: Each feedforward module can have a different hidden dimension.
 */
class Zipformer2EncoderLayer {
public:
    Zipformer2EncoderLayer(
        int d_model,
        int attention_dim,
        int num_heads,
        int ff1_dim,
        int ff2_dim,
        int ff3_dim,
        int kernel_size = 31,
        int pos_head_dim = 4,
        int pos_emb_dim = 48,
        int value_head_dim = 12,
        bool causal = true
    );

    void load_weights(const WeightMap& weights, const std::string& prefix);

    /**
     * Forward pass.
     *
     * Args:
     *   src: (seq_len, batch_size, d_model)
     *   pos_emb: (batch, 2*seq-1, pos_emb_dim)
     *   attn_mask: Optional attention mask
     *   src_key_padding_mask: Optional padding mask
     *
     * Returns:
     *   Output of shape (seq_len, batch_size, d_model)
     */
    array forward(
        const array& src,
        const array& pos_emb,
        const std::optional<array>& attn_mask = std::nullopt,
        const std::optional<array>& src_key_padding_mask = std::nullopt
    ) const;

    /**
     * Streaming forward pass.
     *
     * Args:
     *   src: (seq_len, batch_size, d_model)
     *   pos_emb: (batch, seq_len+kv_len-1, pos_emb_dim)
     *   cached_key: Cached attention keys
     *   cached_val1: Cached values for self_attn1
     *   cached_val2: Cached values for self_attn2
     *   cached_nonlin_attn: Cached values for nonlin_attention
     *   cached_conv1: Cached conv1 context
     *   cached_conv2: Cached conv2 context
     *   left_context_len: Number of left context frames
     *   valid_cache_len: Number of valid cached frames
     *
     * Returns:
     *   Tuple of (output, new caches...)
     */
    std::tuple<array, array, array, array, array, array, array> streaming_forward(
        const array& src,
        const array& pos_emb,
        const array& cached_key,
        const array& cached_val1,
        const array& cached_val2,
        const array& cached_nonlin_attn,
        const array& cached_conv1,
        const array& cached_conv2,
        int left_context_len,
        int valid_cache_len = 0
    ) const;

    // Accessor for testing attention weight computation
    array compute_attn_weights(const array& src, const array& pos_emb) const {
        return self_attn_weights_.forward(src, pos_emb);
    }

    // Accessors for testing intermediate components
    array compute_ff1(const array& x) const { return feed_forward1_.forward(x); }
    array compute_ff2(const array& x) const { return feed_forward2_.forward(x); }
    array compute_ff3(const array& x) const { return feed_forward3_.forward(x); }
    array compute_nonlin_attn(const array& x, const array& attn_weights) const {
        return nonlin_attention_.forward(x, attn_weights);
    }
    array compute_self_attn1(const array& x, const array& attn_weights) const {
        return self_attn1_.forward(x, attn_weights);
    }
    array compute_self_attn2(const array& x, const array& attn_weights) const {
        return self_attn2_.forward(x, attn_weights);
    }
    array compute_conv1(const array& x) const { return conv_module1_.forward(x); }
    array compute_conv2(const array& x) const { return conv_module2_.forward(x); }
    array compute_norm(const array& x) const { return norm_.forward(x); }
    array compute_bypass(const array& src, const array& x) const { return bypass_.forward(src, x); }
    array compute_bypass_mid(const array& src, const array& x) const { return bypass_mid_.forward(src, x); }

private:
    int d_model_;

    // Attention weight computation (shared)
    RelPositionMultiheadAttentionWeights self_attn_weights_;

    // Two self-attention modules
    SelfAttention2 self_attn1_;
    SelfAttention2 self_attn2_;

    // Non-linear attention
    NonlinAttention nonlin_attention_;

    // Three feedforward modules
    FeedforwardModule feed_forward1_;
    FeedforwardModule feed_forward2_;
    FeedforwardModule feed_forward3_;

    // Two convolution modules
    ConvolutionModule conv_module1_;
    ConvolutionModule conv_module2_;

    // Normalization
    BiasNorm norm_;

    // Bypass modules
    BypassModule bypass_;
    BypassModule bypass_mid_;

    // Overall bypass scale (d_model,)
    array bypass_scale_{zeros({1})};
};

/**
 * Zipformer2 encoder stage - wraps multiple Zipformer2EncoderLayers.
 * Stage 0 is direct, stages 1-5 have downsampling/upsampling.
 */
class Zipformer2EncoderStage {
public:
    Zipformer2EncoderStage(
        int d_model,
        int attention_dim,
        int num_heads,
        int ff1_dim,
        int ff2_dim,
        int ff3_dim,
        int num_layers,
        int kernel_size,
        int pos_head_dim,
        int pos_dim,
        int value_head_dim,
        int downsample_factor,
        bool causal
    );

    void load_weights(const WeightMap& weights, const std::string& prefix, int stage_idx);

    array forward(const array& x) const;

    // Streaming forward with state management
    std::tuple<array, std::vector<array>> streaming_forward(
        const array& x,
        const std::vector<array>& cached_states,
        int left_context_len
    ) const;

    int d_model() const { return d_model_; }
    int downsample_factor() const { return downsample_factor_; }

private:
    int d_model_;
    int downsample_factor_;
    bool is_downsampled_;  // true for stages 1-5

    // Positional encoding
    CompactRelPositionalEncoding pos_enc_;

    // Encoder layers
    std::vector<std::unique_ptr<Zipformer2EncoderLayer>> layers_;

    // Softmax-weighted downsampling bias (stages 1-5)
    array downsample_bias_;  // shape: (factor,)

    // Output combiner (bypass) for downsampled stages
    std::unique_ptr<BypassModule> out_combiner_;

    // Helper functions for downsampling/upsampling
    array downsample(const array& x) const;  // Softmax-weighted averaging
    array upsample(const array& x, int target_len) const;  // Repeat to target length
};

/**
 * Full Zipformer encoder with multi-scale stages.
 *
 * Combines outputs from all stages using _get_full_dim_output pattern.
 */
class ZipformerEncoder {
public:
    explicit ZipformerEncoder(const ZipformerConfig& config);

    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Non-streaming forward
    // Input: (batch, time, features)
    // Output: (batch, time', output_dim)
    array forward(const array& x) const;

    // Streaming forward with state management
    array streaming_forward(const array& x, CacheState& state) const;

    const ZipformerConfig& config() const { return config_; }
    int output_dim() const { return config_.output_dim(); }

    // Accessors for testing
    const Conv2dSubsampling& embed() const { return embed_; }
    const std::vector<std::unique_ptr<Zipformer2EncoderStage>>& stages() const { return stages_; }

private:
    ZipformerConfig config_;

    // Conv2d subsampling (encoder_embed)
    Conv2dSubsampling embed_;

    // Multi-scale encoder stages
    std::vector<std::unique_ptr<Zipformer2EncoderStage>> stages_;

    // Final output downsampling
    array downsample_output_bias_;

    // Helper to combine multi-scale outputs
    array get_full_dim_output(const std::vector<array>& outputs) const;
};

} // namespace zipformer
