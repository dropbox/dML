// Copyright 2024-2025 Andrew Yates
// Zipformer2 encoder implementation for MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/zipformer2.hpp"
#include <cmath>
#include <stdexcept>

namespace zipformer {

// ============================================================================
// RelPositionMultiheadAttentionWeights
// ============================================================================

RelPositionMultiheadAttentionWeights::RelPositionMultiheadAttentionWeights(
    int d_model,
    int num_heads,
    int query_head_dim,
    int pos_head_dim,
    int pos_emb_dim
)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , query_head_dim_(query_head_dim)
    , pos_head_dim_(pos_head_dim)
    , pos_emb_dim_(pos_emb_dim)
    // Q, K, P projection dimensions
    , in_proj_(d_model, 2 * num_heads * query_head_dim + num_heads * pos_head_dim, true)
{
    // Initialize linear_pos weight
    linear_pos_weight_ = zeros({num_heads * pos_head_dim, pos_emb_dim});
}

void RelPositionMultiheadAttentionWeights::load_weights(
    const WeightMap& weights,
    const std::string& prefix
) {
    in_proj_.load_weights(weights, prefix + ".in_proj");

    if (has_weight(weights, prefix + ".linear_pos.weight")) {
        linear_pos_weight_ = get_weight(weights, prefix + ".linear_pos.weight");
    }
}

array RelPositionMultiheadAttentionWeights::rel_shift(const array& x, int seq_len) const {
    // Convert relative position scores to absolute position scores.
    // x: (batch_heads, seq_len, 2*seq_len-1)
    // Output: (batch_heads, seq_len, seq_len)
    //
    // For query position q attending to key position k, we need to extract
    // the relative position score at index (seq_len - 1 - q) + k from the
    // input, which represents relative position (k - q).

    // Create index matrix: index[q, k] = (seq_len - 1 - q) + k
    array rows = arange(seq_len - 1.0f, -1.0f, -1.0f);  // [seq_len-1, ..., 0]
    array cols = arange(0.0f, static_cast<float>(seq_len), 1.0f);  // [0, 1, ..., seq_len-1]

    // Broadcast to create (seq_len, seq_len) index matrix
    array row_expanded = expand_dims(rows, 1);  // (seq_len, 1)
    array col_expanded = expand_dims(cols, 0);  // (1, seq_len)
    array indexes = add(row_expanded, col_expanded);  // (seq_len, seq_len)
    indexes = expand_dims(indexes, 0);  // (1, seq_len, seq_len)
    indexes = astype(indexes, int32);

    // Gather along last axis
    return take_along_axis(x, indexes, 2);
}

array RelPositionMultiheadAttentionWeights::rel_shift_streaming(
    const array& x,
    int seq_len,
    int kv_len
) const {
    // Streaming version where key/value has kv_len positions.
    // x: (batch_heads, seq_len, seq_len + kv_len - 1)
    // Output: (batch_heads, seq_len, kv_len)

    array rows = arange(seq_len - 1.0f, -1.0f, -1.0f);
    array cols = arange(0.0f, static_cast<float>(kv_len), 1.0f);

    array row_expanded = expand_dims(rows, 1);
    array col_expanded = expand_dims(cols, 0);
    array indexes = add(row_expanded, col_expanded);
    indexes = expand_dims(indexes, 0);
    indexes = astype(indexes, int32);

    return take_along_axis(x, indexes, 2);
}

array RelPositionMultiheadAttentionWeights::forward(
    const array& x,
    const array& pos_emb,
    const std::optional<array>& attn_mask,
    const std::optional<array>& key_padding_mask
) const {
    // x: (seq_len, batch_size, d_model)
    // pos_emb: (batch, 2*seq-1, pos_emb_dim)

    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];

    // Project input to Q, K, P
    array proj = in_proj_.forward(x);  // (seq, batch, in_proj_dim)

    // Split into Q, K, P
    int query_dim = num_heads_ * query_head_dim_;
    int pos_dim = num_heads_ * pos_head_dim_;

    array q = slice(proj, {0, 0, 0}, {seq_len, batch_size, query_dim});
    array k = slice(proj, {0, 0, query_dim}, {seq_len, batch_size, 2 * query_dim});
    array p = slice(proj, {0, 0, 2 * query_dim}, {seq_len, batch_size, 2 * query_dim + pos_dim});

    // Reshape for multi-head: (seq, batch, heads*dim) -> (seq, batch, heads, dim)
    q = reshape(q, {seq_len, batch_size, num_heads_, query_head_dim_});
    k = reshape(k, {seq_len, batch_size, num_heads_, query_head_dim_});
    p = reshape(p, {seq_len, batch_size, num_heads_, pos_head_dim_});

    // Transpose to (batch, heads, seq, dim)
    q = transpose(q, {1, 2, 0, 3});
    k = transpose(k, {1, 2, 0, 3});
    p = transpose(p, {1, 2, 0, 3});

    // Project position embeddings: (batch, 2*seq-1, pos_emb_dim) @ (pos_emb_dim, heads*pos_head_dim)
    array pos_proj = matmul(pos_emb, transpose(linear_pos_weight_));
    pos_proj = reshape(pos_proj, {batch_size, -1, num_heads_, pos_head_dim_});
    pos_proj = transpose(pos_proj, {0, 2, 1, 3});  // (batch, heads, 2*seq-1, pos_head_dim)

    // Content attention: Q @ K^T (no scaling - matches icefall)
    array content_score = matmul(q, transpose(k, {0, 1, 3, 2}));  // (batch, heads, seq, seq)

    // Position attention: P @ pos_proj^T
    array pos_score = matmul(p, transpose(pos_proj, {0, 1, 3, 2}));  // (batch, heads, seq, 2*seq-1)
    pos_score = reshape(pos_score, {batch_size * num_heads_, seq_len, -1});
    pos_score = rel_shift(pos_score, seq_len);
    pos_score = reshape(pos_score, {batch_size, num_heads_, seq_len, seq_len});

    array attn_score = add(content_score, pos_score);

    // Apply masks
    if (attn_mask.has_value()) {
        attn_score = add(attn_score, *attn_mask);
    }

    if (key_padding_mask.has_value()) {
        // key_padding_mask: (batch, seq)
        array mask = expand_dims(expand_dims(*key_padding_mask, 1), 1);  // (batch, 1, 1, seq)
        attn_score = where(mask, full(attn_score.shape(), -1e9f), attn_score);
    }

    // Softmax
    array attn_weights = softmax(attn_score, -1);

    // Reshape to (batch * heads, seq, seq)
    attn_weights = reshape(attn_weights, {batch_size * num_heads_, seq_len, seq_len});

    return attn_weights;
}

std::tuple<array, array> RelPositionMultiheadAttentionWeights::streaming_forward(
    const array& x,
    const array& pos_emb,
    const array& cached_key,
    int left_context_len,
    int valid_cache_len
) const {
    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];
    int kv_len = left_context_len + seq_len;

    // Project input
    array proj = in_proj_.forward(x);

    int query_dim = num_heads_ * query_head_dim_;
    int pos_dim = num_heads_ * pos_head_dim_;

    array q = slice(proj, {0, 0, 0}, {seq_len, batch_size, query_dim});
    array k = slice(proj, {0, 0, query_dim}, {seq_len, batch_size, 2 * query_dim});
    array p = slice(proj, {0, 0, 2 * query_dim}, {seq_len, batch_size, 2 * query_dim + pos_dim});

    // Concatenate cached keys with new keys
    // cached_key: (left_ctx, batch, query_dim)
    array k_extended = concatenate({cached_key, k}, 0);  // (kv_len, batch, query_dim)

    // Update key cache: keep last left_context_len frames
    int start_idx = std::max(0, static_cast<int>(k_extended.shape()[0]) - left_context_len);
    array new_cached_key = slice(k_extended, {start_idx, 0, 0},
                                 {static_cast<int>(k_extended.shape()[0]), batch_size, query_dim});

    // Reshape for multi-head
    q = reshape(q, {seq_len, batch_size, num_heads_, query_head_dim_});
    k_extended = reshape(k_extended, {kv_len, batch_size, num_heads_, query_head_dim_});
    p = reshape(p, {seq_len, batch_size, num_heads_, pos_head_dim_});

    // Transpose
    q = transpose(q, {1, 2, 0, 3});  // (batch, heads, seq, dim)
    k_extended = transpose(k_extended, {1, 2, 0, 3});  // (batch, heads, kv_len, dim)
    p = transpose(p, {1, 2, 0, 3});

    // Project position embeddings
    array pos_proj = matmul(pos_emb, transpose(linear_pos_weight_));
    int rel_len = seq_len + kv_len - 1;
    pos_proj = reshape(pos_proj, {batch_size, rel_len, num_heads_, pos_head_dim_});
    pos_proj = transpose(pos_proj, {0, 2, 1, 3});

    // Content attention
    array content_score = matmul(q, transpose(k_extended, {0, 1, 3, 2}));  // (batch, heads, seq, kv_len)

    // Position attention with streaming rel_shift
    array pos_score = matmul(p, transpose(pos_proj, {0, 1, 3, 2}));  // (batch, heads, seq, rel_len)
    pos_score = reshape(pos_score, {batch_size * num_heads_, seq_len, rel_len});
    pos_score = rel_shift_streaming(pos_score, seq_len, kv_len);
    pos_score = reshape(pos_score, {batch_size, num_heads_, seq_len, kv_len});

    array attn_score = add(content_score, pos_score);

    // Mask out invalid cache positions (positions beyond valid_cache_len)
    if (valid_cache_len < left_context_len) {
        int invalid_len = left_context_len - valid_cache_len;
        // Create mask for first invalid_len positions
        array mask_indices = arange(0.0f, static_cast<float>(kv_len), 1.0f);
        array mask = less(mask_indices, array(static_cast<float>(invalid_len)));
        mask = expand_dims(expand_dims(expand_dims(mask, 0), 0), 0);  // (1, 1, 1, kv_len)
        attn_score = where(mask, full(attn_score.shape(), -1e9f), attn_score);
    }

    // Softmax
    array attn_weights = softmax(attn_score, -1);
    attn_weights = reshape(attn_weights, {batch_size * num_heads_, seq_len, kv_len});

    return std::make_tuple(attn_weights, new_cached_key);
}

// ============================================================================
// SelfAttention2
// ============================================================================

SelfAttention2::SelfAttention2(int d_model, int num_heads, int value_head_dim)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , value_head_dim_(value_head_dim)
    , attention_dim_(num_heads * value_head_dim)
    , in_proj_(d_model, attention_dim_, true)
    , out_proj_(attention_dim_, d_model, true)
{
}

void SelfAttention2::load_weights(const WeightMap& weights, const std::string& prefix) {
    in_proj_.load_weights(weights, prefix + ".in_proj");
    out_proj_.load_weights(weights, prefix + ".out_proj");
}

array SelfAttention2::forward(const array& x, const array& attn_weights) const {
    // x: (seq_len, batch_size, d_model)
    // attn_weights: (batch * heads, seq, seq)

    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];

    // Project values
    array v = in_proj_.forward(x);  // (seq, batch, attention_dim)
    v = reshape(v, {seq_len, batch_size, num_heads_, value_head_dim_});
    v = transpose(v, {1, 2, 0, 3});  // (batch, heads, seq, head_dim)
    v = reshape(v, {batch_size * num_heads_, seq_len, value_head_dim_});

    // Apply attention
    array out = matmul(attn_weights, v);  // (batch*heads, seq, head_dim)
    out = reshape(out, {batch_size, num_heads_, seq_len, value_head_dim_});
    out = transpose(out, {2, 0, 1, 3});  // (seq, batch, heads, head_dim)
    out = reshape(out, {seq_len, batch_size, attention_dim_});

    return out_proj_.forward(out);
}

std::tuple<array, array> SelfAttention2::streaming_forward(
    const array& x,
    const array& attn_weights,
    const array& cached_val,
    int left_context_len
) const {
    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];

    // Project current chunk values
    array v = in_proj_.forward(x);  // (seq, batch, attention_dim)

    // Concatenate cached values with new values
    array v_extended = concatenate({cached_val, v}, 0);  // (left_ctx + seq, batch, attention_dim)

    // Update cache
    int start_idx = std::max(0, static_cast<int>(v_extended.shape()[0]) - left_context_len);
    array new_cached_val = slice(v_extended, {start_idx, 0, 0},
                                 {static_cast<int>(v_extended.shape()[0]), batch_size, attention_dim_});

    // Reshape for multi-head attention
    int kv_len = left_context_len + seq_len;
    v_extended = reshape(v_extended, {kv_len, batch_size, num_heads_, value_head_dim_});
    v_extended = transpose(v_extended, {1, 2, 0, 3});  // (batch, heads, kv_len, head_dim)
    v_extended = reshape(v_extended, {batch_size * num_heads_, kv_len, value_head_dim_});

    // Apply attention
    array out = matmul(attn_weights, v_extended);  // (batch*heads, seq_len, head_dim)
    out = reshape(out, {batch_size, num_heads_, seq_len, value_head_dim_});
    out = transpose(out, {2, 0, 1, 3});  // (seq, batch, heads, head_dim)
    out = reshape(out, {seq_len, batch_size, attention_dim_});

    return std::make_tuple(out_proj_.forward(out), new_cached_val);
}

// ============================================================================
// NonlinAttention
// ============================================================================

NonlinAttention::NonlinAttention(int d_model, int hidden_channels)
    : d_model_(d_model)
    , hidden_channels_(hidden_channels)
    , in_proj_(d_model, 3 * hidden_channels, true)
    , out_proj_(hidden_channels, d_model, true)
{
}

void NonlinAttention::load_weights(const WeightMap& weights, const std::string& prefix) {
    in_proj_.load_weights(weights, prefix + ".in_proj");
    out_proj_.load_weights(weights, prefix + ".out_proj");
}

array NonlinAttention::forward(const array& x, const array& attn_weights) const {
    // x: (seq_len, batch_size, d_model)
    // attn_weights: (batch * heads, seq, seq)

    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];

    // Project and split
    array proj = in_proj_.forward(x);  // (seq, batch, 3*hidden)
    auto splits = split(proj, 3, -1);
    array s = splits[0];
    array v = splits[1];
    array y = splits[2];

    // Tanh gate
    s = tanh(s);
    v = multiply(v, s);

    // Apply attention using all heads
    int total_heads = attn_weights.shape()[0];
    int num_heads = total_heads / batch_size;
    int head_dim = hidden_channels_ / num_heads;

    // Reshape v for multi-head attention
    v = reshape(v, {seq_len, batch_size, num_heads, head_dim});
    v = transpose(v, {1, 2, 0, 3});  // (batch, heads, seq, head_dim)
    v = reshape(v, {batch_size * num_heads, seq_len, head_dim});

    // Apply attention
    array out = matmul(attn_weights, v);  // (batch*heads, seq, head_dim)

    // Reshape back
    out = reshape(out, {batch_size, num_heads, seq_len, head_dim});
    out = transpose(out, {2, 0, 1, 3});  // (seq, batch, heads, head_dim)
    out = reshape(out, {seq_len, batch_size, hidden_channels_});

    // Output gate and projection
    out = multiply(out, y);
    return out_proj_.forward(out);
}

std::tuple<array, array> NonlinAttention::streaming_forward(
    const array& x,
    const array& attn_weights,
    const array& cached_v,
    int left_context_len
) const {
    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];

    // Project and split
    array proj = in_proj_.forward(x);
    auto splits = split(proj, 3, -1);
    array s = splits[0];
    array v = splits[1];
    array y = splits[2];

    // Tanh gate
    s = tanh(s);
    v = multiply(v, s);

    // Concatenate cached v with new v
    array v_extended = concatenate({cached_v, v}, 0);  // (left_ctx + seq, batch, hidden)

    // Update cache
    int start_idx = std::max(0, static_cast<int>(v_extended.shape()[0]) - left_context_len);
    array new_cached_v = slice(v_extended, {start_idx, 0, 0},
                               {static_cast<int>(v_extended.shape()[0]), batch_size, hidden_channels_});

    // Apply attention using all heads
    int total_heads = attn_weights.shape()[0];
    int num_heads = total_heads / batch_size;
    int head_dim = hidden_channels_ / num_heads;
    int kv_len = left_context_len + seq_len;

    // Reshape v_extended for multi-head attention
    v_extended = reshape(v_extended, {kv_len, batch_size, num_heads, head_dim});
    v_extended = transpose(v_extended, {1, 2, 0, 3});  // (batch, heads, kv_len, head_dim)
    v_extended = reshape(v_extended, {batch_size * num_heads, kv_len, head_dim});

    // Apply attention
    array out = matmul(attn_weights, v_extended);  // (batch*heads, seq_len, head_dim)

    // Reshape back
    out = reshape(out, {batch_size, num_heads, seq_len, head_dim});
    out = transpose(out, {2, 0, 1, 3});  // (seq, batch, heads, head_dim)
    out = reshape(out, {seq_len, batch_size, hidden_channels_});

    // Output gate and projection
    out = multiply(out, y);
    return std::make_tuple(out_proj_.forward(out), new_cached_v);
}

// BypassModule implementation is in encoder.cpp

// ============================================================================
// Zipformer2EncoderLayer
// ============================================================================

Zipformer2EncoderLayer::Zipformer2EncoderLayer(
    int d_model,
    int attention_dim,
    int num_heads,
    int ff1_dim,
    int ff2_dim,
    int ff3_dim,
    int kernel_size,
    int pos_head_dim,
    int pos_emb_dim,
    int value_head_dim,
    bool causal
)
    : d_model_(d_model)
    , self_attn_weights_(d_model, num_heads, attention_dim / num_heads, pos_head_dim, pos_emb_dim)
    , self_attn1_(d_model, num_heads, value_head_dim)
    , self_attn2_(d_model, num_heads, value_head_dim)
    , nonlin_attention_(d_model, 3 * d_model / 4)
    , feed_forward1_(d_model, ff1_dim)
    , feed_forward2_(d_model, ff2_dim)
    , feed_forward3_(d_model, ff3_dim)
    , conv_module1_(d_model, kernel_size, causal)
    , conv_module2_(d_model, kernel_size, causal)
    , norm_(d_model)
    , bypass_(d_model)
    , bypass_mid_(d_model)
{
    // Initialize bypass_scale to 0.5
    bypass_scale_ = multiply(ones({d_model}), array(0.5f));
}

void Zipformer2EncoderLayer::load_weights(const WeightMap& weights, const std::string& prefix) {
    self_attn_weights_.load_weights(weights, prefix + ".self_attn_weights");
    self_attn1_.load_weights(weights, prefix + ".self_attn1");
    self_attn2_.load_weights(weights, prefix + ".self_attn2");
    nonlin_attention_.load_weights(weights, prefix + ".nonlin_attention");
    feed_forward1_.load_weights(weights, prefix + ".feed_forward1");
    feed_forward2_.load_weights(weights, prefix + ".feed_forward2");
    feed_forward3_.load_weights(weights, prefix + ".feed_forward3");
    conv_module1_.load_weights(weights, prefix + ".conv_module1");
    conv_module2_.load_weights(weights, prefix + ".conv_module2");
    norm_.load_weights(weights, prefix + ".norm");
    bypass_.load_weights(weights, prefix + ".bypass");
    bypass_mid_.load_weights(weights, prefix + ".bypass_mid");

    if (has_weight(weights, prefix + ".bypass_scale")) {
        bypass_scale_ = get_weight(weights, prefix + ".bypass_scale");
    }
}

array Zipformer2EncoderLayer::forward(
    const array& src,
    const array& pos_emb,
    const std::optional<array>& attn_mask,
    const std::optional<array>& src_key_padding_mask
) const {
    array src_orig = src;
    array out = src;

    // Compute attention weights (shared)
    array attn_weights = self_attn_weights_.forward(
        out, pos_emb, attn_mask, src_key_padding_mask
    );

    // First feedforward
    out = add(out, feed_forward1_.forward(out));

    // Non-linear attention
    out = add(out, nonlin_attention_.forward(out, attn_weights));

    // First self-attention
    out = add(out, self_attn1_.forward(out, attn_weights));

    // First convolution
    out = add(out, conv_module1_.forward(out, src_key_padding_mask));

    // Second feedforward
    out = add(out, feed_forward2_.forward(out));

    // Mid-layer bypass
    out = bypass_mid_.forward(src_orig, out);

    // Second self-attention
    out = add(out, self_attn2_.forward(out, attn_weights));

    // Second convolution
    out = add(out, conv_module2_.forward(out, src_key_padding_mask));

    // Third feedforward
    out = add(out, feed_forward3_.forward(out));

    // Normalize
    out = norm_.forward(out);

    // Final bypass
    out = bypass_.forward(src_orig, out);

    return out;
}

std::tuple<array, array, array, array, array, array, array>
Zipformer2EncoderLayer::streaming_forward(
    const array& src,
    const array& pos_emb,
    const array& cached_key,
    const array& cached_val1,
    const array& cached_val2,
    const array& cached_nonlin_attn,
    const array& cached_conv1,
    const array& cached_conv2,
    int left_context_len,
    int valid_cache_len
) const {
    array src_orig = src;
    array out = src;

    // Compute streaming attention weights with cache
    auto [attn_weights, new_cached_key] = self_attn_weights_.streaming_forward(
        out, pos_emb, cached_key, left_context_len, valid_cache_len
    );

    // First feedforward
    out = add(out, feed_forward1_.forward(out));

    // First self-attention (streaming)
    auto [attn1_out, new_cached_val1] = self_attn1_.streaming_forward(
        out, attn_weights, cached_val1, left_context_len
    );
    out = add(out, attn1_out);

    // Non-linear attention (streaming)
    auto [nonlin_out, new_cached_nonlin_attn] = nonlin_attention_.streaming_forward(
        out, attn_weights, cached_nonlin_attn, left_context_len
    );
    out = add(out, nonlin_out);

    // First convolution (streaming)
    array conv1_cache = cached_conv1;  // Copy for modification
    array conv1_out = conv_module1_.streaming_forward(out, conv1_cache);
    out = add(out, conv1_out);

    // Second feedforward
    out = add(out, feed_forward2_.forward(out));

    // Mid-layer bypass
    out = bypass_mid_.forward(src_orig, out);

    // Second self-attention (streaming)
    auto [attn2_out, new_cached_val2] = self_attn2_.streaming_forward(
        out, attn_weights, cached_val2, left_context_len
    );
    out = add(out, attn2_out);

    // Second convolution (streaming)
    array conv2_cache = cached_conv2;
    array conv2_out = conv_module2_.streaming_forward(out, conv2_cache);
    out = add(out, conv2_out);

    // Third feedforward
    out = add(out, feed_forward3_.forward(out));

    // Normalize
    out = norm_.forward(out);

    // Final bypass
    out = bypass_.forward(src_orig, out);

    return std::make_tuple(
        out,
        new_cached_key,
        new_cached_val1,
        new_cached_val2,
        new_cached_nonlin_attn,
        conv1_cache,
        conv2_cache
    );
}

} // namespace zipformer
