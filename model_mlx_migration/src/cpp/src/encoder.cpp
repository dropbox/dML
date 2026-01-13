// Copyright 2024-2025 Andrew Yates
// Encoder implementation for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/encoder.hpp"
#include "zipformer/zipformer2.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace zipformer {

// ============================================================================
// Activation Functions
// ============================================================================

array swoosh_l(const array& x) {
    // SwooshL(x) = softplus(x - 4) - 0.08 * x - 0.035
    // Matches icefall Zipformer ConvNext activation
    // Formula: log(1 + exp(x - 4)) - 0.08*x - 0.035
    array z = subtract(x, array(4.0f));
    // Numerically stable softplus: softplus(z) = z if z > 20, else log1p(exp(z))
    array softplus_z = where(
        greater(z, array(20.0f)),
        z,
        log1p(exp(z))
    );
    array linear_term = multiply(array(0.08f), x);
    return subtract(subtract(softplus_z, linear_term), array(0.035f));
}

array swoosh_r(const array& x) {
    // SwooshR(x) = softplus(x - 1) - 0.08 * x - 0.313261687
    // Matches icefall Zipformer activation
    // Formula: log(1 + exp(x - 1)) - 0.08*x - 0.313261687
    array z = subtract(x, array(1.0f));
    // Numerically stable softplus: softplus(z) = z if z > 20, else log1p(exp(z))
    array softplus_z = where(
        greater(z, array(20.0f)),
        z,
        log1p(exp(z))
    );
    array linear_term = multiply(array(0.08f), x);
    return subtract(subtract(softplus_z, linear_term), array(0.31326168f));
}

// ============================================================================
// ConvolutionModule
// ============================================================================

ConvolutionModule::ConvolutionModule(int d_model, int kernel_size, bool causal)
    : d_model_(d_model)
    , kernel_size_(kernel_size)
    , causal_(causal)
    , in_proj_(d_model, 2 * d_model, true)
    , out_proj_(d_model, d_model, true)
{
    if (causal) {
        int half_kernel = (kernel_size + 1) / 2;
        causal_conv_weight_ = zeros({d_model, half_kernel, 1});
        causal_conv_bias_ = zeros({d_model});
        chunkwise_conv_weight_ = zeros({d_model, kernel_size, 1});
        chunkwise_conv_bias_ = zeros({d_model});
        chunkwise_conv_scale_ = zeros({2, d_model, kernel_size});
    } else {
        depthwise_conv_weight_ = zeros({d_model, kernel_size, 1});
        depthwise_conv_bias_ = zeros({d_model});
    }
}

void ConvolutionModule::load_weights(const WeightMap& weights, const std::string& prefix) {
    in_proj_.load_weights(weights, prefix + ".in_proj");
    out_proj_.load_weights(weights, prefix + ".out_proj");

    if (causal_) {
        // Load causal conv weights (checkpoint structure: depthwise_conv.causal_conv.*)
        // Checkpoint: (out, 1, k), need: (out, k, 1) - transpose last two dims
        if (has_weight(weights, prefix + ".depthwise_conv.causal_conv.weight")) {
            array w = get_weight(weights, prefix + ".depthwise_conv.causal_conv.weight");
            causal_conv_weight_ = transpose(w, {0, 2, 1});  // (out, 1, k) -> (out, k, 1)
        }
        if (has_weight(weights, prefix + ".depthwise_conv.causal_conv.bias")) {
            causal_conv_bias_ = get_weight(weights, prefix + ".depthwise_conv.causal_conv.bias");
        }
        // Load chunkwise conv weights - same transpose
        if (has_weight(weights, prefix + ".depthwise_conv.chunkwise_conv.weight")) {
            array w = get_weight(weights, prefix + ".depthwise_conv.chunkwise_conv.weight");
            chunkwise_conv_weight_ = transpose(w, {0, 2, 1});  // (out, 1, k) -> (out, k, 1)
        }
        if (has_weight(weights, prefix + ".depthwise_conv.chunkwise_conv.bias")) {
            chunkwise_conv_bias_ = get_weight(weights, prefix + ".depthwise_conv.chunkwise_conv.bias");
        }
        if (has_weight(weights, prefix + ".depthwise_conv.chunkwise_conv_scale")) {
            chunkwise_conv_scale_ = get_weight(weights, prefix + ".depthwise_conv.chunkwise_conv_scale");
        }
    } else {
        // Non-causal also needs transpose: (out, 1, k) -> (out, k, 1)
        if (has_weight(weights, prefix + ".depthwise_conv.weight")) {
            array w = get_weight(weights, prefix + ".depthwise_conv.weight");
            depthwise_conv_weight_ = transpose(w, {0, 2, 1});
        }
        if (has_weight(weights, prefix + ".depthwise_conv.bias")) {
            depthwise_conv_bias_ = get_weight(weights, prefix + ".depthwise_conv.bias");
        }
    }
}

array ConvolutionModule::get_chunk_scale(int chunk_size) const {
    // Implementation matches Python _get_chunk_scale
    array left_edge = slice(chunkwise_conv_scale_, {0, 0, 0},
                            {1, d_model_, kernel_size_});
    array right_edge = slice(chunkwise_conv_scale_, {1, 0, 0},
                             {2, d_model_, kernel_size_});

    left_edge = squeeze(left_edge, 0);  // (d_model, kernel_size)
    right_edge = squeeze(right_edge, 0);

    if (chunk_size < kernel_size_) {
        left_edge = slice(left_edge, {0, 0}, {d_model_, chunk_size});
        right_edge = slice(right_edge, {0, kernel_size_ - chunk_size},
                           {d_model_, kernel_size_});
    } else {
        int t = chunk_size - kernel_size_;
        array pad_arr = zeros({d_model_, t});
        left_edge = concatenate({left_edge, pad_arr}, 1);
        right_edge = concatenate({pad_arr, right_edge}, 1);
    }

    // scale = 1 + (left + right), then transpose
    array scale = add(array(1.0f), add(left_edge, right_edge));
    scale = transpose(scale);  // (chunk_size, d_model)
    scale = expand_dims(scale, 0);  // (1, chunk_size, d_model)
    return scale;
}

array ConvolutionModule::causal_depthwise_conv(const array& x) const {
    // Chunk-causal depthwise convolution (matches icefall ChunkCausalDepthwiseConv1d)
    // x: (batch, seq, d_model)
    // Returns: x_causal + x_chunk * chunk_scale

    int batch_size = x.shape()[0];
    int seq_len = x.shape()[1];
    int half_kernel = causal_conv_weight_.shape()[1];  // kernel_size after transpose
    int left_pad = kernel_size_ / 2;  // 15 for kernel_size=31

    // === Causal component ===
    // Pad left only
    array x_padded = pad(x, {{0, 0}, {left_pad, 0}, {0, 0}});
    // conv1d with groups=d_model for depthwise conv
    array x_causal = conv1d(x_padded, causal_conv_weight_, /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/d_model_);
    x_causal = add(x_causal, reshape(causal_conv_bias_, {1, 1, d_model_}));

    // === Chunkwise component ===
    // Symmetric padding for chunkwise conv
    int padding = kernel_size_ / 2;
    array x_chunk_padded = pad(x, {{0, 0}, {padding, padding}, {0, 0}});
    array x_chunk = conv1d(x_chunk_padded, chunkwise_conv_weight_, /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/d_model_);
    x_chunk = add(x_chunk, reshape(chunkwise_conv_bias_, {1, 1, d_model_}));

    // Apply chunk edge scaling
    array chunk_scale = get_chunk_scale(seq_len);  // (1, seq_len, d_model)
    x_chunk = multiply(x_chunk, chunk_scale);

    return add(x_causal, x_chunk);
}

array ConvolutionModule::forward(
    const array& x,
    const std::optional<array>& padding_mask
) const {
    // Input: (seq, batch, d_model)
    // Input projection and gating
    array proj = in_proj_.forward(x);
    auto split_result = split(proj, 2, -1);
    array x_proj = split_result[0];
    array gate = split_result[1];
    x_proj = multiply(x_proj, sigmoid(gate));

    // Transpose to (batch, seq, d_model) for conv
    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];
    array x_t = transpose(x_proj, {1, 0, 2});

    // Apply padding mask
    if (padding_mask.has_value()) {
        array mask = expand_dims(padding_mask.value(), -1);
        x_t = where(mask, zeros_like(x_t), x_t);
    }

    // Depthwise convolution
    array conv_out = causal_ ? causal_depthwise_conv(x_t) : x_t;

    // Transpose back
    conv_out = transpose(conv_out, {1, 0, 2});

    // SwooshR and output projection
    conv_out = swoosh_r(conv_out);
    return out_proj_.forward(conv_out);
}

array ConvolutionModule::streaming_forward(
    const array& x,
    array& conv_cache,
    const std::optional<array>& padding_mask
) const {
    // Streaming implementation would use conv_cache
    // For now, delegate to non-streaming
    return forward(x, padding_mask);
}

// ============================================================================
// FeedforwardModule
// ============================================================================

FeedforwardModule::FeedforwardModule(int d_model, int d_ff, int dropout)
    : d_model_(d_model)
    , d_ff_(d_ff)
    , norm_(d_model)
    , linear1_(d_model, d_ff, true)
    , linear2_(d_ff, d_model, true)
{
}

void FeedforwardModule::load_weights(const WeightMap& weights, const std::string& prefix) {
    // Try loading norm - some checkpoints have it, some don't
    if (has_weight(weights, prefix + ".norm.bias")) {
        norm_.load_weights(weights, prefix + ".norm");
        has_norm_ = true;
    } else {
        has_norm_ = false;
    }

    // Try both key naming conventions:
    // Checkpoint style: in_proj/out_proj
    // Original style: linear1/linear2
    if (has_weight(weights, prefix + ".in_proj.weight")) {
        linear1_.load_weights(weights, prefix + ".in_proj");
        linear2_.load_weights(weights, prefix + ".out_proj");
    } else {
        linear1_.load_weights(weights, prefix + ".linear1");
        linear2_.load_weights(weights, prefix + ".linear2");
    }
}

array FeedforwardModule::forward(const array& x) const {
    // x -> [norm] -> linear1 -> SwooshL -> linear2
    array out = x;
    if (has_norm_) {
        out = norm_.forward(out);
    }
    out = linear1_.forward(out);
    out = swoosh_l(out);
    return linear2_.forward(out);
}

// ============================================================================
// RelPositionalEncoding
// ============================================================================

RelPositionalEncoding::RelPositionalEncoding(int d_model, int max_len)
    : d_model_(d_model)
    , max_len_(max_len)
{
    // Pre-compute sinusoidal positional encodings
    // positions: (-max_len+1, ..., max_len-1) for relative positions
    int total_len = 2 * max_len - 1;
    array positions = arange(-max_len + 1.0f, static_cast<float>(max_len), 1.0f);

    // Dimension indices
    array dim_indices = arange(0.0f, static_cast<float>(d_model), 2.0f);
    array freqs = exp(multiply(dim_indices, array(-std::log(10000.0f) / d_model)));

    // (total_len, d_model/2)
    array angles = matmul(
        reshape(positions, {total_len, 1}),
        reshape(freqs, {1, d_model / 2})
    );

    // Interleave sin and cos
    pe_ = zeros({total_len, d_model});
    // This is simplified - full impl needs proper interleaving
}

void RelPositionalEncoding::load_weights(const WeightMap& weights, const std::string& prefix) {
    if (has_weight(weights, prefix + ".pe")) {
        pe_ = get_weight(weights, prefix + ".pe");
    }
}

array RelPositionalEncoding::forward(int seq_len) const {
    // Return positional encodings for sequence length
    int center = max_len_ - 1;
    return slice(pe_, {center - seq_len + 1, 0}, {center + seq_len, d_model_});
}

// ============================================================================
// SelfAttention
// ============================================================================

namespace {

array softmax_last_dim(const array& x) {
    // Stable softmax over last dimension.
    int axis = -1;
    array x_max = max(x, axis, /*keepdims=*/true);
    array exps = exp(subtract(x, x_max));
    array denom = sum(exps, axis, /*keepdims=*/true);
    return divide(exps, denom);
}

} // namespace

SelfAttention::SelfAttention(
    int d_model,
    int attention_dim,
    int num_heads,
    int pos_dim,
    int pos_head_dim,
    int value_head_dim
)
    : d_model_(d_model)
    , attention_dim_(attention_dim)
    , num_heads_(num_heads)
    , pos_dim_(pos_dim)
    , pos_head_dim_(pos_head_dim)
    , value_head_dim_(value_head_dim)
    , norm_(d_model)
    , in_proj_(d_model, num_heads * (2 * (attention_dim / num_heads) + value_head_dim), true)
    , pos_proj_(pos_dim, num_heads * pos_head_dim, true)
    , out_proj_(num_heads * value_head_dim, d_model, true)
{
}

void SelfAttention::load_weights(const WeightMap& weights, const std::string& prefix) {
    norm_.load_weights(weights, prefix + ".norm");
    in_proj_.load_weights(weights, prefix + ".in_proj");
    pos_proj_.load_weights(weights, prefix + ".pos_proj");
    out_proj_.load_weights(weights, prefix + ".out_proj");
}

array SelfAttention::forward(
    const array& x,
    const array& pos_emb,
    const std::optional<array>& attn_mask
) const {
    // Minimal multi-head attention implementation (relative position term TODO).
    array normed = norm_.forward(x);
    array qkv = in_proj_.forward(normed);

    int batch = x.shape()[0];
    int time = x.shape()[1];

    int head_dim = attention_dim_ / num_heads_;
    int per_head_dim = 2 * head_dim + value_head_dim_;

    // (B, T, H*(2*Dq + Dv)) -> (B, T, H, 2*Dq + Dv)
    qkv = reshape(qkv, {batch, time, num_heads_, per_head_dim});

    // Split into q, k, v (B, T, H, D)
    array q = slice(qkv, {0, 0, 0, 0}, {batch, time, num_heads_, head_dim});
    array k = slice(qkv, {0, 0, 0, head_dim}, {batch, time, num_heads_, 2 * head_dim});
    array v = slice(
        qkv,
        {0, 0, 0, 2 * head_dim},
        {batch, time, num_heads_, 2 * head_dim + value_head_dim_}
    );

    // (B, T, H, D) -> (B, H, T, D)
    q = transpose_for_attention(q);
    k = transpose_for_attention(k);
    v = transpose_for_attention(v);

    // Attention scores: (B, H, T, Dq) @ (B, H, Dq, T) -> (B, H, T, T)
    array scores = matmul(q, transpose(k, {0, 1, 3, 2}));
    scores = multiply(scores, array(1.0f / std::sqrt(static_cast<float>(head_dim))));

    if (attn_mask.has_value()) {
        array mask = *attn_mask;
        if (mask.ndim() == 2) {
            mask = expand_dims(expand_dims(mask, 0), 0); // (1,1,T,T)
        }
        scores = add(scores, mask);
    }

    // Relative position term TODO; compute pos projection to keep interface exercised.
    (void)pos_proj_.forward(pos_emb);

    array attn = softmax_last_dim(scores);
    array out = matmul(attn, v);               // (B, H, T, Dv)
    out = transpose_from_attention(out);       // (B, T, H, Dv)
    out = reshape(out, {batch, time, num_heads_ * value_head_dim_}); // (B, T, H*Dv)
    return out_proj_.forward(out);             // (B, T, d_model)
}

array SelfAttention::streaming_forward(
    const array& x,
    const array& pos_emb,
    array& kv_cache,
    const std::optional<array>& attn_mask
) const {
    // Streaming would update kv_cache
    (void)kv_cache;
    return forward(x, pos_emb, attn_mask);
}

// ============================================================================
// ZipformerEncoderLayer
// ============================================================================

ZipformerEncoderLayer::ZipformerEncoderLayer(
    int d_model,
    int attention_dim,
    int feedforward_dim,
    int num_heads,
    int kernel_size,
    int pos_dim,
    int pos_head_dim,
    int value_head_dim,
    bool causal
)
    : d_model_(d_model)
    , self_attn_(d_model, attention_dim, num_heads, pos_dim, pos_head_dim, value_head_dim)
    , ff1_(d_model, feedforward_dim)
    , ff2_(d_model, feedforward_dim)
    , conv_(d_model, kernel_size, causal)
    , norm_(d_model)
{
}

void ZipformerEncoderLayer::load_weights(const WeightMap& weights, const std::string& prefix) {
    self_attn_.load_weights(weights, prefix + ".self_attn");
    ff1_.load_weights(weights, prefix + ".ff1");
    ff2_.load_weights(weights, prefix + ".ff2");
    conv_.load_weights(weights, prefix + ".conv");
    norm_.load_weights(weights, prefix + ".norm");
}

array ZipformerEncoderLayer::forward(
    const array& x,
    const array& pos_emb,
    const std::optional<array>& attn_mask
) const {
    // Macaron-style: FFN -> Attn -> Conv -> FFN -> Norm
    array out = x;
    out = add(out, multiply(array(0.5f), ff1_.forward(out)));
    out = add(out, self_attn_.forward(out, pos_emb, attn_mask));
    out = add(out, conv_.forward(out));
    out = add(out, multiply(array(0.5f), ff2_.forward(out)));
    return norm_.forward(out);
}

array ZipformerEncoderLayer::streaming_forward(
    const array& x,
    const array& pos_emb,
    array& attn_cache,
    array& conv_cache,
    const std::optional<array>& attn_mask
) const {
    // Streaming version with caches
    array out = x;
    out = add(out, multiply(array(0.5f), ff1_.forward(out)));
    out = add(out, self_attn_.streaming_forward(out, pos_emb, attn_cache, attn_mask));
    out = add(out, conv_.streaming_forward(out, conv_cache));
    out = add(out, multiply(array(0.5f), ff2_.forward(out)));
    return norm_.forward(out);
}

// ============================================================================
// Conv2dSubsampling
// ============================================================================

Conv2dSubsampling::Conv2dSubsampling(int in_channels, int out_channels)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
{
    // Initialize weights with correct shapes for the case when no checkpoint is loaded
    // Architecture: 3 conv layers (1->8->32->128) + ConvNeXt + Linear + BiasNorm
    // Input: (B, T, in_channels=80)
    // After conv layers: freq_out = ((in_channels - 1) / 2 - 1) / 2 = 19 for in_channels=80
    // Linear input: freq_out * 128 = 2432

    // Conv layer 0: 3x3, 1->8, MLX shape: (out_ch, H, W, in_ch)
    conv0_weight_ = zeros({8, 3, 3, 1});
    conv0_bias_ = zeros({8});

    // Conv layer 4: 3x3, 8->32
    conv4_weight_ = zeros({32, 3, 3, 8});
    conv4_bias_ = zeros({32});

    // Conv layer 7: 3x3, 32->128
    conv7_weight_ = zeros({128, 3, 3, 32});
    conv7_bias_ = zeros({128});

    // ConvNeXt: depthwise 7x7, groups=128
    convnext_dw_weight_ = zeros({128, 7, 7, 1});  // depthwise: (ch, H, W, 1)
    convnext_dw_bias_ = zeros({128});

    // ConvNeXt pointwise1: 1x1, 128->384
    convnext_pw1_weight_ = zeros({384, 1, 1, 128});
    convnext_pw1_bias_ = zeros({384});

    // ConvNeXt pointwise2: 1x1, 384->128
    convnext_pw2_weight_ = zeros({128, 1, 1, 384});
    convnext_pw2_bias_ = zeros({128});

    // Output linear: (out_channels, 2432) for 80-bin input
    int freq_after_conv = (((in_channels - 1) / 2) - 1) / 2;  // ~19 for in_channels=80
    int linear_input_dim = freq_after_conv * 128;
    out_weight_ = zeros({out_channels, linear_input_dim});
    out_bias_ = zeros({out_channels});

    // BiasNorm
    out_norm_log_scale_ = zeros({});  // scalar
    out_norm_bias_ = zeros({out_channels});
}

void Conv2dSubsampling::load_weights(const WeightMap& weights, const std::string& prefix) {
    // Conv layer 0: encoder_embed.conv.0
    if (has_weight(weights, prefix + ".conv.0.weight")) {
        // PyTorch: (out_ch, in_ch, H, W) -> MLX needs (out_ch, H, W, in_ch)
        array w = get_weight(weights, prefix + ".conv.0.weight");
        conv0_weight_ = transpose(w, {0, 2, 3, 1});
    }
    if (has_weight(weights, prefix + ".conv.0.bias")) {
        conv0_bias_ = get_weight(weights, prefix + ".conv.0.bias");
    }

    // Conv layer 4: encoder_embed.conv.4
    if (has_weight(weights, prefix + ".conv.4.weight")) {
        array w = get_weight(weights, prefix + ".conv.4.weight");
        conv4_weight_ = transpose(w, {0, 2, 3, 1});
    }
    if (has_weight(weights, prefix + ".conv.4.bias")) {
        conv4_bias_ = get_weight(weights, prefix + ".conv.4.bias");
    }

    // Conv layer 7: encoder_embed.conv.7
    if (has_weight(weights, prefix + ".conv.7.weight")) {
        array w = get_weight(weights, prefix + ".conv.7.weight");
        conv7_weight_ = transpose(w, {0, 2, 3, 1});
    }
    if (has_weight(weights, prefix + ".conv.7.bias")) {
        conv7_bias_ = get_weight(weights, prefix + ".conv.7.bias");
    }

    // ConvNeXt depthwise: encoder_embed.convnext.depthwise_conv
    if (has_weight(weights, prefix + ".convnext.depthwise_conv.weight")) {
        array w = get_weight(weights, prefix + ".convnext.depthwise_conv.weight");
        convnext_dw_weight_ = transpose(w, {0, 2, 3, 1});
    }
    if (has_weight(weights, prefix + ".convnext.depthwise_conv.bias")) {
        convnext_dw_bias_ = get_weight(weights, prefix + ".convnext.depthwise_conv.bias");
    }

    // ConvNeXt pointwise1: encoder_embed.convnext.pointwise_conv1
    if (has_weight(weights, prefix + ".convnext.pointwise_conv1.weight")) {
        array w = get_weight(weights, prefix + ".convnext.pointwise_conv1.weight");
        convnext_pw1_weight_ = transpose(w, {0, 2, 3, 1});
    }
    if (has_weight(weights, prefix + ".convnext.pointwise_conv1.bias")) {
        convnext_pw1_bias_ = get_weight(weights, prefix + ".convnext.pointwise_conv1.bias");
    }

    // ConvNeXt pointwise2: encoder_embed.convnext.pointwise_conv2
    if (has_weight(weights, prefix + ".convnext.pointwise_conv2.weight")) {
        array w = get_weight(weights, prefix + ".convnext.pointwise_conv2.weight");
        convnext_pw2_weight_ = transpose(w, {0, 2, 3, 1});
    }
    if (has_weight(weights, prefix + ".convnext.pointwise_conv2.bias")) {
        convnext_pw2_bias_ = get_weight(weights, prefix + ".convnext.pointwise_conv2.bias");
    }

    // Output linear: encoder_embed.out
    if (has_weight(weights, prefix + ".out.weight")) {
        out_weight_ = get_weight(weights, prefix + ".out.weight");
    }
    if (has_weight(weights, prefix + ".out.bias")) {
        out_bias_ = get_weight(weights, prefix + ".out.bias");
    }

    // BiasNorm: encoder_embed.out_norm
    if (has_weight(weights, prefix + ".out_norm.log_scale")) {
        out_norm_log_scale_ = get_weight(weights, prefix + ".out_norm.log_scale");
    }
    if (has_weight(weights, prefix + ".out_norm.bias")) {
        out_norm_bias_ = get_weight(weights, prefix + ".out_norm.bias");
    }
}

array Conv2dSubsampling::forward(const array& x) const {
    // x: (batch, time, features) = (B, T, F)
    // Output: (batch, (T-7)/2, out_channels)

    int batch = x.shape()[0];

    // Add channel dimension: (B, T, F) -> (B, T, F, 1)
    // MLX conv2d expects (N, H, W, C) where H=time, W=freq
    array img = expand_dims(x, 3);

    // Conv layer 0: 3x3, pad=(0, 1) on freq axis, stride=1
    // MLX conv2d: (N, H, W, C_in) * (C_out, kH, kW, C_in) -> (N, H', W', C_out)
    array out = conv2d(img, conv0_weight_, /*stride=*/{1, 1}, /*padding=*/{0, 1});
    out = add(out, reshape(conv0_bias_, {1, 1, 1, 8}));
    out = swoosh_r(out);

    // Conv layer 4: 3x3, stride=2, no padding
    out = conv2d(out, conv4_weight_, /*stride=*/{2, 2}, /*padding=*/{0, 0});
    out = add(out, reshape(conv4_bias_, {1, 1, 1, 32}));
    out = swoosh_r(out);

    // Conv layer 7: 3x3, stride=(1, 2), no padding
    out = conv2d(out, conv7_weight_, /*stride=*/{1, 2}, /*padding=*/{0, 0});
    out = add(out, reshape(conv7_bias_, {1, 1, 1, 128}));
    out = swoosh_r(out);

    // ConvNeXt: depthwise 7x7 -> pointwise1 -> SwooshL -> pointwise2 -> residual
    array bypass = out;

    // Depthwise conv: groups=128, pad=3
    out = conv2d(out, convnext_dw_weight_, /*stride=*/{1, 1}, /*padding=*/{3, 3}, /*dilation=*/{1, 1}, /*groups=*/128);
    out = add(out, reshape(convnext_dw_bias_, {1, 1, 1, 128}));

    // Pointwise conv1: 1x1, 128->384
    out = conv2d(out, convnext_pw1_weight_, /*stride=*/{1, 1}, /*padding=*/{0, 0});
    out = add(out, reshape(convnext_pw1_bias_, {1, 1, 1, 384}));
    out = swoosh_l(out);

    // Pointwise conv2: 1x1, 384->128
    out = conv2d(out, convnext_pw2_weight_, /*stride=*/{1, 1}, /*padding=*/{0, 0});
    out = add(out, reshape(convnext_pw2_bias_, {1, 1, 1, 128}));

    // Residual connection
    out = add(bypass, out);

    // MLX format: (B, T', F', C) = (B, 328, 19, 128)
    // Need to match PyTorch reshape order: (B, C, T', F') -> transpose -> (B, T', C, F') -> reshape -> (B, T', C*F')
    // So we transpose (B, T', F', C) -> (B, T', C, F') then reshape to (B, T', C*F')
    int out_time = out.shape()[1];
    int out_freq = out.shape()[2];
    int out_ch = out.shape()[3];

    // Transpose to (B, T', C, F')
    out = transpose(out, {0, 1, 3, 2});  // (B, T', F', C) -> (B, T', C, F')
    out = reshape(out, {batch, out_time, out_ch * out_freq});  // C*F = 128*19 = 2432

    // Linear projection: (B, T', in_features) @ (out_features, in_features).T
    out = matmul(out, transpose(out_weight_));
    out = add(out, out_bias_);

    // BiasNorm (icefall):
    //   scales = (mean((x - bias)^2, dim=-1, keepdim=True) ^ -0.5) * exp(log_scale)
    //   ans = x * scales
    // Note: bias is subtracted to compute scale, but output is x * scale (not (x-bias) * scale)
    array x_minus_bias = subtract(out, out_norm_bias_);
    array variance = mean(multiply(x_minus_bias, x_minus_bias), /*axis=*/-1, /*keepdims=*/true);
    array inv_rms = rsqrt(add(variance, array(1e-8f)));  // Add epsilon for numerical stability
    array scales = multiply(inv_rms, exp(out_norm_log_scale_));
    out = multiply(out, scales);

    return out;
}

array Conv2dSubsampling::streaming_forward(const array& x, array& cache) const {
    (void)cache;
    return forward(x);
}

// ============================================================================
// SimpleDownsample / SimpleUpsample
// ============================================================================

SimpleDownsample::SimpleDownsample(int in_dim, int out_dim, int factor)
    : in_dim_(in_dim)
    , out_dim_(out_dim)
    , factor_(factor)
    , linear_(in_dim * factor, out_dim, true)
{
}

void SimpleDownsample::load_weights(const WeightMap& weights, const std::string& prefix) {
    linear_.load_weights(weights, prefix + ".linear");
}

array SimpleDownsample::forward(const array& x) const {
    // x: (batch, time, in_dim)
    // Reshape and project: (batch, time/factor, in_dim*factor) -> (batch, time/factor, out_dim)
    int batch = x.shape()[0];
    int time = x.shape()[1];

    // Pad time dimension to be divisible by factor
    int padded_time = ((time + factor_ - 1) / factor_) * factor_;
    array src = x;
    if (padded_time > time) {
        // Pad with last frame repeated
        array last_frame = slice(x, {0, time - 1, 0}, {batch, time, in_dim_});
        int pad_amount = padded_time - time;
        array padding = broadcast_to(last_frame, {batch, pad_amount, in_dim_});
        src = concatenate({x, padding}, 1);
    }

    int new_time = padded_time / factor_;
    array reshaped = reshape(src, {batch, new_time, in_dim_ * factor_});
    return linear_.forward(reshaped);
}

SimpleUpsample::SimpleUpsample(int in_dim, int out_dim, int factor)
    : in_dim_(in_dim)
    , out_dim_(out_dim)
    , factor_(factor)
    , linear_(in_dim, out_dim * factor, true)
    , norm_(out_dim)
{
}

void SimpleUpsample::load_weights(const WeightMap& weights, const std::string& prefix) {
    linear_.load_weights(weights, prefix + ".linear");
    norm_.load_weights(weights, prefix + ".norm");
}

array SimpleUpsample::forward(const array& x) const {
    // x: (batch, time, in_dim)
    // Project and reshape: (batch, time, out_dim*factor) -> (batch, time*factor, out_dim)
    array proj = linear_.forward(x);
    int batch = proj.shape()[0];
    int time = proj.shape()[1];

    array reshaped = reshape(proj, {batch, time * factor_, out_dim_});
    return norm_.forward(reshaped);
}

// ============================================================================
// BypassModule
// ============================================================================

BypassModule::BypassModule(int d_model)
    : d_model_(d_model)
    , bypass_scale_(multiply(ones({d_model}), array(0.5f)))
{
}

void BypassModule::load_weights(const WeightMap& weights, const std::string& prefix) {
    if (has_weight(weights, prefix + ".bypass_scale")) {
        bypass_scale_ = get_weight(weights, prefix + ".bypass_scale");
    }
}

array BypassModule::forward(const array& src_orig, const array& src) const {
    // scale is per-channel, shape (d_model,)
    array scale = clip(bypass_scale_, array(0.0f), array(1.0f));
    // Output = src_orig + scale * (src - src_orig)
    return add(src_orig, multiply(scale, subtract(src, src_orig)));
}

// ============================================================================
// CompactRelPositionalEncoding
// ============================================================================

CompactRelPositionalEncoding::CompactRelPositionalEncoding(int pos_dim)
    : pos_dim_(pos_dim)
{
}

void CompactRelPositionalEncoding::load_weights(const WeightMap& weights, const std::string& prefix) {
    // No learnable parameters - sinusoidal encoding
    (void)weights;
    (void)prefix;
}

array CompactRelPositionalEncoding::forward(const array& x) const {
    // x: (seq_len, batch_size, d_model)
    // Matches icefall CompactRelPositionalEncoding with atan compression
    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];

    // Generate relative position indices: [-seq_len+1, ..., seq_len-1]
    int total_len = 2 * seq_len - 1;
    array positions = arange(-(seq_len - 1.0f), static_cast<float>(seq_len), 1.0f);
    positions = reshape(positions, {total_len, 1});

    // Frequency indices: 1, 2, 3, ..., pos_dim/2
    array freqs = add(arange(0.0f, static_cast<float>(pos_dim_ / 2), 1.0f), array(1.0f));

    // Compression length (heuristic from icefall)
    float compression_length = std::sqrt(static_cast<float>(pos_dim_));
    float length_factor = 1.0f;
    float length_scale = length_factor * pos_dim_ / (2.0f * static_cast<float>(M_PI));

    // Compute compressed positions:
    // x_compressed = compression_length * sign(x) * (log(abs(x) + compression_length) - log(compression_length))
    array pos_sign = sign(positions);
    array pos_abs = abs(positions);
    array x_compressed = multiply(
        array(compression_length),
        multiply(
            pos_sign,
            subtract(
                log(add(pos_abs, array(compression_length))),
                array(std::log(compression_length))
            )
        )
    );

    // Apply atan compression: x_atan = atan(x_compressed / length_scale)
    array x_atan = arctan(divide(x_compressed, array(length_scale)));

    // Compute cosines and sines: (total_len, pos_dim/2)
    array angles = matmul(x_atan, reshape(freqs, {1, pos_dim_ / 2}));
    array cosines = cos(angles);
    array sines = sin(angles);

    // Interleave: pe[:, 0::2] = cosines, pe[:, 1::2] = sines
    // For simplicity, stack and reshape
    array pe = zeros({total_len, pos_dim_});

    // Interleave cosines and sines
    for (int i = 0; i < pos_dim_ / 2; ++i) {
        // pe[:, 2*i] = cosines[:, i]
        array cos_col = slice(cosines, {0, i}, {total_len, i + 1});
        array sin_col = slice(sines, {0, i}, {total_len, i + 1});

        // Build pe column by column using scatter or just concatenate
        // For now, build the full pe tensor differently
    }

    // Actually, let's build pe more efficiently:
    // Stack interleaved: [cos0, sin0, cos1, sin1, ...]
    std::vector<array> cols;
    for (int i = 0; i < pos_dim_ / 2; ++i) {
        cols.push_back(slice(cosines, {0, i}, {total_len, i + 1}));
        cols.push_back(slice(sines, {0, i}, {total_len, i + 1}));
    }
    pe = concatenate(cols, -1);  // (total_len, pos_dim)

    // SET last column to exactly 1.0 (not ADD)
    // This matches Python RelPositionalEncoding: pe[:, -1] = 1.0
    // The Python code does: pe = pe.at[:, -1].add(1.0 - pe[:, -1])
    // which effectively sets the last column to 1.0
    array ones_col = ones({total_len, 1});
    array pe_without_last = slice(pe, {0, 0}, {total_len, pos_dim_ - 1});
    pe = concatenate({pe_without_last, ones_col}, -1);

    // Broadcast to (batch, total_len, pos_dim)
    pe = expand_dims(pe, 0);  // (1, total_len, pos_dim)
    pe = broadcast_to(pe, {batch_size, total_len, pos_dim_});

    return pe;
}

array CompactRelPositionalEncoding::forward(const array& x, int left_context_len) const {
    // Streaming version: generate pos encoding for seq_len + left_context_len
    // Uses same atan-compression as non-streaming version
    int seq_len = x.shape()[0];
    int batch_size = x.shape()[1];
    int kv_len = left_context_len + seq_len;
    int total_len = seq_len + kv_len - 1;

    array positions = arange(-(kv_len - 1.0f), static_cast<float>(seq_len), 1.0f);
    positions = reshape(positions, {total_len, 1});

    array freqs = add(arange(0.0f, static_cast<float>(pos_dim_ / 2), 1.0f), array(1.0f));

    float compression_length = std::sqrt(static_cast<float>(pos_dim_));
    float length_factor = 1.0f;
    float length_scale = length_factor * pos_dim_ / (2.0f * static_cast<float>(M_PI));

    array pos_sign = sign(positions);
    array pos_abs = abs(positions);
    array x_compressed = multiply(
        array(compression_length),
        multiply(
            pos_sign,
            subtract(
                log(add(pos_abs, array(compression_length))),
                array(std::log(compression_length))
            )
        )
    );

    array x_atan = arctan(divide(x_compressed, array(length_scale)));
    array angles = matmul(x_atan, reshape(freqs, {1, pos_dim_ / 2}));
    array cosines = cos(angles);
    array sines = sin(angles);

    std::vector<array> cols;
    for (int i = 0; i < pos_dim_ / 2; ++i) {
        cols.push_back(slice(cosines, {0, i}, {total_len, i + 1}));
        cols.push_back(slice(sines, {0, i}, {total_len, i + 1}));
    }
    array pe = concatenate(cols, -1);

    // SET last column to exactly 1.0 (not ADD)
    // This matches Python RelPositionalEncoding: pe[:, -1] = 1.0
    array ones_col = ones({total_len, 1});
    array pe_without_last = slice(pe, {0, 0}, {total_len, pos_dim_ - 1});
    pe = concatenate({pe_without_last, ones_col}, -1);

    pe = expand_dims(pe, 0);
    pe = broadcast_to(pe, {batch_size, total_len, pos_dim_});

    return pe;
}

// ============================================================================
// Zipformer2EncoderStage
// ============================================================================

Zipformer2EncoderStage::Zipformer2EncoderStage(
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
)
    : d_model_(d_model)
    , downsample_factor_(downsample_factor)
    , is_downsampled_(downsample_factor > 1)
    , pos_enc_(pos_dim)
    , downsample_bias_(zeros({downsample_factor}))
{
    // Create encoder layers
    layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        layers_.push_back(std::make_unique<Zipformer2EncoderLayer>(
            d_model,
            attention_dim,
            num_heads,
            ff1_dim,
            ff2_dim,
            ff3_dim,
            kernel_size,
            pos_head_dim,
            pos_dim,
            value_head_dim,
            causal
        ));
    }

    // Create out_combiner for stages 1-5
    if (is_downsampled_) {
        out_combiner_ = std::make_unique<BypassModule>(d_model);
    }
}

void Zipformer2EncoderStage::load_weights(
    const WeightMap& weights,
    const std::string& prefix,
    int stage_idx
) {
    // Stage 0: encoders.0.layers.{layer_idx}
    // Stages 1-5: encoders.{stage}.encoder.layers.{layer_idx}
    std::string layer_base;
    if (stage_idx == 0) {
        layer_base = prefix + ".layers.";
    } else {
        layer_base = prefix + ".encoder.layers.";
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i]->load_weights(weights, layer_base + std::to_string(i));
    }

    // Load downsample bias and out_combiner for stages 1-5
    if (is_downsampled_) {
        // Downsample is just a softmax bias, not a linear projection
        if (has_weight(weights, prefix + ".downsample.bias")) {
            downsample_bias_ = get_weight(weights, prefix + ".downsample.bias");
        }
        out_combiner_->load_weights(weights, prefix + ".out_combiner");
    }
}

array Zipformer2EncoderStage::downsample(const array& x) const {
    // x: (seq, batch, d_model) - downsample by factor using softmax-weighted averaging
    int seq_len = x.shape()[0];
    int batch = x.shape()[1];
    int d_model = x.shape()[2];

    // Pad sequence length to be divisible by factor
    int ds = downsample_factor_;
    int d_seq_len = (seq_len + ds - 1) / ds;
    int pad = d_seq_len * ds - seq_len;

    array src = x;
    if (pad > 0) {
        // Pad with last frame
        array last_frame = slice(x, {seq_len - 1, 0, 0}, {seq_len, batch, d_model});
        array padding = broadcast_to(last_frame, {pad, batch, d_model});
        src = concatenate({x, padding}, 0);
    }

    // Reshape for weighted average: (seq, batch, d) -> (d_seq, ds, batch, d)
    src = reshape(src, {d_seq_len, ds, batch, d_model});

    // Apply softmax weights
    array weights_softmax = softmax(downsample_bias_, 0);
    weights_softmax = reshape(weights_softmax, {1, ds, 1, 1});
    src = sum(multiply(src, weights_softmax), 1);  // (d_seq, batch, d)

    return src;
}

array Zipformer2EncoderStage::upsample(const array& x, int target_len) const {
    // x: (seq, batch, d_model) - upsample by repeating frames
    int seq_len = x.shape()[0];
    int batch = x.shape()[1];
    int d_model = x.shape()[2];

    // Expand each frame by factor
    // (seq, batch, d) -> (seq, 1, batch, d) -> (seq, factor, batch, d)
    array expanded = expand_dims(x, 1);  // (seq, 1, batch, d)
    expanded = broadcast_to(expanded, {seq_len, downsample_factor_, batch, d_model});

    // Reshape to (seq * factor, batch, d)
    array upsampled = reshape(expanded, {seq_len * downsample_factor_, batch, d_model});

    // Trim or pad to exact target length
    int current_len = upsampled.shape()[0];
    if (current_len > target_len) {
        upsampled = slice(upsampled, {0, 0, 0}, {target_len, batch, d_model});
    } else if (current_len < target_len) {
        array last_frame = slice(upsampled, {current_len - 1, 0, 0}, {current_len, batch, d_model});
        int extra = target_len - current_len;
        array padding = broadcast_to(last_frame, {extra, batch, d_model});
        upsampled = concatenate({upsampled, padding}, 0);
    }

    return upsampled;
}

array Zipformer2EncoderStage::forward(const array& x) const {
    // x: (batch, time, d_model)
    // We use (seq, batch, d_model) convention internally

    // Transpose from (batch, time, d_model) to (seq, batch, d_model)
    array src = transpose(x, {1, 0, 2});
    array src_orig = src;
    int orig_len = src.shape()[0];

    // Downsample for stages 1-5
    if (is_downsampled_) {
        src = downsample(src);  // (seq/ds, batch, d_model)
    }

    // Generate positional embeddings
    array pos_emb = pos_enc_.forward(src);

    // Process through layers
    for (const auto& layer : layers_) {
        src = layer->forward(src, pos_emb);
    }

    // Upsample for stages 1-5
    if (is_downsampled_) {
        src = upsample(src, orig_len);  // (seq, batch, d_model)

        // Apply bypass combiner
        src = out_combiner_->forward(src_orig, src);
    }

    // Transpose back to (batch, time, d_model)
    return transpose(src, {1, 0, 2});
}

std::tuple<array, std::vector<array>> Zipformer2EncoderStage::streaming_forward(
    const array& x,
    const std::vector<array>& cached_states,
    int left_context_len
) const {
    // Streaming forward - TODO: implement properly
    // For now, use non-streaming forward
    (void)cached_states;
    (void)left_context_len;
    array out = forward(x);
    return std::make_tuple(out, std::vector<array>{});
}

// ============================================================================
// ZipformerEncoder
// ============================================================================

ZipformerEncoder::ZipformerEncoder(const ZipformerConfig& config)
    : config_(config)
    , embed_(config.num_features, config.encoder_embed_dim)
    , downsample_output_bias_(zeros({2}))
{
    // Initialize encoder stages
    int num_stages = static_cast<int>(config.num_encoder_layers.size());

    for (int s = 0; s < num_stages; ++s) {
        int num_layers = config.num_encoder_layers[s];
        int d_model = config.encoder_dims[s];
        int attn_dim = config.attention_dims[s];
        int num_heads = config.num_heads[s];
        int kernel = config.cnn_module_kernels[s];
        int ds_factor = config.downsampling_factors[s];

        // Get per-stage FF dimensions
        int ff1 = config.ff1_dims[s];
        int ff2 = config.ff2_dims[s];
        int ff3 = config.ff3_dims[s];

        stages_.push_back(std::make_unique<Zipformer2EncoderStage>(
            d_model,
            attn_dim,
            num_heads,
            ff1,
            ff2,
            ff3,
            num_layers,
            kernel,
            config.pos_head_dim,
            config.pos_dim,
            config.value_head_dim,
            ds_factor,
            config.causal
        ));
    }
}

void ZipformerEncoder::load_weights(const WeightMap& weights, const std::string& prefix) {
    (void)prefix;  // Weight keys are at root level

    // Load encoder_embed weights
    embed_.load_weights(weights, "encoder_embed");

    // Load stage weights
    for (size_t s = 0; s < stages_.size(); ++s) {
        std::string stage_prefix = "encoders." + std::to_string(s);
        stages_[s]->load_weights(weights, stage_prefix, static_cast<int>(s));
    }

    // Load downsample_output bias
    if (has_weight(weights, "downsample_output.bias")) {
        downsample_output_bias_ = get_weight(weights, "downsample_output.bias");
    }
}

array ZipformerEncoder::get_full_dim_output(const std::vector<array>& outputs) const {
    // Combine multi-scale outputs using icefall's _get_full_dim_output pattern
    // Each output is (batch, time, d_model_s)
    // Goal: combine all channels to output_dim = max(encoder_dims)

    // Debug: print output shapes
    #ifdef DEBUG_ENCODER
    std::cerr << "get_full_dim_output: " << outputs.size() << " stages\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cerr << "  outputs[" << i << "]: " << outputs[i].shape()[0] << "x"
                  << outputs[i].shape()[1] << "x" << outputs[i].shape()[2] << "\n";
    }
    #endif

    // Start with the last stage output (has smallest dimension)
    std::vector<array> pieces;
    pieces.push_back(outputs.back());
    int cur_dim = config_.encoder_dims.back();

    // Work backwards through stages, extracting additional channels
    for (int i = static_cast<int>(config_.encoder_dims.size()) - 2; i >= 0; --i) {
        int d = config_.encoder_dims[i];
        if (d > cur_dim) {
            // Extract channels [cur_dim:d] from this stage's output
            array piece = slice(outputs[i], {0, 0, cur_dim},
                               {outputs[i].shape()[0], outputs[i].shape()[1], d});
            pieces.push_back(piece);
            #ifdef DEBUG_ENCODER
            std::cerr << "  Added piece from stage " << i << ": [" << cur_dim
                      << ":" << d << "] -> " << piece.shape()[2] << " channels\n";
            #endif
            cur_dim = d;
        }
    }

    #ifdef DEBUG_ENCODER
    std::cerr << "  Total pieces: " << pieces.size() << ", target dim: " << cur_dim << "\n";
    #endif

    // Concatenate all pieces along last axis
    if (pieces.size() == 1) {
        return pieces[0];
    }
    return concatenate(pieces, 2);  // Use explicit axis=2 for clarity
}

array ZipformerEncoder::forward(const array& x) const {
    // x: (batch, time, features)

    #ifdef DEBUG_ENCODER
    std::cerr << "ZipformerEncoder::forward input: " << x.shape()[0] << "x"
              << x.shape()[1] << "x" << x.shape()[2] << "\n";
    std::cerr << "Number of stages: " << stages_.size() << "\n";
    #endif

    // Step 1: Subsampling via Conv2dSubsampling
    array out = embed_.forward(x);  // (batch, T', encoder_embed_dim=192)

    #ifdef DEBUG_ENCODER
    std::cerr << "After embed: " << out.shape()[0] << "x"
              << out.shape()[1] << "x" << out.shape()[2] << "\n";
    #endif

    // Step 2: Process through all encoder stages
    std::vector<array> stage_outputs;
    stage_outputs.reserve(stages_.size());

    for (size_t s = 0; s < stages_.size(); ++s) {
        // Convert number of channels if needed
        int target_dim = config_.encoder_dims[s];
        int current_dim = out.shape()[2];

        #ifdef DEBUG_ENCODER
        std::cerr << "Stage " << s << ": target_dim=" << target_dim
                  << ", current_dim=" << current_dim << "\n";
        #endif

        if (current_dim != target_dim) {
            // Zero-pad or truncate channels
            if (current_dim < target_dim) {
                // Pad with zeros
                int batch = out.shape()[0];
                int time = out.shape()[1];
                array padding = zeros({batch, time, target_dim - current_dim});
                out = concatenate({out, padding}, -1);
                #ifdef DEBUG_ENCODER
                std::cerr << "  Padded to: " << out.shape()[0] << "x"
                          << out.shape()[1] << "x" << out.shape()[2] << "\n";
                #endif
            } else {
                // Truncate
                out = slice(out, {0, 0, 0}, {out.shape()[0], out.shape()[1], target_dim});
                #ifdef DEBUG_ENCODER
                std::cerr << "  Truncated to: " << out.shape()[0] << "x"
                          << out.shape()[1] << "x" << out.shape()[2] << "\n";
                #endif
            }
        }

        #ifdef DEBUG_ENCODER
        std::cerr << "  Before stage " << s << ": " << out.shape()[0] << "x"
                  << out.shape()[1] << "x" << out.shape()[2] << "\n";
        eval(out);  // Force evaluation to catch errors early
        std::cerr << "  Eval before stage OK\n";
        #endif

        // Process through stage
        out = stages_[s]->forward(out);

        #ifdef DEBUG_ENCODER
        eval(out);  // Force evaluation to catch errors early
        std::cerr << "  After stage " << s << ": " << out.shape()[0] << "x"
                  << out.shape()[1] << "x" << out.shape()[2] << "\n";
        #endif

        stage_outputs.push_back(out);
    }

    // Step 3: Combine multi-scale outputs
    array combined = get_full_dim_output(stage_outputs);

    // Step 4: Final downsampling by factor of 2
    int seq_len = combined.shape()[1];
    int batch_size = combined.shape()[0];
    int d_model = combined.shape()[2];

    // Transpose to (seq, batch, d_model) for processing
    combined = transpose(combined, {1, 0, 2});

    // Pad sequence length to be divisible by 2
    int ds = 2;
    int d_seq_len = (seq_len + ds - 1) / ds;
    int pad = d_seq_len * ds - seq_len;
    if (pad > 0) {
        // Pad with last frame
        array last_frame = slice(combined, {seq_len - 1, 0, 0}, {seq_len, batch_size, d_model});
        array padding = broadcast_to(last_frame, {pad, batch_size, d_model});
        combined = concatenate({combined, padding}, 0);
    }

    // Reshape for weighted average: (seq, batch, d) -> (d_seq, ds, batch, d)
    combined = reshape(combined, {d_seq_len, ds, batch_size, d_model});

    // Apply softmax weights for downsampling
    array weights_softmax = softmax(downsample_output_bias_, 0);
    weights_softmax = reshape(weights_softmax, {1, ds, 1, 1});
    combined = sum(multiply(combined, weights_softmax), 1);  // (d_seq, batch, d)

    // Transpose back to (batch, time, d_model)
    return transpose(combined, {1, 0, 2});
}

array ZipformerEncoder::streaming_forward(const array& x, CacheState& state) const {
    // Streaming forward - TODO: implement properly with state management
    (void)state;
    return forward(x);
}

} // namespace zipformer
