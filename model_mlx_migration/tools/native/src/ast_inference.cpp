/**
 * @file ast_inference.cpp
 * @brief Audio Spectrogram Transformer (AST) Native MLX C++ Inference Implementation
 *
 * High-performance inference using MLX C++ API.
 * Eliminates Python dispatch overhead for batch=1 inference.
 */

#include "ast_inference.h"
#include <mlx/io.h>
#include <mlx/compile.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>

namespace ast {

namespace mx = mlx::core;

ASTInference::ASTInference(const std::string& weights_path,
                           const ASTConfig& config)
    : config_(config) {
    // Load weights from safetensors file
    auto [loaded, metadata] = mx::load_safetensors(weights_path);
    weights_ = std::move(loaded);
}

void ASTInference::load_labels(const std::string& labels_path) {
    std::ifstream file(labels_path);
    if (!file.is_open()) {
        return;
    }

    // Simple JSON parsing for {"0": "label0", "1": "label1", ...}
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // Parse JSON manually (simple format)
    size_t pos = 0;
    while ((pos = content.find("\"", pos)) != std::string::npos) {
        size_t key_start = pos + 1;
        size_t key_end = content.find("\"", key_start);
        if (key_end == std::string::npos) break;

        std::string key_str = content.substr(key_start, key_end - key_start);

        // Check if it's a number
        bool is_number = true;
        for (char c : key_str) {
            if (!std::isdigit(c)) {
                is_number = false;
                break;
            }
        }

        if (is_number) {
            int idx = std::stoi(key_str);

            // Find value
            size_t colon_pos = content.find(":", key_end);
            if (colon_pos == std::string::npos) break;

            size_t val_start = content.find("\"", colon_pos) + 1;
            size_t val_end = content.find("\"", val_start);
            if (val_end == std::string::npos) break;

            std::string value = content.substr(val_start, val_end - val_start);
            labels_[idx] = value;

            pos = val_end + 1;
        } else {
            pos = key_end + 1;
        }
    }
}

std::string ASTInference::get_label(int index) const {
    auto it = labels_.find(index);
    if (it != labels_.end()) {
        return it->second;
    }
    return "unknown";
}

mx::array ASTInference::layer_norm(const mx::array& x, const std::string& prefix) {
    auto weight = weights_.at(prefix + ".weight");
    auto bias = weights_.at(prefix + ".bias");

    // Layer normalization over last dimension
    auto mean = mx::mean(x, -1, true);
    auto var = mx::var(x, -1, true);
    auto x_norm = (x - mean) / mx::sqrt(var + config_.layer_norm_eps);

    return weight * x_norm + bias;
}

mx::array ASTInference::linear(const mx::array& x, const std::string& prefix, bool bias) {
    auto weight = weights_.at(prefix + ".weight");

    // Linear: y = x @ weight.T + bias
    auto y = mx::matmul(x, mx::transpose(weight));

    if (bias) {
        auto b = weights_.at(prefix + ".bias");
        y = y + b;
    }

    return y;
}

mx::array ASTInference::patch_embed(const mx::array& x) {
    // Input: (B, 1, freq, time) or (B, freq, time)
    // Output: (B, num_patches, hidden_size)

    mx::array input = x;

    // Ensure 4D: (B, C, freq, time)
    if (input.ndim() == 3) {
        input = mx::expand_dims(input, 1);  // Add channel dim
    }

    // HuggingFace AST expects (B, 1, time, freq) after transpose
    // Our input is (B, 1, freq, time)
    // Transpose to (B, time, freq, 1) for MLX Conv2d (NHWC format)
    input = mx::transpose(input, {0, 3, 2, 1});

    // Get Conv2d weights and bias
    // MLX Conv2d weight shape: (out_channels, kH, kW, in_channels)
    auto conv_weight = weights_.at("audio_spectrogram_transformer.embeddings.patch_embeddings.weight");
    auto conv_bias = weights_.at("audio_spectrogram_transformer.embeddings.patch_embeddings.bias");

    // Apply Conv2d with stride
    // Input: (B, time, freq, 1), Output: (B, H', W', hidden_size)
    auto y = mx::conv2d(
        input,
        conv_weight,
        {config_.time_stride, config_.frequency_stride},  // stride
        {0, 0},  // padding
        {1, 1},  // dilation
        1        // groups
    );

    // Add bias
    y = y + conv_bias;

    int batch_size = y.shape(0);
    int h = y.shape(1);
    int w = y.shape(2);

    // Flatten spatial dims: (B, H' * W', hidden_size)
    mx::Shape flat_shape = {batch_size, h * w, config_.hidden_size};
    y = mx::reshape(y, flat_shape);

    return y;
}

mx::array ASTInference::add_position_embeddings(const mx::array& x) {
    int batch_size = x.shape(0);

    // Get CLS and distillation tokens
    auto cls_token = weights_.at("audio_spectrogram_transformer.embeddings.cls_token");
    auto dist_token = weights_.at("audio_spectrogram_transformer.embeddings.distillation_token");

    // Broadcast tokens to batch size
    mx::Shape cls_shape = {batch_size, 1, config_.hidden_size};
    mx::Shape dist_shape = {batch_size, 1, config_.hidden_size};
    auto cls_broadcast = mx::broadcast_to(cls_token, cls_shape);
    auto dist_broadcast = mx::broadcast_to(dist_token, dist_shape);

    // Prepend CLS and distillation tokens
    auto x_with_tokens = mx::concatenate({cls_broadcast, dist_broadcast, x}, 1);

    // Add position embeddings
    auto pos_embed = weights_.at("audio_spectrogram_transformer.embeddings.position_embeddings");
    int seq_len = x_with_tokens.shape(1);
    auto pos_embed_slice = mx::slice(pos_embed, {0, 0, 0}, {1, seq_len, config_.hidden_size});

    return x_with_tokens + pos_embed_slice;
}

mx::array ASTInference::embeddings_forward(const mx::array& x) {
    auto patches = patch_embed(x);
    return add_position_embeddings(patches);
}

mx::array ASTInference::self_attention(const mx::array& x, int layer_idx) {
    std::string prefix = "audio_spectrogram_transformer.encoder.layers." +
                         std::to_string(layer_idx) + ".attention";

    int batch_size = x.shape(0);
    int seq_len = x.shape(1);
    int num_heads = config_.num_attention_heads;
    int head_dim = config_.head_dim();

    // QKV projections
    auto q = linear(x, prefix + ".query", config_.qkv_bias);
    auto k = linear(x, prefix + ".key", config_.qkv_bias);
    auto v = linear(x, prefix + ".value", config_.qkv_bias);

    // Reshape for multi-head attention: (B, seq, heads, head_dim)
    mx::Shape qkv_shape = {batch_size, seq_len, num_heads, head_dim};
    q = mx::reshape(q, qkv_shape);
    k = mx::reshape(k, qkv_shape);
    v = mx::reshape(v, qkv_shape);

    // Transpose to (B, heads, seq, head_dim)
    q = mx::transpose(q, {0, 2, 1, 3});
    k = mx::transpose(k, {0, 2, 1, 3});
    v = mx::transpose(v, {0, 2, 1, 3});

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attn_weights = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})) * scale;
    attn_weights = mx::softmax(attn_weights, -1);

    // Apply attention to values
    auto attn_output = mx::matmul(attn_weights, v);

    // Reshape back: (B, seq, hidden_size)
    attn_output = mx::transpose(attn_output, {0, 2, 1, 3});
    mx::Shape out_shape = {batch_size, seq_len, config_.hidden_size};
    attn_output = mx::reshape(attn_output, out_shape);

    // Output projection
    std::string out_prefix = "audio_spectrogram_transformer.encoder.layers." +
                              std::to_string(layer_idx) + ".attention.output_dense";
    return linear(attn_output, out_prefix);
}

mx::array ASTInference::mlp(const mx::array& x, int layer_idx) {
    std::string prefix = "audio_spectrogram_transformer.encoder.layers." +
                         std::to_string(layer_idx) + ".mlp.dense1";
    std::string out_prefix = "audio_spectrogram_transformer.encoder.layers." +
                              std::to_string(layer_idx) + ".mlp.dense2";

    // FC1 -> GELU -> FC2
    auto h = linear(x, prefix);
    // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    constexpr float sqrt2_inv = 0.7071067811865476f;  // 1 / sqrt(2)
    h = h * 0.5f * (1.0f + mx::erf(h * sqrt2_inv));
    h = linear(h, out_prefix);

    return h;
}

mx::array ASTInference::encoder_layer(const mx::array& x, int layer_idx) {
    std::string ln1_prefix = "audio_spectrogram_transformer.encoder.layers." +
                              std::to_string(layer_idx) + ".layernorm_before";
    std::string ln2_prefix = "audio_spectrogram_transformer.encoder.layers." +
                              std::to_string(layer_idx) + ".layernorm_after";

    // Pre-norm self-attention with residual
    auto normed = layer_norm(x, ln1_prefix);
    auto attn_out = self_attention(normed, layer_idx);
    auto y = x + attn_out;

    // Pre-norm MLP with residual
    normed = layer_norm(y, ln2_prefix);
    auto mlp_out = mlp(normed, layer_idx);
    y = y + mlp_out;

    return y;
}

mx::array ASTInference::encoder(const mx::array& x) {
    mx::array h = x;
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        h = encoder_layer(h, i);
    }
    return h;
}

mx::array ASTInference::transformer_forward(const mx::array& x) {
    // Get embeddings
    auto h = embeddings_forward(x);

    // Encoder
    h = encoder(h);

    // Final LayerNorm
    h = layer_norm(h, "audio_spectrogram_transformer.layernorm");

    return h;
}

mx::array ASTInference::classifier_forward(const mx::array& pooled) {
    // Classifier: LayerNorm -> Linear
    auto h = layer_norm(pooled, "classifier.layernorm");
    return linear(h, "classifier.dense");
}

std::vector<mx::array> ASTInference::forward_impl(const std::vector<mx::array>& inputs) {
    auto hidden_states = transformer_forward(inputs[0]);

    // Pool CLS and distillation tokens (positions 0 and 1)
    auto cls_out = mx::slice(hidden_states, {0, 0, 0},
                             {(int)hidden_states.shape(0), 1, config_.hidden_size});
    auto dist_out = mx::slice(hidden_states, {0, 1, 0},
                              {(int)hidden_states.shape(0), 2, config_.hidden_size});

    cls_out = mx::squeeze(cls_out, 1);
    dist_out = mx::squeeze(dist_out, 1);

    auto pooled = (cls_out + dist_out) * 0.5f;

    // Classify
    auto logits = classifier_forward(pooled);
    auto predictions = mx::argmax(logits, -1);

    return {logits, predictions, pooled};
}

void ASTInference::compile_model() {
    if (compiled_) return;

    auto forward_fn = [this](const std::vector<mx::array>& inputs) {
        return this->forward_impl(inputs);
    };

    compiled_forward_ = mx::compile(forward_fn, false);
    compiled_ = true;
}

mx::array ASTInference::extract_features(const mx::array& mel_input) {
    if (compiled_) {
        auto results = compiled_forward_({mel_input});
        mx::eval(results[2]);
        return results[2];
    }

    auto hidden_states = transformer_forward(mel_input);

    // Pool CLS and distillation tokens
    auto cls_out = mx::slice(hidden_states, {0, 0, 0},
                             {(int)hidden_states.shape(0), 1, config_.hidden_size});
    auto dist_out = mx::slice(hidden_states, {0, 1, 0},
                              {(int)hidden_states.shape(0), 2, config_.hidden_size});

    cls_out = mx::squeeze(cls_out, 1);
    dist_out = mx::squeeze(dist_out, 1);

    auto pooled = (cls_out + dist_out) * 0.5f;
    mx::eval(pooled);

    return pooled;
}

std::pair<mx::array, mx::array> ASTInference::classify(const mx::array& mel_input) {
    if (compiled_) {
        auto results = compiled_forward_({mel_input});
        mx::eval(results[0]);
        mx::eval(results[1]);
        return {results[0], results[1]};
    }

    auto hidden_states = transformer_forward(mel_input);

    // Pool CLS and distillation tokens
    auto cls_out = mx::slice(hidden_states, {0, 0, 0},
                             {(int)hidden_states.shape(0), 1, config_.hidden_size});
    auto dist_out = mx::slice(hidden_states, {0, 1, 0},
                              {(int)hidden_states.shape(0), 2, config_.hidden_size});

    cls_out = mx::squeeze(cls_out, 1);
    dist_out = mx::squeeze(dist_out, 1);

    auto pooled = (cls_out + dist_out) * 0.5f;

    auto logits = classifier_forward(pooled);
    auto predictions = mx::argmax(logits, -1);

    mx::eval(logits);
    mx::eval(predictions);

    return {logits, predictions};
}

// Benchmark implementation
double ASTBenchmark::benchmark_latency(
    ASTInference& model,
    int batch_size,
    int num_iterations
) {
    auto config = model.config();

    // Create random input: (B, freq, time)
    auto input = mx::random::normal({batch_size, config.num_mel_bins, config.max_length});
    mx::eval(input);

    // Warm up
    for (int i = 0; i < 10; i++) {
        auto [logits, pred] = model.classify(input);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto [logits, pred] = model.classify(input);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return static_cast<double>(duration.count()) / 1000.0 / num_iterations;
}

double ASTBenchmark::compare_baseline(
    ASTInference& model,
    double baseline_ms,
    int batch_size
) {
    double latency_ms = benchmark_latency(model, batch_size, 100);
    return baseline_ms / latency_ms;
}

} // namespace ast
