/**
 * @file ecapa_inference.cpp
 * @brief ECAPA-TDNN Native MLX C++ Inference Implementation
 *
 * High-performance inference using MLX C++ API.
 * Eliminates Python dispatch overhead for batch=1 inference.
 */

#include "ecapa_inference.h"
#include <mlx/io.h>
#include <mlx/compile.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace ecapa {

namespace mx = mlx::core;

ECAPAInference::ECAPAInference(const std::string& weights_path,
                               const ECAPAConfig& config)
    : config_(config) {
    // Load weights from safetensors file
    auto [loaded, metadata] = mx::load_safetensors(weights_path);
    weights_ = std::move(loaded);
}

void ECAPAInference::load_labels(const std::string& labels_path) {
    std::ifstream file(labels_path);
    if (!file.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Format: 'ab: Abkhazian' => 0
        size_t arrow_pos = line.find("=>");
        if (arrow_pos != std::string::npos) {
            std::string label_part = line.substr(0, arrow_pos);
            std::string idx_part = line.substr(arrow_pos + 2);

            // Extract language code from 'ab: Abkhazian'
            size_t colon_pos = label_part.find(':');
            if (colon_pos != std::string::npos) {
                size_t start = label_part.find('\'') + 1;
                std::string lang_code = label_part.substr(start, colon_pos - start);
                int idx = std::stoi(idx_part);
                labels_[idx] = lang_code;
            }
        }
    }
}

std::string ECAPAInference::get_language_code(int index) const {
    auto it = labels_.find(index);
    if (it != labels_.end()) {
        return it->second;
    }
    return "unknown";
}

mx::array ECAPAInference::conv1d(const mx::array& x, const std::string& prefix) {
    // Get weights
    auto weight_key = prefix + ".weight";
    auto bias_key = prefix + ".bias";

    auto weight = weights_.at(weight_key);
    auto bias = weights_.at(bias_key);

    // Weight shape: (out_channels, kernel_size, in_channels)
    int kernel_size = weight.shape(1);
    int padding = (kernel_size - 1) / 2;

    // Apply padding if needed
    mx::array padded = x;
    if (padding > 0) {
        padded = mx::pad(x, {{0, 0}, {padding, padding}, {0, 0}});
    }

    // MLX conv1d: (B, T, C) -> (B, T_out, C_out)
    auto y = mx::conv1d(padded, weight, 1, 0, 1, 1);  // stride=1, padding=0, dilation=1, groups=1

    // Add bias
    y = y + bias;

    return y;
}

mx::array ECAPAInference::batch_norm(const mx::array& x, const std::string& prefix) {
    // Get parameters
    auto weight = weights_.at(prefix + ".weight");
    auto bias = weights_.at(prefix + ".bias");
    auto running_mean = weights_.at(prefix + ".running_mean");
    auto running_var = weights_.at(prefix + ".running_var");

    constexpr float eps = 1e-5f;

    // Reshape for broadcasting: (C,) -> (1, 1, C)
    auto mean_bc = mx::reshape(running_mean, {1, 1, -1});
    auto var_bc = mx::reshape(running_var, {1, 1, -1});
    auto weight_bc = mx::reshape(weight, {1, 1, -1});
    auto bias_bc = mx::reshape(bias, {1, 1, -1});

    // Normalize
    auto x_norm = (x - mean_bc) / mx::sqrt(var_bc + eps);

    return weight_bc * x_norm + bias_bc;
}

mx::array ECAPAInference::tdnn_block(const mx::array& x, const std::string& prefix) {
    auto y = conv1d(x, prefix + ".conv");
    y = batch_norm(y, prefix + ".norm");
    y = mx::maximum(y, mx::array(0.0f));  // ReLU
    return y;
}

mx::array ECAPAInference::se_block(const mx::array& x, const std::string& prefix) {
    // Global average pooling over time: (B, T, C) -> (B, 1, C)
    auto s = mx::mean(x, 1, true);

    // Squeeze: Conv1d with kernel=1
    s = conv1d(s, prefix + ".conv1");
    s = mx::maximum(s, mx::array(0.0f));  // ReLU

    // Excitation: Conv1d with kernel=1
    s = conv1d(s, prefix + ".conv2");
    s = mx::sigmoid(s);

    // Scale
    return x * s;
}

mx::array ECAPAInference::res2net_block(const mx::array& x, const std::string& prefix, int dilation) {
    int scale = config_.res2net_scale;
    int channels_per_scale = x.shape(2) / scale;

    // Split into scale groups along channel dimension
    std::vector<mx::array> chunks;
    for (int i = 0; i < scale; i++) {
        auto chunk = mx::slice(x, {0, 0, i * channels_per_scale},
                               {(int)x.shape(0), (int)x.shape(1), (i + 1) * channels_per_scale});
        chunks.push_back(chunk);
    }

    // Process hierarchically
    std::vector<mx::array> outputs;
    outputs.push_back(chunks[0]);  // First chunk: identity
    mx::array previous = chunks[0];

    for (int i = 1; i < scale; i++) {
        // Add previous output and process through TDNN block
        auto chunk = chunks[i] + previous;
        auto block_prefix = prefix + ".blocks." + std::to_string(i - 1);
        auto out = tdnn_block(chunk, block_prefix);
        outputs.push_back(out);
        previous = out;
    }

    // Concatenate along channel dimension
    return mx::concatenate(outputs, 2);
}

mx::array ECAPAInference::se_res2net_block(const mx::array& x, const std::string& prefix, int dilation) {
    auto residual = x;

    // TDNN1: pointwise conv
    auto y = tdnn_block(x, prefix + ".tdnn1");

    // Res2Net block
    y = res2net_block(y, prefix + ".res2net_block", dilation);

    // TDNN2: pointwise conv
    y = tdnn_block(y, prefix + ".tdnn2");

    // SE block
    y = se_block(y, prefix + ".se_block");

    // Residual connection
    return y + residual;
}

mx::array ECAPAInference::asp(const mx::array& x, const std::string& prefix) {
    // Compute global statistics for context
    auto mean = mx::mean(x, 1, true);  // (B, 1, C)
    auto var = mx::var(x, 1, true);    // (B, 1, C)
    auto std_val = mx::sqrt(var + 1e-5f);

    // Broadcast to time dimension using Shape (SmallVector<int>)
    mx::Shape target_shape = {(int)x.shape(0), (int)x.shape(1), (int)x.shape(2)};
    auto mean_bc = mx::broadcast_to(mean, target_shape);
    auto std_bc = mx::broadcast_to(std_val, target_shape);

    // Concatenate features with context
    auto context = mx::concatenate({x, mean_bc, std_bc}, 2);

    // Compute attention weights
    auto attn = tdnn_block(context, prefix + ".tdnn");
    attn = conv1d(attn, prefix + ".conv");
    attn = mx::softmax(attn, 1);  // Softmax over time

    // Weighted mean
    auto weighted_mean = mx::sum(x * attn, 1, true);

    // Weighted std
    auto weighted_var = mx::sum(mx::square(x - weighted_mean) * attn, 1, true);
    auto weighted_std = mx::sqrt(weighted_var + 1e-5f);

    // Concatenate mean and std
    return mx::concatenate({weighted_mean, weighted_std}, 2);
}

mx::array ECAPAInference::embedding_forward(const mx::array& x) {
    // Ensure input is (B, T, C) format
    mx::array input = x;
    if (x.shape(2) != config_.n_mels && x.shape(1) == config_.n_mels) {
        // Input is (B, C, T), convert to (B, T, C)
        input = mx::transpose(x, {0, 2, 1});
    }

    // Block 0: Initial TDNN
    auto y = tdnn_block(input, "embedding_model.blocks_0");

    // SE-Res2Net blocks
    auto out1 = se_res2net_block(y, "embedding_model.blocks_1", config_.dilations[1]);
    auto out2 = se_res2net_block(out1, "embedding_model.blocks_2", config_.dilations[2]);
    auto out3 = se_res2net_block(out2, "embedding_model.blocks_3", config_.dilations[3]);

    // Multi-Feature Aggregation
    auto mfa_in = mx::concatenate({out1, out2, out3}, 2);
    auto mfa_out = tdnn_block(mfa_in, "embedding_model.mfa");

    // Attentive Statistics Pooling
    auto asp_out = asp(mfa_out, "embedding_model.asp");

    // BatchNorm
    asp_out = batch_norm(asp_out, "embedding_model.asp_bn");

    // Final embedding layer
    auto embedding = conv1d(asp_out, "embedding_model.fc");

    return embedding;
}

mx::array ECAPAInference::classifier_forward(const mx::array& x) {
    // x: (B, 1, D) or (B, D)
    mx::array input = x;
    if (x.ndim() == 3) {
        input = mx::squeeze(x, 1);  // (B, 1, D) -> (B, D)
    }

    // Reshape for BatchNorm1d: (B, D) -> (B, 1, D)
    input = mx::reshape(input, {(int)input.shape(0), 1, (int)input.shape(1)});
    input = batch_norm(input, "classifier.norm");
    input = mx::squeeze(input, 1);  // (B, 1, D) -> (B, D)

    // DNN block
    auto linear_weight = weights_.at("classifier.linear.weight");
    auto linear_bias = weights_.at("classifier.linear.bias");
    input = mx::matmul(input, mx::transpose(linear_weight)) + linear_bias;

    input = mx::reshape(input, {(int)input.shape(0), 1, (int)input.shape(1)});
    input = batch_norm(input, "classifier.dnn_norm");
    input = mx::squeeze(input, 1);

    // Leaky ReLU
    input = mx::where(input > 0, input, input * 0.01f);

    // Output layer
    auto out_weight = weights_.at("classifier.out.weight");
    auto out_bias = weights_.at("classifier.out.bias");
    auto logits = mx::matmul(input, mx::transpose(out_weight)) + out_bias;

    return logits;
}

std::vector<mx::array> ECAPAInference::forward_impl(const std::vector<mx::array>& inputs) {
    auto embedding = embedding_forward(inputs[0]);
    auto logits = classifier_forward(embedding);
    auto predictions = mx::argmax(logits, -1);
    return {logits, predictions, mx::squeeze(embedding, 1)};
}

void ECAPAInference::compile_model() {
    if (compiled_) return;

    // Create a wrapper function that captures this pointer
    auto forward_fn = [this](const std::vector<mx::array>& inputs) {
        return this->forward_impl(inputs);
    };

    // Compile the function
    compiled_forward_ = mx::compile(forward_fn, false);
    compiled_ = true;
}

mx::array ECAPAInference::extract_embedding(const mx::array& mel_input) {
    if (compiled_) {
        auto results = compiled_forward_({mel_input});
        mx::eval(results[2]);
        return results[2];
    }
    auto embedding = embedding_forward(mel_input);
    mx::eval(embedding);
    return mx::squeeze(embedding, 1);  // (B, 1, D) -> (B, D)
}

std::pair<mx::array, mx::array> ECAPAInference::classify(const mx::array& mel_input) {
    if (compiled_) {
        auto results = compiled_forward_({mel_input});
        mx::eval(results[0]);
        mx::eval(results[1]);
        return {results[0], results[1]};
    }

    auto embedding = embedding_forward(mel_input);
    auto logits = classifier_forward(embedding);

    // Get predictions
    auto predictions = mx::argmax(logits, -1);

    mx::eval(logits);
    mx::eval(predictions);

    return {logits, predictions};
}

// Benchmark implementation
double ECAPABenchmark::benchmark_latency(
    ECAPAInference& model,
    int batch_size,
    int seq_len,
    int num_iterations
) {
    // Create random input
    auto input = mx::random::normal({batch_size, seq_len, model.config().n_mels});
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
    return static_cast<double>(duration.count()) / 1000.0 / num_iterations;  // ms per iteration
}

double ECAPABenchmark::compare_baseline(
    ECAPAInference& model,
    double baseline_ms,
    int batch_size,
    int seq_len
) {
    double latency_ms = benchmark_latency(model, batch_size, seq_len, 100);
    return baseline_ms / latency_ms;
}

} // namespace ecapa
