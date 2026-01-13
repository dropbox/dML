/**
 * @file ast_inference.h
 * @brief Audio Spectrogram Transformer (AST) Native MLX C++ Inference
 *
 * High-performance inference without Python dispatch overhead.
 * Target: <0.5ms latency at batch=1 (vs ~3-4ms with MLX Python)
 *
 * Architecture (MIT/ast-finetuned-audioset-10-10-0.4593):
 *   Input: (B, 1, freq=128, time=1024) mel spectrogram
 *   Patch Embedding: Conv2d(1, 768, 16x16, stride=10x10) -> (B, 1212, 768)
 *   + CLS token + distillation token -> (B, 1214, 768)
 *   + Position embeddings
 *   Encoder: 12 transformer layers (pre-norm)
 *   Pooling: (CLS + DIST) / 2
 *   Classifier: LayerNorm -> Linear -> (B, 527) logits
 */

#pragma once

#include <mlx/mlx.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace ast {

namespace mx = mlx::core;

/**
 * @brief Configuration for Audio Spectrogram Transformer
 */
struct ASTConfig {
    // Transformer architecture
    int hidden_size = 768;
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int intermediate_size = 3072;
    float layer_norm_eps = 1e-12f;
    bool qkv_bias = true;

    // Patch embedding
    int patch_size = 16;
    int num_mel_bins = 128;
    int max_length = 1024;
    int time_stride = 10;
    int frequency_stride = 10;

    // Output
    int num_labels = 527;

    // Derived
    int head_dim() const { return hidden_size / num_attention_heads; }

    int num_time_patches() const {
        return (max_length - patch_size) / time_stride + 1;  // 101
    }

    int num_freq_patches() const {
        return (num_mel_bins - patch_size) / frequency_stride + 1;  // 12
    }

    int num_patches() const {
        return num_time_patches() * num_freq_patches();  // 1212
    }

    static ASTConfig audioset() {
        return ASTConfig();
    }
};

/**
 * @brief High-performance AST inference engine
 *
 * Uses MLX C++ API directly to eliminate Python dispatch overhead.
 */
class ASTInference {
public:
    /**
     * @brief Construct inference engine from weights file
     * @param weights_path Path to weights.npz file
     * @param config Model configuration
     */
    ASTInference(const std::string& weights_path,
                 const ASTConfig& config = ASTConfig::audioset());

    /**
     * @brief Classify audio from mel spectrogram
     * @param mel_input Mel spectrogram (B, freq, time) or (B, 1, freq, time)
     * @return Tuple of (logits, predicted_indices)
     */
    std::pair<mx::array, mx::array> classify(const mx::array& mel_input);

    /**
     * @brief Extract pooled hidden states
     * @param mel_input Mel spectrogram (B, freq, time) or (B, 1, freq, time)
     * @return Pooled hidden states (B, hidden_size)
     */
    mx::array extract_features(const mx::array& mel_input);

    /**
     * @brief Get label from prediction index
     * @param index Prediction index
     * @return Label string
     */
    std::string get_label(int index) const;

    /**
     * @brief Load labels from JSON file
     * @param labels_path Path to labels.json
     */
    void load_labels(const std::string& labels_path);

    /**
     * @brief Get model configuration
     */
    const ASTConfig& config() const { return config_; }

    /**
     * @brief Compile the model for faster inference
     */
    void compile_model();

private:
    ASTConfig config_;
    std::unordered_map<std::string, mx::array> weights_;
    std::unordered_map<int, std::string> labels_;
    bool compiled_ = false;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled_forward_;

    // Embedding operations
    mx::array patch_embed(const mx::array& x);
    mx::array add_position_embeddings(const mx::array& x);

    // Transformer operations
    mx::array layer_norm(const mx::array& x, const std::string& prefix);
    mx::array linear(const mx::array& x, const std::string& prefix, bool bias = true);
    mx::array self_attention(const mx::array& x, int layer_idx);
    mx::array mlp(const mx::array& x, int layer_idx);
    mx::array encoder_layer(const mx::array& x, int layer_idx);
    mx::array encoder(const mx::array& x);

    // Full forward passes
    mx::array embeddings_forward(const mx::array& x);
    mx::array transformer_forward(const mx::array& x);
    mx::array classifier_forward(const mx::array& pooled);

    // Forward implementation for compilation
    std::vector<mx::array> forward_impl(const std::vector<mx::array>& inputs);
};

/**
 * @brief Benchmark utilities for AST
 */
class ASTBenchmark {
public:
    /**
     * @brief Benchmark inference latency
     * @param model Inference model
     * @param batch_size Batch size to test
     * @param num_iterations Number of iterations
     * @return Average latency in milliseconds
     */
    static double benchmark_latency(
        ASTInference& model,
        int batch_size,
        int num_iterations = 100
    );

    /**
     * @brief Compare with baseline timing
     * @param model Inference model
     * @param baseline_ms Baseline latency in ms
     * @param batch_size Batch size
     * @return Speedup factor
     */
    static double compare_baseline(
        ASTInference& model,
        double baseline_ms,
        int batch_size = 1
    );
};

} // namespace ast
