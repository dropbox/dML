/**
 * @file ecapa_inference.h
 * @brief ECAPA-TDNN Native MLX C++ Inference
 *
 * High-performance inference without Python dispatch overhead.
 * Target: <0.3ms latency at batch=1 (vs 2.79ms with MLX Python)
 *
 * Architecture (VoxLingua107):
 *   Input: (B, T, 60) mel filterbanks
 *   Block 0: TDNN -> (B, T, 1024)
 *   Block 1-3: SE-Res2Net -> (B, T, 1024) each
 *   MFA: Concat + TDNN -> (B, T, 3072)
 *   ASP: Attention pooling -> (B, 1, 6144)
 *   FC: Linear -> (B, 256) embedding
 *   Classifier: -> (B, 107) logits
 */

#pragma once

#include <mlx/mlx.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace ecapa {

namespace mx = mlx::core;

/**
 * @brief Configuration for ECAPA-TDNN model
 */
struct ECAPAConfig {
    int n_mels = 60;
    int lin_neurons = 256;
    int attention_channels = 128;
    int res2net_scale = 8;
    int se_channels = 128;
    int classifier_hidden = 512;
    int num_languages = 107;

    // Channel dimensions for each block
    std::vector<int> channels = {1024, 1024, 1024, 1024, 3072};

    // Kernel sizes for each block
    std::vector<int> kernel_sizes = {5, 3, 3, 3};

    // Dilation factors for each block
    std::vector<int> dilations = {1, 2, 3, 4};

    static ECAPAConfig voxlingua107() {
        return ECAPAConfig();
    }
};

/**
 * @brief High-performance ECAPA-TDNN inference engine
 *
 * Uses MLX C++ API directly to eliminate Python dispatch overhead.
 * All operations use native MLX (B, T, C) tensor format.
 */
class ECAPAInference {
public:
    /**
     * @brief Construct inference engine from weights file
     * @param weights_path Path to weights.npz file
     * @param config Model configuration
     */
    ECAPAInference(const std::string& weights_path,
                   const ECAPAConfig& config = ECAPAConfig::voxlingua107());

    /**
     * @brief Extract embeddings from mel features
     * @param mel_input Mel filterbank features (B, T, 60) or (B, 60, T)
     * @return Embeddings (B, 256)
     */
    mx::array extract_embedding(const mx::array& mel_input);

    /**
     * @brief Classify language from mel features
     * @param mel_input Mel filterbank features (B, T, 60) or (B, 60, T)
     * @return Tuple of (logits, predicted_indices)
     */
    std::pair<mx::array, mx::array> classify(const mx::array& mel_input);

    /**
     * @brief Get language code from prediction index
     * @param index Prediction index
     * @return Language code string (e.g., "en", "th", "pt")
     */
    std::string get_language_code(int index) const;

    /**
     * @brief Load label encoder from file
     * @param labels_path Path to label_encoder.txt
     */
    void load_labels(const std::string& labels_path);

    /**
     * @brief Get model configuration
     */
    const ECAPAConfig& config() const { return config_; }

    /**
     * @brief Compile the model for faster inference
     * Must be called after loading weights, before inference
     */
    void compile_model();

private:
    ECAPAConfig config_;
    std::unordered_map<std::string, mx::array> weights_;
    std::unordered_map<int, std::string> labels_;
    bool compiled_ = false;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled_forward_;

    // Helper functions for layer operations
    mx::array conv1d(const mx::array& x, const std::string& prefix);
    mx::array batch_norm(const mx::array& x, const std::string& prefix);
    mx::array tdnn_block(const mx::array& x, const std::string& prefix);
    mx::array se_res2net_block(const mx::array& x, const std::string& prefix, int dilation);
    mx::array res2net_block(const mx::array& x, const std::string& prefix, int dilation);
    mx::array se_block(const mx::array& x, const std::string& prefix);
    mx::array asp(const mx::array& x, const std::string& prefix);

    // Embedding model forward pass
    mx::array embedding_forward(const mx::array& x);

    // Classifier forward pass
    mx::array classifier_forward(const mx::array& x);

    // Forward pass for compilation
    std::vector<mx::array> forward_impl(const std::vector<mx::array>& inputs);
};

/**
 * @brief Benchmark utilities
 */
class ECAPABenchmark {
public:
    /**
     * @brief Benchmark inference latency
     * @param model Inference model
     * @param batch_size Batch size to test
     * @param seq_len Sequence length (time frames)
     * @param num_iterations Number of iterations
     * @return Average latency in milliseconds
     */
    static double benchmark_latency(
        ECAPAInference& model,
        int batch_size,
        int seq_len,
        int num_iterations = 100
    );

    /**
     * @brief Compare with baseline timing
     * @param model Inference model
     * @param baseline_ms Baseline latency in ms (e.g., 2.79 for MLX Python)
     * @return Speedup factor
     */
    static double compare_baseline(
        ECAPAInference& model,
        double baseline_ms,
        int batch_size = 1,
        int seq_len = 300
    );
};

} // namespace ecapa
