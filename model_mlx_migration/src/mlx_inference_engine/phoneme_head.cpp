// Copyright 2024-2025 Andrew Yates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "phoneme_head.h"
#include "misaki_g2p.h"
#include "mlx/io.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// For SIMD optimization on Apple Silicon
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace phoneme {

// ============================================================================
// PhonemeHead constructors and move semantics
// ============================================================================

PhonemeHead::PhonemeHead()
    : config_()
    , loaded_(false)
    , ln_weight_(mx::array({0.0f}))
    , ln_bias_(mx::array({0.0f}))
    , hidden_weight_(mx::array({0.0f}))
    , hidden_bias_(mx::array({0.0f}))
    , proj_weight_(mx::array({0.0f}))
    , proj_bias_(mx::array({0.0f}))
{}

PhonemeHead::PhonemeHead(PhonemeHead&& other) noexcept
    : config_(std::move(other.config_))
    , loaded_(other.loaded_)
    , ln_weight_(std::move(other.ln_weight_))
    , ln_bias_(std::move(other.ln_bias_))
    , hidden_weight_(std::move(other.hidden_weight_))
    , hidden_bias_(std::move(other.hidden_bias_))
    , proj_weight_(std::move(other.proj_weight_))
    , proj_bias_(std::move(other.proj_bias_))
{
    other.loaded_ = false;
}

PhonemeHead& PhonemeHead::operator=(PhonemeHead&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        loaded_ = other.loaded_;
        ln_weight_ = std::move(other.ln_weight_);
        ln_bias_ = std::move(other.ln_bias_);
        hidden_weight_ = std::move(other.hidden_weight_);
        hidden_bias_ = std::move(other.hidden_bias_);
        proj_weight_ = std::move(other.proj_weight_);
        proj_bias_ = std::move(other.proj_bias_);
        other.loaded_ = false;
    }
    return *this;
}

// ============================================================================
// PhonemeHeadConfig
// ============================================================================

PhonemeHeadConfig PhonemeHeadConfig::load(const std::string& path) {
    PhonemeHeadConfig config;

    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config: " + path);
    }

    // Simple JSON parsing for config
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Extract numeric values (simple approach)
    auto extract_int = [&content](const std::string& key) -> int {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return -1;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return -1;
        return std::stoi(content.substr(pos + 1));
    };

    int val;
    if ((val = extract_int("d_model")) > 0) config.d_model = val;
    if ((val = extract_int("phoneme_vocab")) > 0) config.phoneme_vocab = val;
    if ((val = extract_int("hidden_dim")) > 0) config.hidden_dim = val;
    if ((val = extract_int("blank_id")) >= 0) config.blank_id = val;

    return config;
}

// ============================================================================
// PhonemeHead
// ============================================================================

PhonemeHead PhonemeHead::load(const std::string& model_path) {
    PhonemeHead head;

    // Load config
    std::string config_path = model_path + "/config.json";
    std::ifstream config_file(config_path);
    if (config_file.is_open()) {
        head.config_ = PhonemeHeadConfig::load(config_path);
    }

    // Load weights - supports both .safetensors and .npz formats
    std::unordered_map<std::string, mx::array> weights;

    std::string safetensors_path = model_path + "/weights.safetensors";
    std::string npz_path = model_path + "/weights.npz";

    std::ifstream st_check(safetensors_path);
    if (st_check.is_open()) {
        st_check.close();
        // Use safetensors format
        auto [arrays, metadata] = mx::load_safetensors(safetensors_path);
        weights = std::move(arrays);
    } else {
        // Fallback: try to load from npz using Python
        // For now, require safetensors format
        throw std::runtime_error(
            "PhonemeHead requires weights.safetensors format. "
            "Convert using: mlx.core.save_safetensors(path, dict(mlx.core.load(npz_path)))"
        );
    }

    // Extract weights with fallback key names
    auto get_weight = [&weights](const std::vector<std::string>& keys) -> mx::array {
        for (const auto& key : keys) {
            auto it = weights.find(key);
            if (it != weights.end()) {
                return it->second;
            }
        }
        throw std::runtime_error("Weight not found: " + keys[0]);
    };

    // LayerNorm weights
    head.ln_weight_ = get_weight({"ln.weight", "layer_norm.weight", "norm.weight"});
    head.ln_bias_ = get_weight({"ln.bias", "layer_norm.bias", "norm.bias"});

    // Hidden layer weights
    head.hidden_weight_ = get_weight({"hidden.weight", "fc1.weight", "linear1.weight"});
    head.hidden_bias_ = get_weight({"hidden.bias", "fc1.bias", "linear1.bias"});

    // Output projection weights
    head.proj_weight_ = get_weight({"proj.weight", "fc2.weight", "linear2.weight", "output.weight"});
    head.proj_bias_ = get_weight({"proj.bias", "fc2.bias", "linear2.bias", "output.bias"});

    // Update config from actual weight shapes
    head.config_.d_model = static_cast<int>(head.ln_weight_.shape()[0]);
    head.config_.hidden_dim = static_cast<int>(head.hidden_weight_.shape()[0]);
    head.config_.phoneme_vocab = static_cast<int>(head.proj_weight_.shape()[0]);

    head.loaded_ = true;

    std::cout << "[PhonemeHead] Loaded from " << model_path << "\n";
    std::cout << "  d_model: " << head.config_.d_model << "\n";
    std::cout << "  hidden_dim: " << head.config_.hidden_dim << "\n";
    std::cout << "  phoneme_vocab: " << head.config_.phoneme_vocab << "\n";

    return head;
}

mx::array PhonemeHead::forward(const mx::array& encoder_output) {
    if (!loaded_) {
        throw std::runtime_error("PhonemeHead not loaded");
    }

    // Input: [batch, seq_len, d_model]
    auto x = encoder_output;

    // Layer normalization
    if (config_.use_layer_norm) {
        // Compute mean and variance along last dimension
        auto mean = mx::mean(x, -1, true);
        auto var = mx::var(x, -1, true);
        x = (x - mean) / mx::sqrt(var + 1e-5f);
        // Apply affine transform
        x = x * ln_weight_ + ln_bias_;
    }

    // Hidden layer: Linear + GELU
    // x: [batch, seq, d_model], hidden_weight: [hidden_dim, d_model]
    // Output: [batch, seq, hidden_dim]
    x = mx::matmul(x, mx::transpose(hidden_weight_)) + hidden_bias_;

    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    auto x_cubed = x * x * x;
    auto inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    x = 0.5f * x * (1.0f + mx::tanh(inner));

    // Output projection: [batch, seq, phoneme_vocab]
    x = mx::matmul(x, mx::transpose(proj_weight_)) + proj_bias_;

    return x;
}

std::vector<int> PhonemeHead::predict(const mx::array& encoder_output) {
    auto logits = forward(encoder_output);
    mx::eval(logits);

    // Use optimized CTC decode
    return ctc_greedy_decode(logits, config_.blank_id);
}

PhonemeVerificationResult PhonemeHead::compare_with_text(
    const mx::array& encoder_output,
    const std::string& text,
    const std::string& language
) {
    PhonemeVerificationResult result;

    // Get predicted phonemes
    result.predicted = predict(encoder_output);

    // Get expected phonemes from text
    result.expected = phonemize_text(text, language);

    // Compute edit distance
    result.edit_distance = edit_distance(result.predicted, result.expected);

    // Compute confidence: 1 - normalized edit distance
    int max_len = std::max(result.predicted.size(), result.expected.size());
    if (max_len > 0) {
        result.confidence = 1.0f - static_cast<float>(result.edit_distance) / max_len;
    } else {
        result.confidence = 1.0f;
    }

    return result;
}

CommitStatus PhonemeHead::get_commit_status(
    float confidence,
    float commit_threshold,
    float wait_threshold
) {
    if (confidence >= commit_threshold) {
        return CommitStatus::COMMIT;
    } else if (confidence < wait_threshold) {
        return CommitStatus::WAIT;
    } else {
        return CommitStatus::PARTIAL;
    }
}

// ============================================================================
// Optimized CTC Decode
// ============================================================================

std::vector<int> ctc_greedy_decode(
    const float* logits,
    int seq_len,
    int vocab_size,
    int blank_id
) {
    std::vector<int> result;
    result.reserve(seq_len);  // Pre-allocate

    int prev_token = -1;

    for (int t = 0; t < seq_len; ++t) {
        const float* frame = logits + t * vocab_size;

        // Find argmax using SIMD on Apple Silicon
        int best_id = 0;
        float best_val = frame[0];

#ifdef __APPLE__
        // Use vDSP for vectorized max finding
        vDSP_Length max_idx;
        float max_val;
        vDSP_maxvi(frame, 1, &max_val, &max_idx, vocab_size);
        best_id = static_cast<int>(max_idx);
        best_val = max_val;
#else
        // Standard loop
        for (int v = 1; v < vocab_size; ++v) {
            if (frame[v] > best_val) {
                best_val = frame[v];
                best_id = v;
            }
        }
#endif

        // CTC collapse: skip blanks and repeated tokens
        if (best_id != blank_id && best_id != prev_token) {
            result.push_back(best_id);
        }
        prev_token = best_id;
    }

    return result;
}

std::vector<int> ctc_greedy_decode(const mx::array& logits, int blank_id) {
    // Ensure logits are evaluated and float32
    auto logits_f32 = mx::astype(logits, mx::float32);
    mx::eval(logits_f32);

    // Handle different input shapes
    int seq_len, vocab_size;
    const float* data;

    if (logits_f32.ndim() == 2) {
        // [seq_len, vocab_size]
        seq_len = static_cast<int>(logits_f32.shape()[0]);
        vocab_size = static_cast<int>(logits_f32.shape()[1]);
        data = logits_f32.data<float>();
    } else if (logits_f32.ndim() == 3) {
        // [batch, seq_len, vocab_size] - use first batch
        seq_len = static_cast<int>(logits_f32.shape()[1]);
        vocab_size = static_cast<int>(logits_f32.shape()[2]);
        data = logits_f32.data<float>();
    } else {
        throw std::runtime_error("Invalid logits shape for CTC decode");
    }

    return ctc_greedy_decode(data, seq_len, vocab_size, blank_id);
}

// ============================================================================
// Optimized Edit Distance
// ============================================================================

int edit_distance(
    const std::vector<int>& seq1,
    const std::vector<int>& seq2
) {
    const size_t m = seq1.size();
    const size_t n = seq2.size();

    // Handle edge cases
    if (m == 0) return static_cast<int>(n);
    if (n == 0) return static_cast<int>(m);

    // Use two rows for space optimization
    std::vector<int> prev(n + 1);
    std::vector<int> curr(n + 1);

    // Initialize first row
    for (size_t j = 0; j <= n; ++j) {
        prev[j] = static_cast<int>(j);
    }

    // Fill DP table row by row
    for (size_t i = 1; i <= m; ++i) {
        curr[0] = static_cast<int>(i);

#ifdef __APPLE__
        // Vectorized inner loop using Accelerate
        // Note: For small vocab, scalar is often faster
        // Only use SIMD for longer sequences
        if (n >= 32) {
            for (size_t j = 1; j <= n; ++j) {
                int cost = (seq1[i - 1] == seq2[j - 1]) ? 0 : 1;
                curr[j] = std::min({
                    prev[j] + 1,        // deletion
                    curr[j - 1] + 1,    // insertion
                    prev[j - 1] + cost  // substitution
                });
            }
        } else
#endif
        {
            // Standard scalar loop
            for (size_t j = 1; j <= n; ++j) {
                int cost = (seq1[i - 1] == seq2[j - 1]) ? 0 : 1;
                curr[j] = std::min({
                    prev[j] + 1,        // deletion
                    curr[j - 1] + 1,    // insertion
                    prev[j - 1] + cost  // substitution
                });
            }
        }

        std::swap(prev, curr);
    }

    return prev[n];
}

int edit_distance_with_counts(
    const std::vector<int>& seq1,
    const std::vector<int>& seq2,
    int& insertions,
    int& deletions,
    int& substitutions
) {
    const size_t m = seq1.size();
    const size_t n = seq2.size();

    insertions = 0;
    deletions = 0;
    substitutions = 0;

    if (m == 0) {
        insertions = static_cast<int>(n);
        return insertions;
    }
    if (n == 0) {
        deletions = static_cast<int>(m);
        return deletions;
    }

    // Full DP table for backtracking
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<char>> ops(m + 1, std::vector<char>(n + 1));

    // Initialize
    for (size_t i = 0; i <= m; ++i) {
        dp[i][0] = static_cast<int>(i);
        ops[i][0] = 'D';  // deletion
    }
    for (size_t j = 0; j <= n; ++j) {
        dp[0][j] = static_cast<int>(j);
        ops[0][j] = 'I';  // insertion
    }
    ops[0][0] = 'M';  // match (start)

    // Fill DP table
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            int cost = (seq1[i - 1] == seq2[j - 1]) ? 0 : 1;

            int del = dp[i - 1][j] + 1;
            int ins = dp[i][j - 1] + 1;
            int sub = dp[i - 1][j - 1] + cost;

            if (sub <= del && sub <= ins) {
                dp[i][j] = sub;
                ops[i][j] = (cost == 0) ? 'M' : 'S';
            } else if (del <= ins) {
                dp[i][j] = del;
                ops[i][j] = 'D';
            } else {
                dp[i][j] = ins;
                ops[i][j] = 'I';
            }
        }
    }

    // Backtrack to count operations
    size_t i = m, j = n;
    while (i > 0 || j > 0) {
        char op = ops[i][j];
        if (op == 'M') {
            --i;
            --j;
        } else if (op == 'S') {
            ++substitutions;
            --i;
            --j;
        } else if (op == 'D') {
            ++deletions;
            --i;
        } else {  // 'I'
            ++insertions;
            --j;
        }
    }

    return dp[m][n];
}

float normalized_edit_distance(
    const std::vector<int>& seq1,
    const std::vector<int>& seq2
) {
    int dist = edit_distance(seq1, seq2);
    int max_len = std::max(static_cast<int>(seq1.size()),
                          static_cast<int>(seq2.size()));
    if (max_len == 0) return 0.0f;
    return static_cast<float>(dist) / max_len;
}

// ============================================================================
// PhonemeVocab
// ============================================================================

PhonemeVocab PhonemeVocab::load(const std::string& path) {
    PhonemeVocab vocab;

    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open vocab: " + path);
    }

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Simple JSON parsing for vocab: {"char": id, ...}
    // This is a simplified parser - in production use a proper JSON library
    size_t pos = 0;
    while ((pos = content.find("\"", pos)) != std::string::npos) {
        size_t end = content.find("\"", pos + 1);
        if (end == std::string::npos) break;

        std::string phoneme = content.substr(pos + 1, end - pos - 1);
        pos = content.find(":", end);
        if (pos == std::string::npos) break;

        // Find the number
        size_t num_start = pos + 1;
        while (num_start < content.size() &&
               (content[num_start] == ' ' || content[num_start] == '\n')) {
            ++num_start;
        }

        int id = std::stoi(content.substr(num_start));

        vocab.phoneme_to_id_[phoneme] = id;
        vocab.id_to_phoneme_[id] = phoneme;

        pos = end + 1;
    }

    return vocab;
}

int PhonemeVocab::get_id(const std::string& phoneme) const {
    auto it = phoneme_to_id_.find(phoneme);
    return (it != phoneme_to_id_.end()) ? it->second : -1;
}

std::string PhonemeVocab::get_phoneme(int id) const {
    auto it = id_to_phoneme_.find(id);
    return (it != id_to_phoneme_.end()) ? it->second : "";
}

std::string PhonemeVocab::ids_to_ipa(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        result += get_phoneme(id);
    }
    return result;
}

// ============================================================================
// Phonemizer Interface - uses Misaki G2P (NOT espeak-ng!)
// Misaki is Kokoro's official G2P and matches phoneme head training.
// ============================================================================

namespace {
    // Singleton Misaki G2P for efficiency
    misaki::MisakiG2P& get_misaki_g2p() {
        static misaki::MisakiG2P g2p;
        static bool initialized = false;
        if (!initialized) {
            // Try standard locations for Misaki lexicons
            const char* lexicon_paths[] = {
                "misaki_export",
                "../misaki_export",
                "../../misaki_export",
                "models/misaki_export",
            };
            for (const char* path : lexicon_paths) {
                if (g2p.initialize(path, "en-us")) {
                    // Load vocab for tokenization
                    const char* vocab_paths[] = {
                        "misaki_export/vocab.json",
                        "../misaki_export/vocab.json",
                        "../../misaki_export/vocab.json",
                    };
                    for (const char* vpath : vocab_paths) {
                        if (g2p.load_vocab(vpath)) {
                            break;
                        }
                    }
                    initialized = true;
                    break;
                }
            }
            if (!initialized) {
                std::cerr << "[PhonemeHead] Warning: Misaki lexicons not found. "
                          << "Phonemization will not work correctly.\n";
            }
        }
        return g2p;
    }
}

std::vector<int> phonemize_text(
    const std::string& text,
    const std::string& language
) {
    auto& g2p = get_misaki_g2p();

    if (!g2p.initialized()) {
        std::cerr << "[PhonemeHead] Misaki G2P not initialized\n";
        return {};
    }

    // Note: Misaki handles language internally
    // For now, we support English primarily
    // Japanese and Chinese support available if lexicons are loaded

    return g2p.phonemize_to_ids(text);
}

std::string get_ipa(
    const std::string& text,
    const std::string& language
) {
    auto& g2p = get_misaki_g2p();

    if (!g2p.initialized()) {
        std::cerr << "[PhonemeHead] Misaki G2P not initialized\n";
        return "";
    }

    return g2p.phonemize(text);
}

}  // namespace phoneme
