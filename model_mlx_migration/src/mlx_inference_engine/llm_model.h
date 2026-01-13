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

// LLM Model C++ Implementation
// Decoder-only transformer for text generation (LLaMA-style architecture)
//
// Architecture:
// - Token Embeddings + RoPE (Rotary Position Embedding)
// - Decoder blocks: RMSNorm -> Self-Attention -> RMSNorm -> SwiGLU MLP
// - Final RMSNorm + LM Head
//
// Supports: LLaMA, Mistral, Qwen2, and similar architectures

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace llm {

/**
 * Configuration for LLM model.
 */
struct LLMConfig {
    // Model dimensions
    int vocab_size = 32000;        // Vocabulary size
    int hidden_size = 4096;        // Hidden dimension
    int intermediate_size = 11008; // MLP intermediate dimension
    int num_hidden_layers = 32;    // Number of transformer layers
    int num_attention_heads = 32;  // Number of attention heads
    int num_key_value_heads = 32;  // Number of KV heads (for GQA)
    int head_dim = 0;              // Head dimension (default: hidden_size / num_heads)
    int max_position_embeddings = 4096; // Max sequence length

    // Normalization
    float rms_norm_eps = 1e-5f;

    // RoPE parameters
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
    std::string rope_scaling_type;  // "linear", "dynamic", "su", "yarn", etc.
    float rope_scaling_factor = 1.0f;

    // Model behavior
    bool tie_word_embeddings = true;
    bool attention_bias = false;
    bool mlp_bias = false;

    // Model metadata
    std::string model_type = "llama";

    // Quantization parameters
    int quantization_bits = 0;       // 0 = no quantization, 4 = 4-bit, etc.
    int quantization_group_size = 64;

    // Load config from JSON file
    static LLMConfig load(const std::string& path);

    // Get predefined config by name
    static LLMConfig get(const std::string& model_name);

    // Calculate head_dim if not specified
    void finalize() {
        if (head_dim == 0) {
            head_dim = hidden_size / num_attention_heads;
        }
        if (num_key_value_heads == 0) {
            num_key_value_heads = num_attention_heads;
        }
    }
};

/**
 * Weight storage for LLM model with quantization support.
 */
class Weights {
public:
    Weights() = default;

    // Load weights from safetensors file
    void load(const std::string& path);

    // Load weights from multiple sharded safetensors files
    void load_sharded(const std::string& directory);

    // Get a weight by name (auto-dequantizes if quantized)
    // For quantized models, looks for .weight, .scales, .biases triplet
    mx::array get(const std::string& name) const;

    // Get raw weight without dequantization
    mx::array get_raw(const std::string& name) const;

    // Check if weight exists (either raw or quantized)
    bool has(const std::string& name) const;

    // Check if a weight is quantized
    bool is_quantized(const std::string& name) const;

    // Number of weights
    size_t size() const { return weights_.size(); }

    // Set quantization parameters for dequantization
    void set_quantization(int bits, int group_size) {
        quantization_bits_ = bits;
        quantization_group_size_ = group_size;
    }

private:
    std::unordered_map<std::string, mx::array> weights_;
    int quantization_bits_ = 0;
    int quantization_group_size_ = 64;
};

/**
 * KV Cache for decoder self-attention.
 */
struct KVCache {
    // [num_layers] each [batch, num_kv_heads, seq_len, head_dim]
    std::vector<mx::array> keys;
    std::vector<mx::array> values;

    // Current offset (for RoPE)
    int offset = 0;

    void clear() {
        keys.clear();
        values.clear();
        offset = 0;
    }

    bool empty() const { return keys.empty(); }

    // Get current sequence length from cache
    int seq_len() const {
        if (keys.empty()) return 0;
        return static_cast<int>(keys[0].shape()[2]);
    }

    // Update offset after appending tokens
    void update_offset(int new_tokens) {
        offset += new_tokens;
    }
};

/**
 * Tokenizer for LLM (SentencePiece or tiktoken-based).
 */
class LLMTokenizer {
public:
    LLMTokenizer();
    ~LLMTokenizer();

    // Move semantics
    LLMTokenizer(LLMTokenizer&&) noexcept;
    LLMTokenizer& operator=(LLMTokenizer&&) noexcept;

    // Disable copy
    LLMTokenizer(const LLMTokenizer&) = delete;
    LLMTokenizer& operator=(const LLMTokenizer&) = delete;

    /**
     * Load tokenizer from model directory.
     * Supports: tokenizer.json, tokenizer.model (SentencePiece)
     */
    static LLMTokenizer load(const std::string& model_path);

    /**
     * Encode text to token IDs.
     */
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    /**
     * Decode token IDs to text.
     */
    std::string decode(const std::vector<int>& ids) const;

    /**
     * Get vocabulary size.
     */
    int vocab_size() const;

    /**
     * Check if loaded.
     */
    bool loaded() const { return loaded_; }

    // Special tokens
    int bos_token() const { return bos_token_; }
    int eos_token() const { return eos_token_; }
    int pad_token() const { return pad_token_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;

    // Special tokens
    int bos_token_ = 1;    // <s>
    int eos_token_ = 2;    // </s>
    int pad_token_ = 0;    // <pad> or <unk>
};

/**
 * Full LLM model for text generation.
 */
class LLMModel {
public:
    LLMModel();
    ~LLMModel();

    // Move semantics
    LLMModel(LLMModel&&) noexcept;
    LLMModel& operator=(LLMModel&&) noexcept;

    // Disable copy
    LLMModel(const LLMModel&) = delete;
    LLMModel& operator=(const LLMModel&) = delete;

    /**
     * Load model from directory.
     * Expected structure:
     *   model.safetensors or model-*.safetensors (sharded)
     *   config.json
     *   tokenizer.json or tokenizer.model
     */
    static LLMModel load(const std::string& model_path);

    /**
     * Forward pass: compute logits.
     * @param tokens Token IDs [batch, seq_len]
     * @param cache Optional KV cache (updated in-place)
     * @return Logits [batch, seq_len, vocab_size]
     */
    mx::array forward(const mx::array& tokens, KVCache* cache = nullptr);

    /**
     * Generate tokens from prompt.
     * @param tokens Initial token IDs
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @param top_p Top-p (nucleus) sampling
     * @param top_k Top-k sampling (0 to disable)
     * @return Generated token IDs (including prompt)
     */
    std::vector<int> generate(
        const std::vector<int>& tokens,
        int max_tokens = 256,
        float temperature = 0.7f,
        float top_p = 0.9f,
        int top_k = 40
    );

    /**
     * Generate text from prompt.
     */
    std::string generate_text(
        const std::string& prompt,
        int max_tokens = 256,
        float temperature = 0.7f,
        float top_p = 0.9f,
        int top_k = 40
    );

    /**
     * Generate text using chat template (LLaMA 3 Instruct format).
     * @param user_message User's message
     * @param system_prompt Optional system prompt
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @param top_p Top-p (nucleus) sampling
     * @param top_k Top-k sampling (0 to disable)
     * @return Generated assistant response
     */
    std::string chat(
        const std::string& user_message,
        const std::string& system_prompt = "",
        int max_tokens = 256,
        float temperature = 0.7f,
        float top_p = 0.9f,
        int top_k = 40
    );

    /**
     * Format a prompt using the chat template.
     * @param user_message User's message
     * @param system_prompt Optional system prompt
     * @return Formatted prompt string
     */
    std::string format_chat_prompt(
        const std::string& user_message,
        const std::string& system_prompt = ""
    ) const;

    /**
     * Check if model is loaded.
     */
    bool loaded() const { return loaded_; }

    /**
     * Get model configuration.
     */
    const LLMConfig& config() const { return config_; }

    /**
     * Get model info string.
     */
    std::string info() const;

private:
    LLMConfig config_;
    Weights weights_;
    std::unique_ptr<LLMTokenizer> tokenizer_;
    bool loaded_ = false;

    // Rope frequency cache (initialized with placeholder)
    mx::array rope_freqs_{mx::zeros({1})};

    // Internal methods
    void init_rope();
    mx::array apply_rope(const mx::array& x, int offset);
    mx::array decoder_layer(const mx::array& x, int layer_idx, KVCache* cache);
    mx::array self_attention(const mx::array& x, int layer_idx, KVCache* cache);
    mx::array mlp(const mx::array& x, int layer_idx);
    mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps);
    mx::array sample(const mx::array& logits, float temperature, float top_p, int top_k);
};

} // namespace llm
