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

// Translation Model C++ Implementation
// T5/MADLAD/NLLB encoder-decoder transformer for MLX
//
// Based on T5ForConditionalGeneration architecture:
// - Encoder: 32 layers of self-attention + FFN
// - Decoder: 32 layers of self-attention + cross-attention + FFN
// - Relative position bias instead of absolute position embeddings
// - Gated-GELU FFN (wi_0, wi_1, wo)

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace translation {

/**
 * Configuration for T5-style translation model.
 * Compatible with MADLAD-400, NLLB-200, and similar models.
 */
struct T5Config {
    // Model dimensions
    int d_model = 1024;        // Hidden size
    int d_ff = 8192;           // Feed-forward intermediate size
    int d_kv = 128;            // Key/value projection size
    int num_heads = 16;        // Attention heads
    int num_layers = 32;       // Encoder layers
    int num_decoder_layers = 32; // Decoder layers
    int vocab_size = 256000;   // Vocabulary size

    // Attention config
    int relative_attention_num_buckets = 32;
    int relative_attention_max_distance = 128;

    // FFN config
    std::string feed_forward_proj = "gated-gelu";  // "gated-gelu" or "relu"

    // Special tokens
    int decoder_start_token_id = 0;
    int pad_token_id = 1;
    int eos_token_id = 2;

    // Precision
    float layer_norm_epsilon = 1e-6f;
    bool tie_word_embeddings = false;

    // Load from JSON file
    static T5Config load(const std::string& path);
};

/**
 * Weight storage for translation model.
 * Maps weight names to MLX arrays.
 */
class Weights {
public:
    Weights() = default;

    // Load weights from safetensors file
    void load(const std::string& path);

    // Get a weight by name (throws if not found)
    mx::array get(const std::string& name) const;

    // Check if weight exists
    bool has(const std::string& name) const;

    // Number of weights
    size_t size() const { return weights_.size(); }

private:
    std::unordered_map<std::string, mx::array> weights_;
};

/**
 * KV Cache for efficient autoregressive decoding (self-attention).
 */
struct KVCache {
    std::vector<mx::array> keys;    // [num_layers] each [batch, heads, seq, d_kv]
    std::vector<mx::array> values;  // [num_layers] each [batch, heads, seq, d_kv]

    void clear() {
        keys.clear();
        values.clear();
    }

    bool empty() const { return keys.empty(); }
};

/**
 * Cross-attention KV Cache for encoder output projections.
 * Caches K/V projections of encoder output, computed once per translation.
 * This avoids recomputing O(encoder_len * d_model) projections each decode step.
 */
struct CrossKVCache {
    std::vector<mx::array> keys;    // [num_layers] each [batch, heads, enc_seq, d_kv]
    std::vector<mx::array> values;  // [num_layers] each [batch, heads, enc_seq, d_kv]

    void clear() {
        keys.clear();
        values.clear();
    }

    bool empty() const { return keys.empty(); }
};

/**
 * T5 Translation Model.
 *
 * Implements T5ForConditionalGeneration architecture for translation.
 * Thread-safe for inference (multiple threads can call translate() concurrently).
 */
class T5Model {
public:
    T5Model() = default;

    /**
     * Load model from directory.
     * Expected structure:
     *   model.safetensors  - All model weights
     *   config.json        - Model configuration
     *   spiece.model       - SentencePiece tokenizer
     *
     * @param model_path Path to model directory
     * @return Loaded model
     * @throws std::runtime_error if loading fails
     */
    static T5Model load(const std::string& model_path);

    /**
     * Get model configuration.
     */
    const T5Config& config() const { return config_; }

    /**
     * Check if model is loaded.
     */
    bool loaded() const { return loaded_; }

    /**
     * Encode input sequence.
     * @param input_ids Token IDs [batch, seq_len]
     * @return Encoder hidden states [batch, seq_len, d_model]
     */
    mx::array encode(const mx::array& input_ids);

    /**
     * Decode with cross-attention to encoder output.
     * @param decoder_input_ids Decoder input token IDs [batch, seq_len]
     * @param encoder_output Encoder hidden states [batch, src_len, d_model]
     * @param cache Optional KV cache for efficient autoregressive decoding
     * @param cross_cache Optional cross-attention KV cache for encoder projections
     * @return Decoder logits [batch, seq_len, vocab_size]
     */
    mx::array decode(
        const mx::array& decoder_input_ids,
        const mx::array& encoder_output,
        KVCache* cache = nullptr,
        CrossKVCache* cross_cache = nullptr
    );

    /**
     * Full translation pipeline with greedy decoding.
     * @param input_ids Source token IDs [batch, seq_len]
     * @param max_length Maximum output length
     * @return Generated token IDs [batch, out_len]
     */
    mx::array generate(
        const mx::array& input_ids,
        int max_length = 256
    );

    /**
     * Get model info for debugging.
     */
    std::string info() const;

private:
    // =========================================================================
    // Internal components
    // =========================================================================

    // Embedding lookup
    mx::array embed_tokens(const mx::array& input_ids, bool is_decoder);

    // Relative position bias
    // query_offset: for incremental decoding, the position of the first query token
    mx::array compute_relative_position_bias(int query_len, int key_len, bool is_decoder, int query_offset = 0);

    // Layer normalization
    mx::array layer_norm(const mx::array& x, const std::string& prefix);

    // Self-attention layer
    mx::array self_attention(
        const mx::array& hidden,
        const mx::array& position_bias,
        int layer_idx,
        bool is_decoder,
        KVCache* cache = nullptr
    );

    // Cross-attention layer (decoder only)
    // Uses cached K/V projections if cross_cache is provided
    mx::array cross_attention(
        const mx::array& hidden,
        const mx::array& encoder_output,
        int layer_idx,
        CrossKVCache* cross_cache = nullptr
    );

    // Feed-forward network (gated-GELU)
    mx::array feed_forward(
        const mx::array& hidden,
        int layer_idx,
        bool is_decoder
    );

    // Single encoder layer
    mx::array encoder_layer(
        const mx::array& hidden,
        const mx::array& position_bias,
        int layer_idx
    );

    // Single decoder layer
    mx::array decoder_layer(
        const mx::array& hidden,
        const mx::array& encoder_output,
        const mx::array& self_attn_bias,
        int layer_idx,
        KVCache* cache = nullptr,
        CrossKVCache* cross_cache = nullptr
    );

    T5Config config_;
    Weights weights_;
    bool loaded_ = false;
};

/**
 * Tokenizer for translation models.
 * Wraps SentencePiece for encoding/decoding text.
 */
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Move semantics
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // Disable copy (SentencePiece processor is not easily copyable)
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    /**
     * Load tokenizer from SentencePiece model file.
     */
    static Tokenizer load(const std::string& spiece_path);

    /**
     * Encode text to token IDs.
     * @param text Input text (UTF-8)
     * @param add_eos Whether to append EOS token
     * @return Token IDs
     */
    std::vector<int> encode(const std::string& text, bool add_eos = true) const;

    /**
     * Decode token IDs to text.
     * @param ids Token IDs
     * @return Decoded text (UTF-8)
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
    int pad_token_id() const { return pad_token_id_; }
    int eos_token_id() const { return eos_token_id_; }
    int decoder_start_token_id() const { return decoder_start_token_id_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;
    int pad_token_id_ = 1;
    int eos_token_id_ = 2;
    int decoder_start_token_id_ = 0;
};

/**
 * High-level translation interface.
 * Combines model and tokenizer for easy text-to-text translation.
 */
class TranslationModel {
public:
    TranslationModel() = default;

    /**
     * Load translation model.
     * @param model_path Path to model directory
     * @param model_type "madlad", "nllb", or "opus-mt"
     */
    static TranslationModel load(
        const std::string& model_path,
        const std::string& model_type = "madlad"
    );

    /**
     * Translate text.
     * @param text Source text
     * @param source_lang Source language code (e.g., "en")
     * @param target_lang Target language code (e.g., "de")
     * @param max_length Maximum output length
     * @param debug Print debug output during generation
     * @return Translated text
     */
    std::string translate(
        const std::string& text,
        const std::string& source_lang,
        const std::string& target_lang,
        int max_length = 256,
        bool debug = false
    ) const;

    /**
     * Get model info.
     */
    std::string info() const;

    /**
     * Check if loaded.
     */
    bool loaded() const { return model_.loaded() && tokenizer_.loaded(); }

private:
    T5Model model_;
    Tokenizer tokenizer_;
    std::string model_type_;

    // Format source text with language tag (model-specific)
    std::string format_input(
        const std::string& text,
        const std::string& source_lang,
        const std::string& target_lang
    ) const;
};

}  // namespace translation
