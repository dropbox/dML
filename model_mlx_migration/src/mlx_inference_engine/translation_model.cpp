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
// T5/MADLAD encoder-decoder transformer for MLX

#include "translation_model.h"
#include "mlx/mlx.h"
#include "mlx/io.h"

// SentencePiece for tokenization
#include <sentencepiece_processor.h>

#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <optional>

// JSON parsing (minimal implementation)
#include <regex>

namespace mx = mlx::core;

// ============================================================================
// Helper functions
// ============================================================================

namespace {

// GELU activation - match mlx.nn.gelu (erf-based)
// Formula: x * (1 + erf(x / sqrt(2))) / 2
mx::array gelu(const mx::array& x) {
    auto sqrt2 = std::sqrt(2.0f);
    return 0.5f * x * (1.0f + mx::erf(x / sqrt2));
}

}  // anonymous namespace

namespace translation {

// ============================================================================
// JSON parsing helpers
// ============================================================================

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int parse_int(const std::string& json, const std::string& key, int default_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stoi(match[1]);
    }
    return default_val;
}

float parse_float(const std::string& json, const std::string& key, float default_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9.e+-]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stof(match[1]);
    }
    return default_val;
}

std::string parse_string(const std::string& json, const std::string& key, const std::string& default_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1];
    }
    return default_val;
}

bool parse_bool(const std::string& json, const std::string& key, bool default_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1] == "true";
    }
    return default_val;
}

}  // anonymous namespace

// ============================================================================
// T5Config
// ============================================================================

T5Config T5Config::load(const std::string& path) {
    std::string json = read_file(path);

    T5Config config;
    config.d_model = parse_int(json, "d_model", 1024);
    config.d_ff = parse_int(json, "d_ff", 8192);
    config.d_kv = parse_int(json, "d_kv", 128);
    config.num_heads = parse_int(json, "num_heads", 16);
    config.num_layers = parse_int(json, "num_layers", 32);
    config.num_decoder_layers = parse_int(json, "num_decoder_layers", config.num_layers);
    config.vocab_size = parse_int(json, "vocab_size", 256000);
    config.relative_attention_num_buckets = parse_int(json, "relative_attention_num_buckets", 32);
    config.relative_attention_max_distance = parse_int(json, "relative_attention_max_distance", 128);
    config.feed_forward_proj = parse_string(json, "feed_forward_proj", "gated-gelu");
    config.decoder_start_token_id = parse_int(json, "decoder_start_token_id", 0);
    config.pad_token_id = parse_int(json, "pad_token_id", 1);
    config.eos_token_id = parse_int(json, "eos_token_id", 2);
    config.layer_norm_epsilon = parse_float(json, "layer_norm_epsilon", 1e-6f);
    config.tie_word_embeddings = parse_bool(json, "tie_word_embeddings", false);

    return config;
}

// ============================================================================
// Weights
// ============================================================================

void Weights::load(const std::string& path) {
    auto [arrays, metadata] = mx::load_safetensors(path);
    for (auto& kv : arrays) {
        weights_.insert_or_assign(kv.first, kv.second);
    }
}

mx::array Weights::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

bool Weights::has(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

// ============================================================================
// T5Model - Loading
// ============================================================================

T5Model T5Model::load(const std::string& model_path) {
    T5Model model;

    // Load config
    std::string config_path = model_path + "/config.json";
    model.config_ = T5Config::load(config_path);

    // Load weights - search for common weight file names
    std::vector<std::string> weight_names = {
        "model.safetensors",
        "weights.safetensors",
        "pytorch_model.bin"
    };
    std::string weights_path;
    for (const auto& name : weight_names) {
        std::string candidate = model_path + "/" + name;
        std::ifstream f(candidate);
        if (f.good()) {
            weights_path = candidate;
            break;
        }
    }
    if (weights_path.empty()) {
        throw std::runtime_error("No weights file found in " + model_path +
            ". Expected model.safetensors or weights.safetensors");
    }
    model.weights_.load(weights_path);

    model.loaded_ = true;
    return model;
}

std::string T5Model::info() const {
    std::ostringstream oss;
    oss << "T5 Translation Model\n";
    oss << "  d_model: " << config_.d_model << "\n";
    oss << "  d_ff: " << config_.d_ff << "\n";
    oss << "  num_heads: " << config_.num_heads << "\n";
    oss << "  num_layers: " << config_.num_layers << "\n";
    oss << "  vocab_size: " << config_.vocab_size << "\n";
    oss << "  weights: " << weights_.size() << "\n";
    return oss.str();
}

// ============================================================================
// T5Model - Components
// ============================================================================

mx::array T5Model::embed_tokens(const mx::array& input_ids, bool is_decoder) {
    // MADLAD uses decoder.embed_tokens.weight for both encoder and decoder
    // (shared embeddings stored under decoder prefix)
    // Fall back to shared.weight for standard T5 models
    std::string embed_name;
    if (weights_.has("decoder.embed_tokens.weight")) {
        embed_name = "decoder.embed_tokens.weight";
    } else if (weights_.has("shared.weight")) {
        embed_name = "shared.weight";
    } else {
        std::string prefix = is_decoder ? "decoder" : "encoder";
        embed_name = prefix + ".embed_tokens.weight";
    }
    return mx::take(weights_.get(embed_name), input_ids, 0);
}

mx::array T5Model::layer_norm(const mx::array& x, const std::string& prefix) {
    auto weight = weights_.get(prefix + ".weight");
    float eps = config_.layer_norm_epsilon;

    // RMSNorm used in T5
    auto variance = mx::mean(mx::square(x), -1, true);
    auto normalized = x * mx::rsqrt(variance + eps);
    return normalized * weight;
}

mx::array T5Model::compute_relative_position_bias(int query_len, int key_len, bool is_decoder, int query_offset) {
    // T5 relative position bias computation
    int num_buckets = config_.relative_attention_num_buckets;
    int max_distance = config_.relative_attention_max_distance;

    // Create position indices
    // For incremental decoding, query positions need to be offset by cache length
    // so that position i in the query corresponds to position (i + query_offset) in the full sequence
    auto query_pos = mx::arange(query_len) + query_offset;
    auto key_pos = mx::arange(key_len);

    // Compute relative positions: key_pos - query_pos (HuggingFace T5 convention)
    // For position i attending to position j, relative = j - i
    // Positive means key is AFTER query, negative means key is BEFORE query
    auto relative_position = mx::expand_dims(key_pos, 0) - mx::expand_dims(query_pos, 1);

    // Bucket relative positions
    // For bidirectional (encoder): use both positive and negative buckets
    // For unidirectional (decoder): only keep backward-looking (key before query)

    auto relative_buckets = mx::zeros({query_len, key_len}, mx::int32);

    if (!is_decoder) {
        // Bidirectional: split buckets between positive and negative positions
        num_buckets /= 2;
        // HuggingFace T5: add bucket offset for POSITIVE positions (key AFTER query)
        auto pos_mask = relative_position > 0;
        relative_position = mx::abs(relative_position);

        // Offset for positive positions (key is AFTER query in sequence)
        relative_buckets = relative_buckets + mx::where(pos_mask,
            mx::full({query_len, key_len}, num_buckets, mx::int32),
            mx::zeros({query_len, key_len}, mx::int32));
    } else {
        // Unidirectional (decoder): clamp forward-looking positions to 0
        // relative_position = -min(relative_position, 0) per HuggingFace T5
        // This keeps backward distances positive and zeros out forward positions
        auto clamped = mx::minimum(relative_position, mx::zeros_like(relative_position));
        relative_position = mx::negative(clamped);
    }

    // Small positions: linear
    int max_exact = num_buckets / 2;
    auto is_small = relative_position < max_exact;

    // Large positions: logarithmic
    auto relative_position_float = mx::astype(relative_position, mx::float32);
    auto relative_position_large = mx::astype(
        mx::floor(
            mx::astype(mx::full({1}, (float)max_exact, mx::float32), mx::float32) +
            mx::log(relative_position_float / max_exact) /
            std::log((float)max_distance / max_exact) *
            (num_buckets - max_exact)
        ),
        mx::int32
    );
    relative_position_large = mx::minimum(
        relative_position_large,
        mx::full({query_len, key_len}, num_buckets - 1, mx::int32)
    );

    relative_buckets = relative_buckets + mx::where(
        is_small,
        mx::astype(relative_position, mx::int32),
        relative_position_large
    );

    // Get bias values from weights
    std::string bias_name = is_decoder ?
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" :
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight";

    auto bias_weight = weights_.get(bias_name);  // [num_buckets, num_heads]

    // Gather biases for each position pair
    auto bias = mx::take(bias_weight, mx::flatten(relative_buckets), 0);
    bias = mx::reshape(bias, {query_len, key_len, config_.num_heads});
    bias = mx::transpose(bias, {2, 0, 1});  // [num_heads, query_len, key_len]

    return mx::expand_dims(bias, 0);  // [1, num_heads, query_len, key_len]
}

mx::array T5Model::self_attention(
    const mx::array& hidden,
    const mx::array& position_bias,
    int layer_idx,
    bool is_decoder,
    KVCache* cache
) {
    std::string prefix = is_decoder ? "decoder" : "encoder";
    std::string layer_prefix = prefix + ".block." + std::to_string(layer_idx) + ".layer.0.SelfAttention";

    int batch_size = hidden.shape()[0];
    int seq_len = hidden.shape()[1];

    // Get weights
    auto q_weight = weights_.get(layer_prefix + ".q.weight");
    auto k_weight = weights_.get(layer_prefix + ".k.weight");
    auto v_weight = weights_.get(layer_prefix + ".v.weight");
    auto o_weight = weights_.get(layer_prefix + ".o.weight");

    // Project Q, K, V
    auto q = mx::matmul(hidden, mx::transpose(q_weight));
    auto k = mx::matmul(hidden, mx::transpose(k_weight));
    auto v = mx::matmul(hidden, mx::transpose(v_weight));

    // Reshape for multi-head attention
    int head_dim = config_.d_kv;
    int num_heads = config_.num_heads;

    q = mx::reshape(q, {batch_size, seq_len, num_heads, head_dim});
    k = mx::reshape(k, {batch_size, seq_len, num_heads, head_dim});
    v = mx::reshape(v, {batch_size, seq_len, num_heads, head_dim});

    // Transpose to [batch, heads, seq, head_dim]
    q = mx::transpose(q, {0, 2, 1, 3});
    k = mx::transpose(k, {0, 2, 1, 3});
    v = mx::transpose(v, {0, 2, 1, 3});

    // Handle KV cache for decoder
    if (is_decoder && cache != nullptr) {
        if (cache->keys.size() > (size_t)layer_idx) {
            // Append to existing cache
            k = mx::concatenate({cache->keys[layer_idx], k}, 2);
            v = mx::concatenate({cache->values[layer_idx], v}, 2);
            // Update cache in place
            cache->keys[layer_idx] = k;
            cache->values[layer_idx] = v;
        } else {
            // Extend cache by pushing new entries
            while (cache->keys.size() < (size_t)layer_idx) {
                // Fill gap with dummy arrays (should not happen in proper usage)
                cache->keys.push_back(mx::zeros({1}, mx::float32));
                cache->values.push_back(mx::zeros({1}, mx::float32));
            }
            cache->keys.push_back(k);
            cache->values.push_back(v);
        }
    }

    // Attention scores (T5 does NOT scale by 1/sqrt(d_k) - it's folded into initialization)
    auto scores = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2}));

    // Add position bias
    scores = scores + position_bias;

    // Causal mask for decoder self-attention (only needed for full sequence, not incremental)
    // For incremental decoding (q_len=1), the current token can attend to all previous tokens
    if (is_decoder) {
        int kv_len = k.shape()[2];
        int q_len = q.shape()[2];
        // Only apply causal mask when processing full sequence (q_len > 1)
        // For incremental decoding (q_len=1), current position can see all previous positions
        if (q_len > 1) {
            // Standard causal mask: position i can only attend to positions 0..i
            auto causal_mask = mx::triu(
                mx::full({q_len, kv_len}, -1e9f, mx::float32),
                1  // offset=1 means mask positions where key_pos > query_pos
            );
            scores = scores + causal_mask;
        }
    }

    // Softmax and attention
    auto attn_weights = mx::softmax(scores, -1);
    auto attn_output = mx::matmul(attn_weights, v);

    // Reshape back
    attn_output = mx::transpose(attn_output, {0, 2, 1, 3});
    attn_output = mx::reshape(attn_output, {batch_size, -1, num_heads * head_dim});

    // Output projection
    return mx::matmul(attn_output, mx::transpose(o_weight));
}

mx::array T5Model::cross_attention(
    const mx::array& hidden,
    const mx::array& encoder_output,
    int layer_idx,
    CrossKVCache* cross_cache
) {
    std::string layer_prefix = "decoder.block." + std::to_string(layer_idx) + ".layer.1.EncDecAttention";

    int batch_size = hidden.shape()[0];
    int dec_len = hidden.shape()[1];

    // Get weights
    auto q_weight = weights_.get(layer_prefix + ".q.weight");
    auto o_weight = weights_.get(layer_prefix + ".o.weight");

    // Project Q from decoder
    auto q = mx::matmul(hidden, mx::transpose(q_weight));

    int head_dim = config_.d_kv;
    int num_heads = config_.num_heads;

    // Check if K/V are cached
    // Note: size() > layer_idx means we have at least layer_idx+1 elements,
    // so element at index layer_idx exists
    mx::array k = mx::zeros({1}, mx::float32);  // Placeholder, will be replaced
    mx::array v = mx::zeros({1}, mx::float32);
    if (cross_cache != nullptr && cross_cache->keys.size() > (size_t)layer_idx) {
        // Use cached K/V projections
        k = cross_cache->keys[layer_idx];
        v = cross_cache->values[layer_idx];
    } else {
        // Compute K/V projections from encoder output
        auto k_weight = weights_.get(layer_prefix + ".k.weight");
        auto v_weight = weights_.get(layer_prefix + ".v.weight");

        int enc_len = encoder_output.shape()[1];

        auto k_proj = mx::matmul(encoder_output, mx::transpose(k_weight));
        auto v_proj = mx::matmul(encoder_output, mx::transpose(v_weight));

        // Reshape to [batch, seq, heads, head_dim]
        k_proj = mx::reshape(k_proj, {batch_size, enc_len, num_heads, head_dim});
        v_proj = mx::reshape(v_proj, {batch_size, enc_len, num_heads, head_dim});

        // Transpose to [batch, heads, seq, head_dim]
        k = mx::transpose(k_proj, {0, 2, 1, 3});
        v = mx::transpose(v_proj, {0, 2, 1, 3});

        // Cache if cache is provided
        // Note: We don't eval here - let MLX batch the computation
        if (cross_cache != nullptr) {
            while (cross_cache->keys.size() < (size_t)layer_idx) {
                cross_cache->keys.push_back(mx::zeros({1}, mx::float32));
                cross_cache->values.push_back(mx::zeros({1}, mx::float32));
            }
            cross_cache->keys.push_back(k);
            cross_cache->values.push_back(v);
        }
    }

    // Reshape Q for multi-head attention
    q = mx::reshape(q, {batch_size, dec_len, num_heads, head_dim});
    q = mx::transpose(q, {0, 2, 1, 3});

    // Attention scores (T5 does NOT scale by 1/sqrt(d_k) - it's folded into initialization)
    auto scores = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2}));

    // Softmax and attention (no causal mask for cross-attention)
    auto attn_weights = mx::softmax(scores, -1);
    auto attn_output = mx::matmul(attn_weights, v);

    // Reshape back
    attn_output = mx::transpose(attn_output, {0, 2, 1, 3});
    attn_output = mx::reshape(attn_output, {batch_size, -1, num_heads * head_dim});

    // Output projection
    return mx::matmul(attn_output, mx::transpose(o_weight));
}

mx::array T5Model::feed_forward(
    const mx::array& hidden,
    int layer_idx,
    bool is_decoder
) {
    std::string prefix = is_decoder ? "decoder" : "encoder";
    std::string layer_prefix = prefix + ".block." + std::to_string(layer_idx) + ".layer." +
        (is_decoder ? "2" : "1") + ".DenseReluDense";

    // Gated-GELU FFN: GELU(hidden @ wi_0) * (hidden @ wi_1), then @ wo
    auto wi_0 = weights_.get(layer_prefix + ".wi_0.weight");
    auto wi_1 = weights_.get(layer_prefix + ".wi_1.weight");
    auto wo = weights_.get(layer_prefix + ".wo.weight");

    auto gate = mx::matmul(hidden, mx::transpose(wi_0));
    auto hidden_gelu = gelu(gate);

    auto hidden_linear = mx::matmul(hidden, mx::transpose(wi_1));
    auto hidden_act = hidden_gelu * hidden_linear;

    return mx::matmul(hidden_act, mx::transpose(wo));
}

mx::array T5Model::encoder_layer(
    const mx::array& hidden,
    const mx::array& position_bias,
    int layer_idx
) {
    std::string prefix = "encoder.block." + std::to_string(layer_idx);

    // Self-attention with residual (Pre-LN architecture)
    auto ln_hidden = layer_norm(hidden, prefix + ".layer.0.layer_norm");
    auto attn_output = self_attention(ln_hidden, position_bias, layer_idx, false, nullptr);
    auto residual = hidden + attn_output;

    // FFN with residual
    ln_hidden = layer_norm(residual, prefix + ".layer.1.layer_norm");
    auto ffn_output = feed_forward(ln_hidden, layer_idx, false);
    return residual + ffn_output;
}

mx::array T5Model::decoder_layer(
    const mx::array& hidden,
    const mx::array& encoder_output,
    const mx::array& self_attn_bias,
    int layer_idx,
    KVCache* cache,
    CrossKVCache* cross_cache
) {
    std::string prefix = "decoder.block." + std::to_string(layer_idx);

    // Self-attention with residual
    auto ln_hidden = layer_norm(hidden, prefix + ".layer.0.layer_norm");
    auto self_attn_output = self_attention(ln_hidden, self_attn_bias, layer_idx, true, cache);
    auto residual = hidden + self_attn_output;

    // Cross-attention with residual (uses cached K/V if available)
    ln_hidden = layer_norm(residual, prefix + ".layer.1.layer_norm");
    auto cross_attn_output = cross_attention(ln_hidden, encoder_output, layer_idx, cross_cache);
    residual = residual + cross_attn_output;

    // FFN with residual
    ln_hidden = layer_norm(residual, prefix + ".layer.2.layer_norm");
    auto ffn_output = feed_forward(ln_hidden, layer_idx, true);
    return residual + ffn_output;
}

// ============================================================================
// T5Model - Forward Pass
// ============================================================================

mx::array T5Model::encode(const mx::array& input_ids) {
    // Embed input tokens
    auto hidden = embed_tokens(input_ids, false);

    // Compute position bias (only on first layer, shared across all)
    int seq_len = input_ids.shape()[1];
    auto position_bias = compute_relative_position_bias(seq_len, seq_len, false);

    // Apply encoder layers
    for (int i = 0; i < config_.num_layers; i++) {
        hidden = encoder_layer(hidden, position_bias, i);
    }

    // Final layer norm
    hidden = layer_norm(hidden, "encoder.final_layer_norm");

    return hidden;
}

mx::array T5Model::decode(
    const mx::array& decoder_input_ids,
    const mx::array& encoder_output,
    KVCache* cache,
    CrossKVCache* cross_cache
) {
    // Embed decoder tokens
    auto hidden = embed_tokens(decoder_input_ids, true);

    // Compute self-attention position bias
    // Following HuggingFace T5: compute full bias matrix and slice for incremental decoding
    int dec_len = decoder_input_ids.shape()[1];
    int cache_len = (cache && !cache->empty()) ? cache->keys[0].shape()[2] : 0;
    int total_len = dec_len + cache_len;

    // Compute full position bias for entire sequence
    auto full_bias = compute_relative_position_bias(total_len, total_len, true, 0);

    // Slice to get only the bias for current query positions (last dec_len positions)
    // full_bias shape: [1, num_heads, total_len, total_len]
    // We want: [1, num_heads, dec_len, total_len] (last dec_len query positions, all key positions)
    auto self_attn_bias = mx::slice(full_bias, {0, 0, cache_len, 0}, {1, config_.num_heads, total_len, total_len});

    // Apply decoder layers (cross_cache is populated on first call and reused)
    for (int i = 0; i < config_.num_decoder_layers; i++) {
        hidden = decoder_layer(hidden, encoder_output, self_attn_bias, i, cache, cross_cache);
    }

    // Final layer norm
    hidden = layer_norm(hidden, "decoder.final_layer_norm");

    // Project to vocab
    auto lm_head = weights_.get("lm_head.weight");
    auto logits = mx::matmul(hidden, mx::transpose(lm_head));

    return logits;
}

mx::array T5Model::generate(
    const mx::array& input_ids,
    int max_length
) {
    // Encode source
    auto encoder_output = encode(input_ids);

    // Initialize decoder with start token
    int batch_size = input_ids.shape()[0];
    auto decoder_ids = mx::full({batch_size, 1}, config_.decoder_start_token_id, mx::int32);

    // Self-attention KV cache for efficient decoding
    KVCache cache;

    // Note: Cross-attention KV cache disabled for now (causing performance regression)
    // The self-attention KV cache provides the main speedup benefit

    // Greedy decoding loop
    bool use_cache = true;  // Enable KV cache for efficient decoding

    std::vector<mx::array> generated_ids;
    generated_ids.push_back(decoder_ids);

    for (int step = 0; step < max_length; step++) {
        // Get logits
        mx::array logits = mx::zeros({1}, mx::float32);  // Placeholder init
        if (use_cache) {
            // Incremental decoding: only decode new token with KV cache
            logits = decode(decoder_ids, encoder_output, &cache, nullptr);
        } else {
            // Full sequence decoding: decode entire sequence each step (no cache)
            auto full_decoder_ids = mx::concatenate(generated_ids, 1);
            logits = decode(full_decoder_ids, encoder_output, nullptr, nullptr);
        }

        // Get last token's logits (MLX C++ doesn't support Python-style -1 indexing)
        int seq_len_out = logits.shape()[1];
        int vocab = logits.shape()[2];
        auto next_token_logits = mx::slice(logits, {0, seq_len_out - 1, 0}, {batch_size, seq_len_out, vocab});

        // Greedy: take argmax over vocab dimension
        auto next_token = mx::argmax(next_token_logits, -1);
        next_token = mx::reshape(next_token, {batch_size, 1});
        next_token = mx::astype(next_token, mx::int32);

        generated_ids.push_back(next_token);
        decoder_ids = next_token;

        // Check for EOS
        mx::eval(next_token);
        auto next_data = next_token.data<int32_t>();
        bool all_eos = true;
        for (int i = 0; i < batch_size; i++) {
            if (next_data[i] != config_.eos_token_id) {
                all_eos = false;
                break;
            }
        }
        if (all_eos) break;
    }

    // Concatenate all generated tokens
    return mx::concatenate(generated_ids, 1);
}

// ============================================================================
// Tokenizer - SentencePiece implementation
// ============================================================================

struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor processor;
    std::string model_path;
};

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer Tokenizer::load(const std::string& spiece_path) {
    Tokenizer tok;
    tok.impl_ = std::make_unique<Impl>();

    auto status = tok.impl_->processor.Load(spiece_path);
    if (!status.ok()) {
        throw std::runtime_error(
            "Failed to load SentencePiece model: " + spiece_path +
            " - " + status.ToString()
        );
    }

    tok.impl_->model_path = spiece_path;
    tok.loaded_ = true;
    return tok;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_eos) const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer not loaded");
    }

    std::vector<int> ids;
    auto status = impl_->processor.Encode(text, &ids);
    if (!status.ok()) {
        throw std::runtime_error("Failed to encode: " + status.ToString());
    }

    if (add_eos) {
        ids.push_back(eos_token_id_);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer not loaded");
    }

    std::string text;
    auto status = impl_->processor.Decode(ids, &text);
    if (!status.ok()) {
        throw std::runtime_error("Failed to decode: " + status.ToString());
    }
    return text;
}

int Tokenizer::vocab_size() const {
    if (!loaded_) {
        return 256000;  // MADLAD default
    }
    return impl_->processor.GetPieceSize();
}

// ============================================================================
// TranslationModel
// ============================================================================

TranslationModel TranslationModel::load(
    const std::string& model_path,
    const std::string& model_type
) {
    TranslationModel tm;
    tm.model_ = T5Model::load(model_path);
    tm.model_type_ = model_type;

    // Load tokenizer - search for common SentencePiece model names
    std::vector<std::string> tokenizer_names = {
        "spiece.model",
        "sentencepiece.bpe.model",
        "sentencepiece.model",
        "tokenizer.model"
    };
    std::string tokenizer_path;
    for (const auto& name : tokenizer_names) {
        std::string candidate = model_path + "/" + name;
        std::ifstream f(candidate);
        if (f.good()) {
            tokenizer_path = candidate;
            break;
        }
    }
    if (tokenizer_path.empty()) {
        throw std::runtime_error("No tokenizer file found in " + model_path +
            ". Expected spiece.model or sentencepiece.bpe.model");
    }
    tm.tokenizer_ = Tokenizer::load(tokenizer_path);

    return tm;
}

std::string TranslationModel::format_input(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang
) const {
    if (model_type_ == "madlad") {
        // MADLAD format: <2xx>text where xx is target language code
        // source_lang not used for MADLAD (target-only tagging)
        (void)source_lang;
        return "<2" + target_lang + ">" + text;
    } else if (model_type_ == "nllb") {
        // NLLB format: source language embedded in tokenization
        // TODO: Implement NLLB language code prefix
        (void)source_lang;
        return text;
    } else if (model_type_ == "opus-mt") {
        // OPUS-MT format: just the text (language pair determined by model)
        (void)source_lang;
        return text;
    }
    (void)source_lang;
    return text;
}

std::string TranslationModel::translate(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang,
    int max_length,
    bool debug
) const {
    if (!loaded()) {
        throw std::runtime_error("Model or tokenizer not loaded");
    }

    // Format input with language tags
    std::string formatted = format_input(text, source_lang, target_lang);

    // Encode to token IDs
    std::vector<int> input_ids = tokenizer_.encode(formatted, true);

    if (debug) {
        std::cout << "\n[DEBUG] Formatted input: '" << formatted << "'\n";
        std::cout << "[DEBUG] Input token IDs (" << input_ids.size() << "): ";
        for (size_t i = 0; i < std::min(input_ids.size(), (size_t)20); i++) {
            std::cout << input_ids[i] << " ";
        }
        if (input_ids.size() > 20) std::cout << "...";
        std::cout << "\n";
    }

    // Create input array
    mx::array input_arr = mx::array(input_ids.data(), {1, (int)input_ids.size()}, mx::int32);

    // Generate output tokens
    mx::array output_arr = const_cast<T5Model&>(model_).generate(input_arr, max_length);
    mx::eval(output_arr);

    // Extract output token IDs
    auto output_data = output_arr.data<int32_t>();
    int output_len = output_arr.shape()[1];

    if (debug) {
        std::cout << "[DEBUG] Generated token IDs (" << output_len << "): ";
        for (int i = 0; i < std::min(output_len, 30); i++) {
            std::cout << output_data[i] << " ";
        }
        if (output_len > 30) std::cout << "...";
        std::cout << "\n";
    }

    // Filter out special tokens and decode
    std::vector<int> output_ids;
    output_ids.reserve(output_len);
    for (int i = 0; i < output_len; i++) {
        int32_t id = output_data[i];
        // Skip PAD (1), decoder start (0)
        if (id == tokenizer_.pad_token_id() ||
            id == tokenizer_.decoder_start_token_id()) {
            continue;
        }
        // Stop at EOS (2)
        if (id == tokenizer_.eos_token_id()) {
            break;
        }
        output_ids.push_back(id);
    }

    if (debug) {
        std::cout << "[DEBUG] Filtered output IDs: ";
        for (int id : output_ids) {
            std::cout << id << " ";
        }
        std::cout << "\n";
    }

    return tokenizer_.decode(output_ids);
}

std::string TranslationModel::info() const {
    return model_.info();
}

}  // namespace translation
