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

#include "llm_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"

namespace llm {

// ============================================================================
// Utility functions
// ============================================================================

namespace {

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Simple JSON value extractor
std::string get_json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"";
    std::regex re(pattern);
    std::smatch match;
    if (std::regex_search(json, match, re)) {
        return match[1];
    }
    return "";
}

int get_json_int(const std::string& json, const std::string& key, int default_val = 0) {
    std::string pattern = "\"" + key + "\"\\s*:\\s*(-?\\d+)";
    std::regex re(pattern);
    std::smatch match;
    if (std::regex_search(json, match, re)) {
        return std::stoi(match[1]);
    }
    return default_val;
}

float get_json_float(const std::string& json, const std::string& key, float default_val = 0.0f) {
    std::string pattern = "\"" + key + "\"\\s*:\\s*(-?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)";
    std::regex re(pattern);
    std::smatch match;
    if (std::regex_search(json, match, re)) {
        return std::stof(match[1]);
    }
    return default_val;
}

bool get_json_bool(const std::string& json, const std::string& key, bool default_val = false) {
    std::string pattern = "\"" + key + "\"\\s*:\\s*(true|false)";
    std::regex re(pattern);
    std::smatch match;
    if (std::regex_search(json, match, re)) {
        return match[1] == "true";
    }
    return default_val;
}

// Find safetensors files in directory
std::vector<std::string> find_safetensors(const std::string& directory) {
    std::vector<std::string> files;

    // First check for single model.safetensors
    std::string single_path = directory + "/model.safetensors";
    std::ifstream single_file(single_path);
    if (single_file.good()) {
        files.push_back(single_path);
        return files;
    }

    // Check for weights.safetensors
    std::string weights_path = directory + "/weights.safetensors";
    std::ifstream weights_file(weights_path);
    if (weights_file.good()) {
        files.push_back(weights_path);
        return files;
    }

    // Look for sharded files (model-00001-of-00005.safetensors, etc.)
    for (int i = 1; i <= 100; ++i) {  // Up to 100 shards
        char filename[64];
        snprintf(filename, sizeof(filename), "/model-%05d-of-*.safetensors", i);

        // Try common shard counts
        for (int total : {2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20}) {
            char shard_name[64];
            snprintf(shard_name, sizeof(shard_name), "model-%05d-of-%05d.safetensors", i, total);
            std::string shard_path = directory + "/" + shard_name;
            std::ifstream shard_file(shard_path);
            if (shard_file.good()) {
                files.push_back(shard_path);
                break;
            }
        }
    }

    return files;
}

} // anonymous namespace

// ============================================================================
// LLMConfig
// ============================================================================

LLMConfig LLMConfig::load(const std::string& path) {
    std::string json = read_file(path);
    LLMConfig config;

    config.model_type = get_json_string(json, "model_type");
    config.vocab_size = get_json_int(json, "vocab_size", 32000);
    config.hidden_size = get_json_int(json, "hidden_size", 4096);
    config.intermediate_size = get_json_int(json, "intermediate_size", 11008);
    config.num_hidden_layers = get_json_int(json, "num_hidden_layers", 32);
    config.num_attention_heads = get_json_int(json, "num_attention_heads", 32);
    config.num_key_value_heads = get_json_int(json, "num_key_value_heads", config.num_attention_heads);
    config.head_dim = get_json_int(json, "head_dim", 0);
    config.max_position_embeddings = get_json_int(json, "max_position_embeddings", 4096);
    config.rms_norm_eps = get_json_float(json, "rms_norm_eps", 1e-5f);
    config.rope_theta = get_json_float(json, "rope_theta", 10000.0f);
    config.rope_traditional = get_json_bool(json, "rope_traditional", false);
    config.tie_word_embeddings = get_json_bool(json, "tie_word_embeddings", true);
    config.attention_bias = get_json_bool(json, "attention_bias", false);
    config.mlp_bias = get_json_bool(json, "mlp_bias", false);

    // Parse quantization config (both "quantization" and "quantization_config" formats)
    // Check for "bits" in quantization section
    std::regex quant_bits_pattern("\"quantization[_config]*\"\\s*:\\s*\\{[^}]*\"bits\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (std::regex_search(json, match, quant_bits_pattern)) {
        config.quantization_bits = std::stoi(match[1]);
    }
    std::regex quant_gs_pattern("\"quantization[_config]*\"\\s*:\\s*\\{[^}]*\"group_size\"\\s*:\\s*(\\d+)");
    if (std::regex_search(json, match, quant_gs_pattern)) {
        config.quantization_group_size = std::stoi(match[1]);
    }

    config.finalize();
    return config;
}

LLMConfig LLMConfig::get(const std::string& model_name) {
    LLMConfig config;

    // Normalize name
    std::string name = model_name;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    // Common configurations
    if (name.find("7b") != std::string::npos || name.find("8b") != std::string::npos) {
        config.hidden_size = 4096;
        config.intermediate_size = 11008;
        config.num_hidden_layers = 32;
        config.num_attention_heads = 32;
        config.num_key_value_heads = 32;
    } else if (name.find("13b") != std::string::npos) {
        config.hidden_size = 5120;
        config.intermediate_size = 13824;
        config.num_hidden_layers = 40;
        config.num_attention_heads = 40;
        config.num_key_value_heads = 40;
    } else if (name.find("70b") != std::string::npos) {
        config.hidden_size = 8192;
        config.intermediate_size = 28672;
        config.num_hidden_layers = 80;
        config.num_attention_heads = 64;
        config.num_key_value_heads = 8;  // GQA
    }

    // Set model type
    if (name.find("llama") != std::string::npos) {
        config.model_type = "llama";
    } else if (name.find("mistral") != std::string::npos) {
        config.model_type = "mistral";
        config.num_key_value_heads = 8;  // Mistral uses GQA
    } else if (name.find("qwen") != std::string::npos) {
        config.model_type = "qwen2";
    }

    config.finalize();
    return config;
}

// ============================================================================
// Weights
// ============================================================================

void Weights::load(const std::string& path) {
    auto [arrays, metadata] = mx::load_safetensors(path);
    for (const auto& kv : arrays) {
        weights_.insert_or_assign(kv.first, kv.second);
    }
    std::cout << "Loaded " << weights_.size() << " weights from " << path << "\n";
}

void Weights::load_sharded(const std::string& directory) {
    auto files = find_safetensors(directory);
    if (files.empty()) {
        throw std::runtime_error("No safetensors files found in " + directory);
    }

    for (const auto& file : files) {
        auto [arrays, metadata] = mx::load_safetensors(file);
        for (const auto& kv : arrays) {
            weights_.insert_or_assign(kv.first, kv.second);
        }
    }
    std::cout << "Loaded " << weights_.size() << " weights from " << files.size() << " shards\n";
}

mx::array Weights::get_raw(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

bool Weights::is_quantized(const std::string& name) const {
    // A weight is quantized if it has .weight, .scales, and optionally .biases
    std::string weight_key = name + ".weight";
    std::string scales_key = name + ".scales";
    return weights_.find(weight_key) != weights_.end() &&
           weights_.find(scales_key) != weights_.end();
}

mx::array Weights::get(const std::string& name) const {
    // Check if this is a quantized weight
    if (is_quantized(name)) {
        // Dequantize: w_dequant = dequantize(w, scales, biases, group_size, bits)
        auto w = weights_.at(name + ".weight");
        auto scales = weights_.at(name + ".scales");

        std::optional<mx::array> biases = std::nullopt;
        auto biases_it = weights_.find(name + ".biases");
        if (biases_it != weights_.end()) {
            biases = biases_it->second;
        }

        return mx::dequantize(
            w, scales, biases,
            quantization_group_size_,
            quantization_bits_,
            "affine",
            mx::float16  // Output dtype
        );
    }

    // Not quantized - return directly
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

bool Weights::has(const std::string& name) const {
    // Check for regular weight or quantized weight
    return weights_.find(name) != weights_.end() || is_quantized(name);
}

// ============================================================================
// LLMTokenizer
// ============================================================================

struct LLMTokenizer::Impl {
    // Token ID to text mapping
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    // Special token IDs
    int bos_token = 1;
    int eos_token = 2;
    int pad_token = 0;
};

LLMTokenizer::LLMTokenizer() : impl_(std::make_unique<Impl>()) {}
LLMTokenizer::~LLMTokenizer() = default;
LLMTokenizer::LLMTokenizer(LLMTokenizer&&) noexcept = default;
LLMTokenizer& LLMTokenizer::operator=(LLMTokenizer&&) noexcept = default;

LLMTokenizer LLMTokenizer::load(const std::string& model_path) {
    LLMTokenizer tokenizer;

    // Try to load special tokens from config.json first
    std::string config_path = model_path + "/config.json";
    std::ifstream config_file(config_path);
    if (config_file.good()) {
        config_file.close();
        std::string config_json = read_file(config_path);

        // Get bos_token_id (may be single int or first in array)
        int bos_id = get_json_int(config_json, "bos_token_id", 1);
        tokenizer.bos_token_ = bos_id;

        // Get eos_token_id (use first if array)
        int eos_id = get_json_int(config_json, "eos_token_id", 2);
        if (eos_id == 2) {
            // Try to parse as array
            std::regex eos_array_pattern("\"eos_token_id\"\\s*:\\s*\\[\\s*(\\d+)");
            std::smatch match;
            if (std::regex_search(config_json, match, eos_array_pattern)) {
                eos_id = std::stoi(match[1]);
            }
        }
        tokenizer.eos_token_ = eos_id;

        std::cout << "LLMTokenizer: bos_token=" << tokenizer.bos_token_
                  << ", eos_token=" << tokenizer.eos_token_ << "\n";
    }

    // Try to load tokenizer.json
    std::string json_path = model_path + "/tokenizer.json";
    std::ifstream json_file(json_path);
    if (json_file.good()) {
        json_file.close();
        std::string json = read_file(json_path);

        // Extract vocabulary from "vocab" section
        // This is a simplified parser - in production, use a proper JSON library
        std::regex vocab_pattern("\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*(\\d+)");
        std::sregex_iterator it(json.begin(), json.end(), vocab_pattern);
        std::sregex_iterator end;

        while (it != end) {
            std::smatch match = *it;
            std::string token = match[1];
            int id = std::stoi(match[2]);

            // Unescape JSON escapes
            std::string unescaped;
            for (size_t i = 0; i < token.size(); ++i) {
                if (token[i] == '\\' && i + 1 < token.size()) {
                    char next = token[i + 1];
                    switch (next) {
                        case 'n': unescaped += '\n'; ++i; break;
                        case 't': unescaped += '\t'; ++i; break;
                        case 'r': unescaped += '\r'; ++i; break;
                        case '"': unescaped += '"'; ++i; break;
                        case '\\': unescaped += '\\'; ++i; break;
                        default: unescaped += token[i]; break;
                    }
                } else {
                    unescaped += token[i];
                }
            }

            tokenizer.impl_->id_to_token[id] = unescaped;
            tokenizer.impl_->token_to_id[unescaped] = id;
            ++it;
        }

        tokenizer.loaded_ = !tokenizer.impl_->id_to_token.empty();
        if (tokenizer.loaded_) {
            std::cout << "LLMTokenizer: Loaded " << tokenizer.impl_->id_to_token.size()
                      << " tokens from tokenizer.json\n";
        }
    }

    // If no tokenizer.json, try SentencePiece model
    // (Would need to integrate SentencePiece library)
    if (!tokenizer.loaded_) {
        std::cerr << "Warning: Could not load tokenizer from " << model_path << "\n";
    }

    return tokenizer;
}

std::vector<int> LLMTokenizer::encode(const std::string& text, bool add_bos) const {
    // BPE encoding with GPT-style byte mapping
    // - Spaces become Ġ (U+0120 = UTF-8 C4 A0)
    // - Newlines become Ċ (U+010A = UTF-8 C4 8A)
    std::vector<int> tokens;

    if (add_bos) {
        tokens.push_back(bos_token_);
    }

    // Pre-process: convert to GPT-style byte representation
    // Leading space at start of text/after newline should also be Ġ
    std::string processed;
    bool at_word_start = true;  // Start of text is like after a space
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == ' ') {
            // Space becomes Ġ prefix for next word
            processed += "\xC4\xA0";  // UTF-8 for Ġ (U+0120)
            at_word_start = true;
        } else if (c == '\n') {
            processed += "\xC4\x8A";  // UTF-8 for Ċ (U+010A)
            at_word_start = true;
        } else if (c == '\t') {
            processed += "\xC4\x89";  // UTF-8 for ĉ (U+0109) - tab
            at_word_start = true;
        } else {
            // Regular character - add Ġ prefix if not already preceded by space
            // BUT: for first char after BOS or after Ġ, don't add another Ġ
            processed += c;
            at_word_start = false;
        }
    }

    // Greedy longest-match tokenization
    size_t pos = 0;
    while (pos < processed.size()) {
        int best_id = -1;
        size_t best_len = 0;

        // Try to match longest token starting at pos
        for (const auto& [token, id] : impl_->token_to_id) {
            if (token.size() > best_len &&
                pos + token.size() <= processed.size() &&
                processed.substr(pos, token.size()) == token) {
                best_id = id;
                best_len = token.size();
            }
        }

        if (best_id >= 0) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            // Unknown byte - try single character fallback
            // For UTF-8, this might be multi-byte
            if ((processed[pos] & 0x80) == 0) {
                // ASCII - skip 1 byte
                ++pos;
            } else if ((processed[pos] & 0xE0) == 0xC0) {
                // 2-byte UTF-8
                pos += 2;
            } else if ((processed[pos] & 0xF0) == 0xE0) {
                // 3-byte UTF-8
                pos += 3;
            } else {
                // 4-byte UTF-8 or invalid
                pos += std::min(size_t(4), processed.size() - pos);
            }
        }
    }

    return tokens;
}

std::string LLMTokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        // Skip special tokens
        if (id == bos_token_ || id == eos_token_ || id == pad_token_) {
            continue;
        }
        // Skip other special tokens (>=128000 for LLaMA 3)
        if (id >= 128000) {
            continue;
        }

        auto it = impl_->id_to_token.find(id);
        if (it != impl_->id_to_token.end()) {
            std::string token = it->second;

            // Handle BPE special characters:
            // - Ġ (U+0120) represents space prefix in GPT-style BPE
            // - Ċ (U+010A) represents newline
            // Replace the Unicode bytes with actual characters
            std::string processed;
            for (size_t i = 0; i < token.size(); ++i) {
                // Ġ is UTF-8 C4 A0 (U+0120)
                if (i + 1 < token.size() &&
                    static_cast<unsigned char>(token[i]) == 0xC4 &&
                    static_cast<unsigned char>(token[i+1]) == 0xA0) {
                    processed += ' ';
                    i++;  // Skip second byte
                }
                // Ċ is UTF-8 C4 8A (U+010A)
                else if (i + 1 < token.size() &&
                         static_cast<unsigned char>(token[i]) == 0xC4 &&
                         static_cast<unsigned char>(token[i+1]) == 0x8A) {
                    processed += '\n';
                    i++;  // Skip second byte
                }
                else {
                    processed += token[i];
                }
            }
            result += processed;
        }
    }
    return result;
}

int LLMTokenizer::vocab_size() const {
    return static_cast<int>(impl_->id_to_token.size());
}

// ============================================================================
// LLMModel
// ============================================================================

LLMModel::LLMModel() = default;

LLMModel::~LLMModel() {
    // Synchronize GPU before destroying model resources
    // This prevents Metal kernel crashes on repeated model loads
    if (loaded_) {
        mx::synchronize();
    }
}
LLMModel::LLMModel(LLMModel&&) noexcept = default;
LLMModel& LLMModel::operator=(LLMModel&&) noexcept = default;

LLMModel LLMModel::load(const std::string& model_path) {
    LLMModel model;

    // Load config
    std::string config_path = model_path + "/config.json";
    model.config_ = LLMConfig::load(config_path);
    model.config_.finalize();

    std::cout << "LLMModel: Loading " << model.config_.model_type << " model\n";
    if (model.config_.quantization_bits > 0) {
        std::cout << "LLMModel: " << model.config_.quantization_bits << "-bit quantization"
                  << " (group_size=" << model.config_.quantization_group_size << ")\n";
    }

    // Load weights
    auto safetensors_files = find_safetensors(model_path);
    if (safetensors_files.empty()) {
        throw std::runtime_error("No safetensors files found in " + model_path);
    }

    if (safetensors_files.size() == 1) {
        model.weights_.load(safetensors_files[0]);
    } else {
        model.weights_.load_sharded(model_path);
    }

    // Set quantization parameters if model is quantized
    if (model.config_.quantization_bits > 0) {
        model.weights_.set_quantization(
            model.config_.quantization_bits,
            model.config_.quantization_group_size
        );
    }

    // Load tokenizer
    model.tokenizer_ = std::make_unique<LLMTokenizer>(LLMTokenizer::load(model_path));

    // Initialize RoPE frequencies
    model.init_rope();

    model.loaded_ = true;
    return model;
}

void LLMModel::init_rope() {
    // Compute RoPE frequencies: theta_i = 10000^(-2i/dim)
    int head_dim = config_.head_dim;
    std::vector<float> inv_freq(head_dim / 2);

    for (int i = 0; i < head_dim / 2; ++i) {
        float exponent = -2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
        inv_freq[i] = std::pow(config_.rope_theta, exponent);
    }

    rope_freqs_ = mx::array(inv_freq.data(), {head_dim / 2});
}

mx::array LLMModel::apply_rope(const mx::array& x, int offset) {
    // x: [batch, n_heads, seq_len, head_dim]
    int batch_size = static_cast<int>(x.shape()[0]);
    int n_heads = static_cast<int>(x.shape()[1]);
    int seq_len = static_cast<int>(x.shape()[2]);
    int head_dim = static_cast<int>(x.shape()[3]);
    int half_dim = head_dim / 2;

    // Create position indices
    std::vector<float> positions(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        positions[i] = static_cast<float>(offset + i);
    }
    auto pos = mx::array(positions.data(), {seq_len});

    // Compute frequencies: pos * inv_freq
    // pos: [seq_len], inv_freq: [head_dim/2] -> freqs: [seq_len, head_dim/2]
    auto freqs = mx::outer(pos, rope_freqs_);

    // Create rotation matrix components
    auto cos_freqs = mx::cos(freqs);  // [seq_len, head_dim/2]
    auto sin_freqs = mx::sin(freqs);  // [seq_len, head_dim/2]

    // Split x into two halves using explicit dimensions
    auto x1 = mx::slice(x, {0, 0, 0, 0}, {batch_size, n_heads, seq_len, half_dim});
    auto x2 = mx::slice(x, {0, 0, 0, half_dim}, {batch_size, n_heads, seq_len, head_dim});

    // Reshape for broadcasting
    cos_freqs = mx::reshape(cos_freqs, {1, 1, seq_len, half_dim});
    sin_freqs = mx::reshape(sin_freqs, {1, 1, seq_len, half_dim});

    // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    auto rotated_x1 = x1 * cos_freqs - x2 * sin_freqs;
    auto rotated_x2 = x1 * sin_freqs + x2 * cos_freqs;

    // Concatenate back
    return mx::concatenate({rotated_x1, rotated_x2}, -1);
}

mx::array LLMModel::rms_norm(const mx::array& x, const mx::array& weight, float eps) {
    // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    auto x2 = mx::square(x);
    auto mean_x2 = mx::mean(x2, -1, true);
    auto rms = mx::sqrt(mean_x2 + eps);
    return (x / rms) * weight;
}

mx::array LLMModel::self_attention(const mx::array& x, int layer_idx, KVCache* cache) {
    // x: [batch, seq_len, hidden_size]
    int batch_size = static_cast<int>(x.shape()[0]);
    int seq_len = static_cast<int>(x.shape()[1]);
    int hidden_size = config_.hidden_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim;

    // Weight name prefix
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";

    // Q, K, V projections (auto-dequantized if quantized)
    auto q_proj = weights_.get(prefix + "q_proj");
    auto k_proj = weights_.get(prefix + "k_proj");
    auto v_proj = weights_.get(prefix + "v_proj");
    auto o_proj = weights_.get(prefix + "o_proj");

    // Compute Q, K, V: x @ W^T
    auto queries = mx::matmul(x, mx::transpose(q_proj));  // [batch, seq, n_heads * head_dim]
    auto keys = mx::matmul(x, mx::transpose(k_proj));      // [batch, seq, n_kv_heads * head_dim]
    auto values = mx::matmul(x, mx::transpose(v_proj));    // [batch, seq, n_kv_heads * head_dim]

    // Reshape to [batch, seq, n_heads, head_dim] then transpose to [batch, n_heads, seq, head_dim]
    queries = mx::reshape(queries, {batch_size, seq_len, n_heads, head_dim});
    queries = mx::transpose(queries, {0, 2, 1, 3});

    keys = mx::reshape(keys, {batch_size, seq_len, n_kv_heads, head_dim});
    keys = mx::transpose(keys, {0, 2, 1, 3});

    values = mx::reshape(values, {batch_size, seq_len, n_kv_heads, head_dim});
    values = mx::transpose(values, {0, 2, 1, 3});

    // Apply RoPE to queries and keys
    int offset = (cache && !cache->empty()) ? cache->offset : 0;
    queries = apply_rope(queries, offset);
    keys = apply_rope(keys, offset);

    // Update KV cache if provided
    if (cache) {
        // Ensure cache vectors are large enough
        while (cache->keys.size() <= static_cast<size_t>(layer_idx)) {
            // Push placeholder arrays using zeros with minimal shape
            cache->keys.push_back(mx::zeros({1}));
            cache->values.push_back(mx::zeros({1}));
        }

        if (cache->keys[layer_idx].ndim() == 4) {
            // Concatenate with existing cache
            keys = mx::concatenate({cache->keys[layer_idx], keys}, 2);
            values = mx::concatenate({cache->values[layer_idx], values}, 2);
        }

        cache->keys[layer_idx] = keys;
        cache->values[layer_idx] = values;
    }

    // Grouped Query Attention: expand KV heads if needed
    if (n_kv_heads < n_heads) {
        int repeat_factor = n_heads / n_kv_heads;
        // keys: [batch, n_kv_heads, seq, head_dim] -> [batch, n_heads, seq, head_dim]
        keys = mx::repeat(keys, repeat_factor, 1);
        values = mx::repeat(values, repeat_factor, 1);
    }

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attn_weights = mx::matmul(queries, mx::transpose(keys, {0, 1, 3, 2})) * scale;

    // Apply causal mask
    int kv_len = static_cast<int>(keys.shape()[2]);
    if (kv_len > 1) {
        // Create causal mask: upper triangular with -inf
        auto mask = mx::triu(
            mx::full({seq_len, kv_len}, -std::numeric_limits<float>::infinity()),
            kv_len - seq_len + 1
        );
        mask = mx::reshape(mask, {1, 1, seq_len, kv_len});
        attn_weights = attn_weights + mask;
    }

    attn_weights = mx::softmax(attn_weights, -1);

    // Compute attention output
    auto attn_output = mx::matmul(attn_weights, values);  // [batch, n_heads, seq, head_dim]

    // Reshape back: [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads * head_dim]
    attn_output = mx::transpose(attn_output, {0, 2, 1, 3});
    attn_output = mx::reshape(attn_output, {batch_size, seq_len, n_heads * head_dim});

    // Output projection
    auto output = mx::matmul(attn_output, mx::transpose(o_proj));

    return output;
}

mx::array LLMModel::mlp(const mx::array& x, int layer_idx) {
    // SwiGLU: down(silu(gate(x)) * up(x))
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";

    // Auto-dequantized if quantized
    auto gate_proj = weights_.get(prefix + "gate_proj");
    auto up_proj = weights_.get(prefix + "up_proj");
    auto down_proj = weights_.get(prefix + "down_proj");

    auto gate = mx::matmul(x, mx::transpose(gate_proj));
    auto up = mx::matmul(x, mx::transpose(up_proj));

    // SiLU activation: x * sigmoid(x)
    auto silu_gate = gate * mx::sigmoid(gate);

    auto hidden = silu_gate * up;
    auto output = mx::matmul(hidden, mx::transpose(down_proj));

    return output;
}

mx::array LLMModel::decoder_layer(const mx::array& x, int layer_idx, KVCache* cache) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // Get layer norm weights
    auto input_layernorm_weight = weights_.get(prefix + "input_layernorm.weight");
    auto post_attn_layernorm_weight = weights_.get(prefix + "post_attention_layernorm.weight");

    // Pre-norm + attention
    auto normed = rms_norm(x, input_layernorm_weight, config_.rms_norm_eps);
    auto attn_out = self_attention(normed, layer_idx, cache);
    auto h = x + attn_out;

    // Pre-norm + MLP
    normed = rms_norm(h, post_attn_layernorm_weight, config_.rms_norm_eps);
    auto mlp_out = mlp(normed, layer_idx);
    auto output = h + mlp_out;

    return output;
}

mx::array LLMModel::forward(const mx::array& tokens, KVCache* cache) {
    // tokens: [batch, seq_len]

    // Token embeddings (auto-dequantized if quantized)
    auto embed_tokens = weights_.get("model.embed_tokens");
    auto x = mx::take(embed_tokens, tokens, 0);  // [batch, seq_len, hidden_size]

    // Pass through decoder layers
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        x = decoder_layer(x, i, cache);
        mx::eval(x);  // Evaluate to free intermediate memory
    }

    // Final layer norm
    auto norm_weight = weights_.get("model.norm.weight");
    x = rms_norm(x, norm_weight, config_.rms_norm_eps);

    // LM head (may be tied to embeddings)
    mx::array lm_head = embed_tokens;  // Default to tied embeddings
    if (weights_.has("lm_head")) {
        lm_head = weights_.get("lm_head");
    } else if (!config_.tie_word_embeddings) {
        throw std::runtime_error("No lm_head weights and embeddings not tied");
    }

    // Compute logits
    auto logits = mx::matmul(x, mx::transpose(lm_head));  // [batch, seq_len, vocab_size]

    return logits;
}

mx::array LLMModel::sample(const mx::array& logits, float temperature, float top_p, int top_k) {
    // logits: [batch, vocab_size]
    // Implements: temperature scaling, combined top-k and top-p (nucleus) sampling
    //
    // Efficient approach: Use argpartition (O(n)) to get top-k indices, then apply
    // top-p filtering to just those k values. This avoids sorting the full 128K vocabulary.

    // Handle greedy decoding (temperature <= 0)
    if (temperature <= 0.0f) {
        return mx::argmax(logits, -1);
    }

    // Temperature scaling
    auto scaled_logits = logits / temperature;
    mx::eval(scaled_logits);

    // Get vocab_size from the last dimension
    int vocab_size = static_cast<int>(logits.shape().back());
    int batch_size = static_cast<int>(logits.shape()[0]);

    // If top-k or top-p is disabled, use simple categorical sampling
    bool use_topk = (top_k > 0 && top_k < vocab_size);
    bool use_topp = (top_p > 0.0f && top_p < 1.0f);

    if (!use_topk && !use_topp) {
        // No filtering, sample directly from all logits
        auto logits_f32 = mx::astype(scaled_logits, mx::float32);
        mx::eval(logits_f32);
        auto sampled = mx::random::categorical(logits_f32, 1);
        mx::eval(sampled);
        return sampled;
    }

    // Use argpartition for efficient top-k selection (O(n) complexity)
    // argpartition returns indices that partition around the kth element
    // We negate logits to get top-k largest values
    int k = use_topk ? top_k : std::min(100, vocab_size);  // Default to top-100 if only top-p
    auto neg_logits = -scaled_logits;
    mx::eval(neg_logits);

    // argpartition: indices where first k elements are the k smallest of neg_logits
    // (i.e., k largest of original logits)
    auto partitioned_indices = mx::argpartition(neg_logits, k, -1);
    mx::eval(partitioned_indices);

    // Extract just the top-k indices (first k elements after partitioning)
    // Shape: [batch, k]
    auto topk_indices = mx::slice(partitioned_indices, {0, 0}, {batch_size, k});
    mx::eval(topk_indices);

    // Gather the top-k logits
    auto topk_logits = mx::take_along_axis(scaled_logits, topk_indices, -1);
    mx::eval(topk_logits);

    // Apply top-p filtering to just the k values (now sorting k values is fast)
    if (use_topp) {
        // Convert to probabilities (softmax over k values)
        auto topk_probs = mx::softmax(topk_logits, -1);
        mx::eval(topk_probs);

        // Sort probabilities in descending order (negate for ascending argsort)
        auto neg_probs = -topk_probs;
        mx::eval(neg_probs);
        auto sort_indices = mx::argsort(neg_probs, -1);  // Indices to sort descending
        mx::eval(sort_indices);

        // Gather sorted probabilities
        auto sorted_probs = mx::take_along_axis(topk_probs, sort_indices, -1);
        mx::eval(sorted_probs);

        // Compute cumulative sum
        auto cumsum_probs = mx::cumsum(sorted_probs, -1);
        mx::eval(cumsum_probs);

        // Create shifted cumsum: [0, p0, p0+p1, ...] to include token that crosses threshold
        // For each batch, we want: shifted[i] = cumsum[i-1] or 0 for i=0
        auto zeros_col = mx::zeros({batch_size, 1});
        mx::eval(zeros_col);

        // Take cumsum without the last element
        auto cumsum_head = mx::slice(cumsum_probs, {0, 0}, {batch_size, k - 1});
        mx::eval(cumsum_head);

        // Concatenate [0, cumsum[:-1]]
        auto shifted_cumsum = mx::concatenate({zeros_col, cumsum_head}, -1);
        mx::eval(shifted_cumsum);

        // Mask: keep tokens where shifted_cumsum < top_p
        auto keep_mask = shifted_cumsum < top_p;
        mx::eval(keep_mask);

        // Gather logits in sorted order
        auto sorted_logits = mx::take_along_axis(topk_logits, sort_indices, -1);
        mx::eval(sorted_logits);

        // Apply mask: set filtered tokens to -inf
        auto neg_inf = mx::full(sorted_logits.shape(), -std::numeric_limits<float>::infinity());
        auto masked_logits = mx::where(keep_mask, sorted_logits, neg_inf);
        mx::eval(masked_logits);

        // Scatter back to original order within top-k
        auto inv_sort_indices = mx::argsort(sort_indices, -1);
        mx::eval(inv_sort_indices);
        topk_logits = mx::take_along_axis(masked_logits, inv_sort_indices, -1);
        mx::eval(topk_logits);
    }

    // Sample from the filtered top-k distribution
    auto logits_f32 = mx::astype(topk_logits, mx::float32);
    mx::eval(logits_f32);

    // categorical samples an index into the k tokens
    // categorical with axis=1 returns shape [batch] - we need [batch, 1] for take_along_axis
    auto sampled_idx = mx::random::categorical(logits_f32, 1);
    mx::eval(sampled_idx);

    // Reshape to [batch, 1] for take_along_axis
    sampled_idx = mx::reshape(sampled_idx, {batch_size, 1});
    mx::eval(sampled_idx);

    // Map back to original token IDs using topk_indices
    auto sampled_token = mx::take_along_axis(topk_indices, sampled_idx, -1);
    mx::eval(sampled_token);

    // Return as 1D [batch] to match expected output shape
    return mx::reshape(sampled_token, {batch_size});
}

std::vector<int> LLMModel::generate(
    const std::vector<int>& tokens,
    int max_tokens,
    float temperature,
    float top_p,
    int top_k
) {
    std::vector<int> output_tokens = tokens;
    KVCache cache;

    // Initial forward pass with full prompt
    auto token_arr = mx::array(tokens.data(), {1, static_cast<int>(tokens.size())}, mx::int32);
    auto logits = forward(token_arr, &cache);
    mx::eval(logits);
    cache.update_offset(static_cast<int>(tokens.size()));

    // Get logits for last position
    int seq_len = static_cast<int>(logits.shape()[1]);
    auto last_logits = mx::slice(logits, {0, seq_len - 1, 0}, {1, seq_len, config_.vocab_size});
    last_logits = mx::reshape(last_logits, {1, config_.vocab_size});

    // Sample first token
    auto next_token = sample(last_logits, temperature, top_p, top_k);
    mx::eval(next_token);
    int token_id = next_token.item<int32_t>();
    output_tokens.push_back(token_id);

    // Generate remaining tokens
    for (int i = 1; i < max_tokens; ++i) {
        // Check for EOS
        if (tokenizer_ && token_id == tokenizer_->eos_token()) {
            break;
        }

        // Forward pass with single token
        token_arr = mx::array(&token_id, {1, 1}, mx::int32);
        logits = forward(token_arr, &cache);
        mx::eval(logits);
        cache.update_offset(1);

        // Sample next token
        last_logits = mx::reshape(logits, {1, config_.vocab_size});
        next_token = sample(last_logits, temperature, top_p, top_k);
        mx::eval(next_token);

        token_id = next_token.item<int32_t>();
        output_tokens.push_back(token_id);
    }

    return output_tokens;
}

std::string LLMModel::generate_text(
    const std::string& prompt,
    int max_tokens,
    float temperature,
    float top_p,
    int top_k
) {
    if (!tokenizer_ || !tokenizer_->loaded()) {
        throw std::runtime_error("Tokenizer not loaded");
    }

    // Encode prompt
    auto tokens = tokenizer_->encode(prompt, true);

    // Generate
    auto output_tokens = generate(tokens, max_tokens, temperature, top_p, top_k);

    // Decode (skip prompt tokens)
    std::vector<int> new_tokens(output_tokens.begin() + tokens.size(), output_tokens.end());
    return tokenizer_->decode(new_tokens);
}

std::string LLMModel::info() const {
    std::ostringstream oss;
    oss << "LLMModel: " << config_.model_type << "\n";
    oss << "  Hidden size: " << config_.hidden_size << "\n";
    oss << "  Layers: " << config_.num_hidden_layers << "\n";
    oss << "  Attention heads: " << config_.num_attention_heads << "\n";
    oss << "  KV heads: " << config_.num_key_value_heads << "\n";
    oss << "  Head dim: " << config_.head_dim << "\n";
    oss << "  Vocab size: " << config_.vocab_size << "\n";
    oss << "  Max position: " << config_.max_position_embeddings << "\n";
    return oss.str();
}

std::string LLMModel::format_chat_prompt(
    const std::string& user_message,
    const std::string& system_prompt
) const {
    // LLaMA 3 Instruct chat template:
    // <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    // {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    // {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    //
    // For LLaMA 3, the special tokens are:
    // <|begin_of_text|> = 128000 (BOS)
    // <|start_header_id|> = 128006
    // <|end_header_id|> = 128007
    // <|eot_id|> = 128009

    std::ostringstream prompt;

    // Note: We don't add <|begin_of_text|> here since it's added by encode()
    // with add_bos=true

    if (!system_prompt.empty()) {
        prompt << "<|start_header_id|>system<|end_header_id|>\n\n";
        prompt << system_prompt << "<|eot_id|>";
    }

    prompt << "<|start_header_id|>user<|end_header_id|>\n\n";
    prompt << user_message << "<|eot_id|>";
    prompt << "<|start_header_id|>assistant<|end_header_id|>\n\n";

    return prompt.str();
}

std::string LLMModel::chat(
    const std::string& user_message,
    const std::string& system_prompt,
    int max_tokens,
    float temperature,
    float top_p,
    int top_k
) {
    if (!tokenizer_ || !tokenizer_->loaded()) {
        throw std::runtime_error("Tokenizer not loaded");
    }

    // Format the prompt using chat template
    std::string formatted_prompt = format_chat_prompt(user_message, system_prompt);

    // Generate response
    return generate_text(formatted_prompt, max_tokens, temperature, top_p, top_k);
}

} // namespace llm
