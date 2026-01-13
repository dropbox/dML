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

// Whisper STT Model C++ Implementation
// OpenAI Whisper encoder-decoder transformer for MLX

#include "whisper_model.h"
#include "grammar_parser.h"
#include "mlx/io.h"
#include "mlx/random.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <memory>
#include <regex>
#include <chrono>
#include <zlib.h>

// Silero VAD for neural network-based voice activity detection
// (Replaces WebRTC libfvad for exact parity with Python WhisperMLX)

namespace whisper {

// ============================================================================
// Debug helpers for layer-by-layer comparison
// ============================================================================

namespace {

// Helper to save array to binary file for Python comparison
void save_debug_array(const std::string& name, const mx::array& arr) {
    static bool debug_layers = std::getenv("DEBUG_ENCODER_LAYERS") != nullptr;
    if (!debug_layers) return;

    mx::eval(arr);
    std::string path = "/tmp/cpp_" + name + ".bin";

    // Convert to float32 for comparison
    auto arr_f32 = mx::astype(arr, mx::float32);
    mx::eval(arr_f32);

    auto data = arr_f32.data<float>();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), arr_f32.size() * sizeof(float));
    f.close();

    // Print stats
    auto min_val = mx::min(arr_f32);
    auto max_val = mx::max(arr_f32);
    mx::eval(min_val);
    mx::eval(max_val);
    std::cout << "[DEBUG] Saved " << path << ": shape=[";
    for (size_t i = 0; i < arr.shape().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << arr.shape()[i];
    }
    std::cout << "], min=" << min_val.item<float>()
              << ", max=" << max_val.item<float>() << "\n";
}

}  // namespace

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

[[maybe_unused]] float parse_float(const std::string& json, const std::string& key, float default_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9.e+-]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stof(match[1]);
    }
    return default_val;
}

}  // anonymous namespace

// ============================================================================
// Compression ratio calculation for hallucination detection (GAP 12/16)
// ============================================================================

float calculate_compression_ratio(const std::string& text) {
    if (text.empty()) {
        return 1.0f;  // Empty text has ratio 1.0
    }

    // Get original size
    uLong original_size = static_cast<uLong>(text.size());

    // Allocate buffer for compressed data (worst case is slightly larger than input)
    uLong compressed_size = compressBound(original_size);
    std::vector<Bytef> compressed_buffer(compressed_size);

    // Compress using zlib default compression level
    int result = compress(
        compressed_buffer.data(),
        &compressed_size,
        reinterpret_cast<const Bytef*>(text.data()),
        original_size
    );

    if (result != Z_OK) {
        // On compression error, return 1.0 (neutral value)
        return 1.0f;
    }

    // Return ratio: original_size / compressed_size
    // Higher ratio = more compressible = more repetitive = likely hallucination
    return static_cast<float>(original_size) / static_cast<float>(compressed_size);
}

// ============================================================================
// GAP 57: Composable LogitFilter implementations
// ============================================================================
// These filters match Python mlx-whisper's filter chain architecture:
//   for logit_filter in self.logit_filters:
//       logits = logit_filter.apply(logits, tokens)
// ============================================================================

void SuppressBlank::apply(std::vector<float>& logits, const LogitFilterContext& ctx) {
    // SuppressBlank: At sample_begin, suppress space token + EOT
    // Python: mask[tokenizer.encode(" ") + [tokenizer.eot]] = -np.inf
    int current_len = static_cast<int>(ctx.tokens.size());
    if (current_len == ctx.sample_begin) {
        logits[space_token_] = -INFINITY;
        logits[ctx.eot_token] = -INFINITY;
    }
}

void SuppressTokens::apply(std::vector<float>& logits, const LogitFilterContext& ctx) {
    // SuppressTokens: Suppress all tokens in the list
    // Python: mask[self.suppress_tokens] = -np.inf
    for (int token : suppress_tokens_) {
        if (token >= 0 && token < ctx.n_vocab) {
            logits[token] = -INFINITY;
        }
    }
}

// ============================================================================
// SuppressRegex - GAP 43: Regex-based token suppression
// ============================================================================

SuppressRegex::SuppressRegex(const std::string& regex_pattern,
                             const std::unordered_map<int, std::string>& id_to_token)
    : pattern_(regex_pattern) {
    // Pre-compute all token IDs that match the regex
    // This is done once at construction for efficiency
    try {
        std::regex re(regex_pattern);
        for (const auto& [token_id, token_str] : id_to_token) {
            if (std::regex_match(token_str, re)) {
                suppress_tokens_.push_back(token_id);
            }
        }
        // Sort for potential binary search optimization (not needed but good practice)
        std::sort(suppress_tokens_.begin(), suppress_tokens_.end());

        if (!suppress_tokens_.empty()) {
            std::cerr << "SuppressRegex: Pattern '" << regex_pattern
                      << "' matches " << suppress_tokens_.size() << " tokens\n";
        }
    } catch (const std::regex_error& e) {
        std::cerr << "SuppressRegex: Invalid regex pattern '" << regex_pattern
                  << "': " << e.what() << "\n";
        // Leave suppress_tokens_ empty on error - filter will be a no-op
    }
}

void SuppressRegex::apply(std::vector<float>& logits, const LogitFilterContext& ctx) {
    // SuppressRegex: Suppress all tokens matching the pre-computed regex
    // Same as SuppressTokens but with tokens computed from regex at construction
    for (int token : suppress_tokens_) {
        if (token >= 0 && token < ctx.n_vocab) {
            logits[token] = -INFINITY;
        }
    }
}

// ===========================================================================
// GAP 42: GrammarLogitFilter Implementation
// ===========================================================================

// Helper: decode string to UTF-8 code points
static std::vector<uint32_t> decode_utf8_to_codepoints(const std::string& str) {
    std::vector<uint32_t> result;
    const char* pos = str.c_str();
    while (*pos) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        uint32_t value = 0;
        int len = 1;

        if ((first_byte & 0x80) == 0) {
            value = first_byte;
        } else if ((first_byte & 0xE0) == 0xC0) {
            value = first_byte & 0x1F;
            len = 2;
        } else if ((first_byte & 0xF0) == 0xE0) {
            value = first_byte & 0x0F;
            len = 3;
        } else if ((first_byte & 0xF8) == 0xF0) {
            value = first_byte & 0x07;
            len = 4;
        }

        for (int i = 1; i < len && pos[i]; ++i) {
            value = (value << 6) | (static_cast<uint8_t>(pos[i]) & 0x3F);
        }
        result.push_back(value);
        pos += len;
    }
    result.push_back(0);  // Null terminator
    return result;
}

GrammarLogitFilter::GrammarLogitFilter(
    const std::string& grammar_str,
    const std::unordered_map<int, std::string>& id_to_token,
    float penalty,
    size_t start_rule)
    : id_to_token_(id_to_token), penalty_(penalty) {

    // Parse grammar
    auto parsed = grammar_parser::parse(grammar_str.c_str());
    if (!parsed.valid()) {
        std::cerr << "[GrammarLogitFilter] Failed to parse grammar\n";
        return;
    }

    // Initialize grammar state
    grammar_ = std::make_unique<Grammar>();
    grammar_->init(parsed, start_rule);

    // Pre-compute UTF-8 code points for all tokens
    for (const auto& kv : id_to_token_) {
        token_code_points_[kv.first] = decode_utf8_to_codepoints(kv.second);
    }
}

GrammarLogitFilter::~GrammarLogitFilter() = default;

void GrammarLogitFilter::apply(std::vector<float>& logits, const LogitFilterContext& ctx) {
    if (!grammar_ || !grammar_->active()) {
        return;
    }

    // Build candidate list with pre-computed code points
    std::vector<GrammarCandidate> candidates;
    candidates.reserve(ctx.n_vocab);

    for (int i = 0; i < ctx.n_vocab; ++i) {
        auto it = token_code_points_.find(i);
        if (it != token_code_points_.end() && !it->second.empty()) {
            GrammarCandidate cand;
            cand.id = i;
            cand.code_points = it->second.data();
            cand.partial_utf8 = {0, 0};
            candidates.push_back(cand);
        }
    }

    // Get rejected tokens
    auto rejects = grammar_->reject_candidates(candidates);

    // Apply penalty to rejected tokens
    for (const auto& reject : rejects) {
        if (reject.id >= 0 && reject.id < ctx.n_vocab) {
            logits[reject.id] += penalty_;  // penalty_ is typically large negative
        }
    }
}

void GrammarLogitFilter::accept_token(int token_id) {
    if (!grammar_ || !grammar_->active()) {
        return;
    }

    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        grammar_->accept_token(it->second);
    }
}

void GrammarLogitFilter::reset() {
    // Re-parse and re-init would be needed for full reset
    // For now, just clear stacks (simplified implementation)
    if (grammar_) {
        grammar_->stacks.clear();
    }
}

bool GrammarLogitFilter::active() const {
    return grammar_ && grammar_->active();
}

void ApplyTimestampRules::get_timestamp_state(
    const std::vector<int>& tokens,
    int sample_begin,
    int timestamp_begin,
    bool& last_was_timestamp,
    bool& penultimate_was_timestamp,
    std::vector<int>& timestamp_tokens
) {
    // Get sampled tokens (after prompt)
    timestamp_tokens.clear();
    int sampled_len = static_cast<int>(tokens.size()) - sample_begin;

    if (sampled_len <= 0) {
        last_was_timestamp = false;
        penultimate_was_timestamp = true;  // Python: default True when < 2 tokens
        return;
    }

    // Check last two tokens
    // Python: last_was_timestamp = len(seq) >= 1 and seq[-1] >= timestamp_begin
    last_was_timestamp = tokens.back() >= timestamp_begin;

    // Python: penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin
    if (sampled_len < 2) {
        penultimate_was_timestamp = true;
    } else {
        penultimate_was_timestamp = tokens[tokens.size() - 2] >= timestamp_begin;
    }

    // Collect all timestamp tokens in sampled sequence
    for (int i = sample_begin; i < static_cast<int>(tokens.size()); ++i) {
        if (tokens[i] >= timestamp_begin) {
            timestamp_tokens.push_back(tokens[i]);
        }
    }
}

bool ApplyTimestampRules::compute_timestamp_dominance(
    const std::vector<float>& logits,
    int timestamp_begin,
    int n_vocab
) {
    // Compute if sum of timestamp probabilities > max text token probability
    // This must be computed BEFORE applying suppression rules
    // Python: timestamp_logprob = logsumexp(logits[timestamp_begin:])
    //         max_text_token_logprob = max(logits[:timestamp_begin])
    //         if timestamp_logprob > max_text_token_logprob: ...

    float max_logit = *std::max_element(logits.begin(), logits.end());

    // Compute logsumexp for all tokens (normalizing constant)
    float sum_exp = 0.0f;
    for (int t = 0; t < n_vocab; ++t) {
        if (logits[t] > -1e9f) {
            sum_exp += std::exp(logits[t] - max_logit);
        }
    }
    float log_sum_exp = max_logit + std::log(sum_exp + 1e-10f);

    // Compute logsumexp for timestamp tokens only
    float timestamp_sum_exp = 0.0f;
    for (int t = timestamp_begin; t < n_vocab; ++t) {
        if (logits[t] > -1e9f) {
            timestamp_sum_exp += std::exp(logits[t] - max_logit);
        }
    }
    float timestamp_logprob = (timestamp_sum_exp > 0) ?
        (max_logit + std::log(timestamp_sum_exp)) - log_sum_exp : -INFINITY;

    // Find max text token logprob
    float max_text_logit = -INFINITY;
    for (int t = 0; t < timestamp_begin; ++t) {
        if (logits[t] > max_text_logit) {
            max_text_logit = logits[t];
        }
    }
    float max_text_logprob = (max_text_logit > -1e9f) ?
        max_text_logit - log_sum_exp : -INFINITY;

    return timestamp_logprob > max_text_logprob;
}

void ApplyTimestampRules::apply(std::vector<float>& logits, const LogitFilterContext& ctx) {
    // ApplyTimestampRules - from Python mlx-whisper decoding.py
    // Enforces Whisper's timestamp pairing and monotonicity rules

    bool last_was_timestamp, penultimate_was_timestamp;
    std::vector<int> timestamp_tokens;
    get_timestamp_state(ctx.tokens, ctx.sample_begin, ctx.timestamp_begin,
                       last_was_timestamp, penultimate_was_timestamp, timestamp_tokens);

    int current_len = static_cast<int>(ctx.tokens.size());

    // Rule 1: At sample_begin, force timestamp token
    if (current_len == ctx.sample_begin) {
        // Suppress all non-timestamp tokens (except EOT)
        for (int t = 0; t < ctx.timestamp_begin; ++t) {
            if (t != ctx.eot_token) {
                logits[t] = -INFINITY;
            }
        }
        // Apply max_initial_timestamp constraint
        int last_allowed = ctx.timestamp_begin + ctx.max_initial_timestamp_index;
        for (int t = last_allowed + 1; t < ctx.n_vocab; ++t) {
            logits[t] = -INFINITY;
        }
        return;  // Don't apply other rules at sample_begin
    }

    // Rule 2: Timestamp pairing (matching Python ApplyTimestampRules)
    // - After timestamp -> timestamp: force text (suppress timestamps)
    // - After text -> timestamp: force timestamp or EOT (suppress text)
    if (last_was_timestamp) {
        if (penultimate_was_timestamp) {
            // Two timestamps in a row - next must be text
            for (int t = ctx.timestamp_begin; t < ctx.n_vocab; ++t) {
                logits[t] = -INFINITY;
            }
        } else {
            // Timestamp after text - cannot be normal text (force timestamp or EOT)
            for (int t = 0; t < ctx.eot_token; ++t) {
                logits[t] = -INFINITY;
            }
        }
    }

    // Rule 3: Timestamps must be strictly monotonically increasing
    if (!timestamp_tokens.empty()) {
        int timestamp_last = timestamp_tokens.back() + 1;  // +1 for strict monotonicity
        for (int t = ctx.timestamp_begin; t < timestamp_last && t < ctx.n_vocab; ++t) {
            logits[t] = -INFINITY;
        }
    }

    // Rule 4: Suppress timestamps beyond audio duration
    int max_allowed_timestamp = ctx.timestamp_begin + ctx.max_audio_timestamp_index;
    for (int t = max_allowed_timestamp + 1; t < ctx.n_vocab; ++t) {
        logits[t] = -INFINITY;
    }

    // Rule 5: Force EOT when near end of audio with two timestamps
    int sampled_len = static_cast<int>(ctx.tokens.size()) - ctx.sample_begin;
    if (sampled_len >= 2 && last_was_timestamp && penultimate_was_timestamp) {
        int last_tok = ctx.tokens.back();
        int penult_tok = ctx.tokens[ctx.tokens.size() - 2];
        int last_ts_idx = last_tok - ctx.timestamp_begin;
        int penult_ts_idx = penult_tok - ctx.timestamp_begin;
        const int near_end_threshold = 5;
        int near_max = ctx.max_audio_timestamp_index - near_end_threshold;
        if (last_ts_idx >= near_max && penult_ts_idx >= near_max) {
            // Both timestamps near max - force EOT
            for (int t = 0; t < ctx.eot_token; ++t) {
                logits[t] = -INFINITY;
            }
            for (int t = ctx.eot_token + 1; t < ctx.n_vocab; ++t) {
                logits[t] = -INFINITY;
            }
        }
    }
}

// ============================================================================
// WhisperConfig
// ============================================================================

WhisperConfig WhisperConfig::load(const std::string& path) {
    std::string json = read_file(path);

    WhisperConfig config;

    // Read config values with defaults
    // Try different key names used by different Whisper model formats:
    // - MLX community format: n_mels, n_audio_layer, n_text_layer, n_audio_state, etc.
    // - HuggingFace format: num_mel_bins, encoder_layers, decoder_layers, d_model, etc.
    config.n_mels = parse_int(json, "n_mels", parse_int(json, "num_mel_bins", 128));
    config.n_audio_ctx = parse_int(json, "n_audio_ctx", parse_int(json, "max_source_positions", 1500));
    config.n_audio_state = parse_int(json, "n_audio_state", parse_int(json, "d_model", 1280));
    config.n_text_state = parse_int(json, "n_text_state", config.n_audio_state);
    config.n_audio_head = parse_int(json, "n_audio_head", parse_int(json, "encoder_attention_heads", 20));
    config.n_text_head = parse_int(json, "n_text_head", parse_int(json, "decoder_attention_heads", 20));
    config.n_audio_layer = parse_int(json, "n_audio_layer", parse_int(json, "encoder_layers", 32));
    config.n_text_layer = parse_int(json, "n_text_layer", parse_int(json, "decoder_layers", 32));
    config.n_vocab = parse_int(json, "n_vocab", parse_int(json, "vocab_size", 51866));
    config.n_text_ctx = parse_int(json, "n_text_ctx",
                                   parse_int(json, "max_target_positions",
                                             parse_int(json, "max_length", 448)));

    // Special token IDs for Whisper large-v3
    // These must match Python WhisperMLX tokenizer values exactly for Gate 0 parity.
    // Token layout (verified via Python get_whisper_tokenizer(51865)):
    //   50257=EOT, 50258=SOT, 50259=<|en|>, ..., 50357=<|su|>
    //   50358=<|translate|>, 50359=<|transcribe|>, 50360=<|startoflm|>
    //   50361=<|startofprev|>, 50362=<|nospeech|>, 50363=<|notimestamps|>
    //   50364+=timestamps (50364=0.00s, 50365=0.02s, ...)
    // CRITICAL FIX: Previous values were off-by-one, causing C++ to use wrong task token
    config.translate_token = 50358;         // <|translate|>
    config.transcribe_token = 50359;        // <|transcribe|>
    config.no_timestamps_token = 50363;     // <|notimestamps|>
    config.timestamp_begin = 50364;         // <|0.00|> (first timestamp token)

    return config;
}

WhisperConfig WhisperConfig::get(const std::string& model_name) {
    WhisperConfig config;

    // Normalize name
    std::string name = model_name;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    // Predefined configs
    if (name == "tiny" || name == "tiny.en") {
        config.n_mels = 80;
        config.n_audio_state = 384;
        config.n_audio_head = 6;
        config.n_audio_layer = 4;
        config.n_text_state = 384;
        config.n_text_head = 6;
        config.n_text_layer = 4;
        config.n_vocab = (name == "tiny.en") ? 51864 : 51865;
    } else if (name == "base" || name == "base.en") {
        config.n_mels = 80;
        config.n_audio_state = 512;
        config.n_audio_head = 8;
        config.n_audio_layer = 6;
        config.n_text_state = 512;
        config.n_text_head = 8;
        config.n_text_layer = 6;
        config.n_vocab = (name == "base.en") ? 51864 : 51865;
    } else if (name == "small" || name == "small.en") {
        config.n_mels = 80;
        config.n_audio_state = 768;
        config.n_audio_head = 12;
        config.n_audio_layer = 12;
        config.n_text_state = 768;
        config.n_text_head = 12;
        config.n_text_layer = 12;
        config.n_vocab = (name == "small.en") ? 51864 : 51865;
    } else if (name == "medium" || name == "medium.en") {
        config.n_mels = 80;
        config.n_audio_state = 1024;
        config.n_audio_head = 16;
        config.n_audio_layer = 24;
        config.n_text_state = 1024;
        config.n_text_head = 16;
        config.n_text_layer = 24;
        config.n_vocab = (name == "medium.en") ? 51864 : 51865;
    } else if (name == "large" || name == "large-v2") {
        config.n_mels = 80;
        config.n_audio_state = 1280;
        config.n_audio_head = 20;
        config.n_audio_layer = 32;
        config.n_text_state = 1280;
        config.n_text_head = 20;
        config.n_text_layer = 32;
        config.n_vocab = 51865;
    } else if (name == "large-v3") {
        config.n_mels = 128;
        config.n_audio_state = 1280;
        config.n_audio_head = 20;
        config.n_audio_layer = 32;
        config.n_text_state = 1280;
        config.n_text_head = 20;
        config.n_text_layer = 32;
        config.n_vocab = 51866;
    } else if (name == "large-v3-turbo" || name == "turbo") {
        config.n_mels = 128;
        config.n_audio_state = 1280;
        config.n_audio_head = 20;
        config.n_audio_layer = 32;
        config.n_text_state = 1280;
        config.n_text_head = 20;
        config.n_text_layer = 4;  // Turbo has only 4 decoder layers
        config.n_vocab = 51866;
    } else if (name == "distil-large-v3") {
        config.n_mels = 128;
        config.n_audio_state = 1280;
        config.n_audio_head = 20;
        config.n_audio_layer = 32;
        config.n_text_state = 1280;
        config.n_text_head = 20;
        config.n_text_layer = 2;  // Distilled has 2 decoder layers
        config.n_vocab = 51866;
    } else {
        throw std::runtime_error("Unknown model name: " + model_name);
    }

    config.name = name;
    return config;
}

// ============================================================================
// Weights
// ============================================================================

void Weights::load(const std::string& path) {
    // Check if file exists
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + path);
    }
    f.close();

    // Load using MLX safetensors
    auto [arrays, metadata] = mx::load_safetensors(path);
    for (const auto& kv : arrays) {
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
// Helper Functions
// ============================================================================

namespace {

// Generate sinusoidal positional embeddings (utility for reference)
[[maybe_unused]] mx::array sinusoids(int length, int channels, int max_timescale = 10000) {
    float log_timescale_increment = std::log(static_cast<float>(max_timescale)) / (channels / 2 - 1);

    auto inv_timescales = mx::exp(
        -log_timescale_increment * mx::arange(0, channels / 2, mx::float32)
    );

    auto positions = mx::arange(0, length, mx::float32);
    auto scaled_time = mx::expand_dims(positions, 1) * mx::expand_dims(inv_timescales, 0);

    return mx::concatenate({mx::sin(scaled_time), mx::cos(scaled_time)}, 1);
}

// Layer normalization using mx::fast::layer_norm
mx::array layer_norm(const mx::array& x, const mx::array& weight, const mx::array& bias, float eps = 1e-5f) {
    return mx::fast::layer_norm(x, weight, bias, eps);
}

// GELU activation - match mlx.nn.gelu (erf-based) with compilation
// Formula: x * (1 + erf(x / sqrt(2))) / 2
// Using mx::compile for kernel fusion and better performance
mx::array gelu(const mx::array& x) {
    static auto compiled = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            const auto& in = inputs[0];
            constexpr float sqrt2_f32 = 1.4142135381698608f;
            auto y = in * (1.0f + mx::erf(in / sqrt2_f32)) / 2.0f;
            return {y};
        },
        /*shapeless=*/true);
    return compiled({x})[0];
}

// Compiled MLP: matmul(gelu(matmul(x, w1) + b1), w2) + b2
// Takes x, w1, b1, w2, b2 as inputs for full fusion
mx::array compiled_mlp(
    const mx::array& x,
    const mx::array& w1,
    const mx::array& b1,
    const mx::array& w2,
    const mx::array& b2
) {
    static auto compiled = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            const auto& x = inputs[0];
            const auto& w1 = inputs[1];
            const auto& b1 = inputs[2];
            const auto& w2 = inputs[3];
            const auto& b2 = inputs[4];

            // First linear: x @ w1.T + b1
            auto h = mx::matmul(x, mx::transpose(w1)) + b1;

            // GELU activation
            constexpr float sqrt2_f32 = 1.4142135381698608f;
            h = h * (1.0f + mx::erf(h / sqrt2_f32)) / 2.0f;

            // Second linear: h @ w2.T + b2
            h = mx::matmul(h, mx::transpose(w2)) + b2;

            return {h};
        },
        /*shapeless=*/true);
    return compiled({x, w1, b1, w2, b2})[0];
}

// Scaled dot-product attention
mx::array scaled_dot_product_attention(
    const mx::array& q,  // [batch, n_heads, seq_len, head_dim]
    const mx::array& k,  // [batch, n_heads, kv_len, head_dim]
    const mx::array& v,  // [batch, n_heads, kv_len, head_dim]
    const std::string& mask_mode = "",  // "causal" for causal masking
    const std::optional<mx::array>& mask_arr = std::nullopt  // Custom mask array
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(q.shape().back()));
    return mx::fast::scaled_dot_product_attention(q, k, v, scale, mask_mode, mask_arr);
}

// Create causal attention mask (utility for reference)
[[maybe_unused]] mx::array create_causal_mask(int seq_len, mx::Dtype dtype = mx::float16) {
    auto mask = mx::triu(mx::full({seq_len, seq_len}, -INFINITY, dtype), 1);
    return mask;
}

}  // namespace

// ============================================================================
// AudioEncoder
// ============================================================================

void AudioEncoder::init(const WhisperConfig& config, const Weights* weights) {
    config_ = config;
    weights_ = weights;

    // Precompute sinusoidal positional embeddings using MLX ops (matching Python exactly)
    // This ensures numerical parity with Python WhisperMLX's sinusoids() function
    pos_emb_len_ = config.n_audio_ctx;

    // Python equivalent:
    //   log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    //   inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    //   scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    //   return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1).astype(dtype)

    int channels = config.n_audio_state;
    int length = config.n_audio_ctx;
    int half_channels = channels / 2;

    float log_timescale_increment = std::log(10000.0f) / (half_channels - 1);

    // Create inv_timescales using MLX ops (matches mx.exp(-log_timescale_increment * mx.arange(channels // 2)))
    auto channel_indices = mx::arange(0, half_channels, mx::float32);
    auto inv_timescales = mx::exp(-log_timescale_increment * channel_indices);

    // Create position indices
    auto positions = mx::arange(0, length, mx::float32);

    // scaled_time = positions[:, None] * inv_timescales[None, :] -> shape (length, half_channels)
    auto scaled_time = mx::expand_dims(positions, 1) * mx::expand_dims(inv_timescales, 0);

    // Compute sin and cos, then concatenate
    auto sin_emb = mx::sin(scaled_time);
    auto cos_emb = mx::cos(scaled_time);
    auto pos_emb = mx::concatenate({sin_emb, cos_emb}, 1);

    // Convert to float16 to match Python (which uses dtype=mx.float16)
    pos_emb = mx::astype(pos_emb, mx::float16);
    mx::eval(pos_emb);

    // Store in float32 buffer (will be converted back when used)
    // Note: We store float32 because positional_embedding_data_ is std::vector<float>
    // The conversion float32->float16->float32 matches Python's behavior
    auto pos_emb_f32 = mx::astype(pos_emb, mx::float32);
    mx::eval(pos_emb_f32);

    positional_embedding_data_.resize(length * channels);
    auto data_ptr = pos_emb_f32.data<float>();
    std::memcpy(positional_embedding_data_.data(), data_ptr, length * channels * sizeof(float));
}

mx::array AudioEncoder::encode(const mx::array& mel) {
    // Input: [batch, n_frames, n_mels] or [n_frames, n_mels]
    mx::array x = mel;
    if (x.ndim() == 2) {
        x = mx::expand_dims(x, 0);  // Add batch dimension
    }

    // Debug: save input mel
    save_debug_array("mel", x);

    // Get conv weights
    auto conv1_weight = weights_->get("encoder.conv1.weight");
    auto conv1_bias = weights_->get("encoder.conv1.bias");
    auto conv2_weight = weights_->get("encoder.conv2.weight");
    auto conv2_bias = weights_->get("encoder.conv2.bias");

    // Conv1: [batch, n_frames, n_mels] -> [batch, n_frames, n_state]
    x = mx::conv1d(x, conv1_weight, /*stride=*/1, /*padding=*/1);
    x = x + conv1_bias;
    x = gelu(x);

    // Debug: save after conv1
    save_debug_array("after_conv1", x);

    // Conv2: [batch, n_frames, n_state] -> [batch, n_frames/2, n_state]
    x = mx::conv1d(x, conv2_weight, /*stride=*/2, /*padding=*/1);
    x = x + conv2_bias;
    x = gelu(x);

    // Debug: save after conv2
    save_debug_array("after_conv2", x);

    // Add positional embedding
    int seq_len = static_cast<int>(x.shape()[1]);
    auto pos_emb = mx::array(
        positional_embedding_data_.data(),
        {seq_len, config_.n_audio_state}
    );

    // Debug: save positional embedding
    save_debug_array("positional_embedding", pos_emb);

    x = x + pos_emb;

    // Debug: save after pos_emb
    save_debug_array("after_pos_emb", x);

    // Transformer layers
    for (int i = 0; i < config_.n_audio_layer; ++i) {
        x = encoder_layer(x, i);

        // Debug: save more layers around the divergence point (layer 20)
        if (i == 0 || i == config_.n_audio_layer - 1 || i % 4 == 0 ||
            (i >= 17 && i <= 21)) {
            char name[32];
            std::snprintf(name, sizeof(name), "after_layer_%02d", i);
            save_debug_array(name, x);
        }
    }

    // Final layer norm
    auto ln_post_weight = weights_->get("encoder.ln_post.weight");
    auto ln_post_bias = weights_->get("encoder.ln_post.bias");
    x = layer_norm(x, ln_post_weight, ln_post_bias, config_.layer_norm_eps);

    // Debug: save encoder output
    save_debug_array("encoder_output", x);

    return x;
}

mx::array AudioEncoder::encoder_layer(const mx::array& x, int layer_idx) {
    std::string prefix = "encoder.blocks." + std::to_string(layer_idx) + ".";

    // Get layer norm weights
    auto attn_ln_weight = weights_->get(prefix + "attn_ln.weight");
    auto attn_ln_bias = weights_->get(prefix + "attn_ln.bias");
    auto mlp_ln_weight = weights_->get(prefix + "mlp_ln.weight");
    auto mlp_ln_bias = weights_->get(prefix + "mlp_ln.bias");

    // Self-attention with residual
    auto attn_input = layer_norm(x, attn_ln_weight, attn_ln_bias, config_.layer_norm_eps);
    auto attn_out = self_attention(attn_input, layer_idx);
    auto h = x + attn_out;

    // MLP with residual
    auto mlp_input = layer_norm(h, mlp_ln_weight, mlp_ln_bias, config_.layer_norm_eps);
    auto mlp_out = mlp(mlp_input, layer_idx);
    h = h + mlp_out;

    return h;
}

mx::array AudioEncoder::self_attention(const mx::array& x, int layer_idx) {
    std::string prefix = "encoder.blocks." + std::to_string(layer_idx) + ".";

    int batch = static_cast<int>(x.shape()[0]);
    int seq_len = static_cast<int>(x.shape()[1]);
    int n_head = config_.n_audio_head;
    int head_dim = config_.n_audio_state / n_head;

    // Get attention weights
    auto q_weight = weights_->get(prefix + "attn.query.weight");
    auto q_bias = weights_->get(prefix + "attn.query.bias");
    auto k_weight = weights_->get(prefix + "attn.key.weight");
    auto v_weight = weights_->get(prefix + "attn.value.weight");
    auto v_bias = weights_->get(prefix + "attn.value.bias");
    auto out_weight = weights_->get(prefix + "attn.out.weight");
    auto out_bias = weights_->get(prefix + "attn.out.bias");

    // QKV projections
    auto q = mx::matmul(x, mx::transpose(q_weight)) + q_bias;
    auto k = mx::matmul(x, mx::transpose(k_weight));
    auto v = mx::matmul(x, mx::transpose(v_weight)) + v_bias;

    // Reshape to [batch, n_heads, seq_len, head_dim]
    q = mx::transpose(mx::reshape(q, {batch, seq_len, n_head, head_dim}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {batch, seq_len, n_head, head_dim}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {batch, seq_len, n_head, head_dim}), {0, 2, 1, 3});

    // Attention (no mask for encoder self-attention)
    auto attn_out = scaled_dot_product_attention(q, k, v);

    // Reshape back to [batch, seq_len, n_state]
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {batch, seq_len, config_.n_audio_state});

    // Output projection
    return mx::matmul(attn_out, mx::transpose(out_weight)) + out_bias;
}

mx::array AudioEncoder::mlp(const mx::array& x, int layer_idx) {
    std::string prefix = "encoder.blocks." + std::to_string(layer_idx) + ".";

    // MLX community format uses mlp1/mlp2 instead of mlp.0/mlp.2
    auto mlp1_weight = weights_->get(prefix + "mlp1.weight");
    auto mlp1_bias = weights_->get(prefix + "mlp1.bias");
    auto mlp2_weight = weights_->get(prefix + "mlp2.weight");
    auto mlp2_bias = weights_->get(prefix + "mlp2.bias");

    // Use compiled MLP for better performance (fuses matmul+gelu+matmul)
    return compiled_mlp(x, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias);
}

// ============================================================================
// TextDecoder
// ============================================================================

void TextDecoder::init(const WhisperConfig& config, const Weights* weights) {
    config_ = config;
    weights_ = weights;

    // Precompute causal mask as float data (upper triangular with -inf)
    // This avoids storing mx::array as member (no default constructor)
    causal_mask_data_.resize(config.n_text_ctx * config.n_text_ctx);
    for (int i = 0; i < config.n_text_ctx; ++i) {
        for (int j = 0; j < config.n_text_ctx; ++j) {
            // Lower triangle and diagonal = 0, upper triangle = -inf
            causal_mask_data_[i * config.n_text_ctx + j] = (j > i) ? -INFINITY : 0.0f;
        }
    }
}

// Helper to save array to binary file for layer comparison
static void save_layer_array(const std::string& name, const mx::array& arr) {
    static bool debug_step = std::getenv("DEBUG_DECODER_STEP") != nullptr;
    if (!debug_step) return;

    mx::eval(arr);
    auto f32_arr = mx::astype(arr, mx::float32);
    mx::eval(f32_arr);

    std::string path = "/tmp/cpp_" + name + ".bin";
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[DEBUG] Failed to open " << path << "\n";
        return;
    }

    // Write shape
    int ndim = static_cast<int>(f32_arr.shape().size());
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
    for (int i = 0; i < ndim; ++i) {
        int dim = static_cast<int>(f32_arr.shape()[i]);
        file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    }

    // Write data
    size_t numel = f32_arr.size();
    file.write(reinterpret_cast<const char*>(f32_arr.data<float>()), numel * sizeof(float));
    file.close();

    auto min_val = mx::min(f32_arr);
    auto max_val = mx::max(f32_arr);
    mx::eval(min_val, max_val);
    std::cerr << "[DEBUG CPP] Saved " << path << ": min=" << min_val.item<float>()
              << " max=" << max_val.item<float>() << "\n";
}

mx::array TextDecoder::decode(
    const mx::array& tokens,
    const mx::array& encoder_output,
    DecoderKVCache* kv_cache,
    CrossKVCache* cross_cache
) {
    // Input: [batch, seq_len]
    int seq_len = static_cast<int>(tokens.shape()[1]);

    // Get embeddings
    auto token_embedding = weights_->get("decoder.token_embedding.weight");
    auto positional_embedding = weights_->get("decoder.positional_embedding");

    // Debug: check token embedding stats
    static bool debug_decoder = std::getenv("DEBUG_DECODER_LAYERS") != nullptr;

    // Step-specific debug: DEBUG_DECODER_STEP=30 for divergence point
    static int debug_step = std::getenv("DEBUG_DECODER_STEP") ?
        std::atoi(std::getenv("DEBUG_DECODER_STEP")) : -1;
    static int current_step = 0;
    int this_step = current_step++;
    bool debug_this_step = (debug_step >= 0 && this_step == debug_step);
    if (debug_decoder && (kv_cache == nullptr || kv_cache->empty())) {
        mx::eval(token_embedding);
        // Cast to float32 for proper min/max computation
        auto te_f32 = mx::astype(token_embedding, mx::float32);
        mx::eval(te_f32);
        auto te_min = mx::min(te_f32);
        auto te_max = mx::max(te_f32);
        mx::eval(te_min, te_max);
        std::cout << "[DEBUG DECODER] token_embedding dtype=" << token_embedding.dtype()
                  << " shape=[" << token_embedding.shape()[0] << "," << token_embedding.shape()[1]
                  << "] min=" << te_min.item<float>() << " max=" << te_max.item<float>() << "\n";

        mx::eval(positional_embedding);
        auto pe_f32 = mx::astype(positional_embedding, mx::float32);
        mx::eval(pe_f32);
        auto pe_min = mx::min(pe_f32);
        auto pe_max = mx::max(pe_f32);
        mx::eval(pe_min, pe_max);
        std::cout << "[DEBUG DECODER] positional_embedding dtype=" << positional_embedding.dtype()
                  << " shape=[" << positional_embedding.shape()[0] << "," << positional_embedding.shape()[1]
                  << "] min=" << pe_min.item<float>() << " max=" << pe_max.item<float>() << "\n";

        // Print the input tokens
        mx::eval(tokens);
        std::cout << "[DEBUG DECODER] input tokens: [";
        auto tok_data = tokens.data<int32_t>();
        for (int i = 0; i < static_cast<int>(tokens.shape()[1]); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << tok_data[i];
        }
        std::cout << "] (n=" << tokens.shape()[1] << ")\n";
    }

    // Token embedding lookup
    if (debug_decoder && (kv_cache == nullptr || kv_cache->empty())) {
        std::cout << "[DEBUG DECODER] tokens dtype=" << tokens.dtype()
                  << " shape=[" << tokens.shape()[0] << "," << tokens.shape()[1] << "]\n";
    }

    auto x = mx::take(token_embedding, tokens, 0);

    if (debug_decoder && (kv_cache == nullptr || kv_cache->empty())) {
        mx::eval(x);
        // Cast to float32 for accurate min/max
        auto x_f32 = mx::astype(x, mx::float32);
        mx::eval(x_f32);
        auto x_min = mx::min(x_f32);
        auto x_max = mx::max(x_f32);
        mx::eval(x_min, x_max);
        std::cout << "[DEBUG DECODER] after token lookup x dtype=" << x.dtype()
                  << " shape=[" << x.shape()[0] << "," << x.shape()[1] << "," << x.shape()[2]
                  << "] min=" << x_min.item<float>() << " max=" << x_max.item<float>() << "\n";
    }

    // Add positional embedding
    int offset = (kv_cache && !kv_cache->empty()) ? kv_cache->seq_len() : 0;
    auto pos_emb = mx::slice(positional_embedding, {offset, 0}, {offset + seq_len, config_.n_text_state});

    if (debug_decoder && (kv_cache == nullptr || kv_cache->empty())) {
        mx::eval(pos_emb);
        auto pos_f32 = mx::astype(pos_emb, mx::float32);
        mx::eval(pos_f32);
        auto pos_min = mx::min(pos_f32);
        auto pos_max = mx::max(pos_f32);
        mx::eval(pos_min, pos_max);
        std::cout << "[DEBUG DECODER] pos_emb slice shape=["
                  << pos_emb.shape()[0] << "," << pos_emb.shape()[1]
                  << "] min=" << pos_min.item<float>() << " max=" << pos_max.item<float>() << "\n";
        std::cout << "[DEBUG DECODER] x before add shape=[" << x.shape()[0] << "," << x.shape()[1]
                  << "," << x.shape()[2] << "] pos_emb shape=[" << pos_emb.shape()[0]
                  << "," << pos_emb.shape()[1] << "]\n";
    }

    x = x + pos_emb;

    if (debug_decoder && (kv_cache == nullptr || kv_cache->empty())) {
        mx::eval(x);
        auto x_f32 = mx::astype(x, mx::float32);
        mx::eval(x_f32);
        auto x_min = mx::min(x_f32);
        auto x_max = mx::max(x_f32);
        mx::eval(x_min, x_max);
        std::cout << "[DEBUG DECODER] after pos_emb x min=" << x_min.item<float>()
                  << " max=" << x_max.item<float>() << "\n";
    }

    // Step-specific debug: save embedding output
    if (debug_this_step) {
        std::cerr << "\n=== C++ STEP " << this_step << " DECODER DEBUG ===\n";
        std::cerr << "Offset: " << offset << "\n";
        mx::eval(tokens);
        std::cerr << "Input tokens: [";
        auto tok_data = tokens.data<int32_t>();
        for (int i = 0; i < seq_len; ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << tok_data[i];
        }
        std::cerr << "]\n";
        save_layer_array("step" + std::to_string(this_step) + "_embedding", x);
    }

    // Transformer layers
    for (int i = 0; i < config_.n_text_layer; ++i) {
        x = decoder_layer(x, encoder_output, i, kv_cache, cross_cache);

        // Save first and last layer output at debug step
        if (debug_this_step && (i == 0 || i == config_.n_text_layer - 1)) {
            save_layer_array("step" + std::to_string(this_step) + "_layer" + std::to_string(i), x);
        }
    }

    // Final layer norm
    auto ln_weight = weights_->get("decoder.ln.weight");
    auto ln_bias = weights_->get("decoder.ln.bias");
    x = layer_norm(x, ln_weight, ln_bias, config_.layer_norm_eps);

    // Step-specific debug: save hidden states before projection
    if (debug_this_step) {
        save_layer_array("step" + std::to_string(this_step) + "_hidden", x);
    }

    // Project to vocabulary using tied weights
    auto logits = mx::matmul(x, mx::transpose(token_embedding));

    // Step-specific debug: save logits
    if (debug_this_step) {
        save_layer_array("step" + std::to_string(this_step) + "_logits", logits);
        mx::eval(logits);
        auto logits_f32 = mx::astype(logits, mx::float32);
        mx::eval(logits_f32);

        // Print key token logits
        auto logits_ptr = logits_f32.data<float>();
        int vocab_offset = static_cast<int>(logits_f32.shape()[1] - 1) * config_.n_vocab;
        std::cerr << "\nKey tokens at step " << this_step << ":\n";
        std::cerr << "  Token 13 (period): " << logits_ptr[vocab_offset + 13] << "\n";
        std::cerr << "  Token 2221 (Mr): " << logits_ptr[vocab_offset + 2221] << "\n";
    }

    return logits;
}

mx::array TextDecoder::decoder_layer(
    const mx::array& x,
    const mx::array& encoder_output,
    int layer_idx,
    DecoderKVCache* kv_cache,
    CrossKVCache* cross_cache
) {
    std::string prefix = "decoder.blocks." + std::to_string(layer_idx) + ".";

    // Get layer norm weights
    auto self_attn_ln_weight = weights_->get(prefix + "attn_ln.weight");
    auto self_attn_ln_bias = weights_->get(prefix + "attn_ln.bias");
    auto cross_attn_ln_weight = weights_->get(prefix + "cross_attn_ln.weight");
    auto cross_attn_ln_bias = weights_->get(prefix + "cross_attn_ln.bias");
    auto mlp_ln_weight = weights_->get(prefix + "mlp_ln.weight");
    auto mlp_ln_bias = weights_->get(prefix + "mlp_ln.bias");

    // Self-attention with residual
    auto attn_input = layer_norm(x, self_attn_ln_weight, self_attn_ln_bias, config_.layer_norm_eps);
    auto attn_out = self_attention(attn_input, layer_idx, kv_cache);
    auto h = x + attn_out;

    // Cross-attention with residual
    auto cross_input = layer_norm(h, cross_attn_ln_weight, cross_attn_ln_bias, config_.layer_norm_eps);
    auto cross_out = cross_attention(cross_input, encoder_output, layer_idx, cross_cache);
    h = h + cross_out;

    // MLP with residual
    auto mlp_input = layer_norm(h, mlp_ln_weight, mlp_ln_bias, config_.layer_norm_eps);
    auto mlp_out = mlp(mlp_input, layer_idx);
    h = h + mlp_out;

    return h;
}

mx::array TextDecoder::self_attention(
    const mx::array& x,
    int layer_idx,
    DecoderKVCache* kv_cache
) {
    std::string prefix = "decoder.blocks." + std::to_string(layer_idx) + ".";

    int batch = static_cast<int>(x.shape()[0]);
    int seq_len = static_cast<int>(x.shape()[1]);
    int n_head = config_.n_text_head;
    int head_dim = config_.n_text_state / n_head;

    // Get attention weights
    auto q_weight = weights_->get(prefix + "attn.query.weight");
    auto q_bias = weights_->get(prefix + "attn.query.bias");
    auto k_weight = weights_->get(prefix + "attn.key.weight");
    auto v_weight = weights_->get(prefix + "attn.value.weight");
    auto v_bias = weights_->get(prefix + "attn.value.bias");
    auto out_weight = weights_->get(prefix + "attn.out.weight");
    auto out_bias = weights_->get(prefix + "attn.out.bias");

    // QKV projections
    auto q = mx::matmul(x, mx::transpose(q_weight)) + q_bias;
    auto k = mx::matmul(x, mx::transpose(k_weight));
    auto v = mx::matmul(x, mx::transpose(v_weight)) + v_bias;

    // Reshape to [batch, n_heads, seq_len, head_dim]
    q = mx::transpose(mx::reshape(q, {batch, seq_len, n_head, head_dim}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {batch, seq_len, n_head, head_dim}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {batch, seq_len, n_head, head_dim}), {0, 2, 1, 3});

    // Update KV cache (using push_back pattern like translation_model)
    // Keep KV cache in native dtype (float16) to match Python implementation
    if (kv_cache != nullptr) {
        if (kv_cache->keys.size() > static_cast<size_t>(layer_idx)) {
            // Append to existing cache (native dtype)
            k = mx::concatenate({kv_cache->keys[layer_idx], k}, 2);
            v = mx::concatenate({kv_cache->values[layer_idx], v}, 2);
            // Update cache in place
            kv_cache->keys[layer_idx] = k;
            kv_cache->values[layer_idx] = v;
        } else {
            // Extend cache by pushing new entries
            while (kv_cache->keys.size() < static_cast<size_t>(layer_idx)) {
                // Fill gap with dummy arrays (should not happen in proper usage)
                kv_cache->keys.push_back(mx::zeros({1}, k.dtype()));
                kv_cache->values.push_back(mx::zeros({1}, v.dtype()));
            }
            kv_cache->keys.push_back(k);
            kv_cache->values.push_back(v);
        }
    }

    // Use causal masking via mask_mode parameter
    // For incremental decoding (single token), no mask needed as there's only one query position
    std::string mask_mode = (seq_len > 1) ? "causal" : "";

    // Attention with causal mask
    auto attn_out = scaled_dot_product_attention(q, k, v, mask_mode);

    // Reshape back to [batch, seq_len, n_state]
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {batch, seq_len, config_.n_text_state});

    // Output projection
    return mx::matmul(attn_out, mx::transpose(out_weight)) + out_bias;
}

mx::array TextDecoder::cross_attention(
    const mx::array& x,
    const mx::array& encoder_output,
    int layer_idx,
    CrossKVCache* cross_cache
) {
    std::string prefix = "decoder.blocks." + std::to_string(layer_idx) + ".";

    int batch = static_cast<int>(x.shape()[0]);
    int seq_len = static_cast<int>(x.shape()[1]);
    int enc_len = static_cast<int>(encoder_output.shape()[1]);
    int n_head = config_.n_text_head;
    int head_dim = config_.n_text_state / n_head;

    // Get attention weights
    auto q_weight = weights_->get(prefix + "cross_attn.query.weight");
    auto q_bias = weights_->get(prefix + "cross_attn.query.bias");
    auto k_weight = weights_->get(prefix + "cross_attn.key.weight");
    auto v_weight = weights_->get(prefix + "cross_attn.value.weight");
    auto v_bias = weights_->get(prefix + "cross_attn.value.bias");
    auto out_weight = weights_->get(prefix + "cross_attn.out.weight");
    auto out_bias = weights_->get(prefix + "cross_attn.out.bias");

    // Query projection
    auto q = mx::matmul(x, mx::transpose(q_weight)) + q_bias;
    q = mx::transpose(mx::reshape(q, {batch, seq_len, n_head, head_dim}), {0, 2, 1, 3});

    mx::array k = mx::zeros({1}, mx::float32);  // Placeholder
    mx::array v = mx::zeros({1}, mx::float32);

    // Check if we can reuse cached K/V
    if (cross_cache != nullptr && cross_cache->keys.size() > static_cast<size_t>(layer_idx)) {
        k = cross_cache->keys[layer_idx];
        v = cross_cache->values[layer_idx];

        // GAP 73: Handle batch size mismatch for batched beam search
        // Cross cache was created with batch=1, but query may have batch=n_beams
        int cache_batch = static_cast<int>(k.shape()[0]);
        if (cache_batch == 1 && batch > 1) {
            // Broadcast by repeating along batch dimension
            // k/v shape: [1, n_heads, enc_len, head_dim] -> [batch, n_heads, enc_len, head_dim]
            k = mx::repeat(k, batch, 0);
            v = mx::repeat(v, batch, 0);
        }
    } else {
        // Compute K/V projections from encoder output
        k = mx::matmul(encoder_output, mx::transpose(k_weight));
        v = mx::matmul(encoder_output, mx::transpose(v_weight)) + v_bias;

        // Reshape to [batch, n_heads, enc_len, head_dim]
        k = mx::transpose(mx::reshape(k, {batch, enc_len, n_head, head_dim}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {batch, enc_len, n_head, head_dim}), {0, 2, 1, 3});

        // Keep native dtype (float16) to match Python implementation
        // Cache for future decode steps
        if (cross_cache != nullptr) {
            while (cross_cache->keys.size() < static_cast<size_t>(layer_idx)) {
                cross_cache->keys.push_back(mx::zeros({1}, k.dtype()));
                cross_cache->values.push_back(mx::zeros({1}, v.dtype()));
            }
            cross_cache->keys.push_back(k);
            cross_cache->values.push_back(v);
        }
    }

    // Attention (no mask for cross-attention)
    auto attn_out = scaled_dot_product_attention(q, k, v);

    // Reshape back to [batch, seq_len, n_state]
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {batch, seq_len, config_.n_text_state});

    // Output projection
    return mx::matmul(attn_out, mx::transpose(out_weight)) + out_bias;
}

std::pair<mx::array, mx::array> TextDecoder::cross_attention_with_weights(
    const mx::array& x,
    const mx::array& encoder_output,
    int layer_idx
) {
    // This version manually computes attention to extract weights
    // It's slower than SDPA but needed for DTW alignment
    std::string prefix = "decoder.blocks." + std::to_string(layer_idx) + ".";

    int batch = static_cast<int>(x.shape()[0]);
    int seq_len = static_cast<int>(x.shape()[1]);
    int enc_len = static_cast<int>(encoder_output.shape()[1]);
    int n_head = config_.n_text_head;
    int head_dim = config_.n_text_state / n_head;

    // Get attention weights
    auto q_weight = weights_->get(prefix + "cross_attn.query.weight");
    auto q_bias = weights_->get(prefix + "cross_attn.query.bias");
    auto k_weight = weights_->get(prefix + "cross_attn.key.weight");
    auto v_weight = weights_->get(prefix + "cross_attn.value.weight");
    auto v_bias = weights_->get(prefix + "cross_attn.value.bias");
    auto out_weight = weights_->get(prefix + "cross_attn.out.weight");
    auto out_bias = weights_->get(prefix + "cross_attn.out.bias");

    // Query projection
    auto q = mx::matmul(x, mx::transpose(q_weight)) + q_bias;
    q = mx::transpose(mx::reshape(q, {batch, seq_len, n_head, head_dim}), {0, 2, 1, 3});

    // Key/Value projections from encoder output
    auto k = mx::matmul(encoder_output, mx::transpose(k_weight));
    auto v = mx::matmul(encoder_output, mx::transpose(v_weight)) + v_bias;

    // Reshape to [batch, n_heads, enc_len, head_dim]
    k = mx::transpose(mx::reshape(k, {batch, enc_len, n_head, head_dim}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {batch, enc_len, n_head, head_dim}), {0, 2, 1, 3});

    // Manual attention computation to extract weights
    // Q * K^T / sqrt(d_k)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto scores = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})) * scale;
    // scores: [batch, n_head, seq_len, enc_len]

    // Softmax over encoder dimension
    // Python: w = mx.softmax(qk, axis=-1, precise=True)
    auto attn_weights = mx::softmax(scores, -1, true);

    // Weighted sum of values
    auto attn_out = mx::matmul(attn_weights, v);
    // attn_out: [batch, n_head, seq_len, head_dim]

    // Reshape back to [batch, seq_len, n_state]
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {batch, seq_len, config_.n_text_state});

    // Output projection
    auto output = mx::matmul(attn_out, mx::transpose(out_weight)) + out_bias;

    return {output, attn_weights};
}

std::pair<mx::array, mx::array> TextDecoder::decoder_layer_with_attention(
    const mx::array& x,
    const mx::array& encoder_output,
    int layer_idx
) {
    std::string prefix = "decoder.blocks." + std::to_string(layer_idx) + ".";

    // Get layer norm weights
    auto self_attn_ln_weight = weights_->get(prefix + "attn_ln.weight");
    auto self_attn_ln_bias = weights_->get(prefix + "attn_ln.bias");
    auto cross_attn_ln_weight = weights_->get(prefix + "cross_attn_ln.weight");
    auto cross_attn_ln_bias = weights_->get(prefix + "cross_attn_ln.bias");
    auto mlp_ln_weight = weights_->get(prefix + "mlp_ln.weight");
    auto mlp_ln_bias = weights_->get(prefix + "mlp_ln.bias");

    // Self-attention with residual (without KV cache since we decode entire sequence)
    auto attn_input = layer_norm(x, self_attn_ln_weight, self_attn_ln_bias, config_.layer_norm_eps);
    auto attn_out = self_attention(attn_input, layer_idx, nullptr);
    auto h = x + attn_out;

    // Cross-attention with residual - extract attention weights
    auto cross_input = layer_norm(h, cross_attn_ln_weight, cross_attn_ln_bias, config_.layer_norm_eps);
    auto [cross_out, cross_attn_weights] = cross_attention_with_weights(cross_input, encoder_output, layer_idx);
    h = h + cross_out;

    // MLP with residual
    auto mlp_input = layer_norm(h, mlp_ln_weight, mlp_ln_bias, config_.layer_norm_eps);
    auto mlp_out = mlp(mlp_input, layer_idx);
    h = h + mlp_out;

    return {h, cross_attn_weights};
}

DecodeWithAttentionResult TextDecoder::decode_with_attention(
    const mx::array& tokens,
    const mx::array& encoder_output
) {
    // Get token embeddings (handle naming differences between models)
    auto token_embedding = weights_->has("decoder.embed_tokens.weight")
        ? weights_->get("decoder.embed_tokens.weight")
        : weights_->get("decoder.token_embedding.weight");

    // Embed tokens
    auto x = mx::take(token_embedding, tokens, 0);

    // Add positional embeddings
    int seq_len = static_cast<int>(tokens.shape()[1]);
    auto pos_emb = weights_->get("decoder.positional_embedding");
    x = x + mx::slice(pos_emb, {0, 0}, {seq_len, config_.n_text_state});

    // Collect cross-attention weights from all layers
    std::vector<mx::array> all_cross_attn_weights;

    // Transformer layers
    for (int i = 0; i < config_.n_text_layer; ++i) {
        auto [h, cross_attn_weights] = decoder_layer_with_attention(x, encoder_output, i);
        x = h;
        all_cross_attn_weights.push_back(cross_attn_weights);
    }

    // Final layer norm
    auto ln_weight = weights_->get("decoder.ln.weight");
    auto ln_bias = weights_->get("decoder.ln.bias");
    x = layer_norm(x, ln_weight, ln_bias, config_.layer_norm_eps);

    // Project to vocabulary using tied weights
    auto logits = mx::matmul(x, mx::transpose(token_embedding));

    return {logits, all_cross_attn_weights};
}

mx::array TextDecoder::mlp(const mx::array& x, int layer_idx) {
    std::string prefix = "decoder.blocks." + std::to_string(layer_idx) + ".";

    // MLX community format uses mlp1/mlp2 instead of mlp.0/mlp.2
    auto mlp1_weight = weights_->get(prefix + "mlp1.weight");
    auto mlp1_bias = weights_->get(prefix + "mlp1.bias");
    auto mlp2_weight = weights_->get(prefix + "mlp2.weight");
    auto mlp2_bias = weights_->get(prefix + "mlp2.bias");

    // NOTE: Decoder uses small inputs (1 token at a time), so compilation overhead
    // outweighs benefits. Keep uncompiled for better autoregressive performance.
    auto h = mx::matmul(x, mx::transpose(mlp1_weight)) + mlp1_bias;
    h = gelu(h);
    h = mx::matmul(h, mx::transpose(mlp2_weight)) + mlp2_bias;

    return h;
}

// ============================================================================
// WhisperModel
// ============================================================================

WhisperModel::WhisperModel()
    : encoder_(std::make_unique<AudioEncoder>())
    , decoder_(std::make_unique<TextDecoder>())
    , weights_(std::make_unique<Weights>())
    , loaded_(false)
{}

WhisperModel::~WhisperModel() = default;
WhisperModel::WhisperModel(WhisperModel&&) noexcept = default;
WhisperModel& WhisperModel::operator=(WhisperModel&&) noexcept = default;

WhisperModel WhisperModel::load(const std::string& model_path) {
    WhisperModel model;

    // Load config
    std::string config_path = model_path + "/config.json";
    model.config_ = WhisperConfig::load(config_path);

    // Load weights - try multiple naming conventions
    std::string weights_path = model_path + "/model.safetensors";
    if (std::ifstream(weights_path).good()) {
        model.weights_->load(weights_path);
    } else {
        // Try weights.safetensors (MLX community format)
        weights_path = model_path + "/weights.safetensors";
        if (std::ifstream(weights_path).good()) {
            model.weights_->load(weights_path);
        } else {
            // Try weights.npz
            weights_path = model_path + "/weights.npz";
            if (std::ifstream(weights_path).good()) {
                model.weights_->load(weights_path);
            } else {
                throw std::runtime_error("No weights file found in: " + model_path);
            }
        }
    }

    // GAP 26: Load alignment_heads from weights if present
    // These are the specific (layer, head) pairs used for word timestamp DTW alignment
    if (model.weights_->has("alignment_heads")) {
        auto heads_array = model.weights_->get("alignment_heads");
        mx::eval(heads_array);
        // Shape should be [N, 2] where N is number of alignment heads
        // Each row is [layer_idx, head_idx]
        if (heads_array.ndim() == 2 && heads_array.shape()[1] == 2) {
            int n_heads = static_cast<int>(heads_array.shape()[0]);
            auto heads_int = mx::astype(heads_array, mx::int32);
            mx::eval(heads_int);
            const int32_t* data = heads_int.data<int32_t>();
            model.config_.alignment_heads.clear();
            model.config_.alignment_heads.reserve(n_heads);
            for (int i = 0; i < n_heads; ++i) {
                int layer = data[i * 2];
                int head = data[i * 2 + 1];
                model.config_.alignment_heads.push_back({layer, head});
            }
            std::cerr << "[Whisper] Loaded " << n_heads << " alignment heads for word timestamps\n";
        }
    } else {
        // GAP 50: Try to use preset based on model name if no alignment_heads in weights
        // Use model.config_.name which should be set from config.json or model path
        std::string model_name_for_preset = model.config_.name;
        if (model_name_for_preset.empty()) {
            // Extract model name from path (last component)
            size_t last_slash = model_path.find_last_of("/\\");
            model_name_for_preset = (last_slash != std::string::npos) ?
                model_path.substr(last_slash + 1) : model_path;
        }
        auto preset = get_preset_from_model_name(model_name_for_preset);
        if (preset != AlignmentHeadsPreset::NONE) {
            model.config_.alignment_heads = get_alignment_heads_preset(preset);
            model.config_.dtw_aheads_preset = preset;
            std::cerr << "[Whisper] Using preset alignment heads for " << model_name_for_preset
                      << " (" << model.config_.alignment_heads.size() << " heads)\n";
        } else {
            // Default: use all heads from last half of layers (fallback)
            std::cerr << "[Whisper] No alignment_heads in weights, using default (last half layers)\n";
        }
    }

    // Initialize encoder and decoder with pointer to weights
    model.encoder_->init(model.config_, model.weights_.get());
    model.decoder_->init(model.config_, model.weights_.get());

    // Initialize language tokens
    model.init_language_tokens();

    model.loaded_ = true;
    return model;
}

mx::array WhisperModel::encode(const mx::array& mel) {
    return encoder_->encode(mel);
}

mx::array WhisperModel::decode(
    const mx::array& tokens,
    const mx::array& encoder_output,
    DecoderKVCache* kv_cache,
    CrossKVCache* cross_cache
) {
    return decoder_->decode(tokens, encoder_output, kv_cache, cross_cache);
}

std::vector<int> WhisperModel::generate(
    const mx::array& mel,
    const std::string& language,
    const std::string& task,
    int max_tokens,
    float audio_duration_sec,
    float* avg_logprob_out,
    float* no_speech_prob_out,
    const std::vector<int>* prompt_tokens,
    float temperature,
    float repetition_penalty,
    float max_initial_timestamp,
    bool without_timestamps,
    const std::vector<int>* prefix_tokens,
    int sample_len,
    bool skip_logprobs,
    const std::vector<int>* suppress_regex_tokens,  // GAP 43
    bool tdrz_enable  // GAP 41: tinydiarize speaker turn detection
) {
    // =========================================================================
    // PROPER WHISPER DECODING PROTOCOL
    // Reference: OpenAI Whisper decoding.py, whisper.cpp
    // =========================================================================

    // Encode audio
    auto encoder_output = encode(mel);
    // Keep encoder output in native dtype (float16) to match Python exactly
    // Python encoder uses dtype=float16 and outputs float16
    mx::eval(encoder_output);

    // GAP 58: task="lang_id" mode - just return detected language token
    // Python returns DecodingResult with language and language_probs
    // C++ returns vector containing the language token ID
    if (task == "lang_id") {
        auto lang_result = detect_language(encoder_output);
        return {lang_result.language_token};
    }

    // Debug: Save encoder output for Python comparison
    if (std::getenv("DEBUG_ENCODER")) {
        std::cerr << "[DEBUG] Encoder output shape: " << encoder_output.shape()[0]
                  << "x" << encoder_output.shape()[1] << "x" << encoder_output.shape()[2] << "\n";

        // Save encoder output to file
        const float* enc_data = encoder_output.data<float>();
        std::ofstream enc_file("/tmp/cpp_encoder_0004.bin", std::ios::binary);
        enc_file.write(reinterpret_cast<const char*>(enc_data),
                       encoder_output.size() * sizeof(float));
        enc_file.close();
        std::cerr << "[DEBUG] Saved encoder output to /tmp/cpp_encoder_0004.bin\n";

        // Print first few values for quick comparison
        std::cerr << "[DEBUG] First 5 values: ";
        for (int i = 0; i < 5; ++i) {
            std::cerr << enc_data[i] << " ";
        }
        std::cerr << "\n";
    }

    // Initialize caches
    DecoderKVCache kv_cache;
    CrossKVCache cross_cache;

    // Special token constants (use config values for v3 compatibility)
    const int timestamp_begin = config_.timestamp_begin;
    const int no_timestamps_token = config_.no_timestamps_token;

    // Auto-detect language if not specified or set to "auto"
    std::string effective_language = language;
    if (effective_language.empty() || effective_language == "auto") {
        auto lang_result = detect_language(encoder_output);
        effective_language = lang_result.language;
        if (std::getenv("DEBUG_WHISPER")) {
            std::cerr << "[DEBUG] Auto-detected language: " << effective_language
                      << " (p=" << lang_result.probability << ")\n";
        }
    }

    // Build initial token sequence
    // GAP 6: With prompt tokens: [sot_prev, prompt_tokens..., SOT, language, task]
    //        Without prompt:     [SOT, language, task]
    // Reference: Python whisper/decoding.py _get_initial_tokens()
    std::vector<int> tokens;

    // Add prompt tokens if provided (GAP 6: initial_prompt support)
    if (prompt_tokens && !prompt_tokens->empty()) {
        // Prepend sot_prev marker and prompt tokens
        tokens.push_back(config_.sot_prev_token);
        // Truncate prompt to n_text_ctx // 2 - 1 tokens to leave room for transcription
        size_t max_prompt_len = (config_.n_text_ctx / 2) - 1;
        size_t prompt_len = std::min(prompt_tokens->size(), max_prompt_len);
        // Take last prompt_len tokens (keep most recent context)
        size_t start_idx = prompt_tokens->size() - prompt_len;
        tokens.insert(tokens.end(),
                      prompt_tokens->begin() + start_idx,
                      prompt_tokens->end());
    }

    // Add SOT token
    tokens.push_back(config_.sot_token);

    // Add language token
    int lang_token = get_language_token(effective_language);
    tokens.push_back(lang_token);

    // Add task token
    if (task == "translate") {
        tokens.push_back(config_.translate_token);
    } else {
        tokens.push_back(config_.transcribe_token);
    }

    // GAP: without_timestamps mode - add <|notimestamps|> token after task token
    // This tells the model to generate text without timestamp tokens
    if (without_timestamps) {
        tokens.push_back(config_.no_timestamps_token);
    }

    // GAP: decoder prefix - add prefix tokens after SOT sequence but before sampling
    // This is different from prompt (which comes BEFORE SOT)
    // Python: tokens = tokens + prefix_tokens (after sot_sequence)
    if (prefix_tokens && !prefix_tokens->empty()) {
        // Calculate effective sample_len for prefix truncation
        int effective_sample_len = (sample_len > 0) ? sample_len : (config_.n_text_ctx / 2);
        int max_prefix_len = config_.n_text_ctx / 2 - effective_sample_len;

        if (max_prefix_len > 0) {
            size_t prefix_len = std::min(prefix_tokens->size(), static_cast<size_t>(max_prefix_len));
            size_t start_idx = prefix_tokens->size() - prefix_len;
            tokens.insert(tokens.end(),
                          prefix_tokens->begin() + start_idx,
                          prefix_tokens->end());
        }
    }

    // sample_begin marks where we start sampling (after SOT + language + task [+ notimestamps] [+ prefix])
    // This is the position after all prefix tokens where actual transcript begins
    const int sample_begin = static_cast<int>(tokens.size());

    // GAP: sample_len parameter - limit max tokens to sample
    int effective_max_tokens = max_tokens;
    if (sample_len > 0 && sample_len < effective_max_tokens) {
        effective_max_tokens = sample_len;
    }

    // Calculate max initial timestamp based on audio duration
    // Each mel frame = 20ms (hop_length=160*2 at 16kHz -> 20ms per frame after conv2 stride=2)
    int n_mel_frames = static_cast<int>(mel.shape().back());  // Time dimension
    // GAP: max_initial_timestamp is now configurable (Python default: 1.0 second)
    // Convert seconds to timestamp token index: index = seconds / 0.02 (20ms per token)
    const int max_initial_timestamp_index = static_cast<int>(max_initial_timestamp / 0.02f);

    // Calculate max audio timestamp based on actual audio duration
    // If audio_duration_sec is provided (> 0), use it; otherwise use mel frames as fallback
    // Whisper timestamp resolution is 0.02s, so max_audio_timestamp_index = (duration / 0.02)
    float effective_duration = audio_duration_sec;
    if (effective_duration <= 0.0f) {
        // Fallback: estimate from mel frames (assumes unpadded mel)
        effective_duration = static_cast<float>(n_mel_frames) / 100.0f;  // 100 mel frames per second
    }
    int max_audio_timestamp_index = static_cast<int>(effective_duration / 0.02f);  // Each timestamp token = 20ms
    // Clamp to max possible timestamp index (1500 = 30 seconds)
    max_audio_timestamp_index = std::min(max_audio_timestamp_index, 1500);
    const int near_end_threshold = 5;  // Allow 5 timestamp indices (~0.1s) tolerance

    // Build suppress mask for special tokens that should NEVER be generated
    // This is the SuppressTokens filter from OpenAI
    std::vector<float> suppress_mask(config_.n_vocab, 0.0f);

    // Suppress SOT
    suppress_mask[config_.sot_token] = -INFINITY;  // 50258

    // Suppress language tokens (50259 to translate_token-1)
    // v3 has 100 languages (50259-50358), v1/v2 has 99 (50259-50357)
    for (int i = 50259; i < config_.translate_token; ++i) {
        if (i < config_.n_vocab) suppress_mask[i] = -INFINITY;
    }

    // Suppress task tokens
    suppress_mask[config_.translate_token] = -INFINITY;
    suppress_mask[config_.transcribe_token] = -INFINITY;

    // GAP 41: Suppress solm (speaker turn) token when tdrz_enable is false
    // The solm token is used by tinydiarize models to mark speaker changes
    // When disabled, suppress it to prevent unintended behavior
    if (!tdrz_enable) {
        suppress_mask[config_.solm_token] = -INFINITY;
    }

    // Suppress non-speech tokens (punctuation, special characters)
    // This matches Python's non_speech_tokens list from tokenizer
    // These tokens should not appear in speech transcription
    // NOTE: Token 0 ("!") added to prevent spurious exclamation marks after timestamps
    static const std::vector<int> non_speech_tokens = {
        0, 1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
        90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931,
        1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961,
        4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938,
        12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553,
        16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435,
        28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254
    };
    for (int t : non_speech_tokens) {
        if (t < config_.n_vocab) suppress_mask[t] = -INFINITY;
    }

    // GAP 43: Suppress tokens matching user-provided regex pattern
    // These tokens were pre-computed by SuppressRegex filter in the caller
    if (suppress_regex_tokens && !suppress_regex_tokens->empty()) {
        for (int t : *suppress_regex_tokens) {
            if (t >= 0 && t < config_.n_vocab) suppress_mask[t] = -INFINITY;
        }
    }

    // Suppress no_timestamps token (we're in timestamp mode)
    suppress_mask[no_timestamps_token] = -INFINITY;  // 50363

    auto suppress_mask_mx = mx::array(suppress_mask.data(), {1, 1, config_.n_vocab});

    // =========================================================================
    // ApplyTimestampRules state tracking
    // =========================================================================
    // Timestamps work in pairs: [start_ts, text..., end_ts]
    // After start_ts (penultimate was also ts): next must be text (no timestamps)
    // After end_ts (penultimate was NOT ts): next can be text or timestamp
    // This naturally creates: <|0.00|> text <|2.00|> <|2.00|> more text <|4.00|> ...

    // Track last timestamp for monotonicity enforcement
    int last_timestamp_token = timestamp_begin;  // Start from <|0.00|>

    // GAP 10 FIX: Track sum of log probabilities for avg_logprob calculation
    // Python mlx-whisper computes: avg_logprob = sum_logprobs / (len(tokens) + 1)
    // We accumulate log probabilities during greedy decoding
    float sum_logprobs = 0.0f;
    int num_tokens_for_logprob = 0;  // Count tokens (excluding EOT) for averaging

    // GAP 13 FIX: Track no_speech_prob for quality metrics
    // Calculated from softmax(logits)[no_speech_token] at first decode step
    float no_speech_prob = 0.0f;

    // Greedy decoding with proper timestamp rules
    // GAP: Use effective_max_tokens which accounts for sample_len parameter
    for (int i = 0; i < effective_max_tokens; ++i) {
        // Create token tensor
        int n_tokens = kv_cache.empty() ? static_cast<int>(tokens.size()) : 1;
        std::vector<int32_t> token_data(tokens.end() - n_tokens, tokens.end());
        auto token_arr = mx::array(token_data.data(), {1, n_tokens}, mx::int32);

        // Decode
        auto logits = decode(token_arr, encoder_output, &kv_cache, &cross_cache);
        mx::eval(logits);

        // Get logits for last position
        int logits_seq_len = static_cast<int>(logits.shape()[1]);
        auto last_logits = mx::slice(logits, {0, logits_seq_len - 1, 0}, {1, logits_seq_len, config_.n_vocab});

        // Convert to float32 to match Python precision (Gate 0 parity fix)
        // Python does: logits = logits[:, -1].astype(mx.float32)
        last_logits = mx::astype(last_logits, mx::float32);
        mx::eval(last_logits);

        // GAP 13 FIX: Calculate no_speech_prob on first decode step (before suppress mask)
        // Python: probs_at_sot = mx.softmax(logits, axis=-1)
        //         no_speech_prob = float(probs_at_sot[0, tokenizer.no_speech])
        if (i == 0) {
            auto probs = mx::softmax(last_logits, -1);
            mx::eval(probs);
            const float* prob_data = probs.data<float>();
            no_speech_prob = prob_data[config_.no_speech_token];
        }

        // Apply suppress mask (SuppressTokens filter)
        last_logits = last_logits + suppress_mask_mx;

        // Copy logits to CPU for manipulation
        mx::eval(last_logits);
        std::vector<float> logits_data(config_.n_vocab);
        std::memcpy(logits_data.data(), last_logits.data<float>(), config_.n_vocab * sizeof(float));

        // GAP K: Apply repetition penalty to previously generated tokens
        // This helps prevent hallucination loops where the model repeats sequences
        if (repetition_penalty != 1.0f && repetition_penalty > 0.0f) {
            // Track which tokens have been seen (use a set for O(1) lookup)
            std::unordered_set<int> seen_tokens;
            for (int tok : tokens) {
                if (tok >= 0 && tok < config_.n_vocab) {
                    seen_tokens.insert(tok);
                }
            }

            // Apply penalty to seen tokens
            // For positive logits: divide by penalty (reduces probability)
            // For negative logits: multiply by penalty (makes more negative = reduces probability)
            for (int tok : seen_tokens) {
                if (logits_data[tok] > 0) {
                    logits_data[tok] /= repetition_penalty;
                } else {
                    logits_data[tok] *= repetition_penalty;
                }
            }
        }

        // Debug: dump logits at divergence step
        if (std::getenv("DEBUG_LOGITS_STEP")) {
            std::cerr << "\n=== STEP " << i << " LOGITS (C++) ===\n";
            std::cerr << "Token 7031 (Bur): " << logits_data[7031] << "\n";
            std::cerr << "Token 7145 (Bir): " << logits_data[7145] << "\n";
            // Find top 5 tokens
            std::vector<std::pair<float, int>> sorted_logits;
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (logits_data[t] > -1e9f) {
                    sorted_logits.push_back({logits_data[t], t});
                }
            }
            std::sort(sorted_logits.begin(), sorted_logits.end(), std::greater<std::pair<float,int>>());
            std::cerr << "Top 5 BEFORE rules: ";
            for (int k = 0; k < 5 && k < (int)sorted_logits.size(); ++k) {
                std::cerr << "(" << sorted_logits[k].second << ", " << sorted_logits[k].first << ") ";
            }
            std::cerr << "\n";
        }

        // =====================================================================
        // Timestamp probability dominance check - MUST compute BEFORE other rules
        // that suppress timestamps, otherwise the suppression affects the check
        // From OpenAI Whisper: "if sum of probability over timestamps is above
        // any other token, sample timestamp"
        // =====================================================================
        float max_logit_orig = *std::max_element(logits_data.begin(), logits_data.end());
        float sum_exp_orig = 0.0f;
        for (int t = 0; t < config_.n_vocab; ++t) {
            if (logits_data[t] > -1e9f) {
                sum_exp_orig += std::exp(logits_data[t] - max_logit_orig);
            }
        }
        float log_sum_exp_orig = max_logit_orig + std::log(sum_exp_orig + 1e-10f);

        // Log probability of sum of timestamp tokens (logsumexp)
        float timestamp_sum_exp = 0.0f;
        for (int t = timestamp_begin; t < config_.n_vocab; ++t) {
            if (logits_data[t] > -1e9f) {
                timestamp_sum_exp += std::exp(logits_data[t] - max_logit_orig);
            }
        }
        float timestamp_logprob = (timestamp_sum_exp > 0) ?
            (max_logit_orig + std::log(timestamp_sum_exp)) - log_sum_exp_orig : -INFINITY;

        // Max log probability of text tokens
        float max_text_logit = -INFINITY;
        for (int t = 0; t < timestamp_begin; ++t) {
            if (logits_data[t] > max_text_logit) {
                max_text_logit = logits_data[t];
            }
        }
        float max_text_logprob = (max_text_logit > -1e9f) ?
            max_text_logit - log_sum_exp_orig : -INFINITY;

        // Check if timestamp dominates (stored for later application)
        bool timestamp_dominates = (timestamp_logprob > max_text_logprob);

        // =====================================================================
        // ApplyTimestampRules - from OpenAI Whisper decoding.py
        // =====================================================================
        int current_len = static_cast<int>(tokens.size());

        // Get sampled tokens (after prompt)
        std::vector<int> sampled_tokens(tokens.begin() + sample_begin, tokens.end());
        int sampled_len = static_cast<int>(sampled_tokens.size());

        // Check last two tokens (matching Python mlx-whisper ApplyTimestampRules)
        bool last_was_timestamp = (sampled_len >= 1 &&
                                   sampled_tokens[sampled_len - 1] >= timestamp_begin);
        // In Python: penultimate_was_timestamp = (len(seq) < 2 or seq[-2] >= timestamp_begin)
        // This means TRUE when: 1) less than 2 tokens, OR 2) second-to-last is timestamp
        bool penultimate_was_timestamp = (sampled_len < 2 ||
                                          sampled_tokens[sampled_len - 2] >= timestamp_begin);

        // Timestamp pairing rules (matching Python mlx-whisper ApplyTimestampRules)
        // - After timestamp -> timestamp: force text (suppress timestamps)
        // - After text -> timestamp: force timestamp or EOT (suppress text)
        // - After initial timestamp: no suppression (model decides)
        if (last_was_timestamp) {
            if (penultimate_was_timestamp) {
                // Two timestamps in a row - next must be text (suppress all timestamps)
                for (int t = timestamp_begin; t < config_.n_vocab; ++t) {
                    logits_data[t] = -INFINITY;
                }
            } else {
                // Timestamp after text - cannot be normal text (force timestamp or EOT)
                // This matches Python: mask[k, :self.eot] = -np.inf
                for (int t = 0; t < config_.eot_token; ++t) {
                    logits_data[t] = -INFINITY;
                }
            }
        }

        // Find all timestamp tokens in sampled sequence for monotonicity
        std::vector<int> timestamp_tokens;
        for (int t : sampled_tokens) {
            if (t >= timestamp_begin) {
                timestamp_tokens.push_back(t);
            }
        }

        // Timestamps must be strictly monotonically increasing
        // NOTE: Original logic allowed "same or higher" after text+timestamp,
        // but this causes degenerate loops where the same timestamp repeats.
        // We enforce strictly increasing (+1) to prevent repeated timestamps.
        if (!timestamp_tokens.empty()) {
            // Always require strictly increasing timestamps
            int timestamp_last = timestamp_tokens.back() + 1;
            // Suppress all timestamps below the required minimum
            for (int t = timestamp_begin; t < timestamp_last && t < config_.n_vocab; ++t) {
                logits_data[t] = -INFINITY;
            }
            // Update tracking
            last_timestamp_token = timestamp_tokens.back();
        }

        // At the beginning of generation, force timestamp token
        if (current_len == sample_begin) {
            // Suppress all non-timestamp tokens (except EOT which is already handled)
            for (int t = 0; t < timestamp_begin; ++t) {
                if (t != config_.eot_token) {
                    logits_data[t] = -INFINITY;
                }
            }
            // Apply max_initial_timestamp constraint
            int last_allowed = timestamp_begin + max_initial_timestamp_index;
            for (int t = last_allowed + 1; t < config_.n_vocab; ++t) {
                logits_data[t] = -INFINITY;
            }
        }

        // SuppressBlank: At sample_begin, suppress space token + EOT
        // The space token in Whisper is typically 220 (" ")
        if (current_len == sample_begin) {
            logits_data[220] = -INFINITY;  // space token
            logits_data[config_.eot_token] = -INFINITY;  // EOT
        }

        // =====================================================================
        // Suppress timestamps beyond audio duration
        // This prevents hallucination when the model generates timestamps past the audio
        // =====================================================================
        int max_allowed_timestamp = timestamp_begin + max_audio_timestamp_index;
        for (int t = max_allowed_timestamp + 1; t < config_.n_vocab; ++t) {
            logits_data[t] = -INFINITY;
        }

        // =====================================================================
        // Force EOT when we've reached near the end of audio (by timestamp)
        // If last two tokens are both timestamps near max, force EOT
        // This matches Python mlx-whisper ApplyTimestampRules behavior
        // =====================================================================
        if (sampled_len >= 2) {
            int last_tok = sampled_tokens[sampled_len - 1];
            int penult_tok = sampled_tokens[sampled_len - 2];
            if (last_tok >= timestamp_begin && penult_tok >= timestamp_begin) {
                int last_ts_idx = last_tok - timestamp_begin;
                int penult_ts_idx = penult_tok - timestamp_begin;
                int near_max = max_audio_timestamp_index - near_end_threshold;
                if (last_ts_idx >= near_max && penult_ts_idx >= near_max) {
                    // Both timestamps near max - force EOT by suppressing everything else
                    for (int t = 0; t < config_.eot_token; ++t) {
                        logits_data[t] = -INFINITY;
                    }
                    for (int t = config_.eot_token + 1; t < config_.n_vocab; ++t) {
                        logits_data[t] = -INFINITY;
                    }
                }
            }
        }

        // =====================================================================
        // Apply timestamp probability dominance (computed earlier on original logits)
        // If sum of timestamp logprobs > max text token logprob, force timestamp
        //
        // CRITICAL: Do NOT apply when "two timestamps in a row" (text required).
        // When penultimate_was_timestamp && last_was_timestamp, we MUST generate text,
        // so suppress_text would conflict with the pairing rules and deadlock to EOT.
        // =====================================================================
        bool two_timestamps_in_row = last_was_timestamp && penultimate_was_timestamp;
        if (timestamp_dominates && !two_timestamps_in_row) {
            for (int t = 0; t < timestamp_begin; ++t) {
                logits_data[t] = -INFINITY;
            }
        }

        // Debug: dump logits AFTER all timestamp rules
        const char* debug_step_env = std::getenv("DEBUG_LOGITS_STEP");
        int debug_step = debug_step_env ? std::atoi(debug_step_env) : -1;
        if (debug_step >= 0 && i == debug_step) {
            std::cerr << "\n=== STEP " << i << " LOGITS (C++) AFTER TIMESTAMP RULES ===\n";
            std::cerr << "Token 50560: " << logits_data[50560] << "\n";
            std::cerr << "Token 50561: " << logits_data[50561] << "\n";
            std::cerr << "Token 50562: " << logits_data[50562] << "\n";
            std::cerr << "timestamp_tokens.back()=" << (timestamp_tokens.empty() ? -1 : timestamp_tokens.back()) << "\n";
            std::cerr << "last_was_timestamp=" << last_was_timestamp
                      << " penultimate_was_timestamp=" << penultimate_was_timestamp
                      << " timestamp_dominates=" << timestamp_dominates << "\n";
            std::cerr << "timestamp_logprob=" << timestamp_logprob
                      << " max_text_logprob=" << max_text_logprob << "\n";
            // Find top 5 tokens after rules
            std::vector<std::pair<float, int>> sorted_logits;
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (logits_data[t] > -1e9f) {
                    sorted_logits.push_back({logits_data[t], t});
                }
            }
            std::sort(sorted_logits.begin(), sorted_logits.end(), std::greater<std::pair<float,int>>());
            std::cerr << "Top 5 AFTER rules: ";
            for (int k = 0; k < 5 && k < (int)sorted_logits.size(); ++k) {
                std::cerr << "(" << sorted_logits[k].second << ", " << sorted_logits[k].first << ") ";
            }
            std::cerr << "\n";
        }

        // =====================================================================
        // GAP 4/12: Token selection with temperature support
        // Python: if temp == 0: argmax, else mx.random.categorical(logits / temp)
        // =====================================================================
        int token_id = 0;
        float best_logit = -INFINITY;

        if (temperature <= 0.0f) {
            // Greedy selection (argmax) - temperature = 0
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (logits_data[t] > best_logit) {
                    best_logit = logits_data[t];
                    token_id = t;
                }
            }
        } else {
            // Temperature-based categorical sampling
            // Python: mx.random.categorical(logits / temp)

            // First find best_logit for logsumexp calculation (needed later)
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (logits_data[t] > best_logit) {
                    best_logit = logits_data[t];
                }
            }

            // Scale logits by temperature and create MLX array for sampling
            std::vector<float> scaled_logits(config_.n_vocab);
            for (int t = 0; t < config_.n_vocab; ++t) {
                // Keep suppressed tokens suppressed (very negative values)
                if (logits_data[t] < -1e9f) {
                    scaled_logits[t] = -1e10f;  // Effectively zero probability
                } else {
                    scaled_logits[t] = logits_data[t] / temperature;
                }
            }

            // Create MLX array and sample
            mx::array logits_arr = mx::array(scaled_logits.data(), {1, config_.n_vocab}, mx::float32);
            mx::array sampled = mx::random::categorical(logits_arr, 1);
            mx::eval(sampled);

            // Extract sampled token
            const int32_t* sampled_data = sampled.data<int32_t>();
            token_id = sampled_data[0];

            // Update best_logit to the selected token's logit (for logprob calculation)
            best_logit = logits_data[token_id];

            if (std::getenv("DEBUG_DECODER")) {
                std::cerr << "[DEBUG] Step " << i << ": Temperature sampling (T=" << temperature
                          << ") selected token " << token_id << " (logit=" << best_logit << ")\n";
            }
        }

        // =====================================================================
        // GAP 10 FIX: Calculate and accumulate log probability for selected token
        // Python: logprobs = logits - logsumexp(logits)
        //         current_logprob = logprobs[next_token]
        //         sum_logprobs += current_logprob * (prev_token != EOT)
        // We compute logsumexp of the modified logits (after masking)
        // GAP M: Skip this calculation when skip_logprobs is true (for speed)
        // =====================================================================
        if (!skip_logprobs && best_logit > -1e9f) {  // Only if valid token selected and not skipping
            // Compute logsumexp for normalization
            float max_for_lse = -INFINITY;
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (logits_data[t] > max_for_lse) {
                    max_for_lse = logits_data[t];
                }
            }
            float sum_exp = 0.0f;
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (logits_data[t] > -1e9f) {
                    sum_exp += std::exp(logits_data[t] - max_for_lse);
                }
            }
            float logsumexp = max_for_lse + std::log(sum_exp + 1e-10f);

            // Log probability of selected token
            float token_logprob = logits_data[token_id] - logsumexp;

            // Accumulate (Python doesn't add EOT's logprob to sum)
            // We check if PREVIOUS token was EOT, but since we're in greedy
            // and break on EOT, we just always add here (except when forced EOT below)
            if (token_id != config_.eot_token) {
                sum_logprobs += token_logprob;
                num_tokens_for_logprob++;
            }
        }

        // CRITICAL FIX: If all tokens are suppressed, force EOT instead of selecting token 0
        // This can happen when conflicting rules suppress all tokens simultaneously
        if (best_logit == -INFINITY) {
            token_id = config_.eot_token;
            if (std::getenv("DEBUG_DECODER")) {
                std::cerr << "[DEBUG] Step " << i << ": All tokens suppressed, forcing EOT\n";
                std::cerr << "[DEBUG]   last_was_timestamp=" << last_was_timestamp
                          << " penultimate_was_timestamp=" << penultimate_was_timestamp
                          << " timestamp_dominates=" << timestamp_dominates << "\n";
                std::cerr << "[DEBUG]   timestamp_logprob=" << timestamp_logprob
                          << " max_text_logprob=" << max_text_logprob << "\n";
            }
        }

        // Check for EOT
        if (token_id == config_.eot_token) {
            if (std::getenv("DEBUG_DECODER") && best_logit > -INFINITY) {
                std::cerr << "[DEBUG] Step " << i << ": Selected EOT with logit=" << best_logit << "\n";
            }
            tokens.push_back(token_id);
            break;
        }

        tokens.push_back(token_id);

        // Debug: trace when Bir/Bur tokens are selected
        if (std::getenv("DEBUG_LOGITS_STEP") && (token_id == 7031 || token_id == 7145)) {
            std::cerr << "\n=== SELECTED TOKEN " << token_id << " (";
            std::cerr << (token_id == 7031 ? "Bur" : "Bir") << ") at step " << i << " ===\n";
            std::cerr << "  Logit: " << best_logit << "\n";
        }

        // =====================================================================
        // Repetition / Degenerate Output Detection
        // Reference: OpenAI uses compression_ratio_threshold=2.4
        // We use simpler n-gram based detection that works in real-time
        // =====================================================================
        int n = static_cast<int>(tokens.size());
        int generated_len = n - sample_begin;

        // Check 1: 3+ identical tokens of ANY kind in a row (catches "John John John" or "!!! !!!")
        if (n >= 3 &&
            tokens[n-1] == tokens[n-2] && tokens[n-2] == tokens[n-3]) {
            tokens.push_back(config_.eot_token);
            break;
        }

        // Check 1a: Multiple 2-gram repetitions in window (pathological looping detection)
        // A single "teeth teeth" might be valid speech, but multiple 2-gram repetitions
        // of WORD tokens within 30 tokens indicates a degenerate loop.
        // We exclude punctuation tokens (0-50) since "!!!" or "..." can be legitimate.
        if (generated_len >= 15) {
            // Count 2-gram repetitions in last 30 WORD tokens, excluding punctuation
            std::vector<int> recent_word_tokens;
            for (int j = n - 1; j >= sample_begin && recent_word_tokens.size() < 30; --j) {
                // Include text tokens but exclude punctuation (tokens 0-50 are mostly punctuation)
                // Token 0 = "!", 11 = ",", 13 = ".", etc.
                if (tokens[j] < timestamp_begin && tokens[j] > 50) {
                    recent_word_tokens.insert(recent_word_tokens.begin(), tokens[j]);
                }
            }

            int two_gram_count = 0;
            for (size_t j = 1; j < recent_word_tokens.size(); ++j) {
                if (recent_word_tokens[j] == recent_word_tokens[j-1]) {
                    two_gram_count++;
                }
            }

            // If 2 or more WORD token 2-gram repetitions, it's pathological
            // Normal speech rarely has 2 consecutive repeated words in 30 tokens
            if (two_gram_count >= 2) {
                tokens.push_back(config_.eot_token);
                break;
            }
        }

        // Check 1b: Immediate n-gram repetition (catches "word word word word" 4-gram patterns)
        // Check if the last 4 text tokens match the 4 tokens before them
        if (generated_len >= 8) {
            // Extract last 4 text tokens
            std::vector<int> last_ngram;
            int last_ngram_end = n;
            for (int j = n - 1; j >= sample_begin && last_ngram.size() < 4; --j) {
                if (tokens[j] < timestamp_begin) {
                    last_ngram.insert(last_ngram.begin(), tokens[j]);
                    if (last_ngram.size() == 1) last_ngram_end = j;
                }
            }

            // Check if we have 4 text tokens and the 4 before them match
            if (last_ngram.size() == 4) {
                // Find the position where the last 4-gram started
                int pos = n - 1;
                int text_count = 0;
                while (pos >= sample_begin && text_count < 4) {
                    if (tokens[pos] < timestamp_begin) text_count++;
                    pos--;
                }
                int ngram_start = pos + 1;

                // Extract the 4 tokens before the last 4
                std::vector<int> prev_ngram;
                int check_pos = ngram_start - 1;
                while (check_pos >= sample_begin && prev_ngram.size() < 4) {
                    if (tokens[check_pos] < timestamp_begin) {
                        prev_ngram.insert(prev_ngram.begin(), tokens[check_pos]);
                    }
                    check_pos--;
                }

                // Compare ngrams
                if (prev_ngram.size() == 4 && prev_ngram == last_ngram) {
                    // Immediate 4-gram repetition detected, end sequence
                    tokens.push_back(config_.eot_token);
                    break;
                }
            }
        }

        // Check 1c: Repeating n-gram PATTERN detection (catches "A B C A B C A B C" patterns)
        // The model can produce repeating patterns like "Burkitts. Burkitts. Burkitts."
        // which tokenizes as [Burk, itts, ., Burk, itts, ., Burk, itts, .]
        // This is a 3-token pattern repeating 3 times. We detect patterns of length 2-5
        // that repeat 3+ times consecutively.
        if (generated_len >= 6) {
            // Get last 20 text tokens (excluding timestamps)
            std::vector<int> recent_text;
            for (int j = n - 1; j >= sample_begin && recent_text.size() < 20; --j) {
                if (tokens[j] < timestamp_begin) {
                    recent_text.insert(recent_text.begin(), tokens[j]);
                }
            }

            // Check for repeating patterns of length 2, 3, 4, 5
            for (int pattern_len = 2; pattern_len <= 5 && pattern_len * 3 <= static_cast<int>(recent_text.size()); ++pattern_len) {
                // Extract the last pattern_len tokens as the pattern
                std::vector<int> pattern(recent_text.end() - pattern_len, recent_text.end());

                // Count how many consecutive times this pattern appears (from the end)
                int repeat_count = 1;
                int check_pos = static_cast<int>(recent_text.size()) - pattern_len * 2;

                while (check_pos >= 0) {
                    bool matches = true;
                    for (int k = 0; k < pattern_len; ++k) {
                        if (recent_text[check_pos + k] != pattern[k]) {
                            matches = false;
                            break;
                        }
                    }
                    if (matches) {
                        repeat_count++;
                        check_pos -= pattern_len;
                    } else {
                        break;
                    }
                }

                // If pattern repeats 3+ times, it's pathological looping
                if (repeat_count >= 3) {
                    // Truncate to remove the repeated patterns, keeping only first occurrence
                    // Find where in tokens[] the repetition starts
                    int text_count = 0;
                    int truncate_pos = n;
                    for (int j = n - 1; j >= sample_begin; --j) {
                        if (tokens[j] < timestamp_begin) {
                            text_count++;
                            // Keep only the first occurrence of the pattern
                            // repeat_count patterns = pattern_len * repeat_count text tokens
                            // We want to remove (repeat_count - 1) * pattern_len tokens
                            if (text_count == (repeat_count - 1) * pattern_len) {
                                truncate_pos = j;
                                break;
                            }
                        }
                    }
                    if (truncate_pos < n) {
                        tokens.resize(truncate_pos);
                    }
                    tokens.push_back(config_.eot_token);
                    break;
                }
            }
            if (!tokens.empty() && tokens.back() == config_.eot_token) break;
        }

        // Check 1d: Immediate duplicate phrases of 5-8 tokens (catches "A B C D E F G A B C D E F G")
        // Longer phrases (5-8 tokens) repeating even twice is highly suspicious - no natural speech
        // repeats a 5-8 word phrase exactly. This catches patterns like:
        // " one much in the same way that one much in the same way that" (7 tokens x 2)
        // which Check 1c misses because it requires 3+ repeats for patterns up to 5 tokens.
        if (generated_len >= 10) {
            // Get last 20 text tokens (excluding timestamps)
            std::vector<int> recent_text;
            std::vector<int> recent_text_positions;  // Track positions in tokens[]
            for (int j = n - 1; j >= sample_begin && recent_text.size() < 20; --j) {
                if (tokens[j] < timestamp_begin) {
                    recent_text.insert(recent_text.begin(), tokens[j]);
                    recent_text_positions.insert(recent_text_positions.begin(), j);
                }
            }

            // Check for immediate duplicates of length 5, 6, 7, 8
            for (int phrase_len = 5; phrase_len <= 8 && phrase_len * 2 <= static_cast<int>(recent_text.size()); ++phrase_len) {
                // Extract the last phrase_len tokens as the phrase
                std::vector<int> phrase(recent_text.end() - phrase_len, recent_text.end());

                // Check if the phrase_len tokens immediately before match
                bool is_duplicate = true;
                int check_start = static_cast<int>(recent_text.size()) - phrase_len * 2;
                for (int k = 0; k < phrase_len && is_duplicate; ++k) {
                    if (recent_text[check_start + k] != phrase[k]) {
                        is_duplicate = false;
                    }
                }

                if (is_duplicate) {
                    // Found immediate duplicate phrase - truncate to keep only first occurrence
                    // The duplicate starts at position recent_text.size() - phrase_len
                    int truncate_idx = static_cast<int>(recent_text.size()) - phrase_len;
                    if (std::getenv("DEBUG_DECODER")) {
                        std::cerr << "[DEBUG] Check 1d: Found " << phrase_len << "-token duplicate phrase\n";
                        std::cerr << "[DEBUG]   Phrase: ";
                        for (int t : phrase) std::cerr << t << " ";
                        std::cerr << "\n";
                    }
                    if (truncate_idx >= 0 && truncate_idx < static_cast<int>(recent_text_positions.size())) {
                        int truncate_pos = recent_text_positions[truncate_idx];
                        tokens.resize(truncate_pos);
                    }
                    tokens.push_back(config_.eot_token);
                    break;
                }
            }
            if (!tokens.empty() && tokens.back() == config_.eot_token) break;
        }

        // Check 2: High repetition of single token (catches "! ! ! ! !" pattern)
        // Count how many times the last token appears in recent history
        // NOTE: 40% threshold (was 30%) to avoid triggering on common words like "the"
        // which can legitimately appear 30-35% in short segments (e.g., "the X of the Y toward the Z")
        if (generated_len >= 10) {
            int last_tok = tokens[n-1];
            int count = 0;
            for (int j = sample_begin; j < n; ++j) {
                if (tokens[j] == last_tok) count++;
            }
            // If last token appears in >40% of generated tokens, likely degenerate
            if (count > generated_len * 4 / 10) {
                tokens.push_back(config_.eot_token);
                break;
            }
        }

        // Check 3: Look for ANY occurrence of recent 10-gram anywhere earlier in text
        // (increased from 5/7-gram to reduce false positives on natural speech)
        if (generated_len >= 30) {
            const int ngram_len = 10;
            // Extract last ngram (skip timestamps) and track where it starts
            std::vector<int> last_ngram;
            int last_ngram_start = n;
            for (int j = n - 1; j >= sample_begin && static_cast<int>(last_ngram.size()) < ngram_len; --j) {
                if (tokens[j] < timestamp_begin) {
                    last_ngram.insert(last_ngram.begin(), tokens[j]);
                    last_ngram_start = j;
                }
            }

            // Search for this ngram earlier in the sequence
            if (static_cast<int>(last_ngram.size()) == ngram_len) {
                int search_end = last_ngram_start - ngram_len;
                for (int start = sample_begin; start < search_end; ++start) {
                    if (tokens[start] >= timestamp_begin) continue;

                    bool match = true;
                    int ngram_idx = 0;
                    for (int j = start; j < n && ngram_idx < ngram_len; ++j) {
                        if (tokens[j] >= timestamp_begin) continue;
                        if (tokens[j] != last_ngram[ngram_idx]) {
                            match = false;
                            break;
                        }
                        ngram_idx++;
                    }

                    if (match && ngram_idx == ngram_len) {
                        tokens.resize(last_ngram_start);
                        tokens.push_back(config_.eot_token);
                        break;
                    }
                }
                if (tokens.back() == config_.eot_token) break;
            }
        }
    }

    // If we hit max_tokens without EOT, add EOT
    if (tokens.empty() || tokens.back() != config_.eot_token) {
        tokens.push_back(config_.eot_token);
    }

    // GAP 10 FIX: Output average log probability if requested
    // Python formula: avg_logprob = sum_logprobs / (len(tokens) + 1)
    // We use num_tokens_for_logprob which counts actual generated tokens (excluding EOT)
    if (avg_logprob_out) {
        // Number of generated tokens (total - prompt tokens)
        int generated_count = static_cast<int>(tokens.size()) - sample_begin;
        // Python divides by (len(tokens) + 1), where len(tokens) is the generated sequence
        // This is equivalent to generated_count + 1
        if (generated_count > 0) {
            *avg_logprob_out = sum_logprobs / static_cast<float>(generated_count + 1);
        } else {
            *avg_logprob_out = 0.0f;
        }
    }

    // GAP 13 FIX: Output no_speech_prob if requested
    if (no_speech_prob_out) {
        *no_speech_prob_out = no_speech_prob;
    }

    return tokens;
}

// ============================================================================
// Beam Search Decoding
// ============================================================================

namespace {

// Compute normalized score for beam comparison using length penalty
float compute_beam_score(const Beam& beam, int sample_begin, float length_penalty) {
    int n_tokens = static_cast<int>(beam.tokens.size()) - sample_begin;
    if (n_tokens <= 0) {
        return beam.log_prob;
    }
    // Length penalty: divide by ((5 + length) / 6)^alpha
    // alpha = 1.0 means pure average
    float length_factor = std::pow((5.0f + n_tokens) / 6.0f, length_penalty);
    return beam.log_prob / length_factor;
}

// Structure to track a candidate continuation from a beam
struct BeamCandidate {
    int beam_idx;      // Index of parent beam
    int token;         // New token to add
    float new_log_prob; // Total log probability after adding this token
};

}  // anonymous namespace

BeamSearchResult WhisperModel::generate_beam(
    const mx::array& mel,
    const std::string& language,
    const std::string& task,
    int beam_size,
    float length_penalty,
    int max_tokens,
    const std::vector<int>* prompt_tokens,
    float patience,
    float max_initial_timestamp,
    bool without_timestamps
) {
    // =========================================================================
    // BEAM SEARCH DECODING
    // Maintains multiple hypotheses to improve transcription accuracy
    // =========================================================================

    // Encode audio
    auto encoder_output = encode(mel);
    // Keep encoder output in native dtype (float16) to match Python exactly
    mx::eval(encoder_output);

    // Special token constants (use config values for v3 compatibility)
    const int timestamp_begin = config_.timestamp_begin;
    const int no_timestamps_token = config_.no_timestamps_token;

    // Build initial token sequence
    // GAP 6: With prompt tokens: [sot_prev, prompt_tokens..., SOT, language, task]
    //        Without prompt:     [SOT, language, task]
    std::vector<int> initial_tokens;

    // Add prompt tokens if provided (GAP 6: initial_prompt support)
    if (prompt_tokens && !prompt_tokens->empty()) {
        initial_tokens.push_back(config_.sot_prev_token);
        size_t max_prompt_len = (config_.n_text_ctx / 2) - 1;
        size_t prompt_len = std::min(prompt_tokens->size(), max_prompt_len);
        size_t start_idx = prompt_tokens->size() - prompt_len;
        initial_tokens.insert(initial_tokens.end(),
                              prompt_tokens->begin() + start_idx,
                              prompt_tokens->end());
    }

    // Add SOT token
    initial_tokens.push_back(config_.sot_token);

    int lang_token = get_language_token(language);
    initial_tokens.push_back(lang_token);

    if (task == "translate") {
        initial_tokens.push_back(config_.translate_token);
    } else {
        initial_tokens.push_back(config_.transcribe_token);
    }

    // GAP: without_timestamps mode - add <|notimestamps|> token after task token
    if (without_timestamps) {
        initial_tokens.push_back(config_.no_timestamps_token);
    }

    const int sample_begin = static_cast<int>(initial_tokens.size());

    // Calculate max initial timestamp based on audio duration
    int n_mel_frames = static_cast<int>(mel.shape().back());
    // GAP: max_initial_timestamp is now configurable (Python default: 1.0 second)
    const int max_initial_timestamp_index = static_cast<int>(max_initial_timestamp / 0.02f);

    // GAP 9: Beam patience - effective beam size = beam_size * patience
    // patience controls how many beams can finish before we stop
    const int effective_beam_size = static_cast<int>(std::ceil(beam_size * patience));

    // Build suppress mask for special tokens
    std::vector<float> suppress_mask(config_.n_vocab, 0.0f);
    suppress_mask[config_.sot_token] = -INFINITY;
    // Suppress language tokens (50259 to translate_token-1)
    for (int i = 50259; i < config_.translate_token; ++i) {
        if (i < config_.n_vocab) suppress_mask[i] = -INFINITY;
    }
    suppress_mask[config_.translate_token] = -INFINITY;
    suppress_mask[config_.transcribe_token] = -INFINITY;
    // Note: Don't suppress sot_prev, sot_lm - Python mlx-whisper doesn't either
    suppress_mask[config_.no_timestamps_token] = -INFINITY;

    auto suppress_mask_mx = mx::array(suppress_mask.data(), {1, 1, config_.n_vocab});

    // =========================================================================
    // Initialize beams with first token predictions
    // =========================================================================
    CrossKVCache cross_cache;  // Shared across all beams

    // Process initial tokens to get first prediction
    std::vector<int32_t> token_data(initial_tokens.begin(), initial_tokens.end());
    auto token_arr = mx::array(token_data.data(), {1, static_cast<int>(initial_tokens.size())}, mx::int32);

    DecoderKVCache first_kv_cache;
    auto logits = decode(token_arr, encoder_output, &first_kv_cache, &cross_cache);
    mx::eval(logits);

    int logits_seq_len = static_cast<int>(logits.shape()[1]);
    auto last_logits = mx::slice(logits, {0, logits_seq_len - 1, 0}, {1, logits_seq_len, config_.n_vocab});

    // Convert to float32 to match Python precision (Gate 0 parity fix)
    last_logits = mx::astype(last_logits, mx::float32);
    mx::eval(last_logits);

    // GAP 13 FIX: Calculate no_speech_prob from softmax at first decode step
    auto probs = mx::softmax(last_logits, -1);
    mx::eval(probs);
    const float* prob_data = probs.data<float>();
    float no_speech_prob = prob_data[config_.no_speech_token];

    // Apply suppress mask
    last_logits = last_logits + suppress_mask_mx;

    // At sample_begin, force timestamp and apply initial timestamp constraint
    std::vector<float> logits_data(config_.n_vocab);
    mx::eval(last_logits);
    std::memcpy(logits_data.data(), last_logits.data<float>(), config_.n_vocab * sizeof(float));

    // Force timestamp at beginning
    for (int t = 0; t < timestamp_begin; ++t) {
        if (t != config_.eot_token) {
            logits_data[t] = -INFINITY;
        }
    }
    // Apply max_initial_timestamp
    int last_allowed = timestamp_begin + max_initial_timestamp_index;
    for (int t = last_allowed + 1; t < config_.n_vocab; ++t) {
        logits_data[t] = -INFINITY;
    }
    // SuppressBlank at sample_begin
    logits_data[220] = -INFINITY;  // space token
    logits_data[config_.eot_token] = -INFINITY;

    // Compute log softmax
    float max_logit = *std::max_element(logits_data.begin(), logits_data.end());
    float sum_exp = 0.0f;
    for (int t = 0; t < config_.n_vocab; ++t) {
        if (logits_data[t] > -1e9f) {
            sum_exp += std::exp(logits_data[t] - max_logit);
        }
    }
    float log_sum_exp = max_logit + std::log(sum_exp + 1e-10f);

    // Get top-k tokens for initial beams
    std::vector<std::pair<float, int>> token_scores;
    for (int t = 0; t < config_.n_vocab; ++t) {
        if (logits_data[t] > -1e9f) {
            float log_prob = logits_data[t] - log_sum_exp;
            token_scores.emplace_back(log_prob, t);
        }
    }
    std::partial_sort(token_scores.begin(),
                      token_scores.begin() + std::min(beam_size, static_cast<int>(token_scores.size())),
                      token_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Initialize beams
    std::vector<Beam> beams;
    std::vector<DecoderKVCache> kv_caches;
    std::vector<Beam> finished_beams;

    int n_init = std::min(beam_size, static_cast<int>(token_scores.size()));
    for (int i = 0; i < n_init; ++i) {
        float log_prob = token_scores[i].first;
        int token = token_scores[i].second;

        Beam beam;
        beam.tokens = initial_tokens;
        beam.tokens.push_back(token);
        beam.log_prob = log_prob;

        if (token == config_.eot_token) {
            beam.finished = true;
            finished_beams.push_back(beam);
        } else {
            beams.push_back(beam);
            kv_caches.push_back(first_kv_cache.clone());
        }
    }

    // =========================================================================
    // Main beam search decode loop
    // =========================================================================
    float best_finished_score = -INFINITY;

    for (int step = 0; step < max_tokens - 1 && !beams.empty(); ++step) {
        // Collect all candidates from all beams
        std::vector<BeamCandidate> all_candidates;
        std::vector<DecoderKVCache> new_kv_caches_temp;

        int n_beams = static_cast<int>(beams.size());

        // =====================================================================
        // GAP 73: Batched decode for all beams (parallel GPU processing)
        // =====================================================================
        // Batch all last tokens: [n_beams, 1]
        std::vector<int> batch_tokens;
        batch_tokens.reserve(n_beams);
        for (const auto& beam : beams) {
            batch_tokens.push_back(beam.tokens.back());
        }
        auto batch_tokens_arr = mx::array(batch_tokens.data(), {n_beams, 1}, mx::int32);

        // Stack KV caches for batched processing
        auto stacked_kv = DecoderKVCache::stack(kv_caches);

        // Batch encoder output: [1, enc_len, n_state] -> [n_beams, enc_len, n_state]
        auto batched_encoder = mx::repeat(encoder_output, n_beams, 0);

        // Single batched decode call (GPU parallel)
        auto batched_logits = decode(batch_tokens_arr, batched_encoder, &stacked_kv, &cross_cache);
        mx::eval(batched_logits);

        // Unstack KV caches back to individual beams
        kv_caches = DecoderKVCache::unstack(stacked_kv, n_beams);

        // Get logits for last position: [n_beams, 1, vocab] -> [n_beams, vocab]
        int bls = static_cast<int>(batched_logits.shape()[1]);
        auto all_beam_logits = mx::slice(batched_logits, {0, bls - 1, 0}, {n_beams, bls, config_.n_vocab});
        all_beam_logits = mx::squeeze(all_beam_logits, 1);  // [n_beams, vocab]

        // Convert to float32 to match Python precision (Gate 0 parity fix)
        all_beam_logits = mx::astype(all_beam_logits, mx::float32);

        // Apply suppress mask (broadcasts to all beams)
        all_beam_logits = all_beam_logits + suppress_mask_mx;

        // Copy logits to CPU for per-beam processing
        mx::eval(all_beam_logits);

        // Process each beam's logits (timestamp rules are beam-specific)
        for (int beam_idx = 0; beam_idx < n_beams; ++beam_idx) {
            Beam& beam = beams[beam_idx];

            // Extract this beam's logits
            std::vector<float> bl_data(config_.n_vocab);
            const float* src = all_beam_logits.data<float>() + beam_idx * config_.n_vocab;
            std::memcpy(bl_data.data(), src, config_.n_vocab * sizeof(float));

            // Apply timestamp rules based on beam's token history
            int current_len = static_cast<int>(beam.tokens.size());
            std::vector<int> sampled_tokens(beam.tokens.begin() + sample_begin, beam.tokens.end());
            int sampled_len = static_cast<int>(sampled_tokens.size());

            bool last_was_timestamp = sampled_len >= 1 && sampled_tokens[sampled_len - 1] >= timestamp_begin;
            bool penultimate_was_timestamp = sampled_len < 2 || sampled_tokens[sampled_len - 2] >= timestamp_begin;

            // Timestamp pairing rules
            if (last_was_timestamp) {
                if (penultimate_was_timestamp) {
                    for (int t = timestamp_begin; t < config_.n_vocab; ++t) {
                        bl_data[t] = -INFINITY;
                    }
                } else {
                    for (int t = 0; t < config_.eot_token; ++t) {
                        bl_data[t] = -INFINITY;
                    }
                }
            }

            // Timestamps must be monotonically increasing
            std::vector<int> timestamp_tokens;
            for (int t : sampled_tokens) {
                if (t >= timestamp_begin) timestamp_tokens.push_back(t);
            }
            if (!timestamp_tokens.empty()) {
                int timestamp_last;
                if (last_was_timestamp && !penultimate_was_timestamp) {
                    timestamp_last = timestamp_tokens.back();
                } else {
                    timestamp_last = timestamp_tokens.back() + 1;
                }
                for (int t = timestamp_begin; t <= timestamp_last && t < config_.n_vocab; ++t) {
                    bl_data[t] = -INFINITY;
                }
            }

            // Timestamp probability dominance check
            float bl_max = *std::max_element(bl_data.begin(), bl_data.end());
            float bl_sum_exp = 0.0f;
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (bl_data[t] > -1e9f) {
                    bl_sum_exp += std::exp(bl_data[t] - bl_max);
                }
            }
            float bl_log_sum = bl_max + std::log(bl_sum_exp + 1e-10f);

            float ts_sum_exp = 0.0f;
            for (int t = timestamp_begin; t < config_.n_vocab; ++t) {
                if (bl_data[t] > -1e9f) {
                    ts_sum_exp += std::exp(bl_data[t] - bl_max);
                }
            }
            float ts_logprob = (ts_sum_exp > 0) ?
                (bl_max + std::log(ts_sum_exp)) - bl_log_sum : -INFINITY;

            float max_text_logit = -INFINITY;
            for (int t = 0; t < timestamp_begin; ++t) {
                if (bl_data[t] > max_text_logit) max_text_logit = bl_data[t];
            }
            float max_text_logprob = (max_text_logit > -1e9f) ? max_text_logit - bl_log_sum : -INFINITY;

            if (ts_logprob > max_text_logprob) {
                for (int t = 0; t < timestamp_begin; ++t) {
                    bl_data[t] = -INFINITY;
                }
            }

            // Compute log probabilities
            bl_max = *std::max_element(bl_data.begin(), bl_data.end());
            bl_sum_exp = 0.0f;
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (bl_data[t] > -1e9f) {
                    bl_sum_exp += std::exp(bl_data[t] - bl_max);
                }
            }
            bl_log_sum = bl_max + std::log(bl_sum_exp + 1e-10f);

            // Get top-k candidates for this beam
            std::vector<std::pair<float, int>> beam_scores;
            for (int t = 0; t < config_.n_vocab; ++t) {
                if (bl_data[t] > -1e9f) {
                    float lp = bl_data[t] - bl_log_sum;
                    beam_scores.emplace_back(lp, t);
                }
            }

            int n_candidates = std::min(beam_size, static_cast<int>(beam_scores.size()));
            std::partial_sort(beam_scores.begin(),
                              beam_scores.begin() + n_candidates,
                              beam_scores.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            for (int k = 0; k < n_candidates; ++k) {
                BeamCandidate cand;
                cand.beam_idx = beam_idx;
                cand.token = beam_scores[k].second;
                cand.new_log_prob = beam.log_prob + beam_scores[k].first;
                all_candidates.push_back(cand);
            }
        }

        // Sort all candidates by total log probability
        std::sort(all_candidates.begin(), all_candidates.end(),
                  [](const BeamCandidate& a, const BeamCandidate& b) {
                      return a.new_log_prob > b.new_log_prob;
                  });

        // Select top beam_size candidates
        std::vector<Beam> new_beams;
        std::vector<DecoderKVCache> new_kv_caches;
        std::unordered_map<int, bool> used_cache;

        for (const auto& cand : all_candidates) {
            if (static_cast<int>(new_beams.size()) >= beam_size) break;

            Beam new_beam;
            new_beam.tokens = beams[cand.beam_idx].tokens;
            new_beam.tokens.push_back(cand.token);
            new_beam.log_prob = cand.new_log_prob;

            if (cand.token == config_.eot_token) {
                new_beam.finished = true;
                finished_beams.push_back(new_beam);

                float score = compute_beam_score(new_beam, sample_begin, length_penalty);
                if (score > best_finished_score) {
                    best_finished_score = score;
                }
            } else {
                // NOTE: Simple 2-token repetition check was REMOVED in iteration 1486.
                // It caused false positives on legitimate speech like "teeth teeth".
                // The 5-gram repetition check below is sufficient.
                int n = static_cast<int>(new_beam.tokens.size());

                // Check for 5-gram repetition (same as greedy decoder)
                int generated_len = n - sample_begin;
                if (generated_len >= 10) {
                    const int ngram_len = 5;
                    // Extract last ngram (skip timestamps)
                    std::vector<int> last_ngram;
                    int last_ngram_start = n;
                    for (int j = n - 1; j >= sample_begin && static_cast<int>(last_ngram.size()) < ngram_len; --j) {
                        if (new_beam.tokens[j] < timestamp_begin) {
                            last_ngram.insert(last_ngram.begin(), new_beam.tokens[j]);
                            last_ngram_start = j;
                        }
                    }

                    // Search for this ngram earlier in sequence
                    bool has_repetition = false;
                    if (static_cast<int>(last_ngram.size()) == ngram_len) {
                        int search_end = last_ngram_start - ngram_len;
                        for (int start = sample_begin; start < search_end && !has_repetition; ++start) {
                            if (new_beam.tokens[start] >= timestamp_begin) continue;

                            bool match = true;
                            int ngram_idx = 0;
                            for (int j = start; j < n && ngram_idx < ngram_len; ++j) {
                                if (new_beam.tokens[j] >= timestamp_begin) continue;
                                if (new_beam.tokens[j] != last_ngram[ngram_idx]) {
                                    match = false;
                                    break;
                                }
                                ngram_idx++;
                            }
                            if (match && ngram_idx == ngram_len) {
                                has_repetition = true;
                            }
                        }
                    }
                    if (has_repetition) {
                        // Force this beam to finish - truncate and add EOT
                        new_beam.tokens.resize(last_ngram_start);
                        new_beam.tokens.push_back(config_.eot_token);
                        new_beam.finished = true;
                        finished_beams.push_back(new_beam);
                        continue;
                    }
                }

                new_beams.push_back(new_beam);

                // GAP 73 FIX: Always clone KV cache to avoid moved-from empty cache issue
                // The original optimization (move first use, clone subsequent) breaks when
                // the same beam_idx produces multiple candidates - after moving, subsequent
                // clones would clone an empty cache.
                new_kv_caches.push_back(kv_caches[cand.beam_idx].clone());
            }
        }

        beams = std::move(new_beams);
        kv_caches = std::move(new_kv_caches);

        // GAP 9: Beam patience - stop when effective_beam_size beams have finished
        // (effective_beam_size = beam_size * patience)
        if (static_cast<int>(finished_beams.size()) >= effective_beam_size) {
            break;
        }

        // Early stopping: if best active beam can't beat best finished, stop
        if (!finished_beams.empty() && !beams.empty()) {
            float best_active = beams[0].log_prob;
            float best_possible_score = compute_beam_score(
                Beam{beams[0].tokens, best_active, false},
                sample_begin, length_penalty
            );
            if (best_possible_score < best_finished_score) {
                break;
            }
        }
    }

    // =========================================================================
    // Select best beam
    // =========================================================================
    std::vector<Beam> all_beams;
    all_beams.insert(all_beams.end(), finished_beams.begin(), finished_beams.end());
    for (auto& b : beams) {
        if (!b.finished) {
            // Add EOT to unfinished beams
            b.tokens.push_back(config_.eot_token);
        }
        all_beams.push_back(b);
    }

    if (all_beams.empty()) {
        // Shouldn't happen, return empty result
        BeamSearchResult result;
        result.tokens = std::vector<int>(initial_tokens.begin() + sample_begin, initial_tokens.end());
        return result;
    }

    // Score all beams and select best
    float best_score = -INFINITY;
    const Beam* best_beam = nullptr;
    for (const auto& beam : all_beams) {
        float score = compute_beam_score(beam, sample_begin, length_penalty);
        if (score > best_score) {
            best_score = score;
            best_beam = &beam;
        }
    }

    // Prepare result
    BeamSearchResult result;
    result.tokens = std::vector<int>(best_beam->tokens.begin() + sample_begin, best_beam->tokens.end());

    // Remove trailing EOT
    while (!result.tokens.empty() && result.tokens.back() == config_.eot_token) {
        result.tokens.pop_back();
    }

    result.log_prob = best_beam->log_prob;
    result.normalized_score = best_score;
    result.avg_logprob = result.tokens.empty() ? 0.0f :
                         best_beam->log_prob / static_cast<float>(result.tokens.size());
    result.no_speech_prob = no_speech_prob;  // GAP 13 FIX

    return result;
}

void WhisperModel::init_language_tokens() {
    // Whisper multilingual language tokens
    // Token IDs start at 50259 for multilingual models
    language_tokens_ = {
        {"en", 50259}, {"zh", 50260}, {"de", 50261}, {"es", 50262},
        {"ru", 50263}, {"ko", 50264}, {"fr", 50265}, {"ja", 50266},
        {"pt", 50267}, {"tr", 50268}, {"pl", 50269}, {"ca", 50270},
        {"nl", 50271}, {"ar", 50272}, {"sv", 50273}, {"it", 50274},
        {"id", 50275}, {"hi", 50276}, {"fi", 50277}, {"vi", 50278},
        {"he", 50279}, {"uk", 50280}, {"el", 50281}, {"ms", 50282},
        {"cs", 50283}, {"ro", 50284}, {"da", 50285}, {"hu", 50286},
        {"ta", 50287}, {"no", 50288}, {"th", 50289}, {"ur", 50290},
        {"hr", 50291}, {"bg", 50292}, {"lt", 50293}, {"la", 50294},
        {"mi", 50295}, {"ml", 50296}, {"cy", 50297}, {"sk", 50298},
        {"te", 50299}, {"fa", 50300}, {"lv", 50301}, {"bn", 50302},
        {"sr", 50303}, {"az", 50304}, {"sl", 50305}, {"kn", 50306},
        {"et", 50307}, {"mk", 50308}, {"br", 50309}, {"eu", 50310},
        {"is", 50311}, {"hy", 50312}, {"ne", 50313}, {"mn", 50314},
        {"bs", 50315}, {"kk", 50316}, {"sq", 50317}, {"sw", 50318},
        {"gl", 50319}, {"mr", 50320}, {"pa", 50321}, {"si", 50322},
        {"km", 50323}, {"sn", 50324}, {"yo", 50325}, {"so", 50326},
        {"af", 50327}, {"oc", 50328}, {"ka", 50329}, {"be", 50330},
        {"tg", 50331}, {"sd", 50332}, {"gu", 50333}, {"am", 50334},
        {"yi", 50335}, {"lo", 50336}, {"uz", 50337}, {"fo", 50338},
        {"ht", 50339}, {"ps", 50340}, {"tk", 50341}, {"nn", 50342},
        {"mt", 50343}, {"sa", 50344}, {"lb", 50345}, {"my", 50346},
        {"bo", 50347}, {"tl", 50348}, {"mg", 50349}, {"as", 50350},
        {"tt", 50351}, {"haw", 50352}, {"ln", 50353}, {"ha", 50354},
        {"ba", 50355}, {"jw", 50356}, {"su", 50357},
        // GAP 69: TO_LANGUAGE_CODE aliases (Python: tokenizer.py TO_LANGUAGE_CODE)
        // These aliases map common language names to their ISO codes
        {"burmese", 50346},     // -> my (myanmar)
        {"valencian", 50270},   // -> ca (catalan)
        {"flemish", 50271},     // -> nl (dutch)
        {"haitian", 50339},     // -> ht (haitian creole)
        {"letzeburgesch", 50345}, // -> lb (luxembourgish)
        {"pushto", 50340},      // -> ps (pashto)
        {"panjabi", 50321},     // -> pa (punjabi)
        {"moldavian", 50284},   // -> ro (romanian)
        {"moldovan", 50284},    // -> ro (romanian)
        {"sinhalese", 50322},   // -> si (sinhala)
        {"castilian", 50262},   // -> es (spanish)
        {"mandarin", 50260},    // -> zh (chinese)
        // Full language name aliases (matching Python LANGUAGES dict reverse lookup)
        {"english", 50259}, {"chinese", 50260}, {"german", 50261}, {"spanish", 50262},
        {"russian", 50263}, {"korean", 50264}, {"french", 50265}, {"japanese", 50266},
        {"portuguese", 50267}, {"turkish", 50268}, {"polish", 50269}, {"catalan", 50270},
        {"dutch", 50271}, {"arabic", 50272}, {"swedish", 50273}, {"italian", 50274},
        {"indonesian", 50275}, {"hindi", 50276}, {"finnish", 50277}, {"vietnamese", 50278},
        {"hebrew", 50279}, {"ukrainian", 50280}, {"greek", 50281}, {"malay", 50282},
        {"czech", 50283}, {"romanian", 50284}, {"danish", 50285}, {"hungarian", 50286},
        {"tamil", 50287}, {"norwegian", 50288}, {"thai", 50289}, {"urdu", 50290},
        {"croatian", 50291}, {"bulgarian", 50292}, {"lithuanian", 50293}, {"latin", 50294},
        {"maori", 50295}, {"malayalam", 50296}, {"welsh", 50297}, {"slovak", 50298},
        {"telugu", 50299}, {"persian", 50300}, {"latvian", 50301}, {"bengali", 50302},
        {"serbian", 50303}, {"azerbaijani", 50304}, {"slovenian", 50305}, {"kannada", 50306},
        {"estonian", 50307}, {"macedonian", 50308}, {"breton", 50309}, {"basque", 50310},
        {"icelandic", 50311}, {"armenian", 50312}, {"nepali", 50313}, {"mongolian", 50314},
        {"bosnian", 50315}, {"kazakh", 50316}, {"albanian", 50317}, {"swahili", 50318},
        {"galician", 50319}, {"marathi", 50320}, {"punjabi", 50321}, {"sinhala", 50322},
        {"khmer", 50323}, {"shona", 50324}, {"yoruba", 50325}, {"somali", 50326},
        {"afrikaans", 50327}, {"occitan", 50328}, {"georgian", 50329}, {"belarusian", 50330},
        {"tajik", 50331}, {"sindhi", 50332}, {"gujarati", 50333}, {"amharic", 50334},
        {"yiddish", 50335}, {"lao", 50336}, {"uzbek", 50337}, {"faroese", 50338},
        {"haitian creole", 50339}, {"pashto", 50340}, {"turkmen", 50341}, {"nynorsk", 50342},
        {"maltese", 50343}, {"sanskrit", 50344}, {"luxembourgish", 50345}, {"myanmar", 50346},
        {"tibetan", 50347}, {"tagalog", 50348}, {"malagasy", 50349}, {"assamese", 50350},
        {"tatar", 50351}, {"hawaiian", 50352}, {"lingala", 50353}, {"hausa", 50354},
        {"bashkir", 50355}, {"javanese", 50356}, {"sundanese", 50357},
        // Cantonese (yue) - not in original Whisper but in large-v3 extended vocab
        {"yue", 50358}, {"cantonese", 50358},
    };

    // Build reverse lookup map (token ID -> language code)
    // Note: This maps to the primary ISO code, not aliases
    token_to_language_.clear();
    for (const auto& [code, token] : language_tokens_) {
        // Only add if this is a 2-3 letter ISO code (not an alias name)
        if (code.length() <= 3) {
            token_to_language_[token] = code;
        }
    }
}

int WhisperModel::get_language_token(const std::string& language) const {
    auto it = language_tokens_.find(language);
    if (it != language_tokens_.end()) {
        return it->second;
    }
    // Default to English
    return 50259;
}

std::string WhisperModel::get_language_code(int token) const {
    auto it = token_to_language_.find(token);
    if (it != token_to_language_.end()) {
        return it->second;
    }
    // Default to English
    return "en";
}

bool WhisperModel::is_multilingual() const {
    // Multilingual models have vocab >= 51865
    return config_.n_vocab >= 51865;
}

int WhisperModel::num_languages() const {
    // n_vocab - 51765 - int(is_multilingual)
    // For large-v3: 51866 - 51765 - 1 = 100 languages
    return config_.n_vocab - 51765 - (is_multilingual() ? 1 : 0);
}

DetectLanguageResult WhisperModel::detect_language(const mx::array& mel) {
    DetectLanguageResult result;

    if (!is_multilingual()) {
        // English-only model
        result.language = "en";
        result.language_token = 50259;
        result.probability = 1.0f;
        result.language_probs["en"] = 1.0f;
        return result;
    }

    // Check if input is already encoder output (shape: [batch, n_audio_ctx, n_audio_state])
    // vs mel spectrogram (shape: [batch, n_frames, n_mels])
    mx::array encoder_output = mel;  // Start with input
    bool is_encoded = (mel.shape().size() >= 2 &&
        static_cast<int>(mel.shape(-2)) == config_.n_audio_ctx &&
        static_cast<int>(mel.shape(-1)) == config_.n_audio_state);

    if (!is_encoded) {
        // Need to encode
        encoder_output = encode(mel);
    }

    // Ensure batch dimension
    mx::array enc_out = encoder_output;
    if (enc_out.ndim() == 2) {
        enc_out = mx::expand_dims(enc_out, 0);
    }

    // Forward pass with just SOT token
    std::vector<int> sot_tokens = {config_.sot_token};
    mx::array tokens = mx::array(sot_tokens.data(), {1, 1}, mx::int32);

    // Get logits from decoder (no KV cache for fresh inference)
    mx::array logits = decode(tokens, enc_out);
    mx::eval(logits);

    // Get logits at position 0 (after SOT)
    // logits shape: [batch, seq_len, n_vocab] -> [n_vocab]
    // Use mx::slice to extract [0, 0, :] then squeeze
    mx::array position_logits = mx::squeeze(mx::slice(logits, {0, 0, 0}, {1, 1, config_.n_vocab}));
    mx::eval(position_logits);

    // Create mask for non-language tokens (set to -inf)
    // Language tokens are 50259-50357 (99 languages)
    const int first_lang_token = 50259;
    const int last_lang_token = 50357;
    const int n_vocab = config_.n_vocab;

    std::vector<float> mask_data(n_vocab, -std::numeric_limits<float>::infinity());
    for (int i = first_lang_token; i <= last_lang_token && i < n_vocab; ++i) {
        mask_data[i] = 0.0f;
    }
    mx::array mask = mx::array(mask_data.data(), {n_vocab}, mx::float32);

    // Apply mask to logits
    mx::array masked_logits = position_logits + mask;
    mx::eval(masked_logits);

    // Find argmax (detected language token)
    mx::array lang_token_arr = mx::argmax(masked_logits);
    mx::eval(lang_token_arr);

    // Convert to int
    std::vector<int32_t> lang_token_vec(1);
    memcpy(lang_token_vec.data(), lang_token_arr.data<int32_t>(), sizeof(int32_t));
    int detected_token = lang_token_vec[0];

    // Compute softmax over masked logits for probability distribution
    mx::array probs = mx::softmax(masked_logits, -1);
    mx::eval(probs);

    // Extract probabilities
    std::vector<float> probs_vec(n_vocab);
    memcpy(probs_vec.data(), probs.data<float>(), n_vocab * sizeof(float));

    // Build language probability map
    for (const auto& [code, token] : language_tokens_) {
        if (token < n_vocab) {
            result.language_probs[code] = probs_vec[token];
        }
    }

    // Set result
    result.language = get_language_code(detected_token);
    result.language_token = detected_token;
    result.probability = probs_vec[detected_token];

    return result;
}

std::string WhisperModel::info() const {
    std::ostringstream oss;
    oss << "WhisperModel: " << config_.name << "\n";
    oss << "  Encoder: " << config_.n_audio_layer << " layers, "
        << config_.n_audio_state << "d, " << config_.n_audio_head << " heads\n";
    oss << "  Decoder: " << config_.n_text_layer << " layers, "
        << config_.n_text_state << "d, " << config_.n_text_head << " heads\n";
    oss << "  Vocab: " << config_.n_vocab << ", Mels: " << config_.n_mels;
    return oss.str();
}

// ============================================================================
// Word-Level Timestamps via DTW Alignment
// ============================================================================

namespace {

/**
 * Dynamic Time Warping (DTW) for aligning tokens to audio frames.
 *
 * Given a cost matrix of shape [n_tokens, n_frames], finds the optimal
 * alignment path from (0,0) to (n_tokens-1, n_frames-1).
 *
 * Returns vector of (token_idx, frame_idx) pairs representing the alignment.
 */
std::vector<std::pair<int, int>> dtw_backtrace(
    const std::vector<std::vector<float>>& cost_matrix,
    int n_tokens,
    int n_frames
) {
    // Compute cumulative cost matrix with DTW constraints
    // DTW allows: move right, move down, or move diagonally
    std::vector<std::vector<float>> D(n_tokens, std::vector<float>(n_frames, INFINITY));
    std::vector<std::vector<int>> trace(n_tokens, std::vector<int>(n_frames, -1));
    // trace values: 0 = from left, 1 = from below, 2 = from diagonal

    // Initialize first cell
    D[0][0] = cost_matrix[0][0];

    // Initialize first row (can only come from left)
    for (int j = 1; j < n_frames; ++j) {
        D[0][j] = D[0][j-1] + cost_matrix[0][j];
        trace[0][j] = 0;  // from left
    }

    // Initialize first column (can only come from below)
    for (int i = 1; i < n_tokens; ++i) {
        D[i][0] = D[i-1][0] + cost_matrix[i][0];
        trace[i][0] = 1;  // from below
    }

    // Fill rest of matrix
    for (int i = 1; i < n_tokens; ++i) {
        for (int j = 1; j < n_frames; ++j) {
            float from_left = D[i][j-1];      // Move right in time, same token
            float from_below = D[i-1][j];     // Move to next token, same time
            float from_diag = D[i-1][j-1];    // Both advance

            float min_cost = from_diag;
            int trace_val = 2;

            if (from_left < min_cost) {
                min_cost = from_left;
                trace_val = 0;
            }
            if (from_below < min_cost) {
                min_cost = from_below;
                trace_val = 1;
            }

            D[i][j] = min_cost + cost_matrix[i][j];
            trace[i][j] = trace_val;
        }
    }

    // Backtrace from (n_tokens-1, n_frames-1)
    std::vector<std::pair<int, int>> path;
    int i = n_tokens - 1;
    int j = n_frames - 1;

    while (i >= 0 && j >= 0) {
        path.push_back({i, j});
        if (i == 0 && j == 0) break;

        int t = trace[i][j];
        if (t == 0) {
            // from left
            j--;
        } else if (t == 1) {
            // from below
            i--;
        } else {
            // diagonal
            i--;
            j--;
        }
    }

    // Reverse to get path from start to end
    std::reverse(path.begin(), path.end());
    return path;
}

/**
 * Apply 1D median filter along last axis (matching Python scipy.signal.medfilt).
 * GAP 27: Python uses medfilt_width=7, we need to implement this.
 */
void apply_median_filter_1d(std::vector<float>& data, int n_rows, int n_cols, int filter_width) {
    if (filter_width <= 1 || n_cols <= filter_width / 2) {
        return;  // No filtering needed
    }

    int pad = filter_width / 2;
    std::vector<float> padded(n_cols + 2 * pad);
    std::vector<float> window(filter_width);

    for (int row = 0; row < n_rows; ++row) {
        float* row_data = data.data() + row * n_cols;

        // Reflect padding (matching Python mode='reflect')
        for (int i = 0; i < pad; ++i) {
            padded[i] = row_data[pad - i];  // Left reflect
            padded[n_cols + pad + i] = row_data[n_cols - 2 - i];  // Right reflect
        }
        for (int i = 0; i < n_cols; ++i) {
            padded[pad + i] = row_data[i];
        }

        // Apply median filter
        for (int i = 0; i < n_cols; ++i) {
            for (int j = 0; j < filter_width; ++j) {
                window[j] = padded[i + j];
            }
            std::nth_element(window.begin(), window.begin() + filter_width / 2, window.end());
            row_data[i] = window[filter_width / 2];
        }
    }
}

/**
 * Compute alignment from cross-attention weights using DTW.
 * GAP 26: Uses specific alignment_heads (layer, head) pairs from model config.
 * GAP 27: Applies median filter with configurable width.
 *
 * Matches Python mlx-whisper timing.py find_alignment():
 * 1. Extract weights from specific (layer, head) pairs
 * 2. Slice to num_frames // 2 (encoder downsamples by 2)
 * 3. Apply softmax with qk_scale
 * 4. Normalize (mean/std per token)
 * 5. Apply median filter
 * 6. Average across heads
 * 7. Run DTW on negative matrix
 *
 * @param cross_attention_weights Vector of [batch, n_head, n_tokens, n_frames] per layer
 * @param alignment_heads Specific (layer, head) pairs to use. Empty = all heads from last half.
 * @param n_text_layer Total number of decoder layers
 * @param n_text_head Number of attention heads
 * @param max_frames Maximum valid frames (0 = use all). Limits DTW to actual audio, not padding.
 * @param medfilt_width Median filter width (default 7, matching Python)
 * @param qk_scale Cross-attention QK scaling factor (default 1.0)
 * @return Vector of frame indices, one per token
 */
std::vector<int> compute_alignment_from_attention(
    const std::vector<mx::array>& cross_attention_weights,
    const std::vector<std::pair<int, int>>& alignment_heads,
    int n_text_layer,
    int n_text_head,
    int max_frames = 0,
    int medfilt_width = 7,
    float qk_scale = 1.0f
) {
    if (cross_attention_weights.empty()) {
        return {};
    }

    // Determine which heads to use
    std::vector<std::pair<int, int>> heads_to_use;
    if (!alignment_heads.empty()) {
        // GAP 26: Use specific alignment heads from model config
        heads_to_use = alignment_heads;
    } else {
        // Fallback: use all heads from last half of layers (default behavior)
        int start_layer = n_text_layer / 2;
        for (int layer = start_layer; layer < n_text_layer; ++layer) {
            for (int head = 0; head < n_text_head; ++head) {
                heads_to_use.push_back({layer, head});
            }
        }
    }

    if (heads_to_use.empty()) {
        return {};
    }

    // Get dimensions from first valid attention array
    int first_layer = heads_to_use[0].first;
    if (first_layer >= static_cast<int>(cross_attention_weights.size())) {
        return {};
    }
    auto& first = cross_attention_weights[first_layer];
    mx::eval(first);
    auto shape = first.shape();
    int n_tokens = static_cast<int>(shape[2]);
    int n_frames_full = static_cast<int>(shape[3]);

    // Python: weights = weights[:, :, : num_frames // 2]
    // Encoder downsamples by 2, so we use half the frames
    int n_frames = n_frames_full / 2;
    if (max_frames > 0 && max_frames < n_frames) {
        n_frames = max_frames;
    }

    if (n_tokens <= 0 || n_frames <= 0) {
        return {};
    }

    // Extract and stack attention weights from specific heads
    // Python: weights = mx.stack([cross_qk[_l][0, _h] for _l, _h in model.alignment_heads.tolist()])
    std::vector<mx::array> head_weights;
    head_weights.reserve(heads_to_use.size());

    for (const auto& [layer, head] : heads_to_use) {
        if (layer >= static_cast<int>(cross_attention_weights.size())) continue;
        auto& attn = cross_attention_weights[layer];
        mx::eval(attn);

        // attn shape: [batch, n_head, n_tokens, n_frames_full]
        // Extract [batch=0, head=head, :n_tokens, :n_frames]
        auto head_attn = mx::slice(attn, {0, head, 0, 0}, {1, head + 1, n_tokens, n_frames});
        head_attn = mx::squeeze(head_attn, {0, 1});  // [n_tokens, n_frames]
        mx::eval(head_attn);
        head_weights.push_back(head_attn);
    }

    if (head_weights.empty()) {
        return {};
    }

    // Stack heads: [n_heads, n_tokens, n_frames]
    auto weights = mx::stack(head_weights, 0);
    mx::eval(weights);

    // Python: weights = mx.softmax(weights * qk_scale, axis=-1, precise=True)
    weights = mx::softmax(weights * qk_scale, -1, true);
    weights = mx::astype(weights, mx::float32);
    mx::eval(weights);

    // Python: Normalize per head (mean/std across frames)
    // mean = mx.mean(weights, axis=-2, keepdims=True)
    // std = mx.var(weights, axis=-2, keepdims=True, ddof=0).sqrt()
    // weights = (weights - mean) / std
    auto mean = mx::mean(weights, std::vector<int>{1}, true);  // [n_heads, 1, n_frames]
    auto var = mx::var(weights, std::vector<int>{1}, true);     // [n_heads, 1, n_frames]
    auto std_dev = mx::sqrt(mx::maximum(var, mx::array(1e-10f)));
    weights = (weights - mean) / std_dev;
    mx::eval(weights);

    // GAP 27: Apply median filter for smoothing
    // Python: weights = median_filter(np.array(weights), medfilt_width)
    int n_heads = static_cast<int>(heads_to_use.size());
    if (medfilt_width > 1) {
        // Copy to CPU for median filter
        std::vector<float> weights_data(n_heads * n_tokens * n_frames);
        auto weights_cpu = mx::astype(weights, mx::float32);
        mx::eval(weights_cpu);
        std::memcpy(weights_data.data(), weights_cpu.data<float>(),
                    weights_data.size() * sizeof(float));

        // Apply median filter along frames axis for each head's token rows
        for (int h = 0; h < n_heads; ++h) {
            std::vector<float> head_slice(weights_data.begin() + h * n_tokens * n_frames,
                                          weights_data.begin() + (h + 1) * n_tokens * n_frames);
            apply_median_filter_1d(head_slice, n_tokens, n_frames, medfilt_width);
            std::memcpy(weights_data.data() + h * n_tokens * n_frames, head_slice.data(),
                        n_tokens * n_frames * sizeof(float));
        }

        // Convert back to MLX array
        weights = mx::array(weights_data.data(), {n_heads, n_tokens, n_frames}, mx::float32);
        mx::eval(weights);
    }

    // Python: matrix = weights.mean(axis=0)  # Average across heads
    auto matrix = mx::mean(weights, std::vector<int>{0});  // [n_tokens, n_frames]
    mx::eval(matrix);

    // Python: text_indices, time_indices = dtw(-matrix)
    // Convert to cost matrix (negative for DTW)
    auto cost = -matrix;
    mx::eval(cost);

    // Copy to CPU for DTW
    std::vector<float> cost_data(n_tokens * n_frames);
    {
        auto cost_cpu = mx::astype(cost, mx::float32);
        mx::eval(cost_cpu);
        std::memcpy(cost_data.data(), cost_cpu.data<float>(), cost_data.size() * sizeof(float));
    }

    // Build 2D cost matrix
    std::vector<std::vector<float>> cost_matrix(n_tokens, std::vector<float>(n_frames));
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < n_frames; ++j) {
            cost_matrix[i][j] = cost_data[i * n_frames + j];
        }
    }

    // Run DTW
    auto path = dtw_backtrace(cost_matrix, n_tokens, n_frames);

    // Extract frame index for each token (use first occurrence in path)
    std::vector<int> token_frames(n_tokens, -1);
    for (const auto& [tok_idx, frame_idx] : path) {
        if (token_frames[tok_idx] < 0) {
            token_frames[tok_idx] = frame_idx;
        }
    }

    // Fill any gaps with interpolation
    for (int i = 1; i < n_tokens; ++i) {
        if (token_frames[i] < 0) {
            token_frames[i] = token_frames[i-1];
        }
    }

    return token_frames;
}

/**
 * Check if a token is a word boundary (starts with space or is punctuation).
 */
bool is_word_boundary(const std::string& token_text) {
    if (token_text.empty()) return false;

    // Check for leading space (BPE convention)
    // Whisper uses special byte encoding where 0xC4 0xA0 = space
    if (token_text[0] == ' ') return true;
    if (token_text.size() >= 2 &&
        static_cast<unsigned char>(token_text[0]) == 0xC4 &&
        static_cast<unsigned char>(token_text[1]) == 0xA0) {
        return true;
    }

    // Check for punctuation that typically ends words
    char first = token_text[0];
    if (first == '.' || first == ',' || first == '!' || first == '?' ||
        first == ':' || first == ';') {
        return true;
    }

    return false;
}

/**
 * Check if language is CJK (Chinese, Japanese, Korean) or other space-less languages.
 * These languages need special unicode-based word splitting.
 * GAP 30: Matches Python mlx-whisper tokenizer.split_to_word_tokens()
 */
bool is_spaceless_language(const std::string& language) {
    // Languages that don't use spaces between words
    // Matches Python: {"zh", "ja", "th", "lo", "my", "yue"}
    return language == "zh" || language == "ja" || language == "th" ||
           language == "lo" || language == "my" || language == "yue" ||
           language == "chinese" || language == "japanese" || language == "thai" ||
           language == "lao" || language == "myanmar" || language == "cantonese";
}

/**
 * Check if character is a valid unicode code point start.
 * Returns true if the byte sequence starting here is a complete character.
 */
bool is_valid_unicode_char(const std::string& text, size_t pos) {
    if (pos >= text.size()) return false;
    unsigned char c = static_cast<unsigned char>(text[pos]);
    if (c < 0x80) return true;  // ASCII
    if (c >= 0xC0 && c < 0xE0) return pos + 1 < text.size();  // 2-byte
    if (c >= 0xE0 && c < 0xF0) return pos + 2 < text.size();  // 3-byte
    if (c >= 0xF0 && c < 0xF8) return pos + 3 < text.size();  // 4-byte
    return false;
}

/**
 * Check if a word is punctuation only.
 * GAP 30: Used for proper word splitting.
 */
bool is_punctuation_word(const std::string& word) {
    // Strip leading space
    std::string stripped = word;
    if (!stripped.empty() && stripped[0] == ' ') {
        stripped = stripped.substr(1);
    }
    if (stripped.empty()) return false;

    // Check if all characters are punctuation
    for (char c : stripped) {
        if (!std::ispunct(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

/**
 * Check if text is a sentence end marker.
 * GAP 67: Used for truncating long words at sentence boundaries.
 */
bool is_sentence_end_mark(const std::string& text) {
    // Strip leading space
    std::string stripped = text;
    if (!stripped.empty() && stripped[0] == ' ') {
        stripped = stripped.substr(1);
    }
    // Sentence end marks (matches Python: ".。!！?？")
    return stripped == "." || stripped == "。" || stripped == "!" ||
           stripped == "！" || stripped == "?" || stripped == "？";
}

/**
 * Calculate word anomaly score.
 * GAP 18: Matches Python mlx-whisper word_anomaly_score()
 */
float word_anomaly_score(const WhisperWord& word) {
    float probability = word.probability;
    float duration = word.end_time - word.start_time;
    float score = 0.0f;

    if (probability < 0.15f) {
        score += 1.0f;
    }
    if (duration < 0.133f) {
        score += (0.133f - duration) * 15.0f;
    }
    if (duration > 2.0f) {
        score += duration - 2.0f;
    }
    return score;
}

/**
 * Check if a segment is anomalous based on word scores.
 * GAP 19: Matches Python mlx-whisper is_segment_anomaly()
 */
bool is_segment_anomaly(const std::vector<WhisperWord>& words) {
    if (words.empty()) return false;

    // Filter out punctuation words and take first 8
    std::vector<const WhisperWord*> filtered_words;
    for (const auto& w : words) {
        if (!is_punctuation_word(w.word) && filtered_words.size() < 8) {
            filtered_words.push_back(&w);
        }
    }

    if (filtered_words.empty()) return false;

    float score = 0.0f;
    for (const auto* w : filtered_words) {
        score += word_anomaly_score(*w);
    }

    return score >= 3.0f || (score + 0.01f >= static_cast<float>(filtered_words.size()));
}

/**
 * Calculate median of a vector of floats.
 * GAP 67: Used for median_duration calculation.
 */
float calculate_median(std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0f;
    }
    return values[n/2];
}

}  // namespace

void WhisperModel::add_word_timestamps(
    WhisperSegment& segment,
    const mx::array& encoder_output,
    const std::string& language,
    const WhisperTokenizer* tokenizer,
    float actual_audio_duration,
    float* last_speech_timestamp
) {
    if (segment.tokens.empty()) {
        return;
    }

    // Build token array for decoder input (include SOT prefix)
    std::vector<int> decoder_tokens = {config_.sot_token};
    int lang_token = get_language_token(language);
    decoder_tokens.push_back(lang_token);
    decoder_tokens.push_back(config_.transcribe_token);

    // Add segment tokens
    for (int tok : segment.tokens) {
        decoder_tokens.push_back(tok);
    }

    // Create token tensor
    auto tokens_mx = mx::array(decoder_tokens.data(),
                                {1, static_cast<int>(decoder_tokens.size())}, mx::int32);
    mx::eval(tokens_mx);

    // Decode with attention weights
    auto result = decoder_->decode_with_attention(tokens_mx, encoder_output);
    mx::eval(result.logits);
    for (auto& w : result.cross_attention_weights) {
        mx::eval(w);
    }

    // Get token probabilities from logits
    auto probs = mx::softmax(result.logits, -1);
    mx::eval(probs);

    // Time conversion constants
    const float time_per_frame = 0.02f;  // 20ms per encoder frame (stride 2 on 10ms mel frames)
    const float segment_start = segment.start_time;

    // Use actual audio duration if provided, otherwise fall back to segment timing
    // Segment timing from timestamp tokens is often compressed/internal timing
    // Actual audio duration gives correct frame alignment for DTW
    float segment_duration;
    if (actual_audio_duration > 0.0f) {
        segment_duration = actual_audio_duration;
    } else {
        segment_duration = segment.end_time - segment.start_time;
    }

    // Calculate valid frame count from audio duration
    // The encoder output has 1500 frames (30s) but actual audio may be shorter
    const int valid_frames = static_cast<int>(segment_duration / time_per_frame);

    // Compute alignment using DTW on cross-attention
    // GAP 26: Use specific alignment_heads from model config
    // GAP 27: Use medfilt_width from config (default 7, matching Python)
    auto token_frames = compute_alignment_from_attention(
        result.cross_attention_weights,
        config_.alignment_heads,  // GAP 26: specific heads from weights
        config_.n_text_layer,
        config_.n_text_head,
        valid_frames,
        config_.medfilt_width,    // GAP 27: median filter width (default 7)
        1.0f                      // qk_scale (default 1.0)
    );

    // Skip SOT prefix in alignment (first 3 tokens: SOT, lang, task)
    const int prefix_len = 3;
    if (token_frames.size() <= static_cast<size_t>(prefix_len)) {
        return;
    }

    // Build token timings
    std::vector<WhisperTokenTiming> token_timings;
    int n_segment_tokens = static_cast<int>(segment.tokens.size());

    for (int i = 0; i < n_segment_tokens; ++i) {
        int full_idx = prefix_len + i;
        if (full_idx >= static_cast<int>(token_frames.size())) break;

        int frame = token_frames[full_idx];
        float token_start = segment_start + frame * time_per_frame;

        // End time is start of next token, or segment end
        float token_end;
        if (i + 1 < n_segment_tokens && full_idx + 1 < static_cast<int>(token_frames.size())) {
            int next_frame = token_frames[full_idx + 1];
            token_end = segment_start + next_frame * time_per_frame;
        } else {
            token_end = segment.end_time;
        }

        // Get token probability
        float prob = 1.0f;  // Default
        // Note: Could extract from probs tensor if needed

        WhisperTokenTiming timing;
        timing.token_id = segment.tokens[i];
        timing.start_time = token_start;
        timing.end_time = token_end;
        timing.probability = prob;
        timing.dtw_score = 0.0f;  // Could compute from cost matrix

        token_timings.push_back(timing);
    }

    // Group tokens into words
    segment.words.clear();

    if (token_timings.empty()) {
        return;
    }

    if (tokenizer && tokenizer->loaded()) {
        // GAP 30: Handle CJK languages differently (no spaces between words)
        bool spaceless = is_spaceless_language(language);

        // Group tokens into words based on whitespace (or unicode for CJK)
        WhisperWord current_word;
        current_word.start_time = token_timings[0].start_time;
        current_word.probability = 1.0f;

        for (const auto& timing : token_timings) {
            // Decode single token to check for word boundary
            std::string token_text = tokenizer->decode({timing.token_id});

            bool is_word_start = false;

            if (spaceless) {
                // GAP 30: For CJK languages, each token that decodes to valid unicode is a "word"
                // This matches Python split_tokens_on_unicode behavior
                if (!current_word.tokens.empty()) {
                    // Check if current accumulated tokens form valid unicode
                    std::vector<int> current_ids;
                    for (const auto& t : current_word.tokens) {
                        current_ids.push_back(t.token_id);
                    }
                    std::string decoded = tokenizer->decode(current_ids);
                    // If decoded string doesn't contain replacement char, it's valid
                    if (decoded.find('\xEF') == std::string::npos) {  // U+FFFD starts with 0xEF in UTF-8
                        is_word_start = true;  // Start new word
                    }
                }
            } else {
                // For spaced languages, check for space or punctuation boundary
                // Check if this token starts a new word (begins with space)
                is_word_start = !token_text.empty() &&
                                 (token_text[0] == ' ' ||
                                  (token_text.size() >= 2 &&
                                   static_cast<unsigned char>(token_text[0]) == 0xC4 &&
                                   static_cast<unsigned char>(token_text[1]) == 0xA0));

                // Also check for punctuation that should be its own word
                if (!is_word_start && !token_text.empty() && !current_word.tokens.empty()) {
                    if (is_punctuation_word(token_text)) {
                        is_word_start = true;
                    }
                }
            }

            if (is_word_start && !current_word.tokens.empty()) {
                // Finalize current word
                current_word.end_time = timing.start_time;
                // Decode all tokens in the word to get full text
                std::vector<int> word_ids;
                for (const auto& t : current_word.tokens) {
                    word_ids.push_back(t.token_id);
                }
                current_word.word = tokenizer->decode(word_ids);
                segment.words.push_back(current_word);

                // Start new word
                current_word = WhisperWord();
                current_word.start_time = timing.start_time;
                current_word.probability = 1.0f;
            }

            current_word.tokens.push_back(timing);
            // Update probability (geometric mean approximation)
            current_word.probability = std::min(current_word.probability, timing.probability);
        }

        // Don't forget the last word
        if (!current_word.tokens.empty()) {
            current_word.end_time = token_timings.back().end_time;
            std::vector<int> word_ids;
            for (const auto& t : current_word.tokens) {
                word_ids.push_back(t.token_id);
            }
            current_word.word = tokenizer->decode(word_ids);
            segment.words.push_back(current_word);
        }

        // GAP 67: Calculate median_duration and apply truncation at sentence boundaries
        if (!segment.words.empty()) {
            std::vector<float> word_durations;
            for (const auto& w : segment.words) {
                float dur = w.end_time - w.start_time;
                if (dur > 0.0f) {
                    word_durations.push_back(dur);
                }
            }

            float median_duration = calculate_median(word_durations);
            median_duration = std::min(0.7f, median_duration);  // Cap at 0.7s
            float max_duration = median_duration * 2.0f;

            // Truncate long words at sentence boundaries
            if (!word_durations.empty() && max_duration > 0.0f) {
                for (size_t i = 1; i < segment.words.size(); ++i) {
                    float word_dur = segment.words[i].end_time - segment.words[i].start_time;
                    if (word_dur > max_duration) {
                        // Check if current word is sentence end mark
                        if (is_sentence_end_mark(segment.words[i].word)) {
                            segment.words[i].end_time = segment.words[i].start_time + max_duration;
                        }
                        // Check if previous word is sentence end mark
                        else if (is_sentence_end_mark(segment.words[i-1].word)) {
                            segment.words[i].start_time = segment.words[i].end_time - max_duration;
                        }
                    }
                }
            }

            // Merge punctuations (prepend/append punctuation to adjacent words)
            // GAP timing.py merge_punctuations()
            const std::string prepend_punct = "\"'¿([{-";
            const std::string append_punct = "\"'.。,，!！?？:：\")]}、";

            // Merge appended punctuations (scan forward)
            for (size_t i = 0; i + 1 < segment.words.size(); ) {
                auto& prev_word = segment.words[i];
                auto& next_word = segment.words[i + 1];

                // Strip leading space from next word for check
                std::string stripped_next = next_word.word;
                if (!stripped_next.empty() && stripped_next[0] == ' ') {
                    stripped_next = stripped_next.substr(1);
                }

                if (!prev_word.word.empty() && !prev_word.word.empty() &&
                    prev_word.word.back() != ' ' &&
                    stripped_next.size() == 1 &&
                    append_punct.find(stripped_next[0]) != std::string::npos) {
                    // Append punctuation to previous word
                    prev_word.word += next_word.word;
                    prev_word.tokens.insert(prev_word.tokens.end(),
                                           next_word.tokens.begin(),
                                           next_word.tokens.end());
                    prev_word.end_time = next_word.end_time;
                    // Remove the punctuation word
                    segment.words.erase(segment.words.begin() + i + 1);
                } else {
                    ++i;
                }
            }

            // Merge prepended punctuations (scan backward)
            for (int i = static_cast<int>(segment.words.size()) - 2; i >= 0; --i) {
                auto& prev_word = segment.words[i];
                auto& next_word = segment.words[i + 1];

                // Strip leading space from prev word for check
                std::string stripped_prev = prev_word.word;
                if (!stripped_prev.empty() && stripped_prev[0] == ' ') {
                    stripped_prev = stripped_prev.substr(1);
                }

                if (prev_word.word.size() > 0 && prev_word.word[0] == ' ' &&
                    stripped_prev.size() == 1 &&
                    prepend_punct.find(stripped_prev[0]) != std::string::npos) {
                    // Prepend punctuation to following word
                    next_word.word = prev_word.word + next_word.word;
                    next_word.tokens.insert(next_word.tokens.begin(),
                                           prev_word.tokens.begin(),
                                           prev_word.tokens.end());
                    next_word.start_time = prev_word.start_time;
                    // Clear the prepended punctuation word (will be filtered later)
                    prev_word.word = "";
                    prev_word.tokens.clear();
                }
            }

            // Remove empty words from punctuation merging
            segment.words.erase(
                std::remove_if(segment.words.begin(), segment.words.end(),
                              [](const WhisperWord& w) { return w.word.empty() || w.tokens.empty(); }),
                segment.words.end());
        }
    } else {
        // No tokenizer - create one "word" per token
        for (const auto& timing : token_timings) {
            WhisperWord word;
            word.word = "";  // No text without tokenizer
            word.start_time = timing.start_time;
            word.end_time = timing.end_time;
            word.probability = timing.probability;
            word.tokens.push_back(timing);
            segment.words.push_back(word);
        }
    }

    // GAP 78-80: Pause truncation and segment reconciliation
    // Apply after word timestamps are computed
    if (!segment.words.empty()) {
        // Recalculate median/max for reconciliation (may have changed after merging)
        std::vector<float> word_durations;
        for (const auto& w : segment.words) {
            float dur = w.end_time - w.start_time;
            if (dur > 0.0f) {
                word_durations.push_back(dur);
            }
        }
        float median_duration = calculate_median(word_durations);
        median_duration = std::min(0.7f, median_duration);
        float max_duration = median_duration * 2.0f;

        // Get effective last_speech_timestamp
        float prev_speech_ts = 0.0f;
        if (last_speech_timestamp != nullptr) {
            prev_speech_ts = *last_speech_timestamp;
        }

        // GAP 78: First word after pause truncation
        // If first word ends much later than last speech (> 4x median), truncate
        auto& words = segment.words;
        if (words[0].end_time - prev_speech_ts > median_duration * 4.0f &&
            (words[0].end_time - words[0].start_time > max_duration ||
             (words.size() > 1 &&
              words[1].end_time - words[0].start_time > max_duration * 2.0f))) {

            // GAP 79: Second word boundary adjustment
            if (words.size() > 1 &&
                words[1].end_time - words[1].start_time > max_duration) {
                float boundary = std::max(words[1].end_time / 2.0f,
                                         words[1].end_time - max_duration);
                words[0].end_time = boundary;
                words[1].start_time = boundary;
            }
            // Truncate first word
            words[0].start_time = std::max(0.0f, words[0].end_time - max_duration);
        }

        // GAP 80: Prefer segment-level timestamps over word timestamps when appropriate

        // Prefer segment start if first word is too long
        if (segment.start_time < words[0].end_time &&
            segment.start_time - 0.5f > words[0].start_time) {
            words[0].start_time = std::max(
                0.0f,
                std::min(words[0].end_time - median_duration, segment.start_time));
        } else {
            // Update segment start to match first word
            segment.start_time = words[0].start_time;
        }

        // Prefer segment end if last word is too long
        auto& last_word = words.back();
        if (segment.end_time > last_word.start_time &&
            segment.end_time + 0.5f < last_word.end_time) {
            last_word.end_time = std::max(
                last_word.start_time + median_duration,
                segment.end_time);
        } else {
            // Update segment end to match last word
            segment.end_time = last_word.end_time;
        }

        // GAP 68: Update last_speech_timestamp for next segment
        if (last_speech_timestamp != nullptr) {
            *last_speech_timestamp = segment.end_time;
        }
    }
}

// ============================================================================
// Segment-Based Decoding for Long Audio
// ============================================================================

namespace {

// Constants for segment-based decoding
constexpr int N_FRAMES = 3000;        // 30 seconds at 100 frames/sec
constexpr float TIME_PRECISION = 0.02f;  // 20ms per timestamp token
constexpr int N_SAMPLES_PER_FRAME = 160; // hop_length
constexpr int SAMPLE_RATE = 16000;

// GAP 46: Helper to check if a token starts a new word (has leading space)
// Tokens that start with space or special unicode space marker indicate word boundaries
bool is_word_boundary_token(int token_id, const WhisperTokenizer* tokenizer) {
    if (!tokenizer || !tokenizer->loaded()) return false;

    std::string token_text = tokenizer->decode({token_id});
    if (token_text.empty()) return false;

    // Check for leading space (ASCII or Unicode space like Ġ which is 0xC4 0xA0)
    if (token_text[0] == ' ') return true;
    if (token_text.size() >= 2 &&
        static_cast<unsigned char>(token_text[0]) == 0xC4 &&
        static_cast<unsigned char>(token_text[1]) == 0xA0) return true;

    return false;
}

// Parse segments from generated tokens
// Returns a vector of (start_time, end_time, text_tokens) tuples
// GAP 46: When split_on_word is true, segment boundaries prefer word boundaries
std::vector<std::tuple<float, float, std::vector<int>>> parse_segments_from_tokens(
    const std::vector<int>& tokens,
    int timestamp_begin,
    int eot_token,
    bool split_on_word = false,
    const WhisperTokenizer* tokenizer = nullptr
) {
    // Parse segments from timestamp token pairs
    // Whisper produces: <|start|> text tokens... <|end|> <|start|> text... <|end|>
    // Timestamps come in pairs: start timestamp followed by text, then end timestamp
    //
    // Special tokens to skip:
    // - EOT (50257): end of transcript
    // - SOT (50258): start of transcript
    // - Language tokens (50259-50357): language markers like <|en|>
    // - Task tokens (50358-50363): translate, transcribe, notimestamps, etc.
    // - Timestamps (>=50364): handled separately
    //
    // Text tokens are 0-50256 (standard vocabulary)
    constexpr int MAX_TEXT_TOKEN = 50256;  // Last text token before special tokens

    std::vector<std::tuple<float, float, std::vector<int>>> segments;

    float start_time = -1.0f;  // -1 indicates no active segment
    float end_time = -1.0f;
    std::vector<int> current_tokens;
    bool have_start_ts = false;

    for (int tok : tokens) {
        if (tok == eot_token) {
            // End of transcript - close any open segment and return
            if (!current_tokens.empty() && have_start_ts) {
                // Use last known end time or estimate from start
                float final_end = (end_time >= start_time) ? end_time : start_time + 0.5f;
                segments.push_back({start_time, final_end, current_tokens});
            }
            return segments;  // Return immediately - no more processing needed
        }

        if (tok >= timestamp_begin) {
            // This is a timestamp token
            float time = (tok - timestamp_begin) * TIME_PRECISION;

            if (!have_start_ts) {
                // First timestamp - marks start of segment
                start_time = time;
                have_start_ts = true;
            } else if (current_tokens.empty()) {
                // Another timestamp right after a timestamp (consecutive timestamps)
                // This is the end timestamp of an empty segment - update start for next segment
                // OR this is a new start - treat as new start
                start_time = time;
                // keep have_start_ts = true
            } else {
                // Have tokens - this timestamp closes the segment
                end_time = time;
                segments.push_back({start_time, end_time, current_tokens});
                current_tokens.clear();
                // Reset for next segment
                have_start_ts = false;
                start_time = -1.0f;
                end_time = -1.0f;
            }
        } else if (tok <= MAX_TEXT_TOKEN) {
            // Actual text token (0-50256)
            current_tokens.push_back(tok);
            // If no start timestamp yet, use 0.0
            if (!have_start_ts) {
                start_time = 0.0f;
                have_start_ts = true;
            }
        }
        // else: Skip special tokens (SOT, language, task tokens) - they're not text
    }

    // Handle any remaining tokens without closing timestamp
    if (!current_tokens.empty() && have_start_ts) {
        float final_end = (end_time >= start_time) ? end_time : start_time + 0.5f;
        segments.push_back({start_time, final_end, current_tokens});
    }

    // GAP 46: If split_on_word is enabled, post-process to align segment boundaries to word boundaries
    // This splits segments that may have been created mid-word
    if (split_on_word && tokenizer && tokenizer->loaded() && segments.size() > 1) {
        // For each segment boundary, check if we should adjust the split point
        // Look for the nearest word boundary token and redistribute tokens accordingly
        std::vector<std::tuple<float, float, std::vector<int>>> adjusted_segments;

        for (size_t i = 0; i < segments.size(); ++i) {
            auto& [seg_start, seg_end, seg_tokens] = segments[i];

            if (i == 0) {
                // First segment - check if last few tokens should move to next segment
                if (i + 1 < segments.size() && seg_tokens.size() > 1) {
                    // Find last word boundary in current segment
                    size_t last_word_boundary = 0;
                    for (size_t j = 1; j < seg_tokens.size(); ++j) {
                        if (is_word_boundary_token(seg_tokens[j], tokenizer)) {
                            last_word_boundary = j;
                        }
                    }

                    // If word boundary found in last 1/3 of segment, split there
                    if (last_word_boundary > seg_tokens.size() * 2 / 3) {
                        // Move tokens after word boundary to next segment
                        auto& [next_start, next_end, next_tokens] = segments[i + 1];
                        std::vector<int> to_move(seg_tokens.begin() + last_word_boundary, seg_tokens.end());
                        seg_tokens.resize(last_word_boundary);
                        next_tokens.insert(next_tokens.begin(), to_move.begin(), to_move.end());

                        if (std::getenv("DEBUG_WHISPER")) {
                            std::cerr << "[DEBUG] GAP 46: Moved " << to_move.size()
                                      << " tokens to next segment at word boundary\n";
                        }
                    }
                }
            }

            adjusted_segments.push_back(segments[i]);
        }

        return adjusted_segments;
    }

    return segments;
}

// Calculate seek advancement based on timestamp tokens
// Uses mlx-whisper's approach: find consecutive timestamps and use the last one
// Returns the number of mel frames to advance
//
// Whisper outputs timestamps like: <|0.00|> text <|3.20|> <|3.20|> text <|5.40|>
// Consecutive timestamps (two timestamps in a row) indicate segment boundaries
// The second of a consecutive pair is the end timestamp - advance to that position
int calculate_seek_advance(
    const std::vector<int>& tokens,
    int timestamp_begin,
    int segment_size,
    int eot_token
) {
    // input_stride: 2 mel frames per timestamp unit (matches mlx-whisper)
    constexpr int INPUT_STRIDE = 2;

    // Find consecutive timestamp tokens (indicates segment boundary)
    // Looking for patterns where tokens[i] and tokens[i+1] are both timestamps
    int last_timestamp_pos = -1;

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        bool curr_is_ts = tokens[i] >= timestamp_begin && tokens[i] != eot_token;
        bool next_is_ts = tokens[i + 1] >= timestamp_begin && tokens[i + 1] != eot_token;

        if (curr_is_ts && next_is_ts) {
            // Found consecutive timestamps - the second one is the boundary
            last_timestamp_pos = tokens[i + 1] - timestamp_begin;
        }
    }

    // Also check if the last non-EOT token is a timestamp (single timestamp ending)
    // This happens when there's no speech after the last timestamp
    if (!tokens.empty()) {
        int last_tok = tokens.back();
        if (last_tok == eot_token && tokens.size() >= 2) {
            int second_last = tokens[tokens.size() - 2];
            if (second_last >= timestamp_begin) {
                // Single timestamp before EOT - treat as segment boundary
                int pos = second_last - timestamp_begin;
                if (pos > last_timestamp_pos) {
                    last_timestamp_pos = pos;
                }
            }
        }
    }

    if (last_timestamp_pos >= 0) {
        // Advance to the timestamp position (in mel frames)
        // Each timestamp unit = INPUT_STRIDE mel frames
        int advance = last_timestamp_pos * INPUT_STRIDE;

        // Minimum advance to prevent infinite loops
        // But use 1 second (100 frames) instead of 3 seconds
        return std::max(advance, 100);
    }

    // No consecutive timestamps found - likely no speech or hallucination
    // Advance by full segment size
    return segment_size;
}

/**
 * GAP 20: Find next segment that has words.
 * Python: next_words_segment(segments) returns first segment with non-empty words.
 * Used for hallucination_silence_threshold processing.
 */
int next_words_segment_index(const std::vector<WhisperSegment>& segments, int start_idx = 0) {
    for (int i = start_idx; i < static_cast<int>(segments.size()); ++i) {
        if (!segments[i].words.empty()) {
            return i;
        }
    }
    return -1;  // No segment with words found
}

/**
 * GAP 20: Get end time of last word in current segments.
 * Python: _get_end(segments) returns last word's end time or None.
 */
float get_last_word_end(const std::vector<WhisperSegment>& segments) {
    for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
        if (!it->words.empty()) {
            return it->words.back().end_time;
        }
    }
    return -1.0f;  // No words found
}

}  // anonymous namespace

SegmentedTranscriptionResult WhisperModel::generate_segments(
    const mx::array& mel,
    const std::string& language,
    const std::string& task,
    bool condition_on_previous,
    float no_speech_threshold,
    float compression_ratio_threshold,
    int beam_size,
    float length_penalty,
    bool word_timestamps,
    const WhisperTokenizer* tokenizer,
    float actual_audio_duration,
    const std::string& initial_prompt,
    bool carry_initial_prompt,
    const std::vector<float>& temperatures,
    float logprob_threshold,
    float hallucination_silence_threshold,
    const std::vector<float>& clip_timestamps,
    float entropy_thold,
    bool split_on_word,
    bool skip_logprobs,
    const std::string& prefix,
    const std::vector<int>& suppress_regex_tokens,  // GAP 43
    bool token_timestamps,  // GAP 54
    float thold_pt,  // GAP 55
    float thold_ptsum,  // GAP 55
    bool print_realtime,  // GAP 53
    bool tdrz_enable  // GAP 41: tinydiarize speaker turn detection
) {
    SegmentedTranscriptionResult result;

    // GAP 8: Parse clip_timestamps to get time ranges to process
    // Format: [start1, end1, start2, end2, ...] where each pair defines a clip
    // Empty means process entire audio
    std::vector<std::pair<float, float>> clips;
    if (!clip_timestamps.empty()) {
        for (size_t i = 0; i + 1 < clip_timestamps.size(); i += 2) {
            float start = clip_timestamps[i];
            float end = clip_timestamps[i + 1];
            if (end > start) {
                clips.push_back({start, end});
            }
        }
        // If odd number of timestamps, last one is start with end = audio duration
        if (clip_timestamps.size() % 2 == 1) {
            float start = clip_timestamps.back();
            float end = (actual_audio_duration > 0) ? actual_audio_duration :
                        (mel.shape().size() >= 2 ? static_cast<float>(mel.shape()[1]) / 100.0f : 30.0f);
            clips.push_back({start, end});
        }
    }

    // GAP 6: Tokenize initial_prompt if provided
    // Python: initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
    std::vector<int> initial_prompt_tokens;
    if (!initial_prompt.empty() && tokenizer) {
        // Prepend space and strip whitespace like Python does
        std::string trimmed_prompt = initial_prompt;
        // Strip leading/trailing whitespace
        size_t start = trimmed_prompt.find_first_not_of(" \t\n\r");
        size_t end = trimmed_prompt.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            trimmed_prompt = trimmed_prompt.substr(start, end - start + 1);
        }
        // Prepend space (Python: " " + prompt.strip())
        initial_prompt_tokens = tokenizer->encode(" " + trimmed_prompt);
    }

    // GAP: Tokenize decoder prefix if provided
    // Unlike initial_prompt (prepended as context), prefix is added after SOT sequence
    // and treated as if already spoken (forces transcription to start with this text)
    std::vector<int> prefix_tokens;
    if (!prefix.empty() && tokenizer) {
        // Prepend space and strip whitespace like Python does
        std::string trimmed_prefix = prefix;
        size_t start = trimmed_prefix.find_first_not_of(" \t\n\r");
        size_t end = trimmed_prefix.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            trimmed_prefix = trimmed_prefix.substr(start, end - start + 1);
        }
        // Prepend space (Python: " " + prefix.strip())
        prefix_tokens = tokenizer->encode(" " + trimmed_prefix);
        if (std::getenv("DEBUG_WHISPER")) {
            std::cerr << "[DEBUG] Decoder prefix: '" << trimmed_prefix << "' -> " << prefix_tokens.size() << " tokens\n";
        }
    }
    const std::vector<int>* prefix_ptr = prefix_tokens.empty() ? nullptr : &prefix_tokens;

    // Auto-detect language if not specified or set to "auto"
    std::string effective_language = language;
    if (effective_language.empty() || effective_language == "auto") {
        auto lang_result = detect_language(mel);
        effective_language = lang_result.language;
        if (std::getenv("DEBUG_WHISPER")) {
            std::cerr << "[DEBUG] Auto-detected language: " << effective_language
                      << " (p=" << lang_result.probability << ")\n";
        }
    }
    result.language = effective_language;

    // Get mel dimensions
    // mel is [batch, n_frames, n_mels] or [n_frames, n_mels]
    auto shape = mel.shape();
    int total_frames;
    if (shape.size() == 3) {
        total_frames = static_cast<int>(shape[1]);
    } else if (shape.size() == 2) {
        total_frames = static_cast<int>(shape[0]);
    } else {
        throw std::runtime_error("Invalid mel shape: expected 2D or 3D");
    }

    // Track actual audio duration for word timestamps
    // Timestamp tokens from the decoder give internal/model timing which may not match actual audio duration
    // For word-level timestamps to span the full audio, we need the actual duration
    float actual_audio_duration_sec = static_cast<float>(total_frames) / 100.0f;

    // If mel is exactly N_FRAMES, it's been padded to 30s
    // In this case, we can't determine actual duration from mel shape
    // Options: estimate from mel energy, use segment timing, or trust caller to provide duration
    // For now, we'll estimate actual duration by looking at mel spectrogram energy
    if (total_frames == N_FRAMES) {
        // Count frames with significant energy (non-zero audio)
        // This is a heuristic - real audio typically has energy while padding is silent
        // For mel shape [batch, frames, n_mels] or [frames, n_mels]
        mx::array mel_energy = (shape.size() == 3)
            ? mx::squeeze(mx::sum(mx::abs(mel), std::vector<int>{2}), 0)  // [batch, frames] -> [frames]
            : mx::sum(mx::abs(mel), std::vector<int>{1});  // [frames]
        mx::eval(mel_energy);

        // Find last frame with energy above threshold
        float* energy_ptr = mel_energy.data<float>();
        int last_active_frame = 0;
        const float energy_threshold = 0.01f;  // Threshold for detecting active audio
        for (int i = 0; i < total_frames; ++i) {
            if (energy_ptr[i] > energy_threshold) {
                last_active_frame = i;
            }
        }

        // Add small buffer and convert to duration
        actual_audio_duration_sec = static_cast<float>(last_active_frame + 10) / 100.0f;
        // Clamp to reasonable bounds
        actual_audio_duration_sec = std::min(actual_audio_duration_sec, 30.0f);
        actual_audio_duration_sec = std::max(actual_audio_duration_sec, 0.5f);
    }

    // Use provided duration if available, otherwise use mel-based estimate
    if (actual_audio_duration > 0.0f) {
        actual_audio_duration_sec = actual_audio_duration;
    }

    result.total_duration = actual_audio_duration_sec;

    // Determine if we should use single-pass or multi-segment decoding
    // Multi-segment is needed when:
    // 1. Audio is longer than 30s (mel frames > N_FRAMES), OR
    // 2. Audio duration > 10s (may have multiple natural segments that cause EOT)
    // For short audio <= 10s, single-pass is usually sufficient
    constexpr float SINGLE_PASS_MAX_DURATION = 10.0f;  // 10 seconds threshold
    bool use_single_pass = (total_frames <= N_FRAMES) && (actual_audio_duration_sec <= SINGLE_PASS_MAX_DURATION);

    // If audio is short enough, process in one go
    if (use_single_pass) {
        // GAP 4/12: Temperature fallback loop
        // Python: for temp in temperatures: decode, check quality, retry if needed
        std::vector<int> tokens;
        float avg_logprob = 0.0f;
        float no_speech_prob = 0.0f;
        float used_temperature = temperatures.empty() ? 0.0f : temperatures[0];
        float comp_ratio = 0.0f;

        // GAP 6: Pass initial_prompt_tokens if provided
        const std::vector<int>* prompt_ptr = initial_prompt_tokens.empty() ? nullptr : &initial_prompt_tokens;

        // Ensure we have at least one temperature to try
        std::vector<float> temps_to_try = temperatures.empty() ? std::vector<float>{0.0f} : temperatures;

        for (size_t temp_idx = 0; temp_idx < temps_to_try.size(); ++temp_idx) {
            float temp = temps_to_try[temp_idx];
            used_temperature = temp;

            if (beam_size > 1) {
                // Beam search (no temperature support currently)
                auto beam_result = generate_beam(mel, effective_language, task, beam_size, length_penalty, 448, prompt_ptr);
                tokens = beam_result.tokens;
                avg_logprob = beam_result.avg_logprob;
                no_speech_prob = beam_result.no_speech_prob;
            } else {
                // Greedy/sampling with temperature
                // GAP M: Pass skip_logprobs for speed optimization
                // GAP: Pass prefix_ptr for decoder prefix
                tokens = generate(mel, effective_language, task, 448, actual_audio_duration_sec, &avg_logprob, &no_speech_prob, prompt_ptr, temp, 1.0f, 1.0f, false, prefix_ptr, -1, skip_logprobs, suppress_regex_tokens.empty() ? nullptr : &suppress_regex_tokens, tdrz_enable);
            }

            // Calculate compression ratio from decoded text
            std::string decoded_text;
            if (tokenizer && !tokens.empty()) {
                decoded_text = tokenizer->decode(tokens);
                comp_ratio = calculate_compression_ratio(decoded_text);
            } else {
                comp_ratio = 1.0f;
            }

            // GAP 4/12 + GAP J: Check quality for fallback decision
            bool needs_fallback = false;

            // Check compression ratio (detects repetition/hallucination loops)
            if (comp_ratio > compression_ratio_threshold) {
                needs_fallback = true;
                if (std::getenv("DEBUG_WHISPER")) {
                    std::cerr << "[DEBUG] Temperature " << temp << ": compression_ratio " << comp_ratio
                              << " > " << compression_ratio_threshold << " (retry)\n";
                }
            }

            // GAP J: Check logprob threshold (detects low confidence)
            if (logprob_threshold > -std::numeric_limits<float>::infinity() && avg_logprob < logprob_threshold) {
                // Python: If avg_logprob < threshold AND no_speech > threshold, treat as silence
                if (no_speech_prob > no_speech_threshold) {
                    // This is likely silence, not failed transcription
                    tokens.clear();
                    if (std::getenv("DEBUG_WHISPER")) {
                        std::cerr << "[DEBUG] Temperature " << temp << ": detected silence (no_speech=" << no_speech_prob << ")\n";
                    }
                    break;  // Accept empty result
                } else {
                    needs_fallback = true;
                    if (std::getenv("DEBUG_WHISPER")) {
                        std::cerr << "[DEBUG] Temperature " << temp << ": avg_logprob " << avg_logprob
                                  << " < " << logprob_threshold << " (retry)\n";
                    }
                }
            }

            // If quality is acceptable or this is the last temperature, accept result
            if (!needs_fallback || temp_idx == temps_to_try.size() - 1) {
                if (std::getenv("DEBUG_WHISPER") && needs_fallback) {
                    std::cerr << "[DEBUG] Temperature " << temp << ": accepting despite quality issues (last fallback)\n";
                }
                break;  // Accept this result
            }
            // Otherwise, loop continues to next temperature
        }

        // Parse segments from tokens
        // GAP 46: Pass split_on_word and tokenizer for word boundary alignment
        auto parsed = parse_segments_from_tokens(tokens, config_.timestamp_begin, config_.eot_token, split_on_word, tokenizer);

        // Encode mel for word timestamps if requested (lazy evaluation)
        std::unique_ptr<mx::array> encoder_output_ptr;

        for (size_t seg_idx = 0; seg_idx < parsed.size(); ++seg_idx) {
            const auto& [start, end, seg_tokens] = parsed[seg_idx];
            WhisperSegment seg;
            seg.start_time = start;
            // For the last segment, use actual audio duration if it's longer than parsed end time
            bool is_last_segment = (seg_idx == parsed.size() - 1);
            seg.end_time = (is_last_segment && actual_audio_duration_sec > end)
                ? actual_audio_duration_sec : end;
            seg.tokens = seg_tokens;
            seg.avg_logprob = avg_logprob;
            seg.no_speech_prob = no_speech_prob;
            // Note: WhisperSegment doesn't track temperature - quality is tracked via avg_logprob

            // Calculate compression ratio from decoded text (GAP 16)
            if (tokenizer && !seg_tokens.empty()) {
                std::string seg_text = tokenizer->decode(seg_tokens);
                seg.compression_ratio = calculate_compression_ratio(seg_text);
            } else {
                seg.compression_ratio = comp_ratio;  // Use overall ratio if no per-segment calc
            }

            // Add word-level timestamps if requested
            if (word_timestamps && !seg_tokens.empty()) {
                if (!encoder_output_ptr) {
                    encoder_output_ptr = std::make_unique<mx::array>(encode(mel));
                    mx::eval(*encoder_output_ptr);
                }
                add_word_timestamps(seg, *encoder_output_ptr, effective_language, tokenizer, actual_audio_duration_sec, nullptr);
            }

            // GAP 41: Check for speaker turn token (tinydiarize)
            // If solm token is present in the segment, mark speaker_turn_next = true
            if (tdrz_enable) {
                for (int tok : tokens) {
                    if (tok == config_.solm_token) {
                        seg.speaker_turn_next = true;
                        break;
                    }
                }
            }

            result.segments.push_back(seg);

            // GAP 53: Print segment in real-time if requested
            if (print_realtime && tokenizer && tokenizer->loaded()) {
                std::string text = tokenizer->decode(seg.tokens);
                std::cerr << std::fixed << std::setprecision(2)
                          << "[" << seg.start_time << " --> " << seg.end_time << "] "
                          << text << std::endl;
            }
        }

        return result;
    }

    // =========================================================================
    // Segment-based processing for long audio
    // For audio that's been padded to 30s, stop at actual_audio_duration, not total_frames
    // =========================================================================
    int seek = 0;  // Current position in mel frames

    // GAP 68: Track last speech timestamp across segments for pause detection (GAP 78-80)
    float last_speech_ts = 0.0f;

    // Convert actual audio duration to mel frames (100 frames per second)
    int actual_audio_frames = static_cast<int>(actual_audio_duration_sec * 100.0f);
    // Use the smaller of actual audio or total mel frames
    int end_frame = std::min(actual_audio_frames, total_frames);

    // Previous tokens for conditioning (if enabled)
    std::vector<int> previous_tokens;

    while (seek < end_frame) {
        // GAP 8: clip_timestamps filtering - skip segments outside specified time ranges
        // Convert seek position to seconds (100 frames per second)
        float seek_time = static_cast<float>(seek) / 100.0f;
        if (!clips.empty()) {
            bool in_clip = false;
            for (const auto& [clip_start, clip_end] : clips) {
                if (seek_time >= clip_start && seek_time < clip_end) {
                    in_clip = true;
                    break;
                }
            }
            if (!in_clip) {
                // Skip this segment - advance seek to next clip or end
                seek += N_FRAMES;
                continue;
            }
        }

        // Calculate segment size (up to N_FRAMES)
        int segment_size = std::min(N_FRAMES, total_frames - seek);

        // Extract mel segment - always work with 2D then add batch dimension at the end
        // Initialize with slice to avoid default construction
        mx::array mel_segment = (shape.size() == 3) ?
            mx::squeeze(mx::slice(mel, {0, seek, 0}, {1, seek + segment_size, static_cast<int>(shape[2])}), 0) :
            mx::slice(mel, {seek, 0}, {seek + segment_size, static_cast<int>(shape[1])});

        // Pad to N_FRAMES if needed (works with 2D)
        if (segment_size < N_FRAMES) {
            mel_segment = audio::pad_or_trim(mel_segment, N_FRAMES);
        }

        // Add batch dimension for generate/generate_beam
        mel_segment = mx::expand_dims(mel_segment, 0);

        mx::eval(mel_segment);

        // Generate tokens for this segment
        std::vector<int> tokens;
        float avg_logprob = 0.0f;
        float no_speech_prob = 0.0f;  // GAP 13 FIX

        // Calculate actual duration for this chunk to prevent hallucination
        // segment_size is actual mel frames BEFORE padding to N_FRAMES
        float chunk_duration = static_cast<float>(segment_size) / 100.0f;

        // GAP 6: Build prompt tokens for this segment
        // Python behavior:
        //   - If carry_initial_prompt: initial_prompt_tokens + previous_tokens (truncated)
        //   - Else if condition_on_previous: previous_tokens (truncated)
        //   - Else: no prompt
        std::vector<int> segment_prompt_tokens;
        if (condition_on_previous && !previous_tokens.empty()) {
            if (carry_initial_prompt && !initial_prompt_tokens.empty()) {
                // Combine initial_prompt with previous transcript
                segment_prompt_tokens = initial_prompt_tokens;
                // Calculate remaining room after initial_prompt
                size_t max_total = (config_.n_text_ctx / 2) - 1;
                if (segment_prompt_tokens.size() < max_total) {
                    size_t remaining = max_total - segment_prompt_tokens.size();
                    // Take last N previous tokens
                    size_t prev_start = (previous_tokens.size() > remaining)
                        ? previous_tokens.size() - remaining : 0;
                    segment_prompt_tokens.insert(segment_prompt_tokens.end(),
                                                  previous_tokens.begin() + prev_start,
                                                  previous_tokens.end());
                }
            } else {
                // Only use previous transcript tokens
                segment_prompt_tokens = previous_tokens;
            }
        } else if (!initial_prompt_tokens.empty() && seek == 0) {
            // First segment with initial_prompt but no condition_on_previous
            segment_prompt_tokens = initial_prompt_tokens;
        }

        const std::vector<int>* prompt_ptr = segment_prompt_tokens.empty() ? nullptr : &segment_prompt_tokens;

        // GAP 4/12: Temperature fallback loop for each segment
        float used_temperature = temperatures.empty() ? 0.0f : temperatures[0];
        float comp_ratio = 0.0f;
        std::vector<float> temps_to_try = temperatures.empty() ? std::vector<float>{0.0f} : temperatures;

        for (size_t temp_idx = 0; temp_idx < temps_to_try.size(); ++temp_idx) {
            float temp = temps_to_try[temp_idx];
            used_temperature = temp;

            if (beam_size > 1) {
                auto beam_result = generate_beam(mel_segment, effective_language, task, beam_size, length_penalty, 448, prompt_ptr);
                tokens = beam_result.tokens;
                avg_logprob = beam_result.avg_logprob;
                no_speech_prob = beam_result.no_speech_prob;
            } else {
                // GAP M: Pass skip_logprobs for speed optimization
                // GAP: Pass prefix_ptr for decoder prefix
                // GAP 43: Pass suppress_regex_tokens
                tokens = generate(mel_segment, effective_language, task, 448, chunk_duration, &avg_logprob, &no_speech_prob, prompt_ptr, temp, 1.0f, 1.0f, false, prefix_ptr, -1, skip_logprobs, suppress_regex_tokens.empty() ? nullptr : &suppress_regex_tokens, tdrz_enable);
            }

            // Calculate compression ratio from decoded text
            std::string decoded_text;
            if (tokenizer && !tokens.empty()) {
                decoded_text = tokenizer->decode(tokens);
                comp_ratio = calculate_compression_ratio(decoded_text);
            } else {
                comp_ratio = 1.0f;
            }

            // GAP 4/12 + GAP J: Check quality for fallback decision
            bool needs_fallback = false;

            // Check compression ratio (detects repetition/hallucination)
            if (comp_ratio > compression_ratio_threshold) {
                needs_fallback = true;
                if (std::getenv("DEBUG_WHISPER")) {
                    std::cerr << "[DEBUG] Segment at seek=" << seek << ", T=" << temp
                              << ": compression_ratio " << comp_ratio << " > " << compression_ratio_threshold << " (retry)\n";
                }
            }

            // GAP J: Check logprob threshold
            if (logprob_threshold > -std::numeric_limits<float>::infinity() && avg_logprob < logprob_threshold) {
                if (no_speech_prob > no_speech_threshold) {
                    // Silence detected
                    tokens.clear();
                    if (std::getenv("DEBUG_WHISPER")) {
                        std::cerr << "[DEBUG] Segment at seek=" << seek << ", T=" << temp
                                  << ": detected silence (no_speech=" << no_speech_prob << ")\n";
                    }
                    break;
                } else {
                    needs_fallback = true;
                    if (std::getenv("DEBUG_WHISPER")) {
                        std::cerr << "[DEBUG] Segment at seek=" << seek << ", T=" << temp
                                  << ": avg_logprob " << avg_logprob << " < " << logprob_threshold << " (retry)\n";
                    }
                }
            }

            // Accept if quality OK or last temperature
            if (!needs_fallback || temp_idx == temps_to_try.size() - 1) {
                if (std::getenv("DEBUG_WHISPER") && needs_fallback) {
                    std::cerr << "[DEBUG] Segment at seek=" << seek << ", T=" << temp
                              << ": accepting despite quality issues (last fallback)\n";
                }
                break;
            }
        }

        // Parse segments from this chunk's tokens
        // GAP 46: Pass split_on_word and tokenizer for word boundary alignment
        auto parsed = parse_segments_from_tokens(tokens, config_.timestamp_begin, config_.eot_token, split_on_word, tokenizer);

        // Encode mel segment for word timestamps if requested (lazy evaluation)
        std::unique_ptr<mx::array> encoder_output_ptr;

        // Calculate time offset for this chunk
        float time_offset = static_cast<float>(seek) / 100.0f;

        // Add segments with correct timestamps
        for (const auto& [start, end, seg_tokens] : parsed) {
            WhisperSegment seg;
            seg.start_time = time_offset + start;
            seg.end_time = time_offset + end;
            seg.tokens = seg_tokens;
            seg.avg_logprob = avg_logprob;
            seg.no_speech_prob = no_speech_prob;
            seg.temperature = used_temperature;  // GAP 4/12: Track which temperature was used

            // Calculate compression ratio from decoded text (GAP 16)
            if (tokenizer && !seg_tokens.empty()) {
                std::string seg_text = tokenizer->decode(seg_tokens);
                seg.compression_ratio = calculate_compression_ratio(seg_text);
            } else {
                seg.compression_ratio = comp_ratio;  // Use overall ratio if no per-segment calc
            }

            // Add word-level timestamps if requested
            if (word_timestamps && !seg_tokens.empty()) {
                // Lazy encode - only do it once per chunk if needed
                if (!encoder_output_ptr) {
                    encoder_output_ptr = std::make_unique<mx::array>(encode(mel_segment));
                    mx::eval(*encoder_output_ptr);
                }
                // Pass actual segment duration for correct DTW frame alignment
                // segment_size is the actual mel frame count for this chunk
                float chunk_duration = static_cast<float>(segment_size) / 100.0f;
                // GAP 68, 78-80: Pass last_speech_ts for cross-segment pause truncation
                add_word_timestamps(seg, *encoder_output_ptr, effective_language, tokenizer, chunk_duration, &last_speech_ts);
            }

            // GAP 41: Check for speaker turn token (tinydiarize)
            // If solm token is present in this segment, mark speaker_turn_next = true
            if (tdrz_enable) {
                for (int tok : seg_tokens) {
                    if (tok == config_.solm_token) {
                        seg.speaker_turn_next = true;
                        break;
                    }
                }
            }

            result.segments.push_back(seg);

            // GAP 53: Print segment in real-time if requested
            if (print_realtime && tokenizer && tokenizer->loaded()) {
                std::string text = tokenizer->decode(seg.tokens);
                std::cerr << std::fixed << std::setprecision(2)
                          << "[" << seg.start_time << " --> " << seg.end_time << "] "
                          << text << std::endl;
            }
        }

        // Calculate base seek advance from timestamps
        int advance = calculate_seek_advance(tokens, config_.timestamp_begin, segment_size, config_.eot_token);

        // GAP 20: hallucination_silence_threshold processing
        // Skip silence before/after possible hallucinations
        if (hallucination_silence_threshold > 0.0f && word_timestamps) {
            float threshold = hallucination_silence_threshold;
            float window_end_time = time_offset + static_cast<float>(segment_size) / 100.0f;
            float segment_duration = static_cast<float>(segment_size) / 100.0f;

            // Get last word end from current batch of segments
            std::vector<WhisperSegment> current_batch;
            size_t batch_start = result.segments.size() - parsed.size();
            for (size_t i = batch_start; i < result.segments.size(); ++i) {
                current_batch.push_back(result.segments[i]);
            }

            // Skip silence after last word if remaining duration > threshold
            float last_word_end = get_last_word_end(current_batch);
            if (last_word_end > time_offset) {
                float remaining_duration = window_end_time - last_word_end;
                if (remaining_duration > threshold) {
                    // Seek to end of last word
                    advance = static_cast<int>((last_word_end - time_offset) * 100.0f);
                    if (std::getenv("DEBUG_WHISPER")) {
                        std::cerr << "[DEBUG] GAP 20: Skipping " << remaining_duration << "s silence after last word\n";
                    }
                }
            }

            // If first segment might be a hallucination, skip leading silence
            int first_words_idx = next_words_segment_index(current_batch);
            if (first_words_idx >= 0 && is_segment_anomaly(current_batch[first_words_idx].words)) {
                float first_seg_start = current_batch[first_words_idx].start_time;
                float gap = first_seg_start - time_offset;
                if (gap > threshold) {
                    // Skip leading silence and restart iteration
                    seek += static_cast<int>(gap * 100.0f);
                    // Remove segments from this batch that will be re-processed
                    result.segments.resize(batch_start);
                    if (std::getenv("DEBUG_WHISPER")) {
                        std::cerr << "[DEBUG] GAP 20: Skipping " << gap << "s leading silence before anomalous segment\n";
                    }
                    continue;  // Re-process from new position
                }
            }

            // Check each segment for being surrounded by silence (hallucination detection)
            float hal_last_end = last_speech_ts;
            for (size_t si = 0; si < current_batch.size(); ++si) {
                const auto& segment = current_batch[si];
                if (segment.words.empty()) continue;

                if (is_segment_anomaly(segment.words)) {
                    // Find next segment with words
                    int next_idx = next_words_segment_index(current_batch, si + 1);
                    float hal_next_start = (next_idx >= 0)
                        ? current_batch[next_idx].words[0].start_time
                        : time_offset + segment_duration;

                    bool silence_before = (segment.start_time - hal_last_end > threshold) ||
                                          (segment.start_time < threshold) ||
                                          (segment.start_time - time_offset < 2.0f);
                    bool silence_after = (hal_next_start - segment.end_time > threshold) ||
                                         (next_idx >= 0 && is_segment_anomaly(current_batch[next_idx].words)) ||
                                         (window_end_time - segment.end_time < 2.0f);

                    if (silence_before && silence_after) {
                        // Skip to segment start
                        int skip_seek = static_cast<int>(std::max(time_offset + 1.0f, segment.start_time) * 100.0f);
                        float content_duration = static_cast<float>(end_frame - seek) / 100.0f;
                        if (content_duration - segment.end_time < threshold) {
                            skip_seek = end_frame;  // Skip to end
                        }
                        // Remove this segment and all following from this batch
                        result.segments.resize(batch_start + si);
                        seek = skip_seek;
                        if (std::getenv("DEBUG_WHISPER")) {
                            std::cerr << "[DEBUG] GAP 20: Skipping hallucinated segment surrounded by silence\n";
                        }
                        goto next_iteration;  // Exit inner loop
                    }
                }
                hal_last_end = segment.end_time;
            }
        }

        seek += advance;
next_iteration:

        // Store previous tokens for conditioning (if enabled)
        // GAP 60: Python resets context when temperature > 0.5 (indicates poor decode quality)
        // Now that temperature fallback is implemented, use actual used_temperature
        if (condition_on_previous) {
            // Reset context if temperature > 0.5 (Python mlx-whisper behavior)
            // High temperature indicates difficult audio that may hallucinate with context
            if (used_temperature > 0.5f) {
                previous_tokens.clear();
                if (std::getenv("DEBUG_WHISPER")) {
                    std::cerr << "[DEBUG] GAP 60: Resetting context (temperature " << used_temperature << " > 0.5)\n";
                }
            } else {
                // Keep tokens for context conditioning
                previous_tokens = tokens;
            }
        }
    }

    return result;
}

// ============================================================================
// Audio Processing
// ============================================================================

namespace audio {

// ============================================================================
// Mel Filterbank Generation
// ============================================================================

namespace {

// Hz to Mel conversion (standard formula)
float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

// Mel to Hz conversion
float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// Load mel filters from precomputed binary file (exported from mlx-whisper)
// These match librosa's mel filterbank exactly
// Returns [n_mels, n_freqs] matrix where n_freqs = n_fft/2 + 1 = 201
std::vector<float> load_mel_filters_from_file(int n_mels) {
    // Determine path relative to library location
    // For now, try a few common locations
    std::vector<std::string> paths = {
        "mel_filters_" + std::to_string(n_mels) + "_201.bin",
        "src/mlx_inference_engine/mel_filters_" + std::to_string(n_mels) + "_201.bin",
        "/Users/ayates/model_mlx_migration/src/mlx_inference_engine/mel_filters_" + std::to_string(n_mels) + "_201.bin"
    };

    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::binary);
        if (file.good()) {
            const int n_freqs = 201;  // n_fft/2 + 1 for n_fft=400
            std::vector<float> filters(n_mels * n_freqs);
            file.read(reinterpret_cast<char*>(filters.data()), filters.size() * sizeof(float));
            if (file.good()) {
                return filters;
            }
        }
    }

    // Fallback: return empty (will trigger recomputation)
    return std::vector<float>();
}

// Generate mel filterbank matrix
// Returns [n_mels, n_freqs] matrix where n_freqs = n_fft/2 + 1 = 201
std::vector<float> get_mel_filters(int sample_rate, int n_fft, int n_mels) {
    // Try to load pre-computed filters first (match librosa/mlx-whisper exactly)
    auto precomputed = load_mel_filters_from_file(n_mels);
    if (!precomputed.empty()) {
        return precomputed;
    }

    // Fallback: compute mel filters (may differ slightly from librosa)
    int n_freqs = n_fft / 2 + 1;  // Keep all frequency bins including Nyquist

    // Frequency bins
    std::vector<float> freqs(n_freqs);
    for (int i = 0; i < n_freqs; ++i) {
        freqs[i] = static_cast<float>(i) * sample_rate / n_fft;
    }

    // Mel scale bounds
    float min_mel = hz_to_mel(0.0f);
    float max_mel = hz_to_mel(static_cast<float>(sample_rate) / 2.0f);

    // Mel band edges (n_mels + 2 points for triangular filters)
    std::vector<float> mel_points(n_mels + 2);
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = min_mel + (max_mel - min_mel) * i / (n_mels + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Create filterbank [n_mels, n_freqs]
    std::vector<float> filters(n_mels * n_freqs, 0.0f);

    for (int i = 0; i < n_mels; ++i) {
        float left = hz_points[i];
        float center = hz_points[i + 1];
        float right = hz_points[i + 2];

        for (int j = 0; j < n_freqs; ++j) {
            float freq = freqs[j];

            // Triangular filter
            float rising = (freq - left) / (center - left);
            float falling = (right - freq) / (right - center);
            float weight = std::max(0.0f, std::min(rising, falling));

            filters[i * n_freqs + j] = weight;
        }

        // Normalize (same as Whisper - slaney normalization)
        float enorm = 2.0f / (hz_points[i + 2] - hz_points[i]);
        for (int j = 0; j < n_freqs; ++j) {
            filters[i * n_freqs + j] *= enorm;
        }
    }

    return filters;
}

// Generate Hanning window
std::vector<float> get_hanning_window(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        // Match numpy: np.hanning(size+1)[:-1]
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / size));
    }
    return window;
}

}  // anonymous namespace

std::vector<float> load_audio(const std::string& path, int sample_rate, bool from_stdin) {
    // GAP 82: from_stdin mode - read audio directly from stdin via ffmpeg
    // Python: mlx_whisper.audio.load_audio(..., from_stdin=True)
    std::ostringstream cmd;
    if (from_stdin) {
        // Read from stdin - uses "pipe:0" as input
        cmd << "ffmpeg -i pipe:0 "
            << "-threads 0 -f s16le -ac 1 -acodec pcm_s16le "
            << "-ar " << sample_rate << " -";
    } else {
        // Read from file - use -nostdin to prevent ffmpeg from consuming stdin
        cmd << "ffmpeg -nostdin -i \"" << path << "\" "
            << "-threads 0 -f s16le -ac 1 -acodec pcm_s16le "
            << "-ar " << sample_rate << " -";
    }

    std::array<char, 128> buffer;
    std::vector<char> raw_data;

    // Execute ffmpeg and capture output
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.str().c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("Failed to execute ffmpeg");
    }

    while (size_t n = fread(buffer.data(), 1, buffer.size(), pipe.get())) {
        raw_data.insert(raw_data.end(), buffer.begin(), buffer.begin() + n);
    }

    // Convert s16le to float32
    size_t n_samples = raw_data.size() / 2;
    std::vector<float> audio(n_samples);

    const int16_t* raw_samples = reinterpret_cast<const int16_t*>(raw_data.data());
    for (size_t i = 0; i < n_samples; ++i) {
        audio[i] = static_cast<float>(raw_samples[i]) / 32768.0f;
    }

    return audio;
}

mx::array log_mel_spectrogram(
    const std::vector<float>& audio,
    int n_mels,
    int n_fft,
    int hop_length
) {
    // Constants matching Whisper
    const int sample_rate = 16000;
    const int padding = n_fft / 2;  // 200 for n_fft=400

    // Pad audio with reflect padding (like np.pad(..., mode='reflect'))
    // Simplified: use zero padding at edges, reflect in middle
    int audio_len = static_cast<int>(audio.size());
    int padded_len = audio_len + 2 * padding;
    std::vector<float> padded(padded_len, 0.0f);

    // Left reflect padding: [padding, padding-1, ..., 1]
    for (int i = 0; i < padding && i < audio_len; ++i) {
        int src_idx = padding - i;
        if (src_idx < audio_len) {
            padded[i] = audio[src_idx];
        }
    }

    // Copy main audio
    for (int i = 0; i < audio_len; ++i) {
        padded[padding + i] = audio[i];
    }

    // Right reflect padding: [L-2, L-3, ..., L-padding-1]
    for (int i = 0; i < padding && audio_len > 1; ++i) {
        int src_idx = audio_len - 2 - i;
        if (src_idx >= 0) {
            padded[padding + audio_len + i] = audio[src_idx];
        }
    }

    // Calculate number of frames (same as Python: (padded_len - n_fft + hop_length) // hop_length)
    int n_frames = (padded_len - n_fft + hop_length) / hop_length;
    if (n_frames <= 0) {
        throw std::runtime_error("Audio too short for mel spectrogram computation");
    }

    // Match Python: freqs[:-1, :] removes the LAST FRAME
    // So we need n_frames - 1 output frames
    int n_output_frames = n_frames - 1;

    // Get mel filterbank [n_mels, n_freqs] where n_freqs = n_fft/2 + 1 = 201
    auto mel_filters = get_mel_filters(sample_rate, n_fft, n_mels);
    int n_freqs = n_fft / 2 + 1;  // Keep all frequency bins (201 for n_fft=400)

    // Get Hanning window
    auto window = get_hanning_window(n_fft);

    // Frame the audio and apply window
    // frames[i] = padded[i*hop_length : i*hop_length + n_fft] * window
    std::vector<float> framed(n_frames * n_fft);
    for (int i = 0; i < n_frames; ++i) {
        int start = i * hop_length;
        for (int j = 0; j < n_fft; ++j) {
            framed[i * n_fft + j] = padded[start + j] * window[j];
        }
    }

    // Convert to MLX array for GPU-accelerated FFT
    auto framed_mx = mx::array(framed.data(), {n_frames, n_fft});

    // Compute RFFT using MLX
    auto stft = mx::fft::rfft(framed_mx, n_fft, -1);  // [n_frames, n_fft/2+1] complex
    mx::eval(stft);

    // Match Python: magnitudes = freqs[:-1, :].abs().square()
    // Take all n_freqs frequency bins (201), remove last frame
    auto stft_sliced = mx::slice(stft, {0, 0}, {n_output_frames, n_freqs});  // [n_frames-1, 201]
    auto magnitudes = mx::abs(stft_sliced);
    magnitudes = magnitudes * magnitudes;  // magnitude squared
    mx::eval(magnitudes);

    // Apply mel filterbank: [n_frames-1, n_freqs] @ [n_freqs, n_mels] -> [n_frames-1, n_mels]
    // mel_filters is [n_mels, n_freqs], need to transpose
    auto mel_filters_mx = mx::array(mel_filters.data(), {n_mels, n_freqs});
    auto mel_filters_T = mx::transpose(mel_filters_mx);  // [n_freqs, n_mels]

    auto mel_spec = mx::matmul(magnitudes, mel_filters_T);  // [n_frames-1, n_mels]
    mx::eval(mel_spec);

    // Log scale with clipping (same as mlx-whisper)
    // log_spec = log10(max(mel_spec, 1e-10))
    // log_spec = max(log_spec, max_val - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0
    auto log_spec = mx::log10(mx::maximum(mel_spec, mx::array(1e-10f)));
    mx::eval(log_spec);

    auto max_val = mx::max(log_spec);
    mx::eval(max_val);

    log_spec = mx::maximum(log_spec, max_val - mx::array(8.0f));
    log_spec = (log_spec + mx::array(4.0f)) / mx::array(4.0f);
    mx::eval(log_spec);

    return mx::astype(log_spec, mx::float32);
}

mx::array pad_or_trim(const mx::array& mel, int target_length) {
    int current_length = static_cast<int>(mel.shape()[0]);
    int n_mels = static_cast<int>(mel.shape()[1]);

    if (current_length == target_length) {
        return mel;
    } else if (current_length > target_length) {
        // Trim
        return mx::slice(mel, {0, 0}, {target_length, n_mels});
    } else {
        // Pad with zeros
        int pad_length = target_length - current_length;
        auto padding = mx::zeros({pad_length, n_mels}, mel.dtype());
        return mx::concatenate({mel, padding}, 0);
    }
}

}  // namespace audio

// ============================================================================
// WhisperTokenizer - Loads vocab.json for token decoding
// ============================================================================

struct WhisperTokenizer::Impl {
    // Token ID to text mapping (loaded from vocab.json)
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    // Special token IDs
    int eot_token = 50257;
    int sot_token = 50258;
    int timestamp_begin = 50364;
};

WhisperTokenizer::WhisperTokenizer() : impl_(std::make_unique<Impl>()) {}
WhisperTokenizer::~WhisperTokenizer() = default;
WhisperTokenizer::WhisperTokenizer(WhisperTokenizer&&) noexcept = default;
WhisperTokenizer& WhisperTokenizer::operator=(WhisperTokenizer&&) noexcept = default;

WhisperTokenizer WhisperTokenizer::load(const std::string& vocab_path) {
    WhisperTokenizer tokenizer;

    // Read vocab.json file
    std::string json = read_file(vocab_path);

    // Simple JSON parsing for {"id": "text", ...} format
    // Find all key-value pairs
    std::regex kv_pattern("\"(\\d+)\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
    std::sregex_iterator it(json.begin(), json.end(), kv_pattern);
    std::sregex_iterator end;

    while (it != end) {
        std::smatch match = *it;
        int id = std::stoi(match[1]);
        std::string token = match[2];

        // Unescape common JSON escapes
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
                    case 'u': {
                        // Unicode escape: \uXXXX
                        if (i + 5 < token.size()) {
                            std::string hex = token.substr(i + 2, 4);
                            try {
                                int codepoint = std::stoi(hex, nullptr, 16);
                                // Convert to UTF-8
                                if (codepoint < 0x80) {
                                    unescaped += static_cast<char>(codepoint);
                                } else if (codepoint < 0x800) {
                                    unescaped += static_cast<char>(0xC0 | (codepoint >> 6));
                                    unescaped += static_cast<char>(0x80 | (codepoint & 0x3F));
                                } else {
                                    unescaped += static_cast<char>(0xE0 | (codepoint >> 12));
                                    unescaped += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                                    unescaped += static_cast<char>(0x80 | (codepoint & 0x3F));
                                }
                                i += 5;
                            } catch (...) {
                                unescaped += token[i];
                            }
                        } else {
                            unescaped += token[i];
                        }
                        break;
                    }
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

    tokenizer.loaded_ = tokenizer.impl_->id_to_token.size() > 0;

    if (tokenizer.loaded_) {
        std::cout << "WhisperTokenizer: Loaded " << tokenizer.impl_->id_to_token.size() << " tokens\n";
    } else {
        std::cerr << "WhisperTokenizer: Failed to load vocab from " << vocab_path << "\n";
    }

    return tokenizer;
}

std::vector<int> WhisperTokenizer::encode(const std::string& text) const {
    // Encoding not implemented (we only need decode for transcription)
    // Full BPE encoding requires merge rules which are complex
    std::cerr << "WhisperTokenizer::encode not implemented\n";
    return {};
}

std::string WhisperTokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        // Skip special tokens (EOT=50257, SOT=50258, language=50259-50357,
        // task=50358-50359, no_timestamps=50363, timestamps=50364+)
        if (id >= 50257) {
            continue;  // Special token range starts at EOT (50257)
        }

        auto it = impl_->id_to_token.find(id);
        if (it != impl_->id_to_token.end()) {
            result += it->second;
        }
    }
    return result;
}

int WhisperTokenizer::vocab_size() const {
    return static_cast<int>(impl_->id_to_token.size());
}

const std::unordered_map<int, std::string>& WhisperTokenizer::get_id_to_token() const {
    return impl_->id_to_token;
}

// ============================================================================
// Transcriber
// ============================================================================

Transcriber::Transcriber() = default;
Transcriber::~Transcriber() = default;
Transcriber::Transcriber(Transcriber&&) noexcept = default;
Transcriber& Transcriber::operator=(Transcriber&&) noexcept = default;

Transcriber Transcriber::load(const std::string& model_name) {
    Transcriber transcriber;
    // TODO: Download and load model by name
    throw std::runtime_error("Transcriber::load not yet implemented. Use WhisperModel::load with path.");
}

std::string Transcriber::transcribe(
    const std::string& audio_path,
    const std::string& language
) const {
    // Load audio
    auto audio_samples = audio::load_audio(audio_path);
    return transcribe(audio_samples, language);
}

std::string Transcriber::transcribe(
    const std::vector<float>& audio,
    const std::string& language
) const {
    // TODO: Implement full transcription pipeline
    // 1. Compute mel spectrogram
    // 2. Generate tokens
    // 3. Decode tokens to text
    throw std::runtime_error("Transcriber not yet implemented. Use WhisperModel directly.");
}

// ============================================================================
// Streaming Transcription Implementation
// ============================================================================

// ============================================================================
// AudioBuffer
// ============================================================================

AudioBuffer::AudioBuffer(float max_duration, int sample_rate)
    : sample_rate_(sample_rate)
    , max_samples_(static_cast<size_t>(max_duration * sample_rate))
{
    buffer_.resize(max_samples_, 0.0f);
}

void AudioBuffer::append(const float* samples, size_t count) {
    total_samples_ += count;

    if (count >= max_samples_) {
        // Audio larger than buffer - keep only the tail
        std::copy(samples + count - max_samples_, samples + count, buffer_.begin());
        write_pos_ = max_samples_;
    } else if (write_pos_ + count <= max_samples_) {
        // No wrap needed
        std::copy(samples, samples + count, buffer_.begin() + write_pos_);
        write_pos_ += count;
    } else {
        // Wrap around - shift existing data and append
        size_t available = max_samples_ - write_pos_;
        std::copy(samples, samples + available, buffer_.begin() + write_pos_);

        // Remaining goes at the start (overwriting oldest)
        size_t remaining = count - available;
        if (remaining > 0) {
            // Shift data to make room
            size_t shift = std::min(remaining, write_pos_);
            if (write_pos_ > shift) {
                std::memmove(buffer_.data(), buffer_.data() + shift, (write_pos_ - shift) * sizeof(float));
            }
            // Copy remaining new data
            std::copy(samples + available, samples + count, buffer_.data() + write_pos_ - shift);
            write_pos_ = max_samples_;
        } else {
            write_pos_ = max_samples_;
        }
    }
}

void AudioBuffer::append(const std::vector<float>& samples) {
    append(samples.data(), samples.size());
}

std::vector<float> AudioBuffer::get_audio(float duration) const {
    size_t n_samples = std::min(static_cast<size_t>(duration * sample_rate_), write_pos_);
    if (n_samples == 0) {
        return std::vector<float>();
    }

    std::vector<float> result(n_samples);
    std::copy(buffer_.begin() + write_pos_ - n_samples, buffer_.begin() + write_pos_, result.begin());
    return result;
}

std::vector<float> AudioBuffer::get_all() const {
    return std::vector<float>(buffer_.begin(), buffer_.begin() + write_pos_);
}

float AudioBuffer::duration() const {
    return static_cast<float>(write_pos_) / sample_rate_;
}

float AudioBuffer::total_time() const {
    return static_cast<float>(total_samples_) / sample_rate_;
}

void AudioBuffer::clear() {
    write_pos_ = 0;
}

// ============================================================================
// LocalAgreement
// ============================================================================

LocalAgreement::LocalAgreement(int n) : n_(n) {
    if (n < 2) {
        throw std::invalid_argument("LocalAgreement n must be >= 2");
    }
}

std::string LocalAgreement::update(const std::string& new_transcript) {
    // Trim whitespace
    std::string trimmed = new_transcript;
    size_t start = trimmed.find_first_not_of(" \t\r\n");
    size_t end = trimmed.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) {
        trimmed = "";
    } else {
        trimmed = trimmed.substr(start, end - start + 1);
    }

    history_.push_back(trimmed);

    // Keep only last n transcripts
    if (static_cast<int>(history_.size()) > n_) {
        history_.erase(history_.begin(), history_.end() - n_);
    }

    if (static_cast<int>(history_.size()) < n_) {
        return "";
    }

    // Find common prefix among last n transcripts
    std::vector<std::string> recent(history_.end() - n_, history_.end());
    std::string common_prefix = longest_common_prefix(recent);

    // Only commit what's new beyond previously committed
    if (common_prefix.length() > committed_.length()) {
        std::string new_text = common_prefix.substr(committed_.length());
        committed_ = common_prefix;
        return new_text;
    }

    return "";
}

std::string LocalAgreement::get_speculative() const {
    if (history_.empty()) {
        return "";
    }
    const std::string& latest = history_.back();
    if (latest.length() > committed_.length()) {
        return latest.substr(committed_.length());
    }
    return "";
}

void LocalAgreement::reset() {
    history_.clear();
    committed_.clear();
}

std::string LocalAgreement::longest_common_prefix(const std::vector<std::string>& strings) const {
    if (strings.empty()) {
        return "";
    }

    std::string prefix = strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        const std::string& s = strings[i];
        // Find word-aligned prefix for better results
        while (!s.empty() && prefix.length() > 0 &&
               (s.length() < prefix.length() || s.substr(0, prefix.length()) != prefix)) {
            // Try to back off to word boundary
            size_t space_idx = prefix.rfind(' ');
            if (space_idx != std::string::npos && space_idx > 0) {
                prefix = prefix.substr(0, space_idx);
            } else {
                // Character-by-character fallback
                prefix = prefix.substr(0, prefix.length() - 1);
            }
        }
        if (prefix.empty()) {
            return "";
        }
    }
    return prefix;
}

// ============================================================================
// StreamingTranscriber
// ============================================================================

StreamingTranscriber::StreamingTranscriber(WhisperModel& model, const StreamingConfig& config)
    : model_(model)
    , config_(config)
{
    // Initialize buffers
    float max_duration = config.max_chunk_duration + config.context_duration + 1.0f;
    audio_buffer_ = std::make_unique<AudioBuffer>(max_duration, config.sample_rate);
    speech_buffer_ = std::make_unique<AudioBuffer>(max_duration, config.sample_rate);

    // Initialize LocalAgreement if enabled
    if (config.use_local_agreement) {
        local_agreement_ = std::make_unique<LocalAgreement>(config.agreement_n);
    }

    // Initialize VAD if enabled
    if (config.use_vad) {
        init_vad();
    }

    detected_language_ = config.language;
}

StreamingTranscriber::~StreamingTranscriber() {
    // unique_ptr handles cleanup automatically
}

StreamingTranscriber::StreamingTranscriber(StreamingTranscriber&& other) noexcept
    : model_(other.model_)
    , config_(std::move(other.config_))
    , callback_(std::move(other.callback_))
    , state_(other.state_)
    , audio_buffer_(std::move(other.audio_buffer_))
    , speech_buffer_(std::move(other.speech_buffer_))
    , local_agreement_(std::move(other.local_agreement_))
    , segment_start_time_(other.segment_start_time_)
    , last_partial_time_(other.last_partial_time_)
    , silence_frames_(other.silence_frames_)
    , speech_frames_(other.speech_frames_)
    , context_audio_(std::move(other.context_audio_))
    , detected_language_(std::move(other.detected_language_))
    , silero_vad_(std::move(other.silero_vad_))
{
}

StreamingTranscriber& StreamingTranscriber::operator=(StreamingTranscriber&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        callback_ = std::move(other.callback_);
        state_ = other.state_;
        audio_buffer_ = std::move(other.audio_buffer_);
        speech_buffer_ = std::move(other.speech_buffer_);
        local_agreement_ = std::move(other.local_agreement_);
        segment_start_time_ = other.segment_start_time_;
        last_partial_time_ = other.last_partial_time_;
        silence_frames_ = other.silence_frames_;
        speech_frames_ = other.speech_frames_;
        context_audio_ = std::move(other.context_audio_);
        detected_language_ = std::move(other.detected_language_);
        silero_vad_ = std::move(other.silero_vad_);
    }
    return *this;
}

void StreamingTranscriber::set_callback(StreamingCallback callback) {
    callback_ = std::move(callback);
}

void StreamingTranscriber::process_audio(const float* samples, size_t count) {
    process_chunk(samples, count);
}

void StreamingTranscriber::process_audio(const std::vector<float>& samples) {
    process_audio(samples.data(), samples.size());
}

void StreamingTranscriber::process_chunk(const float* samples, size_t count) {
    // Add to main buffer (tracks total time)
    audio_buffer_->append(samples, count);

    // Run VAD if enabled
    bool is_speech = true;
    if (config_.use_vad) {
        is_speech = check_vad(samples, count);
    }

    // State machine
    if (state_ == StreamState::IDLE) {
        if (is_speech) {
            // Speech started
            state_ = StreamState::SPEECH;
            float audio_len = static_cast<float>(count) / config_.sample_rate;
            segment_start_time_ = audio_buffer_->total_time() - audio_len;
            speech_buffer_->clear();
            speech_buffer_->append(samples, count);
            silence_frames_ = 0;
            speech_frames_ = 1;
        }
    } else if (state_ == StreamState::SPEECH) {
        // Accumulate speech
        speech_buffer_->append(samples, count);

        if (is_speech) {
            speech_frames_++;
            silence_frames_ = 0;
        } else {
            silence_frames_++;
        }

        // Check for endpoint conditions
        float current_duration = speech_buffer_->duration();
        float silence_duration = static_cast<float>(silence_frames_ * count) / config_.sample_rate;

        // Endpoint detected (silence threshold)
        if (silence_duration >= config_.silence_threshold_duration) {
            process_segment(true, false);
            state_ = StreamState::IDLE;
            speech_buffer_->clear();
            context_audio_.clear();
        }
        // Max duration reached - forced processing
        else if (current_duration >= config_.max_chunk_duration) {
            process_segment(false, false);

            // Keep context for next chunk
            auto all_audio = speech_buffer_->get_all();
            size_t context_samples = static_cast<size_t>(config_.context_duration * config_.sample_rate);
            if (all_audio.size() > context_samples) {
                context_audio_.assign(all_audio.end() - context_samples, all_audio.end());
                speech_buffer_->clear();
                speech_buffer_->append(context_audio_);
            } else {
                context_audio_ = all_audio;
                speech_buffer_->clear();
            }
        }
        // Emit partial result if configured
        else if (config_.emit_partials) {
            float since_partial = audio_buffer_->total_time() - last_partial_time_;
            if (current_duration >= config_.min_chunk_duration &&
                since_partial >= config_.partial_interval) {
                process_segment(false, true);
                last_partial_time_ = audio_buffer_->total_time();
            }
        }
    }
}

void StreamingTranscriber::process_segment(bool is_final, bool is_partial) {
    size_t min_samples = static_cast<size_t>(config_.min_chunk_duration * config_.sample_rate);
    if (speech_buffer_->size() < min_samples) {
        return;
    }

    // Get audio to transcribe
    std::vector<float> audio_to_transcribe = speech_buffer_->get_all();

    // Add context from previous chunk if available
    if (!context_audio_.empty() && !is_partial) {
        std::vector<float> combined;
        combined.reserve(context_audio_.size() + audio_to_transcribe.size());
        combined.insert(combined.end(), context_audio_.begin(), context_audio_.end());
        combined.insert(combined.end(), audio_to_transcribe.begin(), audio_to_transcribe.end());
        audio_to_transcribe = std::move(combined);
    }

    float audio_duration = static_cast<float>(audio_to_transcribe.size()) / config_.sample_rate;

    // Compute mel spectrogram
    auto start_time = std::chrono::high_resolution_clock::now();

    auto mel = audio::log_mel_spectrogram(
        audio_to_transcribe,
        model_.config().n_mels,
        model_.config().n_fft,
        model_.config().hop_length
    );

    // Pad to 30s (3000 frames) for Whisper
    const int target_frames = 3000;
    if (mel.shape()[0] < target_frames) {
        mel = audio::pad_or_trim(mel, target_frames);
    }

    // Add batch dimension
    mel = mx::expand_dims(mel, 0);
    mx::eval(mel);

    // Generate transcription
    auto tokens = model_.generate(mel, config_.language, config_.task);

    auto end_time = std::chrono::high_resolution_clock::now();
    float processing_time = std::chrono::duration<float>(end_time - start_time).count();

    // Extract text tokens (exclude timestamps and special tokens)
    std::vector<int> text_tokens;
    for (int tok : tokens) {
        if (tok < 50257) {  // Not special token
            text_tokens.push_back(tok);
        }
    }

    // Decode tokens to text
    // For now, we don't have a tokenizer in this context, so we skip text decoding
    // The callback receives the token sequence; caller can decode if needed
    std::string text;

    // If we have a way to decode tokens, we would do it here
    // For now, create a placeholder result
    StreamingResult result;
    result.text = text;  // Will be empty without tokenizer
    result.is_final = is_final && !is_partial;
    result.is_partial = is_partial;
    result.segment_start = segment_start_time_;
    result.segment_end = audio_buffer_->total_time();
    result.language = detected_language_;
    result.processing_time = processing_time;
    result.audio_duration = audio_duration;

    // Apply LocalAgreement for partial results
    if (is_partial && local_agreement_) {
        std::string newly_confirmed = local_agreement_->update(text);
        result.confirmed_text = local_agreement_->get_confirmed();
        result.speculative_text = local_agreement_->get_speculative();
        result.is_confirmed = !newly_confirmed.empty();
    } else if (is_final && !is_partial) {
        result.confirmed_text = text;
        result.speculative_text = "";
        result.is_confirmed = true;
        if (local_agreement_) {
            local_agreement_->reset();
        }
    }

    // Deliver result via callback
    if (callback_) {
        callback_(result);
    }
}

void StreamingTranscriber::finalize() {
    state_ = StreamState::FINALIZING;

    size_t min_samples = static_cast<size_t>(config_.min_chunk_duration * config_.sample_rate);
    if (speech_buffer_->size() >= min_samples) {
        process_segment(true, false);
    }

    reset();
}

void StreamingTranscriber::reset() {
    state_ = StreamState::IDLE;
    audio_buffer_->clear();
    speech_buffer_->clear();
    if (local_agreement_) {
        local_agreement_->reset();
    }
    segment_start_time_ = 0.0f;
    last_partial_time_ = 0.0f;
    silence_frames_ = 0;
    speech_frames_ = 0;
    context_audio_.clear();
}

// ============================================================================
// VAD Implementation (Silero VAD neural network)
// ============================================================================

void StreamingTranscriber::init_vad() {
    // Create Silero VAD instance
    try {
        silero_vad_ = std::make_unique<silero_vad::SileroVAD>(
            config_.vad_weights_path,
            config_.sample_rate
        );
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to create Silero VAD: " << e.what() << std::endl;
        std::cerr << "         Using energy-based fallback" << std::endl;
        silero_vad_.reset();
    }
}

bool StreamingTranscriber::check_vad(const float* samples, size_t count) {
    if (count == 0) {
        return false;
    }

    // If Silero VAD not initialized, fall back to energy-based detection
    if (!silero_vad_) {
        // Simple energy-based fallback
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; ++i) {
            sum_sq += samples[i] * samples[i];
        }
        float rms = std::sqrt(static_cast<float>(sum_sq / count));
        const float speech_threshold = 0.01f;
        return rms > speech_threshold;
    }

    // Process with Silero VAD
    // Silero expects 512 samples at 16kHz (32ms chunks)
    int chunk_size = silero_vad_->chunk_size();

    // Process in chunks, accumulate probabilities
    int num_chunks = 0;
    int speech_chunks = 0;

    for (size_t offset = 0; offset + chunk_size <= count; offset += chunk_size) {
        float prob = silero_vad_->process(samples + offset, chunk_size);
        if (prob > config_.vad_threshold) {
            speech_chunks++;
        }
        num_chunks++;
    }

    // Handle remaining samples (less than full chunk)
    // For partial chunks, use energy-based detection
    size_t remaining = count % chunk_size;
    if (remaining > 0 && num_chunks == 0) {
        // Only partial chunk available - use energy detection
        double sum_sq = 0.0;
        for (size_t i = count - remaining; i < count; ++i) {
            sum_sq += samples[i] * samples[i];
        }
        float rms = std::sqrt(static_cast<float>(sum_sq / remaining));
        return rms > 0.01f;
    }

    // Return true if more than half of chunks contain speech
    if (num_chunks == 0) {
        return false;
    }
    return (speech_chunks * 2 >= num_chunks);
}

}  // namespace whisper
