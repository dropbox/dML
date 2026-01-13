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
//
// Architecture:
// - Encoder: Conv1D frontend + transformer self-attention
// - Decoder: Transformer with self-attention + cross-attention
// - Sinusoidal positional embeddings (encoder), learned (decoder)
//
// Based on WhisperMLX Python implementation (tools/whisper_mlx/)

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <functional>

#include "mlx/mlx.h"
#include "silero_vad.h"

namespace mx = mlx::core;

namespace whisper {

/**
 * GAP 50: Alignment head presets for DTW word timestamps.
 * Pre-configured alignment heads for different Whisper model variants.
 * These match whisper.cpp's whisper_alignment_heads_preset enum.
 */
enum class AlignmentHeadsPreset {
    NONE = 0,           // No preset, use from weights or default
    TINY_EN,            // tiny.en model
    TINY,               // tiny multilingual
    BASE_EN,            // base.en model
    BASE,               // base multilingual
    SMALL_EN,           // small.en model
    SMALL,              // small multilingual
    MEDIUM_EN,          // medium.en model
    MEDIUM,             // medium multilingual
    LARGE_V1,           // large v1
    LARGE_V2,           // large v2
    LARGE_V3,           // large v3
    LARGE_V3_TURBO,     // large-v3-turbo
};

/**
 * Get alignment heads for a preset.
 * Returns vector of (layer, head) pairs for DTW alignment.
 */
inline std::vector<std::pair<int, int>> get_alignment_heads_preset(AlignmentHeadsPreset preset) {
    switch (preset) {
        case AlignmentHeadsPreset::TINY_EN:
            return {{1, 0}, {2, 0}, {2, 5}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}};
        case AlignmentHeadsPreset::TINY:
            return {{2, 2}, {3, 0}, {3, 2}, {3, 3}, {3, 4}, {3, 5}};
        case AlignmentHeadsPreset::BASE_EN:
            return {{3, 3}, {4, 7}, {5, 1}, {5, 5}, {5, 7}};
        case AlignmentHeadsPreset::BASE:
            return {{3, 1}, {4, 2}, {4, 3}, {4, 7}, {5, 1}, {5, 2}, {5, 4}, {5, 6}};
        case AlignmentHeadsPreset::SMALL_EN:
            return {{6, 6}, {7, 0}, {7, 3}, {7, 8}, {8, 2}, {8, 5}, {8, 7}, {9, 0},
                    {9, 4}, {9, 8}, {9, 10}, {10, 0}, {10, 1}, {10, 2}, {10, 3}, {10, 6},
                    {10, 11}, {11, 2}, {11, 4}};
        case AlignmentHeadsPreset::SMALL:
            return {{5, 3}, {5, 9}, {8, 0}, {8, 4}, {8, 7}, {8, 8}, {9, 0}, {9, 7}, {9, 9}, {10, 5}};
        case AlignmentHeadsPreset::MEDIUM_EN:
            return {{11, 4}, {14, 1}, {14, 12}, {14, 14}, {15, 4}, {16, 0}, {16, 4}, {16, 9},
                    {17, 12}, {17, 14}, {18, 7}, {18, 10}, {18, 15}, {20, 0}, {20, 3}, {20, 9},
                    {20, 14}, {21, 12}};
        case AlignmentHeadsPreset::MEDIUM:
            return {{13, 15}, {15, 4}, {15, 15}, {16, 1}, {20, 0}, {23, 4}};
        case AlignmentHeadsPreset::LARGE_V1:
            return {{9, 19}, {11, 2}, {11, 4}, {11, 17}, {22, 7}, {22, 11}, {22, 17}, {23, 2}, {23, 15}};
        case AlignmentHeadsPreset::LARGE_V2:
            return {{10, 12}, {13, 17}, {16, 11}, {16, 12}, {16, 13}, {17, 15}, {17, 16}, {18, 4},
                    {18, 11}, {18, 19}, {19, 11}, {21, 2}, {21, 3}, {22, 3}, {22, 9}, {22, 12},
                    {23, 5}, {23, 7}, {23, 13}, {25, 5}, {26, 1}, {26, 12}, {27, 15}};
        case AlignmentHeadsPreset::LARGE_V3:
            return {{7, 0}, {10, 17}, {12, 18}, {13, 12}, {16, 1}, {17, 14}, {19, 11}, {21, 4}, {24, 1}, {25, 6}};
        case AlignmentHeadsPreset::LARGE_V3_TURBO:
            return {{2, 4}, {2, 11}, {3, 3}, {3, 6}, {3, 11}, {3, 14}};
        case AlignmentHeadsPreset::NONE:
        default:
            return {};  // Empty = use weights or default
    }
}

/**
 * Get alignment head preset from model name string.
 */
inline AlignmentHeadsPreset get_preset_from_model_name(const std::string& model_name) {
    // Match model names to presets
    if (model_name.find("large-v3-turbo") != std::string::npos ||
        model_name.find("turbo") != std::string::npos) {
        return AlignmentHeadsPreset::LARGE_V3_TURBO;
    } else if (model_name.find("large-v3") != std::string::npos) {
        return AlignmentHeadsPreset::LARGE_V3;
    } else if (model_name.find("large-v2") != std::string::npos) {
        return AlignmentHeadsPreset::LARGE_V2;
    } else if (model_name.find("large-v1") != std::string::npos ||
               model_name.find("large") != std::string::npos) {
        return AlignmentHeadsPreset::LARGE_V1;
    } else if (model_name.find("medium.en") != std::string::npos) {
        return AlignmentHeadsPreset::MEDIUM_EN;
    } else if (model_name.find("medium") != std::string::npos) {
        return AlignmentHeadsPreset::MEDIUM;
    } else if (model_name.find("small.en") != std::string::npos) {
        return AlignmentHeadsPreset::SMALL_EN;
    } else if (model_name.find("small") != std::string::npos) {
        return AlignmentHeadsPreset::SMALL;
    } else if (model_name.find("base.en") != std::string::npos) {
        return AlignmentHeadsPreset::BASE_EN;
    } else if (model_name.find("base") != std::string::npos) {
        return AlignmentHeadsPreset::BASE;
    } else if (model_name.find("tiny.en") != std::string::npos) {
        return AlignmentHeadsPreset::TINY_EN;
    } else if (model_name.find("tiny") != std::string::npos) {
        return AlignmentHeadsPreset::TINY;
    }
    return AlignmentHeadsPreset::NONE;
}

/**
 * Configuration for Whisper model.
 */
struct WhisperConfig {
    // Audio encoder
    int n_mels = 128;           // Mel frequency bands (80 for v1-v2, 128 for v3)
    int n_audio_ctx = 1500;     // Max audio context (30s at 50fps)
    int n_audio_state = 1280;   // Encoder hidden dimension
    int n_audio_head = 20;      // Encoder attention heads
    int n_audio_layer = 32;     // Encoder transformer layers

    // Text decoder
    int n_vocab = 51866;        // Vocabulary size (51866 for v3 multilingual)
    int n_text_ctx = 448;       // Max text context length
    int n_text_state = 1280;    // Decoder hidden dimension
    int n_text_head = 20;       // Decoder attention heads
    int n_text_layer = 32;      // Decoder transformer layers (4 for turbo)

    // Audio processing
    int sample_rate = 16000;    // Audio sample rate
    int n_fft = 400;            // FFT window size
    int hop_length = 160;       // Hop length between frames

    // Model metadata
    std::string name = "large-v3";

    // Special tokens
    int sot_token = 50258;      // Start of transcript
    int eot_token = 50257;      // End of transcript
    int translate_token = 50358; // Translate task
    int transcribe_token = 50359; // Transcribe task
    int solm_token = 50360;     // GAP 41: Speaker turn marker (tinydiarize) - "start of last message"
    int sot_prev_token = 50361; // Start of previous transcript (GAP 6: initial_prompt)
    int no_speech_token = 50362; // No speech token (for no_speech_prob calculation)
    int no_timestamps_token = 50363; // No timestamps
    int timestamp_begin = 50364;  // First timestamp token

    // Layer norm epsilon
    float layer_norm_eps = 1e-5f;

    // Alignment heads for word timestamps (GAP 26)
    // Each pair is (layer_index, head_index) for cross-attention heads
    // that are used for DTW alignment. Loaded from weights.safetensors.
    // Default: empty (will use all heads from last half of layers)
    std::vector<std::pair<int, int>> alignment_heads;

    // GAP 50: DTW alignment heads preset
    // If set to a value other than NONE, overrides alignment_heads with preset values
    // This allows using known-good alignment heads for models without weights metadata
    AlignmentHeadsPreset dtw_aheads_preset = AlignmentHeadsPreset::NONE;

    // Median filter width for DTW attention smoothing (GAP 27)
    // Python default is 7, C++ was skipping (effectively 1)
    int medfilt_width = 7;

    // GAP 54: Token-level timestamps flag
    // When enabled, return timestamps for each token (not just words)
    // Independent of word_timestamps. Uses same DTW alignment.
    bool token_timestamps = false;

    // GAP 55: Timestamp probability thresholds
    // thold_pt: Minimum probability for a timestamp token to be accepted
    // thold_ptsum: Minimum cumulative probability for timestamp acceptance
    // Set to 0.0 to disable (default). whisper.cpp uses 0.01 and 0.01.
    float thold_pt = 0.0f;
    float thold_ptsum = 0.0f;

    // Load config from JSON file
    static WhisperConfig load(const std::string& path);

    // Get predefined config by name
    static WhisperConfig get(const std::string& model_name);
};

/**
 * Weight storage for Whisper model.
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
 * KV Cache for decoder self-attention.
 */
struct DecoderKVCache {
    // [num_layers] each [batch, n_heads, seq_len, head_dim]
    // Note: After transpose in self_attention, shape is [batch, n_heads, seq, head_dim]
    std::vector<mx::array> keys;
    std::vector<mx::array> values;

    void clear() {
        keys.clear();
        values.clear();
    }

    bool empty() const { return keys.empty(); }

    // Get current sequence length from cache
    // KV cache shape is [batch, n_heads, seq_len, head_dim], so seq_len is at index 2
    int seq_len() const {
        if (keys.empty()) return 0;
        return static_cast<int>(keys[0].shape()[2]);  // Fixed: was shape()[1] (n_heads), now shape()[2] (seq_len)
    }

    // Clone the KV cache for beam search branching
    DecoderKVCache clone() const {
        DecoderKVCache cloned;
        cloned.keys.reserve(keys.size());
        cloned.values.reserve(values.size());
        for (const auto& k : keys) {
            cloned.keys.push_back(mx::array(k));  // Copy-on-write semantics
        }
        for (const auto& v : values) {
            cloned.values.push_back(mx::array(v));
        }
        return cloned;
    }

    // GAP 73: Stack multiple KV caches along batch dimension for parallel beam processing
    // Each input cache has shape [1, n_heads, seq_len, head_dim]
    // Output has shape [n_caches, n_heads, seq_len, head_dim]
    static DecoderKVCache stack(const std::vector<DecoderKVCache>& caches) {
        if (caches.empty()) return DecoderKVCache{};

        DecoderKVCache stacked;
        int n_layers = static_cast<int>(caches[0].keys.size());
        stacked.keys.reserve(n_layers);
        stacked.values.reserve(n_layers);

        for (int layer = 0; layer < n_layers; ++layer) {
            std::vector<mx::array> layer_keys, layer_values;
            for (const auto& cache : caches) {
                layer_keys.push_back(cache.keys[layer]);
                layer_values.push_back(cache.values[layer]);
            }
            stacked.keys.push_back(mx::concatenate(layer_keys, 0));
            stacked.values.push_back(mx::concatenate(layer_values, 0));
        }
        return stacked;
    }

    // GAP 73: Unstack a batched KV cache into individual caches
    // Input has shape [n_beams, n_heads, seq_len, head_dim]
    // Output is vector of caches each with shape [1, n_heads, seq_len, head_dim]
    static std::vector<DecoderKVCache> unstack(const DecoderKVCache& batched, int n_beams) {
        std::vector<DecoderKVCache> result(n_beams);
        if (batched.empty()) return result;

        int n_layers = static_cast<int>(batched.keys.size());
        for (int i = 0; i < n_beams; ++i) {
            result[i].keys.reserve(n_layers);
            result[i].values.reserve(n_layers);
        }

        for (int layer = 0; layer < n_layers; ++layer) {
            for (int i = 0; i < n_beams; ++i) {
                // Slice [i:i+1, :, :, :] to keep batch dimension
                result[i].keys.push_back(mx::slice(batched.keys[layer],
                    {i, 0, 0, 0},
                    {i + 1, batched.keys[layer].shape(1),
                     batched.keys[layer].shape(2), batched.keys[layer].shape(3)}));
                result[i].values.push_back(mx::slice(batched.values[layer],
                    {i, 0, 0, 0},
                    {i + 1, batched.values[layer].shape(1),
                     batched.values[layer].shape(2), batched.values[layer].shape(3)}));
            }
        }
        return result;
    }
};

/**
 * Cross-attention KV Cache for encoder output projections.
 * Computed once per transcription, reused for all decode steps.
 */
struct CrossKVCache {
    // [num_layers] each [batch, n_heads, enc_seq, head_dim]
    std::vector<mx::array> keys;
    std::vector<mx::array> values;

    void clear() {
        keys.clear();
        values.clear();
    }

    bool empty() const { return keys.empty(); }

    // Clone not needed - cross-attention cache is shared across all beams
};

/**
 * A single beam hypothesis in beam search.
 * Tracks the token sequence, log probability, and state.
 */
struct Beam {
    std::vector<int> tokens;
    float log_prob = 0.0f;
    bool finished = false;

    // Comparison for sorting (higher log_prob = better)
    bool operator<(const Beam& other) const {
        return log_prob > other.log_prob;  // Descending order
    }
};

/**
 * Result from beam search decoding.
 */
struct BeamSearchResult {
    std::vector<int> tokens;
    float log_prob = 0.0f;
    float normalized_score = 0.0f;
    float avg_logprob = 0.0f;
    float no_speech_prob = 0.0f;  // GAP 13: Probability of no speech at SOT
};

/**
 * Result from language detection.
 * Contains detected language code and probability distribution.
 */
struct DetectLanguageResult {
    std::string language;           // ISO 639-1 language code (e.g., "en", "es", "zh")
    int language_token = 0;         // Token ID for detected language
    float probability = 0.0f;       // Probability of detected language
    std::unordered_map<std::string, float> language_probs;  // Full probability distribution
};

/**
 * Token-level timing information.
 * Used for precise alignment via cross-attention weights and DTW.
 */
struct WhisperTokenTiming {
    int token_id = 0;               // Token ID
    float start_time = 0.0f;        // Start time in seconds
    float end_time = 0.0f;          // End time in seconds
    float probability = 0.0f;       // Token probability (softmax)
    float dtw_score = 0.0f;         // DTW alignment score (lower = better)
};

/**
 * Word-level timing information.
 * Groups tokens into words with timing boundaries.
 */
struct WhisperWord {
    std::string word;               // The word text
    float start_time = 0.0f;        // Word start time in seconds
    float end_time = 0.0f;          // Word end time in seconds
    float probability = 1.0f;       // Average token probability
    std::vector<WhisperTokenTiming> tokens;  // Constituent tokens
};

/**
 * A single transcription segment with timing information.
 * Created by segment-based decoding for long audio.
 */
struct WhisperSegment {
    float start_time = 0.0f;        // Segment start in seconds
    float end_time = 0.0f;          // Segment end in seconds
    std::vector<int> tokens;        // Token IDs (excluding timestamps)
    float avg_logprob = 0.0f;       // Quality metric
    float no_speech_prob = 0.0f;    // Probability of no speech
    float compression_ratio = 0.0f; // Text compression ratio (detect hallucinations)
    float temperature = 0.0f;       // Temperature used for this segment (GAP 4/12)
    std::vector<WhisperWord> words; // Word-level timestamps (optional)
    bool speaker_turn_next = false; // GAP 41: Speaker turn detected after this segment (tinydiarize)
};

/**
 * Result from segment-based transcription.
 * Used for long audio (>30s) processing.
 */
struct SegmentedTranscriptionResult {
    std::vector<WhisperSegment> segments;
    std::string language;           // Detected/specified language
    float total_duration = 0.0f;    // Total audio duration in seconds
};

// =============================================================================
// GAP 57: Composable LogitFilter Architecture
// =============================================================================
// Python mlx-whisper uses a composable filter chain:
//   self.logit_filters = [SuppressBlank(...), SuppressTokens(...), ApplyTimestampRules(...)]
//   for logit_filter in self.logit_filters:
//       logits = logit_filter.apply(logits, tokens)
//
// This C++ implementation provides equivalent composability.
// =============================================================================

/**
 * Context passed to logit filters during decoding.
 * Contains all state needed for filter decisions.
 */
struct LogitFilterContext {
    const std::vector<int>& tokens;     // Current token sequence
    int sample_begin;                    // Index where sampling starts (after prompt)
    int n_vocab;                         // Vocabulary size
    int timestamp_begin;                 // First timestamp token ID
    int eot_token;                       // End of transcript token
    int max_initial_timestamp_index;     // Max timestamp at start (from config)
    int max_audio_timestamp_index;       // Max timestamp for this audio duration
    int step;                            // Current decode step (0-indexed)
};

/**
 * Abstract base class for logit filters.
 *
 * Logit filters modify the logits before sampling to enforce decoding constraints.
 * Examples: suppress special tokens, enforce timestamp rules, apply repetition penalty.
 */
class LogitFilter {
public:
    virtual ~LogitFilter() = default;

    /**
     * Apply the filter to logits.
     *
     * @param logits Mutable logit vector to modify in-place (size = n_vocab)
     * @param ctx Context containing token sequence and decoding state
     */
    virtual void apply(std::vector<float>& logits, const LogitFilterContext& ctx) = 0;

    /**
     * Get filter name for debugging.
     */
    virtual const char* name() const = 0;
};

/**
 * SuppressBlank filter (from Python mlx-whisper).
 *
 * At the start of sampling, suppress space token and EOT to prevent
 * empty or whitespace-only outputs.
 */
class SuppressBlank : public LogitFilter {
public:
    explicit SuppressBlank(int space_token = 220) : space_token_(space_token) {}

    void apply(std::vector<float>& logits, const LogitFilterContext& ctx) override;
    const char* name() const override { return "SuppressBlank"; }

private:
    int space_token_;
};

/**
 * SuppressTokens filter (from Python mlx-whisper).
 *
 * Suppress a static set of tokens that should never be generated:
 * - SOT, language tokens, task tokens
 * - Non-speech tokens (punctuation, special characters)
 * - no_timestamps token (when in timestamp mode)
 */
class SuppressTokens : public LogitFilter {
public:
    explicit SuppressTokens(const std::vector<int>& suppress_tokens)
        : suppress_tokens_(suppress_tokens) {}

    void apply(std::vector<float>& logits, const LogitFilterContext& ctx) override;
    const char* name() const override { return "SuppressTokens"; }

private:
    std::vector<int> suppress_tokens_;
};

/**
 * ApplyTimestampRules filter (from Python mlx-whisper decoding.py).
 *
 * Enforces Whisper's timestamp token rules:
 * 1. First token must be a timestamp
 * 2. Timestamps must be monotonically increasing
 * 3. After two timestamps in a row, next must be text
 * 4. After text followed by timestamp, next must be timestamp or EOT
 * 5. Timestamps cannot exceed audio duration
 * 6. Apply max_initial_timestamp constraint at start
 */
class ApplyTimestampRules : public LogitFilter {
public:
    ApplyTimestampRules() = default;

    void apply(std::vector<float>& logits, const LogitFilterContext& ctx) override;
    const char* name() const override { return "ApplyTimestampRules"; }

    /**
     * Compute timestamp probability dominance.
     * If sum of timestamp probabilities > max text probability, returns true.
     * Must be called BEFORE applying suppression rules to get original logits.
     */
    static bool compute_timestamp_dominance(
        const std::vector<float>& logits,
        int timestamp_begin,
        int n_vocab
    );

private:
    // Internal helper to check timestamp state
    void get_timestamp_state(
        const std::vector<int>& tokens,
        int sample_begin,
        int timestamp_begin,
        bool& last_was_timestamp,
        bool& penultimate_was_timestamp,
        std::vector<int>& timestamp_tokens
    );
};

/**
 * SuppressRegex filter (GAP 43 - whisper.cpp feature).
 *
 * Suppresses tokens whose text matches a regular expression.
 * This allows constrained decoding by filtering out specific patterns.
 * Token IDs to suppress are pre-computed at construction for efficiency.
 *
 * Example use cases:
 * - "[0-9]+" - suppress numeric tokens
 * - "\\[.*\\]" - suppress bracketed content like [MUSIC], [LAUGHTER]
 * - "(music|laughter)" - suppress specific words (case-insensitive with std::regex_constants::icase)
 */
class SuppressRegex : public LogitFilter {
public:
    /**
     * Construct filter with regex pattern and vocabulary.
     * Pre-computes all matching token IDs for efficiency.
     *
     * @param regex_pattern The regular expression pattern to match
     * @param id_to_token Map of token IDs to their string representations
     */
    SuppressRegex(const std::string& regex_pattern,
                  const std::unordered_map<int, std::string>& id_to_token);

    void apply(std::vector<float>& logits, const LogitFilterContext& ctx) override;
    const char* name() const override { return "SuppressRegex"; }

    /**
     * Get the number of tokens that will be suppressed.
     */
    size_t num_suppressed() const { return suppress_tokens_.size(); }

    /**
     * Get the regex pattern being used.
     */
    const std::string& pattern() const { return pattern_; }

private:
    std::string pattern_;
    std::vector<int> suppress_tokens_;  // Pre-computed token IDs to suppress
};

/**
 * Chain of logit filters applied in sequence.
 *
 * Matches Python's:
 *   for logit_filter in self.logit_filters:
 *       logits = logit_filter.apply(logits, tokens)
 */
class LogitFilterChain {
public:
    LogitFilterChain() = default;

    /**
     * Add a filter to the chain.
     * Filters are applied in the order they are added.
     */
    void add(std::unique_ptr<LogitFilter> filter) {
        filters_.push_back(std::move(filter));
    }

    /**
     * Add a filter to the chain (takes ownership).
     */
    template<typename T, typename... Args>
    void emplace(Args&&... args) {
        filters_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    /**
     * Apply all filters in sequence.
     */
    void apply(std::vector<float>& logits, const LogitFilterContext& ctx) {
        for (auto& filter : filters_) {
            filter->apply(logits, ctx);
        }
    }

    /**
     * Get number of filters in chain.
     */
    size_t size() const { return filters_.size(); }

    /**
     * Check if chain is empty.
     */
    bool empty() const { return filters_.empty(); }

    /**
     * Clear all filters.
     */
    void clear() { filters_.clear(); }

private:
    std::vector<std::unique_ptr<LogitFilter>> filters_;
};

// Forward declaration for grammar support
namespace grammar_parser { struct ParseState; }
struct Grammar;

/**
 * GAP 42: Grammar-based logit filter for constrained decoding.
 *
 * Suppresses tokens that would violate a BNF grammar constraint.
 * Useful for forcing specific output formats (dates, numbers, commands).
 *
 * Example grammar:
 *   root ::= digit digit "/" digit digit "/" digit digit digit digit
 *   digit ::= [0-9]
 *
 * This would constrain output to dates like "12/31/2024".
 */
class GrammarLogitFilter : public LogitFilter {
public:
    /**
     * Construct filter with BNF grammar string and tokenizer vocabulary.
     *
     * @param grammar_str BNF grammar string (e.g., "root ::= [0-9]+")
     * @param id_to_token Map of token IDs to their string representations
     * @param penalty Penalty to apply to rejected tokens (default: large negative)
     * @param start_rule Index of the start rule (default: 0 = first rule)
     */
    GrammarLogitFilter(
        const std::string& grammar_str,
        const std::unordered_map<int, std::string>& id_to_token,
        float penalty = -1000.0f,
        size_t start_rule = 0);

    ~GrammarLogitFilter();

    void apply(std::vector<float>& logits, const LogitFilterContext& ctx) override;
    const char* name() const override { return "Grammar"; }

    /**
     * Accept a token and update grammar state.
     * Must be called after each token is sampled.
     */
    void accept_token(int token_id);

    /**
     * Reset grammar state to initial configuration.
     */
    void reset();

    /**
     * Check if grammar is active (successfully parsed).
     */
    bool active() const;

private:
    std::unique_ptr<Grammar> grammar_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<int, std::vector<uint32_t>> token_code_points_;  // Cached UTF-8 decode
    float penalty_;
};

/**
 * Audio encoder (ConvNet + Transformer).
 * Uses shared Weights reference to avoid storing mx::array directly.
 */
class AudioEncoder {
public:
    /**
     * Initialize encoder with config and weights.
     */
    void init(const WhisperConfig& config, const Weights* weights);

    /**
     * Encode mel spectrogram to hidden states.
     * @param mel Mel spectrogram [batch, n_frames, n_mels]
     * @return Encoder output [batch, seq_len, n_state]
     */
    mx::array encode(const mx::array& mel);

    /**
     * Get output sequence length for given input frames.
     * Conv2 has stride 2, so output is roughly half input.
     */
    int get_output_length(int n_frames) const {
        return (n_frames + 1) / 2;
    }

    /**
     * Check if encoder is initialized.
     */
    bool initialized() const { return weights_ != nullptr; }

private:
    WhisperConfig config_;
    const Weights* weights_ = nullptr;

    // Positional embedding (precomputed sinusoidal, stored separately)
    std::vector<float> positional_embedding_data_;
    int pos_emb_len_ = 0;

    // Forward through single layer
    mx::array encoder_layer(const mx::array& x, int layer_idx);

    // Self-attention (encoder, no causal mask)
    mx::array self_attention(const mx::array& x, int layer_idx);

    // MLP (GELU activation)
    mx::array mlp(const mx::array& x, int layer_idx);
};

/**
 * Result of decode with attention weights.
 * Used for word-level timestamp extraction via DTW.
 */
struct DecodeWithAttentionResult {
    mx::array logits;                              // [batch, seq_len, n_vocab]
    std::vector<mx::array> cross_attention_weights; // [n_layers] each [batch, n_head, seq_len, enc_len]
};

/**
 * Text decoder (Transformer with cross-attention).
 * Uses shared Weights reference to avoid storing mx::array directly.
 */
class TextDecoder {
public:
    /**
     * Initialize decoder with config and weights.
     */
    void init(const WhisperConfig& config, const Weights* weights);

    /**
     * Decode tokens with encoder output.
     * @param tokens Token IDs [batch, seq_len]
     * @param encoder_output Encoder hidden states [batch, enc_len, n_state]
     * @param kv_cache Optional decoder KV cache
     * @param cross_cache Optional cross-attention KV cache
     * @return Logits [batch, seq_len, n_vocab]
     */
    mx::array decode(
        const mx::array& tokens,
        const mx::array& encoder_output,
        DecoderKVCache* kv_cache = nullptr,
        CrossKVCache* cross_cache = nullptr
    );

    /**
     * Decode tokens and return cross-attention weights for DTW alignment.
     *
     * This is more expensive than regular decode as it cannot use SDPA
     * (which doesn't return attention weights). Use only when word-level
     * timestamps are needed.
     *
     * @param tokens Token IDs [batch, seq_len]
     * @param encoder_output Encoder hidden states [batch, enc_len, n_state]
     * @return Logits and cross-attention weights for each layer
     */
    DecodeWithAttentionResult decode_with_attention(
        const mx::array& tokens,
        const mx::array& encoder_output
    );

    /**
     * Check if decoder is initialized.
     */
    bool initialized() const { return weights_ != nullptr; }

private:
    WhisperConfig config_;
    const Weights* weights_ = nullptr;

    // Causal mask (precomputed, stored as float vector)
    std::vector<float> causal_mask_data_;

    // Forward through single layer
    mx::array decoder_layer(
        const mx::array& x,
        const mx::array& encoder_output,
        int layer_idx,
        DecoderKVCache* kv_cache,
        CrossKVCache* cross_cache
    );

    // Self-attention with causal mask
    mx::array self_attention(
        const mx::array& x,
        int layer_idx,
        DecoderKVCache* kv_cache
    );

    // Cross-attention to encoder output
    mx::array cross_attention(
        const mx::array& x,
        const mx::array& encoder_output,
        int layer_idx,
        CrossKVCache* cross_cache
    );

    // Cross-attention that also returns attention weights for DTW
    // Returns pair of (output, attention_weights)
    // attention_weights shape: [batch, n_head, seq_len, enc_len]
    std::pair<mx::array, mx::array> cross_attention_with_weights(
        const mx::array& x,
        const mx::array& encoder_output,
        int layer_idx
    );

    // Decoder layer with attention weight extraction
    // Returns pair of (output, cross_attention_weights)
    std::pair<mx::array, mx::array> decoder_layer_with_attention(
        const mx::array& x,
        const mx::array& encoder_output,
        int layer_idx
    );

    // MLP (GELU activation)
    mx::array mlp(const mx::array& x, int layer_idx);
};

/**
 * Whisper Tokenizer.
 * Uses tiktoken for BPE encoding/decoding.
 */
class WhisperTokenizer {
public:
    WhisperTokenizer();
    ~WhisperTokenizer();

    // Move semantics
    WhisperTokenizer(WhisperTokenizer&&) noexcept;
    WhisperTokenizer& operator=(WhisperTokenizer&&) noexcept;

    // Disable copy
    WhisperTokenizer(const WhisperTokenizer&) = delete;
    WhisperTokenizer& operator=(const WhisperTokenizer&) = delete;

    /**
     * Load tokenizer from model directory.
     * Expects vocab.json and merges.txt files.
     */
    static WhisperTokenizer load(const std::string& model_path);

    /**
     * Encode text to token IDs.
     */
    std::vector<int> encode(const std::string& text) const;

    /**
     * Decode token IDs to text.
     */
    std::string decode(const std::vector<int>& ids) const;

    /**
     * Get vocabulary size.
     */
    int vocab_size() const;

    /**
     * Get the id_to_token map (for SuppressRegex filter).
     * Returns a reference to internal vocabulary mapping.
     */
    const std::unordered_map<int, std::string>& get_id_to_token() const;

    /**
     * Check if loaded.
     */
    bool loaded() const { return loaded_; }

    // Special tokens
    int sot_token() const { return sot_token_; }
    int eot_token() const { return eot_token_; }
    int timestamp_begin() const { return timestamp_begin_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;

    // Special tokens
    int sot_token_ = 50258;
    int eot_token_ = 50257;
    int timestamp_begin_ = 50364;
};

/**
 * Full Whisper model combining encoder, decoder, and tokenizer.
 */
class WhisperModel {
public:
    WhisperModel();
    ~WhisperModel();

    // Move semantics
    WhisperModel(WhisperModel&&) noexcept;
    WhisperModel& operator=(WhisperModel&&) noexcept;

    // Disable copy
    WhisperModel(const WhisperModel&) = delete;
    WhisperModel& operator=(const WhisperModel&) = delete;

    /**
     * Load model from directory.
     * Expected structure:
     *   model.safetensors or weights.npz
     *   config.json
     *   vocab.json + merges.txt (for tokenizer)
     */
    static WhisperModel load(const std::string& model_path);

    /**
     * Get model configuration.
     */
    const WhisperConfig& config() const { return config_; }

    /**
     * GAP 50: Set alignment heads using a preset.
     * Overrides any alignment heads loaded from model weights.
     */
    void set_alignment_heads_preset(AlignmentHeadsPreset preset) {
        auto heads = get_alignment_heads_preset(preset);
        if (!heads.empty()) {
            config_.alignment_heads = std::move(heads);
            config_.dtw_aheads_preset = preset;
        }
    }

    /**
     * Check if model is loaded.
     */
    bool loaded() const { return loaded_; }

    /**
     * Encode mel spectrogram.
     */
    mx::array encode(const mx::array& mel);

    /**
     * Decode tokens with encoder output.
     */
    mx::array decode(
        const mx::array& tokens,
        const mx::array& encoder_output,
        DecoderKVCache* kv_cache = nullptr,
        CrossKVCache* cross_cache = nullptr
    );

    /**
     * Generate transcription with greedy or temperature-based sampling.
     * @param mel Mel spectrogram [batch, n_frames, n_mels]
     * @param language Language code (e.g., "en", "es")
     * @param task "transcribe" or "translate"
     * @param max_tokens Maximum tokens to generate
     * @param audio_duration_sec Actual audio duration in seconds (for timestamp limiting)
     *                          If <= 0, no timestamp limiting is applied
     * @param avg_logprob_out Optional output for average log probability (quality metric)
     *                        If non-null, will be set to avg_logprob = sum_logprobs / (n_tokens + 1)
     * @param no_speech_prob_out Optional output for no-speech probability (quality metric)
     *                           If non-null, will be set to softmax(logits)[no_speech_token] at first step
     * @param prompt_tokens Optional prompt tokens to prepend (for initial_prompt/context conditioning)
     *                      These are prepended as [sot_prev, prompt_tokens...] before [SOT, lang, task]
     *                      GAP 6: Matches Python's DecodingOptions.prompt behavior
     * @param temperature Sampling temperature (GAP 4/12: temperature fallback support)
     *                    = 0.0: greedy decoding (argmax)
     *                    > 0.0: categorical sampling with logits/temperature
     *                    Python: mx.random.categorical(logits / temp)
     * @param skip_logprobs GAP M: Skip log probability calculation for speed
     *                      When true, avg_logprob_out will not be calculated (saves computation)
     * @return Generated token IDs
     */
    std::vector<int> generate(
        const mx::array& mel,
        const std::string& language = "en",
        const std::string& task = "transcribe",
        int max_tokens = 448,
        float audio_duration_sec = -1.0f,
        float* avg_logprob_out = nullptr,
        float* no_speech_prob_out = nullptr,
        const std::vector<int>* prompt_tokens = nullptr,
        float temperature = 0.0f,
        float repetition_penalty = 1.0f,  // GAP K: 1.0 = disabled, >1.0 discourages repetition
        float max_initial_timestamp = 1.0f,  // GAP: max_initial_timestamp in seconds (Python default: 1.0)
        bool without_timestamps = false,  // GAP: without_timestamps mode (use <|notimestamps|>)
        const std::vector<int>* prefix_tokens = nullptr,  // GAP: decoder prefix (added after SOT sequence)
        int sample_len = -1,  // GAP: sample_len - max tokens to sample (-1 = default to n_text_ctx/2)
        bool skip_logprobs = false,  // GAP M: skip log probability calculation for speed
        const std::vector<int>* suppress_regex_tokens = nullptr,  // GAP 43: pre-computed tokens matching regex
        bool tdrz_enable = false  // GAP 41: tinydiarize speaker turn detection
    );

    /**
     * Generate transcription with beam search decoding.
     *
     * Maintains multiple hypotheses during decoding, selecting the best
     * one based on log probability. Provides better transcription quality
     * than greedy decoding at the cost of beam_size * compute.
     *
     * @param mel Mel spectrogram [batch, n_frames, n_mels]
     * @param language Language code (e.g., "en", "es")
     * @param task "transcribe" or "translate"
     * @param beam_size Number of beams to maintain (default: 5)
     * @param length_penalty Length penalty for beam scoring (default: 1.0)
     *                      > 1.0 encourages longer sequences
     *                      < 1.0 encourages shorter sequences
     * @param max_tokens Maximum tokens to generate
     * @param prompt_tokens Optional prompt tokens to prepend (for initial_prompt/context conditioning)
     *                      GAP 6: Matches Python's DecodingOptions.prompt behavior
     * @return BeamSearchResult with best hypothesis
     */
    BeamSearchResult generate_beam(
        const mx::array& mel,
        const std::string& language = "en",
        const std::string& task = "transcribe",
        int beam_size = 5,
        float length_penalty = 1.0f,
        int max_tokens = 448,
        const std::vector<int>* prompt_tokens = nullptr,
        float patience = 1.0f,  // GAP 9: beam search patience (arxiv:2204.05424)
        float max_initial_timestamp = 1.0f,  // GAP: max_initial_timestamp in seconds
        bool without_timestamps = false  // GAP: without_timestamps mode
    );

    /**
     * Generate transcription for long audio using segment-based decoding.
     *
     * Processes audio in 30-second chunks, using timestamp tokens to
     * determine segment boundaries and advancing seek position accordingly.
     * This is the recommended method for audio longer than 30 seconds.
     *
     * @param mel Full mel spectrogram [batch, n_frames, n_mels]
     * @param language Language code (e.g., "en", "es")
     * @param task "transcribe" or "translate"
     * @param condition_on_previous Use previous segment text as context
     * @param no_speech_threshold Skip segments with no_speech_prob above this
     * @param compression_ratio_threshold Retry segments with high compression
     * @param beam_size Number of beams (1 = greedy, >1 = beam search)
     * @param length_penalty Length penalty for beam search
     * @param word_timestamps Enable word-level timestamps (expensive: requires DTW)
     * @param tokenizer Tokenizer for word text extraction (nullptr = no text)
     * @param actual_audio_duration If > 0, use this instead of mel-based estimate
     * @param initial_prompt Optional prompt string for context conditioning (GAP 6)
     *                       Tokenized and prepended to decoder input as [sot_prev, tokens...]
     *                       Useful for domain-specific vocabulary, speaker names, etc.
     * @param carry_initial_prompt If true, prepend initial_prompt to each segment's context
     *                             If false (default), only use previous transcript as context
     * @param temperatures Temperature fallback sequence (GAP 4/12)
     *                     Default: {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
     *                     If quality fails at T[0], retries with T[1], etc.
     *                     Quality failure = compression_ratio > threshold OR avg_logprob < logprob_threshold
     * @param logprob_threshold Retry if avg_logprob < this (GAP J). Default: -1.0 (Python default)
     *                          Combined with no_speech_prob for silence detection
     * @param hallucination_silence_threshold GAP 20: Skip silence before/after hallucinations.
     *                                         0.0 = disabled (default). Python default is None (disabled).
     *                                         When enabled, skips leading silence before anomalous segments
     *                                         and silence surrounding segments detected as hallucinations.
     * @param split_on_word GAP 46: When true, segment boundaries prefer word boundaries over token boundaries.
     *                      Results in cleaner segments that don't split mid-word.
     *                      Default: false (split on tokens like Python mlx-whisper).
     * @param skip_logprobs GAP M: Skip log probability calculation for speed.
     *                      When true, avg_logprob will not be calculated (saves computation).
     * @param prefix GAP: Decoder prefix text (prepended after SOT sequence).
     *               Unlike initial_prompt (which conditions on previous transcript),
     *               prefix is prepended to current transcript as if already spoken.
     * @param print_realtime GAP 53: Print segments to stderr as they're generated.
     *                       Format: [start_time --> end_time] text
     *                       Useful for real-time feedback during transcription.
     * @return SegmentedTranscriptionResult with all segments
     */
    SegmentedTranscriptionResult generate_segments(
        const mx::array& mel,
        const std::string& language = "en",
        const std::string& task = "transcribe",
        bool condition_on_previous = true,
        float no_speech_threshold = 0.6f,
        float compression_ratio_threshold = 2.4f,
        int beam_size = 1,
        float length_penalty = 1.0f,
        bool word_timestamps = false,
        const WhisperTokenizer* tokenizer = nullptr,
        float actual_audio_duration = -1.0f,
        const std::string& initial_prompt = "",
        bool carry_initial_prompt = false,
        const std::vector<float>& temperatures = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f},
        float logprob_threshold = -1.0f,
        float hallucination_silence_threshold = 0.0f,
        const std::vector<float>& clip_timestamps = {},  // GAP 8: start,end,start,end... timestamps to process
        float entropy_thold = 0.0f,  // GAP 47: entropy threshold for fallback (0 = disabled)
        bool split_on_word = false,  // GAP 46: split segments at word boundaries
        bool skip_logprobs = false,  // GAP M: skip log probability calculation for speed
        const std::string& prefix = "",  // GAP: decoder prefix (prepended after SOT)
        const std::vector<int>& suppress_regex_tokens = {},  // GAP 43: pre-computed tokens matching regex
        bool token_timestamps = false,  // GAP 54: return timestamps for each token
        float thold_pt = 0.0f,  // GAP 55: timestamp token probability threshold
        float thold_ptsum = 0.0f,  // GAP 55: timestamp token sum probability threshold
        bool print_realtime = false,  // GAP 53: print segments to stderr as generated
        bool tdrz_enable = false  // GAP 41: tinydiarize speaker turn detection
    );

    /**
     * Add word-level timestamps to a segment.
     *
     * Uses DTW (Dynamic Time Warping) on cross-attention weights to align
     * tokens to audio frames, then groups tokens into words.
     *
     * This is an expensive operation that requires re-running decoder inference
     * with attention weight extraction enabled.
     *
     * @param segment Segment to add word timestamps to
     * @param encoder_output Encoder output for this segment
     * @param language Language code for tokenizer
     * @param tokenizer Optional tokenizer for word text extraction (nullptr = no text)
     * @param actual_audio_duration Actual audio duration in seconds (overrides segment timing for DTW)
     *                              If <= 0, uses segment.end_time - segment.start_time
     * @param last_speech_timestamp Pointer to track last speech timestamp across segments (GAP 68, 78-80)
     *                              Pass nullptr if not tracking across segments
     * @return Segment with words vector populated, updates last_speech_timestamp if provided
     */
    void add_word_timestamps(
        WhisperSegment& segment,
        const mx::array& encoder_output,
        const std::string& language = "en",
        const WhisperTokenizer* tokenizer = nullptr,
        float actual_audio_duration = -1.0f,
        float* last_speech_timestamp = nullptr
    );

    /**
     * Detect spoken language in audio.
     *
     * Runs encoder and decoder forward pass with SOT token only,
     * then applies softmax over language tokens to determine the
     * most likely language.
     *
     * This is performed outside the main decode loop to not interfere
     * with KV caching.
     *
     * @param mel Mel spectrogram [batch, n_frames, n_mels] or encoder output
     * @return DetectLanguageResult with detected language and probabilities
     */
    DetectLanguageResult detect_language(const mx::array& mel);

    /**
     * Check if model is multilingual.
     * @return true if model supports multiple languages (n_vocab >= 51865)
     */
    bool is_multilingual() const;

    /**
     * Get number of supported languages.
     * @return Number of language tokens in vocabulary
     */
    int num_languages() const;

    /**
     * Get model info string.
     */
    std::string info() const;

private:
    WhisperConfig config_;
    std::unique_ptr<AudioEncoder> encoder_;
    std::unique_ptr<TextDecoder> decoder_;
    std::unique_ptr<Weights> weights_;
    bool loaded_ = false;

    // Language token lookup (code -> token ID)
    std::unordered_map<std::string, int> language_tokens_;
    // Reverse lookup (token ID -> code)
    std::unordered_map<int, std::string> token_to_language_;
    void init_language_tokens();

    // Get language token ID
    int get_language_token(const std::string& language) const;

    // Get language code from token ID
    std::string get_language_code(int token) const;
};

/**
 * Audio processing utilities.
 */
namespace audio {

/**
 * Load audio file and return samples at target sample rate.
 * Uses ffmpeg for decoding (same as Python implementation).
 * @param path Path to audio file
 * @param sample_rate Target sample rate (default 16000)
 * @return Audio samples as float array
 */
std::vector<float> load_audio(const std::string& path, int sample_rate = 16000, bool from_stdin = false);

/**
 * Compute log mel spectrogram from audio samples.
 * @param audio Audio samples (mono, float)
 * @param n_mels Number of mel frequency bands
 * @param n_fft FFT window size
 * @param hop_length Hop length between frames
 * @return Mel spectrogram [n_frames, n_mels]
 */
mx::array log_mel_spectrogram(
    const std::vector<float>& audio,
    int n_mels = 128,
    int n_fft = 400,
    int hop_length = 160
);

/**
 * Pad or trim mel spectrogram to target length.
 */
mx::array pad_or_trim(const mx::array& mel, int target_length);

}  // namespace audio

/**
 * Quality metrics for hallucination detection (GAP 12/16).
 */

/**
 * Calculate compression ratio of text using zlib.
 * High compression ratio indicates repetitive/hallucinated text.
 *
 * Python equivalent:
 *   def compression_ratio(text) -> float:
 *       text_bytes = text.encode("utf-8")
 *       return len(text_bytes) / len(zlib.compress(text_bytes))
 *
 * @param text UTF-8 encoded text
 * @return compression_ratio = original_size / compressed_size
 *         Higher values (>2.4) indicate potential hallucination
 */
float calculate_compression_ratio(const std::string& text);

/**
 * High-level transcription interface.
 */
class Transcriber {
public:
    Transcriber();
    ~Transcriber();

    // Move semantics
    Transcriber(Transcriber&&) noexcept;
    Transcriber& operator=(Transcriber&&) noexcept;

    // Disable copy
    Transcriber(const Transcriber&) = delete;
    Transcriber& operator=(const Transcriber&) = delete;

    /**
     * Load transcriber with specified model.
     * @param model_name Model name (e.g., "large-v3-turbo")
     */
    static Transcriber load(const std::string& model_name);

    /**
     * Transcribe audio file.
     * @param audio_path Path to audio file
     * @param language Language code (empty for auto-detect)
     * @return Transcribed text
     */
    std::string transcribe(
        const std::string& audio_path,
        const std::string& language = ""
    ) const;

    /**
     * Transcribe audio samples directly.
     * @param audio Audio samples (16kHz mono float)
     * @param language Language code (empty for auto-detect)
     * @return Transcribed text
     */
    std::string transcribe(
        const std::vector<float>& audio,
        const std::string& language = ""
    ) const;

    /**
     * Check if loaded.
     */
    bool loaded() const { return model_ && model_->loaded(); }

    /**
     * Get model info.
     */
    std::string info() const { return model_ ? model_->info() : "Not loaded"; }

private:
    std::unique_ptr<WhisperModel> model_;
    std::unique_ptr<WhisperTokenizer> tokenizer_;
};

// ============================================================================
// Streaming Transcription
// ============================================================================

/**
 * Configuration for streaming transcription.
 */
struct StreamingConfig {
    // Audio settings
    int sample_rate = 16000;           // Expected sample rate

    // Chunking settings
    float min_chunk_duration = 0.5f;   // Minimum audio to process (seconds)
    float max_chunk_duration = 10.0f;  // Maximum chunk before forced processing
    float silence_threshold_duration = 0.5f;  // Silence duration to trigger endpoint

    // Context/overlap settings
    float context_duration = 0.5f;     // Audio context from previous chunk

    // Output settings
    bool emit_partials = true;         // Emit partial results during processing
    float partial_interval = 0.5f;     // Minimum interval between partials (seconds)

    // VAD settings (Voice Activity Detection using Silero VAD)
    bool use_vad = true;               // Use VAD for endpoint detection
    int vad_aggressiveness = 2;        // 0-3, higher = more aggressive filtering
    float vad_threshold = 0.5f;        // Speech detection threshold
    std::string vad_weights_path = "models/silero_vad/silero_vad_16k.bin";  // Silero VAD weights

    // LocalAgreement settings (reduces flickering partial results)
    bool use_local_agreement = true;   // Use LocalAgreement for stable partials
    int agreement_n = 2;               // Number of consecutive agreements needed

    // Model settings
    std::string language = "en";       // Language code
    std::string task = "transcribe";   // "transcribe" or "translate"

    // Latency mode: "fast", "balanced", "quality"
    std::string latency_mode = "balanced";
};

/**
 * Result from streaming transcription.
 */
struct StreamingResult {
    std::string text;                  // Transcribed text
    bool is_final = false;             // True if this is a stable final result
    bool is_partial = false;           // True if this is an unstable partial result
    float segment_start = 0.0f;        // Start time in audio stream (seconds)
    float segment_end = 0.0f;          // End time in audio stream (seconds)
    std::string language;              // Detected/specified language
    float confidence = 1.0f;           // Confidence estimate (0-1)

    // Timing metadata
    float processing_time = 0.0f;      // Time to transcribe this chunk (seconds)
    float audio_duration = 0.0f;       // Duration of audio transcribed (seconds)

    // LocalAgreement fields
    std::string confirmed_text;        // Text confirmed by LocalAgreement
    std::string speculative_text;      // Text not yet confirmed (may change)
    bool is_confirmed = false;         // True if this result includes new confirmed text

    // Real-time factor
    float rtf() const {
        return (audio_duration > 0) ? processing_time / audio_duration : 0.0f;
    }
};

/**
 * Streaming transcription state.
 */
enum class StreamState {
    IDLE,           // No speech detected, waiting
    SPEECH,         // Speech in progress
    PROCESSING,     // Processing accumulated audio
    FINALIZING      // Processing remaining audio at end
};

/**
 * Circular audio buffer for efficient streaming.
 */
class AudioBuffer {
public:
    AudioBuffer(float max_duration, int sample_rate);
    ~AudioBuffer() = default;

    /**
     * Append audio samples to buffer.
     * Wraps around if buffer is full.
     */
    void append(const float* samples, size_t count);
    void append(const std::vector<float>& samples);

    /**
     * Get the most recent audio of specified duration.
     */
    std::vector<float> get_audio(float duration) const;

    /**
     * Get all audio currently in buffer.
     */
    std::vector<float> get_all() const;

    /**
     * Current buffer duration in seconds.
     */
    float duration() const;

    /**
     * Total time of audio received since creation.
     */
    float total_time() const;

    /**
     * Clear the buffer.
     */
    void clear();

    /**
     * Get sample rate.
     */
    int sample_rate() const { return sample_rate_; }

    /**
     * Get current write position (samples in buffer).
     */
    size_t size() const { return write_pos_; }

private:
    std::vector<float> buffer_;
    size_t write_pos_ = 0;
    size_t total_samples_ = 0;
    int sample_rate_;
    size_t max_samples_;
};

/**
 * LocalAgreement policy for stable streaming output.
 *
 * Only outputs text when n consecutive transcriptions produce the same prefix.
 * This reduces "flickering" partial results while maintaining low latency.
 */
class LocalAgreement {
public:
    explicit LocalAgreement(int n = 2);

    /**
     * Update with new transcription, return newly confirmed text.
     */
    std::string update(const std::string& new_transcript);

    /**
     * Get all confirmed text so far.
     */
    std::string get_confirmed() const { return committed_; }

    /**
     * Get speculative (unconfirmed) text from latest transcription.
     */
    std::string get_speculative() const;

    /**
     * Reset for new segment.
     */
    void reset();

private:
    int n_;
    std::vector<std::string> history_;
    std::string committed_;

    std::string longest_common_prefix(const std::vector<std::string>& strings) const;
};

/**
 * Callback type for streaming results.
 */
using StreamingCallback = std::function<void(const StreamingResult&)>;

/**
 * Streaming speech-to-text transcriber.
 *
 * Features:
 * - VAD-based endpoint detection for natural sentence boundaries
 * - Configurable chunk duration limits
 * - Partial results during long utterances
 * - Context overlap for better accuracy at chunk boundaries
 * - Callback-based interface
 *
 * Example:
 *     auto model = WhisperModel::load("/path/to/model");
 *     StreamingConfig config;
 *     config.emit_partials = true;
 *
 *     StreamingTranscriber streamer(model, config);
 *     streamer.set_callback([](const StreamingResult& result) {
 *         if (result.is_final) {
 *             std::cout << "[FINAL] " << result.text << std::endl;
 *         } else {
 *             std::cout << "[PARTIAL] " << result.text << std::endl;
 *         }
 *     });
 *
 *     // Feed audio chunks (e.g., from microphone)
 *     while (has_audio()) {
 *         auto chunk = get_audio_chunk();
 *         streamer.process_audio(chunk.data(), chunk.size());
 *     }
 *
 *     // Finalize and get remaining transcription
 *     streamer.finalize();
 */
class StreamingTranscriber {
public:
    /**
     * Create streaming transcriber with model and config.
     */
    StreamingTranscriber(WhisperModel& model, const StreamingConfig& config = StreamingConfig());
    ~StreamingTranscriber();

    // Move semantics
    StreamingTranscriber(StreamingTranscriber&&) noexcept;
    StreamingTranscriber& operator=(StreamingTranscriber&&) noexcept;

    // Disable copy
    StreamingTranscriber(const StreamingTranscriber&) = delete;
    StreamingTranscriber& operator=(const StreamingTranscriber&) = delete;

    /**
     * Set callback for transcription results.
     */
    void set_callback(StreamingCallback callback);

    /**
     * Process audio samples.
     * Results are delivered via callback.
     *
     * @param samples Audio samples (float32, mono)
     * @param count Number of samples
     */
    void process_audio(const float* samples, size_t count);
    void process_audio(const std::vector<float>& samples);

    /**
     * Finalize and process any remaining audio.
     * Call this when the audio stream ends.
     */
    void finalize();

    /**
     * Reset for new streaming session.
     */
    void reset();

    /**
     * Get current state.
     */
    StreamState state() const { return state_; }

    /**
     * Get config.
     */
    const StreamingConfig& config() const { return config_; }

    /**
     * Get detected/specified language.
     */
    std::string language() const { return detected_language_; }

private:
    WhisperModel& model_;
    StreamingConfig config_;
    StreamingCallback callback_;

    // State
    StreamState state_ = StreamState::IDLE;
    std::unique_ptr<AudioBuffer> audio_buffer_;
    std::unique_ptr<AudioBuffer> speech_buffer_;
    std::unique_ptr<LocalAgreement> local_agreement_;

    // Timing
    float segment_start_time_ = 0.0f;
    float last_partial_time_ = 0.0f;
    int silence_frames_ = 0;
    int speech_frames_ = 0;

    // Context from previous chunk
    std::vector<float> context_audio_;

    // Language (cached after first detection)
    std::string detected_language_;

    // VAD state (Silero VAD neural network)
    std::unique_ptr<silero_vad::SileroVAD> silero_vad_;
    std::string silero_weights_path_;  // Path to Silero VAD weights

    // Internal methods
    void process_chunk(const float* samples, size_t count);
    void process_segment(bool is_final, bool is_partial = false);
    bool check_vad(const float* samples, size_t count);
    void init_vad();
};

}  // namespace whisper
