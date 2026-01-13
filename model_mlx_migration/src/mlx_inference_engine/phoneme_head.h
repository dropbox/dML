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

// Kokoro Phoneme Head - CTC phoneme prediction from Whisper encoder output
//
// Architecture:
//   Whisper Encoder Output (1280-dim)
//       -> LayerNorm
//       -> Linear (1280 -> 512) + GELU
//       -> Linear (512 -> 200)
//       -> CTC decode -> IPA phoneme sequence
//
// Used for:
//   1. Transcript verification (compare predicted vs expected phonemes)
//   2. Hallucination detection (phonemes don't match text)
//   3. Pronunciation analysis
//   4. Streaming commit/wait decisions

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace phoneme {

/**
 * Configuration for Kokoro Phoneme Head.
 */
struct PhonemeHeadConfig {
    int d_model = 1280;           // Whisper encoder dimension
    int phoneme_vocab = 200;      // IPA phoneme vocabulary size
    int hidden_dim = 512;         // Hidden layer dimension
    int blank_id = 0;             // CTC blank token
    bool use_layer_norm = true;   // Apply layer norm before projection

    // Load config from JSON file
    static PhonemeHeadConfig load(const std::string& path);
};

/**
 * Phoneme verification result.
 */
struct PhonemeVerificationResult {
    float confidence;             // 0.0-1.0 confidence score
    int edit_distance;            // Edit distance between sequences
    std::vector<int> predicted;   // Predicted phoneme token IDs
    std::vector<int> expected;    // Expected phoneme token IDs (from text)
};

/**
 * Commit decision for streaming ASR.
 */
enum class CommitStatus {
    COMMIT,     // High confidence - finalize text
    PARTIAL,    // Medium confidence - show but may change
    WAIT,       // Low confidence - need more context
    FINAL       // End of utterance - forced commit
};

/**
 * Kokoro Phoneme Head for CTC phoneme prediction.
 *
 * Lightweight CTC head (~750K params) trained to predict IPA phonemes
 * from frozen Whisper encoder output. Used for transcript verification
 * and streaming commit decisions.
 */
class PhonemeHead {
public:
    PhonemeHead();
    ~PhonemeHead() = default;

    // Move semantics
    PhonemeHead(PhonemeHead&&) noexcept;
    PhonemeHead& operator=(PhonemeHead&&) noexcept;

    // Disable copy (weights are large)
    PhonemeHead(const PhonemeHead&) = delete;
    PhonemeHead& operator=(const PhonemeHead&) = delete;

    /**
     * Load phoneme head from directory.
     * Expected files: weights.npz, config.json
     */
    static PhonemeHead load(const std::string& model_path);

    /**
     * Check if model is loaded.
     */
    bool loaded() const { return loaded_; }

    /**
     * Forward pass - compute logits from encoder output.
     * @param encoder_output [batch, seq_len, d_model]
     * @return logits [batch, seq_len, phoneme_vocab]
     */
    mx::array forward(const mx::array& encoder_output);

    /**
     * Predict phoneme sequence using CTC greedy decode.
     * @param encoder_output [batch, seq_len, d_model]
     * @return Vector of phoneme token IDs (blanks removed, collapsed)
     */
    std::vector<int> predict(const mx::array& encoder_output);

    /**
     * Compare predicted phonemes with text-derived phonemes.
     * @param encoder_output Whisper encoder output
     * @param text Transcript to verify
     * @param language Language code for phonemization
     * @return PhonemeVerificationResult with confidence and edit distance
     */
    PhonemeVerificationResult compare_with_text(
        const mx::array& encoder_output,
        const std::string& text,
        const std::string& language = "en"
    );

    /**
     * Get commit status based on phoneme confidence.
     * @param confidence Phoneme verification confidence (0-1)
     * @param commit_threshold Threshold for COMMIT (default 0.75)
     * @param wait_threshold Threshold for WAIT (default 0.50)
     * @return CommitStatus decision
     */
    static CommitStatus get_commit_status(
        float confidence,
        float commit_threshold = 0.75f,
        float wait_threshold = 0.50f
    );

    /**
     * Get configuration.
     */
    const PhonemeHeadConfig& config() const { return config_; }

private:
    PhonemeHeadConfig config_;
    bool loaded_ = false;

    // Weights
    mx::array ln_weight_;      // LayerNorm weight [d_model]
    mx::array ln_bias_;        // LayerNorm bias [d_model]
    mx::array hidden_weight_;  // Hidden layer weight [hidden_dim, d_model]
    mx::array hidden_bias_;    // Hidden layer bias [hidden_dim]
    mx::array proj_weight_;    // Output projection weight [phoneme_vocab, hidden_dim]
    mx::array proj_bias_;      // Output projection bias [phoneme_vocab]
};

// ============================================================================
// Optimized CTC Decode (pure C++, no MLX overhead)
// ============================================================================

/**
 * CTC greedy decode - optimized pure C++ implementation.
 *
 * Performs argmax over vocabulary dimension, then collapses repeated
 * tokens and removes blanks. This is faster than doing it in MLX
 * because it's a simple sequential operation.
 *
 * @param logits Raw float pointer to logits [seq_len, vocab_size]
 * @param seq_len Sequence length (time frames)
 * @param vocab_size Vocabulary size
 * @param blank_id Blank token ID (default 0)
 * @return Vector of decoded token IDs
 */
std::vector<int> ctc_greedy_decode(
    const float* logits,
    int seq_len,
    int vocab_size,
    int blank_id = 0
);

/**
 * CTC greedy decode from MLX array.
 * Convenience wrapper that extracts data pointer.
 */
std::vector<int> ctc_greedy_decode(
    const mx::array& logits,
    int blank_id = 0
);

// ============================================================================
// Optimized Edit Distance (SIMD-accelerated)
// ============================================================================

/**
 * Compute edit distance (Levenshtein distance) between two sequences.
 *
 * Uses SIMD acceleration via Accelerate framework on Apple Silicon.
 * Falls back to standard DP implementation on other platforms.
 *
 * @param seq1 First sequence
 * @param seq2 Second sequence
 * @return Edit distance (number of insertions, deletions, substitutions)
 */
int edit_distance(
    const std::vector<int>& seq1,
    const std::vector<int>& seq2
);

/**
 * Compute edit distance with alignment info.
 *
 * @param seq1 First sequence (reference)
 * @param seq2 Second sequence (hypothesis)
 * @param insertions Output: number of insertions
 * @param deletions Output: number of deletions
 * @param substitutions Output: number of substitutions
 * @return Total edit distance
 */
int edit_distance_with_counts(
    const std::vector<int>& seq1,
    const std::vector<int>& seq2,
    int& insertions,
    int& deletions,
    int& substitutions
);

/**
 * Compute normalized edit distance (0-1 range).
 * @return 1.0 - (edit_distance / max(len1, len2))
 */
float normalized_edit_distance(
    const std::vector<int>& seq1,
    const std::vector<int>& seq2
);

// ============================================================================
// Phoneme Vocabulary
// ============================================================================

/**
 * Phoneme vocabulary for IPA conversion.
 */
class PhonemeVocab {
public:
    PhonemeVocab() = default;

    /**
     * Load vocabulary from JSON file.
     * Format: {"phoneme": token_id, ...}
     */
    static PhonemeVocab load(const std::string& path);

    /**
     * Get token ID for phoneme character.
     */
    int get_id(const std::string& phoneme) const;

    /**
     * Get phoneme character for token ID.
     */
    std::string get_phoneme(int id) const;

    /**
     * Convert token IDs to IPA string.
     */
    std::string ids_to_ipa(const std::vector<int>& ids) const;

    /**
     * Get vocabulary size.
     */
    size_t size() const { return phoneme_to_id_.size(); }

private:
    std::unordered_map<std::string, int> phoneme_to_id_;
    std::unordered_map<int, std::string> id_to_phoneme_;
};

// ============================================================================
// Phonemizer Interface
// ============================================================================

/**
 * Text to phoneme conversion.
 * Uses Misaki G2P for IPA phonemization.
 */
std::vector<int> phonemize_text(
    const std::string& text,
    const std::string& language = "en"
);

/**
 * Get IPA string for text.
 */
std::string get_ipa(
    const std::string& text,
    const std::string& language = "en"
);

}  // namespace phoneme
