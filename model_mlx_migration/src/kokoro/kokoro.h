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

#pragma once

/**
 * Kokoro TTS C++ API
 *
 * This header defines the public API for the Kokoro TTS C++ runtime.
 * The implementation requires MLX C++ library.
 *
 * Build requirements:
 * - MLX C++ (https://github.com/ml-explore/mlx) - build from source
 * - espeak-ng library (for G2P)
 * - CMake 3.25+
 *
 * Usage:
 *   kokoro::Model model = kokoro::Model::load("path/to/weights");
 *   auto audio = model.synthesize("Hello world", "af_bella");
 */

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

namespace kokoro {

/**
 * Callback for streaming audio synthesis.
 * Called with each audio chunk as it's generated.
 *
 * @param samples Audio samples (float32, 24kHz)
 * @param chunk_index 0-based index of this chunk
 * @param is_final True if this is the last chunk
 */
using StreamingCallback = std::function<void(
    const std::vector<float>& samples,
    int chunk_index,
    bool is_final
)>;

// Forward declarations
class G2P;
class Tokenizer;

/**
 * Audio output from synthesis.
 * Contains raw float32 samples at 24kHz sample rate.
 */
struct AudioOutput {
    std::vector<float> samples;    // Audio samples, float32
    int sample_rate = 24000;       // Sample rate in Hz
    float duration_seconds = 0.0f; // Audio duration

    // Convenience methods
    size_t num_samples() const { return samples.size(); }
    bool empty() const { return samples.empty(); }
};

/**
 * Kokoro TTS Model.
 *
 * Thread safety: Multiple threads can call synthesize() concurrently
 * on the same Model instance. Each call is independent.
 */
class Model {
public:
    /**
     * Load model from exported C++ artifacts.
     *
     * Expected directory structure:
     *   weights.safetensors  - All model weights (WeightNorm pre-folded)
     *   config.json          - Model configuration
     *   vocab/phonemes.json  - Phoneme vocabulary
     *   voices/<name>.safetensors - Voice embeddings
     *
     * @param model_path Path to model directory
     * @return Loaded model
     * @throws std::runtime_error if loading fails
     */
    static Model load(const std::string& model_path);

    /**
     * Synthesize speech from text.
     *
     * Full pipeline: text → G2P → tokens → model → audio
     *
     * @param text Input text (UTF-8)
     * @param voice Voice name (default: "af_bella")
     * @param speed Speaking rate (1.0 = normal, <1 = slower, >1 = faster)
     * @return Generated audio
     */
    AudioOutput synthesize(
        const std::string& text,
        const std::string& voice = "af_bella",
        float speed = 1.0f
    ) const;

    /**
     * Synthesize from pre-tokenized phonemes.
     *
     * Use this when you've already done G2P and tokenization externally.
     *
     * @param token_ids Token IDs including BOS/EOS
     * @param voice Voice name
     * @param speed Speaking rate
     * @return Generated audio
     */
    AudioOutput synthesize_tokens(
        const std::vector<int32_t>& token_ids,
        const std::string& voice = "af_bella",
        float speed = 1.0f
    ) const;

    /**
     * Streaming synthesis - generates audio in chunks for low latency.
     *
     * Splits text into sentences and synthesizes each independently,
     * invoking the callback as soon as each chunk is ready. This enables
     * audio playback to begin before the entire text is processed.
     *
     * Time-to-first-audio is typically <200ms for short sentences.
     *
     * @param text Input text (UTF-8)
     * @param callback Called for each audio chunk
     * @param voice Voice name (default: "af_bella")
     * @param speed Speaking rate (1.0 = normal)
     */
    void synthesize_streaming(
        const std::string& text,
        StreamingCallback callback,
        const std::string& voice = "af_bella",
        float speed = 1.0f
    ) const;

    /**
     * Get available voice names.
     */
    std::vector<std::string> available_voices() const;

    /**
     * Check if a voice is available.
     */
    bool has_voice(const std::string& voice) const;

    /**
     * Set current voice for subsequent synthesize calls.
     * Voice is cached in memory - no file I/O on switch.
     * Also auto-sets language based on voice prefix (e.g., "af_" -> American English).
     */
    void set_voice(const std::string& voice);

    /**
     * Get currently active voice.
     */
    std::string current_voice() const;

    /**
     * Get available language codes.
     * Returns: {"a", "b", "j", "z", "e", "f", "h", "i", "p"}
     */
    std::vector<std::string> available_languages() const;

    /**
     * Check if a language is available.
     */
    bool has_language(const std::string& lang_code) const;

    /**
     * Set current language for G2P.
     * Only needed if you want to override auto-detection from voice prefix.
     */
    void set_language(const std::string& lang_code);

    /**
     * Get currently active language code.
     */
    std::string current_language() const;

    /**
     * Get voices available for a specific language.
     * @param lang_code One of: a, b, j, z, e, f, h, i, p
     */
    std::vector<std::string> voices_for_language(const std::string& lang_code) const;

    /**
     * Get model info for debugging.
     */
    std::string info() const;

    // Default constructor creates unloaded model
    Model();
    ~Model();

    // Move-only (no copy)
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Frame bucket sizes for shape-stable compilation.
 * Audio duration at 24kHz / 256 hop_length.
 * ~94 frames per second of audio.
 * Buckets: ~1s, ~2s, ~3s, ~4s, ~6s, ~8s, ~12s, ~16s
 */
constexpr int FRAME_BUCKETS[] = {100, 200, 300, 400, 600, 800, 1200, 1600};
constexpr int NUM_FRAME_BUCKETS = 8;

/**
 * Select appropriate frame bucket for given frame count.
 * Returns the smallest bucket that can fit the requested frames.
 */
inline int select_frame_bucket(int actual_frames) {
    for (int bucket : FRAME_BUCKETS) {
        if (actual_frames <= bucket) {
            return bucket;
        }
    }
    // For very long audio, return next multiple of largest bucket
    int largest = FRAME_BUCKETS[NUM_FRAME_BUCKETS - 1];
    return ((actual_frames + largest - 1) / largest) * largest;
}

/**
 * Supported languages and their espeak-ng codes.
 * Voice naming: {lang_code}{gender}_{name} (e.g., af_bella = American Female Bella)
 */
struct LanguageInfo {
    const char* code;      // Single letter: a, b, j, z, e, f, h, i, p
    const char* espeak;    // espeak-ng voice code
    const char* name;      // Human-readable name
};

constexpr LanguageInfo LANGUAGES[] = {
    {"a", "en-us", "American English"},
    {"b", "gmw/en", "British English"},  // espeak-ng uses file path, not en-gb
    {"j", "ja", "Japanese"},
    {"z", "cmn", "Mandarin Chinese"},
    {"e", "es", "Spanish"},
    {"f", "fr", "French"},
    {"h", "hi", "Hindi"},
    {"i", "it", "Italian"},
    {"p", "pt-br", "Brazilian Portuguese"}
};
constexpr int NUM_LANGUAGES = 9;

/**
 * Total voices: 54 across 9 languages
 * American English (a): 20 voices (11F + 9M)
 * British English (b): 8 voices (4F + 4M)
 * Japanese (j): 5 voices (4F + 1M)
 * Mandarin Chinese (z): 8 voices (4F + 4M)
 * Spanish (e): 3 voices (1F + 2M)
 * French (f): 1 voice (1F)
 * Hindi (h): 4 voices (2F + 2M)
 * Italian (i): 2 voices (1F + 1M)
 * Brazilian Portuguese (p): 3 voices (1F + 2M)
 */
constexpr int TOTAL_VOICES = 54;

}  // namespace kokoro
