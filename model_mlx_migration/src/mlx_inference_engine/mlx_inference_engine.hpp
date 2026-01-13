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

// MLX Inference Engine - Unified C++ Runtime for All MLX Models
// Part of model_mlx_migration project
//
// Provides thread-safe parallel inference for:
// - TTS (Kokoro, CosyVoice)
// - Translation (NLLB, MADLAD, OPUS-MT)
// - STT (WhisperMLX)
// - LLM (LLaMA via mlx-lm)

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <stdexcept>
#include <functional>

namespace mlx_inference {

// Audio output structure
struct AudioOutput {
    std::vector<float> samples;
    int sample_rate;
    float duration_seconds;
};

// Word-level timing information (GAP 3: word timestamps)
struct WordInfo {
    std::string word;           // The word text
    float start_time;           // Word start in seconds
    float end_time;             // Word end in seconds
    float probability;          // Word probability (avg of token probs)
};

// A single transcription segment with timing
struct TranscriptionSegment {
    float start_time;           // Segment start in seconds
    float end_time;             // Segment end in seconds
    std::string text;           // Segment text
    std::vector<int> tokens;    // Tokens for this segment
    float avg_logprob;          // Average log probability (quality metric)
    float no_speech_prob;       // Probability of no speech
    std::vector<WordInfo> words; // Word-level timestamps (GAP 3)
};

// Transcription result structure
struct TranscriptionResult {
    std::string text;           // Full transcription text
    std::string language;       // Detected/specified language
    float confidence;           // Overall confidence (average logprob)
    std::vector<TranscriptionSegment> segments;  // Individual segments with timing
    std::vector<int> tokens;    // All raw token IDs for debugging
};

// Generation result structure
struct GenerationResult {
    std::string text;
    int tokens_generated;
    float tokens_per_second;
};

// TTS Configuration
struct TTSConfig {
    std::string voice = "af_heart";
    float speed = 1.0f;
    std::string emotion = "neutral";
    bool enable_prosody = true;
};

// Translation Configuration
struct TranslationConfig {
    std::string source_lang = "en";
    std::string target_lang = "de";
    int max_length = 512;
    bool use_quantized = true;  // Use 8-bit quantized model
    bool debug = false;  // Print debug output during generation
};

// Transcription Configuration
struct TranscriptionConfig {
    std::string model = "large-v3-turbo";
    std::string language = "";  // Empty for auto-detect
    bool enable_timestamps = false;

    // Temperature for sampling (GAP 4/12: supports tuple for fallback)
    // Single value: decode at that temperature only
    // Multiple values: try each on quality failure (compression_ratio, logprob)
    // Default matches Python: (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    std::vector<float> temperatures = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

    int beam_size = 1;  // 1 = greedy, >1 = beam search (e.g., 5)
    float length_penalty = 1.0f;  // Length penalty for beam search

    // GAP 5: best_of sampling - sample n times at each temperature, keep best
    // Only applies when temperature > 0 and beam_size == 1 (greedy with sampling)
    // Python default is 5 when using non-zero temperature
    // Temperature sampling now implemented (GAP 4/12)
    // TODO: Implement best_of loop: run generate() best_of times, keep best avg_logprob
    int best_of = 1;  // 1 = single sample, >1 = sample n times, pick best avg_logprob

    // VAD preprocessing (matches Python WhisperMLX behavior)
    // When enabled, Silero VAD filters silence before transcription
    bool use_vad = true;  // ALWAYS ON by default (matches Python)

    // Segment-based decoding options (for long audio >30s)
    bool condition_on_previous_text = true;  // Use previous segment as context
    float no_speech_threshold = 0.6f;  // Skip segments with high no-speech prob
    float compression_ratio_threshold = 2.4f;  // Detect degenerate outputs (GAP 16)

    // GAP 12: Quality thresholds for temperature fallback
    // If avg_logprob < logprob_threshold, retry with higher temperature
    // Set to NaN or very negative to disable
    float logprob_threshold = -1.0f;  // Default matches Python mlx-whisper

    // Initial prompt for context (vocabulary hints, proper nouns, etc.)
    std::string initial_prompt = "";

    // GAP 48: carry_initial_prompt - Maintain initial prompt across all segments
    // When true, initial_prompt is prepended to each segment's context
    // When false (default), only the first segment uses initial_prompt directly,
    // subsequent segments use condition_on_previous_text for context
    bool carry_initial_prompt = false;

    // GAP 3: Word-level timestamps via DTW alignment
    // When enabled, computes word boundaries using cross-attention weights
    // Note: More expensive (requires running decoder with attention capture)
    bool word_timestamps = false;

    // GAP K: Repetition penalty to discourage repeated tokens
    // Values > 1.0 discourage repetition, 1.0 = disabled
    // Applied as: logit = logit / penalty (for positive logits) or logit * penalty (for negative)
    // Helps prevent hallucination loops. Typical values: 1.0-1.5
    float repetition_penalty = 1.0f;  // 1.0 = disabled (default)

    // GAP 20: Hallucination silence threshold
    // If set (>0), skip silence before possible hallucinations that are surrounded by silence
    // When a segment has word-level timing info, skip words at the end that have:
    // - Start time >= segment_end - threshold
    // - Low probability (potential hallucination)
    // This helps avoid hallucinated text at the end of segments where audio is silent
    // Python default: None (disabled). Typical value: 2.0 seconds
    float hallucination_silence_threshold = 0.0f;  // 0.0 = disabled (default)

    // GAP 8: clip_timestamps - Timestamp ranges to transcribe
    // Format: list of start/end pairs in seconds: [start1, end1, start2, end2, ...]
    // Empty = transcribe entire audio (default)
    // Example: {0.0, 10.0, 20.0, 30.0} = transcribe 0-10s and 20-30s
    std::vector<float> clip_timestamps;

    // GAP 46: split_on_word - Split segments on word boundaries
    // When enabled, segment boundaries prefer word boundaries over token boundaries
    // Results in cleaner segments that don't split mid-word
    bool split_on_word = false;

    // GAP 47: entropy_thold - Entropy threshold for temperature fallback
    // If segment entropy > threshold, retry with higher temperature
    // Set to NaN or very high to disable. whisper.cpp uses 2.4
    float entropy_threshold = 2.4f;

    // GAP 51: max_speech_duration_s - Maximum speech segment duration (VAD)
    // Prevents runaway segments in VAD preprocessing
    // Set to 0 to disable (default). whisper.cpp default is 0.0 (disabled)
    float max_speech_duration_s = 0.0f;

    // GAP 58: task mode - "transcribe", "translate", or "lang_id"
    // "lang_id" mode only performs language identification, no transcription
    std::string task = "transcribe";

    // GAP M: skip_logprobs - Skip log probability calculation for speed
    // When true, avg_logprob will not be calculated (saves computation)
    bool skip_logprobs = false;

    // GAP: decoder prefix - Text prefix for decoder (after timestamps)
    // Unlike initial_prompt (which conditions on previous transcript),
    // prefix is prepended to the current transcript as if already spoken
    std::string prefix = "";

    // GAP: sample_len - Maximum number of tokens to sample per segment
    // Limits output length independently of max_tokens
    // Set to 0 to use default (224 for timestamps, 448 for no timestamps)
    int sample_len = 0;

    // GAP: without_timestamps - Disable timestamp prediction
    // Uses <|notimestamps|> token instead of timestamp tokens
    bool without_timestamps = false;

    // GAP: max_initial_timestamp - Maximum allowed initial timestamp
    // Prevents transcription from starting with late timestamps
    // Python default: 1.0 seconds
    float max_initial_timestamp = 1.0f;

    // GAP 45: offset_ms - Start offset in milliseconds
    // Skip first offset_ms milliseconds of audio before processing
    // Default: 0 (start from beginning)
    int offset_ms = 0;

    // GAP 45: duration_ms - Duration to process in milliseconds
    // Process only duration_ms of audio from offset_ms
    // Default: 0 (process entire audio from offset)
    int duration_ms = 0;

    // GAP 52: samples_overlap_s - Overlap in seconds for VAD segment extraction
    // Creates overlap between adjacent segments for smoother transitions
    // Each segment's boundaries are extended by half this value in each direction
    // 0 = disabled (default). Typical value: 0.25-0.5 seconds.
    float samples_overlap_s = 0.0f;

    // GAP 43: suppress_regex - Regular expression to suppress matching tokens
    // Tokens whose text matches this regex will have logits set to -inf
    // This is applied during decoding to constrain output vocabulary
    // Empty string = disabled (default)
    // Example: "[0-9]+" suppresses all numeric tokens
    // Example: "\\[.*\\]" suppresses bracketed content like [MUSIC], [LAUGHTER]
    std::string suppress_regex = "";

    // GAP 50: dtw_aheads_preset - Alignment head preset for DTW word timestamps
    // Overrides alignment heads from model weights with known-good presets
    // Values: "none" (default), "tiny.en", "tiny", "base.en", "base",
    //         "small.en", "small", "medium.en", "medium",
    //         "large-v1", "large-v2", "large-v3", "large-v3-turbo"
    // Empty string or "none" = auto-detect from model or use weights
    std::string dtw_aheads_preset = "";

    // GAP 54: token_timestamps - Enable token-level timestamps
    // When true, return timing for each token (not just words)
    // Independent of word_timestamps. Uses same DTW alignment infrastructure.
    bool token_timestamps = false;

    // GAP 55: thold_pt - Timestamp token probability threshold
    // Minimum probability for a timestamp token to be accepted
    // Set to 0.0 to disable (default). whisper.cpp uses 0.01.
    float thold_pt = 0.0f;

    // GAP 55: thold_ptsum - Timestamp token sum probability threshold
    // Minimum cumulative probability for timestamp acceptance
    // Set to 0.0 to disable (default). whisper.cpp uses 0.01.
    float thold_ptsum = 0.0f;

    // GAP 53: print_realtime - Print transcription results in real-time
    // When enabled, prints each segment to stderr as it's generated
    // Format: [start_time --> end_time] text
    // This is useful for streaming/live transcription feedback
    // Note: whisper.cpp recommends using callbacks instead, but this provides
    // a simple built-in option for command-line usage
    bool print_realtime = false;

    // GAP 41: tdrz_enable - Speaker turn detection (tinydiarize)
    // When enabled, allows the model to predict speaker turn markers
    // The model can output a special "solm" token (start of last message) to
    // indicate a speaker change. When this token is detected, the segment's
    // speaker_turn_next field is set to true.
    // Note: This requires a tinydiarize-trained model to be useful.
    // When disabled (default), the solm token is suppressed.
    bool tdrz_enable = false;
};

// Generation Configuration
struct GenerationConfig {
    int max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
};

// Speech-to-Speech configuration
struct S2SConfig {
    TranscriptionConfig stt_config;  // Whisper STT settings
    TTSConfig tts_config;            // Kokoro TTS settings
    bool use_streaming = false;      // Use streaming TTS for low latency
};

// Speech-to-Speech result
struct S2SResult {
    TranscriptionResult transcription;  // STT result (input text)
    AudioOutput audio;                   // TTS output (synthesized speech)
    double stt_latency_ms = 0.0;         // STT processing time
    double tts_latency_ms = 0.0;         // TTS processing time
    double total_latency_ms = 0.0;       // Total pipeline time
};

// Callback for streaming S2S
using S2SStreamingCallback = std::function<void(
    const std::vector<float>& samples,
    int chunk_index,
    bool is_final
)>;

// Callback for true streaming S2S (text + audio as they're generated)
struct S2SStreamingChunk {
    std::string text;                    // Transcribed text for this chunk
    std::vector<float> audio_samples;    // Synthesized audio for this text
    int chunk_index;                     // 0-based chunk index
    bool is_final;                       // True if this is the last chunk
    double stt_latency_ms;               // STT time for this chunk
    double tts_latency_ms;               // TTS time for this chunk
};

using S2SFullStreamingCallback = std::function<void(const S2SStreamingChunk&)>;

// Exception types
class MLXInferenceError : public std::runtime_error {
public:
    explicit MLXInferenceError(const std::string& msg) : std::runtime_error(msg) {}
};

class ModelNotLoadedError : public MLXInferenceError {
public:
    explicit ModelNotLoadedError(const std::string& model)
        : MLXInferenceError("Model not loaded: " + model) {}
};

// Forward declarations for implementation classes
class KokoroModel;
class CosyVoiceModel;
class TranslationModel;
class WhisperModel;
class LLMModel;

// Main inference engine class
class MLXInferenceEngine {
public:
    MLXInferenceEngine();
    ~MLXInferenceEngine();

    // Disable copy (models hold GPU resources)
    MLXInferenceEngine(const MLXInferenceEngine&) = delete;
    MLXInferenceEngine& operator=(const MLXInferenceEngine&) = delete;

    // Allow move
    MLXInferenceEngine(MLXInferenceEngine&&) noexcept;
    MLXInferenceEngine& operator=(MLXInferenceEngine&&) noexcept;

    // =========================================================================
    // Model Loading
    // =========================================================================

    // Load Kokoro TTS model
    // model_path: Path to kokoro_cpp_export/ directory with safetensors
    void load_kokoro(const std::string& model_path);

    // Load CosyVoice TTS model (placeholder - not yet implemented)
    void load_cosyvoice(const std::string& model_path);

    // Load translation model (MADLAD/NLLB/OPUS-MT)
    // model_path: Path to model directory with safetensors
    // model_type: "madlad" | "nllb" | "opus-mt"
    void load_translation(const std::string& model_path, const std::string& model_type = "madlad");

    // Load Whisper STT model
    // model_name: "large-v3-turbo" | "large-v3" | etc.
    void load_whisper(const std::string& model_name);

    // Load LLM model (via mlx-lm patterns)
    void load_llm(const std::string& model_path);

    // =========================================================================
    // Inference - TTS
    // =========================================================================

    // Synthesize speech from text using Kokoro
    // Returns audio samples at 24kHz sample rate
    AudioOutput synthesize(const std::string& text, const TTSConfig& config = TTSConfig());

    // Synthesize with CosyVoice (placeholder)
    AudioOutput synthesize_cosyvoice(const std::string& text, const TTSConfig& config = TTSConfig());

    // =========================================================================
    // Inference - Translation
    // =========================================================================

    // Translate text between languages (requires tokenizer - use translate_tokens for now)
    std::string translate(const std::string& text, const TranslationConfig& config = TranslationConfig());

    // Translate from pre-tokenized input (use Python tokenizer externally)
    // Returns generated token IDs (use Python tokenizer to decode)
    std::vector<int32_t> translate_tokens(const std::vector<int32_t>& input_ids, int max_length = 256);

    // =========================================================================
    // Inference - STT
    // =========================================================================

    // Transcribe audio to text
    TranscriptionResult transcribe(const std::vector<float>& audio,
                                   int sample_rate,
                                   const TranscriptionConfig& config = TranscriptionConfig());

    // Transcribe from file path
    TranscriptionResult transcribe_file(const std::string& audio_path,
                                        const TranscriptionConfig& config = TranscriptionConfig());

    // Transcribe multiple audio files in batch
    // GAP: transcribe_batch - parallel processing of multiple audio files
    // Returns results in the same order as input paths
    std::vector<TranscriptionResult> transcribe_batch(
        const std::vector<std::string>& audio_paths,
        const TranscriptionConfig& config = TranscriptionConfig());

    // =========================================================================
    // Inference - LLM
    // =========================================================================

    // Generate text completion
    GenerationResult generate(const std::string& prompt,
                              const GenerationConfig& config = GenerationConfig());

    // Chat with the model using LLaMA 3 Instruct template
    // @param user_message User's message
    // @param system_prompt Optional system prompt
    // @param config Generation configuration
    // @return Generated assistant response
    GenerationResult chat(const std::string& user_message,
                          const std::string& system_prompt = "",
                          const GenerationConfig& config = GenerationConfig());

    // =========================================================================
    // Speech-to-Speech Pipeline
    // =========================================================================

    // Unified speech-to-speech pipeline
    // Takes audio input, transcribes it, synthesizes the transcription
    // Requires both Whisper and Kokoro models to be loaded
    S2SResult speech_to_speech(const std::vector<float>& audio,
                                int sample_rate,
                                const S2SConfig& config = S2SConfig());

    // Speech-to-speech from audio file
    S2SResult speech_to_speech_file(const std::string& audio_path,
                                     const S2SConfig& config = S2SConfig());

    // Streaming speech-to-speech with callback
    // Callback receives audio chunks as they're generated
    void speech_to_speech_streaming(const std::vector<float>& audio,
                                     int sample_rate,
                                     S2SStreamingCallback callback,
                                     const S2SConfig& config = S2SConfig());

    // True streaming S2S: uses streaming STT piped to streaming TTS
    // Each recognized sentence is immediately synthesized and delivered
    // This provides lowest possible time-to-first-audio
    void speech_to_speech_full_streaming(const std::vector<float>& audio,
                                          int sample_rate,
                                          S2SFullStreamingCallback callback,
                                          const S2SConfig& config = S2SConfig());

    // =========================================================================
    // Utility
    // =========================================================================

    // Check which models are loaded
    bool is_kokoro_loaded() const;
    bool is_cosyvoice_loaded() const;
    bool is_translation_loaded() const;
    bool is_whisper_loaded() const;
    bool is_llm_loaded() const;

    // Get info about loaded models
    std::string get_model_info() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace mlx_inference
