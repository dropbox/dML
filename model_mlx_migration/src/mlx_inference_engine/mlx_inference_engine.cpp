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

// MLX Inference Engine - Implementation
// Integrates all MLX models into a unified C++ interface

#include "mlx_inference_engine.hpp"

// Include Kokoro TTS
#include "../kokoro/kokoro.h"

// Include Translation Model
#include "translation_model.h"

// Include Whisper STT Model
#include "whisper_model.h"

// Include LLM Model
#include "llm_model.h"

// Include CosyVoice3 TTS Model
#include "cosyvoice3_model.h"

// Include Silero VAD for speech preprocessing
#include "silero_vad.h"

#include "mlx/mlx.h"

#include <sstream>
#include <fstream>
#include <mutex>
#include <iostream>
#include <regex>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace mx = mlx::core;

namespace mlx_inference {

// ============================================================================
// Implementation struct (PIMPL pattern)
// ============================================================================

struct MLXInferenceEngine::Impl {
    // TTS Models
    std::unique_ptr<kokoro::Model> kokoro_model;
    std::string kokoro_model_path;

    // Translation Model (T5/MADLAD/NLLB) - with integrated tokenizer
    std::unique_ptr<translation::TranslationModel> translation_model;
    std::string translation_model_path;
    std::string translation_model_type;

    // Whisper STT Model
    std::unique_ptr<whisper::WhisperModel> whisper_model;
    std::unique_ptr<whisper::WhisperTokenizer> whisper_tokenizer;
    std::string whisper_model_path;

    // Silero VAD for speech preprocessing (matches Python behavior)
    std::unique_ptr<silero_vad::SileroVAD> silero_vad;

    // LLM Model
    std::unique_ptr<llm::LLMModel> llm_model;
    std::string llm_model_path;

    // CosyVoice3 TTS Model
    std::unique_ptr<cosyvoice3::CosyVoice3Model> cosyvoice3_model;
    std::string cosyvoice3_model_path;

    // Thread safety (models are thread-safe internally, but loading is not)
    std::mutex load_mutex;

    Impl() = default;
    ~Impl() = default;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

MLXInferenceEngine::MLXInferenceEngine()
    : pimpl_(std::make_unique<Impl>()) {}

MLXInferenceEngine::~MLXInferenceEngine() = default;

MLXInferenceEngine::MLXInferenceEngine(MLXInferenceEngine&&) noexcept = default;
MLXInferenceEngine& MLXInferenceEngine::operator=(MLXInferenceEngine&&) noexcept = default;

// ============================================================================
// Model Loading
// ============================================================================

void MLXInferenceEngine::load_kokoro(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(pimpl_->load_mutex);
    pimpl_->kokoro_model = std::make_unique<kokoro::Model>(kokoro::Model::load(model_path));
    pimpl_->kokoro_model_path = model_path;
}

void MLXInferenceEngine::load_cosyvoice(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(pimpl_->load_mutex);

    // Load CosyVoice3 model from the given path
    // Expected structure:
    //   model_path/
    //     llm/model.safetensors (Qwen2 LLM weights)
    //     flow.safetensors (DiT flow weights)
    //     vocoder.safetensors or hift.safetensors (CausalHiFT vocoder weights)
    //     config.json (optional model config)
    pimpl_->cosyvoice3_model = std::make_unique<cosyvoice3::CosyVoice3Model>(
        cosyvoice3::CosyVoice3Model::load(model_path)
    );
    pimpl_->cosyvoice3_model_path = model_path;
}

void MLXInferenceEngine::load_translation(const std::string& model_path, const std::string& model_type) {
    std::lock_guard<std::mutex> lock(pimpl_->load_mutex);
    pimpl_->translation_model = std::make_unique<translation::TranslationModel>(
        translation::TranslationModel::load(model_path, model_type)
    );
    pimpl_->translation_model_path = model_path;
    pimpl_->translation_model_type = model_type;
}

void MLXInferenceEngine::load_whisper(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(pimpl_->load_mutex);

    // Load the Whisper model
    pimpl_->whisper_model = std::make_unique<whisper::WhisperModel>(
        whisper::WhisperModel::load(model_path)
    );

    // Load tokenizer from models/whisper_vocab.json
    // This file contains the Whisper multilingual tiktoken vocabulary
    try {
        // Try multiple locations for vocab file
        std::vector<std::string> vocab_paths = {
            "models/whisper_vocab.json",
            "../models/whisper_vocab.json",
            "../../models/whisper_vocab.json",  // From build/mlx_inference_engine/
            "./whisper_vocab.json",
            model_path + "/vocab.json"
        };

        bool loaded = false;
        for (const auto& vocab_path : vocab_paths) {
            std::ifstream f(vocab_path);
            if (f.good()) {
                f.close();
                pimpl_->whisper_tokenizer = std::make_unique<whisper::WhisperTokenizer>(
                    whisper::WhisperTokenizer::load(vocab_path)
                );
                loaded = true;
                break;
            }
        }

        if (!loaded) {
            std::cerr << "Warning: Could not find whisper_vocab.json, transcription will return token IDs\n";
            pimpl_->whisper_tokenizer = nullptr;
        }
    } catch (const std::exception& e) {
        // Tokenizer is optional - decoder output can be processed externally
        std::cerr << "Warning: Failed to load tokenizer: " << e.what() << "\n";
        pimpl_->whisper_tokenizer = nullptr;
    }

    // Load Silero VAD for speech preprocessing (matches Python behavior)
    // VAD preprocessing filters silence and speeds up transcription
    try {
        std::vector<std::string> vad_paths = {
            "models/silero_vad/silero_vad_16k.bin",
            "../models/silero_vad/silero_vad_16k.bin",
            "../../models/silero_vad/silero_vad_16k.bin",
            "./silero_vad_16k.bin",
        };

        bool vad_loaded = false;
        for (const auto& vad_path : vad_paths) {
            std::ifstream f(vad_path, std::ios::binary);
            if (f.good()) {
                f.close();
                pimpl_->silero_vad = std::make_unique<silero_vad::SileroVAD>(vad_path, 16000);
                vad_loaded = true;
                if (std::getenv("DEBUG_WHISPER")) {
                    std::cerr << "[DEBUG] Loaded Silero VAD from: " << vad_path << "\n";
                }
                break;
            }
        }

        if (!vad_loaded) {
            std::cerr << "Warning: Silero VAD weights not found, transcription will not use VAD preprocessing\n";
            pimpl_->silero_vad = nullptr;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load Silero VAD: " << e.what() << "\n";
        pimpl_->silero_vad = nullptr;
    }

    pimpl_->whisper_model_path = model_path;
}

void MLXInferenceEngine::load_llm(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(pimpl_->load_mutex);

    pimpl_->llm_model = std::make_unique<llm::LLMModel>(llm::LLMModel::load(model_path));
    pimpl_->llm_model_path = model_path;

    std::cout << "Loaded LLM from: " << model_path << "\n";
    std::cout << pimpl_->llm_model->info();
}

// ============================================================================
// TTS Inference
// ============================================================================

AudioOutput MLXInferenceEngine::synthesize(const std::string& text, const TTSConfig& config) {
    if (!pimpl_->kokoro_model) {
        throw ModelNotLoadedError("Kokoro TTS");
    }

    // Call Kokoro synthesize
    kokoro::AudioOutput kokoro_output = pimpl_->kokoro_model->synthesize(
        text,
        config.voice,
        config.speed
    );

    // Convert to our AudioOutput type
    AudioOutput output;
    output.samples = std::move(kokoro_output.samples);
    output.sample_rate = kokoro_output.sample_rate;
    output.duration_seconds = kokoro_output.duration_seconds;

    return output;
}

AudioOutput MLXInferenceEngine::synthesize_cosyvoice(const std::string& text, const TTSConfig& config) {
    if (!pimpl_->cosyvoice3_model) {
        throw ModelNotLoadedError("CosyVoice3 TTS");
    }

    // NOTE: CosyVoice3 requires:
    // 1. Text tokenization (Qwen2 tokenizer) - not yet implemented in C++
    // 2. LLM speech token generation - returns dummy tokens currently
    // 3. Speaker embedding - using random embedding for now
    //
    // For production use, pre-tokenize text using Python and use synthesize_tokens().
    // This function provides a basic interface for testing the vocoder pipeline.

    (void)config;  // TTSConfig not fully supported yet

    // Create dummy text token IDs based on text length
    // In production, this should use a proper Qwen2 tokenizer
    int text_len = static_cast<int>(text.length());
    int num_tokens = std::max(10, text_len / 4);  // Rough estimate: ~4 chars per token
    auto text_ids = mx::zeros({1, num_tokens}, mx::int32);

    // Create random speaker embedding (192-dim for CAM++ compatibility)
    // In production, this should come from speaker encoder or pre-computed
    auto speaker_emb = mx::random::normal({1, 192});
    speaker_emb = speaker_emb / mx::sqrt(mx::sum(speaker_emb * speaker_emb, -1, true));  // Normalize

    // Synthesize using the model pipeline (LLM -> Flow -> Vocoder)
    auto audio = pimpl_->cosyvoice3_model->synthesize(
        text_ids,
        speaker_emb,
        500,     // max_tokens
        1.0f,    // temperature
        25,      // top_k
        0.8f,    // top_p
        10,      // flow_steps
        0.7f     // cfg_strength
    );
    mx::eval(audio);

    // Convert mx::array to std::vector<float>
    std::vector<float> samples(audio.size());
    std::memcpy(samples.data(), audio.data<float>(), audio.size() * sizeof(float));

    AudioOutput output;
    output.samples = std::move(samples);
    output.sample_rate = 24000;  // CosyVoice3 outputs 24kHz
    output.duration_seconds = static_cast<float>(output.samples.size()) / 24000.0f;

    return output;
}

// ============================================================================
// Translation Inference
// ============================================================================

std::string MLXInferenceEngine::translate(const std::string& text, const TranslationConfig& config) {
    if (!pimpl_->translation_model) {
        throw ModelNotLoadedError("Translation");
    }

    // Use integrated SentencePiece tokenizer for full text-to-text translation
    return pimpl_->translation_model->translate(
        text,
        config.source_lang,
        config.target_lang,
        config.max_length,
        config.debug
    );
}

std::vector<int32_t> MLXInferenceEngine::translate_tokens(
    const std::vector<int32_t>& input_ids,
    int max_length
) {
    (void)input_ids;
    (void)max_length;

    // With SentencePiece tokenizer integrated, use translate() for text-to-text translation
    throw MLXInferenceError(
        "translate_tokens() deprecated. Use translate() for full text-to-text translation "
        "with integrated SentencePiece tokenizer."
    );
}

// ============================================================================
// STT Inference
// ============================================================================

TranscriptionResult MLXInferenceEngine::transcribe(const std::vector<float>& audio,
                                                    int sample_rate,
                                                    const TranscriptionConfig& config) {
    if (!pimpl_->whisper_model) {
        throw ModelNotLoadedError("Whisper STT");
    }

    // GAP 58: task="lang_id" mode - only perform language identification
    if (config.task == "lang_id") {
        TranscriptionResult result;

        // Prepare audio for language detection (just need first 30s)
        std::vector<float> lang_audio = audio;
        if (sample_rate != 16000) {
            float ratio = 16000.0f / sample_rate;
            size_t new_size = static_cast<size_t>(audio.size() * ratio);
            lang_audio.resize(new_size);
            for (size_t i = 0; i < new_size; ++i) {
                float src_idx = i / ratio;
                size_t idx0 = static_cast<size_t>(src_idx);
                size_t idx1 = std::min(idx0 + 1, audio.size() - 1);
                float frac = src_idx - idx0;
                lang_audio[i] = audio[idx0] * (1.0f - frac) + audio[idx1] * frac;
            }
        }

        // Limit to 30s for language detection
        constexpr size_t MAX_SAMPLES = 480000;  // 30s at 16kHz
        if (lang_audio.size() > MAX_SAMPLES) {
            lang_audio.resize(MAX_SAMPLES);
        }

        // Pad to 30s if needed
        if (lang_audio.size() < MAX_SAMPLES) {
            lang_audio.resize(MAX_SAMPLES, 0.0f);
        }

        // Compute mel spectrogram
        auto mel = whisper::audio::log_mel_spectrogram(
            lang_audio,
            pimpl_->whisper_model->config().n_mels,
            pimpl_->whisper_model->config().n_fft,
            pimpl_->whisper_model->config().hop_length
        );

        // Detect language
        auto lang_result = pimpl_->whisper_model->detect_language(mx::expand_dims(mel, 0));

        result.text = "";  // No transcription in lang_id mode
        result.language = lang_result.language;
        result.confidence = lang_result.probability;

        // Create segment with language probabilities info
        TranscriptionSegment seg;
        seg.start_time = 0.0f;
        seg.end_time = static_cast<float>(audio.size()) / sample_rate;
        seg.text = "Detected language: " + lang_result.language + " (" +
                   std::to_string(static_cast<int>(lang_result.probability * 100)) + "%)";
        seg.avg_logprob = std::log(lang_result.probability);
        result.segments.push_back(seg);

        if (std::getenv("DEBUG_WHISPER")) {
            std::cerr << "[DEBUG] GAP 58: task=lang_id - detected " << lang_result.language
                      << " with probability " << lang_result.probability << "\n";
        }

        return result;
    }

    // GAP 45: offset_ms and duration_ms - Process specific time range
    // GAP 8: clip_timestamps - Extract specified time ranges from audio
    // Note: clip_timestamps takes precedence if both are specified
    std::vector<float> clipped_audio = audio;

    if (!config.clip_timestamps.empty()) {
        // clip_timestamps format: [start1, end1, start2, end2, ...]
        // Must have even number of elements (start/end pairs)
        if (config.clip_timestamps.size() % 2 != 0) {
            throw MLXInferenceError("clip_timestamps must have even number of elements (start/end pairs)");
        }

        std::vector<float> extracted;
        for (size_t i = 0; i < config.clip_timestamps.size(); i += 2) {
            float start_sec = config.clip_timestamps[i];
            float end_sec = config.clip_timestamps[i + 1];

            // Convert to sample indices
            int start_sample = static_cast<int>(start_sec * sample_rate);
            int end_sample = static_cast<int>(end_sec * sample_rate);

            // Clamp to audio bounds
            start_sample = std::max(0, start_sample);
            end_sample = std::min(static_cast<int>(audio.size()), end_sample);

            if (start_sample < end_sample) {
                extracted.insert(extracted.end(),
                                 audio.begin() + start_sample,
                                 audio.begin() + end_sample);
            }
        }

        if (!extracted.empty()) {
            clipped_audio = std::move(extracted);
        }

        if (std::getenv("DEBUG_WHISPER")) {
            std::cerr << "[DEBUG] GAP 8: clip_timestamps extracted "
                      << clipped_audio.size() << " samples from "
                      << (config.clip_timestamps.size() / 2) << " ranges\n";
        }
    } else if (config.offset_ms > 0 || config.duration_ms > 0) {
        // GAP 45: offset_ms and duration_ms (only if clip_timestamps not specified)
        // Calculate sample indices from milliseconds
        int start_sample = (config.offset_ms * sample_rate) / 1000;
        int end_sample;

        if (config.duration_ms > 0) {
            // Process duration_ms from offset
            end_sample = start_sample + (config.duration_ms * sample_rate) / 1000;
        } else {
            // Process entire audio from offset
            end_sample = static_cast<int>(audio.size());
        }

        // Clamp to audio bounds
        start_sample = std::max(0, start_sample);
        end_sample = std::min(static_cast<int>(audio.size()), end_sample);

        if (start_sample < end_sample) {
            clipped_audio = std::vector<float>(audio.begin() + start_sample, audio.begin() + end_sample);
        } else {
            // Invalid range - results in empty audio
            if (std::getenv("DEBUG_WHISPER")) {
                std::cerr << "[DEBUG] GAP 45: offset_ms=" << config.offset_ms
                          << " duration_ms=" << config.duration_ms
                          << " results in empty range (audio length: " << audio.size() << " samples)\n";
            }
            clipped_audio.clear();
        }

        if (std::getenv("DEBUG_WHISPER")) {
            std::cerr << "[DEBUG] GAP 45: offset_ms=" << config.offset_ms
                      << " duration_ms=" << config.duration_ms
                      << " extracted " << clipped_audio.size() << " samples from " << audio.size() << "\n";
        }
    }

    // Resample audio to 16kHz if needed
    std::vector<float> resampled_audio = clipped_audio;
    if (sample_rate != 16000) {
        // Simple linear resampling (production should use proper resampling)
        float ratio = 16000.0f / sample_rate;
        size_t new_size = static_cast<size_t>(clipped_audio.size() * ratio);
        resampled_audio.resize(new_size);
        for (size_t i = 0; i < new_size; ++i) {
            float src_idx = i / ratio;
            size_t idx0 = static_cast<size_t>(src_idx);
            size_t idx1 = std::min(idx0 + 1, audio.size() - 1);
            float frac = src_idx - idx0;
            resampled_audio[i] = audio[idx0] * (1.0f - frac) + audio[idx1] * frac;
        }
    }

    // VAD preprocessing (matches Python WhisperMLX behavior)
    // This filters silence and concatenates speech segments to speed up transcription
    if (pimpl_->silero_vad && config.use_vad) {
        pimpl_->silero_vad->reset_state();

        // Get speech segments with default parameters matching Python:
        // threshold=0.5 (aggressiveness=2), min_speech=250ms, min_silence=300ms, speech_pad=30ms
        auto segments = pimpl_->silero_vad->get_speech_segments(
            resampled_audio.data(),
            resampled_audio.size(),
            0.5f,   // threshold (matches Python aggressiveness=2)
            250,    // min_speech_duration_ms
            300,    // min_silence_duration_ms
            30,     // speech_pad_ms (matches Python Silero default)
            config.max_speech_duration_s,  // GAP 51: max speech segment duration
            config.samples_overlap_s       // GAP 52: overlap for smoother segment transitions
        );

        if (std::getenv("DEBUG_WHISPER")) {
            float total_speech = 0.0f;
            for (const auto& seg : segments) {
                total_speech += seg.duration();
            }
            float speech_ratio = resampled_audio.size() > 0 ?
                total_speech / (resampled_audio.size() / 16000.0f) : 0.0f;
            std::cerr << "[DEBUG] VAD: " << (speech_ratio * 100) << "% speech, "
                      << segments.size() << " segments, "
                      << total_speech << "s / " << (resampled_audio.size() / 16000.0f) << "s\n";
            for (size_t i = 0; i < segments.size(); ++i) {
                std::cerr << "[DEBUG] VAD Seg " << i << ": "
                          << (segments[i].start_sample / 16000.0f) << "s - "
                          << (segments[i].end_sample / 16000.0f) << "s ("
                          << segments[i].duration() << "s)\n";
            }
        }

        // Check if audio is mostly silent (< 5% speech)
        float total_speech_duration = 0.0f;
        for (const auto& seg : segments) {
            total_speech_duration += seg.duration();
        }
        float speech_ratio = resampled_audio.size() > 0 ?
            total_speech_duration / (resampled_audio.size() / 16000.0f) : 0.0f;

        if (speech_ratio < 0.05f) {
            // Mostly silent - return empty result
            if (std::getenv("DEBUG_WHISPER")) {
                std::cerr << "[DEBUG] VAD: Audio is mostly silent, skipping transcription\n";
            }
            TranscriptionResult result;
            result.text = "";
            result.language = config.language.empty() ? "en" : config.language;
            result.confidence = 0.0f;
            return result;
        }

        // Extract and concatenate speech segments with 50ms padding
        if (!segments.empty()) {
            const int padding_samples = 50 * 16000 / 1000;  // 50ms padding
            std::vector<float> speech_audio;
            speech_audio.reserve(resampled_audio.size());  // Upper bound

            for (const auto& seg : segments) {
                int start = std::max(0, seg.start_sample - padding_samples);
                int end = std::min(static_cast<int>(resampled_audio.size()),
                                   seg.end_sample + padding_samples);
                speech_audio.insert(speech_audio.end(),
                                    resampled_audio.begin() + start,
                                    resampled_audio.begin() + end);
            }

            if (!speech_audio.empty()) {
                resampled_audio = std::move(speech_audio);
            }
        }
    }

    // Calculate actual audio duration BEFORE padding for timestamp suppression
    float audio_duration_sec = static_cast<float>(resampled_audio.size()) / 16000.0f;

    // PAD AUDIO BEFORE computing mel spectrogram
    // This is CRITICAL: Python pads audio first, then computes mel.
    // Python does: mel = log_mel_spectrogram(audio, n_mels, padding=N_SAMPLES)
    // This adds N_SAMPLES (30s) of silence AT THE END of the audio.
    // For short audio (<30s): pad to 30s total
    // For long audio (>=30s): add 30s padding at the end for seek loop headroom
    constexpr int N_SAMPLES_30S = 480000;  // 30 seconds at 16kHz
    std::vector<float> padded_audio;
    if (resampled_audio.size() < N_SAMPLES_30S) {
        // Pad with zeros to 30 seconds total
        padded_audio.resize(N_SAMPLES_30S, 0.0f);
        std::copy(resampled_audio.begin(), resampled_audio.end(), padded_audio.begin());
    } else {
        // Long audio: add 30s padding at the end (like Python's padding=N_SAMPLES)
        // This gives the seek loop room to process final segments
        padded_audio.resize(resampled_audio.size() + N_SAMPLES_30S, 0.0f);
        std::copy(resampled_audio.begin(), resampled_audio.end(), padded_audio.begin());
    }

    // Compute mel spectrogram from padded audio
    auto mel = whisper::audio::log_mel_spectrogram(
        padded_audio,
        pimpl_->whisper_model->config().n_mels,
        pimpl_->whisper_model->config().n_fft,
        pimpl_->whisper_model->config().hop_length
    );

    // Debug: Save mel spectrogram for comparison with Python
    if (std::getenv("DEBUG_ENCODER")) {
        mx::eval(mel);
        const float* mel_data = mel.data<float>();
        std::ofstream mel_file("/tmp/cpp_mel_0004.bin", std::ios::binary);
        mel_file.write(reinterpret_cast<const char*>(mel_data),
                       mel.size() * sizeof(float));
        mel_file.close();
        std::cerr << "[DEBUG] Saved mel to /tmp/cpp_mel_0004.bin, shape: "
                  << mel.shape()[0] << "x" << mel.shape()[1] << "\n";
        std::cerr << "[DEBUG] Mel first 5 values: ";
        for (int i = 0; i < 5; ++i) {
            std::cerr << mel_data[i] << " ";
        }
        std::cerr << "\n";
    }

    // Check audio length - use segment-based decoding for audio > 30 seconds
    // After VAD preprocessing, speech segments are concatenated into continuous audio.
    // Python WhisperMLX uses single-pass decoding for all VAD-preprocessed audio <= 30s:
    //   1. VAD extracts speech segments
    //   2. Concatenates to continuous speech (~26s for 29.4s file)
    //   3. Pads to 30s
    //   4. Single decode call generates full transcription
    // The segment-based seek loop is only for raw audio > 30s without VAD.
    constexpr float MAX_DURATION_FOR_SINGLE_PASS = 30.0f;  // 30 seconds threshold
    int mel_frames = static_cast<int>(mel.shape()[0]);

    TranscriptionResult result;
    std::vector<int> tokens;
    float avg_logprob = 0.0f;

    // GAP 43: Pre-compute tokens to suppress based on regex pattern
    // Tokens matching the regex will have logits set to -inf during decoding
    std::vector<int> suppress_regex_tokens;
    if (!config.suppress_regex.empty() && pimpl_->whisper_tokenizer && pimpl_->whisper_tokenizer->loaded()) {
        try {
            std::regex re(config.suppress_regex);
            const auto& id_to_token = pimpl_->whisper_tokenizer->get_id_to_token();
            for (const auto& [token_id, token_str] : id_to_token) {
                if (std::regex_match(token_str, re)) {
                    suppress_regex_tokens.push_back(token_id);
                }
            }
            if (!suppress_regex_tokens.empty()) {
                std::cerr << "[GAP 43] suppress_regex '" << config.suppress_regex
                          << "' matches " << suppress_regex_tokens.size() << " tokens\n";
            }
        } catch (const std::regex_error& e) {
            std::cerr << "[GAP 43] Invalid regex pattern '" << config.suppress_regex
                      << "': " << e.what() << "\n";
        }
    }

    // GAP 50: Apply alignment head preset if specified
    if (!config.dtw_aheads_preset.empty() && config.dtw_aheads_preset != "none") {
        whisper::AlignmentHeadsPreset preset = whisper::AlignmentHeadsPreset::NONE;
        if (config.dtw_aheads_preset == "tiny.en") preset = whisper::AlignmentHeadsPreset::TINY_EN;
        else if (config.dtw_aheads_preset == "tiny") preset = whisper::AlignmentHeadsPreset::TINY;
        else if (config.dtw_aheads_preset == "base.en") preset = whisper::AlignmentHeadsPreset::BASE_EN;
        else if (config.dtw_aheads_preset == "base") preset = whisper::AlignmentHeadsPreset::BASE;
        else if (config.dtw_aheads_preset == "small.en") preset = whisper::AlignmentHeadsPreset::SMALL_EN;
        else if (config.dtw_aheads_preset == "small") preset = whisper::AlignmentHeadsPreset::SMALL;
        else if (config.dtw_aheads_preset == "medium.en") preset = whisper::AlignmentHeadsPreset::MEDIUM_EN;
        else if (config.dtw_aheads_preset == "medium") preset = whisper::AlignmentHeadsPreset::MEDIUM;
        else if (config.dtw_aheads_preset == "large-v1") preset = whisper::AlignmentHeadsPreset::LARGE_V1;
        else if (config.dtw_aheads_preset == "large-v2") preset = whisper::AlignmentHeadsPreset::LARGE_V2;
        else if (config.dtw_aheads_preset == "large-v3") preset = whisper::AlignmentHeadsPreset::LARGE_V3;
        else if (config.dtw_aheads_preset == "large-v3-turbo") preset = whisper::AlignmentHeadsPreset::LARGE_V3_TURBO;

        if (preset != whisper::AlignmentHeadsPreset::NONE) {
            pimpl_->whisper_model->set_alignment_heads_preset(preset);
            std::cerr << "[GAP 50] Applied alignment head preset: " << config.dtw_aheads_preset << "\n";
        }
    }

    if (audio_duration_sec > MAX_DURATION_FOR_SINGLE_PASS) {
        // Use segment-based decoding for long audio
        // Pass actual audio duration to help generate_segments choose the right decoding path
        // GAP 3: Pass word_timestamps flag and tokenizer for word-level alignment
        auto segmented_result = pimpl_->whisper_model->generate_segments(
            mx::expand_dims(mel, 0),  // Add batch dimension
            config.language.empty() ? "en" : config.language,
            "transcribe",
            config.condition_on_previous_text,
            config.no_speech_threshold,
            config.compression_ratio_threshold,
            config.beam_size,
            config.length_penalty,
            config.word_timestamps,   // GAP 3: word_timestamps flag
            pimpl_->whisper_tokenizer.get(), // Tokenizer needed for word timestamps
            audio_duration_sec,  // actual_audio_duration - helps determine single-pass vs multi-segment
            config.initial_prompt,  // GAP 6: Pass initial_prompt for context conditioning
            config.carry_initial_prompt,  // GAP 48: Maintain initial_prompt across segments
            config.temperatures,  // GAP 4/12: Temperature fallback sequence
            config.logprob_threshold,  // GAP J: Logprob threshold for quality checking
            config.hallucination_silence_threshold,  // GAP 20: Skip silence around hallucinations
            config.clip_timestamps,  // GAP 8: Timestamp ranges to process
            config.entropy_threshold,  // GAP 47: Entropy threshold for fallback
            config.split_on_word,  // GAP 46: Split segments at word boundaries
            config.skip_logprobs,  // GAP M: Skip log probability calculation for speed
            config.prefix,         // GAP: Decoder prefix (prepended after SOT)
            suppress_regex_tokens,  // GAP 43: Pre-computed tokens matching regex
            config.token_timestamps,  // GAP 54: Token-level timestamps
            config.thold_pt,  // GAP 55: Timestamp probability threshold
            config.thold_ptsum,  // GAP 55: Timestamp sum probability threshold
            config.print_realtime,  // GAP 53: Print segments to stderr as generated
            config.tdrz_enable  // GAP 41: Tinydiarize speaker turn detection
        );

        // Convert WhisperSegments to TranscriptionSegments
        std::ostringstream full_text;
        float total_logprob = 0.0f;

        for (const auto& wseg : segmented_result.segments) {
            TranscriptionSegment seg;
            seg.start_time = wseg.start_time;
            seg.end_time = wseg.end_time;
            seg.tokens = wseg.tokens;
            seg.avg_logprob = wseg.avg_logprob;
            seg.no_speech_prob = wseg.no_speech_prob;

            // GAP 3: Copy word timestamps if available
            for (const auto& wword : wseg.words) {
                WordInfo word;
                word.word = wword.word;
                word.start_time = wword.start_time;
                word.end_time = wword.end_time;
                word.probability = wword.probability;
                seg.words.push_back(word);
            }

            // Decode segment text
            if (pimpl_->whisper_tokenizer && pimpl_->whisper_tokenizer->loaded()) {
                seg.text = pimpl_->whisper_tokenizer->decode(wseg.tokens);
            } else {
                std::ostringstream oss;
                for (int tok : wseg.tokens) {
                    oss << tok << " ";
                }
                seg.text = oss.str();
            }

            full_text << seg.text;
            result.segments.push_back(seg);
            tokens.insert(tokens.end(), wseg.tokens.begin(), wseg.tokens.end());
            total_logprob += wseg.avg_logprob;
        }

        result.text = full_text.str();
        result.language = segmented_result.language;
        result.confidence = segmented_result.segments.empty() ? 0.0f :
            total_logprob / static_cast<float>(segmented_result.segments.size());

        // Debug output
        if (std::getenv("DEBUG_WHISPER")) {
            std::cerr << "[DEBUG] Segment-based decoding: " << segmented_result.segments.size()
                      << " segments, " << segmented_result.total_duration << "s total\n";
            for (size_t i = 0; i < result.segments.size(); ++i) {
                std::cerr << "[DEBUG] Segment " << i << ": ["
                          << result.segments[i].start_time << "s - "
                          << result.segments[i].end_time << "s] '"
                          << result.segments[i].text.substr(0, 50) << "...'\n";
            }
        }
    } else {
        // =================================================================
        // GAP 12: Temperature fallback loop for single-pass decoding
        // Matches Python mlx-whisper decode_with_fallback behavior
        // =================================================================
        const auto& temperatures = config.temperatures;
        float used_temperature = temperatures.empty() ? 0.0f : temperatures[0];
        float compression_ratio = 1.0f;
        std::string decoded_text;

        // GAP 6: Tokenize initial_prompt if provided
        std::vector<int> prompt_tokens;
        const std::vector<int>* prompt_ptr = nullptr;
        if (!config.initial_prompt.empty() && pimpl_->whisper_tokenizer && pimpl_->whisper_tokenizer->loaded()) {
            // Python: tokenizer.encode(" " + initial_prompt.strip())
            std::string trimmed_prompt = config.initial_prompt;
            // Trim whitespace
            size_t start = trimmed_prompt.find_first_not_of(" \t\n\r");
            size_t end = trimmed_prompt.find_last_not_of(" \t\n\r");
            if (start != std::string::npos) {
                trimmed_prompt = trimmed_prompt.substr(start, end - start + 1);
            }
            prompt_tokens = pimpl_->whisper_tokenizer->encode(" " + trimmed_prompt);
            prompt_ptr = &prompt_tokens;

            if (std::getenv("DEBUG_WHISPER")) {
                std::cerr << "[DEBUG] GAP 6: initial_prompt tokenized to " << prompt_tokens.size() << " tokens\n";
            }
        }

        for (size_t temp_idx = 0; temp_idx < temperatures.size(); ++temp_idx) {
            float temp = temperatures[temp_idx];
            used_temperature = temp;

            if (config.beam_size > 1) {
                // Use beam search decoding
                // Calculate effective max_tokens based on sample_len and without_timestamps
                int beam_max_tokens = config.sample_len > 0 ? config.sample_len :
                                      (config.without_timestamps ? 448 : 448);

                auto beam_result = pimpl_->whisper_model->generate_beam(
                    mx::expand_dims(mel, 0),  // Add batch dimension
                    config.language.empty() ? "en" : config.language,
                    "transcribe",
                    config.beam_size,
                    config.length_penalty,
                    beam_max_tokens,  // max_tokens (respects sample_len if set)
                    prompt_ptr,  // GAP 6: initial_prompt tokens
                    1.0f,  // patience (GAP 9: default 1.0)
                    config.max_initial_timestamp,  // GAP: max_initial_timestamp
                    config.without_timestamps  // GAP: without_timestamps mode
                );
                tokens = beam_result.tokens;
                avg_logprob = beam_result.avg_logprob;

                // Debug: print beam search info
                if (std::getenv("DEBUG_WHISPER")) {
                    std::cerr << "[DEBUG] Temp " << temp << ": Beam search (k=" << config.beam_size << ") generated "
                              << tokens.size() << " tokens\n";
                    std::cerr << "[DEBUG] Avg logprob: " << avg_logprob
                              << ", normalized score: " << beam_result.normalized_score << "\n";
                }
            } else {
                // GAP 4/12: Temperature-based sampling
                // When temp == 0: greedy decoding (argmax)
                // When temp > 0: categorical sampling with logits/temp
                // GAP 5: best_of sampling - run multiple times at T>0, keep best avg_logprob
                int effective_best_of = (temp > 0.0f && config.best_of > 1) ? config.best_of : 1;

                std::vector<int> best_tokens;
                float best_avg_logprob = -std::numeric_limits<float>::infinity();

                for (int sample_idx = 0; sample_idx < effective_best_of; ++sample_idx) {
                    float sample_avg_logprob = 0.0f;
                    // Calculate effective max_tokens based on sample_len config
                    // Default: 224 with timestamps, 448 without timestamps
                    int effective_max_tokens = config.sample_len > 0 ? config.sample_len :
                                               (config.without_timestamps ? 448 : 224);

                    auto sample_tokens = pimpl_->whisper_model->generate(
                        mx::expand_dims(mel, 0),  // Add batch dimension
                        config.language.empty() ? "en" : config.language,
                        "transcribe",
                        effective_max_tokens,  // GAP: sample_len parameter
                        audio_duration_sec,  // Pass actual duration for timestamp suppression
                        &sample_avg_logprob,  // GAP 10: capture avg_logprob
                        nullptr,  // no_speech_prob_out (already captured at segment level)
                        prompt_ptr,  // GAP 6: initial_prompt tokens
                        temp,  // GAP 4/12: temperature for sampling
                        config.repetition_penalty,  // GAP K: repetition penalty
                        config.max_initial_timestamp,  // GAP: max_initial_timestamp
                        config.without_timestamps,  // GAP: without_timestamps mode
                        nullptr,  // prefix_tokens (TODO: implement decoder prefix)
                        config.sample_len,  // GAP: sample_len parameter
                        false,  // skip_logprobs
                        suppress_regex_tokens.empty() ? nullptr : &suppress_regex_tokens  // GAP 43
                    );

                    if (std::getenv("DEBUG_WHISPER") && effective_best_of > 1) {
                        std::cerr << "[DEBUG] GAP 5: best_of sample " << (sample_idx + 1)
                                  << "/" << effective_best_of << " avg_logprob=" << sample_avg_logprob << "\n";
                    }

                    // Keep sample with highest avg_logprob
                    if (sample_avg_logprob > best_avg_logprob) {
                        best_avg_logprob = sample_avg_logprob;
                        best_tokens = sample_tokens;
                    }
                }

                tokens = best_tokens;
                avg_logprob = best_avg_logprob;

                if (std::getenv("DEBUG_WHISPER") && effective_best_of > 1) {
                    std::cerr << "[DEBUG] GAP 5: selected best sample with avg_logprob=" << avg_logprob << "\n";
                }
            }

            // Debug: print generated tokens
            if (std::getenv("DEBUG_WHISPER")) {
                std::cerr << "[DEBUG] Temp " << temp << ": Generated " << tokens.size() << " tokens: ";
                for (int tok : tokens) {
                    std::cerr << tok << " ";
                }
                std::cerr << "\n";
            }

            // Decode tokens to text
            if (pimpl_->whisper_tokenizer && pimpl_->whisper_tokenizer->loaded()) {
                decoded_text = pimpl_->whisper_tokenizer->decode(tokens);
            } else {
                // Fallback: output token IDs (skip special tokens)
                std::ostringstream oss;
                for (size_t i = 3; i < tokens.size(); ++i) {  // Skip first 3 (SOT, lang, task)
                    int tok = tokens[i];
                    if (tok == pimpl_->whisper_model->config().eot_token) break;  // Stop at EOT
                    oss << tok << " ";
                }
                decoded_text = oss.str();
            }

            // Calculate compression ratio for quality check (GAP 16)
            compression_ratio = whisper::calculate_compression_ratio(decoded_text);

            // =============================================================
            // GAP 12: Quality threshold checks for fallback decision
            // =============================================================
            bool needs_fallback = false;

            // Check compression ratio (detects repetition/hallucination loops)
            if (compression_ratio > config.compression_ratio_threshold) {
                needs_fallback = true;
                if (std::getenv("DEBUG_WHISPER")) {
                    std::cerr << "[DEBUG] Temp " << temp << ": compression ratio "
                              << compression_ratio << " > " << config.compression_ratio_threshold
                              << " (retry)\n";
                }
            }

            // Check logprob threshold (detects low confidence output)
            if (!std::isnan(config.logprob_threshold) && avg_logprob < config.logprob_threshold) {
                // Only fail on logprob if we also have high no_speech probability
                // This avoids false positives on unusual but correct text
                // Note: no_speech_prob not tracked in single-pass, so just check logprob
                needs_fallback = true;
                if (std::getenv("DEBUG_WHISPER")) {
                    std::cerr << "[DEBUG] Temp " << temp << ": avg_logprob "
                              << avg_logprob << " < " << config.logprob_threshold
                              << " (retry)\n";
                }
            }

            // If quality is acceptable or this is the last temperature, stop
            if (!needs_fallback || temp_idx == temperatures.size() - 1) {
                if (std::getenv("DEBUG_WHISPER") && needs_fallback) {
                    std::cerr << "[DEBUG] Temp " << temp
                              << ": accepting despite quality issues (last fallback)\n";
                }
                break;
            }
        }

        result.text = decoded_text;
        result.language = config.language.empty() ? "en" : config.language;
        result.confidence = avg_logprob;

        // GAP 10 FIX: Create a single segment for single-pass decoding
        // This ensures avg_logprob is available in the output
        TranscriptionSegment seg;
        seg.start_time = 0.0f;
        seg.end_time = audio_duration_sec;
        seg.text = result.text;
        seg.tokens = tokens;
        seg.avg_logprob = avg_logprob;
        seg.no_speech_prob = 0.0f;  // Not tracked in single-pass

        // GAP 3 FIX: Add word timestamps for short audio (<30s)
        // Previously word_timestamps only worked for audio > 30s (segment-based decoding)
        if (config.word_timestamps && !tokens.empty() && pimpl_->whisper_tokenizer && pimpl_->whisper_tokenizer->loaded()) {
            // Create a WhisperSegment to use the add_word_timestamps function
            whisper::WhisperSegment wseg;
            wseg.start_time = 0.0f;
            wseg.end_time = audio_duration_sec;
            wseg.tokens = tokens;

            // Compute encoder output for DTW alignment
            auto encoder_output = pimpl_->whisper_model->encode(mel);
            mx::eval(encoder_output);

            // Add word-level timestamps using DTW alignment
            pimpl_->whisper_model->add_word_timestamps(
                wseg, encoder_output,
                config.language.empty() ? "en" : config.language,
                pimpl_->whisper_tokenizer.get(),
                audio_duration_sec,
                nullptr  // last_speech_timestamp
            );

            // Copy words to TranscriptionSegment
            for (const auto& wword : wseg.words) {
                WordInfo word;
                word.word = wword.word;
                word.start_time = wword.start_time;
                word.end_time = wword.end_time;
                word.probability = wword.probability;
                seg.words.push_back(word);
            }

            if (std::getenv("DEBUG_WHISPER")) {
                std::cerr << "[DEBUG] GAP 3: Added " << seg.words.size() << " word timestamps for short audio\n";
            }
        }

        result.segments.push_back(seg);
    }

    result.tokens = tokens;  // Store all raw tokens for debugging

    return result;
}

TranscriptionResult MLXInferenceEngine::transcribe_file(const std::string& audio_path,
                                                         const TranscriptionConfig& config) {
    if (!pimpl_->whisper_model) {
        throw ModelNotLoadedError("Whisper STT");
    }

    // Load audio file
    auto audio = whisper::audio::load_audio(audio_path);

    // Transcribe
    return transcribe(audio, 16000, config);
}

std::vector<TranscriptionResult> MLXInferenceEngine::transcribe_batch(
    const std::vector<std::string>& audio_paths,
    const TranscriptionConfig& config) {
    if (!pimpl_->whisper_model) {
        throw ModelNotLoadedError("Whisper STT");
    }

    // GAP: transcribe_batch - Process multiple audio files sequentially
    // Note: True batched processing would require model architecture changes
    // This implementation provides the API while processing sequentially
    std::vector<TranscriptionResult> results;
    results.reserve(audio_paths.size());

    for (const auto& path : audio_paths) {
        try {
            results.push_back(transcribe_file(path, config));
        } catch (const std::exception& e) {
            // On error, add empty result with error message
            TranscriptionResult error_result;
            error_result.text = "";
            error_result.language = "";
            error_result.confidence = 0.0f;

            TranscriptionSegment seg;
            seg.text = std::string("Error: ") + e.what();
            error_result.segments.push_back(seg);

            results.push_back(error_result);
        }
    }

    return results;
}

// ============================================================================
// LLM Inference
// ============================================================================

GenerationResult MLXInferenceEngine::generate(const std::string& prompt,
                                               const GenerationConfig& config) {
    if (!pimpl_->llm_model) {
        throw ModelNotLoadedError("LLM");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::string generated_text = pimpl_->llm_model->generate_text(
        prompt,
        config.max_tokens,
        config.temperature,
        config.top_p,
        config.top_k
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float>(end - start).count();

    GenerationResult result;
    result.text = generated_text;
    // Estimate tokens based on ~4 chars per token
    result.tokens_generated = static_cast<int>(generated_text.length() / 4);
    result.tokens_per_second = result.tokens_generated / duration;

    return result;
}

GenerationResult MLXInferenceEngine::chat(const std::string& user_message,
                                          const std::string& system_prompt,
                                          const GenerationConfig& config) {
    if (!pimpl_->llm_model) {
        throw ModelNotLoadedError("LLM");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::string generated_text = pimpl_->llm_model->chat(
        user_message,
        system_prompt,
        config.max_tokens,
        config.temperature,
        config.top_p,
        config.top_k
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float>(end - start).count();

    GenerationResult result;
    result.text = generated_text;
    // Estimate tokens based on ~4 chars per token
    result.tokens_generated = static_cast<int>(generated_text.length() / 4);
    result.tokens_per_second = result.tokens_generated / duration;

    return result;
}

// ============================================================================
// Utility
// ============================================================================

bool MLXInferenceEngine::is_kokoro_loaded() const {
    return pimpl_->kokoro_model != nullptr;
}

bool MLXInferenceEngine::is_cosyvoice_loaded() const {
    return pimpl_->cosyvoice3_model != nullptr && pimpl_->cosyvoice3_model->is_loaded();
}

bool MLXInferenceEngine::is_translation_loaded() const {
    return pimpl_->translation_model != nullptr;
}

bool MLXInferenceEngine::is_whisper_loaded() const {
    return pimpl_->whisper_model != nullptr;
}

bool MLXInferenceEngine::is_llm_loaded() const {
    return pimpl_->llm_model != nullptr && pimpl_->llm_model->loaded();
}

std::string MLXInferenceEngine::get_model_info() const {
    std::ostringstream oss;
    oss << "MLXInferenceEngine Status:\n";
    oss << "  Kokoro TTS: " << (is_kokoro_loaded() ? "LOADED" : "not loaded") << "\n";
    if (is_kokoro_loaded() && pimpl_->kokoro_model) {
        oss << "    Path: " << pimpl_->kokoro_model_path << "\n";
        oss << "    " << pimpl_->kokoro_model->info() << "\n";
    }
    oss << "  CosyVoice3 TTS: " << (is_cosyvoice_loaded() ? "LOADED" : "not loaded") << "\n";
    if (is_cosyvoice_loaded() && pimpl_->cosyvoice3_model) {
        oss << "    Path: " << pimpl_->cosyvoice3_model_path << "\n";
        oss << "    " << pimpl_->cosyvoice3_model->info();
    }
    oss << "  Translation: " << (is_translation_loaded() ? "LOADED" : "not loaded") << "\n";
    if (is_translation_loaded() && pimpl_->translation_model) {
        oss << "    Path: " << pimpl_->translation_model_path << "\n";
        oss << "    Type: " << pimpl_->translation_model_type << "\n";
        oss << "    " << pimpl_->translation_model->info() << "\n";
    }
    oss << "  Whisper STT: " << (is_whisper_loaded() ? "LOADED" : "not loaded") << "\n";
    if (is_whisper_loaded() && pimpl_->whisper_model) {
        oss << "    Path: " << pimpl_->whisper_model_path << "\n";
        oss << "    " << pimpl_->whisper_model->info() << "\n";
    }
    oss << "  LLM: " << (is_llm_loaded() ? "LOADED" : "not loaded") << "\n";
    if (is_llm_loaded() && pimpl_->llm_model) {
        oss << "    Path: " << pimpl_->llm_model_path << "\n";
        oss << "    " << pimpl_->llm_model->info();
    }
    return oss.str();
}

// ============================================================================
// Speech-to-Speech Pipeline
// ============================================================================

S2SResult MLXInferenceEngine::speech_to_speech(
    const std::vector<float>& audio,
    int sample_rate,
    const S2SConfig& config
) {
    // Check both models are loaded
    if (!pimpl_->whisper_model) {
        throw ModelNotLoadedError("Whisper STT (required for S2S pipeline)");
    }
    if (!pimpl_->kokoro_model) {
        throw ModelNotLoadedError("Kokoro TTS (required for S2S pipeline)");
    }

    S2SResult result;
    auto pipeline_start = std::chrono::high_resolution_clock::now();

    // Step 1: Speech-to-Text (Whisper)
    auto stt_start = std::chrono::high_resolution_clock::now();
    result.transcription = transcribe(audio, sample_rate, config.stt_config);
    auto stt_end = std::chrono::high_resolution_clock::now();
    result.stt_latency_ms = std::chrono::duration<double, std::milli>(stt_end - stt_start).count();

    // Step 2: Text-to-Speech (Kokoro)
    auto tts_start = std::chrono::high_resolution_clock::now();

    if (config.use_streaming) {
        // Streaming TTS: collect all chunks
        std::vector<float> all_samples;
        pimpl_->kokoro_model->synthesize_streaming(
            result.transcription.text,
            [&all_samples](const std::vector<float>& samples, int, bool) {
                all_samples.insert(all_samples.end(), samples.begin(), samples.end());
            },
            config.tts_config.voice,
            config.tts_config.speed
        );
        result.audio.samples = std::move(all_samples);
        result.audio.sample_rate = 24000;
        result.audio.duration_seconds = static_cast<float>(result.audio.samples.size()) / 24000.0f;
    } else {
        // Non-streaming TTS
        result.audio = synthesize(result.transcription.text, config.tts_config);
    }

    auto tts_end = std::chrono::high_resolution_clock::now();
    result.tts_latency_ms = std::chrono::duration<double, std::milli>(tts_end - tts_start).count();

    // Total pipeline time
    auto pipeline_end = std::chrono::high_resolution_clock::now();
    result.total_latency_ms = std::chrono::duration<double, std::milli>(pipeline_end - pipeline_start).count();

    return result;
}

S2SResult MLXInferenceEngine::speech_to_speech_file(
    const std::string& audio_path,
    const S2SConfig& config
) {
    // Load audio file
    auto audio = whisper::audio::load_audio(audio_path);

    // Run S2S pipeline
    return speech_to_speech(audio, 16000, config);
}

void MLXInferenceEngine::speech_to_speech_streaming(
    const std::vector<float>& audio,
    int sample_rate,
    S2SStreamingCallback callback,
    const S2SConfig& config
) {
    // Check both models are loaded
    if (!pimpl_->whisper_model) {
        throw ModelNotLoadedError("Whisper STT (required for S2S pipeline)");
    }
    if (!pimpl_->kokoro_model) {
        throw ModelNotLoadedError("Kokoro TTS (required for S2S pipeline)");
    }

    // Step 1: Speech-to-Text (Whisper)
    auto transcription = transcribe(audio, sample_rate, config.stt_config);

    // Step 2: Streaming TTS (Kokoro) - invokes callback for each chunk
    pimpl_->kokoro_model->synthesize_streaming(
        transcription.text,
        callback,
        config.tts_config.voice,
        config.tts_config.speed
    );
}

void MLXInferenceEngine::speech_to_speech_full_streaming(
    const std::vector<float>& audio,
    int sample_rate,
    S2SFullStreamingCallback callback,
    const S2SConfig& config
) {
    // Check both models are loaded
    if (!pimpl_->whisper_model) {
        throw ModelNotLoadedError("Whisper STT (required for S2S pipeline)");
    }
    if (!pimpl_->kokoro_model) {
        throw ModelNotLoadedError("Kokoro TTS (required for S2S pipeline)");
    }

    // For pre-recorded audio, use segment-based transcription to get sentence boundaries,
    // then stream TTS for each segment. This provides lower time-to-first-audio than
    // waiting for full transcription.

    auto pipeline_start = std::chrono::high_resolution_clock::now();

    // Step 1: Get full transcription with segments
    auto stt_start = std::chrono::high_resolution_clock::now();
    auto transcription = transcribe(audio, sample_rate, config.stt_config);
    auto stt_end = std::chrono::high_resolution_clock::now();
    double full_stt_ms = std::chrono::duration<double, std::milli>(stt_end - stt_start).count();

    // Step 2: Process each segment through TTS immediately
    int chunk_index = 0;

    if (transcription.segments.empty()) {
        // No segments - just use full text
        if (!transcription.text.empty()) {
            auto tts_start = std::chrono::high_resolution_clock::now();
            auto tts_output = pimpl_->kokoro_model->synthesize(
                transcription.text,
                config.tts_config.voice,
                config.tts_config.speed
            );
            auto tts_end = std::chrono::high_resolution_clock::now();

            S2SStreamingChunk chunk;
            chunk.text = transcription.text;
            chunk.audio_samples = std::move(tts_output.samples);
            chunk.chunk_index = chunk_index++;
            chunk.is_final = false;
            chunk.stt_latency_ms = full_stt_ms;
            chunk.tts_latency_ms = std::chrono::duration<double, std::milli>(tts_end - tts_start).count();
            callback(chunk);
        }
    } else {
        // Process each segment
        for (size_t i = 0; i < transcription.segments.size(); ++i) {
            const auto& seg = transcription.segments[i];
            if (seg.text.empty()) continue;

            auto tts_start = std::chrono::high_resolution_clock::now();
            auto tts_output = pimpl_->kokoro_model->synthesize(
                seg.text,
                config.tts_config.voice,
                config.tts_config.speed
            );
            auto tts_end = std::chrono::high_resolution_clock::now();

            S2SStreamingChunk chunk;
            chunk.text = seg.text;
            chunk.audio_samples = std::move(tts_output.samples);
            chunk.chunk_index = chunk_index++;
            chunk.is_final = false;
            chunk.stt_latency_ms = (i == 0) ? full_stt_ms : 0;  // Full STT time on first chunk
            chunk.tts_latency_ms = std::chrono::duration<double, std::milli>(tts_end - tts_start).count();
            callback(chunk);
        }
    }

    // Send final chunk to indicate completion
    S2SStreamingChunk final_chunk;
    final_chunk.text = "";
    final_chunk.chunk_index = chunk_index;
    final_chunk.is_final = true;
    final_chunk.stt_latency_ms = 0;
    final_chunk.tts_latency_ms = 0;
    callback(final_chunk);
}

} // namespace mlx_inference
