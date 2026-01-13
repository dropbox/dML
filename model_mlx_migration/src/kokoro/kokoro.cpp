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

// Kokoro C++ Runtime Implementation
// Full text-to-audio pipeline: G2P -> Tokenizer -> Model -> Audio
// With Phase A prosody annotation support

#include "kokoro.h"
#include "g2p.h"
#include "tokenizer.h"
#include "model.h"
#include "prosody_parser.h"
#include "prosody_adjust.h"

#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

namespace kokoro {

// Forward declaration for helper function
static std::vector<float> insert_breaks_audio(
    const std::vector<float>& audio,
    int sample_rate,
    const ParsedProsody& parsed,
    size_t text_length
);

// Helper to get espeak voice code from Kokoro language prefix
static const char* get_espeak_voice(char lang_prefix) {
    for (int i = 0; i < NUM_LANGUAGES; i++) {
        if (LANGUAGES[i].code[0] == lang_prefix) {
            return LANGUAGES[i].espeak;
        }
    }
    return "en-us";  // Default to American English
}

// Helper to extract language prefix from voice name
static char get_lang_prefix(const std::string& voice) {
    if (voice.empty()) return 'a';
    return voice[0];
}

// Model::Impl - private implementation
struct Model::Impl {
    KokoroModel model;
    G2P g2p;
    Tokenizer tokenizer;
    std::string model_path;
    std::string current_voice = "af_bella";
    char current_lang_prefix = 'a';
    bool initialized = false;
};

Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

Model Model::load(const std::string& model_path) {
    Model m;

    // Load the underlying model
    m.impl_->model = KokoroModel::load(model_path);
    m.impl_->model_path = model_path;

    // Initialize G2P (American English)
    if (!m.impl_->g2p.initialize("en-us")) {
        throw std::runtime_error("Failed to initialize G2P");
    }

    // Load tokenizer vocabulary
    fs::path vocab_path = fs::path(model_path) / "vocab" / "phonemes.json";
    if (!m.impl_->tokenizer.load_vocab(vocab_path.string())) {
        throw std::runtime_error("Failed to load tokenizer vocab from: " + vocab_path.string());
    }

    m.impl_->initialized = true;
    return m;
}

AudioOutput Model::synthesize(
    const std::string& text,
    const std::string& voice,
    float speed
) const {
    if (!impl_->initialized) {
        throw std::runtime_error("Model not initialized");
    }

    // Step 0: Parse prosody markers from input text
    ParsedProsody parsed = parse_prosody_markers(text);
    const std::string& clean_text = parsed.clean_text;

    // Debug: show parsed prosody
    if (std::getenv("DEBUG_PROSODY")) {
        std::cerr << "Prosody: clean_text=\"" << clean_text << "\"\n";
        std::cerr << "  annotations: " << parsed.annotations.size() << "\n";
        for (const auto& ann : parsed.annotations) {
            std::cerr << "    [" << ann.char_start << "-" << ann.char_end
                      << "] type=" << static_cast<int>(ann.type) << "\n";
        }
        std::cerr << "  breaks: " << parsed.breaks.size() << "\n";
        for (const auto& brk : parsed.breaks) {
            std::cerr << "    after=" << brk.after_char << " ms=" << brk.duration_ms << "\n";
        }
    }

    // Auto-switch G2P language based on voice prefix
    char lang_prefix = get_lang_prefix(voice);
    if (lang_prefix != impl_->current_lang_prefix) {
        const char* espeak_voice = get_espeak_voice(lang_prefix);
        if (!impl_->g2p.set_language(espeak_voice)) {
            std::cerr << "Warning: Failed to switch G2P to language '" << espeak_voice
                      << "' for voice '" << voice << "'\n";
        } else {
            impl_->current_lang_prefix = lang_prefix;
        }
    }

    // Step 1: G2P - convert clean text to phonemes
    std::string phonemes = impl_->g2p.phonemize(clean_text);
    if (phonemes.empty()) {
        throw std::runtime_error("G2P failed to convert text to phonemes");
    }

    // Debug: print phonemes
    if (std::getenv("DEBUG_TOKENS")) {
        std::cerr << "Phonemes: \"" << phonemes << "\"\n";
    }

    // Step 2: Tokenize phonemes
    auto tokens = impl_->tokenizer.tokenize(phonemes);
    if (tokens.empty()) {
        throw std::runtime_error("Tokenizer failed to convert phonemes to tokens");
    }

    // Debug: print tokens
    if (std::getenv("DEBUG_TOKENS")) {
        std::cerr << "Tokens (" << tokens.size() << "): [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << tokens[i];
        }
        std::cerr << "]\n";
    }

    // Map prosody annotations to phonemes using simple heuristic
    PhonemeProsody phoneme_prosody = map_to_phonemes_simple(parsed, tokens.size());

    // Step 3: Forward through model
    AudioOutput output = synthesize_tokens(tokens, voice, speed);

    // Step 4: Apply prosody post-processing (Phase A)
    // Insert silence breaks if specified
    if (!parsed.breaks.empty()) {
        output.samples = insert_breaks_audio(
            output.samples,
            output.sample_rate,
            parsed,
            clean_text.length()
        );
        output.duration_seconds = static_cast<float>(output.samples.size()) / output.sample_rate;
    }

    return output;
}

// Helper function to insert silence breaks into audio
static std::vector<float> insert_breaks_audio(
    const std::vector<float>& audio,
    int sample_rate,
    const ParsedProsody& parsed,
    size_t text_length
) {
    if (parsed.breaks.empty() || text_length == 0) {
        return audio;
    }

    // Calculate approximate sample position for each break
    // Linear mapping from text position to audio position
    float samples_per_char = static_cast<float>(audio.size()) / text_length;

    // Build list of (sample_position, silence_samples) pairs
    std::vector<std::pair<size_t, size_t>> insertions;
    for (const auto& brk : parsed.breaks) {
        size_t sample_pos = static_cast<size_t>(brk.after_char * samples_per_char);
        sample_pos = std::min(sample_pos, audio.size());
        size_t silence_samples = (brk.duration_ms * sample_rate) / 1000;
        if (silence_samples > 0) {
            insertions.emplace_back(sample_pos, silence_samples);
        }
    }

    // Sort by position
    std::sort(insertions.begin(), insertions.end());

    // Calculate total silence to add
    size_t total_silence = 0;
    for (const auto& ins : insertions) {
        total_silence += ins.second;
    }

    // Create output
    std::vector<float> result;
    result.reserve(audio.size() + total_silence);

    size_t audio_pos = 0;
    for (const auto& [insert_pos, silence_samples] : insertions) {
        // Copy audio up to insertion point
        size_t end_pos = std::min(insert_pos, audio.size());
        while (audio_pos < end_pos) {
            result.push_back(audio[audio_pos++]);
        }

        // Apply fade-out before silence (5ms)
        size_t fade_samples = std::min(static_cast<size_t>((5 * sample_rate) / 1000),
                                       result.size());
        for (size_t i = 0; i < fade_samples && result.size() > i; i++) {
            size_t idx = result.size() - fade_samples + i;
            float fade = static_cast<float>(fade_samples - i) / fade_samples;
            result[idx] *= fade;
        }

        // Insert silence
        for (size_t i = 0; i < silence_samples; i++) {
            result.push_back(0.0f);
        }
    }

    // Copy remaining audio
    while (audio_pos < audio.size()) {
        result.push_back(audio[audio_pos++]);
    }

    return result;
}

AudioOutput Model::synthesize_tokens(
    const std::vector<int32_t>& token_ids,
    const std::string& voice,
    float speed
) const {
    if (!impl_->initialized) {
        throw std::runtime_error("Model not initialized");
    }

    // Convert tokens to MLX array
    mx::array tokens = mx::array(token_ids.data(), {1, (int)token_ids.size()}, mx::int32);

    // Run model inference
    mx::array audio = impl_->model.synthesize(tokens, voice, speed);
    mx::eval(audio);

    // Convert to AudioOutput
    AudioOutput output;
    output.sample_rate = impl_->model.config().sample_rate;

    // Get audio samples
    auto shape = audio.shape();
    int num_samples = shape.size() > 1 ? shape[1] : shape[0];
    output.samples.resize(num_samples);

    // Copy data from MLX array
    // Note: This forces a sync, but we need the data on CPU for output
    auto flat = mx::reshape(audio, {-1});
    mx::eval(flat);

    // MLX arrays use data<T>() to access raw data
    const float* data = flat.data<float>();
    std::copy(data, data + num_samples, output.samples.data());

    output.duration_seconds = (float)num_samples / output.sample_rate;
    return output;
}

std::vector<std::string> Model::available_voices() const {
    if (!impl_->initialized) {
        return {};
    }
    return impl_->model.available_voices();
}

bool Model::has_voice(const std::string& voice) const {
    if (!impl_->initialized) {
        return false;
    }
    return impl_->model.has_voice(voice);
}

std::string Model::info() const {
    if (!impl_->initialized) {
        return "Model not initialized";
    }

    std::string result;
    result += "Kokoro TTS Model\n";
    result += "  Model path: " + impl_->model_path + "\n";
    result += "  Weights: " + std::to_string(impl_->model.num_weights()) + "\n";
    result += "  Sample rate: " + std::to_string(impl_->model.config().sample_rate) + " Hz\n";
    result += "  Voices: ";
    for (const auto& v : impl_->model.available_voices()) {
        result += v + " ";
    }
    result += "\n";
    return result;
}

void Model::set_voice(const std::string& voice) {
    if (!impl_->initialized) {
        throw std::runtime_error("Model not initialized");
    }
    if (!impl_->model.has_voice(voice)) {
        throw std::runtime_error("Voice not found: " + voice);
    }

    impl_->current_voice = voice;

    // Auto-set language based on voice prefix
    char lang_prefix = get_lang_prefix(voice);
    if (lang_prefix != impl_->current_lang_prefix) {
        const char* espeak_voice = get_espeak_voice(lang_prefix);
        if (impl_->g2p.set_language(espeak_voice)) {
            impl_->current_lang_prefix = lang_prefix;
        }
    }
}

std::string Model::current_voice() const {
    return impl_->current_voice;
}

std::vector<std::string> Model::available_languages() const {
    std::vector<std::string> langs;
    for (int i = 0; i < NUM_LANGUAGES; i++) {
        langs.push_back(LANGUAGES[i].code);
    }
    return langs;
}

bool Model::has_language(const std::string& lang_code) const {
    for (int i = 0; i < NUM_LANGUAGES; i++) {
        if (lang_code == LANGUAGES[i].code) {
            return true;
        }
    }
    return false;
}

void Model::set_language(const std::string& lang_code) {
    if (!impl_->initialized) {
        throw std::runtime_error("Model not initialized");
    }

    // Find the espeak voice for this language code
    const char* espeak_voice = nullptr;
    for (int i = 0; i < NUM_LANGUAGES; i++) {
        if (lang_code == LANGUAGES[i].code) {
            espeak_voice = LANGUAGES[i].espeak;
            break;
        }
    }

    if (!espeak_voice) {
        throw std::runtime_error("Unknown language code: " + lang_code);
    }

    if (!impl_->g2p.set_language(espeak_voice)) {
        throw std::runtime_error("Failed to set G2P language: " + lang_code);
    }

    impl_->current_lang_prefix = lang_code[0];
}

std::string Model::current_language() const {
    std::string code(1, impl_->current_lang_prefix);
    return code;
}

std::vector<std::string> Model::voices_for_language(const std::string& lang_code) const {
    if (!impl_->initialized) {
        return {};
    }

    std::vector<std::string> result;
    char prefix = lang_code.empty() ? 'a' : lang_code[0];

    for (const auto& voice : impl_->model.available_voices()) {
        if (!voice.empty() && voice[0] == prefix) {
            result.push_back(voice);
        }
    }
    return result;
}

// Helper to split text into sentences for streaming synthesis
static std::vector<std::string> split_into_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        current += c;

        // Check for sentence-ending punctuation
        bool is_sentence_end = (c == '.' || c == '!' || c == '?' ||
                                c == ';' || c == ':');

        // Handle Unicode sentence-ending punctuation (Japanese/Chinese)
        if (i + 2 < text.size()) {
            unsigned char uc = static_cast<unsigned char>(c);
            if ((uc & 0xF0) == 0xE0) {
                // 3-byte UTF-8
                unsigned char c2 = static_cast<unsigned char>(text[i+1]);
                unsigned char c3 = static_cast<unsigned char>(text[i+2]);
                uint32_t cp = ((uc & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
                // Japanese periods: U+3002 (。), U+FF01 (!), U+FF1F (?)
                if (cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F) {
                    is_sentence_end = true;
                }
            }
        }

        if (is_sentence_end) {
            // Skip whitespace after punctuation
            while (i + 1 < text.size() && std::isspace(text[i + 1])) {
                current += text[++i];
            }

            // Trim leading/trailing whitespace
            size_t start = current.find_first_not_of(" \t\n\r");
            size_t end = current.find_last_not_of(" \t\n\r");
            if (start != std::string::npos && end != std::string::npos) {
                sentences.push_back(current.substr(start, end - start + 1));
            }
            current.clear();
        }
    }

    // Don't forget the last sentence (might not end with punctuation)
    if (!current.empty()) {
        size_t start = current.find_first_not_of(" \t\n\r");
        size_t end = current.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            sentences.push_back(current.substr(start, end - start + 1));
        }
    }

    return sentences;
}

void Model::synthesize_streaming(
    const std::string& text,
    StreamingCallback callback,
    const std::string& voice,
    float speed
) const {
    if (!impl_->initialized) {
        throw std::runtime_error("Model not initialized");
    }

    if (!callback) {
        throw std::runtime_error("Callback is required for streaming synthesis");
    }

    // Split text into sentences for chunked processing
    auto sentences = split_into_sentences(text);

    if (sentences.empty()) {
        // Empty text - just call callback with empty final chunk
        callback({}, 0, true);
        return;
    }

    // Auto-switch G2P language based on voice prefix (do once)
    char lang_prefix = get_lang_prefix(voice);
    if (lang_prefix != impl_->current_lang_prefix) {
        const char* espeak_voice = get_espeak_voice(lang_prefix);
        if (impl_->g2p.set_language(espeak_voice)) {
            impl_->current_lang_prefix = lang_prefix;
        }
    }

    // Process each sentence and invoke callback
    for (size_t i = 0; i < sentences.size(); ++i) {
        const std::string& sentence = sentences[i];
        bool is_final = (i == sentences.size() - 1);

        try {
            // Synthesize this sentence
            auto audio = synthesize(sentence, voice, speed);

            // Invoke callback with this chunk
            callback(audio.samples, static_cast<int>(i), is_final);
        } catch (const std::exception& e) {
            // Log error but continue with other sentences
            std::cerr << "Streaming synthesis error on chunk " << i << ": " << e.what() << "\n";

            // Send empty chunk to indicate error but continue
            if (is_final) {
                callback({}, static_cast<int>(i), true);
            }
        }
    }
}

}  // namespace kokoro
