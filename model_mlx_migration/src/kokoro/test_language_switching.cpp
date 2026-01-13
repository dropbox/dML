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

// Test Kokoro Language Switching: V4 feature
// Tests that G2P auto-switches language based on voice prefix

#include "kokoro.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>

void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate);

void test_language_switching() {
    std::cout << "=== Kokoro Language Switching Test (V4) ===" << std::endl;

    // Load model
    std::string model_path = "/Users/ayates/model_mlx_migration/kokoro_cpp_export";
    kokoro::Model model = kokoro::Model::load(model_path);

    // Test phrases for different languages
    // Each phrase tests: language prefix -> G2P auto-switch -> synthesis
    struct TestCase {
        const char* voice;
        const char* text;
        const char* language;
        const char* expected_lang_code;
    };

    TestCase tests[] = {
        // American English
        {"af_bella", "Hello world", "American English", "a"},

        // British English
        {"bf_emma", "Hello world", "British English", "b"},

        // Japanese (hiragana/katakana work best with espeak-ng)
        {"jf_alpha", "konnichiwa", "Japanese", "j"},

        // Chinese (pinyin works with espeak-ng cmn voice)
        {"zf_xiaobei", "ni hao", "Mandarin Chinese", "z"},

        // Spanish
        {"ef_dora", "hola mundo", "Spanish", "e"},

        // French
        {"ff_siwis", "bonjour le monde", "French", "f"},

        // Italian
        {"if_sara", "ciao mondo", "Italian", "i"},

        // Portuguese (Brazilian)
        {"pf_dora", "ola mundo", "Portuguese", "p"},

        // Hindi (romanized)
        {"hf_alpha", "namaste duniya", "Hindi", "h"},
    };

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        std::cout << "\n--- Testing " << test.language << " ---" << std::endl;
        std::cout << "Voice: " << test.voice << std::endl;
        std::cout << "Text: \"" << test.text << "\"" << std::endl;

        // Check if voice exists
        if (!model.has_voice(test.voice)) {
            std::cout << "SKIP: Voice not found" << std::endl;
            continue;
        }

        try {
            // Synthesize - this should auto-switch language
            auto audio = model.synthesize(test.text, test.voice);

            std::cout << "Duration: " << audio.duration_seconds * 1000 << " ms" << std::endl;
            std::cout << "Samples: " << audio.num_samples() << std::endl;

            // Verify language was set correctly
            std::string current_lang = model.current_language();
            if (current_lang != test.expected_lang_code) {
                std::cout << "FAIL: Expected language '" << test.expected_lang_code
                          << "' but got '" << current_lang << "'" << std::endl;
                failed++;
                continue;
            }

            // Check audio is non-empty and has reasonable values
            if (audio.empty()) {
                std::cout << "FAIL: Empty audio" << std::endl;
                failed++;
                continue;
            }

            // Find max amplitude
            float max_amp = 0.0f;
            for (float s : audio.samples) {
                float abs_s = s < 0 ? -s : s;
                if (abs_s > max_amp) max_amp = abs_s;
            }

            if (max_amp < 0.001f) {
                std::cout << "FAIL: Audio is silent (max amp: " << max_amp << ")" << std::endl;
                failed++;
                continue;
            }

            std::cout << "Max amplitude: " << max_amp << std::endl;
            std::cout << "PASS" << std::endl;

            // Save audio for manual verification
            std::string filename = std::string("test_lang_") + test.expected_lang_code + ".wav";
            write_wav(filename, audio.samples, audio.sample_rate);
            std::cout << "Saved: " << filename << std::endl;

            passed++;

        } catch (const std::exception& e) {
            std::cout << "FAIL: Exception: " << e.what() << std::endl;
            failed++;
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
}

void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Normalize samples
    float max_val = 0.0f;
    for (float s : samples) {
        float abs_s = s < 0 ? -s : s;
        if (abs_s > max_val) max_val = abs_s;
    }
    float scale = max_val > 0.0f ? 0.9f / max_val : 1.0f;

    // Convert to int16
    std::vector<int16_t> int_samples(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        float val = samples[i] * scale * 32767.0f;
        if (val > 32767.0f) val = 32767.0f;
        if (val < -32768.0f) val = -32768.0f;
        int_samples[i] = static_cast<int16_t>(val);
    }

    // WAV header
    uint32_t data_size = static_cast<uint32_t>(int_samples.size() * 2);
    uint32_t file_size = 36 + data_size;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    uint32_t fmt_size = 16;
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    uint16_t audio_format = 1; // PCM
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    uint16_t num_channels = 1;
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    uint32_t sample_rate_u = static_cast<uint32_t>(sample_rate);
    file.write(reinterpret_cast<const char*>(&sample_rate_u), 4);
    uint32_t byte_rate = sample_rate_u * 2;
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    uint16_t block_align = 2;
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    uint16_t bits_per_sample = 16;
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
    file.write(reinterpret_cast<const char*>(int_samples.data()), data_size);
}

int main() {
    try {
        test_language_switching();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
