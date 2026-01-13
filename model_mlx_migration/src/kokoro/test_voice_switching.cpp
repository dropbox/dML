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

// Test voice switching without model reload
#include <iostream>
#include <fstream>
#include <chrono>
#include "kokoro.h"

void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return;

    float max_amp = 0.0f;
    for (const auto& s : samples) max_amp = std::max(max_amp, std::abs(s));

    std::vector<int16_t> pcm(samples.size());
    float scale = (max_amp > 1.0f) ? 0.95f / max_amp : 1.0f;
    for (size_t i = 0; i < samples.size(); ++i) {
        float s = std::max(-1.0f, std::min(1.0f, samples[i] * scale));
        pcm[i] = static_cast<int16_t>(s * 32767.0f);
    }

    uint32_t data_size = pcm.size() * sizeof(int16_t);
    uint32_t file_size = 36 + data_size;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1, num_channels = 1, bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * 2;
    uint16_t block_align = 2;
    file.write(reinterpret_cast<char*>(&fmt_size), 4);
    file.write(reinterpret_cast<char*>(&audio_format), 2);
    file.write(reinterpret_cast<char*>(&num_channels), 2);
    file.write(reinterpret_cast<char*>(&sample_rate), 4);
    file.write(reinterpret_cast<char*>(&byte_rate), 4);
    file.write(reinterpret_cast<char*>(&block_align), 2);
    file.write(reinterpret_cast<char*>(&bits_per_sample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<char*>(&data_size), 4);
    file.write(reinterpret_cast<const char*>(pcm.data()), data_size);
}

int main() {
    std::cout << "=== Voice Switching Test ===\n\n";

    // Load model ONCE
    std::cout << "Loading model...\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto model = kokoro::Model::load("../../kokoro_cpp_export");
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Model loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n\n";

    // Test voices
    struct TestCase {
        std::string voice;
        std::string text;
        std::string wav_file;
    };

    std::vector<TestCase> tests = {
        {"af_bella", "Hello, how are you today?", "test_bella.wav"},
        {"am_adam", "Hello, how are you today?", "test_adam.wav"},
        {"bf_emma", "Hello, how are you today?", "test_emma.wav"},
        {"bm_george", "Hello, how are you today?", "test_george.wav"},
    };

    for (const auto& test : tests) {
        std::cout << "Synthesizing with " << test.voice << "...\n";
        start = std::chrono::high_resolution_clock::now();
        auto output = model.synthesize(test.text, test.voice, 1.0f);
        end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        write_wav(test.wav_file, output.samples, output.sample_rate);
        std::cout << "  Time: " << ms << "ms, Duration: " << output.duration_seconds << "s, File: " << test.wav_file << "\n";
    }

    std::cout << "\n=== Voice switching test PASSED ===\n";
    std::cout << "All 4 voices synthesized from single model load.\n";
    return 0;
}
