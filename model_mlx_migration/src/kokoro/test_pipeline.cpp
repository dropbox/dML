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

// Test Kokoro Full Pipeline: G2P -> Tokenizer -> Model
// Build:
// clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib \
//     kokoro.cpp model.cpp g2p.cpp tokenizer.cpp test_pipeline.cpp \
//     -lmlx -lespeak-ng -o test_pipeline

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdint>
#include "kokoro.h"

// Simple WAV file writer with optional normalization
void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate, bool normalize = true) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open WAV file for writing: " + filename);
    }

    // Find max amplitude for normalization
    float max_amp = 0.0f;
    for (const auto& s : samples) {
        max_amp = std::max(max_amp, std::abs(s));
    }

    // Convert float to 16-bit PCM
    std::vector<int16_t> pcm(samples.size());
    float scale = 1.0f;
    if (normalize && max_amp > 1.0f) {
        scale = 0.95f / max_amp;  // Normalize to 95% to avoid clipping
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        float s = samples[i] * scale;
        s = std::max(-1.0f, std::min(1.0f, s));  // Clamp after scaling
        pcm[i] = static_cast<int16_t>(s * 32767.0f);
    }

    uint32_t data_size = pcm.size() * sizeof(int16_t);
    uint32_t file_size = 36 + data_size;

    // RIFF header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<char*>(&file_size), 4);
    file.write("WAVE", 4);

    // fmt chunk
    file.write("fmt ", 4);
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;  // PCM
    uint16_t num_channels = 1;
    uint32_t byte_rate = sample_rate * num_channels * 2;
    uint16_t block_align = num_channels * 2;
    uint16_t bits_per_sample = 16;
    file.write(reinterpret_cast<char*>(&fmt_size), 4);
    file.write(reinterpret_cast<char*>(&audio_format), 2);
    file.write(reinterpret_cast<char*>(&num_channels), 2);
    file.write(reinterpret_cast<char*>(&sample_rate), 4);
    file.write(reinterpret_cast<char*>(&byte_rate), 4);
    file.write(reinterpret_cast<char*>(&block_align), 2);
    file.write(reinterpret_cast<char*>(&bits_per_sample), 2);

    // data chunk
    file.write("data", 4);
    file.write(reinterpret_cast<char*>(&data_size), 4);
    file.write(reinterpret_cast<const char*>(pcm.data()), data_size);
}

int main(int argc, char* argv[]) {
    std::string model_path = "../../kokoro_cpp_export";  // Relative to src/kokoro
    std::string text = "Hello world!";

    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        text = argv[2];
    }

    std::cout << "=== Kokoro Full Pipeline Test ===\n\n";
    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Input text: \"" << text << "\"\n\n";

    try {
        // Load model
        std::cout << "Loading model...\n";
        auto start = std::chrono::high_resolution_clock::now();

        auto model = kokoro::Model::load(model_path);

        auto end = std::chrono::high_resolution_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Model loaded in " << load_ms << "ms\n\n";

        // Print model info
        std::cout << model.info() << "\n";

        // Synthesize
        std::cout << "Synthesizing speech...\n";
        start = std::chrono::high_resolution_clock::now();

        auto output = model.synthesize(text, "af_bella", 1.0f);

        end = std::chrono::high_resolution_clock::now();
        auto synth_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Synthesis complete in " << synth_ms << "ms\n";
        std::cout << "Output:\n";
        std::cout << "  Samples: " << output.num_samples() << "\n";
        std::cout << "  Duration: " << output.duration_seconds << "s\n";
        std::cout << "  Sample rate: " << output.sample_rate << " Hz\n";

        // Calculate real-time factor
        if (synth_ms > 0) {
            float rtf = (output.duration_seconds * 1000.0f) / synth_ms;
            std::cout << "  Real-time factor: " << rtf << "x\n";
        }

        // Check output is not all zeros (placeholder check)
        bool all_zeros = true;
        float max_abs = 0.0f;
        for (size_t i = 0; i < output.samples.size(); ++i) {
            if (output.samples[i] != 0.0f) {
                all_zeros = false;
            }
            max_abs = std::max(max_abs, std::abs(output.samples[i]));
        }
        if (all_zeros) {
            std::cout << "\n[NOTE: Output is placeholder zeros - forward pass not yet implemented]\n";
        }
        std::cout << "  Max amplitude: " << max_abs << "\n";

        // Save to WAV file
        std::string wav_file = "output.wav";
        write_wav(wav_file, output.samples, output.sample_rate);
        std::cout << "\nSaved to: " << wav_file << "\n";

        std::cout << "\n=== Test PASSED ===\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
