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

// Test Kokoro synthesize_tokens() - bypasses G2P
// Accepts token IDs from command line to enable Python-C++ equivalence validation
//
// Usage:
//   ./test_token_input <model_path> <voice> <token_ids...>
//   ./test_token_input ../../kokoro_cpp_export af_bella 0 70 66 78 82 16 98 101 43 93 60 0
//
// Or with JSON file:
//   ./test_token_input ../../kokoro_cpp_export af_bella --json tokens.json
//
// Build (from src/kokoro):
// clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib \
//     kokoro.cpp model.cpp g2p.cpp tokenizer.cpp test_token_input.cpp \
//     -lmlx -lespeak-ng -o test_token_input

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include "kokoro.h"

// Simple WAV file writer
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
        scale = 0.95f / max_amp;
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        float s = samples[i] * scale;
        s = std::max(-1.0f, std::min(1.0f, s));
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

// Read tokens from JSON file (simple parser for {"token_ids": [0, 70, ...]})
std::vector<int32_t> read_tokens_from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Find token_ids array
    std::vector<int32_t> tokens;
    size_t start = content.find("[");
    size_t end = content.find("]");
    if (start == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("Invalid JSON: no array found");
    }

    // Parse numbers between [ and ]
    std::string arr = content.substr(start + 1, end - start - 1);
    std::stringstream ss(arr);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t\n\r"));
        item.erase(item.find_last_not_of(" \t\n\r") + 1);
        if (!item.empty()) {
            tokens.push_back(std::stoi(item));
        }
    }

    return tokens;
}

void print_usage(const char* prog) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << prog << " <model_path> <voice> <token_ids...>\n";
    std::cerr << "  " << prog << " <model_path> <voice> --json <tokens.json>\n";
    std::cerr << "\nExample:\n";
    std::cerr << "  " << prog << " ../../kokoro_cpp_export af_bella 0 70 66 78 82 16 98 101 43 93 60 0\n";
    std::cerr << "\nOutput:\n";
    std::cerr << "  Saves audio to token_input_output.wav\n";
    std::cerr << "  Prints JSON with timing info for validation\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string voice = argv[2];
    std::vector<int32_t> tokens;

    // Parse tokens
    if (std::string(argv[3]) == "--json") {
        if (argc < 5) {
            std::cerr << "Error: --json requires a file path\n";
            return 1;
        }
        tokens = read_tokens_from_json(argv[4]);
    } else {
        // Read tokens from command line args
        for (int i = 3; i < argc; ++i) {
            tokens.push_back(std::stoi(argv[i]));
        }
    }

    if (tokens.empty()) {
        std::cerr << "Error: No tokens provided\n";
        return 1;
    }

    // Output info
    std::cerr << "Model: " << model_path << "\n";
    std::cerr << "Voice: " << voice << "\n";
    std::cerr << "Tokens (" << tokens.size() << "): [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << tokens[i];
    }
    std::cerr << "]\n";

    try {
        // Load model
        auto start = std::chrono::high_resolution_clock::now();
        auto model = kokoro::Model::load(model_path);
        auto end = std::chrono::high_resolution_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cerr << "Model loaded in " << load_ms << "ms\n";

        // Synthesize from tokens (bypasses G2P!)
        std::cerr << "Synthesizing from " << tokens.size() << " tokens...\n";
        start = std::chrono::high_resolution_clock::now();
        auto output = model.synthesize_tokens(tokens, voice, 1.0f);
        end = std::chrono::high_resolution_clock::now();
        auto synth_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Calculate metrics
        float max_amp = 0.0f;
        double rms_sum = 0.0;
        for (const auto& s : output.samples) {
            max_amp = std::max(max_amp, std::abs(s));
            rms_sum += s * s;
        }
        float rms = std::sqrt(rms_sum / output.samples.size());
        float rtf = (output.duration_seconds * 1000.0f) / synth_ms;

        std::cerr << "Synthesis complete in " << synth_ms << "ms\n";
        std::cerr << "  Duration: " << output.duration_seconds << "s\n";
        std::cerr << "  RTF: " << rtf << "x real-time\n";
        std::cerr << "  Max amplitude: " << max_amp << "\n";
        std::cerr << "  RMS: " << rms << "\n";

        // Save to WAV
        std::string wav_file = "token_input_output.wav";
        write_wav(wav_file, output.samples, output.sample_rate);
        std::cerr << "Saved to: " << wav_file << "\n";

        // Output JSON to stdout for parsing by validation script
        std::cout << "{\n";
        std::cout << "  \"status\": \"success\",\n";
        std::cout << "  \"tokens\": " << tokens.size() << ",\n";
        std::cout << "  \"samples\": " << output.samples.size() << ",\n";
        std::cout << "  \"duration_s\": " << output.duration_seconds << ",\n";
        std::cout << "  \"synth_ms\": " << synth_ms << ",\n";
        std::cout << "  \"rtf\": " << rtf << ",\n";
        std::cout << "  \"max_amp\": " << max_amp << ",\n";
        std::cout << "  \"rms\": " << rms << ",\n";
        std::cout << "  \"wav_file\": \"" << wav_file << "\"\n";
        std::cout << "}\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cout << "{\"status\": \"error\", \"error\": \"" << e.what() << "\"}\n";
        return 1;
    }
}
