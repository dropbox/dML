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

// Test voice ordering to isolate state corruption bug
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
    auto model = kokoro::Model::load("../../kokoro_cpp_export");
    std::string text = "Hello, I am testing.";

    // Test: bf_emma FIRST
    {
        auto out = model.synthesize(text, "bf_emma", 1.0f);
        write_wav("order1_emma_first.wav", out.samples, out.sample_rate);
        std::cout << "bf_emma first: " << out.duration_seconds << "s\n";
    }

    // Test: af_bella then bf_emma
    {
        auto out1 = model.synthesize(text, "af_bella", 1.0f);
        write_wav("order2_bella.wav", out1.samples, out1.sample_rate);
        auto out2 = model.synthesize(text, "bf_emma", 1.0f);
        write_wav("order2_emma_second.wav", out2.samples, out2.sample_rate);
        std::cout << "bf_emma second: " << out2.duration_seconds << "s\n";
    }

    return 0;
}
