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

// Test Kokoro Model Loading
// Build: clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib model.cpp test_model.cpp -lmlx -o test_model

#include <iostream>
#include <string>
#include "model.h"

int main(int argc, char* argv[]) {
    std::string model_path = "../../../kokoro_cpp_export";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "=== Kokoro Model Loading Test ===\n\n";
    std::cout << "Model path: " << model_path << "\n\n";

    try {
        // Load model
        std::cout << "Loading model...\n";
        auto model = kokoro::KokoroModel::load(model_path);
        std::cout << "Model loaded successfully!\n\n";

        // Print config
        const auto& config = model.config();
        std::cout << "--- Configuration ---\n";
        std::cout << "n_token: " << config.n_token << "\n";
        std::cout << "hidden_dim: " << config.hidden_dim << "\n";
        std::cout << "style_dim: " << config.style_dim << "\n";
        std::cout << "plbert_hidden_size: " << config.plbert_hidden_size << "\n";
        std::cout << "sample_rate: " << config.sample_rate << "\n";
        std::cout << "hop_size: " << config.hop_size << "\n";
        std::cout << "weight_norm_folded: " << (config.weight_norm_folded ? "true" : "false") << "\n\n";

        // Print weight count
        std::cout << "--- Weights ---\n";
        std::cout << "Total weights: " << model.num_weights() << "\n";

        // Check specific weights exist
        std::vector<std::string> key_weights = {
            "bert.embeddings.word_embeddings.weight",
            "bert.encoder.albert_layer.attention.query.weight",
            "decoder.generator.resblocks_0.convs1_0.weight",
        };
        for (const auto& key : key_weights) {
            bool exists = model.has_weight(key);
            std::cout << key << ": " << (exists ? "OK" : "MISSING") << "\n";
            if (exists) {
                auto w = model.get_weight(key);
                std::cout << "  shape: [";
                for (size_t i = 0; i < w.shape().size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << w.shape()[i];
                }
                std::cout << "]\n";
            }
        }
        std::cout << "\n";

        // Print available voices
        std::cout << "--- Voices ---\n";
        auto voices = model.available_voices();
        std::cout << "Available voices: " << voices.size() << "\n";
        for (const auto& v : voices) {
            std::cout << "  - " << v << "\n";
        }
        std::cout << "\n";

        // Test synthesize (placeholder)
        std::cout << "--- Testing synthesize ---\n";
        if (!voices.empty()) {
            // Create dummy tokens [BOS, a, b, c, EOS]
            auto tokens = mx::array({0, 43, 44, 45, 0});
            tokens = mx::reshape(tokens, {1, 5});  // [batch=1, seq_len=5]

            auto audio = model.synthesize(tokens, voices[0], 1.0f);
            mx::eval(audio);
            std::cout << "Synthesize returned shape: [" << audio.shape()[0] << ", "
                      << audio.shape()[1] << "]\n";
            std::cout << "(Note: Currently returns placeholder zeros)\n\n";
        }

        std::cout << "=== All tests PASSED ===\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
