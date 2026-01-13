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

// Test MLX C++ safetensors loading
// Build: clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib test_mlx_load.cpp -lmlx -o test_mlx_load

#include <iostream>
#include <string>
#include "mlx/mlx.h"

namespace mx = mlx::core;

int main(int argc, char* argv[]) {
    std::string weights_path = "../../../kokoro_cpp_export/weights.safetensors";
    if (argc > 1) {
        weights_path = argv[1];
    }

    std::cout << "MLX C++ Test: Loading safetensors\n";
    std::cout << "Path: " << weights_path << "\n\n";

    try {
        // Load safetensors
        auto [weights, metadata] = mx::load_safetensors(weights_path);

        std::cout << "Loaded " << weights.size() << " tensors\n\n";

        // Print first 10 weight names and shapes
        int count = 0;
        for (const auto& [name, arr] : weights) {
            if (count >= 10) {
                std::cout << "... (" << (weights.size() - 10) << " more)\n";
                break;
            }
            std::cout << name << ": [";
            const auto& shape = arr.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << shape[i];
            }
            std::cout << "] " << arr.dtype() << "\n";
            ++count;
        }

        // Test basic MLX operations
        std::cout << "\n--- Testing MLX operations ---\n";

        // Create a small test array
        auto x = mx::ones({2, 3}, mx::float32);
        std::cout << "Created ones(2,3): shape=[" << x.shape()[0] << ", " << x.shape()[1] << "]\n";

        // Matrix multiply
        auto y = mx::ones({3, 4}, mx::float32);
        auto z = mx::matmul(x, y);
        mx::eval(z);  // Force evaluation
        std::cout << "matmul result shape: [" << z.shape()[0] << ", " << z.shape()[1] << "]\n";

        // Test activation using maximum(x, 0) = relu
        auto input = mx::ones({1, 10, 8}, mx::float32);  // (batch, seq, channels)
        auto zero = mx::zeros({1, 10, 8}, mx::float32);
        auto relu_out = mx::maximum(input, zero);  // relu = max(x, 0)
        mx::eval(relu_out);
        std::cout << "relu (via max) result shape: [" << relu_out.shape()[0] << ", "
                  << relu_out.shape()[1] << ", " << relu_out.shape()[2] << "]\n";

        // Test transpose
        auto transposed = mx::transpose(input, {0, 2, 1});  // (batch, channels, seq)
        mx::eval(transposed);
        std::cout << "transpose result shape: [" << transposed.shape()[0] << ", "
                  << transposed.shape()[1] << ", " << transposed.shape()[2] << "]\n";

        // Test sigmoid
        auto sig_out = mx::sigmoid(input);
        mx::eval(sig_out);
        std::cout << "sigmoid result shape: [" << sig_out.shape()[0] << ", "
                  << sig_out.shape()[1] << ", " << sig_out.shape()[2] << "]\n";

        std::cout << "\n=== MLX C++ test PASSED ===\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
