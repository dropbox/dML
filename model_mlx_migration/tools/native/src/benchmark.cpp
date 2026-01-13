/**
 * @file benchmark.cpp
 * @brief Benchmark ECAPA-TDNN Native C++ Inference
 *
 * Compares native C++ inference against MLX Python baseline.
 * Target: <0.3ms at batch=1 (baseline: 2.79ms MLX Python)
 */

#include "ecapa_inference.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "ECAPA-TDNN Native C++ Benchmark\n";
    std::cout << "========================================\n\n";

    // Default weights path (safetensors format required)
    std::string weights_path = "models/sota/ecapa-tdnn-mlx/weights.safetensors";
    std::string labels_path = "models/sota/ecapa-tdnn-mlx/label_encoder.txt";

    if (argc > 1) {
        weights_path = argv[1];
    }
    if (argc > 2) {
        labels_path = argv[2];
    }

    std::cout << "Loading model from: " << weights_path << "\n\n";

    try {
        // Load model
        ecapa::ECAPAInference model(weights_path);
        model.load_labels(labels_path);

        // First benchmark WITHOUT compile
        std::cout << "----------------------------------------\n";
        std::cout << "WITHOUT mx.compile() (eager mode)\n";
        std::cout << "----------------------------------------\n";

        auto input_test = mlx::core::random::normal({1, 300, 60});
        mlx::core::eval(input_test);

        // Warm up
        for (int i = 0; i < 5; i++) {
            auto [logits, pred] = model.classify(input_test);
        }

        // Benchmark eager mode
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 50; i++) {
            auto [logits, pred] = model.classify(input_test);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double eager_ms = static_cast<double>(duration.count()) / 1000.0 / 50;
        std::cout << "Eager batch=1: " << std::fixed << std::setprecision(3) << eager_ms << " ms\n\n";

        // Compile model for faster inference
        std::cout << "Compiling model...\n";
        model.compile_model();
        std::cout << "Model compiled.\n\n";

        // Benchmark parameters
        std::vector<int> batch_sizes = {1, 4, 8, 16, 32};
        int seq_len = 300;  // ~3 seconds of audio
        int num_iterations = 100;

        // MLX Python baseline (full model, measured 2026-01-01)
        // Note: Previous 2.79ms was for Block 0 only
        double mlx_python_compiled_ms = 58.5;  // Python with mx.compile()
        double mlx_python_eager_ms = 92.4;     // Python without mx.compile()

        std::cout << "Parameters:\n";
        std::cout << "  Sequence length: " << seq_len << " frames\n";
        std::cout << "  Iterations: " << num_iterations << "\n";
        std::cout << "  Python eager: " << mlx_python_eager_ms << " ms\n";
        std::cout << "  Python compiled: " << mlx_python_compiled_ms << " ms\n\n";

        std::cout << "Results:\n";
        std::cout << std::setw(8) << "Batch"
                  << std::setw(12) << "Latency"
                  << std::setw(12) << "Throughput"
                  << std::setw(12) << "vs Eager"
                  << std::setw(12) << "vs Compiled" << "\n";
        std::cout << std::string(56, '-') << "\n";

        for (int batch_size : batch_sizes) {
            double latency_ms = ecapa::ECAPABenchmark::benchmark_latency(
                model, batch_size, seq_len, num_iterations
            );

            double throughput = batch_size / (latency_ms / 1000.0);
            double speedup_eager = mlx_python_eager_ms / latency_ms;
            double speedup_compiled = mlx_python_compiled_ms / latency_ms;

            std::cout << std::setw(8) << batch_size
                      << std::setw(11) << std::fixed << std::setprecision(2) << latency_ms << "ms"
                      << std::setw(12) << std::fixed << std::setprecision(1) << throughput
                      << std::setw(11) << std::fixed << std::setprecision(2) << speedup_eager << "x"
                      << std::setw(11) << std::fixed << std::setprecision(2) << speedup_compiled << "x\n";
        }

        std::cout << "\n";

        // Test accuracy with sample input
        std::cout << "Testing inference accuracy...\n";

        auto input = mlx::core::random::normal({1, seq_len, 60});
        mlx::core::eval(input);

        auto [logits, predictions] = model.classify(input);

        int pred_idx = static_cast<int>(predictions.item<int32_t>());
        std::string pred_lang = model.get_language_code(pred_idx);

        std::cout << "  Sample prediction: " << pred_lang << " (index " << pred_idx << ")\n";

        // Summary
        std::cout << "\n========================================\n";
        std::cout << "SUMMARY\n";
        std::cout << "========================================\n";

        double batch1_latency = ecapa::ECAPABenchmark::benchmark_latency(model, 1, seq_len, 100);
        double speedup_eager = mlx_python_eager_ms / batch1_latency;
        double speedup_compiled = mlx_python_compiled_ms / batch1_latency;

        std::cout << "C++ Batch=1 latency: " << std::fixed << std::setprecision(2) << batch1_latency << " ms\n";
        std::cout << "Python eager: " << mlx_python_eager_ms << " ms\n";
        std::cout << "Python compiled: " << mlx_python_compiled_ms << " ms\n";
        std::cout << "Speedup vs Python eager: " << std::fixed << std::setprecision(2) << speedup_eager << "x\n";
        std::cout << "Speedup vs Python compiled: " << std::fixed << std::setprecision(2) << speedup_compiled << "x\n";

        if (speedup_compiled > 1.0) {
            std::cout << "\nSUCCESS: C++ is faster than Python compiled!\n";
        } else if (speedup_eager > 1.0) {
            std::cout << "\nGOOD: C++ is faster than Python eager mode.\n";
        } else {
            std::cout << "\nNOTE: C++ implementation needs optimization.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
