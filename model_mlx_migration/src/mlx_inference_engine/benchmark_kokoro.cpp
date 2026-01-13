// Kokoro C++ MLX Benchmark
// Measures actual synthesis RTF (excluding model load)
// Also benchmarks streaming synthesis time-to-first-audio

#include "mlx_inference_engine.hpp"
#include "kokoro.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <atomic>

int main(int argc, char* argv[]) {
    std::string model_path = "kokoro_cpp_export";
    if (argc > 1) model_path = argv[1];

    std::cout << "=== Kokoro C++ MLX Benchmark ===\n\n";

    // Load model
    std::cout << "Loading model from: " << model_path << "\n";
    auto load_start = std::chrono::high_resolution_clock::now();

    mlx_inference::MLXInferenceEngine engine;
    engine.load_kokoro(model_path);

    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    std::cout << "Model load time: " << load_ms << " ms\n\n";

    mlx_inference::TTSConfig config;
    config.voice = "af_heart";
    config.speed = 1.0f;

    // Warmup (critical for MLX compilation)
    std::cout << "Warming up (3 iterations)...\n";
    for (int i = 0; i < 3; i++) {
        engine.synthesize("Hello.", config);
    }
    std::cout << "Warmup complete.\n\n";

    // Benchmark texts
    std::vector<std::pair<std::string, std::string>> tests = {
        {"Short", "Hello world."},
        {"Medium", "The quick brown fox jumps over the lazy dog near the riverbank."},
        {"Long", "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat. It was a hobbit hole, and that means comfort."}
    };

    std::cout << "| Length | Text Len | Audio Dur | Latency | RTF |\n";
    std::cout << "|--------|----------|-----------|---------|-----|\n";

    for (const auto& [name, text] : tests) {
        // Run 3 times, take median
        std::vector<double> latencies;
        std::vector<double> durations;

        for (int i = 0; i < 3; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            auto output = engine.synthesize(text, config);
            auto end = std::chrono::high_resolution_clock::now();

            double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(latency_ms);
            durations.push_back(output.duration_seconds);
        }

        // Median
        std::sort(latencies.begin(), latencies.end());
        double median_latency = latencies[1];
        double audio_dur = durations[1];
        double rtf = audio_dur / (median_latency / 1000.0);

        std::cout << "| " << name << " | " << text.length() << " | "
                  << std::fixed << std::setprecision(2) << audio_dur << "s | "
                  << std::setprecision(1) << median_latency << "ms | "
                  << std::setprecision(1) << rtf << "x |\n";
    }

    // ==========================================
    // Streaming Synthesis Benchmark
    // ==========================================
    std::cout << "\n=== Streaming Synthesis Benchmark ===\n";
    std::cout << "Testing time-to-first-audio (TTFA)\n\n";

    // Load Kokoro model directly for streaming API
    kokoro::Model kokoro_model = kokoro::Model::load(model_path);
    kokoro_model.set_voice("af_heart");

    // Warmup streaming
    std::cout << "Warming up streaming...\n";
    kokoro_model.synthesize_streaming("Hello.", [](const std::vector<float>&, int, bool) {}, "af_heart", 1.0f);
    std::cout << "Warmup complete.\n\n";

    // Multi-sentence test text
    std::string streaming_text = "Hello world. This is a test of streaming synthesis. "
        "The quick brown fox jumps over the lazy dog. "
        "Streaming allows audio playback to begin before the entire text is processed.";

    // Benchmark: Regular synthesis (full text)
    std::cout << "Regular synthesis (full text):\n";
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto output = kokoro_model.synthesize(streaming_text, "af_heart", 1.0f);
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  Total latency: " << std::fixed << std::setprecision(1) << latency_ms << " ms\n";
        std::cout << "  Audio duration: " << std::setprecision(2) << output.duration_seconds << " s\n";
        std::cout << "  Time-to-first-audio: " << std::setprecision(1) << latency_ms << " ms (same as total)\n\n";
    }

    // Benchmark: Streaming synthesis
    std::cout << "Streaming synthesis (sentence-by-sentence):\n";
    {
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point first_audio_time;
        double total_audio_duration = 0.0;
        int chunk_count = 0;
        bool first_chunk = true;

        start_time = std::chrono::high_resolution_clock::now();

        kokoro_model.synthesize_streaming(
            streaming_text,
            [&](const std::vector<float>& samples, int chunk_index, bool is_final) {
                if (first_chunk) {
                    first_audio_time = std::chrono::high_resolution_clock::now();
                    first_chunk = false;
                }
                total_audio_duration += static_cast<double>(samples.size()) / 24000.0;
                chunk_count++;

                double chunk_dur = static_cast<double>(samples.size()) / 24000.0;
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double, std::milli>(now - start_time).count();
                std::cout << "    Chunk " << chunk_index << ": " << samples.size() << " samples ("
                          << std::fixed << std::setprecision(2) << chunk_dur << "s) @ "
                          << std::setprecision(0) << elapsed << "ms"
                          << (is_final ? " [final]" : "") << "\n";
            },
            "af_heart",
            1.0f
        );

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double ttfa = std::chrono::duration<double, std::milli>(first_audio_time - start_time).count();

        std::cout << "\n  Chunks generated: " << chunk_count << "\n";
        std::cout << "  Total latency: " << std::setprecision(1) << total_latency << " ms\n";
        std::cout << "  Total audio duration: " << std::setprecision(2) << total_audio_duration << " s\n";
        std::cout << "  Time-to-first-audio: " << std::setprecision(1) << ttfa << " ms\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
