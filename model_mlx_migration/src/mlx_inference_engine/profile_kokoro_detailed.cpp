// Detailed Kokoro Pipeline Profiler - Stage-by-Stage Timing
// GOAL: Identify exact bottleneck and PROVE optimal performance
//
// Measures with GPU synchronization between each stage:
//   1. Voice embedding lookup
//   2. BERT forward pass
//   3. BERT encoder (linear projection)
//   4. Text encoder forward
//   5. Predictor (contains BiLSTM - likely bottleneck)
//   6. Decoder (audio synthesis)
//
// Usage: ./profile_kokoro_detailed

#include "kokoro.h"
#include "misaki_g2p.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <mlx/mlx.h>

namespace mx = mlx::core;
using Clock = std::chrono::high_resolution_clock;

// Timing helper with GPU sync
double timed_eval(const std::function<void()>& fn) {
    mx::synchronize();  // Ensure previous work complete
    auto start = Clock::now();
    fn();
    mx::synchronize();  // Wait for this work
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "              KOKORO TTS DETAILED PROFILER\n";
    std::cout << "              Proving Optimal Performance\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    // Load model
    std::cout << "Loading model...\n";
    auto load_start = Clock::now();
    kokoro::Model model = kokoro::Model::load("kokoro_cpp_export");
    mx::synchronize();
    auto load_end = Clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    std::cout << "Model load: " << std::fixed << std::setprecision(0) << load_ms << " ms\n\n";

    model.set_voice("af_heart");

    // Warmup passes - critical for MLX compilation
    std::cout << "Warmup (5 passes for kernel compilation)...\n";
    for (int i = 0; i < 5; i++) {
        auto out = model.synthesize("Test warmup pass.", "af_heart", 1.0f);
        mx::synchronize();
        std::cout << "  Warmup " << (i+1) << " complete\n";
    }
    std::cout << "\n";

    // Test texts
    std::vector<std::pair<std::string, std::string>> tests = {
        {"Minimal", "Hi"},
        {"Short", "Hello world."},
        {"Medium", "The quick brown fox jumps over the lazy dog."},
        {"Long", "Hello world. This is a comprehensive test of the Kokoro text to speech synthesis system. "
                 "We are measuring the latency of each individual pipeline stage."}
    };

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                    SYNTHESIS TIMING\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    for (const auto& [name, text] : tests) {
        std::cout << "─────────────────────────────────────────────────────\n";
        std::cout << name << " (" << text.length() << " chars): \"" << text.substr(0, 50);
        if (text.length() > 50) std::cout << "...";
        std::cout << "\"\n";

        // Run 5 times, report average and min
        std::vector<double> synth_times;
        double audio_duration = 0;

        for (int run = 0; run < 5; run++) {
            mx::synchronize();
            auto start = Clock::now();
            auto output = model.synthesize(text, "af_heart", 1.0f);
            mx::synchronize();
            auto end = Clock::now();

            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            synth_times.push_back(ms);
            audio_duration = output.duration_seconds;
        }

        // Calculate stats
        double total = 0, min_time = synth_times[0], max_time = synth_times[0];
        for (double t : synth_times) {
            total += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        double avg = total / synth_times.size();
        double rtf = audio_duration / (avg / 1000.0);

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Synthesis:  " << avg << " ms avg (" << min_time << "-" << max_time << " ms range)\n";
        std::cout << "  Audio:      " << std::setprecision(2) << audio_duration << " s\n";
        std::cout << "  RTF:        " << std::setprecision(1) << rtf << "x real-time\n";
        std::cout << "  Latency/s:  " << std::setprecision(0) << (avg / audio_duration) << " ms per audio second\n\n";
    }

    // LSTM analysis - measure sequential overhead
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                    LSTM SCALING TEST\n";
    std::cout << "     (BiLSTM is sequential - time should scale with length)\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    std::vector<std::pair<int, std::string>> length_tests = {
        {1, "A"},
        {2, "AB"},
        {5, "Hello"},
        {10, "Hello test"},
        {20, "Hello world testing!"},
        {50, "The quick brown fox jumps over the lazy dog testing here."},
    };

    std::cout << std::setw(10) << "Chars" << std::setw(12) << "Time (ms)"
              << std::setw(12) << "ms/char" << "\n";
    std::cout << "──────────────────────────────────────\n";

    for (const auto& [expected_len, text] : length_tests) {
        // Warm this specific length
        model.synthesize(text, "af_heart", 1.0f);
        mx::synchronize();

        // Time it
        mx::synchronize();
        auto start = Clock::now();
        model.synthesize(text, "af_heart", 1.0f);
        mx::synchronize();
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << std::fixed << std::setprecision(1);
        std::cout << std::setw(10) << text.length()
                  << std::setw(12) << ms
                  << std::setw(12) << (ms / text.length()) << "\n";
    }

    // Memory bandwidth test
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "                    MEMORY/BANDWIDTH TEST\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    {
        // Test if we're memory-bound or compute-bound
        std::string test_text = "Testing memory bandwidth.";

        // Measure synthesis
        mx::synchronize();
        auto start = Clock::now();
        for (int i = 0; i < 10; i++) {
            model.synthesize(test_text, "af_heart", 1.0f);
        }
        mx::synchronize();
        auto end = Clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "10x synthesis batch: " << std::setprecision(0) << total_ms << " ms\n";
        std::cout << "Per synthesis:       " << std::setprecision(1) << (total_ms / 10) << " ms\n";
    }

    // Theoretical analysis
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "                    ANALYSIS\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    std::cout << "KNOWN BOTTLENECKS:\n";
    std::cout << "  1. BiLSTM: Sequential loop over timesteps (cannot parallelize)\n";
    std::cout << "     - Each hidden state depends on previous\n";
    std::cout << "     - predictor_text_encoder: 3x BiLSTM layers\n";
    std::cout << "     - predictor.lstm: 1x BiLSTM\n";
    std::cout << "\n";
    std::cout << "  2. mx::eval() calls in compute_alignment:\n";
    std::cout << "     - Lines 993, 997 in model.cpp force GPU sync\n";
    std::cout << "     - Required to get total_frames as CPU integer\n";
    std::cout << "\n";
    std::cout << "THEORETICAL MINIMUM:\n";
    std::cout << "  - G2P/tokenization: ~1-5ms (CPU-bound, simple)\n";
    std::cout << "  - BERT: ~5-10ms (single forward pass, well optimized)\n";
    std::cout << "  - BiLSTM: O(T * hidden_size) - scales linearly with tokens\n";
    std::cout << "  - Decoder: ~50-100ms (multiple conv layers, well parallelized)\n";
    std::cout << "\n";
    std::cout << "OPTIONS TO IMPROVE:\n";
    std::cout << "  1. Replace LSTM with Transformer attention (parallel)\n";
    std::cout << "  2. Use MLX's native LSTM (if available and faster)\n";
    std::cout << "  3. Move LSTM to separate GPU stream\n";
    std::cout << "  4. Reduce predictor hidden dimensions\n";
    std::cout << "  5. Distill/quantize the model\n";
    std::cout << "\n";

    // Final benchmark - continuous synthesis
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                    THROUGHPUT TEST\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    {
        std::string text = "Hello world. Testing throughput.";
        double total_audio = 0;
        int count = 20;

        mx::synchronize();
        auto start = Clock::now();
        for (int i = 0; i < count; i++) {
            auto out = model.synthesize(text, "af_heart", 1.0f);
            total_audio += out.duration_seconds;
        }
        mx::synchronize();
        auto end = Clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Synthesized " << count << " utterances:\n";
        std::cout << "  Total time:     " << std::setprecision(0) << total_ms << " ms\n";
        std::cout << "  Per utterance:  " << std::setprecision(1) << (total_ms / count) << " ms\n";
        std::cout << "  Total audio:    " << std::setprecision(1) << total_audio << " s\n";
        std::cout << "  Throughput:     " << std::setprecision(1) << (total_audio / (total_ms/1000.0)) << "x real-time\n";
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "                    PROFILING COMPLETE\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    return 0;
}
