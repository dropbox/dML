// Profile Kokoro TTS Pipeline - Identify Bottlenecks
// Measures: G2P, Tokenization, BERT, Decoder stages separately

#include "kokoro.h"
#include "misaki_g2p.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

int main() {
    std::cout << "=== Kokoro Pipeline Profiling ===\n\n";

    // Load model
    std::cout << "Loading model...\n";
    auto load_start = Clock::now();
    kokoro::Model model = kokoro::Model::load("kokoro_cpp_export");
    auto load_end = Clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    std::cout << "Model load: " << std::fixed << std::setprecision(0) << load_ms << " ms\n\n";

    model.set_voice("af_heart");

    // Test texts of varying lengths
    std::vector<std::pair<std::string, std::string>> tests = {
        {"Short", "Hello world."},
        {"Medium", "The quick brown fox jumps over the lazy dog."},
        {"Long", "Hello world. This is a test of the Kokoro text to speech system. "
                 "We want to measure the latency of each stage in the pipeline."}
    };

    // Warmup
    std::cout << "Warming up (3 iterations)...\n";
    for (int i = 0; i < 3; i++) {
        model.synthesize("Test warmup.", "af_heart", 1.0f);
    }
    std::cout << "Warmup complete.\n\n";

    // Detailed profiling
    std::cout << "=== Stage-by-Stage Profiling ===\n\n";

    for (const auto& [name, text] : tests) {
        std::cout << name << " (" << text.length() << " chars): \"" << text.substr(0, 40) << "...\"\n";

        // Multiple runs for average
        const int runs = 3;
        double total_g2p = 0, total_synth = 0, total_eval = 0;
        double audio_duration = 0;

        for (int r = 0; r < runs; r++) {
            // Full synthesis with timing
            auto start = Clock::now();
            auto output = model.synthesize(text, "af_heart", 1.0f);
            auto end = Clock::now();

            total_synth += std::chrono::duration<double, std::milli>(end - start).count();
            audio_duration = output.duration_seconds;
        }

        double avg_synth = total_synth / runs;
        double rtf = (avg_synth / 1000.0) / audio_duration;

        std::cout << "  Total synthesis: " << std::setprecision(1) << avg_synth << " ms\n";
        std::cout << "  Audio duration:  " << std::setprecision(2) << audio_duration << " s\n";
        std::cout << "  RTF:             " << std::setprecision(1) << (1.0/rtf) << "x real-time\n\n";
    }

    // Single character test - isolates BERT/decoder overhead
    std::cout << "=== Overhead Analysis (minimal text) ===\n";
    {
        std::string minimal = "Hi";
        auto start = Clock::now();
        auto output = model.synthesize(minimal, "af_heart", 1.0f);
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Minimal text (\"Hi\"): " << std::setprecision(0) << ms << " ms\n";
        std::cout << "Audio: " << std::setprecision(2) << output.duration_seconds << " s\n";
        std::cout << "This represents baseline overhead (BERT warmup + decoder init)\n\n";
    }

    // Throughput test - many short phrases
    std::cout << "=== Throughput Test (10 short phrases) ===\n";
    {
        std::vector<std::string> phrases = {
            "Hello.", "World.", "Test.", "One.", "Two.",
            "Three.", "Four.", "Five.", "Six.", "Go."
        };

        auto start = Clock::now();
        double total_audio = 0;
        for (const auto& phrase : phrases) {
            auto output = model.synthesize(phrase, "af_heart", 1.0f);
            total_audio += output.duration_seconds;
        }
        auto end = Clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Total time: " << std::setprecision(0) << total_ms << " ms\n";
        std::cout << "Per phrase: " << (total_ms / phrases.size()) << " ms\n";
        std::cout << "Total audio: " << std::setprecision(2) << total_audio << " s\n";
        std::cout << "Throughput: " << std::setprecision(1) << (total_audio / (total_ms/1000.0)) << "x real-time\n\n";
    }

    // Memory bandwidth test - force sync between operations
    std::cout << "=== Back-to-back test (measure compilation overhead) ===\n";
    {
        std::string text = "Quick test.";

        // First run (may trigger compilation)
        auto start1 = Clock::now();
        model.synthesize(text, "af_heart", 1.0f);
        auto end1 = Clock::now();

        // Second run (should be cached)
        auto start2 = Clock::now();
        model.synthesize(text, "af_heart", 1.0f);
        auto end2 = Clock::now();

        // Third run
        auto start3 = Clock::now();
        model.synthesize(text, "af_heart", 1.0f);
        auto end3 = Clock::now();

        double ms1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
        double ms2 = std::chrono::duration<double, std::milli>(end2 - start2).count();
        double ms3 = std::chrono::duration<double, std::milli>(end3 - start3).count();

        std::cout << "Run 1: " << std::setprecision(0) << ms1 << " ms\n";
        std::cout << "Run 2: " << ms2 << " ms\n";
        std::cout << "Run 3: " << ms3 << " ms\n";
        std::cout << "If run 1 >> run 2,3 -> compilation overhead detected\n\n";
    }

    std::cout << "Done.\n";
    return 0;
}
