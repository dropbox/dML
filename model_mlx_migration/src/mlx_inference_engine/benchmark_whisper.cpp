// Copyright 2024-2025 Andrew Yates
//
// Whisper STT Benchmark
// Measures encoder and decoder performance separately

#include "whisper_model.h"
#include <iostream>
#include <chrono>
#include <numeric>
#include <cmath>

using namespace whisper;

void benchmark_encoder(WhisperModel& model, int n_warmup = 3, int n_runs = 10) {
    std::cout << "\n=== Encoder Benchmark ===\n";

    // Create 30-second mel spectrogram (3000 frames x 128 mels)
    int n_frames = 3000;
    int n_mels = 128;
    auto mel = mx::random::normal({1, n_frames, n_mels});
    mx::eval(mel);

    std::cout << "Input: [1, " << n_frames << ", " << n_mels << "] mel spectrogram\n";

    // Warmup
    std::cout << "Warmup (" << n_warmup << " runs)... ";
    for (int i = 0; i < n_warmup; ++i) {
        auto output = model.encode(mel);
        mx::eval(output);
    }
    std::cout << "done\n";

    // Benchmark
    std::vector<double> times;
    times.reserve(n_runs);

    for (int i = 0; i < n_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto output = model.encode(mel);
        mx::eval(output);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Statistics
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());

    double sq_sum = 0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double std_dev = std::sqrt(sq_sum / times.size());

    std::cout << "Runs: " << n_runs << "\n";
    std::cout << "Mean: " << mean << " ms\n";
    std::cout << "Min:  " << min_t << " ms\n";
    std::cout << "Max:  " << max_t << " ms\n";
    std::cout << "Std:  " << std_dev << " ms\n";

    // Output shape
    auto output = model.encode(mel);
    mx::eval(output);
    std::cout << "Output: [" << output.shape()[0] << ", " << output.shape()[1]
              << ", " << output.shape()[2] << "]\n";
}

void benchmark_decoder_step(WhisperModel& model, int n_warmup = 5, int n_runs = 50) {
    std::cout << "\n=== Decoder Step Benchmark (single token) ===\n";

    // Create encoder output (1500 frames after conv stride 2)
    int enc_seq = 1500;
    int n_state = 1280;  // Large-v3 turbo
    auto encoder_output = mx::random::normal({1, enc_seq, n_state});
    mx::eval(encoder_output);

    // Single token input
    auto tokens = mx::array({50258}, mx::int32);  // SOT token
    tokens = mx::reshape(tokens, {1, 1});
    mx::eval(tokens);

    std::cout << "Encoder output: [1, " << enc_seq << ", " << n_state << "]\n";
    std::cout << "Token input: [1, 1]\n";

    // Warmup
    std::cout << "Warmup (" << n_warmup << " runs)... ";
    for (int i = 0; i < n_warmup; ++i) {
        auto logits = model.decode(tokens, encoder_output);
        mx::eval(logits);
    }
    std::cout << "done\n";

    // Benchmark without KV cache (first step)
    std::vector<double> times_no_cache;
    for (int i = 0; i < n_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto logits = model.decode(tokens, encoder_output);
        mx::eval(logits);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times_no_cache.push_back(ms);
    }

    double mean_no_cache = std::accumulate(times_no_cache.begin(), times_no_cache.end(), 0.0) / times_no_cache.size();
    double min_no_cache = *std::min_element(times_no_cache.begin(), times_no_cache.end());

    std::cout << "\nWithout KV cache (first step):\n";
    std::cout << "  Mean: " << mean_no_cache << " ms\n";
    std::cout << "  Min:  " << min_no_cache << " ms\n";
}

void benchmark_generate(WhisperModel& model, int n_warmup = 2, int n_runs = 5) {
    std::cout << "\n=== Full Generation Benchmark ===\n";

    // Create 30-second mel spectrogram
    int n_frames = 3000;
    int n_mels = 128;
    auto mel = mx::random::normal({1, n_frames, n_mels});
    mx::eval(mel);

    std::cout << "Input: [1, " << n_frames << ", " << n_mels << "] mel (30s audio)\n";
    std::cout << "Max tokens: 100\n";

    // Warmup
    std::cout << "Warmup (" << n_warmup << " runs)... ";
    for (int i = 0; i < n_warmup; ++i) {
        auto tokens = model.generate(mel, "en", "transcribe", 100);
    }
    std::cout << "done\n";

    // Benchmark
    std::vector<double> times;
    std::vector<int> token_counts;

    for (int i = 0; i < n_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto tokens = model.generate(mel, "en", "transcribe", 100);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
        token_counts.push_back(tokens.size());
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());
    int avg_tokens = std::accumulate(token_counts.begin(), token_counts.end(), 0) / token_counts.size();

    std::cout << "\nResults:\n";
    std::cout << "  Mean time: " << mean << " ms\n";
    std::cout << "  Min time:  " << min_t << " ms\n";
    std::cout << "  Avg tokens: " << avg_tokens << "\n";
    std::cout << "  Tokens/sec: " << (avg_tokens * 1000.0 / mean) << "\n";
}

void benchmark_real_audio(WhisperModel& model, const std::string& audio_path) {
    std::cout << "\n=== Real Audio Benchmark ===\n";
    std::cout << "Audio: " << audio_path << "\n";

    // Load audio
    auto audio = whisper::audio::load_audio(audio_path);
    if (audio.empty()) {
        std::cerr << "Failed to load audio\n";
        return;
    }

    float duration = audio.size() / 16000.0f;
    std::cout << "Duration: " << duration << " seconds\n";

    // Compute mel
    auto mel = whisper::audio::log_mel_spectrogram(audio, 128, 400, 160);
    mx::eval(mel);

    // Pad to 3000 frames
    if (mel.shape()[0] < 3000) {
        mel = whisper::audio::pad_or_trim(mel, 3000);
    }
    mel = mx::reshape(mel, {1, static_cast<int>(mel.shape()[0]), static_cast<int>(mel.shape()[1])});
    mx::eval(mel);

    // Run 5 times
    std::vector<double> times;
    std::string last_text;

    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto tokens = model.generate(mel, "en", "transcribe", 200);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());

    // RTF = processing_time / audio_duration
    double rtf = (mean / 1000.0) / duration;

    std::cout << "\nResults:\n";
    std::cout << "  Mean time: " << mean << " ms\n";
    std::cout << "  Min time:  " << min_t << " ms\n";
    std::cout << "  RTF: " << rtf << " (lower is better, <1 = real-time)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <whisper_model_path> [audio_file]\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string audio_path = (argc > 2) ? argv[2] : "";

    std::cout << "=== Whisper STT Benchmark ===\n";
    std::cout << "Model: " << model_path << "\n";

    // Load model
    std::cout << "\nLoading model... ";
    auto start = std::chrono::high_resolution_clock::now();
    auto model = WhisperModel::load(model_path);
    auto end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "done (" << load_ms << " ms)\n";

    // Print config
    auto config = model.config();
    std::cout << "\nModel config:\n";
    std::cout << "  n_audio_layer: " << config.n_audio_layer << "\n";
    std::cout << "  n_text_layer: " << config.n_text_layer << "\n";
    std::cout << "  n_audio_state: " << config.n_audio_state << "\n";
    std::cout << "  n_mels: " << config.n_mels << "\n";

    // Run benchmarks
    benchmark_encoder(model);
    benchmark_decoder_step(model);
    benchmark_generate(model);

    if (!audio_path.empty()) {
        benchmark_real_audio(model, audio_path);
    }

    std::cout << "\n=== Benchmark Complete ===\n";
    return 0;
}
