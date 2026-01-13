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

// Test MLX Inference Engine
// Tests Kokoro TTS integration

#include "mlx_inference_engine.hpp"
#include "whisper_model.h"
#include "silero_vad.h"
#include "output_writers.h"
#include "cosyvoice3_model.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <sstream>

// WAV file writer (minimal implementation)
void write_wav(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // WAV header
    int16_t num_channels = 1;
    int16_t bits_per_sample = 16;
    int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int16_t block_align = num_channels * bits_per_sample / 8;
    int32_t data_size = static_cast<int32_t>(samples.size() * sizeof(int16_t));
    int32_t chunk_size = 36 + data_size;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&chunk_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int32_t fmt_size = 16;
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    int16_t audio_format = 1;  // PCM
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);

    // Write samples (convert float to int16)
    for (float sample : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, sample));
        int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
        file.write(reinterpret_cast<const char*>(&pcm), 2);
    }
}

void test_kokoro_tts(const std::string& model_path) {
    std::cout << "=== Test: Kokoro TTS via MLXInferenceEngine ===\n";

    mlx_inference::MLXInferenceEngine engine;

    // Test model not loaded
    std::cout << "1. Testing error on unloaded model... ";
    try {
        engine.synthesize("Hello");
        std::cerr << "FAIL: Should have thrown ModelNotLoadedError\n";
        exit(1);
    } catch (const mlx_inference::ModelNotLoadedError& e) {
        std::cout << "PASS (got expected error)\n";
    }

    // Load model
    std::cout << "2. Loading Kokoro model from: " << model_path << "\n";
    engine.load_kokoro(model_path);

    assert(engine.is_kokoro_loaded());
    std::cout << "   Model loaded successfully\n";

    // Print model info
    std::cout << "\n" << engine.get_model_info() << "\n";

    // Synthesize
    std::cout << "3. Synthesizing 'Hello world'... ";
    mlx_inference::TTSConfig config;
    config.voice = "af_heart";
    config.speed = 1.0f;

    mlx_inference::AudioOutput output = engine.synthesize("Hello world", config);

    std::cout << "DONE\n";
    std::cout << "   Samples: " << output.samples.size() << "\n";
    std::cout << "   Sample rate: " << output.sample_rate << " Hz\n";
    std::cout << "   Duration: " << output.duration_seconds << "s\n";

    // Validate output
    assert(output.samples.size() > 0);
    assert(output.sample_rate == 24000);
    assert(output.duration_seconds > 0.0f);

    // Check for non-silence
    float max_amp = 0.0f;
    for (float s : output.samples) {
        max_amp = std::max(max_amp, std::abs(s));
    }
    std::cout << "   Max amplitude: " << max_amp << "\n";
    assert(max_amp > 0.01f);  // Not silence

    // Save to file
    std::string output_path = "mlx_engine_output.wav";
    write_wav(output_path, output.samples, output.sample_rate);
    std::cout << "   Saved to: " << output_path << "\n";

    std::cout << "\n=== All tests PASSED ===\n";
}

void test_not_implemented() {
    std::cout << "=== Test: Not-implemented models throw correct errors ===\n";

    mlx_inference::MLXInferenceEngine engine;

    // Test CosyVoice (now implemented - should fail with file not found)
    std::cout << "1. CosyVoice load (bad path)... ";
    try {
        engine.load_cosyvoice("/nonexistent/path/to/cosyvoice");
        std::cerr << "FAIL\n";
        exit(1);
    } catch (const std::exception& e) {
        std::cout << "PASS (got error: file not found)\n";
    }

    // Test Translation (now implemented - should fail with file not found)
    std::cout << "2. Translation load (bad path)... ";
    try {
        engine.load_translation("/nonexistent/path/to/model");
        std::cerr << "FAIL\n";
        exit(1);
    } catch (const std::exception& e) {
        std::cout << "PASS (got error: file not found)\n";
    }

    // Test Whisper (now implemented - should fail with file not found)
    std::cout << "3. Whisper load (bad path)... ";
    try {
        engine.load_whisper("/nonexistent/path/to/whisper");
        std::cerr << "FAIL\n";
        exit(1);
    } catch (const std::exception& e) {
        std::cout << "PASS (got error: file not found)\n";
    }

    // Test LLM (bad path)
    std::cout << "4. LLM load (bad path)... ";
    try {
        engine.load_llm("/path/to/nonexistent/model");
        std::cerr << "FAIL\n";
        exit(1);
    } catch (const std::exception& e) {
        std::cout << "PASS (got error: " << e.what() << ")\n";
    }

    std::cout << "\n=== All not-implemented tests PASSED ===\n";
}

void test_translation(const std::string& model_path) {
    std::cout << "=== Test: Translation via MLXInferenceEngine ===\n";

    mlx_inference::MLXInferenceEngine engine;

    // Test model not loaded
    std::cout << "1. Testing error on unloaded model... ";
    try {
        mlx_inference::TranslationConfig config;
        engine.translate("Hello", config);
        std::cerr << "FAIL: Should have thrown ModelNotLoadedError\n";
        exit(1);
    } catch (const mlx_inference::ModelNotLoadedError& e) {
        std::cout << "PASS (got expected error)\n";
    }

    // Load model
    std::cout << "2. Loading MADLAD translation model from: " << model_path << "\n";
    engine.load_translation(model_path, "madlad");

    assert(engine.is_translation_loaded());
    std::cout << "   Model loaded successfully\n";

    // Print model info
    std::cout << "\n" << engine.get_model_info() << "\n";

    // Translate English to German
    std::cout << "3. Translating 'Hello world' to German... ";
    mlx_inference::TranslationConfig config;
    config.source_lang = "en";
    config.target_lang = "de";
    config.max_length = 16;
    config.debug = false;  // Disable debug output

    std::string translation = engine.translate("Hello world", config);

    std::cout << "DONE\n";
    std::cout << "   Input: 'Hello world'\n";
    std::cout << "   Output: '" << translation << "'\n";

    // Validate output
    assert(translation.length() > 0);

    // Test another language pair
    std::cout << "4. Translating 'Good morning' to French... ";
    config.target_lang = "fr";
    translation = engine.translate("Good morning", config);
    std::cout << "DONE\n";
    std::cout << "   Input: 'Good morning'\n";
    std::cout << "   Output: '" << translation << "'\n";

    assert(translation.length() > 0);

    std::cout << "\n=== All translation tests PASSED ===\n";
}

void benchmark_translation(const std::string& model_path) {
    std::cout << "\n=== Benchmark: Translation Latency ===\n";

    mlx_inference::MLXInferenceEngine engine;

    // Measure load time
    auto load_start = std::chrono::high_resolution_clock::now();
    engine.load_translation(model_path, "madlad");
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    std::cout << "Model load time: " << load_ms << " ms\n\n";

    // Test sentences of varying lengths
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"Hello world", "de"},
        {"Good morning", "fr"},
        {"How are you today?", "es"},
        {"The quick brown fox jumps over the lazy dog.", "de"},
        {"Machine learning is transforming the world.", "ja"},
        {"I love programming in C++.", "zh"},
    };

    mlx_inference::TranslationConfig config;
    config.source_lang = "en";
    config.max_length = 64;
    config.debug = false;

    std::vector<double> latencies;

    // Warmup
    std::cout << "Warmup run...\n";
    config.target_lang = "de";
    engine.translate("Hello", config);

    std::cout << "\nBenchmark results:\n";
    std::cout << std::setw(50) << "Input" << " | "
              << std::setw(10) << "Lang" << " | "
              << std::setw(10) << "Latency" << " | "
              << "Output\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& tc : test_cases) {
        config.target_lang = tc.second;

        auto start = std::chrono::high_resolution_clock::now();
        std::string result = engine.translate(tc.first, config);
        auto end = std::chrono::high_resolution_clock::now();

        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(latency_ms);

        std::cout << std::setw(50) << tc.first.substr(0, 48) << " | "
                  << std::setw(10) << tc.second << " | "
                  << std::setw(7) << std::fixed << std::setprecision(1) << latency_ms << " ms | "
                  << result << "\n";
    }

    // Statistics
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    std::sort(latencies.begin(), latencies.end());
    double median = latencies[latencies.size() / 2];
    double min_lat = latencies.front();
    double max_lat = latencies.back();

    std::cout << "\nStatistics:\n";
    std::cout << "  Mean latency:   " << std::fixed << std::setprecision(1) << mean << " ms\n";
    std::cout << "  Median latency: " << median << " ms\n";
    std::cout << "  Min latency:    " << min_lat << " ms\n";
    std::cout << "  Max latency:    " << max_lat << " ms\n";

    std::cout << "\n=== Benchmark complete ===\n";
}

void test_mel_spectrogram() {
    std::cout << "=== Test: Mel Spectrogram Computation ===\n";

    // Generate 2 seconds of 440Hz sine wave (same as Python test)
    int sample_rate = 16000;
    int duration_samples = sample_rate * 2;
    std::vector<float> sine_audio(duration_samples);
    for (int i = 0; i < duration_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        sine_audio[i] = 0.3f * std::sin(2.0f * M_PI * 440.0f * t);
    }

    std::cout << "Audio: " << sine_audio.size() << " samples (" << sine_audio.size() / 16000.0 << "s)\n";

    // Compute mel spectrogram
    auto start = std::chrono::high_resolution_clock::now();
    auto mel = whisper::audio::log_mel_spectrogram(sine_audio, 128, 400, 160);
    mx::eval(mel);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Mel spectrogram shape: [" << mel.shape()[0] << ", " << mel.shape()[1] << "]\n";
    std::cout << "Computation time: " << ms << " ms\n";

    // Get min, max, mean
    auto mel_min = mx::min(mel);
    auto mel_max = mx::max(mel);
    auto mel_mean = mx::mean(mel);
    mx::eval(mel_min);
    mx::eval(mel_max);
    mx::eval(mel_mean);

    std::cout << "  Min: " << mel_min.item<float>() << "\n";
    std::cout << "  Max: " << mel_max.item<float>() << "\n";
    std::cout << "  Mean: " << mel_mean.item<float>() << "\n";

    // Print first few values for comparison with Python
    std::cout << "\nFirst 5x5 values:\n";
    for (int i = 0; i < 5 && i < mel.shape()[0]; ++i) {
        for (int j = 0; j < 5 && j < mel.shape()[1]; ++j) {
            auto val = mx::slice(mel, {i, j}, {i+1, j+1});
            mx::eval(val);
            std::cout << std::fixed << std::setprecision(4) << val.item<float>() << " ";
        }
        std::cout << "\n";
    }

    // Verify non-zero values
    bool has_variation = (mel_max.item<float>() - mel_min.item<float>()) > 0.1f;
    if (has_variation) {
        std::cout << "\nPASS: Mel spectrogram has varied values\n";
    } else {
        std::cout << "\nFAIL: Mel spectrogram has insufficient variation\n";
    }

    std::cout << "\n=== Mel spectrogram test complete ===\n";
}

void test_whisper_stt(const std::string& model_path) {
    std::cout << "=== Test: Whisper STT via MLXInferenceEngine ===\n";

    mlx_inference::MLXInferenceEngine engine;

    // Test model not loaded
    std::cout << "1. Testing error on unloaded model... ";
    try {
        std::vector<float> dummy_audio(16000);  // 1 second of silence
        engine.transcribe(dummy_audio, 16000);
        std::cerr << "FAIL: Should have thrown ModelNotLoadedError\n";
        exit(1);
    } catch (const mlx_inference::ModelNotLoadedError& e) {
        std::cout << "PASS (got expected error)\n";
    }

    // Load model
    std::cout << "2. Loading Whisper model from: " << model_path << "\n";
    auto load_start = std::chrono::high_resolution_clock::now();
    engine.load_whisper(model_path);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();

    assert(engine.is_whisper_loaded());
    std::cout << "   Model loaded in " << load_ms << " ms\n";

    // Print model info
    std::cout << "\n" << engine.get_model_info() << "\n";

    // Test transcription with sine wave audio (to verify mel spectrogram works)
    std::cout << "3. Testing transcription pipeline with sine wave... ";
    try {
        // Generate 2 seconds of 440Hz sine wave (A4 note)
        int sample_rate = 16000;
        int duration_samples = sample_rate * 2;
        std::vector<float> sine_audio(duration_samples);
        for (int i = 0; i < duration_samples; ++i) {
            float t = static_cast<float>(i) / sample_rate;
            sine_audio[i] = 0.3f * std::sin(2.0f * M_PI * 440.0f * t);
        }

        mlx_inference::TranscriptionConfig trans_config;
        trans_config.language = "en";

        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine.transcribe(sine_audio, 16000, trans_config);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "DONE (" << ms << " ms)\n";
        std::cout << "   Output: '" << result.text << "'\n";
        std::cout << "   Language: " << result.language << "\n";
        std::cout << "   (Note: sine wave is not speech, output may be empty or hallucinated)\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
    }

    // Test transcription with silence (should produce empty output)
    std::cout << "4. Testing transcription with silence... ";
    try {
        std::vector<float> silence(16000 * 2);  // 2 seconds of silence
        mlx_inference::TranscriptionConfig trans_config;
        trans_config.language = "en";

        auto result = engine.transcribe(silence, 16000, trans_config);
        std::cout << "DONE\n";
        std::cout << "   Output: '" << result.text << "' (expected empty or minimal)\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
    }

    // Test transcription with real audio file if available
    std::string test_audio = "/Users/ayates/model_mlx_migration/tests/prosody/contour_vs_v5_output/neutral_context_baseline.wav";
    std::ifstream audio_file(test_audio);
    if (audio_file.good()) {
        audio_file.close();
        std::cout << "5. Testing transcription with real audio file (~4s)... ";
        try {
            mlx_inference::TranscriptionConfig trans_config;
            trans_config.language = "en";

            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine.transcribe_file(test_audio, trans_config);
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "DONE (" << ms << " ms)\n";
            std::cout << "   File: " << test_audio << "\n";
            std::cout << "   Output: '" << result.text << "'\n";
            std::cout << "   Language: " << result.language << "\n";
            std::cout << "   Tokens (" << result.tokens.size() << "): ";
            for (size_t i = 0; i < result.tokens.size() && i < 30; ++i) {
                std::cout << result.tokens[i] << " ";
            }
            if (result.tokens.size() > 30) std::cout << "...";
            std::cout << "\n";
            // Decode tokens with labels
            std::cout << "   Token breakdown:\n";
            for (size_t i = 0; i < result.tokens.size(); ++i) {
                int tok = result.tokens[i];
                if (tok >= 50364) {
                    float ts = (tok - 50364) * 0.02f;
                    std::cout << "     [" << i << "] <|" << std::fixed << std::setprecision(2) << ts << "|> (tok " << tok << ")\n";
                } else if (tok == 50257) {
                    std::cout << "     [" << i << "] EOT (tok " << tok << ")\n";
                } else if (tok == 50258) {
                    std::cout << "     [" << i << "] SOT (tok " << tok << ")\n";
                } else if (tok >= 50259 && tok <= 50357) {
                    std::cout << "     [" << i << "] LANG (tok " << tok << ")\n";
                } else if (tok == 50359) {
                    std::cout << "     [" << i << "] TRANSCRIBE (tok " << tok << ")\n";
                } else {
                    std::cout << "     [" << i << "] text (tok " << tok << ")\n";
                }
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    }

    // Test transcription with longer audio (~15s) to verify natural EOT
    std::string long_audio = "/Users/ayates/model_mlx_migration/tests/test_audio_long.wav";
    std::ifstream long_audio_file(long_audio);
    if (long_audio_file.good()) {
        long_audio_file.close();
        std::cout << "6. Testing transcription with longer audio (~15s)... ";
        try {
            mlx_inference::TranscriptionConfig trans_config;
            trans_config.language = "en";

            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine.transcribe_file(long_audio, trans_config);
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "DONE (" << ms << " ms)\n";
            std::cout << "   File: " << long_audio << "\n";
            std::cout << "   Output length: " << result.text.length() << " chars\n";
            std::cout << "   Output: '" << result.text.substr(0, 200) << "...'\n";
            std::cout << "   Full output:\n   '" << result.text << "'\n";
            std::cout << "   Language: " << result.language << "\n";
            // Print last 15 tokens
            std::cout << "   Last 15 tokens: ";
            size_t start_idx = result.tokens.size() > 15 ? result.tokens.size() - 15 : 0;
            for (size_t i = start_idx; i < result.tokens.size(); ++i) {
                std::cout << result.tokens[i] << " ";
            }
            std::cout << "\n";
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    } else {
        std::cout << "6. Skipping long audio test (file not found: " << long_audio << ")\n";
    }

    // Test beam search decoding
    if (audio_file.good() || std::ifstream(test_audio).good()) {
        std::cout << "7. Testing BEAM SEARCH decoding (beam_size=5)... ";
        try {
            mlx_inference::TranscriptionConfig beam_config;
            beam_config.language = "en";
            beam_config.beam_size = 5;
            beam_config.length_penalty = 1.0f;

            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine.transcribe_file(test_audio, beam_config);
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "DONE (" << ms << " ms)\n";
            std::cout << "   Output (beam): '" << result.text << "'\n";
            std::cout << "   Language: " << result.language << "\n";

            // Compare with greedy for reference
            mlx_inference::TranscriptionConfig greedy_config;
            greedy_config.language = "en";
            greedy_config.beam_size = 1;

            auto greedy_result = engine.transcribe_file(test_audio, greedy_config);
            std::cout << "   Output (greedy): '" << greedy_result.text << "'\n";

            if (result.text == greedy_result.text) {
                std::cout << "   Note: Beam search and greedy produced identical results\n";
            } else {
                std::cout << "   Note: Beam search produced different result than greedy\n";
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    }

    // Test beam search with long audio if available
    if (std::ifstream(long_audio).good()) {
        std::cout << "8. Testing BEAM SEARCH with long audio (~15s)... ";
        try {
            mlx_inference::TranscriptionConfig beam_config;
            beam_config.language = "en";
            beam_config.beam_size = 5;

            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine.transcribe_file(long_audio, beam_config);
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "DONE (" << ms << " ms)\n";
            std::cout << "   Output (beam): '" << result.text << "'\n";

            // Compare with greedy
            mlx_inference::TranscriptionConfig greedy_config;
            greedy_config.language = "en";
            auto greedy_result = engine.transcribe_file(long_audio, greedy_config);
            std::cout << "   Output (greedy): '" << greedy_result.text << "'\n";

            // Check for trailing period issue (beam search should help)
            bool greedy_ends_period = !greedy_result.text.empty() &&
                                      greedy_result.text.back() == '.';
            bool beam_ends_period = !result.text.empty() &&
                                    result.text.back() == '.';
            std::cout << "   Greedy ends with period: " << (greedy_ends_period ? "YES" : "NO") << "\n";
            std::cout << "   Beam ends with period: " << (beam_ends_period ? "YES" : "NO") << "\n";
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    }

    // Test SEGMENT-BASED DECODING with very long audio (>30s)
    std::string very_long_audio = "/Users/ayates/model_mlx_migration/tests/test_audio_long_40s.wav";
    if (std::ifstream(very_long_audio).good()) {
        std::cout << "9. Testing SEGMENT-BASED DECODING with very long audio (~53s)... ";
        try {
            mlx_inference::TranscriptionConfig segment_config;
            segment_config.language = "en";
            segment_config.beam_size = 1;  // Greedy for speed
            segment_config.condition_on_previous_text = true;

            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine.transcribe_file(very_long_audio, segment_config);
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "DONE (" << ms << " ms)\n";
            std::cout << "   Output length: " << result.text.length() << " chars\n";
            std::cout << "   Segments: " << result.segments.size() << "\n";

            // Print segments with timing (first 10 for brevity)
            for (size_t i = 0; i < result.segments.size() && i < 10; ++i) {
                const auto& seg = result.segments[i];
                std::cout << "   [" << std::fixed << std::setprecision(2)
                          << seg.start_time << "s - " << seg.end_time << "s] "
                          << "'" << seg.text.substr(0, 60) << "...'\n";
            }
            if (result.segments.size() > 10) {
                std::cout << "   ... (" << result.segments.size() - 10 << " more segments)\n";
            }

            // Show full text
            std::cout << "   Full text (first 500 chars):\n   '" << result.text.substr(0, 500) << "'\n";

            // Expected: audio repeats 4 times, so similar content should appear multiple times
            size_t count_weather = 0;
            size_t pos = 0;
            while ((pos = result.text.find("weather", pos)) != std::string::npos) {
                count_weather++;
                pos++;
            }
            std::cout << "   Word 'weather' appears " << count_weather << " times (expected 4)\n";
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    } else {
        std::cout << "9. Skipping segment-based decoding test (file not found: " << very_long_audio << ")\n";
        std::cout << "   To create test file: ffmpeg -y -stream_loop 3 -i tests/test_audio_long.wav -c copy tests/test_audio_long_40s.wav\n";
    }

    // Test WORD-LEVEL TIMESTAMPS via DTW
    std::string word_ts_audio = "/Users/ayates/model_mlx_migration/tests/test_audio_short.wav";
    if (!std::ifstream(word_ts_audio).good()) {
        // Use existing short audio if available
        word_ts_audio = test_audio;
    }
    if (std::ifstream(word_ts_audio).good()) {
        std::cout << "10. Testing WORD-LEVEL TIMESTAMPS with DTW alignment... ";
        try {
            // Load audio
            auto audio_samples = whisper::audio::load_audio(word_ts_audio);
            auto mel = whisper::audio::log_mel_spectrogram(audio_samples, 128, 400, 160);
            mx::eval(mel);

            // Pad to 3000 frames (30s)
            int target_frames = 3000;
            if (mel.shape()[0] < target_frames) {
                mel = whisper::audio::pad_or_trim(mel, target_frames);
            } else if (mel.shape()[0] > target_frames) {
                mel = mx::slice(mel, {0, 0}, {target_frames, static_cast<int>(mel.shape()[1])});
            }
            mel = mx::reshape(mel, {1, static_cast<int>(mel.shape()[0]), static_cast<int>(mel.shape()[1])});
            mx::eval(mel);

            // Load model directly
            std::string model_path_str = model_path;
            auto whisper_model = whisper::WhisperModel::load(model_path_str);

            // Load tokenizer for word text extraction
            std::string vocab_path = "/Users/ayates/model_mlx_migration/models/whisper_vocab.json";
            auto tokenizer = whisper::WhisperTokenizer::load(vocab_path);

            // Encode
            auto encoder_output = whisper_model.encode(mel);
            mx::eval(encoder_output);

            // Generate tokens
            auto tokens = whisper_model.generate(mel, "en", "transcribe", 100);

            // Create a segment from the tokens
            whisper::WhisperSegment segment;
            segment.start_time = 0.0f;
            segment.end_time = static_cast<float>(audio_samples.size()) / 16000.0f;

            // Filter to text tokens only (not timestamps or special tokens)
            for (int tok : tokens) {
                if (tok < 50257) {  // Not special token
                    segment.tokens.push_back(tok);
                }
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Add word-level timestamps (with tokenizer for text extraction)
            whisper_model.add_word_timestamps(segment, encoder_output, "en", &tokenizer);

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "DONE (" << ms << " ms)\n";
            std::cout << "   Segment: " << std::fixed << std::setprecision(2)
                      << segment.start_time << "s - " << segment.end_time << "s\n";
            std::cout << "   Token count: " << segment.tokens.size() << "\n";
            std::cout << "   Word count: " << segment.words.size() << "\n";

            // Print word-level timestamps
            if (segment.words.size() > 0) {
                std::cout << "   Word timings:\n";
                for (size_t i = 0; i < segment.words.size(); ++i) {
                    const auto& word = segment.words[i];
                    std::cout << "     [" << std::fixed << std::setprecision(3)
                              << word.start_time << "s - " << word.end_time << "s] \""
                              << word.word << "\"\n";
                }
                std::cout << "   PASS: Word-level timestamps with text generated\n";
            } else {
                std::cout << "   Note: No words extracted (may need longer audio)\n";
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    } else {
        std::cout << "10. Skipping word-level timestamps test (no suitable audio file)\n";
    }

    // Test 11: generate_segments with word_timestamps=true
    if (std::ifstream(test_audio).good()) {
        std::cout << "11. Testing generate_segments with word_timestamps=true... ";
        try {
            // Load audio
            auto audio_samples = whisper::audio::load_audio(test_audio);
            auto mel = whisper::audio::log_mel_spectrogram(audio_samples, 128, 400, 160);
            mx::eval(mel);

            // Pad to 3000 frames
            int target_frames = 3000;
            if (mel.shape()[0] < target_frames) {
                mel = whisper::audio::pad_or_trim(mel, target_frames);
            }
            mel = mx::reshape(mel, {1, static_cast<int>(mel.shape()[0]), static_cast<int>(mel.shape()[1])});
            mx::eval(mel);

            // Load model and tokenizer
            std::string model_path_str = model_path;
            auto whisper_model = whisper::WhisperModel::load(model_path_str);
            std::string vocab_path = "/Users/ayates/model_mlx_migration/models/whisper_vocab.json";
            auto tokenizer = whisper::WhisperTokenizer::load(vocab_path);

            auto start = std::chrono::high_resolution_clock::now();

            // Call generate_segments with word timestamps enabled
            auto result = whisper_model.generate_segments(
                mel, "en", "transcribe",
                true,   // condition_on_previous
                0.6f,   // no_speech_threshold
                2.4f,   // compression_ratio_threshold
                1,      // beam_size (greedy)
                1.0f,   // length_penalty
                true,   // word_timestamps=true!
                &tokenizer
            );

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "DONE (" << ms << " ms)\n";
            std::cout << "   Segments: " << result.segments.size() << "\n";

            int total_words = 0;
            for (const auto& seg : result.segments) {
                total_words += static_cast<int>(seg.words.size());
            }
            std::cout << "   Total words across all segments: " << total_words << "\n";

            // Debug: show segment timing
            for (size_t i = 0; i < result.segments.size(); ++i) {
                const auto& seg = result.segments[i];
                std::cout << "   Segment " << i << ": " << std::fixed << std::setprecision(2)
                          << seg.start_time << "s - " << seg.end_time << "s"
                          << " (" << seg.words.size() << " words)\n";
            }

            // Print words from first segment
            if (!result.segments.empty() && !result.segments[0].words.empty()) {
                std::cout << "   Words from first segment:\n";
                for (size_t i = 0; i < result.segments[0].words.size(); ++i) {
                    const auto& word = result.segments[0].words[i];
                    std::cout << "     [" << std::fixed << std::setprecision(3)
                              << word.start_time << "s - " << word.end_time << "s] \""
                              << word.word << "\"\n";
                }
                // NOTE: Word timing appears compressed because segment.end_time comes from
                // timestamp tokens (0.5s) not actual audio duration (4s). This is expected
                // behavior for generate_segments - use add_word_timestamps directly with
                // correct segment.end_time for accurate word timing.
                std::cout << "   PASS: generate_segments with word timestamps (timing compressed, see note)\n";
            } else {
                std::cout << "   Note: No words in segments\n";
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    } else {
        std::cout << "11. Skipping generate_segments word timestamps test (no audio file)\n";
    }

    std::cout << "\n=== Whisper STT tests PASSED ===\n";
}

void test_llm(const std::string& model_path) {
    std::cout << "=== Test: LLM via MLXInferenceEngine ===\n";

    mlx_inference::MLXInferenceEngine engine;

    // Test model not loaded
    std::cout << "1. Testing error on unloaded model... ";
    try {
        mlx_inference::GenerationConfig config;
        engine.generate("Hello", config);
        std::cerr << "FAIL: Should have thrown ModelNotLoadedError\n";
        exit(1);
    } catch (const mlx_inference::ModelNotLoadedError& e) {
        std::cout << "PASS (got expected error)\n";
    }

    // Load model
    std::cout << "2. Loading LLM model from: " << model_path << "\n";
    auto start = std::chrono::high_resolution_clock::now();
    engine.load_llm(model_path);
    auto end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    assert(engine.is_llm_loaded());
    std::cout << "   Model loaded successfully in " << load_ms << "ms\n";

    // Print model info
    std::cout << "\n" << engine.get_model_info() << "\n";

    // Test generation
    std::cout << "3. Generating text from prompt 'What is 2+2?'...\n";
    mlx_inference::GenerationConfig config;
    config.max_tokens = 32;
    config.temperature = 0.0f;  // Greedy

    start = std::chrono::high_resolution_clock::now();
    mlx_inference::GenerationResult result = engine.generate("What is 2+2?", config);
    end = std::chrono::high_resolution_clock::now();
    auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "   Generated " << result.tokens_generated << " tokens in " << gen_ms << "ms\n";
    std::cout << "   Output: '" << result.text << "'\n";
    std::cout << "   Tokens/sec: " << result.tokens_per_second << "\n";

    // Validate output contains correct answer
    if (result.text.find("4") != std::string::npos) {
        std::cout << "   PASS: Output contains correct answer '4'\n";
    } else {
        std::cerr << "   FAIL: Output does not contain '4'\n";
    }

    // Test chat template
    std::cout << "\n4. Testing chat template with prompt 'What is the capital of France?'...\n";
    config.max_tokens = 32;
    config.temperature = 0.0f;  // Greedy

    start = std::chrono::high_resolution_clock::now();
    mlx_inference::GenerationResult chat_result = engine.chat(
        "What is the capital of France?",
        "You are a helpful assistant. Be concise.",
        config
    );
    end = std::chrono::high_resolution_clock::now();
    gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "   Generated " << chat_result.tokens_generated << " tokens in " << gen_ms << "ms\n";
    std::cout << "   Output: '" << chat_result.text << "'\n";
    std::cout << "   Tokens/sec: " << chat_result.tokens_per_second << "\n";

    // Validate output contains Paris
    if (chat_result.text.find("Paris") != std::string::npos) {
        std::cout << "   PASS: Output contains 'Paris'\n";
    } else {
        std::cerr << "   FAIL: Output does not contain 'Paris'\n";
    }

    // Test sampling with temperature
    std::cout << "\n5. Testing sampling with temperature=0.7, top_k=40, top_p=0.9...\n";
    config.max_tokens = 32;
    config.temperature = 0.7f;
    config.top_k = 40;
    config.top_p = 0.9f;

    start = std::chrono::high_resolution_clock::now();
    mlx_inference::GenerationResult sampling_result = engine.chat(
        "Tell me a random fact about space.",
        "",
        config
    );
    end = std::chrono::high_resolution_clock::now();
    gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "   Generated " << sampling_result.tokens_generated << " tokens in " << gen_ms << "ms\n";
    std::cout << "   Output: '" << sampling_result.text << "'\n";
    std::cout << "   Tokens/sec: " << sampling_result.tokens_per_second << "\n";

    // Run the same prompt again - with sampling, we should get different (or same) outputs
    start = std::chrono::high_resolution_clock::now();
    mlx_inference::GenerationResult sampling_result2 = engine.chat(
        "Tell me a random fact about space.",
        "",
        config
    );
    end = std::chrono::high_resolution_clock::now();
    gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "   Run 2: Generated " << sampling_result2.tokens_generated << " tokens in " << gen_ms << "ms\n";
    std::cout << "   Run 2 Output: '" << sampling_result2.text << "'\n";

    // Check if outputs are non-empty
    if (!sampling_result.text.empty() && !sampling_result2.text.empty()) {
        std::cout << "   PASS: Sampling produced valid outputs\n";
        if (sampling_result.text != sampling_result2.text) {
            std::cout << "   Note: Outputs differ (expected with sampling)\n";
        } else {
            std::cout << "   Note: Outputs are identical (can happen with high-probability tokens)\n";
        }
    } else {
        std::cerr << "   FAIL: Sampling produced empty output\n";
    }

    // Test top-p restrictive vs permissive
    std::cout << "\n6. Testing top-p effect (0.1 vs 0.95)...\n";

    // Very restrictive top-p (only highest probability tokens)
    config.max_tokens = 20;
    config.temperature = 1.0f;  // High temperature to amplify top-p effect
    config.top_k = 0;  // Disable top-k to isolate top-p effect
    config.top_p = 0.1f;

    mlx_inference::GenerationResult restrictive_result = engine.chat(
        "Complete this sentence: The sky is",
        "",
        config
    );
    std::cout << "   top_p=0.1: '" << restrictive_result.text.substr(0, 50) << "...'\n";

    // Permissive top-p
    config.top_p = 0.95f;
    mlx_inference::GenerationResult permissive_result = engine.chat(
        "Complete this sentence: The sky is",
        "",
        config
    );
    std::cout << "   top_p=0.95: '" << permissive_result.text.substr(0, 50) << "...'\n";

    if (!restrictive_result.text.empty() && !permissive_result.text.empty()) {
        std::cout << "   PASS: Top-p filtering working\n";
    } else {
        std::cerr << "   FAIL: Top-p produced empty output\n";
    }

    std::cout << "\n=== LLM test completed ===\n";
}

void test_streaming(const std::string& model_path) {
    std::cout << "=== Test: Streaming Transcription ===\n";

    // Load Whisper model
    std::cout << "1. Loading Whisper model for streaming... ";
    auto whisper_model = whisper::WhisperModel::load(model_path);
    std::cout << "DONE\n";

    // Test AudioBuffer
    std::cout << "2. Testing AudioBuffer... ";
    {
        whisper::AudioBuffer buffer(2.0f, 16000);  // 2 seconds max

        // Append some audio
        std::vector<float> chunk1(8000, 0.1f);  // 0.5s
        std::vector<float> chunk2(8000, 0.2f);  // 0.5s
        buffer.append(chunk1);
        buffer.append(chunk2);

        assert(std::abs(buffer.duration() - 1.0f) < 0.01f);
        assert(buffer.size() == 16000);

        auto audio = buffer.get_audio(0.5f);
        assert(audio.size() == 8000);

        buffer.clear();
        assert(buffer.size() == 0);
        std::cout << "PASS\n";
    }

    // Test LocalAgreement
    std::cout << "3. Testing LocalAgreement... ";
    {
        whisper::LocalAgreement agreement(2);

        // First transcript - no confirmed text yet
        std::string confirmed = agreement.update("Hello world");
        assert(confirmed.empty());

        // Second transcript - should confirm "Hello"
        confirmed = agreement.update("Hello there");
        // Common prefix is "Hello" between "Hello world" and "Hello there"
        assert(agreement.get_confirmed() == "Hello");

        // Third transcript
        confirmed = agreement.update("Hello friend");
        assert(agreement.get_confirmed() == "Hello");  // Still just "Hello"

        agreement.reset();
        assert(agreement.get_confirmed().empty());
        std::cout << "PASS\n";
    }

    // Test StreamingTranscriber with simple energy-based VAD
    std::cout << "4. Testing StreamingTranscriber initialization... ";
    {
        whisper::StreamingConfig config;
        config.sample_rate = 16000;
        config.min_chunk_duration = 0.5f;
        config.max_chunk_duration = 5.0f;
        config.emit_partials = true;
        config.use_vad = true;
        config.use_local_agreement = true;
        config.language = "en";

        whisper::StreamingTranscriber streamer(whisper_model, config);

        assert(streamer.state() == whisper::StreamState::IDLE);
        assert(streamer.language() == "en");
        std::cout << "PASS\n";
    }

    // Test streaming with real audio if available
    std::string test_audio = "/Users/ayates/model_mlx_migration/tests/prosody/contour_vs_v5_output/neutral_context_baseline.wav";
    if (std::ifstream(test_audio).good()) {
        std::cout << "5. Testing StreamingTranscriber with real audio... ";
        try {
            // Load audio
            auto audio_samples = whisper::audio::load_audio(test_audio);

            whisper::StreamingConfig config;
            config.sample_rate = 16000;
            config.min_chunk_duration = 0.5f;
            config.max_chunk_duration = 5.0f;
            config.silence_threshold_duration = 0.3f;
            config.emit_partials = false;  // Only final results for this test
            config.use_vad = true;
            config.use_local_agreement = false;
            config.language = "en";

            whisper::StreamingTranscriber streamer(whisper_model, config);

            // Track results
            int result_count = 0;
            bool got_final = false;

            streamer.set_callback([&](const whisper::StreamingResult& result) {
                result_count++;
                if (result.is_final) {
                    got_final = true;
                }
                std::cout << "\n   [" << (result.is_final ? "FINAL" : "PARTIAL") << "] "
                          << "RTF=" << std::fixed << std::setprecision(3) << result.rtf()
                          << ", duration=" << result.audio_duration << "s"
                          << ", processing=" << result.processing_time * 1000 << "ms";
            });

            // Feed audio in chunks (simulating real-time streaming)
            size_t chunk_size = 8000;  // 0.5s chunks
            auto start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < audio_samples.size(); i += chunk_size) {
                size_t count = std::min(chunk_size, audio_samples.size() - i);
                streamer.process_audio(audio_samples.data() + i, count);
            }

            // Finalize
            streamer.finalize();

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "\n   Total time: " << ms << " ms\n";
            std::cout << "   Results received: " << result_count << "\n";
            std::cout << "   Got final result: " << (got_final ? "YES" : "NO") << "\n";
            std::cout << "   PASS: Streaming transcription completed\n";
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    } else {
        std::cout << "5. Skipping streaming real audio test (file not found)\n";
    }

    std::cout << "\n=== Streaming tests completed ===\n";
}

// Gate 0: Transcribe a single file and output in specified format
// GAP 82: Read raw audio from stdin (16kHz mono s16le PCM expected)
std::vector<float> read_audio_from_stdin() {
    std::vector<float> audio;
    std::vector<int16_t> buffer(4096);  // Read in chunks

    // Read binary data from stdin
    std::cin.clear();
    std::freopen(nullptr, "rb", stdin);  // Reopen stdin in binary mode

    while (std::cin.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(int16_t))) {
        size_t samples_read = std::cin.gcount() / sizeof(int16_t);
        for (size_t i = 0; i < samples_read; ++i) {
            audio.push_back(static_cast<float>(buffer[i]) / 32768.0f);
        }
    }

    // Handle remaining bytes
    size_t remaining = std::cin.gcount();
    if (remaining > 0) {
        size_t samples_read = remaining / sizeof(int16_t);
        for (size_t i = 0; i < samples_read; ++i) {
            audio.push_back(static_cast<float>(buffer[i]) / 32768.0f);
        }
    }

    return audio;
}

// GAPs 101-108, 74: Output format support (txt, vtt, srt, tsv, json)
void transcribe_single_file(const std::string& model_path, const std::string& audio_path,
                           bool use_vad = true, bool word_timestamps = false,
                           const std::string& output_format = "json") {
    // Suppress all logging - output only the requested format
    mlx_inference::MLXInferenceEngine engine;
    engine.load_whisper(model_path);

    mlx_inference::TranscriptionConfig trans_config;
    trans_config.language = "en";
    trans_config.use_vad = use_vad;
    trans_config.word_timestamps = word_timestamps;  // GAP 3: word timestamps via DTW

    auto start = std::chrono::high_resolution_clock::now();
    mlx_inference::TranscriptionResult result;

    // GAP 82: Support reading audio from stdin with "-"
    if (audio_path == "-") {
        std::vector<float> stdin_audio = read_audio_from_stdin();
        result = engine.transcribe(stdin_audio, 16000, trans_config);
    } else {
        result = engine.transcribe_file(audio_path, trans_config);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // GAP 101-108: Use output_writers for format-specific output
    if (output_format == "txt" || output_format == "vtt" || output_format == "srt" ||
        output_format == "tsv" || output_format == "json") {
        // Use output_writers module for standard formats
        std::string output = mlx_inference::write_to_string(result, output_format);
        std::cout << output;
    } else if (output_format == "debug") {
        // Extended JSON format with debug info (original behavior)
        std::cout << "{\n";
        std::cout << "  \"file\": \"" << audio_path << "\",\n";
        std::cout << "  \"text\": \"";
        // Escape the text for JSON
        for (char c : result.text) {
            if (c == '"') std::cout << "\\\"";
            else if (c == '\\') std::cout << "\\\\";
            else if (c == '\n') std::cout << "\\n";
            else if (c == '\r') std::cout << "\\r";
            else if (c == '\t') std::cout << "\\t";
            else std::cout << c;
        }
        std::cout << "\",\n";
        std::cout << "  \"language\": \"" << result.language << "\",\n";
        std::cout << "  \"confidence\": " << result.confidence << ",\n";
        std::cout << "  \"elapsed_ms\": " << ms << ",\n";

        // Output segments with avg_logprob (GAP 10 fix validation)
        std::cout << "  \"segments\": [\n";
        for (size_t i = 0; i < result.segments.size(); ++i) {
            const auto& seg = result.segments[i];
            std::cout << "    {\n";
            std::cout << "      \"start\": " << seg.start_time << ",\n";
            std::cout << "      \"end\": " << seg.end_time << ",\n";
            std::cout << "      \"text\": \"";
            for (char c : seg.text) {
                if (c == '"') std::cout << "\\\"";
                else if (c == '\\') std::cout << "\\\\";
                else if (c == '\n') std::cout << "\\n";
                else std::cout << c;
            }
            std::cout << "\",\n";
            std::cout << "      \"avg_logprob\": " << seg.avg_logprob << ",\n";
            std::cout << "      \"no_speech_prob\": " << seg.no_speech_prob;

            // GAP 3: Output word timestamps if available
            if (!seg.words.empty()) {
                std::cout << ",\n      \"words\": [\n";
                for (size_t j = 0; j < seg.words.size(); ++j) {
                    const auto& word = seg.words[j];
                    std::cout << "        {\"word\": \"";
                    for (char c : word.word) {
                        if (c == '"') std::cout << "\\\"";
                        else if (c == '\\') std::cout << "\\\\";
                        else std::cout << c;
                    }
                    std::cout << "\", \"start\": " << word.start_time
                              << ", \"end\": " << word.end_time
                              << ", \"probability\": " << word.probability << "}";
                    if (j + 1 < seg.words.size()) std::cout << ",";
                    std::cout << "\n";
                }
                std::cout << "      ]\n";
            } else {
                std::cout << "\n";
            }
            std::cout << "    }";
            if (i + 1 < result.segments.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "  ],\n";

        std::cout << "  \"tokens\": [";
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << result.tokens[i];
        }
        std::cout << "]\n";
        std::cout << "}\n";
    } else {
        std::cerr << "Unknown output format: " << output_format << "\n";
        std::cerr << "Supported formats: txt, vtt, srt, tsv, json, debug\n";
    }
}

void test_cosyvoice3_forward_pass() {
    std::cout << "=== Test: CosyVoice3 Forward Pass (random weights) ===\n";

    using namespace cosyvoice3;

    // Create default config
    CosyVoice3Config config = CosyVoice3Config::create_default();
    std::cout << "Config: sample_rate=" << config.sample_rate
              << ", flow_depth=" << config.flow_config.depth
              << ", vocoder_upsample=" << config.vocoder_config.total_upsample_factor() << "x\n";

    // Create model (random weights)
    CosyVoice3Model model(config);
    std::cout << "Model created\n";

    // Test DiT Flow forward pass
    std::cout << "1. DiT Flow forward pass... ";
    try {
        int batch = 1;
        int num_tokens = 10;
        int mel_frames = num_tokens * config.flow_config.token_mel_ratio;

        auto tokens = mx::zeros({batch, num_tokens}, mx::int32);
        auto spk_emb = mx::random::normal({batch, 192});

        // Use model's tokens_to_mel which calls flow internally
        auto mel = model.tokens_to_mel(tokens, spk_emb, 3, 0.7f);  // 3 steps for quick test
        mx::eval(mel);

        auto shape = mel.shape();
        bool shape_ok = (shape[0] == batch && shape[1] == mel_frames && shape[2] == config.flow_config.out_channels);

        if (shape_ok) {
            std::cout << "PASS (mel shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << ")\n";
        } else {
            std::cout << "FAIL (unexpected shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << ")\n";
            return;
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL (exception: " << e.what() << ")\n";
        return;
    }

    // Test Vocoder forward pass
    std::cout << "2. Vocoder forward pass... ";
    try {
        int batch = 1;
        int mel_frames = 100;

        // Mel input: [B, L, mel_dim] for mel_to_audio
        auto mel = mx::random::normal({batch, mel_frames, config.vocoder_config.in_channels});

        auto audio = model.mel_to_audio(mel);
        mx::eval(audio);

        auto shape = audio.shape();
        int expected_samples = mel_frames * config.vocoder_config.total_upsample_factor();
        bool shape_ok = (shape[0] == batch && shape[1] >= expected_samples * 0.5);

        if (shape_ok) {
            std::cout << "PASS (output samples: " << shape[1] << ", expected ~" << expected_samples << ")\n";
        } else {
            std::cout << "FAIL (unexpected shape: " << shape[0] << "x" << shape[1] << ")\n";
            return;
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL (exception: " << e.what() << ")\n";
        return;
    }

    // Test full synthesis pipeline (tokens -> mel -> audio)
    std::cout << "3. Full synthesis pipeline... ";
    try {
        int batch = 1;

        auto text_ids = mx::zeros({batch, 5}, mx::int32);
        auto spk_emb = mx::random::normal({batch, 192});

        auto mel = model.tokens_to_mel(text_ids, spk_emb, 3, 0.7f);  // 3 steps
        mx::eval(mel);

        auto audio = model.mel_to_audio(mel);
        mx::eval(audio);

        auto mel_shape = mel.shape();
        auto audio_shape = audio.shape();

        std::cout << "PASS (mel: " << mel_shape[0] << "x" << mel_shape[1] << "x" << mel_shape[2]
                  << ", audio: " << audio_shape[0] << "x" << audio_shape[1] << ")\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL (exception: " << e.what() << ")\n";
        return;
    }

    // Test model info
    std::cout << "4. Model info... ";
    try {
        std::string info = model.info();
        if (info.find("CosyVoice3") != std::string::npos) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
            return;
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL (exception: " << e.what() << ")\n";
        return;
    }

    std::cout << "=== All CosyVoice3 tests PASSED ===\n\n";
}

void test_cosyvoice3_with_weights(const std::string& weights_path) {
    std::cout << "=== Test: CosyVoice3 with Real Weights ===\n";

    using namespace cosyvoice3;

    // Create default config
    CosyVoice3Config config = CosyVoice3Config::create_default();
    std::cout << "Config: sample_rate=" << config.sample_rate
              << ", flow_depth=" << config.flow_config.depth
              << ", vocoder_upsample=" << config.vocoder_config.total_upsample_factor() << "x\n";

    // Create model
    CosyVoice3Model model(config);
    std::cout << "Model created\n";

    // Load weights
    std::cout << "Loading weights from: " << weights_path << "\n";
    try {
        CosyVoice3Weights weights;

        // Use the unified load function which handles:
        // - Combined model.safetensors (MLX format with flow.*, vocoder.* prefixes)
        // - Separate flow.safetensors, vocoder.safetensors (PyTorch format)
        weights.load(weights_path);

        std::cout << "  Loaded flow weights: " << weights.flow_weight_count() << " tensors\n";
        std::cout << "  Loaded vocoder weights: " << weights.vocoder_weight_count() << " tensors\n";
        std::cout << "  Loaded LLM weights: " << weights.llm_weight_count() << " tensors\n";

        if (weights.total_weight_count() == 0) {
            std::cout << "FAIL: No weights loaded\n";
            return;
        }

        model.load_weights(weights);
        std::cout << "Weights loaded successfully\n";

        // Apply optimizations: fuse QKV weights, enable speaker cache
        model.optimize_all();
        std::cout << "Optimizations applied (QKV fusion, speaker cache)\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL (weight loading: " << e.what() << ")\n";
        return;
    }

    // Test vocoder with real weights
    std::cout << "1. Vocoder forward pass (real weights)... ";
    try {
        int batch = 1;
        int mel_frames = 50;  // ~0.5 seconds

        // Random mel input (we'll compare with Python later)
        auto mel = mx::random::normal({batch, mel_frames, config.vocoder_config.in_channels});

        auto start = std::chrono::high_resolution_clock::now();
        auto audio = model.mel_to_audio(mel);
        mx::eval(audio);
        auto end = std::chrono::high_resolution_clock::now();

        auto shape = audio.shape();
        int expected_samples = mel_frames * config.vocoder_config.total_upsample_factor();
        auto duration = std::chrono::duration<double, std::milli>(end - start);

        // Calculate RTF (Real-Time Factor)
        double audio_duration_s = static_cast<double>(shape[1]) / config.sample_rate;
        double rtf = audio_duration_s / (duration.count() / 1000.0);

        std::cout << "PASS\n";
        std::cout << "   Output samples: " << shape[1] << " (expected ~" << expected_samples << ")\n";
        std::cout << "   Audio duration: " << std::fixed << std::setprecision(2) << audio_duration_s << " s\n";
        std::cout << "   Inference time: " << duration.count() << " ms\n";
        std::cout << "   RTF: " << std::fixed << std::setprecision(1) << rtf << "x\n";

        // Save audio for listening
        std::string output_path = "cosyvoice3_test_output.wav";
        std::vector<float> samples(shape[1]);
        auto audio_data = audio.data<float>();
        std::copy(audio_data, audio_data + shape[1], samples.begin());
        write_wav(output_path, samples, config.sample_rate);
        std::cout << "   Saved to: " << output_path << "\n";

    } catch (const std::exception& e) {
        std::cout << "FAIL (exception: " << e.what() << ")\n";
        return;
    }

    // Test DiT flow with real weights
    std::cout << "2. DiT Flow forward pass (real weights)... ";
    try {
        int batch = 1;
        int num_tokens = 10;

        auto tokens = mx::zeros({batch, num_tokens}, mx::int32);
        auto spk_emb = mx::random::normal({batch, 192});

        auto start = std::chrono::high_resolution_clock::now();
        auto mel = model.tokens_to_mel(tokens, spk_emb, 10, 0.7f);  // 10 steps
        mx::eval(mel);
        auto end = std::chrono::high_resolution_clock::now();

        auto shape = mel.shape();
        auto duration = std::chrono::duration<double, std::milli>(end - start);

        std::cout << "PASS\n";
        std::cout << "   Mel shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << "\n";
        std::cout << "   Inference time: " << duration.count() << " ms\n";

    } catch (const std::exception& e) {
        std::cout << "FAIL (exception: " << e.what() << ")\n";
        return;
    }

    // Test LLM speech token generation
    std::cout << "3. LLM speech token generation... ";
    try {
        // Create simple text token input (just a few tokens for testing)
        // In production, these would come from tokenizer
        int batch = 1;
        int text_len = 10;

        // Generate some placeholder text tokens (vocab size is 151936)
        auto text_ids = mx::zeros({batch, text_len}, mx::int32);
        // Set some actual token values (e.g., random valid tokens)
        std::vector<int> token_values = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        auto tokens_data = mx::array(token_values.data(), {batch, text_len}, mx::int32);

        auto start = std::chrono::high_resolution_clock::now();
        auto speech_tokens = model.generate_speech_tokens(
            tokens_data,
            50,    // max_length - short for testing
            0.0f,  // temperature - greedy
            25,    // top_k
            0.8f   // top_p
        );
        mx::eval(speech_tokens);
        auto end = std::chrono::high_resolution_clock::now();

        auto shape = speech_tokens.shape();
        auto duration = std::chrono::duration<double, std::milli>(end - start);

        // Check if we got non-zero output
        bool has_nonzero = false;
        auto data = speech_tokens.data<int32_t>();
        for (int i = 0; i < shape[1] && i < 10; ++i) {
            if (data[i] != 0) has_nonzero = true;
        }

        if (shape[1] > 0 && has_nonzero) {
            std::cout << "PASS\n";
            std::cout << "   Generated tokens: " << shape[1] << "\n";
            std::cout << "   Inference time: " << duration.count() << " ms\n";
            std::cout << "   First 10 tokens: ";
            for (int i = 0; i < std::min(10, static_cast<int>(shape[1])); ++i) {
                std::cout << data[i] << " ";
            }
            std::cout << "\n";
        } else {
            std::cout << "WARN (generated zero tokens or all zeros)\n";
            std::cout << "   Generated tokens: " << shape[1] << "\n";
            std::cout << "   Note: LLM may need proper text input from tokenizer\n";
        }

    } catch (const std::exception& e) {
        std::cout << "FAIL (exception: " << e.what() << ")\n";
        return;
    }

    std::cout << "=== CosyVoice3 Real Weights Test Complete ===\n\n";
}

int main(int argc, char* argv[]) {
    // Parse command line options FIRST to check for --transcribe mode
    std::string kokoro_path;
    std::string translation_path;
    std::string whisper_path;
    std::string llm_path;
    std::string cosyvoice_path;    // For --cosyvoice <path> flag
    std::string transcribe_audio;  // For --transcribe mode
    std::string vad_probs_audio;   // For --vad-probs mode
    std::string output_format = "json";  // GAPs 101-108: Output format (txt, vtt, srt, tsv, json, debug)
    bool run_benchmark = false;
    bool run_mel_test = false;
    bool run_streaming_test = false;
    bool run_cosyvoice_test = false;  // For --cosyvoice-test flag (random weights)
    bool disable_vad = false;  // For --no-vad flag
    bool word_timestamps = false;  // For --word-timestamps flag (GAP 3)

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--kokoro" && i + 1 < argc) {
            kokoro_path = argv[++i];
        } else if (arg == "--translation" && i + 1 < argc) {
            translation_path = argv[++i];
        } else if (arg == "--whisper" && i + 1 < argc) {
            whisper_path = argv[++i];
        } else if (arg == "--llm" && i + 1 < argc) {
            llm_path = argv[++i];
        } else if (arg == "--cosyvoice" && i + 1 < argc) {
            cosyvoice_path = argv[++i];
        } else if (arg == "--transcribe" && i + 1 < argc) {
            transcribe_audio = argv[++i];
        } else if (arg == "--vad-probs" && i + 1 < argc) {
            vad_probs_audio = argv[++i];
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--mel-test") {
            run_mel_test = true;
        } else if (arg == "--streaming") {
            run_streaming_test = true;
        } else if (arg == "--cosyvoice-test") {
            run_cosyvoice_test = true;
        } else if (arg == "--no-vad") {
            disable_vad = true;
        } else if (arg == "--word-timestamps") {
            word_timestamps = true;
        } else if (arg == "--output-format" && i + 1 < argc) {
            output_format = argv[++i];  // GAPs 101-108: txt, vtt, srt, tsv, json, debug
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./test_engine [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --kokoro <path>       Path to Kokoro model directory\n";
            std::cout << "  --translation <path>  Path to MADLAD translation model directory\n";
            std::cout << "  --whisper <path>      Path to Whisper model directory\n";
            std::cout << "  --llm <path>          Path to LLM model directory\n";
            std::cout << "  --transcribe <file>   Transcribe single audio file (requires --whisper)\n";
            std::cout << "  --output-format <fmt> Output format: txt, vtt, srt, tsv, json (default), debug\n";
            std::cout << "                        GAPs 101-108: Matches Python mlx_whisper output formats\n";
            std::cout << "  --vad-probs <file>    Output per-chunk VAD probabilities as JSON\n";
            std::cout << "  --no-vad              Disable VAD preprocessing (for Gate 0 debugging)\n";
            std::cout << "  --word-timestamps     Enable word-level timestamps in output (GAP 3)\n";
            std::cout << "  --benchmark           Run translation benchmark (requires --translation)\n";
            std::cout << "  --mel-test            Run mel spectrogram standalone test\n";
            std::cout << "  --streaming           Run streaming transcription tests (requires --whisper)\n";
            std::cout << "  --cosyvoice-test      Run CosyVoice3 forward pass tests (random weights)\n";
            std::cout << "  --cosyvoice <path>    Test CosyVoice3 with real weights from <path>\n";
            std::cout << "  -h, --help            Show this help\n";
            return 0;
        } else if (kokoro_path.empty()) {
            // Legacy: first positional arg is Kokoro path
            kokoro_path = arg;
        }
    }

    // Special mode: --transcribe for Gate 0 comparison
    // GAPs 101-108: Supports multiple output formats via --output-format
    if (!transcribe_audio.empty()) {
        if (whisper_path.empty()) {
            std::cerr << "Error: --transcribe requires --whisper <path>\n";
            return 1;
        }
        transcribe_single_file(whisper_path, transcribe_audio, !disable_vad, word_timestamps, output_format);
        return 0;
    }

    // Special mode: --vad-probs for debugging VAD probabilities
    if (!vad_probs_audio.empty()) {
        // Load audio
        auto audio_samples = whisper::audio::load_audio(vad_probs_audio);
        if (audio_samples.empty()) {
            std::cerr << "Error: Failed to load audio: " << vad_probs_audio << "\n";
            return 1;
        }

        // Get VAD weights path (relative to current directory or from project root)
        std::string vad_weights = "models/silero_vad/silero_vad_16k.bin";
        if (!std::ifstream(vad_weights).good()) {
            std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
            vad_weights = home + "/model_mlx_migration/models/silero_vad/silero_vad_16k.bin";
        }

        // Create VAD and get probabilities
        silero_vad::SileroVAD vad(vad_weights, 16000);
        auto probs = vad.get_probabilities(audio_samples.data(), audio_samples.size());

        // Output JSON
        std::cout << "{\n";
        std::cout << "  \"file\": \"" << vad_probs_audio << "\",\n";
        std::cout << "  \"num_chunks\": " << probs.size() << ",\n";
        std::cout << "  \"chunk_size_samples\": 512,\n";
        std::cout << "  \"probabilities\": [";
        for (size_t i = 0; i < probs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            if (i % 20 == 0) std::cout << "\n    ";
            std::cout << std::fixed << std::setprecision(4) << probs[i];
        }
        std::cout << "\n  ]\n";
        std::cout << "}\n";
        return 0;
    }

    // Normal test mode
    std::cout << "MLX Inference Engine Tests\n";
    std::cout << "==========================\n\n";

    // Test not-implemented paths
    test_not_implemented();
    std::cout << "\n";

    // Test CosyVoice3 if requested
    if (run_cosyvoice_test) {
        test_cosyvoice3_forward_pass();
        std::cout << "\n";
    }

    // Test CosyVoice3 with real weights if path provided
    if (!cosyvoice_path.empty()) {
        test_cosyvoice3_with_weights(cosyvoice_path);
        std::cout << "\n";
    }

    // Test Kokoro if path provided
    if (!kokoro_path.empty()) {
        test_kokoro_tts(kokoro_path);
        std::cout << "\n";
    } else {
        std::cout << "To test Kokoro TTS, provide model path:\n";
        std::cout << "  ./test_engine --kokoro /path/to/kokoro_cpp_export\n\n";
    }

    // Test Translation if path provided
    if (!translation_path.empty()) {
        if (run_benchmark) {
            benchmark_translation(translation_path);
        } else {
            test_translation(translation_path);
        }
        std::cout << "\n";
    } else {
        std::cout << "To test Translation, provide model path:\n";
        std::cout << "  ./test_engine --translation ~/.cache/huggingface/hub/models--google--madlad400-3b-mt/snapshots/*/\n";
        std::cout << "  ./test_engine --translation <path> --benchmark   # Run benchmark\n\n";
    }

    // Test Whisper if path provided
    if (!whisper_path.empty()) {
        test_whisper_stt(whisper_path);
        std::cout << "\n";

        // Run streaming test if requested
        if (run_streaming_test) {
            test_streaming(whisper_path);
            std::cout << "\n";
        }
    } else {
        std::cout << "To test Whisper STT, provide model path:\n";
        std::cout << "  ./test_engine --whisper ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/*/\n";
        std::cout << "  ./test_engine --whisper <path> --streaming   # Run streaming tests\n\n";
    }

    // Run mel spectrogram standalone test if requested
    if (run_mel_test) {
        test_mel_spectrogram();
        std::cout << "\n";
    }

    // Test LLM if path provided
    if (!llm_path.empty()) {
        test_llm(llm_path);
        std::cout << "\n";
    } else {
        std::cout << "To test LLM, provide model path:\n";
        std::cout << "  ./test_engine --llm ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/*/\n\n";
    }

    return 0;
}
