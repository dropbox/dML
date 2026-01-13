// Speech-to-Speech Pipeline Benchmark
// Tests unified Whisper STT -> Kokoro TTS pipeline performance

#include "mlx_inference_engine.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>

void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create " << filename << "\n";
        return;
    }

    // WAV header
    int data_size = samples.size() * sizeof(int16_t);
    int file_size = 36 + data_size;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);

    int fmt_size = 16;
    short audio_format = 1;  // PCM
    short num_channels = 1;
    int byte_rate = sample_rate * 2;
    short block_align = 2;
    short bits_per_sample = 16;

    file.write(reinterpret_cast<char*>(&fmt_size), 4);
    file.write(reinterpret_cast<char*>(&audio_format), 2);
    file.write(reinterpret_cast<char*>(&num_channels), 2);
    file.write(reinterpret_cast<char*>(&sample_rate), 4);
    file.write(reinterpret_cast<char*>(&byte_rate), 4);
    file.write(reinterpret_cast<char*>(&block_align), 2);
    file.write(reinterpret_cast<char*>(&bits_per_sample), 2);

    file.write("data", 4);
    file.write(reinterpret_cast<char*>(&data_size), 4);

    // Convert float to int16
    for (float s : samples) {
        int16_t sample = static_cast<int16_t>(std::clamp(s, -1.0f, 1.0f) * 32767.0f);
        file.write(reinterpret_cast<char*>(&sample), 2);
    }
}

int main(int argc, char* argv[]) {
    std::string whisper_model = "models/whisper-large-v3-turbo-mlx";
    std::string kokoro_model = "kokoro_cpp_export";
    std::string audio_file = "";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--whisper" && i + 1 < argc) {
            whisper_model = argv[++i];
        } else if (arg == "--kokoro" && i + 1 < argc) {
            kokoro_model = argv[++i];
        } else if (arg == "--audio" && i + 1 < argc) {
            audio_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --whisper <path>  Path to Whisper model (default: models/whisper-large-v3-turbo-mlx)\n"
                      << "  --kokoro <path>   Path to Kokoro model (default: kokoro_cpp_export)\n"
                      << "  --audio <path>    Path to audio file to process\n";
            return 0;
        }
    }

    std::cout << "=== Speech-to-Speech Pipeline Benchmark ===\n\n";

    // Load models
    mlx_inference::MLXInferenceEngine engine;

    std::cout << "Loading Whisper model from: " << whisper_model << "\n";
    auto whisper_start = std::chrono::high_resolution_clock::now();
    engine.load_whisper(whisper_model);
    auto whisper_end = std::chrono::high_resolution_clock::now();
    double whisper_load_ms = std::chrono::duration<double, std::milli>(whisper_end - whisper_start).count();
    std::cout << "  Load time: " << std::fixed << std::setprecision(0) << whisper_load_ms << " ms\n";

    std::cout << "Loading Kokoro model from: " << kokoro_model << "\n";
    auto kokoro_start = std::chrono::high_resolution_clock::now();
    engine.load_kokoro(kokoro_model);
    auto kokoro_end = std::chrono::high_resolution_clock::now();
    double kokoro_load_ms = std::chrono::duration<double, std::milli>(kokoro_end - kokoro_start).count();
    std::cout << "  Load time: " << std::fixed << std::setprecision(0) << kokoro_load_ms << " ms\n";

    std::cout << "\nTotal model load time: " << (whisper_load_ms + kokoro_load_ms) << " ms\n\n";

    // Warmup
    std::cout << "Warming up models...\n";
    {
        // Create a short test audio (1 second of silence with a brief tone)
        std::vector<float> warmup_audio(16000, 0.0f);
        for (int i = 0; i < 8000; i++) {
            warmup_audio[i] = 0.1f * std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f);
        }

        mlx_inference::S2SConfig config;
        config.tts_config.voice = "af_heart";

        // Warmup run
        try {
            engine.speech_to_speech(warmup_audio, 16000, config);
        } catch (...) {
            // Warmup may fail on silence, that's OK
        }
    }
    std::cout << "Warmup complete.\n\n";

    // If audio file provided, run S2S pipeline on it
    if (!audio_file.empty()) {
        std::cout << "=== Processing: " << audio_file << " ===\n\n";

        mlx_inference::S2SConfig config;
        config.tts_config.voice = "af_heart";
        config.tts_config.speed = 1.0f;

        // Non-streaming mode
        std::cout << "Non-streaming mode:\n";
        auto result = engine.speech_to_speech_file(audio_file, config);

        std::cout << "  Transcription: \"" << result.transcription.text << "\"\n";
        std::cout << "  STT latency: " << std::fixed << std::setprecision(1) << result.stt_latency_ms << " ms\n";
        std::cout << "  TTS latency: " << result.tts_latency_ms << " ms\n";
        std::cout << "  Total latency: " << result.total_latency_ms << " ms\n";
        std::cout << "  Output audio: " << result.audio.duration_seconds << " s\n";

        // Calculate RTF
        float input_duration = static_cast<float>(result.transcription.segments.empty() ? 0 :
            result.transcription.segments.back().end_time);
        if (input_duration > 0) {
            float rtf = (result.total_latency_ms / 1000.0f) / input_duration;
            std::cout << "  RTF (input): " << std::setprecision(2) << rtf << "x real-time\n";
        }

        // Save output
        std::string output_file = "s2s_output.wav";
        write_wav(output_file, result.audio.samples, result.audio.sample_rate);
        std::cout << "  Saved to: " << output_file << "\n\n";

        // Streaming mode
        std::cout << "Streaming mode:\n";
        config.use_streaming = true;

        auto stream_start = std::chrono::high_resolution_clock::now();
        double first_chunk_latency = 0.0;
        bool first_chunk = true;

        std::vector<float> streaming_samples;
        engine.speech_to_speech_streaming(
            result.transcription.tokens.empty() ? std::vector<float>() : std::vector<float>(),  // Use same audio
            16000,
            [&](const std::vector<float>& samples, int chunk_idx, bool is_final) {
                auto now = std::chrono::high_resolution_clock::now();
                if (first_chunk) {
                    first_chunk_latency = std::chrono::duration<double, std::milli>(now - stream_start).count();
                    first_chunk = false;
                }
                streaming_samples.insert(streaming_samples.end(), samples.begin(), samples.end());
                std::cout << "    Chunk " << chunk_idx << ": " << samples.size() << " samples"
                          << (is_final ? " [final]" : "") << "\n";
            },
            config
        );

        if (first_chunk_latency > 0) {
            std::cout << "  Time-to-first-audio: " << std::setprecision(1) << first_chunk_latency << " ms\n";
        }
    }

    // Synthetic benchmark
    std::cout << "=== Synthetic Benchmark ===\n\n";

    // Generate synthetic speech-like audio (simple tone bursts)
    std::vector<std::pair<std::string, int>> test_durations = {
        {"1 second", 16000},
        {"5 seconds", 80000},
        {"10 seconds", 160000}
    };

    for (const auto& [name, samples] : test_durations) {
        std::vector<float> test_audio(samples, 0.0f);

        // Add some tone bursts to simulate speech
        for (int burst = 0; burst < samples / 4000; burst++) {
            int start = burst * 4000;
            for (int i = 0; i < 2000 && start + i < samples; i++) {
                float t = static_cast<float>(i) / 16000.0f;
                test_audio[start + i] = 0.3f * std::sin(2.0f * 3.14159f * (200 + burst * 50) * t);
            }
        }

        std::cout << name << " audio:\n";

        mlx_inference::S2SConfig config;
        config.tts_config.voice = "af_heart";

        try {
            auto result = engine.speech_to_speech(test_audio, 16000, config);
            std::cout << "  STT: " << std::fixed << std::setprecision(0) << result.stt_latency_ms << " ms\n";
            std::cout << "  TTS: " << result.tts_latency_ms << " ms\n";
            std::cout << "  Total: " << result.total_latency_ms << " ms\n";
            std::cout << "  Transcription: \"" << result.transcription.text.substr(0, 50)
                      << (result.transcription.text.length() > 50 ? "..." : "") << "\"\n\n";
        } catch (const std::exception& e) {
            std::cout << "  Error: " << e.what() << "\n\n";
        }
    }

    std::cout << "Done.\n";
    return 0;
}
