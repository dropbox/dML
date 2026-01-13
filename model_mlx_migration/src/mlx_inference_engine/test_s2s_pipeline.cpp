// Test S2S pipeline by generating speech first, then running it through S2S
// This ensures we have valid speech audio for testing

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

    int data_size = samples.size() * sizeof(int16_t);
    int file_size = 36 + data_size;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);

    int fmt_size = 16;
    short audio_format = 1;
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

    for (float s : samples) {
        int16_t sample = static_cast<int16_t>(std::clamp(s, -1.0f, 1.0f) * 32767.0f);
        file.write(reinterpret_cast<char*>(&sample), 2);
    }
}

// Resample from 24kHz to 16kHz for Whisper
std::vector<float> resample_24k_to_16k(const std::vector<float>& input) {
    // Simple linear interpolation resampling
    double ratio = 16000.0 / 24000.0;  // 2/3
    size_t output_size = static_cast<size_t>(input.size() * ratio);
    std::vector<float> output(output_size);

    for (size_t i = 0; i < output_size; ++i) {
        double src_idx = i / ratio;
        size_t idx0 = static_cast<size_t>(src_idx);
        size_t idx1 = std::min(idx0 + 1, input.size() - 1);
        double frac = src_idx - idx0;
        output[i] = input[idx0] * (1.0 - frac) + input[idx1] * frac;
    }

    return output;
}

int main() {
    std::cout << "=== S2S Pipeline Test ===\n\n";

    mlx_inference::MLXInferenceEngine engine;

    // Load models
    std::cout << "Loading Whisper model...\n";
    engine.load_whisper("models/whisper-large-v3-turbo-mlx");

    std::cout << "Loading Kokoro model...\n";
    engine.load_kokoro("kokoro_cpp_export");

    std::cout << "\nModels loaded.\n\n";

    // Step 1: Generate speech with Kokoro
    std::string test_text = "Hello world. This is a test of the speech to speech pipeline.";
    std::cout << "Step 1: Generating speech for: \"" << test_text << "\"\n";

    mlx_inference::TTSConfig tts_config;
    tts_config.voice = "af_heart";
    tts_config.speed = 1.0f;

    auto tts_start = std::chrono::high_resolution_clock::now();
    auto tts_output = engine.synthesize(test_text, tts_config);
    auto tts_end = std::chrono::high_resolution_clock::now();
    double tts_ms = std::chrono::duration<double, std::milli>(tts_end - tts_start).count();

    std::cout << "  TTS generated " << tts_output.duration_seconds << "s audio in "
              << std::fixed << std::setprecision(0) << tts_ms << "ms\n";

    // Save original TTS output
    write_wav("s2s_test_input.wav", tts_output.samples, tts_output.sample_rate);
    std::cout << "  Saved to: s2s_test_input.wav\n\n";

    // Step 2: Resample to 16kHz for Whisper
    std::cout << "Step 2: Resampling from 24kHz to 16kHz...\n";
    auto resampled = resample_24k_to_16k(tts_output.samples);
    std::cout << "  Resampled from " << tts_output.samples.size() << " to " << resampled.size() << " samples\n\n";

    // Step 3: Run full S2S pipeline
    std::cout << "Step 3: Running S2S pipeline...\n";

    mlx_inference::S2SConfig s2s_config;
    s2s_config.tts_config.voice = "af_heart";
    s2s_config.tts_config.speed = 1.0f;

    auto s2s_start = std::chrono::high_resolution_clock::now();
    auto s2s_result = engine.speech_to_speech(resampled, 16000, s2s_config);
    auto s2s_end = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== S2S Results ===\n";
    std::cout << "Input text:       \"" << test_text << "\"\n";
    std::cout << "Transcription:    \"" << s2s_result.transcription.text << "\"\n";
    std::cout << "STT latency:      " << std::setprecision(0) << s2s_result.stt_latency_ms << " ms\n";
    std::cout << "TTS latency:      " << s2s_result.tts_latency_ms << " ms\n";
    std::cout << "Total latency:    " << s2s_result.total_latency_ms << " ms\n";
    std::cout << "Output audio:     " << std::setprecision(2) << s2s_result.audio.duration_seconds << " s\n";

    // Calculate pipeline RTF
    float input_duration = static_cast<float>(resampled.size()) / 16000.0f;
    float pipeline_rtf = (s2s_result.total_latency_ms / 1000.0f) / input_duration;
    std::cout << "Pipeline RTF:     " << std::setprecision(2) << pipeline_rtf << "x real-time\n";

    // Save S2S output
    write_wav("s2s_test_output.wav", s2s_result.audio.samples, s2s_result.audio.sample_rate);
    std::cout << "\nSaved output to: s2s_test_output.wav\n";

    // Step 4: Test Full Streaming S2S (streaming STT -> streaming TTS)
    std::cout << "\nStep 4: Testing full streaming S2S pipeline...\n";
    {
        mlx_inference::S2SConfig s2s_config;
        s2s_config.tts_config.voice = "af_heart";
        s2s_config.tts_config.speed = 1.0f;

        auto stream_start = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point first_chunk_time;
        bool got_first_chunk = false;
        int total_chunks = 0;
        std::vector<float> all_audio;

        engine.speech_to_speech_full_streaming(
            resampled, 16000,
            [&](const mlx_inference::S2SStreamingChunk& chunk) {
                if (!got_first_chunk && !chunk.is_final) {
                    first_chunk_time = std::chrono::high_resolution_clock::now();
                    got_first_chunk = true;
                }

                if (!chunk.is_final) {
                    all_audio.insert(all_audio.end(),
                                     chunk.audio_samples.begin(),
                                     chunk.audio_samples.end());
                    total_chunks++;
                    std::cout << "    Chunk " << chunk.chunk_index << ": \"" << chunk.text.substr(0, 40)
                              << "...\" STT:" << std::setprecision(0) << chunk.stt_latency_ms
                              << "ms TTS:" << chunk.tts_latency_ms << "ms\n";
                }
            },
            s2s_config
        );

        auto stream_end = std::chrono::high_resolution_clock::now();

        if (got_first_chunk) {
            double ttfa = std::chrono::duration<double, std::milli>(first_chunk_time - stream_start).count();
            double total = std::chrono::duration<double, std::milli>(stream_end - stream_start).count();
            std::cout << "\n  Full Streaming Results:\n";
            std::cout << "    Total chunks: " << total_chunks << "\n";
            std::cout << "    Time-to-first-audio: " << std::setprecision(0) << ttfa << " ms\n";
            std::cout << "    Total time: " << total << " ms\n";
            std::cout << "    Output audio: " << std::setprecision(2)
                      << (all_audio.size() / 24000.0f) << " s\n";
        } else {
            std::cout << "  No chunks generated (streaming may have failed)\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
