// Copyright 2024-2025 Andrew Yates
// High-level inference API for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#pragma once

#include "zipformer/zipformer.hpp"
#include <string>
#include <vector>
#include <functional>

namespace zipformer {

/**
 * Callback for streaming results.
 *
 * Called whenever a partial or final result is available.
 */
using StreamingCallback = std::function<void(const DecodingResult&, bool is_final)>;

/**
 * Transcribe a single audio file.
 *
 * @param model_path Path to model checkpoint directory
 * @param audio_path Path to audio file (WAV, FLAC, etc.)
 * @return Transcription result
 */
DecodingResult transcribe_file(
    const std::string& model_path,
    const std::string& audio_path
);

/**
 * Transcribe multiple audio files (batch).
 *
 * @param model_path Path to model checkpoint directory
 * @param audio_paths Paths to audio files
 * @return Vector of transcription results
 */
std::vector<DecodingResult> transcribe_files(
    const std::string& model_path,
    const std::vector<std::string>& audio_paths
);

/**
 * Streaming transcription from audio file.
 *
 * Simulates streaming by processing the file in chunks.
 *
 * @param model_path Path to model checkpoint directory
 * @param audio_path Path to audio file
 * @param callback Called for each partial/final result
 * @param chunk_ms Chunk size in milliseconds (default 320ms)
 */
void transcribe_streaming(
    const std::string& model_path,
    const std::string& audio_path,
    StreamingCallback callback,
    int chunk_ms = 320
);

/**
 * Benchmark model performance.
 *
 * @param model_path Path to model checkpoint directory
 * @param audio_path Path to test audio file
 * @param num_iterations Number of iterations for timing
 * @return Performance metrics
 */
struct BenchmarkResult {
    float total_audio_seconds;
    float total_inference_seconds;
    float rtf;  // Real-time factor (inference_time / audio_time)
    float first_token_latency_ms;
    float tokens_per_second;
};

BenchmarkResult benchmark_model(
    const std::string& model_path,
    const std::string& audio_path,
    int num_iterations = 10
);

/**
 * Validate model against Python reference.
 *
 * Compares C++ inference output against pre-computed Python output.
 *
 * @param model_path Path to C++ model
 * @param reference_path Path to Python reference output (.npz)
 * @param tolerance Maximum allowed difference
 * @return True if outputs match within tolerance
 */
bool validate_against_reference(
    const std::string& model_path,
    const std::string& reference_path,
    float tolerance = 1e-4f
);

} // namespace zipformer
