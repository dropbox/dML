// Copyright 2024-2025 Andrew Yates
// High-level inference API implementation for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/inference.hpp"
#include <chrono>
#include <iostream>

namespace zipformer {

DecodingResult transcribe_file(
    const std::string& model_path,
    const std::string& audio_path
) {
    // Load model
    auto model = load_model(model_path);

    // Load audio
    array audio = load_audio(audio_path, model->config().features.sample_rate);

    // Transcribe
    return model->transcribe(audio);
}

std::vector<DecodingResult> transcribe_files(
    const std::string& model_path,
    const std::vector<std::string>& audio_paths
) {
    // Load model once
    auto model = load_model(model_path);

    std::vector<DecodingResult> results;
    results.reserve(audio_paths.size());

    for (const auto& path : audio_paths) {
        array audio = load_audio(path, model->config().features.sample_rate);
        results.push_back(model->transcribe(audio));
    }

    return results;
}

void transcribe_streaming(
    const std::string& model_path,
    const std::string& audio_path,
    StreamingCallback callback,
    int chunk_ms
) {
    // Load streaming model
    auto model = load_streaming_model(model_path);
    model->reset();

    // Load full audio
    array audio = load_audio(audio_path, model->current_result().confidence);

    // Calculate chunk size in samples
    int chunk_samples = static_cast<int>(16000 * chunk_ms / 1000);

    // Process in chunks
    int total_samples = audio.shape()[0];
    for (int offset = 0; offset < total_samples; offset += chunk_samples) {
        int end = std::min(offset + chunk_samples, total_samples);
        array chunk = slice(audio, {offset}, {end});

        bool is_final = (end >= total_samples);
        DecodingResult result = model->process_chunk(chunk, is_final);

        callback(result, is_final);
    }
}

BenchmarkResult benchmark_model(
    const std::string& model_path,
    const std::string& audio_path,
    int num_iterations
) {
    using Clock = std::chrono::high_resolution_clock;

    // Load model
    auto model = load_model(model_path);

    // Load audio
    array audio = load_audio(audio_path, model->config().features.sample_rate);
    float audio_seconds = static_cast<float>(audio.shape()[0]) / 16000.0f;

    BenchmarkResult result;
    result.total_audio_seconds = audio_seconds * num_iterations;

    // Warm up
    model->transcribe(audio);

    // Benchmark
    auto start = Clock::now();
    int total_tokens = 0;

    for (int i = 0; i < num_iterations; ++i) {
        auto decode_start = Clock::now();
        DecodingResult output = model->transcribe(audio);
        auto decode_end = Clock::now();

        total_tokens += output.tokens.size();

        // First iteration: measure first token latency
        if (i == 0) {
            // Simplified - would need more precise measurement
            result.first_token_latency_ms = std::chrono::duration<float, std::milli>(
                decode_end - decode_start).count() / output.tokens.size();
        }
    }

    auto end = Clock::now();
    result.total_inference_seconds = std::chrono::duration<float>(end - start).count();
    result.rtf = result.total_inference_seconds / result.total_audio_seconds;
    result.tokens_per_second = static_cast<float>(total_tokens) / result.total_inference_seconds;

    return result;
}

bool validate_against_reference(
    const std::string& model_path,
    const std::string& reference_path,
    float tolerance
) {
    // Load model
    auto model = load_model(model_path);

    // Load reference data (expects npz or safetensors format)
    // Reference should contain: input_fbank (or audio), expected_encoder_out
    WeightMap reference = load_weights(reference_path);

    if (!has_weight(reference, "expected_encoder_out")) {
        std::cerr << "Reference file missing 'expected_encoder_out' key" << std::endl;
        return false;
    }

    array expected_encoder = get_weight(reference, "expected_encoder_out");

    // Try to use pre-computed fbank if available (more deterministic)
    array feats = zeros({1});  // Placeholder - will be reassigned
    if (has_weight(reference, "input_fbank")) {
        feats = get_weight(reference, "input_fbank");
        std::cout << "Using pre-computed fbank from reference: "
                  << feats.shape()[0] << "x" << feats.shape()[1] << "x" << feats.shape()[2] << std::endl;
    } else if (has_weight(reference, "audio")) {
        array audio = get_weight(reference, "audio");
        feats = model->features().extract(audio);
        feats = expand_dims(feats, 0);
        std::cout << "Extracted fbank from audio: "
                  << feats.shape()[0] << "x" << feats.shape()[1] << "x" << feats.shape()[2] << std::endl;
    } else {
        std::cerr << "Reference file missing 'input_fbank' or 'audio' key" << std::endl;
        return false;
    }

    // Debug: Check if model weights were loaded
    std::cout << "Model features sample rate: " << model->config().features.sample_rate << std::endl;

    // Run encoder
    array encoder_out = model->encoder().forward(feats);
    std::cout << "Encoder output shape: "
              << encoder_out.shape()[0] << "x" << encoder_out.shape()[1] << "x" << encoder_out.shape()[2] << std::endl;

    // Debug: Check output statistics
    eval(encoder_out);
    auto out_min = min(encoder_out);
    auto out_max = max(encoder_out);
    auto out_mean = mean(encoder_out);
    eval(out_min, out_max, out_mean);
    std::cout << "C++ encoder output stats: min=" << out_min.item<float>()
              << " max=" << out_max.item<float>()
              << " mean=" << out_mean.item<float>() << std::endl;

    auto exp_min = min(expected_encoder);
    auto exp_max = max(expected_encoder);
    auto exp_mean = mean(expected_encoder);
    eval(exp_min, exp_max, exp_mean);
    std::cout << "Expected output stats: min=" << exp_min.item<float>()
              << " max=" << exp_max.item<float>()
              << " mean=" << exp_mean.item<float>() << std::endl;
    std::cout << "Expected shape: "
              << expected_encoder.shape()[0] << "x" << expected_encoder.shape()[1] << "x" << expected_encoder.shape()[2] << std::endl;

    // Compare
    auto validation = compare_arrays(encoder_out, expected_encoder, tolerance, tolerance);

    if (!validation.passed) {
        std::cerr << "Validation failed: " << validation.message << std::endl;
        std::cerr << "  max_diff: " << validation.max_diff << std::endl;
        std::cerr << "  mean_diff: " << validation.mean_diff << std::endl;
        return false;
    }

    std::cout << "Validation passed!" << std::endl;
    std::cout << "  max_diff: " << validation.max_diff << std::endl;
    std::cout << "  mean_diff: " << validation.mean_diff << std::endl;

    return true;
}

} // namespace zipformer
