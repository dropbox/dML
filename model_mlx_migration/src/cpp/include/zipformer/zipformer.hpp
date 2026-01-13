// Copyright 2024-2025 Andrew Yates
// Full Zipformer ASR model for MLX C++
//
// Licensed under the Apache License, Version 2.0

#pragma once

#include "zipformer/array_utils.hpp"
#include "zipformer/scaling.hpp"
#include "zipformer/encoder.hpp"
#include "zipformer/zipformer2.hpp"  // For ZipformerEncoder
#include "zipformer/decoder.hpp"
#include "zipformer/joiner.hpp"
#include "zipformer/features.hpp"

#include <string>
#include <vector>
#include <memory>

namespace zipformer {

using namespace mlx::core;

/**
 * Configuration for complete ASR model.
 */
struct ASRModelConfig {
    ZipformerConfig encoder;
    DecoderConfig decoder;
    JoinerConfig joiner;
    FbankConfig features;

    // Decoding parameters
    int beam_size = 4;
    float temperature = 1.0f;

    // Create default config matching icefall streaming checkpoint
    static ASRModelConfig default_streaming();
};

/**
 * Result from decoding.
 */
struct DecodingResult {
    std::string text;
    std::vector<int> tokens;
    std::vector<float> timestamps;  // Per-token timestamps (seconds)
    float confidence;
};

/**
 * Complete Zipformer ASR model.
 *
 * Combines encoder, decoder, and joiner for speech recognition.
 */
class ASRModel {
public:
    explicit ASRModel(const ASRModelConfig& config);

    // Load model weights from safetensors file
    void load_weights(const std::string& path);

    // Non-streaming inference
    // audio: raw waveform (samples,) at 16kHz
    DecodingResult transcribe(const array& audio) const;

    // Batch inference
    // audio: (batch, max_samples) with padding
    // lengths: actual sample lengths per batch item
    std::vector<DecodingResult> transcribe_batch(
        const array& audio,
        const std::vector<int>& lengths
    ) const;

    // Access components
    const ZipformerEncoder& encoder() const { return encoder_; }
    const Decoder& decoder() const { return decoder_; }
    const Joiner& joiner() const { return joiner_; }
    const FbankExtractor& features() const { return features_; }
    const ASRModelConfig& config() const { return config_; }

private:
    ASRModelConfig config_;

    FbankExtractor features_;
    ZipformerEncoder encoder_;
    Decoder decoder_;
    Joiner joiner_;

    // Greedy search decoding
    DecodingResult greedy_search(const array& encoder_out) const;

    // Beam search decoding
    DecodingResult beam_search(const array& encoder_out) const;
};

/**
 * State for streaming inference.
 */
struct StreamingState {
    CacheState encoder_cache;
    DecoderState decoder_state;
    array feature_buffer{zeros({0})};  // Buffered audio samples
    int processed_samples{0};

    void reset();
};

/**
 * Streaming ASR pipeline.
 *
 * Processes audio chunks incrementally and returns partial results.
 */
class StreamingASR {
public:
    explicit StreamingASR(const ASRModelConfig& config);

    // Load model weights
    void load_weights(const std::string& path);

    // Process audio chunk and get partial result
    // chunk: raw audio samples (at 16kHz)
    // is_final: whether this is the last chunk
    DecodingResult process_chunk(const array& chunk, bool is_final = false);

    // Reset state for new utterance
    void reset();

    // Get current transcription
    const DecodingResult& current_result() const { return current_result_; }

private:
    ASRModelConfig config_;
    std::unique_ptr<ASRModel> model_;
    StreamingState state_;
    DecodingResult current_result_;

    // Chunk processing parameters
    int chunk_size_samples_;  // Samples per chunk
    int lookahead_samples_;   // Future context for streaming encoder
};

/**
 * Load model from checkpoint directory.
 *
 * Expects:
 *   - model.safetensors or model.gguf
 *   - config.json or config.yaml
 *   - tokenizer files
 */
std::unique_ptr<ASRModel> load_model(const std::string& checkpoint_dir);

/**
 * Load streaming model from checkpoint.
 */
std::unique_ptr<StreamingASR> load_streaming_model(const std::string& checkpoint_dir);

} // namespace zipformer
