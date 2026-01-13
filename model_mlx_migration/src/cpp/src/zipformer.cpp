// Copyright 2024-2025 Andrew Yates
// Full Zipformer ASR model implementation for MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/zipformer.hpp"
#include <fstream>
#include <stdexcept>

namespace zipformer {

// ============================================================================
// ASRModelConfig
// ============================================================================

ASRModelConfig ASRModelConfig::default_streaming() {
    ASRModelConfig config;

    // Encoder config (streaming Zipformer)
    config.encoder.num_features = 80;
    config.encoder.num_encoder_layers = {2, 2, 3, 4, 3, 2};
    config.encoder.encoder_dims = {192, 256, 384, 512, 384, 256};
    config.encoder.attention_dims = {128, 128, 128, 256, 128, 128};
    config.encoder.ff1_dims = {384, 576, 768, 1152, 768, 576};
    config.encoder.ff2_dims = {512, 768, 1024, 1536, 1024, 768};
    config.encoder.ff3_dims = {640, 960, 1280, 1920, 1280, 960};
    config.encoder.num_heads = {4, 4, 4, 8, 4, 4};
    config.encoder.downsampling_factors = {1, 2, 4, 8, 4, 2};
    config.encoder.cnn_module_kernels = {31, 31, 15, 15, 15, 31};
    config.encoder.causal = true;

    // Decoder config
    config.decoder.vocab_size = 500;
    config.decoder.decoder_dim = 512;
    config.decoder.blank_id = 0;
    config.decoder.context_size = 2;

    // Joiner config
    config.joiner.encoder_dim = 512;
    config.joiner.decoder_dim = 512;
    config.joiner.joiner_dim = 512;
    config.joiner.vocab_size = 500;

    // Feature config
    config.features.sample_rate = 16000;
    config.features.num_mel_bins = 80;

    return config;
}

// ============================================================================
// ASRModel
// ============================================================================

ASRModel::ASRModel(const ASRModelConfig& config)
    : config_(config)
    , features_(config.features)
    , encoder_(config.encoder)
    , decoder_(config.decoder)
    , joiner_(config.joiner)
{
}

void ASRModel::load_weights(const std::string& path) {
    WeightMap weights = zipformer::load_weights(path);

    encoder_.load_weights(weights, "encoder");
    decoder_.load_weights(weights, "decoder");
    joiner_.load_weights(weights, "joiner");
}

DecodingResult ASRModel::transcribe(const array& audio) const {
    // Extract features
    array feats = features_.extract(audio);

    // Add batch dimension
    feats = expand_dims(feats, 0);

    // Encode
    array encoder_out = encoder_.forward(feats);

    // Decode
    return greedy_search(encoder_out);
}

std::vector<DecodingResult> ASRModel::transcribe_batch(
    const array& audio,
    const std::vector<int>& lengths
) const {
    // Extract features for batch
    array feats = features_.extract(audio);

    // Encode
    array encoder_out = encoder_.forward(feats);

    // Decode each item
    std::vector<DecodingResult> results;
    for (size_t i = 0; i < lengths.size(); ++i) {
        // Get this item's encoder output
        array item_out = slice(encoder_out, {static_cast<int>(i), 0, 0},
                               {static_cast<int>(i) + 1,
                                encoder_out.shape()[1],
                                encoder_out.shape()[2]});
        results.push_back(greedy_search(item_out));
    }

    return results;
}

DecodingResult ASRModel::greedy_search(const array& encoder_out) const {
    DecodingResult result;
    result.tokens.clear();
    result.confidence = 1.0f;

    int T = encoder_out.shape()[1];  // Time steps

    // Initialize decoder state with blank tokens
    std::vector<int> context(config_.decoder.context_size, config_.decoder.blank_id);
    array context_ids = array(context.data(), {1, static_cast<int>(context.size())}, int32);
    array decoder_rep = decoder_.forward(context_ids);

    // Greedy search through time steps
    for (int t = 0; t < T; ++t) {
        // Get encoder frame
        array enc_frame = slice(encoder_out, {0, t, 0},
                               {1, t + 1, encoder_out.shape()[2]});
        enc_frame = squeeze(enc_frame, 1);

        // Get joiner output (logits)
        array logits = joiner_.forward_single(enc_frame, decoder_rep);
        eval(logits);

        // Get best token
        array best_idx = argmax(logits, -1);
        eval(best_idx);
        int token = best_idx.item<int>();

        // If not blank, emit token and update decoder
        if (token != config_.decoder.blank_id) {
            result.tokens.push_back(token);

            // Update context
            context.erase(context.begin());
            context.push_back(token);
            context_ids = array(context.data(), {1, static_cast<int>(context.size())}, int32);
            decoder_rep = decoder_.forward(context_ids);

            // Store timestamp (simplified - frame index to seconds)
            float time = static_cast<float>(t) * config_.features.frame_shift_ms / 1000.0f;
            result.timestamps.push_back(time);
        }
    }

    // TODO: Convert tokens to text using tokenizer
    result.text = "[tokens: " + std::to_string(result.tokens.size()) + "]";

    return result;
}

DecodingResult ASRModel::beam_search(const array& encoder_out) const {
    // Beam search implementation - for now, fall back to greedy
    return greedy_search(encoder_out);
}

// ============================================================================
// StreamingState
// ============================================================================

void StreamingState::reset() {
    encoder_cache.reset();
    decoder_state = DecoderState();
    feature_buffer = zeros({0});  // Empty buffer
    processed_samples = 0;
}

// ============================================================================
// StreamingASR
// ============================================================================

StreamingASR::StreamingASR(const ASRModelConfig& config)
    : config_(config)
    , model_(std::make_unique<ASRModel>(config))
{
    // Default chunk size: 320ms
    chunk_size_samples_ = static_cast<int>(config.features.sample_rate * 0.32f);
    lookahead_samples_ = 0;  // No lookahead for now
}

void StreamingASR::load_weights(const std::string& path) {
    model_->load_weights(path);
}

DecodingResult StreamingASR::process_chunk(const array& chunk, bool is_final) {
    // Buffer audio
    if (state_.feature_buffer.size() > 0) {
        state_.feature_buffer = concatenate({state_.feature_buffer, chunk}, 0);
    } else {
        state_.feature_buffer = chunk;
    }

    // Extract features if we have enough samples
    int samples = state_.feature_buffer.shape()[0];
    int min_samples = model_->features().get_num_samples(1);

    if (samples < min_samples && !is_final) {
        return current_result_;
    }

    // Extract features
    array feats = model_->features().extract(state_.feature_buffer);

    if (feats.shape()[0] == 0) {
        return current_result_;
    }

    // Add batch dimension
    feats = expand_dims(feats, 0);

    // Encode (streaming)
    array encoder_out = model_->encoder().streaming_forward(feats, state_.encoder_cache);

    // Decode new frames
    // (Simplified - full impl would do incremental decoding)
    current_result_ = model_->transcribe(state_.feature_buffer);

    // Update processed samples
    state_.processed_samples += state_.feature_buffer.shape()[0];

    // Clear buffer (streaming processes all available)
    state_.feature_buffer = zeros({0});

    return current_result_;
}

void StreamingASR::reset() {
    state_.reset();
    current_result_ = DecodingResult();
    current_result_.text = "";
}

// ============================================================================
// Model Loading
// ============================================================================

std::unique_ptr<ASRModel> load_model(const std::string& checkpoint_dir) {
    // Try to find model file
    std::string model_path = checkpoint_dir + "/model.safetensors";

    // Check if file exists
    std::ifstream f(model_path);
    if (!f.good()) {
        // Try GGUF
        model_path = checkpoint_dir + "/model.gguf";
        f = std::ifstream(model_path);
        if (!f.good()) {
            throw std::runtime_error("Could not find model file in " + checkpoint_dir);
        }
    }

    // Load with default config for now
    // TODO: Load config from config.json
    auto config = ASRModelConfig::default_streaming();
    auto model = std::make_unique<ASRModel>(config);
    model->load_weights(model_path);

    return model;
}

std::unique_ptr<StreamingASR> load_streaming_model(const std::string& checkpoint_dir) {
    auto config = ASRModelConfig::default_streaming();
    auto model = std::make_unique<StreamingASR>(config);

    std::string model_path = checkpoint_dir + "/model.safetensors";
    model->load_weights(model_path);

    return model;
}

} // namespace zipformer
