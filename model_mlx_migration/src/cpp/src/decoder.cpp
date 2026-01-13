// Copyright 2024-2025 Andrew Yates
// Decoder implementation for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/decoder.hpp"
#include <stdexcept>

namespace zipformer {

// ============================================================================
// Decoder
// ============================================================================

Decoder::Decoder(const DecoderConfig& config)
    : config_(config)
    , embedding_weight_(zeros({config.vocab_size, config.decoder_dim}))
    , conv_proj_(config.context_size * config.decoder_dim, config.decoder_dim, true)
    , out_proj_(config.decoder_dim, config.decoder_dim, true)
{
}

void Decoder::load_weights(const WeightMap& weights, const std::string& prefix) {
    if (zipformer::has_weight(weights, prefix + ".embedding.weight")) {
        embedding_weight_ = zipformer::get_weight(weights, prefix + ".embedding.weight");
    }
    conv_proj_.load_weights(weights, prefix + ".conv_proj");
    out_proj_.load_weights(weights, prefix + ".out_proj");
}

array Decoder::forward(const array& tokens) const {
    // tokens: (batch, context_size) - previous token IDs
    // Returns: (batch, decoder_dim)

    int batch = tokens.shape()[0];
    int context = tokens.shape()[1];

    // Gather embeddings for each token
    array embedded = take(embedding_weight_, flatten(tokens), 0);
    embedded = reshape(embedded, {batch, context * config_.decoder_dim});

    // Project context
    array proj = conv_proj_.forward(embedded);

    // Activation and output projection
    proj = multiply(proj, sigmoid(proj));  // SiLU/Swish
    return out_proj_.forward(proj);
}

array Decoder::get_blank_representation() const {
    // Get representation for blank token (typically token 0)
    array blank_ids = full({1, config_.context_size}, config_.blank_id, int32);
    return forward(blank_ids);
}

// ============================================================================
// DecoderState
// ============================================================================

void DecoderState::init(int context_size, int blank_id) {
    context.clear();
    context.resize(context_size, blank_id);
    decoder_rep = zeros({1});  // Sentinel - will be recomputed on first use
}

void DecoderState::update(int new_token, const Decoder& decoder) {
    // Shift context and add new token
    context.erase(context.begin());
    context.push_back(new_token);

    // Recompute decoder representation
    array token_ids = array(context.data(), {1, static_cast<int>(context.size())}, int32);
    decoder_rep = decoder.forward(token_ids);
}

} // namespace zipformer
