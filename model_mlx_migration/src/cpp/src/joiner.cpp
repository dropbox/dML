// Copyright 2024-2025 Andrew Yates
// Joiner implementation for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/joiner.hpp"
#include <stdexcept>

namespace zipformer {

Joiner::Joiner(const JoinerConfig& config)
    : config_(config)
    , encoder_proj_(config.encoder_dim, config.joiner_dim, true)
    , decoder_proj_(config.decoder_dim, config.joiner_dim, true)
    , output_proj_(config.joiner_dim, config.vocab_size, true)
{
}

void Joiner::load_weights(const WeightMap& weights, const std::string& prefix) {
    encoder_proj_.load_weights(weights, prefix + ".encoder_proj");
    decoder_proj_.load_weights(weights, prefix + ".decoder_proj");
    output_proj_.load_weights(weights, prefix + ".output_proj");
}

array Joiner::forward(const array& encoder_out, const array& decoder_out) const {
    // encoder_out: (batch, time, encoder_dim)
    // decoder_out: (batch, decoder_dim)
    // Returns: (batch, time, vocab_size)

    // Project encoder output
    array enc_proj = encoder_proj_.forward(encoder_out);  // (batch, time, joiner_dim)

    // Project decoder output and expand for broadcasting
    array dec_proj = decoder_proj_.forward(decoder_out);  // (batch, joiner_dim)
    dec_proj = expand_dims(dec_proj, 1);  // (batch, 1, joiner_dim)

    // Combine with tanh activation
    array joint = tanh(add(enc_proj, dec_proj));  // (batch, time, joiner_dim)

    // Output projection to vocabulary
    return output_proj_.forward(joint);  // (batch, time, vocab_size)
}

array Joiner::forward_single(const array& encoder_frame, const array& decoder_rep) const {
    // encoder_frame: (batch, encoder_dim)
    // decoder_rep: (batch, decoder_dim)
    // Returns: (batch, vocab_size)

    // Project both
    array enc_proj = encoder_proj_.forward(encoder_frame);
    array dec_proj = decoder_proj_.forward(decoder_rep);

    // Combine
    array joint = tanh(add(enc_proj, dec_proj));

    // Output
    return output_proj_.forward(joint);
}

} // namespace zipformer
