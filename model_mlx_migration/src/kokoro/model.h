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

// Kokoro MLX C++ Model Implementation
// Weight loading and forward pass for Kokoro TTS

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace kokoro {

/**
 * Configuration for Kokoro model.
 * Loaded from config.json in the model directory.
 */
struct KokoroConfig {
    // Text encoder
    int dim_in = 64;
    int hidden_dim = 512;
    int style_dim = 128;
    int max_conv_dim = 512;
    int n_token = 178;
    int n_mels = 80;
    int n_layer = 3;
    int max_dur = 50;
    float dropout = 0.2f;
    int text_encoder_kernel_size = 5;
    bool multispeaker = true;

    // ALBERT/BERT
    int plbert_hidden_size = 768;
    int plbert_num_attention_heads = 12;
    int plbert_intermediate_size = 2048;
    int plbert_max_position_embeddings = 512;
    int plbert_num_hidden_layers = 12;
    float plbert_dropout = 0.1f;
    int albert_embedding_dim = 128;

    // ISTFT Generator
    std::vector<int> istft_upsample_rates = {10, 6};
    std::vector<int> istft_upsample_kernel_sizes = {20, 12};
    int istft_gen_istft_n_fft = 20;
    int istft_gen_istft_hop_size = 5;
    std::vector<int> istft_resblock_kernel_sizes = {3, 7, 11};
    std::vector<std::vector<int>> istft_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    int istft_upsample_initial_channel = 512;

    // Audio
    int sample_rate = 24000;
    int hop_size = 256;

    // Special tokens
    int vocab_size = 178;
    int bos_token_id = 0;
    int eos_token_id = 0;
    bool weight_norm_folded = true;

    // Load from JSON file
    static KokoroConfig load(const std::string& path);
};

/**
 * Weight storage for Kokoro model.
 * Maps weight names to MLX arrays.
 */
class Weights {
public:
    Weights() = default;

    // Load weights from safetensors file
    void load(const std::string& path);

    // Get a weight by name (throws if not found)
    mx::array get(const std::string& name) const;

    // Check if weight exists
    bool has(const std::string& name) const;

    // Get all weights with prefix
    std::vector<std::pair<std::string, mx::array>> get_prefix(const std::string& prefix) const;

    // Number of weights
    size_t size() const { return weights_.size(); }

    // Iterate over all weights
    auto begin() const { return weights_.begin(); }
    auto end() const { return weights_.end(); }

private:
    std::unordered_map<std::string, mx::array> weights_;
};

/**
 * Voice pack for Kokoro.
 * Contains style embeddings indexed by phoneme length.
 */
class VoicePack {
public:
    VoicePack() = default;

    // Load from safetensors file
    void load(const std::string& path);

    // Select style embedding for given phoneme length
    mx::array select(int phoneme_length) const;

    // Check if loaded
    bool loaded() const { return !embeddings_.empty(); }

private:
    std::vector<mx::array> embeddings_;  // Indexed by phoneme length
};

/**
 * Kokoro TTS Model Implementation.
 *
 * This class holds all model weights and provides forward pass methods.
 * The model is organized into components:
 * - BERT: Text understanding
 * - Predictor: Duration/F0/Noise prediction
 * - Decoder: Audio synthesis
 */
class KokoroModel {
public:
    KokoroModel() = default;

    // Load model from directory
    // Expected structure:
    //   weights.safetensors
    //   config.json
    //   vocab/phonemes.json
    //   voices/*.safetensors
    static KokoroModel load(const std::string& model_path);

    // Get model configuration
    const KokoroConfig& config() const { return config_; }

    // Load a voice pack
    void load_voice(const std::string& name, const std::string& path);

    // Get available voices
    std::vector<std::string> available_voices() const;

    // Check if voice is loaded
    bool has_voice(const std::string& name) const;

    // Full text-to-audio synthesis
    // tokens: token IDs including BOS/EOS [batch, seq_len]
    // voice: voice name
    // speed: speaking rate (1.0 = normal)
    // Returns: audio samples [batch, samples]
    mx::array synthesize(
        const mx::array& tokens,
        const std::string& voice,
        float speed = 1.0f
    );

    // Get the weight at a specific key (for debugging)
    mx::array get_weight(const std::string& key) const {
        return weights_.get(key);
    }

    // Check if a weight exists
    bool has_weight(const std::string& key) const {
        return weights_.has(key);
    }

    // Get number of weights
    size_t num_weights() const { return weights_.size(); }

    // Check if model is loaded
    bool loaded() const { return loaded_; }

private:
    // BERT embeddings (word + position + type + layernorm)
    mx::array bert_embeddings(const mx::array& input_ids);

    // ALBERT attention layer
    mx::array albert_attention(const mx::array& hidden, const mx::array& attention_mask);

    // ALBERT FFN layer
    mx::array albert_ffn(const mx::array& hidden);

    // BERT forward pass (embeddings + 12 ALBERT layers)
    mx::array bert_forward(const mx::array& tokens, const mx::array& attention_mask);

    // Text encoder forward pass (Conv1d + BiLSTM)
    // Takes token IDs, returns encoded features [batch, seq_len, 512]
    mx::array text_encoder_forward(const mx::array& tokens);

    // Predictor text encoder (style-conditioned BiLSTM stack)
    mx::array predictor_text_encoder(const mx::array& bert_enc, const mx::array& style);

    // Compute alignment from duration logits
    // Returns: (indices, bucket_size, actual_frames)
    std::tuple<mx::array, int, int> compute_alignment(const mx::array& duration_logits, float speed);

    // Expand features from text rate to audio frame rate
    mx::array expand_features(const mx::array& features, const mx::array& indices, int total_frames);

    // Predictor forward pass (duration, F0, noise)
    // style: voice_embed[0:128] - for text_encoder
    // speaker: voice_embed[128:256] - for F0/N blocks
    // Returns: (asr_features, f0, noise, style_128, actual_frames)
    std::tuple<mx::array, mx::array, mx::array, mx::array, int> predictor_forward(
        const mx::array& bert_enc,
        const mx::array& text_enc,
        const mx::array& style,
        const mx::array& speaker,
        float speed
    );

    // Decoder forward pass
    mx::array decoder_forward(
        const mx::array& asr_features,
        const mx::array& f0,
        const mx::array& noise,
        const mx::array& style
    );

    KokoroConfig config_;
    Weights weights_;
    std::unordered_map<std::string, VoicePack> voices_;
    bool loaded_ = false;
};

}  // namespace kokoro
