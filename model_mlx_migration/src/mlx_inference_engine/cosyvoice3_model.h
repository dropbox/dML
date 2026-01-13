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

// CosyVoice3 Model - C++ MLX Implementation
// High-performance TTS: Text -> LLM -> DiT Flow -> Vocoder -> Audio
//
// Target: 70x+ RTF (exceeding Kokoro's 67.8x)
// Architecture:
//   1. Qwen2 LLM (642M params) - Text to speech tokens
//   2. DiT Flow (113M params) - Speech tokens to mel spectrogram
//   3. CausalHiFT Vocoder (21M params) - Mel to 24kHz audio
//
// Optimizations:
//   - KV cache for LLM autoregressive generation
//   - mx::fast::scaled_dot_product_attention for DiT
//   - mx::fast::rope for rotary position embeddings
//   - mx::compile for graph optimization
//   - Quantization support (4-bit LLM)
//   - Speaker embedding caching
//   - Streaming inference support

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <functional>

#include "mlx/mlx.h"
#include "llm_model.h"  // Reuse LLM infrastructure

namespace mx = mlx::core;

namespace cosyvoice3 {

// ============================================================================
// Configuration
// ============================================================================

/**
 * CausalHiFT Vocoder configuration.
 */
struct VocoderConfig {
    // Input/output
    int in_channels = 80;           // Mel channels
    int sample_rate = 24000;        // Output sample rate

    // Generator
    int base_channels = 512;
    std::vector<int> upsample_rates = {8, 5, 3};        // Total: 120x
    std::vector<int> upsample_kernel_sizes = {16, 11, 7};

    // ResBlocks - 3 per upsample stage = 9 total
    std::vector<int> resblock_kernel_sizes = {3, 7, 11};
    std::vector<std::vector<int>> resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};

    // Source module
    int nb_harmonics = 8;
    float nsf_alpha = 0.1f;
    float nsf_sigma = 0.003f;
    float nsf_voiced_threshold = 10.0f;

    // Source ResBlocks
    std::vector<int> source_resblock_kernel_sizes = {7, 7, 11};
    std::vector<std::vector<int>> source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    std::vector<int> source_down_channels = {256, 128, 64};
    std::vector<int> source_down_kernels = {30, 6, 1};

    // iSTFT
    int istft_n_fft = 16;
    int istft_hop_len = 4;

    // F0 predictor
    int f0_channels = 512;
    int f0_num_convs = 5;
    std::vector<int> f0_kernel_sizes = {4, 3, 3, 3, 3};

    // Other
    float lrelu_slope = 0.1f;
    float audio_limit = 0.99f;

    int total_upsample_factor() const {
        int factor = 1;
        for (int r : upsample_rates) factor *= r;
        return factor;
    }
};

/**
 * DiT Flow model configuration.
 */
struct DiTConfig {
    // Core dimensions
    int dim = 1024;                 // Transformer dimension
    int depth = 22;                 // Number of DiT blocks
    int heads = 16;                 // Number of attention heads
    int dim_head = 64;              // Dimension per head
    int ff_mult = 2;                // FFN multiplier

    // Audio dimensions
    int mel_dim = 80;               // Mel spectrogram dimension
    int mu_dim = 80;                // Text embedding dimension
    int spk_dim = 80;               // Speaker embedding dimension
    int out_channels = 80;          // Output channels

    // Streaming/causal settings
    int static_chunk_size = 50;     // 25 tokens * 2
    int num_decoding_left_chunks = -1;  // -1 = use all

    // Flow matching
    int vocab_size = 6561;          // Speech token vocabulary
    int input_frame_rate = 25;      // Tokens per second
    int token_mel_ratio = 2;        // Mel frames per token

    // Pre-lookahead
    int pre_lookahead_len = 3;
    int pre_lookahead_channels = 1024;

    // Other
    float dropout = 0.0f;
    bool use_long_skip = true;

    // RoPE
    float rope_theta = 10000.0f;

    // Time embedding
    int sinusoidal_dim = 256;       // Sinusoidal embedding dimension
};

/**
 * Full CosyVoice3 configuration.
 */
struct CosyVoice3Config {
    // Sample rate
    int sample_rate = 24000;

    // Token settings
    int token_frame_rate = 25;      // 25 tokens per second
    int token_mel_ratio = 2;        // 2 mel frames per token
    int speech_token_size = 6561;   // Speech token vocabulary

    // Streaming
    int chunk_size = 25;            // Streaming chunk size in tokens

    // Component configs
    llm::LLMConfig llm_config;
    DiTConfig flow_config;
    VocoderConfig vocoder_config;

    // Initialize with Qwen2 defaults for CosyVoice3
    // Qwen2 GQA: 7 query heads, 1 key-value head, head_dim=128
    void init_qwen2_config() {
        llm_config.vocab_size = 151936;
        llm_config.hidden_size = 896;
        llm_config.intermediate_size = 4864;
        llm_config.num_hidden_layers = 24;
        llm_config.num_attention_heads = 7;    // 7 query heads (GQA)
        llm_config.num_key_value_heads = 1;    // 1 KV head (GQA)
        llm_config.head_dim = 128;             // 896 / 7 = 128
        llm_config.rope_theta = 1000000.0f;
        llm_config.attention_bias = true;      // Qwen2 has QKV biases
        llm_config.rms_norm_eps = 1e-6f;
        llm_config.model_type = "qwen2";
    }

    // Create default config
    static CosyVoice3Config create_default() {
        CosyVoice3Config config;
        config.init_qwen2_config();
        return config;
    }
};

// ============================================================================
// Weight Storage
// ============================================================================

/**
 * Weight storage for CosyVoice3 components.
 */
class CosyVoice3Weights {
public:
    CosyVoice3Weights() = default;

    // Load all weights from directory
    void load(const std::string& model_path);

    // Load specific component weights
    void load_llm(const std::string& path);
    void load_flow(const std::string& path);
    void load_vocoder(const std::string& path);
    void load_combined(const std::string& path);  // MLX single-file format

    // Get weight by name
    mx::array get(const std::string& name) const;
    bool has(const std::string& name) const;

    // Get weight-normalized conv weight (computes w = g * v / ||v||)
    mx::array get_weight_norm_conv(const std::string& prefix) const;

    // Get transposed conv weight (PyTorch [out,in,k] -> MLX [out,k,in])
    mx::array get_transposed_conv(const std::string& name) const;

    // Component weight counts
    size_t llm_weight_count() const { return llm_weights_.size(); }
    size_t flow_weight_count() const { return flow_weights_.size(); }
    size_t vocoder_weight_count() const { return vocoder_weights_.size(); }
    size_t total_weight_count() const {
        return llm_weight_count() + flow_weight_count() + vocoder_weight_count();
    }

private:
    std::unordered_map<std::string, mx::array> llm_weights_;
    std::unordered_map<std::string, mx::array> flow_weights_;
    std::unordered_map<std::string, mx::array> vocoder_weights_;
};

// ============================================================================
// CausalHiFT Vocoder
// ============================================================================

/**
 * Snake activation: x + (1/alpha) * sin^2(alpha * x)
 */
class SnakeActivation {
public:
    SnakeActivation(int channels);

    mx::array forward(const mx::array& x);  // Input: [B, C, L] (PyTorch format)
    mx::array forward_mlx(const mx::array& x);  // Input: [B, L, C] (MLX native format)

    void load_weights(const mx::array& alpha);

private:
    mx::array alpha_;
    int channels_;
};

/**
 * ResBlock with Snake activations.
 */
class VocoderResBlock {
public:
    VocoderResBlock(int channels, int kernel_size, const std::vector<int>& dilations);

    mx::array forward(const mx::array& x);

    void load_weights(
        const std::vector<mx::array>& conv1_weights,
        const std::vector<mx::array>& conv1_biases,
        const std::vector<mx::array>& conv2_weights,
        const std::vector<mx::array>& conv2_biases,
        const std::vector<mx::array>& alpha1,
        const std::vector<mx::array>& alpha2
    );

private:
    int channels_;
    int kernel_size_;
    std::vector<int> dilations_;

    std::vector<mx::array> conv1_weights_;
    std::vector<mx::array> conv1_biases_;
    std::vector<mx::array> conv2_weights_;
    std::vector<mx::array> conv2_biases_;
    std::vector<SnakeActivation> activations1_;
    std::vector<SnakeActivation> activations2_;
};

/**
 * F0 Predictor for source-filter vocoder.
 */
class F0Predictor {
public:
    F0Predictor(const VocoderConfig& config);

    mx::array forward(const mx::array& mel);

    void load_weights(
        const std::vector<mx::array>& conv_weights,
        const std::vector<mx::array>& conv_biases,
        const mx::array& classifier_weight,
        const mx::array& classifier_bias
    );

private:
    VocoderConfig config_;
    std::vector<mx::array> conv_weights_;
    std::vector<mx::array> conv_biases_;
    mx::array classifier_weight_;
    mx::array classifier_bias_;
};

/**
 * Neural source-filter source module.
 */
class SourceModule {
public:
    SourceModule(const VocoderConfig& config);

    mx::array forward(const mx::array& f0, int upsample_factor);

    void load_weights(const mx::array& linear_weight, const mx::array& linear_bias);

private:
    VocoderConfig config_;
    mx::array linear_weight_;
    mx::array linear_bias_;
};

/**
 * CausalHiFT Generator - Mel to Audio.
 *
 * Architecture:
 *   mel -> conv_pre -> [ups + source + resblocks] -> conv_post -> iSTFT -> audio
 *
 * Optimizations:
 *   - Fused Snake activation (custom Metal kernel when available)
 *   - mx::compile for graph optimization
 *   - Streaming support with internal state
 */
class CausalHiFTGenerator {
public:
    CausalHiFTGenerator(const VocoderConfig& config);

    /**
     * Generate audio from mel spectrogram.
     * @param mel Mel spectrogram [B, mel_dim, L]
     * @param f0 Optional pre-computed F0 [B, 1, L]
     * @return Audio waveform [B, L * upsample_factor]
     */
    mx::array forward(const mx::array& mel, const mx::array* f0 = nullptr);

    /**
     * Generate audio with streaming support.
     * @param mel Current mel chunk [B, mel_dim, L]
     * @param is_first First chunk (resets state)
     * @param is_last Last chunk (finalizes output)
     * @return Audio chunk
     */
    mx::array forward_streaming(
        const mx::array& mel,
        bool is_first = false,
        bool is_last = true
    );

    // Weight loading
    void load_weights(const CosyVoice3Weights& weights);

    // Compile for faster inference
    void compile();
    bool is_compiled() const { return compiled_; }

    // Config access
    const VocoderConfig& config() const { return config_; }

private:
    VocoderConfig config_;
    bool compiled_ = false;

    // Layers
    F0Predictor f0_predictor_;
    SourceModule source_module_;

    // Pre/post convolutions
    mx::array conv_pre_weight_;
    mx::array conv_pre_bias_;
    mx::array conv_post_weight_;
    mx::array conv_post_bias_;

    // Upsampling layers
    std::vector<mx::array> up_weights_;
    std::vector<mx::array> up_biases_;
    std::vector<SnakeActivation> up_activations_;

    // ResBlocks
    std::vector<VocoderResBlock> resblocks_;

    // Source downsampling and ResBlocks
    std::vector<mx::array> source_down_weights_;
    std::vector<mx::array> source_down_biases_;
    std::vector<VocoderResBlock> source_resblocks_;

    // Streaming state
    mx::array streaming_state_;

    // Internal methods
    mx::array istft(const mx::array& x);
    mx::array snake(const mx::array& x, const mx::array& alpha);
    mx::array leaky_relu(const mx::array& x, float slope = 0.1f);
};

// ============================================================================
// DiT Flow Model
// ============================================================================

/**
 * DiT Attention with RoPE and fast operations.
 *
 * Uses:
 *   - mx::fast::scaled_dot_product_attention
 *   - mx::fast::rope for rotary embeddings
 *   - Fused QKV computation (Q18 optimization)
 */
class DiTAttention {
public:
    DiTAttention(const DiTConfig& config);

    mx::array forward(
        const mx::array& x,
        const mx::array& inv_freq,
        int offset = 0,
        const mx::array* mask = nullptr,
        bool streaming = false
    );

    void load_weights(
        const mx::array& q_weight, const mx::array& q_bias,
        const mx::array& k_weight, const mx::array& k_bias,
        const mx::array& v_weight, const mx::array& v_bias,
        const mx::array& out_weight, const mx::array& out_bias
    );

    // Fuse QKV weights for faster matmul (Q18 optimization)
    void fuse_qkv_weights();
    bool is_qkv_fused() const { return qkv_fused_; }

private:
    DiTConfig config_;
    bool qkv_fused_ = false;

    // Separate QKV (for weight loading)
    mx::array q_weight_, q_bias_;
    mx::array k_weight_, k_bias_;
    mx::array v_weight_, v_bias_;

    // Fused QKV
    mx::array qkv_weight_, qkv_bias_;

    // Output projection
    mx::array out_weight_, out_bias_;
};

/**
 * DiT Feed-Forward Network.
 */
class DiTFeedForward {
public:
    DiTFeedForward(const DiTConfig& config);

    mx::array forward(const mx::array& x);

    void load_weights(
        const mx::array& w1, const mx::array& b1,
        const mx::array& w2, const mx::array& b2
    );

private:
    DiTConfig config_;
    mx::array w1_, b1_;
    mx::array w2_, b2_;
};

/**
 * Adaptive Layer Normalization for DiT.
 * Produces 6 modulation params: scale1, shift1, gate1, scale2, shift2, gate2
 */
class AdaptiveLayerNorm {
public:
    AdaptiveLayerNorm(int dim);

    // Returns: scale1, shift1, gate1, scale2, shift2, gate2
    std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array>
    forward(const mx::array& x, const mx::array& cond);

    void load_weights(const mx::array& weight, const mx::array& bias);

private:
    int dim_;
    mx::array weight_, bias_;
};

/**
 * Single DiT Block with attention and FFN.
 */
class DiTBlock {
public:
    DiTBlock(const DiTConfig& config);

    mx::array forward(
        const mx::array& x,
        const mx::array& cond,
        const mx::array& inv_freq,
        int offset = 0,
        const mx::array* mask = nullptr,
        bool streaming = false
    );

    void load_weights(
        // Attention weights
        const mx::array& q_w, const mx::array& q_b,
        const mx::array& k_w, const mx::array& k_b,
        const mx::array& v_w, const mx::array& v_b,
        const mx::array& out_w, const mx::array& out_b,
        // Adaptive norm
        const mx::array& norm_w, const mx::array& norm_b,
        // FFN
        const mx::array& ff1_w, const mx::array& ff1_b,
        const mx::array& ff2_w, const mx::array& ff2_b
    );

    void fuse_qkv_weights() { attn_.fuse_qkv_weights(); }

private:
    DiTConfig config_;
    AdaptiveLayerNorm norm_;
    DiTAttention attn_;
    DiTFeedForward ff_;
};

/**
 * Time embedding for diffusion.
 */
class TimeEmbedding {
public:
    TimeEmbedding(int dim, int sinusoidal_dim = 256);

    mx::array forward(const mx::array& t);

    void load_weights(
        const mx::array& w1, const mx::array& b1,
        const mx::array& w2, const mx::array& b2
    );

private:
    int dim_;
    int sinusoidal_dim_;
    mx::array w1_, b1_;
    mx::array w2_, b2_;
};

/**
 * Pre-lookahead convolution layer.
 */
class PreLookaheadLayer {
public:
    PreLookaheadLayer(int in_channels = 80, int hidden_channels = 1024);

    mx::array forward(const mx::array& x);

    void load_weights(
        const mx::array& conv1_w, const mx::array& conv1_b,
        const mx::array& conv2_w, const mx::array& conv2_b
    );

private:
    int in_channels_;
    int hidden_channels_;
    mx::array conv1_w_, conv1_b_;
    mx::array conv2_w_, conv2_b_;
};

/**
 * DiT Flow Model - Speech tokens to mel spectrogram.
 *
 * Architecture:
 *   tokens -> embedding -> pre_lookahead -> DiT blocks -> output projection -> mel
 *
 * Uses Euler ODE solver for inference.
 */
class DiTFlowModel {
public:
    DiTFlowModel(const DiTConfig& config);

    /**
     * Forward pass for training/inference step.
     * @param x Noised mel [B, L, mel_dim]
     * @param tokens Speech tokens [B, L_tokens]
     * @param t Time step [B] in [0, 1]
     * @param spk_emb Speaker embedding [B, 192]
     * @return Predicted velocity [B, L, out_channels]
     */
    mx::array forward(
        const mx::array& x,
        const mx::array& tokens,
        const mx::array& t,
        const mx::array& spk_emb,
        const mx::array* mask = nullptr,
        bool streaming = false
    );

    /**
     * Inference using Euler ODE solver.
     * @param tokens Speech tokens [B, L_tokens]
     * @param spk_emb Speaker embedding [B, 192]
     * @param num_steps Number of ODE steps (default: 10)
     * @param cfg_strength CFG strength (default: 0.7)
     * @return Generated mel [B, L, mel_dim]
     */
    mx::array inference(
        const mx::array& tokens,
        const mx::array& spk_emb,
        int num_steps = 10,
        float cfg_strength = 0.7f,
        bool streaming = false
    );

    // Weight loading
    void load_weights(const CosyVoice3Weights& weights);

    // Optimizations
    void compile();
    void fuse_qkv_weights();

    // Speaker embedding caching (Q6 optimization)
    void cache_speaker_projection(const mx::array& spk_emb, const std::string& cache_id = "default");
    void enable_speaker_cache(bool enable = true);
    void clear_speaker_cache();

    bool is_compiled() const { return compiled_; }
    const DiTConfig& config() const { return config_; }

private:
    DiTConfig config_;
    bool compiled_ = false;
    bool speaker_cache_enabled_ = false;
    std::unordered_map<std::string, mx::array> speaker_cache_;

    // Input embedding
    mx::array input_embedding_;

    // Pre-lookahead
    PreLookaheadLayer pre_lookahead_;

    // Speaker embedding projection
    mx::array spk_proj_weight_, spk_proj_bias_;

    // Input embedding projection (combined input -> transformer dim)
    mx::array input_embed_proj_weight_, input_embed_proj_bias_;

    // Time embedding
    TimeEmbedding time_embed_;

    // RoPE
    mx::array rope_inv_freq_;

    // DiT blocks
    std::vector<DiTBlock> blocks_;

    // Long skip projection
    mx::array skip_proj_weight_, skip_proj_bias_;

    // Output
    mx::array norm_out_weight_, norm_out_bias_;
    mx::array proj_out_weight_, proj_out_bias_;
};

// ============================================================================
// Full CosyVoice3 Model
// ============================================================================

/**
 * Complete CosyVoice3 TTS Model.
 *
 * Pipeline:
 *   Text -> LLM -> Speech Tokens -> DiT Flow -> Mel -> Vocoder -> Audio
 *
 * Performance target: 70x+ RTF (24kHz audio)
 */
class CosyVoice3Model {
public:
    CosyVoice3Model(const CosyVoice3Config& config);
    ~CosyVoice3Model();

    // Disable copy
    CosyVoice3Model(const CosyVoice3Model&) = delete;
    CosyVoice3Model& operator=(const CosyVoice3Model&) = delete;

    // Enable move
    CosyVoice3Model(CosyVoice3Model&&) noexcept;
    CosyVoice3Model& operator=(CosyVoice3Model&&) noexcept;

    /**
     * Load model from directory.
     * Expected files:
     *   - llm/model.safetensors (Qwen2 weights)
     *   - flow.safetensors (DiT weights)
     *   - vocoder.safetensors (CausalHiFT weights)
     *   - config.json
     */
    static CosyVoice3Model load(const std::string& model_path);

    /**
     * Generate speech tokens from text tokens.
     * @param text_ids Text token IDs [B, L_text]
     * @param max_length Maximum tokens to generate
     * @param temperature Sampling temperature
     * @param top_k Top-k sampling
     * @param top_p Top-p sampling
     * @return Speech tokens [B, L_speech]
     */
    mx::array generate_speech_tokens(
        const mx::array& text_ids,
        int max_length = 1000,
        float temperature = 1.0f,
        int top_k = 25,
        float top_p = 0.8f
    );

    /**
     * Convert speech tokens to mel spectrogram.
     * @param tokens Speech tokens [B, L_tokens]
     * @param speaker_emb Speaker embedding [B, 192]
     * @param num_steps ODE steps
     * @param cfg_strength CFG strength
     * @return Mel spectrogram [B, L_mel, mel_dim]
     */
    mx::array tokens_to_mel(
        const mx::array& tokens,
        const mx::array& speaker_emb,
        int num_steps = 10,
        float cfg_strength = 0.7f,
        bool streaming = false
    );

    /**
     * Convert mel spectrogram to audio.
     * @param mel Mel spectrogram [B, mel_dim, L] or [B, L, mel_dim]
     * @return Audio waveform [B, L_audio]
     */
    mx::array mel_to_audio(const mx::array& mel);

    /**
     * End-to-end text-to-speech synthesis.
     * @param text_ids Text token IDs [B, L_text]
     * @param speaker_emb Speaker embedding [B, 192]
     * @return Audio waveform [B, L_audio]
     */
    mx::array synthesize(
        const mx::array& text_ids,
        const mx::array& speaker_emb,
        int max_tokens = 1000,
        float temperature = 1.0f,
        int top_k = 25,
        float top_p = 0.8f,
        int flow_steps = 10,
        float cfg_strength = 0.7f
    );

    /**
     * Streaming callback for audio chunks.
     */
    using StreamCallback = std::function<void(const mx::array& audio_chunk, bool is_last)>;

    /**
     * Streaming synthesis with callback.
     * @param text_ids Text token IDs
     * @param speaker_emb Speaker embedding
     * @param callback Callback for each audio chunk
     * @param chunk_size Tokens per chunk
     */
    void synthesize_streaming(
        const mx::array& text_ids,
        const mx::array& speaker_emb,
        StreamCallback callback,
        int chunk_size = 25
    );

    // Optimizations
    void compile();
    void optimize_all();  // Apply all optimizations

    // Weight loading
    void load_weights(const CosyVoice3Weights& weights);

    // Model info
    bool is_loaded() const { return loaded_; }
    const CosyVoice3Config& config() const { return config_; }
    std::string info() const;

    // Component access
    llm::LLMModel* llm() { return llm_.get(); }
    DiTFlowModel* flow() { return flow_.get(); }
    CausalHiFTGenerator* vocoder() { return vocoder_.get(); }

private:
    CosyVoice3Config config_;
    bool loaded_ = false;
    bool llm_loaded_ = false;

    // Components
    std::unique_ptr<llm::LLMModel> llm_;
    std::unique_ptr<DiTFlowModel> flow_;
    std::unique_ptr<CausalHiFTGenerator> vocoder_;

    // LLM KV cache for autoregressive generation
    llm::KVCache kv_cache_;

    // CosyVoice3 LLM-specific weights (dual embedding/decoder architecture)
    // These are separate from the standard LLM because CosyVoice3's LLM has:
    // - Dual embeddings: text (151936) + speech (6564)
    // - Dual outputs: lm_head (text) + llm_decoder (speech)
    mx::array llm_embedding_;        // [2, 896] - SOS/EOS special tokens
    mx::array speech_embedding_;     // [6564, 896] - Speech token embeddings
    mx::array embed_tokens_;         // [151936, 896] - Text token embeddings
    mx::array llm_decoder_weight_;   // [6564, 896] - Speech logits projection
    mx::array llm_decoder_bias_;     // [6564] - Speech logits bias
    mx::array lm_head_;              // [151936, 896] - Text logits projection (tied or separate)

    // Qwen2 transformer weights (24 layers) - using maps since mx::array has no default ctor
    // Each layer has: q_proj, k_proj, v_proj, o_proj (attention)
    //                 gate_proj, up_proj, down_proj (MLP)
    //                 input_layernorm, post_attention_layernorm
    std::unordered_map<int, mx::array> q_proj_w_, q_proj_b_;
    std::unordered_map<int, mx::array> k_proj_w_, k_proj_b_;
    std::unordered_map<int, mx::array> v_proj_w_, v_proj_b_;
    std::unordered_map<int, mx::array> o_proj_w_;
    std::unordered_map<int, mx::array> gate_proj_w_, up_proj_w_, down_proj_w_;
    std::unordered_map<int, mx::array> input_layernorm_w_, post_attn_layernorm_w_;
    mx::array final_norm_w_;

    // RoPE parameters
    mx::array rope_freqs_;
    float rope_theta_ = 1000000.0f;

    // Internal helper methods for LLM
    void load_llm_weights(const CosyVoice3Weights& weights);
    mx::array llm_forward(const mx::array& hidden_states, llm::KVCache* cache);
    mx::array llm_attention(const mx::array& x, int layer_idx, llm::KVCache* cache);
    mx::array llm_mlp(const mx::array& x, int layer_idx);
    mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps = 1e-6f);
    mx::array apply_rope(const mx::array& x, int offset);
    mx::array sample_token(const mx::array& logits, float temperature, int top_k, float top_p);
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Compute weight normalization: w = g * (v / ||v||)
 */
mx::array compute_weight_norm(const mx::array& g, const mx::array& v);

/**
 * Transpose Conv1d weight from PyTorch [out, in, kernel] to MLX [out, kernel, in]
 */
mx::array transpose_conv_weight(const mx::array& w);

} // namespace cosyvoice3
