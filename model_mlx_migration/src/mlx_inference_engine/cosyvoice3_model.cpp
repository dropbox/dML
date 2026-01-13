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
// Minimal skeleton for compilation - to be incrementally enhanced

#include "cosyvoice3_model.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

#include "mlx/fft.h"  // For irfft

namespace cosyvoice3 {

// ============================================================================
// Helper activation functions (file-local)
// ============================================================================

// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static mx::array gelu_approx(const mx::array& x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    auto x3 = x * x * x;
    auto inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return x * 0.5f * (1.0f + mx::tanh(inner));
}

// SiLU activation: x * sigmoid(x)
static mx::array silu_approx(const mx::array& x) {
    return x * mx::sigmoid(x);
}

// ReLU activation: max(0, x)
static mx::array relu_approx(const mx::array& x) {
    return mx::maximum(x, mx::array(0.0f));
}

// ============================================================================
// Utility Functions
// ============================================================================

mx::array compute_weight_norm(const mx::array& g, const mx::array& v) {
    auto v_norm = mx::sqrt(mx::sum(v * v, {1, 2}, true) + 1e-12f);
    return g * (v / v_norm);
}

mx::array transpose_conv_weight(const mx::array& w) {
    return mx::transpose(w, {0, 2, 1});
}

// ============================================================================
// Weight Storage
// ============================================================================

void CosyVoice3Weights::load(const std::string& model_path) {
    // Check if model_path exists
    std::ifstream path_check(model_path);
    struct stat path_stat;
    if (stat(model_path.c_str(), &path_stat) != 0) {
        throw std::runtime_error("CosyVoice3 model path does not exist: " + model_path);
    }

    // Try to load combined model.safetensors first (MLX format)
    std::string combined_path = model_path + "/model.safetensors";
    std::ifstream combined_file(combined_path);
    if (combined_file.good()) {
        load_combined(combined_path);
        return;
    }

    // Otherwise load from separate safetensors files (PyTorch format)
    std::string llm_path = model_path + "/llm/model.safetensors";
    std::ifstream llm_file(llm_path);
    if (llm_file.good()) {
        load_llm(llm_path);
    }

    std::string flow_path = model_path + "/flow.safetensors";
    std::ifstream flow_file(flow_path);
    if (flow_file.good()) {
        load_flow(flow_path);
    }

    std::string vocoder_path = model_path + "/vocoder.safetensors";
    std::ifstream vocoder_file(vocoder_path);
    if (vocoder_file.good()) {
        load_vocoder(vocoder_path);
    } else {
        // Also try hift.safetensors (CosyVoice3 naming)
        std::string hift_path = model_path + "/hift.safetensors";
        std::ifstream hift_file(hift_path);
        if (hift_file.good()) {
            load_vocoder(hift_path);
        }
    }

    // Validate that at least some weights were loaded
    if (total_weight_count() == 0) {
        throw std::runtime_error("No CosyVoice3 weights found in: " + model_path +
            ". Expected model.safetensors or flow.safetensors/vocoder.safetensors");
    }
}

void CosyVoice3Weights::load_combined(const std::string& path) {
    auto result = mx::load_safetensors(path);
    auto& weights = result.first;

    // Separate weights by prefix using iterator to avoid copy
    for (auto it = weights.begin(); it != weights.end(); ++it) {
        const std::string& key = it->first;
        mx::array value = it->second;  // copy the array

        if (key.rfind("flow.", 0) == 0) {
            // Strip "flow." prefix and store
            flow_weights_.emplace(key.substr(5), std::move(value));
        } else if (key.rfind("vocoder.", 0) == 0) {
            // Strip "vocoder." prefix and store
            vocoder_weights_.emplace(key.substr(8), std::move(value));
        } else if (key.rfind("llm.", 0) == 0) {
            // Strip "llm." prefix and store
            llm_weights_.emplace(key.substr(4), std::move(value));
        } else {
            // Store as-is in flow_weights for compatibility
            flow_weights_.emplace(key, std::move(value));
        }
    }
}

void CosyVoice3Weights::load_llm(const std::string& path) {
    auto result = mx::load_safetensors(path);
    llm_weights_ = std::move(result.first);
}

void CosyVoice3Weights::load_flow(const std::string& path) {
    auto result = mx::load_safetensors(path);
    flow_weights_ = std::move(result.first);
}

void CosyVoice3Weights::load_vocoder(const std::string& path) {
    auto result = mx::load_safetensors(path);
    vocoder_weights_ = std::move(result.first);
}

mx::array CosyVoice3Weights::get(const std::string& name) const {
    auto it = llm_weights_.find(name);
    if (it != llm_weights_.end()) return it->second;

    it = flow_weights_.find(name);
    if (it != flow_weights_.end()) return it->second;

    it = vocoder_weights_.find(name);
    if (it != vocoder_weights_.end()) return it->second;

    throw std::runtime_error("Weight not found: " + name);
}

bool CosyVoice3Weights::has(const std::string& name) const {
    return llm_weights_.count(name) > 0 ||
           flow_weights_.count(name) > 0 ||
           vocoder_weights_.count(name) > 0;
}

mx::array CosyVoice3Weights::get_weight_norm_conv(const std::string& prefix) const {
    std::string orig0_key = prefix + ".parametrizations.weight.original0";
    std::string orig1_key = prefix + ".parametrizations.weight.original1";

    if (!has(orig0_key) || !has(orig1_key)) {
        throw std::runtime_error("Weight norm params not found: " + prefix);
    }

    // Compute weight norm, then transpose to MLX format
    auto w = compute_weight_norm(get(orig0_key), get(orig1_key));
    return transpose_conv_weight(w);
}

mx::array CosyVoice3Weights::get_transposed_conv(const std::string& name) const {
    return transpose_conv_weight(get(name));
}

// ============================================================================
// Snake Activation
// ============================================================================

SnakeActivation::SnakeActivation(int channels)
    : alpha_(mx::ones({channels})), channels_(channels) {
}

mx::array SnakeActivation::forward(const mx::array& x) {
    // Input: [B, C, L] (PyTorch format)
    auto alpha = mx::expand_dims(mx::expand_dims(alpha_, 0), 2);
    auto sin_ax = mx::sin(alpha * x);
    return x + (1.0f / (alpha + 1e-9f)) * sin_ax * sin_ax;
}

mx::array SnakeActivation::forward_mlx(const mx::array& x) {
    // Input: [B, L, C] (MLX native format) - avoids transpose overhead
    auto alpha = mx::expand_dims(mx::expand_dims(alpha_, 0), 1);  // [1, 1, C]
    auto sin_ax = mx::sin(alpha * x);
    return x + (1.0f / (alpha + 1e-9f)) * sin_ax * sin_ax;
}

void SnakeActivation::load_weights(const mx::array& alpha) {
    alpha_ = alpha;
}

// ============================================================================
// VocoderResBlock
// ============================================================================

VocoderResBlock::VocoderResBlock(int channels, int kernel_size, const std::vector<int>& dilations)
    : channels_(channels), kernel_size_(kernel_size), dilations_(dilations) {
    for (size_t i = 0; i < dilations.size(); ++i) {
        activations1_.emplace_back(channels);
        activations2_.emplace_back(channels);
    }
}

mx::array VocoderResBlock::forward(const mx::array& x) {
    // x: [B, C, L] (PyTorch format)
    // Convert to MLX format at start, stay in MLX format internally
    mx::array out = mx::transpose(x, {0, 2, 1});  // [B, L, C]

    for (size_t i = 0; i < dilations_.size(); ++i) {
        // First activation in MLX format
        mx::array h = activations1_[i].forward_mlx(out);

        // Apply dilated conv (already in MLX format)
        if (i < conv1_weights_.size()) {
            int dilation = dilations_[i];
            int padding = (kernel_size_ * dilation - dilation) / 2;
            h = mx::conv1d(h, conv1_weights_[i], /*stride=*/1, padding, dilation);
            if (i < conv1_biases_.size()) {
                h = h + conv1_biases_[i];
            }
        }

        // Second activation in MLX format
        h = activations2_[i].forward_mlx(h);

        // Second conv
        if (i < conv2_weights_.size()) {
            int padding2 = (kernel_size_ - 1) / 2;
            h = mx::conv1d(h, conv2_weights_[i], /*stride=*/1, padding2, /*dilation=*/1);
            if (i < conv2_biases_.size()) {
                h = h + conv2_biases_[i];
            }
        }

        // Residual connection (both in MLX format)
        out = out + h;
    }

    // Convert back to PyTorch format at end
    return mx::transpose(out, {0, 2, 1});  // [B, C, L]
}

void VocoderResBlock::load_weights(
    const std::vector<mx::array>& conv1_weights,
    const std::vector<mx::array>& conv1_biases,
    const std::vector<mx::array>& conv2_weights,
    const std::vector<mx::array>& conv2_biases,
    const std::vector<mx::array>& alpha1,
    const std::vector<mx::array>& alpha2
) {
    conv1_weights_ = conv1_weights;
    conv1_biases_ = conv1_biases;
    conv2_weights_ = conv2_weights;
    conv2_biases_ = conv2_biases;

    for (size_t i = 0; i < alpha1.size() && i < activations1_.size(); ++i) {
        activations1_[i].load_weights(alpha1[i]);
    }
    for (size_t i = 0; i < alpha2.size() && i < activations2_.size(); ++i) {
        activations2_[i].load_weights(alpha2[i]);
    }
}

// ============================================================================
// F0 Predictor
// ============================================================================

F0Predictor::F0Predictor(const VocoderConfig& config)
    : config_(config),
      classifier_weight_(mx::zeros({1, config.f0_channels})),
      classifier_bias_(mx::zeros({1})) {
    // Initialize conv layers - progressively increase channels from mel_dim to f0_channels
    int in_ch = config.in_channels;  // 80
    int out_ch = config.f0_channels; // 512
    for (int i = 0; i < config.f0_num_convs; ++i) {
        int k = (i < static_cast<int>(config.f0_kernel_sizes.size())) ? config.f0_kernel_sizes[i] : 3;
        // MLX Conv1d weight shape: [out, kernel, in]
        conv_weights_.push_back(mx::zeros({out_ch, k, in_ch}));
        conv_biases_.push_back(mx::zeros({out_ch}));
        in_ch = out_ch;  // All subsequent layers use f0_channels
    }
}

mx::array F0Predictor::forward(const mx::array& mel) {
    // mel: [B, mel_dim, L]
    mx::array x = mel;

    // Apply conv layers with leaky ReLU
    for (size_t i = 0; i < conv_weights_.size(); ++i) {
        // Transpose: [B, C, L] -> [B, L, C]
        x = mx::transpose(x, {0, 2, 1});

        int ks = (i < config_.f0_kernel_sizes.size()) ? config_.f0_kernel_sizes[i] : 3;
        int padding = (ks - 1) / 2;

        x = mx::conv1d(x, conv_weights_[i], /*stride=*/1, padding);
        if (i < conv_biases_.size()) {
            x = x + conv_biases_[i];
        }

        // Transpose back: [B, L, C] -> [B, C, L]
        x = mx::transpose(x, {0, 2, 1});

        // Leaky ReLU
        x = mx::maximum(x, x * config_.lrelu_slope);
    }

    // Classifier: [B, C, L] -> [B, L, C] -> Linear -> [B, L, 1] -> [B, 1, L]
    x = mx::transpose(x, {0, 2, 1});  // [B, L, C]
    x = mx::matmul(x, mx::transpose(classifier_weight_)) + classifier_bias_;  // [B, L, 1]
    x = mx::transpose(x, {0, 2, 1});  // [B, 1, L]

    // F0 should be positive
    return mx::maximum(x, mx::array(0.0f));
}

void F0Predictor::load_weights(
    const std::vector<mx::array>& conv_weights,
    const std::vector<mx::array>& conv_biases,
    const mx::array& classifier_weight,
    const mx::array& classifier_bias
) {
    conv_weights_ = conv_weights;
    conv_biases_ = conv_biases;
    classifier_weight_ = classifier_weight;
    classifier_bias_ = classifier_bias;
}

// ============================================================================
// Source Module
// ============================================================================

SourceModule::SourceModule(const VocoderConfig& config)
    : config_(config),
      linear_weight_(mx::zeros({1, config.nb_harmonics + 1})),
      linear_bias_(mx::zeros({1})) {
}

mx::array SourceModule::forward(const mx::array& f0, int upsample_factor) {
    // f0: [B, 1, L]
    auto shape = f0.shape();
    int B = static_cast<int>(shape[0]);
    int L = static_cast<int>(shape[2]);
    int L_audio = L * upsample_factor;

    // Upsample F0 to audio rate (repeat each value)
    auto f0_up = mx::repeat(f0, upsample_factor, 2);  // [B, 1, L_audio]

    // Generate phase
    auto phase_inc = f0_up / static_cast<float>(config_.sample_rate);
    auto phase = mx::cumsum(phase_inc, 2) * (2.0f * static_cast<float>(M_PI));  // [B, 1, L_audio]

    // Generate harmonics (sin and cos for each harmonic)
    std::vector<mx::array> components;
    for (int h = 0; h < config_.nb_harmonics; ++h) {
        auto harmonic_phase = phase * static_cast<float>(h + 1);
        components.push_back(mx::sin(harmonic_phase));
        components.push_back(mx::cos(harmonic_phase));
    }

    // Add noise channels
    auto noise = mx::random::normal({B, 2, L_audio}) * config_.nsf_sigma;
    components.push_back(mx::slice(noise, {0, 0, 0}, {B, 1, L_audio}));
    components.push_back(mx::slice(noise, {0, 1, 0}, {B, 2, L_audio}));

    // Stack: [B, nb_harmonics*2 + 2, L_audio]
    return mx::concatenate(components, 1);
}

void SourceModule::load_weights(const mx::array& linear_weight, const mx::array& linear_bias) {
    linear_weight_ = linear_weight;
    linear_bias_ = linear_bias;
}

// ============================================================================
// CausalHiFT Generator
// ============================================================================

CausalHiFTGenerator::CausalHiFTGenerator(const VocoderConfig& config)
    : config_(config),
      f0_predictor_(config),
      source_module_(config),
      conv_pre_weight_(mx::zeros({config.base_channels, 5, config.in_channels})),
      conv_pre_bias_(mx::zeros({config.base_channels})),
      conv_post_weight_(mx::zeros({config.istft_n_fft + 2, 7, config.base_channels / 8})),
      conv_post_bias_(mx::zeros({config.istft_n_fft + 2})),
      streaming_state_(mx::zeros({1})) {

    // Initialize upsample layers (ConvTranspose1d)
    int in_ch = config.base_channels;
    for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
        int out_ch = in_ch / 2;
        int ks = config.upsample_kernel_sizes[i];
        // ConvTranspose1d weight: [out, kernel, in]
        up_weights_.push_back(mx::zeros({out_ch, ks, in_ch}));
        up_biases_.push_back(mx::zeros({out_ch}));

        // Initialize snake activations for upsampling
        up_activations_.emplace_back(in_ch);

        // ResBlocks for this stage
        for (size_t j = 0; j < config.resblock_kernel_sizes.size(); ++j) {
            resblocks_.emplace_back(
                out_ch,
                config.resblock_kernel_sizes[j],
                config.resblock_dilation_sizes[j]
            );
        }
        in_ch = out_ch;
    }

    // Initialize source downsampling layers
    // Source has nb_harmonics*2+2 channels = 18
    // Each downsample layer operates on the ORIGINAL source signal (not progressively)
    // The stride varies per stage to match the decoder length at that stage
    int source_input_ch = config.nb_harmonics * 2 + 2;  // Always 18
    for (size_t i = 0; i < config.source_down_channels.size(); ++i) {
        int out_ch = config.source_down_channels[i];
        int ks = config.source_down_kernels[i];
        // Conv1d weight: [out, kernel, in]
        // All layers take the original source (18 ch) as input
        source_down_weights_.push_back(mx::zeros({out_ch, ks, source_input_ch}));
        source_down_biases_.push_back(mx::zeros({out_ch}));

        source_resblocks_.emplace_back(
            out_ch,
            config.source_resblock_kernel_sizes[i],
            config.source_resblock_dilation_sizes[i]
        );
    }
}

mx::array CausalHiFTGenerator::forward(const mx::array& mel, const mx::array* f0_ptr) {
    // mel: [B, mel_dim, L]
    auto shape = mel.shape();
    int B = static_cast<int>(shape[0]);
    int L_mel = static_cast<int>(shape[2]);
    int total_up = config_.total_upsample_factor();

    // Predict F0 if not provided
    mx::array f0 = mx::zeros({1});
    if (f0_ptr) {
        f0 = *f0_ptr;
    } else {
        f0 = f0_predictor_.forward(mel);
    }

    // Generate source signal
    auto source = source_module_.forward(f0, total_up);

    // Main filter path: conv_pre
    // [B, mel_dim, L] -> [B, L, mel_dim] -> conv -> [B, L, base_ch]
    auto x = mx::transpose(mel, {0, 2, 1});
    x = mx::conv1d(x, conv_pre_weight_, 1, 2);  // kernel=5, padding=2
    x = x + conv_pre_bias_;
    x = mx::transpose(x, {0, 2, 1});  // [B, base_ch, L]

    // Upsampling with ResBlocks and source injection
    int resblock_idx = 0;
    int cumulative_up = 1;

    for (size_t i = 0; i < config_.upsample_rates.size(); ++i) {
        // Leaky ReLU
        x = leaky_relu(x, config_.lrelu_slope);

        int rate = config_.upsample_rates[i];
        int ks = config_.upsample_kernel_sizes[i];
        int padding = (ks - rate) / 2;

        // ConvTranspose1d: [B, C, L] -> [B, L, C] -> convT -> [B, L*rate, C/2]
        x = mx::transpose(x, {0, 2, 1});
        if (i < up_weights_.size()) {
            x = mx::conv_transpose1d(x, up_weights_[i], rate, padding);
            if (i < up_biases_.size()) {
                x = x + up_biases_[i];
            }
        }
        x = mx::transpose(x, {0, 2, 1});  // [B, C/2, L*rate]

        cumulative_up *= rate;

        // Downsample source and add
        if (i < source_down_weights_.size()) {
            int source_stride = total_up / cumulative_up;
            int source_ks = config_.source_down_kernels[i];
            int source_padding = (source_stride > 1) ? (source_ks - source_stride) / 2 : 0;

            auto source_ds = mx::transpose(source, {0, 2, 1});
            source_ds = mx::conv1d(source_ds, source_down_weights_[i], source_stride, source_padding);
            if (i < source_down_biases_.size()) {
                source_ds = source_ds + source_down_biases_[i];
            }
            source_ds = mx::transpose(source_ds, {0, 2, 1});

            // Apply source ResBlock
            if (i < source_resblocks_.size()) {
                source_ds = source_resblocks_[i].forward(source_ds);
            }

            // Match lengths and add
            int x_len = static_cast<int>(x.shape()[2]);
            int s_len = static_cast<int>(source_ds.shape()[2]);
            int sB = static_cast<int>(source_ds.shape()[0]);
            int sC = static_cast<int>(source_ds.shape()[1]);
            if (s_len < x_len) {
                source_ds = mx::pad(source_ds, {{0, 0}, {0, 0}, {0, x_len - s_len}});
            } else if (s_len > x_len) {
                source_ds = mx::slice(source_ds, {0, 0, 0}, {sB, sC, x_len});
            }

            x = x + source_ds;
        }

        // Apply ResBlocks (3 per upsample stage)
        int num_resblocks = static_cast<int>(config_.resblock_kernel_sizes.size());
        mx::array xs = mx::zeros_like(x);
        for (int j = 0; j < num_resblocks; ++j) {
            if (static_cast<size_t>(resblock_idx) < resblocks_.size()) {
                xs = xs + resblocks_[resblock_idx].forward(x);
                resblock_idx++;
            }
        }
        x = xs / static_cast<float>(num_resblocks);
    }

    // Post-conv
    x = leaky_relu(x, config_.lrelu_slope);
    x = mx::transpose(x, {0, 2, 1});
    x = mx::conv1d(x, conv_post_weight_, 1, 3);  // kernel=7, padding=3
    x = x + conv_post_bias_;
    x = mx::transpose(x, {0, 2, 1});  // [B, n_fft+2, L*total_up]

    // iSTFT synthesis
    auto audio = istft(x);

    // Clamp output
    return mx::clip(audio, mx::array(-config_.audio_limit), mx::array(config_.audio_limit));
}

mx::array CausalHiFTGenerator::forward_streaming(
    const mx::array& mel,
    bool /*is_first*/,
    bool /*is_last*/
) {
    return forward(mel, nullptr);
}

mx::array CausalHiFTGenerator::istft(const mx::array& x) {
    // x: [B, n_fft + 2, L] containing magnitude/phase
    // n_fft=16, hop_len=4 for CosyVoice3
    int n_fft = config_.istft_n_fft;
    int hop_len = config_.istft_hop_len;

    auto shape = x.shape();
    int B = static_cast<int>(shape[0]);
    int L = static_cast<int>(shape[2]);
    int L_out = L * hop_len;

    // Split into magnitude and phase
    int half = n_fft / 2 + 1;  // 9 for n_fft=16
    auto mag = mx::slice(x, {0, 0, 0}, {B, half, L});
    auto phase = mx::slice(x, {0, half, 0}, {B, 2 * half, L});

    // Convert to real/imag
    auto real_part = mag * mx::cos(phase);
    auto imag_part = mag * mx::sin(phase);

    // Create Hann window [n_fft] - static cache
    static mx::array window = []() {
        int n = 16;  // config_.istft_n_fft
        std::vector<float> window_data(n);
        for (int i = 0; i < n; ++i) {
            window_data[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / n));
        }
        return mx::array(window_data.data(), {n});
    }();

    // Build complex array by interleaving real and imag
    // Stack [real, imag] along last axis: [B, half, L] -> [B, half, L, 2]
    auto real_expanded = mx::expand_dims(real_part, -1);
    auto imag_expanded = mx::expand_dims(imag_part, -1);
    auto complex_interleaved = mx::concatenate({real_expanded, imag_expanded}, -1);

    // View as complex: [B, half, L, 2] float32 -> [B, half, L] complex64
    auto freq_domain = mx::view(complex_interleaved, mx::complex64);
    freq_domain = mx::squeeze(freq_domain, -1);  // [B, half, L]

    // Transpose for irfft: [B, L, half]
    freq_domain = mx::transpose(freq_domain, {0, 2, 1});

    // Apply irfft with output size n_fft on last axis: [B, L, half] -> [B, L, n_fft]
    auto frames = mx::fft::irfft(freq_domain, n_fft, -1);

    // Apply window to each frame: [B, L, n_fft] * [n_fft]
    frames = frames * window;

    // Vectorized overlap-add
    // With n_fft=16, hop_len=4, we have n_fft/hop_len=4 overlapping contributions
    int num_overlaps = n_fft / hop_len;  // 4

    // Reshape [B, L, n_fft] -> [B, L, num_overlaps, hop_len] -> shift and sum
    auto frames_reshape = mx::reshape(frames, {B, L, num_overlaps, hop_len});

    // Transpose to [B, num_overlaps, L, hop_len]
    frames_reshape = mx::transpose(frames_reshape, {0, 2, 1, 3});

    // Reshape to [B, num_overlaps, L * hop_len]
    frames_reshape = mx::reshape(frames_reshape, {B, num_overlaps, L * hop_len});

    // Sum shifted versions using stack and sum (more efficient than loop)
    std::vector<mx::array> shifted;
    shifted.reserve(num_overlaps);
    for (int k = 0; k < num_overlaps; ++k) {
        auto slice_k = mx::slice(frames_reshape, {0, k, 0}, {B, k + 1, L * hop_len});
        slice_k = mx::squeeze(slice_k, 1);  // [B, L * hop_len]

        // Pad: [0...k*hop_len] at front, [(num_overlaps-1-k)*hop_len...] at back
        int pad_front = k * hop_len;
        int pad_back = (num_overlaps - 1 - k) * hop_len;
        if (pad_front > 0 || pad_back > 0) {
            slice_k = mx::pad(slice_k, {{0, 0}, {pad_front, pad_back}});
        }
        shifted.push_back(slice_k);
    }

    // Stack and sum (one reduction op instead of sequential additions)
    auto stacked = mx::stack(shifted, 0);  // [num_overlaps, B, total_len]
    auto audio = mx::sum(stacked, 0);  // [B, total_len]

    // Trim to final length
    audio = mx::slice(audio, {0, 0}, {B, L_out});

    return audio;
}

mx::array CausalHiFTGenerator::snake(const mx::array& x, const mx::array& alpha) {
    auto a = mx::expand_dims(mx::expand_dims(alpha, 0), 2);
    auto sin_ax = mx::sin(a * x);
    return x + (1.0f / (a + 1e-9f)) * sin_ax * sin_ax;
}

mx::array CausalHiFTGenerator::leaky_relu(const mx::array& x, float slope) {
    return mx::maximum(x, x * slope);
}

void CausalHiFTGenerator::load_weights(const CosyVoice3Weights& weights) {
    // Check if we have MLX format (merged weight norm) or PyTorch format (separate g/v)
    bool use_mlx_format = weights.has("conv_pre.conv.weight");

    // Pre/post convolutions
    if (use_mlx_format) {
        // MLX format: weights already processed (weight norm merged, transposed)
        if (weights.has("conv_pre.conv.weight")) {
            conv_pre_weight_ = weights.get("conv_pre.conv.weight");
            conv_pre_bias_ = weights.get("conv_pre.conv.bias");
        }
        if (weights.has("conv_post.conv.weight")) {
            conv_post_weight_ = weights.get("conv_post.conv.weight");
            conv_post_bias_ = weights.get("conv_post.conv.bias");
        }
    } else {
        // PyTorch format: compute weight norm
        if (weights.has("conv_pre.parametrizations.weight.original0")) {
            conv_pre_weight_ = weights.get_weight_norm_conv("conv_pre");
            conv_pre_bias_ = weights.get("conv_pre.bias");
        }
        if (weights.has("conv_post.parametrizations.weight.original0")) {
            conv_post_weight_ = weights.get_weight_norm_conv("conv_post");
            conv_post_bias_ = weights.get("conv_post.bias");
        }
    }

    // F0 predictor
    if (use_mlx_format && weights.has("f0_predictor.condnet.0.conv.weight")) {
        std::vector<mx::array> conv_w, conv_b;
        for (int i = 0; i < 5; ++i) {
            std::string prefix = "f0_predictor.condnet." + std::to_string(i);
            if (weights.has(prefix + ".conv.weight")) {
                conv_w.push_back(weights.get(prefix + ".conv.weight"));
                conv_b.push_back(weights.get(prefix + ".conv.bias"));
            }
        }
        if (!conv_w.empty()) {
            f0_predictor_.load_weights(
                conv_w, conv_b,
                weights.get("f0_predictor.classifier.weight"),
                weights.get("f0_predictor.classifier.bias")
            );
        }
    } else if (weights.has("f0_predictor.condnet.0.parametrizations.weight.original0")) {
        std::vector<mx::array> conv_w, conv_b;
        for (int i = 0; i < 5; ++i) {
            std::string prefix = "f0_predictor.condnet." + std::to_string(i * 2);
            if (weights.has(prefix + ".parametrizations.weight.original0")) {
                conv_w.push_back(weights.get_weight_norm_conv(prefix));
                conv_b.push_back(weights.get(prefix + ".bias"));
            }
        }
        if (!conv_w.empty()) {
            f0_predictor_.load_weights(
                conv_w, conv_b,
                weights.get("f0_predictor.classifier.weight"),
                weights.get("f0_predictor.classifier.bias")
            );
        }
    }

    // Source module
    if (weights.has("m_source.l_linear.weight")) {
        source_module_.load_weights(
            weights.get("m_source.l_linear.weight"),
            weights.get("m_source.l_linear.bias")
        );
    }

    // Upsampling layers
    for (size_t i = 0; i < up_weights_.size(); ++i) {
        std::string prefix = "ups." + std::to_string(i);
        if (use_mlx_format && weights.has(prefix + ".conv.weight")) {
            // MLX format: already processed
            up_weights_[i] = weights.get(prefix + ".conv.weight");
            up_biases_[i] = weights.get(prefix + ".conv.bias");
        } else if (weights.has(prefix + ".parametrizations.weight.original0")) {
            // PyTorch ConvTranspose1d weight: [out, in, kernel]
            // MLX conv_transpose1d weight: [out, kernel, in]
            auto g = weights.get(prefix + ".parametrizations.weight.original0");
            auto v = weights.get(prefix + ".parametrizations.weight.original1");
            auto w = compute_weight_norm(g, v);
            up_weights_[i] = mx::transpose(w, {0, 2, 1});  // [out, in, kernel] -> [out, kernel, in]
            up_biases_[i] = weights.get(prefix + ".bias");
        }
    }

    // Upsampling activations (snake alpha)
    for (size_t i = 0; i < up_activations_.size(); ++i) {
        std::string alpha_key = "ups_activations." + std::to_string(i) + ".alpha";
        if (weights.has(alpha_key)) {
            up_activations_[i].load_weights(weights.get(alpha_key));
        }
    }

    // Residual blocks - extract weights from CosyVoice3Weights
    for (size_t i = 0; i < resblocks_.size(); ++i) {
        std::string prefix = "resblocks." + std::to_string(i);

        // Check MLX format first
        bool has_mlx_resblock = weights.has(prefix + ".convs1.0.conv.weight");
        bool has_pt_resblock = weights.has(prefix + ".convs1.0.parametrizations.weight.original0");

        if (!has_mlx_resblock && !has_pt_resblock) {
            continue;
        }

        std::vector<mx::array> c1_w, c1_b, c2_w, c2_b, a1, a2;
        // 3 conv layers per resblock (dilations: 1, 3, 5)
        for (int j = 0; j < 3; ++j) {
            std::string c1_prefix = prefix + ".convs1." + std::to_string(j);
            std::string c2_prefix = prefix + ".convs2." + std::to_string(j);

            if (has_mlx_resblock) {
                // MLX format: already processed
                c1_w.push_back(weights.get(c1_prefix + ".conv.weight"));
                c1_b.push_back(weights.get(c1_prefix + ".conv.bias"));
                c2_w.push_back(weights.get(c2_prefix + ".conv.weight"));
                c2_b.push_back(weights.get(c2_prefix + ".conv.bias"));
            } else {
                // PyTorch format: compute weight norm
                c1_w.push_back(weights.get_weight_norm_conv(c1_prefix));
                c1_b.push_back(weights.get(c1_prefix + ".bias"));
                c2_w.push_back(weights.get_weight_norm_conv(c2_prefix));
                c2_b.push_back(weights.get(c2_prefix + ".bias"));
            }
            a1.push_back(weights.get(prefix + ".activations1." + std::to_string(j) + ".alpha"));
            a2.push_back(weights.get(prefix + ".activations2." + std::to_string(j) + ".alpha"));
        }
        resblocks_[i].load_weights(c1_w, c1_b, c2_w, c2_b, a1, a2);
    }

    // Source downsampling - check MLX format
    for (size_t i = 0; i < source_down_weights_.size(); ++i) {
        std::string prefix = "source_downs." + std::to_string(i);
        if (use_mlx_format && weights.has(prefix + ".conv.weight")) {
            source_down_weights_[i] = weights.get(prefix + ".conv.weight");
            source_down_biases_[i] = weights.get(prefix + ".conv.bias");
        }
    }

    // Source resblocks
    for (size_t i = 0; i < source_resblocks_.size(); ++i) {
        std::string prefix = "source_resblocks." + std::to_string(i);
        bool has_mlx_resblock = weights.has(prefix + ".convs1.0.conv.weight");

        if (!has_mlx_resblock) continue;

        std::vector<mx::array> c1_w, c1_b, c2_w, c2_b, a1, a2;
        for (int j = 0; j < 3; ++j) {
            std::string c1_prefix = prefix + ".convs1." + std::to_string(j);
            std::string c2_prefix = prefix + ".convs2." + std::to_string(j);

            if (weights.has(c1_prefix + ".conv.weight")) {
                c1_w.push_back(weights.get(c1_prefix + ".conv.weight"));
                c1_b.push_back(weights.get(c1_prefix + ".conv.bias"));
                c2_w.push_back(weights.get(c2_prefix + ".conv.weight"));
                c2_b.push_back(weights.get(c2_prefix + ".conv.bias"));
                a1.push_back(weights.get(prefix + ".activations1." + std::to_string(j) + ".alpha"));
                a2.push_back(weights.get(prefix + ".activations2." + std::to_string(j) + ".alpha"));
            }
        }
        if (!c1_w.empty()) {
            source_resblocks_[i].load_weights(c1_w, c1_b, c2_w, c2_b, a1, a2);
        }
    }
}

void CausalHiFTGenerator::compile() {
    compiled_ = true;
}

// ============================================================================
// DiT Attention
// ============================================================================

DiTAttention::DiTAttention(const DiTConfig& config)
    : config_(config),
      q_weight_(mx::zeros({config.heads * config.dim_head, config.dim})),
      q_bias_(mx::zeros({config.heads * config.dim_head})),
      k_weight_(mx::zeros({config.heads * config.dim_head, config.dim})),
      k_bias_(mx::zeros({config.heads * config.dim_head})),
      v_weight_(mx::zeros({config.heads * config.dim_head, config.dim})),
      v_bias_(mx::zeros({config.heads * config.dim_head})),
      qkv_weight_(mx::zeros({3 * config.heads * config.dim_head, config.dim})),
      qkv_bias_(mx::zeros({3 * config.heads * config.dim_head})),
      out_weight_(mx::zeros({config.dim, config.heads * config.dim_head})),
      out_bias_(mx::zeros({config.dim})) {
}

mx::array DiTAttention::forward(
    const mx::array& x,
    const mx::array& inv_freq,
    int offset,
    const mx::array* mask,
    bool /*streaming*/
) {
    auto shape = x.shape();
    int B = static_cast<int>(shape[0]);
    int L = static_cast<int>(shape[1]);

    mx::array q = mx::zeros({1}), k = mx::zeros({1}), v = mx::zeros({1});

    if (qkv_fused_) {
        auto qkv = mx::matmul(x, mx::transpose(qkv_weight_)) + qkv_bias_;
        int inner_dim = config_.heads * config_.dim_head;
        q = mx::slice(qkv, {0, 0, 0}, {B, L, inner_dim});
        k = mx::slice(qkv, {0, 0, inner_dim}, {B, L, 2 * inner_dim});
        v = mx::slice(qkv, {0, 0, 2 * inner_dim}, {B, L, 3 * inner_dim});
    } else {
        q = mx::matmul(x, mx::transpose(q_weight_)) + q_bias_;
        k = mx::matmul(x, mx::transpose(k_weight_)) + k_bias_;
        v = mx::matmul(x, mx::transpose(v_weight_)) + v_bias_;
    }

    q = mx::reshape(q, {B, L, config_.heads, config_.dim_head});
    k = mx::reshape(k, {B, L, config_.heads, config_.dim_head});
    v = mx::reshape(v, {B, L, config_.heads, config_.dim_head});

    q = mx::transpose(q, {0, 2, 1, 3});
    k = mx::transpose(k, {0, 2, 1, 3});
    v = mx::transpose(v, {0, 2, 1, 3});

    // Use base (rope_theta) for RoPE - don't pass precomputed freqs
    q = mx::fast::rope(q, config_.dim_head, false, config_.rope_theta, 1.0f, offset);
    k = mx::fast::rope(k, config_.dim_head, false, config_.rope_theta, 1.0f, offset);

    float scale = 1.0f / std::sqrt(static_cast<float>(config_.dim_head));

    // Use optimized attention kernel - mx::fast::scaled_dot_product_attention
    // This uses fused Metal kernels for much better performance
    // Signature: (queries, keys, values, scale, mask_mode, mask_arr)
    mx::array attn = mx::fast::scaled_dot_product_attention(
        q, k, v,
        scale,
        mask ? "" : "",   // mask_mode: empty for no preset mode
        mask ? std::optional<mx::array>(*mask) : std::nullopt  // custom mask
    );

    attn = mx::transpose(attn, {0, 2, 1, 3});
    attn = mx::reshape(attn, {B, L, config_.heads * config_.dim_head});

    return mx::matmul(attn, mx::transpose(out_weight_)) + out_bias_;
}

void DiTAttention::load_weights(
    const mx::array& q_weight, const mx::array& q_bias,
    const mx::array& k_weight, const mx::array& k_bias,
    const mx::array& v_weight, const mx::array& v_bias,
    const mx::array& out_weight, const mx::array& out_bias
) {
    q_weight_ = q_weight;
    q_bias_ = q_bias;
    k_weight_ = k_weight;
    k_bias_ = k_bias;
    v_weight_ = v_weight;
    v_bias_ = v_bias;
    out_weight_ = out_weight;
    out_bias_ = out_bias;
    qkv_fused_ = false;
}

void DiTAttention::fuse_qkv_weights() {
    if (qkv_fused_) return;
    qkv_weight_ = mx::concatenate({q_weight_, k_weight_, v_weight_}, 0);
    qkv_bias_ = mx::concatenate({q_bias_, k_bias_, v_bias_}, 0);
    qkv_fused_ = true;
}

// ============================================================================
// DiT Feed-Forward
// ============================================================================

DiTFeedForward::DiTFeedForward(const DiTConfig& config)
    : config_(config),
      w1_(mx::zeros({config.dim * config.ff_mult, config.dim})),
      b1_(mx::zeros({config.dim * config.ff_mult})),
      w2_(mx::zeros({config.dim, config.dim * config.ff_mult})),
      b2_(mx::zeros({config.dim})) {
}

mx::array DiTFeedForward::forward(const mx::array& x) {
    auto h = mx::matmul(x, mx::transpose(w1_)) + b1_;
    h = gelu_approx(h);
    return mx::matmul(h, mx::transpose(w2_)) + b2_;
}

void DiTFeedForward::load_weights(
    const mx::array& w1, const mx::array& b1,
    const mx::array& w2, const mx::array& b2
) {
    w1_ = w1;
    b1_ = b1;
    w2_ = w2;
    b2_ = b2;
}

// ============================================================================
// Adaptive Layer Norm
// ============================================================================

AdaptiveLayerNorm::AdaptiveLayerNorm(int dim)
    : dim_(dim),
      weight_(mx::zeros({dim * 6, dim})),
      bias_(mx::zeros({dim * 6})) {
}

std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array>
AdaptiveLayerNorm::forward(const mx::array& /*x*/, const mx::array& cond) {
    auto params = mx::matmul(cond, mx::transpose(weight_)) + bias_;

    int d = dim_;
    int B = static_cast<int>(params.shape()[0]);
    auto scale1 = mx::slice(params, {0, 0}, {B, d});
    auto shift1 = mx::slice(params, {0, d}, {B, 2 * d});
    auto gate1 = mx::slice(params, {0, 2 * d}, {B, 3 * d});
    auto scale2 = mx::slice(params, {0, 3 * d}, {B, 4 * d});
    auto shift2 = mx::slice(params, {0, 4 * d}, {B, 5 * d});
    auto gate2 = mx::slice(params, {0, 5 * d}, {B, 6 * d});

    return {scale1, shift1, gate1, scale2, shift2, gate2};
}

void AdaptiveLayerNorm::load_weights(const mx::array& weight, const mx::array& bias) {
    weight_ = weight;
    bias_ = bias;
}

// ============================================================================
// DiT Block
// ============================================================================

DiTBlock::DiTBlock(const DiTConfig& config)
    : config_(config), norm_(config.dim), attn_(config), ff_(config) {
}

mx::array DiTBlock::forward(
    const mx::array& x,
    const mx::array& cond,
    const mx::array& inv_freq,
    int offset,
    const mx::array* mask,
    bool streaming
) {
    auto [scale1, shift1, gate1, scale2, shift2, gate2] = norm_.forward(x, cond);

    scale1 = mx::expand_dims(scale1, 1);
    shift1 = mx::expand_dims(shift1, 1);
    gate1 = mx::expand_dims(gate1, 1);
    scale2 = mx::expand_dims(scale2, 1);
    shift2 = mx::expand_dims(shift2, 1);
    gate2 = mx::expand_dims(gate2, 1);

    auto h = mx::fast::layer_norm(x, std::nullopt, std::nullopt, 1e-5f);
    h = h * (1.0f + scale1) + shift1;
    h = attn_.forward(h, inv_freq, offset, mask, streaming);
    auto out = x + gate1 * h;

    h = mx::fast::layer_norm(out, std::nullopt, std::nullopt, 1e-5f);
    h = h * (1.0f + scale2) + shift2;
    h = ff_.forward(h);
    out = out + gate2 * h;

    return out;
}

void DiTBlock::load_weights(
    const mx::array& q_w, const mx::array& q_b,
    const mx::array& k_w, const mx::array& k_b,
    const mx::array& v_w, const mx::array& v_b,
    const mx::array& out_w, const mx::array& out_b,
    const mx::array& norm_w, const mx::array& norm_b,
    const mx::array& ff1_w, const mx::array& ff1_b,
    const mx::array& ff2_w, const mx::array& ff2_b
) {
    attn_.load_weights(q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b);
    norm_.load_weights(norm_w, norm_b);
    ff_.load_weights(ff1_w, ff1_b, ff2_w, ff2_b);
}

// ============================================================================
// Time Embedding
// ============================================================================

TimeEmbedding::TimeEmbedding(int dim, int sinusoidal_dim)
    : dim_(dim), sinusoidal_dim_(sinusoidal_dim),
      w1_(mx::zeros({dim, sinusoidal_dim})),
      b1_(mx::zeros({dim})),
      w2_(mx::zeros({dim, dim})),
      b2_(mx::zeros({dim})) {
}

mx::array TimeEmbedding::forward(const mx::array& t) {
    int half_dim = sinusoidal_dim_ / 2;
    float log_10000 = std::log(10000.0f);

    std::vector<float> freq_data(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        freq_data[i] = std::exp(-log_10000 * i / (half_dim - 1));
    }
    auto freqs = mx::array(freq_data.data(), {half_dim});

    auto t_exp = mx::expand_dims(t, 1);
    auto angles = t_exp * freqs;
    auto emb = mx::concatenate({mx::sin(angles), mx::cos(angles)}, 1);

    auto h = mx::matmul(emb, mx::transpose(w1_)) + b1_;
    h = silu_approx(h);
    return mx::matmul(h, mx::transpose(w2_)) + b2_;
}

void TimeEmbedding::load_weights(
    const mx::array& w1, const mx::array& b1,
    const mx::array& w2, const mx::array& b2
) {
    w1_ = w1;
    b1_ = b1;
    w2_ = w2;
    b2_ = b2;
}

// ============================================================================
// Pre-Lookahead Layer
// ============================================================================

PreLookaheadLayer::PreLookaheadLayer(int in_channels, int hidden_channels)
    : in_channels_(in_channels), hidden_channels_(hidden_channels),
      conv1_w_(mx::zeros({hidden_channels, 4, in_channels})),
      conv1_b_(mx::zeros({hidden_channels})),
      conv2_w_(mx::zeros({in_channels, 3, hidden_channels})),
      conv2_b_(mx::zeros({in_channels})) {
}

mx::array PreLookaheadLayer::forward(const mx::array& x) {
    auto h = mx::pad(x, {{0, 0}, {3, 0}, {0, 0}});
    h = mx::conv1d(h, conv1_w_);
    h = h + conv1_b_;
    h = relu_approx(h);

    h = mx::pad(h, {{0, 0}, {2, 0}, {0, 0}});
    h = mx::conv1d(h, conv2_w_);
    h = h + conv2_b_;
    h = relu_approx(h);

    return h;
}

void PreLookaheadLayer::load_weights(
    const mx::array& conv1_w, const mx::array& conv1_b,
    const mx::array& conv2_w, const mx::array& conv2_b
) {
    conv1_w_ = conv1_w;
    conv1_b_ = conv1_b;
    conv2_w_ = conv2_w;
    conv2_b_ = conv2_b;
}

// ============================================================================
// DiT Flow Model
// ============================================================================

DiTFlowModel::DiTFlowModel(const DiTConfig& config)
    : config_(config),
      input_embedding_(mx::zeros({config.vocab_size, config.mel_dim})),
      pre_lookahead_(config.mel_dim, config.pre_lookahead_channels),
      spk_proj_weight_(mx::zeros({config.spk_dim, 192})),
      spk_proj_bias_(mx::zeros({config.spk_dim})),
      // Input embedding projection: combined_dim -> transformer_dim
      input_embed_proj_weight_(mx::zeros({config.dim, 320})),  // Projects padded combined input
      input_embed_proj_bias_(mx::zeros({config.dim})),
      time_embed_(config.dim, config.sinusoidal_dim),
      rope_inv_freq_(mx::zeros({config.dim_head / 2})),
      skip_proj_weight_(mx::zeros({config.dim, config.dim * 2})),
      skip_proj_bias_(mx::zeros({config.dim})),
      norm_out_weight_(mx::zeros({config.dim * 2, config.dim})),
      norm_out_bias_(mx::zeros({config.dim * 2})),
      proj_out_weight_(mx::zeros({config.out_channels, config.dim})),
      proj_out_bias_(mx::zeros({config.out_channels})) {

    for (int i = 0; i < config.depth; ++i) {
        blocks_.emplace_back(config);
    }

    int half_dim = config.dim_head / 2;
    std::vector<float> inv_freq_data(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        inv_freq_data[i] = 1.0f / std::pow(config.rope_theta, 2.0f * i / config.dim_head);
    }
    rope_inv_freq_ = mx::array(inv_freq_data.data(), {half_dim});
}

mx::array DiTFlowModel::forward(
    const mx::array& x,
    const mx::array& tokens,
    const mx::array& t,
    const mx::array& spk_emb,
    const mx::array* mask,
    bool streaming
) {
    auto shape = x.shape();
    int B = static_cast<int>(shape[0]);
    int L = static_cast<int>(shape[1]);

    auto token_emb = mx::take(input_embedding_, tokens, 0);
    token_emb = mx::repeat(token_emb, config_.token_mel_ratio, 1);

    auto mu = pre_lookahead_.forward(token_emb);

    mx::array spks = mx::zeros({1});
    if (speaker_cache_enabled_ && speaker_cache_.count("default") > 0) {
        spks = speaker_cache_.at("default");
    } else {
        spks = mx::matmul(spk_emb, mx::transpose(spk_proj_weight_)) + spk_proj_bias_;
    }

    auto t_emb = time_embed_.forward(t);

    auto spks_exp = mx::broadcast_to(
        mx::expand_dims(spks, 1),
        {B, L, static_cast<int>(spks.shape()[1])}
    );
    auto combined = mx::concatenate({x, mu, spks_exp}, 2);

    int combined_dim = static_cast<int>(combined.shape()[2]);
    if (combined_dim < 320) {
        combined = mx::pad(combined, {{0, 0}, {0, 0}, {0, 320 - combined_dim}});
    }

    // Project combined input to transformer dimension
    auto h = mx::matmul(combined, mx::transpose(input_embed_proj_weight_)) + input_embed_proj_bias_;
    auto h_init = h;

    int mid_block = config_.depth / 2;
    for (int i = 0; i < config_.depth; ++i) {
        h = blocks_[i].forward(h, t_emb, rope_inv_freq_, 0, mask, streaming);

        if (i == mid_block && config_.use_long_skip) {
            auto h_cat = mx::concatenate({h, h_init}, 2);
            h = mx::matmul(h_cat, mx::transpose(skip_proj_weight_)) + skip_proj_bias_;
        }
    }

    auto norm_params = mx::matmul(t_emb, mx::transpose(norm_out_weight_)) + norm_out_bias_;
    int dim = config_.dim;
    int nB = static_cast<int>(norm_params.shape()[0]);
    auto scale = mx::slice(norm_params, {0, 0}, {nB, dim});
    auto shift = mx::slice(norm_params, {0, dim}, {nB, 2 * dim});
    scale = mx::expand_dims(scale, 1);
    shift = mx::expand_dims(shift, 1);

    h = mx::fast::layer_norm(h, std::nullopt, std::nullopt, 1e-5f);
    h = h * (1.0f + scale) + shift;

    return mx::matmul(h, mx::transpose(proj_out_weight_)) + proj_out_bias_;
}

mx::array DiTFlowModel::inference(
    const mx::array& tokens,
    const mx::array& spk_emb,
    int num_steps,
    float /*cfg_strength*/,
    bool streaming
) {
    auto shape = tokens.shape();
    int B = static_cast<int>(shape[0]);
    int L_tokens = static_cast<int>(shape[1]);
    int L = L_tokens * config_.token_mel_ratio;

    auto x = mx::random::normal({B, L, config_.mel_dim});

    float dt = 1.0f / num_steps;

    for (int i = 0; i < num_steps; ++i) {
        float t_val = 1.0f - i * dt;
        auto t = mx::full({B}, t_val);

        auto v = forward(x, tokens, t, spk_emb, nullptr, streaming);
        x = x - v * dt;
    }

    return x;
}

void DiTFlowModel::load_weights(const CosyVoice3Weights& weights) {
    // MLX key format uses: dit.*, while PyTorch uses: decoder.estimator.*
    // Try MLX format first, then fall back to PyTorch format

    // Input embedding projection
    if (weights.has("dit.input_embed.proj.weight")) {
        input_embed_proj_weight_ = weights.get("dit.input_embed.proj.weight");
        input_embed_proj_bias_ = weights.get("dit.input_embed.proj.bias");
    } else if (weights.has("decoder.estimator.input_embed.proj.weight")) {
        input_embed_proj_weight_ = weights.get("decoder.estimator.input_embed.proj.weight");
        input_embed_proj_bias_ = weights.get("decoder.estimator.input_embed.proj.bias");
    }

    // Time embedding MLP
    if (weights.has("dit.time_embed.mlp.layers.0.weight")) {
        // MLX format: dit.time_embed.mlp.layers.{0,2}
        time_embed_.load_weights(
            weights.get("dit.time_embed.mlp.layers.0.weight"),
            weights.get("dit.time_embed.mlp.layers.0.bias"),
            weights.get("dit.time_embed.mlp.layers.2.weight"),
            weights.get("dit.time_embed.mlp.layers.2.bias")
        );
    } else if (weights.has("decoder.estimator.time_embed.time_mlp.0.weight")) {
        // PyTorch format
        time_embed_.load_weights(
            weights.get("decoder.estimator.time_embed.time_mlp.0.weight"),
            weights.get("decoder.estimator.time_embed.time_mlp.0.bias"),
            weights.get("decoder.estimator.time_embed.time_mlp.2.weight"),
            weights.get("decoder.estimator.time_embed.time_mlp.2.bias")
        );
    }

    // RoPE inv_freq
    if (weights.has("dit.rotary_embed.inv_freq")) {
        rope_inv_freq_ = weights.get("dit.rotary_embed.inv_freq");
    } else if (weights.has("decoder.estimator.rotary_embed.inv_freq")) {
        rope_inv_freq_ = weights.get("decoder.estimator.rotary_embed.inv_freq");
    }

    // Transformer blocks - try MLX format (dit.blocks) first
    bool use_mlx_format = weights.has("dit.blocks.0.attn.to_q.weight");

    for (size_t i = 0; i < blocks_.size(); ++i) {
        std::string prefix = use_mlx_format
            ? "dit.blocks." + std::to_string(i) + "."
            : "decoder.estimator.transformer_blocks." + std::to_string(i) + ".";

        if (!weights.has(prefix + "attn.to_q.weight")) {
            continue;
        }

        // MLX format uses attn.to_out.weight, PyTorch uses attn.to_out.0.weight
        std::string out_w_key = use_mlx_format ? prefix + "attn.to_out.weight" : prefix + "attn.to_out.0.weight";
        std::string out_b_key = use_mlx_format ? prefix + "attn.to_out.bias" : prefix + "attn.to_out.0.bias";

        // MLX format uses ff.layers.{0,1}, PyTorch uses ff.ff.0.0 and ff.ff.2
        std::string ff1_w_key = use_mlx_format ? prefix + "ff.layers.0.weight" : prefix + "ff.ff.0.0.weight";
        std::string ff1_b_key = use_mlx_format ? prefix + "ff.layers.0.bias" : prefix + "ff.ff.0.0.bias";
        std::string ff2_w_key = use_mlx_format ? prefix + "ff.layers.1.weight" : prefix + "ff.ff.2.weight";
        std::string ff2_b_key = use_mlx_format ? prefix + "ff.layers.1.bias" : prefix + "ff.ff.2.bias";

        blocks_[i].load_weights(
            // Attention Q/K/V/Out
            weights.get(prefix + "attn.to_q.weight"),
            weights.get(prefix + "attn.to_q.bias"),
            weights.get(prefix + "attn.to_k.weight"),
            weights.get(prefix + "attn.to_k.bias"),
            weights.get(prefix + "attn.to_v.weight"),
            weights.get(prefix + "attn.to_v.bias"),
            weights.get(out_w_key),
            weights.get(out_b_key),
            // Adaptive norm
            weights.get(prefix + "attn_norm.linear.weight"),
            weights.get(prefix + "attn_norm.linear.bias"),
            // FFN
            weights.get(ff1_w_key),
            weights.get(ff1_b_key),
            weights.get(ff2_w_key),
            weights.get(ff2_b_key)
        );
    }

    // Output layer norm (norm_out.linear)
    if (weights.has("dit.norm_out.linear.weight")) {
        norm_out_weight_ = weights.get("dit.norm_out.linear.weight");
        norm_out_bias_ = weights.get("dit.norm_out.linear.bias");
    } else if (weights.has("decoder.estimator.norm_out.linear.weight")) {
        norm_out_weight_ = weights.get("decoder.estimator.norm_out.linear.weight");
        norm_out_bias_ = weights.get("decoder.estimator.norm_out.linear.bias");
    }

    // Output projection (proj_out)
    if (weights.has("dit.proj_out.weight")) {
        proj_out_weight_ = weights.get("dit.proj_out.weight");
        proj_out_bias_ = weights.get("dit.proj_out.bias");
    } else if (weights.has("decoder.estimator.proj_out.weight")) {
        proj_out_weight_ = weights.get("decoder.estimator.proj_out.weight");
        proj_out_bias_ = weights.get("decoder.estimator.proj_out.bias");
    }

    // Speaker embedding projection
    if (weights.has("spk_embed_affine_layer.weight")) {
        spk_proj_weight_ = weights.get("spk_embed_affine_layer.weight");
        spk_proj_bias_ = weights.get("spk_embed_affine_layer.bias");
    }

    // Input embedding for speech tokens
    if (weights.has("input_embedding.weight")) {
        input_embedding_ = weights.get("input_embedding.weight");
    }

    // Pre-lookahead layer
    // MLX format weights are already in [out, kernel, in] format
    // PyTorch format needs transposition: [out, in, kernel] -> [out, kernel, in]
    if (weights.has("pre_lookahead_layer.conv1.weight")) {
        auto w1 = weights.get("pre_lookahead_layer.conv1.weight");
        auto w2 = weights.get("pre_lookahead_layer.conv2.weight");

        // Check if weights need transposition by looking at shape
        // MLX format: [out, kernel, in] where kernel=4 for conv1, kernel=3 for conv2
        // PyTorch format: [out, in, kernel]
        auto s1 = w1.shape();
        bool is_mlx_format = (s1.size() == 3 && s1[1] == 4);  // kernel_size=4 is in dim 1

        if (!is_mlx_format) {
            // PyTorch format - transpose
            w1 = transpose_conv_weight(w1);
            w2 = transpose_conv_weight(w2);
        }

        pre_lookahead_.load_weights(
            w1,
            weights.get("pre_lookahead_layer.conv1.bias"),
            w2,
            weights.get("pre_lookahead_layer.conv2.bias")
        );
    }
}

void DiTFlowModel::compile() {
    compiled_ = true;
}

void DiTFlowModel::fuse_qkv_weights() {
    for (auto& block : blocks_) {
        block.fuse_qkv_weights();
    }
}

void DiTFlowModel::cache_speaker_projection(const mx::array& spk_emb, const std::string& cache_id) {
    auto spks = mx::matmul(spk_emb, mx::transpose(spk_proj_weight_)) + spk_proj_bias_;
    mx::eval(spks);
    speaker_cache_.insert_or_assign(cache_id, spks);
}

void DiTFlowModel::enable_speaker_cache(bool enable) {
    speaker_cache_enabled_ = enable;
    if (!enable) {
        speaker_cache_.clear();
    }
}

void DiTFlowModel::clear_speaker_cache() {
    speaker_cache_.clear();
}

// ============================================================================
// CosyVoice3 Model
// ============================================================================

CosyVoice3Model::CosyVoice3Model(const CosyVoice3Config& config)
    : config_(config),
      // Initialize mx::array members with placeholder zeros (will be loaded later)
      llm_embedding_(mx::zeros({0})),
      speech_embedding_(mx::zeros({0})),
      embed_tokens_(mx::zeros({0})),
      llm_decoder_weight_(mx::zeros({0})),
      llm_decoder_bias_(mx::zeros({0})),
      lm_head_(mx::zeros({0})),
      final_norm_w_(mx::zeros({0})),
      rope_freqs_(mx::zeros({0})) {
    flow_ = std::make_unique<DiTFlowModel>(config.flow_config);
    vocoder_ = std::make_unique<CausalHiFTGenerator>(config.vocoder_config);
}

CosyVoice3Model::~CosyVoice3Model() = default;

CosyVoice3Model::CosyVoice3Model(CosyVoice3Model&&) noexcept = default;
CosyVoice3Model& CosyVoice3Model::operator=(CosyVoice3Model&&) noexcept = default;

CosyVoice3Model CosyVoice3Model::load(const std::string& model_path) {
    CosyVoice3Config config = CosyVoice3Config::create_default();

    CosyVoice3Model model(config);

    CosyVoice3Weights weights;
    weights.load(model_path);

    // Load all components: flow, vocoder, and LLM
    model.load_weights(weights);

    return model;
}

mx::array CosyVoice3Model::generate_speech_tokens(
    const mx::array& text_ids,
    int max_length,
    float temperature,
    int top_k,
    float top_p
) {
    // Check if LLM weights are loaded
    if (!llm_loaded_) {
        // Fall back to dummy tokens if LLM not loaded
        auto shape = text_ids.shape();
        int B = static_cast<int>(shape[0]);
        int L = std::min(max_length, 100);
        return mx::zeros({B, L}, mx::int32);
    }

    auto shape = text_ids.shape();
    int batch = static_cast<int>(shape[0]);
    int text_len = static_cast<int>(shape[1]);

    // Speech token vocabulary parameters (from CosyVoice2)
    const int speech_token_size = 6561;  // Speech tokens are 0-6560
    // Stop tokens: 6561, 6562, 6563 (any token >= speech_token_size is EOS)

    // Min/max length based on text length
    int min_len = static_cast<int>(text_len * 2.0f);  // min_token_text_ratio = 2.0
    int max_len = std::min(max_length, static_cast<int>(text_len * 20.0f));  // max_token_text_ratio = 20.0

    // Reset KV cache
    kv_cache_.clear();

    // Phase 1: Process text tokens to build KV cache
    // Get text embeddings
    auto text_embeddings = mx::take(embed_tokens_, text_ids, 0);  // [batch, text_len, hidden_size]

    // Run through transformer (builds KV cache)
    auto hidden_states = llm_forward(text_embeddings, &kv_cache_);
    kv_cache_.update_offset(text_len);
    mx::eval(hidden_states);

    // Phase 2: Autoregressive speech token generation
    std::vector<std::vector<int>> decoded_tokens(batch);

    // Start with SOS token (index 0 in llm_embedding)
    auto sos_embedding = mx::slice(llm_embedding_, {0, 0}, {1, static_cast<int>(llm_embedding_.shape()[1])});
    sos_embedding = mx::broadcast_to(sos_embedding, {batch, 1, static_cast<int>(llm_embedding_.shape()[1])});

    // Current input is SOS embedding
    auto current_input = sos_embedding;

    for (int step = 0; step < max_len; ++step) {
        // Run through transformer
        auto hidden = llm_forward(current_input, &kv_cache_);
        kv_cache_.update_offset(1);

        // Get speech logits using llm_decoder
        // hidden: [batch, 1, hidden_size]
        // llm_decoder_weight_: [6564, hidden_size]
        auto speech_logits = mx::matmul(hidden, mx::transpose(llm_decoder_weight_));
        speech_logits = speech_logits + llm_decoder_bias_;  // [batch, 1, 6564]

        // Take last token's logits
        auto logits = mx::squeeze(speech_logits, 1);  // [batch, 6564]
        mx::eval(logits);

        // Sample next tokens
        std::vector<int> next_tokens(batch);
        bool all_stopped = true;

        for (int b = 0; b < batch; ++b) {
            // Check if this batch item already stopped
            if (!decoded_tokens[b].empty() && decoded_tokens[b].back() >= speech_token_size) {
                next_tokens[b] = speech_token_size;  // Keep at EOS
                continue;
            }

            // Get logits for this batch item
            auto batch_logits = mx::slice(logits, {b, 0}, {b + 1, static_cast<int>(logits.shape()[1])});

            // Sample with temperature, top_k, top_p
            auto sampled = sample_token(batch_logits, temperature, top_k, top_p);
            mx::eval(sampled);
            int token = static_cast<int>(sampled.item<int32_t>());

            // Before min_len, avoid EOS tokens
            if (step < min_len && token >= speech_token_size) {
                // Resample without EOS tokens (set EOS logits to -inf)
                auto filtered_logits = mx::slice(batch_logits, {0, 0}, {1, speech_token_size});
                auto padded_logits = mx::concatenate({
                    filtered_logits,
                    mx::full({1, 3}, -1e9f)  // 3 EOS tokens
                }, -1);
                sampled = sample_token(padded_logits, temperature, top_k, top_p);
                mx::eval(sampled);
                token = static_cast<int>(sampled.item<int32_t>());
            }

            next_tokens[b] = token;
            decoded_tokens[b].push_back(token);

            if (token < speech_token_size) {
                all_stopped = false;
            }
        }

        // Check if all sequences have stopped
        if (all_stopped) {
            // Remove EOS tokens from decoded
            for (int b = 0; b < batch; ++b) {
                while (!decoded_tokens[b].empty() && decoded_tokens[b].back() >= speech_token_size) {
                    decoded_tokens[b].pop_back();
                }
            }
            break;
        }

        // Prepare next input embeddings
        // Use speech_embedding for speech tokens (0-6563)
        std::vector<mx::array> next_embeddings;
        for (int b = 0; b < batch; ++b) {
            int token = next_tokens[b];
            int emb_idx = (token < static_cast<int>(speech_embedding_.shape()[0])) ? token : 0;
            auto emb = mx::slice(speech_embedding_, {emb_idx, 0}, {emb_idx + 1, static_cast<int>(speech_embedding_.shape()[1])});
            next_embeddings.push_back(emb);
        }
        current_input = mx::stack(next_embeddings, 0);  // [batch, 1, hidden_size]
        mx::eval(current_input);
    }

    // Convert decoded tokens to output array
    if (decoded_tokens[0].empty()) {
        return mx::zeros({batch, 0}, mx::int32);
    }

    // Find max length across batch
    size_t max_gen_len = 0;
    for (int b = 0; b < batch; ++b) {
        max_gen_len = std::max(max_gen_len, decoded_tokens[b].size());
    }

    // Create output array with padding
    std::vector<int> output_data(batch * max_gen_len, 0);
    for (int b = 0; b < batch; ++b) {
        for (size_t t = 0; t < decoded_tokens[b].size(); ++t) {
            output_data[b * max_gen_len + t] = decoded_tokens[b][t];
        }
    }

    return mx::array(output_data.data(), {batch, static_cast<int>(max_gen_len)}, mx::int32);
}

mx::array CosyVoice3Model::tokens_to_mel(
    const mx::array& tokens,
    const mx::array& speaker_emb,
    int num_steps,
    float cfg_strength,
    bool streaming
) {
    return flow_->inference(tokens, speaker_emb, num_steps, cfg_strength, streaming);
}

mx::array CosyVoice3Model::mel_to_audio(const mx::array& mel) {
    mx::array mel_input = mel;
    if (mel.shape()[2] == config_.vocoder_config.in_channels) {
        mel_input = mx::transpose(mel, {0, 2, 1});
    }
    return vocoder_->forward(mel_input);
}

mx::array CosyVoice3Model::synthesize(
    const mx::array& text_ids,
    const mx::array& speaker_emb,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int flow_steps,
    float cfg_strength
) {
    auto tokens = generate_speech_tokens(text_ids, max_tokens, temperature, top_k, top_p);
    auto mel = tokens_to_mel(tokens, speaker_emb, flow_steps, cfg_strength);
    return mel_to_audio(mel);
}

void CosyVoice3Model::synthesize_streaming(
    const mx::array& text_ids,
    const mx::array& speaker_emb,
    StreamCallback callback,
    int chunk_size
) {
    auto tokens = generate_speech_tokens(text_ids, 1000);

    auto shape = tokens.shape();
    int B = static_cast<int>(shape[0]);
    int num_tokens = static_cast<int>(shape[1]);
    int num_chunks = (num_tokens + chunk_size - 1) / chunk_size;

    for (int i = 0; i < num_chunks; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, num_tokens);

        auto chunk_tokens = mx::slice(tokens, {0, start}, {B, end});
        auto mel = flow_->inference(chunk_tokens, speaker_emb, 10, 0.7f, true);

        bool is_final = (i == num_chunks - 1);
        auto audio = vocoder_->forward_streaming(mel, i == 0, is_final);

        callback(audio, is_final);
    }
}

void CosyVoice3Model::compile() {
    flow_->compile();
    vocoder_->compile();
}

void CosyVoice3Model::optimize_all() {
    flow_->fuse_qkv_weights();
    flow_->enable_speaker_cache(true);
    compile();
}

void CosyVoice3Model::load_llm_weights(const CosyVoice3Weights& weights) {
    // Load CosyVoice3-specific speech embeddings and decoder
    // After load_combined() strips "llm." prefix, keys become:
    // - llm_embedding.weight (SOS/EOS tokens)
    // - speech_embedding.weight (speech token embeddings)
    // - early_exit_head.weight/bias (speech logits decoder)
    // - llm.embed_tokens.weight (text embeddings)
    // - lm_head.weight (text logits)
    // - llm.norm.weight (final norm)
    // - llm.layers.N.* (transformer layers)

    if (weights.has("llm_embedding.weight")) {
        llm_embedding_ = weights.get("llm_embedding.weight");
    }
    if (weights.has("speech_embedding.weight")) {
        speech_embedding_ = weights.get("speech_embedding.weight");
    }
    // Speech decoder is called "early_exit_head" in CosyVoice3
    if (weights.has("early_exit_head.weight")) {
        llm_decoder_weight_ = weights.get("early_exit_head.weight");
    }
    if (weights.has("early_exit_head.bias")) {
        llm_decoder_bias_ = weights.get("early_exit_head.bias");
    }

    // Load text embeddings (key is llm.embed_tokens.weight after stripping outer llm. prefix)
    if (weights.has("llm.embed_tokens.weight")) {
        embed_tokens_ = weights.get("llm.embed_tokens.weight");
    }

    // Load LM head (text logits projection)
    if (weights.has("lm_head.weight")) {
        lm_head_ = weights.get("lm_head.weight");
    }

    // Load final norm
    if (weights.has("llm.norm.weight")) {
        final_norm_w_ = weights.get("llm.norm.weight");
    }

    // Load transformer layer weights (using maps - no resize needed)
    // Keys are llm.layers.N.* after stripping outer llm. prefix
    int num_layers = config_.llm_config.num_hidden_layers;

    for (int i = 0; i < num_layers; ++i) {
        std::string prefix = "llm.layers." + std::to_string(i);

        // Attention projections (Qwen2 has biases on Q, K, V)
        if (weights.has(prefix + ".self_attn.q_proj.weight")) {
            q_proj_w_.emplace(i, weights.get(prefix + ".self_attn.q_proj.weight"));
            q_proj_b_.emplace(i, weights.get(prefix + ".self_attn.q_proj.bias"));
        }
        if (weights.has(prefix + ".self_attn.k_proj.weight")) {
            k_proj_w_.emplace(i, weights.get(prefix + ".self_attn.k_proj.weight"));
            k_proj_b_.emplace(i, weights.get(prefix + ".self_attn.k_proj.bias"));
        }
        if (weights.has(prefix + ".self_attn.v_proj.weight")) {
            v_proj_w_.emplace(i, weights.get(prefix + ".self_attn.v_proj.weight"));
            v_proj_b_.emplace(i, weights.get(prefix + ".self_attn.v_proj.bias"));
        }
        if (weights.has(prefix + ".self_attn.o_proj.weight")) {
            o_proj_w_.emplace(i, weights.get(prefix + ".self_attn.o_proj.weight"));
        }

        // MLP projections (Qwen2: SwiGLU with gate, up, down)
        if (weights.has(prefix + ".mlp.gate_proj.weight")) {
            gate_proj_w_.emplace(i, weights.get(prefix + ".mlp.gate_proj.weight"));
        }
        if (weights.has(prefix + ".mlp.up_proj.weight")) {
            up_proj_w_.emplace(i, weights.get(prefix + ".mlp.up_proj.weight"));
        }
        if (weights.has(prefix + ".mlp.down_proj.weight")) {
            down_proj_w_.emplace(i, weights.get(prefix + ".mlp.down_proj.weight"));
        }

        // Layer norms
        if (weights.has(prefix + ".input_layernorm.weight")) {
            input_layernorm_w_.emplace(i, weights.get(prefix + ".input_layernorm.weight"));
        }
        if (weights.has(prefix + ".post_attention_layernorm.weight")) {
            post_attn_layernorm_w_.emplace(i, weights.get(prefix + ".post_attention_layernorm.weight"));
        }
    }

    // Initialize RoPE frequencies
    int head_dim = config_.llm_config.head_dim;
    int half_dim = head_dim / 2;
    std::vector<float> freqs(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        freqs[i] = 1.0f / std::pow(rope_theta_, static_cast<float>(i) / half_dim);
    }
    rope_freqs_ = mx::array(freqs.data(), {half_dim});

    llm_loaded_ = true;
}

mx::array CosyVoice3Model::rms_norm(const mx::array& x, const mx::array& weight, float eps) {
    // x: [batch, seq_len, hidden_size]
    // Compute RMS and normalize
    auto variance = mx::mean(x * x, -1, true);
    auto x_norm = x * mx::rsqrt(variance + eps);
    return x_norm * weight;
}

mx::array CosyVoice3Model::apply_rope(const mx::array& x, int offset) {
    // x: [batch, num_heads, seq_len, head_dim]
    auto shape = x.shape();
    int seq_len = static_cast<int>(shape[2]);
    int head_dim = static_cast<int>(shape[3]);
    int half_dim = head_dim / 2;

    // Create position indices
    std::vector<int> positions(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        positions[i] = offset + i;
    }
    auto pos = mx::array(positions.data(), {seq_len});

    // Compute angles: [seq_len, half_dim]
    auto angles = mx::expand_dims(mx::astype(pos, mx::float32), 1) * mx::expand_dims(rope_freqs_, 0);

    // Compute cos and sin
    auto cos_vals = mx::cos(angles);  // [seq_len, half_dim]
    auto sin_vals = mx::sin(angles);  // [seq_len, half_dim]

    // Reshape for broadcasting: [1, 1, seq_len, half_dim]
    cos_vals = mx::expand_dims(mx::expand_dims(cos_vals, 0), 0);
    sin_vals = mx::expand_dims(mx::expand_dims(sin_vals, 0), 0);

    // Split x into two halves
    auto x1 = mx::slice(x, {0, 0, 0, 0}, {static_cast<int>(shape[0]), static_cast<int>(shape[1]), seq_len, half_dim});
    auto x2 = mx::slice(x, {0, 0, 0, half_dim}, {static_cast<int>(shape[0]), static_cast<int>(shape[1]), seq_len, head_dim});

    // Apply rotation
    auto x_rotated = mx::concatenate({
        x1 * cos_vals - x2 * sin_vals,
        x1 * sin_vals + x2 * cos_vals
    }, -1);

    return x_rotated;
}

mx::array CosyVoice3Model::llm_attention(const mx::array& x, int layer_idx, llm::KVCache* cache) {
    // x: [batch, seq_len, hidden_size]
    auto shape = x.shape();
    int batch = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);
    int num_heads = config_.llm_config.num_attention_heads;
    int num_kv_heads = config_.llm_config.num_key_value_heads;
    int head_dim = config_.llm_config.head_dim;

    // Project Q, K, V
    auto q = mx::matmul(x, mx::transpose(q_proj_w_.at(layer_idx))) + q_proj_b_.at(layer_idx);
    auto k = mx::matmul(x, mx::transpose(k_proj_w_.at(layer_idx))) + k_proj_b_.at(layer_idx);
    auto v = mx::matmul(x, mx::transpose(v_proj_w_.at(layer_idx))) + v_proj_b_.at(layer_idx);

    // Reshape to [batch, num_heads, seq_len, head_dim]
    q = mx::reshape(q, {batch, seq_len, num_heads, head_dim});
    q = mx::transpose(q, {0, 2, 1, 3});

    k = mx::reshape(k, {batch, seq_len, num_kv_heads, head_dim});
    k = mx::transpose(k, {0, 2, 1, 3});

    v = mx::reshape(v, {batch, seq_len, num_kv_heads, head_dim});
    v = mx::transpose(v, {0, 2, 1, 3});

    // Apply RoPE
    int offset = cache ? cache->offset : 0;
    q = apply_rope(q, offset);
    k = apply_rope(k, offset);

    // Handle KV cache
    if (cache) {
        // Expand cache vectors if needed (use dummy arrays as placeholders)
        while (cache->keys.size() <= static_cast<size_t>(layer_idx)) {
            cache->keys.push_back(mx::zeros({0}));  // Dummy placeholder
            cache->values.push_back(mx::zeros({0}));  // Dummy placeholder
        }

        if (cache->keys[layer_idx].size() > 0) {
            k = mx::concatenate({cache->keys[layer_idx], k}, 2);
            v = mx::concatenate({cache->values[layer_idx], v}, 2);
        }
        cache->keys[layer_idx] = k;
        cache->values[layer_idx] = v;
    }

    // GQA: Repeat K,V for each query head group
    int heads_per_kv = num_heads / num_kv_heads;
    if (heads_per_kv > 1) {
        // Repeat K and V along head dimension
        k = mx::repeat(k, heads_per_kv, 1);
        v = mx::repeat(v, heads_per_kv, 1);
    }

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto scores = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})) * scale;

    // Causal mask
    int kv_len = static_cast<int>(k.shape()[2]);
    if (seq_len > 1) {
        // Create causal mask
        auto mask = mx::triu(mx::full({seq_len, kv_len}, -1e9f), kv_len - seq_len + 1);
        scores = scores + mask;
    }

    // Softmax and attention
    auto attn_weights = mx::softmax(scores, -1);
    auto attn_output = mx::matmul(attn_weights, v);

    // Reshape back to [batch, seq_len, hidden_size]
    attn_output = mx::transpose(attn_output, {0, 2, 1, 3});
    attn_output = mx::reshape(attn_output, {batch, seq_len, num_heads * head_dim});

    // Output projection
    auto output = mx::matmul(attn_output, mx::transpose(o_proj_w_.at(layer_idx)));

    return output;
}

mx::array CosyVoice3Model::llm_mlp(const mx::array& x, int layer_idx) {
    // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    auto gate = mx::matmul(x, mx::transpose(gate_proj_w_.at(layer_idx)));
    auto up = mx::matmul(x, mx::transpose(up_proj_w_.at(layer_idx)));

    // SiLU activation on gate
    auto silu_gate = gate * mx::sigmoid(gate);

    // Element-wise multiply and down projection
    auto hidden = silu_gate * up;
    return mx::matmul(hidden, mx::transpose(down_proj_w_.at(layer_idx)));
}

mx::array CosyVoice3Model::llm_forward(const mx::array& hidden_states, llm::KVCache* cache) {
    // hidden_states: [batch, seq_len, hidden_size]
    mx::array x = hidden_states;

    int num_layers = config_.llm_config.num_hidden_layers;
    for (int i = 0; i < num_layers; ++i) {
        // Pre-norm attention
        auto residual = x;
        x = rms_norm(x, input_layernorm_w_.at(i), config_.llm_config.rms_norm_eps);
        x = llm_attention(x, i, cache);
        x = residual + x;

        // Pre-norm MLP
        residual = x;
        x = rms_norm(x, post_attn_layernorm_w_.at(i), config_.llm_config.rms_norm_eps);
        x = llm_mlp(x, i);
        x = residual + x;

        mx::eval(x);  // Free intermediate memory
    }

    // Final norm
    x = rms_norm(x, final_norm_w_, config_.llm_config.rms_norm_eps);

    return x;
}

mx::array CosyVoice3Model::sample_token(const mx::array& logits, float temperature, int top_k, float top_p) {
    // logits: [batch, vocab_size]
    // Returns: [batch] sampled token IDs

    // Greedy decoding
    if (temperature <= 0.0f) {
        return mx::argmax(logits, -1);
    }

    // Temperature scaling
    auto scaled_logits = logits / temperature;

    int vocab_size = static_cast<int>(logits.shape().back());
    int batch_size = static_cast<int>(logits.shape()[0]);

    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        // Get top-k values using topk
        auto topk_vals = mx::topk(scaled_logits, top_k, -1);
        auto threshold = mx::slice(topk_vals, {0, 0}, {batch_size, 1});
        auto neg_inf = mx::full(scaled_logits.shape(), -1e9f);
        scaled_logits = mx::where(scaled_logits < threshold, neg_inf, scaled_logits);
    }

    // Top-p (nucleus) filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        // Sort descending
        auto sorted_indices = mx::argsort(-scaled_logits, -1);
        auto sorted_logits = mx::take_along_axis(scaled_logits, sorted_indices, -1);
        auto sorted_probs = mx::softmax(sorted_logits, -1);
        auto cumsum_probs = mx::cumsum(sorted_probs, -1);

        // Create cutoff mask
        auto cutoff_mask = cumsum_probs > top_p;
        // Shift to include first token above threshold
        auto zeros_col = mx::zeros({batch_size, 1});
        cutoff_mask = mx::concatenate({mx::astype(zeros_col, mx::bool_), mx::slice(cutoff_mask, {0, 0}, {batch_size, vocab_size - 1})}, -1);

        // Get threshold from sorted order
        auto cutoff_idx = mx::argmax(mx::astype(cutoff_mask, mx::int32), -1);
        auto threshold = mx::take_along_axis(sorted_logits, mx::expand_dims(cutoff_idx, -1), -1);
        auto neg_inf = mx::full(scaled_logits.shape(), -1e9f);
        scaled_logits = mx::where(scaled_logits < threshold, neg_inf, scaled_logits);
    }

    // Categorical sampling
    auto logits_f32 = mx::astype(scaled_logits, mx::float32);
    mx::eval(logits_f32);
    auto sampled = mx::random::categorical(logits_f32, 1);
    mx::eval(sampled);
    return sampled;
}

void CosyVoice3Model::load_weights(const CosyVoice3Weights& weights) {
    if (weights.flow_weight_count() > 0) {
        flow_->load_weights(weights);
    }
    if (weights.vocoder_weight_count() > 0) {
        vocoder_->load_weights(weights);
    }
    // Load LLM weights if available
    if (weights.llm_weight_count() > 0) {
        load_llm_weights(weights);
    }
    loaded_ = true;
}

std::string CosyVoice3Model::info() const {
    std::string info = "CosyVoice3 Model\n";
    info += "  Sample Rate: " + std::to_string(config_.sample_rate) + " Hz\n";
    info += "  Token Frame Rate: " + std::to_string(config_.token_frame_rate) + " tokens/s\n";
    info += "  DiT Depth: " + std::to_string(config_.flow_config.depth) + " layers\n";
    info += "  DiT Dim: " + std::to_string(config_.flow_config.dim) + "\n";
    info += "  Vocoder Upsample: " + std::to_string(config_.vocoder_config.total_upsample_factor()) + "x\n";
    info += "  Loaded: " + std::string(loaded_ ? "Yes" : "No") + "\n";
    return info;
}

} // namespace cosyvoice3
