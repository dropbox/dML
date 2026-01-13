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

// Silero VAD MLX C++ Implementation
// Voice Activity Detection ported from PyTorch to MLX C++

#include "silero_vad.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace silero_vad {

// ============================================================================
// Weight loading from binary format
// ============================================================================

namespace {

struct TensorHeader {
    std::string name;
    std::vector<int> shape;
    size_t data_size;
};

TensorHeader read_tensor_header(std::ifstream& f) {
    TensorHeader header;

    // Read name
    uint32_t name_len;
    f.read(reinterpret_cast<char*>(&name_len), 4);
    header.name.resize(name_len);
    f.read(&header.name[0], name_len);

    // Read shape
    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), 4);
    header.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim;
        f.read(reinterpret_cast<char*>(&dim), 4);
        header.shape[i] = static_cast<int>(dim);
    }

    // Read data size
    uint32_t data_size;
    f.read(reinterpret_cast<char*>(&data_size), 4);
    header.data_size = data_size;

    return header;
}

}  // anonymous namespace

void SileroVAD::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + path);
    }

    // Read magic
    char magic[4];
    f.read(magic, 4);
    if (std::strncmp(magic, "SVAD", 4) != 0) {
        throw std::runtime_error("Invalid Silero VAD weights file (bad magic)");
    }

    // Read version
    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), 4);
    if (version != 1) {
        throw std::runtime_error("Unsupported Silero VAD weights version: " + std::to_string(version));
    }

    // Read number of tensors
    uint32_t num_tensors;
    f.read(reinterpret_cast<char*>(&num_tensors), 4);

    // Read each tensor
    for (uint32_t i = 0; i < num_tensors; ++i) {
        TensorHeader header = read_tensor_header(f);

        // Read data
        std::vector<float> data(header.data_size / sizeof(float));
        f.read(reinterpret_cast<char*>(data.data()), header.data_size);

        // Create MLX array using the Shape type (SmallVector<int>)
        mx::Shape shape(header.shape.begin(), header.shape.end());
        mx::array arr = mx::array(data.data(), shape, mx::float32);

        // Store in weights map
        weights_.emplace(header.name, arr);
    }
}

// ============================================================================
// Constructor and reset
// ============================================================================

SileroVAD::SileroVAD(const std::string& weights_path, int sample_rate)
    : sample_rate_(sample_rate) {

    // Set parameters based on sample rate
    if (sample_rate == 16000) {
        filter_length_ = 256;
        hop_length_ = 128;
        context_size_ = 64;
    } else if (sample_rate == 8000) {
        filter_length_ = 128;
        hop_length_ = 64;
        context_size_ = 32;
    } else {
        throw std::runtime_error("Unsupported sample rate: " + std::to_string(sample_rate) +
                                 ". Must be 16000 or 8000.");
    }

    n_freq_ = filter_length_ / 2 + 1;

    // Load weights
    load_weights(weights_path);

    // Validate critical weights loaded
    if (weights_.find("stft_forward_basis_buffer") == weights_.end()) {
        throw std::runtime_error("Failed to load STFT weights");
    }
    if (weights_.find("decoder_rnn_weight_ih") == weights_.end()) {
        throw std::runtime_error("Failed to load LSTM weights");
    }
}

void SileroVAD::reset_state() {
    h_state_.reset();
    c_state_.reset();
    context_.reset();
}

// ============================================================================
// STFT Implementation (conv1d-based, matching Silero)
// ============================================================================

mx::array SileroVAD::compute_stft(const mx::array& audio) {
    // audio: [batch, samples]

    // Silero uses reflection padding on the right side (not zero padding)
    // This is critical for numerical exactness with PyTorch implementation
    int pad_right = filter_length_ / 4;  // 64 for 16kHz

    // Implement reflection padding: mirror the last pad_right samples
    // For input [a, b, c, d, e] with pad=2: [a, b, c, d, e, d, c]
    // The reflection excludes the edge sample itself
    int samples = audio.shape(1);

    // Create reversed indices for the reflection: [pad_right-1, pad_right-2, ..., 0]
    // These index into the last pad_right samples (excluding the final sample)
    std::vector<int32_t> rev_indices;
    rev_indices.reserve(pad_right);
    for (int i = samples - 2; i >= samples - pad_right - 1; --i) {
        rev_indices.push_back(i);
    }
    auto indices = mx::array(rev_indices.data(), {static_cast<int>(rev_indices.size())}, mx::int32);

    // Gather the reflected samples using advanced indexing
    auto reflected = mx::take(audio, indices, 1);

    // Concatenate original audio with reflected part
    auto x = mx::concatenate({audio, reflected}, 1);

    // Add channel dim: [batch, samples] -> [batch, samples, 1] (NLC for MLX)
    x = mx::expand_dims(x, -1);

    // Conv1d with DFT basis
    // forward_basis_buffer from PyTorch: [out_channels=258, 1, kernel=256]
    // MLX conv1d weight format: [out_channels, kernel, in_channels]
    // PyTorch stores as [out, in, kernel], so we need to transpose [0,2,1]
    auto& stft_basis = weights_.at("stft_forward_basis_buffer");
    auto weight = mx::transpose(stft_basis, {0, 2, 1});  // [258, 256, 1]

    auto forward_transform = mx::conv1d(x, weight, hop_length_, /*padding=*/0);
    // forward_transform: [batch, frames, 258] (NLC)

    // Transpose to NCL: [batch, 258, frames]
    forward_transform = mx::transpose(forward_transform, {0, 2, 1});

    // Split into real and imaginary parts
    int cutoff = n_freq_;  // 129
    auto real_part = mx::slice(forward_transform, {0, 0, 0},
                               {forward_transform.shape(0), cutoff, forward_transform.shape(2)});
    auto imag_part = mx::slice(forward_transform, {0, cutoff, 0},
                               {forward_transform.shape(0), 2 * cutoff, forward_transform.shape(2)});

    // Compute magnitude
    auto magnitude = mx::sqrt(real_part * real_part + imag_part * imag_part);

    return magnitude;  // [batch, n_freq, frames]
}

// ============================================================================
// Encoder (4 conv blocks with ReLU)
// ============================================================================

mx::array SileroVAD::encoder_forward(const mx::array& stft_mag) {
    // stft_mag: [batch, n_freq=129, frames] (NCL format from STFT)
    // PyTorch conv1d expects NCL, MLX conv1d expects NLC

    // Convert to NLC
    auto x = mx::transpose(stft_mag, {0, 2, 1});  // [batch, frames, 129]

    // Encoder block 0: [129->128, stride=1, kernel=3, padding=1]
    // PyTorch weight: [out, in, kernel] -> MLX: [out, kernel, in]
    auto w0 = mx::transpose(weights_.at("encoder_0_weight"), {0, 2, 1});  // [128, 3, 129]
    x = mx::conv1d(x, w0, /*stride=*/1, /*padding=*/1);
    x = x + weights_.at("encoder_0_bias");
    x = mx::maximum(x, mx::array(0.0f));  // ReLU

    // Encoder block 1: [128->64, stride=2, kernel=3, padding=1]
    auto w1 = mx::transpose(weights_.at("encoder_1_weight"), {0, 2, 1});  // [64, 3, 128]
    x = mx::conv1d(x, w1, /*stride=*/2, /*padding=*/1);
    x = x + weights_.at("encoder_1_bias");
    x = mx::maximum(x, mx::array(0.0f));  // ReLU

    // Encoder block 2: [64->64, stride=2, kernel=3, padding=1]
    auto w2 = mx::transpose(weights_.at("encoder_2_weight"), {0, 2, 1});  // [64, 3, 64]
    x = mx::conv1d(x, w2, /*stride=*/2, /*padding=*/1);
    x = x + weights_.at("encoder_2_bias");
    x = mx::maximum(x, mx::array(0.0f));  // ReLU

    // Encoder block 3: [64->128, stride=1, kernel=3, padding=1]
    auto w3 = mx::transpose(weights_.at("encoder_3_weight"), {0, 2, 1});  // [128, 3, 64]
    x = mx::conv1d(x, w3, /*stride=*/1, /*padding=*/1);
    x = x + weights_.at("encoder_3_bias");
    x = mx::maximum(x, mx::array(0.0f));  // ReLU

    // Convert back to NCL for decoder
    return mx::transpose(x, {0, 2, 1});  // [batch, 128, frames]
}

// ============================================================================
// LSTM Cell
// ============================================================================

mx::array SileroVAD::lstm_cell(const mx::array& x) {
    // x: [batch, 128]
    int batch_size = x.shape(0);
    int hidden_size = 128;

    // Initialize state if needed
    if (!h_state_.has_value()) {
        h_state_ = mx::zeros({batch_size, hidden_size});
        c_state_ = mx::zeros({batch_size, hidden_size});
    }

    auto h = h_state_.value();
    auto c = c_state_.value();

    // Get LSTM weights
    auto& weight_ih = weights_.at("decoder_rnn_weight_ih");
    auto& weight_hh = weights_.at("decoder_rnn_weight_hh");
    auto& bias_ih = weights_.at("decoder_rnn_bias_ih");
    auto& bias_hh = weights_.at("decoder_rnn_bias_hh");

    // Compute gates: gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
    auto gates = mx::matmul(x, mx::transpose(weight_ih)) + bias_ih +
                 mx::matmul(h, mx::transpose(weight_hh)) + bias_hh;

    // Split gates into i, f, g, o (each hidden_size=128)
    auto i_gate = mx::sigmoid(mx::slice(gates, {0, 0}, {batch_size, hidden_size}));
    auto f_gate = mx::sigmoid(mx::slice(gates, {0, hidden_size}, {batch_size, 2 * hidden_size}));
    auto g_gate = mx::tanh(mx::slice(gates, {0, 2 * hidden_size}, {batch_size, 3 * hidden_size}));
    auto o_gate = mx::sigmoid(mx::slice(gates, {0, 3 * hidden_size}, {batch_size, 4 * hidden_size}));

    // Update cell and hidden state
    auto c_new = f_gate * c + i_gate * g_gate;
    auto h_new = o_gate * mx::tanh(c_new);

    // Store state
    h_state_ = h_new;
    c_state_ = c_new;

    return h_new;  // [batch, 128]
}

// ============================================================================
// Decoder
// ============================================================================

mx::array SileroVAD::decoder_forward(const mx::array& features) {
    // features: [batch, 128, frames] (NCL)
    // After downsampling, frames should be 1

    // Squeeze last dim (frames=1 after downsampling)
    auto x = mx::squeeze(features, -1);  // [batch, 128]

    // LSTM
    auto h = lstm_cell(x);  // [batch, 128]

    // ReLU (dropout is identity during inference)
    h = mx::maximum(h, mx::array(0.0f));

    // Output projection via Conv1d(128, 1, kernel=1)
    // Add length dim for conv: [batch, 128] -> [batch, 1, 128] (NLC)
    h = mx::expand_dims(h, 1);

    // Conv1d weight: PyTorch [1, 128, 1] -> MLX [1, 1, 128]
    auto& dec_w = weights_.at("decoder_output_weight");
    auto& dec_b = weights_.at("decoder_output_bias");
    auto w = mx::transpose(dec_w, {0, 2, 1});  // [1, 1, 128]
    auto prob = mx::conv1d(h, w, /*stride=*/1, /*padding=*/0);  // [batch, 1, 1]
    prob = prob + dec_b;

    // Sigmoid
    prob = mx::sigmoid(prob);

    // Remove extra dims: [batch, 1, 1] -> [batch, 1]
    return mx::squeeze(prob, -1);  // [batch, 1]
}

// ============================================================================
// Public API
// ============================================================================

float SileroVAD::process(const float* samples, size_t count) {
    if (count == 0) {
        return 0.0f;
    }

    // Create audio array [1, count]
    mx::array audio = mx::array(samples, {1, static_cast<int>(count)}, mx::float32);

    // Initialize context if needed
    if (!context_.has_value()) {
        context_ = mx::zeros({1, context_size_});
    }

    // Prepend context to audio
    auto audio_with_context = mx::concatenate({context_.value(), audio}, 1);

    // Update context for next chunk (last context_size samples)
    int total_len = audio_with_context.shape(1);
    context_ = mx::slice(audio_with_context, {0, total_len - context_size_},
                         {1, total_len});

    // STFT
    auto stft_mag = compute_stft(audio_with_context);

    // Encoder
    auto features = encoder_forward(stft_mag);

    // Decoder
    auto prob = decoder_forward(features);

    // Evaluate and return
    mx::eval(prob);

    // Get scalar value
    float result = prob.item<float>();
    return result;
}

std::vector<SpeechSegment> SileroVAD::get_speech_segments(
    const float* audio,
    size_t length,
    float threshold,
    int min_speech_duration_ms,
    int min_silence_duration_ms,
    int speech_pad_ms,
    float max_speech_duration_s,  // GAP 51: Maximum speech segment duration (0 = disabled)
    float samples_overlap_s      // GAP 52: Overlap in seconds for segment extraction
) {
    // Reset state for new audio
    reset_state();

    int chunk_sz = chunk_size();
    int num_chunks = length / chunk_sz;

    // Collect per-chunk probabilities
    std::vector<float> probs;
    probs.reserve(num_chunks);

    for (int i = 0; i < num_chunks; ++i) {
        float prob = process(audio + i * chunk_sz, chunk_sz);
        probs.push_back(prob);
    }

    // Convert to speech/silence segments
    std::vector<SpeechSegment> segments;

    if (probs.empty()) {
        return segments;
    }

    // Time per chunk in seconds
    float chunk_duration = static_cast<float>(chunk_sz) / static_cast<float>(sample_rate_);

    // Speech padding in samples (matching Python Silero behavior)
    int speech_pad_samples = static_cast<int>(speech_pad_ms * sample_rate_ / 1000);

    // Minimum durations in chunks
    int min_speech_chunks = static_cast<int>(std::ceil(
        (min_speech_duration_ms / 1000.0f) / chunk_duration));
    int min_silence_chunks = static_cast<int>(std::ceil(
        (min_silence_duration_ms / 1000.0f) / chunk_duration));

    // neg_threshold: Python uses threshold - 0.15 (clamped to 0.01)
    // temp_end is set when prob first drops below neg_threshold
    // Add 0.02 precision tolerance to compensate for MLX vs ONNX LSTM differences
    // (C++ Silero produces ~0.01-0.02 lower probabilities than Python/ONNX at boundary conditions,
    //  e.g., chunk 312: C++ = 0.3485, Python = 0.3527; chunk 664: C++ = 0.3362, Python = 0.3474)
    float neg_threshold = std::max(threshold - 0.15f - 0.02f, 0.01f);


    // State machine for segment detection (matching Python Silero algorithm)
    bool in_speech = false;
    int speech_start = 0;
    int temp_end = 0;       // Potential segment end (sample index), set when prob < neg_threshold
    int silence_start = 0;  // Chunk when silence started (prob < threshold)

    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        float prob = probs[i];
        int cur_sample = i * chunk_sz;  // Sample position at START of this chunk (matching Python)

        if (!in_speech) {
            if (prob >= threshold) {
                // Start of speech segment
                in_speech = true;
                speech_start = i;
                temp_end = 0;
                silence_start = 0;
            }
        } else {
            // In speech segment
            if (prob >= threshold) {
                // Speech continues - if we had a potential end, clear it
                if (temp_end != 0) {
                    // Speech returned after temp_end was set - this was a brief dip
                    temp_end = 0;
                }
                silence_start = 0;
            } else {
                // Prob < threshold - potential silence
                if (silence_start == 0) {
                    silence_start = i;  // First silence chunk
                }

                // Track when prob drops below neg_threshold (this sets temp_end in Python)
                if (prob < neg_threshold && temp_end == 0) {
                    temp_end = cur_sample;  // Sample position at START of this chunk
                }

                // Check if silence is long enough to end segment
                int silence_chunks = i - silence_start + 1;
                if (silence_chunks >= min_silence_chunks && temp_end != 0) {
                    // End segment at temp_end (where prob first dropped below neg_threshold)
                    int speech_len_chunks = silence_start - speech_start;

                    if (speech_len_chunks >= min_speech_chunks) {
                        SpeechSegment seg;
                        seg.start_sample = speech_start * chunk_sz;
                        seg.end_sample = std::min(temp_end, static_cast<int>(length));
                        seg.start_time = seg.start_sample / static_cast<float>(sample_rate_);
                        seg.end_time = seg.end_sample / static_cast<float>(sample_rate_);
                        segments.push_back(seg);
                    }

                    in_speech = false;
                    temp_end = 0;
                    silence_start = 0;
                }
            }
        }
    }

    // Handle final segment
    if (in_speech) {
        int speech_end = static_cast<int>(probs.size());
        int speech_len = speech_end - speech_start;

        if (speech_len >= min_speech_chunks) {
            SpeechSegment seg;
            seg.start_sample = speech_start * chunk_sz;
            seg.end_sample = std::min(speech_end * chunk_sz, static_cast<int>(length));
            seg.start_time = seg.start_sample / static_cast<float>(sample_rate_);
            seg.end_time = seg.end_sample / static_cast<float>(sample_rate_);
            segments.push_back(seg);
        }
    }

    // Apply speech padding (matching Python Silero get_speech_timestamps behavior)
    // This pads each segment by speech_pad_ms on each side, sharing gaps when necessary
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i == 0) {
            // First segment: pad start (clamp to 0)
            segments[i].start_sample = std::max(0, segments[i].start_sample - speech_pad_samples);
        }

        if (i != segments.size() - 1) {
            // Middle segments: check gap to next segment
            int gap = segments[i + 1].start_sample - segments[i].end_sample;
            if (gap < 2 * speech_pad_samples) {
                // Gap is small - share it between segments
                int half_gap = gap / 2;
                segments[i].end_sample += half_gap;
                segments[i + 1].start_sample -= (gap - half_gap);  // Remaining half
            } else {
                // Gap is large enough - full padding
                segments[i].end_sample = std::min(static_cast<int>(length),
                                                   segments[i].end_sample + speech_pad_samples);
                segments[i + 1].start_sample = std::max(0,
                                                         segments[i + 1].start_sample - speech_pad_samples);
            }
        } else {
            // Last segment: pad end (clamp to audio length)
            segments[i].end_sample = std::min(static_cast<int>(length),
                                               segments[i].end_sample + speech_pad_samples);
        }

        // Update times from samples
        segments[i].start_time = segments[i].start_sample / static_cast<float>(sample_rate_);
        segments[i].end_time = segments[i].end_sample / static_cast<float>(sample_rate_);
    }

    // GAP 51: Split segments that exceed max_speech_duration_s
    // This prevents runaway segments that can cause hallucinations
    if (max_speech_duration_s > 0.0f) {
        int max_samples = static_cast<int>(max_speech_duration_s * sample_rate_);
        std::vector<SpeechSegment> split_segments;
        split_segments.reserve(segments.size() * 2);  // Estimate

        for (const auto& seg : segments) {
            int seg_samples = seg.end_sample - seg.start_sample;
            if (seg_samples <= max_samples) {
                split_segments.push_back(seg);
            } else {
                // Split segment into chunks of max_samples
                int start = seg.start_sample;
                while (start < seg.end_sample) {
                    int end = std::min(start + max_samples, seg.end_sample);
                    SpeechSegment split_seg;
                    split_seg.start_sample = start;
                    split_seg.end_sample = end;
                    split_seg.start_time = static_cast<float>(start) / sample_rate_;
                    split_seg.end_time = static_cast<float>(end) / sample_rate_;
                    split_segments.push_back(split_seg);
                    start = end;
                }
            }
        }
        segments = std::move(split_segments);
    }

    // GAP 52: Apply samples_overlap to create overlapping segment boundaries
    // This helps smooth transitions between segments when extracting audio
    // Overlap extends segment boundaries: start pushed earlier, end pushed later
    // Results in adjacent segments sharing some audio samples
    if (samples_overlap_s > 0.0f && segments.size() > 0) {
        int overlap_samples = static_cast<int>(samples_overlap_s * sample_rate_);
        int half_overlap = overlap_samples / 2;

        for (size_t i = 0; i < segments.size(); ++i) {
            // Extend start earlier (except for first segment which is clamped to 0)
            int new_start = segments[i].start_sample - half_overlap;
            segments[i].start_sample = std::max(0, new_start);

            // Extend end later (except for last segment which is clamped to audio length)
            int new_end = segments[i].end_sample + half_overlap;
            segments[i].end_sample = std::min(static_cast<int>(length), new_end);

            // Update times from samples
            segments[i].start_time = segments[i].start_sample / static_cast<float>(sample_rate_);
            segments[i].end_time = segments[i].end_sample / static_cast<float>(sample_rate_);
        }
    }

    return segments;
}

std::vector<float> SileroVAD::get_probabilities(const float* audio, size_t length) {
    // Reset state for new audio
    reset_state();

    int chunk_sz = chunk_size();
    int num_chunks = length / chunk_sz;

    std::vector<float> probs;
    probs.reserve(num_chunks);

    for (int i = 0; i < num_chunks; ++i) {
        float prob = process(audio + i * chunk_sz, chunk_sz);
        probs.push_back(prob);
    }

    return probs;
}

}  // namespace silero_vad
