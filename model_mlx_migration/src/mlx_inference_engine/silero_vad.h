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
//
// Original: https://github.com/snakers4/silero-vad (MIT License)
//
// Architecture:
// - STFT preprocessing (conv1d-based spectral features)
// - 4x Conv1d encoder blocks with ReLU
// - LSTM decoder with hidden state
// - Conv1d -> Sigmoid output (speech probability)
//
// Input: Audio chunks (512 samples at 16kHz = 32ms)
// Output: Speech probability [0, 1]

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace silero_vad {

/**
 * Speech segment detected by VAD.
 */
struct SpeechSegment {
    float start_time;      // Start time in seconds
    float end_time;        // End time in seconds
    int start_sample;      // Start sample index
    int end_sample;        // End sample index

    float duration() const { return end_time - start_time; }
};

/**
 * Silero VAD model for MLX C++.
 *
 * Implements neural network-based Voice Activity Detection matching
 * the Python Silero VAD implementation for exact parity with
 * WhisperMLX Python VAD processing.
 */
class SileroVAD {
public:
    /**
     * Construct Silero VAD model.
     *
     * @param weights_path Path to silero_vad_16k.bin weights file
     * @param sample_rate Audio sample rate (16000 or 8000)
     */
    explicit SileroVAD(const std::string& weights_path, int sample_rate = 16000);

    /**
     * Process a single audio chunk and return speech probability.
     *
     * @param samples Pointer to audio samples (float32, mono)
     * @param count Number of samples (should be 512 for 16kHz)
     * @return Speech probability [0, 1]
     */
    float process(const float* samples, size_t count);

    /**
     * Get speech segments from full audio file.
     *
     * @param audio Pointer to full audio signal
     * @param length Number of samples
     * @param threshold Speech detection threshold (default 0.5)
     * @param min_speech_duration_ms Minimum speech segment duration in ms
     * @param min_silence_duration_ms Minimum silence gap to split segments
     * @param speech_pad_ms Padding to add on each side of speech segment (default 30)
     * @param max_speech_duration_s GAP 51: Maximum speech segment duration in seconds
     *                              0 = disabled (default). Prevents runaway segments.
     * @param samples_overlap_s GAP 52: Overlap in seconds when extracting segment audio.
     *                          Creates overlap between adjacent segments for smoother transitions.
     *                          0 = disabled (default). Typical value: 0.25-0.5 seconds.
     * @return Vector of detected speech segments
     */
    std::vector<SpeechSegment> get_speech_segments(
        const float* audio,
        size_t length,
        float threshold = 0.5f,
        int min_speech_duration_ms = 250,
        int min_silence_duration_ms = 300,
        int speech_pad_ms = 30,
        float max_speech_duration_s = 0.0f,
        float samples_overlap_s = 0.0f
    );

    /**
     * Reset LSTM state and context for new audio stream.
     * Call this before processing a new audio file/stream.
     */
    void reset_state();

    /**
     * Get sample rate.
     */
    int sample_rate() const { return sample_rate_; }

    /**
     * Get chunk size (samples per process() call).
     */
    int chunk_size() const { return sample_rate_ == 16000 ? 512 : 256; }

    /**
     * Get per-chunk probabilities for full audio (for debugging).
     *
     * @param audio Pointer to full audio signal
     * @param length Number of samples
     * @return Vector of per-chunk probabilities
     */
    std::vector<float> get_probabilities(const float* audio, size_t length);

private:
    // Sample rate
    int sample_rate_;

    // STFT parameters
    int filter_length_;  // 256 for 16kHz, 128 for 8kHz
    int hop_length_;     // 128 for 16kHz, 64 for 8kHz
    int n_freq_;         // filter_length/2 + 1

    // Context size for streaming
    int context_size_;   // 64 for 16kHz, 32 for 8kHz

    // Weights (stored in a map since mx::array has no default constructor)
    std::unordered_map<std::string, mx::array> weights_;

    // LSTM state: [batch, hidden_size]
    std::optional<mx::array> h_state_;
    std::optional<mx::array> c_state_;

    // Audio context for streaming: [batch, context_size]
    std::optional<mx::array> context_;

    // Internal methods
    void load_weights(const std::string& path);

    // STFT: audio [batch, samples] -> magnitude [batch, n_freq, frames]
    mx::array compute_stft(const mx::array& audio);

    // Encoder: stft [batch, n_freq, frames] -> features [batch, 128, frames]
    mx::array encoder_forward(const mx::array& stft_mag);

    // Decoder: features [batch, 128, 1] -> prob [batch, 1]
    mx::array decoder_forward(const mx::array& features);

    // LSTM cell: x [batch, 128], state -> h [batch, 128], new_state
    mx::array lstm_cell(const mx::array& x);
};

}  // namespace silero_vad
