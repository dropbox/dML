// Copyright 2024-2025 Andrew Yates
// Feature extraction (Fbank) for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <string>

namespace zipformer {

using namespace mlx::core;

/**
 * Configuration for filterbank (Fbank) feature extraction.
 */
struct FbankConfig {
    int sample_rate = 16000;
    float frame_length_ms = 25.0f;
    float frame_shift_ms = 10.0f;
    int num_mel_bins = 80;
    float low_freq = 20.0f;
    float high_freq = -400.0f;  // Nyquist - 400 Hz
    bool use_energy = false;
    bool use_log_fbank = true;
    float dither = 0.0f;
    std::string window_type = "povey";  // hann with sqrt
};

/**
 * Filterbank feature extractor.
 *
 * Extracts log-Mel filterbank features from raw audio waveforms.
 * Compatible with Kaldi and torchaudio feature extraction.
 */
class FbankExtractor {
public:
    explicit FbankExtractor(const FbankConfig& config = FbankConfig());

    // Extract features from waveform
    // waveform: (samples,) or (batch, samples) - normalized audio [-1, 1]
    // Returns: (frames, num_mel_bins) or (batch, frames, num_mel_bins)
    array extract(const array& waveform) const;

    // Streaming extraction with state
    // Returns features for current chunk, updates internal buffer
    array extract_streaming(const array& chunk, array& state) const;

    // Get number of frames for given number of samples
    int get_num_frames(int num_samples) const;

    // Get number of samples for given number of frames
    int get_num_samples(int num_frames) const;

    const FbankConfig& config() const { return config_; }

private:
    FbankConfig config_;

    // Pre-computed
    int frame_length_samples_;
    int frame_shift_samples_;
    int fft_size_;

    // Mel filterbank matrix: (num_mel_bins, fft_size/2+1)
    array mel_filterbank_{zeros({1})};

    // Window function: (frame_length,)
    array window_{zeros({1})};

    void init_filterbank();
    void init_window();
    array apply_stft(const array& frames) const;
};

/**
 * Load audio from file.
 *
 * Supports WAV, FLAC, MP3 (via external decoder).
 * Returns normalized audio in [-1, 1] range.
 */
array load_audio(const std::string& path, int target_sample_rate = 16000);

/**
 * Resample audio to target sample rate.
 */
array resample_audio(const array& audio, int from_rate, int to_rate);

} // namespace zipformer
