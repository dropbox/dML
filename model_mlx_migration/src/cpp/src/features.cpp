// Copyright 2024-2025 Andrew Yates
// Feature extraction implementation for Zipformer MLX C++
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/features.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace zipformer {

FbankExtractor::FbankExtractor(const FbankConfig& config)
    : config_(config)
{
    // Compute derived parameters
    frame_length_samples_ = static_cast<int>(config.sample_rate * config.frame_length_ms / 1000.0f);
    frame_shift_samples_ = static_cast<int>(config.sample_rate * config.frame_shift_ms / 1000.0f);

    // FFT size: next power of 2 >= frame_length
    fft_size_ = 1;
    while (fft_size_ < frame_length_samples_) {
        fft_size_ *= 2;
    }

    init_window();
    init_filterbank();
}

void FbankExtractor::init_window() {
    // Povey window: power(0.5) of Hann window
    // Equivalent to sqrt(0.5 - 0.5 * cos(2*pi*n/(N-1)))
    std::vector<float> win(frame_length_samples_);
    for (int i = 0; i < frame_length_samples_; ++i) {
        float hann = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / (frame_length_samples_ - 1));
        win[i] = std::sqrt(hann);
    }
    window_ = array(win.data(), {frame_length_samples_});
}

void FbankExtractor::init_filterbank() {
    // Create mel filterbank matrix
    int num_fft_bins = fft_size_ / 2 + 1;
    float nyquist = config_.sample_rate / 2.0f;
    float high_freq = config_.high_freq;
    if (high_freq < 0) {
        high_freq = nyquist + high_freq;
    }

    // Mel scale conversion
    auto hz_to_mel = [](float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    };
    auto mel_to_hz = [](float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };

    float mel_low = hz_to_mel(config_.low_freq);
    float mel_high = hz_to_mel(high_freq);

    // Mel center frequencies
    std::vector<float> mel_points(config_.num_mel_bins + 2);
    for (int i = 0; i < config_.num_mel_bins + 2; ++i) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (config_.num_mel_bins + 1);
    }

    // Convert to Hz and then to FFT bin indices
    std::vector<int> bin_indices(config_.num_mel_bins + 2);
    for (int i = 0; i < config_.num_mel_bins + 2; ++i) {
        float hz = mel_to_hz(mel_points[i]);
        bin_indices[i] = static_cast<int>(std::floor((fft_size_ + 1) * hz / config_.sample_rate));
    }

    // Create filterbank matrix
    std::vector<float> fb_data(config_.num_mel_bins * num_fft_bins, 0.0f);
    for (int m = 0; m < config_.num_mel_bins; ++m) {
        int left = bin_indices[m];
        int center = bin_indices[m + 1];
        int right = bin_indices[m + 2];

        // Rising slope
        for (int k = left; k < center && k < num_fft_bins; ++k) {
            if (center > left) {
                fb_data[m * num_fft_bins + k] = static_cast<float>(k - left) / (center - left);
            }
        }

        // Falling slope
        for (int k = center; k < right && k < num_fft_bins; ++k) {
            if (right > center) {
                fb_data[m * num_fft_bins + k] = static_cast<float>(right - k) / (right - center);
            }
        }
    }

    mel_filterbank_ = array(fb_data.data(), {config_.num_mel_bins, num_fft_bins});
}

int FbankExtractor::get_num_frames(int num_samples) const {
    if (num_samples < frame_length_samples_) {
        return 0;
    }
    return (num_samples - frame_length_samples_) / frame_shift_samples_ + 1;
}

int FbankExtractor::get_num_samples(int num_frames) const {
    return (num_frames - 1) * frame_shift_samples_ + frame_length_samples_;
}

array FbankExtractor::extract(const array& waveform) const {
    // waveform: (samples,) or (batch, samples)
    bool batched = waveform.ndim() == 2;
    array wave = batched ? waveform : expand_dims(waveform, 0);

    int batch = wave.shape()[0];
    int samples = wave.shape()[1];
    int num_frames = get_num_frames(samples);

    if (num_frames == 0) {
        return zeros({batch, 0, config_.num_mel_bins});
    }

    // Frame the signal
    std::vector<array> frames_list;
    for (int i = 0; i < num_frames; ++i) {
        int start = i * frame_shift_samples_;
        int end = start + frame_length_samples_;
        array frame = slice(wave, {0, start}, {batch, end});
        frames_list.push_back(frame);
    }

    // Stack frames: (batch, num_frames, frame_length)
    array frames = stack(frames_list, 1);

    // Apply window
    frames = multiply(frames, window_);

    // Pad to FFT size
    if (fft_size_ > frame_length_samples_) {
        int pad_amount = fft_size_ - frame_length_samples_;
        frames = pad(frames, {{0, 0}, {0, 0}, {0, pad_amount}});
    }

    // FFT - using MLX fft
    array spectrum = fft::rfft(frames);

    // Power spectrum
    array power = add(multiply(real(spectrum), real(spectrum)),
                      multiply(imag(spectrum), imag(spectrum)));

    // Apply mel filterbank
    // power: (batch, num_frames, fft_size/2+1)
    // filterbank: (num_mel_bins, fft_size/2+1)
    // result: (batch, num_frames, num_mel_bins)
    array mel_spec = matmul(power, transpose(mel_filterbank_));

    // Log compression
    if (config_.use_log_fbank) {
        mel_spec = log(add(mel_spec, array(1e-10f)));
    }

    if (!batched) {
        mel_spec = squeeze(mel_spec, 0);
    }

    return mel_spec;
}

array FbankExtractor::extract_streaming(const array& chunk, array& state) const {
    // Concatenate with buffered samples
    array combined = chunk;  // Initialize with chunk
    if (state.size() > 0) {
        combined = concatenate({state, chunk}, 0);
    }

    int samples = combined.shape()[0];
    int num_frames = get_num_frames(samples);

    if (num_frames == 0) {
        state = combined;
        return zeros({0, config_.num_mel_bins});
    }

    // Extract features
    array features = extract(combined);

    // Update state with remaining samples
    int processed = num_frames * frame_shift_samples_;
    if (processed < samples) {
        state = slice(combined, {processed}, {samples});
    } else {
        state = zeros({0});  // Empty state
    }

    return features;
}

// Audio loading placeholder - would need external library
array load_audio(const std::string& path, int target_sample_rate) {
    // This is a placeholder - actual implementation would use
    // libsndfile, dr_wav, or similar library

    // For now, return empty array
    throw std::runtime_error("Audio loading not yet implemented in C++. "
                            "Use Python to convert audio to numpy format first.");
}

array resample_audio(const array& audio, int from_rate, int to_rate) {
    if (from_rate == to_rate) {
        return audio;
    }

    // Simple linear interpolation resampling
    // For production, use proper sinc interpolation
    float ratio = static_cast<float>(to_rate) / from_rate;
    int new_length = static_cast<int>(audio.shape()[0] * ratio);

    array indices = linspace(0.0f, static_cast<float>(audio.shape()[0] - 1),
                            new_length);

    // Linear interpolation (simplified)
    array idx_floor = floor(indices);
    array idx_ceil = minimum(add(idx_floor, array(1.0f)),
                            array(static_cast<float>(audio.shape()[0] - 1)));
    array frac = subtract(indices, idx_floor);

    array val_floor = take(audio, astype(idx_floor, int32), 0);
    array val_ceil = take(audio, astype(idx_ceil, int32), 0);

    return add(multiply(val_floor, subtract(array(1.0f), frac)),
               multiply(val_ceil, frac));
}

} // namespace zipformer
