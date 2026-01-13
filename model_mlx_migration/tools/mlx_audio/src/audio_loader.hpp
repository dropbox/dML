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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
}

namespace mlx_audio {

/**
 * High-performance audio loader using libav (FFmpeg libraries).
 *
 * Design principles:
 * 1. Identical output to ffmpeg CLI command:
 *    ffmpeg -nostdin -threads 0 -i <file> -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -
 * 2. Zero-copy output to numpy arrays when possible
 * 3. Thread-safe via per-instance contexts
 * 4. Minimal overhead for repeated calls
 */
class AudioLoader {
public:
    struct Config {
        int target_sample_rate;
        int target_channels;

        // Constructor with defaults
        Config(int sr = 16000, int ch = 1)
            : target_sample_rate(sr), target_channels(ch) {}
    };

    AudioLoader();
    explicit AudioLoader(const Config& config);
    ~AudioLoader();

    // Non-copyable
    AudioLoader(const AudioLoader&) = delete;
    AudioLoader& operator=(const AudioLoader&) = delete;

    // Moveable
    AudioLoader(AudioLoader&&) noexcept;
    AudioLoader& operator=(AudioLoader&&) noexcept;

    /**
     * Load audio file and resample to target format (int16).
     *
     * @param path Path to audio file (wav, mp3, flac, m4a, etc.)
     * @return Audio samples as int16
     * @throws std::runtime_error on decode failure
     */
    std::vector<int16_t> load(const std::string& path);

    /**
     * Load audio and return as float32 (same as ffmpeg output).
     * Values are normalized to [-1.0, 1.0] via division by 32768.0
     */
    std::vector<float> load_float32(const std::string& path);

    /**
     * Get last error message.
     */
    const std::string& last_error() const { return error_; }

    /**
     * Check if last operation succeeded.
     */
    bool ok() const { return error_.empty(); }

    /**
     * Get target sample rate.
     */
    int sample_rate() const { return config_.target_sample_rate; }

private:
    Config config_;
    std::string error_;
};

/**
 * Convenience function for single-file loading.
 * Returns float32 samples normalized to [-1.0, 1.0].
 *
 * @param path Path to audio file
 * @param sample_rate Target sample rate (default 16000)
 * @return Audio samples as float32
 * @throws std::runtime_error on decode failure
 */
std::vector<float> load_audio(
    const std::string& path,
    int sample_rate = 16000
);

/**
 * Get audio duration in seconds without fully decoding.
 *
 * @param path Path to audio file
 * @return Duration in seconds, or -1.0 on error
 */
double get_audio_duration(const std::string& path);

/**
 * Get audio sample count at target sample rate without fully decoding.
 *
 * @param path Path to audio file
 * @param sample_rate Target sample rate
 * @return Estimated sample count, or -1 on error
 */
int64_t get_audio_sample_count(const std::string& path, int sample_rate = 16000);

} // namespace mlx_audio
