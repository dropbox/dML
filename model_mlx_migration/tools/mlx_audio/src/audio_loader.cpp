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

#include "audio_loader.hpp"
#include <stdexcept>
#include <cmath>

namespace mlx_audio {

AudioLoader::AudioLoader()
    : config_()
{
}

AudioLoader::AudioLoader(const Config& config)
    : config_(config)
{
}

AudioLoader::~AudioLoader() = default;

AudioLoader::AudioLoader(AudioLoader&&) noexcept = default;
AudioLoader& AudioLoader::operator=(AudioLoader&&) noexcept = default;

std::vector<float> AudioLoader::load_float32(const std::string& path) {
    auto samples = load(path);

    // Convert to float32 (same as ffmpeg: divide by 32768.0)
    std::vector<float> result(samples.size());
    constexpr float scale = 1.0f / 32768.0f;

    for (size_t i = 0; i < samples.size(); ++i) {
        result[i] = static_cast<float>(samples[i]) * scale;
    }

    return result;
}

std::vector<int16_t> AudioLoader::load(const std::string& path) {
    error_.clear();
    std::vector<int16_t> output;

    // Open input file
    AVFormatContext* format_ctx = nullptr;
    if (avformat_open_input(&format_ctx, path.c_str(), nullptr, nullptr) < 0) {
        error_ = "Failed to open file: " + path;
        return output;
    }

    // Find stream info
    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        error_ = "Failed to find stream info";
        avformat_close_input(&format_ctx);
        return output;
    }

    // Find audio stream
    int audio_stream_idx = -1;
    const AVCodec* codec = nullptr;
    for (unsigned i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = static_cast<int>(i);
            codec = avcodec_find_decoder(
                format_ctx->streams[i]->codecpar->codec_id
            );
            break;
        }
    }

    if (audio_stream_idx < 0 || !codec) {
        error_ = "No audio stream found in: " + path;
        avformat_close_input(&format_ctx);
        return output;
    }

    // Create codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        error_ = "Failed to allocate codec context";
        avformat_close_input(&format_ctx);
        return output;
    }

    AVCodecParameters* codecpar = format_ctx->streams[audio_stream_idx]->codecpar;
    if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
        error_ = "Failed to copy codec parameters";
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return output;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        error_ = "Failed to open codec";
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return output;
    }

    // Create resampler
    // Match ffmpeg CLI: -ac 1 -ar 16000 -f s16le
    SwrContext* swr_ctx = nullptr;

    AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;
    AVChannelLayout in_layout = {};  // Zero-initialize (critical for FFmpeg 6.0+)

    // Copy input channel layout
    if (av_channel_layout_copy(&in_layout, &codec_ctx->ch_layout) < 0) {
        // Fallback to default stereo if unknown
        av_channel_layout_uninit(&in_layout);  // Clean up any partial state
        in_layout = AV_CHANNEL_LAYOUT_STEREO;
    }

    int ret = swr_alloc_set_opts2(
        &swr_ctx,
        &out_layout,                    // out channel layout (mono)
        AV_SAMPLE_FMT_S16,              // out format (s16)
        config_.target_sample_rate,     // out sample rate (16000)
        &in_layout,                     // in channel layout
        codec_ctx->sample_fmt,          // in format
        codec_ctx->sample_rate,         // in sample rate
        0, nullptr
    );

    if (ret < 0 || !swr_ctx) {
        error_ = "Failed to allocate resampler";
        av_channel_layout_uninit(&in_layout);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return output;
    }

    if (swr_init(swr_ctx) < 0) {
        error_ = "Failed to initialize resampler";
        swr_free(&swr_ctx);
        av_channel_layout_uninit(&in_layout);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return output;
    }

    // Pre-estimate output size based on duration
    int64_t duration_samples = 0;
    if (format_ctx->duration > 0) {
        duration_samples = format_ctx->duration * config_.target_sample_rate
                          / AV_TIME_BASE;
    }
    output.reserve(duration_samples > 0 ? duration_samples : 480000);

    // Allocate packet and frame
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    if (!packet || !frame) {
        error_ = "Failed to allocate packet/frame";
        if (packet) av_packet_free(&packet);
        if (frame) av_frame_free(&frame);
        swr_free(&swr_ctx);
        av_channel_layout_uninit(&in_layout);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return output;
    }

    // Decode and resample
    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == audio_stream_idx) {
            if (avcodec_send_packet(codec_ctx, packet) >= 0) {
                while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
                    // Calculate output samples for this frame
                    int out_samples = swr_get_out_samples(
                        swr_ctx, frame->nb_samples
                    );

                    if (out_samples > 0) {
                        // Resize output buffer
                        size_t old_size = output.size();
                        output.resize(old_size + out_samples);

                        // Resample directly into output buffer
                        uint8_t* out_buf = reinterpret_cast<uint8_t*>(
                            output.data() + old_size
                        );
                        int converted = swr_convert(
                            swr_ctx,
                            &out_buf,
                            out_samples,
                            const_cast<const uint8_t**>(frame->extended_data),
                            frame->nb_samples
                        );

                        // Adjust size if fewer samples were converted
                        if (converted >= 0) {
                            output.resize(old_size + converted);
                        } else {
                            output.resize(old_size);
                        }
                    }
                }
            }
        }
        av_packet_unref(packet);
    }

    // Flush decoder
    avcodec_send_packet(codec_ctx, nullptr);
    while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
        int out_samples = swr_get_out_samples(swr_ctx, frame->nb_samples);
        if (out_samples > 0) {
            size_t old_size = output.size();
            output.resize(old_size + out_samples);
            uint8_t* out_buf = reinterpret_cast<uint8_t*>(
                output.data() + old_size
            );
            int converted = swr_convert(
                swr_ctx,
                &out_buf,
                out_samples,
                const_cast<const uint8_t**>(frame->extended_data),
                frame->nb_samples
            );
            if (converted >= 0) {
                output.resize(old_size + converted);
            } else {
                output.resize(old_size);
            }
        }
    }

    // Flush resampler (get remaining samples)
    int remaining = swr_get_out_samples(swr_ctx, 0);
    if (remaining > 0) {
        size_t old_size = output.size();
        output.resize(old_size + remaining);
        uint8_t* out_buf = reinterpret_cast<uint8_t*>(
            output.data() + old_size
        );
        int flushed = swr_convert(swr_ctx, &out_buf, remaining, nullptr, 0);
        if (flushed >= 0) {
            output.resize(old_size + flushed);
        } else {
            output.resize(old_size);
        }
    }

    // Cleanup
    av_frame_free(&frame);
    av_packet_free(&packet);
    swr_free(&swr_ctx);
    av_channel_layout_uninit(&in_layout);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);

    return output;
}

// Convenience function
std::vector<float> load_audio(const std::string& path, int sample_rate) {
    AudioLoader loader(AudioLoader::Config{sample_rate, 1});
    auto result = loader.load_float32(path);
    if (!loader.ok()) {
        throw std::runtime_error(loader.last_error());
    }
    return result;
}

double get_audio_duration(const std::string& path) {
    AVFormatContext* format_ctx = nullptr;
    if (avformat_open_input(&format_ctx, path.c_str(), nullptr, nullptr) < 0) {
        return -1.0;
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        avformat_close_input(&format_ctx);
        return -1.0;
    }

    double duration = -1.0;
    if (format_ctx->duration > 0) {
        duration = static_cast<double>(format_ctx->duration) / AV_TIME_BASE;
    }

    avformat_close_input(&format_ctx);
    return duration;
}

int64_t get_audio_sample_count(const std::string& path, int sample_rate) {
    double duration = get_audio_duration(path);
    if (duration < 0) {
        return -1;
    }
    return static_cast<int64_t>(duration * sample_rate);
}

} // namespace mlx_audio
