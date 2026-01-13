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

/**
 * Kokoro Prosody Adjustment Implementation
 *
 * See: prosody_adjust.h for interface documentation
 */

#include "prosody_adjust.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace kokoro {

void apply_prosody_adjustments(
    std::vector<float>& durations,
    std::vector<float>& f0,
    const PhonemeProsody& prosody
) {
    std::vector<float> volume_dummy;
    apply_prosody_adjustments(durations, f0, volume_dummy, prosody);
}

void apply_prosody_adjustments(
    std::vector<float>& durations,
    std::vector<float>& f0,
    std::vector<float>& volume_mult,
    const PhonemeProsody& prosody
) {
    if (prosody.empty()) return;

    size_t n = std::min({durations.size(), f0.size(), prosody.size()});

    // Calculate mean F0 for variance adjustments
    float f0_sum = 0;
    int f0_count = 0;
    for (size_t i = 0; i < n; i++) {
        if (f0[i] > 0) {  // Only voiced frames
            f0_sum += f0[i];
            f0_count++;
        }
    }
    float f0_mean = (f0_count > 0) ? f0_sum / f0_count : 200.0f;

    // Resize volume if needed
    if (!volume_mult.empty() || prosody.has_prosody()) {
        volume_mult.resize(n, 1.0f);
    }

    // Apply adjustments per phoneme
    for (size_t i = 0; i < n; i++) {
        ProsodyType type = prosody.mask[i];
        if (type == ProsodyType::NEUTRAL) continue;

        const auto& adj = get_adjustment(type);

        // Duration adjustment
        durations[i] *= adj.duration_mult;

        // F0 adjustment: pitch shift + variance scaling
        if (f0[i] > 0) {  // Only adjust voiced frames
            // First apply pitch multiplier
            f0[i] *= adj.f0_mult;

            // Then apply variance scaling (expand/contract from mean)
            if (adj.f0_var_mult != 1.0f) {
                float deviation = f0[i] - f0_mean;
                f0[i] = f0_mean + deviation * adj.f0_var_mult;
            }

            // Clamp F0 to reasonable range
            f0[i] = std::clamp(f0[i], 50.0f, 600.0f);
        }

        // Volume adjustment
        if (!volume_mult.empty()) {
            volume_mult[i] = adj.volume_mult;
        }
    }
}

std::vector<float> insert_prosody_breaks(
    const std::vector<float>& audio,
    int sample_rate,
    const std::vector<float>& durations,
    const PhonemeProsody& prosody
) {
    // Check if any breaks to insert
    bool has_breaks = false;
    for (int ms : prosody.break_after_ms) {
        if (ms > 0) {
            has_breaks = true;
            break;
        }
    }
    if (!has_breaks) return audio;

    // Calculate phoneme boundaries in samples
    // Durations are in frames, each frame = hop_length samples
    constexpr int hop_length = 256;

    std::vector<size_t> phoneme_end_samples;
    size_t current_sample = 0;
    for (size_t i = 0; i < durations.size(); i++) {
        current_sample += static_cast<size_t>(durations[i] * hop_length);
        phoneme_end_samples.push_back(current_sample);
    }

    // Calculate total silence to add
    size_t total_silence_samples = 0;
    for (size_t i = 0; i < prosody.break_after_ms.size(); i++) {
        if (prosody.break_after_ms[i] > 0) {
            total_silence_samples += (prosody.break_after_ms[i] * sample_rate) / 1000;
        }
    }

    // Create output with extra space for silence
    std::vector<float> output;
    output.reserve(audio.size() + total_silence_samples);

    // Copy audio with silence insertions
    size_t audio_pos = 0;
    for (size_t i = 0; i < phoneme_end_samples.size() && audio_pos < audio.size(); i++) {
        size_t end_pos = std::min(phoneme_end_samples[i], audio.size());

        // Copy audio up to this phoneme's end
        while (audio_pos < end_pos) {
            output.push_back(audio[audio_pos++]);
        }

        // Insert silence if specified
        if (i < prosody.break_after_ms.size() && prosody.break_after_ms[i] > 0) {
            size_t silence_samples = (prosody.break_after_ms[i] * sample_rate) / 1000;

            // Fade out before silence (10ms)
            size_t fade_samples = std::min(static_cast<size_t>((10 * sample_rate) / 1000),
                                          output.size());
            for (size_t j = 0; j < fade_samples && output.size() > j; j++) {
                size_t idx = output.size() - fade_samples + j;
                float fade = static_cast<float>(fade_samples - j) / fade_samples;
                output[idx] *= fade;
            }

            // Insert silence
            for (size_t j = 0; j < silence_samples; j++) {
                output.push_back(0.0f);
            }
        }
    }

    // Copy remaining audio
    while (audio_pos < audio.size()) {
        output.push_back(audio[audio_pos++]);
    }

    return output;
}

void apply_volume_to_audio(
    std::vector<float>& audio,
    const std::vector<float>& volume_mult,
    const std::vector<float>& durations,
    int hop_length
) {
    if (volume_mult.empty() || durations.empty()) return;

    // Calculate phoneme boundaries in samples
    std::vector<std::pair<size_t, size_t>> phoneme_ranges;
    size_t current = 0;
    for (size_t i = 0; i < durations.size(); i++) {
        size_t start = current;
        size_t len = static_cast<size_t>(durations[i] * hop_length);
        current += len;
        phoneme_ranges.emplace_back(start, current);
    }

    // Apply volume per phoneme
    for (size_t i = 0; i < phoneme_ranges.size() && i < volume_mult.size(); i++) {
        float vol = volume_mult[i];
        if (std::abs(vol - 1.0f) < 0.01f) continue;  // Skip if no change

        auto [start, end] = phoneme_ranges[i];
        end = std::min(end, audio.size());

        for (size_t j = start; j < end; j++) {
            audio[j] *= vol;
        }
    }

    // Normalize to prevent clipping
    float max_abs = 0.0f;
    for (float sample : audio) {
        max_abs = std::max(max_abs, std::abs(sample));
    }
    if (max_abs > 1.0f) {
        float scale = 0.99f / max_abs;
        for (float& sample : audio) {
            sample *= scale;
        }
    }
}

}  // namespace kokoro
