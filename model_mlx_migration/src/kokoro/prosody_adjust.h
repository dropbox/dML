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

/**
 * Kokoro Prosody Adjustment
 *
 * Phase A post-prediction adjustments for duration and F0.
 * Applies prosody masks to model predictions before audio generation.
 *
 * See: reports/main/PROSODY_DESIGN.md
 */

#include <vector>
#include "prosody_types.h"
#include "prosody_parser.h"

namespace kokoro {

/**
 * Apply prosody adjustments to predicted durations and F0.
 *
 * This is the Phase A approach: post-prediction multipliers.
 * Applied after Kokoro's duration/F0 predictors but before audio synthesis.
 *
 * @param durations Predicted durations per phoneme (modified in place)
 * @param f0 Predicted F0 values (modified in place)
 * @param prosody Per-phoneme prosody information
 */
void apply_prosody_adjustments(
    std::vector<float>& durations,
    std::vector<float>& f0,
    const PhonemeProsody& prosody
);

/**
 * Apply prosody adjustments with separate volume output.
 *
 * @param durations Predicted durations per phoneme (modified in place)
 * @param f0 Predicted F0 values (modified in place)
 * @param volume_mult Output: per-sample volume multipliers
 * @param prosody Per-phoneme prosody information
 */
void apply_prosody_adjustments(
    std::vector<float>& durations,
    std::vector<float>& f0,
    std::vector<float>& volume_mult,
    const PhonemeProsody& prosody
);

/**
 * Insert silence breaks into audio based on prosody breaks.
 *
 * @param audio Input audio samples
 * @param sample_rate Audio sample rate (e.g., 24000)
 * @param durations Per-phoneme durations (used to find phoneme boundaries)
 * @param prosody Prosody information with break_after_ms
 * @return New audio with silence inserted
 */
std::vector<float> insert_prosody_breaks(
    const std::vector<float>& audio,
    int sample_rate,
    const std::vector<float>& durations,
    const PhonemeProsody& prosody
);

/**
 * Apply volume multipliers to audio.
 *
 * @param audio Audio samples (modified in place)
 * @param volume_mult Per-sample volume multipliers
 * @param durations Per-phoneme durations (for sample mapping)
 * @param hop_length Samples per frame (typically 256)
 */
void apply_volume_to_audio(
    std::vector<float>& audio,
    const std::vector<float>& volume_mult,
    const std::vector<float>& durations,
    int hop_length = 256
);

}  // namespace kokoro
