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
 * Kokoro Prosody Types and Adjustment Tables
 *
 * Defines prosody marker types and their corresponding duration/F0 adjustments
 * for Phase A (manual tuning) of prosody annotation support.
 *
 * See: reports/main/PROSODY_DESIGN.md
 */

#include <cstdint>

namespace kokoro {

/**
 * Prosody type enumeration.
 * Values chosen to allow efficient table lookup.
 */
enum class ProsodyType : uint8_t {
    // Base
    NEUTRAL = 0,

    // Emphasis (1-9)
    EMPHASIS = 1,
    STRONG_EMPHASIS = 2,
    REDUCED_EMPHASIS = 3,

    // Rate/Speed (10-19)
    RATE_X_SLOW = 10,
    RATE_SLOW = 11,
    RATE_FAST = 12,
    RATE_X_FAST = 13,

    // Pitch (20-29)
    PITCH_X_LOW = 20,
    PITCH_LOW = 21,
    PITCH_HIGH = 22,
    PITCH_X_HIGH = 23,

    // Volume (30-39)
    VOLUME_X_SOFT = 30,
    VOLUME_SOFT = 31,
    VOLUME_LOUD = 32,
    VOLUME_X_LOUD = 33,
    VOLUME_WHISPER = 34,

    // Emotions (40-59)
    EMOTION_ANGRY = 40,
    EMOTION_SAD = 41,
    EMOTION_EXCITED = 42,
    EMOTION_WORRIED = 43,
    EMOTION_ALARMED = 44,
    EMOTION_CALM = 45,
    EMOTION_EMPATHETIC = 46,
    EMOTION_CONFIDENT = 47,
    EMOTION_FRUSTRATED = 48,
    EMOTION_NERVOUS = 49,
    EMOTION_SURPRISED = 50,
    EMOTION_DISAPPOINTED = 51,

    // Special (60-69)
    QUESTION = 60,
    WHISPER = 61,
    LOUD = 62,

    // Max value for array sizing
    NUM_TYPES = 70
};

/**
 * Prosody adjustment parameters.
 * Used for Phase A manual tuning.
 */
struct ProsodyAdjustment {
    float duration_mult;  // Duration multiplier (1.0 = no change)
    float f0_mult;        // F0 (pitch) multiplier (1.0 = no change)
    float f0_var_mult;    // F0 variance multiplier (for emotions)
    float volume_mult;    // Volume multiplier (1.0 = no change)
};

/**
 * Default adjustment values for each prosody type.
 * These are empirically tuned starting values for Phase A.
 */
constexpr ProsodyAdjustment PROSODY_ADJUSTMENTS[] = {
    // NEUTRAL (0)
    {1.00f, 1.00f, 1.00f, 1.00f},

    // EMPHASIS (1)
    {1.30f, 1.10f, 1.00f, 1.10f},
    // STRONG_EMPHASIS (2)
    {1.50f, 1.20f, 1.10f, 1.20f},
    // REDUCED_EMPHASIS (3)
    {0.85f, 0.95f, 0.90f, 0.90f},

    // Padding 4-9
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},

    // RATE_X_SLOW (10)
    {2.00f, 1.00f, 1.00f, 1.00f},
    // RATE_SLOW (11)
    {1.40f, 1.00f, 1.00f, 1.00f},
    // RATE_FAST (12)
    {0.75f, 1.00f, 1.00f, 1.00f},
    // RATE_X_FAST (13)
    {0.60f, 1.00f, 1.00f, 1.00f},

    // Padding 14-19
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},

    // PITCH_X_LOW (20)
    {1.00f, 0.75f, 1.00f, 1.00f},
    // PITCH_LOW (21)
    {1.00f, 0.90f, 1.00f, 1.00f},
    // PITCH_HIGH (22)
    {1.00f, 1.15f, 1.00f, 1.00f},
    // PITCH_X_HIGH (23)
    {1.00f, 1.30f, 1.00f, 1.00f},

    // Padding 24-29
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},

    // VOLUME_X_SOFT (30)
    {1.00f, 0.95f, 0.90f, 0.40f},
    // VOLUME_SOFT (31)
    {1.00f, 0.98f, 0.95f, 0.70f},
    // VOLUME_LOUD (32)
    {0.95f, 1.05f, 1.05f, 1.40f},
    // VOLUME_X_LOUD (33)
    {0.90f, 1.10f, 1.10f, 1.70f},
    // VOLUME_WHISPER (34)
    {1.20f, 0.80f, 0.60f, 0.30f},

    // Padding 35-39
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},

    // EMOTION_ANGRY (40)
    {0.90f, 1.15f, 1.30f, 1.30f},
    // EMOTION_SAD (41)
    {1.20f, 0.90f, 0.80f, 0.80f},
    // EMOTION_EXCITED (42)
    {0.85f, 1.20f, 1.40f, 1.20f},
    // EMOTION_WORRIED (43)
    {1.10f, 1.05f, 1.20f, 0.95f},
    // EMOTION_ALARMED (44)
    {0.80f, 1.25f, 1.50f, 1.30f},
    // EMOTION_CALM (45)
    {1.20f, 0.95f, 0.70f, 0.90f},
    // EMOTION_EMPATHETIC (46)
    {1.10f, 1.00f, 1.10f, 0.95f},
    // EMOTION_CONFIDENT (47)
    {0.95f, 1.05f, 0.90f, 1.10f},
    // EMOTION_FRUSTRATED (48)
    {0.95f, 1.10f, 1.25f, 1.15f},
    // EMOTION_NERVOUS (49)
    {0.90f, 1.08f, 1.50f, 0.95f},
    // EMOTION_SURPRISED (50)
    {0.85f, 1.25f, 1.60f, 1.10f},
    // EMOTION_DISAPPOINTED (51)
    {1.15f, 0.92f, 0.85f, 0.85f},

    // Padding 52-59
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},

    // QUESTION (60)
    {1.00f, 1.08f, 1.20f, 1.00f},
    // WHISPER (61)
    {1.20f, 0.80f, 0.60f, 0.30f},
    // LOUD (62)
    {0.90f, 1.10f, 1.10f, 1.50f},

    // Padding 63-69
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, 1.00f, 1.00f, 1.00f},
};

/**
 * Get adjustment parameters for a prosody type.
 */
inline const ProsodyAdjustment& get_adjustment(ProsodyType type) {
    auto idx = static_cast<uint8_t>(type);
    if (idx >= static_cast<uint8_t>(ProsodyType::NUM_TYPES)) {
        idx = 0;  // Default to NEUTRAL
    }
    return PROSODY_ADJUSTMENTS[idx];
}

/**
 * Break strength to milliseconds mapping.
 */
inline int break_strength_to_ms(const char* strength) {
    if (!strength) return 500;

    if (strcmp(strength, "none") == 0) return 0;
    if (strcmp(strength, "x-weak") == 0) return 100;
    if (strcmp(strength, "weak") == 0) return 250;
    if (strcmp(strength, "medium") == 0) return 500;
    if (strcmp(strength, "strong") == 0) return 750;
    if (strcmp(strength, "x-strong") == 0) return 1000;

    return 500;  // Default
}

}  // namespace kokoro
