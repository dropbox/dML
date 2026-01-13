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

// SIMD utilities for Kokoro C++ inference
// Optimized for Apple Silicon (ARM NEON)

#pragma once

#include <cstddef>
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

namespace kokoro {
namespace simd {

// Check if NEON is available at compile time
constexpr bool has_neon() {
#if HAS_NEON
    return true;
#else
    return false;
#endif
}

#if HAS_NEON

// Vectorized overlap-add for iSTFT
// Processes 4 samples at a time using NEON
inline void overlap_add_neon(
    float* __restrict audio,
    float* __restrict window_sum,
    const float* __restrict frame_data,
    const float* __restrict window_squared,
    int n_fft,
    int start,
    int audio_size
) {
    int n = 0;

    // Process 4 elements at a time
    for (; n + 4 <= n_fft && start + n + 4 <= audio_size; n += 4) {
        // Load current audio values
        float32x4_t audio_vec = vld1q_f32(audio + start + n);
        float32x4_t wsum_vec = vld1q_f32(window_sum + start + n);

        // Load frame and window data
        float32x4_t frame_vec = vld1q_f32(frame_data + n);
        float32x4_t win_sq_vec = vld1q_f32(window_squared + n);

        // Accumulate
        audio_vec = vaddq_f32(audio_vec, frame_vec);
        wsum_vec = vaddq_f32(wsum_vec, win_sq_vec);

        // Store results
        vst1q_f32(audio + start + n, audio_vec);
        vst1q_f32(window_sum + start + n, wsum_vec);
    }

    // Handle remaining elements
    for (; n < n_fft && start + n < audio_size; ++n) {
        audio[start + n] += frame_data[n];
        window_sum[start + n] += window_squared[n];
    }
}

// Vectorized normalization with threshold check
// Divides audio by window_sum where window_sum > threshold
inline void normalize_neon(
    float* __restrict audio,
    const float* __restrict window_sum,
    int size,
    float threshold = 1e-8f
) {
    float32x4_t thresh_vec = vdupq_n_f32(threshold);
    float32x4_t one_vec = vdupq_n_f32(1.0f);

    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t audio_vec = vld1q_f32(audio + i);
        float32x4_t wsum_vec = vld1q_f32(window_sum + i);

        // Create mask: window_sum > threshold
        uint32x4_t mask = vcgtq_f32(wsum_vec, thresh_vec);

        // Safe division: divide where mask is true, keep original otherwise
        // Replace zeros with 1.0 to avoid division by zero
        float32x4_t safe_wsum = vbslq_f32(mask, wsum_vec, one_vec);
        float32x4_t divided = vdivq_f32(audio_vec, safe_wsum);

        // Select: use divided where mask is true, keep audio otherwise
        audio_vec = vbslq_f32(mask, divided, audio_vec);

        vst1q_f32(audio + i, audio_vec);
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        if (window_sum[i] > threshold) {
            audio[i] /= window_sum[i];
        }
    }
}

// Vectorized Hann window generation
inline void generate_hann_window_neon(float* __restrict window, int n_fft) {
    const float pi = 3.14159265358979323846f;
    const float scale = 2.0f * pi / static_cast<float>(n_fft);

    // For small n_fft (like 20), scalar is fine
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(scale * static_cast<float>(i)));
    }
}

// Vectorized element-wise multiply
inline void elementwise_multiply_neon(
    float* __restrict result,
    const float* __restrict a,
    const float* __restrict b,
    int size
) {
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t result_vec = vmulq_f32(a_vec, b_vec);
        vst1q_f32(result + i, result_vec);
    }

    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

// Vectorized element-wise add
inline void elementwise_add_neon(
    float* __restrict result,
    const float* __restrict a,
    const float* __restrict b,
    int size
) {
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t result_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(result + i, result_vec);
    }

    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

#else  // No NEON

// Scalar fallbacks for non-ARM platforms

inline void overlap_add_neon(
    float* __restrict audio,
    float* __restrict window_sum,
    const float* __restrict frame_data,
    const float* __restrict window_squared,
    int n_fft,
    int start,
    int audio_size
) {
    for (int n = 0; n < n_fft && start + n < audio_size; ++n) {
        audio[start + n] += frame_data[n];
        window_sum[start + n] += window_squared[n];
    }
}

inline void normalize_neon(
    float* __restrict audio,
    const float* __restrict window_sum,
    int size,
    float threshold = 1e-8f
) {
    for (int i = 0; i < size; ++i) {
        if (window_sum[i] > threshold) {
            audio[i] /= window_sum[i];
        }
    }
}

inline void generate_hann_window_neon(float* __restrict window, int n_fft) {
    const float pi = 3.14159265358979323846f;
    const float scale = 2.0f * pi / static_cast<float>(n_fft);

    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(scale * static_cast<float>(i)));
    }
}

inline void elementwise_multiply_neon(
    float* __restrict result,
    const float* __restrict a,
    const float* __restrict b,
    int size
) {
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

inline void elementwise_add_neon(
    float* __restrict result,
    const float* __restrict a,
    const float* __restrict b,
    int size
) {
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

#endif  // HAS_NEON

}  // namespace simd
}  // namespace kokoro
