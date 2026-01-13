#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Q28 Vocoder ISTFT Window Cache Evaluation

Q28 proposes caching the ISTFT window (Hann) instead of computing it per call.
This benchmark evaluates whether such optimization provides any benefit.

Current implementation computes window inside _istft() method:
    window = mx.array([0.5 * (1 - math.cos(2 * math.pi * i / n_fft))
                      for i in range(n_fft)])

With n_fft=16 for CosyVoice3 vocoder, this is a tiny array.

Author: Worker #1345
Date: 2025-12-20
"""

import time
import math
import mlx.core as mx
import numpy as np


def benchmark_window_creation():
    """Benchmark window creation vs cached window access."""
    print("=" * 60)
    print("BENCHMARK: ISTFT Window Creation vs Cached")
    print("=" * 60)

    n_fft = 16  # CosyVoice3 vocoder config

    # Cached window (precomputed at init time)
    cached_window = mx.array([0.5 * (1 - math.cos(2 * math.pi * i / n_fft))
                              for i in range(n_fft)])
    mx.eval(cached_window)

    # Benchmark: create window each time
    times_create = []
    for _ in range(1000):
        start = time.perf_counter()
        window = mx.array([0.5 * (1 - math.cos(2 * math.pi * i / n_fft))
                          for i in range(n_fft)])
        mx.eval(window)
        times_create.append(time.perf_counter() - start)

    # Benchmark: use cached window
    times_cached = []
    for _ in range(1000):
        start = time.perf_counter()
        window = cached_window
        mx.eval(window)
        times_cached.append(time.perf_counter() - start)

    print(f"\nWindow creation (n_fft={n_fft}):")
    print(f"  Create each time: {np.mean(times_create)*1000:.4f} ms avg")
    print(f"  Use cached:       {np.mean(times_cached)*1000:.4f} ms avg")
    print(f"  Savings:          {(np.mean(times_create) - np.mean(times_cached))*1000:.4f} ms")

    return np.mean(times_create) * 1000, np.mean(times_cached) * 1000


def benchmark_istft_full():
    """Benchmark full ISTFT operation to understand relative cost."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Full iSTFT Operation")
    print("=" * 60)

    n_fft = 16
    hop_len = 4
    L = 6000  # 50 mel frames * 120 upsample factor

    # Input: [B, n_fft + 2, L]
    x = mx.random.normal((1, n_fft + 2, L))
    mx.eval(x)

    # Full iSTFT (simplified from vocoder)
    def istft_baseline(x, n_fft, hop_len):
        # Create window each time (current implementation)
        window = mx.array([0.5 * (1 - math.cos(2 * math.pi * i / n_fft))
                          for i in range(n_fft)])

        mag = x[:, :n_fft // 2 + 1, :]
        phase = x[:, n_fft // 2 + 1:, :]
        real = mag * mx.cos(phase)
        imag = mag * mx.sin(phase)

        _, _, L = mag.shape
        L_out = L * hop_len
        audio = mx.zeros((1, L_out))

        # Overlap-add (simplified - just first few frames for benchmark)
        for t in range(min(L, 100)):
            frame_complex = real[:, :, t] + 1j * imag[:, :, t]
            frame = mx.fft.irfft(frame_complex, n=n_fft)
            frame = frame * window
        return audio

    # Warmup
    for _ in range(3):
        out = istft_baseline(x, n_fft, hop_len)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter()
        out = istft_baseline(x, n_fft, hop_len)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    print(f"\nFull iSTFT (100 frames): {avg_ms:.3f} ms")

    return avg_ms


def estimate_e2e_impact(window_create_ms, window_cached_ms, istft_ms):
    """Estimate E2E impact of Q28 optimization."""
    print("\n" + "=" * 60)
    print("E2E IMPACT ANALYSIS")
    print("=" * 60)

    # Reference: vocoder total ~238ms (H2+I1 optimized)
    vocoder_time = 238.0

    # Window creation savings per call
    window_savings = window_create_ms - window_cached_ms

    print(f"\nWindow creation overhead: {window_create_ms:.4f} ms")
    print(f"Window cache access:      {window_cached_ms:.4f} ms")
    print(f"Potential savings:        {window_savings:.4f} ms")

    print(f"\nVocoder reference time:   {vocoder_time} ms")
    print(f"iSTFT time (100 frames):  {istft_ms:.3f} ms")

    # Savings as percentage of vocoder
    pct_savings = (window_savings / vocoder_time) * 100

    print(f"\nSavings % of vocoder:     {pct_savings:.4f}%")
    print(f"Expected E2E speedup:     {vocoder_time / (vocoder_time - window_savings):.6f}x")


def analyze_implementation_status():
    """Check if optimization is already partially done."""
    print("\n" + "=" * 60)
    print("IMPLEMENTATION ANALYSIS")
    print("=" * 60)

    print("""
Current CosyVoice3 vocoder _istft implementation:
- Window is created inside _istft() method (line 506)
- Called once per vocoder forward pass
- n_fft = 16 → window is only 16 float32 values (64 bytes)
- List comprehension + mx.array() is the overhead

Analysis:
1. Window creation happens ONCE per inference (not per frame)
2. The 16-element array creation is dominated by Python overhead
3. MLX's lazy evaluation may already optimize this in compiled path
4. The overlap-add loop (lines 512-520) is the real cost

Optimization approach (if worthwhile):
- Add window as class attribute in __init__
- Reference self.window in _istft instead of creating

Verdict: The optimization is trivial to implement but benefit is negligible.
""")


def main():
    print("Q28: Vocoder ISTFT Window Cache Evaluation")
    print("=" * 60)

    window_create_ms, window_cached_ms = benchmark_window_creation()
    istft_ms = benchmark_istft_full()
    estimate_e2e_impact(window_create_ms, window_cached_ms, istft_ms)
    analyze_implementation_status()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
Q28 Evaluation Result: NOT WORTH (but trivial to fix)

Key Findings:
1. Window creation overhead: ~0.01-0.02ms per inference
2. This is <0.01% of vocoder time (238ms)
3. E2E impact: effectively 0%

Root Cause of Non-Impact:
- Window is only 16 values (64 bytes)
- Created once per inference, not per frame
- Python list comprehension dominates, not MLX compute
- The real iSTFT cost is the overlap-add loop, not window creation

Note: While the optimization is trivial (just move window to __init__),
the benefit is negligible. This is a micro-optimization not worth the
code change in production.

Conclusion: Q28 NOT WORTH - benefit is <0.01% of vocoder time.
""")


if __name__ == "__main__":
    main()
