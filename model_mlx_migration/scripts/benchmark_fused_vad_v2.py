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
Benchmark FFT computation overhead - v2 with warm-up.

More accurate measurement of actual runtime after initialization.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools/pytorch_to_mlx/converters/models"))

import mlx.core as mx


def benchmark_with_warmup(func, audio, warmup=5, iterations=50):
    """Benchmark with warm-up iterations."""
    # Warm-up
    for _ in range(warmup):
        result = func(audio)
        if isinstance(result, tuple):
            for r in result:
                mx.eval(r)
        else:
            mx.eval(result)
    mx.synchronize()

    # Actual timing
    times = []
    for _ in range(iterations):
        mx.synchronize()
        start = time.perf_counter()
        result = func(audio)
        if isinstance(result, tuple):
            for r in result:
                mx.eval(r)
        else:
            mx.eval(result)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "p50_ms": np.percentile(times, 50) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
    }


def main():
    from tools.whisper_mlx.audio import log_mel_spectrogram, compute_stft_and_mel

    print("=" * 70)
    print("Phase 1.2: Fused VAD-Whisper FFT Benchmark (with warm-up)")
    print("=" * 70)

    # Test durations
    durations = [5, 15, 30]

    for duration in durations:
        sample_rate = 16000
        n_samples = int(duration * sample_rate)

        # Generate test audio
        np.random.seed(42)
        audio = np.random.randn(n_samples).astype(np.float32) * 0.1

        print(f"\n--- {duration}s audio ({n_samples} samples) ---")

        # 1. Whisper mel (using mlx-whisper)
        def whisper_mel(a):
            return log_mel_spectrogram(a, n_mels=128)

        mel_result = benchmark_with_warmup(whisper_mel, audio)
        print(f"1. Whisper mel (mlx-whisper): {mel_result['mean_ms']:.3f}ms (p50: {mel_result['p50_ms']:.3f}ms)")

        # 2. Fused mel + STFT (single FFT, fallback implementation)
        def fused_mel_stft(a):
            return compute_stft_and_mel(a, return_stft=True)

        fused_result = benchmark_with_warmup(fused_mel_stft, audio)
        print(f"2. Fused mel+STFT (fallback): {fused_result['mean_ms']:.3f}ms (p50: {fused_result['p50_ms']:.3f}ms)")

        # 3. Silero STFT
        try:
            from silero_vad_mlx import STFT
            stft = STFT(filter_length=256, hop_length=128)

            def silero_stft(a):
                return stft(mx.array(a[None, :]))

            silero_result = benchmark_with_warmup(silero_stft, audio)
            print(f"3. Silero STFT:              {silero_result['mean_ms']:.3f}ms (p50: {silero_result['p50_ms']:.3f}ms)")

            # Analysis
            separate = mel_result['mean_ms'] + silero_result['mean_ms']
            fused_result['mean_ms'] + silero_result['mean_ms']  # Still need Silero STFT for VAD
            print(f"\n   Separate (mel + Silero STFT): {separate:.3f}ms")
            print("   Note: Can't directly share FFT due to different params")

        except ImportError as e:
            print(f"3. Silero STFT: Error - {e}")

    # Conclusion
    print("\n" + "=" * 70)
    print("Key Finding")
    print("=" * 70)
    print("""
The fused mel+STFT computation (compute_stft_and_mel) is available and works,
but Silero VAD's trained weights expect n_fft=256, not Whisper's n_fft=400.

Options:
1. Train FusedVAD model with Whisper's FFT params (~20 commits, needs data)
2. Create simple mel-energy-based VAD (no training, ~5 commits)
3. Keep separate FFT - overhead is only ~1-2ms for 30s audio

Given mel computation is <10ms for 30s audio, and total transcription is
1000-2000ms, saving a few ms has minimal impact (<1% speedup).

Recommendation: Phase 1.2's expected 1.1x speedup is not achievable through
FFT sharing alone. Consider pivoting to higher-ROI optimizations.
""")


if __name__ == "__main__":
    main()
