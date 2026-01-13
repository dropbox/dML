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
Benchmark: F10 Skip Logprobs optimization on real transcription.

Tests the impact of skip_logprobs=True on actual WhisperMLX transcription.
"""

import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    from tools.whisper_mlx import WhisperMLX

    print("=" * 60)
    print("F10 Benchmark: Skip Logprobs on Real Transcription")
    print("=" * 60)

    # Use a sample audio file
    audio_samples = [
        "test_data/jfk.wav",
        "test_data/librispeech_1.wav",
        "tests/prosody/contour_vs_v5_output/neutral_context_baseline.wav",
    ]

    # Find first existing audio file
    audio_file = None
    for f in audio_samples:
        if os.path.exists(f):
            audio_file = f
            break

    if not audio_file:
        # Generate synthetic audio for benchmark
        print("No test audio found, generating synthetic audio...")
        import mlx.core as mx
        sample_rate = 16000
        duration = 10  # 10 seconds
        audio = mx.random.normal((sample_rate * duration,))
        audio = np.array(audio)
    else:
        print(f"Using: {audio_file}")
        audio = audio_file

    # Load model (use large-v3-turbo for faster testing)
    print("\nLoading WhisperMLX (large-v3-turbo)...")
    model = WhisperMLX.from_pretrained("large-v3-turbo")
    print("Model loaded, device: Metal")

    # Warmup
    print("\nWarmup...")
    for _ in range(2):
        _ = model.transcribe(audio, temperature=0.0, skip_logprobs=False)
        _ = model.transcribe(audio, temperature=0.0, skip_logprobs=True)

    # Benchmark parameters
    NUM_ITERS = 5

    # Benchmark with logprobs (default)
    print(f"\nBenchmarking WITH logprobs ({NUM_ITERS} iterations)...")
    times_with_logprobs = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        result = model.transcribe(audio, temperature=0.0, skip_logprobs=False)
        elapsed = time.perf_counter() - start
        times_with_logprobs.append(elapsed)
        if i == 0:
            text_with = result["text"]
            avg_logprob_with = result.get("avg_logprob", float("nan"))

    # Benchmark without logprobs (F10 optimization)
    print(f"Benchmarking WITHOUT logprobs ({NUM_ITERS} iterations)...")
    times_without_logprobs = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        result = model.transcribe(audio, temperature=0.0, skip_logprobs=True)
        elapsed = time.perf_counter() - start
        times_without_logprobs.append(elapsed)
        if i == 0:
            text_without = result["text"]
            avg_logprob_without = result.get("avg_logprob", float("nan"))

    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    mean_with = np.mean(times_with_logprobs)
    std_with = np.std(times_with_logprobs)
    mean_without = np.mean(times_without_logprobs)
    std_without = np.std(times_without_logprobs)

    print(f"\nWith logprobs:    {mean_with*1000:.1f}ms +/- {std_with*1000:.1f}ms")
    print(f"Without logprobs: {mean_without*1000:.1f}ms +/- {std_without*1000:.1f}ms")
    print(f"\nSpeedup: {mean_with/mean_without:.2f}x ({(1 - mean_without/mean_with)*100:.1f}% reduction)")

    print(f"\navg_logprob with: {avg_logprob_with:.4f}")
    print(f"avg_logprob without: {avg_logprob_without}")

    # Verify text is the same
    print("\n--- Text Verification ---")
    if text_with == text_without:
        print("PASS: Transcription text is IDENTICAL")
    else:
        print("WARN: Transcription text differs")
        print(f"  With:    {text_with[:100]}...")
        print(f"  Without: {text_without[:100]}...")

    # Conclusion
    print("\n" + "=" * 60)
    speedup = mean_with / mean_without
    if speedup > 1.1:
        print(f"F10 RESULT: {speedup:.2f}x speedup - RECOMMENDED")
    elif speedup > 1.02:
        print(f"F10 RESULT: {speedup:.2f}x speedup - Modest improvement")
    else:
        print(f"F10 RESULT: {speedup:.2f}x - Minimal impact")


if __name__ == "__main__":
    main()
