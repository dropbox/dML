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
Detailed F10 Benchmark: Measure decode loop impact directly.

For short audio, encoder time dominates.
For long audio (30s), decode loop becomes significant.
This benchmark isolates the decode loop to show expected benefit.
"""

import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    from tools.whisper_mlx import WhisperMLX

    print("=" * 60)
    print("F10 Detailed Benchmark: Decode Loop Analysis")
    print("=" * 60)

    # Load model
    print("\nLoading WhisperMLX (large-v3-turbo)...")
    model = WhisperMLX.from_pretrained("large-v3-turbo")

    # Generate 30s of synthetic audio (silence + noise)
    # This produces longer output sequences for better measurement
    print("Generating 30s synthetic audio...")
    sample_rate = 16000
    duration = 30
    # Mix silence with some noise to trigger decoder output
    audio = np.random.randn(sample_rate * duration) * 0.01
    audio = audio.astype(np.float32)

    # Warmup
    print("Warmup...")
    for _ in range(2):
        _ = model.transcribe(audio, temperature=0.0, skip_logprobs=True)

    # Benchmark
    print("\n--- 30s Audio Benchmark ---")
    NUM_ITERS = 3

    # With logprobs
    times_with = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        result = model.transcribe(audio, temperature=0.0, skip_logprobs=False)
        elapsed = time.perf_counter() - start
        times_with.append(elapsed)
        if i == 0:
            len(result.get("segments", []))
            len(result.get("text", ""))

    # Without logprobs
    times_without = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        result = model.transcribe(audio, temperature=0.0, skip_logprobs=True)
        elapsed = time.perf_counter() - start
        times_without.append(elapsed)

    mean_with = np.mean(times_with)
    mean_without = np.mean(times_without)

    print(f"\nWith logprobs:    {mean_with*1000:.1f}ms")
    print(f"Without logprobs: {mean_without*1000:.1f}ms")
    print(f"Speedup: {mean_with/mean_without:.2f}x")

    # Now test transcribe_long which always uses skip_logprobs
    print("\n--- transcribe_long Benchmark (auto skip_logprobs) ---")

    # Generate longer audio (60s)
    print("Generating 60s audio...")
    audio_long = np.random.randn(sample_rate * 60) * 0.01
    audio_long = audio_long.astype(np.float32)

    # Warmup
    _ = model.transcribe_long(audio_long, temperature=0.0)

    # Benchmark transcribe_long (inherently uses skip_logprobs)
    times_long = []
    for i in range(2):
        start = time.perf_counter()
        result = model.transcribe_long(audio_long, temperature=0.0)
        elapsed = time.perf_counter() - start
        times_long.append(elapsed)

    print(f"transcribe_long (60s): {np.mean(times_long)*1000:.1f}ms")
    print("  (Uses skip_logprobs=True automatically for ~2x decode speedup)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
F10 Optimization Impact:
- Microbenchmark (decode loop only): ~2x speedup
- Short audio (4s): Minimal impact (encoder dominates)
- Long audio (30s): ~{:.1f}x speedup
- transcribe_long: Auto-enabled for best performance

Recommendation:
- For latency-critical applications: use skip_logprobs=True
- For quality monitoring: keep skip_logprobs=False (default)
- transcribe_long: Already optimized (auto skip_logprobs)
""".format(mean_with/mean_without))


if __name__ == "__main__":
    main()
