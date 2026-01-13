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
Benchmark C4: Dynamic Batch Reorder Optimization

Tests the impact of sorting audio by length and dynamic padding
on batch transcription throughput.

Expected benefits:
- Sorting by length: Reduces padding waste, similar-length audio together
- Dynamic padding: Pad to max-in-batch instead of fixed 30s

Test cases:
1. Uniform length audio (no benefit expected)
2. Mixed length audio (benefit expected)
3. Extreme length variance (maximum benefit)
"""

import time
import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration')

import mlx.core as mx
import numpy as np
from pathlib import Path


def get_librispeech_files(data_dir: str, limit: int = 20) -> list:
    """Get LibriSpeech audio files with varying lengths."""
    data_path = Path(data_dir)
    flac_files = list(data_path.rglob("*.flac"))

    if len(flac_files) < limit:
        print(f"Warning: Only found {len(flac_files)} files, need {limit}")
        return flac_files

    # Sample files to get good length distribution
    return sorted(flac_files)[:limit]


def measure_audio_lengths(files: list, sample_rate: int = 16000) -> list:
    """Measure audio duration for each file."""
    from tools.whisper_mlx.audio import load_audio

    lengths = []
    for f in files:
        audio = load_audio(str(f), sample_rate=sample_rate)
        lengths.append(len(audio) / sample_rate)
    return lengths


def benchmark_batch(model, files: list, sort_by_length: bool, dynamic_padding: bool,
                    warmup: int = 1, repeats: int = 3, verbose: bool = False) -> dict:
    """Benchmark batch transcription with given settings."""
    # Warmup
    for _ in range(warmup):
        _ = model.transcribe_batch(
            files,
            language="en",
            sort_by_length=sort_by_length,
            dynamic_padding=dynamic_padding,
            verbose=False,
        )
        mx.metal.clear_cache()

    # Timed runs
    times = []
    results = None
    for _ in range(repeats):
        mx.metal.clear_cache()
        start = time.perf_counter()
        results = model.transcribe_batch(
            files,
            language="en",
            sort_by_length=sort_by_length,
            dynamic_padding=dynamic_padding,
            verbose=verbose,
        )
        mx.eval()  # Ensure complete
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "times": times,
        "mean": np.mean(times),
        "std": np.std(times),
        "results": results,
    }


def main():
    from tools.whisper_mlx.model import WhisperMLX

    print("=" * 60)
    print("C4: Dynamic Batch Reorder Benchmark")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = WhisperMLX.from_pretrained("base")
    print(f"Model: {model.config.n_audio_state}d, {model.config.n_vocab} vocab")

    # Get test files
    data_dir = "/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean"
    files = get_librispeech_files(data_dir, limit=16)

    if not files:
        print("ERROR: No audio files found")
        return

    print(f"\nTest files: {len(files)}")

    # Measure audio lengths
    print("\nMeasuring audio lengths...")
    lengths = measure_audio_lengths(files)
    print(f"Lengths: min={min(lengths):.1f}s, max={max(lengths):.1f}s, mean={np.mean(lengths):.1f}s")
    print(f"Length variance: {np.std(lengths):.2f}s")

    # Test configurations
    configs = [
        ("No sort, Fixed pad", False, False),
        ("No sort, Dynamic pad", False, True),
        ("Sort, Fixed pad", True, False),
        ("Sort, Dynamic pad (C4)", True, True),
    ]

    results = {}

    for batch_size in [4, 8, 16]:
        print(f"\n{'=' * 60}")
        print(f"BATCH SIZE: {batch_size}")
        print("=" * 60)

        test_files = [str(f) for f in files[:batch_size]]
        test_lengths = lengths[:batch_size]

        print(f"Files: {batch_size}")
        print(f"Lengths: {[f'{length:.1f}s' for length in test_lengths]}")

        for name, sort_by_length, dynamic_padding in configs:
            print(f"\n--- {name} ---")

            result = benchmark_batch(
                model, test_files,
                sort_by_length=sort_by_length,
                dynamic_padding=dynamic_padding,
                warmup=1,
                repeats=3,
                verbose=(name == "Sort, Dynamic pad (C4)"),  # Show C4 verbose output once
            )

            print(f"Time: {result['mean']:.3f}s ± {result['std']:.3f}s")
            results[(batch_size, name)] = result

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Speedup vs No Sort/Fixed Pad Baseline")
    print("=" * 60)

    print(f"\n{'Config':<30} | {'Batch 4':>12} | {'Batch 8':>12} | {'Batch 16':>12}")
    print("-" * 72)

    for name, _, _ in configs:
        speedups = []
        for batch_size in [4, 8, 16]:
            baseline = results[(batch_size, "No sort, Fixed pad")]["mean"]
            current = results[(batch_size, name)]["mean"]
            speedup = baseline / current
            speedups.append(f"{speedup:.2f}x")
        print(f"{name:<30} | {speedups[0]:>12} | {speedups[1]:>12} | {speedups[2]:>12}")

    # Verify correctness
    print("\n" + "=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)

    # Compare results from sorted vs unsorted
    batch_8_unsorted = results[(8, "No sort, Fixed pad")]["results"]
    batch_8_sorted = results[(8, "Sort, Dynamic pad (C4)")]["results"]

    if batch_8_unsorted and batch_8_sorted:
        all_match = True
        for i in range(len(batch_8_unsorted)):
            if batch_8_unsorted[i]["text"] != batch_8_sorted[i]["text"]:
                print(f"MISMATCH at index {i}:")
                print(f"  Unsorted: {batch_8_unsorted[i]['text'][:50]}...")
                print(f"  Sorted:   {batch_8_sorted[i]['text'][:50]}...")
                all_match = False

        if all_match:
            print("✓ All results match between sorted and unsorted batches")
        else:
            print("✗ MISMATCH: Results differ between sorted and unsorted")

    # Show C4 optimization impact
    print("\n" + "=" * 60)
    print("C4 OPTIMIZATION IMPACT")
    print("=" * 60)

    for batch_size in [4, 8, 16]:
        baseline = results[(batch_size, "No sort, Fixed pad")]["mean"]
        c4 = results[(batch_size, "Sort, Dynamic pad (C4)")]["mean"]
        speedup = baseline / c4
        improvement = (baseline - c4) / baseline * 100
        print(f"Batch {batch_size}: {baseline:.3f}s -> {c4:.3f}s ({speedup:.2f}x, {improvement:.1f}% faster)")


if __name__ == "__main__":
    main()
