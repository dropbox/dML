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
Benchmark streaming vs non-streaming Kokoro synthesis.

Measures:
1. Time-to-first-audio (TTFA): Time until first audio samples are available
2. Total time: Time to generate complete audio
3. Audio samples: Total output samples

Usage:
    python scripts/benchmark_streaming.py [--model-dir MODEL_DIR] [--warmup N] [--runs N]
"""

import argparse
import time

import mlx.core as mx


def benchmark_non_streaming(model, input_ids, voice, warmup: int = 2, runs: int = 5):
    """Benchmark non-streaming synthesis."""
    # Warmup
    for _ in range(warmup):
        _ = model(input_ids, voice, validate_output=False)
        mx.eval(_)

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_total_ms": sum(times) / len(times) * 1000,
        "min_total_ms": min(times) * 1000,
        "max_total_ms": max(times) * 1000,
        "samples": audio.shape[1],
        # TTFA for non-streaming is same as total time (no early output)
        "ttfa_ms": sum(times) / len(times) * 1000,
    }


def benchmark_streaming(
    model, input_ids, voice, chunk_frames: int, overlap_frames: int,
    warmup: int = 2, runs: int = 5
):
    """Benchmark streaming synthesis."""
    # Warmup
    for _ in range(warmup):
        for chunk in model.synthesize_streaming(
            input_ids, voice, chunk_frames=chunk_frames, overlap_frames=overlap_frames
        ):
            mx.eval(chunk)

    # Benchmark
    ttfa_times = []
    total_times = []
    chunk_counts = []
    total_samples = []

    for _ in range(runs):
        start = time.perf_counter()
        first_chunk_time = None
        chunks = []
        num_chunks = 0

        for chunk in model.synthesize_streaming(
            input_ids, voice, chunk_frames=chunk_frames, overlap_frames=overlap_frames
        ):
            mx.eval(chunk)
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            chunks.append(chunk)
            num_chunks += 1

        end = time.perf_counter()

        ttfa_times.append(first_chunk_time - start)
        total_times.append(end - start)
        chunk_counts.append(num_chunks)
        total_samples.append(sum(c.shape[1] for c in chunks))

    return {
        "mean_ttfa_ms": sum(ttfa_times) / len(ttfa_times) * 1000,
        "min_ttfa_ms": min(ttfa_times) * 1000,
        "max_ttfa_ms": max(ttfa_times) * 1000,
        "mean_total_ms": sum(total_times) / len(total_times) * 1000,
        "min_total_ms": min(total_times) * 1000,
        "max_total_ms": max(total_times) * 1000,
        "chunks": chunk_counts[0],
        "samples": total_samples[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Kokoro streaming synthesis")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to model directory (default: use random weights)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark runs")
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 64, 128, 256],
                        help="Token counts to benchmark")
    args = parser.parse_args()

    from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

    # Load model
    if args.model_dir:
        print(f"Loading model from {args.model_dir}")
        model = KokoroModel.from_pretrained(args.model_dir)
    else:
        print("Using random weights (for benchmarking only)")
        config = KokoroConfig()
        model = KokoroModel(config)

    model.set_deterministic(True)

    # Create voice embedding
    voice = mx.zeros((1, 256))

    print("\n" + "=" * 80)
    print("Kokoro Streaming Synthesis Benchmark")
    print("=" * 80)

    # Streaming configurations to test
    streaming_configs = [
        {"chunk_frames": 50, "overlap_frames": 5, "name": "50 frames, 5 overlap"},
        {"chunk_frames": 100, "overlap_frames": 10, "name": "100 frames, 10 overlap"},
        {"chunk_frames": 200, "overlap_frames": 20, "name": "200 frames, 20 overlap"},
    ]

    for num_tokens in args.tokens:
        print(f"\n{'='*80}")
        print(f"Input: {num_tokens} tokens")
        print("=" * 80)

        input_ids = mx.zeros((1, num_tokens), dtype=mx.int32)

        # Non-streaming benchmark
        print("\nNon-streaming:")
        ns_results = benchmark_non_streaming(
            model, input_ids, voice, warmup=args.warmup, runs=args.runs
        )
        print(f"  TTFA: {ns_results['ttfa_ms']:.1f} ms (same as total)")
        print(f"  Total: {ns_results['mean_total_ms']:.1f} ms "
              f"(min: {ns_results['min_total_ms']:.1f}, max: {ns_results['max_total_ms']:.1f})")
        print(f"  Samples: {ns_results['samples']:,}")

        # Streaming benchmarks
        for config in streaming_configs:
            print(f"\nStreaming ({config['name']}):")
            s_results = benchmark_streaming(
                model, input_ids, voice,
                chunk_frames=config["chunk_frames"],
                overlap_frames=config["overlap_frames"],
                warmup=args.warmup, runs=args.runs
            )
            ttfa_improvement = ns_results['ttfa_ms'] / s_results['mean_ttfa_ms']
            total_overhead = (s_results['mean_total_ms'] - ns_results['mean_total_ms']) / ns_results['mean_total_ms'] * 100

            print(f"  TTFA: {s_results['mean_ttfa_ms']:.1f} ms "
                  f"({ttfa_improvement:.1f}x faster than non-streaming)")
            print(f"  Total: {s_results['mean_total_ms']:.1f} ms "
                  f"({total_overhead:+.1f}% overhead)")
            print(f"  Chunks: {s_results['chunks']}, Samples: {s_results['samples']:,}")

    print("\n" + "=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
