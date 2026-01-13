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
Test and benchmark continuous batching (OPT-6) for translation models.

Compares sequential translation vs batched parallel translation to measure
throughput improvements.

Usage:
    python scripts/test_continuous_batching.py
    python scripts/test_continuous_batching.py --batch-sizes 2 4 8 16
    python scripts/test_continuous_batching.py --save-report
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import MADLADConverter


def test_basic_functionality():
    """Test that continuous batching produces correct translations."""
    print("=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)

    converter = MADLADConverter(quantize=8)
    converter.load()

    texts = [
        "Hello world",
        "How are you?",
        "Good morning",
    ]

    # Sequential translation
    print("\nSequential translation:")
    sequential_results = converter.translate_batch(texts, tgt_lang="fr")
    for i, r in enumerate(sequential_results):
        print(f"  [{i}] {texts[i]!r} -> {r.text!r}")

    # Continuous batching
    print("\nContinuous batching:")
    batch_result = converter.translate_batch_continuous(texts, tgt_lang="fr")
    for i, r in enumerate(batch_result.results):
        print(f"  [{i}] {texts[i]!r} -> {r.text!r}")

    # Compare results
    print("\nComparison:")
    matches = 0
    for i, (seq, cont) in enumerate(
        zip(sequential_results, batch_result.results)
    ):
        match = seq.text == cont.text
        matches += int(match)
        status = "MATCH" if match else "DIFF"
        print(f"  [{i}] {status}")
        if not match:
            print(f"       Sequential: {seq.text!r}")
            print(f"       Continuous: {cont.text!r}")

    print(f"\nResult: {matches}/{len(texts)} translations match")
    return matches == len(texts)


def test_varying_lengths():
    """Test continuous batching with varying input lengths."""
    print("\n" + "=" * 60)
    print("TEST 2: Varying Input Lengths")
    print("=" * 60)

    converter = MADLADConverter(quantize=8)
    converter.load()

    texts = [
        "Hi",  # Very short
        "Hello world, how are you today?",  # Medium
        "The quick brown fox jumps over the lazy dog. This is a longer sentence.",  # Long
        "OK",  # Very short
        "Machine translation has improved significantly with the advent of neural networks and transformer architectures.",  # Very long
    ]

    print(f"\nInput lengths: {[len(t) for t in texts]}")

    # Test continuous batching
    result = converter.translate_batch_continuous(texts, tgt_lang="de")

    print("\nResults:")
    for i, (src, r) in enumerate(zip(texts, result.results)):
        print(f"  [{i}] ({len(src)} chars) -> ({len(r.text)} chars, {r.tokens_generated} tokens)")
        print(f"       Input:  {src[:50]}{'...' if len(src) > 50 else ''}")
        print(f"       Output: {r.text[:50]}{'...' if len(r.text) > 50 else ''}")

    print(f"\nTotal latency: {result.total_latency_ms:.1f}ms")
    print(f"Throughput: {result.throughput_texts_per_second:.2f} texts/sec")

    return True


def benchmark_batch_sizes(batch_sizes=None):
    """Benchmark different batch sizes."""
    print("\n" + "=" * 60)
    print("TEST 3: Batch Size Scaling")
    print("=" * 60)

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    converter = MADLADConverter(quantize=8)
    converter.load()

    # Base sentences to replicate
    base_texts = [
        "Hello world.",
        "How are you today?",
        "The weather is nice.",
        "I love programming.",
    ]

    results = {}

    for batch_size in batch_sizes:
        # Create batch of desired size
        texts = (base_texts * ((batch_size // len(base_texts)) + 1))[:batch_size]

        print(f"\n--- Batch size: {batch_size} ---")

        # Warmup
        _ = converter.translate_batch_continuous(texts, tgt_lang="fr")

        # Sequential timing
        start = time.time()
        for _ in range(3):
            _ = converter.translate_batch(texts, tgt_lang="fr")
        seq_time = ((time.time() - start) / 3) * 1000

        # Continuous batching timing
        start = time.time()
        for _ in range(3):
            converter.translate_batch_continuous(texts, tgt_lang="fr")
        cont_time = ((time.time() - start) / 3) * 1000

        speedup = seq_time / cont_time if cont_time > 0 else 0

        results[batch_size] = {
            "sequential_ms": seq_time,
            "continuous_ms": cont_time,
            "speedup": speedup,
            "throughput_texts_per_sec": (batch_size / cont_time) * 1000,
        }

        print(f"  Sequential:  {seq_time:.1f}ms")
        print(f"  Continuous:  {cont_time:.1f}ms")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"  Throughput:  {results[batch_size]['throughput_texts_per_sec']:.2f} texts/sec")

    return results


def run_full_benchmark(iterations=5, warmup=2):
    """Run the built-in benchmark method."""
    print("\n" + "=" * 60)
    print("TEST 4: Full Benchmark (built-in method)")
    print("=" * 60)

    converter = MADLADConverter(quantize=8)
    converter.load()

    print(f"\nRunning {iterations} iterations with {warmup} warmup...")
    result = converter.benchmark_continuous_batching(
        iterations=iterations, warmup=warmup
    )

    print(f"\nBatch size: {result['batch_size']}")
    print("\nSequential:")
    print(f"  Mean latency:     {result['sequential']['mean_latency_ms']:.1f}ms")
    print(f"  Median latency:   {result['sequential']['median_latency_ms']:.1f}ms")
    print(f"  Throughput:       {result['sequential']['throughput_texts_per_second']:.2f} texts/sec")
    print(f"  Token throughput: {result['sequential']['throughput_tokens_per_second']:.2f} tokens/sec")

    print("\nContinuous Batching:")
    print(f"  Mean latency:     {result['continuous_batching']['mean_latency_ms']:.1f}ms")
    print(f"  Median latency:   {result['continuous_batching']['median_latency_ms']:.1f}ms")
    print(f"  Throughput:       {result['continuous_batching']['throughput_texts_per_second']:.2f} texts/sec")
    print(f"  Token throughput: {result['continuous_batching']['throughput_tokens_per_second']:.2f} tokens/sec")

    print(f"\nLatency speedup:    {result['speedup_latency']:.2f}x")
    print(f"Throughput speedup: {result['speedup_throughput']:.2f}x")

    return result


def test_quality_comparison(num_samples=20):
    """Compare translation quality between sequential and batched."""
    print("\n" + "=" * 60)
    print("TEST 5: Quality Comparison")
    print("=" * 60)

    converter = MADLADConverter(quantize=8)
    converter.load()

    # Diverse test sentences
    texts = [
        "Hello, how are you?",
        "The weather is beautiful today.",
        "I need to go to the store.",
        "What time does the movie start?",
        "Please help me with this problem.",
        "The quick brown fox jumps over the lazy dog.",
        "I love eating pizza with my friends.",
        "Can you recommend a good restaurant?",
        "This is a very important document.",
        "We should meet tomorrow at noon.",
        "The book was really interesting.",
        "I'm learning a new programming language.",
        "The concert was absolutely amazing.",
        "Please send me the report by Friday.",
        "I've never been to that country before.",
        "The coffee here is excellent.",
        "We need to find a better solution.",
        "Thank you for your help.",
        "The train leaves at 3 o'clock.",
        "I hope you have a great day.",
    ][:num_samples]

    print(f"\nComparing {len(texts)} translations...")

    # Sequential
    sequential = converter.translate_batch(texts, tgt_lang="de")

    # Continuous batching
    batch_result = converter.translate_batch_continuous(texts, tgt_lang="de")
    continuous = batch_result.results

    # Compare
    exact_matches = 0
    for i, (seq, cont) in enumerate(zip(sequential, continuous)):
        if seq.text == cont.text:
            exact_matches += 1
        else:
            print(f"\n[{i}] Difference found:")
            print(f"  Input:      {texts[i]}")
            print(f"  Sequential: {seq.text}")
            print(f"  Continuous: {cont.text}")

    match_rate = exact_matches / len(texts) * 100
    print(f"\n\nExact match rate: {exact_matches}/{len(texts)} ({match_rate:.1f}%)")

    return match_rate >= 95.0  # Allow 5% tolerance


def main():
    parser = argparse.ArgumentParser(
        description="Test and benchmark continuous batching for translation"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Benchmark iterations",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save results to JSON report",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CONTINUOUS BATCHING (OPT-6) TEST SUITE")
    print("=" * 60)

    all_results = {}

    # Test 1: Basic functionality
    try:
        passed = test_basic_functionality()
        all_results["basic_functionality"] = "PASS" if passed else "FAIL"
    except Exception as e:
        print(f"ERROR: {e}")
        all_results["basic_functionality"] = f"ERROR: {e}"

    # Test 2: Varying lengths
    try:
        passed = test_varying_lengths()
        all_results["varying_lengths"] = "PASS" if passed else "FAIL"
    except Exception as e:
        print(f"ERROR: {e}")
        all_results["varying_lengths"] = f"ERROR: {e}"

    # Test 3: Batch size scaling
    try:
        batch_results = benchmark_batch_sizes(args.batch_sizes)
        all_results["batch_scaling"] = batch_results
    except Exception as e:
        print(f"ERROR: {e}")
        all_results["batch_scaling"] = f"ERROR: {e}"

    # Test 4: Full benchmark
    try:
        benchmark_results = run_full_benchmark(iterations=args.iterations)
        all_results["full_benchmark"] = benchmark_results
    except Exception as e:
        print(f"ERROR: {e}")
        all_results["full_benchmark"] = f"ERROR: {e}"

    # Test 5: Quality comparison
    try:
        passed = test_quality_comparison()
        all_results["quality_comparison"] = "PASS" if passed else "FAIL"
    except Exception as e:
        print(f"ERROR: {e}")
        all_results["quality_comparison"] = f"ERROR: {e}"

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nTest Results:")
    for test, result in all_results.items():
        if isinstance(result, str):
            print(f"  {test}: {result}")
        elif isinstance(result, dict) and "speedup_latency" in result:
            print(f"  {test}: {result['speedup_latency']:.2f}x speedup")

    # Save report if requested
    if args.save_report:
        report_path = Path("tests/translation/continuous_batching_results.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nReport saved to: {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
