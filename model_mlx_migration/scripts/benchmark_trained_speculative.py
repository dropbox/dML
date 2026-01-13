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
Benchmark speculative decoding with trained early exit head (OPT-4).

Compares:
1. Baseline translation (no speculative decoding)
2. Untrained speculative decoding (expected slowdown)
3. Trained speculative decoding (expected speedup)

Usage:
    python scripts/benchmark_trained_speculative.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))



def benchmark():
    from tools.pytorch_to_mlx.converters.madlad_converter import MADLADConverter

    # Test sentences
    test_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming many industries.",
        "Climate change affects weather patterns globally.",
        "The company announced record profits today.",
    ]

    print("=" * 60)
    print("Benchmark: Speculative Decoding with Trained Early Exit Head")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    converter = MADLADConverter(dtype="bfloat16", quantize=8)
    converter.load()

    # Benchmark 1: Baseline translation
    print("\n" + "-" * 40)
    print("1. Baseline Translation (no speculative decoding)")
    print("-" * 40)

    baseline_times = []
    baseline_tokens = []
    for text in test_texts:
        start = time.time()
        result = converter.translate(text, tgt_lang="de")
        elapsed = (time.time() - start) * 1000
        baseline_times.append(elapsed)
        baseline_tokens.append(result.tokens_generated)
        print(f"  {text[:40]}... -> {elapsed:.1f}ms ({result.tokens_generated} tokens)")

    avg_baseline = sum(baseline_times) / len(baseline_times)
    print(f"\n  Average: {avg_baseline:.1f}ms/translation")

    # Benchmark 2: Untrained speculative decoding
    print("\n" + "-" * 40)
    print("2. Speculative Decoding (untrained)")
    print("-" * 40)

    untrained_times = []
    untrained_acceptance = []
    for text in test_texts:
        result = converter.translate_speculative(text, tgt_lang="de")
        untrained_times.append(result.latency_ms)
        untrained_acceptance.append(result.acceptance_rate)
        print(f"  {text[:40]}... -> {result.latency_ms:.1f}ms (acc: {result.acceptance_rate:.1%})")

    avg_untrained = sum(untrained_times) / len(untrained_times)
    avg_untrained_acc = sum(untrained_acceptance) / len(untrained_acceptance)
    print(f"\n  Average: {avg_untrained:.1f}ms/translation (acceptance: {avg_untrained_acc:.1%})")
    print(f"  Speedup vs baseline: {avg_baseline / avg_untrained:.2f}x")

    # Benchmark 3: Trained speculative decoding
    print("\n" + "-" * 40)
    print("3. Speculative Decoding (trained early_exit_head)")
    print("-" * 40)

    # Load trained head
    head_path = Path("models/early_exit_heads/madlad400-3b-mt_exit4_head.safetensors")
    if not head_path.exists():
        print(f"  ERROR: Trained head not found at {head_path}")
        print("  Run train_early_exit_head.py first")
        return

    converter.load_early_exit_head(str(head_path))

    trained_times = []
    trained_acceptance = []
    for text in test_texts:
        result = converter.translate_speculative(text, tgt_lang="de")
        trained_times.append(result.latency_ms)
        trained_acceptance.append(result.acceptance_rate)
        print(f"  {text[:40]}... -> {result.latency_ms:.1f}ms (acc: {result.acceptance_rate:.1%})")

    avg_trained = sum(trained_times) / len(trained_times)
    avg_trained_acc = sum(trained_acceptance) / len(trained_acceptance)
    print(f"\n  Average: {avg_trained:.1f}ms/translation (acceptance: {avg_trained_acc:.1%})")
    print(f"  Speedup vs baseline: {avg_baseline / avg_trained:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline:            {avg_baseline:.1f}ms")
    print(f"Untrained spec:      {avg_untrained:.1f}ms (acceptance: {avg_untrained_acc:.1%})")
    print(f"Trained spec:        {avg_trained:.1f}ms (acceptance: {avg_trained_acc:.1%})")
    print("")
    print(f"Speedup (untrained): {avg_baseline / avg_untrained:.2f}x")
    print(f"Speedup (trained):   {avg_baseline / avg_trained:.2f}x")
    print("")
    print(f"Acceptance improvement: {avg_untrained_acc:.1%} -> {avg_trained_acc:.1%}")


if __name__ == "__main__":
    benchmark()
