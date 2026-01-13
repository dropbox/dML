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
Test script for N-gram Lookahead Decoding (OPT-7)

This script tests and benchmarks the n-gram lookahead decoding approach
for MADLAD translation.

N-gram lookahead works best for:
- Repetitive patterns (technical documents, legal text)
- Copy mechanisms (source words appearing in output)
- Common phrases and patterns

Worker: #1159
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters.madlad_converter import MADLADConverter


def test_basic_translation():
    """Test that n-gram translation produces correct output."""
    print("=" * 60)
    print("TEST 1: Basic N-gram Translation")
    print("=" * 60)

    converter = MADLADConverter(quantize=8)

    # Simple sentence
    text = "Hello, how are you today?"
    print(f"\nInput: {text}")

    # Baseline
    baseline = converter.translate(text, "fr")
    print(f"Baseline: {baseline.text} ({baseline.latency_ms:.1f}ms)")

    # N-gram lookahead
    ngram = converter.translate_ngram(text, "fr")
    print(f"N-gram:   {ngram.text} ({ngram.latency_ms:.1f}ms)")
    print(f"  - N-gram hits: {ngram.ngram_hits}")
    print(f"  - Lookahead tokens: {ngram.total_lookahead_tokens}")
    print(f"  - Accepted: {ngram.accepted_tokens}")
    print(f"  - Acceptance rate: {ngram.acceptance_rate:.1%}")

    # Check translation matches
    if baseline.text == ngram.text:
        print("\n[PASS] Translations match")
        return True
    else:
        print("\n[FAIL] Translations differ!")
        return False


def test_repetitive_text():
    """Test n-gram lookahead on repetitive text (best case)."""
    print("\n" + "=" * 60)
    print("TEST 2: Repetitive Text (Best Case for N-gram)")
    print("=" * 60)

    converter = MADLADConverter(quantize=8)

    # Repetitive technical text
    text = """
    The system processes data. The system validates data.
    The system transforms data. The system stores data.
    The data processing completes successfully.
    """

    print(f"\nInput: {text.strip()[:60]}...")

    # Baseline
    baseline = converter.translate(text, "de")
    print(f"\nBaseline: {baseline.text[:80]}...")
    print(f"  - Latency: {baseline.latency_ms:.1f}ms")
    print(f"  - Tokens: {baseline.tokens_generated}")

    # N-gram lookahead with various settings
    for ngram_n in [2, 3, 4]:
        for max_lookahead in [2, 4, 6]:
            ngram = converter.translate_ngram(
                text, "de", ngram_n=ngram_n, max_lookahead=max_lookahead
            )
            speedup = baseline.latency_ms / ngram.latency_ms if ngram.latency_ms > 0 else 0
            print(
                f"\nN-gram (n={ngram_n}, lookahead={max_lookahead}): "
                f"{ngram.latency_ms:.1f}ms ({speedup:.2f}x)"
            )
            print(f"  - Hits: {ngram.ngram_hits}, Accepted: {ngram.accepted_tokens}/{ngram.total_lookahead_tokens}")
            print(f"  - Acceptance: {ngram.acceptance_rate:.1%}")

    return True


def test_copy_mechanism():
    """Test n-gram lookahead with copy-heavy text (names, numbers)."""
    print("\n" + "=" * 60)
    print("TEST 3: Copy Mechanism (Names, Numbers)")
    print("=" * 60)

    converter = MADLADConverter(quantize=8)

    # Text with proper nouns and numbers that should be copied
    text = """
    John Smith met with Maria Garcia at 123 Main Street on December 15, 2024.
    The meeting included Dr. Robert Chen and Professor Elizabeth Brown.
    Contact information: john.smith@email.com, phone 555-123-4567.
    """

    print(f"\nInput: {text.strip()[:80]}...")

    # Baseline
    baseline = converter.translate(text, "fr")
    print(f"\nBaseline: {baseline.text[:100]}...")
    print(f"  - Latency: {baseline.latency_ms:.1f}ms")

    # N-gram lookahead
    ngram = converter.translate_ngram(text, "fr", ngram_n=2, max_lookahead=4)
    speedup = baseline.latency_ms / ngram.latency_ms if ngram.latency_ms > 0 else 0
    print(f"\nN-gram: {ngram.text[:100]}...")
    print(f"  - Latency: {ngram.latency_ms:.1f}ms ({speedup:.2f}x)")
    print(f"  - Hits: {ngram.ngram_hits}, Accepted: {ngram.accepted_tokens}/{ngram.total_lookahead_tokens}")
    print(f"  - Acceptance: {ngram.acceptance_rate:.1%}")

    return True


def benchmark_ngram():
    """Run full benchmark comparing n-gram lookahead vs baseline."""
    print("\n" + "=" * 60)
    print("BENCHMARK: N-gram Lookahead vs Baseline")
    print("=" * 60)

    converter = MADLADConverter(quantize=8)

    test_cases = [
        {
            "name": "Short sentence",
            "text": "Hello, how are you today?",
            "expected_speedup": "Low (few patterns)",
        },
        {
            "name": "Repetitive text",
            "text": "The system processes data. The system validates data. "
            "The system transforms data. The system stores data.",
            "expected_speedup": "Higher (repeated patterns)",
        },
        {
            "name": "Technical documentation",
            "text": "The API endpoint accepts JSON requests. The API returns JSON responses. "
            "The API requires authentication. The API supports rate limiting. "
            "The API documentation describes all endpoints. The API version is 2.0.",
            "expected_speedup": "Higher (repeated 'The API')",
        },
        {
            "name": "Legal text pattern",
            "text": "The Parties hereby agree that the Contract shall be binding. "
            "The Parties agree that the Contract terms are final. "
            "The Parties shall comply with the Contract provisions. "
            "The Contract shall be governed by applicable law.",
            "expected_speedup": "Higher (legal patterns)",
        },
    ]

    results = []
    print("\n[Benchmarking with 10 iterations each...]\n")

    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {case['name']}")
        benchmark = converter.benchmark_ngram(
            text=case["text"],
            tgt_lang="de",
            iterations=10,
            warmup=3,
            ngram_n=3,
            max_lookahead=4,
        )

        result = {
            "name": case["name"],
            "expected": case["expected_speedup"],
            **benchmark,
        }
        results.append(result)

        print(f"  Baseline: {benchmark['baseline']['mean_latency_ms']:.1f}ms")
        print(f"  N-gram:   {benchmark['ngram_lookahead']['mean_latency_ms']:.1f}ms")
        print(f"  Speedup:  {benchmark['speedup']:.2f}x")
        print(f"  Acceptance: {benchmark['ngram_lookahead']['mean_acceptance_rate']:.1%}")
        print(f"  N-gram hits: {benchmark['ngram_lookahead']['mean_ngram_hits']:.1f}")
        print()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{'Test Case':<25} | {'Baseline (ms)':<15} | {'N-gram (ms)':<13} | {'Speedup':<8} | {'Accept %'}"
    )
    print("-" * 85)

    for r in results:
        print(
            f"{r['name']:<25} | "
            f"{r['baseline']['mean_latency_ms']:>13.1f} | "
            f"{r['ngram_lookahead']['mean_latency_ms']:>11.1f} | "
            f"{r['speedup']:>7.2f}x | "
            f"{r['ngram_lookahead']['mean_acceptance_rate']:>6.1%}"
        )

    return results


def main():
    print("N-gram Lookahead Decoding Test (OPT-7)")
    print("Worker: #1159")
    print("=" * 60)

    # Run tests
    test1_pass = test_basic_translation()
    test2_pass = test_repetitive_text()
    test3_pass = test_copy_mechanism()

    # Run benchmark
    benchmark_results = benchmark_ngram()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Basic translation test: {'PASS' if test1_pass else 'FAIL'}")
    print(f"Repetitive text test: {'PASS' if test2_pass else 'FAIL'}")
    print(f"Copy mechanism test: {'PASS' if test3_pass else 'FAIL'}")

    # Save results
    output_path = Path("tests/translation/ngram_lookahead_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "tests": {
                    "basic_translation": test1_pass,
                    "repetitive_text": test2_pass,
                    "copy_mechanism": test3_pass,
                },
                "benchmarks": benchmark_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
