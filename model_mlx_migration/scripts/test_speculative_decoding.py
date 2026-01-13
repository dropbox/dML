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
Test script for speculative decoding in translation models.

Tests early-exit speculative decoding for MADLAD translation.
"""

import sys

sys.path.insert(0, ".")

from tools.pytorch_to_mlx.converters.madlad_converter import MADLADConverter


def test_basic_translation():
    """Test that speculative decoding produces correct translations."""
    print("=" * 60)
    print("Test 1: Basic Translation Quality")
    print("=" * 60)

    converter = MADLADConverter()

    test_cases = [
        ("Hello, how are you?", "fr"),
        ("The quick brown fox jumps over the lazy dog.", "de"),
        ("I love programming in Python.", "es"),
    ]

    for text, lang in test_cases:
        print(f"\nInput: {text}")
        print(f"Target language: {lang}")

        # Baseline translation
        baseline = converter.translate(text, lang)
        print(f"Baseline: {baseline.text}")

        # Speculative translation
        speculative = converter.translate_speculative(text, lang)
        print(f"Speculative: {speculative.text}")
        print(f"  Acceptance rate: {speculative.acceptance_rate:.1%}")
        print(f"  Accepted/Total draft: {speculative.accepted_tokens}/{speculative.total_draft_tokens}")

        # Check if they produce the same result
        match = baseline.text == speculative.text
        print(f"  Match: {'✓' if match else '✗'}")

    print()


def test_benchmark():
    """Benchmark speculative vs baseline decoding."""
    print("=" * 60)
    print("Test 2: Speculative Decoding Benchmark")
    print("=" * 60)

    converter = MADLADConverter()

    # Test different early_exit_layers configurations
    configs = [
        {"early_exit_layers": 2, "num_draft_tokens": 4},
        {"early_exit_layers": 4, "num_draft_tokens": 4},
        {"early_exit_layers": 6, "num_draft_tokens": 4},
        {"early_exit_layers": 4, "num_draft_tokens": 2},
        {"early_exit_layers": 4, "num_draft_tokens": 6},
    ]

    test_text = "Hello, how are you today? I hope you are doing well and having a wonderful day."

    print(f"\nTest text: {test_text}")
    print("Target: French")
    print()

    for config in configs:
        print(f"Config: layers={config['early_exit_layers']}, draft_tokens={config['num_draft_tokens']}")

        result = converter.benchmark_speculative(
            text=test_text,
            tgt_lang="fr",
            iterations=5,
            warmup=2,
            **config,
        )

        print(f"  Baseline:    {result['baseline']['mean_latency_ms']:.1f}ms")
        print(f"  Speculative: {result['speculative']['mean_latency_ms']:.1f}ms")
        print(f"  Acceptance:  {result['speculative']['mean_acceptance_rate']:.1%}")
        print(f"  Speedup:     {result['speedup']:.2f}x")
        print()


def test_longer_text():
    """Test with longer text to see if acceptance rate improves."""
    print("=" * 60)
    print("Test 3: Longer Text Test")
    print("=" * 60)

    converter = MADLADConverter()

    long_text = """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing computer programs that can access data and use it
    to learn for themselves. The process begins with observations or data,
    such as examples, direct experience, or instruction.
    """.strip().replace("\n", " ")

    print(f"\nTest text (truncated): {long_text[:100]}...")
    print("Target: French")
    print()

    # Baseline
    baseline = converter.translate(long_text, "fr")
    print(f"Baseline latency: {baseline.latency_ms:.1f}ms")
    print(f"Baseline tokens: {baseline.tokens_generated}")

    # Speculative
    speculative = converter.translate_speculative(long_text, "fr")
    print(f"Speculative latency: {speculative.latency_ms:.1f}ms")
    print(f"Speculative tokens: {speculative.tokens_generated}")
    print(f"Acceptance rate: {speculative.acceptance_rate:.1%}")
    print(f"Accepted/Total draft: {speculative.accepted_tokens}/{speculative.total_draft_tokens}")

    speedup = baseline.latency_ms / speculative.latency_ms
    print(f"Speedup: {speedup:.2f}x")

    print(f"\nBaseline translation: {baseline.text[:200]}...")
    print(f"\nSpeculative translation: {speculative.text[:200]}...")


def main():
    print("Speculative Decoding Test Suite")
    print("================================")
    print()

    try:
        test_basic_translation()
        test_benchmark()
        test_longer_text()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
