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
Benchmark CosyVoice2 Speculative Decoding

Compares standard generation vs speculative decoding with trained early exit head.

Usage:
    python scripts/benchmark_cosyvoice2_speculative.py
    python scripts/benchmark_cosyvoice2_speculative.py --runs 5
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def benchmark_standard(model, text_ids, max_tokens, runs):
    """Benchmark standard (non-speculative) generation."""
    times = []
    tokens_generated = []

    for r in range(runs):
        start = time.time()
        speech_tokens = model.llm.generate_speech_tokens(
            text_ids,
            max_length=max_tokens,
            temperature=1.0,
            top_k=25,
            top_p=0.8,
        )
        mx.eval(speech_tokens)
        elapsed = time.time() - start

        times.append(elapsed)
        tokens_generated.append(speech_tokens.shape[-1])

    return {
        "method": "standard",
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "mean_tokens": np.mean(tokens_generated),
        "tokens_per_sec": np.mean(tokens_generated) / np.mean(times),
    }


def benchmark_speculative(model, text_ids, max_tokens, runs, num_draft_tokens=4):
    """Benchmark speculative decoding generation."""
    times = []
    tokens_generated = []
    acceptance_rates = []

    for r in range(runs):
        start = time.time()
        speech_tokens, stats = model.llm.generate_speech_tokens_speculative(
            text_ids,
            max_length=max_tokens,
            temperature=1.0,
            top_k=25,
            top_p=0.8,
            num_draft_tokens=num_draft_tokens,
        )
        mx.eval(speech_tokens)
        elapsed = time.time() - start

        times.append(elapsed)
        tokens_generated.append(speech_tokens.shape[-1])
        acceptance_rates.append(stats["acceptance_rate"])

    return {
        "method": f"speculative (draft={num_draft_tokens})",
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "mean_tokens": np.mean(tokens_generated),
        "tokens_per_sec": np.mean(tokens_generated) / np.mean(times),
        "acceptance_rate": np.mean(acceptance_rates),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CosyVoice2 Speculative Decoding")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to CosyVoice2 model",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--exit-layer",
        type=int,
        default=6,
        help="Early exit layer",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()

    from tools.pytorch_to_mlx.converters.models import CosyVoice2Model

    # Load model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = CosyVoice2Model.get_default_model_path()

    print("="*70)
    print("CosyVoice2 Speculative Decoding Benchmark")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Exit layer: {args.exit_layer}")
    print(f"Runs: {args.runs}")
    print()

    print("Loading model...")
    model = CosyVoice2Model.from_pretrained(model_path)
    model.llm.early_exit_layer = args.exit_layer

    # Check for trained early exit head
    head_path = Path(f"models/early_exit_heads/cosyvoice2_exit{args.exit_layer}_head.npz")
    if head_path.exists():
        print(f"Loading trained early exit head from {head_path}")
        weights = mx.load(str(head_path))
        model.llm.early_exit_head.load_weights(list(weights.items()))
    else:
        print("No trained head found, using initialized head (lower acceptance rate)")
        model.llm.initialize_early_exit_head()
    print()

    # Test texts
    test_texts = [
        "Hello, how are you today?",
        "This is a test of the speech synthesis system.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("="*70)
    print("Results")
    print("="*70)

    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: \"{text[:50]}...\"" if len(text) > 50 else f"\nTest {i+1}: \"{text}\"")
        print("-"*70)

        text_ids = model.tokenizer.encode(text)
        if text_ids.ndim == 1:
            text_ids = text_ids[None, :]

        # Warmup
        _ = model.llm.generate_speech_tokens(text_ids, max_length=20)
        mx.eval(_)

        # Benchmark standard
        std_result = benchmark_standard(model, text_ids, args.max_tokens, args.runs)

        # Benchmark speculative with different draft sizes
        spec_results = []
        for draft in [2, 4, 6]:
            result = benchmark_speculative(model, text_ids, args.max_tokens, args.runs, draft)
            spec_results.append(result)

        # Print results
        print(f"{'Method':<30} {'Time':>8} {'Tokens':>8} {'Tok/s':>8} {'Accept':>8} {'Speedup':>8}")
        print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        baseline_time = std_result["mean_time"]
        print(f"{std_result['method']:<30} {std_result['mean_time']:>7.2f}s {std_result['mean_tokens']:>7.0f} {std_result['tokens_per_sec']:>7.1f} {'N/A':>8} {'1.00x':>8}")

        for result in spec_results:
            speedup = baseline_time / result["mean_time"]
            accept = f"{result['acceptance_rate']:.1%}"
            print(f"{result['method']:<30} {result['mean_time']:>7.2f}s {result['mean_tokens']:>7.0f} {result['tokens_per_sec']:>7.1f} {accept:>8} {speedup:>7.2f}x")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("\nNotes:")
    print("- CosyVoice2 LLM is 87% of total synthesis time")
    print("- Speculative decoding targets the LLM bottleneck")
    print(f"- Trained early exit head at layer {args.exit_layer}/24")
    print("- Higher acceptance rates = better speedup potential")

    return 0


if __name__ == "__main__":
    sys.exit(main())
