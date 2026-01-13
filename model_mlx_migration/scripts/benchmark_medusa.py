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
Benchmark script for Medusa multi-token prediction.

Compares transcription speed and accuracy between:
1. Standard greedy decoding
2. Medusa multi-token decoding

Usage:
    python scripts/benchmark_medusa.py --medusa-weights path/to/weights.npz
    python scripts/benchmark_medusa.py --audio-file path/to/audio.wav
    python scripts/benchmark_medusa.py --generate-random-weights  # For testing

Note: Without trained Medusa weights, this script generates random weights
for testing the pipeline. Actual speedups require trained weights.
"""

import argparse
import time
from typing import Dict, Optional

import numpy as np

from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.medusa import create_medusa_module
from tools.whisper_mlx.medusa_training import save_medusa_weights


def generate_test_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio with speech-like characteristics."""
    # Generate a simple tone modulated to simulate speech
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Mix of frequencies to simulate speech
    audio = np.zeros_like(t)
    for freq in [200, 400, 800, 1200]:
        # Add harmonics with amplitude modulation
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # 3 Hz modulation
        audio += envelope * np.sin(2 * np.pi * freq * t) / 4

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.5

    return audio.astype(np.float32)


def generate_random_medusa_weights(
    model: WhisperMLX,
    n_heads: int = 5,
    output_path: str = "/tmp/medusa_random_weights.npz",
) -> str:
    """
    Generate random Medusa weights for testing.

    These weights won't produce good predictions, but they allow
    testing the full pipeline.
    """
    print(f"Generating random Medusa weights with {n_heads} heads...")

    # Create Medusa module with random weights
    medusa = create_medusa_module(
        model.config,
        n_heads=n_heads,
        use_block=False,
        dtype=model.dtype,
    )

    # Save weights
    save_medusa_weights(medusa, output_path)
    print(f"Saved random weights to {output_path}")

    return output_path


def benchmark_standard_decode(
    model: WhisperMLX,
    audio: np.ndarray,
    n_runs: int = 3,
    warmup: int = 1,
) -> Dict:
    """Benchmark standard greedy decoding."""
    results = {
        "times": [],
        "texts": [],
    }

    # Warmup
    for _ in range(warmup):
        _ = model.transcribe(audio, temperature=0)

    # Benchmark runs
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = model.transcribe(audio, temperature=0)
        elapsed = time.perf_counter() - t0

        results["times"].append(elapsed)
        results["texts"].append(result["text"])
        print(f"  Run {i+1}/{n_runs}: {elapsed*1000:.1f}ms")

    results["mean_time"] = np.mean(results["times"])
    results["std_time"] = np.std(results["times"])
    results["text"] = results["texts"][0]  # Use first result

    return results


def benchmark_medusa_decode(
    model: WhisperMLX,
    audio: np.ndarray,
    n_runs: int = 3,
    warmup: int = 1,
) -> Dict:
    """Benchmark Medusa multi-token decoding."""
    if not model.medusa_loaded:
        raise RuntimeError("Medusa heads not loaded")

    results = {
        "times": [],
        "texts": [],
        "acceptance_rates": [],
        "tokens_per_step": [],
    }

    # Warmup
    for _ in range(warmup):
        _ = model.transcribe_medusa(audio, temperature=0)

    # Benchmark runs
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = model.transcribe_medusa(audio, temperature=0)
        elapsed = time.perf_counter() - t0

        results["times"].append(elapsed)
        results["texts"].append(result["text"])
        results["acceptance_rates"].append(result["acceptance_rate"])
        results["tokens_per_step"].append(result["tokens_per_step"])

        print(f"  Run {i+1}/{n_runs}: {elapsed*1000:.1f}ms "
              f"(acc={result['acceptance_rate']:.1%}, "
              f"tps={result['tokens_per_step']:.2f})")

    results["mean_time"] = np.mean(results["times"])
    results["std_time"] = np.std(results["times"])
    results["mean_acceptance_rate"] = np.mean(results["acceptance_rates"])
    results["mean_tokens_per_step"] = np.mean(results["tokens_per_step"])
    results["text"] = results["texts"][0]

    return results


def compare_results(
    standard_results: Dict,
    medusa_results: Dict,
) -> Dict:
    """Compare standard and Medusa results."""
    speedup = standard_results["mean_time"] / medusa_results["mean_time"]

    # Check if texts match
    standard_text = standard_results["text"].strip()
    medusa_text = medusa_results["text"].strip()
    texts_match = standard_text == medusa_text

    comparison = {
        "standard_time_ms": standard_results["mean_time"] * 1000,
        "medusa_time_ms": medusa_results["mean_time"] * 1000,
        "speedup": speedup,
        "acceptance_rate": medusa_results.get("mean_acceptance_rate", 0),
        "tokens_per_step": medusa_results.get("mean_tokens_per_step", 1),
        "texts_match": texts_match,
        "standard_text": standard_text,
        "medusa_text": medusa_text,
    }

    return comparison


def run_benchmark(
    model_name: str = "mlx-community/whisper-large-v3-mlx",
    medusa_weights: Optional[str] = None,
    audio_file: Optional[str] = None,
    audio_duration: float = 10.0,
    n_heads: int = 5,
    n_runs: int = 3,
    warmup: int = 1,
    generate_random: bool = False,
) -> Dict:
    """
    Run full benchmark comparing standard and Medusa decoding.

    Args:
        model_name: Whisper model to load
        medusa_weights: Path to trained Medusa weights
        audio_file: Optional audio file to transcribe
        audio_duration: Duration of generated test audio (if no file)
        n_heads: Number of Medusa heads
        n_runs: Number of benchmark runs
        warmup: Number of warmup runs
        generate_random: Generate random Medusa weights for testing

    Returns:
        Benchmark results dictionary
    """
    print("=" * 60)
    print("Medusa Multi-Token Benchmark")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {model_name}")
    model = WhisperMLX.from_pretrained(model_name, warmup=True)
    print(f"Model loaded: {model.config.n_vocab} vocab, {model.config.n_text_layer} layers")

    # Prepare audio
    if audio_file:
        print(f"\nLoading audio: {audio_file}")
        from tools.whisper_mlx.audio import load_audio
        audio = load_audio(audio_file, sample_rate=16000)
    else:
        print(f"\nGenerating {audio_duration}s test audio...")
        audio = generate_test_audio(duration=audio_duration)

    print(f"Audio shape: {audio.shape}, duration: {len(audio)/16000:.2f}s")

    # Handle Medusa weights
    if generate_random or (medusa_weights is None):
        if medusa_weights is None:
            print("\nNo Medusa weights provided. Generating random weights for testing.")
            print("NOTE: Random weights won't give real speedups - train weights first!")
        medusa_weights = generate_random_medusa_weights(model, n_heads=n_heads)

    # Benchmark standard decoding
    print(f"\n--- Standard Decoding ({n_runs} runs, {warmup} warmup) ---")
    standard_results = benchmark_standard_decode(model, audio, n_runs=n_runs, warmup=warmup)
    print(f"Mean: {standard_results['mean_time']*1000:.1f}ms ± {standard_results['std_time']*1000:.1f}ms")
    print(f"Text: {standard_results['text'][:100]}...")

    # Load Medusa heads and benchmark
    print(f"\n--- Medusa Decoding ({n_runs} runs, {warmup} warmup) ---")
    print(f"Loading Medusa weights: {medusa_weights}")
    model.load_medusa_heads(medusa_weights, n_heads=n_heads)

    medusa_results = benchmark_medusa_decode(model, audio, n_runs=n_runs, warmup=warmup)
    print(f"Mean: {medusa_results['mean_time']*1000:.1f}ms ± {medusa_results['std_time']*1000:.1f}ms")
    print(f"Acceptance rate: {medusa_results['mean_acceptance_rate']:.1%}")
    print(f"Tokens per step: {medusa_results['mean_tokens_per_step']:.2f}")
    print(f"Text: {medusa_results['text'][:100]}...")

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    comparison = compare_results(standard_results, medusa_results)

    print(f"\nStandard decoding: {comparison['standard_time_ms']:.1f}ms")
    print(f"Medusa decoding:   {comparison['medusa_time_ms']:.1f}ms")
    print(f"Speedup:           {comparison['speedup']:.2f}x")
    print(f"Acceptance rate:   {comparison['acceptance_rate']:.1%}")
    print(f"Tokens per step:   {comparison['tokens_per_step']:.2f}")
    print(f"Texts match:       {comparison['texts_match']}")

    if not comparison['texts_match']:
        print("\nWARNING: Texts do not match!")
        print(f"  Standard: {comparison['standard_text'][:80]}...")
        print(f"  Medusa:   {comparison['medusa_text'][:80]}...")
        print("\n(This is expected with random weights)")

    # Theoretical analysis
    if comparison['tokens_per_step'] > 1:
        theoretical_speedup = comparison['tokens_per_step']
        efficiency = comparison['speedup'] / theoretical_speedup
        print(f"\nTheoretical speedup: {theoretical_speedup:.2f}x")
        print(f"Actual speedup:      {comparison['speedup']:.2f}x")
        print(f"Efficiency:          {efficiency:.1%}")

    return {
        "standard": standard_results,
        "medusa": medusa_results,
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Medusa multi-token prediction for Whisper"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-large-v3-mlx",
        help="Whisper model to benchmark",
    )
    parser.add_argument(
        "--medusa-weights",
        type=str,
        default=None,
        help="Path to trained Medusa weights (.npz)",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Audio file to transcribe (generates test audio if not provided)",
    )
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=10.0,
        help="Duration of generated test audio in seconds",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=5,
        help="Number of Medusa heads",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--generate-random-weights",
        action="store_true",
        help="Generate random Medusa weights for testing",
    )

    args = parser.parse_args()

    results = run_benchmark(
        model_name=args.model,
        medusa_weights=args.medusa_weights,
        audio_file=args.audio_file,
        audio_duration=args.audio_duration,
        n_heads=args.n_heads,
        n_runs=args.n_runs,
        warmup=args.warmup,
        generate_random=args.generate_random_weights,
    )

    # Summary
    print("\n" + "=" * 60)
    if results["comparison"]["speedup"] >= 1.5:
        print("SUCCESS: Medusa provides significant speedup!")
    elif results["comparison"]["speedup"] >= 1.0:
        print("MARGINAL: Medusa provides slight improvement")
    else:
        print("NOTE: Medusa slower than standard (train weights or check config)")
    print("=" * 60)


if __name__ == "__main__":
    main()
