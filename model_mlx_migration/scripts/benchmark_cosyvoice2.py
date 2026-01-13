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
CosyVoice2 Performance Benchmark

Measures throughput and latency for different text lengths.
Includes batch synthesis comparison for throughput workloads.

Usage:
    python scripts/benchmark_cosyvoice2.py
    python scripts/benchmark_cosyvoice2.py --runs 5
    python scripts/benchmark_cosyvoice2.py --batch-sizes 1,2,4
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def benchmark_synthesis(model, text: str, runs: int = 3, max_tokens: int = 100):
    """Benchmark synthesis for a given text."""

    speaker_embedding = model.tokenizer.random_speaker_embedding(seed=42)

    # Warm up
    audio = model.synthesize_text(
        text, speaker_embedding=speaker_embedding, max_tokens=max_tokens
    )
    mx.eval(audio)

    # Benchmark
    times = []
    audio_durations = []

    for _ in range(runs):
        start = time.time()
        audio = model.synthesize_text(
            text, speaker_embedding=speaker_embedding, max_tokens=max_tokens
        )
        mx.eval(audio)
        elapsed = time.time() - start

        times.append(elapsed)
        audio_durations.append(len(np.array(audio)) / model.config.sample_rate)

    return {
        "text_chars": len(text),
        "max_tokens": max_tokens,
        "audio_duration_mean": np.mean(audio_durations),
        "synthesis_time_mean": np.mean(times),
        "synthesis_time_std": np.std(times),
        "rtf_mean": np.mean([t / d for t, d in zip(times, audio_durations)]),
    }


def benchmark_batch_synthesis(model, texts: list, runs: int = 3, max_tokens: int = 100):
    """Benchmark batch synthesis for multiple texts."""

    speaker_embeddings = model.tokenizer.random_speaker_embedding(seed=42)

    # Warm up with single synthesis
    audio = model.synthesize_text(
        texts[0], speaker_embedding=speaker_embeddings, max_tokens=max_tokens
    )
    mx.eval(audio)

    # Sequential benchmark
    seq_times = []
    seq_audio_durations = []

    for _ in range(runs):
        start = time.time()
        audios = []
        for text in texts:
            audio = model.synthesize_text(
                text, speaker_embedding=speaker_embeddings, max_tokens=max_tokens
            )
            mx.eval(audio)
            audios.append(audio)
        elapsed = time.time() - start

        seq_times.append(elapsed)
        total_audio = sum(len(np.array(a)) / model.config.sample_rate for a in audios)
        seq_audio_durations.append(total_audio)

    # Batch benchmark
    batch_times = []
    batch_audio_durations = []

    for _ in range(runs):
        start = time.time()
        batch_audio, lengths = model.synthesize_batch_text(
            texts, speaker_embeddings=speaker_embeddings, max_tokens=max_tokens
        )
        mx.eval(batch_audio, lengths)
        elapsed = time.time() - start

        batch_times.append(elapsed)
        total_audio = float(mx.sum(lengths).item()) / model.config.sample_rate
        batch_audio_durations.append(total_audio)

    return {
        "batch_size": len(texts),
        "seq_time_mean": np.mean(seq_times),
        "seq_time_std": np.std(seq_times),
        "batch_time_mean": np.mean(batch_times),
        "batch_time_std": np.std(batch_times),
        "seq_audio_duration": np.mean(seq_audio_durations),
        "batch_audio_duration": np.mean(batch_audio_durations),
        "speedup": np.mean(seq_times) / np.mean(batch_times),
        "seq_rtf": np.mean(seq_times) / np.mean(seq_audio_durations),
        "batch_rtf": np.mean(batch_times) / np.mean(batch_audio_durations),
    }


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2 Benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--model-path", type=str, default=None, help="Model path")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes for batch benchmark (e.g., '2,3,4')",
    )
    args = parser.parse_args()

    from tools.pytorch_to_mlx.converters.models import CosyVoice2Model

    # Model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = CosyVoice2Model.get_default_model_path()

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    print("=" * 70)
    print("CosyVoice2 Performance Benchmark")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Runs per test: {args.runs}")
    print()

    # Load model
    print("Loading model...")
    start = time.time()
    model = CosyVoice2Model.from_pretrained(model_path)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    print()

    # Test texts of different lengths with max_tokens estimates
    # Roughly 2-3 tokens per character for English
    test_texts = [
        ("Short", "Hello world.", 30),
        ("Medium", "This is a test of the CosyVoice2 text to speech system.", 100),
        (
            "Long",
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the alphabet.",
            200,
        ),
        (
            "Very Long",
            "In the beginning, there was silence. "
            "Then came the first words, spoken by ancient peoples "
            "who sought to communicate their thoughts and feelings.",
            300,
        ),
    ]

    print("=" * 70)
    cols = f"{'Test':<12} {'Chars':>6} {'Tokens':>7}"
    cols += f" {'Audio':>8} {'Synth':>8} {'RTF':>8}"
    print(cols)
    print("-" * 70)

    results = []
    for name, text, max_tokens in test_texts:
        result = benchmark_synthesis(model, text, runs=args.runs, max_tokens=max_tokens)
        results.append((name, result))

        print(
            f"{name:<12} {result['text_chars']:>6} {result['max_tokens']:>7} "
            f"{result['audio_duration_mean']:>7.2f}s "
            f"{result['synthesis_time_mean']:>7.2f}s "
            f"{result['rtf_mean']:>7.3f}x"
        )

    print("=" * 70)
    print()

    # Summary statistics
    avg_rtf = np.mean([r["rtf_mean"] for _, r in results])
    print(f"Average RTF: {avg_rtf:.3f}x")
    print(f"Average speedup: {1 / avg_rtf:.1f}x real-time")
    print()

    # Throughput
    total_audio = sum(r["audio_duration_mean"] for _, r in results)
    total_synth = sum(r["synthesis_time_mean"] for _, r in results)
    print(f"Total audio generated: {total_audio:.2f}s")
    print(f"Total synthesis time: {total_synth:.2f}s")
    print(f"Overall throughput: {total_audio / total_synth:.1f}x real-time")
    print()

    print("Benchmark complete.")

    # Batch synthesis benchmark (if requested)
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

        print()
        print("=" * 70)
        print("Batch Synthesis Benchmark")
        print("=" * 70)
        print()

        # Sample texts for batch synthesis
        batch_texts = [
            "Hello, how are you today?",
            "The weather is beautiful this morning.",
            "I hope you have a wonderful day ahead.",
            "Technology is changing the world rapidly.",
            "Music has the power to move our souls.",
            "Reading expands the mind and imagination.",
            "Nature provides us with endless beauty.",
            "Friendship is one of life's greatest gifts.",
        ]

        print(
            f"{'Batch':<6} {'Seq Time':>10} {'Batch Time':>12} "
            f"{'Speedup':>10} {'Seq RTF':>10} {'Batch RTF':>12}"
        )
        print("-" * 70)

        for batch_size in batch_sizes:
            texts = batch_texts[:batch_size]
            if len(texts) < batch_size:
                # Repeat if we need more texts
                texts = (texts * (batch_size // len(texts) + 1))[:batch_size]

            result = benchmark_batch_synthesis(model, texts, runs=args.runs)

            print(
                f"{batch_size:<6} {result['seq_time_mean']:>9.2f}s "
                f"{result['batch_time_mean']:>11.2f}s "
                f"{result['speedup']:>9.2f}x "
                f"{1/result['seq_rtf']:>9.1f}x "
                f"{1/result['batch_rtf']:>11.1f}x"
            )

        print("=" * 70)
        print()
        print("Notes:")
        print("- Seq Time: Total time for sequential synthesis")
        print("- Batch Time: Total time for batch synthesis")
        print("- Speedup: Seq Time / Batch Time (higher = better)")
        print("- RTF columns show real-time factor (higher = faster)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
