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
MADLAD Model Size Benchmark

Compares MADLAD-3B, 7B, and 10B models on:
- Translation latency
- Translation quality
- Memory usage

Usage:
    python scripts/benchmark_madlad_sizes.py [--sizes 3b 7b] [--sentences 20]
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pytorch_to_mlx.converters.madlad_converter import MADLADConverter


@dataclass
class BenchmarkResult:
    model_size: str
    latency_mean_ms: float
    latency_median_ms: float
    tokens_per_second: float
    translations: List[str]
    memory_bytes: Optional[int] = None


# Test sentences for quality evaluation
TEST_SENTENCES = [
    # Short
    "Hello world.",
    "How are you?",
    "Good morning!",
    # Medium
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning has transformed artificial intelligence.",
    "I love programming in Python and using MLX on Apple Silicon.",
    # Long
    "Neural machine translation has made significant advances in recent years, enabling real-time translation between hundreds of languages.",
    "The MADLAD-400 model supports over 400 languages and is licensed under Apache 2.0, making it suitable for commercial applications.",
    # Technical
    "The transformer architecture uses self-attention mechanisms to process sequences in parallel.",
    "Quantization reduces model size and improves inference speed while maintaining quality.",
    # Named entities and numbers
    "Apple released the M3 Max chip in November 2023.",
    "Tokyo hosted the 2020 Olympic Games in 2021.",
    # Questions
    "What is the capital of France?",
    "How do neural networks learn?",
    # Emotional/conversational
    "I'm so excited about this new technology!",
    "That's really disappointing news.",
    # CJK (Chinese characters often problematic with low-bit quantization)
    "Artificial intelligence will change the world.",
    "The weather is beautiful today.",
    # Ambiguous
    "Bank can mean a financial institution or a river side.",
    "I saw the man with the telescope.",
]

# Target languages to test
TARGET_LANGUAGES = ["fr", "de", "es", "ja", "zh", "ko"]


def get_memory_usage():
    """Get current memory usage in bytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    except ImportError:
        return None


def benchmark_model(
    model_id: str,
    model_size: str,
    sentences: List[str],
    tgt_lang: str = "fr",
    iterations: int = 3,
    warmup: int = 2,
    quantize: int = 8,
) -> BenchmarkResult:
    """Benchmark a specific MADLAD model size."""
    print(f"\n{'='*60}")
    print(f"Benchmarking MADLAD-{model_size.upper()}")
    print(f"Model: {model_id}")
    print(f"Quantization: {quantize}-bit")
    print("=" * 60)

    # Force garbage collection before loading
    gc.collect()
    memory_before = get_memory_usage()

    # Load model
    print("Loading model...")
    start_load = time.time()
    converter = MADLADConverter(model_path=model_id, quantize=quantize)
    converter.load()
    load_time = time.time() - start_load
    print(f"Load time: {load_time:.1f}s")

    memory_after = get_memory_usage()
    memory_used = (memory_after - memory_before) if memory_before and memory_after else None
    if memory_used:
        print(f"Memory used: {memory_used / 1e9:.2f} GB")

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = converter.translate(sentences[0], tgt_lang)

    # Benchmark
    print(f"Benchmarking ({iterations} iterations per sentence)...")
    latencies = []
    translations = []
    total_tokens = 0

    for sentence in sentences:
        sentence_latencies = []
        translation = None

        for _ in range(iterations):
            result = converter.translate(sentence, tgt_lang)
            sentence_latencies.append(result.latency_ms)
            translation = result.text
            total_tokens += result.tokens_generated

        latencies.extend(sentence_latencies)
        translations.append(translation)

    # Calculate statistics
    latencies.sort()
    latency_mean = sum(latencies) / len(latencies)
    latency_median = latencies[len(latencies) // 2]
    total_time_ms = sum(latencies)
    tokens_per_second = (total_tokens / total_time_ms) * 1000

    print(f"Mean latency: {latency_mean:.1f}ms")
    print(f"Median latency: {latency_median:.1f}ms")
    print(f"Tokens/sec: {tokens_per_second:.1f}")

    # Clean up to free memory for next model
    del converter
    gc.collect()

    return BenchmarkResult(
        model_size=model_size,
        latency_mean_ms=latency_mean,
        latency_median_ms=latency_median,
        tokens_per_second=tokens_per_second,
        translations=translations,
        memory_bytes=memory_used,
    )


def compare_quality(results: Dict[str, BenchmarkResult], sentences: List[str]):
    """Compare translation quality across model sizes."""
    print(f"\n{'='*60}")
    print("QUALITY COMPARISON")
    print("=" * 60 + "\n")

    sizes = list(results.keys())

    # Compare each sentence across models
    for i, sentence in enumerate(sentences):
        print(f"\n[{i+1}] Source: {sentence}")
        for size in sizes:
            translation = results[size].translations[i]
            print(f"    {size.upper()}: {translation}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MADLAD model sizes")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["3b", "7b"],
        choices=["3b", "7b", "10b"],
        help="Model sizes to benchmark",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=10,
        help="Number of test sentences to use",
    )
    parser.add_argument(
        "--lang",
        default="fr",
        choices=["fr", "de", "es", "ja", "zh", "ko", "ru", "ar"],
        help="Target language for translation",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Benchmark iterations per sentence",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bits (8 recommended for quality)",
    )
    args = parser.parse_args()

    # Model IDs
    model_ids = {
        "3b": "google/madlad400-3b-mt",
        "7b": "google/madlad400-7b-mt",
        "10b": "google/madlad400-10b-mt",
    }

    sentences = TEST_SENTENCES[:args.sentences]
    print(f"Testing with {len(sentences)} sentences, target language: {args.lang}")

    # Benchmark each model size
    results: Dict[str, BenchmarkResult] = {}

    for size in args.sizes:
        try:
            result = benchmark_model(
                model_id=model_ids[size],
                model_size=size,
                sentences=sentences,
                tgt_lang=args.lang,
                iterations=args.iterations,
                quantize=args.quantize,
            )
            results[size] = result
        except Exception as e:
            print(f"\nERROR benchmarking {size}: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print("=" * 60 + "\n")

    print(f"{'Model':<10} {'Mean (ms)':<12} {'Median (ms)':<12} {'Tok/s':<10} {'Memory (GB)':<12}")
    print("-" * 56)

    for size in args.sizes:
        if size in results:
            r = results[size]
            memory_str = f"{r.memory_bytes/1e9:.2f}" if r.memory_bytes else "N/A"
            print(f"{size.upper():<10} {r.latency_mean_ms:<12.1f} {r.latency_median_ms:<12.1f} {r.tokens_per_second:<10.1f} {memory_str:<12}")

    # Speedup comparison
    if "3b" in results and len(results) > 1:
        baseline = results["3b"]
        print("\nSpeedup vs 3B:")
        for size in args.sizes:
            if size != "3b" and size in results:
                speedup = baseline.latency_mean_ms / results[size].latency_mean_ms
                print(f"  {size.upper()}: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

    # Quality comparison
    if len(results) > 1:
        compare_quality(results, sentences)

    print(f"\n{'='*60}")
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
