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
Test and benchmark Mixed Precision Quantization (OPT-9)

Compares:
- 8-bit quantization (baseline, lossless quality)
- 4-bit quantization (fastest, some quality loss)
- Mixed precision (8-bit attention, 4-bit FFN)

Expected outcome:
- Mixed precision should have quality closer to 8-bit
- Mixed precision should have speed closer to 4-bit
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import MADLADConverter

# Test sentences - diverse set to reveal quality differences
TEST_SENTENCES = [
    # Basic sentences
    "Hello, how are you today?",
    "The weather is beautiful.",
    "I would like to order coffee.",
    # Complex sentences
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Could you please tell me where the nearest train station is located?",
    "Artificial intelligence is transforming how we work and live.",
    # CJK stress test (4-bit has known issues with CJK)
    "I love Japanese food, especially sushi and ramen.",
    "The Great Wall of China is a UNESCO World Heritage site.",
    "Korean culture has become popular worldwide through K-pop.",
    # Technical/business
    "Please find attached the quarterly financial report for your review.",
    "The software development team will deploy the new feature next week.",
    # Numbers and proper nouns
    "The meeting is scheduled for March 15th at 2:30 PM.",
    "Dr. Smith published her research paper in the Nature journal.",
    # Emotional/expressive
    "I am so excited to see you again after all these years!",
    "This is absolutely unacceptable and must be addressed immediately.",
    # Longer sentences
    (
        "The scientific community has reached a consensus that climate change "
        "poses significant risks to ecosystems, economies, and human health."
    ),
    (
        "When planning your next vacation, consider visiting countries "
        "that offer historical landmarks, natural beauty, and culture."
    ),
    # Short sentences
    "Yes, please.",
    "Thank you very much.",
    "Good morning everyone.",
]

TARGET_LANGUAGES = ["de", "fr", "ja", "ko", "zh"]


def benchmark_quantization(quantize_mode, name, sentences, target_lang, iterations=3):
    """
    Benchmark a specific quantization mode.

    Returns:
        dict with latency, throughput, and translations
    """
    print(f"\n{'='*60}")
    print(f"Testing: {name} (target: {target_lang})")
    print(f"{'='*60}")

    # Load model with specific quantization
    load_start = time.time()
    converter = MADLADConverter(quantize=quantize_mode)
    converter.load()
    load_time = time.time() - load_start
    print(f"Model load time: {load_time:.2f}s")

    # Warmup
    print("Warming up...")
    for _ in range(2):
        converter.translate("Hello world", tgt_lang=target_lang)

    # Benchmark
    print(f"Benchmarking {len(sentences)} sentences, {iterations} iterations...")
    all_latencies = []
    translations = []

    for iteration in range(iterations):
        iter_start = time.time()
        iter_translations = []

        for sentence in sentences:
            result = converter.translate(sentence, tgt_lang=target_lang)
            all_latencies.append(result.latency_ms)
            if iteration == 0:  # Only store first iteration's translations
                iter_translations.append({
                    "source": sentence,
                    "target": result.text,
                    "latency_ms": result.latency_ms,
                    "tokens": result.tokens_generated,
                })

        if iteration == 0:
            translations = iter_translations

        iter_time = time.time() - iter_start
        rate = len(sentences) / iter_time
        print(f"  Iteration {iteration + 1}: {iter_time:.2f}s ({rate:.1f} texts/sec)")

    # Calculate statistics
    avg_latency = sum(all_latencies) / len(all_latencies)
    median_latency = sorted(all_latencies)[len(all_latencies) // 2]
    min_latency = min(all_latencies)
    max_latency = max(all_latencies)
    throughput = len(sentences) * iterations / (sum(all_latencies) / 1000)

    print(f"\nResults for {name}:")
    print(f"  Average latency: {avg_latency:.1f}ms")
    print(f"  Median latency: {median_latency:.1f}ms")
    print(f"  Min/Max latency: {min_latency:.1f}ms / {max_latency:.1f}ms")
    print(f"  Throughput: {throughput:.2f} texts/sec")

    return {
        "name": name,
        "quantize_mode": str(quantize_mode),
        "target_lang": target_lang,
        "load_time_s": load_time,
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_texts_per_sec": throughput,
        "num_sentences": len(sentences),
        "iterations": iterations,
        "translations": translations,
    }


def compare_quality(results_8bit, results_mixed, results_4bit):
    """
    Compare translation quality between quantization modes.

    Uses 8-bit as reference (known lossless).
    """
    print(f"\n{'='*60}")
    print("Quality Comparison (8-bit as reference)")
    print(f"{'='*60}")

    trans_8bit = results_8bit["translations"]
    trans_mixed = results_mixed["translations"]
    trans_4bit = results_4bit["translations"]
    translations_8bit = {t["source"]: t["target"] for t in trans_8bit}
    translations_mixed = {t["source"]: t["target"] for t in trans_mixed}
    translations_4bit = {t["source"]: t["target"] for t in trans_4bit}

    # Compare mixed vs 8-bit
    mixed_exact = 0
    mixed_diff = []
    for source, ref in translations_8bit.items():
        mixed_trans = translations_mixed.get(source, "")
        if ref == mixed_trans:
            mixed_exact += 1
        else:
            mixed_diff.append({
                "source": source,
                "8bit": ref,
                "mixed": mixed_trans,
            })

    # Compare 4-bit vs 8-bit
    bit4_exact = 0
    bit4_diff = []
    for source, ref in translations_8bit.items():
        bit4_trans = translations_4bit.get(source, "")
        if ref == bit4_trans:
            bit4_exact += 1
        else:
            bit4_diff.append({
                "source": source,
                "8bit": ref,
                "4bit": bit4_trans,
            })

    total = len(translations_8bit)

    print("\nMixed Precision vs 8-bit:")
    print(f"  Exact match: {mixed_exact}/{total} ({100*mixed_exact/total:.1f}%)")
    print(f"  Different: {len(mixed_diff)}")

    print("\n4-bit vs 8-bit:")
    print(f"  Exact match: {bit4_exact}/{total} ({100*bit4_exact/total:.1f}%)")
    print(f"  Different: {len(bit4_diff)}")

    # Show sample differences
    if mixed_diff:
        print("\nSample Mixed vs 8-bit differences:")
        for diff in mixed_diff[:3]:
            print(f"  Source: {diff['source'][:50]}...")
            print(f"    8-bit: {diff['8bit'][:50]}...")
            print(f"    Mixed: {diff['mixed'][:50]}...")

    if bit4_diff:
        print("\nSample 4-bit vs 8-bit differences:")
        for diff in bit4_diff[:3]:
            print(f"  Source: {diff['source'][:50]}...")
            print(f"    8-bit: {diff['8bit'][:50]}...")
            print(f"    4-bit: {diff['4bit'][:50]}...")

    return {
        "mixed_exact_match": mixed_exact,
        "mixed_exact_match_pct": 100 * mixed_exact / total,
        "mixed_differences": len(mixed_diff),
        "4bit_exact_match": bit4_exact,
        "4bit_exact_match_pct": 100 * bit4_exact / total,
        "4bit_differences": len(bit4_diff),
        "total_sentences": total,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test mixed precision quantization")
    parser.add_argument(
        "--target-lang", default="de",
        help="Target language (default: de)"
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Benchmark iterations"
    )
    parser.add_argument(
        "--output", default="tests/translation/mixed_precision_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with fewer sentences"
    )
    args = parser.parse_args()

    sentences = TEST_SENTENCES[:5] if args.quick else TEST_SENTENCES

    print("=" * 60)
    print("Mixed Precision Quantization Benchmark (OPT-9)")
    print("=" * 60)
    print(f"Test sentences: {len(sentences)}")
    print(f"Target language: {args.target_lang}")
    print(f"Iterations: {args.iterations}")

    # Run benchmarks
    results_8bit = benchmark_quantization(
        8, "8-bit Quantization (Baseline)",
        sentences, args.target_lang, args.iterations
    )

    results_mixed = benchmark_quantization(
        "mixed", "Mixed Precision (8-bit attn, 4-bit FFN)",
        sentences, args.target_lang, args.iterations
    )

    results_4bit = benchmark_quantization(
        4, "4-bit Quantization",
        sentences, args.target_lang, args.iterations
    )

    # Compare quality
    quality_comparison = compare_quality(results_8bit, results_mixed, results_4bit)

    # Calculate speedups
    lat_8 = results_8bit["avg_latency_ms"]
    lat_mix = results_mixed["avg_latency_ms"]
    lat_4 = results_4bit["avg_latency_ms"]
    speedup_mixed_vs_8bit = lat_8 / lat_mix
    speedup_4bit_vs_8bit = lat_8 / lat_4

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nLatency Comparison (lower is better):")
    print(f"  8-bit:  {lat_8:.1f}ms (baseline)")
    print(f"  Mixed:  {lat_mix:.1f}ms ({speedup_mixed_vs_8bit:.2f}x vs 8-bit)")
    print(f"  4-bit:  {lat_4:.1f}ms ({speedup_4bit_vs_8bit:.2f}x vs 8-bit)")

    print("\nQuality Comparison (higher is better):")
    print("  8-bit:  100% (reference)")
    print(f"  Mixed:  {quality_comparison['mixed_exact_match_pct']:.1f}% exact match")
    print(f"  4-bit:  {quality_comparison['4bit_exact_match_pct']:.1f}% exact match")

    print("\nRecommendation:")
    mixed_pct = quality_comparison["mixed_exact_match_pct"]
    bit4_pct = quality_comparison["4bit_exact_match_pct"]
    if mixed_pct > bit4_pct:
        if speedup_mixed_vs_8bit > 1.1:
            print("  Mixed precision provides BEST quality/speed tradeoff!")
        else:
            print("  Mixed precision has better quality but similar speed to 8-bit.")
    else:
        print("  Mixed precision does not improve over 4-bit quality.")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {
        "8bit": results_8bit,
        "mixed": results_mixed,
        "4bit": results_4bit,
        "quality_comparison": quality_comparison,
        "summary": {
            "speedup_mixed_vs_8bit": speedup_mixed_vs_8bit,
            "speedup_4bit_vs_8bit": speedup_4bit_vs_8bit,
            "mixed_quality_pct": quality_comparison["mixed_exact_match_pct"],
            "4bit_quality_pct": quality_comparison["4bit_exact_match_pct"],
        }
    }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
