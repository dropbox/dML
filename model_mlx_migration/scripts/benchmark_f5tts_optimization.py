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
F5-TTS Optimization Benchmark

DEPRECATED: F5-TTS is deprecated in favor of CosyVoice2.
CosyVoice2 is 18x faster (35x vs 2x RTF) with equal/better quality.
This script is kept for historical reference only.

Test performance impact of:
1. Reduced step count (8 → 4)
2. Quantization (8-bit vs fp32)

Usage:
    python scripts/benchmark_f5tts_optimization.py
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_f5tts(steps: int, quantization_bits: int | None, text: str) -> dict:
    """Benchmark F5-TTS with specific settings."""
    from f5_tts_mlx.generate import generate

    output_path = tempfile.mktemp(suffix=".wav")

    # Generate audio
    start_time = time.time()
    generate(
        generation_text=text,
        output_path=output_path,
        steps=steps,
        quantization_bits=quantization_bits,
    )
    generation_time = time.time() - start_time

    # Load generated audio to get metrics
    audio_np, sample_rate = sf.read(output_path)
    audio_np = np.array(audio_np).flatten()
    audio_duration_s = len(audio_np) / sample_rate
    rtf = generation_time / max(audio_duration_s, 0.001)
    audio_rms = float(np.sqrt(np.mean(audio_np**2)))

    # Clean up
    Path(output_path).unlink(missing_ok=True)

    return {
        "steps": steps,
        "quantization_bits": quantization_bits,
        "generation_time_s": generation_time,
        "audio_duration_s": audio_duration_s,
        "rtf": rtf,
        "audio_rms": audio_rms,
    }


def main():
    # Check F5-TTS available
    try:
        from f5_tts_mlx.generate import generate  # noqa: F401
    except ImportError:
        print("F5-TTS not available. Install: pip install f5-tts-mlx")
        return 1

    test_text = "The quick brown fox jumps over the lazy dog."

    print("=" * 70)
    print("F5-TTS Optimization Benchmark")
    print("=" * 70)
    print(f"Test text: '{test_text}'")
    print()

    # Test configurations
    configs = [
        {"steps": 8, "quantization_bits": None, "name": "Baseline (8 steps, fp32)"},
        {"steps": 4, "quantization_bits": None, "name": "4 steps, fp32"},
        {"steps": 8, "quantization_bits": 8, "name": "8 steps, 8-bit"},
        {"steps": 4, "quantization_bits": 8, "name": "4 steps, 8-bit"},
    ]

    results = []
    for config in configs:
        print(f"Testing: {config['name']}...")
        try:
            result = benchmark_f5tts(
                steps=config["steps"],
                quantization_bits=config["quantization_bits"],
                text=test_text,
            )
            result["name"] = config["name"]
            results.append(result)
            print(
                f"  RTF={result['rtf']:.3f}x, "
                f"Duration={result['audio_duration_s']:.2f}s, "
                f"Time={result['generation_time_s']:.2f}s"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": config["name"], "error": str(e)})
        print()

    # Summary table
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Config':<25} | {'RTF':>8} | {'Gen Time':>10} | {'Duration':>10}")
    print("-" * 70)

    baseline_rtf = None
    for result in results:
        if "error" in result:
            print(f"{result['name']:<25} | {'ERROR':>8} | {'-':>10} | {'-':>10}")
            continue

        if baseline_rtf is None:
            baseline_rtf = result["rtf"]

        speedup = baseline_rtf / result["rtf"] if result["rtf"] > 0 else 0
        print(
            f"{result['name']:<25} | "
            f"{result['rtf']:>7.3f}x | "
            f"{result['generation_time_s']:>9.2f}s | "
            f"{result['audio_duration_s']:>9.2f}s | "
            f"({speedup:.2f}x speedup)"
        )

    print("-" * 70)

    # Recommendations
    print()
    print("ANALYSIS:")
    successful = [r for r in results if "error" not in r]
    if len(successful) >= 2:
        best = min(successful, key=lambda r: r["rtf"])
        baseline = results[0] if "error" not in results[0] else None
        if baseline:
            improvement = (baseline["rtf"] - best["rtf"]) / baseline["rtf"] * 100
            print(f"  Best config: {best['name']}")
            print(f"  RTF improvement: {improvement:.1f}% faster than baseline")
            if best["rtf"] < 1.0:
                print(f"  Status: Faster than real-time ({1 / best['rtf']:.1f}x)")
            else:
                print(f"  Status: Still slower than real-time ({best['rtf']:.2f}x)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
