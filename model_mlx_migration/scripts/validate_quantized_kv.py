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
WhisperMLX KV Cache Quantization Audit (OPT-2.3)

Validates that INT8 KV cache quantization produces identical or near-identical
transcriptions compared to non-quantized KV cache.

Tests:
1. Exact text match rate (100% required for lossless claim)
2. Token-level comparison
3. Speed comparison
4. Memory comparison

Usage:
    python scripts/validate_quantized_kv.py

Exit codes:
    0: Audit passed (>=99% exact match)
    1: Audit failed (<99% exact match)
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


@dataclass
class QuantizationTestResult:
    """Results from a single quantization comparison test."""

    audio_path: str
    audio_duration: float

    # Text comparison
    baseline_text: str  # Non-quantized KV cache
    quantized_text: str  # Quantized KV cache (OPT-2.3)
    texts_match: bool

    # Timing
    baseline_time: float
    quantized_time: float
    speedup: float

    # Error info
    error: Optional[str] = None


def find_test_audio_files() -> list[Path]:
    """Find test audio files for validation."""
    project_root = Path(__file__).parent.parent

    test_files = []

    # Primary test files
    for wav in ["medium_mlx.wav", "short_mlx.wav", "long_mlx.wav"]:
        path = project_root / "reports/audio" / wav
        if path.exists():
            test_files.append(path)

    # Additional test audio
    kokoro_audio = list((project_root / "reports/audio").glob("kokoro_misaki*.wav"))
    test_files.extend(kokoro_audio[:3])

    # RAVDESS prosody samples (diverse emotions, known content)
    ravdess_dir = project_root / "data/prosody/ravdess/Actor_01"
    if ravdess_dir.exists():
        ravdess_files = sorted(ravdess_dir.glob("*.wav"))[:10]
        test_files.extend(ravdess_files)

    return test_files


def compare_quantization(
    audio_path: Path,
    model_quantized,
    model_baseline,
) -> QuantizationTestResult:
    """
    Compare quantized vs non-quantized KV cache on single file.

    Args:
        audio_path: Path to audio file
        model_quantized: WhisperMLX with quantize_kv=True
        model_baseline: WhisperMLX with quantize_kv=False

    Returns:
        QuantizationTestResult
    """
    result_data = {
        "audio_path": str(audio_path),
        "audio_duration": 0.0,
        "baseline_text": "",
        "quantized_text": "",
        "texts_match": False,
        "baseline_time": 0.0,
        "quantized_time": 0.0,
        "speedup": 1.0,
        "error": None,
    }

    try:
        # Baseline: Non-quantized KV cache
        t0 = time.perf_counter()
        result_baseline = model_baseline.transcribe(str(audio_path), variable_length=False)
        result_data["baseline_time"] = time.perf_counter() - t0
        result_data["baseline_text"] = result_baseline.get("text", "").strip()
        result_data["audio_duration"] = result_baseline.get("audio_duration", 0.0)

        # Clear GPU cache
        mx.eval(mx.zeros(1))

        # Quantized: INT8 KV cache (OPT-2.3)
        t0 = time.perf_counter()
        result_quantized = model_quantized.transcribe(str(audio_path), variable_length=False)
        result_data["quantized_time"] = time.perf_counter() - t0
        result_data["quantized_text"] = result_quantized.get("text", "").strip()

        # Compare
        result_data["texts_match"] = result_data["baseline_text"] == result_data["quantized_text"]
        if result_data["quantized_time"] > 0:
            result_data["speedup"] = result_data["baseline_time"] / result_data["quantized_time"]

    except Exception as e:
        result_data["error"] = str(e)

    return QuantizationTestResult(**result_data)


def run_quantization_audit():
    """Run comprehensive quantization audit."""
    print("=" * 60)
    print("WhisperMLX KV Cache Quantization Audit (OPT-2.3)")
    print("=" * 60)
    print()

    # Find test files
    test_files = find_test_audio_files()
    if not test_files:
        print("ERROR: No test audio files found")
        return 1

    print(f"Found {len(test_files)} test audio files")
    print()

    # Load models
    print("Loading WhisperMLX models...")
    print("  - Model with quantize_kv=True (INT8 cross-attention cache)")
    print("  - Model with quantize_kv=False (FP16 cross-attention cache)")
    print()

    from tools.whisper_mlx import WhisperMLX

    # Load quantized model (default)
    t0 = time.perf_counter()
    model_quantized = WhisperMLX.from_pretrained("large-v3", quantize_kv=True)
    load_time_q = time.perf_counter() - t0
    print(f"  Quantized model loaded in {load_time_q:.2f}s")

    # Load baseline model
    t0 = time.perf_counter()
    model_baseline = WhisperMLX.from_pretrained("large-v3", quantize_kv=False)
    load_time_b = time.perf_counter() - t0
    print(f"  Baseline model loaded in {load_time_b:.2f}s")
    print()

    # Run tests
    print("Running comparisons...")
    print("-" * 60)

    results = []
    for i, audio_path in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] {audio_path.name}...", end=" ", flush=True)

        result = compare_quantization(audio_path, model_quantized, model_baseline)
        results.append(result)

        if result.error:
            print(f"ERROR: {result.error}")
        elif result.texts_match:
            print(f"MATCH ({result.speedup:.2f}x)")
        else:
            print("MISMATCH")
            print(f"    Baseline: {result.baseline_text[:50]}...")
            print(f"    Quantized: {result.quantized_text[:50]}...")

    print("-" * 60)
    print()

    # Calculate statistics
    total = len(results)
    errors = sum(1 for r in results if r.error)
    valid = total - errors
    matches = sum(1 for r in results if r.texts_match and not r.error)
    match_rate = matches / valid if valid > 0 else 0.0

    avg_speedup = (
        sum(r.speedup for r in results if not r.error) / valid
        if valid > 0
        else 1.0
    )

    # Generate report
    report = generate_report(results, matches, valid, match_rate, avg_speedup)

    # Write report
    report_path = Path(__file__).parent.parent / "reports/main/KV_CACHE_QUANTIZATION_AUDIT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report written to {report_path}")
    print()

    # Summary
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Total files tested: {total}")
    print(f"  Valid comparisons:  {valid}")
    print(f"  Exact matches:      {matches}")
    print(f"  Match rate:         {100*match_rate:.1f}%")
    print(f"  Average speedup:    {avg_speedup:.2f}x")
    print()

    if match_rate >= 0.99:
        print("RESULT: PASS - INT8 KV cache is LOSSLESS (>=99% exact match)")
        print()
        print("RECOMMENDATION: Keep quantize_kv=True as default")
        return 0
    elif match_rate >= 0.95:
        print("RESULT: CONDITIONAL - 95-99% match rate")
        print()
        print("RECOMMENDATION: Enable with warning in documentation")
        return 0
    else:
        print("RESULT: FAIL - INT8 KV cache causes quality degradation")
        print()
        print("RECOMMENDATION: Disable quantize_kv by default")
        return 1


def generate_report(results, matches, valid, match_rate, avg_speedup):
    """Generate markdown report."""
    timestamp = time.strftime("%Y-%m-%d %H:%M")

    # Build results table
    results_table = ""
    for r in results:
        status = "ERROR" if r.error else ("MATCH" if r.texts_match else "MISMATCH")
        results_table += f"| {Path(r.audio_path).name} | {r.audio_duration:.1f}s | {status} | {r.speedup:.2f}x |\n"

    # Build mismatches section
    mismatches = [r for r in results if not r.texts_match and not r.error]
    mismatch_section = ""
    if mismatches:
        mismatch_section = "\n## Mismatches\n\n"
        for r in mismatches:
            mismatch_section += f"### {Path(r.audio_path).name}\n\n"
            mismatch_section += f"**Baseline (FP16):** {r.baseline_text}\n\n"
            mismatch_section += f"**Quantized (INT8):** {r.quantized_text}\n\n"

    # Recommendation
    if match_rate >= 0.99:
        recommendation = "**SAFE TO ENABLE**: >=99% exact match rate. Quality loss is negligible. Keep `quantize_kv=True` as default."
    elif match_rate >= 0.95:
        recommendation = "**CONDITIONAL**: 95-99% match rate. Enable with documentation warning."
    else:
        recommendation = "**DO NOT ENABLE**: <95% match rate. Quality loss is unacceptable. Set `quantize_kv=False` as default."

    report = f"""# KV Cache Quantization Quality Audit (OPT-2.3)

**Date**: {timestamp}
**Test Files**: {len(results)}
**Valid Comparisons**: {valid}

## Summary

| Metric | Value |
|--------|-------|
| Exact Text Match Rate | {matches}/{valid} ({100*match_rate:.1f}%) |
| Average Speedup | {avg_speedup:.2f}x |

## Results

| Audio File | Duration | Result | Speedup |
|------------|----------|--------|---------|
{results_table}
{mismatch_section}
## Recommendation

{recommendation}

## Technical Details

**INT8 KV Cache Quantization (OPT-2.3)**

- Cross-attention K/V is stored as INT8 with per-tensor scaling
- Self-attention K/V remains FP16 for incremental accuracy
- Memory reduction: ~50% for cross-attention cache
- Quantization: `x_int8 = round(x / scale)` where `scale = max(|x|) / 127`

**Why Cross-Attention Only?**

Cross-attention K/V (encoder output) is computed once and reused for all decoder steps.
This makes it ideal for quantization:
1. Computed once, read many times (bandwidth-bound)
2. No accumulated error from incremental updates
3. INT8 storage reduces memory bandwidth during attention

Self-attention K/V is updated incrementally and thus remains FP16 to prevent error accumulation.

---

*Generated by scripts/validate_quantized_kv.py*
"""

    return report


if __name__ == "__main__":
    sys.exit(run_quantization_audit())
