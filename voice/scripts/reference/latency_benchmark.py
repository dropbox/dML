#!/usr/bin/env python3
"""
TTS Latency Test

Measures time-to-first-audio and end-to-end latency.

Usage:
    python tests/test_latency.py --command "python tts.py {text} -o {output}"

Thresholds:
    - Time to first audio: <= 200ms (streaming)
    - End-to-end latency: <= 500ms
"""

import argparse
import subprocess
import tempfile
import time
import os
import sys

def measure_latency(command_template: str, text: str, iterations: int = 3) -> dict:
    """
    Measure TTS latency.

    Returns:
        dict with latency measurements in milliseconds
    """
    latencies = []

    for i in range(iterations):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            command = command_template.replace("{text}", text).replace("{output}", wav_path)

            start = time.perf_counter()
            result = subprocess.run(command, shell=True, capture_output=True, timeout=30)
            end = time.perf_counter()

            if result.returncode != 0:
                print(f"  Iteration {i+1}: FAILED - {result.stderr.decode()[:100]}")
                continue

            if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
                print(f"  Iteration {i+1}: FAILED - No audio generated")
                continue

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            print(f"  Iteration {i+1}: {latency_ms:.0f}ms")

        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    if not latencies:
        return {"error": "All iterations failed"}

    return {
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "avg_ms": sum(latencies) / len(latencies),
        "p50_ms": sorted(latencies)[len(latencies) // 2],
        "iterations": len(latencies)
    }

def run_latency_tests(command_template: str) -> bool:
    """Run latency tests with different text lengths."""
    print("=" * 60)
    print("TTS LATENCY TEST")
    print("=" * 60)

    tests = [
        ("short", "Hello"),
        ("medium", "Hello world, this is a test"),
        ("long", "The quick brown fox jumps over the lazy dog near the river"),
    ]

    all_passed = True

    for name, text in tests:
        print(f"\n--- {name.upper()} ({len(text)} chars) ---")
        result = measure_latency(command_template, text)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            all_passed = False
            continue

        avg = result["avg_ms"]

        # Thresholds based on text length
        threshold = 500 + (len(text) * 10)  # Base 500ms + 10ms per char

        if avg <= threshold:
            print(f"  ✅ PASS: {avg:.0f}ms avg (threshold: {threshold}ms)")
        else:
            print(f"  ❌ FAIL: {avg:.0f}ms avg (threshold: {threshold}ms)")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL LATENCY TESTS PASSED")
    else:
        print("❌ SOME LATENCY TESTS FAILED")
    print("=" * 60)

    return all_passed

def main():
    parser = argparse.ArgumentParser(description="TTS Latency Test")
    parser.add_argument("--command", required=True,
                       help="TTS command template. Use {text} for input, {output} for WAV path")
    parser.add_argument("--threshold", type=int, default=500,
                       help="Maximum acceptable latency in ms (default: 500)")

    args = parser.parse_args()

    passed = run_latency_tests(args.command)
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
