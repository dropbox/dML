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
Test GAP 10 Fix: avg_logprob tracking for greedy decoding

This script verifies that the C++ implementation calculates avg_logprob
correctly by comparing with Python mlx-whisper.

Reference: Python mlx-whisper decoding.py:677-678
    avg_logprobs: List[float] = [
        lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
    ]
"""

import subprocess
import json
import os
import sys

# Test audio file
TEST_AUDIO = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
MODEL_PATH = os.environ.get("WHISPER_MODEL", "models/whisper/mlx_models/whisper-large-v3-mlx")


def run_python_whisper():
    """Run Python mlx-whisper and extract avg_logprob."""
    import mlx_whisper

    result = mlx_whisper.transcribe(
        TEST_AUDIO,
        path_or_hf_repo=MODEL_PATH,
        language="en",
        word_timestamps=False,
        temperature=0.0,  # Greedy decoding
    )

    # Extract avg_logprob from first segment
    if result.get("segments"):
        return {
            "text": result["text"],
            "avg_logprob": result["segments"][0].get("avg_logprob", None),
            "segments": len(result["segments"]),
        }
    return None


def run_cpp_whisper():
    """Run C++ whisper and extract avg_logprob."""
    cmd = [
        "./build/test_mlx_engine",
        "--whisper", MODEL_PATH,
        "--no-vad",
        "--transcribe", TEST_AUDIO,
        "--json",  # Get JSON output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"C++ failed: {result.stderr}")
        return None

    # Parse JSON output (look for the JSON line)
    for line in result.stdout.split("\n"):
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if "segments" in data and data["segments"]:
                    return {
                        "text": data.get("text", ""),
                        "avg_logprob": data["segments"][0].get("avg_logprob", None),
                        "segments": len(data["segments"]),
                    }
            except json.JSONDecodeError:
                pass

    # Try parsing the entire output as JSON
    try:
        data = json.loads(result.stdout)
        if "segments" in data and data["segments"]:
            return {
                "text": data.get("text", ""),
                "avg_logprob": data["segments"][0].get("avg_logprob", None),
                "segments": len(data["segments"]),
            }
    except json.JSONDecodeError:
        pass

    return None


def main():
    print("=" * 70)
    print("GAP 10 Test: avg_logprob tracking for greedy decoding")
    print("=" * 70)
    print(f"Test audio: {TEST_AUDIO}")
    print(f"Model: {MODEL_PATH}")
    print()

    # Check if test audio exists
    if not os.path.exists(TEST_AUDIO):
        print(f"ERROR: Test audio not found: {TEST_AUDIO}")
        sys.exit(1)

    # Run Python
    print("Running Python mlx-whisper...")
    try:
        py_result = run_python_whisper()
        if py_result:
            print(f"  Text: {py_result['text'][:80]}...")
            print(f"  avg_logprob: {py_result['avg_logprob']}")
            print(f"  Segments: {py_result['segments']}")
        else:
            print("  ERROR: No result from Python")
    except Exception as e:
        print(f"  ERROR: {e}")
        py_result = None

    print()

    # Run C++
    print("Running C++ implementation...")
    cpp_result = run_cpp_whisper()
    if cpp_result:
        print(f"  Text: {cpp_result['text'][:80]}...")
        print(f"  avg_logprob: {cpp_result['avg_logprob']}")
        print(f"  Segments: {cpp_result['segments']}")
    else:
        print("  ERROR: No result from C++")

    print()
    print("=" * 70)

    # Compare results
    if py_result and cpp_result:
        py_lp = py_result['avg_logprob']
        cpp_lp = cpp_result['avg_logprob']

        if py_lp is not None and cpp_lp is not None:
            diff = abs(py_lp - cpp_lp)
            print(f"Python avg_logprob: {py_lp:.6f}")
            print(f"C++ avg_logprob:    {cpp_lp:.6f}")
            print(f"Difference:         {diff:.6f}")

            # Check if C++ is no longer using placeholder -0.5
            if abs(cpp_lp - (-0.5)) < 0.01:
                print("\nWARNING: C++ avg_logprob is near -0.5 (placeholder value)")
                print("This suggests the fix may not be working correctly.")
            elif cpp_lp < 0.0 and cpp_lp > -2.0:
                print("\nGood: C++ avg_logprob is in expected range (negative, > -2)")

            # Tolerance check
            if diff < 0.1:
                print("\nPASS: avg_logprob values are close (diff < 0.1)")
            else:
                print(f"\nFAIL: avg_logprob difference ({diff:.4f}) is too large")
                print("Note: Some difference is expected due to numerical precision")
        else:
            print("Could not compare - one or both avg_logprob values are None")
    else:
        print("Could not compare - missing results from one or both implementations")

    print("=" * 70)


if __name__ == "__main__":
    main()
