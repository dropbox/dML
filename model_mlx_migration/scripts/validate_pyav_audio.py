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
Validate PyAV audio loading matches ffmpeg subprocess exactly.

This script compares the audio output from both backends to ensure
bit-exact equivalence. If differences are found, it analyzes them
to help debug the issue.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.audio import (
    _is_pyav_available,
    _load_audio_ffmpeg,
    _load_audio_pyav,
)


def validate_pyav_matches_ffmpeg(audio_path: str, verbose: bool = True) -> dict:
    """
    Validate that PyAV produces identical output to ffmpeg subprocess.

    Returns:
        dict with keys: match, ffmpeg_samples, pyav_samples, max_diff, mean_diff, time_ffmpeg, time_pyav
    """
    # Load with ffmpeg subprocess (reference)
    t0 = time.perf_counter()
    ffmpeg_audio = _load_audio_ffmpeg(audio_path)
    time_ffmpeg = (time.perf_counter() - t0) * 1000

    # Load with PyAV
    t0 = time.perf_counter()
    pyav_audio = _load_audio_pyav(audio_path)
    time_pyav = (time.perf_counter() - t0) * 1000

    # Compare
    len_ffmpeg = len(ffmpeg_audio)
    len_pyav = len(pyav_audio)
    len_diff = len_ffmpeg - len_pyav

    # Compare overlapping region
    min_len = min(len_ffmpeg, len_pyav)
    if min_len > 0:
        diff = np.abs(ffmpeg_audio[:min_len] - pyav_audio[:min_len])
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
    else:
        max_diff = float('inf')
        mean_diff = float('inf')

    # Check for exact match
    exact_match = (len_ffmpeg == len_pyav) and (max_diff < 1e-7)

    result = {
        'match': exact_match,
        'ffmpeg_samples': len_ffmpeg,
        'pyav_samples': len_pyav,
        'sample_diff': len_diff,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'time_ffmpeg_ms': time_ffmpeg,
        'time_pyav_ms': time_pyav,
        'speedup': time_ffmpeg / time_pyav if time_pyav > 0 else float('inf'),
    }

    if verbose:
        status = "MATCH" if exact_match else "MISMATCH"
        print(f"\n{audio_path}:")
        print(f"  Status: {status}")
        print(f"  ffmpeg samples: {len_ffmpeg}")
        print(f"  PyAV samples:   {len_pyav}")
        print(f"  Sample diff:    {len_diff:+d}")
        print(f"  Max value diff: {max_diff:.2e}")
        print(f"  Mean value diff: {mean_diff:.2e}")
        print(f"  ffmpeg time:    {time_ffmpeg:.2f}ms")
        print(f"  PyAV time:      {time_pyav:.2f}ms")
        print(f"  Speedup:        {result['speedup']:.1f}x")

        if not exact_match:
            # Analyze where differences are
            if len_diff != 0:
                print(f"\n  Sample count difference: {len_diff} samples ({len_diff / 16000 * 1000:.2f}ms)")
                if len_diff > 0:
                    print(f"  ffmpeg has {len_diff} MORE samples")
                    # Show tail samples
                    print(f"  ffmpeg tail (last 10): {ffmpeg_audio[-10:]}")
                else:
                    print(f"  PyAV has {-len_diff} MORE samples")
                    print(f"  PyAV tail (last 10): {pyav_audio[-10:]}")

            if max_diff > 1e-7:
                # Find where max diff occurs
                diff_full = np.abs(ffmpeg_audio[:min_len] - pyav_audio[:min_len])
                max_idx = np.argmax(diff_full)
                print(f"\n  Max diff at sample {max_idx}:")
                print(f"    ffmpeg: {ffmpeg_audio[max_idx]}")
                print(f"    PyAV:   {pyav_audio[max_idx]}")

    return result


def find_test_audio_files():
    """Find audio files for testing."""
    base = Path(__file__).parent.parent

    # Look in common locations
    search_paths = [
        base / "tests" / "fixtures" / "audio",
        base / "reports" / "audio",
        base / "voices",
        Path.home() / "voice" / "test_audio",
    ]

    extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    files = []

    for path in search_paths:
        if path.exists():
            for ext in extensions:
                files.extend(path.rglob(f"*{ext}"))

    return files[:20]  # Limit to 20 files


def main():
    if not _is_pyav_available():
        print("ERROR: PyAV not available. Install with: pip install av")
        sys.exit(1)

    print("=" * 60)
    print("PyAV Audio Loading Validation")
    print("=" * 60)

    # Find test files
    test_files = find_test_audio_files()

    if not test_files:
        print("\nNo test audio files found. Please provide audio paths as arguments.")
        print("Usage: python validate_pyav_audio.py [audio_file1] [audio_file2] ...")
        sys.exit(1)

    # Use command line args if provided
    if len(sys.argv) > 1:
        test_files = [Path(p) for p in sys.argv[1:]]

    print(f"\nTesting {len(test_files)} audio files...")

    results = []
    for audio_path in test_files:
        if audio_path.exists():
            try:
                result = validate_pyav_matches_ffmpeg(str(audio_path))
                result['file'] = str(audio_path)
                results.append(result)
            except Exception as e:
                print(f"\n{audio_path}: ERROR - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    matches = sum(1 for r in results if r['match'])
    total = len(results)

    print(f"\nMatches: {matches}/{total} ({100*matches/total:.1f}%)")

    if matches == total:
        print("\nPyAV produces IDENTICAL output to ffmpeg subprocess.")
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"Average speedup: {avg_speedup:.1f}x")
    else:
        print("\nWARNING: PyAV output differs from ffmpeg!")
        print("The following files have mismatches:")
        for r in results:
            if not r['match']:
                print(f"  - {r['file']}: {r['sample_diff']:+d} samples, max_diff={r['max_diff']:.2e}")


if __name__ == "__main__":
    main()
