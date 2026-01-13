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
Compare C++ Kokoro output against Python MLX reference.

This script loads:
1. Python reference from /tmp/kokoro_cpp_ref/audio.npy
2. C++ output from src/kokoro/token_input_output.wav

And computes error metrics.

Usage:
  python scripts/compare_cpp_python.py
  python scripts/compare_cpp_python.py --python-ref /tmp/kokoro_cpp_ref/audio.npy --cpp-wav src/kokoro/token_input_output.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav


def main():
    parser = argparse.ArgumentParser(description="Compare C++ vs Python Kokoro output")
    parser.add_argument(
        "--python-ref",
        type=Path,
        default=Path("/tmp/kokoro_cpp_ref/audio.npy"),
        help="Python reference .npy file",
    )
    parser.add_argument(
        "--cpp-wav",
        type=Path,
        default=Path("/Users/ayates/model_mlx_migration/src/kokoro/token_input_output.wav"),
        help="C++ output .wav file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Max abs threshold for pass/fail (default: 0.01)",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.99,
        help="Minimum correlation for pass/fail (default: 0.99)",
    )
    args = parser.parse_args()

    # Load Python reference
    if not args.python_ref.exists():
        print(f"ERROR: Python reference not found: {args.python_ref}")
        print("Run: python scripts/generate_cpp_reference.py")
        return 1

    py_audio = np.load(args.python_ref)
    print(f"Python reference: {args.python_ref}")
    print(f"  Samples: {len(py_audio)}")
    print(f"  RMS: {np.sqrt(np.mean(py_audio**2)):.6f}")

    # Load C++ output
    if not args.cpp_wav.exists():
        print(f"ERROR: C++ output not found: {args.cpp_wav}")
        print("Run: cd src/kokoro && ./test_token_input ../../kokoro_cpp_export af_bella <tokens>")
        return 1

    sr, cpp_audio_int16 = wav.read(args.cpp_wav)
    # Convert from int16 to float32
    cpp_audio = cpp_audio_int16.astype(np.float32) / 32767.0
    print(f"\nC++ output: {args.cpp_wav}")
    print(f"  Samples: {len(cpp_audio)}")
    print(f"  Sample rate: {sr}")
    print(f"  RMS: {np.sqrt(np.mean(cpp_audio**2)):.6f}")

    # Compare
    min_len = min(len(py_audio), len(cpp_audio))
    py_trim = py_audio[:min_len]
    cpp_trim = cpp_audio[:min_len]

    diff = np.abs(py_trim - cpp_trim)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rmse = float(np.sqrt(np.mean((py_trim - cpp_trim) ** 2)))

    # Correlation
    corr = 0.0
    if np.std(py_trim) > 0 and np.std(cpp_trim) > 0:
        corr = float(np.corrcoef(py_trim, cpp_trim)[0, 1])

    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}")
    print(f"Python samples: {len(py_audio)}")
    print(f"C++ samples:    {len(cpp_audio)}")
    print(f"Compared:       {min_len} samples")
    print(f"Max abs error:  {max_abs:.6f}")
    print(f"Mean abs error: {mean_abs:.6f}")
    print(f"RMSE:           {rmse:.6f}")
    print(f"Correlation:    {corr:.6f}")

    # Pass requires both max_abs AND correlation thresholds
    # Note: 0.06 max_abs is acceptable - error originates from BERT transformer
    # which has inherent floating-point accumulation across 12 layers.
    # See reports/main/error_source_investigation_2025-12-15.md
    max_abs_ok = max_abs <= args.threshold
    corr_ok = corr >= args.min_correlation
    passed = max_abs_ok and corr_ok

    print("\nThresholds:")
    print(f"  Max abs: {max_abs:.6f} <= {args.threshold} -> {'PASS' if max_abs_ok else 'FAIL'}")
    print(f"  Correlation: {corr:.6f} >= {args.min_correlation} -> {'PASS' if corr_ok else 'FAIL'}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    if not passed and corr_ok:
        # High correlation but exceeded max_abs - this is expected due to BERT error
        print("\nNote: High correlation (>0.99) with max_abs > 0.01 is expected.")
        print("Error originates from BERT transformer (0.005) and amplifies downstream.")
        print("See: reports/main/error_source_investigation_2025-12-15.md")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
