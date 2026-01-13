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

"""Test whether the float32 vs float64 cumsum difference explains error.

Current state:
- C++: Uses float64 cumsum on CPU
- Python: Uses float32 cumsum (default)

Question: Does the float32 vs float64 difference cause the 0.057 error floor?
"""

import math

import mlx.core as mx
import numpy as np


def test_cumsum_precision():
    """Test cumsum precision at different dtypes."""
    np.random.seed(42)
    batch = 1
    length = 155
    harmonics = 9

    # rad_values_down typically in range [0, 1]
    rad_values = np.random.rand(batch, length, harmonics).astype(np.float32)

    print("Testing cumsum precision:")
    print("=" * 60)

    # Method 1: Python default (float32 on GPU)
    rad_mx = mx.array(rad_values)
    phase_low_f32 = mx.cumsum(rad_mx, axis=1) * 2 * math.pi
    mx.eval(phase_low_f32)
    result_f32 = np.array(phase_low_f32)

    # Method 2: Numpy float64 (ground truth)
    rad_64 = rad_values.astype(np.float64)
    phase_64 = np.cumsum(rad_64, axis=1) * 2 * math.pi
    result_f64 = phase_64.astype(np.float32)

    # Method 3: Numpy float32 (should match MLX GPU)
    phase_32_np = np.cumsum(rad_values.astype(np.float32), axis=1) * 2 * np.float32(math.pi)
    result_f32_np = phase_32_np

    # Compare results
    diff_f32_vs_f64 = np.abs(result_f32 - result_f64).max()
    diff_f32_vs_np32 = np.abs(result_f32 - result_f32_np).max()

    print(f"Shape: {result_f32.shape}")
    print("\nMax absolute differences:")
    print(f"  MLX float32 vs NumPy float64:   {diff_f32_vs_f64:.6e}")
    print(f"  MLX float32 vs NumPy float32:   {diff_f32_vs_np32:.6e}")

    # Check last values (where cumsum differences accumulate)
    last_f32 = result_f32[0, -1, :]
    last_f64 = result_f64[0, -1, :]

    print("\nLast frame values (where cumsum accumulation is largest):")
    print(f"  MLX float32: mean={last_f32.mean():.6f}, max={last_f32.max():.6f}")
    print(f"  NumPy float64: mean={last_f64.mean():.6f}, max={last_f64.max():.6f}")

    print("\nDifference at last frame (float32 vs float64):")
    print(f"  Max: {np.abs(last_f32 - last_f64).max():.6e}")

    return diff_f32_vs_f64


def test_full_phase_pipeline():
    """Test the full phase computation pipeline to see error propagation."""
    np.random.seed(42)
    batch = 1
    length = 147     # Typical length for encoder output
    harmonics = 9
    total_upp = 256  # upsampling factor

    print("\n" + "=" * 60)
    print("Full phase pipeline test:")
    print("=" * 60)

    # Simulate rad_values_down
    rad_values_down = np.random.rand(batch, length, harmonics).astype(np.float32)
    rad_mx = mx.array(rad_values_down)

    # Python default method (current implementation - float32)
    phase_low = mx.cumsum(rad_mx, axis=1) * 2 * math.pi
    phase_scaled = phase_low * total_upp
    mx.eval(phase_scaled)
    result_py = np.array(phase_scaled)

    # C++ method simulation (float64, then back to float32)
    rad_64 = rad_values_down.astype(np.float64)
    phase_low_64 = np.cumsum(rad_64, axis=1) * 2 * math.pi
    phase_scaled_64 = phase_low_64 * total_upp
    result_cpp = phase_scaled_64.astype(np.float32)

    # Compare
    diff = np.abs(result_py - result_cpp)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Shape: {result_py.shape}")
    print(f"Max phase_scaled value: Python={result_py.max():.2f}, C++={result_cpp.max():.2f}")
    print("\nPhase difference (Python float32 vs C++ float64):")
    print(f"  Max:  {max_diff:.6f}")
    print(f"  Mean: {mean_diff:.6f}")

    # Check where max diff occurs
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"\nMax diff at index {idx}:")
    print(f"  Python: {result_py[idx]:.6f}")
    print(f"  C++:    {result_cpp[idx]:.6f}")

    # This is the key insight - the phase difference propagates through sin()
    sin_py = np.sin(result_py)
    sin_cpp = np.sin(result_cpp)
    sin_diff = np.abs(sin_py - sin_cpp).max()
    print("\nAfter sin() (final harmonics):")
    print(f"  Max diff: {sin_diff:.6f}")

    return max_diff, sin_diff


def test_with_actual_model_values():
    """Test with realistic values from the actual model."""
    print("\n" + "=" * 60)
    print("Test with realistic model values:")
    print("=" * 60)

    # These values are typical for Kokoro:
    # - F0 range: 50-500 Hz
    # - Sample rate: 24000 Hz
    # - length: ~150 frames for "Hello world"
    # - rad_values are normalized F0 / sample_rate

    np.random.seed(42)
    batch = 1
    length = 147
    harmonics = 9
    sample_rate = 24000

    # Simulate F0 values (in Hz)
    f0_base = 200 + 100 * np.random.randn(batch, length, 1)  # ~200 Hz ± 100
    f0_base = np.clip(f0_base, 50, 500).astype(np.float32)

    # Create harmonic ratios: [1, 2, 3, ..., 9]
    harmonic_ratios = np.arange(1, harmonics + 1).reshape(1, 1, harmonics).astype(np.float32)

    # rad_values = f0 * harmonic_ratio / sample_rate
    rad_values = f0_base * harmonic_ratios / sample_rate

    print(f"rad_values range: [{rad_values.min():.6f}, {rad_values.max():.6f}]")

    # Compare float32 vs float64 cumsum
    rad_mx = mx.array(rad_values)

    # Float32 (Python default)
    phase_f32 = mx.cumsum(rad_mx, axis=1) * 2 * math.pi
    mx.eval(phase_f32)
    result_f32 = np.array(phase_f32)

    # Float64 (C++ approach)
    rad_64 = rad_values.astype(np.float64)
    phase_f64 = np.cumsum(rad_64, axis=1) * 2 * math.pi
    result_f64 = phase_f64.astype(np.float32)

    # Compare
    diff = np.abs(result_f32 - result_f64)
    print("\nPhase difference (float32 vs float64):")
    print(f"  Max: {diff.max():.6f}")
    print(f"  Mean: {diff.mean():.6f}")

    # After sin (final harmonics)
    sin_f32 = np.sin(result_f32)
    sin_f64 = np.sin(result_f64)
    sin_diff = np.abs(sin_f32 - sin_f64).max()
    print("\nAfter sin():")
    print(f"  Max diff: {sin_diff:.6f}")

    return diff.max(), sin_diff


if __name__ == "__main__":
    diff1 = test_cumsum_precision()
    phase_diff, sin_diff = test_full_phase_pipeline()
    phase_real, sin_real = test_with_actual_model_values()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print(f"Cumsum precision difference (float32 vs float64): {diff1:.6e}")
    print(f"Phase pipeline difference: {phase_diff:.4f}")
    print(f"Final harmonics difference after sin(): {sin_diff:.6f}")
    print(f"With realistic values - sin diff: {sin_real:.6f}")

    if sin_diff > 0.01:
        print("\nThe float32 vs float64 cumsum difference CONTRIBUTES to the error.")
        print("But it may not explain the full 0.057 error floor.")
    else:
        print("\nThe cumsum precision difference does NOT explain the error.")
        print("There must be additional algorithmic differences.")
