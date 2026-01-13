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

"""Test whether matmul transpose order affects numerical results.

C++ uses: mx::matmul(x, mx::transpose(W))
Python may have pre-transposed weights: mx.matmul(x, W_transposed)

Test if these produce different results.
"""

import mlx.core as mx
import numpy as np


def test_transpose_methods():
    """Compare different transpose methods in matmul."""
    np.random.seed(42)

    batch = 1
    seq_len = 150
    input_size = 512
    hidden_size = 2048  # 4 * 512 for LSTM gates

    # Create random input and weight
    x = np.random.randn(batch, seq_len, input_size).astype(np.float32)
    W = np.random.randn(hidden_size, input_size).astype(np.float32)

    x_mx = mx.array(x)
    W_mx = mx.array(W)

    print("Testing transpose methods in matmul:")
    print("=" * 60)

    # Method 1: Runtime transpose (C++ pattern)
    # matmul(x, transpose(W)) where W is [hidden, input]
    result1 = mx.matmul(x_mx, mx.transpose(W_mx))
    mx.eval(result1)
    r1 = np.array(result1)

    # Method 2: Pre-transposed weight (Python pattern)
    # matmul(x, W.T) where W.T is pre-computed
    W_T_mx = mx.array(W.T)  # Pre-transpose in numpy/python
    result2 = mx.matmul(x_mx, W_T_mx)
    mx.eval(result2)
    r2 = np.array(result2)

    # Method 3: Direct MLX transpose as intermediate
    W_T_mx2 = mx.transpose(W_mx)
    mx.eval(W_T_mx2)  # Force materialization
    result3 = mx.matmul(x_mx, W_T_mx2)
    mx.eval(result3)
    r3 = np.array(result3)

    # Compare
    print(f"Shape: {r1.shape}")

    diff12 = np.abs(r1 - r2).max()
    diff13 = np.abs(r1 - r3).max()
    diff23 = np.abs(r2 - r3).max()

    print("\nMax absolute differences:")
    print(f"  Runtime transpose vs pre-transposed:         {diff12:.6e}")
    print(f"  Runtime transpose vs materialized transpose: {diff13:.6e}")
    print(f"  Pre-transposed vs materialized transpose:    {diff23:.6e}")

    # Also test with numpy for ground truth
    r_np = np.matmul(x, W.T)
    diff_mlx_np = np.abs(r1 - r_np).max()
    print(f"  MLX vs NumPy ground truth:                   {diff_mlx_np:.6e}")

    return max(diff12, diff13, diff23)


def test_lstm_with_transpose_patterns():
    """Test full LSTM step with different transpose patterns."""
    np.random.seed(42)

    batch = 1
    input_size = 512
    hidden_size = 512

    # Create data
    x_t = np.random.randn(batch, input_size).astype(np.float32)
    h = np.random.randn(batch, hidden_size).astype(np.float32)
    c = np.random.randn(batch, hidden_size).astype(np.float32)

    # Weights: [4*hidden, input] for W_ih, [4*hidden, hidden] for W_hh
    W_ih = np.random.randn(4 * hidden_size, input_size).astype(np.float32)
    W_hh = np.random.randn(4 * hidden_size, hidden_size).astype(np.float32)
    b_ih = np.random.randn(4 * hidden_size).astype(np.float32)
    b_hh = np.random.randn(4 * hidden_size).astype(np.float32)

    x_mx = mx.array(x_t)
    h_mx = mx.array(h)
    mx.array(c)
    W_ih_mx = mx.array(W_ih)
    W_hh_mx = mx.array(W_hh)
    b_ih_mx = mx.array(b_ih)
    b_hh_mx = mx.array(b_hh)

    print("\n" + "=" * 60)
    print("Testing LSTM step with different transpose patterns:")
    print("=" * 60)

    # Pattern 1: C++ style (runtime transpose)
    gates1 = (mx.matmul(x_mx, mx.transpose(W_ih_mx)) +
              mx.matmul(h_mx, mx.transpose(W_hh_mx)) +
              b_ih_mx + b_hh_mx)
    mx.eval(gates1)
    g1 = np.array(gates1)

    # Pattern 2: Python style (pre-transposed)
    W_ih_T_mx = mx.array(W_ih.T)
    W_hh_T_mx = mx.array(W_hh.T)
    gates2 = (mx.matmul(x_mx, W_ih_T_mx) +
              mx.matmul(h_mx, W_hh_T_mx) +
              b_ih_mx + b_hh_mx)
    mx.eval(gates2)
    g2 = np.array(gates2)

    # Pattern 3: NumPy ground truth
    g_np = (np.matmul(x_t, W_ih.T) +
            np.matmul(h, W_hh.T) +
            b_ih + b_hh)

    diff12 = np.abs(g1 - g2).max()
    diff1_np = np.abs(g1 - g_np).max()
    diff2_np = np.abs(g2 - g_np).max()

    print(f"Shape: {g1.shape}")
    print("\nMax absolute differences:")
    print(f"  C++ pattern vs Python pattern:  {diff12:.6e}")
    print(f"  C++ pattern vs NumPy:           {diff1_np:.6e}")
    print(f"  Python pattern vs NumPy:        {diff2_np:.6e}")

    return diff12


if __name__ == "__main__":
    diff1 = test_transpose_methods()
    diff2 = test_lstm_with_transpose_patterns()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print(f"Transpose method max diff: {diff1:.6e}")
    print(f"LSTM transpose pattern max diff: {diff2:.6e}")

    if max(diff1, diff2) > 1e-5:
        print("\nTranspose method DOES affect numerical results!")
        print("This could explain the C++ vs Python difference.")
    else:
        print("\nTranspose method does NOT significantly affect results.")
        print("The error must come from another source.")
