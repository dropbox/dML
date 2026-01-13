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

"""Test LSTM error accumulation from transpose pattern differences.

Hypothesis: 2.3e-5 error per LSTM step accumulates over 150 steps to ~0.008.
"""

import mlx.core as mx
import numpy as np


def lstm_step_cpp_pattern(
    x_t, h, c,
    W_ih, W_hh, b_ih, b_hh
):
    """LSTM step using C++ pattern (runtime transpose)."""
    gates = (mx.matmul(x_t, mx.transpose(W_ih)) +
             mx.matmul(h, mx.transpose(W_hh)) +
             b_ih + b_hh)
    i, f, g, o = mx.split(gates, 4, axis=-1)
    i = mx.sigmoid(i)
    f = mx.sigmoid(f)
    g = mx.tanh(g)
    o = mx.sigmoid(o)
    c_new = f * c + i * g
    h_new = o * mx.tanh(c_new)
    return h_new, c_new


def lstm_step_python_pattern(
    x_t, h, c,
    W_ih_T, W_hh_T, b_ih, b_hh
):
    """LSTM step using Python pattern (pre-transposed weights)."""
    gates = (mx.matmul(x_t, W_ih_T) +
             mx.matmul(h, W_hh_T) +
             b_ih + b_hh)
    i, f, g, o = mx.split(gates, 4, axis=-1)
    i = mx.sigmoid(i)
    f = mx.sigmoid(f)
    g = mx.tanh(g)
    o = mx.sigmoid(o)
    c_new = f * c + i * g
    h_new = o * mx.tanh(c_new)
    return h_new, c_new


def test_lstm_error_accumulation():
    """Test how error accumulates over an LSTM sequence."""
    np.random.seed(42)

    batch = 1
    seq_len = 150
    input_size = 512
    hidden_size = 512

    # Create data
    x = np.random.randn(batch, seq_len, input_size).astype(np.float32)
    x_mx = mx.array(x)

    # Weights
    W_ih = np.random.randn(4 * hidden_size, input_size).astype(np.float32)
    W_hh = np.random.randn(4 * hidden_size, hidden_size).astype(np.float32)
    b_ih = np.random.randn(4 * hidden_size).astype(np.float32)
    b_hh = np.random.randn(4 * hidden_size).astype(np.float32)

    W_ih_mx = mx.array(W_ih)
    W_hh_mx = mx.array(W_hh)
    W_ih_T_mx = mx.array(W_ih.T)  # Pre-transposed
    W_hh_T_mx = mx.array(W_hh.T)  # Pre-transposed
    b_ih_mx = mx.array(b_ih)
    b_hh_mx = mx.array(b_hh)

    print("Testing LSTM error accumulation:")
    print("=" * 60)

    # Run LSTM with C++ pattern
    h_cpp = mx.zeros((batch, hidden_size))
    c_cpp = mx.zeros((batch, hidden_size))
    outputs_cpp = []
    for t in range(seq_len):
        h_cpp, c_cpp = lstm_step_cpp_pattern(
            x_mx[:, t, :], h_cpp, c_cpp,
            W_ih_mx, W_hh_mx, b_ih_mx, b_hh_mx
        )
        outputs_cpp.append(h_cpp)
    output_cpp = mx.stack(outputs_cpp, axis=1)
    mx.eval(output_cpp)
    result_cpp = np.array(output_cpp)

    # Run LSTM with Python pattern
    h_py = mx.zeros((batch, hidden_size))
    c_py = mx.zeros((batch, hidden_size))
    outputs_py = []
    for t in range(seq_len):
        h_py, c_py = lstm_step_python_pattern(
            x_mx[:, t, :], h_py, c_py,
            W_ih_T_mx, W_hh_T_mx, b_ih_mx, b_hh_mx
        )
        outputs_py.append(h_py)
    output_py = mx.stack(outputs_py, axis=1)
    mx.eval(output_py)
    result_py = np.array(output_py)

    # Compute error at each timestep
    errors = np.abs(result_cpp - result_py).max(axis=(0, 2))  # Max error per timestep

    print(f"Shape: {result_cpp.shape}")
    print("\nError accumulation over sequence:")
    print(f"  Step   1: max_diff = {errors[0]:.6e}")
    print(f"  Step  25: max_diff = {errors[24]:.6e}")
    print(f"  Step  50: max_diff = {errors[49]:.6e}")
    print(f"  Step  75: max_diff = {errors[74]:.6e}")
    print(f"  Step 100: max_diff = {errors[99]:.6e}")
    print(f"  Step 125: max_diff = {errors[124]:.6e}")
    print(f"  Step 150: max_diff = {errors[149]:.6e}")

    # Overall stats
    max_diff = np.abs(result_cpp - result_py).max()
    mean_diff = np.abs(result_cpp - result_py).mean()

    print("\nOverall:")
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    return max_diff, errors


def test_bilstm_error_accumulation():
    """Test BiLSTM error accumulation (forward + backward)."""
    np.random.seed(42)

    batch = 1
    seq_len = 150
    input_size = 512
    hidden_size = 512

    x = np.random.randn(batch, seq_len, input_size).astype(np.float32)
    x_mx = mx.array(x)

    # Weights for both directions (same weights for simplicity)
    W_ih = np.random.randn(4 * hidden_size, input_size).astype(np.float32)
    W_hh = np.random.randn(4 * hidden_size, hidden_size).astype(np.float32)
    b_ih = np.random.randn(4 * hidden_size).astype(np.float32)
    b_hh = np.random.randn(4 * hidden_size).astype(np.float32)

    W_ih_mx = mx.array(W_ih)
    W_hh_mx = mx.array(W_hh)
    W_ih_T_mx = mx.array(W_ih.T)
    W_hh_T_mx = mx.array(W_hh.T)
    b_ih_mx = mx.array(b_ih)
    b_hh_mx = mx.array(b_hh)

    print("\n" + "=" * 60)
    print("Testing BiLSTM error accumulation:")
    print("=" * 60)

    # Forward - C++ pattern
    h = mx.zeros((batch, hidden_size))
    c = mx.zeros((batch, hidden_size))
    fwd_cpp = []
    for t in range(seq_len):
        h, c = lstm_step_cpp_pattern(x_mx[:, t, :], h, c, W_ih_mx, W_hh_mx, b_ih_mx, b_hh_mx)
        fwd_cpp.append(h)
    fwd_cpp = mx.stack(fwd_cpp, axis=1)

    # Backward - C++ pattern
    h = mx.zeros((batch, hidden_size))
    c = mx.zeros((batch, hidden_size))
    bwd_cpp = []
    for t in range(seq_len - 1, -1, -1):
        h, c = lstm_step_cpp_pattern(x_mx[:, t, :], h, c, W_ih_mx, W_hh_mx, b_ih_mx, b_hh_mx)
        bwd_cpp.insert(0, h)
    bwd_cpp = mx.stack(bwd_cpp, axis=1)

    bilstm_cpp = mx.concatenate([fwd_cpp, bwd_cpp], axis=-1)
    mx.eval(bilstm_cpp)
    result_cpp = np.array(bilstm_cpp)

    # Forward - Python pattern
    h = mx.zeros((batch, hidden_size))
    c = mx.zeros((batch, hidden_size))
    fwd_py = []
    for t in range(seq_len):
        h, c = lstm_step_python_pattern(x_mx[:, t, :], h, c, W_ih_T_mx, W_hh_T_mx, b_ih_mx, b_hh_mx)
        fwd_py.append(h)
    fwd_py = mx.stack(fwd_py, axis=1)

    # Backward - Python pattern
    h = mx.zeros((batch, hidden_size))
    c = mx.zeros((batch, hidden_size))
    bwd_py = []
    for t in range(seq_len - 1, -1, -1):
        h, c = lstm_step_python_pattern(x_mx[:, t, :], h, c, W_ih_T_mx, W_hh_T_mx, b_ih_mx, b_hh_mx)
        bwd_py.insert(0, h)
    bwd_py = mx.stack(bwd_py, axis=1)

    bilstm_py = mx.concatenate([fwd_py, bwd_py], axis=-1)
    mx.eval(bilstm_py)
    result_py = np.array(bilstm_py)

    # Compare
    max_diff = np.abs(result_cpp - result_py).max()
    mean_diff = np.abs(result_cpp - result_py).mean()

    print(f"Shape: {result_cpp.shape}")
    print("\nBiLSTM comparison:")
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    # Find where max diff is
    idx = np.unravel_index(np.argmax(np.abs(result_cpp - result_py)), result_cpp.shape)
    print(f"\nMax diff at index {idx}:")
    print(f"  C++:    {result_cpp[idx]:.6f}")
    print(f"  Python: {result_py[idx]:.6f}")

    return max_diff


if __name__ == "__main__":
    lstm_max_diff, errors = test_lstm_error_accumulation()
    bilstm_max_diff = test_bilstm_error_accumulation()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print(f"LSTM max accumulated error (150 steps): {lstm_max_diff:.6e}")
    print(f"BiLSTM max error: {bilstm_max_diff:.6e}")
    print("Expected x_shared error from reports: ~0.008")

    if lstm_max_diff > 0.001:
        print("\nThe transpose pattern difference EXPLAINS part of the x_shared error!")
        print(f"Observed: {lstm_max_diff:.6f} vs Expected: 0.008")
    else:
        print("\nThe transpose pattern difference is too small to explain the full error.")
        print("There must be additional differences beyond transpose patterns.")
