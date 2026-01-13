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

"""Test whether mx.eval() placement affects numerical results.

Hypothesis: C++ has more mx::eval() calls than Python, which forces different
intermediate materializations and causes the 0.008 BiLSTM error.

Test: Compare Python LSTM with lazy evaluation vs eager evaluation (many evals).
"""

import mlx.core as mx
import numpy as np


def single_lstm_step_lazy(
    x_t: mx.array,
    h_t: mx.array,
    c_t: mx.array,
    weight_ih: mx.array,
    weight_hh: mx.array,
    bias_ih: mx.array,
    bias_hh: mx.array,
) -> tuple[mx.array, mx.array]:
    """Single LSTM step with lazy evaluation (Python default)."""
    # All in one lazy graph
    gates = x_t @ weight_ih.T + h_t @ weight_hh.T + bias_ih + bias_hh
    i, f, g, o = mx.split(gates, 4, axis=-1)
    i = mx.sigmoid(i)
    f = mx.sigmoid(f)
    g = mx.tanh(g)
    o = mx.sigmoid(o)
    c_new = f * c_t + i * g
    h_new = o * mx.tanh(c_new)
    return h_new, c_new


def single_lstm_step_eager(
    x_t: mx.array,
    h_t: mx.array,
    c_t: mx.array,
    weight_ih: mx.array,
    weight_hh: mx.array,
    bias_ih: mx.array,
    bias_hh: mx.array,
) -> tuple[mx.array, mx.array]:
    """Single LSTM step with eager evaluation (matching C++ pattern)."""
    # Force evaluation at each step to match C++ pattern
    gates_ih = x_t @ weight_ih.T
    mx.eval(gates_ih)  # C++ pattern

    gates_hh = h_t @ weight_hh.T
    mx.eval(gates_hh)  # C++ pattern

    gates = gates_ih + gates_hh + bias_ih + bias_hh
    mx.eval(gates)  # C++ pattern

    i, f, g, o = mx.split(gates, 4, axis=-1)

    i = mx.sigmoid(i)
    f = mx.sigmoid(f)
    g = mx.tanh(g)
    o = mx.sigmoid(o)
    mx.eval(i, f, g, o)  # C++ pattern

    c_new = f * c_t + i * g
    mx.eval(c_new)  # C++ pattern

    h_new = o * mx.tanh(c_new)
    mx.eval(h_new)  # C++ pattern

    return h_new, c_new


def test_lstm_eval_patterns():
    """Compare lazy vs eager LSTM evaluation."""
    np.random.seed(42)

    batch_size = 1
    seq_len = 150  # Typical for Kokoro
    input_size = 512
    hidden_size = 512

    # Create random input
    x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
    x_mx = mx.array(x)

    # Create random weights (matching MLX LSTM format)
    weight_ih = np.random.randn(4 * hidden_size, input_size).astype(np.float32)
    weight_hh = np.random.randn(4 * hidden_size, hidden_size).astype(np.float32)
    bias_ih = np.random.randn(4 * hidden_size).astype(np.float32)
    bias_hh = np.random.randn(4 * hidden_size).astype(np.float32)

    weight_ih_mx = mx.array(weight_ih)
    weight_hh_mx = mx.array(weight_hh)
    bias_ih_mx = mx.array(bias_ih)
    bias_hh_mx = mx.array(bias_hh)

    print("Testing LSTM evaluation patterns:")
    print("=" * 60)

    # Method 1: Lazy evaluation
    h_lazy = mx.zeros((batch_size, hidden_size))
    c_lazy = mx.zeros((batch_size, hidden_size))
    outputs_lazy = []

    for t in range(seq_len):
        h_lazy, c_lazy = single_lstm_step_lazy(
            x_mx[:, t, :], h_lazy, c_lazy,
            weight_ih_mx, weight_hh_mx, bias_ih_mx, bias_hh_mx
        )
        outputs_lazy.append(h_lazy)

    output_lazy = mx.stack(outputs_lazy, axis=1)
    mx.eval(output_lazy)
    result_lazy = np.array(output_lazy)

    # Method 2: Eager evaluation (matching C++ pattern)
    h_eager = mx.zeros((batch_size, hidden_size))
    c_eager = mx.zeros((batch_size, hidden_size))
    outputs_eager = []

    for t in range(seq_len):
        h_eager, c_eager = single_lstm_step_eager(
            x_mx[:, t, :], h_eager, c_eager,
            weight_ih_mx, weight_hh_mx, bias_ih_mx, bias_hh_mx
        )
        outputs_eager.append(h_eager)

    output_eager = mx.stack(outputs_eager, axis=1)
    mx.eval(output_eager)
    result_eager = np.array(output_eager)

    # Compare
    diff = np.abs(result_lazy - result_eager)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Shape: {result_lazy.shape}")
    print("\nLazy vs Eager LSTM comparison:")
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if max_diff > 0:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\nMax diff at index {idx}:")
        print(f"  Lazy:  {result_lazy[idx]:.6f}")
        print(f"  Eager: {result_eager[idx]:.6f}")

    # Check last hidden state (most accumulated error)
    last_diff = np.abs(result_lazy[:, -1, :] - result_eager[:, -1, :])
    print("\nLast hidden state:")
    print(f"  Max diff: {last_diff.max():.6e}")

    return max_diff


def test_bilstm_eval_patterns():
    """Test BiLSTM with different eval patterns."""
    np.random.seed(42)

    batch_size = 1
    seq_len = 150
    input_size = 512
    hidden_size = 512

    # Create random input
    x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
    x_mx = mx.array(x)

    # Create random weights
    weight_ih = np.random.randn(4 * hidden_size, input_size).astype(np.float32)
    weight_hh = np.random.randn(4 * hidden_size, hidden_size).astype(np.float32)
    bias_ih = np.random.randn(4 * hidden_size).astype(np.float32)
    bias_hh = np.random.randn(4 * hidden_size).astype(np.float32)

    weight_ih_mx = mx.array(weight_ih)
    weight_hh_mx = mx.array(weight_hh)
    bias_ih_mx = mx.array(bias_ih)
    bias_hh_mx = mx.array(bias_hh)

    print("\n" + "=" * 60)
    print("Testing BiLSTM evaluation patterns:")
    print("=" * 60)

    # Forward pass - lazy
    h = mx.zeros((batch_size, hidden_size))
    c = mx.zeros((batch_size, hidden_size))
    fwd_outputs_lazy = []
    for t in range(seq_len):
        h, c = single_lstm_step_lazy(
            x_mx[:, t, :], h, c,
            weight_ih_mx, weight_hh_mx, bias_ih_mx, bias_hh_mx
        )
        fwd_outputs_lazy.append(h)
    fwd_lazy = mx.stack(fwd_outputs_lazy, axis=1)

    # Backward pass - lazy (reversed input)
    h = mx.zeros((batch_size, hidden_size))
    c = mx.zeros((batch_size, hidden_size))
    bwd_outputs_lazy = []
    for t in range(seq_len - 1, -1, -1):
        h, c = single_lstm_step_lazy(
            x_mx[:, t, :], h, c,
            weight_ih_mx, weight_hh_mx, bias_ih_mx, bias_hh_mx
        )
        bwd_outputs_lazy.insert(0, h)
    bwd_lazy = mx.stack(bwd_outputs_lazy, axis=1)

    # Concatenate - lazy
    bilstm_lazy = mx.concatenate([fwd_lazy, bwd_lazy], axis=-1)
    mx.eval(bilstm_lazy)
    result_lazy = np.array(bilstm_lazy)

    # Forward pass - eager
    h = mx.zeros((batch_size, hidden_size))
    c = mx.zeros((batch_size, hidden_size))
    fwd_outputs_eager = []
    for t in range(seq_len):
        h, c = single_lstm_step_eager(
            x_mx[:, t, :], h, c,
            weight_ih_mx, weight_hh_mx, bias_ih_mx, bias_hh_mx
        )
        fwd_outputs_eager.append(h)
    fwd_eager = mx.stack(fwd_outputs_eager, axis=1)
    mx.eval(fwd_eager)  # Extra eval

    # Backward pass - eager (reversed input)
    h = mx.zeros((batch_size, hidden_size))
    c = mx.zeros((batch_size, hidden_size))
    bwd_outputs_eager = []
    for t in range(seq_len - 1, -1, -1):
        h, c = single_lstm_step_eager(
            x_mx[:, t, :], h, c,
            weight_ih_mx, weight_hh_mx, bias_ih_mx, bias_hh_mx
        )
        bwd_outputs_eager.insert(0, h)
    bwd_eager = mx.stack(bwd_outputs_eager, axis=1)
    mx.eval(bwd_eager)  # Extra eval

    # Concatenate - eager
    bilstm_eager = mx.concatenate([fwd_eager, bwd_eager], axis=-1)
    mx.eval(bilstm_eager)
    result_eager = np.array(bilstm_eager)

    # Compare
    diff = np.abs(result_lazy - result_eager)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Shape: {result_lazy.shape}")
    print("\nLazy vs Eager BiLSTM comparison:")
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if max_diff > 0:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\nMax diff at index {idx}:")
        print(f"  Lazy:  {result_lazy[idx]:.6f}")
        print(f"  Eager: {result_eager[idx]:.6f}")

    return max_diff


if __name__ == "__main__":
    lstm_diff = test_lstm_eval_patterns()
    bilstm_diff = test_bilstm_eval_patterns()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print(f"LSTM lazy vs eager max diff: {lstm_diff:.6e}")
    print(f"BiLSTM lazy vs eager max diff: {bilstm_diff:.6e}")

    if max(lstm_diff, bilstm_diff) > 1e-5:
        print("\nmx.eval() placement DOES affect numerical results!")
        print("The C++ eager evaluation pattern is a potential cause of the error.")
    else:
        print("\nmx.eval() placement does NOT affect numerical results significantly.")
        print("The error must come from another source.")
