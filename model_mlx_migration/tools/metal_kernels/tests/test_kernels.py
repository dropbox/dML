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
Tests for custom Metal kernels.

Run with: python -m pytest tools/metal_kernels/tests/test_kernels.py -v
"""

import sys

import mlx.core as mx

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

from tools.metal_kernels.ops import (
    benchmark_kernel,
    fused_adain_conv1d_baseline,
    fused_instance_norm_baseline,
    int4_linear_baseline,
    int8_linear_baseline,
    snake1d_baseline,
    verify_numerical_accuracy,
)


class TestSnake1D:
    """Tests for Snake1D activation kernel."""

    def test_baseline_correctness(self):
        """Test that baseline Snake1D produces correct output."""
        x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # [B=2, C=3]
        alpha = mx.array([1.0, 2.0, 0.5])  # [C=3]

        out = snake1d_baseline(x, alpha)
        mx.eval(out)

        # Manual calculation for x[0, 0]: 1.0 + (1/1.0) * sin(1.0*1.0)^2
        # = 1.0 + sin(1.0)^2 = 1.0 + 0.708... = 1.708...
        expected_00 = 1.0 + (1.0 / 1.0) * (mx.sin(mx.array(1.0)) ** 2)
        assert abs(float(out[0, 0]) - float(expected_00)) < 1e-5

    def test_baseline_shape(self):
        """Test that baseline preserves shape."""
        for shape in [(4,), (4, 8), (2, 16, 8), (1, 32, 64, 8)]:
            x = mx.random.normal(shape)
            alpha = mx.random.uniform(0.1, 2.0, shape=(shape[-1],))
            out = snake1d_baseline(x, alpha)
            mx.eval(out)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_baseline_dtypes(self):
        """Test baseline works with different dtypes."""
        for dtype in [mx.float32, mx.float16]:
            x = mx.random.normal((4, 8)).astype(dtype)
            alpha = mx.random.uniform(0.1, 2.0, shape=(8,)).astype(dtype)
            out = snake1d_baseline(x, alpha)
            mx.eval(out)
            assert out.dtype == dtype


class TestInstanceNorm:
    """Tests for Instance Normalization kernel."""

    def test_baseline_correctness(self):
        """Test baseline instance norm produces normalized output."""
        B, L, C = 2, 16, 8
        x = mx.random.normal((B, L, C))  # [B, L, C]
        # gamma/beta are per-batch, per-channel for AdaIN style
        gamma = mx.ones((B, C))
        beta = mx.zeros((B, C))

        out = fused_instance_norm_baseline(x, gamma, beta)
        mx.eval(out)

        # After instance norm, each (batch, channel) should have ~0 mean, ~1 var
        mean = mx.mean(out, axis=1)
        var = mx.var(out, axis=1)
        mx.eval(mean, var)

        assert mx.allclose(mean, mx.zeros_like(mean), atol=1e-5)
        assert mx.allclose(var, mx.ones_like(var), atol=1e-4)

    def test_baseline_shape(self):
        """Test baseline preserves shape."""
        B, L, C = 4, 32, 16
        x = mx.random.normal((B, L, C))
        gamma = mx.ones((B, C))
        beta = mx.zeros((B, C))

        out = fused_instance_norm_baseline(x, gamma, beta)
        mx.eval(out)
        assert out.shape == x.shape


class TestAdaINConv:
    """Tests for Fused AdaIN + Conv1d kernel."""

    def test_baseline_shape(self):
        """Test baseline produces correct output shape."""
        B, L, C_in, C_out, K = 2, 32, 16, 24, 3

        x = mx.random.normal((B, L, C_in))
        weight = mx.random.normal((C_out, K, C_in))
        gamma = mx.random.uniform(0.5, 1.5, shape=(B, C_in))
        beta = mx.random.uniform(-0.5, 0.5, shape=(B, C_in))

        out = fused_adain_conv1d_baseline(x, weight, gamma, beta, kernel_size=K)
        mx.eval(out)

        # Output length: L - K + 1 (no padding)
        expected_L_out = L - K + 1
        assert out.shape == (B, expected_L_out, C_out), f"Shape: {out.shape}"

    def test_baseline_with_padding(self):
        """Test baseline with padding."""
        B, L, C_in, C_out, K = 2, 32, 16, 24, 3

        x = mx.random.normal((B, L, C_in))
        weight = mx.random.normal((C_out, K, C_in))
        gamma = mx.random.uniform(0.5, 1.5, shape=(B, C_in))
        beta = mx.random.uniform(-0.5, 0.5, shape=(B, C_in))

        # Same padding
        padding = K // 2
        out = fused_adain_conv1d_baseline(
            x, weight, gamma, beta, kernel_size=K, padding=padding,
        )
        mx.eval(out)

        # With same padding, output length should be L
        assert out.shape == (B, L, C_out), f"Shape: {out.shape}"


class TestINT8Linear:
    """Tests for INT8 quantized linear kernel."""

    def test_baseline_correctness(self):
        """Test INT8 baseline produces correct output."""
        B, K, N = 4, 64, 32

        x = mx.random.normal((B, K))

        # Create quantized weights
        weight_fp = mx.random.normal((N, K))
        # Quantize to INT8
        scales = mx.max(mx.abs(weight_fp), axis=1, keepdims=True) / 127.0
        scales = mx.squeeze(scales)  # Remove keepdims for scale shape
        zeros = mx.zeros((N,))
        weight_q = mx.round(weight_fp / scales[:, None]).astype(mx.int8)

        # Dequantized output
        out = int8_linear_baseline(x, weight_q, scales, zeros)

        # Reference: x @ weight_fp.T
        out_ref = x @ weight_fp.T
        mx.eval(out, out_ref)

        # Should be close (quantization error) - INT8 has limited precision
        diff = mx.abs(out - out_ref)
        max_diff = float(mx.max(diff))
        assert max_diff < 0.5, f"Max diff {max_diff} too large"

    def test_baseline_shape(self):
        """Test INT8 baseline shape."""
        B, K, N = 8, 128, 64

        x = mx.random.normal((B, K))
        weight_q = mx.zeros((N, K), dtype=mx.int8)
        scales = mx.ones((N,))
        zeros = mx.zeros((N,))

        out = int8_linear_baseline(x, weight_q, scales, zeros)
        mx.eval(out)
        assert out.shape == (B, N)


class TestINT4Linear:
    """Tests for INT4 quantized linear kernel."""

    def test_baseline_shape(self):
        """Test INT4 baseline shape."""
        B, K, N = 4, 128, 64  # K must be even for INT4

        x = mx.random.normal((B, K))
        # Packed weights: 2 INT4 per byte
        weight_q = mx.zeros((N, K // 2), dtype=mx.uint8)
        # Group-wise scales
        group_size = 32
        num_groups = K // group_size
        scales = mx.ones((N, num_groups))
        zeros = mx.zeros((N, num_groups))

        out = int4_linear_baseline(x, weight_q, scales, zeros, group_size=group_size)
        mx.eval(out)
        assert out.shape == (B, N)


class TestBenchmark:
    """Tests for benchmark utilities."""

    def test_benchmark_function(self):
        """Test benchmark utility works."""
        # Use larger input to reduce timing variance
        x = mx.random.normal((16, 256))
        alpha = mx.random.uniform(0.1, 2.0, shape=(256,))

        result = benchmark_kernel(
            kernel_fn=snake1d_baseline,
            baseline_fn=snake1d_baseline,
            inputs=[x, alpha],
            warmup=5,
            iterations=20,
            name="snake1d_self",
        )

        assert "speedup" in result
        assert "custom_ms" in result
        assert "baseline_ms" in result
        # Same function should have ~1.0x speedup (widen tolerance for timing variance)
        assert 0.5 < result["speedup"] < 2.0

    def test_verify_accuracy(self):
        """Test accuracy verification utility."""
        x = mx.random.normal((4, 8))
        alpha = mx.random.uniform(0.1, 2.0, shape=(8,))

        result = verify_numerical_accuracy(
            kernel_fn=snake1d_baseline,
            baseline_fn=snake1d_baseline,
            inputs=[x, alpha],
            name="snake1d_self",
        )

        assert result["is_close"]
        assert result["max_diff"] < 1e-10


if __name__ == "__main__":
    # Run quick smoke test
    print("Running smoke tests...")

    # Snake1D
    x = mx.random.normal((4, 8))
    alpha = mx.random.uniform(0.1, 2.0, shape=(8,))
    out = snake1d_baseline(x, alpha)
    mx.eval(out)
    print(f"Snake1D baseline: input {x.shape} -> output {out.shape}")

    # Instance Norm
    x = mx.random.normal((2, 16, 8))
    gamma = mx.ones((8,))
    beta = mx.zeros((8,))
    out = fused_instance_norm_baseline(x, gamma, beta)
    mx.eval(out)
    print(f"InstanceNorm baseline: input {x.shape} -> output {out.shape}")

    # AdaIN + Conv
    x = mx.random.normal((2, 32, 16))
    weight = mx.random.normal((24, 3, 16))
    gamma = mx.random.uniform(0.5, 1.5, shape=(2, 16))
    beta = mx.random.uniform(-0.5, 0.5, shape=(2, 16))
    out = fused_adain_conv1d_baseline(x, weight, gamma, beta)
    mx.eval(out)
    print(f"AdaIN+Conv baseline: input {x.shape} -> output {out.shape}")

    # INT8 Linear
    x = mx.random.normal((4, 64))
    weight_q = mx.zeros((32, 64), dtype=mx.int8)
    scales = mx.ones((32,))
    zeros = mx.zeros((32,))
    out = int8_linear_baseline(x, weight_q, scales, zeros)
    mx.eval(out)
    print(f"INT8 Linear baseline: input {x.shape} -> output {out.shape}")

    print("\nAll smoke tests passed!")
