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
Tests for MLX implementation of Zipformer scaling modules.

Validates numerical equivalence against PyTorch reference implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add icefall to path for PyTorch reference
icefall_zipformer_path = Path("tools/third_party/icefall/egs/librispeech/ASR/zipformer")
if icefall_zipformer_path.exists():
    sys.path.insert(0, str(icefall_zipformer_path))


class TestBiasNorm:
    """Test BiasNorm MLX implementation against PyTorch reference."""

    @pytest.mark.zipformer
    @pytest.mark.numerical_validation
    def test_bias_norm_forward_basic(self, compare_arrays):
        """Test BiasNorm forward pass with basic input."""
        import mlx.core as mx

        from src.models.zipformer.scaling import BiasNorm as BiasNormMLX

        # Create MLX BiasNorm
        num_channels = 256
        mlx_bn = BiasNormMLX(num_channels=num_channels, channel_dim=-1)

        # Create test input
        batch_size, seq_len = 2, 10
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((batch_size, seq_len, num_channels)).astype(np.float32)

        # MLX forward
        x_mlx = mx.array(x_np)
        # Override bias and log_scale with fixed values for reproducibility
        mlx_bn.bias = mx.zeros((num_channels,))
        mlx_bn.log_scale = mx.array(0.0)  # exp(0) = 1

        out_mlx = mlx_bn(x_mlx)
        mx.eval(out_mlx)

        # Manual computation for reference
        # scale = mean(x^2)^(-0.5) * 1.0
        variance = np.mean(x_np ** 2, axis=-1, keepdims=True)
        scale = 1.0 / np.sqrt(variance + 1e-8)
        expected = x_np * scale

        result = compare_arrays(expected, out_mlx, atol=1e-5, rtol=1e-5)
        assert result.is_close, f"BiasNorm basic test failed: {result}"

    @pytest.mark.zipformer
    @pytest.mark.numerical_validation
    def test_bias_norm_with_bias(self, compare_arrays):
        """Test BiasNorm with non-zero bias."""
        import mlx.core as mx

        from src.models.zipformer.scaling import BiasNorm as BiasNormMLX

        num_channels = 128
        mlx_bn = BiasNormMLX(num_channels=num_channels, channel_dim=-1)

        batch_size, seq_len = 4, 8
        rng = np.random.default_rng(123)
        x_np = rng.standard_normal((batch_size, seq_len, num_channels)).astype(np.float32)
        bias_np = rng.standard_normal(num_channels).astype(np.float32) * 0.1
        log_scale = 0.5

        x_mlx = mx.array(x_np)
        mlx_bn.bias = mx.array(bias_np)
        mlx_bn.log_scale = mx.array(log_scale)

        out_mlx = mlx_bn(x_mlx)
        mx.eval(out_mlx)

        # Manual computation
        centered = x_np - bias_np
        variance = np.mean(centered ** 2, axis=-1, keepdims=True)
        scale = 1.0 / np.sqrt(variance + 1e-8) * np.exp(log_scale)
        expected = x_np * scale

        result = compare_arrays(expected, out_mlx, atol=1e-5, rtol=1e-5)
        assert result.is_close, f"BiasNorm with bias test failed: {result}"

    @pytest.mark.zipformer
    @pytest.mark.numerical_validation
    def test_bias_norm_channel_dim_0(self, compare_arrays):
        """Test BiasNorm with channel_dim=0."""
        import mlx.core as mx

        from src.models.zipformer.scaling import BiasNorm as BiasNormMLX

        num_channels = 64
        mlx_bn = BiasNormMLX(num_channels=num_channels, channel_dim=0)

        rng = np.random.default_rng(456)
        x_np = rng.standard_normal((num_channels, 4, 8)).astype(np.float32)

        x_mlx = mx.array(x_np)
        mlx_bn.bias = mx.zeros((num_channels,))
        mlx_bn.log_scale = mx.array(0.0)

        out_mlx = mlx_bn(x_mlx)
        mx.eval(out_mlx)

        # Manual computation with axis=0
        variance = np.mean(x_np ** 2, axis=0, keepdims=True)
        scale = 1.0 / np.sqrt(variance + 1e-8)
        expected = x_np * scale

        result = compare_arrays(expected, out_mlx, atol=1e-5, rtol=1e-5)
        assert result.is_close, f"BiasNorm channel_dim=0 test failed: {result}"

    @pytest.mark.zipformer
    @pytest.mark.numerical_validation
    @pytest.mark.skipif(
        not Path("tools/third_party/icefall/egs/librispeech/ASR/zipformer/scaling.py").exists(),
        reason="icefall not available for PyTorch reference",
    )
    def test_bias_norm_pytorch_equivalence(self, compare_arrays):
        """Test BiasNorm against PyTorch reference implementation.

        Note: Requires k2 library for PyTorch reference. Skip if not available.
        """
        pytest.importorskip("k2", reason="k2 required for PyTorch reference")
        import mlx.core as mx
        import torch

        # Import PyTorch reference
        from scaling import BiasNorm as BiasNormPT

        from src.models.zipformer.scaling import BiasNorm as BiasNormMLX

        num_channels = 256
        batch_size, seq_len = 2, 10

        # Create PyTorch BiasNorm
        pt_bn = BiasNormPT(num_channels=num_channels, channel_dim=-1)
        pt_bn.eval()  # Set to eval mode

        # Create MLX BiasNorm with same parameters
        mlx_bn = BiasNormMLX(num_channels=num_channels, channel_dim=-1)

        # Copy parameters from PyTorch to MLX
        mlx_bn.bias = mx.array(pt_bn.bias.detach().numpy())
        mlx_bn.log_scale = mx.array(pt_bn.log_scale.detach().numpy())

        # Create test input
        rng = np.random.default_rng(789)
        x_np = rng.standard_normal((batch_size, seq_len, num_channels)).astype(np.float32)

        # PyTorch forward
        x_pt = torch.tensor(x_np)
        with torch.no_grad():
            out_pt = pt_bn(x_pt)

        # MLX forward
        x_mlx = mx.array(x_np)
        out_mlx = mlx_bn(x_mlx)
        mx.eval(out_mlx)

        result = compare_arrays(out_pt, out_mlx, atol=1e-4, rtol=1e-4)
        assert result.is_close, f"BiasNorm PyTorch equivalence test failed: {result}"


class TestScaledLinear:
    """Test ScaledLinear MLX implementation."""

    @pytest.mark.zipformer
    def test_scaled_linear_forward(self, compare_arrays):
        """Test ScaledLinear forward pass."""
        import mlx.core as mx

        from src.models.zipformer.scaling import ScaledLinear

        in_features, out_features = 128, 256
        sl = ScaledLinear(in_features, out_features, initial_scale=0.5)

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 10, in_features)).astype(np.float32)
        x_mlx = mx.array(x_np)

        out_mlx = sl(x_mlx)
        mx.eval(out_mlx)

        # Just verify output shape and that it runs
        assert out_mlx.shape == (2, 10, out_features)


class TestActivations:
    """Test activation functions."""

    @pytest.mark.zipformer
    def test_swoosh_l(self, compare_arrays):
        """Test SwooshL activation."""
        import mlx.core as mx

        from src.models.zipformer.scaling import ActivationDropoutAndLinear

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 10, 64)).astype(np.float32)
        x_mlx = mx.array(x_np)

        out_mlx = ActivationDropoutAndLinear._swoosh_l(x_mlx)
        mx.eval(out_mlx)

        # Manual computation: x * sigmoid(x - 1) + 0.1 * sigmoid(-x)
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        expected = x_np * sigmoid(x_np - 1) + 0.1 * sigmoid(-x_np)

        result = compare_arrays(expected, out_mlx, atol=1e-5, rtol=1e-5)
        assert result.is_close, f"SwooshL test failed: {result}"

    @pytest.mark.zipformer
    def test_swoosh_r(self, compare_arrays):
        """Test SwooshR activation."""
        import mlx.core as mx

        from src.models.zipformer.scaling import ActivationDropoutAndLinear

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 10, 64)).astype(np.float32)
        x_mlx = mx.array(x_np)

        out_mlx = ActivationDropoutAndLinear._swoosh_r(x_mlx)
        mx.eval(out_mlx)

        # Manual computation: x * sigmoid(x + 1) - 0.1
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        expected = x_np * sigmoid(x_np + 1) - 0.1

        result = compare_arrays(expected, out_mlx, atol=1e-5, rtol=1e-5)
        assert result.is_close, f"SwooshR test failed: {result}"
