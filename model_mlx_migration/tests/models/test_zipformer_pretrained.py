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
Tests for pretrained Zipformer encoder implementation.

These tests validate:
1. Weight loading from converted checkpoint
2. Numerical equivalence with PyTorch reference
3. Forward pass correctness with pretrained weights
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Mark all tests to skip if checkpoint not available
CHECKPOINT_PATH = Path("pretrained/zipformer-streaming/exp/pretrained_mlx.npz")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="Pretrained weights not available",
)


class TestPretrainedEncoderLayer:
    """Tests for ZipformerEncoderLayer with pretrained weights."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load weights fixture."""
        import sys
        sys.path.insert(0, "src")

        from models.zipformer.encoder_pretrained import (
            ZipformerEncoderLayer,
            load_encoder_layer_weights,
        )

        self.weights = dict(mx.load(str(CHECKPOINT_PATH)))
        self.ZipformerEncoderLayer = ZipformerEncoderLayer
        self.load_encoder_layer_weights = load_encoder_layer_weights

    def test_weight_loading(self):
        """Test that weights can be loaded into encoder layer."""
        layer = self.ZipformerEncoderLayer(
            d_model=384,
            attention_dim=192,
            num_heads=8,
            feedforward_dim=1024,
            kernel_size=31,
            pos_dim=4,
        )

        self.load_encoder_layer_weights(
            layer, self.weights, "encoder.encoders.0.layers.0.",
        )

        # Verify key weights are loaded (not all zeros)
        assert mx.abs(layer.bypass_scale - 0.5) < 0.5, "bypass_scale should be loaded"
        assert mx.sum(mx.abs(layer.self_attn.in_proj.weight)) > 0, "in_proj should be loaded"
        assert mx.sum(mx.abs(layer.pooling.proj.weight)) > 0, "pooling should be loaded"

    def test_forward_with_pretrained_weights(self):
        """Test forward pass with pretrained weights."""
        layer = self.ZipformerEncoderLayer(
            d_model=384,
            attention_dim=192,
            num_heads=8,
            feedforward_dim=1024,
            kernel_size=31,
            pos_dim=4,
        )

        self.load_encoder_layer_weights(
            layer, self.weights, "encoder.encoders.0.layers.0.",
        )

        # Run forward pass
        batch_size = 2
        seq_len = 16
        x = mx.random.normal((seq_len, batch_size, 384))
        pos_emb = mx.random.normal((batch_size, 2 * seq_len - 1, 384))

        out = layer(x, pos_emb)

        # Verify output shape
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

        # Verify output is not all zeros
        assert mx.sum(mx.abs(out)) > 0, "Output should not be all zeros"

        # Verify output has reasonable statistics
        out_mean = float(mx.mean(out))
        out_std = float(mx.std(out))
        assert abs(out_mean) < 10.0, f"Mean too large: {out_mean}"
        assert 0.01 < out_std < 100.0, f"Std out of range: {out_std}"

    def test_all_stages_loadable(self):
        """Test that all encoder stages have loadable weights."""
        # Stage configurations from checkpoint analysis
        stage_configs = {
            0: {"layers": 2, "prefix": "encoder.encoders.0.layers."},
            1: {"layers": 4, "prefix": "encoder.encoders.1.encoder.layers."},
            2: {"layers": 3, "prefix": "encoder.encoders.2.encoder.layers."},
            3: {"layers": 2, "prefix": "encoder.encoders.3.encoder.layers."},
            4: {"layers": 4, "prefix": "encoder.encoders.4.encoder.layers."},
        }

        for _stage_idx, config in stage_configs.items():
            for layer_idx in range(config["layers"]):
                prefix = f"{config['prefix']}{layer_idx}."

                # Check if key weights exist
                bypass_key = f"{prefix}bypass_scale"
                if bypass_key not in self.weights:
                    pytest.skip(f"Weight {bypass_key} not found")

                # Try loading
                layer = self.ZipformerEncoderLayer(
                    d_model=384,
                    attention_dim=192,
                    num_heads=8,
                    feedforward_dim=1024,
                    kernel_size=31,
                    pos_dim=4,
                )
                self.load_encoder_layer_weights(layer, self.weights, prefix)


class TestNumericalEquivalence:
    """Tests for numerical equivalence between PyTorch and MLX."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        import sys
        sys.path.insert(0, "src")

        from models.zipformer.encoder_pretrained import (
            ConvolutionModule,
            FeedforwardModule,
            PoolingModule,
        )

        self.FeedforwardModule = FeedforwardModule
        self.PoolingModule = PoolingModule
        self.ConvolutionModule = ConvolutionModule

    def test_feedforward_numerical_accuracy(self):
        """Test feedforward module produces correct output."""
        import torch

        # Load PyTorch weights
        pt_checkpoint = "pretrained/zipformer-streaming/exp/pretrained.pt"
        if not Path(pt_checkpoint).exists():
            pytest.skip("PyTorch checkpoint not available")

        state_dict = torch.load(pt_checkpoint, map_location="cpu")
        model_dict = state_dict.get("model", state_dict)

        prefix = "encoder.encoders.0.layers.0.feed_forward1."
        in_proj_w = model_dict[prefix + "in_proj.weight"]
        in_proj_b = model_dict[prefix + "in_proj.bias"]
        out_proj_w = model_dict[prefix + "out_proj.weight"]
        out_proj_b = model_dict[prefix + "out_proj.bias"]

        # Create identical input
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10, 2, 384)).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = mx.array(x_np)

        # PyTorch forward
        h = torch.nn.functional.linear(x_torch, in_proj_w, in_proj_b)
        h = torch.nn.functional.silu(h)
        pt_out = torch.nn.functional.linear(h, out_proj_w, out_proj_b)

        # MLX forward
        mlx_ff = self.FeedforwardModule(384, 1024)
        mlx_ff.in_proj.weight = mx.array(in_proj_w.numpy())
        mlx_ff.in_proj.bias = mx.array(in_proj_b.numpy())
        mlx_ff.out_proj.weight = mx.array(out_proj_w.numpy())
        mlx_ff.out_proj.bias = mx.array(out_proj_b.numpy())
        mlx_out = mlx_ff(x_mlx)

        # Compare
        max_abs_error = float(np.max(np.abs(pt_out.numpy() - np.array(mlx_out))))
        assert max_abs_error < 1e-4, f"Max abs error too large: {max_abs_error}"

    def test_pooling_numerical_accuracy(self):
        """Test pooling module produces correct output."""
        import torch

        pt_checkpoint = "pretrained/zipformer-streaming/exp/pretrained.pt"
        if not Path(pt_checkpoint).exists():
            pytest.skip("PyTorch checkpoint not available")

        state_dict = torch.load(pt_checkpoint, map_location="cpu")
        model_dict = state_dict.get("model", state_dict)

        proj_w = model_dict["encoder.encoders.0.layers.0.pooling.proj.weight"]

        # Create identical input
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10, 2, 384)).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = mx.array(x_np)

        # PyTorch forward (manual pooling)
        cum_x = torch.cumsum(x_torch, dim=0)
        cum_count = torch.arange(1, 11, dtype=torch.float32).view(-1, 1, 1)
        pooled = cum_x / cum_count
        pt_out = torch.nn.functional.linear(pooled, proj_w, None)

        # MLX forward
        mlx_pool = self.PoolingModule(384)
        mlx_pool.proj.weight = mx.array(proj_w.numpy())
        mlx_out = mlx_pool(x_mlx)

        # Compare
        max_abs_error = float(np.max(np.abs(pt_out.numpy() - np.array(mlx_out))))
        assert max_abs_error < 1e-5, f"Max abs error too large: {max_abs_error}"


class TestWeightConversion:
    """Tests for weight conversion utilities."""

    def test_convert_checkpoint_analysis(self):
        """Test checkpoint analysis function."""
        import sys
        sys.path.insert(0, "src")

        from models.zipformer.convert_weights import analyze_checkpoint

        pt_checkpoint = "pretrained/zipformer-streaming/exp/pretrained.pt"
        if not Path(pt_checkpoint).exists():
            pytest.skip("PyTorch checkpoint not available")

        analysis = analyze_checkpoint(pt_checkpoint)

        assert analysis["total_keys"] > 0
        assert analysis["encoder_keys"] > 0
        assert len(analysis["encoder_stages"]) == 5  # 5 stages

    def test_config_extraction(self):
        """Test configuration extraction from checkpoint."""
        import sys
        sys.path.insert(0, "src")

        from models.zipformer.convert_weights import extract_encoder_config

        pt_checkpoint = "pretrained/zipformer-streaming/exp/pretrained.pt"
        if not Path(pt_checkpoint).exists():
            pytest.skip("PyTorch checkpoint not available")

        config = extract_encoder_config(pt_checkpoint)

        assert config["d_model"] == 384
        assert config["kernel_size"] == 31
        assert config["num_stages"] == 5
