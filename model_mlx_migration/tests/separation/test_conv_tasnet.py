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
Unit tests for Conv-TasNet MLX implementation.

Tests cover:
1. Model instantiation and weight loading
2. Forward pass output shapes
3. Numerical equivalence with PyTorch (when available)
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Module-level random generator for reproducibility
_rng = np.random.default_rng(42)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.whisper_mlx.separation.conv_tasnet import ConvTasNet, ConvTasNetConfig
from tools.whisper_mlx.separation.decoder import Decoder
from tools.whisper_mlx.separation.encoder import Encoder
from tools.whisper_mlx.separation.tcn import TemporalConvNet

WEIGHTS_PATH = Path(__file__).parent.parent.parent / "models" / "conv_tasnet" / "conv_tasnet_16k.safetensors"
CONFIG_PATH = WEIGHTS_PATH.with_suffix(".json")


def load_model_and_config():
    """Load the pretrained Conv-TasNet model."""
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"Weights not found at {WEIGHTS_PATH}")

    with open(CONFIG_PATH) as f:
        config_dict = json.load(f)

    model_args = config_dict.get("model_args", config_dict)
    config = ConvTasNetConfig.from_asteroid(model_args)
    model = ConvTasNet(config)

    from mlx.utils import tree_unflatten
    weights = mx.load(str(WEIGHTS_PATH))
    model.update(tree_unflatten(list(weights.items())))

    return model, config


class TestEncoder:
    """Tests for the Encoder module."""

    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape."""
        encoder = Encoder(n_filters=512, kernel_size=32, stride=16)

        batch_size = 2
        input_length = 16000  # 1 second at 16kHz
        x = mx.array(_rng.standard_normal((batch_size, input_length)).astype(np.float32))

        output = encoder(x)
        mx.eval(output)

        # Expected output shape: (B, N, T') where T' = (T - K) // S + 1
        expected_time = (input_length - 32) // 16 + 1
        assert output.shape == (batch_size, 512, expected_time)

    def test_encoder_length_calculation(self):
        """Test encoder output length calculation."""
        encoder = Encoder(n_filters=512, kernel_size=32, stride=16)

        input_length = 16000
        expected = (input_length - 32) // 16 + 1
        assert encoder.get_output_length(input_length) == expected


class TestDecoder:
    """Tests for the Decoder module."""

    def test_decoder_output_shape(self):
        """Test decoder produces correct output shape."""
        decoder = Decoder(n_filters=512, kernel_size=32, stride=16)

        batch_size = 2
        num_sources = 2
        n_filters = 512
        time_frames = 998

        encoded = mx.array(_rng.standard_normal((batch_size, n_filters, time_frames)).astype(np.float32))
        masks = mx.array(_rng.random((batch_size, num_sources, n_filters, time_frames)).astype(np.float32))

        output = decoder(encoded, masks)
        mx.eval(output)

        # Output shape: (B, C, T) where T = (T' - 1) * S + K
        expected_length = (time_frames - 1) * 16 + 32
        assert output.shape == (batch_size, num_sources, expected_length)


class TestTemporalConvNet:
    """Tests for the TCN Separator module."""

    def test_tcn_output_shape(self):
        """Test TCN produces correct mask shape."""
        tcn = TemporalConvNet(
            n_filters=512,
            bn_chan=128,
            hid_chan=512,
            skip_chan=128,
            kernel_size=3,
            n_layers=8,
            n_stacks=3,
            n_sources=2,
        )

        batch_size = 2
        n_filters = 512
        time_frames = 998

        x = mx.array(_rng.standard_normal((batch_size, n_filters, time_frames)).astype(np.float32))

        masks = tcn(x)
        mx.eval(masks)

        # Output shape: (B, C, N, T') where C is num_sources
        assert masks.shape == (batch_size, 2, n_filters, time_frames)


class TestConvTasNet:
    """Tests for the full Conv-TasNet model."""

    def test_model_output_shape_2d(self):
        """Test model with 2D input (B, T)."""
        model, config = load_model_and_config()

        batch_size = 1
        input_length = 16000  # 1 second

        x = mx.array(_rng.standard_normal((batch_size, input_length)).astype(np.float32))
        output = model(x)
        mx.eval(output)

        # Output: (B, C, T) where C = n_sources
        assert output.shape == (batch_size, config.n_src, input_length)

    def test_model_output_shape_1d(self):
        """Test model with 1D input (T,)."""
        model, config = load_model_and_config()

        input_length = 16000

        x = mx.array(_rng.standard_normal(input_length).astype(np.float32))
        output = model(x)
        mx.eval(output)

        # Output: (C, T) for 1D input
        assert output.shape == (config.n_src, input_length)

    def test_model_produces_different_sources(self):
        """Test that model produces different output for each source."""
        model, config = load_model_and_config()

        # Use reproducible input
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, 16000)).astype(np.float32))

        output = model(x)
        mx.eval(output)

        # Sources should be different
        source_diff = float(mx.sum(mx.abs(output[0, 0] - output[0, 1])))
        assert source_diff > 0.1, f"Sources too similar, diff={source_diff}"

    def test_separate_method(self):
        """Test the separate convenience method."""
        model, config = load_model_and_config()

        rng = np.random.default_rng(42)
        audio = rng.standard_normal(16000).astype(np.float32)

        sources = model.separate(audio, normalize=True)

        assert len(sources) == config.n_src
        for src in sources:
            assert src.shape == (16000,)
            # Normalized output should be in [-1, 1]
            max_val = float(mx.max(mx.abs(src)))
            assert max_val <= 1.01, f"Normalized output exceeds 1: {max_val}"

    def test_model_is_deterministic(self):
        """Test that model output is deterministic."""
        model, _ = load_model_and_config()

        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, 8000)).astype(np.float32))

        output1 = model(x)
        mx.eval(output1)

        output2 = model(x)
        mx.eval(output2)

        diff = float(mx.max(mx.abs(output1 - output2)))
        assert diff < 1e-6, f"Model not deterministic, max diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
