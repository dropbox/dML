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
Tests for Zipformer RNN-T decoder and joiner modules.

Validates:
1. Basic forward pass shapes
2. Weight loading from checkpoint
3. Numerical comparison with PyTorch reference
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.zipformer.decoder import Decoder, DecoderConfig
from models.zipformer.joiner import Joiner, JoinerConfig


class TestDecoder:
    """Tests for the Decoder module."""

    def test_decoder_init(self):
        """Test decoder initialization."""
        config = DecoderConfig(
            vocab_size=500,
            decoder_dim=512,
            blank_id=0,
            context_size=2,
        )
        decoder = Decoder(config)

        assert decoder.vocab_size == 500
        assert decoder.decoder_dim == 512
        assert decoder.context_size == 2
        assert decoder.embedding is not None
        assert decoder.conv is not None

    def test_decoder_forward_shape(self):
        """Test decoder forward pass output shapes."""
        config = DecoderConfig(
            vocab_size=500,
            decoder_dim=512,
            blank_id=0,
            context_size=2,
        )
        decoder = Decoder(config)

        # Test input
        batch_size = 2
        seq_len = 10
        y = mx.zeros((batch_size, seq_len), dtype=mx.int32)

        # Forward with padding (training mode)
        output = decoder(y, need_pad=True)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, 512)

    def test_decoder_no_context(self):
        """Test decoder without context (context_size=1)."""
        config = DecoderConfig(
            vocab_size=500,
            decoder_dim=512,
            blank_id=0,
            context_size=1,
        )
        decoder = Decoder(config)

        # Should not have conv layer
        assert decoder.conv is None

        # Forward pass
        y = mx.zeros((2, 10), dtype=mx.int32)
        output = decoder(y)
        mx.eval(output)

        assert output.shape == (2, 10, 512)

    def test_decoder_streaming_step(self):
        """Test decoder forward_one_step for streaming."""
        config = DecoderConfig(
            vocab_size=500,
            decoder_dim=512,
            blank_id=0,
            context_size=2,
        )
        decoder = Decoder(config)

        batch_size = 2
        # Current token
        y = mx.zeros((batch_size, 1), dtype=mx.int32)
        # Previous context (context_size - 1 embeddings)
        context = mx.zeros((batch_size, 1, 512))

        output = decoder.forward_one_step(y, context)
        mx.eval(output)

        assert output.shape == (batch_size, 1, 512)

    def test_decoder_relu_activation(self):
        """Test that ReLU is applied (no negative values in output)."""
        config = DecoderConfig(
            vocab_size=500,
            decoder_dim=512,
            blank_id=0,
            context_size=2,
        )
        decoder = Decoder(config)

        # Use random input tokens
        y = mx.random.randint(0, 500, (2, 10))

        output = decoder(y)
        mx.eval(output)

        # ReLU should make all values non-negative
        assert mx.all(output >= 0).item()


class TestJoiner:
    """Tests for the Joiner module."""

    def test_joiner_init(self):
        """Test joiner initialization."""
        config = JoinerConfig(
            encoder_dim=512,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        joiner = Joiner(config)

        assert joiner.encoder_proj is not None
        assert joiner.decoder_proj is not None
        assert joiner.output_linear is not None

    def test_joiner_forward_3d(self):
        """Test joiner with 3D inputs (full computation)."""
        config = JoinerConfig(
            encoder_dim=512,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        joiner = Joiner(config)

        batch_size = 2
        T = 50  # encoder frames
        U = 10  # decoder tokens

        encoder_out = mx.random.normal((batch_size, T, 512))
        decoder_out = mx.random.normal((batch_size, U, 512))

        logits = joiner(encoder_out, decoder_out)
        mx.eval(logits)

        # Output should be (batch, T, U, vocab_size)
        assert logits.shape == (batch_size, T, U, 500)

    def test_joiner_forward_4d(self):
        """Test joiner with 4D inputs (pruned computation)."""
        config = JoinerConfig(
            encoder_dim=512,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        joiner = Joiner(config)

        batch_size = 2
        T = 50
        s_range = 5  # pruning range

        encoder_out = mx.random.normal((batch_size, T, s_range, 512))
        decoder_out = mx.random.normal((batch_size, T, s_range, 512))

        logits = joiner(encoder_out, decoder_out)
        mx.eval(logits)

        # Output should be (batch, T, s_range, vocab_size)
        assert logits.shape == (batch_size, T, s_range, 500)

    def test_joiner_streaming(self):
        """Test joiner for streaming inference."""
        config = JoinerConfig(
            encoder_dim=512,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        joiner = Joiner(config)

        batch_size = 2
        encoder_out = mx.random.normal((batch_size, 1, 512))
        decoder_out = mx.random.normal((batch_size, 1, 512))

        logits = joiner.forward_streaming(encoder_out, decoder_out)
        mx.eval(logits)

        assert logits.shape == (batch_size, 1, 500)

    def test_joiner_tanh_activation(self):
        """Test that tanh is applied (values between -1 and 1 before output)."""
        config = JoinerConfig(
            encoder_dim=512,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        joiner = Joiner(config)

        encoder_out = mx.random.normal((2, 10, 512)) * 10  # Large values
        decoder_out = mx.random.normal((2, 5, 512)) * 10

        # Get intermediate values before output linear
        enc_proj = joiner.encoder_proj(encoder_out)
        dec_proj = joiner.decoder_proj(decoder_out)

        enc_proj = mx.expand_dims(enc_proj, axis=2)
        dec_proj = mx.expand_dims(dec_proj, axis=1)

        combined = mx.tanh(enc_proj + dec_proj)
        mx.eval(combined)

        # tanh output should be in [-1, 1]
        assert mx.all(combined >= -1).item()
        assert mx.all(combined <= 1).item()


class TestWeightLoading:
    """Tests for loading pretrained weights."""

    @pytest.fixture
    def checkpoint_path(self):
        return Path('checkpoints/zipformer/en-streaming/exp/pretrained.pt')

    @pytest.mark.skipif(
        not Path('checkpoints/zipformer/en-streaming/exp/pretrained.pt').exists(),
        reason="Pretrained checkpoint not available",
    )
    def test_load_decoder_weights(self, checkpoint_path):
        """Test loading decoder weights from checkpoint."""
        import torch

        # Load checkpoint
        ckpt = torch.load(str(checkpoint_path), map_location='cpu')
        model_dict = ckpt.get('model', ckpt)

        # Extract decoder config from weights
        emb_weight = model_dict['decoder.embedding.weight']
        conv_weight = model_dict['decoder.conv.weight']

        vocab_size = emb_weight.shape[0]  # 500
        decoder_dim = emb_weight.shape[1]  # 512
        context_size = conv_weight.shape[2]  # 2

        # Create decoder
        config = DecoderConfig(
            vocab_size=vocab_size,
            decoder_dim=decoder_dim,
            blank_id=0,
            context_size=context_size,
        )
        decoder = Decoder(config)

        # Load weights
        decoder.embedding.weight = mx.array(emb_weight.numpy())

        # Convert conv weight: PyTorch (O, I/G, K) -> MLX (O, K, I/G)
        conv_np = conv_weight.numpy()
        conv_np = np.transpose(conv_np, (0, 2, 1))
        decoder.conv.weight = mx.array(conv_np)

        # Verify shapes
        assert decoder.embedding.weight.shape == (500, 512)
        assert decoder.conv.weight.shape == (512, 2, 4)  # (O, K, I/G)

        # Test forward pass
        y = mx.zeros((2, 10), dtype=mx.int32)
        output = decoder(y)
        mx.eval(output)

        assert output.shape == (2, 10, 512)
        assert not mx.any(mx.isnan(output)).item()
        assert not mx.any(mx.isinf(output)).item()

    @pytest.mark.skipif(
        not Path('checkpoints/zipformer/en-streaming/exp/pretrained.pt').exists(),
        reason="Pretrained checkpoint not available",
    )
    def test_load_joiner_weights(self, checkpoint_path):
        """Test loading joiner weights from checkpoint."""
        import torch

        # Load checkpoint
        ckpt = torch.load(str(checkpoint_path), map_location='cpu')
        model_dict = ckpt.get('model', ckpt)

        # Extract joiner config from weights
        enc_proj_weight = model_dict['joiner.encoder_proj.weight']
        dec_proj_weight = model_dict['joiner.decoder_proj.weight']
        output_weight = model_dict['joiner.output_linear.weight']

        encoder_dim = enc_proj_weight.shape[1]  # 512
        decoder_dim = dec_proj_weight.shape[1]  # 512
        joiner_dim = enc_proj_weight.shape[0]  # 512
        vocab_size = output_weight.shape[0]  # 500

        # Create joiner
        config = JoinerConfig(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            joiner_dim=joiner_dim,
            vocab_size=vocab_size,
        )
        joiner = Joiner(config)

        # Load weights
        joiner.encoder_proj.weight = mx.array(enc_proj_weight.numpy())
        joiner.encoder_proj.bias = mx.array(
            model_dict['joiner.encoder_proj.bias'].numpy(),
        )
        joiner.decoder_proj.weight = mx.array(dec_proj_weight.numpy())
        joiner.decoder_proj.bias = mx.array(
            model_dict['joiner.decoder_proj.bias'].numpy(),
        )
        joiner.output_linear.weight = mx.array(output_weight.numpy())
        joiner.output_linear.bias = mx.array(
            model_dict['joiner.output_linear.bias'].numpy(),
        )

        # Test forward pass
        encoder_out = mx.random.normal((2, 50, 512))
        decoder_out = mx.random.normal((2, 10, 512))

        logits = joiner(encoder_out, decoder_out)
        mx.eval(logits)

        assert logits.shape == (2, 50, 10, 500)
        assert not mx.any(mx.isnan(logits)).item()
        assert not mx.any(mx.isinf(logits)).item()


class TestPyTorchComparison:
    """Numerical comparison with PyTorch reference."""

    @pytest.mark.skipif(
        not Path('checkpoints/zipformer/en-streaming/exp/pretrained.pt').exists(),
        reason="Pretrained checkpoint not available",
    )
    def test_decoder_numerical_match(self):
        """Test decoder output matches PyTorch reference."""
        import torch
        import torch.nn as pt_nn

        checkpoint_path = 'checkpoints/zipformer/en-streaming/exp/pretrained.pt'
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_dict = ckpt.get('model', ckpt)

        # Extract weights
        emb_weight = model_dict['decoder.embedding.weight']
        conv_weight = model_dict['decoder.conv.weight']

        vocab_size = emb_weight.shape[0]
        decoder_dim = emb_weight.shape[1]
        context_size = conv_weight.shape[2]

        # Create PyTorch decoder
        class PTDecoder(pt_nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = pt_nn.Embedding(vocab_size, decoder_dim)
                self.conv = pt_nn.Conv1d(
                    decoder_dim, decoder_dim,
                    kernel_size=context_size,
                    groups=decoder_dim // 4,
                    padding=0,
                    bias=False,
                )

            def forward(self, y, blank_id=0):
                embed = self.embedding(y)
                # Pad with blank embedding
                blank = self.embedding(
                    torch.full((embed.size(0), context_size - 1), blank_id),
                )
                embed = torch.cat([blank, embed], dim=1)
                # Conv expects (batch, channels, seq)
                embed = embed.transpose(1, 2)
                embed = self.conv(embed)
                embed = embed.transpose(1, 2)
                return torch.relu(embed)

        pt_decoder = PTDecoder()
        pt_decoder.embedding.weight.data = emb_weight
        pt_decoder.conv.weight.data = conv_weight
        pt_decoder.eval()

        # Create MLX decoder
        config = DecoderConfig(
            vocab_size=vocab_size,
            decoder_dim=decoder_dim,
            blank_id=0,
            context_size=context_size,
        )
        mlx_decoder = Decoder(config)
        mlx_decoder.embedding.weight = mx.array(emb_weight.numpy())
        conv_np = conv_weight.numpy()
        mlx_decoder.conv.weight = mx.array(np.transpose(conv_np, (0, 2, 1)))

        # Test input
        y_np = np.array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4]], dtype=np.int32)
        y_pt = torch.from_numpy(y_np).long()
        y_mlx = mx.array(y_np)

        # Forward pass
        with torch.no_grad():
            pt_out = pt_decoder(y_pt).numpy()

        mlx_out = mlx_decoder(y_mlx, need_pad=True)
        mx.eval(mlx_out)
        mlx_out_np = np.array(mlx_out)

        # Compare
        max_diff = np.abs(pt_out - mlx_out_np).max()
        mean_diff = np.abs(pt_out - mlx_out_np).mean()

        print("Decoder comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        # Tolerance of 2e-5 accounts for float32 precision differences between frameworks
        assert max_diff < 2e-5, f"Max diff {max_diff} exceeds tolerance"

    @pytest.mark.skipif(
        not Path('checkpoints/zipformer/en-streaming/exp/pretrained.pt').exists(),
        reason="Pretrained checkpoint not available",
    )
    def test_joiner_numerical_match(self):
        """Test joiner output matches PyTorch reference."""
        import torch
        import torch.nn as pt_nn

        checkpoint_path = 'checkpoints/zipformer/en-streaming/exp/pretrained.pt'
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_dict = ckpt.get('model', ckpt)

        # Create PyTorch joiner
        class PTJoiner(pt_nn.Module):
            def __init__(self, encoder_dim, decoder_dim, joiner_dim, vocab_size):
                super().__init__()
                self.encoder_proj = pt_nn.Linear(encoder_dim, joiner_dim)
                self.decoder_proj = pt_nn.Linear(decoder_dim, joiner_dim)
                self.output_linear = pt_nn.Linear(joiner_dim, vocab_size)

            def forward(self, encoder_out, decoder_out):
                encoder_out = self.encoder_proj(encoder_out)
                decoder_out = self.decoder_proj(decoder_out)
                # Expand for broadcasting
                encoder_out = encoder_out.unsqueeze(2)
                decoder_out = decoder_out.unsqueeze(1)
                logits = torch.tanh(encoder_out + decoder_out)
                return self.output_linear(logits)

        pt_joiner = PTJoiner(512, 512, 512, 500)
        pt_joiner.encoder_proj.weight.data = model_dict['joiner.encoder_proj.weight']
        pt_joiner.encoder_proj.bias.data = model_dict['joiner.encoder_proj.bias']
        pt_joiner.decoder_proj.weight.data = model_dict['joiner.decoder_proj.weight']
        pt_joiner.decoder_proj.bias.data = model_dict['joiner.decoder_proj.bias']
        pt_joiner.output_linear.weight.data = model_dict['joiner.output_linear.weight']
        pt_joiner.output_linear.bias.data = model_dict['joiner.output_linear.bias']
        pt_joiner.eval()

        # Create MLX joiner
        config = JoinerConfig(
            encoder_dim=512,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        mlx_joiner = Joiner(config)
        mlx_joiner.encoder_proj.weight = mx.array(
            model_dict['joiner.encoder_proj.weight'].numpy(),
        )
        mlx_joiner.encoder_proj.bias = mx.array(
            model_dict['joiner.encoder_proj.bias'].numpy(),
        )
        mlx_joiner.decoder_proj.weight = mx.array(
            model_dict['joiner.decoder_proj.weight'].numpy(),
        )
        mlx_joiner.decoder_proj.bias = mx.array(
            model_dict['joiner.decoder_proj.bias'].numpy(),
        )
        mlx_joiner.output_linear.weight = mx.array(
            model_dict['joiner.output_linear.weight'].numpy(),
        )
        mlx_joiner.output_linear.bias = mx.array(
            model_dict['joiner.output_linear.bias'].numpy(),
        )

        # Test input (smaller for speed)
        rng = np.random.default_rng(42)
        enc_np = rng.standard_normal((2, 10, 512)).astype(np.float32)
        dec_np = rng.standard_normal((2, 5, 512)).astype(np.float32)

        enc_pt = torch.from_numpy(enc_np)
        dec_pt = torch.from_numpy(dec_np)
        enc_mlx = mx.array(enc_np)
        dec_mlx = mx.array(dec_np)

        # Forward pass
        with torch.no_grad():
            pt_out = pt_joiner(enc_pt, dec_pt).numpy()

        mlx_out = mlx_joiner(enc_mlx, dec_mlx)
        mx.eval(mlx_out)
        mlx_out_np = np.array(mlx_out)

        # Compare
        max_diff = np.abs(pt_out - mlx_out_np).max()
        mean_diff = np.abs(pt_out - mlx_out_np).mean()

        print("Joiner comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        # Tolerance of 2e-5 accounts for float32 precision differences between frameworks
        assert max_diff < 2e-5, f"Max diff {max_diff} exceeds tolerance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
