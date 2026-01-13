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
Unit tests for Zipformer2 encoder components.
"""

import mlx.core as mx
import pytest


class TestChunkCausalDepthwiseConv1d:
    """Tests for ChunkCausalDepthwiseConv1d module."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import ChunkCausalDepthwiseConv1d

        conv = ChunkCausalDepthwiseConv1d(channels=64, kernel_size=31)
        assert conv.channels == 64
        assert conv.kernel_size == 31
        # MLX conv1d weight shape: (C_out, K, C_in) with C_in=1 for depthwise
        assert conv.causal_conv_weight.shape == (64, 16, 1)  # (kernel_size + 1) // 2
        assert conv.chunkwise_conv_weight.shape == (64, 31, 1)

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import ChunkCausalDepthwiseConv1d

        conv = ChunkCausalDepthwiseConv1d(channels=64, kernel_size=31)
        x = mx.random.normal(shape=(4, 64, 100))  # (batch, channels, time)

        # Without chunking
        y = conv(x, chunk_size=-1)
        assert y.shape == x.shape

        # With chunking
        y = conv(x, chunk_size=20)
        assert y.shape == x.shape

    def test_streaming_forward(self):
        """Test streaming forward pass."""
        from src.models.zipformer import ChunkCausalDepthwiseConv1d

        conv = ChunkCausalDepthwiseConv1d(channels=64, kernel_size=31)
        batch_size = 4
        left_pad = 31 // 2  # kernel_size // 2

        x = mx.random.normal(shape=(batch_size, 64, 20))
        cache = mx.zeros((batch_size, 64, left_pad))

        y, new_cache = conv.streaming_forward(x, cache)
        assert y.shape == x.shape
        assert new_cache.shape == cache.shape


class TestCompactRelPositionalEncoding:
    """Tests for CompactRelPositionalEncoding module."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import CompactRelPositionalEncoding

        pos_enc = CompactRelPositionalEncoding(embed_dim=192)
        assert pos_enc.embed_dim == 192
        assert pos_enc.length_factor == 1.0

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import CompactRelPositionalEncoding

        pos_enc = CompactRelPositionalEncoding(embed_dim=192)
        x = mx.random.normal(shape=(20, 4, 384))  # (seq, batch, dim)

        pos_emb = pos_enc(x)
        # Output should be (1, 2*seq_len - 1, embed_dim)
        assert pos_emb.shape == (1, 39, 192)

    def test_streaming_forward_shape(self):
        """Test streaming output shape."""
        from src.models.zipformer import CompactRelPositionalEncoding

        pos_enc = CompactRelPositionalEncoding(embed_dim=192)
        x = mx.random.normal(shape=(20, 4, 384))
        left_context_len = 10

        pos_emb = pos_enc(x, left_context_len=left_context_len)
        # Output should be (1, left_context_len + 2*seq_len - 1, embed_dim)
        assert pos_emb.shape == (1, 49, 192)


class TestSelfAttention:
    """Tests for SelfAttention module."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import SelfAttention

        attn = SelfAttention(embed_dim=384, num_heads=8, value_head_dim=12)
        assert attn.num_heads == 8
        assert attn.value_head_dim == 12

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import SelfAttention

        attn = SelfAttention(embed_dim=384, num_heads=8, value_head_dim=12)
        seq_len, batch_size, embed_dim = 20, 4, 384

        x = mx.random.normal(shape=(seq_len, batch_size, embed_dim))
        attn_weights = mx.softmax(
            mx.random.normal(shape=(8, batch_size, seq_len, seq_len)),
            axis=-1,
        )

        y = attn(x, attn_weights)
        assert y.shape == x.shape


class TestFeedforwardModule:
    """Tests for FeedforwardModule."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import FeedforwardModule

        ff = FeedforwardModule(embed_dim=384, feedforward_dim=1536)
        assert ff.in_proj.weight.shape[0] == 1536

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import FeedforwardModule

        ff = FeedforwardModule(embed_dim=384, feedforward_dim=1536)
        x = mx.random.normal(shape=(20, 4, 384))

        y = ff(x)
        assert y.shape == x.shape


class TestNonlinAttention:
    """Tests for NonlinAttention module."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import NonlinAttention

        na = NonlinAttention(channels=384, hidden_channels=288)
        assert na.hidden_channels == 288

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import NonlinAttention

        na = NonlinAttention(channels=384, hidden_channels=288)
        seq_len, batch_size = 20, 4

        x = mx.random.normal(shape=(seq_len, batch_size, 384))
        # NonlinAttention uses a single head of attention weights
        attn_weights = mx.softmax(
            mx.random.normal(shape=(1, batch_size, seq_len, seq_len)),
            axis=-1,
        )

        y = na(x, attn_weights)
        assert y.shape == x.shape


class TestConvolutionModule:
    """Tests for ConvolutionModule."""

    def test_init_non_causal(self):
        """Test non-causal initialization."""
        from src.models.zipformer import ConvolutionModule

        conv = ConvolutionModule(channels=384, kernel_size=31, causal=False)
        assert conv.channels == 384
        assert conv.kernel_size == 31
        assert not conv.causal

    def test_init_causal(self):
        """Test causal initialization."""
        from src.models.zipformer import ConvolutionModule

        conv = ConvolutionModule(channels=384, kernel_size=31, causal=True)
        assert conv.causal

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import ConvolutionModule

        conv = ConvolutionModule(channels=384, kernel_size=31, causal=False)
        x = mx.random.normal(shape=(20, 4, 384))  # (seq, batch, channels)

        y = conv(x)
        assert y.shape == x.shape


class TestBypassModule:
    """Tests for BypassModule."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import BypassModule

        bypass = BypassModule(embed_dim=384)
        assert bypass.bypass_scale.shape == (384,)

    def test_forward(self):
        """Test forward pass."""
        from src.models.zipformer import BypassModule

        bypass = BypassModule(embed_dim=384)
        src_orig = mx.random.normal(shape=(20, 4, 384))
        src = mx.random.normal(shape=(20, 4, 384))

        y = bypass(src_orig, src)
        assert y.shape == src_orig.shape

        # With bypass_scale = 0.5, output should be (src_orig + src) / 2
        # approximately
        mx.eval(y)


class TestZipformer2EncoderLayer:
    """Tests for Zipformer2EncoderLayer."""

    def test_init(self):
        """Test layer initialization."""
        from src.models.zipformer import Zipformer2EncoderLayer

        layer = Zipformer2EncoderLayer(
            embed_dim=384,
            pos_dim=192,
            num_heads=8,
            query_head_dim=24,
            pos_head_dim=4,
            value_head_dim=12,
            feedforward_dim=1536,
            cnn_module_kernel=31,
            causal=False,
        )
        assert layer.embed_dim == 384

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import (
            CompactRelPositionalEncoding,
            Zipformer2EncoderLayer,
        )

        layer = Zipformer2EncoderLayer(
            embed_dim=384,
            pos_dim=192,
            num_heads=8,
            query_head_dim=24,
            pos_head_dim=4,
            value_head_dim=12,
            feedforward_dim=1536,
            cnn_module_kernel=31,
            causal=False,
        )

        pos_enc = CompactRelPositionalEncoding(embed_dim=192)

        seq_len, batch_size, embed_dim = 20, 4, 384
        x = mx.random.normal(shape=(seq_len, batch_size, embed_dim))
        pos_emb = pos_enc(x)

        y = layer(x, pos_emb)
        assert y.shape == x.shape


class TestSimpleDownsample:
    """Tests for SimpleDownsample."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import SimpleDownsample

        ds = SimpleDownsample(channels=384, downsample=2)
        assert ds.downsample == 2

    def test_forward_shape(self):
        """Test output shape with exact multiple."""
        from src.models.zipformer import SimpleDownsample

        ds = SimpleDownsample(channels=384, downsample=2)
        x = mx.random.normal(shape=(20, 4, 384))

        y = ds(x)
        assert y.shape == (10, 4, 384)

    def test_forward_shape_with_padding(self):
        """Test output shape with non-exact multiple."""
        from src.models.zipformer import SimpleDownsample

        ds = SimpleDownsample(channels=384, downsample=4)
        x = mx.random.normal(shape=(21, 4, 384))  # 21 frames

        y = ds(x)
        # (21 + 4 - 1) // 4 = 6
        assert y.shape == (6, 4, 384)


class TestSimpleUpsample:
    """Tests for SimpleUpsample."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import SimpleUpsample

        us = SimpleUpsample(num_channels=384, upsample=2)
        assert us.upsample == 2

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import SimpleUpsample

        us = SimpleUpsample(num_channels=384, upsample=2)
        x = mx.random.normal(shape=(10, 4, 384))

        y = us(x)
        assert y.shape == (20, 4, 384)


class TestRelPositionMultiheadAttentionWeights:
    """Tests for RelPositionMultiheadAttentionWeights."""

    def test_init(self):
        """Test module initialization."""
        from src.models.zipformer import RelPositionMultiheadAttentionWeights

        attn = RelPositionMultiheadAttentionWeights(
            embed_dim=384,
            pos_dim=192,
            num_heads=8,
            query_head_dim=24,
            pos_head_dim=4,
        )
        assert attn.num_heads == 8
        assert attn.query_head_dim == 24
        assert attn.pos_head_dim == 4

    def test_forward_shape(self):
        """Test output shape."""
        from src.models.zipformer import (
            CompactRelPositionalEncoding,
            RelPositionMultiheadAttentionWeights,
        )

        attn = RelPositionMultiheadAttentionWeights(
            embed_dim=384,
            pos_dim=192,
            num_heads=8,
            query_head_dim=24,
            pos_head_dim=4,
        )
        pos_enc = CompactRelPositionalEncoding(embed_dim=192)

        seq_len, batch_size = 20, 4
        x = mx.random.normal(shape=(seq_len, batch_size, 384))
        pos_emb = pos_enc(x)

        attn_weights = attn(x, pos_emb)
        # Output should be (num_heads, batch_size, seq_len, seq_len)
        assert attn_weights.shape == (8, batch_size, seq_len, seq_len)

        # Check softmax property (sums to 1 along last dim)
        sums = mx.sum(attn_weights, axis=-1)
        mx.eval(sums)
        # Should be approximately 1.0
        assert mx.allclose(sums, mx.ones_like(sums), atol=1e-5)


class TestIntegration:
    """Integration tests for the full encoder."""

    def test_encoder_layer_forward_backward(self):
        """Test that we can run forward pass without errors."""
        from src.models.zipformer import (
            CompactRelPositionalEncoding,
            Zipformer2EncoderLayer,
        )

        layer = Zipformer2EncoderLayer(
            embed_dim=128,  # smaller for speed
            pos_dim=64,
            num_heads=4,
            query_head_dim=16,
            pos_head_dim=4,
            value_head_dim=16,
            feedforward_dim=256,
            cnn_module_kernel=15,  # smaller kernel
            causal=False,
        )

        pos_enc = CompactRelPositionalEncoding(embed_dim=64)

        x = mx.random.normal(shape=(10, 2, 128))
        pos_emb = pos_enc(x)

        y = layer(x, pos_emb)
        mx.eval(y)

        assert y.shape == x.shape
        # Check that output is finite
        assert mx.all(mx.isfinite(y))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
