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

"""Tests for full Zipformer model implementation."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from src.models.zipformer.zipformer import (
    BypassModule,
    Conv2dSubsampling,
    ConvolutionModule,
    DownsampledZipformer2Encoder,
    FeedforwardModule,
    NonlinAttention,
    RelPositionalEncoding,
    RelPositionMultiheadAttentionWeights,
    SelfAttention,
    SimpleDownsample,
    SimpleUpsample,
    Zipformer,
    Zipformer2Encoder,
    Zipformer2EncoderLayer,
    ZipformerConfig,
)


class TestSimpleDownsample:
    """Test SimpleDownsample module."""

    def test_init(self):
        ds = SimpleDownsample(d_model=192, downsample=2)
        assert ds.downsample == 2
        assert ds.bias.shape == (2,)

    def test_forward_shape(self):
        ds = SimpleDownsample(d_model=192, downsample=2)
        x = mx.random.normal((100, 4, 192))  # (seq, batch, d_model)
        mx.eval(x)

        out = ds(x)
        mx.eval(out)

        assert out.shape == (50, 4, 192)

    def test_forward_shape_odd_length(self):
        ds = SimpleDownsample(d_model=192, downsample=2)
        x = mx.random.normal((101, 4, 192))
        mx.eval(x)

        out = ds(x)
        mx.eval(out)

        assert out.shape == (51, 4, 192)


class TestSimpleUpsample:
    """Test SimpleUpsample module."""

    def test_init(self):
        us = SimpleUpsample(d_model=192, upsample=2)
        assert us.upsample == 2

    def test_forward_shape(self):
        us = SimpleUpsample(d_model=192, upsample=2)
        x = mx.random.normal((50, 4, 192))
        mx.eval(x)

        out = us(x)
        mx.eval(out)

        assert out.shape == (100, 4, 192)


class TestBypassModule:
    """Test BypassModule."""

    def test_forward(self):
        bypass = BypassModule(d_model=192)
        src_orig = mx.random.normal((10, 4, 192))
        src = mx.random.normal((10, 4, 192))
        mx.eval(src_orig, src)

        out = bypass(src_orig, src)
        mx.eval(out)

        assert out.shape == (10, 4, 192)


class TestFeedforwardModule:
    """Test FeedforwardModule."""

    def test_init(self):
        ff = FeedforwardModule(d_model=192, feedforward_dim=384)
        assert ff.in_proj.weight.shape == (384, 192)
        assert ff.out_proj.weight.shape == (192, 384)

    def test_forward_shape(self):
        ff = FeedforwardModule(d_model=192, feedforward_dim=384)
        x = mx.random.normal((20, 4, 192))
        mx.eval(x)

        out = ff(x)
        mx.eval(out)

        assert out.shape == (20, 4, 192)


class TestConvolutionModule:
    """Test ConvolutionModule."""

    def test_init_causal(self):
        conv = ConvolutionModule(d_model=192, kernel_size=31, causal=True)
        assert conv.d_model == 192
        assert conv.kernel_size == 31
        assert conv.causal is True

    def test_init_non_causal(self):
        conv = ConvolutionModule(d_model=192, kernel_size=31, causal=False)
        assert conv.causal is False

    def test_forward_shape_causal(self):
        conv = ConvolutionModule(d_model=192, kernel_size=31, causal=True)
        x = mx.random.normal((20, 4, 192))
        mx.eval(x)

        out = conv(x)
        mx.eval(out)

        assert out.shape == (20, 4, 192)

    def test_forward_shape_non_causal(self):
        conv = ConvolutionModule(d_model=192, kernel_size=31, causal=False)
        x = mx.random.normal((20, 4, 192))
        mx.eval(x)

        out = conv(x)
        mx.eval(out)

        assert out.shape == (20, 4, 192)

    def test_streaming_forward_shape(self):
        """Test streaming_forward output shapes."""
        conv = ConvolutionModule(d_model=192, kernel_size=31, causal=True)
        seq_len = 16
        batch_size = 2
        cache_len = 31 // 2  # 15

        x = mx.random.normal((seq_len, batch_size, 192))
        cached_conv = mx.zeros((batch_size, 192, cache_len))
        mx.eval(x, cached_conv)

        out, new_cache = conv.streaming_forward(x, cached_conv)
        mx.eval(out, new_cache)

        assert out.shape == (seq_len, batch_size, 192)
        assert new_cache.shape == (batch_size, 192, cache_len)

    def test_streaming_forward_various_kernels(self):
        """Test streaming_forward with different kernel sizes."""
        for kernel_size in [15, 31]:
            conv = ConvolutionModule(d_model=192, kernel_size=kernel_size, causal=True)
            seq_len = 16
            batch_size = 2
            cache_len = kernel_size // 2

            x = mx.random.normal((seq_len, batch_size, 192))
            cached_conv = mx.zeros((batch_size, 192, cache_len))
            mx.eval(x, cached_conv)

            out, new_cache = conv.streaming_forward(x, cached_conv)
            mx.eval(out, new_cache)

            assert out.shape == (seq_len, batch_size, 192), f"Failed for kernel_size={kernel_size}"
            assert new_cache.shape == (batch_size, 192, cache_len), f"Failed for kernel_size={kernel_size}"

    def test_streaming_forward_cache_update(self):
        """Test that cache is properly updated with last frames."""
        conv = ConvolutionModule(d_model=4, kernel_size=7, causal=True)
        seq_len = 10
        batch_size = 1
        cache_len = 7 // 2  # 3

        # Create deterministic input
        x = mx.arange(seq_len * batch_size * 4, dtype=mx.float32)
        x = mx.reshape(x, (seq_len, batch_size, 4))
        cached_conv = mx.zeros((batch_size, 4, cache_len))
        mx.eval(x, cached_conv)

        _, new_cache = conv.streaming_forward(x, cached_conv)
        mx.eval(new_cache)

        # New cache should be the last cache_len frames after GLU
        assert new_cache.shape == (batch_size, 4, cache_len)

    def test_streaming_multiple_chunks(self):
        """Test that streaming produces consistent cache across chunks."""
        conv = ConvolutionModule(d_model=64, kernel_size=15, causal=True)
        chunk_size = 8
        batch_size = 2
        cache_len = 15 // 2  # 7

        # Initialize cache
        cached_conv = mx.zeros((batch_size, 64, cache_len))

        # Process multiple chunks
        for _i in range(3):
            x = mx.random.normal((chunk_size, batch_size, 64))
            mx.eval(x, cached_conv)

            out, cached_conv = conv.streaming_forward(x, cached_conv)
            mx.eval(out, cached_conv)

            assert out.shape == (chunk_size, batch_size, 64)
            assert cached_conv.shape == (batch_size, 64, cache_len)


class TestRelPositionalEncoding:
    """Test RelPositionalEncoding."""

    def test_init(self):
        pe = RelPositionalEncoding(pos_dim=48)
        assert pe.pos_dim == 48

    def test_forward_shape(self):
        pe = RelPositionalEncoding(pos_dim=48)
        x = mx.random.normal((20, 4, 192))  # (seq, batch, d_model) - only shape used
        mx.eval(x)

        pos_emb = pe(x)
        mx.eval(pos_emb)

        # Should be (batch, 2*seq-1, pos_dim)
        assert pos_emb.shape == (4, 39, 48)


class TestRelPositionMultiheadAttentionWeights:
    """Test attention weight computation."""

    def test_init(self):
        attn = RelPositionMultiheadAttentionWeights(
            d_model=192, num_heads=4, query_head_dim=32, pos_head_dim=4, pos_emb_dim=48,
        )
        assert attn.num_heads == 4
        assert attn.query_head_dim == 32
        assert attn.pos_head_dim == 4

    def test_forward_shape(self):
        attn = RelPositionMultiheadAttentionWeights(
            d_model=192, num_heads=4, query_head_dim=32, pos_head_dim=4, pos_emb_dim=48,
        )
        x = mx.random.normal((20, 4, 192))  # (seq, batch, d_model)
        pos_emb = mx.random.normal((4, 39, 48))  # (batch, 2*seq-1, pos_emb_dim)
        mx.eval(x, pos_emb)

        weights = attn(x, pos_emb)
        mx.eval(weights)

        # Should be (batch * heads, seq, seq)
        assert weights.shape == (16, 20, 20)

    def test_streaming_forward_shape(self):
        """Test streaming_forward output shapes."""
        attn = RelPositionMultiheadAttentionWeights(
            d_model=192, num_heads=4, query_head_dim=32, pos_head_dim=4, pos_emb_dim=48,
        )
        seq_len = 16
        batch_size = 2
        left_context_len = 64
        kv_len = left_context_len + seq_len  # 80
        query_dim = 4 * 32  # num_heads * query_head_dim = 128

        x = mx.random.normal((seq_len, batch_size, 192))
        # Extended positional embedding for streaming: seq_len + kv_len - 1 = 95
        pos_emb = mx.random.normal((batch_size, seq_len + kv_len - 1, 48))
        cached_key = mx.zeros((left_context_len, batch_size, query_dim))
        mx.eval(x, pos_emb, cached_key)

        weights, new_cached_key = attn.streaming_forward(x, pos_emb, cached_key, left_context_len)
        mx.eval(weights, new_cached_key)

        # Weights: (batch*heads, seq_len, kv_len)
        assert weights.shape == (batch_size * 4, seq_len, kv_len)
        # Cached key: (left_ctx, batch, query_dim)
        assert new_cached_key.shape == (left_context_len, batch_size, query_dim)

    def test_streaming_multiple_chunks(self):
        """Test streaming over multiple chunks."""
        attn = RelPositionMultiheadAttentionWeights(
            d_model=192, num_heads=4, query_head_dim=32, pos_head_dim=4, pos_emb_dim=48,
        )
        seq_len = 16
        batch_size = 2
        left_context_len = 32
        query_dim = 4 * 32  # 128

        # Initialize cache
        cached_key = mx.zeros((left_context_len, batch_size, query_dim))

        for _i in range(3):
            x = mx.random.normal((seq_len, batch_size, 192))
            kv_len = left_context_len + seq_len
            pos_emb = mx.random.normal((batch_size, seq_len + kv_len - 1, 48))
            mx.eval(x, pos_emb, cached_key)

            weights, cached_key = attn.streaming_forward(x, pos_emb, cached_key, left_context_len)
            mx.eval(weights, cached_key)

            assert weights.shape == (batch_size * 4, seq_len, kv_len)
            assert cached_key.shape == (left_context_len, batch_size, query_dim)


class TestSelfAttention:
    """Test SelfAttention module."""

    def test_init(self):
        attn = SelfAttention(d_model=192, num_heads=4, value_head_dim=24)
        assert attn.num_heads == 4
        assert attn.value_head_dim == 24

    def test_forward_shape(self):
        attn = SelfAttention(d_model=192, num_heads=4, value_head_dim=24)
        x = mx.random.normal((20, 4, 192))
        attn_weights = mx.random.normal((16, 20, 20))  # (batch*heads, seq, seq)
        attn_weights = mx.softmax(attn_weights, axis=-1)
        mx.eval(x, attn_weights)

        out = attn(x, attn_weights)
        mx.eval(out)

        assert out.shape == (20, 4, 192)

    def test_streaming_forward_shape(self):
        """Test streaming_forward output shapes."""
        attn = SelfAttention(d_model=192, num_heads=4, value_head_dim=24)
        seq_len = 16
        batch_size = 2
        left_context_len = 64
        kv_len = left_context_len + seq_len  # 80
        attention_dim = 4 * 24  # num_heads * value_head_dim = 96

        x = mx.random.normal((seq_len, batch_size, 192))
        # Attention weights from streaming attention: (batch*heads, seq_len, kv_len)
        attn_weights = mx.random.normal((batch_size * 4, seq_len, kv_len))
        attn_weights = mx.softmax(attn_weights, axis=-1)
        cached_val = mx.zeros((left_context_len, batch_size, attention_dim))
        mx.eval(x, attn_weights, cached_val)

        out, new_cached_val = attn.streaming_forward(x, attn_weights, cached_val, left_context_len)
        mx.eval(out, new_cached_val)

        # Output: (seq_len, batch, d_model)
        assert out.shape == (seq_len, batch_size, 192)
        # Cached value: (left_ctx, batch, attention_dim)
        assert new_cached_val.shape == (left_context_len, batch_size, attention_dim)

    def test_streaming_multiple_chunks(self):
        """Test streaming over multiple chunks."""
        attn = SelfAttention(d_model=192, num_heads=4, value_head_dim=24)
        seq_len = 16
        batch_size = 2
        left_context_len = 32
        kv_len = left_context_len + seq_len
        attention_dim = 4 * 24  # 96

        # Initialize cache
        cached_val = mx.zeros((left_context_len, batch_size, attention_dim))

        for _i in range(3):
            x = mx.random.normal((seq_len, batch_size, 192))
            attn_weights = mx.random.normal((batch_size * 4, seq_len, kv_len))
            attn_weights = mx.softmax(attn_weights, axis=-1)
            mx.eval(x, attn_weights, cached_val)

            out, cached_val = attn.streaming_forward(x, attn_weights, cached_val, left_context_len)
            mx.eval(out, cached_val)

            assert out.shape == (seq_len, batch_size, 192)
            assert cached_val.shape == (left_context_len, batch_size, attention_dim)


class TestNonlinAttention:
    """Test NonlinAttention module."""

    def test_init(self):
        na = NonlinAttention(d_model=192, hidden_channels=144)
        assert na.hidden_channels == 144

    def test_forward_shape(self):
        na = NonlinAttention(d_model=192, hidden_channels=144)
        x = mx.random.normal((20, 4, 192))
        attn_weights = mx.random.normal((16, 20, 20))
        attn_weights = mx.softmax(attn_weights, axis=-1)
        mx.eval(x, attn_weights)

        out = na(x, attn_weights)
        mx.eval(out)

        assert out.shape == (20, 4, 192)

    def test_streaming_forward_shape(self):
        """Test streaming_forward output shapes."""
        na = NonlinAttention(d_model=192, hidden_channels=144)
        seq_len = 16
        batch_size = 2
        left_context_len = 64
        kv_len = left_context_len + seq_len

        x = mx.random.normal((seq_len, batch_size, 192))
        # Attention weights from streaming attention: (batch*heads, seq_len, kv_len)
        # Assuming 4 heads
        attn_weights = mx.random.normal((batch_size * 4, seq_len, kv_len))
        attn_weights = mx.softmax(attn_weights, axis=-1)
        cached_v = mx.zeros((left_context_len, batch_size, 144))
        mx.eval(x, attn_weights, cached_v)

        out, new_cached_v = na.streaming_forward(x, attn_weights, cached_v, left_context_len)
        mx.eval(out, new_cached_v)

        assert out.shape == (seq_len, batch_size, 192)
        assert new_cached_v.shape == (left_context_len, batch_size, 144)


class TestZipformer2EncoderLayer:
    """Test Zipformer2EncoderLayer."""

    def test_init(self):
        layer = Zipformer2EncoderLayer(
            d_model=192,
            attention_dim=128,
            num_heads=4,
            ff1_dim=384,
            ff2_dim=512,
            ff3_dim=640,
            kernel_size=31,
            pos_head_dim=4,
            pos_emb_dim=48,
            value_head_dim=12,
            causal=True,
        )
        assert layer.d_model == 192

    def test_forward_shape(self):
        layer = Zipformer2EncoderLayer(
            d_model=192,
            attention_dim=128,
            num_heads=4,
            ff1_dim=384,
            ff2_dim=512,
            ff3_dim=640,
            kernel_size=31,
            pos_head_dim=4,
            pos_emb_dim=48,
            value_head_dim=12,
            causal=True,
        )
        x = mx.random.normal((20, 4, 192))  # (seq, batch, d_model)
        pos_emb = mx.random.normal((4, 39, 48))  # (batch, 2*seq-1, pos_emb_dim)
        mx.eval(x, pos_emb)

        out = layer(x, pos_emb)
        mx.eval(out)

        assert out.shape == (20, 4, 192)

    def test_streaming_forward_shape(self):
        """Test streaming_forward output shapes."""
        d_model = 192
        attention_dim = 128
        num_heads = 4
        kernel_size = 31
        value_head_dim = 12
        pos_emb_dim = 48

        layer = Zipformer2EncoderLayer(
            d_model=d_model,
            attention_dim=attention_dim,
            num_heads=num_heads,
            ff1_dim=384,
            ff2_dim=512,
            ff3_dim=640,
            kernel_size=kernel_size,
            pos_head_dim=4,
            pos_emb_dim=pos_emb_dim,
            value_head_dim=value_head_dim,
            causal=True,
        )

        seq_len = 16
        batch_size = 2
        left_context_len = 64
        kv_len = left_context_len + seq_len

        # Compute dimensions
        query_dim = num_heads * (attention_dim // num_heads)  # 128
        attention_value_dim = num_heads * value_head_dim  # 48
        hidden_channels = 3 * d_model // 4  # 144
        conv_cache_len = kernel_size // 2  # 15

        x = mx.random.normal((seq_len, batch_size, d_model))
        # Extended positional embedding for streaming
        pos_emb = mx.random.normal((batch_size, seq_len + kv_len - 1, pos_emb_dim))

        # Initialize caches
        cached_key = mx.zeros((left_context_len, batch_size, query_dim))
        cached_val1 = mx.zeros((left_context_len, batch_size, attention_value_dim))
        cached_val2 = mx.zeros((left_context_len, batch_size, attention_value_dim))
        cached_nonlin_attn = mx.zeros((left_context_len, batch_size, hidden_channels))
        cached_conv1 = mx.zeros((batch_size, d_model, conv_cache_len))
        cached_conv2 = mx.zeros((batch_size, d_model, conv_cache_len))

        mx.eval(x, pos_emb, cached_key, cached_val1, cached_val2, cached_nonlin_attn, cached_conv1, cached_conv2)

        result = layer.streaming_forward(
            x, pos_emb,
            cached_key, cached_val1, cached_val2, cached_nonlin_attn,
            cached_conv1, cached_conv2,
            left_context_len,
        )
        (out, new_key, new_val1, new_val2, new_nonlin, new_conv1, new_conv2) = result
        mx.eval(out, new_key, new_val1, new_val2, new_nonlin, new_conv1, new_conv2)

        # Check output shape
        assert out.shape == (seq_len, batch_size, d_model)
        # Check cache shapes
        assert new_key.shape == cached_key.shape
        assert new_val1.shape == cached_val1.shape
        assert new_val2.shape == cached_val2.shape
        assert new_nonlin.shape == cached_nonlin_attn.shape
        assert new_conv1.shape == cached_conv1.shape
        assert new_conv2.shape == cached_conv2.shape


class TestZipformer2Encoder:
    """Test Zipformer2Encoder (stack of layers)."""

    def test_init(self):
        encoder = Zipformer2Encoder(
            d_model=192,
            attention_dim=128,
            num_heads=4,
            ff1_dim=384,
            ff2_dim=512,
            ff3_dim=640,
            num_layers=2,
            pos_dim=48,
            pos_head_dim=4,
            value_head_dim=12,
        )
        assert encoder.num_layers == 2
        assert len(encoder.layers) == 2

    def test_forward_shape(self):
        encoder = Zipformer2Encoder(
            d_model=192,
            attention_dim=128,
            num_heads=4,
            ff1_dim=384,
            ff2_dim=512,
            ff3_dim=640,
            num_layers=2,
            pos_dim=48,
            pos_head_dim=4,
            value_head_dim=12,
        )
        x = mx.random.normal((20, 4, 192))
        mx.eval(x)

        out = encoder(x)
        mx.eval(out)

        assert out.shape == (20, 4, 192)

    def test_streaming_forward_shape(self):
        """Test streaming_forward output and state shapes."""
        d_model = 192
        attention_dim = 128
        num_heads = 4
        ff1_dim = 384
        ff2_dim = 512
        ff3_dim = 640
        num_layers = 2
        kernel_size = 31
        value_head_dim = 12
        pos_dim = 48

        encoder = Zipformer2Encoder(
            d_model=d_model,
            attention_dim=attention_dim,
            num_heads=num_heads,
            ff1_dim=ff1_dim,
            ff2_dim=ff2_dim,
            ff3_dim=ff3_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            pos_dim=pos_dim,
            pos_head_dim=4,
            value_head_dim=value_head_dim,
        )

        seq_len = 16
        batch_size = 2
        left_context_len = 64

        # Initialize states
        states = encoder.init_states(batch_size, left_context_len)
        mx.eval(*states)

        # Check state count: 6 states per layer
        assert len(states) == num_layers * 6

        # Run streaming forward
        x = mx.random.normal((seq_len, batch_size, d_model))
        mx.eval(x)

        out, new_states = encoder.streaming_forward(x, states, left_context_len)
        mx.eval(out, *new_states)

        # Check output shape
        assert out.shape == (seq_len, batch_size, d_model)

        # Check state shapes are preserved
        assert len(new_states) == len(states)
        for old_state, new_state in zip(states, new_states, strict=False):
            assert old_state.shape == new_state.shape

    def test_streaming_multiple_chunks(self):
        """Test processing multiple chunks sequentially."""
        d_model = 192
        attention_dim = 128
        num_heads = 4
        num_layers = 2
        kernel_size = 31
        value_head_dim = 12
        pos_dim = 48

        encoder = Zipformer2Encoder(
            d_model=d_model,
            attention_dim=attention_dim,
            num_heads=num_heads,
            ff1_dim=384,
            ff2_dim=512,
            ff3_dim=640,
            num_layers=num_layers,
            kernel_size=kernel_size,
            pos_dim=pos_dim,
            pos_head_dim=4,
            value_head_dim=value_head_dim,
        )

        seq_len = 16
        batch_size = 2
        left_context_len = 64

        # Initialize states
        states = encoder.init_states(batch_size, left_context_len)
        mx.eval(*states)

        # Process multiple chunks
        outputs = []
        for _ in range(3):
            x = mx.random.normal((seq_len, batch_size, d_model))
            mx.eval(x)

            out, states = encoder.streaming_forward(x, states, left_context_len)
            mx.eval(out, *states)
            outputs.append(out)

        # All outputs should have correct shape
        for out in outputs:
            assert out.shape == (seq_len, batch_size, d_model)


class TestDownsampledZipformer2Encoder:
    """Test DownsampledZipformer2Encoder."""

    def test_init(self):
        encoder = DownsampledZipformer2Encoder(
            d_model=256,
            attention_dim=128,
            num_heads=4,
            ff1_dim=576,
            ff2_dim=768,
            ff3_dim=960,
            num_layers=2,
            downsample=2,
            pos_dim=48,
            pos_head_dim=4,
            value_head_dim=12,
        )
        assert encoder.downsample_factor == 2

    def test_forward_shape(self):
        encoder = DownsampledZipformer2Encoder(
            d_model=256,
            attention_dim=128,
            num_heads=4,
            ff1_dim=576,
            ff2_dim=768,
            ff3_dim=960,
            num_layers=2,
            downsample=2,
            pos_dim=48,
            pos_head_dim=4,
            value_head_dim=12,
        )
        x = mx.random.normal((100, 4, 256))
        mx.eval(x)

        out = encoder(x)
        mx.eval(out)

        # Output should have same shape as input (upsample restores length)
        assert out.shape == (100, 4, 256)

    def test_streaming_forward_shape(self):
        """Test streaming_forward output and state shapes."""
        d_model = 256
        attention_dim = 128
        num_heads = 4
        ff1_dim = 576
        ff2_dim = 768
        ff3_dim = 960
        num_layers = 2
        downsample = 2
        kernel_size = 31
        value_head_dim = 12
        pos_dim = 48

        encoder = DownsampledZipformer2Encoder(
            d_model=d_model,
            attention_dim=attention_dim,
            num_heads=num_heads,
            ff1_dim=ff1_dim,
            ff2_dim=ff2_dim,
            ff3_dim=ff3_dim,
            num_layers=num_layers,
            downsample=downsample,
            kernel_size=kernel_size,
            pos_dim=pos_dim,
            pos_head_dim=4,
            value_head_dim=value_head_dim,
        )

        seq_len = 32  # Must be multiple of downsample for clean processing
        batch_size = 2
        left_context_len = 32  # In downsampled space

        # Initialize states
        states = encoder.init_states(batch_size, left_context_len)
        mx.eval(*states)

        # Check state count: 6 states per layer
        assert len(states) == num_layers * 6

        # Run streaming forward
        x = mx.random.normal((seq_len, batch_size, d_model))
        mx.eval(x)

        out, new_states = encoder.streaming_forward(x, states, left_context_len)
        mx.eval(out, *new_states)

        # Check output shape - should match input
        assert out.shape == (seq_len, batch_size, d_model)

        # Check state shapes are preserved
        assert len(new_states) == len(states)
        for old_state, new_state in zip(states, new_states, strict=False):
            assert old_state.shape == new_state.shape

    def test_streaming_multiple_chunks(self):
        """Test processing multiple chunks sequentially."""
        d_model = 256
        attention_dim = 128
        num_heads = 4
        num_layers = 2
        downsample = 2
        kernel_size = 31
        value_head_dim = 12
        pos_dim = 48

        encoder = DownsampledZipformer2Encoder(
            d_model=d_model,
            attention_dim=attention_dim,
            num_heads=num_heads,
            ff1_dim=576,
            ff2_dim=768,
            ff3_dim=960,
            num_layers=num_layers,
            downsample=downsample,
            kernel_size=kernel_size,
            pos_dim=pos_dim,
            pos_head_dim=4,
            value_head_dim=value_head_dim,
        )

        seq_len = 32
        batch_size = 2
        left_context_len = 32

        # Initialize states
        states = encoder.init_states(batch_size, left_context_len)
        mx.eval(*states)

        # Process multiple chunks
        outputs = []
        for _ in range(3):
            x = mx.random.normal((seq_len, batch_size, d_model))
            mx.eval(x)

            out, states = encoder.streaming_forward(x, states, left_context_len)
            mx.eval(out, *states)
            outputs.append(out)

        # All outputs should have correct shape
        for out in outputs:
            assert out.shape == (seq_len, batch_size, d_model)


class TestConv2dSubsampling:
    """Test Conv2dSubsampling (encoder_embed)."""

    def test_init(self):
        embed = Conv2dSubsampling(
            num_features=80,
            output_dim=192,
            intermediate_dim=128,
        )
        assert embed.output_dim == 192
        assert embed.out_width == 19  # (((80-1)//2)-1)//2 = 19
        # Linear layer should expect 128 * 19 = 2432 input features
        assert embed.out.weight.shape == (192, 2432)

    def test_forward_shape(self):
        """Test forward pass with correct architecture dimensions.

        Architecture follows icefall subsampling.py:
        - conv0: padding=(0,1), stride=1 -> T-2, F
        - conv4: padding=0, stride=2 -> (T-5)//2+1, 39
        - conv7: padding=0, stride=(1,2) -> (T-7)//2, 19

        Final: T' = (T-7)//2, F' = 19, linear_in = 128 * 19 = 2432
        """
        embed = Conv2dSubsampling(
            num_features=80,
            output_dim=192,
            intermediate_dim=128,
        )

        # Input: (batch, time, features)
        # Time must be >= 7 to produce valid output
        x = mx.random.normal((4, 107, 80))  # T=107 -> (107-7)//2 = 50
        mx.eval(x)

        out = embed(x)
        mx.eval(out)

        # Output: (time', batch, output_dim)
        expected_time = (107 - 7) // 2  # = 50
        assert out.shape == (expected_time, 4, 192)
        assert out.shape[0] == 50
        assert out.shape[1] == 4
        assert out.shape[2] == 192

    def test_forward_shape_various_lengths(self):
        """Test forward pass with various input lengths."""
        embed = Conv2dSubsampling(
            num_features=80,
            output_dim=192,
            intermediate_dim=128,
        )

        # Test various input lengths
        for T in [17, 27, 67, 107, 207]:
            x = mx.random.normal((2, T, 80))
            mx.eval(x)

            out = embed(x)
            mx.eval(out)

            expected_time = (T - 7) // 2
            assert out.shape == (expected_time, 2, 192), f"Failed for T={T}"


class TestConv2dSubsamplingStreaming:
    """Streaming-specific tests for Conv2dSubsampling."""

    def test_streaming_forward_shape(self):
        embed = Conv2dSubsampling(
            num_features=80,
            output_dim=192,
            intermediate_dim=128,
        )

        batch_size = 2
        x = mx.random.normal((batch_size, 45, 80))
        # State is a tuple: (convnext_state, fbank_cache)
        convnext_state = mx.zeros((batch_size, 128, 3, 19))
        fbank_cache = mx.zeros((batch_size, 7, 80))
        state = (convnext_state, fbank_cache)
        mx.eval(x, convnext_state, fbank_cache)

        out, new_state = embed.streaming_forward(x, state)
        mx.eval(out, new_state[0], new_state[1])

        # For 45 fbank frames + 7 cached = 52 total -> conv stack produces (52-7)//2 = 22 frames
        assert out.shape == (22, batch_size, 192)
        # new_state is tuple: (convnext_state, fbank_cache)
        assert new_state[0].shape == (batch_size, 128, 3, 19)
        assert new_state[1].shape == (batch_size, 7, 80)

    @pytest.mark.skip(
        reason=(
            "ONNX model uses old state structure without fbank caching. "
            "MLX implementation has been updated with streaming frame alignment fix "
            "that adds fbank caching for improved WER (2-6% vs old 17-21%). "
            "Numerical equivalence validated via production tests (test_streaming_asr.py)."
        ),
    )
    def test_new_embed_states_matches_onnx(self):
        try:
            import onnxruntime as ort
        except Exception as e:  # pragma: no cover
            pytest.skip(f"onnxruntime not available: {e}")

        try:
            import torch
        except Exception as e:  # pragma: no cover
            pytest.skip(f"torch not available: {e}")

        from src.models.zipformer.convert_weights import convert_conv2d_weight

        onnx_path = Path(
            "checkpoints/zipformer/sherpa-onnx-streaming-zipformer-en-2023-06-26/"
            "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        )
        pt_path = Path("checkpoints/zipformer/en-streaming/exp/pretrained.pt")

        # Load embed weights from the PyTorch checkpoint to match the ONNX model weights.
        ckpt = torch.load(pt_path, map_location="cpu")
        model_dict = ckpt.get("model", ckpt)

        embed = Conv2dSubsampling(
            num_features=80,
            output_dim=192,
            intermediate_dim=128,
        )

        prefix = "encoder_embed."
        embed.conv0_weight = convert_conv2d_weight(model_dict[f"{prefix}conv.0.weight"])
        embed.conv0_bias = mx.array(model_dict[f"{prefix}conv.0.bias"].numpy())
        embed.conv4_weight = convert_conv2d_weight(model_dict[f"{prefix}conv.4.weight"])
        embed.conv4_bias = mx.array(model_dict[f"{prefix}conv.4.bias"].numpy())
        embed.conv7_weight = convert_conv2d_weight(model_dict[f"{prefix}conv.7.weight"])
        embed.conv7_bias = mx.array(model_dict[f"{prefix}conv.7.bias"].numpy())

        embed.convnext_dw_weight = convert_conv2d_weight(
            model_dict[f"{prefix}convnext.depthwise_conv.weight"],
        )
        embed.convnext_dw_bias = mx.array(
            model_dict[f"{prefix}convnext.depthwise_conv.bias"].numpy(),
        )
        embed.convnext_pw1_weight = convert_conv2d_weight(
            model_dict[f"{prefix}convnext.pointwise_conv1.weight"],
        )
        embed.convnext_pw1_bias = mx.array(
            model_dict[f"{prefix}convnext.pointwise_conv1.bias"].numpy(),
        )
        embed.convnext_pw2_weight = convert_conv2d_weight(
            model_dict[f"{prefix}convnext.pointwise_conv2.weight"],
        )
        embed.convnext_pw2_bias = mx.array(
            model_dict[f"{prefix}convnext.pointwise_conv2.bias"].numpy(),
        )

        embed.out.weight = mx.array(model_dict[f"{prefix}out.weight"].numpy())
        embed.out.bias = mx.array(model_dict[f"{prefix}out.bias"].numpy())
        embed.out_norm_bias = mx.array(model_dict[f"{prefix}out_norm.bias"].numpy())
        embed.out_norm_log_scale = mx.array(model_dict[f"{prefix}out_norm.log_scale"].numpy())

        # Deterministic test input
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((1, 45, 80)).astype(np.float32)

        x_mx = mx.array(x_np)
        # State is now a tuple: (convnext_state, fbank_cache)
        convnext_state_mx = mx.zeros((1, 128, 3, 19), dtype=mx.float32)
        fbank_cache_mx = mx.zeros((1, 7, 80), dtype=mx.float32)
        state_mx = (convnext_state_mx, fbank_cache_mx)
        mx.eval(x_mx, convnext_state_mx, fbank_cache_mx)

        _, new_state = embed.streaming_forward(x_mx, state_mx)
        # new_state is tuple: (new_convnext_state, new_fbank_cache)
        mx.eval(new_state[0], new_state[1])

        # Run ONNX encoder just far enough to compute new_embed_states.
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feed = {}
        for inp in sess.get_inputs():
            shape = [1 if (d is None or d == "N") else int(d) for d in inp.shape]
            if inp.name == "x":
                feed[inp.name] = x_np
            elif inp.name == "embed_states":
                feed[inp.name] = np.zeros(shape, dtype=np.float32)
            elif inp.name == "processed_lens":
                feed[inp.name] = np.zeros(shape, dtype=np.int64)
            else:
                if inp.type == "tensor(int64)":
                    feed[inp.name] = np.zeros(shape, dtype=np.int64)
                else:
                    feed[inp.name] = np.zeros(shape, dtype=np.float32)

        (onnx_new_state,) = sess.run(["new_embed_states"], feed)

        # Compare only the convnext_state portion (ONNX doesn't have fbank_cache)
        np.testing.assert_allclose(
            np.array(new_state[0]),
            onnx_new_state,
            rtol=1e-3,
            atol=1e-3,
        )


class TestZipformerConfig:
    """Test ZipformerConfig."""

    def test_default_config(self):
        config = ZipformerConfig()
        assert config.num_features == 80
        assert len(config.num_encoder_layers) == 6
        assert config.encoder_dims == (192, 256, 384, 512, 384, 256)
        assert config.downsampling_factors == (1, 2, 4, 8, 4, 2)


class TestZipformer:
    """Test full Zipformer model."""

    def test_init(self):
        config = ZipformerConfig()
        model = Zipformer(config)
        assert len(model.encoders) == 6

    def test_forward_shape(self):
        """Test forward pass with corrected architecture.

        Time subsampling:
        - Conv2dSubsampling: T -> (T-7)//2
        - Final downsampling: T -> ceil(T/2)

        For T=207: (207-7)//2 = 100 -> ceil(100/2) = 50
        """
        config = ZipformerConfig()
        model = Zipformer(config)

        # Input: (batch, time, features)
        # Using T=207 for clean arithmetic
        x = mx.random.normal((4, 207, 80))
        mx.eval(x)

        out, out_lens = model(x)
        mx.eval(out)

        # Output: (batch, time', d_model)
        # T' = ceil((T-7)//2 / 2) = ceil(100/2) = 50
        expected_time = ((207 - 7) // 2 + 1) // 2  # ceil div by 2
        assert out.shape[0] == 4
        # Output dimension is max(encoder_dims), not encoder_dims[-1]
        assert out.shape[2] == max(config.encoder_dims)  # max(encoder_dims) = 512
        assert out.shape[1] == expected_time

    def test_forward_with_lengths(self):
        config = ZipformerConfig()
        model = Zipformer(config)

        x = mx.random.normal((4, 207, 80))
        x_lens = mx.array([207, 187, 167, 147])
        mx.eval(x, x_lens)

        out, out_lens = model(x, x_lens)
        mx.eval(out, out_lens)

        assert out.shape[0] == 4
        assert out_lens is not None
        assert out_lens.shape == (4,)

    def test_streaming_init_states(self):
        """Test init_states creates correct number and shapes of states."""
        config = ZipformerConfig()
        model = Zipformer(config)

        batch_size = 2
        left_context_len = 128

        states = model.init_states(batch_size, left_context_len)
        # Note: states[0] is a tuple (convnext_state, fbank_cache) for embed state
        mx.eval(*[s[0] if isinstance(s, tuple) else s for s in states],
                *[s[1] for s in states if isinstance(s, tuple)])

        # Total states: 1 embed + 6 per layer for all stages
        total_layers = sum(config.num_encoder_layers)  # 2+2+3+4+3+2 = 16
        expected_num_states = 1 + total_layers * 6  # 1 + 96 = 97
        assert len(states) == expected_num_states, f"Expected {expected_num_states} states, got {len(states)}"

        # Check embed state shape - now a tuple (convnext_state, fbank_cache)
        assert isinstance(states[0], tuple), "Embed state should be a tuple"
        convnext_state, fbank_cache = states[0]
        assert convnext_state.shape == (batch_size, 128, 3, 19)  # out_width = 19
        assert fbank_cache.shape == (batch_size, 7, 80)  # 7-frame fbank cache

        # Verify get_num_states
        assert model.get_num_states() == expected_num_states

    def test_streaming_forward_shape(self):
        """Test streaming_forward output and state shapes."""
        config = ZipformerConfig()
        model = Zipformer(config)

        batch_size = 2
        chunk_frames = 45  # Input fbank frames per chunk
        left_context_len = 128

        # Initialize states
        states = model.init_states(batch_size, left_context_len)
        # Handle tuple states (embed state is now a tuple)
        mx.eval(*[s[0] if isinstance(s, tuple) else s for s in states],
                *[s[1] for s in states if isinstance(s, tuple)])

        # Input chunk
        x = mx.random.normal((batch_size, chunk_frames, 80))
        mx.eval(x)

        # Run streaming forward
        out, new_states = model.streaming_forward(x, states, left_context_len=left_context_len)
        mx.eval(out, *[s[0] if isinstance(s, tuple) else s for s in new_states],
                *[s[1] for s in new_states if isinstance(s, tuple)])

        # Output: (batch, out_frames, output_dim)
        # Conv2dSubsampling: 45+7 fbank -> (52-7)//2 = 22 conv frames
        # Multi-stage encoders with downsampling factors 1,2,4,8,4,2
        # Final output: 11 frames after encoder processing
        expected_out_frames = 11
        assert out.shape == (batch_size, expected_out_frames, model.output_dim)
        assert model.output_dim == max(config.encoder_dims)  # 512

        # States should be preserved
        assert len(new_states) == len(states)
        for old_state, new_state in zip(states, new_states, strict=False):
            if isinstance(old_state, tuple):
                assert isinstance(new_state, tuple)
                assert old_state[0].shape == new_state[0].shape
                assert old_state[1].shape == new_state[1].shape
            else:
                assert old_state.shape == new_state.shape

    def test_streaming_multiple_chunks(self):
        """Test processing multiple chunks sequentially."""
        config = ZipformerConfig()
        model = Zipformer(config)

        batch_size = 2
        chunk_frames = 45
        left_context_len = 128
        num_chunks = 3

        # Initialize states
        states = model.init_states(batch_size, left_context_len)
        # Handle tuple states (embed state is now a tuple)
        mx.eval(*[s[0] if isinstance(s, tuple) else s for s in states],
                *[s[1] for s in states if isinstance(s, tuple)])

        outputs = []
        for _ in range(num_chunks):
            x = mx.random.normal((batch_size, chunk_frames, 80))
            mx.eval(x)

            out, states = model.streaming_forward(x, states, left_context_len=left_context_len)
            mx.eval(out, *[s[0] if isinstance(s, tuple) else s for s in states],
                    *[s[1] for s in states if isinstance(s, tuple)])
            outputs.append(out)

        # All outputs should have correct shape
        # Conv2dSubsampling: 45+7 fbank -> 22 conv frames
        # Final output: 11 frames after encoder processing
        expected_out_frames = 11
        for out in outputs:
            assert out.shape == (batch_size, expected_out_frames, model.output_dim)


class TestWeightLoading:
    """Test weight loading functionality."""

    @pytest.mark.skipif(
        not __import__('pathlib').Path('checkpoints/zipformer/en-streaming/exp/pretrained.pt').exists(),
        reason="Pretrained checkpoint not available",
    )
    def test_weight_loading_keys(self):
        """Test that we can identify all weight keys correctly."""
        import torch

        ckpt_path = 'checkpoints/zipformer/en-streaming/exp/pretrained.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_dict = ckpt['model']

        # Check that we have expected encoder keys
        encoder_keys = [k for k in model_dict.keys() if k.startswith('encoder.')]
        assert len(encoder_keys) > 0

        # Check encoder_embed keys exist
        embed_keys = [k for k in model_dict.keys() if k.startswith('encoder_embed.')]
        assert len(embed_keys) > 0

        # Check layer structure
        layer_keys = [k for k in model_dict.keys() if 'layers.0.' in k]
        assert len(layer_keys) > 0
