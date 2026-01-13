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
Tests for T5 MLX Model Implementation (MADLAD/T5)

Tests the encoder-decoder transformer architecture with relative position biases
and KV-cache preallocation without requiring actual HuggingFace weights.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

# Skip if MLX not available
mlx = pytest.importorskip("mlx.core")
mx = mlx


def make_config(
    vocab_size=1000,
    d_model=64,
    d_kv=16,
    d_ff=128,
    num_layers=2,
    num_heads=4,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    layer_norm_epsilon=1e-6,
    feed_forward_proj="gated-gelu",
    tie_word_embeddings=True,
):
    """Create a small test config as SimpleNamespace."""
    return SimpleNamespace(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        relative_attention_num_buckets=relative_attention_num_buckets,
        relative_attention_max_distance=relative_attention_max_distance,
        layer_norm_epsilon=layer_norm_epsilon,
        feed_forward_proj=feed_forward_proj,
        tie_word_embeddings=tie_word_embeddings,
    )


class TestT5KVCache:
    """Test T5KVCache for preallocated KV cache (OPT-5)."""

    def test_cache_initialization(self):
        """Test cache starts empty."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        cache = T5KVCache()
        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0

    def test_cache_update_and_fetch(self):
        """Test updating cache with new keys/values."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        cache = T5KVCache()

        # Create test keys/values (T5 uses transposed key layout)
        B, H, D, S = 2, 4, 16, 3
        keys = mx.random.normal((B, H, D, S))
        values = mx.random.normal((B, H, S, D))

        cached_k, cached_v = cache.update_and_fetch(keys, values)
        mx.eval(cached_k, cached_v)

        assert cache.offset == S
        assert cached_k.shape == (B, H, D, S)
        assert cached_v.shape == (B, H, S, D)

    def test_cache_incremental_update(self):
        """Test incremental cache updates."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        cache = T5KVCache()
        B, H, D = 2, 4, 16

        # First update
        keys1 = mx.random.normal((B, H, D, 3))
        values1 = mx.random.normal((B, H, 3, D))
        k1, v1 = cache.update_and_fetch(keys1, values1)
        mx.eval(k1, v1)

        assert cache.offset == 3

        # Second update
        keys2 = mx.random.normal((B, H, D, 1))
        values2 = mx.random.normal((B, H, 1, D))
        k2, v2 = cache.update_and_fetch(keys2, values2)
        mx.eval(k2, v2)

        assert cache.offset == 4
        assert k2.shape[3] == 4
        assert v2.shape[2] == 4

    def test_cache_get_state(self):
        """Test getting cache state as tuple."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        cache = T5KVCache()

        # Empty cache
        state = cache.get_state()
        assert state is None

        # After update
        B, H, D, S = 2, 4, 16, 3
        keys = mx.random.normal((B, H, D, S))
        values = mx.random.normal((B, H, S, D))
        cache.update_and_fetch(keys, values)

        state = cache.get_state()
        assert state is not None
        assert len(state) == 2
        assert state[0].shape == (B, H, D, S)
        assert state[1].shape == (B, H, S, D)

    def test_cache_trim(self):
        """Test trimming tokens from cache."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        cache = T5KVCache()
        B, H, D = 2, 4, 16

        # Fill cache
        keys = mx.random.normal((B, H, D, 5))
        values = mx.random.normal((B, H, 5, D))
        cache.update_and_fetch(keys, values)

        assert cache.offset == 5

        # Trim 2 tokens
        trimmed = cache.trim(2)
        assert trimmed == 2
        assert cache.offset == 3

    def test_cache_trim_clamped(self):
        """Test trim doesn't go below zero."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        cache = T5KVCache()
        B, H, D = 2, 4, 16

        keys = mx.random.normal((B, H, D, 3))
        values = mx.random.normal((B, H, 3, D))
        cache.update_and_fetch(keys, values)

        # Trim more than available
        trimmed = cache.trim(10)
        assert trimmed == 3
        assert cache.offset == 0

    def test_cache_from_tuple(self):
        """Test creating cache from existing tuple."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        B, H, D, S = 2, 4, 16, 5
        keys = mx.random.normal((B, H, D, S))
        values = mx.random.normal((B, H, S, D))

        cache = T5KVCache.from_tuple((keys, values))

        assert cache.offset == S
        assert cache.keys is not None

    def test_cache_copy(self):
        """Test deep copy of cache."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        cache = T5KVCache()
        B, H, D = 2, 4, 16

        keys = mx.random.normal((B, H, D, 5))
        values = mx.random.normal((B, H, 5, D))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        # Copy
        cache_copy = cache.copy()

        # Modify original
        cache.trim(2)

        # Copy should be unaffected
        assert cache.offset == 3
        assert cache_copy.offset == 5

    def test_cache_is_trimmable(self):
        """Test is_trimmable always returns True."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        cache = T5KVCache()
        assert cache.is_trimmable() is True


class TestCopyCacheList:
    """Test copy_cache_list utility function."""

    def test_copy_none(self):
        """Test copying None returns None."""
        from pytorch_to_mlx.converters.models.t5_mlx import copy_cache_list

        assert copy_cache_list(None) is None

    def test_copy_cache_list(self):
        """Test deep copying a list of caches."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache, copy_cache_list

        B, H, D = 2, 4, 16

        # Create cache list
        cache_list = []
        for _ in range(3):
            cache = T5KVCache()
            keys = mx.random.normal((B, H, D, 5))
            values = mx.random.normal((B, H, 5, D))
            cache.update_and_fetch(keys, values)
            mx.eval(cache.keys, cache.values)
            cache_list.append(cache)

        # Copy
        copied = copy_cache_list(cache_list)

        # Modify original
        cache_list[0].trim(2)

        # Copy should be unaffected
        assert copied[0].offset == 5


class TestRelativePositionBucket:
    """Test relative position bucket computation."""

    def test_relative_position_bucket_bidirectional(self):
        """Test bidirectional relative position buckets."""
        from pytorch_to_mlx.converters.models.t5_mlx import _relative_position_bucket

        # Create relative positions
        rel_pos = mx.array([[-5, -3, -1, 0, 1, 3, 5]])

        buckets = _relative_position_bucket(rel_pos, bidirectional=True)
        mx.eval(buckets)

        # Buckets should be valid indices
        assert buckets.shape == rel_pos.shape
        assert mx.all(buckets >= 0)
        assert mx.all(buckets < 32)

    def test_relative_position_bucket_unidirectional(self):
        """Test unidirectional (causal) relative position buckets."""
        from pytorch_to_mlx.converters.models.t5_mlx import _relative_position_bucket

        rel_pos = mx.array([[-5, -3, -1, 0, 1, 3, 5]])

        buckets = _relative_position_bucket(rel_pos, bidirectional=False)
        mx.eval(buckets)

        assert buckets.shape == rel_pos.shape
        assert mx.all(buckets >= 0)
        assert mx.all(buckets < 32)

    def test_relative_position_bucket_matrix(self):
        """Test with full position matrix."""
        from pytorch_to_mlx.converters.models.t5_mlx import _relative_position_bucket

        seq_len = 10
        query_pos = mx.arange(seq_len)[:, None]
        key_pos = mx.arange(seq_len)[None, :]
        rel_pos = key_pos - query_pos

        buckets = _relative_position_bucket(rel_pos, bidirectional=True)
        mx.eval(buckets)

        assert buckets.shape == (seq_len, seq_len)


class TestRelativePositionBias:
    """Test RelativePositionBias module."""

    @pytest.fixture
    def config(self):
        return make_config()

    def test_position_bias_forward(self, config):
        """Test relative position bias forward pass."""
        from pytorch_to_mlx.converters.models.t5_mlx import RelativePositionBias

        bias = RelativePositionBias(config, bidirectional=True)

        query_len, key_len = 10, 10
        output = bias(query_len, key_len)
        mx.eval(output)

        assert output.shape == (config.num_heads, query_len, key_len)

    def test_position_bias_asymmetric(self, config):
        """Test with different query/key lengths."""
        from pytorch_to_mlx.converters.models.t5_mlx import RelativePositionBias

        bias = RelativePositionBias(config, bidirectional=True)

        output = bias(query_length=5, key_length=10)
        mx.eval(output)

        assert output.shape == (config.num_heads, 5, 10)

    def test_position_bias_with_offset(self, config):
        """Test position bias with cache offset."""
        from pytorch_to_mlx.converters.models.t5_mlx import RelativePositionBias

        bias = RelativePositionBias(config, bidirectional=False)

        # Simulate cached decoding
        output = bias(query_length=10, key_length=10, offset=5)
        mx.eval(output)

        assert output.shape == (config.num_heads, 5, 10)


class TestMultiHeadAttention:
    """Test MultiHeadAttention module."""

    @pytest.fixture
    def config(self):
        return make_config()

    @pytest.fixture
    def attention(self, config):
        from pytorch_to_mlx.converters.models.t5_mlx import MultiHeadAttention

        return MultiHeadAttention(config)

    def test_attention_self(self, attention):
        """Test self-attention forward pass."""
        B, L, D = 2, 10, 64
        x = mx.random.normal((B, L, D))

        output, cache = attention(x, x, x, mask=None)
        mx.eval(output)

        assert output.shape == (B, L, D)

    def test_attention_cross(self, attention):
        """Test cross-attention forward pass."""
        B, L_q, L_kv, D = 2, 8, 12, 64
        queries = mx.random.normal((B, L_q, D))
        keys = mx.random.normal((B, L_kv, D))
        values = mx.random.normal((B, L_kv, D))

        output, _ = attention(queries, keys, values, mask=None)
        mx.eval(output)

        assert output.shape == (B, L_q, D)

    def test_attention_with_cache(self, attention):
        """Test attention with KV cache."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        B, D = 2, 64

        cache = T5KVCache()

        # First token
        x = mx.random.normal((B, 1, D))
        output1, cache = attention(x, x, x, mask=None, cache=cache)
        mx.eval(output1)

        assert cache.offset == 1

        # Second token
        x2 = mx.random.normal((B, 1, D))
        output2, cache = attention(x2, x2, x2, mask=None, cache=cache)
        mx.eval(output2)

        assert cache.offset == 2


class TestDenseActivation:
    """Test DenseActivation (FFN) module."""

    def test_dense_gated_gelu(self):
        """Test gated GELU FFN."""
        from pytorch_to_mlx.converters.models.t5_mlx import DenseActivation

        config = make_config(feed_forward_proj="gated-gelu")
        ffn = DenseActivation(config)

        B, L, D = 2, 10, 64
        x = mx.random.normal((B, L, D))

        output = ffn(x)
        mx.eval(output)

        assert output.shape == (B, L, D)

    def test_dense_gated_silu(self):
        """Test gated SiLU FFN."""
        from pytorch_to_mlx.converters.models.t5_mlx import DenseActivation

        config = make_config(feed_forward_proj="gated-silu")
        ffn = DenseActivation(config)

        B, L, D = 2, 10, 64
        x = mx.random.normal((B, L, D))

        output = ffn(x)
        mx.eval(output)

        assert output.shape == (B, L, D)

    def test_dense_relu(self):
        """Test non-gated ReLU FFN."""
        from pytorch_to_mlx.converters.models.t5_mlx import DenseActivation

        # Non-gated config (no feed_forward_proj attribute)
        config = SimpleNamespace(
            d_model=64,
            d_ff=128,
        )
        ffn = DenseActivation(config)

        B, L, D = 2, 10, 64
        x = mx.random.normal((B, L, D))

        output = ffn(x)
        mx.eval(output)

        assert output.shape == (B, L, D)


class TestTransformerEncoderLayer:
    """Test TransformerEncoderLayer module."""

    @pytest.fixture
    def config(self):
        return make_config()

    def test_encoder_layer_forward(self, config):
        """Test encoder layer forward pass."""
        from pytorch_to_mlx.converters.models.t5_mlx import TransformerEncoderLayer

        layer = TransformerEncoderLayer(config)

        B, L, D = 2, 10, 64
        x = mx.random.normal((B, L, D))
        mask = mx.zeros((config.num_heads, L, L))

        output = layer(x, mask)
        mx.eval(output)

        assert output.shape == (B, L, D)


class TestTransformerEncoder:
    """Test TransformerEncoder module."""

    @pytest.fixture
    def config(self):
        return make_config()

    @pytest.fixture
    def encoder(self, config):
        from pytorch_to_mlx.converters.models.t5_mlx import TransformerEncoder

        return TransformerEncoder(config)

    def test_encoder_forward(self, encoder):
        """Test encoder forward pass."""
        B, L, D = 2, 10, 64
        x = mx.random.normal((B, L, D))

        output = encoder(x)
        mx.eval(output)

        assert output.shape == (B, L, D)

    def test_encoder_variable_length(self, encoder):
        """Test encoder with different sequence lengths."""
        B, D = 2, 64

        x_short = mx.random.normal((B, 5, D))
        x_long = mx.random.normal((B, 20, D))

        out_short = encoder(x_short)
        out_long = encoder(x_long)
        mx.eval(out_short, out_long)

        assert out_short.shape == (B, 5, D)
        assert out_long.shape == (B, 20, D)


class TestTransformerDecoderLayer:
    """Test TransformerDecoderLayer module."""

    @pytest.fixture
    def config(self):
        return make_config()

    def test_decoder_layer_forward(self, config):
        """Test decoder layer forward pass."""
        from pytorch_to_mlx.converters.models.t5_mlx import TransformerDecoderLayer

        layer = TransformerDecoderLayer(config)

        B, L_dec, L_enc, D = 2, 8, 10, 64
        x = mx.random.normal((B, L_dec, D))
        memory = mx.random.normal((B, L_enc, D))
        mask = mx.zeros((config.num_heads, L_dec, L_dec))

        output, cache = layer(x, memory, mask, memory_mask=None)
        mx.eval(output)

        assert output.shape == (B, L_dec, D)

    def test_decoder_layer_with_cache(self, config):
        """Test decoder layer with KV cache."""
        from pytorch_to_mlx.converters.models.t5_mlx import (
            T5KVCache,
            TransformerDecoderLayer,
        )

        layer = TransformerDecoderLayer(config)

        B, L_enc, D = 2, 10, 64
        memory = mx.random.normal((B, L_enc, D))
        cache = T5KVCache()

        # First token
        x1 = mx.random.normal((B, 1, D))
        mask = mx.zeros((config.num_heads, 1, 1))
        out1, cache = layer(x1, memory, mask, memory_mask=None, cache=cache)
        mx.eval(out1)

        assert cache.offset == 1

        # Second token
        x2 = mx.random.normal((B, 1, D))
        mask2 = mx.zeros((config.num_heads, 1, 2))
        out2, cache = layer(x2, memory, mask2, memory_mask=None, cache=cache)
        mx.eval(out2)

        assert cache.offset == 2


class TestTransformerDecoder:
    """Test TransformerDecoder module."""

    @pytest.fixture
    def config(self):
        return make_config()

    @pytest.fixture
    def decoder(self, config):
        from pytorch_to_mlx.converters.models.t5_mlx import TransformerDecoder

        return TransformerDecoder(config)

    def test_decoder_forward(self, decoder, config):
        """Test decoder forward pass."""
        B, L_dec, L_enc, D = 2, 8, 10, 64
        x = mx.random.normal((B, L_dec, D))
        memory = mx.random.normal((B, L_enc, D))

        output, cache = decoder(x, memory, mask=None, memory_mask=None)
        mx.eval(output)

        assert output.shape == (B, L_dec, D)
        assert len(cache) == config.num_layers

    def test_decoder_make_cache(self, decoder, config):
        """Test cache creation."""
        cache = decoder.make_cache()

        assert len(cache) == config.num_layers
        assert all(c.offset == 0 for c in cache)

    def test_decoder_autoregressive(self, decoder, config):
        """Test autoregressive decoding."""
        B, L_enc, D = 2, 10, 64
        memory = mx.random.normal((B, L_enc, D))

        cache = None

        # Generate 5 tokens
        for i in range(5):
            x = mx.random.normal((B, 1, D))
            output, cache = decoder(x, memory, mask=None, memory_mask=None, cache=cache)
            mx.eval(output)

            assert output.shape == (B, 1, D)
            assert cache[0].offset == i + 1


class TestT5Model:
    """Test T5Model complete encoder-decoder."""

    @pytest.fixture
    def config(self):
        return make_config()

    @pytest.fixture
    def model(self, config):
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        return T5Model(config)

    def test_model_encode(self, model):
        """Test model encode method."""
        B, L = 2, 10
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * B)

        encoder_output = model.encode(input_ids)
        mx.eval(encoder_output)

        assert encoder_output.shape == (B, L, 64)

    def test_model_decode(self, model):
        """Test model decode method."""
        B, L_enc, L_dec = 2, 10, 5
        memory = mx.random.normal((B, L_enc, 64))
        decoder_input_ids = mx.array([[0, 1, 2, 3, 4]] * B)

        logits, cache = model.decode(decoder_input_ids, memory)
        mx.eval(logits)

        assert logits.shape == (B, L_dec, 1000)

    def test_model_full_forward(self, model):
        """Test full encoder-decoder via __call__."""
        B, L_dec = 2, 5
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * B)
        decoder_input_ids = mx.array([[0, 1, 2, 3, 4]] * B)

        logits = model(input_ids, decoder_input_ids)
        mx.eval(logits)

        assert logits.shape == (B, L_dec, 1000)

    def test_model_autoregressive_generation(self, model):
        """Test autoregressive generation with cache."""
        B = 2
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * B)

        # Encode once
        memory = model.encode(input_ids)
        mx.eval(memory)

        # Generate token by token
        decoder_input_ids = mx.array([[0]] * B)  # Start token
        cache = None

        for _ in range(5):
            logits, cache = model.decode(decoder_input_ids, memory, cache=cache)
            mx.eval(logits)

            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            decoder_input_ids = next_token

        assert logits.shape == (B, 1, 1000)

    def test_model_tied_embeddings(self, config):
        """Test tied word embeddings."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        config.tie_word_embeddings = True
        model = T5Model(config)

        # Should not have separate lm_head
        assert not hasattr(model, "lm_head") or model.tie_word_embeddings

    def test_model_untied_embeddings(self, config):
        """Test untied word embeddings."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        config.tie_word_embeddings = False
        model = T5Model(config)

        assert hasattr(model, "lm_head")


class TestT5ModelEarlyExit:
    """Test T5Model early exit for speculative decoding."""

    @pytest.fixture
    def config(self):
        return make_config(num_layers=6)

    @pytest.fixture
    def model(self, config):
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        return T5Model(config)

    def test_early_exit_decode(self, model):
        """Test decode with early exit."""
        B, L_enc = 2, 10
        memory = mx.random.normal((B, L_enc, 64))
        decoder_input_ids = mx.array([[0, 1, 2]] * B)

        logits, cache = model.decode_early_exit(
            decoder_input_ids, memory, num_layers=2,
        )
        mx.eval(logits)

        assert logits.shape == (B, 3, 1000)
        # Cache should only have num_layers entries
        assert len(cache) == 2

    def test_early_exit_with_cache(self, model):
        """Test early exit with existing cache."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        B, L_enc = 2, 10
        memory = mx.random.normal((B, L_enc, 64))

        # Create initial cache with 2 layers
        cache = [T5KVCache() for _ in range(2)]

        # First token
        input1 = mx.array([[0]] * B)
        out1, cache = model.decode_early_exit(input1, memory, num_layers=2, cache=cache)
        mx.eval(out1)

        assert cache[0].offset == 1

        # Second token
        input2 = mx.array([[1]] * B)
        out2, cache = model.decode_early_exit(input2, memory, num_layers=2, cache=cache)
        mx.eval(out2)

        assert cache[0].offset == 2


class TestT5ModelTrimCache:
    """Test T5Model cache trimming."""

    @pytest.fixture
    def config(self):
        return make_config()

    @pytest.fixture
    def model(self, config):
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        return T5Model(config)

    def test_trim_decoder_cache(self, model):
        """Test trimming decoder cache."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        B, H, D = 2, 4, 16

        # Create cache list
        cache = []
        for _ in range(2):
            c = T5KVCache()
            keys = mx.random.normal((B, H, D, 5))
            values = mx.random.normal((B, H, 5, D))
            c.update_and_fetch(keys, values)
            mx.eval(c.keys, c.values)
            cache.append(c)

        # Trim 2 tokens
        model.trim_decoder_cache(cache, 2)

        assert all(c.offset == 3 for c in cache)

    def test_trim_decoder_cache_none(self, model):
        """Test trimming None cache."""
        result = model.trim_decoder_cache(None, 5)
        assert result is None

    def test_trim_decoder_cache_zero(self, model):
        """Test trimming zero tokens."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5KVCache

        B, H, D = 2, 4, 16
        cache = [T5KVCache() for _ in range(2)]
        for c in cache:
            keys = mx.random.normal((B, H, D, 5))
            values = mx.random.normal((B, H, 5, D))
            c.update_and_fetch(keys, values)
            mx.eval(c.keys, c.values)

        model.trim_decoder_cache(cache, 0)

        assert all(c.offset == 5 for c in cache)


class TestT5ModelSanitize:
    """Test T5Model weight sanitization."""

    def test_sanitize_encoder_weights(self):
        """Test encoder weight key transformation."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        weights = {
            "encoder.block.0.layer.0.SelfAttention.q.weight": mx.zeros((64, 64)),
            "encoder.block.0.layer.1.DenseReluDense.wi.weight": mx.zeros((128, 64)),
        }

        sanitized = T5Model.sanitize(weights)

        assert "encoder.layers.0.attention.query_proj.weight" in sanitized
        assert "encoder.layers.0.dense.wi.weight" in sanitized

    def test_sanitize_decoder_weights(self):
        """Test decoder weight key transformation."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        weights = {
            "decoder.block.0.layer.0.SelfAttention.q.weight": mx.zeros((64, 64)),
            "decoder.block.0.layer.1.EncDecAttention.k.weight": mx.zeros((64, 64)),
            "decoder.block.0.layer.2.DenseReluDense.wi_0.weight": mx.zeros((128, 64)),
        }

        sanitized = T5Model.sanitize(weights)

        assert "decoder.layers.0.self_attention.query_proj.weight" in sanitized
        assert "decoder.layers.0.cross_attention.key_proj.weight" in sanitized
        assert "decoder.layers.0.dense.wi_0.weight" in sanitized

    def test_sanitize_shared_embedding(self):
        """Test shared embedding weight transformation."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        weights = {
            "shared.weight": mx.zeros((1000, 64)),
        }

        sanitized = T5Model.sanitize(weights)

        assert "wte.weight" in sanitized

    def test_sanitize_madlad_embedding(self):
        """Test MADLAD-style decoder embed tokens."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        weights = {
            "decoder.embed_tokens.weight": mx.zeros((1000, 64)),
        }

        sanitized = T5Model.sanitize(weights)

        assert "wte.weight" in sanitized

    def test_sanitize_relative_attention_bias(self):
        """Test relative attention bias transformation."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        weights = {
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": mx.zeros(
                (32, 4),
            ),
        }

        sanitized = T5Model.sanitize(weights)

        assert "encoder.relative_attention_bias.embeddings.weight" in sanitized

    def test_sanitize_ignores_cross_attn_bias(self):
        """Test that cross-attention relative bias is ignored."""
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        weights = {
            "decoder.layers.0.cross_attention.relative_attention_bias.weight": mx.zeros(
                (32, 4),
            ),
            "decoder.layers.0.self_attention.query_proj.weight": mx.zeros((64, 64)),
        }

        sanitized = T5Model.sanitize(weights)

        # Cross-attention bias should be removed
        assert (
            "decoder.layers.0.cross_attention.relative_attention_bias.weight"
            not in sanitized
        )
        # Other weights preserved
        assert "decoder.layers.0.self_attention.query_proj.weight" in sanitized


class TestT5NumericalStability:
    """Test numerical stability of T5 model."""

    @pytest.fixture
    def config(self):
        return make_config()

    @pytest.fixture
    def model(self, config):
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        return T5Model(config)

    def test_deterministic_output(self, model):
        """Test that same inputs produce same outputs."""
        B = 2
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * B)
        decoder_input_ids = mx.array([[0, 1, 2, 3, 4]] * B)

        logits1 = model(input_ids, decoder_input_ids)
        mx.eval(logits1)

        logits2 = model(input_ids, decoder_input_ids)
        mx.eval(logits2)

        assert mx.allclose(logits1, logits2)

    def test_no_nan_outputs(self, model):
        """Test that outputs never contain NaN."""
        B = 2
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * B)
        decoder_input_ids = mx.array([[0, 1, 2, 3, 4]] * B)

        logits = model(input_ids, decoder_input_ids)
        mx.eval(logits)

        assert not mx.any(mx.isnan(logits))

    def test_softmax_stability(self, config):
        """Test numerical stability of softmax in attention."""
        from pytorch_to_mlx.converters.models.t5_mlx import MultiHeadAttention

        attention = MultiHeadAttention(config)

        # Large values that might cause overflow
        B, L, D = 1, 10, 64
        x = mx.random.normal((B, L, D)) * 100

        output, _ = attention(x, x, x, mask=None)
        mx.eval(output)

        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))


class TestT5EdgeCases:
    """Test edge cases for T5 model."""

    @pytest.fixture
    def config(self):
        return make_config()

    @pytest.fixture
    def model(self, config):
        from pytorch_to_mlx.converters.models.t5_mlx import T5Model

        return T5Model(config)

    def test_single_token_input(self, model):
        """Test with single token input."""
        B = 1
        input_ids = mx.array([[1]])
        decoder_input_ids = mx.array([[0]])

        logits = model(input_ids, decoder_input_ids)
        mx.eval(logits)

        assert logits.shape == (B, 1, 1000)

    def test_batch_size_one(self, model):
        """Test with batch size of 1."""
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        decoder_input_ids = mx.array([[0, 1, 2]])

        logits = model(input_ids, decoder_input_ids)
        mx.eval(logits)

        assert logits.shape == (1, 3, 1000)

    def test_large_batch_size(self, model):
        """Test with larger batch size."""
        B = 8
        input_ids = mx.array([[1, 2, 3, 4, 5]] * B)
        decoder_input_ids = mx.array([[0, 1, 2]] * B)

        logits = model(input_ids, decoder_input_ids)
        mx.eval(logits)

        assert logits.shape == (B, 3, 1000)

    def test_different_batch_elements_independent(self, model):
        """Test that different batch elements are processed independently."""
        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        decoder_input_ids = mx.array([[0, 1], [0, 2]])

        logits = model(input_ids, decoder_input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 2, 1000)
        # Logits should not be identical
        assert not mx.array_equal(logits[0], logits[1])

    def test_long_sequence(self, model):
        """Test with longer sequence."""
        B = 1
        L = 100
        input_ids = mx.arange(1, L + 1).reshape((B, L))
        decoder_input_ids = mx.arange(0, 50).reshape((B, 50))

        logits = model(input_ids, decoder_input_ids)
        mx.eval(logits)

        assert logits.shape == (B, 50, 1000)
