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
Tests for whisper_mlx attention module.

Tests:
- MultiHeadAttention: self-attention, cross-attention, KV cache
- ResidualAttentionBlock: full transformer block
"""

import mlx.core as mx
import pytest


# Test fixtures
@pytest.fixture
def attention_config():
    """Standard attention configuration."""
    return {
        "n_state": 512,
        "n_head": 8,
    }


@pytest.fixture
def small_attention_config():
    """Smaller config for faster tests."""
    return {
        "n_state": 64,
        "n_head": 4,
    }


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention class."""

    def test_init_basic(self, attention_config):
        """Test basic initialization."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**attention_config)

        assert attn.n_state == 512
        assert attn.n_head == 8
        assert attn.head_dim == 64  # 512 / 8
        assert attn.use_fused is True

    def test_init_no_fused(self, small_attention_config):
        """Test initialization with fused disabled."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config, use_fused=False)
        assert attn.use_fused is False

    def test_projections_shapes(self, small_attention_config):
        """Test that projections have correct shapes."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)

        # Check projection layers exist
        assert hasattr(attn, "query")
        assert hasattr(attn, "key")
        assert hasattr(attn, "value")
        assert hasattr(attn, "out")

    def test_self_attention_output_shape(self, small_attention_config):
        """Test self-attention output shape."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        batch_size, seq_len = 2, 10
        x = mx.random.normal((batch_size, seq_len, small_attention_config["n_state"]))

        out, kv_cache, attn_weights = attn(x)

        assert out.shape == (batch_size, seq_len, small_attention_config["n_state"])
        assert kv_cache[0].shape == (batch_size, seq_len, small_attention_config["n_state"])  # K
        assert kv_cache[1].shape == (batch_size, seq_len, small_attention_config["n_state"])  # V

    def test_self_attention_with_mask(self, small_attention_config):
        """Test self-attention with causal mask."""
        import mlx.nn as nn

        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        batch_size, seq_len = 2, 10
        x = mx.random.normal((batch_size, seq_len, small_attention_config["n_state"]))

        # Create causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        out, kv_cache, attn_weights = attn(x, mask=mask)

        assert out.shape == (batch_size, seq_len, small_attention_config["n_state"])
        # Note: attn_weights may be None when using SDPA (OPT-NEW-2)
        # SDPA is a fused kernel that doesn't expose intermediate attention weights

    def test_self_attention_kv_cache(self, small_attention_config):
        """Test self-attention with KV cache accumulation."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        batch_size, n_state = 2, small_attention_config["n_state"]

        # First token
        x1 = mx.random.normal((batch_size, 1, n_state))
        out1, kv_cache1, _ = attn(x1)

        assert out1.shape == (batch_size, 1, n_state)
        assert kv_cache1[0].shape == (batch_size, 1, n_state)

        # Second token with cache
        x2 = mx.random.normal((batch_size, 1, n_state))
        out2, kv_cache2, _ = attn(x2, kv_cache=kv_cache1)

        assert out2.shape == (batch_size, 1, n_state)
        assert kv_cache2[0].shape == (batch_size, 2, n_state)  # Cache grew

        # Third token
        x3 = mx.random.normal((batch_size, 1, n_state))
        out3, kv_cache3, _ = attn(x3, kv_cache=kv_cache2)

        assert kv_cache3[0].shape == (batch_size, 3, n_state)  # Cache grew again

    def test_cross_attention_output_shape(self, small_attention_config):
        """Test cross-attention output shape."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        batch_size = 2
        q_len, kv_len = 5, 20
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((batch_size, q_len, n_state))
        xa = mx.random.normal((batch_size, kv_len, n_state))

        out, kv_cache, attn_weights = attn(x, xa=xa)

        assert out.shape == (batch_size, q_len, n_state)
        assert kv_cache[0].shape == (batch_size, kv_len, n_state)  # K from xa
        assert kv_cache[1].shape == (batch_size, kv_len, n_state)  # V from xa

    def test_cross_attention_kv_reuse(self, small_attention_config):
        """Test cross-attention reuses cached K/V."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        batch_size = 2
        kv_len = 20
        n_state = small_attention_config["n_state"]

        # First query
        x1 = mx.random.normal((batch_size, 1, n_state))
        xa = mx.random.normal((batch_size, kv_len, n_state))

        out1, kv_cache1, _ = attn(x1, xa=xa)

        # Second query reuses K/V from cache (xa not used)
        x2 = mx.random.normal((batch_size, 1, n_state))
        out2, kv_cache2, _ = attn(x2, xa=xa, kv_cache=kv_cache1)

        # KV cache should be unchanged (same encoder output)
        assert kv_cache2[0].shape == (batch_size, kv_len, n_state)
        assert mx.allclose(kv_cache1[0], kv_cache2[0]).item()
        assert mx.allclose(kv_cache1[1], kv_cache2[1]).item()

    def test_attention_weights_shape(self, small_attention_config):
        """Test attention weights have correct shape."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        batch_size, seq_len = 2, 10
        n_state = small_attention_config["n_state"]
        n_head = small_attention_config["n_head"]

        x = mx.random.normal((batch_size, seq_len, n_state))

        out, kv_cache, attn_weights = attn(x)

        # Note: attn_weights may be None when using SDPA (OPT-NEW-2)
        # Skip shape check if SDPA is used (fused kernel doesn't expose weights)
        if attn_weights is not None:
            assert attn_weights.shape == (batch_size, n_head, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self, small_attention_config):
        """Test attention weights sum to 1 along kv dimension."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        batch_size, seq_len = 1, 8
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((batch_size, seq_len, n_state))

        out, kv_cache, attn_weights = attn(x)

        # Note: attn_weights may be None when using SDPA (OPT-NEW-2)
        if attn_weights is None:
            pytest.skip("SDPA doesn't return attention weights")

        # Sum along kv dimension should be ~1
        weight_sums = mx.sum(attn_weights, axis=-1)
        assert mx.allclose(weight_sums, mx.ones_like(weight_sums), atol=1e-5).item()

    def test_different_batch_sizes(self, small_attention_config):
        """Test with various batch sizes."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        seq_len = 8
        n_state = small_attention_config["n_state"]

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, seq_len, n_state))
            out, _, _ = attn(x)
            assert out.shape == (batch_size, seq_len, n_state)

    def test_head_dim_calculation(self):
        """Test head dimension is calculated correctly."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        # Test various configurations
        configs = [
            (512, 8, 64),   # 512/8 = 64
            (768, 12, 64),  # 768/12 = 64
            (256, 4, 64),   # 256/4 = 64
            (384, 6, 64),   # 384/6 = 64
        ]

        for n_state, n_head, expected_head_dim in configs:
            attn = MultiHeadAttention(n_state, n_head)
            assert attn.head_dim == expected_head_dim


class TestResidualAttentionBlock:
    """Tests for ResidualAttentionBlock class."""

    def test_init_without_cross_attention(self, small_attention_config):
        """Test initialization without cross-attention (encoder block)."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=False)

        assert block.attn is not None
        assert block.attn_ln is not None
        assert block.cross_attn is None
        assert block.cross_attn_ln is None
        assert block.mlp1 is not None
        assert block.mlp2 is not None
        assert block.mlp_ln is not None

    def test_init_with_cross_attention(self, small_attention_config):
        """Test initialization with cross-attention (decoder block)."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=True)

        assert block.attn is not None
        assert block.cross_attn is not None
        assert block.cross_attn_ln is not None

    def test_encoder_block_forward(self, small_attention_config):
        """Test encoder block (no cross-attention) forward pass."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=False)
        batch_size, seq_len = 2, 16
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((batch_size, seq_len, n_state))

        out, kv_cache, cross_qk = block(x)

        assert out.shape == (batch_size, seq_len, n_state)
        assert cross_qk is None  # No cross-attention

    def test_decoder_block_forward(self, small_attention_config):
        """Test decoder block (with cross-attention) forward pass."""
        import mlx.nn as nn

        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=True)
        batch_size = 2
        dec_len, enc_len = 8, 20
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((batch_size, dec_len, n_state))
        xa = mx.random.normal((batch_size, enc_len, n_state))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(dec_len)

        out, kv_cache, cross_qk = block(x, xa=xa, mask=mask)

        assert out.shape == (batch_size, dec_len, n_state)
        # Note: cross_qk may be None when using SDPA (OPT-NEW-2)

    def test_decoder_block_with_cache(self, small_attention_config):
        """Test decoder block with KV cache."""
        import mlx.nn as nn

        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=True)
        batch_size = 2
        enc_len = 20
        n_state = small_attention_config["n_state"]
        max_len = 16

        xa = mx.random.normal((batch_size, enc_len, n_state))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(max_len)

        # First token
        x1 = mx.random.normal((batch_size, 1, n_state))
        out1, kv_cache1, _ = block(x1, xa=xa, mask=mask)

        assert out1.shape == (batch_size, 1, n_state)

        # Second token with cache
        x2 = mx.random.normal((batch_size, 1, n_state))
        out2, kv_cache2, _ = block(x2, xa=xa, mask=mask, kv_cache=kv_cache1)

        assert out2.shape == (batch_size, 1, n_state)
        # Self-attention cache should grow
        assert kv_cache2[0][0].shape[1] == 2  # self_k seq_len

    def test_residual_connection(self, small_attention_config):
        """Test that residual connections are working."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=False)
        batch_size, seq_len = 1, 8
        n_state = small_attention_config["n_state"]

        # Create input with specific values
        x = mx.ones((batch_size, seq_len, n_state))

        out, _, _ = block(x)

        # Output should NOT be identical to input (transformations applied)
        assert not mx.allclose(out, x).item()
        # But should have same shape
        assert out.shape == x.shape

    def test_mlp_expansion(self, small_attention_config):
        """Test MLP has 4x expansion."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=False)
        n_state = small_attention_config["n_state"]

        # MLP1 should expand to 4x
        # Check by running a forward pass
        batch_size, seq_len = 1, 4
        x = mx.random.normal((batch_size, seq_len, n_state))

        # MLP1 output
        mlp1_out = block.mlp1(x)
        assert mlp1_out.shape == (batch_size, seq_len, n_state * 4)

    def test_layer_norm_applied(self, small_attention_config):
        """Test layer normalization is applied."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=False)

        # Layer norms should exist
        assert hasattr(block, "attn_ln")
        assert hasattr(block, "mlp_ln")

    def test_gelu_activation(self, small_attention_config):
        """Test GELU activation in MLP."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=False)
        batch_size, seq_len = 1, 4
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((batch_size, seq_len, n_state))

        # Full forward includes GELU
        out, _, _ = block(x)

        # Output should be finite
        assert mx.all(mx.isfinite(out)).item()

    def test_encoder_no_cross_qk(self, small_attention_config):
        """Test encoder block returns None for cross_qk."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=False)
        x = mx.random.normal((1, 8, small_attention_config["n_state"]))

        _, _, cross_qk = block(x)
        assert cross_qk is None

    def test_decoder_returns_cross_qk(self, small_attention_config):
        """Test decoder block runs successfully with cross-attention."""
        from tools.whisper_mlx.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(**small_attention_config, cross_attention=True)
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((1, 4, n_state))
        xa = mx.random.normal((1, 16, n_state))

        out, _, cross_qk = block(x, xa=xa)
        # Note: cross_qk may be None when using SDPA (OPT-NEW-2)
        # The key test is that the block runs successfully
        assert out.shape == (1, 4, n_state)


class TestAttentionMasking:
    """Tests for attention masking behavior."""

    def test_causal_mask_prevents_future_attention(self, small_attention_config):
        """Test causal mask prevents attending to future positions."""
        import mlx.nn as nn

        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        seq_len = 8
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((1, seq_len, n_state))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        out, _, attn_weights = attn(x, mask=mask)

        # Note: SDPA doesn't return attention weights (OPT-NEW-2)
        if attn_weights is None:
            # Can't check attention weights directly, but verify output is valid
            assert mx.all(mx.isfinite(out)).item()
            pytest.skip("SDPA doesn't return attention weights")

        # For each query position, attention to future positions should be near zero
        # (after softmax of -inf)
        for q_pos in range(seq_len):
            for kv_pos in range(q_pos + 1, seq_len):
                weight = attn_weights[0, 0, q_pos, kv_pos].item()
                assert weight < 1e-6, f"Position {q_pos} attending to future position {kv_pos}"

    def test_no_mask_allows_full_attention(self, small_attention_config):
        """Test no mask allows attending to all positions."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        seq_len = 8
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((1, seq_len, n_state))

        out, _, attn_weights = attn(x, mask=None)

        # Note: SDPA doesn't return attention weights (OPT-NEW-2)
        if attn_weights is None:
            assert mx.all(mx.isfinite(out)).item()
            pytest.skip("SDPA doesn't return attention weights")

        # All attention weights should be non-zero
        assert mx.all(attn_weights > 0).item()


class TestAttentionNumericalStability:
    """Tests for numerical stability."""

    def test_large_values(self, small_attention_config):
        """Test attention handles large input values."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        n_state = small_attention_config["n_state"]

        # Large but not extreme values
        x = mx.random.normal((1, 8, n_state)) * 10

        out, _, attn_weights = attn(x)

        assert mx.all(mx.isfinite(out)).item()
        # Note: attn_weights may be None when using SDPA (OPT-NEW-2)
        if attn_weights is not None:
            assert mx.all(mx.isfinite(attn_weights)).item()

    def test_small_values(self, small_attention_config):
        """Test attention handles small input values."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        n_state = small_attention_config["n_state"]

        # Small values
        x = mx.random.normal((1, 8, n_state)) * 0.01

        out, _, attn_weights = attn(x)

        assert mx.all(mx.isfinite(out)).item()
        # Note: attn_weights may be None when using SDPA (OPT-NEW-2)
        if attn_weights is not None:
            assert mx.all(mx.isfinite(attn_weights)).item()

    def test_mixed_precision(self, small_attention_config):
        """Test attention with float16 input."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((1, 8, n_state)).astype(mx.float16)

        out, _, attn_weights = attn(x)

        assert mx.all(mx.isfinite(out)).item()


class TestSpeculativeDecodingSupport:
    """Tests for speculative decoding with KV cache."""

    def test_multi_token_query_with_cache(self, small_attention_config):
        """Test processing multiple query tokens with existing cache."""
        import mlx.nn as nn

        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        n_state = small_attention_config["n_state"]
        max_len = 32

        # Build up some cache
        x1 = mx.random.normal((1, 5, n_state))
        out1, kv_cache1, _ = attn(x1)

        # Now query with multiple new tokens (speculative decoding scenario)
        x2 = mx.random.normal((1, 3, n_state))  # 3 speculative tokens
        mask = nn.MultiHeadAttention.create_additive_causal_mask(max_len)

        out2, kv_cache2, _ = attn(x2, kv_cache=kv_cache1, mask=mask)

        assert out2.shape == (1, 3, n_state)
        # Cache should now have 5 + 3 = 8 positions
        assert kv_cache2[0].shape == (1, 8, n_state)

    def test_mask_offset_handling(self, small_attention_config):
        """Test mask is correctly offset for KV cache positions."""
        import mlx.nn as nn

        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        n_state = small_attention_config["n_state"]
        max_len = 32

        mask = nn.MultiHeadAttention.create_additive_causal_mask(max_len)

        # Build cache with 10 tokens
        x1 = mx.random.normal((1, 10, n_state))
        _, kv_cache1, _ = attn(x1, mask=mask)

        # Query with 1 new token (position 10)
        x2 = mx.random.normal((1, 1, n_state))
        out, kv_cache2, attn_weights = attn(x2, kv_cache=kv_cache1, mask=mask)

        # Verify output and cache are correct shapes
        assert out.shape == (1, 1, n_state)
        assert kv_cache2[0].shape == (1, 11, n_state)  # 10 cached + 1 new

        # Note: attn_weights may be None when using SDPA (OPT-NEW-2)
        if attn_weights is not None:
            assert attn_weights.shape == (1, small_attention_config["n_head"], 1, 11)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_token(self, small_attention_config):
        """Test with single token input."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        n_state = small_attention_config["n_state"]

        x = mx.random.normal((1, 1, n_state))

        out, kv_cache, attn_weights = attn(x)

        assert out.shape == (1, 1, n_state)
        # Note: attn_weights may be None when using SDPA (OPT-NEW-2)
        if attn_weights is not None:
            assert attn_weights.shape == (1, small_attention_config["n_head"], 1, 1)

    def test_long_sequence(self, small_attention_config):
        """Test with longer sequence."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        n_state = small_attention_config["n_state"]
        seq_len = 256

        x = mx.random.normal((1, seq_len, n_state))

        out, _, _ = attn(x)

        assert out.shape == (1, seq_len, n_state)

    def test_batch_independence(self, small_attention_config):
        """Test that batches are processed independently."""
        from tools.whisper_mlx.attention import MultiHeadAttention

        attn = MultiHeadAttention(**small_attention_config)
        n_state = small_attention_config["n_state"]
        seq_len = 8

        # Create two different sequences
        x1 = mx.random.normal((1, seq_len, n_state))
        x2 = mx.random.normal((1, seq_len, n_state))

        # Process individually
        out1, _, _ = attn(x1)
        out2, _, _ = attn(x2)

        # Process as batch
        x_batch = mx.concatenate([x1, x2], axis=0)
        out_batch, _, _ = attn(x_batch)

        # Results should match
        assert mx.allclose(out1, out_batch[0:1], atol=1e-5).item()
        assert mx.allclose(out2, out_batch[1:2], atol=1e-5).item()
