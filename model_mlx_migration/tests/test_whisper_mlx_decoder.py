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
Tests for whisper_mlx decoder module.

Tests:
- TextDecoder: Text decoding with timestamp precision
- TimestampLogitFilter: Timestamp constraint filtering
"""

from unittest.mock import Mock

import mlx.core as mx
import pytest


class TestTextDecoder:
    """Tests for TextDecoder class."""

    @pytest.fixture
    def decoder_config(self):
        """Standard decoder configuration."""
        return {
            "n_vocab": 51865,
            "n_ctx": 448,
            "n_state": 384,
            "n_head": 6,
            "n_layer": 4,
        }

    @pytest.fixture
    def small_decoder_config(self):
        """Smaller config for faster tests."""
        return {
            "n_vocab": 1000,
            "n_ctx": 64,
            "n_state": 64,
            "n_head": 4,
            "n_layer": 2,
        }

    def test_init(self, small_decoder_config):
        """Test decoder initialization."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        assert decoder.n_ctx == small_decoder_config["n_ctx"]
        assert decoder.n_state == small_decoder_config["n_state"]
        assert decoder.n_layer == small_decoder_config["n_layer"]

    def test_components_exist(self, small_decoder_config):
        """Test all components are created."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        assert hasattr(decoder, "token_embedding")
        assert hasattr(decoder, "positional_embedding")
        assert hasattr(decoder, "blocks")
        assert hasattr(decoder, "ln")
        assert hasattr(decoder, "_mask")

    def test_blocks_count(self, small_decoder_config):
        """Test correct number of decoder blocks."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        assert len(decoder.blocks) == small_decoder_config["n_layer"]

    def test_mask_shape(self, small_decoder_config):
        """Test causal mask has correct shape."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        assert decoder._mask.shape == (
            small_decoder_config["n_ctx"],
            small_decoder_config["n_ctx"],
        )

    def test_forward_basic(self, small_decoder_config):
        """Test basic forward pass."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)
        batch_size, seq_len = 1, 10
        enc_len = 50
        n_state = small_decoder_config["n_state"]
        n_vocab = small_decoder_config["n_vocab"]

        # Token IDs
        x = mx.array([[0] + [1] * (seq_len - 1)])  # Batch of token IDs
        xa = mx.random.normal((batch_size, enc_len, n_state))

        logits, kv_cache, cross_qk, _ = decoder(x, xa)

        assert logits.shape == (batch_size, seq_len, n_vocab)
        assert len(kv_cache) == small_decoder_config["n_layer"]
        assert len(cross_qk) == small_decoder_config["n_layer"]

    def test_forward_with_cache(self, small_decoder_config):
        """Test forward pass with KV cache."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)
        batch_size = 1
        enc_len = 50
        n_state = small_decoder_config["n_state"]
        n_vocab = small_decoder_config["n_vocab"]

        xa = mx.random.normal((batch_size, enc_len, n_state))

        # First pass - multiple tokens
        x1 = mx.array([[0, 1, 2, 3, 4]])
        logits1, kv_cache1, _, _ = decoder(x1, xa)

        assert logits1.shape == (batch_size, 5, n_vocab)

        # Second pass - single token with cache
        x2 = mx.array([[5]])
        logits2, kv_cache2, _, _ = decoder(x2, xa, kv_cache=kv_cache1)

        assert logits2.shape == (batch_size, 1, n_vocab)
        # Cache should have grown
        assert kv_cache2[0][0][0].shape[1] == 6  # 5 + 1 tokens


class TestTextDecoderPrecision:
    """Tests for timestamp precision handling."""

    @pytest.fixture
    def small_decoder_config(self):
        """Smaller config for faster tests."""
        return {
            "n_vocab": 1000,
            "n_ctx": 64,
            "n_state": 64,
            "n_head": 4,
            "n_layer": 2,
        }

    def test_default_precision(self, small_decoder_config):
        """Test default precision is 0.02s."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        expected = 30.0 / 1500  # 0.02
        assert abs(decoder.precision - expected) < 1e-10

    def test_set_precision(self, small_decoder_config):
        """Test setting custom precision."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        # 5 second audio with 250 encoder positions
        decoder.set_precision(5.0, 250)

        expected = 5.0 / 250
        assert abs(decoder.precision - expected) < 1e-10
        assert decoder.audio_duration == 5.0

    def test_reset_precision(self, small_decoder_config):
        """Test resetting precision to default."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        # Set custom precision
        decoder.set_precision(10.0, 500)
        assert decoder.audio_duration == 10.0

        # Reset
        decoder.reset_precision()

        expected = 30.0 / 1500
        assert abs(decoder.precision - expected) < 1e-10
        assert decoder.audio_duration is None

    def test_set_precision_zero_positions(self, small_decoder_config):
        """Test set_precision handles zero positions gracefully."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        decoder.set_precision(5.0, 0)

        # Should use default precision
        expected = 30.0 / 1500
        assert abs(decoder.precision - expected) < 1e-10

    def test_timestamp_to_position(self, small_decoder_config):
        """Test timestamp to position conversion."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        # Default precision: 0.02
        assert decoder.timestamp_to_position(0.0) == 0
        assert decoder.timestamp_to_position(0.02) == 1
        assert decoder.timestamp_to_position(1.0) == 50
        assert decoder.timestamp_to_position(30.0) == 1500

    def test_position_to_timestamp(self, small_decoder_config):
        """Test position to timestamp conversion."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        # Default precision: 0.02
        assert abs(decoder.position_to_timestamp(0) - 0.0) < 1e-10
        assert abs(decoder.position_to_timestamp(1) - 0.02) < 1e-10
        assert abs(decoder.position_to_timestamp(50) - 1.0) < 1e-10
        assert abs(decoder.position_to_timestamp(1500) - 30.0) < 1e-10

    def test_timestamp_conversion_roundtrip(self, small_decoder_config):
        """Test roundtrip conversion preserves values."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        for pos in [0, 1, 10, 100, 500, 1000]:
            timestamp = decoder.position_to_timestamp(pos)
            recovered_pos = decoder.timestamp_to_position(timestamp)
            assert recovered_pos == pos

    def test_precision_with_short_audio(self, small_decoder_config):
        """Test precision calculation for short audio."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**small_decoder_config)

        # 5 second audio produces ~250 encoder positions
        # (5s * 100 fps / 2 for conv stride)
        decoder.set_precision(5.0, 250)

        # Position 0 should be 0s
        assert abs(decoder.position_to_timestamp(0) - 0.0) < 1e-10

        # Position 250 should be 5s
        assert abs(decoder.position_to_timestamp(250) - 5.0) < 1e-10

        # Position 125 should be 2.5s
        assert abs(decoder.position_to_timestamp(125) - 2.5) < 1e-10


class TestTimestampLogitFilter:
    """Tests for TimestampLogitFilter class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.timestamp_begin = 50364
        tokenizer.no_timestamps = 50363
        tokenizer.eot = 50257
        return tokenizer

    def test_init(self, mock_tokenizer):
        """Test filter initialization."""
        from tools.whisper_mlx.decoder import TimestampLogitFilter

        filter = TimestampLogitFilter(
            tokenizer=mock_tokenizer,
            sample_begin=4,
            max_initial_timestamp=1.0,
            precision=0.02,
        )

        assert filter.tokenizer is mock_tokenizer
        assert filter.sample_begin == 4
        assert filter.max_initial_timestamp == 1.0
        assert filter.precision == 0.02
        assert filter._max_initial_timestamp_index == 50  # 1.0 / 0.02

    def test_init_no_max_timestamp(self, mock_tokenizer):
        """Test init without max initial timestamp."""
        from tools.whisper_mlx.decoder import TimestampLogitFilter

        filter = TimestampLogitFilter(
            tokenizer=mock_tokenizer,
            sample_begin=4,
            max_initial_timestamp=None,
        )

        assert filter._max_initial_timestamp_index is None

    def test_set_precision(self, mock_tokenizer):
        """Test updating precision."""
        from tools.whisper_mlx.decoder import TimestampLogitFilter

        filter = TimestampLogitFilter(
            tokenizer=mock_tokenizer,
            sample_begin=4,
            max_initial_timestamp=1.0,
            precision=0.02,
        )

        # Update precision (e.g., for shorter audio)
        filter.set_precision(0.01, max_initial_timestamp=0.5)

        assert filter.precision == 0.01
        assert filter.max_initial_timestamp == 0.5
        assert filter._max_initial_timestamp_index == 50  # 0.5 / 0.01

    def test_set_precision_keeps_max_timestamp(self, mock_tokenizer):
        """Test set_precision keeps existing max_initial_timestamp if not provided."""
        from tools.whisper_mlx.decoder import TimestampLogitFilter

        filter = TimestampLogitFilter(
            tokenizer=mock_tokenizer,
            sample_begin=4,
            max_initial_timestamp=1.0,
            precision=0.02,
        )

        # Update only precision
        filter.set_precision(0.04)

        assert filter.precision == 0.04
        assert filter.max_initial_timestamp == 1.0
        assert filter._max_initial_timestamp_index == 25  # 1.0 / 0.04


class TestTimestampLogitFilterApply:
    """Tests for TimestampLogitFilter.apply method."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.timestamp_begin = 100
        tokenizer.no_timestamps = 99
        tokenizer.eot = 50
        return tokenizer

    def test_suppresses_no_timestamps_token(self, mock_tokenizer):
        """Test no_timestamps token is always suppressed."""
        from tools.whisper_mlx.decoder import TimestampLogitFilter

        filter = TimestampLogitFilter(
            tokenizer=mock_tokenizer,
            sample_begin=4,
            max_initial_timestamp=None,
        )

        vocab_size = 200
        logits = mx.zeros((1, vocab_size))
        tokens = mx.array([[0, 1, 2, 3, 10, 20]])  # After sample_begin

        result = filter.apply(logits, tokens)

        # no_timestamps should be -inf
        assert result[0, mock_tokenizer.no_timestamps].item() == float("-inf")

    def test_at_start_suppresses_text_tokens(self, mock_tokenizer):
        """Test at start, text tokens are suppressed."""
        from tools.whisper_mlx.decoder import TimestampLogitFilter

        filter = TimestampLogitFilter(
            tokenizer=mock_tokenizer,
            sample_begin=4,
            max_initial_timestamp=1.0,
            precision=0.02,
        )

        vocab_size = 200
        logits = mx.zeros((1, vocab_size))
        tokens = mx.array([[0, 1, 2, 3]])  # Exactly at sample_begin

        result = filter.apply(logits, tokens)

        # Text tokens (< timestamp_begin) should be suppressed
        for i in range(mock_tokenizer.timestamp_begin):
            assert result[0, i].item() == float("-inf") or result[0, i].item() < -1e10


class TestDecoderEdgeCases:
    """Edge case tests for decoder."""

    @pytest.fixture
    def tiny_decoder_config(self):
        """Minimal config for edge case tests."""
        return {
            "n_vocab": 500,
            "n_ctx": 32,
            "n_state": 32,
            "n_head": 2,
            "n_layer": 1,
        }

    def test_single_token_forward(self, tiny_decoder_config):
        """Test forward with single token."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config)
        n_state = tiny_decoder_config["n_state"]
        n_vocab = tiny_decoder_config["n_vocab"]

        x = mx.array([[0]])
        xa = mx.random.normal((1, 20, n_state))

        logits, kv_cache, cross_qk, _ = decoder(x, xa)

        assert logits.shape == (1, 1, n_vocab)

    def test_max_length_tokens(self, tiny_decoder_config):
        """Test forward with max context length."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config)
        n_ctx = tiny_decoder_config["n_ctx"]
        n_state = tiny_decoder_config["n_state"]
        n_vocab = tiny_decoder_config["n_vocab"]

        x = mx.array([list(range(n_ctx))])
        xa = mx.random.normal((1, 20, n_state))

        logits, kv_cache, cross_qk, _ = decoder(x, xa)

        assert logits.shape == (1, n_ctx, n_vocab)

    def test_logits_are_finite(self, tiny_decoder_config):
        """Test output logits are finite."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config)
        n_state = tiny_decoder_config["n_state"]

        x = mx.array([[0, 1, 2, 3]])
        xa = mx.random.normal((1, 20, n_state))

        logits, _, _, _ = decoder(x, xa)

        assert mx.all(mx.isfinite(logits)).item()

    def test_batch_processing(self, tiny_decoder_config):
        """Test batch processing works correctly."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config)
        batch_size = 4
        seq_len = 8
        n_state = tiny_decoder_config["n_state"]
        n_vocab = tiny_decoder_config["n_vocab"]

        x = mx.array([[0] * seq_len] * batch_size)
        xa = mx.random.normal((batch_size, 20, n_state))

        logits, kv_cache, cross_qk, _ = decoder(x, xa)

        assert logits.shape == (batch_size, seq_len, n_vocab)

    def test_cross_qk_shapes(self, tiny_decoder_config):
        """Test cross-attention weights have correct shapes."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config)
        batch_size = 2
        dec_len = 6
        enc_len = 30
        n_state = tiny_decoder_config["n_state"]
        n_head = tiny_decoder_config["n_head"]
        n_layer = tiny_decoder_config["n_layer"]

        x = mx.array([[0] * dec_len] * batch_size)
        xa = mx.random.normal((batch_size, enc_len, n_state))

        out, _, cross_qk, _ = decoder(x, xa)

        # Output should always be valid
        assert out.shape == (batch_size, dec_len, tiny_decoder_config["n_vocab"])

        assert len(cross_qk) == n_layer
        for layer_qk in cross_qk:
            # Note: cross_qk entries may be None when using SDPA (OPT-NEW-2)
            if layer_qk is not None:
                # Shape: (batch, n_head, dec_len, enc_len)
                assert layer_qk.shape == (batch_size, n_head, dec_len, enc_len)


class TestDecoderCacheGrowth:
    """Tests for KV cache growth during decoding."""

    @pytest.fixture
    def tiny_decoder_config(self):
        """Minimal config."""
        return {
            "n_vocab": 500,
            "n_ctx": 32,
            "n_state": 32,
            "n_head": 2,
            "n_layer": 2,
        }

    def test_cache_grows_incrementally(self, tiny_decoder_config):
        """Test cache grows by 1 each step."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config)
        n_state = tiny_decoder_config["n_state"]

        xa = mx.random.normal((1, 20, n_state))

        # Start with 1 token
        x = mx.array([[0]])
        _, cache, _, _ = decoder(x, xa)

        for step in range(1, 10):
            x_next = mx.array([[step]])
            _, cache, _, _ = decoder(x_next, xa, kv_cache=cache)

            # Self-attention cache should have step+1 positions
            self_k_len = cache[0][0][0].shape[1]
            assert self_k_len == step + 1

    def test_cross_attention_cache_fixed(self, tiny_decoder_config):
        """Test cross-attention cache doesn't grow."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config)
        n_state = tiny_decoder_config["n_state"]
        enc_len = 25

        xa = mx.random.normal((1, enc_len, n_state))

        # First token
        x = mx.array([[0]])
        _, cache, _, _ = decoder(x, xa)

        initial_cross_len = cache[0][1][0].shape[1]

        # More tokens
        for step in range(1, 10):
            x_next = mx.array([[step]])
            _, cache, _, _ = decoder(x_next, xa, kv_cache=cache)

            # Cross-attention cache should stay fixed
            cross_len = cache[0][1][0].shape[1]
            assert cross_len == initial_cross_len


class TestDecoderDtype:
    """Tests for dtype handling."""

    @pytest.fixture
    def tiny_decoder_config(self):
        """Minimal config."""
        return {
            "n_vocab": 500,
            "n_ctx": 32,
            "n_state": 32,
            "n_head": 2,
            "n_layer": 1,
        }

    def test_float16_mask(self, tiny_decoder_config):
        """Test causal mask is float16 when specified."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config, dtype=mx.float16)

        assert decoder._mask.dtype == mx.float16

    def test_float32_mask(self, tiny_decoder_config):
        """Test causal mask is float32 when specified."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(**tiny_decoder_config, dtype=mx.float32)

        assert decoder._mask.dtype == mx.float32
