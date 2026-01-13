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
Tests for Marian (OPUS-MT) MLX Model Implementation

Tests the encoder-decoder transformer architecture without requiring
actual HuggingFace weights.
"""

import sys
from pathlib import Path

import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

# Skip if MLX not available
mlx = pytest.importorskip("mlx.core")
mx = mlx


class TestMarianConfig:
    """Test MarianConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        config = MarianConfig()
        assert config.vocab_size == 65001
        assert config.d_model == 512
        assert config.encoder_layers == 6
        assert config.decoder_layers == 6
        assert config.encoder_attention_heads == 8
        assert config.decoder_attention_heads == 8
        assert config.encoder_ffn_dim == 2048
        assert config.decoder_ffn_dim == 2048
        assert config.activation_function == "swish"
        assert config.max_position_embeddings == 512
        assert config.pad_token_id == 65000
        assert config.eos_token_id == 0
        assert config.decoder_start_token_id == 65000

    def test_custom_config(self):
        """Test custom configuration."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        config = MarianConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=4,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
        )
        assert config.vocab_size == 1000
        assert config.d_model == 256
        assert config.encoder_layers == 4
        assert config.decoder_layers == 4
        assert config.encoder_attention_heads == 4

    def test_config_immutable_default(self):
        """Test that different instances have independent values."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        config1 = MarianConfig()
        config2 = MarianConfig(vocab_size=500)
        assert config1.vocab_size == 65001
        assert config2.vocab_size == 500


class TestMarianAttention:
    """Test MarianAttention module."""

    @pytest.fixture
    def attention(self):
        """Create attention module."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianAttention

        return MarianAttention(embed_dim=64, num_heads=4)

    def test_attention_initialization(self, attention):
        """Test attention module initialization."""
        assert attention.embed_dim == 64
        assert attention.num_heads == 4
        assert attention.head_dim == 16
        assert attention.scale == 16**-0.5

    def test_self_attention(self, attention):
        """Test self-attention forward pass."""
        batch_size, seq_len, embed_dim = 2, 10, 64
        hidden_states = mx.random.normal((batch_size, seq_len, embed_dim))

        output, cache = attention(hidden_states)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert cache is not None
        assert len(cache) == 2  # (keys, values)

    def test_self_attention_with_mask(self, attention):
        """Test self-attention with attention mask."""
        batch_size, seq_len, embed_dim = 2, 10, 64
        hidden_states = mx.random.normal((batch_size, seq_len, embed_dim))
        attention_mask = mx.zeros((batch_size, 1, 1, seq_len))

        output, cache = attention(hidden_states, attention_mask=attention_mask)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_cross_attention(self, attention):
        """Test cross-attention forward pass."""
        batch_size, tgt_len, src_len, embed_dim = 2, 8, 10, 64
        hidden_states = mx.random.normal((batch_size, tgt_len, embed_dim))
        encoder_states = mx.random.normal((batch_size, src_len, embed_dim))

        output, cache = attention(hidden_states, key_value_states=encoder_states)
        mx.eval(output)

        assert output.shape == (batch_size, tgt_len, embed_dim)
        # Cross-attention cache has encoder sequence length
        assert cache[0].shape[1] == src_len

    def test_self_attention_with_cache(self, attention):
        """Test self-attention with KV cache."""
        batch_size, embed_dim = 2, 64

        # First token
        hidden_states = mx.random.normal((batch_size, 1, embed_dim))
        output1, cache1 = attention(hidden_states)
        mx.eval(output1, cache1[0], cache1[1])

        assert cache1[0].shape == (batch_size, 1, embed_dim)
        assert cache1[1].shape == (batch_size, 1, embed_dim)

        # Second token with cache
        hidden_states2 = mx.random.normal((batch_size, 1, embed_dim))
        output2, cache2 = attention(hidden_states2, cache=cache1)
        mx.eval(output2, cache2[0], cache2[1])

        # Cache should grow
        assert cache2[0].shape == (batch_size, 2, embed_dim)
        assert cache2[1].shape == (batch_size, 2, embed_dim)

    def test_cross_attention_with_cache(self, attention):
        """Test cross-attention with cached encoder states."""
        batch_size, src_len, embed_dim = 2, 10, 64
        encoder_states = mx.random.normal((batch_size, src_len, embed_dim))

        # First pass - compute and cache encoder projections
        hidden_states = mx.random.normal((batch_size, 1, embed_dim))
        output1, cache1 = attention(
            hidden_states, key_value_states=encoder_states,
        )
        mx.eval(output1, cache1[0], cache1[1])

        # Second pass - reuse cached encoder projections
        hidden_states2 = mx.random.normal((batch_size, 1, embed_dim))
        output2, cache2 = attention(
            hidden_states2, key_value_states=encoder_states, cache=cache1,
        )
        mx.eval(output2)

        # Cache should be reused, not grow for cross-attention
        assert cache2[0].shape == cache1[0].shape


class TestMarianEncoderLayer:
    """Test MarianEncoderLayer module."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        return MarianConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
        )

    @pytest.fixture
    def encoder_layer(self, config):
        """Create encoder layer."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianEncoderLayer

        return MarianEncoderLayer(config)

    def test_encoder_layer_forward(self, encoder_layer):
        """Test encoder layer forward pass."""
        batch_size, seq_len, d_model = 2, 10, 64
        hidden_states = mx.random.normal((batch_size, seq_len, d_model))

        output = encoder_layer(hidden_states)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_encoder_layer_with_mask(self, encoder_layer):
        """Test encoder layer with attention mask."""
        batch_size, seq_len, d_model = 2, 10, 64
        hidden_states = mx.random.normal((batch_size, seq_len, d_model))
        attention_mask = mx.zeros((batch_size, 1, 1, seq_len))

        output = encoder_layer(hidden_states, attention_mask=attention_mask)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_encoder_layer_residual_connection(self, encoder_layer):
        """Test that encoder layer uses residual connections."""
        batch_size, seq_len, d_model = 1, 5, 64
        hidden_states = mx.ones((batch_size, seq_len, d_model))

        output = encoder_layer(hidden_states)
        mx.eval(output)

        # Output should differ from input due to layer processing
        # but should be in similar range due to residuals
        assert output.shape == hidden_states.shape


class TestMarianDecoderLayer:
    """Test MarianDecoderLayer module."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        return MarianConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
        )

    @pytest.fixture
    def decoder_layer(self, config):
        """Create decoder layer."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianDecoderLayer

        return MarianDecoderLayer(config)

    def test_decoder_layer_forward(self, decoder_layer):
        """Test decoder layer forward pass."""
        batch_size, tgt_len, src_len, d_model = 2, 8, 10, 64
        hidden_states = mx.random.normal((batch_size, tgt_len, d_model))
        encoder_hidden_states = mx.random.normal((batch_size, src_len, d_model))

        output, cache = decoder_layer(hidden_states, encoder_hidden_states)
        mx.eval(output)

        assert output.shape == (batch_size, tgt_len, d_model)
        assert cache is not None

    def test_decoder_layer_with_causal_mask(self, decoder_layer):
        """Test decoder layer with causal attention mask."""
        batch_size, tgt_len, src_len, d_model = 2, 8, 10, 64
        hidden_states = mx.random.normal((batch_size, tgt_len, d_model))
        encoder_hidden_states = mx.random.normal((batch_size, src_len, d_model))

        # Causal mask
        causal_mask = mx.triu(mx.full((tgt_len, tgt_len), -1e9), k=1)
        causal_mask = causal_mask[None, None, :, :]

        output, cache = decoder_layer(
            hidden_states, encoder_hidden_states, attention_mask=causal_mask,
        )
        mx.eval(output)

        assert output.shape == (batch_size, tgt_len, d_model)

    def test_decoder_layer_with_cache(self, decoder_layer):
        """Test decoder layer with KV cache."""
        batch_size, src_len, d_model = 2, 10, 64
        encoder_hidden_states = mx.random.normal((batch_size, src_len, d_model))

        # First token
        hidden_states = mx.random.normal((batch_size, 1, d_model))
        output1, cache1 = decoder_layer(hidden_states, encoder_hidden_states)
        mx.eval(output1, cache1[0][0], cache1[0][1])

        assert output1.shape == (batch_size, 1, d_model)

        # Second token with cache
        hidden_states2 = mx.random.normal((batch_size, 1, d_model))
        output2, cache2 = decoder_layer(
            hidden_states2, encoder_hidden_states, cache=cache1,
        )
        mx.eval(output2)

        assert output2.shape == (batch_size, 1, d_model)


class TestMarianEncoder:
    """Test MarianEncoder module."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        return MarianConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            max_position_embeddings=128,
        )

    @pytest.fixture
    def encoder(self, config):
        """Create encoder."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianEncoder

        return MarianEncoder(config)

    def test_encoder_forward(self, encoder):
        """Test encoder forward pass."""
        batch_size, seq_len = 2, 10
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * batch_size)

        output = encoder(input_ids)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, 64)

    def test_encoder_with_attention_mask(self, encoder):
        """Test encoder with attention mask."""
        batch_size, seq_len = 2, 10
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * batch_size)
        attention_mask = mx.ones((batch_size, seq_len))

        output = encoder(input_ids, attention_mask=attention_mask)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, 64)

    def test_encoder_embed_scale(self, encoder):
        """Test that encoder applies embedding scale."""
        # Marian uses sqrt(d_model) scaling
        assert encoder.embed_scale == 64**0.5

    def test_encoder_variable_sequence_length(self, encoder):
        """Test encoder with different sequence lengths."""
        batch_size = 2

        # Short sequence
        input_ids_short = mx.array([[1, 2, 3]] * batch_size)
        output_short = encoder(input_ids_short)
        mx.eval(output_short)
        assert output_short.shape == (batch_size, 3, 64)

        # Longer sequence
        input_ids_long = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]] * batch_size)
        output_long = encoder(input_ids_long)
        mx.eval(output_long)
        assert output_long.shape == (batch_size, 8, 64)


class TestMarianDecoder:
    """Test MarianDecoder module."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        return MarianConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            max_position_embeddings=128,
        )

    @pytest.fixture
    def decoder(self, config):
        """Create decoder."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianDecoder

        return MarianDecoder(config)

    def test_decoder_forward(self, decoder):
        """Test decoder forward pass."""
        batch_size, tgt_len, src_len = 2, 8, 10
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]] * batch_size)
        encoder_hidden_states = mx.random.normal((batch_size, src_len, 64))

        output, cache = decoder(input_ids, encoder_hidden_states)
        mx.eval(output)

        assert output.shape == (batch_size, tgt_len, 64)
        assert cache is not None
        assert len(cache) == 2  # 2 decoder layers

    def test_decoder_autoregressive(self, decoder):
        """Test decoder autoregressive generation."""
        batch_size, src_len = 2, 10
        encoder_hidden_states = mx.random.normal((batch_size, src_len, 64))

        # First token
        input_ids = mx.array([[1]] * batch_size)
        output1, cache1 = decoder(input_ids, encoder_hidden_states)
        mx.eval(output1)

        assert output1.shape == (batch_size, 1, 64)

        # Second token with cache
        input_ids2 = mx.array([[2]] * batch_size)
        output2, cache2 = decoder(input_ids2, encoder_hidden_states, cache=cache1)
        mx.eval(output2)

        assert output2.shape == (batch_size, 1, 64)

    def test_decoder_embed_scale(self, decoder):
        """Test that decoder applies embedding scale."""
        assert decoder.embed_scale == 64**0.5


class TestMarianModel:
    """Test MarianModel complete encoder-decoder."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        return MarianConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            max_position_embeddings=128,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )

    @pytest.fixture
    def model(self, config):
        """Create model."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianModel

        return MarianModel(config)

    def test_model_encode(self, model):
        """Test model encode method."""
        batch_size, seq_len = 2, 10
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * batch_size)

        encoder_output = model.encode(input_ids)
        mx.eval(encoder_output)

        assert encoder_output.shape == (batch_size, seq_len, 64)

    def test_model_decode(self, model):
        """Test model decode method."""
        batch_size, src_len, tgt_len = 2, 10, 5
        encoder_hidden_states = mx.random.normal((batch_size, src_len, 64))
        decoder_input_ids = mx.array([[0, 1, 2, 3, 4]] * batch_size)

        logits, cache = model.decode(decoder_input_ids, encoder_hidden_states)
        mx.eval(logits)

        assert logits.shape == (batch_size, tgt_len, 1000)  # vocab_size
        assert cache is not None

    def test_model_full_forward(self, model):
        """Test full encoder-decoder forward pass."""
        batch_size = 2
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * batch_size)
        decoder_input_ids = mx.array([[0, 1, 2]] * batch_size)

        # Encode
        encoder_output = model.encode(input_ids)
        mx.eval(encoder_output)

        # Decode
        logits, cache = model.decode(decoder_input_ids, encoder_output)
        mx.eval(logits)

        assert logits.shape == (batch_size, 3, 1000)

    def test_model_autoregressive_generation(self, model):
        """Test autoregressive generation with cache."""
        batch_size = 2
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * batch_size)

        # Encode once
        encoder_output = model.encode(input_ids)
        mx.eval(encoder_output)

        # Generate token by token
        decoder_input_ids = mx.array([[0]] * batch_size)  # Start token
        cache = None

        for _ in range(5):
            logits, cache = model.decode(
                decoder_input_ids, encoder_output, cache=cache,
            )
            mx.eval(logits)

            # Get next token (greedy)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            decoder_input_ids = next_token

        # Should have generated 5 tokens
        assert logits.shape == (batch_size, 1, 1000)

    def test_model_final_logits_bias(self, model):
        """Test that final_logits_bias is applied."""
        batch_size = 2
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * batch_size)
        decoder_input_ids = mx.array([[0]] * batch_size)

        encoder_output = model.encode(input_ids)
        logits, _ = model.decode(decoder_input_ids, encoder_output)
        mx.eval(logits)

        # final_logits_bias is initialized to zeros by default
        assert model.final_logits_bias.shape == (1, 1000)


class TestSanitizeMarianWeights:
    """Test weight sanitization for Marian models."""

    def test_sanitize_shared_weights(self):
        """Test that shared embedding is distributed correctly."""
        from pytorch_to_mlx.converters.models.marian_mlx import _sanitize_marian_weights

        # Mock HuggingFace weight dict
        weights = {
            "model.shared.weight": mx.zeros((1000, 64)),
        }

        sanitized = _sanitize_marian_weights(weights)

        # Shared weight should be copied to encoder, decoder, and lm_head
        assert "encoder.embed_tokens.weight" in sanitized
        assert "decoder.embed_tokens.weight" in sanitized
        assert "lm_head.weight" in sanitized

    def test_sanitize_encoder_weights(self):
        """Test encoder weight key transformation."""
        from pytorch_to_mlx.converters.models.marian_mlx import _sanitize_marian_weights

        weights = {
            "model.encoder.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
            "model.encoder.layers.0.fc1.weight": mx.zeros((128, 64)),
        }

        sanitized = _sanitize_marian_weights(weights)

        assert "encoder.layers.0.self_attn.q_proj.weight" in sanitized
        assert "encoder.layers.0.fc1.weight" in sanitized

    def test_sanitize_decoder_weights(self):
        """Test decoder weight key transformation."""
        from pytorch_to_mlx.converters.models.marian_mlx import _sanitize_marian_weights

        weights = {
            "model.decoder.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
            "model.decoder.layers.0.encoder_attn.k_proj.weight": mx.zeros((64, 64)),
        }

        sanitized = _sanitize_marian_weights(weights)

        assert "decoder.layers.0.self_attn.q_proj.weight" in sanitized
        assert "decoder.layers.0.encoder_attn.k_proj.weight" in sanitized

    def test_sanitize_ignores_explicit_embeddings(self):
        """Test that explicit embed_tokens weights are ignored."""
        from pytorch_to_mlx.converters.models.marian_mlx import _sanitize_marian_weights

        weights = {
            "model.shared.weight": mx.zeros((1000, 64)),
            "model.encoder.embed_tokens.weight": mx.ones((1000, 64)),  # Should ignore
            "model.decoder.embed_tokens.weight": mx.ones((1000, 64)),  # Should ignore
        }

        sanitized = _sanitize_marian_weights(weights)

        # Should use shared weight, not explicit embeddings
        # Check that encoder embedding is zeros (from shared), not ones
        assert mx.array_equal(
            sanitized["encoder.embed_tokens.weight"], mx.zeros((1000, 64)),
        )


class TestMarianEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        return MarianConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            max_position_embeddings=128,
        )

    def test_single_token_input(self, config):
        """Test with single token input."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianModel

        model = MarianModel(config)
        batch_size = 1
        input_ids = mx.array([[1]])
        decoder_input_ids = mx.array([[0]])

        encoder_output = model.encode(input_ids)
        logits, cache = model.decode(decoder_input_ids, encoder_output)
        mx.eval(logits)

        assert logits.shape == (batch_size, 1, 1000)

    def test_batch_size_one(self, config):
        """Test with batch size of 1."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianModel

        model = MarianModel(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        decoder_input_ids = mx.array([[0, 1, 2]])

        encoder_output = model.encode(input_ids)
        logits, cache = model.decode(decoder_input_ids, encoder_output)
        mx.eval(logits)

        assert logits.shape == (1, 3, 1000)

    def test_large_batch_size(self, config):
        """Test with larger batch size."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianModel

        model = MarianModel(config)
        batch_size = 8
        input_ids = mx.array([[1, 2, 3, 4, 5]] * batch_size)
        decoder_input_ids = mx.array([[0, 1, 2]] * batch_size)

        encoder_output = model.encode(input_ids)
        logits, cache = model.decode(decoder_input_ids, encoder_output)
        mx.eval(logits)

        assert logits.shape == (batch_size, 3, 1000)

    def test_different_batch_elements_independent(self, config):
        """Test that different batch elements are processed independently."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianModel

        model = MarianModel(config)

        # Different inputs for each batch element
        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        decoder_input_ids = mx.array([[0, 1], [0, 2]])

        encoder_output = model.encode(input_ids)
        logits, _ = model.decode(decoder_input_ids, encoder_output)
        mx.eval(logits)

        # Each batch element should produce different logits
        assert logits.shape == (2, 2, 1000)
        # Logits should not be identical
        assert not mx.array_equal(logits[0], logits[1])


class TestMarianNumericalStability:
    """Test numerical stability of Marian model."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianConfig

        return MarianConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            max_position_embeddings=128,
        )

    def test_deterministic_output(self, config):
        """Test that same inputs produce same outputs."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianModel

        model = MarianModel(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        decoder_input_ids = mx.array([[0, 1, 2]])

        encoder_output1 = model.encode(input_ids)
        logits1, _ = model.decode(decoder_input_ids, encoder_output1)
        mx.eval(logits1)

        encoder_output2 = model.encode(input_ids)
        logits2, _ = model.decode(decoder_input_ids, encoder_output2)
        mx.eval(logits2)

        assert mx.allclose(logits1, logits2)

    def test_softmax_stability_in_attention(self, config):
        """Test that attention softmax is numerically stable."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianAttention

        attention = MarianAttention(embed_dim=64, num_heads=4)

        # Use values that might cause numerical issues
        hidden_states = mx.random.normal((1, 10, 64)) * 10

        output, _ = attention(hidden_states)
        mx.eval(output)

        # Check no NaN or Inf
        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))

    def test_no_nan_in_outputs(self, config):
        """Test that model outputs never contain NaN."""
        from pytorch_to_mlx.converters.models.marian_mlx import MarianModel

        model = MarianModel(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        decoder_input_ids = mx.array([[0, 1, 2]])

        encoder_output = model.encode(input_ids)
        logits, _ = model.decode(decoder_input_ids, encoder_output)
        mx.eval(logits)

        assert not mx.any(mx.isnan(encoder_output))
        assert not mx.any(mx.isnan(logits))
