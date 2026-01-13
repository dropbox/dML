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
Tests for NLLB MLX Model Implementation

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


class TestNLLBConfig:
    """Test NLLBConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        from pytorch_to_mlx.converters.models.nllb import NLLBConfig

        config = NLLBConfig()
        assert config.vocab_size == 256206
        assert config.d_model == 1024
        assert config.encoder_layers == 12
        assert config.decoder_layers == 12
        assert config.encoder_attention_heads == 16

    def test_custom_config(self):
        """Test custom configuration."""
        from pytorch_to_mlx.converters.models.nllb import NLLBConfig

        config = NLLBConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=4,
        )
        assert config.vocab_size == 1000
        assert config.d_model == 256
        assert config.encoder_layers == 4


class TestNLLBAttention:
    """Test NLLBAttention module."""

    @pytest.fixture
    def attention(self):
        """Create attention module."""
        from pytorch_to_mlx.converters.models.nllb import NLLBAttention

        return NLLBAttention(embed_dim=64, num_heads=4)

    def test_self_attention(self, attention):
        """Test self-attention forward pass."""
        import mlx.core as mx

        batch_size, seq_len, embed_dim = 2, 10, 64
        hidden_states = mx.random.normal((batch_size, seq_len, embed_dim))

        output, cache = attention(hidden_states)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, embed_dim)
        # Cache is always returned for generation support
        assert cache is not None
        assert len(cache) == 2  # (keys, values)
        assert cache[0].shape == (batch_size, seq_len, embed_dim)  # keys
        assert cache[1].shape == (batch_size, seq_len, embed_dim)  # values

    def test_cross_attention(self, attention):
        """Test cross-attention forward pass."""
        import mlx.core as mx

        batch_size, tgt_len, src_len, embed_dim = 2, 8, 10, 64
        hidden_states = mx.random.normal((batch_size, tgt_len, embed_dim))
        encoder_states = mx.random.normal((batch_size, src_len, embed_dim))

        output, cache = attention(hidden_states, key_value_states=encoder_states)
        mx.eval(output)

        assert output.shape == (batch_size, tgt_len, embed_dim)


class TestNLLBEncoder:
    """Test NLLBEncoder module."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.nllb import NLLBConfig

        return NLLBConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=256,
        )

    @pytest.fixture
    def encoder(self, config):
        """Create encoder."""
        from pytorch_to_mlx.converters.models.nllb import NLLBEncoder

        return NLLBEncoder(config)

    def test_forward(self, encoder, config):
        """Test encoder forward pass."""
        import mlx.core as mx

        batch_size, seq_len = 2, 10
        input_ids = mx.ones((batch_size, seq_len), dtype=mx.int32)

        output = encoder(input_ids)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, config.d_model)


class TestNLLBDecoder:
    """Test NLLBDecoder module."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.nllb import NLLBConfig

        return NLLBConfig(
            vocab_size=1000,
            d_model=64,
            decoder_layers=2,
            decoder_attention_heads=4,
            decoder_ffn_dim=256,
        )

    @pytest.fixture
    def decoder(self, config):
        """Create decoder."""
        from pytorch_to_mlx.converters.models.nllb import NLLBDecoder

        return NLLBDecoder(config)

    def test_forward(self, decoder, config):
        """Test decoder forward pass."""
        import mlx.core as mx

        batch_size, tgt_len, src_len = 2, 8, 10
        input_ids = mx.ones((batch_size, tgt_len), dtype=mx.int32)
        encoder_hidden = mx.random.normal((batch_size, src_len, config.d_model))

        output, cache = decoder(input_ids, encoder_hidden)
        mx.eval(output)

        assert output.shape == (batch_size, tgt_len, config.d_model)

    def test_with_cache(self, decoder, config):
        """Test decoder with KV cache."""
        import mlx.core as mx

        batch_size, src_len = 2, 10
        encoder_hidden = mx.random.normal((batch_size, src_len, config.d_model))

        # First token
        input_ids = mx.ones((batch_size, 1), dtype=mx.int32)
        output1, cache = decoder(input_ids, encoder_hidden, cache=None)
        mx.eval(output1)

        assert output1.shape == (batch_size, 1, config.d_model)
        assert cache is not None

        # Second token with cache
        input_ids2 = mx.ones((batch_size, 1), dtype=mx.int32) * 2
        output2, cache = decoder(input_ids2, encoder_hidden, cache=cache)
        mx.eval(output2)

        assert output2.shape == (batch_size, 1, config.d_model)


class TestNLLBModel:
    """Test complete NLLBModel."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        from pytorch_to_mlx.converters.models.nllb import NLLBConfig

        return NLLBConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
        )

    @pytest.fixture
    def model(self, config):
        """Create model."""
        from pytorch_to_mlx.converters.models.nllb import NLLBModel

        return NLLBModel(config)

    def test_full_forward(self, model, config):
        """Test full model forward pass."""
        import mlx.core as mx

        batch_size, src_len, tgt_len = 2, 10, 8
        input_ids = mx.ones((batch_size, src_len), dtype=mx.int32)
        decoder_input_ids = mx.ones((batch_size, tgt_len), dtype=mx.int32)

        logits = model(input_ids, decoder_input_ids)
        mx.eval(logits)

        assert logits.shape == (batch_size, tgt_len, config.vocab_size)

    def test_encode_decode(self, model, config):
        """Test separate encode/decode API."""
        import mlx.core as mx

        batch_size, src_len = 2, 10
        input_ids = mx.ones((batch_size, src_len), dtype=mx.int32)

        # Encode
        encoder_output = model.encode(input_ids)
        mx.eval(encoder_output)
        assert encoder_output.shape == (batch_size, src_len, config.d_model)

        # Decode first token
        decoder_input = mx.ones((batch_size, 1), dtype=mx.int32)
        logits, cache = model.decode(decoder_input, encoder_output)
        mx.eval(logits)

        assert logits.shape == (batch_size, 1, config.vocab_size)
        assert cache is not None

        # Decode next token with cache
        decoder_input2 = mx.ones((batch_size, 1), dtype=mx.int32) * 2
        logits2, cache = model.decode(decoder_input2, encoder_output, cache=cache)
        mx.eval(logits2)

        assert logits2.shape == (batch_size, 1, config.vocab_size)

    def test_generation_loop(self, model, config):
        """Test autoregressive generation."""
        import mlx.core as mx

        batch_size, src_len, max_new_tokens = 1, 10, 5
        input_ids = mx.ones((batch_size, src_len), dtype=mx.int32)

        # Encode source
        encoder_output = model.encode(input_ids)
        mx.eval(encoder_output)

        # Start with BOS token
        generated = [config.bos_token_id]
        cache = None

        for _ in range(max_new_tokens):
            decoder_input = mx.array([[generated[-1]]], dtype=mx.int32)
            logits, cache = model.decode(decoder_input, encoder_output, cache=cache)
            mx.eval(logits)

            # Greedy selection
            next_token = mx.argmax(logits[0, -1]).item()
            generated.append(next_token)

            if next_token == config.eos_token_id:
                break

        assert len(generated) > 1
        print(f"Generated tokens: {generated}")


def run_quick_test():
    """Quick test without pytest."""
    print("Testing NLLB Model...")
    print("=" * 50)

    import mlx.core as mx
    from pytorch_to_mlx.converters.models.nllb import NLLBConfig, NLLBModel

    # Create small model
    config = NLLBConfig(
        vocab_size=1000,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=256,
        decoder_ffn_dim=256,
    )

    model = NLLBModel(config)
    print("Model created!")

    # Test forward
    input_ids = mx.ones((2, 10), dtype=mx.int32)
    decoder_input_ids = mx.ones((2, 8), dtype=mx.int32)

    logits = model(input_ids, decoder_input_ids)
    mx.eval(logits)

    print(f"Forward pass: input {input_ids.shape} -> logits {logits.shape}")
    assert logits.shape == (2, 8, 1000)

    print("\n" + "=" * 50)
    print("Quick test PASSED!")

    return True


if __name__ == "__main__":
    sys.exit(0 if run_quick_test() else 1)
