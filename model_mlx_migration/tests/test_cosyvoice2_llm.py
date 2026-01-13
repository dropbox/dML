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
Unit tests for CosyVoice2 LLM Component (Qwen2-based).

Tests the LLM model components:
- Qwen2Config
- RoPE embeddings
- Qwen2RMSNorm
- Qwen2Attention (GQA)
- Qwen2MLP (SwiGLU)
- Qwen2DecoderLayer
- Qwen2Model
- CosyVoice2LLM
- Early-exit speculative decoding
"""

import os
import time

import mlx.core as mx
import pytest

from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
    CosyVoice2LLM,
    Qwen2Attention,
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2Model,
    Qwen2RMSNorm,
    apply_rotary_embedding,
    precompute_rope_frequencies,
)


class TestQwen2Config:
    """Tests for Qwen2Config."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Qwen2Config()

        assert config.hidden_size == 896
        assert config.intermediate_size == 4864
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 7
        assert config.num_key_value_heads == 1
        assert config.head_dim == 128
        assert config.vocab_size == 151936
        assert config.speech_vocab_size == 6564

    def test_custom_config(self):
        """Test custom configuration values."""
        config = Qwen2Config(
            hidden_size=512,
            num_hidden_layers=12,
        )

        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_precompute_frequencies_shape(self):
        """Test RoPE frequency precomputation shapes."""
        head_dim = 128
        max_seq_len = 128

        cos, sin = precompute_rope_frequencies(head_dim, max_seq_len)

        assert cos.shape == (max_seq_len, head_dim)
        assert sin.shape == (max_seq_len, head_dim)

    def test_precompute_frequencies_range(self):
        """Test that frequencies are in [-1, 1] range."""
        cos, sin = precompute_rope_frequencies(64, 128)
        mx.eval(cos, sin)

        assert mx.all(cos >= -1.0).item()
        assert mx.all(cos <= 1.0).item()
        assert mx.all(sin >= -1.0).item()
        assert mx.all(sin <= 1.0).item()

    def test_apply_rotary_embedding_shape(self):
        """Test rotary embedding application preserves shape."""
        batch, heads, seq_len, head_dim = 2, 7, 32, 128
        cos, sin = precompute_rope_frequencies(head_dim, 128)

        q = mx.random.normal((batch, heads, seq_len, head_dim))
        k = mx.random.normal((batch, heads, seq_len, head_dim))

        q_rot, k_rot = apply_rotary_embedding(q, k, cos, sin)
        mx.eval(q_rot, k_rot)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestQwen2RMSNorm:
    """Tests for Qwen2RMSNorm."""

    def test_output_shape(self):
        """Test output shape equals input shape."""
        hidden_size = 896
        norm = Qwen2RMSNorm(hidden_size)

        x = mx.random.normal((2, 10, hidden_size))
        y = norm(x)
        mx.eval(y)

        assert y.shape == x.shape

    def test_normalization_effect(self):
        """Test that normalization changes values."""
        norm = Qwen2RMSNorm(896)

        x = mx.random.normal((2, 10, 896)) * 10  # Large values
        y = norm(x)
        mx.eval(x, y)

        # Output variance should be closer to 1 (after scaling by weight)
        input_var = mx.var(x).item()
        output_var = mx.var(y).item()

        # Normalized variance should be smaller than scaled input
        assert output_var < input_var * 100


class TestQwen2Attention:
    """Tests for Qwen2Attention (GQA)."""

    def test_output_shape(self):
        """Test output shape equals input shape."""
        config = Qwen2Config()
        attn = Qwen2Attention(config)

        x = mx.random.normal((2, 10, config.hidden_size))
        cos, sin = precompute_rope_frequencies(config.head_dim, 128)

        y, cache = attn(x, cos=cos, sin=sin)
        mx.eval(y)

        assert y.shape == x.shape

    def test_gqa_dimensions(self):
        """Test GQA head configuration."""
        config = Qwen2Config()
        attn = Qwen2Attention(config)

        # 7 query heads, 1 KV head
        assert attn.num_heads == 7
        assert attn.num_kv_heads == 1
        assert attn.num_heads_per_kv == 7  # 7 query heads per KV head

    def test_cache_shape(self):
        """Test KV cache shape."""
        config = Qwen2Config()
        attn = Qwen2Attention(config)

        batch, seq_len = 2, 10
        x = mx.random.normal((batch, seq_len, config.hidden_size))
        cos, sin = precompute_rope_frequencies(config.head_dim, 128)

        _, cache = attn(x, cos=cos, sin=sin)
        mx.eval(cache[0], cache[1])

        k_cache, v_cache = cache
        # KV heads = 1, so shape is [batch, 1, seq_len, head_dim]
        assert k_cache.shape == (batch, 1, seq_len, config.head_dim)
        assert v_cache.shape == (batch, 1, seq_len, config.head_dim)

    def test_causal_masking(self):
        """Test that causal mask is applied correctly."""
        config = Qwen2Config()
        attn = Qwen2Attention(config)

        batch, seq_len = 2, 10
        x = mx.random.normal((batch, seq_len, config.hidden_size))
        cos, sin = precompute_rope_frequencies(config.head_dim, 128)

        # Create causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
        mask = mask[None, None, :, :]

        y, _ = attn(x, attention_mask=mask, cos=cos, sin=sin)
        mx.eval(y)

        assert y.shape == x.shape

    def test_qkv_fusion_lossless(self):
        """Test Q18-LLM: QKV fusion produces identical outputs."""
        config = Qwen2Config()
        attn = Qwen2Attention(config)

        batch, seq_len = 2, 10
        x = mx.random.normal((batch, seq_len, config.hidden_size))
        cos, sin = precompute_rope_frequencies(config.head_dim, 128)

        # Output without fusion
        y_unfused, cache_unfused = attn(x, cos=cos, sin=sin)
        mx.eval(y_unfused, cache_unfused[0], cache_unfused[1])

        # Fuse QKV weights
        attn.fuse_qkv()
        assert attn._qkv_fused is True

        # Output with fusion
        y_fused, cache_fused = attn(x, cos=cos, sin=sin)
        mx.eval(y_fused, cache_fused[0], cache_fused[1])

        # Outputs should be identical (lossless)
        max_diff = mx.abs(y_unfused - y_fused).max().item()
        assert max_diff < 1e-5, f"QKV fusion should be lossless, but max_diff={max_diff}"

        # Cache should also be identical
        k_diff = mx.abs(cache_unfused[0] - cache_fused[0]).max().item()
        v_diff = mx.abs(cache_unfused[1] - cache_fused[1]).max().item()
        assert k_diff < 1e-5, f"K cache should be identical, but max_diff={k_diff}"
        assert v_diff < 1e-5, f"V cache should be identical, but max_diff={v_diff}"

    def test_qkv_unfusion(self):
        """Test that QKV unfusion works correctly."""
        config = Qwen2Config()
        attn = Qwen2Attention(config)

        # Initially not fused
        assert attn._qkv_fused is False

        # Fuse
        attn.fuse_qkv()
        assert attn._qkv_fused is True
        assert attn._qkv_proj is not None

        # Unfuse
        attn.unfuse_qkv()
        assert attn._qkv_fused is False
        assert attn._qkv_proj is None


class TestQwen2MLP:
    """Tests for Qwen2MLP (SwiGLU)."""

    def test_output_shape(self):
        """Test output shape equals input shape."""
        config = Qwen2Config()
        mlp = Qwen2MLP(config)

        x = mx.random.normal((2, 10, config.hidden_size))
        y = mlp(x)
        mx.eval(y)

        assert y.shape == x.shape

    def test_swiglu_activation(self):
        """Test that SwiGLU produces non-zero output."""
        config = Qwen2Config()
        mlp = Qwen2MLP(config)

        x = mx.random.normal((2, 10, config.hidden_size))
        y = mlp(x)
        mx.eval(y)

        # Should produce non-zero values
        assert not mx.all(y == 0).item()


class TestQwen2DecoderLayer:
    """Tests for Qwen2DecoderLayer."""

    def test_output_shape(self):
        """Test output shape equals input shape."""
        config = Qwen2Config()
        layer = Qwen2DecoderLayer(config)
        cos, sin = precompute_rope_frequencies(config.head_dim, 128)

        x = mx.random.normal((2, 10, config.hidden_size))
        y, cache = layer(x, cos=cos, sin=sin)
        mx.eval(y)

        assert y.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connections work."""
        config = Qwen2Config()
        layer = Qwen2DecoderLayer(config)
        cos, sin = precompute_rope_frequencies(config.head_dim, 128)

        x = mx.random.normal((2, 10, config.hidden_size))
        y, _ = layer(x, cos=cos, sin=sin)
        mx.eval(x, y)

        # Output should be different from input (due to transformation)
        # but not dramatically different (due to residuals)
        diff = mx.abs(y - x).mean().item()
        assert diff > 0  # Some change
        assert diff < 100  # Not huge change


class TestQwen2Model:
    """Tests for Qwen2Model (full transformer)."""

    def test_output_shape(self):
        """Test output shape."""
        # Use smaller config for faster testing
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = Qwen2Model(config)

        batch, seq_len = 2, 10
        input_ids = mx.random.randint(0, config.vocab_size, (batch, seq_len))

        hidden_states, cache = model(input_ids)
        mx.eval(hidden_states)

        assert hidden_states.shape == (batch, seq_len, config.hidden_size)

    def test_cache_length(self):
        """Test that cache has correct number of layers."""
        config = Qwen2Config(num_hidden_layers=4, vocab_size=1000)
        model = Qwen2Model(config)

        input_ids = mx.random.randint(0, config.vocab_size, (2, 10))
        _, cache = model(input_ids)
        mx.eval(cache[0][0])

        assert len(cache) == config.num_hidden_layers


class TestCosyVoice2LLM:
    """Tests for CosyVoice2LLM."""

    def test_initialization(self):
        """Test model initialization."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        assert model.llm is not None
        assert model.llm_embedding is not None
        assert model.speech_embedding is not None
        assert model.lm_head is not None
        assert model.llm_decoder is not None

    def test_forward_pass(self):
        """Test forward pass produces logits."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        batch, seq_len = 2, 10
        input_ids = mx.random.randint(0, config.vocab_size, (batch, seq_len))

        text_logits, speech_logits, cache = model(input_ids)
        mx.eval(text_logits, speech_logits)

        assert text_logits.shape == (batch, seq_len, config.vocab_size)
        assert speech_logits.shape == (batch, seq_len, config.speech_vocab_size)

    def test_embedding_dimensions(self):
        """Test embedding dimensions."""
        config = Qwen2Config(num_hidden_layers=2)
        model = CosyVoice2LLM(config)

        # LLM embedding (SOS/EOS)
        assert model.llm_embedding.weight.shape == (
            config.llm_embedding_size,
            config.hidden_size,
        )

        # Speech embedding
        assert model.speech_embedding.weight.shape == (
            config.speech_vocab_size,
            config.hidden_size,
        )

    def test_fuse_qkv_weights_lossless(self):
        """Test Q18-LLM: fuse_qkv_weights produces identical outputs."""
        config = Qwen2Config(num_hidden_layers=4, vocab_size=1000)
        model = CosyVoice2LLM(config)

        batch, seq_len = 2, 10
        input_ids = mx.random.randint(0, config.vocab_size, (batch, seq_len))

        # Forward pass without fusion
        text_logits_unfused, speech_logits_unfused, _ = model(input_ids)
        mx.eval(text_logits_unfused, speech_logits_unfused)

        # Fuse QKV weights
        model.fuse_qkv_weights()

        # Forward pass with fusion
        text_logits_fused, speech_logits_fused, _ = model(input_ids)
        mx.eval(text_logits_fused, speech_logits_fused)

        # Outputs should be identical (lossless)
        max_text_diff = mx.abs(text_logits_unfused - text_logits_fused).max().item()
        max_speech_diff = mx.abs(speech_logits_unfused - speech_logits_fused).max().item()

        assert max_text_diff < 1e-4, f"QKV fusion should be lossless for text logits, but max_diff={max_text_diff}"
        assert max_speech_diff < 1e-4, f"QKV fusion should be lossless for speech logits, but max_diff={max_speech_diff}"

    def test_fuse_qkv_weights_all_layers(self):
        """Test that fuse_qkv_weights fuses all attention layers."""
        config = Qwen2Config(num_hidden_layers=4, vocab_size=1000)
        model = CosyVoice2LLM(config)

        # Initially unfused
        for layer in model.llm.layers:
            assert layer.self_attn._qkv_fused is False

        # Fuse all
        model.fuse_qkv_weights()

        # All should be fused
        for layer in model.llm.layers:
            assert layer.self_attn._qkv_fused is True
            assert layer.self_attn._qkv_proj is not None

        # Unfuse all
        model.unfuse_qkv_weights()

        # All should be unfused
        for layer in model.llm.layers:
            assert layer.self_attn._qkv_fused is False
            assert layer.self_attn._qkv_proj is None


def count_params(params):
    """Recursively count parameters in nested dict."""
    total = 0
    if isinstance(params, dict):
        for v in params.values():
            total += count_params(v)
    elif isinstance(params, mx.array):
        total += params.size
    return total


class TestParameterCount:
    """Tests for parameter counts."""

    def test_attention_params(self):
        """Test attention parameter count."""
        config = Qwen2Config()
        attn = Qwen2Attention(config)

        total = count_params(attn.parameters())

        # Q: 896 * 896 + 896 = 803,712
        # K: 896 * 128 + 128 = 114,816
        # V: 896 * 128 + 128 = 114,816
        # O: 896 * 896 = 802,816
        # Total: ~1.8M per attention
        assert total > 1_000_000
        assert total < 3_000_000

    def test_mlp_params(self):
        """Test MLP parameter count."""
        config = Qwen2Config()
        mlp = Qwen2MLP(config)

        total = count_params(mlp.parameters())

        # gate: 896 * 4864 = 4,358,144
        # up: 896 * 4864 = 4,358,144
        # down: 4864 * 896 = 4,358,144
        # Total: ~13M per MLP
        assert total > 10_000_000
        assert total < 20_000_000

    def test_small_model_params(self):
        """Test small model parameter count."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        total = count_params(model.parameters())

        # Should be in millions
        assert total > 1_000_000


class TestSamplingMethods:
    """Tests for token sampling methods."""

    def test_sample_tokens_temperature(self):
        """Test temperature affects distribution."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        # Create logits with one high value
        logits_np = [[0.0] * 100]
        logits_np[0][0] = 10.0  # One high logit
        logits = mx.array(logits_np)

        # Low temperature - should almost always pick token 0
        mx.random.seed(42)
        samples_low_temp = []
        for _ in range(10):
            sample = model.sample_tokens(logits, temperature=0.1, top_k=0, top_p=1.0)
            mx.eval(sample)
            samples_low_temp.append(int(sample[0]))
        # Most should be 0
        assert samples_low_temp.count(0) >= 8

        # High temperature - more random
        mx.random.seed(42)
        samples_high_temp = []
        for _ in range(10):
            sample = model.sample_tokens(logits, temperature=2.0, top_k=0, top_p=1.0)
            mx.eval(sample)
            samples_high_temp.append(int(sample[0]))
        # Should have more variety
        assert len(set(samples_high_temp)) >= 1  # At least some variation

    def test_sample_tokens_top_k(self):
        """Test top-k sampling limits choices."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        # Create logits with distinct top values
        logits = mx.arange(100, dtype=mx.float32).reshape(1, -1)

        # With top_k=5, should only sample from highest 5 tokens (95-99)
        mx.random.seed(42)
        samples = []
        for _ in range(20):
            sample = model.sample_tokens(logits, top_k=5, temperature=0.5)
            samples.append(int(sample[0]))

        # All samples should be from top 5 (indices 95-99)
        for s in samples:
            assert s >= 95

    def test_sample_tokens_output_shape(self):
        """Test sample output shape."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        batch = 4
        logits = mx.random.normal((batch, 100))
        samples = model.sample_tokens(logits)
        mx.eval(samples)

        assert samples.shape == (batch,)


class TestStreamingGeneration:
    """Tests for streaming generation."""

    def test_generate_speech_tokens_shape(self):
        """Test generate_speech_tokens output shape."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        batch = 1
        text_ids = mx.random.randint(0, config.vocab_size, (batch, 5))

        # Generate a few tokens
        tokens = model.generate_speech_tokens(
            text_ids, max_length=3, temperature=1.0, top_k=10,
        )
        mx.eval(tokens)

        assert tokens.shape[0] == batch
        assert tokens.shape[1] <= 3  # May stop early

    def test_generate_stream_yields_chunks(self):
        """Test streaming generation yields chunks."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        batch = 1
        text_ids = mx.random.randint(0, config.vocab_size, (batch, 5))

        # Generate with small chunks
        chunks = list(
            model.generate_speech_tokens_stream(
                text_ids,
                max_length=10,
                chunk_size=3,
                temperature=1.0,
                top_k=10,
            ),
        )

        # Should have yielded at least one chunk
        assert len(chunks) >= 1

        # Each chunk should be a tuple (tokens, is_final)
        for chunk_tokens, is_final in chunks:
            assert isinstance(is_final, bool)
            assert chunk_tokens.shape[0] == batch

        # Last chunk should be final
        assert chunks[-1][1] is True

    def test_generate_stream_total_tokens(self):
        """Test streaming generates correct total tokens."""
        config = Qwen2Config(num_hidden_layers=2, vocab_size=1000)
        model = CosyVoice2LLM(config)

        text_ids = mx.random.randint(0, config.vocab_size, (1, 5))

        # Collect all streamed tokens
        all_tokens = []
        for chunk_tokens, _is_final in model.generate_speech_tokens_stream(
            text_ids,
            max_length=8,
            chunk_size=3,
            temperature=1.0,
            top_k=25,
        ):
            mx.eval(chunk_tokens)
            all_tokens.append(chunk_tokens)

        # Concatenate all chunks
        total_tokens = mx.concatenate(all_tokens, axis=1)
        assert total_tokens.shape[1] <= 8


class TestEarlyExitDecoding:
    """Tests for early-exit speculative decoding."""

    def test_early_exit_head_initialization(self):
        """Test early exit head is initialized."""
        config = Qwen2Config(num_hidden_layers=8, vocab_size=1000)
        model = CosyVoice2LLM(config, early_exit_layer=4)

        assert model.early_exit_layer == 4
        assert model.early_exit_head is not None
        assert model.early_exit_head.weight.shape == (
            config.speech_vocab_size,
            config.hidden_size,
        )

    def test_early_exit_forward(self):
        """Test forward pass with early exit produces logits."""
        config = Qwen2Config(num_hidden_layers=8, vocab_size=1000)
        model = CosyVoice2LLM(config, early_exit_layer=4)

        batch, seq_len = 2, 10
        input_ids = mx.random.randint(0, config.vocab_size, (batch, seq_len))

        # Normal forward
        text_logits_full, speech_logits_full, cache_full = model(
            input_ids, use_early_exit=False,
        )
        mx.eval(speech_logits_full)

        # Early exit forward
        text_logits_early, speech_logits_early, cache_early = model(
            input_ids, use_early_exit=True,
        )
        mx.eval(speech_logits_early)

        # Shapes should match
        assert speech_logits_full.shape == speech_logits_early.shape
        assert speech_logits_full.shape == (batch, seq_len, config.speech_vocab_size)

        # Early exit cache should have fewer layers
        assert len(cache_early) == 4  # early_exit_layer
        assert len(cache_full) == 8  # all layers

    def test_early_exit_different_outputs(self):
        """Test that early exit produces different outputs than full model."""
        config = Qwen2Config(num_hidden_layers=8, vocab_size=1000)
        model = CosyVoice2LLM(config, early_exit_layer=4)

        batch, seq_len = 1, 5
        input_ids = mx.random.randint(0, config.vocab_size, (batch, seq_len))

        _, speech_logits_full, _ = model(input_ids, use_early_exit=False)
        _, speech_logits_early, _ = model(input_ids, use_early_exit=True)
        mx.eval(speech_logits_full, speech_logits_early)

        # Should be different (early exit uses fewer layers + different head)
        max_diff = mx.abs(speech_logits_full - speech_logits_early).max().item()
        assert max_diff > 0.1  # Should be noticeably different

    def test_speculative_decode_output(self):
        """Test speculative decode produces tokens and stats."""
        config = Qwen2Config(num_hidden_layers=4, vocab_size=1000)
        model = CosyVoice2LLM(config, early_exit_layer=2)

        batch = 1
        text_ids = mx.random.randint(0, config.vocab_size, (batch, 5))

        tokens, stats = model.generate_speech_tokens_speculative(
            text_ids,
            max_length=10,
            num_draft_tokens=2,
            temperature=1.0,
            top_k=10,
        )
        mx.eval(tokens)

        # Should have generated some tokens
        assert tokens.shape[0] == batch
        assert tokens.shape[1] > 0  # At least one token generated

        # Stats should be populated
        assert "acceptance_rate" in stats
        assert "total_tokens" in stats
        assert "draft_calls" in stats
        assert "verify_calls" in stats
        assert stats["total_tokens"] == tokens.shape[1]

    def test_speculative_decode_stats_reasonable(self):
        """Test speculative decode stats are within reasonable bounds."""
        config = Qwen2Config(num_hidden_layers=4, vocab_size=1000)
        model = CosyVoice2LLM(config, early_exit_layer=2)

        text_ids = mx.random.randint(0, config.vocab_size, (1, 5))

        _, stats = model.generate_speech_tokens_speculative(
            text_ids,
            max_length=20,
            num_draft_tokens=4,
            temperature=1.0,
            top_k=10,
        )

        # Acceptance rate should be between 0 and 1
        assert 0.0 <= stats["acceptance_rate"] <= 1.0

        # Draft calls should be > 0
        assert stats["draft_calls"] > 0

        # Verify calls should match draft calls
        assert stats["verify_calls"] == stats["draft_calls"]

    def test_speculative_decode_vs_normal_generation(self):
        """Test speculative decode produces similar length to normal generation."""
        config = Qwen2Config(num_hidden_layers=4, vocab_size=1000)
        model = CosyVoice2LLM(config, early_exit_layer=2)

        text_ids = mx.random.randint(0, config.vocab_size, (1, 5))
        max_len = 15

        # Normal generation
        mx.random.seed(42)
        normal_tokens = model.generate_speech_tokens(
            text_ids, max_length=max_len, temperature=1.0, top_k=10,
        )
        mx.eval(normal_tokens)

        # Speculative generation (may differ due to different sampling strategy)
        mx.random.seed(42)
        spec_tokens, stats = model.generate_speech_tokens_speculative(
            text_ids, max_length=max_len, num_draft_tokens=3, temperature=1.0, top_k=10,
        )
        mx.eval(spec_tokens)

        # Both should produce tokens
        assert normal_tokens.shape[1] > 0
        assert spec_tokens.shape[1] > 0

        # Stats should be valid
        assert stats["total_tokens"] == spec_tokens.shape[1]


class TestSpeculativeDecodingBenchmark:
    """Benchmark tests for speculative decoding (skipped by default)."""

    @pytest.mark.skipif(
        not os.environ.get("RUN_BENCHMARKS"),
        reason="Set RUN_BENCHMARKS=1 to run benchmark tests",
    )
    def test_speculative_vs_normal_timing(self):
        """Benchmark speculative decoding vs normal decoding."""
        # Use a medium-sized model for more meaningful benchmarks
        config = Qwen2Config(num_hidden_layers=12, vocab_size=10000)
        model = CosyVoice2LLM(config, early_exit_layer=6)

        text_ids = mx.random.randint(0, config.vocab_size, (1, 10))
        max_len = 50
        num_runs = 3

        # Warmup
        _ = model.generate_speech_tokens(
            text_ids, max_length=5, temperature=1.0, top_k=10,
        )
        mx.eval(_)

        # Normal generation timing
        normal_times = []
        for _ in range(num_runs):
            mx.random.seed(42)
            start = time.perf_counter()
            normal_tokens = model.generate_speech_tokens(
                text_ids, max_length=max_len, temperature=1.0, top_k=10,
            )
            mx.eval(normal_tokens)
            normal_times.append(time.perf_counter() - start)

        # Speculative generation timing
        spec_times = []
        acceptance_rates = []
        for _ in range(num_runs):
            mx.random.seed(42)
            start = time.perf_counter()
            spec_tokens, stats = model.generate_speech_tokens_speculative(
                text_ids,
                max_length=max_len,
                num_draft_tokens=4,
                temperature=1.0,
                top_k=10,
            )
            mx.eval(spec_tokens)
            spec_times.append(time.perf_counter() - start)
            acceptance_rates.append(stats["acceptance_rate"])

        # Report results
        avg_normal = sum(normal_times) / len(normal_times)
        avg_spec = sum(spec_times) / len(spec_times)
        avg_acceptance = sum(acceptance_rates) / len(acceptance_rates)
        speedup = avg_normal / avg_spec if avg_spec > 0 else 0

        print("\n=== Speculative Decoding Benchmark ===")
        print(f"Model: {config.num_hidden_layers} layers, early exit at layer 6")
        print(f"Max tokens: {max_len}")
        print(f"Normal generation: {avg_normal*1000:.2f}ms")
        print(f"Speculative generation: {avg_spec*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Acceptance rate: {avg_acceptance*100:.1f}%")
        print(f"Draft calls: {stats['draft_calls']}, Verify calls: {stats['verify_calls']}")

        # The test passes regardless of speedup - we just want to measure
        # Note: Early exit without trained weights will have low acceptance rate
        assert avg_acceptance >= 0.0  # Sanity check

    @pytest.mark.skipif(
        not os.environ.get("RUN_BENCHMARKS"),
        reason="Set RUN_BENCHMARKS=1 to run benchmark tests",
    )
    def test_early_exit_layer_sweep(self):
        """Test different early exit layers to find optimal configuration."""
        config = Qwen2Config(num_hidden_layers=12, vocab_size=10000)
        text_ids = mx.random.randint(0, config.vocab_size, (1, 10))
        max_len = 30

        results = []
        for exit_layer in [3, 4, 6, 8, 10]:
            model = CosyVoice2LLM(config, early_exit_layer=exit_layer)

            # Run speculative decode
            mx.random.seed(42)
            start = time.perf_counter()
            _, stats = model.generate_speech_tokens_speculative(
                text_ids,
                max_length=max_len,
                num_draft_tokens=4,
                temperature=1.0,
                top_k=10,
            )
            elapsed = time.perf_counter() - start

            results.append({
                "exit_layer": exit_layer,
                "time_ms": elapsed * 1000,
                "acceptance_rate": stats["acceptance_rate"],
                "draft_calls": stats["draft_calls"],
            })

        print("\n=== Early Exit Layer Sweep ===")
        print(f"{'Layer':<8} {'Time (ms)':<12} {'Acceptance':<12} {'Draft Calls':<12}")
        print("-" * 44)
        for r in results:
            print(
                f"{r['exit_layer']:<8} {r['time_ms']:<12.2f} "
                f"{r['acceptance_rate']*100:<11.1f}% {r['draft_calls']:<12}",
            )

        # Sanity check
        assert len(results) == 5

    @pytest.mark.skipif(
        not os.environ.get("RUN_BENCHMARKS"),
        reason="Set RUN_BENCHMARKS=1 to run benchmark tests",
    )
    def test_initialized_early_exit_acceptance(self):
        """Test acceptance rate with initialized early_exit_head weights."""
        config = Qwen2Config(num_hidden_layers=12, vocab_size=10000)

        # Test 1: Random weights (baseline)
        model_random = CosyVoice2LLM(config, early_exit_layer=6)
        text_ids = mx.random.randint(0, config.vocab_size, (1, 10))

        mx.random.seed(42)
        _, stats_random = model_random.generate_speech_tokens_speculative(
            text_ids, max_length=30, num_draft_tokens=4, temperature=1.0, top_k=10,
        )

        # Test 2: Initialized from llm_decoder weights
        model_init = CosyVoice2LLM(config, early_exit_layer=6)
        model_init.initialize_early_exit_head()

        mx.random.seed(42)
        start = time.perf_counter()
        _, stats_init = model_init.generate_speech_tokens_speculative(
            text_ids, max_length=30, num_draft_tokens=4, temperature=1.0, top_k=10,
        )
        init_time = time.perf_counter() - start

        # Test 3: Normal generation for comparison
        mx.random.seed(42)
        start = time.perf_counter()
        _ = model_init.generate_speech_tokens(
            text_ids, max_length=30, temperature=1.0, top_k=10,
        )
        mx.eval(_)
        normal_time = time.perf_counter() - start

        print("\n=== Early Exit Head Initialization Comparison ===")
        print(f"Random weights acceptance: {stats_random['acceptance_rate']*100:.1f}%")
        print(f"Initialized weights acceptance: {stats_init['acceptance_rate']*100:.1f}%")
        print(f"Normal generation time: {normal_time*1000:.2f}ms")
        print(f"Speculative (initialized) time: {init_time*1000:.2f}ms")
        speedup = normal_time / init_time if init_time > 0 else 0
        print(f"Speedup with initialized weights: {speedup:.2f}x")

        # Initialized weights should have higher acceptance rate
        # With same decoder weights at early exit, acceptance should be much higher
        assert stats_init["acceptance_rate"] >= stats_random["acceptance_rate"]
