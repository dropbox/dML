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
CosyVoice3 LLM Validation Tests

Tests the MLX LLM implementation for:
1. Weight loading from llm.pt
2. Forward pass shape validation
3. Speech token generation
4. KV cache functionality
5. Streaming generation

Run: pytest tests/test_cosyvoice3_llm.py -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import pytest

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def pytorch_llm_weights():
    """Load PyTorch LLM weights."""
    try:
        import torch
        return torch.load(
            'models/cosyvoice3/llm.pt',
            map_location='cpu',
            weights_only=False,
        )
    except Exception as e:
        pytest.skip(f"PyTorch LLM weights not available: {e}")


@pytest.fixture(scope="module")
def mlx_llm_model(pytorch_llm_weights):
    """Create and load MLX LLM model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM,
        Qwen2Config,
    )

    # CosyVoice3 LLM config
    config = Qwen2Config(
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=7,   # GQA: 7 query heads
        num_key_value_heads=1,   # GQA: 1 KV head
        head_dim=128,
        vocab_size=151936,
        speech_vocab_size=6761,  # CosyVoice3 actual size
        llm_embedding_size=2,
    )

    model = CosyVoice2LLM(config)

    # Load weights
    def to_mlx(t):
        return mx.array(t.numpy())

    state_dict = pytorch_llm_weights

    # Load backbone
    model.llm.embed_tokens.weight = to_mlx(state_dict['llm.model.model.embed_tokens.weight'])

    for i in range(24):
        prefix = f'llm.model.model.layers.{i}'
        layer = model.llm.layers[i]
        layer.self_attn.q_proj.weight = to_mlx(state_dict[f'{prefix}.self_attn.q_proj.weight'])
        layer.self_attn.q_proj.bias = to_mlx(state_dict[f'{prefix}.self_attn.q_proj.bias'])
        layer.self_attn.k_proj.weight = to_mlx(state_dict[f'{prefix}.self_attn.k_proj.weight'])
        layer.self_attn.k_proj.bias = to_mlx(state_dict[f'{prefix}.self_attn.k_proj.bias'])
        layer.self_attn.v_proj.weight = to_mlx(state_dict[f'{prefix}.self_attn.v_proj.weight'])
        layer.self_attn.v_proj.bias = to_mlx(state_dict[f'{prefix}.self_attn.v_proj.bias'])
        layer.self_attn.o_proj.weight = to_mlx(state_dict[f'{prefix}.self_attn.o_proj.weight'])
        layer.mlp.gate_proj.weight = to_mlx(state_dict[f'{prefix}.mlp.gate_proj.weight'])
        layer.mlp.up_proj.weight = to_mlx(state_dict[f'{prefix}.mlp.up_proj.weight'])
        layer.mlp.down_proj.weight = to_mlx(state_dict[f'{prefix}.mlp.down_proj.weight'])
        layer.input_layernorm.weight = to_mlx(state_dict[f'{prefix}.input_layernorm.weight'])
        layer.post_attention_layernorm.weight = to_mlx(state_dict[f'{prefix}.post_attention_layernorm.weight'])

    model.llm.norm.weight = to_mlx(state_dict['llm.model.model.norm.weight'])
    model.lm_head.weight = to_mlx(state_dict['llm.model.lm_head.weight'])
    model.speech_embedding.weight = to_mlx(state_dict['speech_embedding.weight'])
    model.llm_decoder.weight = to_mlx(state_dict['llm_decoder.weight'])
    model.llm_decoder.bias = mx.zeros((config.speech_vocab_size,))
    model.llm_embedding.weight = mx.random.normal((2, 896)) * 0.02

    mx.eval(model.parameters())
    return model


# ============================================================================
# CATEGORY 1: WEIGHT LOADING
# ============================================================================

class TestLLMWeightLoading:
    """Test LLM weight loading matches PyTorch."""

    def test_embed_tokens_shape(self, pytorch_llm_weights):
        """Test embedding weights shape."""
        key = 'llm.model.model.embed_tokens.weight'
        assert key in pytorch_llm_weights
        shape = pytorch_llm_weights[key].shape
        assert shape == (151936, 896), f"Expected (151936, 896), got {shape}"

    def test_speech_embedding_shape(self, pytorch_llm_weights):
        """Test speech embedding weights shape."""
        key = 'speech_embedding.weight'
        assert key in pytorch_llm_weights
        shape = pytorch_llm_weights[key].shape
        assert shape[1] == 896, f"Expected dim 896, got {shape[1]}"

    def test_attention_shapes(self, pytorch_llm_weights):
        """Test attention projection shapes confirm GQA config."""
        # Q projection: 7 heads * 128 dim = 896
        q_shape = pytorch_llm_weights['llm.model.model.layers.0.self_attn.q_proj.weight'].shape
        assert q_shape == (896, 896), f"Q proj shape: {q_shape}"

        # K, V projections: 1 head * 128 dim = 128
        k_shape = pytorch_llm_weights['llm.model.model.layers.0.self_attn.k_proj.weight'].shape
        assert k_shape == (128, 896), f"K proj shape: {k_shape}"

        v_shape = pytorch_llm_weights['llm.model.model.layers.0.self_attn.v_proj.weight'].shape
        assert v_shape == (128, 896), f"V proj shape: {v_shape}"

    def test_mlp_shapes(self, pytorch_llm_weights):
        """Test MLP shapes."""
        gate = pytorch_llm_weights['llm.model.model.layers.0.mlp.gate_proj.weight'].shape
        assert gate == (4864, 896), f"Gate proj shape: {gate}"

        down = pytorch_llm_weights['llm.model.model.layers.0.mlp.down_proj.weight'].shape
        assert down == (896, 4864), f"Down proj shape: {down}"


# ============================================================================
# CATEGORY 2: FORWARD PASS
# ============================================================================

class TestLLMForwardPass:
    """Test LLM forward pass."""

    def test_forward_shape(self, mlx_llm_model):
        """Test forward pass output shapes."""
        batch, seq_len = 1, 10
        input_ids = mx.zeros((batch, seq_len), dtype=mx.int32)

        text_logits, speech_logits, cache = mlx_llm_model(input_ids)
        mx.eval(text_logits, speech_logits)

        assert text_logits.shape == (1, 10, 151936), f"Text logits: {text_logits.shape}"
        assert speech_logits.shape == (1, 10, 6761), f"Speech logits: {speech_logits.shape}"
        assert len(cache) == 24, f"Cache layers: {len(cache)}"

    def test_forward_no_nan(self, mlx_llm_model):
        """Test forward pass produces no NaN values."""
        input_ids = mx.array([[100, 200, 300]], dtype=mx.int32)

        text_logits, speech_logits, _ = mlx_llm_model(input_ids)
        mx.eval(text_logits, speech_logits)

        assert not bool(mx.any(mx.isnan(text_logits))), "Text logits contain NaN"
        assert not bool(mx.any(mx.isnan(speech_logits))), "Speech logits contain NaN"

    def test_forward_reasonable_values(self, mlx_llm_model):
        """Test forward pass produces reasonable logit values."""
        input_ids = mx.array([[100, 200, 300]], dtype=mx.int32)

        text_logits, speech_logits, _ = mlx_llm_model(input_ids)
        mx.eval(text_logits, speech_logits)

        # Logits should be in reasonable range (not exploding)
        text_max = float(mx.abs(text_logits).max())
        speech_max = float(mx.abs(speech_logits).max())

        assert text_max < 1000, f"Text logits too large: {text_max}"
        assert speech_max < 1000, f"Speech logits too large: {speech_max}"


# ============================================================================
# CATEGORY 3: SPEECH TOKEN GENERATION
# ============================================================================

class TestSpeechTokenGeneration:
    """Test speech token generation."""

    def test_ras_sampling_generates_tokens(self, mlx_llm_model):
        """Test ras_sampling generates valid tokens."""
        mx.random.seed(42)
        text_ids = mx.array([[100, 200, 300, 400, 500]], dtype=mx.int32)

        tokens = mlx_llm_model.generate_speech_tokens_ras(
            text_ids,
            max_length=50,
            top_k=25,
            top_p=0.8,
            speech_token_size=6561,
        )
        mx.eval(tokens)

        assert tokens.shape[0] == 1, "Batch size should be 1"
        assert tokens.shape[1] > 0, "Should generate some tokens"
        assert tokens.shape[1] <= 50, f"Generated {tokens.shape[1]} > max 50"

    def test_token_range_valid(self, mlx_llm_model):
        """Test generated tokens are in valid range."""
        mx.random.seed(123)
        text_ids = mx.array([[100, 200, 300]], dtype=mx.int32)

        tokens = mlx_llm_model.generate_speech_tokens_ras(
            text_ids,
            max_length=100,
            speech_token_size=6561,
        )
        mx.eval(tokens)

        min_token = int(tokens.min())
        max_token = int(tokens.max())

        assert min_token >= 0, f"Min token {min_token} < 0"
        assert max_token < 6561, f"Max token {max_token} >= 6561"

    def test_deterministic_with_seed(self, mlx_llm_model):
        """Test generation is deterministic with same seed."""
        text_ids = mx.array([[100, 200, 300]], dtype=mx.int32)

        mx.random.seed(999)
        tokens1 = mlx_llm_model.generate_speech_tokens_ras(
            text_ids, max_length=30, speech_token_size=6561,
        )
        mx.eval(tokens1)

        mx.random.seed(999)
        tokens2 = mlx_llm_model.generate_speech_tokens_ras(
            text_ids, max_length=30, speech_token_size=6561,
        )
        mx.eval(tokens2)

        assert tokens1.shape == tokens2.shape, "Shapes differ"
        diff = np.abs(np.array(tokens1) - np.array(tokens2)).max()
        assert diff == 0, f"Non-deterministic: max diff = {diff}"


# ============================================================================
# CATEGORY 4: KV CACHE
# ============================================================================

class TestKVCache:
    """Test KV cache functionality."""

    def test_cache_grows(self, mlx_llm_model):
        """Test cache grows with each token."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import make_kv_cache

        cache = make_kv_cache(24)
        input_ids = mx.array([[100, 200, 300]], dtype=mx.int32)

        _, _, cache = mlx_llm_model(input_ids, cache=cache)
        mx.eval(mlx_llm_model.parameters())

        cache_len1 = cache[0].offset
        assert cache_len1 == 3, f"Cache should have 3 tokens, got {cache_len1}"

        next_input = mx.array([[400]], dtype=mx.int32)
        _, _, cache = mlx_llm_model(next_input, cache=cache)
        mx.eval(mlx_llm_model.parameters())

        cache_len2 = cache[0].offset
        assert cache_len2 == 4, f"Cache should have 4 tokens, got {cache_len2}"

    def test_cache_reset(self, mlx_llm_model):
        """Test cache can be reset."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import make_kv_cache

        cache = make_kv_cache(24)
        input_ids = mx.array([[100, 200, 300]], dtype=mx.int32)

        _, _, cache = mlx_llm_model(input_ids, cache=cache)
        mx.eval(mlx_llm_model.parameters())

        # Reset
        for layer_cache in cache:
            layer_cache.reset()

        assert cache[0].offset == 0, "Cache should be empty after reset"


# ============================================================================
# CATEGORY 5: STREAMING
# ============================================================================

class TestStreamingGeneration:
    """Test streaming token generation."""

    def test_streaming_yields_chunks(self, mlx_llm_model):
        """Test streaming generation yields chunks."""
        text_ids = mx.array([[100, 200, 300]], dtype=mx.int32)

        chunks = []
        for chunk, is_final in mlx_llm_model.generate_speech_tokens_stream(
            text_ids,
            max_length=50,
            chunk_size=10,
        ):
            mx.eval(chunk)
            chunks.append(chunk)
            if is_final:
                break

        assert len(chunks) > 0, "Should yield at least one chunk"
        total_tokens = sum(c.shape[1] for c in chunks)
        assert total_tokens <= 50, f"Total tokens {total_tokens} > max 50"

    def test_streaming_chunk_size(self, mlx_llm_model):
        """Test streaming respects chunk_size."""
        text_ids = mx.array([[100, 200, 300]], dtype=mx.int32)
        chunk_size = 5

        for chunk, is_final in mlx_llm_model.generate_speech_tokens_stream(
            text_ids,
            max_length=50,
            chunk_size=chunk_size,
        ):
            mx.eval(chunk)
            # All chunks should be <= chunk_size (last chunk may be smaller)
            assert chunk.shape[1] <= chunk_size, f"Chunk size {chunk.shape[1]} > {chunk_size}"
            if is_final:
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
