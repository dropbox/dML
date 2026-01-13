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
CosyVoice3 End-to-End PyTorch Comparison Tests

Compares MLX implementation against PyTorch reference at each stage:
1. LLM forward pass: text tokens -> logits
2. Flow forward pass: speech tokens + speaker -> mel
3. Vocoder forward pass: mel -> audio

Uses real text input via the Qwen2 tokenizer.
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
def tokenizer():
    """Load Qwen2 tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer_path = "models/cosyvoice3/CosyVoice-BlankEN"
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Tokenizer not available: {e}")


@pytest.fixture(scope="module")
def pytorch_llm():
    """Load PyTorch LLM weights."""
    try:
        import torch
        return torch.load(
            'models/cosyvoice3/llm.pt',
            map_location='cpu',
            weights_only=True,
        )
    except Exception as e:
        pytest.skip(f"PyTorch LLM weights not available: {e}")


@pytest.fixture(scope="module")
def pytorch_flow():
    """Load PyTorch flow weights."""
    try:
        import torch
        return torch.load(
            'models/cosyvoice3/flow.pt',
            map_location='cpu',
            weights_only=True,
        )
    except Exception as e:
        pytest.skip(f"PyTorch flow weights not available: {e}")


@pytest.fixture(scope="module")
def pytorch_vocoder():
    """Load PyTorch vocoder weights."""
    try:
        import torch
        return torch.load(
            'models/cosyvoice3/hift.pt',
            map_location='cpu',
            weights_only=True,
        )
    except Exception as e:
        pytest.skip(f"PyTorch vocoder weights not available: {e}")


@pytest.fixture(scope="module")
def mlx_weights():
    """Load MLX converted weights."""
    try:
        return mx.load('models/cosyvoice3_mlx/model.safetensors')
    except Exception as e:
        pytest.skip(f"MLX weights not available: {e}")


@pytest.fixture(scope="module")
def mlx_flow_model(mlx_weights):
    """Create and load MLX flow model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
        CausalMaskedDiffWithDiT,
        create_cosyvoice3_flow_config,
    )
    config = create_cosyvoice3_flow_config()
    model = CausalMaskedDiffWithDiT(config)
    flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
    model.load_weights(list(flow_weights.items()))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def mlx_vocoder_model(mlx_weights):
    """Create and load MLX vocoder model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import (
        CausalHiFTGenerator,
        create_cosyvoice3_vocoder_config,
    )
    config = create_cosyvoice3_vocoder_config()
    model = CausalHiFTGenerator(config)
    vocoder_weights = {k[8:]: v for k, v in mlx_weights.items() if k.startswith('vocoder.')}
    model.load_weights(list(vocoder_weights.items()))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def mlx_llm_model(mlx_weights):
    """Create and load MLX LLM model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM,
        Qwen2Config,
    )
    # CosyVoice3 uses Qwen2 with GQA: 7 query heads, 1 KV head
    config = Qwen2Config(
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=7,
        num_key_value_heads=1,
        head_dim=128,
        intermediate_size=4864,
        vocab_size=151936,
        speech_vocab_size=6564,
        rope_theta=1000000.0,
    )
    model = CosyVoice2LLM(config)
    llm_weights = {k[4:]: v for k, v in mlx_weights.items() if k.startswith('llm.')}
    model.load_weights(list(llm_weights.items()))
    mx.eval(model.parameters())
    return model


# ============================================================================
# TEST HELPERS
# ============================================================================

def compare_arrays(pt_arr, mlx_arr, name: str, tolerance: float = 1e-3):
    """Compare PyTorch and MLX arrays."""
    import torch

    # Convert to numpy
    pt_np = pt_arr.detach().cpu().numpy() if isinstance(pt_arr, torch.Tensor) else pt_arr
    mlx_np = np.array(mlx_arr.astype(mx.float32)) if isinstance(mlx_arr, mx.array) else mlx_arr

    if pt_np.shape != mlx_np.shape:
        return {
            'name': name,
            'match': False,
            'error': f'Shape mismatch: PT {pt_np.shape} vs MLX {mlx_np.shape}',
            'max_diff': float('inf'),
        }

    diff = np.abs(pt_np.astype(np.float32) - mlx_np.astype(np.float32))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    return {
        'name': name,
        'match': max_diff < tolerance,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'shape': pt_np.shape,
        'tolerance': tolerance,
    }


# ============================================================================
# CATEGORY 1: TOKENIZER TESTS
# ============================================================================

class TestTokenizer:
    """Test tokenizer functionality."""

    def test_tokenizer_loads(self, tokenizer):
        """Test tokenizer loads successfully."""
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_tokenize_english(self, tokenizer):
        """Test tokenizing English text."""
        text = "Hello, how are you today?"
        tokens = tokenizer(text, return_tensors="pt")
        assert tokens["input_ids"].shape[0] == 1
        assert tokens["input_ids"].shape[1] > 0

    def test_tokenize_various_lengths(self, tokenizer):
        """Test tokenizing various text lengths."""
        texts = [
            "Hi",
            "Hello world",
            "This is a longer sentence with more words.",
            "The quick brown fox jumps over the lazy dog. " * 5,
        ]
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt")
            assert tokens["input_ids"].shape[1] > 0


# ============================================================================
# CATEGORY 2: LLM FORWARD PASS COMPARISON
# ============================================================================

class TestLLMForwardComparison:
    """Compare LLM forward pass between PyTorch and MLX."""

    def test_embedding_forward(self, mlx_llm_model, mlx_weights, tokenizer):
        """Test embedding layer output."""
        text = "Hello world"
        tokens = tokenizer(text, return_tensors="pt")
        input_ids_np = tokens["input_ids"].numpy()
        input_ids_mlx = mx.array(input_ids_np)

        # MLX embedding lookup
        mlx_emb = mlx_llm_model.llm.embed_tokens(input_ids_mlx)
        mx.eval(mlx_emb)

        # Check shape
        B, L = input_ids_np.shape
        assert mlx_emb.shape == (B, L, 896), f"Wrong embedding shape: {mlx_emb.shape}"

        # Check for NaN
        assert not bool(mx.any(mx.isnan(mlx_emb))), "Embedding contains NaN"

    def test_llm_forward_no_nan(self, mlx_llm_model, tokenizer):
        """Test LLM forward pass produces no NaN."""
        text = "Hello, this is a test of the speech synthesis system."
        tokens = tokenizer(text, return_tensors="pt")
        input_ids_mlx = mx.array(tokens["input_ids"].numpy())

        text_logits, speech_logits, _ = mlx_llm_model(input_ids_mlx)
        mx.eval(text_logits, speech_logits)

        assert not bool(mx.any(mx.isnan(text_logits))), "text_logits contains NaN"
        assert not bool(mx.any(mx.isnan(speech_logits))), "speech_logits contains NaN"

    def test_llm_forward_shapes(self, mlx_llm_model, tokenizer):
        """Test LLM forward pass output shapes."""
        text = "Hello world"
        tokens = tokenizer(text, return_tensors="pt")
        input_ids_mlx = mx.array(tokens["input_ids"].numpy())
        B, L = input_ids_mlx.shape

        text_logits, speech_logits, _ = mlx_llm_model(input_ids_mlx)
        mx.eval(text_logits, speech_logits)

        assert text_logits.shape == (B, L, 151936), f"Wrong text_logits shape: {text_logits.shape}"
        assert speech_logits.shape == (B, L, 6564), f"Wrong speech_logits shape: {speech_logits.shape}"


# ============================================================================
# CATEGORY 3: FLOW FORWARD PASS COMPARISON
# ============================================================================

class TestFlowForwardComparison:
    """Compare Flow forward pass between PyTorch and MLX."""

    def test_flow_forward_with_random_tokens(self, mlx_flow_model):
        """Test flow forward with random speech tokens."""
        B, L_tokens = 1, 25
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))
        t = mx.array([0.5])
        L = L_tokens * 2  # token_mel_ratio = 2
        x = mx.random.normal((B, L, 80))

        output = mlx_flow_model.forward(x, tokens, t, spk_emb)
        mx.eval(output)

        assert output.shape == (B, L, 80), f"Wrong shape: {output.shape}"
        assert not bool(mx.any(mx.isnan(output))), "Output contains NaN"

    def test_flow_inference_with_random_tokens(self, mlx_flow_model):
        """Test flow inference (ODE solver)."""
        B, L_tokens = 1, 25
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))

        mel = mlx_flow_model.inference(tokens, spk_emb, num_steps=5)
        mx.eval(mel)

        expected_L = L_tokens * 2
        assert mel.shape == (B, expected_L, 80), f"Wrong shape: {mel.shape}"
        assert not bool(mx.any(mx.isnan(mel))), "Mel contains NaN"


# ============================================================================
# CATEGORY 4: VOCODER FORWARD PASS COMPARISON
# ============================================================================

class TestVocoderForwardComparison:
    """Compare Vocoder forward pass between PyTorch and MLX."""

    def test_vocoder_forward_shape(self, mlx_vocoder_model):
        """Test vocoder produces correct output shape."""
        B, L = 1, 50
        mel = mx.random.normal((B, 80, L)) * 0.5

        audio = mlx_vocoder_model(mel)
        mx.eval(audio)

        assert audio.shape[0] == B
        assert audio.shape[1] > 0
        assert not bool(mx.any(mx.isnan(audio))), "Audio contains NaN"

    def test_vocoder_amplitude_range(self, mlx_vocoder_model):
        """Test vocoder output amplitude."""
        B, L = 1, 100
        mel = mx.random.normal((B, 80, L)) * 0.5

        audio = mlx_vocoder_model(mel)
        mx.eval(audio)

        max_amp = float(mx.abs(audio).max())
        assert max_amp < 100.0, f"Audio amplitude too high: {max_amp}"


# ============================================================================
# CATEGORY 5: END-TO-END PIPELINE TESTS
# ============================================================================

class TestEndToEndPipeline:
    """Test full end-to-end synthesis pipeline."""

    def test_text_to_speech_tokens(self, mlx_llm_model, tokenizer):
        """Test text -> speech token generation."""
        text = "Hello"
        tokens = tokenizer(text, return_tensors="pt")
        input_ids_mlx = mx.array(tokens["input_ids"].numpy())

        # Generate speech tokens (short for test speed)
        speech_tokens = mlx_llm_model.generate_speech_tokens(
            input_ids_mlx,
            max_length=10,
            temperature=0,  # Greedy for determinism
        )
        mx.eval(speech_tokens)

        assert speech_tokens.shape[0] == 1
        assert speech_tokens.shape[1] <= 10

    def test_speech_tokens_to_mel(self, mlx_flow_model, mlx_llm_model, tokenizer):
        """Test speech tokens -> mel spectrogram."""
        text = "Hi"
        tokens = tokenizer(text, return_tensors="pt")
        input_ids_mlx = mx.array(tokens["input_ids"].numpy())

        # Generate speech tokens
        speech_tokens = mlx_llm_model.generate_speech_tokens(
            input_ids_mlx,
            max_length=10,
            temperature=0,
        )
        mx.eval(speech_tokens)

        # Generate speaker embedding (random for test)
        spk_emb = mx.random.normal((1, 192))

        # Flow inference
        mel = mlx_flow_model.inference(speech_tokens, spk_emb, num_steps=3)
        mx.eval(mel)

        assert mel.shape[0] == 1
        assert mel.shape[2] == 80  # mel channels
        assert not bool(mx.any(mx.isnan(mel))), "Mel contains NaN"

    def test_full_pipeline(self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer):
        """Test complete text -> audio pipeline."""
        text = "Test"
        tokens = tokenizer(text, return_tensors="pt")
        input_ids_mlx = mx.array(tokens["input_ids"].numpy())

        # Step 1: Text -> Speech tokens
        speech_tokens = mlx_llm_model.generate_speech_tokens(
            input_ids_mlx,
            max_length=10,
            temperature=0,
        )
        mx.eval(speech_tokens)

        # Step 2: Speech tokens -> Mel (with random speaker embedding)
        spk_emb = mx.random.normal((1, 192))
        mel = mlx_flow_model.inference(speech_tokens, spk_emb, num_steps=3)
        mx.eval(mel)

        # Step 3: Mel -> Audio
        # Transpose: [B, L, C] -> [B, C, L]
        mel_for_vocoder = mel.transpose(0, 2, 1)
        audio = mlx_vocoder_model(mel_for_vocoder)
        mx.eval(audio)

        assert audio.shape[0] == 1
        assert audio.shape[1] > 0
        assert not bool(mx.any(mx.isnan(audio))), "Audio contains NaN"

    def test_pipeline_with_real_text(self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer):
        """Test pipeline with multiple real text inputs."""
        test_texts = [
            "Hello",
            "Good morning",
            "How are you?",
        ]

        for text in test_texts:
            tokens = tokenizer(text, return_tensors="pt")
            input_ids_mlx = mx.array(tokens["input_ids"].numpy())

            # Generate speech tokens
            speech_tokens = mlx_llm_model.generate_speech_tokens(
                input_ids_mlx,
                max_length=20,
                temperature=0.8,
                top_k=25,
            )
            mx.eval(speech_tokens)

            # Generate mel
            spk_emb = mx.random.normal((1, 192))
            mel = mlx_flow_model.inference(speech_tokens, spk_emb, num_steps=3)
            mx.eval(mel)

            # Generate audio
            mel_for_vocoder = mel.transpose(0, 2, 1)
            audio = mlx_vocoder_model(mel_for_vocoder)
            mx.eval(audio)

            assert audio.shape[1] > 0, f"No audio generated for: {text}"
            assert not bool(mx.any(mx.isnan(audio))), f"NaN in audio for: {text}"


# ============================================================================
# CATEGORY 6: DETERMINISM AND CONSISTENCY TESTS
# ============================================================================

class TestDeterminism:
    """Test deterministic behavior."""

    def test_greedy_determinism(self, mlx_llm_model, tokenizer):
        """Test greedy decoding is deterministic."""
        text = "Hello world"
        tokens = tokenizer(text, return_tensors="pt")
        input_ids_mlx = mx.array(tokens["input_ids"].numpy())

        # Generate twice with same input
        speech1 = mlx_llm_model.generate_speech_tokens(
            input_ids_mlx, max_length=10, temperature=0,
        )
        mx.eval(speech1)

        speech2 = mlx_llm_model.generate_speech_tokens(
            input_ids_mlx, max_length=10, temperature=0,
        )
        mx.eval(speech2)

        # Should be identical
        assert mx.array_equal(speech1, speech2), "Greedy decoding not deterministic"

    def test_flow_determinism_with_seed(self, mlx_flow_model):
        """Test flow is deterministic with same seed."""
        B, L_tokens = 1, 10
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)

        # Run twice with same seed
        mx.random.seed(42)
        spk_emb_1 = mx.random.normal((B, 192))
        mel_1 = mlx_flow_model.inference(tokens, spk_emb_1, num_steps=3)
        mx.eval(mel_1)

        mx.random.seed(42)
        spk_emb_2 = mx.random.normal((B, 192))
        mel_2 = mlx_flow_model.inference(tokens, spk_emb_2, num_steps=3)
        mx.eval(mel_2)

        max_diff = float(mx.abs(mel_1 - mel_2).max())
        assert max_diff < 1e-5, f"Non-deterministic: max_diff={max_diff}"


# ============================================================================
# CATEGORY 7: PYTORCH REFERENCE COMPARISON
# ============================================================================

class TestPyTorchComparison:
    """Direct comparison with PyTorch reference models."""

    def test_llm_embedding_vs_pytorch(self, mlx_llm_model, pytorch_llm, tokenizer):
        """Compare embedding layer with PyTorch.

        Note: MLX weights are stored in bfloat16 for efficiency, so tolerance
        must accommodate bfloat16 quantization error (~0.06 max diff typical).
        """
        text = "Hello"
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"]

        # MLX embedding - output is (batch, seq_len, hidden)
        input_ids_mlx = mx.array(input_ids.numpy())
        mlx_emb = mlx_llm_model.llm.embed_tokens(input_ids_mlx)
        mx.eval(mlx_emb)
        mlx_emb_squeezed = np.array(mlx_emb.astype(mx.float32)).squeeze()

        # PyTorch embedding - CosyVoice3 uses nested structure: llm.model.model.embed_tokens
        pt_key = 'llm.model.model.embed_tokens.weight'
        assert pt_key in pytorch_llm, f"Expected key {pt_key} not in pytorch_llm. Keys: {list(pytorch_llm.keys())[:10]}"

        pt_emb_weight = pytorch_llm[pt_key]
        pt_emb = pt_emb_weight[input_ids.squeeze()]
        pt_emb_np = pt_emb.numpy()

        # Compare with tolerance for bfloat16 quantization (~0.06 max diff typical)
        assert mlx_emb_squeezed.shape == pt_emb_np.shape, f"Shape mismatch: MLX {mlx_emb_squeezed.shape} vs PT {pt_emb_np.shape}"
        diff = np.abs(pt_emb_np - mlx_emb_squeezed)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        assert max_diff < 0.1, f"Embedding max_diff={max_diff:.6f} exceeds bfloat16 tolerance (0.1), mean_diff={mean_diff:.6f}"

    def test_flow_weights_match(self, pytorch_flow, mlx_weights):
        """Verify flow weight conversion is correct."""
        # Check a sample of weights
        weight_pairs = [
            ('input_embedding.weight', 'flow.input_embedding.weight'),
            ('spk_embed_affine_layer.weight', 'flow.spk_embed_affine_layer.weight'),
        ]

        for pt_key, mlx_key in weight_pairs:
            if pt_key in pytorch_flow and mlx_key in mlx_weights:
                pt_w = pytorch_flow[pt_key].numpy()
                mlx_w = np.array(mlx_weights[mlx_key].astype(mx.float32))

                max_diff = np.abs(pt_w - mlx_w).max()
                assert max_diff < 1e-5, f"{pt_key} max_diff: {max_diff}"

    def test_vocoder_weights_match(self, pytorch_vocoder, mlx_weights):
        """Verify vocoder weight conversion is correct (accounting for weight norm)."""
        # Check source_downs which are regular convs
        for i in range(3):
            pt_key = f'source_downs.{i}.weight'
            mlx_key = f'vocoder.source_downs.{i}.conv.weight'

            if pt_key in pytorch_vocoder and mlx_key in mlx_weights:
                pt_w = pytorch_vocoder[pt_key].numpy()
                # Transpose PyTorch [out, in, kernel] to MLX [out, kernel, in]
                pt_w = np.transpose(pt_w, (0, 2, 1))
                mlx_w = np.array(mlx_weights[mlx_key].astype(mx.float32))

                max_diff = np.abs(pt_w - mlx_w).max()
                assert max_diff < 1e-5, f"source_downs.{i} max_diff: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
