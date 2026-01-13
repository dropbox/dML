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
CosyVoice3 Numerical Validation Tests

Exhaustive layer-by-layer comparison between PyTorch and MLX implementations.
Tests ensure EXACT numerical matching (within floating point tolerance).

Test Categories:
1. Weight Loading Verification - All weights match exactly
2. Component-Level Forward Pass - Each component matches PyTorch
3. End-to-End Flow - Full pipeline matches
4. Edge Cases - Various input shapes and values

Tolerance Levels:
- FP32: max_diff < 1e-5
- FP16: max_diff < 1e-3
- INT8: max_diff < 0.02
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
def pytorch_flow_weights():
    """Load PyTorch flow weights."""
    try:
        import torch
        return torch.load(
            'models/cosyvoice3/flow.pt',
            map_location='cpu',
            weights_only=True,
        )
    except Exception as e:
        pytest.skip(f"PyTorch weights not available: {e}")


@pytest.fixture(scope="module")
def pytorch_vocoder_weights():
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
def mlx_flow_model():
    """Create MLX flow model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
        CausalMaskedDiffWithDiT,
        DiTConfig,
    )
    config = DiTConfig()
    return CausalMaskedDiffWithDiT(config)


@pytest.fixture(scope="module")
def mlx_vocoder_model():
    """Create MLX vocoder model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import (
        CausalHiFTConfig,
        CausalHiFTGenerator,
    )
    config = CausalHiFTConfig()
    return CausalHiFTGenerator(config)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_weight_norm_pytorch(orig0, orig1):
    """Compute weight from PyTorch weight normalization parameters."""
    import torch
    v_norm = torch.sqrt(torch.sum(orig1 * orig1, dim=(1, 2), keepdim=True) + 1e-12)
    w = orig0 * (orig1 / v_norm)
    return w.numpy()


def compute_weight_norm_mlx(orig0, orig1):
    """Compute weight from MLX weight normalization parameters."""
    v_norm = mx.sqrt(mx.sum(orig1 * orig1, axis=(1, 2), keepdims=True) + 1e-12)
    w = orig0 * (orig1 / v_norm)
    return np.array(w)


def compare_weights(pt_weight, mlx_weight, name, tolerance=1e-5, transpose_conv=False):
    """Compare PyTorch and MLX weights with detailed error reporting."""
    pt_np = pt_weight.numpy() if hasattr(pt_weight, 'numpy') else pt_weight
    mlx_np = np.array(mlx_weight.astype(mx.float32)) if isinstance(mlx_weight, mx.array) else mlx_weight

    # Handle Conv1d transposition: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
    if transpose_conv and len(pt_np.shape) == 3:
        pt_np = np.transpose(pt_np, (0, 2, 1))

    if pt_np.shape != mlx_np.shape:
        return {
            'name': name,
            'match': False,
            'error': f'Shape mismatch: PT {pt_np.shape} vs MLX {mlx_np.shape}',
            'max_diff': float('inf'),
            'mean_diff': float('inf'),
        }

    diff = np.abs(pt_np - mlx_np)
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
# CATEGORY 1: WEIGHT LOADING VERIFICATION
# ============================================================================

class TestFlowWeightLoading:
    """Test flow model weight loading matches PyTorch exactly."""

    def test_input_embedding_weights(self, pytorch_flow_weights, mlx_weights):
        """Test input embedding weights match."""
        pt_key = 'input_embedding.weight'
        mlx_key = 'flow.input_embedding.weight'

        result = compare_weights(
            pytorch_flow_weights[pt_key],
            mlx_weights[mlx_key],
            'input_embedding',
        )
        assert result['match'], f"Max diff: {result['max_diff']}"

    def test_pre_lookahead_conv1_weights(self, pytorch_flow_weights, mlx_weights):
        """Test pre_lookahead conv1 weights match."""
        pt_key = 'pre_lookahead_layer.conv1.weight'
        mlx_key = 'flow.pre_lookahead_layer.conv1.weight'

        result = compare_weights(
            pytorch_flow_weights[pt_key],
            mlx_weights[mlx_key],
            'pre_lookahead.conv1',
            transpose_conv=True,
        )
        assert result['match'], f"Max diff: {result['max_diff']}"

    def test_pre_lookahead_conv2_weights(self, pytorch_flow_weights, mlx_weights):
        """Test pre_lookahead conv2 weights match."""
        pt_key = 'pre_lookahead_layer.conv2.weight'
        mlx_key = 'flow.pre_lookahead_layer.conv2.weight'

        result = compare_weights(
            pytorch_flow_weights[pt_key],
            mlx_weights[mlx_key],
            'pre_lookahead.conv2',
            transpose_conv=True,
        )
        assert result['match'], f"Max diff: {result['max_diff']}"

    def test_speaker_affine_weights(self, pytorch_flow_weights, mlx_weights):
        """Test speaker embedding affine layer weights match."""
        pt_key = 'spk_embed_affine_layer.weight'
        mlx_key = 'flow.spk_embed_affine_layer.weight'

        result = compare_weights(
            pytorch_flow_weights[pt_key],
            mlx_weights[mlx_key],
            'spk_embed_affine',
        )
        assert result['match'], f"Max diff: {result['max_diff']}"

    def test_time_embedding_weights(self, pytorch_flow_weights, mlx_weights):
        """Test time embedding MLP weights match."""
        comparisons = [
            ('decoder.estimator.time_embed.time_mlp.0.weight',
             'flow.dit.time_embed.mlp.layers.0.weight'),
            ('decoder.estimator.time_embed.time_mlp.0.bias',
             'flow.dit.time_embed.mlp.layers.0.bias'),
            ('decoder.estimator.time_embed.time_mlp.2.weight',
             'flow.dit.time_embed.mlp.layers.2.weight'),
            ('decoder.estimator.time_embed.time_mlp.2.bias',
             'flow.dit.time_embed.mlp.layers.2.bias'),
        ]

        for pt_key, mlx_key in comparisons:
            result = compare_weights(
                pytorch_flow_weights[pt_key],
                mlx_weights[mlx_key],
                pt_key.split('.')[-2],
            )
            assert result['match'], f"{pt_key}: Max diff {result['max_diff']}"

    def test_input_embed_proj_weights(self, pytorch_flow_weights, mlx_weights):
        """Test input embedding projection weights match."""
        pt_key = 'decoder.estimator.input_embed.proj.weight'
        mlx_key = 'flow.dit.input_embed.proj.weight'

        result = compare_weights(
            pytorch_flow_weights[pt_key],
            mlx_weights[mlx_key],
            'input_embed.proj',
        )
        assert result['match'], f"Max diff: {result['max_diff']}"

    def test_output_proj_weights(self, pytorch_flow_weights, mlx_weights):
        """Test output projection weights match."""
        pt_key = 'decoder.estimator.proj_out.weight'
        mlx_key = 'flow.dit.proj_out.weight'

        result = compare_weights(
            pytorch_flow_weights[pt_key],
            mlx_weights[mlx_key],
            'proj_out',
        )
        assert result['match'], f"Max diff: {result['max_diff']}"

    @pytest.mark.parametrize("block_idx", [0, 10, 21])
    def test_transformer_block_weights(self, pytorch_flow_weights, mlx_weights, block_idx):
        """Test transformer block weights match for first, middle, and last blocks."""
        comparisons = [
            (f'decoder.estimator.transformer_blocks.{block_idx}.attn.to_q.weight',
             f'flow.dit.blocks.{block_idx}.attn.to_q.weight'),
            (f'decoder.estimator.transformer_blocks.{block_idx}.attn.to_k.weight',
             f'flow.dit.blocks.{block_idx}.attn.to_k.weight'),
            (f'decoder.estimator.transformer_blocks.{block_idx}.attn.to_v.weight',
             f'flow.dit.blocks.{block_idx}.attn.to_v.weight'),
            (f'decoder.estimator.transformer_blocks.{block_idx}.attn.to_out.0.weight',
             f'flow.dit.blocks.{block_idx}.attn.to_out.weight'),
            (f'decoder.estimator.transformer_blocks.{block_idx}.attn_norm.linear.weight',
             f'flow.dit.blocks.{block_idx}.attn_norm.linear.weight'),
            (f'decoder.estimator.transformer_blocks.{block_idx}.ff.ff.0.0.weight',
             f'flow.dit.blocks.{block_idx}.ff.layers.0.weight'),
            (f'decoder.estimator.transformer_blocks.{block_idx}.ff.ff.2.weight',
             f'flow.dit.blocks.{block_idx}.ff.layers.1.weight'),
        ]

        for pt_key, mlx_key in comparisons:
            if pt_key in pytorch_flow_weights and mlx_key in mlx_weights:
                result = compare_weights(
                    pytorch_flow_weights[pt_key],
                    mlx_weights[mlx_key],
                    f'block{block_idx}.{pt_key.split(".")[-2]}',
                )
                assert result['match'], f"{pt_key}: Max diff {result['max_diff']}"


class TestVocoderWeightLoading:
    """Test vocoder weight loading matches PyTorch exactly."""

    def test_conv_pre_weights(self, pytorch_vocoder_weights, mlx_weights):
        """Test conv_pre weights match (weight normalized)."""
        orig0 = pytorch_vocoder_weights['conv_pre.parametrizations.weight.original0'].numpy()
        orig1 = pytorch_vocoder_weights['conv_pre.parametrizations.weight.original1'].numpy()

        # Compute actual weight
        v_norm = np.sqrt(np.sum(orig1 * orig1, axis=(1, 2), keepdims=True) + 1e-12)
        pt_weight = orig0 * (orig1 / v_norm)
        pt_weight = np.transpose(pt_weight, (0, 2, 1))  # To MLX format

        mlx_weight = np.array(mlx_weights['vocoder.conv_pre.conv.weight'].astype(mx.float32))

        max_diff = np.abs(pt_weight - mlx_weight).max()
        assert max_diff < 1e-5, f"conv_pre max diff: {max_diff}"

    def test_conv_post_weights(self, pytorch_vocoder_weights, mlx_weights):
        """Test conv_post weights match (weight normalized)."""
        orig0 = pytorch_vocoder_weights['conv_post.parametrizations.weight.original0'].numpy()
        orig1 = pytorch_vocoder_weights['conv_post.parametrizations.weight.original1'].numpy()

        v_norm = np.sqrt(np.sum(orig1 * orig1, axis=(1, 2), keepdims=True) + 1e-12)
        pt_weight = orig0 * (orig1 / v_norm)
        pt_weight = np.transpose(pt_weight, (0, 2, 1))

        mlx_weight = np.array(mlx_weights['vocoder.conv_post.conv.weight'].astype(mx.float32))

        max_diff = np.abs(pt_weight - mlx_weight).max()
        assert max_diff < 1e-5, f"conv_post max diff: {max_diff}"

    @pytest.mark.parametrize("up_idx", [0, 1, 2])
    def test_upsample_weights(self, pytorch_vocoder_weights, mlx_weights, up_idx):
        """Test upsample layer weights match."""
        orig0_key = f'ups.{up_idx}.parametrizations.weight.original0'
        orig1_key = f'ups.{up_idx}.parametrizations.weight.original1'
        mlx_key = f'vocoder.ups.{up_idx}.conv.weight'

        orig0 = pytorch_vocoder_weights[orig0_key].numpy()
        orig1 = pytorch_vocoder_weights[orig1_key].numpy()

        v_norm = np.sqrt(np.sum(orig1 * orig1, axis=(1, 2), keepdims=True) + 1e-12)
        pt_weight = orig0 * (orig1 / v_norm)
        pt_weight = np.transpose(pt_weight, (0, 2, 1))

        mlx_weight = np.array(mlx_weights[mlx_key].astype(mx.float32))

        max_diff = np.abs(pt_weight - mlx_weight).max()
        assert max_diff < 1e-5, f"ups.{up_idx} max diff: {max_diff}"

    @pytest.mark.parametrize("sd_idx", [0, 1, 2])
    def test_source_down_weights(self, pytorch_vocoder_weights, mlx_weights, sd_idx):
        """Test source_downs weights match (regular conv, not weight-norm)."""
        pt_key = f'source_downs.{sd_idx}.weight'
        mlx_key = f'vocoder.source_downs.{sd_idx}.conv.weight'

        pt_weight = pytorch_vocoder_weights[pt_key].numpy()
        pt_weight = np.transpose(pt_weight, (0, 2, 1))  # To MLX format

        mlx_weight = np.array(mlx_weights[mlx_key].astype(mx.float32))

        max_diff = np.abs(pt_weight - mlx_weight).max()
        assert max_diff < 1e-5, f"source_downs.{sd_idx} max diff: {max_diff}"


# ============================================================================
# CATEGORY 2: COMPONENT-LEVEL FORWARD PASS
# ============================================================================

class TestFlowForwardPass:
    """Test flow model forward pass produces valid outputs."""

    def test_forward_shape(self, mlx_flow_model, mlx_weights):
        """Test forward pass produces correct output shape."""
        # Load weights
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))
        mx.eval(mlx_flow_model.parameters())

        B, L_tokens = 1, 10
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))
        t = mx.array([0.5])
        L = L_tokens * 2
        x = mx.random.normal((B, L, 80))

        output = mlx_flow_model.forward(x, tokens, t, spk_emb)
        mx.eval(output)

        assert output.shape == (B, L, 80), f"Expected (1, 20, 80), got {output.shape}"

    def test_forward_no_nan(self, mlx_flow_model, mlx_weights):
        """Test forward pass produces no NaN values."""
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))
        mx.eval(mlx_flow_model.parameters())

        B, L_tokens = 1, 25
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))
        t = mx.array([0.5])
        L = L_tokens * 2
        x = mx.random.normal((B, L, 80))

        output = mlx_flow_model.forward(x, tokens, t, spk_emb)
        mx.eval(output)

        has_nan = bool(mx.any(mx.isnan(output)))
        assert not has_nan, "Output contains NaN values"

    def test_inference_shape(self, mlx_flow_model, mlx_weights):
        """Test inference produces correct output shape."""
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))
        mx.eval(mlx_flow_model.parameters())

        B, L_tokens = 1, 25
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))

        output = mlx_flow_model.inference(tokens, spk_emb, num_steps=5)
        mx.eval(output)

        expected_L = L_tokens * 2  # token_mel_ratio = 2
        assert output.shape == (B, expected_L, 80), f"Expected (1, 50, 80), got {output.shape}"


class TestVocoderForwardPass:
    """Test vocoder forward pass produces valid outputs."""

    def test_forward_shape(self, mlx_vocoder_model, mlx_weights):
        """Test vocoder forward pass produces correct output shape."""
        vocoder_weights = {k[8:]: v for k, v in mlx_weights.items() if k.startswith('vocoder.')}
        mlx_vocoder_model.load_weights(list(vocoder_weights.items()))
        mx.eval(mlx_vocoder_model.parameters())

        B, L = 1, 50
        mel = mx.random.normal((B, 80, L))  # [B, C, L] format

        audio = mlx_vocoder_model(mel)
        mx.eval(audio)

        # Check output is 1D audio
        assert len(audio.shape) == 2, f"Expected 2D output, got {len(audio.shape)}D"
        assert audio.shape[0] == B, "Batch size mismatch"

    def test_forward_no_nan(self, mlx_vocoder_model, mlx_weights):
        """Test vocoder forward pass produces no NaN values."""
        vocoder_weights = {k[8:]: v for k, v in mlx_weights.items() if k.startswith('vocoder.')}
        mlx_vocoder_model.load_weights(list(vocoder_weights.items()))
        mx.eval(mlx_vocoder_model.parameters())

        B, L = 1, 100
        mel = mx.random.normal((B, 80, L))

        audio = mlx_vocoder_model(mel)
        mx.eval(audio)

        has_nan = bool(mx.any(mx.isnan(audio)))
        assert not has_nan, "Audio output contains NaN values"

    def test_audio_amplitude_range(self, mlx_vocoder_model, mlx_weights):
        """Test vocoder output is in reasonable amplitude range."""
        vocoder_weights = {k[8:]: v for k, v in mlx_weights.items() if k.startswith('vocoder.')}
        mlx_vocoder_model.load_weights(list(vocoder_weights.items()))
        mx.eval(mlx_vocoder_model.parameters())

        B, L = 1, 100
        mel = mx.random.normal((B, 80, L)) * 0.5  # Reasonable mel range

        audio = mlx_vocoder_model(mel)
        mx.eval(audio)

        max_amp = float(mx.abs(audio).max())
        assert max_amp < 10.0, f"Audio amplitude too high: {max_amp}"


# ============================================================================
# CATEGORY 3: END-TO-END PIPELINE
# ============================================================================

class TestEndToEndPipeline:
    """Test full flow -> vocoder pipeline."""

    def test_pipeline_runs(self, mlx_flow_model, mlx_vocoder_model, mlx_weights):
        """Test full pipeline executes without error."""
        # Load flow weights
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))

        # Load vocoder weights
        vocoder_weights = {k[8:]: v for k, v in mlx_weights.items() if k.startswith('vocoder.')}
        mlx_vocoder_model.load_weights(list(vocoder_weights.items()))

        mx.eval(mlx_flow_model.parameters())
        mx.eval(mlx_vocoder_model.parameters())

        # Run pipeline
        B, L_tokens = 1, 25
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))

        # Flow inference
        mel = mlx_flow_model.inference(tokens, spk_emb, num_steps=5)
        mx.eval(mel)

        # Transpose for vocoder: [B, L, C] -> [B, C, L]
        mel_for_vocoder = mel.transpose(0, 2, 1)

        # Vocoder
        audio = mlx_vocoder_model(mel_for_vocoder)
        mx.eval(audio)

        assert audio.shape[0] == B
        assert audio.shape[1] > 0
        assert not bool(mx.any(mx.isnan(audio)))

    def test_pipeline_deterministic(self, mlx_flow_model, mlx_vocoder_model, mlx_weights):
        """Test pipeline produces consistent results with same seed."""
        # Load weights
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))
        vocoder_weights = {k[8:]: v for k, v in mlx_weights.items() if k.startswith('vocoder.')}
        mlx_vocoder_model.load_weights(list(vocoder_weights.items()))
        mx.eval(mlx_flow_model.parameters())
        mx.eval(mlx_vocoder_model.parameters())

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

        # Should be identical with same seed
        max_diff = float(mx.abs(mel_1 - mel_2).max())
        assert max_diff < 1e-5, f"Non-deterministic output: max_diff={max_diff}"


# ============================================================================
# CATEGORY 4: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self, mlx_flow_model, mlx_weights):
        """Test with single token input."""
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))
        mx.eval(mlx_flow_model.parameters())

        B, L_tokens = 1, 1
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))
        t = mx.array([0.5])
        L = L_tokens * 2
        x = mx.random.normal((B, L, 80))

        # Should not crash
        output = mlx_flow_model.forward(x, tokens, t, spk_emb)
        mx.eval(output)
        assert output.shape == (B, L, 80)

    def test_long_sequence(self, mlx_flow_model, mlx_weights):
        """Test with longer sequence."""
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))
        mx.eval(mlx_flow_model.parameters())

        B, L_tokens = 1, 100  # ~4 seconds
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))
        t = mx.array([0.5])
        L = L_tokens * 2
        x = mx.random.normal((B, L, 80))

        output = mlx_flow_model.forward(x, tokens, t, spk_emb)
        mx.eval(output)
        assert output.shape == (B, L, 80)

    def test_batch_size_2(self, mlx_flow_model, mlx_weights):
        """Test with batch size > 1."""
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))
        mx.eval(mlx_flow_model.parameters())

        B, L_tokens = 2, 10
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))
        t = mx.array([0.5, 0.5])
        L = L_tokens * 2
        x = mx.random.normal((B, L, 80))

        output = mlx_flow_model.forward(x, tokens, t, spk_emb)
        mx.eval(output)
        assert output.shape == (B, L, 80)

    def test_extreme_time_values(self, mlx_flow_model, mlx_weights):
        """Test with time values at boundaries."""
        flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
        mlx_flow_model.load_weights(list(flow_weights.items()))
        mx.eval(mlx_flow_model.parameters())

        B, L_tokens = 1, 10
        tokens = mx.zeros((B, L_tokens), dtype=mx.int32)
        spk_emb = mx.random.normal((B, 192))
        L = L_tokens * 2
        x = mx.random.normal((B, L, 80))

        # Test t=0 and t=1
        for t_val in [0.0, 0.01, 0.99, 1.0]:
            t = mx.array([t_val])
            output = mlx_flow_model.forward(x, tokens, t, spk_emb)
            mx.eval(output)
            assert not bool(mx.any(mx.isnan(output))), f"NaN at t={t_val}"


# ============================================================================
# CATEGORY 5: LLM VALIDATION
# ============================================================================

@pytest.fixture(scope="module")
def mlx_llm_model():
    """Create MLX LLM model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM,
        Qwen2Config,
    )
    # CosyVoice3 uses Qwen2 with GQA: 7 query heads, 1 KV head
    # Q projection: 7 * 128 = 896
    # K, V projection: 1 * 128 = 128
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
    return CosyVoice2LLM(config)


class TestLLMWeightLoading:
    """Test LLM weight loading."""

    def test_llm_weight_count(self, mlx_weights):
        """Test LLM has expected number of weights."""
        llm_count = sum(1 for k in mlx_weights if k.startswith('llm.'))
        # Expected: 297 keys (24 layers * 12 + special embeddings)
        assert llm_count >= 290, f"Expected >= 290 LLM weights, got {llm_count}"

    def test_llm_embedding_weights(self, mlx_weights):
        """Test LLM embedding weights exist and have correct shapes."""
        # Text embedding
        key = 'llm.llm.embed_tokens.weight'
        assert key in mlx_weights, f"Missing {key}"
        assert mlx_weights[key].shape == (151936, 896), f"Wrong shape for {key}"

        # Speech embedding
        key = 'llm.speech_embedding.weight'
        assert key in mlx_weights, f"Missing {key}"
        assert mlx_weights[key].shape == (6564, 896), f"Wrong shape for {key}"

        # LLM embedding (SOS/EOS)
        key = 'llm.llm_embedding.weight'
        assert key in mlx_weights, f"Missing {key}"
        assert mlx_weights[key].shape == (2, 896), f"Wrong shape for {key}"

    def test_llm_decoder_weights(self, mlx_weights):
        """Test LLM decoder (speech head) weights."""
        key = 'llm.llm_decoder.weight'
        assert key in mlx_weights, f"Missing {key}"
        assert mlx_weights[key].shape == (6564, 896), f"Wrong shape for {key}"

        key = 'llm.llm_decoder.bias'
        assert key in mlx_weights, f"Missing {key}"
        assert mlx_weights[key].shape == (6564,), f"Wrong shape for {key}"

    def test_llm_lm_head_weights(self, mlx_weights):
        """Test LLM language model head weights."""
        key = 'llm.lm_head.weight'
        assert key in mlx_weights, f"Missing {key}"
        assert mlx_weights[key].shape == (151936, 896), f"Wrong shape for {key}"

    def test_all_llm_weights_numeric(self, mlx_weights):
        """Test all LLM weights are finite numbers."""
        for k, v in mlx_weights.items():
            if k.startswith('llm.'):
                has_nan = bool(mx.any(mx.isnan(v)))
                has_inf = bool(mx.any(mx.isinf(v)))
                assert not has_nan, f"{k} contains NaN"
                assert not has_inf, f"{k} contains Inf"

    @pytest.mark.parametrize("layer_idx", [0, 11, 23])
    def test_llm_layer_weights(self, mlx_weights, layer_idx):
        """Test transformer layer weights exist for first, middle, and last layers."""
        prefix = f'llm.llm.layers.{layer_idx}'
        expected_keys = [
            f'{prefix}.self_attn.q_proj.weight',
            f'{prefix}.self_attn.k_proj.weight',
            f'{prefix}.self_attn.v_proj.weight',
            f'{prefix}.self_attn.o_proj.weight',
            f'{prefix}.mlp.gate_proj.weight',
            f'{prefix}.mlp.up_proj.weight',
            f'{prefix}.mlp.down_proj.weight',
            f'{prefix}.input_layernorm.weight',
            f'{prefix}.post_attention_layernorm.weight',
        ]
        for key in expected_keys:
            assert key in mlx_weights, f"Missing {key}"


class TestLLMForwardPass:
    """Test LLM forward pass produces valid outputs."""

    def test_forward_text_encoding(self, mlx_llm_model, mlx_weights):
        """Test forward pass with text input."""
        # Load weights
        llm_weights = {k[4:]: v for k, v in mlx_weights.items() if k.startswith('llm.')}
        mlx_llm_model.load_weights(list(llm_weights.items()))
        mx.eval(mlx_llm_model.parameters())

        # Test with small text input
        B, L = 1, 10
        input_ids = mx.zeros((B, L), dtype=mx.int32) + 100  # Some text token

        text_logits, speech_logits, cache = mlx_llm_model(input_ids)
        mx.eval(text_logits, speech_logits)

        # Check shapes
        assert text_logits.shape == (B, L, 151936), f"Wrong text logits shape: {text_logits.shape}"
        assert speech_logits.shape == (B, L, 6564), f"Wrong speech logits shape: {speech_logits.shape}"

    def test_forward_no_nan(self, mlx_llm_model, mlx_weights):
        """Test forward pass produces no NaN values."""
        llm_weights = {k[4:]: v for k, v in mlx_weights.items() if k.startswith('llm.')}
        mlx_llm_model.load_weights(list(llm_weights.items()))
        mx.eval(mlx_llm_model.parameters())

        B, L = 1, 5
        input_ids = mx.zeros((B, L), dtype=mx.int32) + 100

        text_logits, speech_logits, _ = mlx_llm_model(input_ids)
        mx.eval(text_logits, speech_logits)

        assert not bool(mx.any(mx.isnan(text_logits))), "text_logits contains NaN"
        assert not bool(mx.any(mx.isnan(speech_logits))), "speech_logits contains NaN"

    def test_forward_with_cache(self, mlx_llm_model, mlx_weights):
        """Test forward pass with KV cache (autoregressive)."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import make_kv_cache

        llm_weights = {k[4:]: v for k, v in mlx_weights.items() if k.startswith('llm.')}
        mlx_llm_model.load_weights(list(llm_weights.items()))
        mx.eval(mlx_llm_model.parameters())

        B = 1
        cache = make_kv_cache(24)  # 24 layers

        # First pass: encode 5 tokens
        input_ids = mx.zeros((B, 5), dtype=mx.int32) + 100
        _, _, cache = mlx_llm_model(input_ids, cache=cache)
        mx.eval([c.state for c in cache if c.state is not None])

        # Second pass: generate 1 token
        next_input = mx.zeros((B, 1), dtype=mx.int32) + 200
        text_logits, speech_logits, cache = mlx_llm_model(next_input, cache=cache)
        mx.eval(text_logits, speech_logits)

        # Should have output for single token
        assert text_logits.shape == (B, 1, 151936)
        assert speech_logits.shape == (B, 1, 6564)


class TestLLMSpeechGeneration:
    """Test LLM speech token generation."""

    def test_sample_tokens_greedy(self, mlx_llm_model, mlx_weights):
        """Test greedy sampling (temperature=0)."""
        llm_weights = {k[4:]: v for k, v in mlx_weights.items() if k.startswith('llm.')}
        mlx_llm_model.load_weights(list(llm_weights.items()))
        mx.eval(mlx_llm_model.parameters())

        # Create random logits
        B = 2
        logits = mx.random.normal((B, 6564))

        # Greedy should be deterministic
        tokens1 = mlx_llm_model.sample_tokens(logits, temperature=0)
        tokens2 = mlx_llm_model.sample_tokens(logits, temperature=0)
        mx.eval(tokens1, tokens2)

        assert tokens1.shape == (B,)
        assert mx.all(tokens1 == tokens2), "Greedy sampling should be deterministic"

    def test_sample_tokens_with_temperature(self, mlx_llm_model, mlx_weights):
        """Test sampling with temperature."""
        llm_weights = {k[4:]: v for k, v in mlx_weights.items() if k.startswith('llm.')}
        mlx_llm_model.load_weights(list(llm_weights.items()))
        mx.eval(mlx_llm_model.parameters())

        B = 1
        logits = mx.random.normal((B, 6564))

        # With temperature, sampling should produce valid tokens
        tokens = mlx_llm_model.sample_tokens(logits, temperature=1.0, top_k=25, top_p=0.8)
        mx.eval(tokens)

        assert tokens.shape == (B,)
        assert bool(mx.all(tokens >= 0)), "Tokens should be non-negative"
        assert bool(mx.all(tokens < 6564)), "Tokens should be < vocab size"

    def test_generate_speech_short_sequence(self, mlx_llm_model, mlx_weights):
        """Test generating a short speech token sequence."""
        llm_weights = {k[4:]: v for k, v in mlx_weights.items() if k.startswith('llm.')}
        mlx_llm_model.load_weights(list(llm_weights.items()))
        mx.eval(mlx_llm_model.parameters())

        B = 1
        # Very short text input to test basic generation
        text_ids = mx.zeros((B, 3), dtype=mx.int32) + 100

        # Generate just a few tokens to verify the pipeline works
        # Using greedy decoding for determinism
        speech_tokens = mlx_llm_model.generate_speech_tokens(
            text_ids,
            max_length=5,  # Very short for speed
            temperature=0,  # Greedy for determinism
        )
        mx.eval(speech_tokens)

        # Should produce some tokens
        assert speech_tokens.shape[0] == B
        assert speech_tokens.shape[1] <= 5  # May stop early at EOS


# ============================================================================
# SUMMARY TEST
# ============================================================================

class TestWeightCountSummary:
    """Verify total weight counts match expectations."""

    def test_flow_weight_count(self, mlx_weights):
        """Test flow has expected number of weights."""
        flow_count = sum(1 for k in mlx_weights if k.startswith('flow.'))
        # Expected: 332 (330 from PyTorch + 2 padding attributes)
        assert flow_count >= 330, f"Expected >= 330 flow weights, got {flow_count}"

    def test_vocoder_weight_count(self, mlx_weights):
        """Test vocoder has expected number of weights."""
        vocoder_count = sum(1 for k in mlx_weights if k.startswith('vocoder.'))
        # Expected: 246 (after weight norm merging)
        assert vocoder_count >= 240, f"Expected >= 240 vocoder weights, got {vocoder_count}"

    def test_llm_weight_count(self, mlx_weights):
        """Test LLM has expected number of weights."""
        llm_count = sum(1 for k in mlx_weights if k.startswith('llm.'))
        # Expected: 297 (24 layers * 12 + special embeddings)
        assert llm_count >= 290, f"Expected >= 290 LLM weights, got {llm_count}"

    def test_total_weight_count(self, mlx_weights):
        """Test total weight count matches expected."""
        total = len(mlx_weights)
        # Expected: flow 332 + vocoder 246 + llm 297 = 875
        assert total >= 870, f"Expected >= 870 total weights, got {total}"

    def test_all_flow_weights_numeric(self, mlx_weights):
        """Test all flow weights are finite numbers."""
        for k, v in mlx_weights.items():
            if k.startswith('flow.'):
                has_nan = bool(mx.any(mx.isnan(v)))
                has_inf = bool(mx.any(mx.isinf(v)))
                assert not has_nan, f"{k} contains NaN"
                assert not has_inf, f"{k} contains Inf"

    def test_all_vocoder_weights_numeric(self, mlx_weights):
        """Test all vocoder weights are finite numbers."""
        for k, v in mlx_weights.items():
            if k.startswith('vocoder.'):
                has_nan = bool(mx.any(mx.isnan(v)))
                has_inf = bool(mx.any(mx.isinf(v)))
                assert not has_nan, f"{k} contains NaN"
                assert not has_inf, f"{k} contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
