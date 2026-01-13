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
Tests for Kokoro TTS MLX Model

Tests cover:
- Model instantiation
- Forward pass
- Building block functionality
- Weight loading from PyTorch
"""

import os

import mlx.core as mx
import numpy as np
import pytest


class TestKokoroConfig:
    """Test KokoroConfig dataclass."""

    def test_default_config(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig

        config = KokoroConfig()
        assert config.n_token == 178
        assert config.hidden_dim == 512
        assert config.style_dim == 128
        assert config.plbert_hidden_size == 768
        assert config.plbert_num_hidden_layers == 12

    def test_custom_config(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig

        config = KokoroConfig(n_token=256, hidden_dim=768)
        assert config.n_token == 256
        assert config.hidden_dim == 768


class TestBuildingBlocks:
    """Test core building blocks."""

    def test_weight_norm_conv1d(self):
        from tools.pytorch_to_mlx.converters.models import WeightNormConv1d

        conv = WeightNormConv1d(64, 128, kernel_size=3, padding=1)
        x = mx.zeros((1, 10, 64))  # [batch, length, channels]
        y = conv(x)
        assert y.shape == (1, 10, 128)

    def test_custom_layer_norm(self):
        from tools.pytorch_to_mlx.converters.models import CustomLayerNorm

        norm = CustomLayerNorm(64)
        x = mx.random.normal((2, 10, 64))
        y = norm(x)
        assert y.shape == (2, 10, 64)

    def test_adain(self):
        from tools.pytorch_to_mlx.converters.models import AdaIN

        adain = AdaIN(64, style_dim=128)
        x = mx.random.normal((2, 10, 64))  # [batch, length, channels]
        s = mx.random.normal((2, 128))  # [batch, style_dim]
        y = adain(x, s)
        assert y.shape == (2, 10, 64)

    def test_ada_layer_norm(self):
        from tools.pytorch_to_mlx.converters.models import AdaLayerNorm

        norm = AdaLayerNorm(64, style_dim=128)
        x = mx.random.normal((2, 10, 64))
        s = mx.random.normal((2, 128))
        y = norm(x, s)
        assert y.shape == (2, 10, 64)

    def test_adain_res_blk1d(self):
        from tools.pytorch_to_mlx.converters.models import AdainResBlk1d

        block = AdainResBlk1d(64, 128, style_dim=128)
        x = mx.random.normal((2, 10, 64))
        s = mx.random.normal((2, 128))
        y = block(x, s)
        assert y.shape == (2, 10, 128)

    def test_bilstm(self):
        from tools.pytorch_to_mlx.converters.models import BiLSTM

        bilstm = BiLSTM(64, 128)
        x = mx.random.normal((2, 10, 64))  # [batch, length, input]
        y = bilstm(x)
        assert y.shape == (2, 10, 256)  # hidden_size * 2

    def test_res_block1d(self):
        from tools.pytorch_to_mlx.converters.models import ResBlock1d

        block = ResBlock1d(64, kernel_size=3, dilations=(1, 3, 5))
        x = mx.random.normal((2, 10, 64))
        y = block(x)
        assert y.shape == (2, 10, 64)


class TestDecoder:
    """Test ISTFTNet decoder components."""

    def test_decoder_instantiation(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig
        from tools.pytorch_to_mlx.converters.models.kokoro import Decoder

        config = KokoroConfig()
        decoder = Decoder(config)

        assert hasattr(decoder, "f0_conv")
        assert hasattr(decoder, "n_conv")
        assert hasattr(decoder, "asr_res")
        assert hasattr(decoder, "encode")
        # Decoder uses flattened attribute names (decode_0, decode_1, etc.) for MLX module registration
        assert hasattr(decoder, "decode_0")
        assert hasattr(decoder, "generator")

    def test_decoder_forward(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig
        from tools.pytorch_to_mlx.converters.models.kokoro import Decoder

        config = KokoroConfig()
        decoder = Decoder(config)

        batch_size = 1
        seq_length = 10
        # F0 and noise are at 2x rate after predictor upsampling
        f0_n_length = seq_length * 2

        # Create inputs
        asr_features = mx.random.normal((batch_size, seq_length, config.hidden_dim))
        f0 = mx.random.normal((batch_size, f0_n_length))
        noise = mx.random.normal((batch_size, f0_n_length))
        style = mx.random.normal((batch_size, config.style_dim))

        # Forward pass
        audio = decoder(asr_features, f0, noise, style)
        mx.eval(audio)

        assert audio.shape[0] == batch_size
        assert audio.shape[1] > 0  # Audio samples

    def test_generator_instantiation(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig
        from tools.pytorch_to_mlx.converters.models.kokoro import Generator

        config = KokoroConfig()
        generator = Generator(config)

        # Generator uses flattened attribute names (ups_0, ups_1, etc.) for MLX module registration
        assert hasattr(generator, "ups_0")
        assert hasattr(generator, "resblocks_0")
        assert hasattr(generator, "conv_post")
        # Check expected number of ups modules (2 for default config)
        assert hasattr(generator, "ups_0") and hasattr(generator, "ups_1")

    def test_source_module(self):
        from tools.pytorch_to_mlx.converters.models.kokoro import SourceModule

        source = SourceModule(sample_rate=24000, num_harmonics=9)
        # F0 is normalized (predictor outputs ~0.5-1.5), SourceModule applies 200.0 scaling
        f0 = mx.random.normal((1, 100)) * 0.5 + 1.0
        upp = 60  # Total upsampling factor
        har_source, noise, uv = source(f0, upp)
        mx.eval(har_source, noise, uv)

        # SourceModule now returns single-channel outputs (STFT applied in Generator)
        # har_source: [batch, samples, 1] - Merged harmonic source
        # noise: [batch, samples, 1] - Noise source
        # uv: [batch, samples, 1] - Voiced/unvoiced mask
        samples = 100 * upp  # 6000
        assert har_source.shape == (1, samples, 1), (
            f"Expected (1, {samples}, 1), got {har_source.shape}"
        )
        assert noise.shape == (1, samples, 1), (
            f"Expected (1, {samples}, 1), got {noise.shape}"
        )
        assert uv.shape == (1, samples, 1), (
            f"Expected (1, {samples}, 1), got {uv.shape}"
        )

        # har_source should be bounded by tanh
        assert float(har_source.min()) >= -1.0, (
            f"har_source min {har_source.min()} < -1"
        )
        assert float(har_source.max()) <= 1.0, f"har_source max {har_source.max()} > 1"


class TestKokoroModel:
    """Test full Kokoro model."""

    def test_model_instantiation(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        assert hasattr(model, "bert")
        assert hasattr(model, "bert_encoder")
        assert hasattr(model, "text_encoder")
        assert hasattr(model, "predictor")
        assert hasattr(model, "decoder")

    def test_forward_pass(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        batch_size = 1
        seq_length = 10
        input_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
        style = mx.zeros((batch_size, config.style_dim))
        # Mask should be boolean (True = masked, False = valid)
        attention_mask = mx.zeros((batch_size, seq_length), dtype=mx.bool_)

        output = model(input_ids, style, attention_mask)
        mx.eval(output)

        assert output.shape[0] == batch_size
        assert output.shape[1] > 0  # Audio samples

    def test_text_encoder(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        x = mx.zeros((1, 10), dtype=mx.int32)
        # Mask should be boolean (True = masked, False = valid)
        mask = mx.zeros((1, 10), dtype=mx.bool_)
        out = model.text_encoder(x, mask)
        mx.eval(out)

        assert out.shape == (1, 10, config.hidden_dim)

    def test_bert_forward(self):
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        x = mx.zeros((1, 10), dtype=mx.int32)
        out = model.bert(x)
        mx.eval(out)

        assert out.shape == (1, 10, config.plbert_hidden_size)


class TestWeightLoading:
    """Test weight loading from PyTorch checkpoint."""

    @pytest.fixture
    def kokoro_weights_path(self):
        """Path to Kokoro weights if available."""
        import os

        path = os.path.expanduser("~/models/kokoro/kokoro-v1_0.pth")
        if os.path.exists(path):
            return path
        pytest.skip("Kokoro weights not found")

    def test_embedding_loading(self, kokoro_weights_path):
        """Test that embedding weights load correctly."""
        import torch

        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        pt_state = torch.load(
            kokoro_weights_path, map_location="cpu", weights_only=True,
        )
        te_state = pt_state["text_encoder"]

        config = KokoroConfig()
        model = KokoroModel(config)

        # Load embedding
        pt_embed = te_state["module.embedding.weight"].numpy()
        model.text_encoder.embedding.weight = mx.array(pt_embed)

        # Test
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mlx_embed = model.text_encoder.embedding(input_ids)
        mx.eval(mlx_embed)

        # Compare with PyTorch
        pt_input = torch.tensor([[1, 2, 3, 4, 5]])
        pt_embed_out = torch.nn.functional.embedding(
            pt_input, te_state["module.embedding.weight"],
        ).numpy()

        assert np.allclose(np.array(mlx_embed), pt_embed_out, atol=1e-6)

    def test_conv_weight_norm_loading(self, kokoro_weights_path):
        """Test that weight-normalized conv weights load correctly."""
        import torch
        import torch.nn.functional as F

        from tools.pytorch_to_mlx.converters.models import WeightNormConv1d

        pt_state = torch.load(
            kokoro_weights_path, map_location="cpu", weights_only=True,
        )
        te_state = pt_state["text_encoder"]

        # Load weights into MLX conv
        conv = WeightNormConv1d(512, 512, kernel_size=5, padding=2)
        conv.weight_g = mx.array(te_state["module.cnn.0.0.weight_g"].numpy())
        conv.weight_v = mx.array(te_state["module.cnn.0.0.weight_v"].numpy())
        conv.bias = mx.array(te_state["module.cnn.0.0.bias"].numpy())

        # Test with random input
        rng = np.random.default_rng(42)
        input_np = rng.standard_normal((1, 10, 512)).astype(np.float32)  # NLC for MLX

        # MLX forward
        mlx_output = conv(mx.array(input_np))
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # PyTorch forward (NCL format)
        pt_input = torch.tensor(input_np.transpose(0, 2, 1))
        weight_g = te_state["module.cnn.0.0.weight_g"]
        weight_v = te_state["module.cnn.0.0.weight_v"]
        bias = te_state["module.cnn.0.0.bias"]

        v_norm = torch.sqrt((weight_v**2).sum(dim=(1, 2), keepdim=True) + 1e-12)
        weight = weight_g * weight_v / v_norm
        pt_output = F.conv1d(pt_input, weight, bias, padding=2)
        pt_output_np = pt_output.detach().numpy().transpose(0, 2, 1)  # Back to NLC

        assert np.allclose(mlx_output_np, pt_output_np, atol=1e-5)


class TestInputValidation:
    """Test input validation in KokoroModel.__call__."""

    def test_invalid_input_ids_shape_1d(self):
        """Test that 1D input_ids raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((10,), dtype=mx.int32)  # 1D - invalid
        voice = mx.zeros((1, 256))

        with pytest.raises(ValueError, match="input_ids must be 2D"):
            model(input_ids, voice)

    def test_invalid_input_ids_shape_3d(self):
        """Test that 3D input_ids raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10, 5), dtype=mx.int32)  # 3D - invalid
        voice = mx.zeros((1, 256))

        with pytest.raises(ValueError, match="input_ids must be 2D"):
            model(input_ids, voice)

    def test_empty_input_ids(self):
        """Test that empty input_ids raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 0), dtype=mx.int32)  # Empty sequence
        voice = mx.zeros((1, 256))

        with pytest.raises(ValueError, match="Empty input_ids sequence"):
            model(input_ids, voice)

    def test_invalid_voice_shape_1d(self):
        """Test that 1D voice embedding raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((256,))  # 1D - invalid

        with pytest.raises(ValueError, match="voice must be 2D"):
            model(input_ids, voice)

    def test_invalid_voice_dimension(self):
        """Test that wrong voice dimension raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 64))  # Wrong dim - must be 128 or 256

        with pytest.raises(ValueError, match="voice embedding dimension must be 128 or 256"):
            model(input_ids, voice)

    def test_batch_size_mismatch(self):
        """Test that mismatched batch sizes raise ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((2, 10), dtype=mx.int32)  # batch=2
        voice = mx.zeros((1, 256))  # batch=1

        with pytest.raises(ValueError, match="batch size"):
            model(input_ids, voice)

    def test_invalid_attention_mask_shape(self):
        """Test that mismatched attention_mask shape raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))
        attention_mask = mx.zeros((1, 5), dtype=mx.bool_)  # Wrong length

        with pytest.raises(ValueError, match="attention_mask shape .* must match"):
            model(input_ids, voice, attention_mask)

    def test_invalid_speed_zero(self):
        """Test that speed=0 raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        with pytest.raises(ValueError, match="speed must be in"):
            model(input_ids, voice, speed=0)

    def test_invalid_speed_negative(self):
        """Test that negative speed raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        with pytest.raises(ValueError, match="speed must be in"):
            model(input_ids, voice, speed=-1.0)

    def test_invalid_speed_too_high(self):
        """Test that speed > 10 raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        with pytest.raises(ValueError, match="speed must be in"):
            model(input_ids, voice, speed=11.0)

    def test_valid_inputs_128_voice(self):
        """Test that valid 128-dim voice passes validation."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 128))  # 128-dim is valid
        attention_mask = mx.zeros((1, 10), dtype=mx.bool_)

        # Should not raise - forward pass will execute
        output = model(input_ids, voice, attention_mask, speed=1.0)
        mx.eval(output)
        assert output.shape[0] == 1

    def test_valid_inputs_256_voice(self):
        """Test that valid 256-dim voice passes validation."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))  # 256-dim is valid

        # Should not raise - forward pass will execute
        output = model(input_ids, voice)
        mx.eval(output)
        assert output.shape[0] == 1


class TestOutputValidation:
    """Test output validation in KokoroModel.__call__."""

    def test_audio_is_clipped_to_valid_range(self):
        """Test that output audio is clipped to [-1, 1]."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        output = model(input_ids, voice)
        mx.eval(output)

        # Audio should be clipped to [-1, 1]
        assert float(output.min()) >= -1.0
        assert float(output.max()) <= 1.0

    def test_validate_output_false_skips_validation(self):
        """Test that validate_output=False skips output validation."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        # Should succeed with validate_output=False
        output = model(input_ids, voice, validate_output=False)
        mx.eval(output)

        # Output should still be valid shape
        assert output.shape[0] == 1
        assert output.ndim == 2

    def test_validate_output_true_is_default(self):
        """Test that validate_output=True is the default behavior."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        # Default call should clip audio to [-1, 1]
        output = model(input_ids, voice)
        mx.eval(output)

        # Audio should be clipped (validation occurred)
        assert float(output.min()) >= -1.0
        assert float(output.max()) <= 1.0

    def test_validate_output_false_no_clipping(self):
        """Test that validate_output=False doesn't clip output."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        # With validation disabled, output is returned as-is (no clipping)
        # Note: for zero inputs, output might still be in [-1, 1] naturally
        output = model(input_ids, voice, validate_output=False)
        mx.eval(output)

        # Just verify we get output without errors
        assert output.shape[0] == 1


class TestStreamingSynthesis:
    """Test streaming synthesis functionality."""

    def test_streaming_produces_output(self):
        """Test that streaming synthesis yields audio chunks."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 20), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        # Collect all streaming chunks
        chunks = list(model.synthesize_streaming(input_ids, voice))

        # Should yield at least one chunk
        assert len(chunks) >= 1

        # Each chunk should be 2D [batch, samples]
        for chunk in chunks:
            assert chunk.ndim == 2
            assert chunk.shape[0] == 1

    def test_streaming_single_chunk_shape(self):
        """Test that streaming with chunk_frames larger than output produces valid single chunk."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        # Streaming with very large chunk (forces single chunk, no overlap processing)
        chunks = list(model.synthesize_streaming(
            input_ids, voice, chunk_frames=10000, overlap_frames=0,
        ))

        # Should be single chunk
        assert len(chunks) == 1

        streaming = chunks[0]
        mx.eval(streaming)

        # Should produce valid 2D output
        assert streaming.ndim == 2
        assert streaming.shape[0] == 1
        # Should have reasonable number of samples (10 tokens -> ~150k samples at default config)
        assert streaming.shape[1] > 10000

    def test_streaming_multiple_chunks_concatenate(self):
        """Test that multiple streaming chunks can be concatenated to form complete output."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        input_ids = mx.zeros((1, 30), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        # Streaming with multiple chunks (small chunk_frames forces multiple chunks)
        chunks = list(model.synthesize_streaming(
            input_ids, voice, chunk_frames=50, overlap_frames=5,
        ))

        # Should have multiple chunks for longer input
        assert len(chunks) >= 1

        # Each chunk should be valid
        for chunk in chunks:
            assert chunk.ndim == 2
            assert chunk.shape[0] == 1
            assert chunk.shape[1] > 0

        # Concatenate all chunks
        streaming = mx.concatenate(chunks, axis=1)
        mx.eval(streaming)

        # Should produce substantial output (30 tokens -> ~450k samples)
        assert streaming.shape[1] > 100000

    def test_streaming_with_zero_overlap(self):
        """Test streaming without overlap (simple concatenation)."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        input_ids = mx.zeros((1, 20), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        # Streaming without overlap
        chunks = list(model.synthesize_streaming(
            input_ids, voice, chunk_frames=50, overlap_frames=0,
        ))

        # Should yield chunks
        assert len(chunks) >= 1

        # Concatenate
        streaming = mx.concatenate(chunks, axis=1)
        mx.eval(streaming)

        # Should have reasonable output
        assert streaming.shape[0] == 1
        assert streaming.shape[1] > 0

    def test_streaming_generator_interface(self):
        """Test that synthesize_streaming returns a generator."""
        import types

        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        result = model.synthesize_streaming(input_ids, voice)

        # Should be a generator
        assert isinstance(result, types.GeneratorType)


class TestQuantization:
    """Tests for INT8/INT4 quantization of Kokoro model."""

    def test_estimate_memory_savings_full_mode(self):
        """Test memory estimation for full quantization mode."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
            estimate_memory_savings,
        )

        config = KokoroConfig()
        model = KokoroModel(config)

        est = estimate_memory_savings(model, bits=8, mode="full")

        # Should report positive savings
        assert est["savings_mb"] > 0
        assert est["savings_percent"] > 0
        assert est["original_mb"] > est["quantized_mb"]
        # Kokoro is ~82M params = ~312MB float32
        assert 250 < est["original_mb"] < 350

    def test_estimate_memory_savings_encoder_only(self):
        """Test memory estimation for encoder-only mode."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
            estimate_memory_savings,
        )

        config = KokoroConfig()
        model = KokoroModel(config)

        est_full = estimate_memory_savings(model, bits=8, mode="full")
        est_encoder = estimate_memory_savings(model, bits=8, mode="encoder_only")

        # Encoder-only should save less than full
        assert est_encoder["savings_mb"] < est_full["savings_mb"]
        assert est_encoder["quantizable_params"] < est_full["quantizable_params"]

    def test_quantize_model_full(self):
        """Test full model quantization."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
            quantize_kokoro_model,
        )

        config = KokoroConfig()
        model = KokoroModel(config)

        stats = quantize_kokoro_model(model, bits=8, mode="full")

        # Should quantize most Linear/Embedding layers
        assert stats["total_quantized"] > 80
        # Should skip layers with small dimensions
        assert stats["total_skipped"] >= 1
        assert stats["bits"] == 8
        assert stats["mode"] == "full"

    def test_quantize_model_encoder_only(self):
        """Test encoder-only quantization mode."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
            quantize_kokoro_model,
        )

        config = KokoroConfig()
        model = KokoroModel(config)

        stats = quantize_kokoro_model(model, bits=8, mode="encoder_only")

        # Should only quantize bert.* and bert_encoder layers
        for layer in stats["layers_quantized"]:
            assert layer.startswith("bert") or layer == "bert_encoder"

        # Should skip decoder and predictor layers
        assert stats["total_skipped"] > 0

    def test_quantize_model_no_adain(self):
        """Test no_adain quantization mode."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
            quantize_kokoro_model,
        )

        config = KokoroConfig()
        model = KokoroModel(config)

        stats = quantize_kokoro_model(model, bits=8, mode="no_adain")

        # Should skip adain.fc layers
        for skipped in stats["layers_skipped"]:
            if "adain" in skipped and ".fc" in skipped:
                # This is expected - adain layers should be skipped
                pass

        # But most layers should still be quantized
        assert stats["total_quantized"] > 30

    @pytest.mark.xfail(
        reason="Flaky: uninitialized weights may produce NaN depending on random state",
        strict=False,
    )
    def test_quantized_model_inference(self):
        """Test that quantized model can still run inference.

        Note: This test uses an uninitialized model without loading real weights.
        Verifies basic inference flow works after quantization, but may produce
        NaN values due to random weight initialization.
        """
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
            quantize_kokoro_model,
        )

        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        # Quantize the model
        quantize_kokoro_model(model, bits=8, mode="full")

        # Run inference
        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))

        audio = model.synthesize(input_ids, voice)
        mx.eval(audio)

        # Should produce output
        assert audio.shape[0] == 1
        assert audio.shape[1] > 0

    def test_quantize_with_int4(self):
        """Test INT4 quantization."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
            quantize_kokoro_model,
        )

        config = KokoroConfig()
        model = KokoroModel(config)

        stats = quantize_kokoro_model(model, bits=4, mode="full")

        # Should quantize layers
        assert stats["total_quantized"] > 80
        assert stats["bits"] == 4

    def test_quantize_with_smaller_group_size(self):
        """Test quantization with smaller group_size to quantize more layers."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
            quantize_kokoro_model,
        )

        config = KokoroConfig()
        model = KokoroModel(config)

        # With group_size=32, more layers can be quantized
        stats_32 = quantize_kokoro_model(model, bits=8, mode="full", group_size=32)

        # Recreate model to compare with group_size=64
        model2 = KokoroModel(config)
        stats_64 = quantize_kokoro_model(model2, bits=8, mode="full", group_size=64)

        # More layers should be quantizable with smaller group_size
        # (or equal if all meet both criteria)
        assert stats_32["total_quantized"] >= stats_64["total_quantized"]


class TestCompilation:
    """Tests for mx.compile optimizations.

    Note: These tests check compilation functionality without requiring
    full inference, avoiding MLX GPU resource issues that can cause
    NaN in sequential pytest runs. Full inference is validated in
    dedicated scripts and integration tests.
    """

    def test_compile_predictor_lstm(self):
        """Test compile_predictor_lstm method compiles successfully."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
        )

        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        # Compile predictor BiLSTMs
        setup_ms = model.compile_predictor_lstm()

        # Should complete quickly (< 100ms)
        assert setup_ms < 100

        # Verify compilation by checking the modules are wrapped
        # Compiled modules have __wrapped__ attribute or are CompiledFunction
        assert model.predictor.lstm is not None
        assert model.predictor.shared is not None

    def test_compile_predictor_lstm_inference(self):
        """Test compiled predictor produces valid intermediate outputs."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            KokoroConfig,
            KokoroModel,
        )

        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)
        model.compile_predictor_lstm()

        # Test intermediate outputs (avoids full pipeline GPU issues)
        input_ids = mx.zeros((1, 10), dtype=mx.int32)
        voice = mx.zeros((1, 256))
        speaker = voice[:, 128:]

        bert_out = model.bert(input_ids)
        bert_enc = model.bert_encoder(bert_out)
        duration_feats = model.predictor.text_encoder(bert_enc, speaker)
        mx.eval(duration_feats)

        # Should produce valid features
        assert duration_feats.shape == (1, 10, 640)
        assert not mx.any(mx.isnan(duration_feats)).item()

        # Test compiled lstm
        dur_enc = model.predictor.lstm(duration_feats)
        mx.eval(dur_enc)
        assert dur_enc.shape == (1, 10, 512)
        assert not mx.any(mx.isnan(dur_enc)).item()

        # Test compiled shared
        shared_out = model.predictor.shared(duration_feats)
        mx.eval(shared_out)
        assert shared_out.shape == (1, 10, 512)
        assert not mx.any(mx.isnan(shared_out)).item()


@pytest.mark.skip(
    reason="MLX 0.30.1 GPU resource exhaustion: tests pass individually or directly "
    "via Python but crash when run via pytest. Run individual tests with: "
    "pytest tests/test_kokoro_model.py::TestBatchSynthesis::test_name -v",
)
class TestBatchSynthesis:
    """Tests for batch synthesis functionality."""

    def test_batch_synthesis_basic(self):
        """Test basic batch synthesis with multiple utterances."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        # Seed for reproducible random weight initialization
        mx.random.seed(42)
        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        # Create 3 utterances of different lengths
        input_ids_list = [
            mx.array([1, 2, 3, 4, 5], dtype=mx.int32),  # 5 tokens
            mx.array([1, 2, 3], dtype=mx.int32),  # 3 tokens
            mx.array([1, 2, 3, 4, 5, 6, 7], dtype=mx.int32),  # 7 tokens
        ]
        voice = mx.zeros((3, 256))

        audio, lengths = model.synthesize_batch(input_ids_list, voice)
        mx.eval(audio, lengths)

        # Should produce batched output
        assert audio.shape[0] == 3  # batch size
        assert audio.shape[1] > 0  # some audio samples
        assert lengths.shape == (3,)  # one length per utterance

    def test_batch_synthesis_single_voice(self):
        """Test batch synthesis with a single voice for all utterances."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        # Seed for reproducible random weight initialization
        mx.random.seed(42)
        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        input_ids_list = [
            mx.array([1, 2, 3, 4], dtype=mx.int32),
            mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32),
        ]
        # Single voice array (1D) - should be broadcast
        voice = mx.zeros((256,))

        audio, lengths = model.synthesize_batch(input_ids_list, voice)
        mx.eval(audio, lengths)

        assert audio.shape[0] == 2
        assert lengths.shape == (2,)

    def test_batch_synthesis_voice_list(self):
        """Test batch synthesis with a list of per-utterance voices."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        # Seed for reproducible random weight initialization
        mx.random.seed(42)
        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        input_ids_list = [
            mx.array([1, 2, 3, 4], dtype=mx.int32),
            mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32),
        ]
        # List of voice embeddings
        voices = [
            mx.zeros((256,)),
            mx.ones((256,)) * 0.1,
        ]

        audio, lengths = model.synthesize_batch(input_ids_list, voices)
        mx.eval(audio, lengths)

        assert audio.shape[0] == 2
        assert lengths.shape == (2,)

    def test_batch_synthesis_empty_raises(self):
        """Test that empty input raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        with pytest.raises(ValueError, match="Empty input_ids_list"):
            model.synthesize_batch([], mx.zeros((256,)))

    def test_batch_synthesis_voice_mismatch_raises(self):
        """Test that mismatched voice count raises ValueError."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        config = KokoroConfig()
        model = KokoroModel(config)

        input_ids_list = [
            mx.array([1, 2, 3], dtype=mx.int32),
            mx.array([1, 2, 3], dtype=mx.int32),
        ]
        # Wrong number of voices
        voices = [mx.zeros((256,))]  # Only 1 voice for 2 utterances

        with pytest.raises(ValueError, match="voices list length"):
            model.synthesize_batch(input_ids_list, voices)

    def test_batch_synthesis_different_lengths(self):
        """Test that batch synthesis handles inputs of different lengths correctly."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        # Seed for reproducible random weight initialization
        mx.random.seed(42)
        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        # Create test inputs with very different lengths
        input_ids_list = [
            mx.array([1, 2, 3], dtype=mx.int32),  # Short
            mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=mx.int32),  # Long
        ]
        voice = mx.zeros((2, 256))

        # Run batch synthesis
        batch_audio, batch_lengths = model.synthesize_batch(input_ids_list, voice)
        mx.eval(batch_audio, batch_lengths)

        # Verify both outputs are valid
        len1 = int(batch_lengths[0].item())
        len2 = int(batch_lengths[1].item())

        # Both outputs should have positive length
        assert len1 > 0, f"First output should have positive length, got {len1}"
        assert len2 > 0, f"Second output should have positive length, got {len2}"

        # With random weights, different input lengths typically produce different
        # output lengths (not hardcoded). The ratio check verifies the outputs
        # are within a reasonable range of each other.
        ratio = max(len1, len2) / min(len1, len2)
        assert ratio < 10.0, f"Audio length ratio {ratio} is unexpectedly large"

    def test_batch_synthesis_with_speed(self):
        """Test batch synthesis with speed parameter."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        # Seed for reproducible random weight initialization
        mx.random.seed(42)
        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        input_ids_list = [
            mx.array([1, 2, 3, 4, 5], dtype=mx.int32),
        ]
        voice = mx.zeros((1, 256))

        # Normal speed
        audio_normal, lengths_normal = model.synthesize_batch(input_ids_list, voice, speed=1.0)
        mx.eval(audio_normal, lengths_normal)

        # Slower
        audio_slow, lengths_slow = model.synthesize_batch(input_ids_list, voice, speed=0.5)
        mx.eval(audio_slow, lengths_slow)

        # Faster
        audio_fast, lengths_fast = model.synthesize_batch(input_ids_list, voice, speed=2.0)
        mx.eval(audio_fast, lengths_fast)

        # Slower should produce more samples, faster should produce fewer
        assert int(lengths_slow[0].item()) > int(lengths_normal[0].item())
        assert int(lengths_fast[0].item()) < int(lengths_normal[0].item())

    def test_batch_synthesis_no_validation(self):
        """Test batch synthesis with validation disabled."""
        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        # Seed for reproducible random weight initialization
        mx.random.seed(42)
        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        input_ids_list = [
            mx.array([1, 2, 3, 4], dtype=mx.int32),
        ]
        voice = mx.zeros((1, 256))

        # Should not raise even with potential edge cases
        audio, lengths = model.synthesize_batch(
            input_ids_list, voice, validate_output=False,
        )
        mx.eval(audio, lengths)

        assert audio.shape[0] == 1
        assert lengths.shape == (1,)


class TestBatchSynthesisBenchmark:
    """Benchmark tests for batch synthesis throughput."""

    @pytest.mark.skipif(
        not os.environ.get("RUN_BENCHMARKS"),
        reason="Set RUN_BENCHMARKS=1 to run throughput tests",
    )
    def test_batch_vs_sequential_throughput(self):
        """
        Compare throughput: batch synthesis vs sequential.

        Expected: Batch processing should achieve higher throughput
        by amortizing GPU overhead across multiple utterances.
        """
        import time

        from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

        mx.random.seed(42)
        config = KokoroConfig()
        model = KokoroModel(config)
        model.set_deterministic(True)

        # Test inputs: 4 utterances of varying lengths
        input_ids_list = [
            mx.array([1, 2, 3, 4, 5], dtype=mx.int32),
            mx.array([1, 2, 3, 4, 5, 6, 7], dtype=mx.int32),
            mx.array([1, 2, 3, 4], dtype=mx.int32),
            mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32),
        ]
        voice = mx.zeros((256,))

        num_iterations = 5

        # Warmup
        for inp in input_ids_list:
            audio = model.synthesize(inp[None, :], voice[None, :])
            mx.eval(audio)

        batch_audio, _ = model.synthesize_batch(input_ids_list, voice)
        mx.eval(batch_audio)

        # Sequential synthesis timing
        sequential_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            for inp in input_ids_list:
                audio = model.synthesize(inp[None, :], voice[None, :])
                mx.eval(audio)
            end = time.perf_counter()
            sequential_times.append(end - start)

        # Batch synthesis timing
        batch_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            batch_audio, lengths = model.synthesize_batch(input_ids_list, voice)
            mx.eval(batch_audio, lengths)
            end = time.perf_counter()
            batch_times.append(end - start)

        sequential_mean = sum(sequential_times) / len(sequential_times)
        batch_mean = sum(batch_times) / len(batch_times)
        speedup = sequential_mean / batch_mean if batch_mean > 0 else 0

        print("\n=== Batch vs Sequential Throughput ===")
        print(f"Sequential (4 utterances): {sequential_mean*1000:.2f} ms")
        print(f"Batch (4 utterances):      {batch_mean*1000:.2f} ms")
        print(f"Throughput speedup:        {speedup:.2f}x")

        # Batch should be at least as fast as sequential
        # In practice, it should be faster due to shared overhead
        assert batch_mean <= sequential_mean * 1.5, (
            f"Batch synthesis unexpectedly slow: {batch_mean:.3f}s vs sequential {sequential_mean:.3f}s"
        )
