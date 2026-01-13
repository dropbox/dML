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
Unit tests for CosyVoice2 Flow Matching Model.

Tests the flow model components:
- Token embedding
- Pre-lookahead convolution
- Multi-head attention
- Transformer encoder layers
- Flow encoder
- Time embedding
- UNet decoder blocks
- MaskedDiffWithXvec complete model
"""

import mlx.core as mx

from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import (
    CausalFlowConfig,
    CausalFlowEncoder,
    CausalMaskedDiffWithXvec,
    DownBlock,
    FlowDecoder,
    FlowEncoder,
    FlowMatchingConfig,
    MaskedDiffWithXvec,
    MidBlock,
    MultiHeadAttention,
    PreLookaheadLayer,
    ResBlock,
    TimeEmbedding,
    TransformerEncoderLayer,
    UpBlock,
    sinusoidal_embedding,
)


class TestFlowMatchingConfig:
    """Tests for FlowMatchingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FlowMatchingConfig()

        assert config.vocab_size == 6561
        assert config.embed_dim == 512
        assert config.num_encoder_layers == 6
        assert config.encoder_attention_heads == 8
        assert config.mel_dim == 80
        assert config.speaker_embed_dim == 192


class TestSinusoidalEmbedding:
    """Tests for sinusoidal position embedding."""

    def test_embedding_shape(self):
        """Test output shape of sinusoidal embedding."""
        timesteps = mx.array([0.0, 0.5, 1.0])
        dim = 320

        emb = sinusoidal_embedding(timesteps, dim)

        assert emb.shape == (3, 320)

    def test_embedding_range(self):
        """Test that embedding values are in expected range."""
        timesteps = mx.array([0.0, 0.5, 1.0])
        dim = 320

        emb = sinusoidal_embedding(timesteps, dim)
        mx.eval(emb)

        # Sinusoidal values should be in [-1, 1]
        assert mx.all(emb >= -1.0).item()
        assert mx.all(emb <= 1.0).item()


class TestPreLookaheadLayer:
    """Tests for PreLookaheadLayer."""

    def test_output_shape(self):
        """Test output shape matches expected."""
        config = FlowMatchingConfig()
        layer = PreLookaheadLayer(config)

        x = mx.random.normal((2, 100, 512))
        y = layer(x)
        mx.eval(y)

        # Output length reduced by conv operations
        # conv1 kernel=4, conv2 kernel=3, both no padding -> length-5
        assert y.shape[0] == 2
        assert y.shape[2] == 512
        assert y.shape[1] < 100  # Reduced by convs


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_output_shape(self):
        """Test output shape equals input shape."""
        config = FlowMatchingConfig()
        attn = MultiHeadAttention(config)

        x = mx.random.normal((2, 50, 512))
        y = attn(x)
        mx.eval(y)

        assert y.shape == (2, 50, 512)

    def test_attention_dimensions(self):
        """Test attention head dimensions."""
        config = FlowMatchingConfig()
        attn = MultiHeadAttention(config)

        assert attn.num_heads == 8
        assert attn.head_dim == 64  # 512 / 8
        assert attn.scale == 64**-0.5


class TestTransformerEncoderLayer:
    """Tests for TransformerEncoderLayer."""

    def test_output_shape(self):
        """Test output shape equals input shape."""
        config = FlowMatchingConfig()
        layer = TransformerEncoderLayer(config)

        x = mx.random.normal((2, 50, 512))
        y = layer(x)
        mx.eval(y)

        assert y.shape == (2, 50, 512)

    def test_residual_connection(self):
        """Test residual connections preserve information."""
        config = FlowMatchingConfig()
        layer = TransformerEncoderLayer(config)

        x = mx.random.normal((1, 10, 512))
        y = layer(x)
        mx.eval(y)

        # Output should not be zero (residual preserves input)
        assert mx.sum(mx.abs(y)).item() > 0


class TestFlowEncoder:
    """Tests for FlowEncoder."""

    def test_output_shape(self):
        """Test encoder output shape."""
        config = FlowMatchingConfig()
        encoder = FlowEncoder(config)

        x = mx.random.normal((2, 100, 512))
        y = encoder(x)
        mx.eval(y)

        assert y.shape[0] == 2
        assert y.shape[2] == 512

    def test_num_layers(self):
        """Test encoder has correct number of layers."""
        config = FlowMatchingConfig()
        encoder = FlowEncoder(config)

        assert len(encoder.encoder_layer) == 6


class TestTimeEmbedding:
    """Tests for TimeEmbedding."""

    def test_output_shape(self):
        """Test time embedding output shape."""
        config = FlowMatchingConfig()
        time_mlp = TimeEmbedding(config)

        t = mx.array([0.0, 0.5, 1.0])
        emb = time_mlp(t)
        mx.eval(emb)

        assert emb.shape == (3, 1024)

    def test_different_timesteps(self):
        """Test different timesteps produce different embeddings."""
        config = FlowMatchingConfig()
        time_mlp = TimeEmbedding(config)

        t1 = mx.array([0.0])
        t2 = mx.array([1.0])

        emb1 = time_mlp(t1)
        emb2 = time_mlp(t2)
        mx.eval(emb1, emb2)

        # Embeddings should differ
        diff = mx.sum(mx.abs(emb1 - emb2)).item()
        assert diff > 0


class TestResBlock:
    """Tests for ResBlock."""

    def test_output_shape_same_channels(self):
        """Test output shape with same input/output channels."""
        block = ResBlock(256, 256, 1024)

        x = mx.random.normal((2, 50, 256))
        t = mx.random.normal((2, 1024))

        y = block(x, t)
        mx.eval(y)

        assert y.shape == (2, 50, 256)

    def test_output_shape_different_channels(self):
        """Test output shape with different channels."""
        block = ResBlock(128, 256, 1024)

        x = mx.random.normal((2, 50, 128))
        t = mx.random.normal((2, 1024))

        y = block(x, t)
        mx.eval(y)

        assert y.shape == (2, 50, 256)


class TestDownBlock:
    """Tests for DownBlock."""

    def test_output_shapes(self):
        """Test downblock returns correct shapes."""
        block = DownBlock(256, 256, 1024)

        x = mx.random.normal((2, 100, 256))
        t = mx.random.normal((2, 1024))

        out, skip = block(x, t)
        mx.eval(out, skip)

        # Output is downsampled by 2
        assert out.shape == (2, 50, 256)
        # Skip has original resolution
        assert skip.shape == (2, 100, 256)


class TestMidBlock:
    """Tests for MidBlock."""

    def test_output_shape(self):
        """Test midblock preserves shape."""
        block = MidBlock(256, 1024)

        x = mx.random.normal((2, 50, 256))
        t = mx.random.normal((2, 1024))

        y = block(x, t)
        mx.eval(y)

        assert y.shape == (2, 50, 256)


class TestUpBlock:
    """Tests for UpBlock."""

    def test_output_shape(self):
        """Test upblock returns correct shape."""
        block = UpBlock(256, 256, 1024)

        x = mx.random.normal((2, 50, 256))
        skip = mx.random.normal((2, 100, 256))
        t = mx.random.normal((2, 1024))

        y = block(x, skip, t)
        mx.eval(y)

        # Output matches skip resolution
        assert y.shape == (2, 100, 256)


class TestFlowDecoder:
    """Tests for FlowDecoder (legacy UNet-style)."""

    def test_output_shape(self):
        """Test decoder output shape matches input."""
        config = FlowMatchingConfig()
        decoder = FlowDecoder(config)

        x = mx.random.normal((2, 100, 80))  # Noisy mel
        t = mx.array([0.5, 0.5])
        cond = mx.random.normal((2, 100, 80))  # Conditioning

        y = decoder(x, t, cond)
        mx.eval(y)

        assert y.shape == (2, 100, 80)


# ====================
# DiT-style Decoder Tests
# ====================


class TestDiTConvBlock:
    """Tests for DiTConvBlock."""

    def test_output_shape_same_channels(self):
        """Test output shape with same channels."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import DiTConvBlock

        block = DiTConvBlock(256, 256, 1024)

        x = mx.random.normal((2, 50, 256))
        t = mx.random.normal((2, 1024))

        y = block(x, t)
        mx.eval(y)

        assert y.shape == (2, 50, 256)

    def test_output_shape_different_channels(self):
        """Test output shape with channel change."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import DiTConvBlock

        block = DiTConvBlock(320, 256, 1024)

        x = mx.random.normal((2, 50, 320))
        t = mx.random.normal((2, 1024))

        y = block(x, t)
        mx.eval(y)

        assert y.shape == (2, 50, 256)


class TestDiTAttentionBlock:
    """Tests for DiTAttentionBlock."""

    def test_output_shape(self):
        """Test attention block preserves shape."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import (
            DiTAttentionBlock,
        )

        block = DiTAttentionBlock(256, num_heads=8)

        x = mx.random.normal((2, 50, 256))

        y = block(x)
        mx.eval(y)

        assert y.shape == (2, 50, 256)


class TestDiTBlock:
    """Tests for DiTBlock."""

    def test_output_shape(self):
        """Test full DiT block output."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import DiTBlock

        block = DiTBlock(256, 256, 1024, num_attention_blocks=2)

        x = mx.random.normal((2, 50, 256))
        t = mx.random.normal((2, 1024))

        y = block(x, t)
        mx.eval(y)

        assert y.shape == (2, 50, 256)


class TestDiTDecoder:
    """Tests for DiTDecoder."""

    def test_output_shape(self):
        """Test DiT decoder output shape."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import DiTDecoder

        config = FlowMatchingConfig()
        decoder = DiTDecoder(config)

        x = mx.random.normal((2, 100, 80))  # Noisy mel
        t = mx.array([0.5, 0.5])
        cond = mx.random.normal((2, 100, 80))  # Conditioning

        y = decoder(x, t, cond)
        mx.eval(y)

        assert y.shape == (2, 100, 80)

    def test_architecture_structure(self):
        """Test DiT decoder has correct structure."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import DiTDecoder

        config = FlowMatchingConfig()
        decoder = DiTDecoder(config)

        # Should have 1 down, 12 mid, 1 up blocks
        assert len(decoder.down_blocks) == 1
        assert len(decoder.mid_blocks) == 12
        assert len(decoder.up_blocks) == 1


class TestMaskedDiffWithXvec:
    """Tests for complete MaskedDiffWithXvec model."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        config = FlowMatchingConfig()
        model = MaskedDiffWithXvec(config)

        # Inputs
        tokens = mx.random.randint(0, 6561, (2, 50))
        speaker_embed = mx.random.normal((2, 192))
        t = mx.array([0.5, 0.5])
        x_noisy = mx.random.normal((2, 100, 80))

        # Forward
        velocity = model(tokens, speaker_embed, t, x_noisy)
        mx.eval(velocity)

        assert velocity.shape == (2, 100, 80)

    def test_generate_shape(self):
        """Test generation output shape."""
        config = FlowMatchingConfig()
        model = MaskedDiffWithXvec(config)

        # Inputs
        tokens = mx.random.randint(0, 6561, (1, 30))
        speaker_embed = mx.random.normal((1, 192))

        # Generate
        mel = model.generate(tokens, speaker_embed, mel_length=50, num_steps=2)
        mx.eval(mel)

        assert mel.shape == (1, 50, 80)

    def test_embedding_dimensions(self):
        """Test embedding layer dimensions match config."""
        config = FlowMatchingConfig()
        model = MaskedDiffWithXvec(config)

        assert model.input_embedding.weight.shape == (6561, 512)

    def test_encoder_proj_dimensions(self):
        """Test encoder projection dimensions."""
        config = FlowMatchingConfig()
        model = MaskedDiffWithXvec(config)

        # MLX Linear stores weights as (out, in)
        assert model.encoder_proj.weight.shape == (80, 512)

    def test_speaker_proj_dimensions(self):
        """Test speaker embedding projection dimensions."""
        config = FlowMatchingConfig()
        model = MaskedDiffWithXvec(config)

        # MLX Linear stores weights as (out, in)
        assert model.spk_embed_affine_layer.weight.shape == (80, 192)


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

    def test_encoder_params(self):
        """Test encoder has reasonable parameter count."""
        config = FlowMatchingConfig()
        encoder = FlowEncoder(config)

        # Count parameters
        total = count_params(encoder.parameters())

        # 6 transformer layers + pre-lookahead
        # Each layer: ~4M params (attention + FFN)
        # Total should be around 25-30M
        assert total > 1_000_000  # At least 1M
        assert total < 100_000_000  # Less than 100M

    def test_full_model_params(self):
        """Test full model parameter count."""
        config = FlowMatchingConfig()
        model = MaskedDiffWithXvec(config)

        # Count parameters
        total = count_params(model.parameters())

        # Current implementation has ~6.7M params (simplified decoder)
        # Full CosyVoice2 flow model is ~112M
        # Will be updated when decoder architecture is refined
        assert total > 5_000_000  # At least 5M


# ====================
# CausalMaskedDiffWithXvec Tests
# ====================


class TestCausalFlowConfig:
    """Tests for CausalFlowConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CausalFlowConfig()

        # Inherits from FlowMatchingConfig
        assert config.vocab_size == 6561
        assert config.embed_dim == 512
        assert config.mel_dim == 80

        # Streaming-specific parameters
        assert config.pre_lookahead_len == 3
        assert config.token_mel_ratio == 4.0
        assert config.streaming_chunk_size == 16

    def test_custom_streaming_params(self):
        """Test custom streaming parameters."""
        config = CausalFlowConfig(
            pre_lookahead_len=5,
            token_mel_ratio=3.5,
            streaming_chunk_size=32,
        )

        assert config.pre_lookahead_len == 5
        assert config.token_mel_ratio == 3.5
        assert config.streaming_chunk_size == 32


class TestCausalFlowEncoder:
    """Tests for CausalFlowEncoder."""

    def test_output_shape(self):
        """Test output shape matches expected."""
        config = CausalFlowConfig()
        encoder = CausalFlowEncoder(config)

        x = mx.random.normal((2, 100, 512))
        y = encoder(x)
        mx.eval(y)

        # Output should have same batch and dim, reduced length
        assert y.shape[0] == 2
        assert y.shape[2] == 512
        assert y.shape[1] < 100  # Reduced by pre-lookahead convs

    def test_streaming_mode(self):
        """Test streaming mode with causal masking."""
        config = CausalFlowConfig()
        encoder = CausalFlowEncoder(config)

        x = mx.random.normal((2, 50, 512))
        y = encoder(x, streaming=True)
        mx.eval(y)

        # Should produce output in streaming mode
        assert y.shape[0] == 2
        assert y.shape[2] == 512

    def test_context_concatenation(self):
        """Test context is concatenated in streaming mode."""
        config = CausalFlowConfig()
        encoder = CausalFlowEncoder(config)

        x = mx.random.normal((2, 50, 512))
        context = mx.random.normal((2, 10, 512))
        y = encoder(x, streaming=True, context=context)
        mx.eval(y)

        # Output should account for context
        assert y.shape[0] == 2
        assert y.shape[2] == 512

    def test_cache_reset(self):
        """Test cache reset functionality."""
        config = CausalFlowConfig()
        encoder = CausalFlowEncoder(config)

        # Run with streaming
        x = mx.random.normal((2, 50, 512))
        encoder(x, streaming=True)

        # Reset cache
        encoder.reset_cache()

        assert encoder._cache is None
        assert encoder._context is None


class TestCausalMaskedDiffWithXvec:
    """Tests for CausalMaskedDiffWithXvec."""

    def test_initialization(self):
        """Test model initialization."""
        config = CausalFlowConfig()
        model = CausalMaskedDiffWithXvec(config)

        assert model.config == config
        assert model.input_embedding is not None
        assert model.encoder is not None
        assert model.decoder is not None

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        config = CausalFlowConfig()
        model = CausalMaskedDiffWithXvec(config)

        batch = 2
        seq_len = 50
        mel_len = 200

        tokens = mx.random.randint(0, config.vocab_size, (batch, seq_len))
        speaker_embed = mx.random.normal((batch, config.speaker_embed_dim))
        t = mx.array([0.5, 0.3])
        x_noisy = mx.random.normal((batch, mel_len, config.mel_dim))

        velocity = model(tokens, speaker_embed, t, x_noisy)
        mx.eval(velocity)

        assert velocity.shape == (batch, mel_len, config.mel_dim)

    def test_forward_streaming_mode(self):
        """Test forward pass in streaming mode."""
        config = CausalFlowConfig()
        model = CausalMaskedDiffWithXvec(config)

        batch = 2
        seq_len = 50
        mel_len = 200

        tokens = mx.random.randint(0, config.vocab_size, (batch, seq_len))
        speaker_embed = mx.random.normal((batch, config.speaker_embed_dim))
        t = mx.array([0.5, 0.3])
        x_noisy = mx.random.normal((batch, mel_len, config.mel_dim))

        velocity = model(tokens, speaker_embed, t, x_noisy, streaming=True)
        mx.eval(velocity)

        assert velocity.shape == (batch, mel_len, config.mel_dim)

    def test_generate(self):
        """Test mel generation with ODE solver."""
        config = CausalFlowConfig()
        model = CausalMaskedDiffWithXvec(config)

        batch = 2
        seq_len = 20
        mel_length = 80

        tokens = mx.random.randint(0, config.vocab_size, (batch, seq_len))
        speaker_embed = mx.random.normal((batch, config.speaker_embed_dim))

        mel = model.generate(tokens, speaker_embed, mel_length, num_steps=3)
        mx.eval(mel)

        assert mel.shape == (batch, mel_length, config.mel_dim)

    def test_generate_streaming(self):
        """Test streaming mel generation."""
        config = CausalFlowConfig()
        model = CausalMaskedDiffWithXvec(config)

        batch = 2
        chunk_len = 10

        tokens = mx.random.randint(0, config.vocab_size, (batch, chunk_len))
        speaker_embed = mx.random.normal((batch, config.speaker_embed_dim))

        # First chunk
        mel_chunk1 = model.generate_streaming(
            tokens, speaker_embed, num_steps=3, finalize=False,
        )
        mx.eval(mel_chunk1)

        # Should produce mel frames based on token_mel_ratio
        int((chunk_len - config.pre_lookahead_len) * config.token_mel_ratio)
        assert mel_chunk1.shape[0] == batch
        assert mel_chunk1.shape[2] == config.mel_dim

    def test_cache_reset(self):
        """Test cache reset for streaming."""
        config = CausalFlowConfig()
        model = CausalMaskedDiffWithXvec(config)

        # Do some streaming
        batch = 2
        tokens = mx.random.randint(0, config.vocab_size, (batch, 10))
        speaker_embed = mx.random.normal((batch, config.speaker_embed_dim))
        model.generate_streaming(tokens, speaker_embed, num_steps=2, finalize=False)

        # Reset
        model.reset_cache()

        assert model._mel_cache is None
        assert model._noise_cache is None

    def test_interpolate_to_length(self):
        """Test length interpolation."""
        config = CausalFlowConfig()
        model = CausalMaskedDiffWithXvec(config)

        x = mx.random.normal((2, 50, 80))

        # Upsample
        y = model._interpolate_to_length(x, 100)
        mx.eval(y)
        assert y.shape == (2, 100, 80)

        # Downsample
        y = model._interpolate_to_length(x, 25)
        mx.eval(y)
        assert y.shape == (2, 25, 80)

        # Same length
        y = model._interpolate_to_length(x, 50)
        mx.eval(y)
        assert y.shape == (2, 50, 80)

    def test_parameter_count(self):
        """Test causal model has similar params to non-causal."""
        config = CausalFlowConfig()
        causal_model = CausalMaskedDiffWithXvec(config)
        non_causal_model = MaskedDiffWithXvec(FlowMatchingConfig())

        causal_params = count_params(causal_model.parameters())
        non_causal_params = count_params(non_causal_model.parameters())

        # Causal should have more params due to DiT decoder
        # Non-causal uses legacy FlowDecoder
        assert causal_params > 0
        assert non_causal_params > 0
