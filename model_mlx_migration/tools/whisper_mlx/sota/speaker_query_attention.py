#!/usr/bin/env python3
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
Speaker Query Attention - Speaker-Conditioned Whisper Encoder

Phase 10.3 implementation: Inject speaker information into Whisper encoder
using cross-attention at strategic layers.

Architecture based on SQ-Whisper (Guo et al., 2024):
- Cross-attention layers at blocks 8, 16, 24, 32
- Audio features attend to speaker embedding
- Gated residual connection for smooth blending
- Whisper weights frozen, only cross-attention trainable

Benefits:
- Better speaker separation in multi-speaker audio
- Improved recognition for target speaker in cocktail party scenarios
- Works with ECAPA-TDNN 192-dim speaker embeddings
"""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SpeakerQueryConfig:
    """Configuration for speaker query attention."""

    # Model dimensions
    d_model: int = 1280  # Whisper large-v3 hidden dimension
    speaker_dim: int = 192  # ECAPA-TDNN embedding dimension
    n_heads: int = 8  # Cross-attention heads

    # Layer injection points (after these transformer blocks)
    injection_layers: tuple[int, ...] = (8, 16, 24, 32)

    # Training
    dropout: float = 0.1

    # Gating
    gate_init_bias: float = -2.0  # Initialize gate to favor original features


class SpeakerQueryAttention(nn.Module):
    """Cross-attention layer that injects speaker information into encoder.

    Inserted after specified transformer blocks in Whisper encoder.
    Trainable while keeping Whisper weights frozen.

    Architecture:
        1. Project speaker embedding to d_model
        2. Cross-attention: audio queries, speaker key/value
        3. Gated residual: blend attended features with original

    The gating mechanism allows the model to learn how much speaker
    conditioning to apply at each position, enabling position-dependent
    speaker focus (e.g., more focus during speech, less during silence).
    """

    def __init__(
        self,
        d_model: int = 1280,
        speaker_dim: int = 192,
        n_heads: int = 8,
        dropout: float = 0.1,
        gate_init_bias: float = -2.0,
    ):
        """Initialize speaker query attention.

        Args:
            d_model: Hidden dimension of Whisper encoder
            speaker_dim: Speaker embedding dimension (192 for ECAPA-TDNN)
            n_heads: Number of attention heads
            dropout: Dropout rate
            gate_init_bias: Initial bias for gate (negative = favor original)
        """
        super().__init__()

        self.d_model = d_model
        self.speaker_dim = speaker_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"

        # Project speaker embedding to model dimension
        self.speaker_proj = nn.Linear(speaker_dim, d_model)

        # Cross-attention projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # Gating mechanism: learn to blend original and conditioned features
        # Gate input: concatenated (original, attended) features
        self.gate_proj = nn.Linear(d_model * 2, d_model)

        # Initialize gate bias to favor original features initially
        # This stabilizes training by starting with minimal speaker conditioning
        self._gate_init_bias = gate_init_bias

        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        audio_features: mx.array,
        speaker_embedding: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Apply speaker-conditioned cross-attention.

        Args:
            audio_features: Encoder hidden states (B, T, d_model)
            speaker_embedding: Target speaker embedding (B, speaker_dim) or (speaker_dim,)
            mask: Optional attention mask (not typically used)

        Returns:
            conditioned_features: (B, T, d_model) speaker-conditioned features
        """
        # Handle unbatched speaker embedding
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding[None, :]  # (1, speaker_dim)

        batch_size, seq_len, _ = audio_features.shape

        # Broadcast speaker embedding to batch size if needed
        if speaker_embedding.shape[0] == 1 and batch_size > 1:
            speaker_embedding = mx.broadcast_to(
                speaker_embedding, (batch_size, self.speaker_dim),
            )

        # Project speaker embedding to d_model
        speaker_proj = self.speaker_proj(speaker_embedding)  # (B, d_model)
        speaker_proj = speaker_proj[:, None, :]  # (B, 1, d_model)

        # Cross-attention: audio queries, speaker key/value
        # Q from audio, K/V from speaker
        q = self.query(audio_features)  # (B, T, d_model)
        k = self.key(speaker_proj)  # (B, 1, d_model)
        v = self.value(speaker_proj)  # (B, 1, d_model)

        # Reshape for multi-head attention
        # (B, T, d_model) -> (B, n_heads, T, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, 1, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, 1, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, n_heads, T, 1)

        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = mx.softmax(attn_weights, axis=-1)  # (B, n_heads, T, 1)

        # Apply attention to values
        attn_out = attn_weights @ v  # (B, n_heads, T, head_dim)

        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Output projection
        attn_out = self.out(attn_out)  # (B, T, d_model)
        attn_out = self.dropout(attn_out)

        # Gated residual connection
        # Concatenate original and attended features
        combined = mx.concatenate([audio_features, attn_out], axis=-1)  # (B, T, 2*d_model)

        # Compute gate (sigmoid gives values in [0, 1])
        gate_logits = self.gate_proj(combined)  # (B, T, d_model)
        gate = mx.sigmoid(gate_logits + self._gate_init_bias)  # (B, T, d_model)

        # Blend: gate * attended + (1 - gate) * original
        output = gate * attn_out + (1 - gate) * audio_features

        # Layer norm
        return self.norm(output)


class SpeakerConditionedEncoder(nn.Module):
    """Whisper encoder with speaker query attention layers.

    Wraps an existing AudioEncoder and injects speaker conditioning
    at specified transformer block depths.

    Training strategy:
    1. Freeze all Whisper encoder weights
    2. Train only SpeakerQueryAttention layers
    3. Loss: CTC + speaker-attributed transcription accuracy

    Usage:
        # Wrap existing encoder
        conditioned = SpeakerConditionedEncoder(whisper_encoder)

        # Enroll speaker
        speaker_emb = speaker_encoder.encode(enrollment_audio)

        # Encode with speaker conditioning
        features = conditioned(mel, speaker_embedding=speaker_emb)
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: SpeakerQueryConfig | None = None,
    ):
        """Initialize speaker-conditioned encoder.

        Args:
            encoder: Base Whisper encoder (AudioEncoder)
            config: Speaker query configuration
        """
        super().__init__()

        self.config = config or SpeakerQueryConfig()
        self.encoder = encoder  # Will be frozen during training

        # Create speaker query layers at specified depths
        self.speaker_query_layers = {}
        for layer_idx in self.config.injection_layers:
            self.speaker_query_layers[layer_idx] = SpeakerQueryAttention(
                d_model=self.config.d_model,
                speaker_dim=self.config.speaker_dim,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                gate_init_bias=self.config.gate_init_bias,
            )

        # Track frozen state
        self._encoder_frozen = False

    def freeze_encoder(self):
        """Freeze encoder weights (call before training)."""
        self.encoder.freeze()
        self._encoder_frozen = True

    def unfreeze_encoder(self):
        """Unfreeze encoder weights (for fine-tuning)."""
        self.encoder.unfreeze()
        self._encoder_frozen = False

    def __call__(
        self,
        x: mx.array,
        speaker_embedding: mx.array | None = None,
        variable_length: bool = True,
    ) -> mx.array:
        """Encode audio with optional speaker conditioning.

        Args:
            x: Mel spectrogram (n_frames, n_mels) or (B, n_frames, n_mels)
            speaker_embedding: Target speaker embedding (B, 192) or (192,)
                              If None, returns standard encoder output
            variable_length: Use variable-length encoding (default True)

        Returns:
            Encoded features (B, seq_len, d_model)
        """
        # If no speaker embedding, just run standard encoder
        if speaker_embedding is None:
            return self.encoder(x, variable_length=variable_length)

        # Run encoder with speaker injection at specified layers
        return self._forward_with_speaker(x, speaker_embedding, variable_length)

    def _forward_with_speaker(
        self,
        x: mx.array,
        speaker_embedding: mx.array,
        variable_length: bool,
    ) -> mx.array:
        """Forward pass with speaker injection at specified layers.

        This reimplements the encoder forward pass to inject speaker
        conditioning at the right points. We need access to intermediate
        transformer block outputs.
        """
        # Handle unbatched input
        if x.ndim == 2:
            x = x[None]

        # Determine sequence length after conv2 (stride=2)
        input_frames = x.shape[1]
        seq_len = (input_frames + 1) // 2

        # Get positional embeddings
        if variable_length:
            if seq_len > self.encoder.n_ctx:
                raise ValueError(
                    f"Input sequence length {seq_len} exceeds maximum context {self.encoder.n_ctx}",
                )
            pos = self.encoder._positional_embedding[:seq_len]  # noqa: SLF001
        else:
            pos = self.encoder._positional_embedding  # noqa: SLF001

        # Convolutional frontend
        x = nn.gelu(self.encoder.conv1(x))
        x = nn.gelu(self.encoder.conv2(x))

        # Add positional embedding
        x = x + pos

        # Transformer blocks with speaker injection
        for i, block in enumerate(self.encoder.blocks):
            x, _, _ = block(x)

            # Inject speaker conditioning at specified layers
            if i in self.speaker_query_layers:
                x = self.speaker_query_layers[i](x, speaker_embedding)

        # Final layer norm
        return mx.fast.layer_norm(
            x,
            self.encoder.ln_post.weight,
            self.encoder.ln_post.bias,
            eps=1e-5,
        )

    def trainable_parameters(self) -> dict:
        """Get only the trainable parameters (speaker query layers)."""
        params = {}
        for layer_idx, layer in self.speaker_query_layers.items():
            layer_params = layer.parameters()
            for name, param in layer_params.items():
                params[f"speaker_query_{layer_idx}.{name}"] = param
        return params

    def save_speaker_query_weights(self, path: str):
        """Save only the speaker query attention weights.

        Args:
            path: Path to save weights (creates directory if needed)
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            config_dict = {
                "d_model": self.config.d_model,
                "speaker_dim": self.config.speaker_dim,
                "n_heads": self.config.n_heads,
                "injection_layers": list(self.config.injection_layers),
                "dropout": self.config.dropout,
                "gate_init_bias": self.config.gate_init_bias,
            }
            json.dump(config_dict, f, indent=2)

        # Save speaker query weights only
        weights = {}
        for layer_idx, layer in self.speaker_query_layers.items():
            layer_params = dict(layer.parameters())
            for name, param in layer_params.items():
                weights[f"speaker_query_{layer_idx}.{name}"] = param

        mx.savez(str(path / "speaker_query_weights.npz"), **weights)

    def load_speaker_query_weights(self, path: str):
        """Load speaker query attention weights.

        Args:
            path: Path to load weights from
        """
        path = Path(path)

        # Load weights
        weights = mx.load(str(path / "speaker_query_weights.npz"))

        # Assign to layers
        for layer_idx, layer in self.speaker_query_layers.items():
            prefix = f"speaker_query_{layer_idx}."
            layer_weights = [
                (k[len(prefix):], v)
                for k, v in weights.items()
                if k.startswith(prefix)
            ]
            layer.load_weights(layer_weights)


# =============================================================================
# Utility Functions
# =============================================================================


def create_speaker_conditioned_encoder(
    encoder: nn.Module,
    speaker_dim: int = 192,
    injection_layers: tuple[int, ...] = (8, 16, 24, 32),
) -> SpeakerConditionedEncoder:
    """Create a speaker-conditioned encoder from an existing encoder.

    This is a convenience function for the common use case.

    Args:
        encoder: Base Whisper encoder
        speaker_dim: Speaker embedding dimension (192 for ECAPA-TDNN)
        injection_layers: Transformer block indices for speaker injection

    Returns:
        SpeakerConditionedEncoder instance
    """
    # Infer d_model from encoder
    d_model = encoder.n_state

    config = SpeakerQueryConfig(
        d_model=d_model,
        speaker_dim=speaker_dim,
        injection_layers=injection_layers,
    )

    return SpeakerConditionedEncoder(encoder, config)


# =============================================================================
# Tests
# =============================================================================


def test_speaker_query_attention():
    """Test SpeakerQueryAttention layer."""
    print("Testing SpeakerQueryAttention...")

    # Create layer
    layer = SpeakerQueryAttention(
        d_model=512,
        speaker_dim=192,
        n_heads=8,
    )

    # Test inputs
    batch_size = 2
    seq_len = 100
    audio_features = mx.random.normal((batch_size, seq_len, 512))
    speaker_embedding = mx.random.normal((batch_size, 192))

    # Forward pass
    output = layer(audio_features, speaker_embedding)

    # Check output shape
    assert output.shape == (batch_size, seq_len, 512), f"Expected (2, 100, 512), got {output.shape}"

    # Test with unbatched speaker embedding
    speaker_single = mx.random.normal((192,))
    output2 = layer(audio_features, speaker_single)
    assert output2.shape == (batch_size, seq_len, 512)

    print("  Output shape: OK")
    print("  Unbatched speaker: OK")
    print("SpeakerQueryAttention: PASS\n")


def test_speaker_conditioned_encoder():
    """Test SpeakerConditionedEncoder with a mock encoder."""
    print("Testing SpeakerConditionedEncoder...")

    # Create mock encoder (simplified for testing)
    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_ctx = 1500
            self.n_state = 512
            self.conv1 = nn.Conv1d(80, 512, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)
            self._positional_embedding = mx.zeros((1500, 512))

            # Simple transformer blocks (just linear layers for test)
            self.blocks = [
                MockBlock(512) for _ in range(32)
            ]
            self.ln_post = nn.LayerNorm(512)

        def __call__(self, x, variable_length=True):
            if x.ndim == 2:
                x = x[None]
            x = nn.gelu(self.conv1(x))
            x = nn.gelu(self.conv2(x))
            seq_len = x.shape[1]
            x = x + self._positional_embedding[:seq_len]
            for block in self.blocks:
                x, _, _ = block(x)
            return self.ln_post(x)

    class MockBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def __call__(self, x, mask=None):
            return self.linear(x), None, None

    # Create mock encoder
    mock_encoder = MockEncoder()

    # Create speaker-conditioned encoder
    config = SpeakerQueryConfig(
        d_model=512,
        speaker_dim=192,
        injection_layers=(8, 16, 24),  # Fewer layers for mock
    )
    conditioned = SpeakerConditionedEncoder(mock_encoder, config)

    # Test inputs
    batch_size = 2
    n_frames = 200
    mel = mx.random.normal((batch_size, n_frames, 80))
    speaker_emb = mx.random.normal((batch_size, 192))

    # Forward without speaker (should work)
    output_no_speaker = conditioned(mel, speaker_embedding=None)
    expected_seq_len = (n_frames + 1) // 2
    assert output_no_speaker.shape == (batch_size, expected_seq_len, 512), \
        f"Expected ({batch_size}, {expected_seq_len}, 512), got {output_no_speaker.shape}"
    print("  Forward without speaker: OK")

    # Forward with speaker
    output_with_speaker = conditioned(mel, speaker_embedding=speaker_emb)
    assert output_with_speaker.shape == (batch_size, expected_seq_len, 512), \
        f"Expected ({batch_size}, {expected_seq_len}, 512), got {output_with_speaker.shape}"
    print("  Forward with speaker: OK")

    # Check that outputs differ (speaker conditioning has effect)
    diff = mx.abs(output_with_speaker - output_no_speaker).max()
    assert float(diff) > 0, "Speaker conditioning should change output"
    print(f"  Max diff with/without speaker: {float(diff):.4f}")

    # Test trainable parameters
    trainable = conditioned.trainable_parameters()
    assert len(trainable) > 0, "Should have trainable parameters"
    print(f"  Trainable params: {len(trainable)} tensors")

    print("SpeakerConditionedEncoder: PASS\n")


if __name__ == "__main__":
    test_speaker_query_attention()
    test_speaker_conditioned_encoder()
    print("All tests passed!")
