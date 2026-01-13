"""Emotion2vec MLX Implementation.

This module implements Emotion2vec for speech emotion recognition in MLX.
Paper: https://arxiv.org/abs/2312.15185
Checkpoint: iic/emotion2vec_base (FunASR/ModelScope)

Architecture:
    - Local encoder: 7 Conv1d layers for raw audio feature extraction
    - Projection: LayerNorm + Linear (512 -> 768)
    - Relative positional encoder: 5 Conv1d layers (grouped)
    - Context encoder (prenet): 4 transformer blocks
    - Main encoder: 8 transformer blocks
    - ALiBi attention bias with learned scaling
    - Extra tokens: 10 prepended learnable tokens

This is a Data2Vec-style self-supervised model fine-tuned for emotion.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .emotion2vec_config import Emotion2vecConfig


class Conv1dBlock(nn.Module):
    """Conv1d block with optional LayerNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        # Original emotion2vec uses no padding (valid convolution)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply conv block.

        Args:
            x: Input tensor (batch, length, channels)

        Returns:
            Output tensor (batch, new_length, out_channels)
        """
        # MLX Conv1d expects (batch, length, channels)
        x = self.conv(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return nn.gelu(x)


class LocalEncoder(nn.Module):
    """Local encoder: Conv1d stack for feature extraction from raw audio."""

    def __init__(self, config: Emotion2vecConfig):
        super().__init__()
        self.config = config

        # Build conv layers based on spec
        # spec format: [(out_ch, kernel_size, stride), ...]
        self.conv_layers = []
        in_ch = 1  # Raw audio input
        for _i, (out_ch, kernel_size, stride) in enumerate(config.local_encoder_spec):
            layer = Conv1dBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                use_layer_norm=True,
            )
            self.conv_layers.append(layer)
            in_ch = out_ch

    def __call__(self, x: mx.array) -> mx.array:
        """Extract features from raw audio.

        Args:
            x: Raw audio (batch, samples) or (batch, samples, 1)

        Returns:
            Features (batch, frames, 512)
        """
        # Ensure 3D input
        if x.ndim == 2:
            x = x[:, :, None]  # Add channel dim

        # Apply conv layers
        for conv in self.conv_layers:
            x = conv(x)

        return x


class FeatureProjection(nn.Module):
    """Project features from local encoder dimension to transformer dimension."""

    def __init__(self, config: Emotion2vecConfig):
        super().__init__()
        # LayerNorm on 512-dim features
        self.layer_norm = nn.LayerNorm(512, eps=config.layer_norm_eps)
        # Project to embed_dim
        self.projection = nn.Linear(512, config.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Project features.

        Args:
            x: Features (batch, frames, 512)

        Returns:
            Projected features (batch, frames, embed_dim)
        """
        x = self.layer_norm(x)
        return self.projection(x)


class RelativePositionalEncoder(nn.Module):
    """Relative positional encoding using grouped convolutions."""

    def __init__(self, config: Emotion2vecConfig):
        super().__init__()
        self.config = config

        # Stack of grouped Conv1d layers
        self.conv_layers = []
        kernel_size = 19  # From weight shape [768, 48, 19]
        groups = config.conv_pos_groups  # 16

        for _ in range(config.conv_pos_depth):
            conv = nn.Conv1d(
                in_channels=config.embed_dim,
                out_channels=config.embed_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
            )
            self.conv_layers.append(conv)

    def __call__(self, x: mx.array) -> mx.array:
        """Add relative positional information.

        Args:
            x: Input features (batch, seq_len, embed_dim)

        Returns:
            Features with positional info (batch, seq_len, embed_dim)
        """
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            x = nn.gelu(x)
            x = x + residual
        return x


class TransformerBlock(nn.Module):
    """Transformer block with post-norm architecture (layer_norm_first=False)."""

    def __init__(self, config: Emotion2vecConfig):
        super().__init__()
        self.config = config

        # Self-attention with combined QKV
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # MLP
        self.fc1 = nn.Linear(config.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.embed_dim)

        # Layer norms
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def __call__(
        self,
        x: mx.array,
        alibi_bias: mx.array | None = None,
    ) -> mx.array:
        """Apply transformer block.

        Args:
            x: Input (batch, seq_len, embed_dim)
            alibi_bias: Optional ALiBi attention bias (batch, heads, seq_len, seq_len)

        Returns:
            Output (batch, seq_len, embed_dim)
        """
        # Self-attention
        residual = x
        x = self._attention(x, alibi_bias)
        x = residual + x
        x = self.norm1(x)

        # MLP
        residual = x
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        x = residual + x
        return self.norm2(x)


    def _attention(
        self,
        x: mx.array,
        alibi_bias: mx.array | None = None,
    ) -> mx.array:
        """Multi-head self-attention."""
        batch_size, seq_len, _ = x.shape
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Add ALiBi bias if provided
        if alibi_bias is not None:
            attn = attn + alibi_bias

        attn = mx.softmax(attn, axis=-1)

        # Apply attention to values
        out = attn @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.proj(out)



class Emotion2vecEncoder(nn.Module):
    """Emotion2vec encoder (main transformer stack)."""

    def __init__(self, config: Emotion2vecConfig):
        super().__init__()
        self.config = config

        # Context encoder (prenet) - 4 blocks
        self.context_encoder = [
            TransformerBlock(config) for _ in range(config.prenet_depth)
        ]
        self.context_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        # Main encoder - 8 blocks
        self.blocks = [
            TransformerBlock(config) for _ in range(config.depth)
        ]

        # ALiBi scale (learnable)
        if config.learned_alibi_scale_per_head:
            self.alibi_scale = mx.ones((1, 1, config.num_alibi_heads, 1, 1))
        else:
            self.alibi_scale = mx.ones((1,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply encoder.

        Args:
            x: Input features (batch, seq_len, embed_dim)

        Returns:
            Encoded features (batch, seq_len, embed_dim)
        """
        # Compute ALiBi bias
        seq_len = x.shape[1]
        alibi_bias = self._compute_alibi_bias(seq_len)

        # Context encoder
        for block in self.context_encoder:
            x = block(x, alibi_bias)
        x = self.context_norm(x)

        # Main encoder
        for block in self.blocks:
            x = block(x, alibi_bias)

        return x

    def _compute_alibi_bias(self, seq_len: int) -> mx.array:
        """Compute ALiBi attention bias.

        ALiBi adds a linear bias to attention scores based on
        relative position distance.
        """
        # Create position indices
        positions = mx.arange(seq_len)
        # Compute relative distances (query_pos - key_pos)
        relative_pos = positions[None, :] - positions[:, None]  # (seq_len, seq_len)

        # ALiBi slopes (head-specific)
        num_heads = self.config.num_alibi_heads
        slopes = mx.array([2 ** (-8 * i / num_heads) for i in range(1, num_heads + 1)])

        # Compute bias: slope * relative_position
        # Shape: (1, num_heads, seq_len, seq_len)
        alibi_bias = slopes[None, :, None, None] * relative_pos[None, None, :, :]

        # Apply learned scale - squeeze from [1, 1, 12, 1, 1] to [1, 12, 1, 1]
        scale = self.alibi_scale
        if scale.ndim == 5:
            scale = scale.squeeze(1)  # [1, 1, 12, 1, 1] -> [1, 12, 1, 1]
        return alibi_bias * scale



class Emotion2vecModel(nn.Module):
    """Emotion2vec model for speech emotion recognition.

    This model extracts emotion-aware representations from raw audio.
    Can be used for:
    - Emotion classification (with classification head)
    - Emotion embedding extraction (for downstream tasks)
    """

    def __init__(self, config: Emotion2vecConfig | None = None):
        super().__init__()
        self.config = config or Emotion2vecConfig.base()

        # Local encoder (raw audio -> features)
        self.local_encoder = LocalEncoder(self.config)

        # Feature projection (512 -> 768)
        self.project_features = FeatureProjection(self.config)

        # Relative positional encoder
        self.relative_positional_encoder = RelativePositionalEncoder(self.config)

        # Extra tokens (prepended to sequence)
        self.extra_tokens = mx.zeros((1, self.config.num_extra_tokens, self.config.embed_dim))

        # Transformer encoder
        self.encoder = Emotion2vecEncoder(self.config)

        # Compiled forward pass (lazy initialization)
        self._compiled_forward = None

    def _forward_impl(self, audio: mx.array) -> mx.array:
        """Internal forward implementation for mx.compile().

        Args:
            audio: Raw audio (batch, samples)

        Returns:
            Hidden states (batch, seq_len, embed_dim)
        """
        # Local encoder: raw audio -> features
        x = self.local_encoder(audio)

        # Project to transformer dimension
        x = self.project_features(x)

        # Add relative positional encoding
        x = self.relative_positional_encoder(x)

        # Prepend extra tokens
        batch_size = x.shape[0]
        extra = mx.broadcast_to(
            self.extra_tokens,
            (batch_size, self.config.num_extra_tokens, self.config.embed_dim),
        )
        x = mx.concatenate([extra, x], axis=1)

        # Transformer encoder
        return self.encoder(x)


    def __call__(
        self,
        audio: mx.array,
        return_all_layers: bool = False,
        use_compile: bool = True,
    ) -> mx.array:
        """Extract emotion representations from audio.

        Args:
            audio: Raw audio (batch, samples) or mel spectrogram (batch, frames, features)
            return_all_layers: If True, return representations from all layers
            use_compile: If True, use mx.compile() for faster inference

        Returns:
            Emotion representations (batch, seq_len, embed_dim) or
            averaged over extra tokens (batch, embed_dim)
        """
        if use_compile:
            if self._compiled_forward is None:
                self._compiled_forward = mx.compile(self._forward_impl)
            return self._compiled_forward(audio)
        return self._forward_impl(audio)

    def extract_emotion(self, audio: mx.array) -> mx.array:
        """Extract emotion embedding (averaged over extra tokens).

        Args:
            audio: Raw audio (batch, samples)

        Returns:
            Emotion embedding (batch, embed_dim)
        """
        x = self(audio)
        # Average over extra tokens
        return x[:, :self.config.num_extra_tokens, :].mean(axis=1)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Emotion2vecConfig | None = None,
    ) -> "Emotion2vecModel":
        """Load model from converted MLX weights.

        Args:
            path: Path to directory containing:
                - weights.npz: MLX weights
                - config.json: Configuration (optional)

        Returns:
            Loaded model
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                config = Emotion2vecConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "weights.npz"
        if weights_path.exists():
            weights = dict(mx.load(str(weights_path)))
            model._load_weights(weights)

        return model

    def _load_weights(self, weights: dict[str, mx.array]) -> None:
        """Load weights from flat dictionary into model.

        Args:
            weights: Dictionary of weight arrays with dotted keys
        """
        # Convert flat dict to nested dict, handling list indices
        nested = {}
        for key, value in weights.items():
            parts = key.split(".")
            current = nested
            for _i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        # Convert numeric dict keys to lists (for layers)
        nested = self._convert_numeric_keys_to_lists(nested)

        self.update(nested)

    def _convert_numeric_keys_to_lists(self, d: dict) -> dict:
        """Recursively convert dict with numeric string keys to lists.

        This handles the case where encoder.blocks is a list of modules,
        and the weights have keys like 'blocks.0', 'blocks.1', etc.
        """
        if not isinstance(d, dict):
            return d

        # Check if all keys are numeric strings
        keys = list(d.keys())
        if keys and all(k.isdigit() for k in keys):
            # Convert to list
            max_idx = max(int(k) for k in keys)
            result = [None] * (max_idx + 1)
            for k, v in d.items():
                result[int(k)] = self._convert_numeric_keys_to_lists(v)
            return result
        # Recursively process
        return {k: self._convert_numeric_keys_to_lists(v) for k, v in d.items()}
