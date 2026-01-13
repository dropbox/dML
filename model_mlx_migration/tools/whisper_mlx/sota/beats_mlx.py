"""BEATs MLX Implementation.

Audio Pre-Training with Acoustic Tokenizers.
Reference: https://github.com/microsoft/unilm/tree/master/beats

Architecture:
    - Patch embedding: 16x16 patches from log-mel spectrogram
    - Positional conv: Grouped Conv1d with weight normalization
    - 12 transformer layers with relative position bias (T5-style bucketing)
    - Deep norm scaling for stable training
"""

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .beats_config import BEATsConfig


def compute_relative_position_bucket(
    relative_position: mx.array,
    bidirectional: bool = True,
    num_buckets: int = 320,
    max_distance: int = 800,
) -> mx.array:
    """Compute relative position bucket indices (T5-style).

    Maps relative positions to bucket indices for efficient relative attention.

    Args:
        relative_position: Relative position matrix (query_len, key_len)
        bidirectional: Whether attention is bidirectional
        num_buckets: Number of position buckets
        max_distance: Maximum distance to consider

    Returns:
        Bucket indices (query_len, key_len)
    """
    relative_buckets = mx.zeros(relative_position.shape, dtype=mx.int32)

    if bidirectional:
        num_buckets //= 2
        relative_buckets = relative_buckets + (relative_position > 0).astype(mx.int32) * num_buckets
        relative_position = mx.abs(relative_position)
    else:
        relative_position = -mx.minimum(relative_position, mx.zeros(relative_position.shape, dtype=relative_position.dtype))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # Log-space bucketing for large distances
    relative_position_if_large = max_exact + (
        mx.log(relative_position.astype(mx.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(mx.int32)

    relative_position_if_large = mx.minimum(
        relative_position_if_large,
        mx.full(relative_position_if_large.shape, num_buckets - 1, dtype=mx.int32),
    )

    return relative_buckets + mx.where(
        is_small,
        relative_position.astype(mx.int32),
        relative_position_if_large,
    )



class PatchEmbedding(nn.Module):
    """Patch embedding using 2D convolution."""

    def __init__(self, config: BEATsConfig):
        super().__init__()
        self.config = config
        # Conv2d: (out_channels, in_channels, kernel_height, kernel_width)
        # Input: log-mel spectrogram (batch, 1, freq, time)
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=config.embed_dim,
            kernel_size=config.input_patch_size,
            stride=config.input_patch_size,
            bias=config.conv_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Extract patches from spectrogram.

        Args:
            x: Log-mel spectrogram (batch, 1, freq_bins, time_steps)

        Returns:
            Patches (batch, num_patches, embed_dim)
        """
        # MLX Conv2d: (N, H, W, C) format, need to adapt
        # Input expected as (batch, freq, time, 1) for MLX
        if x.ndim == 3:
            x = x[:, :, :, None]  # Add channel dim
        if x.shape[-1] == 1:
            pass  # Already (batch, freq, time, 1)
        else:
            x = x.transpose(0, 2, 3, 1)  # (batch, 1, freq, time) -> (batch, freq, time, 1)

        x = self.proj(x)  # (batch, freq//16, time//16, embed_dim)

        # Flatten spatial dimensions
        batch_size, h, w, c = x.shape
        return x.reshape(batch_size, h * w, c)



class PositionalConvEmbedding(nn.Module):
    """Positional conv embedding with weight normalization (pre-computed)."""

    def __init__(self, config: BEATsConfig):
        super().__init__()
        self.config = config
        kernel_size = config.conv_pos
        groups = config.conv_pos_groups
        group_size = config.encoder_embed_dim // groups

        # Pre-computed effective weight
        self.weight = mx.zeros((config.encoder_embed_dim, kernel_size, group_size))
        self.bias = mx.zeros((config.encoder_embed_dim,))
        self.groups = groups
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def __call__(self, x: mx.array) -> mx.array:
        """Apply positional convolution."""
        batch_size, seq_len, hidden_size = x.shape
        group_size = hidden_size // self.groups

        # Symmetric padding
        x_padded = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])

        # Manual grouped convolution
        outputs = []
        for g in range(self.groups):
            x_g = x_padded[:, :, g * group_size:(g + 1) * group_size]
            w_g = self.weight[g * group_size:(g + 1) * group_size]
            conv_out = mx.conv1d(x_g, w_g)
            outputs.append(conv_out)

        out = mx.concatenate(outputs, axis=-1)
        out = out[:, :-1, :]  # Trim to match input length
        out = out + self.bias
        return nn.gelu(out)



class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position bias (T5-style)."""

    def __init__(self, config: BEATsConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.encoder_embed_dim // self.num_heads
        self.embed_dim = config.encoder_embed_dim

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Relative position bias
        self.relative_attention_bias = nn.Embedding(
            config.num_buckets, config.encoder_attention_heads,
        )

        # GRU-relative position (grep) parameters
        self.grep_linear = nn.Linear(self.head_dim, 8)
        self.grep_a = mx.ones((1, self.num_heads, 1, 1))

        self.scale = self.head_dim ** -0.5

        # Deep norm scaling
        self.deep_norm_alpha = 1.0
        if config.deep_norm:
            self.deep_norm_alpha = (2.0 * config.encoder_layers) ** 0.25

    def compute_bias(self, query_length: int, key_length: int) -> mx.array:
        """Compute relative position bias."""
        context_position = mx.arange(query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]
        relative_position = memory_position - context_position

        bucket = compute_relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.config.num_buckets,
            max_distance=self.config.max_distance,
        )

        values = self.relative_attention_bias(bucket)  # (q, k, heads)
        return values.transpose(2, 0, 1)[None, :, :, :]  # (1, heads, q, k)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, _ = x.shape

        # BEATs uses alpha=32 for scaling
        alpha = 32.0

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply scaling to q (matching PyTorch: q *= scaling / alpha)
        q = q * self.scale / alpha

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores (no additional scaling needed since q is pre-scaled)
        attn = q @ k.transpose(0, 1, 3, 2)

        # Get relative position bias
        position_bias = self.compute_bias(seq_len, seq_len)  # (1, heads, q, k)

        # GRU relative position gating (grep)
        # For grep, we need to undo the scaling: q * alpha / self.scale
        query_layer = q * alpha / self.scale

        # grep_linear: head_dim -> 8, then reshape and sum
        grep_out = self.grep_linear(query_layer)  # (batch, heads, seq, 8)
        grep_out = grep_out.reshape(batch_size, self.num_heads, seq_len, 2, 4)
        grep_out = grep_out.sum(axis=-1)  # (batch, heads, seq, 2)
        grep_out = mx.sigmoid(grep_out)

        # Split into gate_a and gate_b
        gate_a = grep_out[:, :, :, 0:1]  # (batch, heads, seq, 1)
        gate_b = grep_out[:, :, :, 1:2]  # (batch, heads, seq, 1)

        # Compute gating factor: gate_a * (gate_b * grep_a - 1.0) + 2.0
        gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0  # (batch, heads, seq, 1)

        # Apply gating to position bias
        # position_bias is (1, heads, q, k), gate_a_1 is (batch, heads, seq, 1)
        attn_mask_rel_pos = gate_a_1 * position_bias  # broadcast over k dimension

        # Add gated position bias to attention
        attn = attn + attn_mask_rel_pos

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)



class BEATsEncoderLayer(nn.Module):
    """BEATs encoder layer (post-norm architecture)."""

    def __init__(self, config: BEATsConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = RelativeMultiHeadAttention(config, layer_idx)
        self.self_attn_layer_norm = nn.LayerNorm(config.encoder_embed_dim, eps=config.layer_norm_eps)

        self.fc1 = nn.Linear(config.encoder_embed_dim, config.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_embed_dim, config.encoder_embed_dim)
        self.final_layer_norm = nn.LayerNorm(config.encoder_embed_dim, eps=config.layer_norm_eps)

        # Deep norm scaling
        self.deep_norm_alpha = 1.0
        self.deep_norm_beta = 1.0
        if config.deep_norm:
            self.deep_norm_alpha = (2.0 * config.encoder_layers) ** 0.25
            self.deep_norm_beta = (8.0 * config.encoder_layers) ** -0.25

    def __call__(self, x: mx.array) -> mx.array:
        # Post-norm architecture (layer_norm_first = False)
        residual = x
        x = self.self_attn(x)
        x = residual * self.deep_norm_alpha + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        x = residual * self.deep_norm_alpha + x
        return self.final_layer_norm(x)



class BEATsEncoder(nn.Module):
    """BEATs encoder (transformer stack)."""

    def __init__(self, config: BEATsConfig):
        super().__init__()
        self.config = config
        self.pos_conv = PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim, eps=config.layer_norm_eps)
        self.layers = [BEATsEncoderLayer(config, i) for i in range(config.encoder_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        # Add positional encoding
        x = x + self.pos_conv(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        return self.layer_norm(x)



class BEATsModel(nn.Module):
    """BEATs model for audio understanding.

    Takes log-mel spectrogram and outputs contextualized features.
    """

    def __init__(self, config: BEATsConfig | None = None):
        super().__init__()
        self.config = config or BEATsConfig()

        # Patch embedding
        self.patch_embedding = PatchEmbedding(self.config)

        # Layer norm after patch embedding
        self.layer_norm = nn.LayerNorm(self.config.embed_dim, eps=self.config.layer_norm_eps)

        # Project patches to encoder dimension
        self.post_extract_proj = nn.Linear(self.config.embed_dim, self.config.encoder_embed_dim)

        # Transformer encoder
        self.encoder = BEATsEncoder(self.config)

        # Compiled forward pass (lazy initialization)
        self._compiled_forward = None

    def _forward_impl(self, fbank: mx.array) -> mx.array:
        """Internal forward implementation for mx.compile()."""
        # Ensure 4D input
        if fbank.ndim == 3:
            fbank = fbank[:, None, :, :]  # Add channel dim

        # Patch embedding
        x = self.patch_embedding(fbank)

        # Layer norm
        x = self.layer_norm(x)

        # Project to encoder dimension
        x = self.post_extract_proj(x)

        # Transformer encoder
        return self.encoder(x)


    def __call__(self, fbank: mx.array, use_compile: bool = True) -> mx.array:
        """Extract audio features.

        Args:
            fbank: Log-mel spectrogram (batch, freq_bins, time_frames)
                   or (batch, 1, freq_bins, time_frames)
            use_compile: If True, use mx.compile() for faster inference

        Returns:
            Features (batch, num_patches, encoder_embed_dim)
        """
        if use_compile:
            if self._compiled_forward is None:
                self._compiled_forward = mx.compile(self._forward_impl)
            return self._compiled_forward(fbank)
        return self._forward_impl(fbank)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: BEATsConfig | None = None,
    ) -> "BEATsModel":
        """Load model from converted MLX weights."""
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                config = BEATsConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "weights.npz"
        if weights_path.exists():
            weights = dict(mx.load(str(weights_path)))
            model._load_weights(weights)

        return model

    def _load_weights(self, weights: dict[str, mx.array]) -> None:
        """Load weights from flat dictionary."""
        nested = {}
        for key, value in weights.items():
            parts = key.split(".")
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        nested = self._convert_numeric_keys_to_lists(nested)
        self.update(nested)

    def _convert_numeric_keys_to_lists(self, d: dict) -> dict:
        """Convert dict with numeric string keys to lists."""
        if not isinstance(d, dict):
            return d

        keys = list(d.keys())
        if keys and all(k.isdigit() for k in keys):
            max_idx = max(int(k) for k in keys)
            result = [None] * (max_idx + 1)
            for k, v in d.items():
                result[int(k)] = self._convert_numeric_keys_to_lists(v)
            return result
        return {k: self._convert_numeric_keys_to_lists(v) for k, v in d.items()}
