"""WavLM MLX Implementation.

This module implements WavLM for speech representation in MLX.
Reference: https://huggingface.co/microsoft/wavlm-large

Architecture:
    - Feature extractor: 7 Conv1d layers with layer norm
    - Feature projection: LayerNorm + Linear (512 -> 1024)
    - Positional encoding: Grouped Conv1d with weight normalization
    - 24 transformer encoder layers with bucket-based relative position bias
    - GRU-style gating for relative position embeddings
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .wavlm_config import WavLMConfig


class FeatureExtractorLayer(nn.Module):
    """Single conv layer in feature extractor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply conv layer.

        Args:
            x: Input (batch, length, channels)

        Returns:
            Output (batch, new_length, out_channels)
        """
        x = self.conv(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return nn.gelu(x)


class FeatureExtractor(nn.Module):
    """Feature extractor: 7 Conv1d layers."""

    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        self.conv_layers = []

        in_ch = 1
        for i in range(len(config.conv_dim)):
            out_ch = config.conv_dim[i]
            kernel_size = config.conv_kernel[i]
            stride = config.conv_stride[i]
            layer = FeatureExtractorLayer(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                bias=config.conv_bias,
                use_layer_norm=(config.feat_extract_norm == "layer"),
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
        if x.ndim == 2:
            x = x[:, :, None]

        for conv in self.conv_layers:
            x = conv(x)

        return x


class FeatureProjection(nn.Module):
    """Project features to transformer dimension."""

    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        return self.projection(x)


class PositionalConvEmbedding(nn.Module):
    """Positional encoding using grouped convolution.

    The weight is pre-computed from weight normalization during conversion.
    """

    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        kernel_size = config.num_conv_pos_embeddings
        groups = config.num_conv_pos_embedding_groups

        # Pre-computed effective weight (after weight normalization)
        # Shape: (hidden_size, kernel_size, hidden_size // groups) in MLX format
        self.weight = mx.zeros((config.hidden_size, kernel_size, config.hidden_size // groups))
        self.bias = mx.zeros((config.hidden_size,))
        self.groups = groups
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def __call__(self, x: mx.array) -> mx.array:
        """Apply positional convolution.

        Args:
            x: Input (batch, seq_len, hidden_size)

        Returns:
            Output (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        group_size = hidden_size // self.groups

        # Pad input - use symmetric padding like PyTorch F.conv1d(padding=K//2)
        x_padded = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])

        # Manual grouped convolution
        outputs = []
        for g in range(self.groups):
            # Get group input and weight
            x_g = x_padded[:, :, g * group_size:(g + 1) * group_size]
            w_g = self.weight[g * group_size:(g + 1) * group_size]  # (group_size, kernel, group_size)

            # Apply conv1d
            conv_out = mx.conv1d(x_g, w_g)
            outputs.append(conv_out)

        out = mx.concatenate(outputs, axis=-1)

        # Trim by 1 to match PyTorch output length
        out = out[:, :-1, :]

        out = out + self.bias
        return nn.gelu(out)



def compute_bucket_indices(
    seq_len: int,
    num_buckets: int,
    max_distance: int,
) -> mx.array:
    """Compute relative position bucket indices.

    Maps relative positions to bucket indices using logarithmic bucketing.

    Args:
        seq_len: Sequence length
        num_buckets: Number of buckets (320 for WavLM)
        max_distance: Maximum distance for bucketing (800 for WavLM)

    Returns:
        Bucket indices of shape (seq_len, seq_len)
    """
    import math

    # Create position indices
    context_position = mx.arange(seq_len)[:, None]
    memory_position = mx.arange(seq_len)[None, :]

    # Relative position
    relative_position = memory_position - context_position  # (seq_len, seq_len)

    # Initialize buckets - positive and negative have separate buckets
    num_buckets_half = num_buckets // 2

    # Negative positions
    negative_mask = relative_position < 0
    relative_position_abs = mx.abs(relative_position)

    # Half buckets for exact positions, half for log-scaled positions
    max_exact = num_buckets_half // 2

    # Small positions get exact buckets
    is_small = relative_position_abs < max_exact
    small_buckets = relative_position_abs

    # Larger positions get log-scaled buckets
    relative_position_if_large = max_exact + (
        mx.log(relative_position_abs.astype(mx.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets_half - max_exact)
    ).astype(mx.int32)

    # Clamp to max bucket
    relative_position_if_large = mx.minimum(
        relative_position_if_large, mx.array(num_buckets_half - 1),
    )

    # Choose based on whether position is small or large
    positive_buckets = mx.where(is_small, small_buckets, relative_position_if_large)

    # Apply offset for negative positions
    return mx.where(negative_mask, num_buckets_half + positive_buckets, positive_buckets)



class WavLMAttention(nn.Module):
    """Multi-head self-attention with bucket-based relative position bias and GRU gating."""

    def __init__(self, config: WavLMConfig, has_rel_pos: bool = True, shared_rel_attn_embed: nn.Embedding | None = None):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.has_rel_pos = has_rel_pos

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.scale = self.head_dim ** -0.5

        if has_rel_pos:
            # Use shared relative position embedding (only layer 0 creates it)
            # Other layers receive it via shared_rel_attn_embed parameter
            if shared_rel_attn_embed is not None:
                self.rel_attn_embed = shared_rel_attn_embed
            else:
                # Layer 0 creates the embedding
                self.rel_attn_embed = nn.Embedding(config.num_buckets, config.num_attention_heads)

            # GRU-style gating (each layer has its own)
            # gru_rel_pos_linear: Linear(head_dim, head_dim // 8)
            # Output is split into gate (sigmoid) and a (tanh)
            self.gru_rel_pos_linear = nn.Linear(config.head_dim, config.head_dim // 8)
            # gru_rel_pos_const: (1, num_heads, 1, 1) - per-head learnable constant
            self.gru_rel_pos_const = mx.zeros((1, config.num_attention_heads, 1, 1))

    def compute_rel_pos_bias(
        self,
        query: mx.array,
        seq_len: int,
    ) -> mx.array:
        """Compute relative position bias with GRU gating.

        Args:
            query: Query tensor (batch, num_heads, seq_len, head_dim)
            seq_len: Sequence length

        Returns:
            Relative position bias (batch, num_heads, seq_len, seq_len)
        """
        # Get bucket indices
        bucket_indices = compute_bucket_indices(
            seq_len, self.config.num_buckets, self.config.max_bucket_distance,
        )  # (seq_len, seq_len)

        # Get position embeddings
        pos_embed = self.rel_attn_embed(bucket_indices)  # (seq_len, seq_len, num_heads)
        pos_embed = pos_embed.transpose(2, 0, 1)  # (num_heads, seq_len, seq_len)

        # GRU-style gating
        # Apply linear to query: (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim // 8)
        gate_input = self.gru_rel_pos_linear(query)

        # Split into gate and a components (each head_dim // 16)
        half_dim = gate_input.shape[-1] // 2
        gate = mx.sigmoid(gate_input[..., :half_dim])  # (batch, num_heads, seq_len, head_dim // 16)
        a = mx.tanh(gate_input[..., half_dim:])  # (batch, num_heads, seq_len, head_dim // 16)

        # Compute gated output: gate * a + (1 - gate) * const
        gru_out = gate * a + (1 - gate) * self.gru_rel_pos_const  # (batch, num_heads, seq_len, head_dim // 16)

        # Sum over last dimension to get scalar per (batch, head, position)
        gru_weight = gru_out.sum(axis=-1, keepdims=True)  # (batch, num_heads, seq_len, 1)

        # Apply gating to position bias
        # pos_embed: (num_heads, seq_len, seq_len)
        # gru_weight: (batch, num_heads, seq_len, 1)
        # Result: (batch, num_heads, seq_len, seq_len)
        return pos_embed[None, :, :, :] * gru_weight


    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Add relative position bias if enabled
        if self.has_rel_pos:
            pos_bias = self.compute_rel_pos_bias(q, seq_len)
            attn = attn + pos_bias

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)



class WavLMFeedForward(nn.Module):
    """Feed-forward network for WavLM."""

    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.intermediate_dense(x)
        x = nn.gelu(x)
        return self.output_dense(x)


class WavLMEncoderLayer(nn.Module):
    """WavLM encoder layer (pre-norm / stable layer norm)."""

    def __init__(self, config: WavLMConfig, has_rel_pos: bool = True, shared_rel_attn_embed: nn.Embedding | None = None):
        super().__init__()
        self.attention = WavLMAttention(config, has_rel_pos=has_rel_pos, shared_rel_attn_embed=shared_rel_attn_embed)
        self.feed_forward = WavLMFeedForward(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.do_stable_layer_norm = config.do_stable_layer_norm

    def __call__(self, x: mx.array) -> mx.array:
        if self.do_stable_layer_norm:
            # Pre-norm (stable layer norm)
            residual = x
            x = self.layer_norm(x)
            x = self.attention(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.feed_forward(x)
            x = residual + x
        else:
            # Post-norm
            residual = x
            x = self.attention(x)
            x = residual + x
            x = self.layer_norm(x)

            residual = x
            x = self.feed_forward(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x


class WavLMEncoder(nn.Module):
    """WavLM encoder (transformer stack)."""

    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        self.pos_conv_embed = PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Create shared relative attention embedding (only used by layer 0's weights)
        # All layers share this embedding but have their own GRU gating
        nn.Embedding(config.num_buckets, config.num_attention_heads)

        # Create layers - layer 0 owns the embedding, others share it
        self.layers = []
        for i in range(config.num_hidden_layers):
            if i == 0:
                # Layer 0 creates and owns the embedding
                layer = WavLMEncoderLayer(config, has_rel_pos=True, shared_rel_attn_embed=None)
            else:
                # Other layers share layer 0's embedding
                layer = WavLMEncoderLayer(config, has_rel_pos=True, shared_rel_attn_embed=self.layers[0].attention.rel_attn_embed)
            self.layers.append(layer)

    def __call__(self, x: mx.array) -> mx.array:
        # Add positional encoding
        x = x + self.pos_conv_embed(x)
        x = self.layer_norm(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        return x


class WavLMModel(nn.Module):
    """WavLM model for speech representation.

    This is the base encoder without classification head.
    """

    def __init__(self, config: WavLMConfig | None = None):
        super().__init__()
        self.config = config or WavLMConfig.wavlm_large()

        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_projection = FeatureProjection(self.config)
        self.encoder = WavLMEncoder(self.config)

        # Compiled forward pass (lazy initialization)
        self._compiled_forward = None

    def _forward_impl(self, audio: mx.array) -> mx.array:
        """Internal forward implementation for mx.compile()."""
        # Feature extraction
        x = self.feature_extractor(audio)

        # Project to transformer dimension
        x = self.feature_projection(x)

        # Transformer encoder
        return self.encoder(x)


    def __call__(self, audio: mx.array, use_compile: bool = True) -> mx.array:
        """Extract speech representations.

        Args:
            audio: Raw audio (batch, samples)
            use_compile: If True, use mx.compile() for faster inference

        Returns:
            Features (batch, frames, hidden_size)
        """
        if use_compile:
            if self._compiled_forward is None:
                self._compiled_forward = mx.compile(self._forward_impl)
            return self._compiled_forward(audio)
        return self._forward_impl(audio)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: WavLMConfig | None = None,
    ) -> "WavLMModel":
        """Load model from converted MLX weights."""
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                config = WavLMConfig.from_dict(config_dict)

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
