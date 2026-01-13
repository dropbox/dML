"""Wav2Vec2-XLSR MLX Implementation.

This module implements Wav2Vec2-XLSR for speech representation in MLX.
Reference: https://huggingface.co/facebook/wav2vec2-large-xlsr-53

Architecture:
    - Feature extractor: 7 Conv1d layers with layer norm
    - Feature projection: LayerNorm + Linear (512 -> 1024)
    - Positional encoding: Grouped Conv1d with weight normalization
    - 24 transformer encoder layers (pre-norm)
    - Used for phoneme recognition, ASR, and speech features
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .wav2vec2_xlsr_config import Wav2Vec2XLSRConfig


class FeatureExtractorLayer(nn.Module):
    """Single conv layer in feature extractor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
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

    def __init__(self, config: Wav2Vec2XLSRConfig):
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

    def __init__(self, config: Wav2Vec2XLSRConfig):
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

    def __init__(self, config: Wav2Vec2XLSRConfig):
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
        # PyTorch padding=P adds P zeros on both left and right sides
        # Then produces output of length = input + 2*P - K + 1 = input + 1
        # We trim by 1 at the end to get same length as input
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



class Wav2Vec2Attention(nn.Module):
    """Multi-head self-attention for Wav2Vec2."""

    def __init__(self, config: Wav2Vec2XLSRConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)



class Wav2Vec2FeedForward(nn.Module):
    """Feed-forward network for Wav2Vec2."""

    def __init__(self, config: Wav2Vec2XLSRConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.intermediate_dense(x)
        x = nn.gelu(x)
        return self.output_dense(x)


class Wav2Vec2EncoderLayer(nn.Module):
    """Wav2Vec2 encoder layer (pre-norm for stable layer norm)."""

    def __init__(self, config: Wav2Vec2XLSRConfig):
        super().__init__()
        self.attention = Wav2Vec2Attention(config)
        self.feed_forward = Wav2Vec2FeedForward(config)
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


class Wav2Vec2Encoder(nn.Module):
    """Wav2Vec2 encoder (transformer stack)."""

    def __init__(self, config: Wav2Vec2XLSRConfig):
        super().__init__()
        self.config = config
        self.pos_conv_embed = PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        # Add positional encoding
        x = x + self.pos_conv_embed(x)
        x = self.layer_norm(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        return x


class Wav2Vec2Model(nn.Module):
    """Wav2Vec2 model for speech representation.

    This is the base encoder without quantizer or classification head.
    """

    def __init__(self, config: Wav2Vec2XLSRConfig | None = None):
        super().__init__()
        self.config = config or Wav2Vec2XLSRConfig.xlsr_53()

        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_projection = FeatureProjection(self.config)
        self.encoder = Wav2Vec2Encoder(self.config)

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
        config: Wav2Vec2XLSRConfig | None = None,
    ) -> "Wav2Vec2Model":
        """Load model from converted MLX weights."""
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                config = Wav2Vec2XLSRConfig.from_dict(config_dict)

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


class Wav2Vec2ForSequenceClassification(nn.Module):
    """Wav2Vec2 for sequence classification (e.g., speech emotion recognition).

    Architecture:
        - Base Wav2Vec2 encoder
        - Mean pooling over sequence
        - Projector: Linear (hidden_size -> hidden_size)
        - Classifier: Linear (hidden_size -> num_labels)
    """

    def __init__(self, config: Wav2Vec2XLSRConfig | None = None, num_labels: int = 8):
        super().__init__()
        self.config = config or Wav2Vec2XLSRConfig.xlsr_53()
        self.num_labels = num_labels

        # Base encoder
        self.wav2vec2 = Wav2Vec2Model(self.config)

        # Classification head
        self.projector = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def __call__(self, audio: mx.array) -> mx.array:
        """Classify audio.

        Args:
            audio: Raw audio (batch, samples)

        Returns:
            Logits (batch, num_labels)
        """
        # Get encoder hidden states
        hidden_states = self.wav2vec2(audio)  # (batch, frames, hidden_size)

        # Mean pooling
        pooled = hidden_states.mean(axis=1)  # (batch, hidden_size)

        # Classification
        x = self.projector(pooled)
        x = nn.tanh(x)
        return self.classifier(x)


    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Wav2Vec2XLSRConfig | None = None,
        num_labels: int = 8,
    ) -> "Wav2Vec2ForSequenceClassification":
        """Load model from converted MLX weights."""
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                config = Wav2Vec2XLSRConfig.from_dict(config_dict)
                num_labels = len(config_dict.get("id2label", {})) or num_labels

        # Create model
        model = cls(config, num_labels)

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
