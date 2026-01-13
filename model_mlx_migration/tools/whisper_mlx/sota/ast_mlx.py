"""Audio Spectrogram Transformer (AST) MLX Implementation.

This module implements AST for audio classification in MLX.
Paper: https://arxiv.org/abs/2104.01778
Checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593

Architecture:
    - Vision Transformer (ViT) adapted for audio spectrograms
    - Input: Mel spectrogram (batch, freq, time) or (batch, 1, freq, time)
    - Patch embedding: Conv2d projection
    - 12 transformer encoder layers (pre-norm)
    - CLS token pooling for classification
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .ast_config import ASTConfig


class ASTEmbeddings(nn.Module):
    """AST Embeddings: patch projection + positional embeddings."""

    def __init__(self, config: ASTConfig):
        super().__init__()
        self.config = config

        # CLS and distillation tokens
        self.cls_token = mx.zeros((1, 1, config.hidden_size))
        self.distillation_token = mx.zeros((1, 1, config.hidden_size))

        # Patch projection: Conv2d(1, hidden_size, kernel=patch_size, stride=(time_stride, freq_stride))
        # MLX Conv2d: weight shape is (out_channels, H, W, in_channels)
        self.patch_embeddings = nn.Conv2d(
            in_channels=1,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=(config.time_stride, config.frequency_stride),
            bias=True,
        )

        # Position embeddings: (1, num_patches + 2, hidden_size)
        # +2 for CLS and distillation tokens
        num_positions = config.num_patches + 2
        self.position_embeddings = mx.zeros((1, num_positions, config.hidden_size))

    def __call__(self, x: mx.array) -> mx.array:
        """Embed mel spectrogram into patch tokens.

        Args:
            x: Input spectrogram (batch, freq, time) or (batch, 1, freq, time)

        Returns:
            Embedded tokens (batch, num_patches + 2, hidden_size)
        """
        batch_size = x.shape[0]

        # Ensure 4D input: (batch, channels, freq, time)
        if x.ndim == 3:
            x = x[:, None, :, :]  # Add channel dimension

        # HuggingFace AST does: unsqueeze(1) then transpose(2,3) to get (B, 1, time, freq)
        # Input: (B, 1, freq, time) -> after HF transpose: (B, 1, time, freq)
        # MLX expects (batch, H, W, channels) so: (B, time, freq, 1)
        # Transpose from (B, C, freq, time) to (B, time, freq, C)
        x = mx.transpose(x, (0, 3, 2, 1))

        # Patch projection
        # Input: (B, time, freq, 1), Output: (B, H', W', hidden_size)
        # where H' = (time - patch_size) // time_stride + 1 = (1024 - 16) // 10 + 1 = 101
        #       W' = (freq - patch_size) // freq_stride + 1 = (128 - 16) // 10 + 1 = 12
        x = self.patch_embeddings(x)

        # Flatten spatial dims: (batch, H' * W', hidden_size)
        x = x.reshape(batch_size, -1, self.config.hidden_size)

        # Prepend CLS and distillation tokens
        cls_tokens = mx.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        dist_tokens = mx.broadcast_to(self.distillation_token, (batch_size, 1, self.config.hidden_size))
        x = mx.concatenate([cls_tokens, dist_tokens, x], axis=1)

        # Add position embeddings
        return x + self.position_embeddings[:, :x.shape[1], :]



class ASTSelfAttention(nn.Module):
    """Multi-head self-attention for AST."""

    def __init__(self, config: ASTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Q, K, V projections
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)

        # Output projection
        self.output_dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention: (batch, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Reshape back: (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        return self.output_dense(attn_output)



class ASTMLP(nn.Module):
    """MLP (Feed-Forward Network) for AST."""

    def __init__(self, config: ASTConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply feed-forward network.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        x = self.dense1(x)
        x = nn.gelu(x)
        return self.dense2(x)


class ASTEncoderLayer(nn.Module):
    """Single transformer encoder layer for AST (pre-norm architecture)."""

    def __init__(self, config: ASTConfig):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = ASTSelfAttention(config)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ASTMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply transformer encoder layer.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        # Pre-norm self-attention with residual
        normed = self.layernorm_before(x)
        attn_output = self.attention(normed)
        x = x + attn_output

        # Pre-norm MLP with residual
        normed = self.layernorm_after(x)
        mlp_output = self.mlp(normed)
        return x + mlp_output



class ASTEncoder(nn.Module):
    """Transformer encoder stack for AST."""

    def __init__(self, config: ASTConfig):
        super().__init__()
        self.layers = [ASTEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        """Apply all encoder layers.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ASTModel(nn.Module):
    """Audio Spectrogram Transformer base model."""

    def __init__(self, config: ASTConfig):
        super().__init__()
        self.config = config
        self.embeddings = ASTEmbeddings(config)
        self.encoder = ASTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through AST.

        Args:
            x: Input spectrogram (batch, freq, time) or (batch, 1, freq, time)

        Returns:
            Hidden states (batch, seq_len, hidden_size)
        """
        x = self.embeddings(x)
        x = self.encoder(x)
        return self.layernorm(x)


class ASTClassifier(nn.Module):
    """Classification head for AST."""

    def __init__(self, config: ASTConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, x: mx.array) -> mx.array:
        """Classify from pooled hidden states.

        Args:
            x: Pooled hidden state (batch, hidden_size)

        Returns:
            Logits (batch, num_labels)
        """
        x = self.layernorm(x)
        return self.dense(x)


class ASTForAudioClassification(nn.Module):
    """AST model for audio classification (e.g., AudioSet).

    Uses average of CLS and distillation token outputs.
    """

    def __init__(self, config: ASTConfig | None = None):
        super().__init__()
        self.config = config or ASTConfig.audioset()
        self.audio_spectrogram_transformer = ASTModel(self.config)
        self.classifier = ASTClassifier(self.config)

        # Label mapping (loaded from file)
        self.id2label: dict[int, str] | None = None

        # Compiled forward pass
        self._compiled_forward = None

    def _forward_impl(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Internal forward implementation for compilation."""
        # Get hidden states
        hidden_states = self.audio_spectrogram_transformer(x)

        # Pool CLS and distillation tokens (positions 0 and 1)
        cls_output = hidden_states[:, 0, :]
        dist_output = hidden_states[:, 1, :]
        pooled = (cls_output + dist_output) / 2.0

        # Classify
        logits = self.classifier(pooled)

        return logits, pooled

    def __call__(
        self,
        x: mx.array,
        return_hidden: bool = False,
        use_compile: bool = True,
    ) -> mx.array:
        """Classify audio from mel spectrogram.

        Args:
            x: Mel spectrogram (batch, freq, time) or (batch, 1, freq, time)
            return_hidden: If True, also return pooled hidden states
            use_compile: If True, use compiled forward pass

        Returns:
            Logits (batch, num_labels) or (logits, hidden_states) if return_hidden
        """
        if use_compile:
            if self._compiled_forward is None:
                self._compiled_forward = mx.compile(self._forward_impl)
            logits, pooled = self._compiled_forward(x)
        else:
            logits, pooled = self._forward_impl(x)

        if return_hidden:
            return logits, pooled
        return logits

    def classify(self, x: mx.array) -> tuple[mx.array, mx.array, list]:
        """Classify audio and return predictions with labels.

        Args:
            x: Mel spectrogram (batch, freq, time)

        Returns:
            Tuple of (logits, predicted_indices, label_names)
        """
        logits = self(x)
        predictions = mx.argmax(logits, axis=-1)

        if self.id2label:
            labels = [self.id2label.get(int(p), "unknown") for p in predictions.tolist()]
        else:
            labels = [str(int(p)) for p in predictions.tolist()]

        return logits, predictions, labels

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: ASTConfig | None = None,
    ) -> "ASTForAudioClassification":
        """Load model from converted MLX weights.

        Args:
            path: Path to directory containing:
                - weights.npz: MLX weights
                - config.json: Configuration (optional)
                - labels.json: Label mapping (optional)

        Returns:
            Loaded model
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                config = ASTConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "weights.npz"
        if weights_path.exists():
            weights = dict(mx.load(str(weights_path)))
            model.load_weights_from_dict(weights)

        # Load labels
        labels_path = path / "labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                model.id2label = {int(k): v for k, v in json.load(f).items()}

        return model

    def load_weights_from_dict(self, weights: dict[str, mx.array]) -> None:
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

        # Convert numeric dict keys to lists (for encoder.layers)
        nested = self._convert_numeric_keys_to_lists(nested)

        self.update(nested)

    def _convert_numeric_keys_to_lists(self, d: dict) -> dict:
        """Recursively convert dict with numeric string keys to lists.

        This handles the case where encoder.layers is a list of modules,
        and the weights have keys like 'layers.0', 'layers.1', etc.
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
