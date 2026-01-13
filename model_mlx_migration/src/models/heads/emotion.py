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
Emotion classification head for Zipformer encoder.

Predicts emotion class from encoder output using attention pooling
followed by classification layers.

Target: >92% accuracy on CREMA-D/RAVDESS (matching Whisper baseline)

Emotion Classes (8):
- Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise, Contempt

Reference:
- Whisper-AT achieves 92.07% on emotion recognition
- SUPERB emotion benchmark uses IEMOCAP (9 classes)
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

_ARANGE_CACHE: dict[int, mx.array] = {}


def _arange_cached(n: int) -> mx.array:
    arr = _ARANGE_CACHE.get(n)
    if arr is None:
        arr = mx.arange(n)
        _ARANGE_CACHE[n] = arr
    return arr


@dataclass
class EmotionConfig:
    """Configuration for emotion head."""

    # Input dimension from encoder
    encoder_dim: int = 384

    # Number of emotion classes
    num_classes: int = 0

    # Hidden dimension for classifier
    hidden_dim: int = 256

    # Number of attention heads for pooling
    num_attention_heads: int = 4

    # Dropout rate (applied during training)
    dropout_rate: float = 0.1

    # Whether to use attention pooling vs mean pooling
    use_attention_pooling: bool = True

    # Label smoothing for training
    label_smoothing: float = 0.1

    # Class names for reference
    class_names: tuple[str, ...] = (
        "neutral",
        "happy",
        "sad",
        "angry",
        "fear",
        "disgust",
        "surprise",
        "contempt",
    )

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            self.num_classes = len(self.class_names)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence-to-vector.

    Uses multi-head self-attention to compute weighted sum of encoder
    frames, producing a fixed-size representation for classification.

    Args:
        embed_dim: Input embedding dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query vector (learned parameter)
        self.query = mx.random.normal(shape=(1, 1, embed_dim)) * 0.02
        self._cached_batch_size: int | None = None
        self._cached_q: mx.array | None = None

        # Key and value projections
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Compute attention-pooled representation.

        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask of shape (batch_size, seq_len), True = masked

        Returns:
            Pooled representation of shape (batch_size, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project keys and values
        k = self.key_proj(x)  # (batch, seq, embed_dim)
        v = self.value_proj(x)  # (batch, seq, embed_dim)

        # Expand query for batch
        if self._cached_q is None or self._cached_batch_size != batch_size:
            self._cached_q = mx.broadcast_to(self.query, (batch_size, 1, embed_dim))
            self._cached_batch_size = batch_size
        q = self._cached_q

        # Reshape for multi-head attention
        q = mx.reshape(q, (batch_size, 1, self.num_heads, self.head_dim))
        k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch, heads, seq, dim)
        q = mx.transpose(q, (0, 2, 1, 3))  # (batch, heads, 1, dim)
        k = mx.transpose(k, (0, 2, 1, 3))  # (batch, heads, seq, dim)
        v = mx.transpose(v, (0, 2, 1, 3))  # (batch, heads, seq, dim)

        # Compute attention scores
        scale = self.head_dim ** -0.5
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        # scores: (batch, heads, 1, seq)

        # Apply mask if provided
        if mask is not None:
            mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
            scores = mx.where(mask, mx.full(scores.shape, -1e9), scores)

        # Softmax attention weights
        attn_weights = mx.softmax(scores, axis=-1)

        # Apply attention to values
        out = mx.matmul(attn_weights, v)  # (batch, heads, 1, dim)

        # Reshape back
        out = mx.transpose(out, (0, 2, 1, 3))  # (batch, 1, heads, dim)
        out = mx.reshape(out, (batch_size, embed_dim))

        # Output projection
        out = self.out_proj(out)

        return out


class EmotionHead(nn.Module):
    """
    Emotion classification head for Zipformer encoder.

    Takes encoder output and predicts emotion class per utterance.
    Uses attention pooling to aggregate frame-level features into
    utterance-level representation.

    Args:
        config: EmotionConfig instance with hyperparameters.
    """

    def __init__(self, config: EmotionConfig | None = None):
        super().__init__()
        if config is None:
            config = EmotionConfig()

        self.config = config

        # Pooling layer
        if config.use_attention_pooling:
            self.pooling = AttentionPooling(
                config.encoder_dim,
                config.num_attention_heads,
            )
        else:
            self.pooling = None

        # Dropout for regularization (applied during training only)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Classification layers with dropout between hidden layers
        self.classifier = nn.Sequential(
            nn.Linear(config.encoder_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(config.encoder_dim)

    def __call__(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> mx.array:
        """
        Predict emotion from encoder output.

        Args:
            encoder_out: Encoder output of shape (batch_size, seq_len, encoder_dim)
            encoder_lengths: Optional sequence lengths of shape (batch_size,)

        Returns:
            Logits of shape (batch_size, num_classes)

        Note:
            The Zipformer encoder outputs (seq_len, batch_size, encoder_dim).
            Transpose before calling this method:
            encoder_out = mx.transpose(encoder_out, (1, 0, 2))
        """
        encoder_out.shape[0]
        seq_len = encoder_out.shape[1]

        # Create mask from lengths if provided
        mask = None
        if encoder_lengths is not None:
            # Create mask where True = position to ignore
            positions = _arange_cached(seq_len)
            mask = positions >= mx.expand_dims(encoder_lengths, axis=1)

        # Layer norm
        encoder_out = self.layer_norm(encoder_out)

        # Pool to utterance-level representation
        if self.pooling is not None:
            # Attention pooling
            pooled = self.pooling(encoder_out, mask=mask)
        else:
            # Mean pooling
            if mask is not None:
                # Masked mean
                mask_expanded = mx.expand_dims(~mask, axis=-1)
                encoder_out = encoder_out * mask_expanded
                lengths = mx.sum(~mask, axis=1, keepdims=True)
                pooled = mx.sum(encoder_out, axis=1) / mx.maximum(lengths, 1)
            else:
                pooled = mx.mean(encoder_out, axis=1)

        # Apply dropout after pooling (dropout in classifier handles hidden layers)
        pooled = self.dropout(pooled)

        # Classify
        logits = self.classifier(pooled)

        return logits

    def predict(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Get predicted emotion classes and probabilities.

        Args:
            encoder_out: Encoder output
            encoder_lengths: Optional sequence lengths

        Returns:
            Tuple of:
            - Predicted class indices of shape (batch_size,)
            - Class probabilities of shape (batch_size, num_classes)
        """
        logits = self(encoder_out, encoder_lengths)
        probs = mx.softmax(logits, axis=-1)
        predictions = mx.argmax(logits, axis=-1)
        return predictions, probs

    def get_class_name(self, class_idx: int) -> str:
        """Get emotion name from class index."""
        return self.config.class_names[class_idx]


def emotion_loss(
    logits: mx.array,
    targets: mx.array,
    label_smoothing: float = 0.1,
    reduction: str = "mean",
) -> mx.array:
    """
    Cross-entropy loss for emotion classification with label smoothing.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes)
        targets: Target class indices of shape (batch_size,)
        label_smoothing: Label smoothing factor (0 = no smoothing)
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    return nn.losses.cross_entropy(
        logits,
        targets,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


class EmotionLoss(nn.Module):
    """Emotion loss as nn.Module wrapper."""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(
        self,
        logits: mx.array,
        targets: mx.array,
    ) -> mx.array:
        return emotion_loss(
            logits=logits,
            targets=targets,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
