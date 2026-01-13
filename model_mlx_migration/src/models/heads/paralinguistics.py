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
Paralinguistics classification head for Zipformer encoder.

Predicts non-speech vocalization classes from encoder output using
attention pooling followed by classification layers.

Target: >96% accuracy (matching Whisper-AT baseline 96.96%)

Paralinguistics Classes (50 categories):
- Laughter, crying, sighing, breathing, coughing, etc.
- Fillers: um, uh, hmm, etc.
- Speaker states: hesitation, emphasis, etc.

Datasets:
- VocalSound: 21 non-speech classes
- Fillers corpus: filler word detection
- AudioSet: broad paralinguistic events

Reference:
- Whisper-AT achieves 96.96% on paralinguistics
- SUPERB uses VoxCeleb for speaker verification
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# Paralinguistic event inventory
PARALINGUISTIC_CLASSES: tuple[str, ...] = (
    # VocalSound core classes (21)
    "laughter",
    "sigh",
    "cough",
    "throat_clear",
    "sneeze",
    "sniff",
    "breathing",
    "yawn",
    "gasp",
    "groan",
    "cry",
    "scream",
    "moan",
    "hum",
    "whistle",
    "click",
    "babble",
    "hiccup",
    "burp",
    "wheeze",
    "chew",
    # Fillers (10)
    "um",
    "uh",
    "hmm",
    "ah",
    "er",
    "like",
    "you_know",
    "so",
    "well",
    "basically",
    # Speaker states (10)
    "hesitation",
    "emphasis",
    "question_intonation",
    "excitement",
    "disappointment",
    "surprise_vocal",
    "agreement",
    "disagreement",
    "thinking",
    "doubt",
    # Background/noise (5)
    "background_speech",
    "music_vocals",
    "applause",
    "cheering",
    "booing",
    # Special (4)
    "<silence>",
    "<speech>",
    "<noise>",
    "<unknown>",
)


@dataclass
class ParalinguisticsConfig:
    """Configuration for paralinguistics head."""

    # Input dimension from encoder
    encoder_dim: int = 384

    # Number of paralinguistic classes
    num_classes: int = len(PARALINGUISTIC_CLASSES)

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

    # Multi-label mode (detect multiple events per utterance)
    multi_label: bool = True

    # Detection threshold for multi-label mode
    detection_threshold: float = 0.5

    # Class names for reference
    class_names: tuple[str, ...] = PARALINGUISTIC_CLASSES


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
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Expand query for batch
        q = mx.broadcast_to(self.query, (batch_size, 1, embed_dim))

        # Reshape for multi-head attention
        q = mx.reshape(q, (batch_size, 1, self.num_heads, self.head_dim))
        k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch, heads, seq, dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Compute attention scores
        scale = self.head_dim ** -0.5
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

        # Apply mask if provided
        if mask is not None:
            mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
            scores = mx.where(mask, mx.full(scores.shape, -1e9), scores)

        # Softmax attention weights
        attn_weights = mx.softmax(scores, axis=-1)

        # Apply attention to values
        out = mx.matmul(attn_weights, v)

        # Reshape back
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (batch_size, embed_dim))

        # Output projection
        out = self.out_proj(out)

        return out


class ParalinguisticsHead(nn.Module):
    """
    Paralinguistics classification head for Zipformer encoder.

    Takes encoder output and predicts paralinguistic events per utterance.
    Supports both single-label and multi-label classification.

    Args:
        config: ParalinguisticsConfig instance with hyperparameters.
    """

    def __init__(self, config: ParalinguisticsConfig | None = None):
        super().__init__()
        if config is None:
            config = ParalinguisticsConfig()

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
        Predict paralinguistic events from encoder output.

        Args:
            encoder_out: Encoder output of shape (batch_size, seq_len, encoder_dim)
            encoder_lengths: Optional sequence lengths of shape (batch_size,)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        encoder_out.shape[0]
        seq_len = encoder_out.shape[1]

        # Create mask from lengths if provided
        mask = None
        if encoder_lengths is not None:
            positions = mx.arange(seq_len)
            mask = positions >= mx.expand_dims(encoder_lengths, axis=1)

        # Layer norm
        encoder_out = self.layer_norm(encoder_out)

        # Pool to utterance-level representation
        if self.pooling is not None:
            pooled = self.pooling(encoder_out, mask=mask)
        else:
            if mask is not None:
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
        Get predicted paralinguistic classes and probabilities.

        For single-label mode: returns argmax class
        For multi-label mode: returns all classes above threshold

        Args:
            encoder_out: Encoder output
            encoder_lengths: Optional sequence lengths

        Returns:
            Tuple of:
            - Predicted class indices (single-label) or binary mask (multi-label)
            - Class probabilities/scores
        """
        logits = self(encoder_out, encoder_lengths)

        if self.config.multi_label:
            # Sigmoid for multi-label
            probs = mx.sigmoid(logits)
            predictions = (probs > self.config.detection_threshold).astype(mx.int32)
        else:
            # Softmax for single-label
            probs = mx.softmax(logits, axis=-1)
            predictions = mx.argmax(logits, axis=-1)

        return predictions, probs

    def get_class_name(self, class_idx: int) -> str:
        """Get paralinguistic class name from index."""
        return self.config.class_names[class_idx]

    def decode_predictions(
        self,
        predictions: mx.array,
        probs: mx.array,
    ) -> list[list[tuple[str, float]]]:
        """
        Decode predictions to class names with confidence scores.

        Args:
            predictions: Binary mask of shape (batch, num_classes) for multi-label
                        or class indices of shape (batch,) for single-label
            probs: Probability scores

        Returns:
            List of (class_name, confidence) tuples per batch item
        """
        batch_size = predictions.shape[0]
        results = []

        if self.config.multi_label:
            for b in range(batch_size):
                detected = []
                for c in range(self.config.num_classes):
                    if predictions[b, c]:
                        detected.append(
                            (self.config.class_names[c], float(probs[b, c])),
                        )
                # Sort by confidence
                detected.sort(key=lambda x: x[1], reverse=True)
                results.append(detected)
        else:
            for b in range(batch_size):
                class_idx = int(predictions[b])
                conf = float(probs[b, class_idx])
                results.append([(self.config.class_names[class_idx], conf)])

        return results


def paralinguistics_loss(
    logits: mx.array,
    targets: mx.array,
    multi_label: bool = True,
    label_smoothing: float = 0.1,
    reduction: str = "mean",
) -> mx.array:
    """
    Loss for paralinguistics classification.

    For multi-label: binary cross-entropy per class
    For single-label: cross-entropy with label smoothing

    Args:
        logits: Predicted logits of shape (batch_size, num_classes)
        targets: Target labels (multi-label: binary mask, single-label: class indices)
        multi_label: Whether to use multi-label mode
        label_smoothing: Label smoothing factor for single-label mode
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    if multi_label:
        # Binary cross-entropy for multi-label
        probs = mx.sigmoid(logits)
        # Clamp for numerical stability
        eps = 1e-7
        probs = mx.clip(probs, eps, 1 - eps)

        # BCE loss
        targets_float = targets.astype(mx.float32)
        loss = -(
            targets_float * mx.log(probs) +
            (1 - targets_float) * mx.log(1 - probs)
        )
        loss = mx.mean(loss, axis=-1)  # Average over classes
    else:
        # Cross-entropy for single-label
        num_classes = logits.shape[-1]
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        if label_smoothing > 0:
            smooth_targets = mx.full(logits.shape, label_smoothing / num_classes)
            one_hot = mx.zeros_like(logits)
            batch_indices = mx.arange(logits.shape[0])
            one_hot = one_hot.at[batch_indices, targets].add(1.0)
            smooth_targets = smooth_targets + one_hot * (1.0 - label_smoothing)
            loss = -mx.sum(smooth_targets * log_probs, axis=-1)
        else:
            batch_indices = mx.arange(logits.shape[0])
            loss = -log_probs[batch_indices, targets]

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


class ParalinguisticsLoss(nn.Module):
    """Paralinguistics loss as nn.Module wrapper."""

    def __init__(
        self,
        multi_label: bool = True,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.multi_label = multi_label
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(
        self,
        logits: mx.array,
        targets: mx.array,
    ) -> mx.array:
        return paralinguistics_loss(
            logits=logits,
            targets=targets,
            multi_label=self.multi_label,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


def compute_paralinguistics_accuracy(
    predictions: mx.array,
    targets: mx.array,
    multi_label: bool = True,
) -> mx.array:
    """
    Compute accuracy for paralinguistics classification.

    For multi-label: F1 score (harmonic mean of precision and recall)
    For single-label: standard accuracy

    Args:
        predictions: Predicted labels
        targets: Target labels
        multi_label: Whether in multi-label mode

    Returns:
        Accuracy or F1 score
    """
    if multi_label:
        # F1 score for multi-label
        predictions = predictions.astype(mx.float32)
        targets = targets.astype(mx.float32)

        tp = mx.sum(predictions * targets)
        fp = mx.sum(predictions * (1 - targets))
        fn = mx.sum((1 - predictions) * targets)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
    # Standard accuracy for single-label
    correct = (predictions == targets).astype(mx.float32)
    return mx.mean(correct)
