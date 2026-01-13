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
Singing detection and technique classification head for Zipformer encoder.

Predicts:
1. Binary singing vs speech detection
2. Singing technique classification (10 techniques)

Target: >95% binary accuracy, >90% technique classification

Singing Techniques (10):
- Belt, Falsetto, Head voice, Chest voice, Mixed voice
- Vibrato, Straight tone, Breathy, Twang, Opera

Datasets:
- OpenSinger: Chinese singing dataset
- VocalSet: 11 singing techniques
- DAMP: Digital Archive of Mobile Performances
- NUS-48E: English singing

Reference:
- Singing voice detection: ~95% typical accuracy
- Technique classification more challenging: ~85-90%
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# Singing technique inventory
SINGING_TECHNIQUES: tuple[str, ...] = (
    "belt",           # Powerful chest-dominant technique
    "falsetto",       # Light, airy head register
    "head_voice",     # Connected head register
    "chest_voice",    # Full chest register
    "mixed_voice",    # Blend of chest and head
    "vibrato",        # Oscillating pitch
    "straight_tone",  # No vibrato
    "breathy",        # Airy tone with airflow
    "twang",          # Bright, focused tone
    "opera",          # Classical operatic style
)


@dataclass
class SingingConfig:
    """Configuration for singing head."""

    # Input dimension from encoder
    encoder_dim: int = 384

    # Number of singing technique classes
    num_techniques: int = len(SINGING_TECHNIQUES)

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

    # Detection threshold for binary singing detection
    singing_threshold: float = 0.5

    # Technique confidence threshold
    technique_threshold: float = 0.3

    # Multi-technique mode (can have multiple techniques)
    multi_technique: bool = True

    # Technique names for reference
    technique_names: tuple[str, ...] = SINGING_TECHNIQUES


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence-to-vector."""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = mx.random.normal(shape=(1, 1, embed_dim)) * 0.02
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len, embed_dim = x.shape

        k = self.key_proj(x)
        v = self.value_proj(x)
        q = mx.broadcast_to(self.query, (batch_size, 1, embed_dim))

        q = mx.reshape(q, (batch_size, 1, self.num_heads, self.head_dim))
        k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        scale = self.head_dim ** -0.5
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

        if mask is not None:
            mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
            scores = mx.where(mask, mx.full(scores.shape, -1e9), scores)

        attn_weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn_weights, v)

        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (batch_size, embed_dim))
        out = self.out_proj(out)

        return out


class SingingHead(nn.Module):
    """
    Singing detection and technique classification head.

    Two-stage output:
    1. Binary singing vs speech detection
    2. If singing detected, technique classification

    Args:
        config: SingingConfig instance with hyperparameters.
    """

    def __init__(self, config: SingingConfig | None = None):
        super().__init__()
        if config is None:
            config = SingingConfig()

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

        # Shared hidden layer with dropout between layers
        self.shared_layers = nn.Sequential(
            nn.Linear(config.encoder_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
        )

        # Binary singing detection head
        self.singing_classifier = nn.Linear(config.hidden_dim, 1)

        # Technique classification head
        self.technique_classifier = nn.Linear(config.hidden_dim, config.num_techniques)

        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(config.encoder_dim)

    def __call__(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Predict singing detection and technique from encoder output.

        Args:
            encoder_out: Encoder output of shape (batch_size, seq_len, encoder_dim)
            encoder_lengths: Optional sequence lengths of shape (batch_size,)

        Returns:
            Tuple of:
            - singing_logits: Binary singing logits of shape (batch_size, 1)
            - technique_logits: Technique logits of shape (batch_size, num_techniques)
        """
        encoder_out.shape[0]
        seq_len = encoder_out.shape[1]

        mask = None
        if encoder_lengths is not None:
            positions = mx.arange(seq_len)
            mask = positions >= mx.expand_dims(encoder_lengths, axis=1)

        encoder_out = self.layer_norm(encoder_out)

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

        # Apply dropout after pooling
        pooled = self.dropout(pooled)

        # Shared features
        features = self.shared_layers(pooled)

        # Binary detection
        singing_logits = self.singing_classifier(features)

        # Technique classification
        technique_logits = self.technique_classifier(features)

        return singing_logits, technique_logits

    def predict(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Get predictions for singing detection and techniques.

        Args:
            encoder_out: Encoder output
            encoder_lengths: Optional sequence lengths

        Returns:
            Tuple of:
            - is_singing: Binary prediction (batch,)
            - singing_prob: Probability of singing (batch,)
            - technique_preds: Technique predictions (batch, num_techniques) if multi-technique
                              or (batch,) if single-technique
            - technique_probs: Technique probabilities (batch, num_techniques)
        """
        singing_logits, technique_logits = self(encoder_out, encoder_lengths)

        # Binary singing detection
        singing_prob = mx.sigmoid(singing_logits.squeeze(-1))
        is_singing = (singing_prob > self.config.singing_threshold).astype(mx.int32)

        # Technique classification
        if self.config.multi_technique:
            technique_probs = mx.sigmoid(technique_logits)
            technique_preds = (technique_probs > self.config.technique_threshold).astype(mx.int32)
        else:
            technique_probs = mx.softmax(technique_logits, axis=-1)
            technique_preds = mx.argmax(technique_logits, axis=-1)

        return is_singing, singing_prob, technique_preds, technique_probs

    def get_technique_name(self, technique_idx: int) -> str:
        """Get technique name from index."""
        return self.config.technique_names[technique_idx]

    def decode_predictions(
        self,
        is_singing: mx.array,
        singing_prob: mx.array,
        technique_preds: mx.array,
        technique_probs: mx.array,
    ) -> list[dict]:
        """
        Decode predictions to human-readable format.

        Args:
            is_singing: Binary singing predictions
            singing_prob: Singing probabilities
            technique_preds: Technique predictions
            technique_probs: Technique probabilities

        Returns:
            List of result dicts per batch item
        """
        batch_size = is_singing.shape[0]
        results = []

        for b in range(batch_size):
            result = {
                "is_singing": bool(is_singing[b]),
                "singing_confidence": float(singing_prob[b]),
                "techniques": [],
            }

            if result["is_singing"]:
                if self.config.multi_technique:
                    # Multi-technique: list all detected
                    for t in range(self.config.num_techniques):
                        if technique_preds[b, t]:
                            result["techniques"].append({
                                "name": self.config.technique_names[t],
                                "confidence": float(technique_probs[b, t]),
                            })
                    # Sort by confidence
                    result["techniques"].sort(key=lambda x: x["confidence"], reverse=True)
                else:
                    # Single technique: argmax
                    t = int(technique_preds[b])
                    result["techniques"].append({
                        "name": self.config.technique_names[t],
                        "confidence": float(technique_probs[b, t]),
                    })

            results.append(result)

        return results


def singing_loss(
    singing_logits: mx.array,
    technique_logits: mx.array,
    singing_targets: mx.array,
    technique_targets: mx.array,
    multi_technique: bool = True,
    label_smoothing: float = 0.1,
    technique_weight: float = 0.5,
    reduction: str = "mean",
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Combined loss for singing detection and technique classification.

    Args:
        singing_logits: Binary singing logits (batch, 1)
        technique_logits: Technique logits (batch, num_techniques)
        singing_targets: Binary singing targets (batch,)
        technique_targets: Technique targets (multi-technique: binary mask, single: indices)
        multi_technique: Whether multi-technique mode
        label_smoothing: Label smoothing for technique classification
        technique_weight: Weight for technique loss relative to singing loss
        reduction: "mean", "sum", or "none"

    Returns:
        Tuple of (total_loss, singing_loss, technique_loss)
    """
    # Binary cross-entropy for singing detection
    singing_logits = singing_logits.squeeze(-1)
    singing_targets_float = singing_targets.astype(mx.float32)
    singing_prob = mx.sigmoid(singing_logits)
    eps = 1e-7
    singing_prob = mx.clip(singing_prob, eps, 1 - eps)

    singing_loss_val = -(
        singing_targets_float * mx.log(singing_prob) +
        (1 - singing_targets_float) * mx.log(1 - singing_prob)
    )

    # Technique loss (only for singing samples)
    if multi_technique:
        # BCE for multi-technique
        tech_prob = mx.sigmoid(technique_logits)
        tech_prob = mx.clip(tech_prob, eps, 1 - eps)
        tech_targets_float = technique_targets.astype(mx.float32)
        technique_loss_val = -(
            tech_targets_float * mx.log(tech_prob) +
            (1 - tech_targets_float) * mx.log(1 - tech_prob)
        )
        technique_loss_val = mx.mean(technique_loss_val, axis=-1)  # Average over techniques
    else:
        # CE for single-technique
        num_classes = technique_logits.shape[-1]
        log_probs = technique_logits - mx.logsumexp(technique_logits, axis=-1, keepdims=True)

        if label_smoothing > 0:
            smooth_targets = mx.full(technique_logits.shape, label_smoothing / num_classes)
            one_hot = mx.zeros_like(technique_logits)
            batch_indices = mx.arange(technique_logits.shape[0])
            one_hot = one_hot.at[batch_indices, technique_targets].add(1.0)
            smooth_targets = smooth_targets + one_hot * (1.0 - label_smoothing)
            technique_loss_val = -mx.sum(smooth_targets * log_probs, axis=-1)
        else:
            batch_indices = mx.arange(technique_logits.shape[0])
            technique_loss_val = -log_probs[batch_indices, technique_targets]

    # Mask technique loss for non-singing samples
    technique_loss_val = technique_loss_val * singing_targets_float

    # Combine losses
    total_loss = singing_loss_val + technique_weight * technique_loss_val

    if reduction == "mean":
        total_loss = mx.mean(total_loss)
        singing_loss_val = mx.mean(singing_loss_val)
        # Average technique loss only over singing samples
        num_singing = mx.sum(singing_targets_float) + eps
        technique_loss_val = mx.sum(technique_loss_val) / num_singing
    elif reduction == "sum":
        total_loss = mx.sum(total_loss)
        singing_loss_val = mx.sum(singing_loss_val)
        technique_loss_val = mx.sum(technique_loss_val)

    return total_loss, singing_loss_val, technique_loss_val


class SingingLoss(nn.Module):
    """Singing loss as nn.Module wrapper."""

    def __init__(
        self,
        multi_technique: bool = True,
        label_smoothing: float = 0.1,
        technique_weight: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.multi_technique = multi_technique
        self.label_smoothing = label_smoothing
        self.technique_weight = technique_weight
        self.reduction = reduction

    def __call__(
        self,
        singing_logits: mx.array,
        technique_logits: mx.array,
        singing_targets: mx.array,
        technique_targets: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        return singing_loss(
            singing_logits=singing_logits,
            technique_logits=technique_logits,
            singing_targets=singing_targets,
            technique_targets=technique_targets,
            multi_technique=self.multi_technique,
            label_smoothing=self.label_smoothing,
            technique_weight=self.technique_weight,
            reduction=self.reduction,
        )


def compute_singing_accuracy(
    is_singing_pred: mx.array,
    is_singing_target: mx.array,
) -> mx.array:
    """
    Compute binary singing detection accuracy.

    Args:
        is_singing_pred: Predicted binary labels
        is_singing_target: Target binary labels

    Returns:
        Accuracy (0-1)
    """
    correct = (is_singing_pred == is_singing_target).astype(mx.float32)
    return mx.mean(correct)


def compute_technique_accuracy(
    technique_preds: mx.array,
    technique_targets: mx.array,
    singing_mask: mx.array,
    multi_technique: bool = True,
) -> mx.array:
    """
    Compute technique classification accuracy (only for singing samples).

    Args:
        technique_preds: Predicted techniques
        technique_targets: Target techniques
        singing_mask: Mask for singing samples
        multi_technique: Whether multi-technique mode

    Returns:
        Accuracy or F1 score
    """
    singing_mask_float = singing_mask.astype(mx.float32)
    num_singing = mx.sum(singing_mask_float) + 1e-8

    if multi_technique:
        # F1 for multi-label
        preds_float = technique_preds.astype(mx.float32)
        targets_float = technique_targets.astype(mx.float32)

        # Only compute for singing samples
        preds_float = preds_float * singing_mask_float[:, None]
        targets_float = targets_float * singing_mask_float[:, None]

        tp = mx.sum(preds_float * targets_float)
        fp = mx.sum(preds_float * (1 - targets_float))
        fn = mx.sum((1 - preds_float) * targets_float)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
    # Standard accuracy for single-label
    correct = (technique_preds == technique_targets).astype(mx.float32)
    correct = correct * singing_mask_float
    return mx.sum(correct) / num_singing
