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
Pitch (F0) prediction head for Zipformer encoder.

Predicts fundamental frequency (F0) in Hz for each frame, along with
voicing probability. Uses frame-level regression.

Target: <10Hz MAE on voiced frames

F0 Range:
- Speech: ~50-500Hz (male 85-180Hz, female 165-255Hz typical)
- Singing: ~50-1000Hz

Output:
- f0_hz: Predicted F0 in Hz (valid for voiced frames)
- voicing_prob: Probability that frame is voiced (0-1)

Reference:
- CREPE: CNN for pitch estimation (~5.9 cents accuracy)
- PYIN: Probabilistic YIN algorithm
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class PitchConfig:
    """Configuration for pitch head."""

    # Input dimension from encoder
    encoder_dim: int = 384

    # Hidden dimension for predictor
    hidden_dim: int = 256

    # Number of hidden layers
    num_layers: int = 2

    # F0 prediction range
    f0_min_hz: float = 50.0  # Minimum F0 (Hz)
    f0_max_hz: float = 800.0  # Maximum F0 (Hz)

    # Output representation
    # "hz": direct Hz prediction
    # "log_hz": log-scale Hz prediction (better for gradient flow)
    # "cents": cents relative to reference (like CREPE)
    output_mode: str = "log_hz"

    # Reference frequency for cents mode (A4 = 440Hz)
    reference_hz: float = 440.0

    # Whether to predict voicing probability
    predict_voicing: bool = True

    # Dropout rate
    dropout_rate: float = 0.1


class PitchHead(nn.Module):
    """
    Pitch prediction head for Zipformer encoder.

    Predicts F0 value and voicing probability for each frame.

    Args:
        config: PitchConfig instance with hyperparameters.
    """

    def __init__(self, config: PitchConfig | None = None):
        super().__init__()
        if config is None:
            config = PitchConfig()

        self.config = config

        # Dropout for regularization (applied during training only)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Build hidden layers with dropout between layers
        layers = []
        in_dim = config.encoder_dim
        for i in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.GELU())
            # Add dropout between hidden layers (not after last layer)
            if i < config.num_layers - 1:
                layers.append(nn.Dropout(config.dropout_rate))
            in_dim = config.hidden_dim

        self.hidden = nn.Sequential(*layers) if layers else None

        # F0 prediction head (outputs 1 value per frame)
        self.f0_proj = nn.Linear(config.hidden_dim, 1)

        # Voicing prediction head (outputs probability)
        if config.predict_voicing:
            self.voicing_proj = nn.Linear(config.hidden_dim, 1)
        else:
            self.voicing_proj = None

        # Layer norm before prediction
        self.layer_norm = nn.LayerNorm(config.encoder_dim)

        # Precompute log range for log_hz mode
        self.log_f0_min = mx.log(mx.array(config.f0_min_hz))
        self.log_f0_max = mx.log(mx.array(config.f0_max_hz))

    def __call__(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Predict F0 and voicing from encoder output.

        Args:
            encoder_out: Encoder output of shape (batch_size, seq_len, encoder_dim)
            encoder_lengths: Optional sequence lengths of shape (batch_size,)

        Returns:
            Tuple of:
            - f0_hz: Predicted F0 in Hz of shape (batch_size, seq_len)
            - voicing_prob: Voicing probability of shape (batch_size, seq_len) or None

        Note:
            The Zipformer encoder outputs (seq_len, batch_size, encoder_dim).
            Transpose before calling this method:
            encoder_out = mx.transpose(encoder_out, (1, 0, 2))
        """
        # Layer norm
        x = self.layer_norm(encoder_out)

        # Hidden layers
        if self.hidden is not None:
            x = self.hidden(x)

        # Apply dropout before final projection
        x = self.dropout(x)

        # F0 prediction
        f0_raw = self.f0_proj(x)  # (batch, seq, 1)
        f0_raw = mx.squeeze(f0_raw, axis=-1)  # (batch, seq)

        # Convert to Hz based on output mode
        if self.config.output_mode == "log_hz":
            # Sigmoid to [0, 1], then scale to log range
            f0_normalized = mx.sigmoid(f0_raw)
            log_f0 = self.log_f0_min + f0_normalized * (self.log_f0_max - self.log_f0_min)
            f0_hz = mx.exp(log_f0)
        elif self.config.output_mode == "hz":
            # Sigmoid to [0, 1], then scale to Hz range
            f0_normalized = mx.sigmoid(f0_raw)
            f0_hz = self.config.f0_min_hz + f0_normalized * (
                self.config.f0_max_hz - self.config.f0_min_hz
            )
        elif self.config.output_mode == "cents":
            # Raw output is cents relative to reference, convert to Hz
            # cents = 1200 * log2(f0 / ref)
            # f0 = ref * 2^(cents / 1200)
            f0_hz = self.config.reference_hz * mx.power(2.0, f0_raw / 1200.0)
            # Clamp to valid range
            f0_hz = mx.clip(f0_hz, self.config.f0_min_hz, self.config.f0_max_hz)
        else:
            raise ValueError(f"Unknown output_mode: {self.config.output_mode}")

        # Voicing prediction
        voicing_prob = None
        if self.voicing_proj is not None:
            voicing_logit = self.voicing_proj(x)  # (batch, seq, 1)
            voicing_prob = mx.sigmoid(mx.squeeze(voicing_logit, axis=-1))  # (batch, seq)

        return f0_hz, voicing_prob

    def predict(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
        voicing_threshold: float = 0.5,
    ) -> tuple[mx.array, mx.array]:
        """
        Get predicted F0 with voicing mask.

        Args:
            encoder_out: Encoder output
            encoder_lengths: Optional sequence lengths
            voicing_threshold: Threshold for voicing decision

        Returns:
            Tuple of:
            - f0_hz: Predicted F0 in Hz (0 for unvoiced frames)
            - voiced: Boolean mask of voiced frames
        """
        f0_hz, voicing_prob = self(encoder_out, encoder_lengths)

        if voicing_prob is not None:
            voiced = voicing_prob >= voicing_threshold
            # Zero out F0 for unvoiced frames
            f0_hz = mx.where(voiced, f0_hz, mx.zeros_like(f0_hz))
        else:
            voiced = mx.ones(f0_hz.shape, dtype=mx.bool_)

        return f0_hz, voiced


def pitch_loss(
    f0_pred: mx.array,
    f0_target: mx.array,
    voicing_pred: mx.array | None,
    voicing_target: mx.array | None,
    mask: mx.array | None = None,
    f0_loss_type: str = "l1",
    voicing_weight: float = 0.5,
    reduction: str = "mean",
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Combined loss for pitch prediction.

    Args:
        f0_pred: Predicted F0 in Hz of shape (batch, seq)
        f0_target: Target F0 in Hz of shape (batch, seq)
        voicing_pred: Predicted voicing probability of shape (batch, seq) or None
        voicing_target: Target voicing mask (1=voiced, 0=unvoiced) of shape (batch, seq) or None
        mask: Optional mask of shape (batch, seq), True = valid position
        f0_loss_type: "l1" or "l2" for F0 regression
        voicing_weight: Weight for voicing loss (F0 loss weight = 1 - voicing_weight)
        reduction: "mean", "sum", or "none"

    Returns:
        Tuple of (total_loss, f0_loss, voicing_loss)
    """
    batch_size, seq_len = f0_pred.shape

    # Create mask if not provided
    if mask is None:
        mask = mx.ones((batch_size, seq_len), dtype=mx.bool_)

    # F0 loss (only on voiced frames if voicing_target is provided)
    if voicing_target is not None:
        # Only compute F0 loss on voiced frames
        f0_mask = mask & (voicing_target > 0.5)
    else:
        f0_mask = mask

    # Compute F0 loss
    if f0_loss_type == "l1":
        f0_diff = mx.abs(f0_pred - f0_target)
    elif f0_loss_type == "l2":
        f0_diff = (f0_pred - f0_target) ** 2
    else:
        raise ValueError(f"Unknown f0_loss_type: {f0_loss_type}")

    # Apply mask
    f0_diff = mx.where(f0_mask, f0_diff, mx.zeros_like(f0_diff))
    num_voiced = mx.sum(f0_mask.astype(mx.float32)) + 1e-8

    if reduction == "mean":
        f0_loss = mx.sum(f0_diff) / num_voiced
    elif reduction == "sum":
        f0_loss = mx.sum(f0_diff)
    else:
        f0_loss = f0_diff

    # Voicing loss (binary cross-entropy)
    if voicing_pred is not None and voicing_target is not None:
        # Clamp predictions to avoid log(0)
        voicing_pred_clamped = mx.clip(voicing_pred, 1e-7, 1 - 1e-7)
        voicing_bce = -(
            voicing_target * mx.log(voicing_pred_clamped)
            + (1 - voicing_target) * mx.log(1 - voicing_pred_clamped)
        )
        voicing_bce = mx.where(mask, voicing_bce, mx.zeros_like(voicing_bce))
        num_valid = mx.sum(mask.astype(mx.float32)) + 1e-8

        if reduction == "mean":
            voicing_loss = mx.sum(voicing_bce) / num_valid
        elif reduction == "sum":
            voicing_loss = mx.sum(voicing_bce)
        else:
            voicing_loss = voicing_bce
    else:
        voicing_loss = mx.array(0.0)

    # Combined loss
    total_loss = (1 - voicing_weight) * f0_loss + voicing_weight * voicing_loss

    return total_loss, f0_loss, voicing_loss


class PitchLoss(nn.Module):
    """Pitch loss as nn.Module wrapper."""

    def __init__(
        self,
        f0_loss_type: str = "l1",
        voicing_weight: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.f0_loss_type = f0_loss_type
        self.voicing_weight = voicing_weight
        self.reduction = reduction

    def __call__(
        self,
        f0_pred: mx.array,
        f0_target: mx.array,
        voicing_pred: mx.array | None = None,
        voicing_target: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        return pitch_loss(
            f0_pred=f0_pred,
            f0_target=f0_target,
            voicing_pred=voicing_pred,
            voicing_target=voicing_target,
            mask=mask,
            f0_loss_type=self.f0_loss_type,
            voicing_weight=self.voicing_weight,
            reduction=self.reduction,
        )


def hz_to_cents(f0_hz: mx.array, reference_hz: float = 440.0) -> mx.array:
    """Convert Hz to cents relative to reference frequency."""
    return 1200 * mx.log2(f0_hz / reference_hz)


def cents_to_hz(cents: mx.array, reference_hz: float = 440.0) -> mx.array:
    """Convert cents to Hz relative to reference frequency."""
    return reference_hz * mx.power(2.0, cents / 1200.0)


def compute_pitch_mae(
    f0_pred: mx.array,
    f0_target: mx.array,
    voicing_target: mx.array | None = None,
) -> mx.array:
    """
    Compute Mean Absolute Error for F0 prediction.

    Only computes error on voiced frames if voicing_target is provided.

    Args:
        f0_pred: Predicted F0 in Hz
        f0_target: Target F0 in Hz
        voicing_target: Optional voicing mask (1=voiced)

    Returns:
        MAE in Hz
    """
    if voicing_target is not None:
        mask = voicing_target > 0.5
        diff = mx.abs(f0_pred - f0_target)
        diff = mx.where(mask, diff, mx.zeros_like(diff))
        num_voiced = mx.sum(mask.astype(mx.float32)) + 1e-8
        return mx.sum(diff) / num_voiced
    return mx.mean(mx.abs(f0_pred - f0_target))


def compute_voicing_accuracy(
    voicing_pred: mx.array,
    voicing_target: mx.array,
    threshold: float = 0.5,
) -> mx.array:
    """
    Compute accuracy of voicing prediction.

    Args:
        voicing_pred: Predicted voicing probability
        voicing_target: Target voicing mask
        threshold: Threshold for voicing decision

    Returns:
        Accuracy (0-1)
    """
    pred_voiced = voicing_pred >= threshold
    target_voiced = voicing_target > 0.5
    correct = pred_voiced == target_voiced
    return mx.mean(correct.astype(mx.float32))
