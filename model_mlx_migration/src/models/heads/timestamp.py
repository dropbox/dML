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
Word timestamp prediction head for Zipformer encoder.

Predicts word-level timing information (start/end timestamps) from
encoder output. Uses frame-level boundary detection with peak finding.

Target: 80-90% timestamp accuracy (within 50ms of ground truth)

Output Format:
- Per-frame boundary probability
- Start/end time in milliseconds per word

Methods:
1. Frame-level boundary detection (predict word boundaries)
2. Forced alignment refinement (CTC alignment)
3. Cross-attention to decoder states (if available)

Datasets:
- LibriSpeech (forced aligned)
- MLS (forced aligned)
- CommonVoice (forced aligned)

Reference:
- CTC forced alignment: 20-50ms typical accuracy
- Whisper word timestamps: ~100ms accuracy
"""

from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn


class WordTimestamp(NamedTuple):
    """Word timestamp result."""
    word: str
    start_ms: float
    end_ms: float
    confidence: float


@dataclass
class TimestampConfig:
    """Configuration for timestamp head."""

    # Input dimension from encoder
    encoder_dim: int = 384

    # Hidden dimension for predictor
    hidden_dim: int = 256

    # Number of hidden layers
    num_layers: int = 2

    # Dropout rate
    dropout_rate: float = 0.1

    # Frame duration in milliseconds (depends on encoder stride)
    # Zipformer typically uses 40ms frames (10ms base * 4 subsampling)
    frame_duration_ms: float = 40.0

    # Boundary detection threshold
    boundary_threshold: float = 0.5

    # Minimum word duration in milliseconds
    min_word_duration_ms: float = 50.0

    # Maximum word duration in milliseconds
    max_word_duration_ms: float = 2000.0

    # Use regression head for offset refinement
    use_offset_regression: bool = True

    # Offset range in frames (for regression)
    max_offset_frames: int = 2


class TimestampHead(nn.Module):
    """
    Word timestamp prediction head for Zipformer encoder.

    Uses frame-level boundary detection to identify word boundaries,
    then computes timestamps based on frame positions.

    Two outputs per frame:
    1. Boundary probability (is this frame a word boundary?)
    2. Optional offset regression (sub-frame refinement)

    Args:
        config: TimestampConfig instance with hyperparameters.
    """

    def __init__(self, config: TimestampConfig | None = None):
        super().__init__()
        if config is None:
            config = TimestampConfig()

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

        # Boundary probability head (binary classification per frame)
        self.boundary_head = nn.Linear(config.hidden_dim, 1)

        # Optional offset regression head (start offset, end offset)
        if config.use_offset_regression:
            self.offset_head = nn.Linear(config.hidden_dim, 2)
        else:
            self.offset_head = None

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.encoder_dim)

    def __call__(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Predict word boundaries from encoder output.

        Args:
            encoder_out: Encoder output of shape (batch_size, seq_len, encoder_dim)
            encoder_lengths: Optional sequence lengths of shape (batch_size,)

        Returns:
            Tuple of:
            - boundary_logits: Boundary logits of shape (batch, seq_len, 1)
            - offset_preds: Offset predictions of shape (batch, seq_len, 2) or None
        """
        # Layer norm
        x = self.layer_norm(encoder_out)

        # Hidden layers
        if self.hidden is not None:
            x = self.hidden(x)

        # Apply dropout before final projection
        x = self.dropout(x)

        # Boundary prediction
        boundary_logits = self.boundary_head(x)

        # Offset regression
        offset_preds = None
        if self.offset_head is not None:
            offset_preds = self.offset_head(x)
            # Scale offsets to reasonable range
            offset_preds = mx.tanh(offset_preds) * self.config.max_offset_frames

        return boundary_logits, offset_preds

    def predict_boundaries(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        """
        Get predicted word boundaries and probabilities.

        Args:
            encoder_out: Encoder output
            encoder_lengths: Optional sequence lengths

        Returns:
            Tuple of:
            - boundary_mask: Binary boundary mask of shape (batch, seq_len)
            - boundary_probs: Boundary probabilities of shape (batch, seq_len)
            - offsets: Offset predictions of shape (batch, seq_len, 2) or None
        """
        boundary_logits, offset_preds = self(encoder_out, encoder_lengths)

        # Squeeze last dimension
        boundary_logits = boundary_logits.squeeze(-1)

        # Sigmoid for probability
        boundary_probs = mx.sigmoid(boundary_logits)

        # Threshold for binary prediction
        boundary_mask = (boundary_probs > self.config.boundary_threshold).astype(mx.int32)

        return boundary_mask, boundary_probs, offset_preds

    def extract_word_timestamps(
        self,
        boundary_probs: mx.array,
        tokens: list[str],
        lengths: mx.array | None = None,
        offsets: mx.array | None = None,
    ) -> list[list[WordTimestamp]]:
        """
        Extract word timestamps from boundary predictions and tokens.

        Simple algorithm:
        1. Find peaks in boundary probability
        2. Assign each token to a boundary region
        3. Compute timestamps based on frame positions

        Args:
            boundary_probs: Boundary probabilities (batch, seq_len)
            tokens: List of token strings per batch item
            lengths: Optional sequence lengths
            offsets: Optional offset refinements

        Returns:
            List of WordTimestamp lists per batch item
        """
        batch_size = boundary_probs.shape[0]
        seq_len = boundary_probs.shape[1]
        frame_ms = self.config.frame_duration_ms

        results = []

        for b in range(batch_size):
            length = int(lengths[b]) if lengths is not None else seq_len
            probs = boundary_probs[b, :length].tolist()

            # Find boundary peaks (local maxima above threshold)
            boundaries = []
            for i in range(length):
                if probs[i] > self.config.boundary_threshold:
                    # Check if local maximum
                    is_peak = True
                    if i > 0 and probs[i] <= probs[i - 1]:
                        is_peak = False
                    if i < length - 1 and probs[i] < probs[i + 1]:
                        is_peak = False
                    if is_peak:
                        boundaries.append(i)

            # Add start and end boundaries
            if not boundaries or boundaries[0] != 0:
                boundaries.insert(0, 0)
            if boundaries[-1] != length - 1:
                boundaries.append(length - 1)

            # Get tokens for this batch item
            batch_tokens = tokens[b] if isinstance(tokens[0], list) else tokens

            # Create word timestamps
            word_timestamps = []
            num_words = len(batch_tokens)
            num_boundaries = len(boundaries)

            if num_words > 0 and num_boundaries > 1:
                # Distribute words across boundary regions
                words_per_region = max(1, num_words // (num_boundaries - 1))
                word_idx = 0

                for i in range(num_boundaries - 1):
                    if word_idx >= num_words:
                        break

                    start_frame = boundaries[i]
                    end_frame = boundaries[i + 1]

                    # Apply offset refinement if available
                    start_offset = 0.0
                    end_offset = 0.0
                    if offsets is not None:
                        start_offset = float(offsets[b, start_frame, 0])
                        end_offset = float(offsets[b, end_frame, 1])

                    start_ms = (start_frame + start_offset) * frame_ms
                    end_ms = (end_frame + end_offset) * frame_ms

                    # Clamp to valid range
                    start_ms = max(0, start_ms)
                    end_ms = min(length * frame_ms, end_ms)

                    # Get confidence (average boundary prob)
                    conf = (probs[start_frame] + probs[end_frame]) / 2

                    # Assign words to this region
                    words_in_region = []
                    for _ in range(words_per_region):
                        if word_idx < num_words:
                            words_in_region.append(batch_tokens[word_idx])
                            word_idx += 1

                    # If last region, include remaining words
                    if i == num_boundaries - 2:
                        while word_idx < num_words:
                            words_in_region.append(batch_tokens[word_idx])
                            word_idx += 1

                    # Create timestamps for each word in region
                    if words_in_region:
                        region_duration = end_ms - start_ms
                        word_duration = region_duration / len(words_in_region)

                        for j, word in enumerate(words_in_region):
                            word_start = start_ms + j * word_duration
                            word_end = start_ms + (j + 1) * word_duration
                            word_timestamps.append(WordTimestamp(
                                word=word,
                                start_ms=word_start,
                                end_ms=word_end,
                                confidence=conf,
                            ))

            results.append(word_timestamps)

        return results

    def frame_to_time(self, frame_idx: int, offset: float = 0.0) -> float:
        """Convert frame index to time in milliseconds."""
        return (frame_idx + offset) * self.config.frame_duration_ms

    def time_to_frame(self, time_ms: float) -> int:
        """Convert time in milliseconds to frame index."""
        return int(time_ms / self.config.frame_duration_ms)


def timestamp_loss(
    boundary_logits: mx.array,
    boundary_targets: mx.array,
    offset_preds: mx.array | None = None,
    offset_targets: mx.array | None = None,
    mask: mx.array | None = None,
    offset_weight: float = 0.1,
    reduction: str = "mean",
) -> tuple[mx.array, mx.array, mx.array | None]:
    """
    Loss for timestamp prediction.

    Args:
        boundary_logits: Boundary logits (batch, seq, 1)
        boundary_targets: Binary boundary targets (batch, seq)
        offset_preds: Optional offset predictions (batch, seq, 2)
        offset_targets: Optional offset targets (batch, seq, 2)
        mask: Optional mask (batch, seq), True = valid position
        offset_weight: Weight for offset loss
        reduction: "mean", "sum", or "none"

    Returns:
        Tuple of (total_loss, boundary_loss, offset_loss)
    """
    batch_size, seq_len, _ = boundary_logits.shape
    boundary_logits = boundary_logits.squeeze(-1)

    # Create mask if not provided
    if mask is None:
        mask = mx.ones((batch_size, seq_len), dtype=mx.bool_)

    # Binary cross-entropy for boundaries
    boundary_targets_float = boundary_targets.astype(mx.float32)
    boundary_prob = mx.sigmoid(boundary_logits)
    eps = 1e-7
    boundary_prob = mx.clip(boundary_prob, eps, 1 - eps)

    boundary_loss = -(
        boundary_targets_float * mx.log(boundary_prob) +
        (1 - boundary_targets_float) * mx.log(1 - boundary_prob)
    )
    boundary_loss = mx.where(mask, boundary_loss, mx.zeros_like(boundary_loss))

    # Offset loss (L1 on boundary frames only)
    offset_loss = None
    if offset_preds is not None and offset_targets is not None:
        # Only compute offset loss on boundary frames
        boundary_mask = mask & (boundary_targets > 0)
        offset_diff = mx.abs(offset_preds - offset_targets)
        offset_loss = mx.mean(offset_diff, axis=-1)  # Average over start/end
        offset_loss = mx.where(boundary_mask, offset_loss, mx.zeros_like(offset_loss))

    # Reduction
    num_valid = mx.sum(mask.astype(mx.float32)) + eps
    if offset_loss is not None:
        num_boundaries = mx.sum((mask & (boundary_targets > 0)).astype(mx.float32)) + eps

    if reduction == "mean":
        boundary_loss = mx.sum(boundary_loss) / num_valid
        if offset_loss is not None:
            offset_loss = mx.sum(offset_loss) / num_boundaries
    elif reduction == "sum":
        boundary_loss = mx.sum(boundary_loss)
        if offset_loss is not None:
            offset_loss = mx.sum(offset_loss)

    # Combined loss
    if offset_loss is not None:
        total_loss = boundary_loss + offset_weight * offset_loss
    else:
        total_loss = boundary_loss

    return total_loss, boundary_loss, offset_loss


class TimestampLoss(nn.Module):
    """Timestamp loss as nn.Module wrapper."""

    def __init__(
        self,
        offset_weight: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.offset_weight = offset_weight
        self.reduction = reduction

    def __call__(
        self,
        boundary_logits: mx.array,
        boundary_targets: mx.array,
        offset_preds: mx.array | None = None,
        offset_targets: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        return timestamp_loss(
            boundary_logits=boundary_logits,
            boundary_targets=boundary_targets,
            offset_preds=offset_preds,
            offset_targets=offset_targets,
            mask=mask,
            offset_weight=self.offset_weight,
            reduction=self.reduction,
        )


def compute_boundary_accuracy(
    predictions: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """
    Compute frame-level boundary detection accuracy.

    Args:
        predictions: Predicted boundary mask (batch, seq)
        targets: Target boundary mask (batch, seq)
        mask: Optional validity mask

    Returns:
        Accuracy (0-1)
    """
    if mask is None:
        mask = mx.ones(predictions.shape, dtype=mx.bool_)

    correct = (predictions == targets).astype(mx.float32)
    correct = mx.where(mask, correct, mx.zeros_like(correct))

    num_correct = mx.sum(correct)
    num_total = mx.sum(mask.astype(mx.float32)) + 1e-8

    return num_correct / num_total


def compute_boundary_f1(
    predictions: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Compute boundary detection F1 score.

    Args:
        predictions: Predicted boundary mask (batch, seq)
        targets: Target boundary mask (batch, seq)
        mask: Optional validity mask

    Returns:
        Tuple of (f1, precision, recall)
    """
    if mask is None:
        mask = mx.ones(predictions.shape, dtype=mx.bool_)

    preds_float = predictions.astype(mx.float32)
    targets_float = targets.astype(mx.float32)
    mask_float = mask.astype(mx.float32)

    # Apply mask
    preds_float = preds_float * mask_float
    targets_float = targets_float * mask_float

    tp = mx.sum(preds_float * targets_float)
    fp = mx.sum(preds_float * (1 - targets_float))
    fn = mx.sum((1 - preds_float) * targets_float)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1, precision, recall


def compute_timestamp_error(
    predicted_timestamps: list[list[WordTimestamp]],
    target_timestamps: list[list[WordTimestamp]],
    tolerance_ms: float = 50.0,
) -> tuple[float, float, float]:
    """
    Compute timestamp error metrics.

    Args:
        predicted_timestamps: Predicted word timestamps
        target_timestamps: Ground truth word timestamps
        tolerance_ms: Tolerance for "correct" classification

    Returns:
        Tuple of (mean_abs_error_ms, accuracy_within_tolerance, word_error_rate)
    """
    total_error = 0.0
    total_correct = 0
    total_words = 0
    total_word_errors = 0

    for pred_list, target_list in zip(predicted_timestamps, target_timestamps, strict=False):
        # Simple alignment: match by word order
        for i, target in enumerate(target_list):
            total_words += 1

            if i < len(pred_list):
                pred = pred_list[i]

                # Compute errors
                start_error = abs(pred.start_ms - target.start_ms)
                end_error = abs(pred.end_ms - target.end_ms)
                avg_error = (start_error + end_error) / 2

                total_error += avg_error

                if avg_error <= tolerance_ms:
                    total_correct += 1

                # Word error (different word)
                if pred.word != target.word:
                    total_word_errors += 1
            else:
                # Missing prediction
                total_word_errors += 1
                total_error += tolerance_ms * 2  # Penalty

    mean_error = total_error / max(total_words, 1)
    accuracy = total_correct / max(total_words, 1)
    wer = total_word_errors / max(total_words, 1)

    return mean_error, accuracy, wer
