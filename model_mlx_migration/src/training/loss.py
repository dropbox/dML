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
Loss functions for Zipformer ASR training.

Implements:
- CTC Loss (for auxiliary CTC head)
- Transducer Loss (for RNN-T)
- CR-CTC Loss (joint transducer + CTC, used in icefall)

References:
- CTC: Graves et al., "Connectionist Temporal Classification"
- RNN-T: Graves, "Sequence Transduction with Recurrent Neural Networks"
- CR-CTC: k2-fsa/icefall implementation
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

_NEG_INF = -1e30


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Numerically stable log softmax."""
    x_max = mx.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    return x_shifted - mx.log(mx.sum(mx.exp(x_shifted), axis=axis, keepdims=True))


def logsumexp(a: mx.array, b: mx.array) -> mx.array:
    """
    Numerically stable log(exp(a) + exp(b)).

    Handles edge cases where both inputs are -inf.
    """
    neg_inf = mx.array(_NEG_INF)
    # If both are -inf (or very negative), return -inf directly
    both_neg_inf = mx.logical_and(a < neg_inf + 1e10, b < neg_inf + 1e10)
    max_val = mx.maximum(a, b)
    result = max_val + mx.log(mx.exp(a - max_val) + mx.exp(b - max_val))
    # Return -inf when both inputs are effectively -inf
    return mx.where(both_neg_inf, neg_inf, result)


def logsumexp3(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    return logsumexp(logsumexp(a, b), c)


def ctc_loss(
    logits: mx.array,
    targets: mx.array,
    input_lengths: mx.array,
    target_lengths: mx.array,
    blank_id: int = 0,
    reduction: str = "mean",
) -> mx.array:
    """
    CTC Loss implementation in MLX.

    Computes the CTC loss for a batch of sequences using the forward-backward
    algorithm.

    Args:
        logits: Log probabilities of shape (batch, time, vocab).
        targets: Target sequences of shape (batch, max_target_len).
        input_lengths: Input sequence lengths of shape (batch,).
        target_lengths: Target sequence lengths of shape (batch,).
        blank_id: Index of blank label.
        reduction: "mean", "sum", or "none".

    Returns:
        CTC loss value.
    """
    batch_size, max_time, _ = logits.shape
    max_target_len = targets.shape[1] if targets.ndim == 2 else 0
    max_expanded_len = 2 * max_target_len + 1

    # Apply log softmax
    log_probs = log_softmax(logits, axis=-1)

    neg_inf = mx.array(_NEG_INF)

    # Build expanded label sequences (batch, S)
    expanded = mx.full((batch_size, max_expanded_len), blank_id, dtype=mx.int32)
    if max_target_len > 0:
        odd_positions = mx.arange(max_target_len, dtype=mx.int32) * 2 + 1  # (U,)
        batch_idx = mx.arange(batch_size, dtype=mx.int32)[:, None]  # (B,1)
        pos = mx.broadcast_to(odd_positions[None, :], (batch_size, max_target_len))
        # MLX ArrayAt doesn't support `.set()`: use add(delta) to overwrite blank_id with target label.
        expanded = expanded.at[batch_idx, pos].add(targets.astype(mx.int32) - blank_id)

    expanded_lens = target_lengths.astype(mx.int32) * 2 + 1  # (B,)
    state_positions = mx.arange(max_expanded_len, dtype=mx.int32)[None, :]
    state_mask = state_positions < expanded_lens[:, None]

    # Skip transitions from s-2 are allowed only when:
    # - current label is not blank
    # - current label != label at s-2 (avoid repeated labels)
    # Use concatenation instead of ArrayAt slice to avoid gradient bug in MLX 0.30.1
    if max_expanded_len > 2:
        shifted_by_2 = mx.concatenate(
            [mx.full((batch_size, 2), blank_id, dtype=mx.int32), expanded[:, :-2]],
            axis=1,
        )
    else:
        shifted_by_2 = mx.full((batch_size, max_expanded_len), blank_id, dtype=mx.int32)
    s_ge_2 = state_positions >= 2
    skip_allowed = mx.logical_and(
        s_ge_2,
        mx.logical_and(expanded != blank_id, expanded != shifted_by_2),
    )

    # Initialize alpha for t=0
    # Use explicit indices for ArrayAt to avoid gradient bug in MLX 0.30.1 with slice notation
    batch_idx_col = mx.arange(batch_size, dtype=mx.int32)[:, None]  # (B, 1)
    alpha = mx.full((batch_size, max_expanded_len), neg_inf)
    has_time0 = input_lengths > 0
    if max_time > 0 and max_expanded_len > 0:
        emit0 = mx.take_along_axis(
            log_probs[:, 0, :],
            expanded[:, 0:1],
            axis=-1,
        ).squeeze(-1)
        col0_idx = mx.zeros((batch_size, 1), dtype=mx.int32)
        alpha = alpha.at[batch_idx_col, col0_idx].add(
            mx.where(has_time0, emit0 - neg_inf, mx.array(0.0))[:, None],
        )

        if max_expanded_len > 1:
            emit1 = mx.take_along_axis(
                log_probs[:, 0, :],
                expanded[:, 1:2],
                axis=-1,
            ).squeeze(-1)
            valid_state1 = mx.logical_and(has_time0, expanded_lens > 1)
            col1_idx = mx.ones((batch_size, 1), dtype=mx.int32)
            alpha = alpha.at[batch_idx_col, col1_idx].add(
                mx.where(valid_state1, emit1 - neg_inf, mx.array(0.0))[:, None],
            )

    alpha = mx.where(state_mask, alpha, neg_inf)

    # Forward DP across time (vectorized over batch + states)
    for t in range(1, max_time):
        emit = mx.take_along_axis(log_probs[:, t, :], expanded, axis=-1)  # (B, S)

        a0 = alpha
        a1 = mx.concatenate([mx.full((batch_size, 1), neg_inf), alpha[:, :-1]], axis=1)
        # Use concatenation instead of ArrayAt slice to avoid gradient bug in MLX 0.30.1
        if max_expanded_len > 2:
            a2 = mx.concatenate([mx.full((batch_size, 2), neg_inf), alpha[:, :-2]], axis=1)
        else:
            a2 = mx.full((batch_size, max_expanded_len), neg_inf)
        a2 = mx.where(skip_allowed, a2, neg_inf)

        new_alpha = logsumexp3(a0, a1, a2) + emit
        new_alpha = mx.where(state_mask, new_alpha, neg_inf)

        time_mask = input_lengths > t
        alpha = mx.where(time_mask[:, None], new_alpha, alpha)

    # Final probability from last two states at the final time for each sequence
    last_idx = mx.maximum(expanded_lens - 1, mx.array(0, dtype=mx.int32))
    last_val = mx.take_along_axis(alpha, last_idx[:, None], axis=1).squeeze(-1)

    second_idx = mx.maximum(expanded_lens - 2, mx.array(0, dtype=mx.int32))
    second_val = mx.take_along_axis(alpha, second_idx[:, None], axis=1).squeeze(-1)
    second_val = mx.where(expanded_lens > 1, second_val, neg_inf)

    log_prob = logsumexp(last_val, second_val)

    # Handle empty input edge cases explicitly
    empty_input = input_lengths == 0
    empty_target = target_lengths == 0
    log_prob = mx.where(mx.logical_and(empty_input, empty_target), mx.array(0.0), log_prob)
    log_prob = mx.where(mx.logical_and(empty_input, mx.logical_not(empty_target)), neg_inf, log_prob)

    loss = -log_prob

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def transducer_loss_simple(
    encoder_out: mx.array,
    decoder_out: mx.array,
    joiner: "nn.Module",
    targets: mx.array,
    encoder_lengths: mx.array,
    target_lengths: mx.array,
    blank_id: int = 0,
    reduction: str = "mean",
) -> mx.array:
    """
    Simple transducer loss implementation.

    Uses the full lattice computation (O(T*U) per sequence).
    For production, use pruned transducer loss for efficiency.

    Args:
        encoder_out: Encoder output of shape (batch, T, encoder_dim).
        decoder_out: Decoder output of shape (batch, U+1, decoder_dim).
        joiner: Joiner network that computes logits from encoder+decoder.
        targets: Target sequences of shape (batch, U).
        encoder_lengths: Encoder output lengths of shape (batch,).
        target_lengths: Target sequence lengths of shape (batch,).
        blank_id: Index of blank label.
        reduction: "mean", "sum", or "none".

    Returns:
        Transducer loss value.
    """
    batch_size = encoder_out.shape[0]
    neg_inf = mx.array(_NEG_INF)

    losses = []

    for b in range(batch_size):
        T = int(encoder_lengths[b].item())
        U = int(target_lengths[b].item())

        if T == 0:
            losses.append(mx.array(0.0 if U == 0 else -_NEG_INF))
            continue

        # Get encoder and decoder outputs for this sequence
        enc = encoder_out[b : b + 1, :T]  # (1, T, encoder_dim)
        dec = decoder_out[b : b + 1, : U + 1]  # (1, U+1, decoder_dim)

        # Memory-friendly forward algorithm:
        # keep only alpha[t, :] (size U+1), avoid materializing full lattice logits.
        alpha = mx.full((U + 1,), neg_inf)
        alpha = alpha.at[0].add(mx.array(0.0) - neg_inf)

        for t in range(T):
            logits_row = joiner(enc[:, t : t + 1, :], dec)  # (1, 1, U+1, vocab)
            logits_row = logits_row[0, 0]  # (U+1, vocab)
            log_probs_row = log_softmax(logits_row, axis=-1)

            next_alpha = mx.full((U + 1,), neg_inf)

            for u in range(U + 1):
                a = alpha[u]
                prev = next_alpha[u]
                updated = logsumexp(prev, a + log_probs_row[u, blank_id])
                next_alpha = next_alpha.at[u].add(updated - prev)

                if u < U:
                    label = int(targets[b, u].item())
                    prev = alpha[u + 1]
                    updated = logsumexp(prev, a + log_probs_row[u, label])
                    alpha = alpha.at[u + 1].add(updated - prev)

            alpha = next_alpha

        log_prob = alpha[U]
        losses.append(-log_prob)

    loss = mx.stack(losses)

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def cr_ctc_loss(
    encoder_out: mx.array,
    decoder_out: mx.array,
    joiner: "nn.Module",
    ctc_output: mx.array,
    targets: mx.array,
    encoder_lengths: mx.array,
    target_lengths: mx.array,
    blank_id: int = 0,
    ctc_weight: float = 0.3,
    reduction: str = "mean",
) -> tuple[mx.array, mx.array, mx.array]:
    """
    CR-CTC Loss: Joint Transducer + CTC loss.

    This is the loss function used in k2-fsa/icefall for joint training
    of transducer and CTC heads on the same encoder.

    Benefits:
    - CTC provides auxiliary gradient signal
    - CTC head can be used for ROVER voting
    - More stable training

    Args:
        encoder_out: Encoder output of shape (batch, T, encoder_dim).
        decoder_out: Decoder output of shape (batch, U+1, decoder_dim).
        joiner: Joiner network.
        ctc_output: CTC logits of shape (batch, T, vocab).
        targets: Target sequences of shape (batch, U).
        encoder_lengths: Encoder output lengths of shape (batch,).
        target_lengths: Target sequence lengths of shape (batch,).
        blank_id: Index of blank label.
        ctc_weight: Weight for CTC loss (transducer weight = 1 - ctc_weight).
        reduction: "mean", "sum", or "none".

    Returns:
        Tuple of (total_loss, transducer_loss, ctc_loss).
    """
    # Compute transducer loss
    trans_loss = transducer_loss_simple(
        encoder_out=encoder_out,
        decoder_out=decoder_out,
        joiner=joiner,
        targets=targets,
        encoder_lengths=encoder_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id,
        reduction=reduction,
    )

    # Compute CTC loss
    ctc_loss_val = ctc_loss(
        logits=ctc_output,
        targets=targets,
        input_lengths=encoder_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id,
        reduction=reduction,
    )

    # Combine losses
    total_loss = (1.0 - ctc_weight) * trans_loss + ctc_weight * ctc_loss_val

    return total_loss, trans_loss, ctc_loss_val


class CTCLoss(nn.Module):
    """CTC Loss as an nn.Module wrapper."""

    def __init__(self, blank_id: int = 0, reduction: str = "mean"):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction

    def __call__(
        self,
        logits: mx.array,
        targets: mx.array,
        input_lengths: mx.array,
        target_lengths: mx.array,
    ) -> mx.array:
        return ctc_loss(
            logits=logits,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank_id=self.blank_id,
            reduction=self.reduction,
        )


class TransducerLoss(nn.Module):
    """Transducer Loss as an nn.Module wrapper."""

    def __init__(self, blank_id: int = 0, reduction: str = "mean"):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction

    def __call__(
        self,
        encoder_out: mx.array,
        decoder_out: mx.array,
        joiner: nn.Module,
        targets: mx.array,
        encoder_lengths: mx.array,
        target_lengths: mx.array,
    ) -> mx.array:
        return transducer_loss_simple(
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            joiner=joiner,
            targets=targets,
            encoder_lengths=encoder_lengths,
            target_lengths=target_lengths,
            blank_id=self.blank_id,
            reduction=self.reduction,
        )


class CRCTCLoss(nn.Module):
    """CR-CTC Loss as an nn.Module wrapper."""

    def __init__(
        self,
        blank_id: int = 0,
        ctc_weight: float = 0.3,
        reduction: str = "mean",
    ):
        super().__init__()
        self.blank_id = blank_id
        self.ctc_weight = ctc_weight
        self.reduction = reduction

    def __call__(
        self,
        encoder_out: mx.array,
        decoder_out: mx.array,
        joiner: nn.Module,
        ctc_output: mx.array,
        targets: mx.array,
        encoder_lengths: mx.array,
        target_lengths: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        return cr_ctc_loss(
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            joiner=joiner,
            ctc_output=ctc_output,
            targets=targets,
            encoder_lengths=encoder_lengths,
            target_lengths=target_lengths,
            blank_id=self.blank_id,
            ctc_weight=self.ctc_weight,
            reduction=self.reduction,
        )


# =============================================================================
# Rich Audio Heads Loss Functions
# =============================================================================


def cross_entropy_loss(
    logits: mx.array,
    targets: mx.array,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> mx.array:
    """
    Cross entropy loss for classification heads.

    Uses vectorized operations compatible with MLX autodiff.
    Efficient label smoothing without materializing full one-hot tensor.

    Args:
        logits: Predicted logits (batch, num_classes) or (batch, seq, num_classes)
        targets: Target class indices (batch,) or (batch, seq)
        label_smoothing: Label smoothing factor
        reduction: "mean", "sum", or "none"

    Returns:
        Cross entropy loss
    """
    logits.shape[-1]
    log_probs = log_softmax(logits, axis=-1)

    # Gather log prob at target index (used for both paths)
    targets_expanded = mx.expand_dims(targets, axis=-1)
    nll = -mx.squeeze(mx.take_along_axis(log_probs, targets_expanded, axis=-1), axis=-1)

    if label_smoothing > 0:
        # Efficient label smoothing without materializing one-hot tensor
        # loss = (1-eps) * nll + eps * uniform_loss
        # where uniform_loss = -mean(log_probs) over classes
        smooth_loss = -mx.mean(log_probs, axis=-1)
        loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
    else:
        loss = nll

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def binary_cross_entropy_loss(
    logits: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """
    Binary cross entropy loss with logits.

    Args:
        logits: Predicted logits (any shape)
        targets: Binary targets (same shape as logits)
        reduction: "mean", "sum", or "none"

    Returns:
        BCE loss
    """
    # Numerically stable BCE: max(x, 0) - x*t + log(1 + exp(-|x|))
    pos_part = mx.maximum(logits, 0)
    neg_part = logits * targets
    log_part = mx.log(1 + mx.exp(-mx.abs(logits)))
    loss = pos_part - neg_part + log_part

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def mse_loss(
    predictions: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
    reduction: str = "mean",
) -> mx.array:
    """
    Mean squared error loss for regression heads.

    Args:
        predictions: Predicted values
        targets: Target values (same shape)
        mask: Optional mask (1 = valid, 0 = invalid)
        reduction: "mean", "sum", or "none"

    Returns:
        MSE loss
    """
    diff = predictions - targets
    loss = diff * diff

    if mask is not None:
        loss = loss * mask
        if reduction == "mean":
            return mx.sum(loss) / (mx.sum(mask) + 1e-8)
        if reduction == "sum":
            return mx.sum(loss)
        return loss

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def l1_loss(
    predictions: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
    reduction: str = "mean",
) -> mx.array:
    """
    L1 (absolute) loss for regression heads.

    Args:
        predictions: Predicted values
        targets: Target values (same shape)
        mask: Optional mask (1 = valid, 0 = invalid)
        reduction: "mean", "sum", or "none"

    Returns:
        L1 loss
    """
    loss = mx.abs(predictions - targets)

    if mask is not None:
        loss = loss * mask
        if reduction == "mean":
            return mx.sum(loss) / (mx.sum(mask) + 1e-8)
        if reduction == "sum":
            return mx.sum(loss)
        return loss

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


@dataclass
class RichAudioLossOutput:
    """Output from RichAudioLoss computation."""

    total_loss: mx.array
    losses: dict[str, mx.array]  # Individual head losses


class RichAudioLoss(nn.Module):
    """
    Combined loss function for all rich audio heads.

    Computes weighted sum of losses for:
    - Emotion (CE)
    - Language (CE)
    - Paralinguistics (CE)
    - Pitch F0 (MSE) + Voiced (BCE)
    - Phoneme (CE frame-level)
    - Singing binary (BCE) + technique (CE)
    - Timestamp boundary (BCE) + offset (L1)
    """

    def __init__(
        self,
        emotion_weight: float = 0.1,
        language_weight: float = 0.1,
        paralinguistics_weight: float = 0.1,
        pitch_weight: float = 0.05,
        phoneme_weight: float = 0.2,
        singing_weight: float = 0.05,
        timestamp_weight: float = 0.1,
        pitch_f0_weight: float = 0.5,
        pitch_voiced_weight: float = 0.5,
        singing_binary_weight: float = 0.5,
        singing_technique_weight: float = 0.5,
        timestamp_boundary_weight: float = 0.7,
        timestamp_offset_weight: float = 0.3,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.language_weight = language_weight
        self.paralinguistics_weight = paralinguistics_weight
        self.pitch_weight = pitch_weight
        self.phoneme_weight = phoneme_weight
        self.singing_weight = singing_weight
        self.timestamp_weight = timestamp_weight
        self.pitch_f0_weight = pitch_f0_weight
        self.pitch_voiced_weight = pitch_voiced_weight
        self.singing_binary_weight = singing_binary_weight
        self.singing_technique_weight = singing_technique_weight
        self.timestamp_boundary_weight = timestamp_boundary_weight
        self.timestamp_offset_weight = timestamp_offset_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(
        self,
        outputs: dict[str, mx.array],
        labels: dict[str, mx.array],
    ) -> RichAudioLossOutput:
        """
        Compute rich audio heads loss.

        Args:
            outputs: Dict from RichAudioHeads containing:
                - emotion_logits: (batch, 8)
                - language_logits: (batch, num_languages)
                - paralinguistics_logits: (batch, num_classes)
                - pitch_f0_hz: (batch, seq)
                - pitch_voiced_logits: (batch, seq)
                - phoneme_logits: (batch, seq, num_phonemes)
                - singing_binary_logits: (batch, 1)
                - singing_technique_logits: (batch, num_techniques)
                - timestamp_boundary_logits: (batch, seq, 1)
                - timestamp_offset_preds: (batch, seq, 2) or None

            labels: Dict containing (only compute loss for present keys):
                - emotion_labels: (batch,) int
                - language_labels: (batch,) int
                - paralinguistics_labels: (batch,) int
                - pitch_f0_targets: (batch, seq) float
                - pitch_voiced_targets: (batch, seq) float
                - pitch_mask: (batch, seq) optional
                - phoneme_labels: (batch, seq) int
                - phoneme_mask: (batch, seq) optional
                - singing_binary_labels: (batch, 1) float
                - singing_technique_labels: (batch,) int
                - timestamp_boundary_targets: (batch, seq, 1) float
                - timestamp_offset_targets: (batch, seq, 2) float
                - timestamp_mask: (batch, seq) optional

        Returns:
            RichAudioLossOutput with total loss and per-head losses
        """
        losses: dict[str, mx.array] = {}
        total_loss = mx.array(0.0)

        # Emotion loss
        if "emotion_labels" in labels and "emotion_logits" in outputs:
            emotion_loss = cross_entropy_loss(
                outputs["emotion_logits"],
                labels["emotion_labels"],
                label_smoothing=self.label_smoothing,
                reduction=self.reduction,
            )
            losses["emotion"] = emotion_loss
            total_loss = total_loss + self.emotion_weight * emotion_loss

        # Language loss
        if "language_labels" in labels and "language_logits" in outputs:
            language_loss = cross_entropy_loss(
                outputs["language_logits"],
                labels["language_labels"],
                label_smoothing=self.label_smoothing,
                reduction=self.reduction,
            )
            losses["language"] = language_loss
            total_loss = total_loss + self.language_weight * language_loss

        # Paralinguistics loss
        if "paralinguistics_labels" in labels and "paralinguistics_logits" in outputs:
            para_loss = cross_entropy_loss(
                outputs["paralinguistics_logits"],
                labels["paralinguistics_labels"],
                label_smoothing=self.label_smoothing,
                reduction=self.reduction,
            )
            losses["paralinguistics"] = para_loss
            total_loss = total_loss + self.paralinguistics_weight * para_loss

        # Pitch loss (F0 + voiced)
        if "pitch_f0_targets" in labels and "pitch_f0_hz" in outputs:
            pitch_mask = labels.get("pitch_mask")

            f0_loss = mse_loss(
                outputs["pitch_f0_hz"],
                labels["pitch_f0_targets"],
                mask=pitch_mask,
                reduction=self.reduction,
            )
            losses["pitch_f0"] = f0_loss

            voiced_loss = mx.array(0.0)
            if "pitch_voiced_targets" in labels and "pitch_voiced_logits" in outputs:
                voiced_loss = binary_cross_entropy_loss(
                    outputs["pitch_voiced_logits"],
                    labels["pitch_voiced_targets"],
                    reduction=self.reduction,
                )
                losses["pitch_voiced"] = voiced_loss

            pitch_total = (
                self.pitch_f0_weight * f0_loss + self.pitch_voiced_weight * voiced_loss
            )
            losses["pitch"] = pitch_total
            total_loss = total_loss + self.pitch_weight * pitch_total

        # Phoneme loss (frame-level CE)
        if "phoneme_labels" in labels and "phoneme_logits" in outputs:
            phoneme_loss = cross_entropy_loss(
                outputs["phoneme_logits"],
                labels["phoneme_labels"],
                label_smoothing=self.label_smoothing,
                reduction=self.reduction,
            )
            losses["phoneme"] = phoneme_loss
            total_loss = total_loss + self.phoneme_weight * phoneme_loss

        # Singing loss (binary + technique)
        if "singing_binary_labels" in labels and "singing_binary_logits" in outputs:
            binary_loss = binary_cross_entropy_loss(
                outputs["singing_binary_logits"],
                labels["singing_binary_labels"],
                reduction=self.reduction,
            )
            losses["singing_binary"] = binary_loss

            technique_loss = mx.array(0.0)
            if (
                "singing_technique_labels" in labels
                and "singing_technique_logits" in outputs
            ):
                technique_loss = cross_entropy_loss(
                    outputs["singing_technique_logits"],
                    labels["singing_technique_labels"],
                    label_smoothing=self.label_smoothing,
                    reduction=self.reduction,
                )
                losses["singing_technique"] = technique_loss

            singing_total = (
                self.singing_binary_weight * binary_loss
                + self.singing_technique_weight * technique_loss
            )
            losses["singing"] = singing_total
            total_loss = total_loss + self.singing_weight * singing_total

        # Timestamp loss (boundary + offset)
        if (
            "timestamp_boundary_targets" in labels
            and "timestamp_boundary_logits" in outputs
        ):
            ts_mask = labels.get("timestamp_mask")

            boundary_loss = binary_cross_entropy_loss(
                outputs["timestamp_boundary_logits"],
                labels["timestamp_boundary_targets"],
                reduction=self.reduction,
            )
            losses["timestamp_boundary"] = boundary_loss

            offset_loss = mx.array(0.0)
            if (
                "timestamp_offset_targets" in labels
                and "timestamp_offset_preds" in outputs
            ):
                offset_loss = l1_loss(
                    outputs["timestamp_offset_preds"],
                    labels["timestamp_offset_targets"],
                    mask=ts_mask,
                    reduction=self.reduction,
                )
                losses["timestamp_offset"] = offset_loss

            ts_total = (
                self.timestamp_boundary_weight * boundary_loss
                + self.timestamp_offset_weight * offset_loss
            )
            losses["timestamp"] = ts_total
            total_loss = total_loss + self.timestamp_weight * ts_total

        return RichAudioLossOutput(total_loss=total_loss, losses=losses)
