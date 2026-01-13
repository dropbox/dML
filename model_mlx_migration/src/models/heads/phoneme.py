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
Phoneme prediction head for Zipformer encoder.

Predicts IPA phoneme classes for each frame. Uses frame-level
classification with cross-entropy loss.

Target: <18% PER (Phoneme Error Rate) on LibriSpeech

Phoneme Inventory:
- 178 IPA phonemes (covering major world languages)
- Includes special tokens: blank, silence, word boundary

Loss Options:
- Frame-level cross-entropy (with forced alignment labels)
- CTC loss (with sequence labels)

Reference:
- Whisper achieves ~19.7% PER
- CMU phoneme set: 39 phonemes (English only)
- IPA extended: 178+ phonemes (multilingual)
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# IPA phoneme inventory - subset of most common phonemes
# Full IPA has 100+ consonants and 50+ vowels
IPA_PHONEMES: tuple[str, ...] = (
    # Special tokens
    "<blank>",  # CTC blank
    "<sil>",    # Silence
    "<unk>",    # Unknown
    "<wb>",     # Word boundary

    # English consonants (plosives)
    "p", "b", "t", "d", "k", "ɡ",
    # Affricates
    "tʃ", "dʒ",
    # Fricatives
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    # Nasals
    "m", "n", "ŋ",
    # Liquids
    "l", "ɹ",
    # Glides
    "w", "j",

    # English vowels (monophthongs)
    "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "ʊ", "u", "ʌ", "ə",
    # Diphthongs
    "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ",

    # Additional consonants (other languages)
    "ʔ",  # Glottal stop (Arabic, German)
    "x",  # Voiceless velar fricative (German, Spanish)
    "ɣ",  # Voiced velar fricative (Spanish, Arabic)
    "ç",  # Voiceless palatal fricative (German)
    "ʁ",  # Voiced uvular fricative (French, German)
    "r",  # Alveolar trill (Spanish, Italian)
    "ɾ",  # Alveolar tap (Spanish, Japanese)
    "ɲ",  # Palatal nasal (Spanish ñ, Italian gn)
    "ʎ",  # Palatal lateral (Italian, Spanish)

    # Additional vowels (other languages)
    "y",  # Close front rounded (French, German)
    "ø",  # Close-mid front rounded (French, German)
    "œ",  # Open-mid front rounded (French)
    "ɯ",  # Close back unrounded (Japanese, Korean)
    "ɤ",  # Close-mid back unrounded (Korean, Mandarin)

    # Tonal markers (for tonal languages)
    "˥",  # High tone (Mandarin tone 1)
    "˧˥", # Rising tone (Mandarin tone 2)
    "˨˩˦", # Dipping tone (Mandarin tone 3)
    "˥˩", # Falling tone (Mandarin tone 4)

    # Length markers
    "ː",  # Long vowel marker

    # Stress markers
    "ˈ",  # Primary stress
    "ˌ",  # Secondary stress
)


@dataclass
class PhonemeConfig:
    """Configuration for phoneme head."""

    # Input dimension from encoder
    encoder_dim: int = 384

    # Number of phoneme classes
    num_phonemes: int = len(IPA_PHONEMES)

    # Hidden dimension for classifier
    hidden_dim: int = 256

    # Number of hidden layers
    num_layers: int = 2

    # Dropout rate
    dropout_rate: float = 0.1

    # Blank token index (for CTC)
    blank_id: int = 0

    # Whether to use CTC loss or frame-level CE
    use_ctc: bool = False

    # Label smoothing for frame-level CE
    label_smoothing: float = 0.1

    # Phoneme inventory (can be customized)
    phoneme_inventory: tuple[str, ...] = IPA_PHONEMES


class PhonemeHead(nn.Module):
    """
    Phoneme prediction head for Zipformer encoder.

    Predicts phoneme class for each frame. Can be used with
    frame-level cross-entropy (aligned labels) or CTC (sequence labels).

    Args:
        config: PhonemeConfig instance with hyperparameters.
    """

    def __init__(self, config: PhonemeConfig | None = None):
        super().__init__()
        if config is None:
            config = PhonemeConfig()

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

        # Phoneme classification head
        self.classifier = nn.Linear(config.hidden_dim, config.num_phonemes)

        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(config.encoder_dim)

    def __call__(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> mx.array:
        """
        Predict phonemes from encoder output.

        Args:
            encoder_out: Encoder output of shape (batch_size, seq_len, encoder_dim)
            encoder_lengths: Optional sequence lengths of shape (batch_size,)

        Returns:
            Logits of shape (batch_size, seq_len, num_phonemes)

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

        # Classify
        logits = self.classifier(x)

        return logits

    def predict(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Get predicted phoneme classes and probabilities.

        Args:
            encoder_out: Encoder output
            encoder_lengths: Optional sequence lengths

        Returns:
            Tuple of:
            - Predicted phoneme indices of shape (batch_size, seq_len)
            - Phoneme probabilities of shape (batch_size, seq_len, num_phonemes)
        """
        logits = self(encoder_out, encoder_lengths)
        probs = mx.softmax(logits, axis=-1)
        predictions = mx.argmax(logits, axis=-1)
        return predictions, probs

    def get_phoneme(self, phoneme_idx: int) -> str:
        """Get phoneme symbol from index."""
        return self.config.phoneme_inventory[phoneme_idx]

    def get_phoneme_id(self, phoneme: str) -> int:
        """Get index from phoneme symbol."""
        return self.config.phoneme_inventory.index(phoneme)

    def decode_greedy(
        self,
        logits: mx.array,
        lengths: mx.array | None = None,
        collapse_repeated: bool = True,
        remove_blank: bool = True,
    ) -> list[list[str]]:
        """
        Greedy decode phoneme logits to phoneme sequences.

        Args:
            logits: Phoneme logits of shape (batch_size, seq_len, num_phonemes)
            lengths: Optional sequence lengths
            collapse_repeated: Whether to collapse repeated phonemes (CTC-style)
            remove_blank: Whether to remove blank tokens

        Returns:
            List of phoneme sequences (one per batch item)
        """
        predictions = mx.argmax(logits, axis=-1)  # (batch, seq)
        batch_size = predictions.shape[0]
        seq_len = predictions.shape[1]

        results = []
        for b in range(batch_size):
            length = int(lengths[b]) if lengths is not None else seq_len
            phoneme_ids = predictions[b, :length].tolist()

            # Decode
            phonemes = []
            prev_id = -1
            for pid in phoneme_ids:
                # Skip blank
                if remove_blank and pid == self.config.blank_id:
                    prev_id = pid
                    continue

                # Collapse repeated
                if collapse_repeated and pid == prev_id:
                    continue

                phonemes.append(self.config.phoneme_inventory[pid])
                prev_id = pid

            results.append(phonemes)

        return results


def phoneme_ce_loss(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
    label_smoothing: float = 0.1,
    reduction: str = "mean",
) -> mx.array:
    """
    Frame-level cross-entropy loss for phoneme prediction.

    Args:
        logits: Predicted logits of shape (batch, seq, num_phonemes)
        targets: Target phoneme indices of shape (batch, seq)
        mask: Optional mask of shape (batch, seq), True = valid position
        label_smoothing: Label smoothing factor
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    batch_size, seq_len, num_classes = logits.shape

    # Create mask if not provided
    if mask is None:
        mask = mx.ones((batch_size, seq_len), dtype=mx.bool_)

    # Log softmax
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    if label_smoothing > 0:
        # Create smoothed target distribution
        smooth_targets = mx.full(logits.shape, label_smoothing / num_classes)
        # One-hot encode targets
        # Create indices for batch and sequence dimensions
        batch_idx = mx.arange(batch_size)[:, None]  # (batch, 1)
        seq_idx = mx.arange(seq_len)[None, :]  # (1, seq)
        batch_idx = mx.broadcast_to(batch_idx, (batch_size, seq_len))
        seq_idx = mx.broadcast_to(seq_idx, (batch_size, seq_len))

        # Manual one-hot construction
        one_hot = mx.zeros_like(logits)
        # This is a workaround since MLX doesn't have simple scatter
        for b in range(batch_size):
            for s in range(seq_len):
                t = int(targets[b, s].item())
                one_hot = one_hot.at[b, s, t].add(1.0)

        # Combine smooth and one-hot
        smooth_targets = smooth_targets + one_hot * (1.0 - label_smoothing)

        # Cross entropy
        loss = -mx.sum(smooth_targets * log_probs, axis=-1)  # (batch, seq)
    else:
        # Standard cross entropy
        # Gather log probs at target indices
        loss = mx.zeros((batch_size, seq_len))
        for b in range(batch_size):
            for s in range(seq_len):
                t = int(targets[b, s].item())
                loss = loss.at[b, s].add(-log_probs[b, s, t])

    # Apply mask
    loss = mx.where(mask, loss, mx.zeros_like(loss))
    num_valid = mx.sum(mask.astype(mx.float32)) + 1e-8

    if reduction == "mean":
        return mx.sum(loss) / num_valid
    if reduction == "sum":
        return mx.sum(loss)
    return loss


class PhonemeFrameLoss(nn.Module):
    """Phoneme frame-level loss as nn.Module wrapper."""

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
        mask: mx.array | None = None,
    ) -> mx.array:
        return phoneme_ce_loss(
            logits=logits,
            targets=targets,
            mask=mask,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


def compute_per(
    predictions: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
    blank_id: int = 0,
) -> mx.array:
    """
    Compute Phoneme Error Rate (frame-level accuracy).

    For CTC-style evaluation, use a proper edit distance metric instead.

    Args:
        predictions: Predicted phoneme indices of shape (batch, seq)
        targets: Target phoneme indices of shape (batch, seq)
        mask: Optional mask of shape (batch, seq), True = valid position
        blank_id: Blank token ID to exclude from evaluation

    Returns:
        Error rate (0-1)
    """
    if mask is None:
        mask = mx.ones(predictions.shape, dtype=mx.bool_)

    # Exclude blank tokens from evaluation
    non_blank_mask = mask & (targets != blank_id)

    # Count errors (mismatches)
    errors = (predictions != targets).astype(mx.float32)
    errors = mx.where(non_blank_mask, errors, mx.zeros(errors.shape))

    num_errors = mx.sum(errors)
    num_total = mx.sum(non_blank_mask.astype(mx.float32)) + 1e-8

    return num_errors / num_total


def compute_frame_accuracy(
    predictions: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """
    Compute frame-level accuracy.

    Args:
        predictions: Predicted phoneme indices of shape (batch, seq)
        targets: Target phoneme indices of shape (batch, seq)
        mask: Optional mask of shape (batch, seq), True = valid position

    Returns:
        Accuracy (0-1)
    """
    if mask is None:
        mask = mx.ones(predictions.shape, dtype=mx.bool_)

    correct = (predictions == targets).astype(mx.float32)
    correct = mx.where(mask, correct, mx.zeros(correct.shape))

    num_correct = mx.sum(correct)
    num_total = mx.sum(mask.astype(mx.float32)) + 1e-8

    return num_correct / num_total
