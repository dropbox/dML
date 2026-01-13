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
Language identification head for Zipformer encoder.

Predicts spoken language from encoder output using attention pooling
followed by classification layers.

Target: >98% accuracy on supported languages (matching Whisper baseline 98.61%)

Supported Languages (initial 9):
- English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese

Extensible to 100+ languages via fine-tuning.

Datasets:
- CommonVoice: Multi-language speech corpus
- VoxLingua107: 107 languages
- FLEURS: 102 languages

Reference:
- Whisper achieves 98.61% on language identification
- VoxLingua107 baseline: ~95%
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# Core language inventory (ISO 639-1 codes)
CORE_LANGUAGES: tuple[str, ...] = (
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "pt",  # Portuguese
    "nl",  # Dutch
    "ru",  # Russian
    "zh",  # Chinese (Mandarin)
)

# Extended language inventory (for multi-language systems)
EXTENDED_LANGUAGES: tuple[str, ...] = (
    # Core 9
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh",
    # European
    "pl", "cs", "sk", "hu", "ro", "bg", "sr", "hr", "sl",
    "el", "tr", "fi", "sv", "da", "no", "uk", "be",
    # Asian
    "ja", "ko", "vi", "th", "id", "ms", "tl", "hi", "bn",
    "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur",
    # Middle Eastern
    "ar", "he", "fa", "ku",
    # African
    "sw", "am", "yo", "ig", "ha", "zu",
    # Special
    "<unknown>",
    "<mixed>",  # Code-switching
)

# Language names for display
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "uk": "Ukrainian",
    "he": "Hebrew",
    "fa": "Persian",
    "bn": "Bengali",
    "ta": "Tamil",
    "<unknown>": "Unknown",
    "<mixed>": "Mixed/Code-switching",
}


@dataclass
class LanguageConfig:
    """Configuration for language identification head."""

    # Input dimension from encoder
    encoder_dim: int = 384

    # Number of language classes
    num_languages: int = len(CORE_LANGUAGES)

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

    # Minimum audio duration (seconds) for reliable ID
    min_duration_sec: float = 0.5

    # Confidence threshold for prediction
    confidence_threshold: float = 0.7

    # Language inventory
    language_codes: tuple[str, ...] = CORE_LANGUAGES

    # Whether to use extended language set
    use_extended: bool = False


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence-to-vector.

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


class LanguageHead(nn.Module):
    """
    Language identification head for Zipformer encoder.

    Takes encoder output and predicts spoken language per utterance.
    Uses attention pooling to aggregate frame-level features.

    Args:
        config: LanguageConfig instance with hyperparameters.
    """

    def __init__(self, config: LanguageConfig | None = None):
        super().__init__()
        if config is None:
            config = LanguageConfig()

        self.config = config

        # Update language set if using extended
        if config.use_extended:
            config = LanguageConfig(
                encoder_dim=config.encoder_dim,
                num_languages=len(EXTENDED_LANGUAGES),
                hidden_dim=config.hidden_dim,
                num_attention_heads=config.num_attention_heads,
                dropout_rate=config.dropout_rate,
                use_attention_pooling=config.use_attention_pooling,
                label_smoothing=config.label_smoothing,
                min_duration_sec=config.min_duration_sec,
                confidence_threshold=config.confidence_threshold,
                language_codes=EXTENDED_LANGUAGES,
                use_extended=True,
            )
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
            nn.Linear(config.hidden_dim, config.num_languages),
        )

        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(config.encoder_dim)

    def __call__(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> mx.array:
        """
        Predict language from encoder output.

        Args:
            encoder_out: Encoder output of shape (batch_size, seq_len, encoder_dim)
            encoder_lengths: Optional sequence lengths of shape (batch_size,)

        Returns:
            Logits of shape (batch_size, num_languages)
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

        # Apply dropout after pooling (dropout in classifier handles hidden layers)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        return logits

    def predict(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Get predicted language classes and probabilities.

        Args:
            encoder_out: Encoder output
            encoder_lengths: Optional sequence lengths

        Returns:
            Tuple of:
            - Predicted language indices of shape (batch_size,)
            - Language probabilities of shape (batch_size, num_languages)
        """
        logits = self(encoder_out, encoder_lengths)
        probs = mx.softmax(logits, axis=-1)
        predictions = mx.argmax(logits, axis=-1)
        return predictions, probs

    def get_language_code(self, lang_idx: int) -> str:
        """Get ISO 639-1 language code from index."""
        return self.config.language_codes[lang_idx]

    def get_language_name(self, lang_idx: int) -> str:
        """Get human-readable language name from index."""
        code = self.config.language_codes[lang_idx]
        return LANGUAGE_NAMES.get(code, code)

    def get_language_id(self, lang_code: str) -> int:
        """Get index from language code."""
        return self.config.language_codes.index(lang_code)

    def decode_predictions(
        self,
        predictions: mx.array,
        probs: mx.array,
        top_k: int = 3,
    ) -> list[list[tuple[str, str, float]]]:
        """
        Decode predictions to language names with confidence scores.

        Args:
            predictions: Predicted class indices of shape (batch,)
            probs: Probability scores of shape (batch, num_languages)
            top_k: Return top-k predictions

        Returns:
            List of (code, name, confidence) tuples per batch item
        """
        batch_size = predictions.shape[0]
        results = []

        for b in range(batch_size):
            # Get top-k languages
            batch_probs = probs[b]
            top_indices = mx.argsort(batch_probs)[::-1][:top_k]

            langs = []
            for idx in top_indices.tolist():
                code = self.config.language_codes[idx]
                name = LANGUAGE_NAMES.get(code, code)
                conf = float(batch_probs[idx])
                langs.append((code, name, conf))

            results.append(langs)

        return results


def language_loss(
    logits: mx.array,
    targets: mx.array,
    label_smoothing: float = 0.1,
    reduction: str = "mean",
) -> mx.array:
    """
    Cross-entropy loss for language identification with label smoothing.

    Args:
        logits: Predicted logits of shape (batch_size, num_languages)
        targets: Target language indices of shape (batch_size,)
        label_smoothing: Label smoothing factor (0 = no smoothing)
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
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


class LanguageLoss(nn.Module):
    """Language loss as nn.Module wrapper."""

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
        return language_loss(
            logits=logits,
            targets=targets,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


def compute_language_accuracy(
    predictions: mx.array,
    targets: mx.array,
) -> mx.array:
    """
    Compute language identification accuracy.

    Args:
        predictions: Predicted language indices of shape (batch,)
        targets: Target language indices of shape (batch,)

    Returns:
        Accuracy (0-1)
    """
    correct = (predictions == targets).astype(mx.float32)
    return mx.mean(correct)
