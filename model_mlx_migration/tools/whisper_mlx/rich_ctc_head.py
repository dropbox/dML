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
Rich CTC Head - Multi-task CTC head with all frame-aligned outputs.

Combines all trained heads into a unified interface for rich audio understanding:
- Text CTC (51865 Whisper vocab + 50 paralinguistic special tokens)
- Emotion (8 RAVDESS classes or 34 extended classes)
- Pitch (CREPE-style 361-bin classification -> F0 Hz)
- Paralinguistics (50 classes: non-verbal, fillers, singing)
- Phonemes (Misaki 178-200 phonemes via CTC)

All outputs are frame-aligned at 50Hz (20ms per frame).

Architecture:
    Audio -> Whisper Encoder (frozen) -> RichCTCHead -> {
        text_logits: (T, vocab_size)
        emotion: (T, num_emotions)
        pitch_bins: (T, 361)
        pitch_hz: (T, 1)
        para: (T, 50)
        phoneme: (T, phoneme_vocab)
    }

Usage:
    from tools.whisper_mlx.rich_ctc_head import RichCTCHead

    # Load with trained weights
    head = RichCTCHead.from_pretrained()

    # Forward pass
    outputs = head(encoder_output)

    # Access individual outputs
    text_logits = outputs["text_logits"]
    emotion_logits = outputs["emotion"]
    pitch_hz = outputs["pitch_hz"]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

# Canonical 9-class emotion taxonomy (v2.0 - adds contempt)
# Import from label_taxonomy.py for the single source of truth
from .label_taxonomy import EMOTION_CLASSES_9

# =============================================================================
# Constants
# =============================================================================

WHISPER_VOCAB_SIZE = 51865
PARA_SPECIAL_TOKEN_COUNT = 50
EXTENDED_VOCAB_SIZE = WHISPER_VOCAB_SIZE + PARA_SPECIAL_TOKEN_COUNT  # 51915

# Re-export for backward compatibility
EMOTION_CLASSES_8 = EMOTION_CLASSES_9  # Alias for compatibility (now 9 classes)

# Extended 34-class emotion taxonomy (from trained checkpoint)
EMOTION_CLASSES_34 = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
    "contempt",  # Now at index 8, moved up from index 11
    "amused", "bored", "confused", "desire", "disappointment",
    "embarrassment", "excited", "interested", "pain", "proud", "relieved",
    "satisfied", "sympathetic", "tired", "default", "enunciated", "laughing",
    "nonverbal", "projected", "singing", "sleepy", "whisper",
    "confused_voice", "concerned", "thoughtful",
]

# 50 Paralinguistics classes (from UNIFIED_RICH_AUDIO_ARCHITECTURE.md)
PARA_CLASSES = {
    # Universal non-verbal (0-10)
    "speech": 0, "laughter": 1, "cough": 2, "sigh": 3, "breath": 4,
    "cry": 5, "yawn": 6, "throat_clear": 7, "sneeze": 8, "gasp": 9, "groan": 10,
    # English fillers (11-15)
    "um_en": 11, "uh_en": 12, "hmm_en": 13, "er_en": 14, "ah_en": 15,
    # Chinese fillers (16-19)
    "nage_zh": 16, "zhege_zh": 17, "jiushi_zh": 18, "en_zh": 19,
    # Japanese fillers (20-24)
    "eto_ja": 20, "ano_ja": 21, "ee_ja": 22, "maa_ja": 23, "un_ja": 24,
    # Korean fillers (25-28)
    "eo_ko": 25, "eum_ko": 26, "geuge_ko": 27, "mwo_ko": 28,
    # Hindi fillers (29-32)
    "matlab_hi": 29, "wo_hi": 30, "yeh_hi": 31, "haan_hi": 32,
    # Other languages (33-39)
    "este_es": 33, "pues_es": 34, "euh_fr": 35, "ben_fr": 36,
    "aeh_de": 37, "also_de": 38, "yani_ar": 39,
    # Singing vocalizations (40-49)
    "sing_a": 40, "sing_e": 41, "sing_i": 42, "sing_o": 43, "sing_u": 44,
    "vibrato": 45, "trill": 46, "vocal_fry": 47, "falsetto": 48, "belt": 49,
}

PARA_CLASSES_INV = {v: k for k, v in PARA_CLASSES.items()}

# Special tokens for paralinguistics in text stream
PARA_SPECIAL_TOKENS = [
    # Universal (0-10)
    "<|SPEECH|>", "<|LAUGH|>", "<|COUGH|>", "<|SIGH|>", "<|BREATH|>",
    "<|CRY|>", "<|YAWN|>", "<|THROAT_CLEAR|>", "<|SNEEZE|>", "<|GASP|>", "<|GROAN|>",
    # English fillers (11-15)
    "<|UM_EN|>", "<|UH_EN|>", "<|HMM_EN|>", "<|ER_EN|>", "<|AH_EN|>",
    # Chinese fillers (16-19)
    "<|NAGE_ZH|>", "<|ZHEGE_ZH|>", "<|JIUSHI_ZH|>", "<|EN_ZH|>",
    # Japanese fillers (20-24)
    "<|ETO_JA|>", "<|ANO_JA|>", "<|EE_JA|>", "<|MAA_JA|>", "<|UN_JA|>",
    # Korean fillers (25-28)
    "<|EO_KO|>", "<|UEM_KO|>", "<|GEUGE_KO|>", "<|MWO_KO|>",
    # Hindi fillers (29-32)
    "<|MATLAB_HI|>", "<|WO_HI|>", "<|YEH_HI|>", "<|HAAN_HI|>",
    # Other languages (33-39)
    "<|ESTE_ES|>", "<|PUES_ES|>", "<|EUH_FR|>", "<|BEN_FR|>",
    "<|AEH_DE|>", "<|ALSO_DE|>", "<|YANI_AR|>",
    # Singing (40-49)
    "<|SING_A|>", "<|SING_E|>", "<|SING_I|>", "<|SING_O|>", "<|SING_U|>",
    "<|VIBRATO|>", "<|TRILL|>", "<|VOCAL_FRY|>", "<|FALSETTO|>", "<|BELT|>",
]

# CREPE pitch constants
CREPE_NUM_BINS = 361  # 360 pitch bins + 1 unvoiced
CREPE_MIN_HZ = 32.7   # C1
CREPE_MAX_HZ = 2093.0  # C7

# Whisper language tokens (from tokenizer)
# These are the language codes Whisper recognizes
WHISPER_LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

# Whisper language token IDs
# Base ID varies by model size - these are computed from tokenizer
# large-v3: 50259, large-v2/v1: 50259, medium: 50259, small: 50259, base/tiny: 50259
# The base ID is actually consistent across models, but we parameterize for safety
WHISPER_LANGUAGE_BASE_ID_BY_MODEL = {
    "tiny": 50259,
    "base": 50259,
    "small": 50259,
    "medium": 50259,
    "large": 50259,
    "large-v1": 50259,
    "large-v2": 50259,
    "large-v3": 50259,
    "large-v3-turbo": 50259,
}
# Default to large-v3
WHISPER_LANGUAGE_BASE_ID = 50259


def get_language_token_ids(model_size: str = "large-v3") -> dict:
    """Get language token IDs for a specific model size."""
    base_id = WHISPER_LANGUAGE_BASE_ID_BY_MODEL.get(model_size, 50259)
    return {
        lang_code: base_id + idx
        for idx, lang_code in enumerate(WHISPER_LANGUAGES.keys())
    }


# Default mappings (large-v3)
WHISPER_LANGUAGE_IDS = get_language_token_ids("large-v3")
WHISPER_ID_TO_LANGUAGE = {v: k for k, v in WHISPER_LANGUAGE_IDS.items()}

# Default checkpoint paths
DEFAULT_CHECKPOINTS = {
    "ctc": "checkpoints/ctc_english_full/step_49000.npz",
    "emotion": "checkpoints/emotion_unified_v2/best.npz",
    "pitch": "checkpoints/pitch_combined_v4/best.npz",
    # v1 step 1400 has best external PER (19.5%) - see TRAINING_OPTIMIZATION_ROADMAP.md
    "phoneme": "checkpoints/kokoro_phoneme_head_v1/kokoro_phoneme_head_1400_best.npz",
    # Paralinguistics v3 checkpoint - 11 classes (speech, laughter, cough, etc.)
    # Note: RichCTCHead uses 50 classes; only compatible weights (ln, fc1) loaded
    "para": "checkpoints/paralinguistics_v3/best.npz",
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RichCTCConfig:
    """Configuration for RichCTCHead."""
    d_model: int = 1280  # Whisper large-v3 encoder dim

    # CTC text
    text_vocab_size: int = WHISPER_VOCAB_SIZE  # Can extend to EXTENDED_VOCAB_SIZE

    # Emotion
    num_emotions: int = 34  # Extended emotion classes (from checkpoint)
    emotion_hidden_dim: int = 512

    # Pitch (CREPE-style)
    pitch_bins: int = CREPE_NUM_BINS
    pitch_hidden_dim: int = 256

    # Paralinguistics
    num_para_classes: int = 50
    para_hidden_dim: int = 256

    # Phonemes
    phoneme_vocab: int = 200  # Kokoro phonemes
    phoneme_hidden_dim: int = 512

    # Prosody conditioning
    use_prosody_ctc: bool = False  # Enable prosody-conditioned CTC
    prosody_dim: int = 64  # Prosody embedding dimension

    # Speaker embedding (Phase 8)
    use_speaker_embedding: bool = False  # Enable speaker embedding head
    speaker_embed_dim: int = 256  # Speaker embedding output dimension
    speaker_hidden_dim: int = 512  # Speaker head hidden dimension
    speaker_normalize: bool = True  # L2 normalize speaker embeddings

    # Frame rate
    frame_rate_hz: float = 50.0  # Whisper encoder: 20ms per frame


# =============================================================================
# Sub-heads (mirroring trained architecture)
# =============================================================================

class TextCTCHead(nn.Module):
    """CTC head for text transcription."""

    def __init__(self, d_model: int = 1280, vocab_size: int = WHISPER_VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: (batch, T, d_model) -> (batch, T, vocab_size)"""
        x = self.ln(x)
        return self.proj(x)


class EmotionHead(nn.Module):
    """Emotion classification head with frame-level output."""

    def __init__(
        self,
        d_model: int = 1280,
        num_emotions: int = 34,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_emotions = num_emotions

        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_emotions)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: (batch, T, d_model) -> (batch, T, num_emotions)"""
        x = self.ln(x)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


class DilatedConv1D(nn.Module):
    """
    1D Dilated Convolution for temporal pattern extraction.

    MLX Conv1d expects input (N, L, C_in) and outputs (N, L_out, C_out).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Receptive field = dilation * (kernel_size - 1)
        self.padding = dilation * (kernel_size - 1)

        # MLX Conv1d expects (N, L, C_in)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,  # Handle padding manually
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with causal padding.

        Args:
            x: (batch, T, channels) input

        Returns:
            (batch, T, out_channels) output
        """
        # Causal padding: pad only on the left
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])

        # Apply dilated conv (MLX doesn't support dilation directly)
        if self.dilation > 1:
            # Subsample input for dilation effect
            batch, seq_len, channels = x.shape
            # Gather dilated indices
            indices = mx.arange(0, seq_len - self.kernel_size + 1)
            kernel_indices = indices[:, None] + mx.arange(self.kernel_size) * self.dilation

            # Check bounds and use valid indices only
            valid_mask = kernel_indices[:, -1] < seq_len
            if not mx.all(valid_mask):
                # Truncate to valid positions
                valid_count = int(mx.sum(valid_mask))
                kernel_indices = kernel_indices[:valid_count]

            # Manual dilated convolution via einsum
            # For simplicity, use strided approach
            out = self._manual_dilated_conv(x)
        else:
            out = self.conv(x)

        return out

    def _manual_dilated_conv(self, x: mx.array) -> mx.array:
        """Manual dilated conv for dilation > 1."""
        batch, seq_len, channels = x.shape
        k = self.kernel_size
        d = self.dilation

        # Calculate output length
        out_len = seq_len - (k - 1) * d

        if out_len <= 0:
            return mx.zeros((batch, 1, self.conv.weight.shape[0]))

        # Gather dilated windows
        outputs = []
        for i in range(out_len):
            # Indices for dilated kernel
            indices = [i + j * d for j in range(k)]
            # Gather and stack: (batch, k, channels)
            window = mx.stack([x[:, idx, :] for idx in indices], axis=1)
            # Apply conv weights manually
            # weight shape: (out_channels, kernel_size, in_channels)
            out_i = mx.einsum('bkc,okc->bo', window, self.conv.weight)
            if self.conv.bias is not None:
                out_i = out_i + self.conv.bias
            outputs.append(out_i)

        # Stack outputs: (batch, out_len, out_channels)
        return mx.stack(outputs, axis=1)


class CREPEPitchHead(nn.Module):
    """CREPE-style pitch head with dilated convolutions."""

    CREPE_FREF = 32.70  # C1 in Hz
    CREPE_BINS = 360     # 6 octaves * 60 bins/octave

    def __init__(
        self,
        d_model: int = 1280,
        hidden_dim: int = 256,
        num_bins: int = CREPE_NUM_BINS,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins

        # Input projection and layer norm
        self.ln_input = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(d_model, hidden_dim)

        # Dilated conv layers (dilations: 1, 2, 4, 8, 16)
        self.conv_layers = []
        for dilation in [1, 2, 4, 8, 16]:
            self.conv_layers.append({
                "conv": DilatedConv1D(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                ),
                "ln": nn.LayerNorm(hidden_dim),
            })

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_bins)

        # Precompute bin frequencies
        self._bin_frequencies = None

    def _get_bin_frequencies(self) -> mx.array:
        """Compute center frequency for each bin."""
        if self._bin_frequencies is None:
            bins = mx.arange(self.CREPE_BINS)
            self._bin_frequencies = self.CREPE_FREF * (2.0 ** (bins / 60.0))
        return self._bin_frequencies

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Forward pass.

        Args:
            x: (batch, T, d_model) encoder output

        Returns:
            pitch_bins: (batch, T, num_bins) logits over pitch bins
            pitch_hz: (batch, T, 1) expected F0 in Hz
        """
        # Input projection
        x = self.ln_input(x)
        x = self.input_proj(x)  # (batch, T, hidden_dim)
        x = nn.gelu(x)

        # Dilated conv layers with residual connections
        # MLX Conv1d input: (batch, T, channels)
        for layer in self.conv_layers:
            residual = x
            x = layer["conv"](x)  # (batch, T', hidden_dim)
            # Handle length mismatch from convolution
            if x.shape[1] < residual.shape[1]:
                residual = residual[:, :x.shape[1], :]
            elif x.shape[1] > residual.shape[1]:
                x = x[:, :residual.shape[1], :]
            x = layer["ln"](x)
            x = nn.relu(x) + residual

        # Output projection
        pitch_bins = self.output_proj(x)  # (batch, T, num_bins)

        # Convert to Hz via expected value
        pitch_hz = self._bins_to_hz(pitch_bins)

        return pitch_bins, pitch_hz

    def _bins_to_hz(self, logits: mx.array) -> mx.array:
        """Convert bin logits to Hz via softmax + expected value."""
        # Softmax over pitch bins (exclude last bin = unvoiced)
        probs = mx.softmax(logits[..., :360], axis=-1)  # (batch, T, 360)

        # Bin center frequencies
        bin_freqs = self._get_bin_frequencies()  # (360,)

        # Expected value
        pitch_hz = mx.sum(probs * bin_freqs, axis=-1, keepdims=True)  # (batch, T, 1)

        # Mask unvoiced (when unvoiced bin has highest prob)
        unvoiced_prob = mx.sigmoid(logits[..., 360:361])  # (batch, T, 1)
        return pitch_hz * (1 - unvoiced_prob)



class ProsodyConditionedCTC(nn.Module):
    """
    CTC head that sees emotion/pitch at every frame for better punctuation.

    Key insight: CTC knows "this frame sounds like a question" before it
    decodes the text. By conditioning on prosody features (emotion, pitch),
    we can improve punctuation prediction.

    Architecture:
        encoder_out (batch, T, d_model = 1280)
        emotion_seq (batch, T, num_emotions = 34)
        pitch_seq (batch, T, 1)

        prosody = concat(emotion, pitch)  # (batch, T, 35)
        prosody_emb = prosody_proj(prosody)  # (batch, T, prosody_dim)
        combined = concat(encoder_out, prosody_emb)  # (batch, T, d_model + prosody_dim)
        logits = proj(combined)  # (batch, T, vocab_size)
    """

    def __init__(
        self,
        d_model: int = 1280,
        emotion_dim: int = 34,  # Extended emotions from checkpoint
        pitch_dim: int = 1,
        prosody_dim: int = 64,
        vocab_size: int = WHISPER_VOCAB_SIZE,
    ):
        super().__init__()
        self.d_model = d_model
        self.emotion_dim = emotion_dim
        self.pitch_dim = pitch_dim
        self.prosody_dim = prosody_dim
        self.vocab_size = vocab_size

        # Input layer norm for encoder output
        self.ln_encoder = nn.LayerNorm(d_model)

        # Prosody projection: (emotion_dim + pitch_dim) -> prosody_dim
        prosody_input_dim = emotion_dim + pitch_dim
        self.prosody_proj = nn.Linear(prosody_input_dim, prosody_dim)

        # Combined projection: (d_model + prosody_dim) -> vocab_size
        combined_dim = d_model + prosody_dim
        self.proj = nn.Linear(combined_dim, vocab_size)

    def __call__(
        self,
        encoder_output: mx.array,
        emotion_seq: mx.array,
        pitch_seq: mx.array,
    ) -> mx.array:
        """
        Forward pass with prosody conditioning.

        Args:
            encoder_output: (batch, T, d_model) Whisper encoder output
            emotion_seq: (batch, T, emotion_dim) per-frame emotion logits/probs
            pitch_seq: (batch, T, pitch_dim) per-frame pitch values (normalized)

        Returns:
            logits: (batch, T, vocab_size) CTC output logits
        """
        # Normalize encoder output
        x = self.ln_encoder(encoder_output)

        # Concatenate prosody features
        prosody = mx.concatenate([emotion_seq, pitch_seq], axis=-1)  # (batch, T, 35)

        # Project prosody to embedding
        prosody_emb = nn.gelu(self.prosody_proj(prosody))  # (batch, T, prosody_dim)

        # Combine encoder and prosody
        combined = mx.concatenate([x, prosody_emb], axis=-1)  # (batch, T, d_model + prosody_dim)

        # Project to vocabulary
        return self.proj(combined)  # (batch, T, vocab_size)


    @classmethod
    def from_ctc_head(
        cls,
        ctc_weights_path: str,
        emotion_dim: int = 34,
        pitch_dim: int = 1,
        prosody_dim: int = 64,
    ) -> ProsodyConditionedCTC:
        """
        Initialize from existing CTC head weights.

        Copies the CTC projection weights and initializes prosody projection
        randomly. The CTC projection is expanded to accommodate prosody input.

        Args:
            ctc_weights_path: Path to trained CTC head weights
            emotion_dim: Emotion feature dimension
            pitch_dim: Pitch feature dimension
            prosody_dim: Prosody embedding dimension

        Returns:
            ProsodyConditionedCTC with partial weight initialization
        """
        # Load existing CTC weights
        ctc_weights = dict(mx.load(ctc_weights_path))

        # Infer dimensions
        if "proj.weight" in ctc_weights:
            vocab_size, d_model = ctc_weights["proj.weight"].shape
        else:
            vocab_size = WHISPER_VOCAB_SIZE
            d_model = 1280

        # Create model
        model = cls(
            d_model=d_model,
            emotion_dim=emotion_dim,
            pitch_dim=pitch_dim,
            prosody_dim=prosody_dim,
            vocab_size=vocab_size,
        )

        # Copy encoder layer norm weights if available
        if "ln.weight" in ctc_weights:
            model.ln_encoder.weight = ctc_weights["ln.weight"]
            model.ln_encoder.bias = ctc_weights["ln.bias"]

        # Initialize projection: first d_model columns from CTC, rest random
        # CTC weight shape: (vocab_size, d_model)
        # New weight shape: (vocab_size, d_model + prosody_dim)
        if "proj.weight" in ctc_weights:
            old_weight = ctc_weights["proj.weight"]
            old_bias = ctc_weights["proj.bias"]

            # Create new weight matrix
            new_weight = mx.zeros((vocab_size, d_model + prosody_dim))

            # Copy existing weights (encoder portion)
            new_weight = new_weight.at[:, :d_model].add(old_weight)

            # Initialize prosody portion with small random values
            prosody_init = mx.random.normal((vocab_size, prosody_dim)) * 0.02
            new_weight = new_weight.at[:, d_model:].add(prosody_init)

            model.proj.weight = new_weight
            model.proj.bias = old_bias

        mx.eval(model.parameters())
        return model


class ParalinguisticsHead(nn.Module):
    """Paralinguistics classification head (50 classes)."""

    def __init__(
        self,
        d_model: int = 1280,
        num_classes: int = 50,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: (batch, T, d_model) -> (batch, T, num_classes)"""
        x = self.ln(x)
        x = nn.gelu(self.fc1(x))
        return self.fc2(x)


class PhonemeHead(nn.Module):
    """Kokoro phoneme head (CTC-based)."""

    def __init__(
        self,
        d_model: int = 1280,
        phoneme_vocab: int = 200,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.phoneme_vocab = phoneme_vocab

        self.ln = nn.LayerNorm(d_model)
        self.hidden = nn.Linear(d_model, hidden_dim)
        self.proj = nn.Linear(hidden_dim, phoneme_vocab)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: (batch, T, d_model) -> (batch, T, phoneme_vocab)"""
        x = self.ln(x)
        x = nn.gelu(self.hidden(x))
        return self.proj(x)


class SpeakerEmbeddingHead(nn.Module):
    """
    Speaker embedding head for downstream clustering/verification.

    Outputs a fixed-size embedding per utterance by mean-pooling encoder
    outputs over time, then projecting to a lower-dimensional space.

    NOTE: We do NOT do speaker diarization. That's the downstream system's
    responsibility. We just output embeddings that can be compared using
    cosine similarity.

    Architecture:
        encoder_out (batch, T, d_model=1280)
        -> mean_pool (batch, d_model)
        -> layer_norm (batch, d_model)
        -> linear (batch, hidden_dim)
        -> gelu (batch, hidden_dim)
        -> linear (batch, embed_dim=256)
        -> L2_normalize (optional) (batch, embed_dim)

    Use cases for downstream systems:
        - Compare embeddings across segments for diarization
        - Speaker verification ("is this the same person?")
        - Speaker identification (match against known voices)
        - Per-speaker fine-tuning data collection
    """

    def __init__(
        self,
        d_model: int = 1280,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        normalize: bool = True,
    ):
        """
        Initialize SpeakerEmbeddingHead.

        Args:
            d_model: Whisper encoder output dimension
            embed_dim: Output embedding dimension (256 typical for speaker ID)
            hidden_dim: Hidden layer dimension
            normalize: L2 normalize output embeddings (for cosine similarity)
        """
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.normalize = normalize

        # Layer norm before projection
        self.ln = nn.LayerNorm(d_model)

        # Two-layer projection
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def __call__(
        self,
        encoder_output: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass: mean pool over time and project.

        Args:
            encoder_output: (batch, T, d_model) Whisper encoder output
            attention_mask: (batch, T) Optional mask for variable-length inputs
                           1 = include frame, 0 = exclude frame

        Returns:
            speaker_embedding: (batch, embed_dim) L2-normalized embedding
        """
        # Handle unbatched input
        if encoder_output.ndim == 2:
            encoder_output = encoder_output[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, seq_len, _ = encoder_output.shape

        # Mean pool over time dimension
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask[:, :, None]  # (batch, T, 1)
            masked_output = encoder_output * mask_expanded
            pooled = mx.sum(masked_output, axis=1)  # (batch, d_model)
            lengths = mx.sum(attention_mask, axis=1, keepdims=True)  # (batch, 1)
            pooled = pooled / mx.maximum(lengths, mx.array(1.0))  # Avoid div by zero
        else:
            # Simple mean pooling
            pooled = mx.mean(encoder_output, axis=1)  # (batch, d_model)

        # Layer norm
        pooled = self.ln(pooled)

        # Two-layer projection
        x = nn.gelu(self.fc1(pooled))  # (batch, hidden_dim)
        embedding = self.fc2(x)  # (batch, embed_dim)

        # L2 normalization (for cosine similarity)
        if self.normalize:
            norm = mx.sqrt(mx.sum(embedding ** 2, axis=-1, keepdims=True))
            embedding = embedding / mx.maximum(norm, mx.array(1e-8))

        if squeeze_batch:
            embedding = embedding[0]

        return embedding

    def compute_similarity(
        self,
        embedding1: mx.array,
        embedding2: mx.array,
    ) -> mx.array:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: (embed_dim,) or (batch, embed_dim)
            embedding2: (embed_dim,) or (batch, embed_dim)

        Returns:
            similarity: Scalar or (batch,) cosine similarity [-1, 1]
        """
        if embedding1.ndim == 1:
            embedding1 = embedding1[None, :]
        if embedding2.ndim == 1:
            embedding2 = embedding2[None, :]

        # If normalized, cosine similarity is just dot product
        if self.normalize:
            similarity = mx.sum(embedding1 * embedding2, axis=-1)
        else:
            # Compute proper cosine similarity
            norm1 = mx.sqrt(mx.sum(embedding1 ** 2, axis=-1, keepdims=True))
            norm2 = mx.sqrt(mx.sum(embedding2 ** 2, axis=-1, keepdims=True))
            embedding1_norm = embedding1 / mx.maximum(norm1, mx.array(1e-8))
            embedding2_norm = embedding2 / mx.maximum(norm2, mx.array(1e-8))
            similarity = mx.sum(embedding1_norm * embedding2_norm, axis=-1)

        return similarity


# =============================================================================
# Rich CTC Head (Unified)
# =============================================================================

class RichCTCHead(nn.Module):
    """
    Multi-task CTC head with all frame-aligned outputs at 50Hz.

    Combines:
    - Text CTC (Whisper vocab) - optionally prosody-conditioned
    - Emotion (34 classes)
    - Pitch (CREPE-style, outputs Hz)
    - Paralinguistics (50 classes)
    - Phonemes (Kokoro 200 phonemes)

    All outputs are frame-aligned and can be decoded independently
    or used together for rich audio understanding.

    Prosody Conditioning:
        When config.use_prosody_ctc=True, the text CTC head is replaced
        with ProsodyConditionedCTC which sees emotion and pitch at every
        frame. This improves punctuation prediction by leveraging prosody.
    """

    def __init__(self, config: RichCTCConfig | None = None):
        super().__init__()
        self.config = config or RichCTCConfig()

        # Initialize text CTC - standard or prosody-conditioned
        if self.config.use_prosody_ctc:
            self.text_ctc = ProsodyConditionedCTC(
                d_model=self.config.d_model,
                emotion_dim=self.config.num_emotions,
                pitch_dim=1,
                prosody_dim=self.config.prosody_dim,
                vocab_size=self.config.text_vocab_size,
            )
            self._use_prosody_ctc = True
        else:
            self.text_ctc = TextCTCHead(
                d_model=self.config.d_model,
                vocab_size=self.config.text_vocab_size,
            )
            self._use_prosody_ctc = False

        self.emotion = EmotionHead(
            d_model=self.config.d_model,
            num_emotions=self.config.num_emotions,
            hidden_dim=self.config.emotion_hidden_dim,
        )

        self.pitch = CREPEPitchHead(
            d_model=self.config.d_model,
            hidden_dim=self.config.pitch_hidden_dim,
            num_bins=self.config.pitch_bins,
        )

        self.para = ParalinguisticsHead(
            d_model=self.config.d_model,
            num_classes=self.config.num_para_classes,
            hidden_dim=self.config.para_hidden_dim,
        )

        self.phoneme = PhonemeHead(
            d_model=self.config.d_model,
            phoneme_vocab=self.config.phoneme_vocab,
            hidden_dim=self.config.phoneme_hidden_dim,
        )

        # Speaker embedding head (Phase 8) - optional, utterance-level
        if self.config.use_speaker_embedding:
            self.speaker = SpeakerEmbeddingHead(
                d_model=self.config.d_model,
                embed_dim=self.config.speaker_embed_dim,
                hidden_dim=self.config.speaker_hidden_dim,
                normalize=self.config.speaker_normalize,
            )
            self._use_speaker_embedding = True
        else:
            self.speaker = None
            self._use_speaker_embedding = False

    def __call__(
        self,
        encoder_output: mx.array,
        return_timing: bool = True,
    ) -> dict[str, mx.array]:
        """
        Forward pass through all heads.

        Args:
            encoder_output: (batch, T, d_model) Whisper encoder output
            return_timing: Include frame timing information

        Returns:
            Dictionary with all outputs:
                text_logits: (batch, T, vocab_size)
                emotion: (batch, T, num_emotions)
                pitch_bins: (batch, T, 361)
                pitch_hz: (batch, T, 1)
                para: (batch, T, 50)
                phoneme: (batch, T, phoneme_vocab)
                start_time_ms: (batch, T) - if return_timing
                end_time_ms: (batch, T) - if return_timing
                speaker_embedding: (batch, embed_dim) - if use_speaker_embedding
        """
        # Add batch dimension if needed
        if encoder_output.ndim == 2:
            encoder_output = encoder_output[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, num_frames, _ = encoder_output.shape

        # Run emotion and pitch heads first (needed for prosody conditioning)
        emotion_logits = self.emotion(encoder_output)
        pitch_bins, pitch_hz = self.pitch(encoder_output)

        # Run text CTC - with or without prosody conditioning
        if self._use_prosody_ctc:
            # Prosody-conditioned CTC sees emotion and pitch per frame
            # Normalize pitch to [0, 1] range for stable training
            pitch_normalized = pitch_hz / 500.0  # Rough normalization
            text_logits = self.text_ctc(encoder_output, emotion_logits, pitch_normalized)
        else:
            text_logits = self.text_ctc(encoder_output)

        # Run remaining heads
        para_logits = self.para(encoder_output)
        phoneme_logits = self.phoneme(encoder_output)

        outputs = {
            "text_logits": text_logits,
            "emotion": emotion_logits,
            "pitch_bins": pitch_bins,
            "pitch_hz": pitch_hz,
            "para": para_logits,
            "phoneme": phoneme_logits,
        }

        # Speaker embedding (utterance-level, not frame-aligned)
        if self._use_speaker_embedding and self.speaker is not None:
            speaker_embedding = self.speaker(encoder_output)
            outputs["speaker_embedding"] = speaker_embedding

        # Add timing information
        if return_timing:
            frame_duration_ms = 1000.0 / self.config.frame_rate_hz  # 20ms
            frame_indices = mx.arange(num_frames)

            # Start time of each frame (in ms)
            start_times = frame_indices * frame_duration_ms
            # End time of each frame
            end_times = (frame_indices + 1) * frame_duration_ms

            # Broadcast to batch dimension
            outputs["start_time_ms"] = mx.broadcast_to(
                start_times[None, :], (batch_size, num_frames),
            )
            outputs["end_time_ms"] = mx.broadcast_to(
                end_times[None, :], (batch_size, num_frames),
            )
            outputs["frame_indices"] = mx.broadcast_to(
                frame_indices[None, :], (batch_size, num_frames),
            )

        # Remove batch dimension if we added it
        if squeeze_batch:
            outputs = {k: v[0] for k, v in outputs.items()}

        return outputs

    def get_speaker_embedding(
        self,
        encoder_output: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array | None:
        """
        Get speaker embedding from encoder output.

        This is a convenience method that can be called separately from
        the main forward pass if you only need the speaker embedding.

        Args:
            encoder_output: (batch, T, d_model) or (T, d_model) Whisper encoder output
            attention_mask: (batch, T) or (T,) Optional mask for variable-length inputs

        Returns:
            speaker_embedding: (batch, embed_dim) or (embed_dim,) L2-normalized embedding,
                             or None if speaker embedding head is not enabled
        """
        if not self._use_speaker_embedding or self.speaker is None:
            return None

        return self.speaker(encoder_output, attention_mask)

    def compute_speaker_similarity(
        self,
        embedding1: mx.array,
        embedding2: mx.array,
    ) -> mx.array:
        """
        Compute cosine similarity between two speaker embeddings.

        Args:
            embedding1: (embed_dim,) or (batch, embed_dim) First embedding
            embedding2: (embed_dim,) or (batch, embed_dim) Second embedding

        Returns:
            similarity: Scalar or (batch,) cosine similarity in range [-1, 1]
                       Values > 0.6 typically indicate same speaker

        Raises:
            ValueError: If speaker embedding head is not enabled
        """
        if not self._use_speaker_embedding or self.speaker is None:
            raise ValueError("Speaker embedding head is not enabled. "
                           "Set use_speaker_embedding=True in config.")

        return self.speaker.compute_similarity(embedding1, embedding2)

    def decode_text_greedy(
        self,
        outputs: dict[str, mx.array],
        blank_id: int = 0,
    ) -> tuple[list[int], list[tuple[int, float, float]]]:
        """
        CTC greedy decoding for text with timestamps.

        Args:
            outputs: Output dict from forward pass
            blank_id: CTC blank token ID

        Returns:
            tokens: Collapsed token sequence
            tokens_with_timing: List of (token_id, start_ms, end_ms)
        """
        text_logits = outputs["text_logits"]
        if text_logits.ndim == 3:
            text_logits = text_logits[0]

        predictions = mx.argmax(text_logits, axis=-1)
        mx.eval(predictions)

        start_times = outputs.get("start_time_ms", None)
        end_times = outputs.get("end_time_ms", None)

        if start_times is not None and start_times.ndim == 2:
            start_times = start_times[0]
            end_times = end_times[0]

        tokens = []
        tokens_with_timing = []
        prev_token = blank_id
        token_start_frame = 0

        predictions_list = predictions.tolist()
        for frame_idx, token in enumerate(predictions_list):
            if token != blank_id and token != prev_token:
                # New token - record end of previous and start of new
                start_ms = float(start_times[token_start_frame]) if start_times is not None else frame_idx * 20.0
                end_ms = float(end_times[frame_idx]) if end_times is not None else (frame_idx + 1) * 20.0

                tokens.append(token)
                tokens_with_timing.append((token, start_ms, end_ms))
                token_start_frame = frame_idx
            prev_token = token

        return tokens, tokens_with_timing

    def decode_phonemes_greedy(
        self,
        outputs: dict[str, mx.array],
        blank_id: int = 0,
    ) -> list[int]:
        """CTC greedy decoding for phonemes."""
        phoneme_logits = outputs["phoneme"]
        if phoneme_logits.ndim == 3:
            phoneme_logits = phoneme_logits[0]

        predictions = mx.argmax(phoneme_logits, axis=-1)
        mx.eval(predictions)

        # Collapse blanks and repeats
        tokens = []
        prev = blank_id
        for t in predictions.tolist():
            if t != blank_id and t != prev:
                tokens.append(t)
            prev = t

        return tokens

    def get_emotion_per_frame(
        self,
        outputs: dict[str, mx.array],
    ) -> tuple[list[str], list[float]]:
        """Get emotion labels and confidences per frame."""
        emotion_logits = outputs["emotion"]
        if emotion_logits.ndim == 3:
            emotion_logits = emotion_logits[0]

        probs = mx.softmax(emotion_logits, axis=-1)
        mx.eval(probs)

        predictions = mx.argmax(emotion_logits, axis=-1)
        mx.eval(predictions)

        labels = []
        confidences = []

        num_classes = emotion_logits.shape[-1]
        emotion_classes = EMOTION_CLASSES_34 if num_classes == 34 else EMOTION_CLASSES_8

        for frame_idx, pred in enumerate(predictions.tolist()):
            if pred < len(emotion_classes):
                labels.append(emotion_classes[pred])
            else:
                labels.append(f"unknown_{pred}")
            confidences.append(float(probs[frame_idx, pred]))

        return labels, confidences

    def get_utterance_emotion(
        self,
        outputs: dict[str, mx.array],
    ) -> tuple[str, float]:
        """Get utterance-level emotion via mean pooling."""
        emotion_logits = outputs["emotion"]
        if emotion_logits.ndim == 3:
            emotion_logits = emotion_logits[0]

        # Mean pool across frames
        pooled = mx.mean(emotion_logits, axis=0)
        probs = mx.softmax(pooled, axis=-1)
        mx.eval(probs)

        pred = int(mx.argmax(pooled))

        num_classes = emotion_logits.shape[-1]
        emotion_classes = EMOTION_CLASSES_34 if num_classes == 34 else EMOTION_CLASSES_8

        label = emotion_classes[pred] if pred < len(emotion_classes) else f"unknown_{pred}"
        confidence = float(probs[pred])

        return label, confidence

    def get_pitch_hz(
        self,
        outputs: dict[str, mx.array],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get pitch in Hz and voicing probability per frame."""
        pitch_hz = outputs["pitch_hz"]
        pitch_bins = outputs["pitch_bins"]

        if pitch_hz.ndim == 3:
            pitch_hz = pitch_hz[0]
            pitch_bins = pitch_bins[0]

        mx.eval(pitch_hz)
        mx.eval(pitch_bins)

        # Voicing probability is 1 - unvoiced probability
        unvoiced_prob = mx.sigmoid(pitch_bins[..., 360])
        voicing = 1.0 - unvoiced_prob
        mx.eval(voicing)

        return np.array(pitch_hz.squeeze(-1)), np.array(voicing)

    def get_paralinguistics(
        self,
        outputs: dict[str, mx.array],
        threshold: float = 0.5,
    ) -> list[tuple[int, str, float, float, float]]:
        """
        Detect paralinguistic events.

        Args:
            outputs: Output dict from forward pass
            threshold: Detection threshold

        Returns:
            List of (class_id, class_name, start_ms, end_ms, confidence)
        """
        para_logits = outputs["para"]
        if para_logits.ndim == 3:
            para_logits = para_logits[0]

        start_times = outputs.get("start_time_ms", None)
        end_times = outputs.get("end_time_ms", None)

        if start_times is not None and start_times.ndim == 2:
            start_times = start_times[0]
            end_times = end_times[0]

        probs = mx.softmax(para_logits, axis=-1)
        mx.eval(probs)

        predictions = mx.argmax(para_logits, axis=-1)
        mx.eval(predictions)

        events = []
        current_class = None
        current_start = 0

        for frame_idx, (pred, _prob_vec) in enumerate(
            zip(predictions.tolist(), np.array(probs), strict=False),
        ):
            if pred == 0:  # speech
                if current_class is not None and current_class != 0:
                    # End of non-speech event
                    end_ms = float(end_times[frame_idx - 1]) if end_times is not None else frame_idx * 20.0
                    start_ms = float(start_times[current_start]) if start_times is not None else current_start * 20.0
                    avg_conf = float(np.mean([probs[i, current_class].item() for i in range(current_start, frame_idx)]))

                    if avg_conf >= threshold:
                        class_name = PARA_CLASSES_INV.get(current_class, f"class_{current_class}")
                        events.append((current_class, class_name, start_ms, end_ms, avg_conf))

                current_class = 0
                current_start = frame_idx
            elif pred != current_class:
                # New non-speech event
                if current_class is not None and current_class != 0:
                    end_ms = float(end_times[frame_idx - 1]) if end_times is not None else frame_idx * 20.0
                    start_ms = float(start_times[current_start]) if start_times is not None else current_start * 20.0
                    avg_conf = float(np.mean([probs[i, current_class].item() for i in range(current_start, frame_idx)]))

                    if avg_conf >= threshold:
                        class_name = PARA_CLASSES_INV.get(current_class, f"class_{current_class}")
                        events.append((current_class, class_name, start_ms, end_ms, avg_conf))

                current_class = pred
                current_start = frame_idx

        # Handle final event
        if current_class is not None and current_class != 0:
            num_frames = len(predictions.tolist())
            end_ms = float(end_times[num_frames - 1]) if end_times is not None else num_frames * 20.0
            start_ms = float(start_times[current_start]) if start_times is not None else current_start * 20.0
            avg_conf = float(np.mean([probs[i, current_class].item() for i in range(current_start, num_frames)]))

            if avg_conf >= threshold:
                class_name = PARA_CLASSES_INV.get(current_class, f"class_{current_class}")
                events.append((current_class, class_name, start_ms, end_ms, avg_conf))

        return events

    @classmethod
    def from_pretrained(
        cls,
        ctc_path: str | None = None,
        emotion_path: str | None = None,
        pitch_path: str | None = None,
        phoneme_path: str | None = None,
        para_path: str | None = None,
        config: RichCTCConfig | None = None,
    ) -> RichCTCHead:
        """
        Load RichCTCHead from pretrained checkpoint files.

        Uses default paths if not specified. Will skip heads where
        checkpoint is not found (weights remain random).

        Args:
            ctc_path: Path to CTC head weights
            emotion_path: Path to emotion head weights
            pitch_path: Path to CREPE pitch head weights
            phoneme_path: Path to phoneme head weights
            para_path: Path to paralinguistics head weights
            config: Override configuration

        Returns:
            RichCTCHead with loaded weights
        """
        # Use defaults if not specified
        ctc_path = ctc_path or DEFAULT_CHECKPOINTS["ctc"]
        emotion_path = emotion_path or DEFAULT_CHECKPOINTS["emotion"]
        pitch_path = pitch_path or DEFAULT_CHECKPOINTS["pitch"]
        phoneme_path = phoneme_path or DEFAULT_CHECKPOINTS["phoneme"]
        para_path = para_path or DEFAULT_CHECKPOINTS["para"]

        # Create model
        model = cls(config=config)

        # Load CTC head
        if Path(ctc_path).exists():
            weights = dict(mx.load(str(ctc_path)))
            if "ln.weight" in weights:
                model.text_ctc.ln.weight = weights["ln.weight"]
                model.text_ctc.ln.bias = weights["ln.bias"]
            if "proj.weight" in weights:
                model.text_ctc.proj.weight = weights["proj.weight"]
                model.text_ctc.proj.bias = weights["proj.bias"]
            print(f"Loaded CTC head from {ctc_path}")
        else:
            print(f"CTC checkpoint not found: {ctc_path}")

        # Load emotion head
        if Path(emotion_path).exists():
            weights = dict(mx.load(str(emotion_path)))
            if "emotion.ln.weight" in weights:
                model.emotion.ln.weight = weights["emotion.ln.weight"]
                model.emotion.ln.bias = weights["emotion.ln.bias"]
                model.emotion.fc1.weight = weights["emotion.fc1.weight"]
                model.emotion.fc1.bias = weights["emotion.fc1.bias"]
                model.emotion.fc2.weight = weights["emotion.fc2.weight"]
                model.emotion.fc2.bias = weights["emotion.fc2.bias"]
                print(f"Loaded emotion head from {emotion_path}")
        else:
            print(f"Emotion checkpoint not found: {emotion_path}")

        # Load pitch head (CREPE)
        if Path(pitch_path).exists():
            weights = dict(mx.load(str(pitch_path)))
            if "pitch.ln_input.weight" in weights:
                model.pitch.ln_input.weight = weights["pitch.ln_input.weight"]
                model.pitch.ln_input.bias = weights["pitch.ln_input.bias"]
                model.pitch.input_proj.weight = weights["pitch.input_proj.weight"]
                model.pitch.input_proj.bias = weights["pitch.input_proj.bias"]
                model.pitch.output_proj.weight = weights["pitch.output_proj.weight"]
                model.pitch.output_proj.bias = weights["pitch.output_proj.bias"]

                # Load conv layers
                for i in range(5):
                    prefix = f"pitch.conv_layers.{i}"
                    if f"{prefix}.conv.conv.weight" in weights:
                        model.pitch.conv_layers[i]["conv"].weight = weights[f"{prefix}.conv.conv.weight"]
                        model.pitch.conv_layers[i]["conv"].bias = weights[f"{prefix}.conv.conv.bias"]
                        model.pitch.conv_layers[i]["ln"].weight = weights[f"{prefix}.ln.weight"]
                        model.pitch.conv_layers[i]["ln"].bias = weights[f"{prefix}.ln.bias"]

                print(f"Loaded pitch head from {pitch_path}")
        else:
            print(f"Pitch checkpoint not found: {pitch_path}")

        # Load phoneme head
        if Path(phoneme_path).exists():
            weights = dict(mx.load(str(phoneme_path)))
            if "ln.weight" in weights:
                model.phoneme.ln.weight = weights["ln.weight"]
                model.phoneme.ln.bias = weights["ln.bias"]
            if "hidden.weight" in weights:
                model.phoneme.hidden.weight = weights["hidden.weight"]
                model.phoneme.hidden.bias = weights["hidden.bias"]
            if "proj.weight" in weights:
                model.phoneme.proj.weight = weights["proj.weight"]
                model.phoneme.proj.bias = weights["proj.bias"]
            print(f"Loaded phoneme head from {phoneme_path}")
        else:
            print(f"Phoneme checkpoint not found: {phoneme_path}")

        # Load paralinguistics head
        if Path(para_path).exists():
            weights = dict(mx.load(str(para_path)))
            loaded_parts = []

            # Load LayerNorm (always compatible - d_model doesn't change)
            if "paralinguistics.ln.weight" in weights:
                model.para.ln.weight = weights["paralinguistics.ln.weight"]
                model.para.ln.bias = weights["paralinguistics.ln.bias"]
                loaded_parts.append("ln")

            # Load fc1 (d_model -> hidden_dim, compatible if hidden_dim matches)
            if "paralinguistics.fc1.weight" in weights:
                ckpt_fc1_shape = weights["paralinguistics.fc1.weight"].shape
                model_fc1_shape = model.para.fc1.weight.shape
                if ckpt_fc1_shape == model_fc1_shape:
                    model.para.fc1.weight = weights["paralinguistics.fc1.weight"]
                    model.para.fc1.bias = weights["paralinguistics.fc1.bias"]
                    loaded_parts.append("fc1")
                else:
                    print(f"  Para fc1 shape mismatch: ckpt={ckpt_fc1_shape}, model={model_fc1_shape}")

            # Load fc2 (hidden_dim -> num_classes, may not match if class count differs)
            if "paralinguistics.fc2.weight" in weights:
                ckpt_fc2_shape = weights["paralinguistics.fc2.weight"].shape
                model_fc2_shape = model.para.fc2.weight.shape
                if ckpt_fc2_shape == model_fc2_shape:
                    model.para.fc2.weight = weights["paralinguistics.fc2.weight"]
                    model.para.fc2.bias = weights["paralinguistics.fc2.bias"]
                    loaded_parts.append("fc2")
                else:
                    # Class count mismatch - copy compatible portion
                    ckpt_classes = ckpt_fc2_shape[0]
                    model_classes = model_fc2_shape[0]
                    min_classes = min(ckpt_classes, model_classes)
                    # Copy first min_classes rows (shared classes)
                    # MLX: create new array with partial update
                    new_weight = mx.array(model.para.fc2.weight)
                    new_bias = mx.array(model.para.fc2.bias)
                    new_weight[:min_classes, :] = weights["paralinguistics.fc2.weight"][:min_classes, :]
                    new_bias[:min_classes] = weights["paralinguistics.fc2.bias"][:min_classes]
                    model.para.fc2.weight = new_weight
                    model.para.fc2.bias = new_bias
                    loaded_parts.append(f"fc2 (partial: {min_classes}/{model_classes} classes)")
                    print(f"  Para class mismatch: ckpt={ckpt_classes}, model={model_classes} - loaded first {min_classes}")

            if loaded_parts:
                print(f"Loaded para head from {para_path}: {', '.join(loaded_parts)}")
        else:
            print(f"Para checkpoint not found: {para_path}")

        mx.eval(model.parameters())
        return model

    def save(self, path: str) -> None:
        """
        Save all head weights to a single file.

        Args:
            path: Output path (.npz file)
        """
        weights = {}

        # CTC head
        weights["text_ctc.ln.weight"] = self.text_ctc.ln.weight
        weights["text_ctc.ln.bias"] = self.text_ctc.ln.bias
        weights["text_ctc.proj.weight"] = self.text_ctc.proj.weight
        weights["text_ctc.proj.bias"] = self.text_ctc.proj.bias

        # Emotion head
        weights["emotion.ln.weight"] = self.emotion.ln.weight
        weights["emotion.ln.bias"] = self.emotion.ln.bias
        weights["emotion.fc1.weight"] = self.emotion.fc1.weight
        weights["emotion.fc1.bias"] = self.emotion.fc1.bias
        weights["emotion.fc2.weight"] = self.emotion.fc2.weight
        weights["emotion.fc2.bias"] = self.emotion.fc2.bias

        # Pitch head
        weights["pitch.ln_input.weight"] = self.pitch.ln_input.weight
        weights["pitch.ln_input.bias"] = self.pitch.ln_input.bias
        weights["pitch.input_proj.weight"] = self.pitch.input_proj.weight
        weights["pitch.input_proj.bias"] = self.pitch.input_proj.bias
        weights["pitch.output_proj.weight"] = self.pitch.output_proj.weight
        weights["pitch.output_proj.bias"] = self.pitch.output_proj.bias
        for i, layer in enumerate(self.pitch.conv_layers):
            weights[f"pitch.conv_layers.{i}.conv.weight"] = layer["conv"].weight
            weights[f"pitch.conv_layers.{i}.conv.bias"] = layer["conv"].bias
            weights[f"pitch.conv_layers.{i}.ln.weight"] = layer["ln"].weight
            weights[f"pitch.conv_layers.{i}.ln.bias"] = layer["ln"].bias

        # Para head
        weights["para.ln.weight"] = self.para.ln.weight
        weights["para.ln.bias"] = self.para.ln.bias
        weights["para.fc1.weight"] = self.para.fc1.weight
        weights["para.fc1.bias"] = self.para.fc1.bias
        weights["para.fc2.weight"] = self.para.fc2.weight
        weights["para.fc2.bias"] = self.para.fc2.bias

        # Phoneme head
        weights["phoneme.ln.weight"] = self.phoneme.ln.weight
        weights["phoneme.ln.bias"] = self.phoneme.ln.bias
        weights["phoneme.hidden.weight"] = self.phoneme.hidden.weight
        weights["phoneme.hidden.bias"] = self.phoneme.hidden.bias
        weights["phoneme.proj.weight"] = self.phoneme.proj.weight
        weights["phoneme.proj.bias"] = self.phoneme.proj.bias

        mx.savez(path, **weights)


# =============================================================================
# Language Detection Utilities
# =============================================================================

class LanguageHead(nn.Module):
    """
    Per-token language ID for code-switching detection.

    Detects language at each frame (50Hz), enabling detection of
    mixed-language utterances like "I went to the 商店 yesterday".

    This is separate from Whisper's utterance-level language detection.
    """

    def __init__(
        self,
        d_model: int = 1280,
        num_languages: int = 100,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_languages = num_languages

        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_languages)

    def __call__(self, encoder_output: mx.array) -> mx.array:
        """
        Forward pass: (batch, T, d_model) -> (batch, T, num_languages)

        Returns language logits per frame.
        """
        x = self.ln(encoder_output)
        x = nn.gelu(self.fc1(x))
        return self.fc2(x)


def detect_language_from_tokens(
    token_ids: list[int],
) -> str | None:
    """
    Detect language from Whisper output tokens.

    Whisper outputs a language token early in the sequence.
    This function extracts the language code from that token.

    Args:
        token_ids: Output token IDs from Whisper decoder

    Returns:
        Language code (e.g., "en", "zh") or None if not found
    """
    for token_id in token_ids[:10]:  # Language token is early
        if token_id in WHISPER_ID_TO_LANGUAGE:
            return WHISPER_ID_TO_LANGUAGE[token_id]
    return None


def get_language_probs_from_logits(
    decoder_logits: mx.array,
    position: int = 1,
) -> dict[str, float]:
    """
    Get language probabilities from decoder logits at a specific position.

    Whisper's decoder outputs language probabilities at position 1
    (after the <|startoftranscript|> token).

    Args:
        decoder_logits: (vocab_size,) or (batch, vocab_size) logits
        position: Position in sequence for language prediction (default: 1)

    Returns:
        Dictionary mapping language codes to probabilities
    """
    if decoder_logits.ndim == 2:
        decoder_logits = decoder_logits[0]

    # Extract logits for language tokens only
    lang_indices = list(WHISPER_LANGUAGE_IDS.values())
    min_idx = min(lang_indices)
    max_idx = max(lang_indices)

    lang_logits = decoder_logits[min_idx:max_idx + 1]
    lang_probs = mx.softmax(lang_logits, axis=-1)
    mx.eval(lang_probs)

    # Map to language codes
    results = {}
    for lang_code, token_id in WHISPER_LANGUAGE_IDS.items():
        idx = token_id - min_idx
        if 0 <= idx < len(lang_probs):
            results[lang_code] = float(lang_probs[idx])

    # Sort by probability
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


def get_top_languages(
    decoder_logits: mx.array,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Get top-k most likely languages from decoder logits.

    Args:
        decoder_logits: Decoder logits at language prediction position
        top_k: Number of languages to return

    Returns:
        List of (language_code, probability) tuples, sorted by probability
    """
    all_probs = get_language_probs_from_logits(decoder_logits)
    return list(all_probs.items())[:top_k]


# =============================================================================
# Rich Token Output (Data Structure)
# =============================================================================

@dataclass
class RichToken:
    """
    Single token with all rich information.

    Same schema for CTC and Decoder outputs, enabling easy comparison.
    """
    # Alignment
    alignment_id: str
    stream: str  # "ctc" or "decoder"
    timestamp_ms: float

    # Timing (in audio)
    start_time_ms: float
    end_time_ms: float
    start_frame: int
    end_frame: int

    # Content
    token: str
    token_id: int
    confidence: float

    # Language
    language: str
    language_confidence: float

    # Emotion
    emotion: str
    emotion_confidence: float

    # Pitch
    pitch_hz: float
    pitch_confidence: float

    # Phonemes
    phonemes: list[str]
    phoneme_confidence: list[float]
    phoneme_deviation: float

    # Paralinguistics
    para_class: int | None
    para_confidence: float | None

    # Speaker (optional, for downstream clustering)
    speaker_embedding: list[float] | None = None


def outputs_to_rich_tokens(
    outputs: dict[str, mx.array],
    token_ids: list[int],
    tokenizer: Any = None,
    language: str = "en",
    stream: str = "ctc",
) -> list[RichToken]:
    """
    Convert RichCTCHead outputs to a list of RichToken objects.

    Args:
        outputs: Output dict from RichCTCHead forward pass
        token_ids: Decoded text token IDs (from CTC greedy decode)
        tokenizer: Whisper tokenizer for decoding tokens to text
        language: Detected language code
        stream: Stream identifier ("ctc" or "decoder")

    Returns:
        List of RichToken objects
    """
    import uuid

    # Get timing info
    start_times = outputs.get("start_time_ms", None)
    end_times = outputs.get("end_time_ms", None)

    if start_times is not None and start_times.ndim == 2:
        start_times = np.array(start_times[0])
        end_times = np.array(end_times[0])
    elif start_times is not None:
        start_times = np.array(start_times)
        end_times = np.array(end_times)

    # Get emotion per frame
    emotion_logits = outputs["emotion"]
    if emotion_logits.ndim == 3:
        emotion_logits = emotion_logits[0]
    emotion_probs = mx.softmax(emotion_logits, axis=-1)
    emotion_preds = mx.argmax(emotion_logits, axis=-1)
    mx.eval(emotion_probs)
    mx.eval(emotion_preds)

    # Get pitch per frame
    pitch_hz = outputs["pitch_hz"]
    if pitch_hz.ndim == 3:
        pitch_hz = pitch_hz[0]
    mx.eval(pitch_hz)
    pitch_np = np.array(pitch_hz.squeeze(-1))

    # Get voicing probability from pitch bins (for pitch_confidence)
    pitch_bins = outputs.get("pitch_bins", None)
    if pitch_bins is not None:
        if pitch_bins.ndim == 3:
            pitch_bins = pitch_bins[0]
        # Last bin (360) is unvoiced bin - voicing = 1 - sigmoid(unvoiced_logit)
        unvoiced_prob = mx.sigmoid(pitch_bins[..., 360])
        voicing_np = np.array(1.0 - unvoiced_prob)
        mx.eval(voicing_np)
    else:
        voicing_np = np.ones_like(pitch_np)  # Default to confident voicing

    # Get paralinguistics per frame
    para_logits = outputs["para"]
    if para_logits.ndim == 3:
        para_logits = para_logits[0]
    para_probs = mx.softmax(para_logits, axis=-1)
    para_preds = mx.argmax(para_logits, axis=-1)
    mx.eval(para_probs)
    mx.eval(para_preds)

    # Get phonemes
    phoneme_logits = outputs["phoneme"]
    if phoneme_logits.ndim == 3:
        phoneme_logits = phoneme_logits[0]
    phoneme_probs = mx.softmax(phoneme_logits, axis=-1)
    phoneme_preds = mx.argmax(phoneme_logits, axis=-1)
    mx.eval(phoneme_probs)
    mx.eval(phoneme_preds)

    # CTC decode with frame alignment
    text_logits = outputs["text_logits"]
    if text_logits.ndim == 3:
        text_logits = text_logits[0]
    text_preds = mx.argmax(text_logits, axis=-1)
    text_probs = mx.softmax(text_logits, axis=-1)
    mx.eval(text_preds)
    mx.eval(text_probs)

    # Build tokens
    rich_tokens = []
    num_classes = emotion_logits.shape[-1]
    emotion_classes = EMOTION_CLASSES_34 if num_classes == 34 else EMOTION_CLASSES_8

    prev_token_id = 0  # CTC blank
    token_start_frame = 0

    for frame_idx, token_id in enumerate(text_preds.tolist()):
        if token_id != 0 and token_id != prev_token_id:
            # New token - create RichToken
            alignment_id = str(uuid.uuid4())[:8]
            start_ms = float(start_times[token_start_frame]) if start_times is not None else token_start_frame * 20.0
            end_ms = float(end_times[frame_idx]) if end_times is not None else (frame_idx + 1) * 20.0

            # Get token text
            if tokenizer is not None:
                try:
                    token_text = tokenizer.decode([token_id])
                except Exception:
                    token_text = f"<{token_id}>"
            else:
                token_text = f"<{token_id}>"

            # Get emotion at this frame
            emotion_pred = int(emotion_preds[frame_idx])
            emotion_label = emotion_classes[emotion_pred] if emotion_pred < len(emotion_classes) else "unknown"
            emotion_conf = float(emotion_probs[frame_idx, emotion_pred])

            # Get pitch at this frame
            frame_pitch = float(pitch_np[frame_idx])

            # Get paralinguistics at this frame
            para_pred = int(para_preds[frame_idx])
            para_conf = float(para_probs[frame_idx, para_pred])
            para_class_val = para_pred if para_pred != 0 else None  # 0 = speech
            para_conf_val = para_conf if para_pred != 0 else None

            # Get phonemes across token's frame span with CTC collapse
            # Collect frame-level predictions and collapse (remove blanks/duplicates)
            token_phonemes = []
            token_phoneme_confs = []
            prev_phoneme = -1
            for f_idx in range(token_start_frame, frame_idx + 1):
                p_id = int(phoneme_preds[f_idx])
                if p_id != 0 and p_id != prev_phoneme:  # 0 is blank
                    token_phonemes.append(str(p_id))
                    token_phoneme_confs.append(float(phoneme_probs[f_idx, p_id]))
                prev_phoneme = p_id

            # If no valid phonemes, use frame phoneme
            if not token_phonemes:
                phoneme_pred = int(phoneme_preds[frame_idx])
                token_phonemes = [str(phoneme_pred)]
                token_phoneme_confs = [float(phoneme_probs[frame_idx, phoneme_pred])]

            # Compute phoneme_deviation as (1 - mean_confidence)
            # Higher deviation = lower confidence = potential mismatch
            avg_phoneme_conf = sum(token_phoneme_confs) / len(token_phoneme_confs)
            phoneme_dev = 1.0 - avg_phoneme_conf

            rich_token = RichToken(
                alignment_id=alignment_id,
                stream=stream,
                timestamp_ms=end_ms,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                start_frame=token_start_frame,
                end_frame=frame_idx,
                token=token_text,
                token_id=token_id,
                confidence=float(text_probs[frame_idx, token_id]),
                language=language,
                language_confidence=1.0,  # Set by Whisper utterance-level
                emotion=emotion_label,
                emotion_confidence=emotion_conf,
                pitch_hz=frame_pitch,
                pitch_confidence=float(voicing_np[frame_idx]),  # Voicing probability from pitch bins
                phonemes=token_phonemes,
                phoneme_confidence=token_phoneme_confs,
                phoneme_deviation=phoneme_dev,
                para_class=para_class_val,
                para_confidence=para_conf_val,
            )
            rich_tokens.append(rich_token)
            token_start_frame = frame_idx

        prev_token_id = token_id

    return rich_tokens


# =============================================================================
# Test
# =============================================================================

def test_rich_ctc_head():
    """Test RichCTCHead forward pass and methods."""
    print("Testing RichCTCHead...")

    if not HAS_MLX:
        print("MLX not available, skipping")
        return False

    # Create head
    config = RichCTCConfig()
    head = RichCTCHead(config)
    print(f"  Created head with config: d_model={config.d_model}")

    # Mock encoder output (batch=1, T=100 frames, d_model=1280)
    encoder_output = mx.random.normal((1, 100, 1280))

    # Forward pass
    outputs = head(encoder_output)
    mx.eval(outputs)

    # Verify shapes
    print("  Output shapes:")
    expected_shapes = {
        "text_logits": (1, 100, config.text_vocab_size),
        "emotion": (1, 100, config.num_emotions),
        "pitch_bins": (1, 100, config.pitch_bins),
        "pitch_hz": (1, 100, 1),
        "para": (1, 100, config.num_para_classes),
        "phoneme": (1, 100, config.phoneme_vocab),
        "start_time_ms": (1, 100),
        "end_time_ms": (1, 100),
        "frame_indices": (1, 100),
    }

    all_correct = True
    for key, expected in expected_shapes.items():
        actual = outputs[key].shape
        status = "OK" if actual == expected else "FAIL"
        print(f"    {key}: {actual} (expected {expected}) [{status}]")
        if actual != expected:
            all_correct = False

    # Test decoding methods
    print("  Testing decoding methods...")

    tokens, tokens_with_timing = head.decode_text_greedy(outputs)
    print(f"    decode_text_greedy: {len(tokens)} tokens")

    phonemes = head.decode_phonemes_greedy(outputs)
    print(f"    decode_phonemes_greedy: {len(phonemes)} phonemes")

    emotion_label, emotion_conf = head.get_utterance_emotion(outputs)
    print(f"    get_utterance_emotion: {emotion_label} ({emotion_conf:.3f})")

    pitch_hz, voicing = head.get_pitch_hz(outputs)
    print(f"    get_pitch_hz: shape={pitch_hz.shape}, mean={pitch_hz.mean():.1f} Hz")

    para_events = head.get_paralinguistics(outputs, threshold=0.1)
    print(f"    get_paralinguistics: {len(para_events)} events")

    # Verify timing
    start_times = np.array(outputs["start_time_ms"][0])
    end_times = np.array(outputs["end_time_ms"][0])
    print(f"  Timing: first frame {start_times[0]:.1f}-{end_times[0]:.1f} ms")
    print(f"          last frame {start_times[-1]:.1f}-{end_times[-1]:.1f} ms")

    if all_correct:
        print("RichCTCHead tests PASSED")
    else:
        print("RichCTCHead tests FAILED")

    return all_correct


def test_load_pretrained():
    """Test loading from pretrained checkpoints."""
    print("\nTesting from_pretrained...")

    if not HAS_MLX:
        print("MLX not available, skipping")
        return False

    # Load with defaults (will print which checkpoints are found)
    head = RichCTCHead.from_pretrained()

    # Run forward pass
    encoder_output = mx.random.normal((1, 100, 1280))
    outputs = head(encoder_output)
    mx.eval(outputs)

    print(f"  Forward pass successful, {len(outputs)} outputs")

    return True


if __name__ == "__main__":
    test_rich_ctc_head()
    test_load_pretrained()
