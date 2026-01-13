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
Multi-Head Architecture for Whisper Encoder.

This module implements multiple output heads that share a frozen Whisper encoder:
1. CTC Head - Text transcription via CTC decoding
2. Emotion Head - 8-class emotion classification (RAVDESS: angry, calm, disgust, fear, happy, neutral, sad, surprise)
3. Singing Head - Binary singing vs speaking detection
4. Pitch Head - Continuous F0 prediction for real-time harmonization

Architecture:
    Audio -> Mel Spectrogram -> Whisper Encoder (frozen) -> [CTC Head, Emotion Head, Singing Head, Pitch Head]

Key Features:
- Shared encoder: Efficient multi-task learning
- Independent heads: Can train any subset
- Real-time capable: Frame-level predictions for streaming
- Paralinguistics support: Pitch, emotion, singing detection for expressive AI

Target Use Cases:
- Real-time harmonization: Detect singing + track pitch to harmonize in real-time
- Emotional STT: Transcribe text with emotion labels
- Expressive AI: Enable AI to understand and respond to human emotional expression

References:
- RAVDESS Dataset: https://zenodo.org/record/1188976
- CTC: Graves et al. "Connectionist Temporal Classification" (2006)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tools.whisper_mlx.ctc_decoder import CTCDraftHead

import mlx.core as mx
import mlx.nn as nn

# Canonical 9-class emotion taxonomy (v2.0)
# Import from label_taxonomy.py for the single source of truth
from .label_taxonomy import EMOTION_CLASSES_9

# RAVDESS_EMOTIONS is now an alias for EMOTION_CLASSES_9 (9 classes including contempt)
RAVDESS_EMOTIONS = EMOTION_CLASSES_9  # ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised", "contempt"]

# Extended emotion taxonomy (includes Expresso's 34 styles)
# Grouped by category for easier mapping
EXTENDED_EMOTIONS = [
    # Core emotions (0-7, compatible with RAVDESS)
    "neutral",      # 0
    "calm",         # 1
    "happy",        # 2
    "sad",          # 3
    "angry",        # 4
    "fearful",      # 5
    "disgust",      # 6
    "surprised",    # 7 - changed from "surprise" to match RAVDESS_EMOTIONS
    # Expresso additional styles (8-33)
    "confused",     # 8
    "enunciated",   # 9
    "whisper",      # 10
    "laughing",     # 11
    "narration",    # 12
    "projected",    # 13  (loud/projected voice)
    "singing",      # 14  (singing style)
    "default",      # 15  (neutral read)
    # Conversational styles
    "empathetic",   # 16
    "concerned",    # 17
    "apologetic",   # 18
    "encouraging",  # 19
    "excited",      # 20
    "hesitant",     # 21
    "bored",        # 22
    "tired",        # 23
    "sarcastic",    # 24
    "dismissive",   # 25
    "annoyed",      # 26
    "amused",       # 27
    "curious",      # 28
    "sympathetic",  # 29
    "defensive",    # 30
    "flirty",       # 31
    "assertive",    # 32
    "nervous",      # 33
]

# Mapping from Expresso style names to emotion IDs
EXPRESSO_STYLE_MAP = {
    # Read speech styles
    "default": 15,
    "confused": 8,
    "enunciated": 9,
    "whisper": 10,
    "laughing": 11,
    "narration": 12,
    "projected": 13,
    "singing": 14,
    # Improvised dialogue styles
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "fear": 5,  # alias
    "disgust": 6,
    "disgusted": 6,  # alias
    "surprise": 7,
    "surprised": 7,  # alias
    "empathetic": 16,
    "concerned": 17,
    "apologetic": 18,
    "encouraging": 19,
    "excited": 20,
    "hesitant": 21,
    "bored": 22,
    "tired": 23,
    "sarcastic": 24,
    "dismissive": 25,
    "annoyed": 26,
    "amused": 27,
    "curious": 28,
    "sympathetic": 29,
    "defensive": 30,
    "flirty": 31,
    "assertive": 32,
    "nervous": 33,
}

# Mapping from RAVDESS emotion IDs to extended taxonomy (identity for 0-7)
RAVDESS_TO_EXTENDED = {i: i for i in range(8)}

# Singing vs Speaking (binary)
VOCAL_MODES = ["speaking", "singing"]

# Paralinguistics V2 (50 classes: universal + international fillers + singing)
PARALINGUISTICS_CLASSES = [
    # ═══ UNIVERSAL NON-VERBAL (0-10) ═══
    "speech",         # 0 - Normal speech (default)
    "laughter",       # 1 - Laughing sounds
    "cough",          # 2 - Coughing
    "sigh",           # 3 - Sighing (emotional exhale)
    "breath",         # 4 - Breathing (in/out)
    "cry",            # 5 - Crying/sobbing
    "yawn",           # 6 - Yawning
    "throat_clear",   # 7 - Throat clearing
    "sneeze",         # 8 - Sneezing
    "gasp",           # 9 - Sharp inhale (surprise)
    "groan",          # 10 - Groaning

    # ═══ ENGLISH FILLERS (11-15) ═══
    "um_en",          # 11 - English "um"
    "uh_en",          # 12 - English "uh"
    "hmm_en",         # 13 - English "hmm"
    "er_en",          # 14 - English "er"
    "ah_en",          # 15 - English "ah"

    # ═══ MANDARIN CHINESE FILLERS (16-19) ═══
    "nage_zh",        # 16 - 那个 (nàge) - most common Chinese filler
    "zhege_zh",       # 17 - 这个 (zhège) - "this"
    "jiushi_zh",      # 18 - 就是 (jiùshì) - "that is"
    "en_zh",          # 19 - 嗯 (en) - Chinese "mmm/um"

    # ═══ JAPANESE FILLERS (20-24) ═══
    "eto_ja",         # 20 - えと (eto) - hesitation
    "ano_ja",         # 21 - あの (ano) - "um"
    "ee_ja",          # 22 - ええ (ee) - "yeah/well"
    "maa_ja",         # 23 - まあ (maa) - "well"
    "un_ja",          # 24 - うん (un) - informal "yeah"

    # ═══ KOREAN FILLERS (25-28) ═══
    "eo_ko",          # 25 - 어 (eo) - "uh"
    "eum_ko",         # 26 - 음 (eum) - "um"
    "geuge_ko",       # 27 - 그게 (geuge) - "that thing"
    "mwo_ko",         # 28 - 뭐 (mwo) - "what"

    # ═══ HINDI FILLERS (29-32) ═══
    "matlab_hi",      # 29 - मतलब (matlab) - "meaning"
    "wo_hi",          # 30 - वो (wo) - "that"
    "yeh_hi",         # 31 - ये (yeh) - "this"
    "haan_hi",        # 32 - हाँ (haan) - "yes/hmm"

    # ═══ OTHER MAJOR LANGUAGES (33-39) ═══
    "este_es",        # 33 - Spanish "this/um"
    "pues_es",        # 34 - Spanish "well"
    "euh_fr",         # 35 - French "uh"
    "ben_fr",         # 36 - French "well"
    "aeh_de",         # 37 - German "uh"
    "also_de",        # 38 - German "so/well"
    "yani_ar",        # 39 - Arabic يعني (ya'ni) - "meaning"

    # ═══ SINGING VOCALIZATIONS (40-49) ═══
    "sing_a",         # 40 - "ahhh" vowel
    "sing_e",         # 41 - "ehh" vowel
    "sing_i",         # 42 - "eee" vowel
    "sing_o",         # 43 - "ooo" vowel
    "sing_u",         # 44 - "ooo" (as in "you")
    "vibrato",        # 45 - vibrato technique
    "trill",          # 46 - trill/trillo
    "vocal_fry",      # 47 - creaky voice
    "falsetto",       # 48 - head voice
    "belt",           # 49 - chest voice projection
]

# Mapping from common labels to paralinguistics indices (V2 - 50 classes)
PARALINGUISTICS_MAP = {
    # ═══ UNIVERSAL NON-VERBAL ═══
    "speech": 0, "talking": 0, "speaking": 0, "Speech": 0, "none": 0,
    "laughter": 1, "laugh": 1, "laughing": 1, "chuckle": 1, "giggle": 1,
    "Laughter": 1, "Laughing": 1, "Laugh": 1,
    "cough": 2, "coughing": 2, "Cough": 2, "Coughing": 2,
    "sigh": 3, "sighing": 3, "Sigh": 3, "Sighing": 3,
    "breath": 4, "breathing": 4, "Breathing": 4, "Breath": 4,
    "breath_in": 4, "inhale": 4, "breath_out": 4, "exhale": 4,
    "cry": 5, "crying": 5, "sob": 5, "sobbing": 5, "Crying": 5,
    "yawn": 6, "yawning": 6, "Yawn": 6, "Yawning": 6,
    "throat_clear": 7, "throat-clear": 7, "throatclear": 7, "clear_throat": 7,
    "Throat clearing": 7, "throat clearing": 7, "Throat_clear": 7,
    "sneeze": 8, "sneezing": 8, "Sneeze": 8, "Sneezing": 8,
    "sniff": 8, "sniffing": 8, "Sniff": 8, "Sniffing": 8,  # map to sneeze category
    "gasp": 9, "Gasp": 9, "sharp_inhale": 9,
    "groan": 10, "groaning": 10, "Groan": 10, "moan": 10,

    # ═══ ENGLISH FILLERS ═══
    "um": 11, "umm": 11, "ummm": 11, "um_en": 11,
    "uh": 12, "uhh": 12, "uhhh": 12, "uh_en": 12,
    "hmm": 13, "hmmm": 13, "hmm_en": 13,
    "er": 14, "err": 14, "erm": 14, "er_en": 14,
    "ah": 15, "ahh": 15, "ahhh": 15, "ah_en": 15,

    # ═══ MANDARIN CHINESE FILLERS ═══
    "nage": 16, "nàge": 16, "那个": 16, "nage_zh": 16,
    "zhege": 17, "zhège": 17, "这个": 17, "zhege_zh": 17,
    "jiushi": 18, "jiùshì": 18, "就是": 18, "jiushi_zh": 18,
    "en_zh": 19, "嗯": 19,

    # ═══ JAPANESE FILLERS ═══
    "eto": 20, "えと": 20, "eto_ja": 20,
    "ano": 21, "あの": 21, "ano_ja": 21,
    "ee_ja": 22, "ええ": 22,
    "maa": 23, "まあ": 23, "maa_ja": 23,
    "un_ja": 24, "うん": 24,

    # ═══ KOREAN FILLERS ═══
    "eo": 25, "어": 25, "eo_ko": 25,
    "eum": 26, "음": 26, "eum_ko": 26,
    "geuge": 27, "그게": 27, "geuge_ko": 27,
    "mwo": 28, "뭐": 28, "mwo_ko": 28,

    # ═══ HINDI FILLERS ═══
    "matlab": 29, "मतलब": 29, "matlab_hi": 29,
    "wo_hi": 30, "वो": 30,
    "yeh_hi": 31, "ये": 31,
    "haan": 32, "हाँ": 32, "haan_hi": 32,

    # ═══ OTHER LANGUAGES ═══
    "este": 33, "este_es": 33,  # Spanish
    "pues": 34, "pues_es": 34,  # Spanish
    "euh": 35, "euh_fr": 35,    # French
    "ben": 36, "ben_fr": 36,    # French
    "äh": 37, "aeh": 37, "aeh_de": 37,  # German
    "also": 38, "also_de": 38,   # German
    "yani": 39, "يعني": 39, "yani_ar": 39,  # Arabic

    # ═══ SINGING VOCALIZATIONS ═══
    "sing_a": 40, "aaa": 40, "vowel_a": 40,
    "sing_e": 41, "eee": 41, "vowel_e": 41,
    "sing_i": 42, "iii": 42, "vowel_i": 42,
    "sing_o": 43, "ooo": 43, "vowel_o": 43,
    "sing_u": 44, "uuu": 44, "vowel_u": 44,
    "vibrato": 45, "Vibrato": 45,
    "trill": 46, "trillo": 46, "Trill": 46,
    "vocal_fry": 47, "creaky": 47, "creak": 47,
    "falsetto": 48, "head_voice": 48, "Falsetto": 48,
    "belt": 49, "belting": 49, "projected": 49, "Belt": 49,

    # ═══ LEGACY MAPPINGS (for backward compatibility) ═══
    "filler": 11,  # generic filler → um_en
    "hesitation": 11,
    "silence": 0,  # silence → speech (as "no paralinguistic event")
    "pause": 0,
    "other": 10,  # other → groan (catch-all)
    "other_vocalization": 10,
    "noise": 0,  # noise → speech
}

# Singing styles (10 classes) - for extended singing head
SINGING_STYLES = [
    "belt",       # 0 - Loud, projected (Broadway, anthems)
    "breathy",    # 1 - Airy, intimate
    "classical",  # 2 - Operatic, trained technique
    "folk",       # 3 - Natural, conversational
    "jazz",       # 4 - Improvisational, rhythmic
    "pop",        # 5 - Contemporary commercial
    "rock",       # 6 - Energetic, rough
    "soft",       # 7 - Gentle, lullaby
    "vibrato",    # 8 - Prominent pitch oscillation
    "neutral",    # 9 - Unclassified/default style
]

# Mapping from common style aliases to indices
SINGING_STYLE_MAP = {
    "belt": 0, "power": 0, "projected": 0, "belting": 0,
    "breathy": 1, "airy": 1, "intimate": 1, "whisper_sing": 1,
    "classical": 2, "operatic": 2, "opera": 2, "trained": 2,
    "folk": 3, "natural": 3, "conversational": 3, "acoustic": 3,
    "jazz": 4, "scat": 4, "improvisational": 4,
    "pop": 5, "contemporary": 5, "commercial": 5, "mainstream": 5,
    "rock": 6, "energetic": 6, "rough": 6, "raspy": 6,
    "soft": 7, "gentle": 7, "lullaby": 7, "quiet": 7,
    "vibrato": 8, "tremolo": 8, "oscillating": 8,
    "neutral": 9, "default": 9, "unclassified": 9, "other": 9,
}


@dataclass
class MultiHeadConfig:
    """Configuration for multi-head architecture."""

    # Encoder dimensions (from Whisper)
    d_model: int = 1280  # large-v3

    # CTC Head
    ctc_vocab_size: int = 51865  # Whisper vocabulary
    ctc_blank_id: int = 0

    # Emotion Head
    num_emotions: int = 34  # Extended taxonomy (RAVDESS + Expresso)
    emotion_hidden_dim: int = 512  # Larger for more classes

    # Emotion Head (Attention-based)
    use_attention_emotion: bool = False  # Use attention instead of mean pooling
    emotion_num_heads: int = 8  # Number of attention heads
    emotion_num_queries: int = 1  # Number of learnable query vectors

    # Singing Head (Legacy binary)
    num_vocal_modes: int = 2  # speaking/singing
    singing_hidden_dim: int = 128

    # Extended Singing Head
    use_extended_singing: bool = False  # Use extended singing head with style + intensity
    num_singing_styles: int = 17  # Number of singing style classes (VocalSet techniques)
    singing_style_hidden_dim: int = 256  # Hidden dim for style classifier

    # Pitch Head (Legacy MLP)
    pitch_hidden_dim: int = 128
    pitch_output_dim: int = 1  # F0 in Hz (or normalized)
    pitch_min_hz: float = 50.0   # Lowest pitch
    pitch_max_hz: float = 800.0  # Highest pitch (soprano range)

    # Pitch Head (CREPE-style)
    use_crepe_pitch: bool = False  # Use CREPE-style pitch head
    crepe_hidden_dim: int = 256   # Hidden dimension for CREPE conv layers
    crepe_num_bins: int = 360     # Number of pitch bins (6 octaves * 60)

    # Paralinguistics Head V2 (50 classes: universal + intl fillers + singing)
    use_paralinguistics: bool = False  # Enable paralinguistics detection
    num_paralinguistics_classes: int = 50  # V2: universal(11) + intl_fillers(29) + singing(10)
    paralinguistics_hidden_dim: int = 512  # Increased for 50 classes

    # Training
    use_layer_norm: bool = True
    dropout_rate: float = 0.1

    # Punctuation Head (emotion-aware punctuation prediction)
    use_punctuation: bool = False  # Enable punctuation prediction
    num_punctuation_classes: int = 6  # [., ,, ?, !, ..., NONE]
    punctuation_hidden_dim: int = 256  # Hidden dim for classifier
    punctuation_use_emotion: bool = True  # Use emotion features
    punctuation_use_pitch: bool = True  # Use pitch features


# Punctuation class labels
PUNCTUATION_CLASSES = [
    "PERIOD",      # 0 - .
    "COMMA",       # 1 - ,
    "QUESTION",    # 2 - ?
    "EXCLAMATION", # 3 - !
    "ELLIPSIS",    # 4 - ...
    "NONE",        # 5 - No punctuation
]


class EmotionHead(nn.Module):
    """
    Emotion classification head.

    Predicts frame-level emotions and aggregates to utterance-level.
    Uses RAVDESS 8-class emotion taxonomy.

    Architecture:
        encoder_output (batch, T, d_model)
        -> Linear(d_model, hidden_dim)
        -> ReLU
        -> LayerNorm
        -> Dropout
        -> Linear(hidden_dim, num_emotions)
        -> frame_logits (batch, T, num_emotions)
        -> mean pooling
        -> utterance_logits (batch, num_emotions)
    """

    def __init__(self, config: MultiHeadConfig):
        super().__init__()
        self.config = config

        # Projection layers
        self.ln = nn.LayerNorm(config.d_model) if config.use_layer_norm else None
        self.fc1 = nn.Linear(config.d_model, config.emotion_hidden_dim)
        self.fc2 = nn.Linear(config.emotion_hidden_dim, config.num_emotions)
        self.dropout = nn.Dropout(config.dropout_rate)

    def __call__(
        self,
        encoder_output: mx.array,
        return_frame_logits: bool = False,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            return_frame_logits: If True, return per-frame logits (for streaming)

        Returns:
            logits: (batch, num_emotions) utterance-level logits
                    or (batch, T, num_emotions) if return_frame_logits=True
        """
        x = encoder_output

        # Optional layer norm
        if self.ln is not None:
            x = self.ln(x)

        # Hidden layer
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x)

        # Output projection
        frame_logits = self.fc2(x)  # (batch, T, num_emotions)

        if return_frame_logits:
            return frame_logits

        # Mean pooling for utterance-level
        return mx.mean(frame_logits, axis=1)  # (batch, num_emotions)


    def predict(self, encoder_output: mx.array) -> tuple[int, float]:
        """
        Predict emotion label and confidence.

        Args:
            encoder_output: (1, T, d_model) single utterance

        Returns:
            Tuple of (emotion_id, confidence)
        """
        logits = self.__call__(encoder_output)
        probs = mx.softmax(logits, axis=-1)
        emotion_id = int(mx.argmax(probs, axis=-1)[0])
        confidence = float(probs[0, emotion_id])
        return emotion_id, confidence

    def predict_label(self, encoder_output: mx.array) -> tuple[str, float]:
        """
        Predict emotion label string and confidence.
        """
        emotion_id, confidence = self.predict(encoder_output)
        # Use extended taxonomy if available, fallback to RAVDESS
        if emotion_id < len(EXTENDED_EMOTIONS):
            return EXTENDED_EMOTIONS[emotion_id], confidence
        if emotion_id < len(RAVDESS_EMOTIONS):
            return RAVDESS_EMOTIONS[emotion_id], confidence
        return f"emotion_{emotion_id}", confidence


class AttentionEmotionHead(nn.Module):
    """
    Attention-based emotion classification head.

    Uses multi-head attention with learnable query vectors to aggregate
    information from encoder frames. The attention mechanism learns which
    frames are most relevant for emotion classification, rather than
    treating all frames equally (mean pooling).

    Architecture:
        encoder_output (batch, T, d_model)
        -> LayerNorm
        -> MultiHeadAttention(queries, keys=encoder, values=encoder)
        -> attended_features (batch, num_queries, d_model)
        -> Flatten or mean over queries
        -> Linear(d_model, hidden_dim)
        -> GELU
        -> Dropout
        -> Linear(hidden_dim, num_emotions)
        -> logits (batch, num_emotions)

    Key Differences from EmotionHead:
        - Uses learnable query vectors instead of mean pooling
        - Attention learns which frames are emotionally salient
        - Can use multiple queries for multi-aspect emotion understanding
        - More parameters but better modeling capacity
    """

    def __init__(self, config: MultiHeadConfig):
        super().__init__()
        self.config = config

        # Layer norm on input
        self.ln = nn.LayerNorm(config.d_model) if config.use_layer_norm else None

        # Learnable query vectors (emotion probes)
        # Shape: (num_queries, d_model)
        self.num_queries = config.emotion_num_queries
        self.queries = mx.zeros((config.emotion_num_queries, config.d_model))
        # Initialize with small random values
        import math
        scale = 1.0 / math.sqrt(config.d_model)
        self.queries = mx.random.normal((config.emotion_num_queries, config.d_model)) * scale

        # Multi-head attention
        self.num_heads = config.emotion_num_heads
        self.head_dim = config.d_model // config.emotion_num_heads
        assert config.d_model % config.emotion_num_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by num_heads ({config.emotion_num_heads})"

        # Projection layers for attention
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Classification layers
        self.fc1 = nn.Linear(config.d_model, config.emotion_hidden_dim)
        self.fc2 = nn.Linear(config.emotion_hidden_dim, config.num_emotions)
        self.dropout = nn.Dropout(config.dropout_rate)

    def _multi_head_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
    ) -> mx.array:
        """
        Multi-head attention.

        Args:
            queries: (batch, num_queries, d_model)
            keys: (batch, T, d_model)
            values: (batch, T, d_model)

        Returns:
            (batch, num_queries, d_model)
        """
        batch = keys.shape[0]
        T = keys.shape[1]
        num_queries = queries.shape[1]

        # Project queries, keys, values
        q = self.q_proj(queries)  # (batch, num_queries, d_model)
        k = self.k_proj(keys)      # (batch, T, d_model)
        v = self.v_proj(values)    # (batch, T, d_model)

        # Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, num_heads, seq, head_dim)
        q = q.reshape(batch, num_queries, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # (batch, heads, queries, T)
        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v)  # (batch, heads, queries, head_dim)

        # Reshape back to (batch, num_queries, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, num_queries, -1)

        # Output projection
        return self.out_proj(attn_output)

    def __call__(
        self,
        encoder_output: mx.array,
        return_attention_weights: bool = False,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            return_attention_weights: If True, also return attention weights

        Returns:
            logits: (batch, num_emotions) utterance-level logits
        """
        x = encoder_output
        batch = x.shape[0]

        # Optional layer norm
        if self.ln is not None:
            x = self.ln(x)

        # Expand queries for batch
        # (num_queries, d_model) -> (batch, num_queries, d_model)
        queries = mx.broadcast_to(self.queries[None, :, :], (batch, self.num_queries, self.config.d_model))

        # Multi-head cross-attention
        attended = self._multi_head_attention(queries, x, x)  # (batch, num_queries, d_model)

        # Aggregate over queries (if multiple)
        if self.num_queries > 1:
            attended = mx.mean(attended, axis=1)  # (batch, d_model)
        else:
            attended = attended[:, 0, :]  # (batch, d_model)

        # Classification MLP
        h = self.fc1(attended)
        h = nn.gelu(h)
        h = self.dropout(h)
        return self.fc2(h)  # (batch, num_emotions)


    def predict(self, encoder_output: mx.array) -> tuple[int, float]:
        """
        Predict emotion label and confidence.

        Args:
            encoder_output: (1, T, d_model) single utterance

        Returns:
            Tuple of (emotion_id, confidence)
        """
        logits = self.__call__(encoder_output)
        probs = mx.softmax(logits, axis=-1)
        emotion_id = int(mx.argmax(probs, axis=-1)[0])
        confidence = float(probs[0, emotion_id])
        return emotion_id, confidence

    def predict_label(self, encoder_output: mx.array) -> tuple[str, float]:
        """
        Predict emotion label string and confidence.
        """
        emotion_id, confidence = self.predict(encoder_output)
        # Use extended taxonomy if available, fallback to RAVDESS
        if emotion_id < len(EXTENDED_EMOTIONS):
            return EXTENDED_EMOTIONS[emotion_id], confidence
        if emotion_id < len(RAVDESS_EMOTIONS):
            return RAVDESS_EMOTIONS[emotion_id], confidence
        return f"emotion_{emotion_id}", confidence

    def get_attention_map(self, encoder_output: mx.array) -> mx.array:
        """
        Get attention weights showing which frames are attended to.

        Useful for visualization and interpretability.

        Args:
            encoder_output: (1, T, d_model) single utterance

        Returns:
            (num_heads, num_queries, T) attention weights
        """
        x = encoder_output
        batch = x.shape[0]
        T = x.shape[1]

        if self.ln is not None:
            x = self.ln(x)

        queries = mx.broadcast_to(self.queries[None, :, :], (batch, self.num_queries, self.config.d_model))

        # Get attention weights (before softmax)
        q = self.q_proj(queries).reshape(batch, self.num_queries, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = mx.softmax(mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale, axis=-1)

        # Return (num_heads, num_queries, T) for single sample
        return attn_weights[0]


# Alias for backward compatibility - EmotionHeadMLP
EmotionHeadMLP = EmotionHead


class SingingHead(nn.Module):
    """
    Binary singing vs speaking classification head.

    Detects whether the user is singing (for harmonization) or speaking (for STT).
    Frame-level predictions enable streaming detection.

    Architecture:
        encoder_output -> Linear -> ReLU -> Linear -> sigmoid
    """

    def __init__(self, config: MultiHeadConfig):
        super().__init__()
        self.config = config

        # Projection layers
        self.ln = nn.LayerNorm(config.d_model) if config.use_layer_norm else None
        self.fc1 = nn.Linear(config.d_model, config.singing_hidden_dim)
        self.fc2 = nn.Linear(config.singing_hidden_dim, 1)  # Binary

    def __call__(
        self,
        encoder_output: mx.array,
        return_frame_logits: bool = False,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            return_frame_logits: If True, return per-frame logits

        Returns:
            logits: (batch, 1) utterance-level logit
                    or (batch, T, 1) if return_frame_logits=True
        """
        x = encoder_output

        if self.ln is not None:
            x = self.ln(x)

        x = self.fc1(x)
        x = nn.relu(x)
        frame_logits = self.fc2(x)  # (batch, T, 1)

        if return_frame_logits:
            return frame_logits

        # Mean pooling
        return mx.mean(frame_logits, axis=1)  # (batch, 1)

    def predict(self, encoder_output: mx.array, threshold: float = 0.5) -> tuple[bool, float]:
        """
        Predict if singing.

        Args:
            encoder_output: (1, T, d_model) single utterance
            threshold: Classification threshold

        Returns:
            Tuple of (is_singing, confidence)
        """
        logit = self.__call__(encoder_output)
        prob = mx.sigmoid(logit)
        is_singing = float(prob[0, 0]) > threshold
        confidence = float(prob[0, 0]) if is_singing else 1.0 - float(prob[0, 0])
        return is_singing, confidence

    def predict_streaming(
        self,
        encoder_output: mx.array,
        threshold: float = 0.5,
    ) -> list[tuple[int, bool, float]]:
        """
        Streaming frame-level predictions.

        Returns list of (frame_idx, is_singing, confidence) for each frame.
        Useful for detecting singing onset in real-time.
        """
        frame_logits = self.__call__(encoder_output, return_frame_logits=True)
        frame_probs = mx.sigmoid(frame_logits)  # (1, T, 1)

        results = []
        for t in range(frame_probs.shape[1]):
            prob = float(frame_probs[0, t, 0])
            is_singing = prob > threshold
            confidence = prob if is_singing else 1.0 - prob
            results.append((t, is_singing, confidence))

        return results


class ExtendedSingingHead(nn.Module):
    """
    Extended singing head with style classification and intensity.

    Outputs three predictions:
    1. P(singing) - Binary singing vs speaking probability
    2. Style - 10-class singing style classification
    3. Intensity - Continuous singing intensity [0, 1]

    Architecture:
        encoder_output (batch, T, d_model)
        -> LayerNorm
        -> Shared projection (d_model -> hidden_dim)
        -> GELU + Dropout
        -> Three parallel heads:
            - Singing head: Linear -> sigmoid -> P(singing)
            - Style head: Linear -> 10 classes (masked when speaking)
            - Intensity head: Linear -> sigmoid -> intensity

    Key Features:
        - Style prediction is masked during speaking (not trained)
        - Intensity represents vocal effort/power
        - Compatible with VocalSet and similar singing datasets
        - Supports streaming frame-level predictions
    """

    def __init__(self, config: MultiHeadConfig):
        super().__init__()
        self.config = config
        self.num_styles = config.num_singing_styles
        hidden_dim = config.singing_style_hidden_dim

        # Input layer norm
        self.ln = nn.LayerNorm(config.d_model) if config.use_layer_norm else None

        # Shared feature extraction
        self.shared_fc = nn.Linear(config.d_model, hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Singing detection head (binary)
        self.singing_fc = nn.Linear(hidden_dim, 1)

        # Style classification head
        self.style_fc = nn.Linear(hidden_dim, self.num_styles)

        # Intensity regression head
        self.intensity_fc = nn.Linear(hidden_dim, 1)

    def __call__(
        self,
        encoder_output: mx.array,
        return_frame_logits: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            return_frame_logits: If True, return per-frame predictions

        Returns:
            singing_logit: (batch, 1) or (batch, T, 1) singing probability logit
            style_logits: (batch, num_styles) or (batch, T, num_styles) style logits
            intensity: (batch, 1) or (batch, T, 1) singing intensity [0, 1]
        """
        x = encoder_output

        # Layer norm
        if self.ln is not None:
            x = self.ln(x)

        # Shared features
        h = self.shared_fc(x)
        h = nn.gelu(h)
        h = self.dropout(h)

        # Three parallel heads
        singing_logit = self.singing_fc(h)  # (batch, T, 1)
        style_logits = self.style_fc(h)     # (batch, T, num_styles)
        intensity = mx.sigmoid(self.intensity_fc(h))  # (batch, T, 1)

        if return_frame_logits:
            return singing_logit, style_logits, intensity

        # Mean pooling for utterance-level
        singing_logit = mx.mean(singing_logit, axis=1)  # (batch, 1)
        style_logits = mx.mean(style_logits, axis=1)    # (batch, num_styles)
        intensity = mx.mean(intensity, axis=1)          # (batch, 1)

        return singing_logit, style_logits, intensity

    def predict(
        self,
        encoder_output: mx.array,
        threshold: float = 0.5,
    ) -> tuple[bool, float, str, float, float]:
        """
        Predict singing status, style, and intensity.

        Args:
            encoder_output: (1, T, d_model) single utterance
            threshold: Classification threshold for singing detection

        Returns:
            Tuple of:
                - is_singing: bool
                - singing_confidence: float [0, 1]
                - style: str (style name, only valid if singing)
                - style_confidence: float [0, 1]
                - intensity: float [0, 1]
        """
        singing_logit, style_logits, intensity = self.__call__(encoder_output)

        # Singing probability
        singing_prob = float(mx.sigmoid(singing_logit)[0, 0])
        is_singing = singing_prob > threshold
        singing_confidence = singing_prob if is_singing else 1.0 - singing_prob

        # Style prediction (only meaningful if singing)
        style_probs = mx.softmax(style_logits, axis=-1)
        style_id = int(mx.argmax(style_probs, axis=-1)[0])
        style_confidence = float(style_probs[0, style_id])
        style_name = SINGING_STYLES[style_id] if style_id < len(SINGING_STYLES) else f"style_{style_id}"

        # Intensity
        intensity_value = float(intensity[0, 0])

        return is_singing, singing_confidence, style_name, style_confidence, intensity_value

    def predict_label(
        self,
        encoder_output: mx.array,
        threshold: float = 0.5,
    ) -> dict[str, any]:
        """
        Predict with full label information.

        Returns:
            Dict with keys: is_singing, singing_confidence, style, style_confidence, intensity
        """
        is_singing, singing_conf, style, style_conf, intensity = self.predict(encoder_output, threshold)
        return {
            "is_singing": is_singing,
            "singing_confidence": singing_conf,
            "style": style if is_singing else None,
            "style_confidence": style_conf if is_singing else None,
            "intensity": intensity if is_singing else 0.0,
        }

    def predict_streaming(
        self,
        encoder_output: mx.array,
        threshold: float = 0.5,
    ) -> list[dict[str, any]]:
        """
        Streaming frame-level predictions.

        Returns list of dicts for each frame with:
            - frame_idx: int
            - is_singing: bool
            - singing_prob: float
            - style: str
            - style_prob: float
            - intensity: float
        """
        singing_logits, style_logits, intensity = self.__call__(
            encoder_output, return_frame_logits=True,
        )

        singing_probs = mx.sigmoid(singing_logits)  # (1, T, 1)
        style_probs = mx.softmax(style_logits, axis=-1)  # (1, T, num_styles)

        results = []
        for t in range(singing_probs.shape[1]):
            singing_prob = float(singing_probs[0, t, 0])
            is_singing = singing_prob > threshold

            style_id = int(mx.argmax(style_probs[0, t, :]))
            style_prob = float(style_probs[0, t, style_id])
            style_name = SINGING_STYLES[style_id] if style_id < len(SINGING_STYLES) else f"style_{style_id}"

            results.append({
                "frame_idx": t,
                "is_singing": is_singing,
                "singing_prob": singing_prob,
                "style": style_name if is_singing else None,
                "style_prob": style_prob if is_singing else None,
                "intensity": float(intensity[0, t, 0]) if is_singing else 0.0,
            })

        return results


# Alias for backwards compatibility
SingingHeadMLP = SingingHead


class ParalinguisticsHead(nn.Module):
    """
    Paralinguistics detection head for non-speech vocalizations.

    Detects frame-level paralinguistic events:
    - speech (default)
    - laughter, breath_in, breath_out, cough, sniff
    - sigh, throat_clear, filler (um/uh), silence, other

    Architecture:
        encoder_output (batch, T, d_model)
        -> LayerNorm
        -> Linear(d_model, hidden_dim)
        -> GELU
        -> Dropout
        -> Linear(hidden_dim, num_classes)
        -> frame_logits (batch, T, num_classes)

    Key Features:
        - Frame-level classification for streaming
        - Can aggregate to utterance-level or segment-level
        - Multi-class (not multi-label) - one class per frame
        - Designed for correlation with emotion (laughter → amused, sigh → tired)

    Use Cases:
        - Detect laughter in conversation
        - Identify hesitation/filler for turn-taking
        - Remove non-speech for clean transcription
        - Understand emotional context from vocalizations
    """

    def __init__(self, config: MultiHeadConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_paralinguistics_classes
        hidden_dim = config.paralinguistics_hidden_dim

        # Input processing
        self.ln = nn.LayerNorm(config.d_model) if config.use_layer_norm else None

        # Classification layers
        self.fc1 = nn.Linear(config.d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(config.dropout_rate)

    def __call__(
        self,
        encoder_output: mx.array,
        return_frame_logits: bool = True,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            return_frame_logits: If True (default), return per-frame logits.
                                 If False, return utterance-level (mean pooled).

        Returns:
            logits: (batch, T, num_classes) frame-level logits
                    or (batch, num_classes) if return_frame_logits=False
        """
        x = encoder_output

        # Layer norm
        if self.ln is not None:
            x = self.ln(x)

        # Hidden layer
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)

        # Output projection
        frame_logits = self.fc2(x)  # (batch, T, num_classes)

        if return_frame_logits:
            return frame_logits

        # Mean pooling for utterance-level
        return mx.mean(frame_logits, axis=1)  # (batch, num_classes)

    def predict(
        self,
        encoder_output: mx.array,
        threshold: float = 0.0,  # Not used, kept for API consistency
    ) -> tuple[int, float]:
        """
        Predict dominant paralinguistic class for utterance.

        Args:
            encoder_output: (1, T, d_model) single utterance

        Returns:
            Tuple of (class_id, confidence)
        """
        logits = self.__call__(encoder_output, return_frame_logits=False)
        probs = mx.softmax(logits, axis=-1)
        class_id = int(mx.argmax(probs, axis=-1)[0])
        confidence = float(probs[0, class_id])
        return class_id, confidence

    def predict_label(self, encoder_output: mx.array) -> tuple[str, float]:
        """
        Predict paralinguistic label string and confidence.

        Returns:
            Tuple of (class_name, confidence)
        """
        class_id, confidence = self.predict(encoder_output)
        if class_id < len(PARALINGUISTICS_CLASSES):
            return PARALINGUISTICS_CLASSES[class_id], confidence
        return f"class_{class_id}", confidence

    def predict_streaming(
        self,
        encoder_output: mx.array,
        min_confidence: float = 0.5,
    ) -> list[dict[str, any]]:
        """
        Frame-level paralinguistics predictions for streaming.

        Args:
            encoder_output: (1, T, d_model) single utterance
            min_confidence: Minimum confidence to report non-speech event

        Returns:
            List of dicts for each frame:
                - frame_idx: int
                - class: str
                - class_id: int
                - confidence: float
                - is_speech: bool (True if speech, False if paralinguistic event)
        """
        frame_logits = self.__call__(encoder_output, return_frame_logits=True)
        frame_probs = mx.softmax(frame_logits, axis=-1)  # (1, T, num_classes)

        results = []
        for t in range(frame_probs.shape[1]):
            probs = frame_probs[0, t, :]  # (num_classes,)
            class_id = int(mx.argmax(probs))
            confidence = float(probs[class_id])

            # Determine if this is a paralinguistic event
            is_speech = (class_id == 0)  # class 0 is "speech"

            # Only report non-speech if confidence is high enough
            if not is_speech and confidence < min_confidence:
                is_speech = True
                class_id = 0
                confidence = float(probs[0])

            class_name = PARALINGUISTICS_CLASSES[class_id] if class_id < len(PARALINGUISTICS_CLASSES) else f"class_{class_id}"

            results.append({
                "frame_idx": t,
                "class": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "is_speech": is_speech,
            })

        return results

    def get_events(
        self,
        encoder_output: mx.array,
        frame_rate: float = 50.0,
        min_duration_frames: int = 3,
        min_confidence: float = 0.6,
    ) -> list[dict[str, any]]:
        """
        Extract discrete paralinguistic events with timestamps.

        Groups consecutive frames of the same class into events.

        Args:
            encoder_output: (1, T, d_model) single utterance
            frame_rate: Frames per second (50 for Whisper)
            min_duration_frames: Minimum frames for an event to be reported
            min_confidence: Minimum average confidence for event

        Returns:
            List of event dicts:
                - class: str (event class name)
                - class_id: int
                - start_time: float (seconds)
                - end_time: float (seconds)
                - duration: float (seconds)
                - confidence: float (average confidence over event)
        """
        frame_predictions = self.predict_streaming(encoder_output, min_confidence=0.0)

        events = []
        current_event = None

        for pred in frame_predictions:
            t = pred["frame_idx"]
            class_id = pred["class_id"]
            conf = pred["confidence"]

            # Skip speech frames (class 0) for event extraction
            if class_id == 0:
                # Finalize any current event
                if current_event is not None:
                    events.append(current_event)
                    current_event = None
                continue

            # Start new event or extend current
            if current_event is None or current_event["class_id"] != class_id:
                # Finalize previous event
                if current_event is not None:
                    events.append(current_event)

                # Start new event
                current_event = {
                    "class": pred["class"],
                    "class_id": class_id,
                    "start_frame": t,
                    "end_frame": t,
                    "confidences": [conf],
                }
            else:
                # Extend current event
                current_event["end_frame"] = t
                current_event["confidences"].append(conf)

        # Finalize last event
        if current_event is not None:
            events.append(current_event)

        # Post-process events: filter and add timestamps
        filtered_events = []
        for event in events:
            duration_frames = event["end_frame"] - event["start_frame"] + 1
            avg_confidence = sum(event["confidences"]) / len(event["confidences"])

            if duration_frames >= min_duration_frames and avg_confidence >= min_confidence:
                filtered_events.append({
                    "class": event["class"],
                    "class_id": event["class_id"],
                    "start_time": event["start_frame"] / frame_rate,
                    "end_time": (event["end_frame"] + 1) / frame_rate,
                    "duration": duration_frames / frame_rate,
                    "confidence": avg_confidence,
                })

        return filtered_events


class PitchHeadMLP(nn.Module):
    """
    Simple MLP pitch head (legacy version).

    Kept for backwards compatibility with existing checkpoints.
    For new training, use CREPEPitchHead instead.
    """

    def __init__(self, config: MultiHeadConfig):
        super().__init__()
        self.config = config

        # Projection layers
        self.ln = nn.LayerNorm(config.d_model) if config.use_layer_norm else None
        self.fc1 = nn.Linear(config.d_model, config.pitch_hidden_dim)
        self.ln2 = nn.LayerNorm(config.pitch_hidden_dim)
        self.fc2 = nn.Linear(config.pitch_hidden_dim, 2)  # pitch_normalized, voicing_probability

    def __call__(self, encoder_output: mx.array) -> tuple[mx.array, mx.array]:
        x = encoder_output

        if self.ln is not None:
            x = self.ln(x)

        x = self.fc1(x)
        x = nn.relu(x)
        x = self.ln2(x)

        output = self.fc2(x)  # (batch, T, 2)

        # Split outputs
        pitch_normalized = mx.sigmoid(output[:, :, 0])  # [0, 1]
        voicing_logit = output[:, :, 1]
        voicing_prob = mx.sigmoid(voicing_logit)

        # Convert to Hz
        pitch_range = self.config.pitch_max_hz - self.config.pitch_min_hz
        pitch_hz = self.config.pitch_min_hz + pitch_normalized * pitch_range

        # Zero out unvoiced frames
        pitch_hz = pitch_hz * (voicing_prob > 0.5)

        return pitch_hz, voicing_prob

    def predict_melody(
        self,
        encoder_output: mx.array,
        frame_rate: float = 50.0,
    ) -> list[tuple[float, float, float]]:
        """
        Extract melody as list of (time, pitch_hz, confidence).

        Args:
            encoder_output: (1, T, d_model) single utterance
            frame_rate: Frames per second (50 for Whisper)

        Returns:
            List of (time_seconds, pitch_hz, voicing_confidence)
        """
        pitch_hz, voicing_prob = self.__call__(encoder_output)

        melody = []
        for t in range(pitch_hz.shape[1]):
            time_s = t / frame_rate
            hz = float(pitch_hz[0, t])
            conf = float(voicing_prob[0, t])
            melody.append((time_s, hz, conf))

        return melody

    def hz_to_midi(self, hz: float) -> int:
        """Convert Hz to MIDI note number."""
        if hz <= 0:
            return 0
        import math
        return int(round(69 + 12 * math.log2(hz / 440.0)))

    def midi_to_note_name(self, midi: int) -> str:
        """Convert MIDI note number to note name (e.g., C4, A#3)."""
        if midi <= 0:
            return ""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        note = note_names[midi % 12]
        return f"{note}{octave}"


class DilatedConv1D(nn.Module):
    """
    1D Dilated Convolution for temporal pattern extraction.

    Applies causal padding to maintain temporal alignment.
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

        # Conv1d: input (batch, seq_len, in_channels)
        # MLX Conv1d expects (N, L, C_in) and outputs (N, L_out, C_out)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,  # We'll handle padding manually for causal
        )
        self.dilation_value = dilation

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
            # Pad on time dimension (axis=1)
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])

        # Apply dilated conv
        # MLX doesn't support dilation in Conv1d directly, so we simulate it
        if self.dilation_value > 1:
            # Select every dilation-th element for each conv position
            # This is a simplified approach - expand input with zeros
            batch, seq_len, channels = x.shape
            # Reshape for dilated access
            out = self._dilated_forward(x)
        else:
            out = self.conv(x)

        return out

    def _dilated_forward(self, x: mx.array) -> mx.array:
        """
        Implement dilated convolution manually.

        For dilation d and kernel size k:
        - The effective kernel spans d*(k-1)+1 positions
        - We sample positions 0, d, 2d, ..., (k-1)*d
        """
        batch, seq_len, channels = x.shape
        k = self.kernel_size
        d = self.dilation_value

        # Effective kernel length
        effective_len = d * (k - 1) + 1

        # Output length after convolution
        out_len = seq_len - effective_len + 1

        if out_len <= 0:
            # Input too short, return zeros
            return mx.zeros((batch, max(1, seq_len), self.conv.weight.shape[0]))

        # Get conv weights: (out_channels, kernel_size, in_channels)
        w = self.conv.weight
        b = self.conv.bias if self.conv.bias is not None else mx.zeros(w.shape[0])

        # Build output frame by frame (vectorized where possible)
        outputs = []
        for t in range(out_len):
            # Gather input at dilated positions: t, t+d, t+2d, ..., t+(k-1)*d
            positions = [t + i * d for i in range(k)]
            # Extract frames: (batch, k, channels)
            frames = mx.stack([x[:, p, :] for p in positions], axis=1)
            # Compute convolution: einsum('bkc,okc->bo', frames, w)
            # w shape: (out_channels, kernel_size, in_channels)
            out_t = mx.sum(frames[:, None, :, :] * w[None, :, :, :], axis=(2, 3)) + b
            outputs.append(out_t)

        # Stack outputs: (batch, out_len, out_channels)
        return mx.stack(outputs, axis=1)


class CREPEPitchHead(nn.Module):
    """
    CREPE-style pitch head with dilated convolutions and 360-bin classification.

    Based on "CREPE: A Convolutional Representation for Pitch Estimation"
    by Jong Wook Kim et al., but adapted for Whisper encoder output.

    Architecture:
        encoder_output (batch, T, d_model)
        -> Input projection (d_model -> hidden_dim)
        -> 5 dilated conv layers (dilations: 1, 2, 4, 8, 16)
        -> Output projection (hidden_dim -> 361 bins)
        -> Softmax over bins
        -> Expected value gives pitch in Hz

    360 bins cover C1 (32.7 Hz) to C7 (2093 Hz):
        - 6 octaves, 60 bins per octave
        - Each bin = 20 cents (1/5 semitone)
        - Bin 360 = unvoiced/no pitch

    Key Differences from Original CREPE:
        - Input: Whisper encoder features (not raw audio mel)
        - Lighter architecture (5 layers, fewer filters)
        - Frame-synchronous output (no pooling)
        - Additional unvoiced bin for voicing detection
    """

    # CREPE frequency mapping constants
    CREPE_FREF = 32.70  # C1 in Hz
    CREPE_BINS = 360     # 6 octaves * 60 bins/octave
    CREPE_CENTS_PER_BIN = 20  # 20 cents per bin

    def __init__(self, config: MultiHeadConfig):
        super().__init__()
        self.config = config

        # Get hidden dim from config or use default
        hidden_dim = getattr(config, 'crepe_hidden_dim', 256)

        # Input projection
        self.ln_input = nn.LayerNorm(config.d_model) if config.use_layer_norm else None
        self.input_proj = nn.Linear(config.d_model, hidden_dim)

        # Dilated conv stack (5 layers with increasing dilation)
        # Dilations: 1, 2, 4, 8, 16 → receptive field covers ~31 frames
        self.conv_layers = []
        dilations = [1, 2, 4, 8, 16]

        for i, dilation in enumerate(dilations):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < len(dilations) - 1 else hidden_dim

            self.conv_layers.append({
                'conv': DilatedConv1D(in_ch, out_ch, kernel_size=3, dilation=dilation),
                'ln': nn.LayerNorm(out_ch),
            })

        # Store as module list for proper parameter tracking
        self._conv_modules = [layer['conv'] for layer in self.conv_layers]
        self._ln_modules = [layer['ln'] for layer in self.conv_layers]

        # Output projection: 360 pitch bins + 1 unvoiced bin
        self.output_proj = nn.Linear(hidden_dim, self.CREPE_BINS + 1)

        # Precompute bin frequencies for Hz conversion
        self._bin_frequencies = self._compute_bin_frequencies()

    def _compute_bin_frequencies(self) -> mx.array:
        """Compute center frequency for each bin."""
        import numpy as np
        bins = np.arange(self.CREPE_BINS)
        # f = 32.70 * 2^(bin / 60)
        freqs = self.CREPE_FREF * (2.0 ** (bins / 60.0))
        return mx.array(freqs, dtype=mx.float32)

    def __call__(
        self,
        encoder_output: mx.array,
        return_bins: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            return_bins: If True, also return bin logits

        Returns:
            pitch_hz: (batch, T) predicted F0 in Hz (0 for unvoiced)
            voicing_prob: (batch, T) probability frame is voiced
            (optional) bin_logits: (batch, T, 361) raw logits
        """
        x = encoder_output

        # Input layer norm
        if self.ln_input is not None:
            x = self.ln_input(x)

        # Project to hidden dim
        x = self.input_proj(x)
        x = nn.gelu(x)

        # Dilated conv stack with residual connections
        for _i, (conv, ln) in enumerate(zip(self._conv_modules, self._ln_modules, strict=False)):
            residual = x
            x = conv(x)
            x = ln(x)
            x = nn.gelu(x)

            # Residual connection (if shapes match)
            if x.shape == residual.shape:
                x = x + residual

        # Output projection to bins
        logits = self.output_proj(x)  # (batch, T, 361)

        # Split into pitch bins and unvoiced
        pitch_logits = logits[:, :, :self.CREPE_BINS]  # (batch, T, 360)
        unvoiced_logit = logits[:, :, self.CREPE_BINS]  # (batch, T)

        # Softmax over pitch bins for frequency estimation
        pitch_probs = mx.softmax(pitch_logits, axis=-1)  # (batch, T, 360)

        # Voicing probability: sigmoid of voiced vs unvoiced
        # Using difference between max pitch logit and unvoiced logit
        max_pitch_logit = mx.max(pitch_logits, axis=-1)  # (batch, T)
        voicing_prob = mx.sigmoid(max_pitch_logit - unvoiced_logit)

        # Expected frequency: weighted sum of bin frequencies
        # pitch_hz = sum(prob[b] * freq[b]) for b in bins
        pitch_hz = mx.sum(pitch_probs * self._bin_frequencies[None, None, :], axis=-1)

        # Zero out unvoiced frames
        pitch_hz = pitch_hz * (voicing_prob > 0.5)

        if return_bins:
            return pitch_hz, voicing_prob, logits

        return pitch_hz, voicing_prob

    def get_bin_probabilities(self, encoder_output: mx.array) -> mx.array:
        """
        Get full probability distribution over pitch bins.

        Useful for visualization and uncertainty estimation.

        Returns:
            (batch, T, 360) probability distribution over bins
        """
        pitch_hz, voicing_prob, logits = self.__call__(encoder_output, return_bins=True)
        return mx.softmax(logits[:, :, :self.CREPE_BINS], axis=-1)

    def cents_to_hz(self, cents: float) -> float:
        """Convert cents above C1 to Hz."""
        return self.CREPE_FREF * (2.0 ** (cents / 1200.0))

    def hz_to_cents(self, hz: float) -> float:
        """Convert Hz to cents above C1."""
        if hz <= 0:
            return 0
        import math
        return 1200.0 * math.log2(hz / self.CREPE_FREF)

    def hz_to_bin(self, hz: float) -> int:
        """Convert Hz to nearest bin index."""
        if hz <= 0:
            return self.CREPE_BINS  # Unvoiced bin
        cents = self.hz_to_cents(hz)
        bin_idx = int(round(cents / self.CREPE_CENTS_PER_BIN))
        return max(0, min(self.CREPE_BINS - 1, bin_idx))

    def bin_to_hz(self, bin_idx: int) -> float:
        """Convert bin index to Hz."""
        if bin_idx >= self.CREPE_BINS:
            return 0.0  # Unvoiced
        return float(self.CREPE_FREF * (2.0 ** (bin_idx / 60.0)))


# Alias for backwards compatibility
PitchHead = PitchHeadMLP


def create_crepe_pitch_head(config: MultiHeadConfig) -> CREPEPitchHead:
    """Factory function to create CREPE-style pitch head."""
    return CREPEPitchHead(config)


# Add predict_melody and MIDI helper methods to CREPEPitchHead
def _crepe_predict_melody(
    self,
    encoder_output: mx.array,
    frame_rate: float = 50.0,
) -> list[tuple[float, float, float]]:
    """
    Extract melody as list of (time, pitch_hz, confidence).

    Args:
        encoder_output: (1, T, d_model) single utterance
        frame_rate: Frames per second (50 for Whisper)

    Returns:
        List of (time_seconds, pitch_hz, voicing_confidence)
    """
    pitch_hz, voicing_prob = self.__call__(encoder_output)

    melody = []
    for t in range(pitch_hz.shape[1]):
        time_s = t / frame_rate
        hz = float(pitch_hz[0, t])
        conf = float(voicing_prob[0, t])
        melody.append((time_s, hz, conf))

    return melody


def _crepe_hz_to_midi(self, hz: float) -> int:
    """Convert Hz to MIDI note number."""
    if hz <= 0:
        return 0
    import math
    return int(round(69 + 12 * math.log2(hz / 440.0)))


def _crepe_midi_to_note_name(self, midi: int) -> str:
    """Convert MIDI note number to note name (e.g., C4, A#3)."""
    if midi <= 0:
        return ""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"


# Bind methods to class
CREPEPitchHead.predict_melody = _crepe_predict_melody
CREPEPitchHead.hz_to_midi = _crepe_hz_to_midi
CREPEPitchHead.midi_to_note_name = _crepe_midi_to_note_name


class WhisperMultiHead(nn.Module):
    """
    Multi-head architecture with shared Whisper encoder.

    Combines:
    - CTC Head: Text transcription
    - Emotion Head: Emotion classification
    - Singing Head: Singing detection
    - Pitch Head: F0 tracking

    The encoder is FROZEN - only heads are trained.
    """

    def __init__(
        self,
        config: MultiHeadConfig,
        ctc_head: Optional["CTCDraftHead"] = None,
    ):
        super().__init__()
        self.config = config

        # Optional CTC head (can be provided separately)
        self.ctc_head = ctc_head

        # Emotion head: attention-based or legacy mean pooling
        if config.use_attention_emotion:
            self.emotion_head = AttentionEmotionHead(config)
        else:
            self.emotion_head = EmotionHead(config)

        # Singing head: extended (style + intensity) or legacy binary
        if config.use_extended_singing:
            self.singing_head = ExtendedSingingHead(config)
        else:
            self.singing_head = SingingHead(config)

        # Pitch head: CREPE-style or legacy MLP
        if config.use_crepe_pitch:
            self.pitch_head = CREPEPitchHead(config)
        else:
            self.pitch_head = PitchHeadMLP(config)

        # Paralinguistics head (optional)
        if config.use_paralinguistics:
            self.paralinguistics_head = ParalinguisticsHead(config)
        else:
            self.paralinguistics_head = None

    def __call__(
        self,
        encoder_output: mx.array,
        compute_ctc: bool = True,
        compute_emotion: bool = True,
        compute_singing: bool = True,
        compute_pitch: bool = True,
        compute_paralinguistics: bool = True,
    ) -> dict[str, mx.array]:
        """
        Forward pass through all heads.

        Args:
            encoder_output: (batch, T, d_model) from frozen Whisper encoder
            compute_*: Flags to enable/disable specific heads

        Returns:
            Dict with outputs from each enabled head
        """
        outputs = {}

        if compute_ctc and self.ctc_head is not None:
            outputs["ctc_logits"] = self.ctc_head(encoder_output)

        if compute_emotion:
            outputs["emotion_logits"] = self.emotion_head(encoder_output)

        if compute_singing:
            if self.config.use_extended_singing:
                # ExtendedSingingHead returns (singing_logit, style_logits, intensity)
                singing_logit, style_logits, intensity = self.singing_head(encoder_output)
                outputs["singing_logit"] = singing_logit
                outputs["style_logits"] = style_logits
                outputs["intensity"] = intensity
            else:
                outputs["singing_logit"] = self.singing_head(encoder_output)

        if compute_pitch:
            pitch_hz, voicing_prob = self.pitch_head(encoder_output)
            outputs["pitch_hz"] = pitch_hz
            outputs["voicing_prob"] = voicing_prob

        if compute_paralinguistics and self.paralinguistics_head is not None:
            outputs["paralinguistics_logits"] = self.paralinguistics_head(encoder_output)

        return outputs

    def predict_all(
        self,
        encoder_output: mx.array,
        tokenizer=None,
    ) -> dict[str, any]:
        """
        Predict all outputs with human-readable labels.

        Args:
            encoder_output: (1, T, d_model) single utterance
            tokenizer: Whisper tokenizer for CTC decoding

        Returns:
            Dict with decoded predictions
        """
        outputs = self.__call__(encoder_output)

        results = {}

        # CTC text
        if "ctc_logits" in outputs and self.ctc_head is not None:
            tokens = self.ctc_head.decode_greedy(outputs["ctc_logits"])
            if tokenizer is not None:
                results["text"] = tokenizer.decode(tokens)
            else:
                results["tokens"] = tokens

        # Emotion
        if "emotion_logits" in outputs:
            probs = mx.softmax(outputs["emotion_logits"], axis=-1)
            emotion_id = int(mx.argmax(probs, axis=-1)[0])
            # Use extended taxonomy if available, fallback to RAVDESS
            if emotion_id < len(EXTENDED_EMOTIONS):
                results["emotion"] = EXTENDED_EMOTIONS[emotion_id]
            elif emotion_id < len(RAVDESS_EMOTIONS):
                results["emotion"] = RAVDESS_EMOTIONS[emotion_id]
            else:
                results["emotion"] = f"emotion_{emotion_id}"
            results["emotion_confidence"] = float(probs[0, emotion_id])

        # Singing
        if "singing_logit" in outputs:
            prob = mx.sigmoid(outputs["singing_logit"])
            results["is_singing"] = float(prob[0, 0]) > 0.5
            results["singing_confidence"] = float(prob[0, 0])

            # Extended singing head outputs (style + intensity)
            if "style_logits" in outputs:
                style_probs = mx.softmax(outputs["style_logits"], axis=-1)
                style_id = int(mx.argmax(style_probs, axis=-1)[0])
                style_name = SINGING_STYLES[style_id] if style_id < len(SINGING_STYLES) else f"style_{style_id}"
                results["style"] = style_name if results["is_singing"] else None
                results["style_confidence"] = float(style_probs[0, style_id]) if results["is_singing"] else None

            if "intensity" in outputs:
                results["intensity"] = float(outputs["intensity"][0, 0]) if results["is_singing"] else 0.0

        # Pitch
        if "pitch_hz" in outputs:
            pitch_hz = outputs["pitch_hz"][0]  # (T,)
            voiced_mask = pitch_hz > 0
            num_voiced = mx.sum(voiced_mask.astype(mx.float32))
            if float(num_voiced) > 0:
                # Compute mean of voiced frames using mask
                voiced_sum = mx.sum(pitch_hz * voiced_mask.astype(mx.float32))
                mean_pitch = float(voiced_sum / num_voiced)
                results["mean_pitch_hz"] = mean_pitch
                results["pitch_note"] = self.pitch_head.midi_to_note_name(
                    self.pitch_head.hz_to_midi(mean_pitch),
                )
            else:
                results["mean_pitch_hz"] = 0.0
                results["pitch_note"] = ""

        # Paralinguistics (non-speech vocalizations)
        if "paralinguistics_logits" in outputs and self.paralinguistics_head is not None:
            # Get events (laughter, cough, etc.)
            events = self.paralinguistics_head.get_events(encoder_output)
            results["paralinguistic_events"] = events

            # Also get dominant class at utterance level
            para_logits = outputs["paralinguistics_logits"]
            # Mean pool over time for utterance-level
            para_logits_mean = mx.mean(para_logits, axis=1)  # (batch, num_classes)
            para_probs = mx.softmax(para_logits_mean, axis=-1)
            para_class_id = int(mx.argmax(para_probs, axis=-1)[0])
            para_confidence = float(para_probs[0, para_class_id])

            # Only report if non-speech with high confidence
            if para_class_id > 0 and para_confidence > 0.5:
                para_class_name = PARALINGUISTICS_CLASSES[para_class_id] if para_class_id < len(PARALINGUISTICS_CLASSES) else f"class_{para_class_id}"
                results["paralinguistics"] = {
                    "class": para_class_name,
                    "class_id": para_class_id,
                    "confidence": para_confidence,
                }
            else:
                results["paralinguistics"] = None

        return results


def create_multi_head(
    model_size: str = "large-v3",
    ctc_head: Optional["CTCDraftHead"] = None,
    use_crepe_pitch: bool = False,
    use_attention_emotion: bool = False,
    use_extended_singing: bool = False,
    use_paralinguistics: bool = False,
) -> WhisperMultiHead:
    """
    Factory function to create multi-head architecture.

    Args:
        model_size: Whisper model size
        ctc_head: Optional pre-existing CTC head
        use_crepe_pitch: Use CREPE-style pitch head (360 bins) instead of MLP
        use_attention_emotion: Use attention-based emotion head
        use_extended_singing: Use extended singing head with style + intensity
        use_paralinguistics: Use paralinguistics head for non-speech vocalizations

    Returns:
        WhisperMultiHead configured for the model
    """
    # Model dimensions
    d_model_map = {
        "tiny": 384,
        "base": 512,
        "small": 768,
        "medium": 1024,
        "large": 1280,
        "large-v2": 1280,
        "large-v3": 1280,
        "turbo": 1280,
    }

    d_model = d_model_map.get(model_size, 1280)

    config = MultiHeadConfig(
        d_model=d_model,
        use_crepe_pitch=use_crepe_pitch,
        use_attention_emotion=use_attention_emotion,
        use_extended_singing=use_extended_singing,
        use_paralinguistics=use_paralinguistics,
    )

    return WhisperMultiHead(config, ctc_head=ctc_head)


# Emotion loss functions
def emotion_loss(
    logits: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """
    Cross-entropy loss for emotion classification.

    Args:
        logits: (batch, num_emotions) predicted logits
        targets: (batch,) target emotion indices
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Softmax cross entropy
    log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

    # Gather target log probs
    batch_size = logits.shape[0]
    target_log_probs = log_probs[mx.arange(batch_size), targets]

    loss = -target_log_probs

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def singing_loss(
    logits: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """
    Binary cross-entropy loss for singing detection.

    Args:
        logits: (batch, 1) predicted logits
        targets: (batch,) target labels (0=speaking, 1=singing)
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Binary cross entropy with logits
    targets = targets.astype(mx.float32).reshape(-1, 1)

    # Stable BCE: max(logits, 0) - logits * targets + log(1 + exp(-|logits|))
    loss = mx.maximum(logits, 0) - logits * targets + mx.log(1 + mx.exp(-mx.abs(logits)))

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def extended_singing_loss(
    singing_logits: mx.array,
    style_logits: mx.array,
    intensity_pred: mx.array,
    target_singing: mx.array,
    target_style: mx.array,
    target_intensity: mx.array,
    style_weight: float = 1.0,
    intensity_weight: float = 0.5,
    reduction: str = "mean",
) -> tuple[mx.array, dict[str, mx.array]]:
    """
    Multi-task loss for extended singing head.

    Computes:
    1. Binary cross-entropy for singing detection
    2. Cross-entropy for style classification (only on singing frames)
    3. MSE for intensity regression (only on singing frames)

    Args:
        singing_logits: (batch, 1) singing probability logits
        style_logits: (batch, num_styles) style classification logits
        intensity_pred: (batch, 1) predicted intensity [0, 1]
        target_singing: (batch,) target singing labels (0=speaking, 1=singing)
        target_style: (batch,) target style indices (only valid when singing)
        target_intensity: (batch,) target intensity values [0, 1]
        style_weight: Weight for style loss component
        intensity_weight: Weight for intensity loss component
        reduction: "mean", "sum", or "none"

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual components
    """
    batch_size = singing_logits.shape[0]

    # 1. Singing detection loss (binary cross-entropy)
    target_singing_float = target_singing.astype(mx.float32).reshape(-1, 1)
    singing_bce = (
        mx.maximum(singing_logits, 0)
        - singing_logits * target_singing_float
        + mx.log(1 + mx.exp(-mx.abs(singing_logits)))
    )

    # 2. Style classification loss (cross-entropy, masked for speaking frames)
    log_probs = mx.log(mx.softmax(style_logits, axis=-1) + 1e-10)
    target_style_int = target_style.astype(mx.int32)
    style_ce = -log_probs[mx.arange(batch_size), target_style_int]

    # Mask style loss for speaking frames (target_singing == 0)
    singing_mask = target_singing > 0.5
    style_ce_masked = mx.where(singing_mask, style_ce, mx.zeros_like(style_ce))

    # 3. Intensity regression loss (MSE, masked for speaking frames)
    target_intensity_reshaped = target_intensity.reshape(-1, 1)
    intensity_mse = (intensity_pred - target_intensity_reshaped) ** 2
    intensity_mse = intensity_mse.squeeze(-1)  # (batch,)
    intensity_mse_masked = mx.where(singing_mask, intensity_mse, mx.zeros_like(intensity_mse))

    # Combined loss with weights
    total_loss = (
        singing_bce.squeeze(-1)
        + style_weight * style_ce_masked
        + intensity_weight * intensity_mse_masked
    )

    # Create loss dict for logging
    loss_dict = {
        "singing_loss": singing_bce.squeeze(-1),
        "style_loss": style_ce_masked,
        "intensity_loss": intensity_mse_masked,
    }

    if reduction == "mean":
        total_loss = mx.mean(total_loss)
        loss_dict = {k: mx.mean(v) for k, v in loss_dict.items()}
    elif reduction == "sum":
        total_loss = mx.sum(total_loss)
        loss_dict = {k: mx.sum(v) for k, v in loss_dict.items()}

    return total_loss, loss_dict


def pitch_loss(
    predicted_hz: mx.array,
    predicted_voicing: mx.array,
    target_hz: mx.array,
    target_voiced: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """
    Combined pitch and voicing loss.

    Args:
        predicted_hz: (batch, T) predicted F0 in Hz
        predicted_voicing: (batch, T) predicted voicing probability
        target_hz: (batch, T) target F0 in Hz (0 for unvoiced)
        target_voiced: (batch, T) target voicing labels
        reduction: "mean", "sum", or "none"

    Returns:
        Combined loss value
    """
    # Voicing loss: binary cross entropy
    voicing_loss = mx.maximum(predicted_voicing, 0) - predicted_voicing * target_voiced + mx.log(1 + mx.exp(-mx.abs(predicted_voicing)))

    # Pitch loss: MSE only on voiced frames
    pitch_diff = (predicted_hz - target_hz) ** 2
    voiced_mask = target_voiced > 0.5
    pitch_loss_values = mx.where(voiced_mask, pitch_diff, mx.zeros_like(pitch_diff))

    # Normalize pitch loss by Hz range (make scale-invariant)
    pitch_loss_normalized = pitch_loss_values / (300.0 ** 2)  # ~300 Hz typical range

    # Combined loss
    total_loss = voicing_loss + pitch_loss_normalized

    if reduction == "mean":
        return mx.mean(total_loss)
    if reduction == "sum":
        return mx.sum(total_loss)
    return total_loss


def crepe_pitch_loss(
    logits: mx.array,
    target_hz: mx.array,
    target_voiced: mx.array,
    fref: float = 32.70,
    cents_per_bin: float = 20.0,
    num_bins: int = 360,
    gaussian_blur: float = 25.0,
    reduction: str = "mean",
) -> mx.array:
    """
    CREPE-style pitch loss using cross-entropy over bins with Gaussian blur.

    Instead of hard labels, creates a Gaussian distribution centered on the
    target bin. This allows the model to learn smoother pitch transitions
    and handle labeling noise.

    Args:
        logits: (batch, T, num_bins+1) raw logits from CREPEPitchHead
        target_hz: (batch, T) target F0 in Hz (0 for unvoiced)
        target_voiced: (batch, T) target voicing labels (0 or 1)
        fref: Reference frequency (C1 = 32.70 Hz)
        cents_per_bin: Cents per bin (default 20 for CREPE)
        num_bins: Number of pitch bins (default 360)
        gaussian_blur: Standard deviation in cents for label smoothing
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    import math

    batch, T, num_outputs = logits.shape

    # Split logits into pitch bins and unvoiced
    pitch_logits = logits[:, :, :num_bins]  # (batch, T, 360)
    unvoiced_logit = logits[:, :, num_bins:num_bins+1]  # (batch, T, 1)

    # Convert target Hz to bin indices
    # bin = 60 * log2(hz / fref) for pitch bins
    # Clamp to valid range
    eps = 1e-6
    target_hz_clamped = mx.maximum(target_hz, fref)  # Clamp to min frequency
    target_cents = 1200.0 * mx.log(target_hz_clamped / fref + eps) / math.log(2)
    target_bins = target_cents / cents_per_bin  # Fractional bin index

    # Create Gaussian distribution around target bin for voiced frames
    bin_indices = mx.arange(num_bins)[None, None, :]  # (1, 1, 360)
    target_bins_expanded = target_bins[:, :, None]  # (batch, T, 1)

    # Gaussian weights centered on target bin
    # sigma in bins = gaussian_blur (cents) / cents_per_bin
    sigma = gaussian_blur / cents_per_bin
    gaussian_weights = mx.exp(-0.5 * ((bin_indices - target_bins_expanded) / sigma) ** 2)
    gaussian_weights = gaussian_weights / (mx.sum(gaussian_weights, axis=-1, keepdims=True) + eps)

    # Log softmax over pitch bins
    log_probs = pitch_logits - mx.logsumexp(pitch_logits, axis=-1, keepdims=True)

    # Cross entropy with soft labels (voiced frames)
    pitch_ce = -mx.sum(gaussian_weights * log_probs, axis=-1)  # (batch, T)

    # Voicing loss: BCE between voiced prediction and target
    # Voicing is determined by max pitch logit vs unvoiced logit
    max_pitch_logit = mx.max(pitch_logits, axis=-1, keepdims=True)  # (batch, T, 1)
    voiced_logit = max_pitch_logit - unvoiced_logit  # (batch, T, 1)
    voiced_logit = voiced_logit.squeeze(-1)  # (batch, T)

    target_voiced_float = target_voiced.astype(mx.float32)
    voicing_bce = (
        mx.maximum(voiced_logit, 0)
        - voiced_logit * target_voiced_float
        + mx.log(1 + mx.exp(-mx.abs(voiced_logit)))
    )

    # Combine losses: pitch CE only for voiced, voicing BCE for all
    voiced_mask = target_voiced > 0.5
    pitch_loss_masked = mx.where(voiced_mask, pitch_ce, mx.zeros_like(pitch_ce))

    # Weight voicing loss higher for unvoiced frames (class imbalance)
    voicing_weight = mx.where(voiced_mask, mx.ones_like(voicing_bce), mx.ones_like(voicing_bce) * 2.0)
    voicing_loss_weighted = voicing_bce * voicing_weight

    total_loss = pitch_loss_masked + voicing_loss_weighted

    if reduction == "mean":
        return mx.mean(total_loss)
    if reduction == "sum":
        return mx.sum(total_loss)
    return total_loss


def paralinguistics_loss(
    logits: mx.array,
    targets: mx.array,
    class_weights: mx.array | None = None,
    reduction: str = "mean",
) -> mx.array:
    """
    Cross-entropy loss for paralinguistics classification.

    Supports optional class weighting to handle imbalanced datasets
    (speech is usually dominant, paralinguistic events are rare).

    Args:
        logits: (batch, T, num_classes) or (batch, num_classes) predicted logits
        targets: (batch, T) or (batch,) target class indices
        class_weights: Optional (num_classes,) weights per class
                       Higher weight = more importance for that class
                       Default: uniform weights
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Handle both frame-level and utterance-level inputs
    if logits.ndim == 3:
        # Frame-level: (batch, T, num_classes)
        batch, T, num_classes = logits.shape
        # Reshape for easier indexing
        logits_flat = logits.reshape(batch * T, num_classes)
        targets_flat = targets.reshape(batch * T)
    else:
        # Utterance-level: (batch, num_classes)
        batch, num_classes = logits.shape
        T = 1
        logits_flat = logits
        targets_flat = targets

    # Softmax cross entropy
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

    # Gather target log probs
    n_samples = logits_flat.shape[0]
    target_log_probs = log_probs[mx.arange(n_samples), targets_flat.astype(mx.int32)]

    loss = -target_log_probs

    # Apply class weights if provided
    if class_weights is not None:
        # Weight each sample by its target class weight
        sample_weights = class_weights[targets_flat.astype(mx.int32)]
        loss = loss * sample_weights

    # Reshape back if frame-level
    if T > 1:
        loss = loss.reshape(batch, T)

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def create_paralinguistics_class_weights(
    speech_weight: float = 0.5,
    event_weight: float = 2.0,
    num_classes: int = 18,
) -> mx.array:
    """
    Create class weights for paralinguistics loss.

    Since speech is dominant (~90%+ of frames), we down-weight it
    and up-weight rare paralinguistic events.

    Args:
        speech_weight: Weight for speech class (default 0.5, lower = less important)
        event_weight: Weight for all non-speech classes (default 2.0, higher = more important)
        num_classes: Total number of classes (default 18 with disfluencies)

    Returns:
        (num_classes,) array of class weights
    """
    weights = [speech_weight]  # class 0: speech
    weights.extend([event_weight] * (num_classes - 1))  # classes 1+: paralinguistic events + disfluencies
    return mx.array(weights, dtype=mx.float32)


class PunctuationHead(nn.Module):
    """
    Emotion-aware punctuation prediction head.

    Predicts frame-level punctuation classes using encoder output,
    optionally enhanced with emotion and pitch features.

    6 Output Classes:
        0: PERIOD (.)
        1: COMMA (,)
        2: QUESTION (?)
        3: EXCLAMATION (!)
        4: ELLIPSIS (...)
        5: NONE (no punctuation)

    Architecture:
        encoder_output (batch, T, d_model)
        + emotion_probs (batch, T, num_emotions) [optional]
        + pitch_values (batch, T, 1) [optional]
        -> Concat -> Linear(d_model + extras, hidden_dim)
        -> ReLU -> LayerNorm -> Dropout
        -> Linear(hidden_dim, num_classes)
        -> frame_logits (batch, T, num_classes)

    Key Insight:
        Questions are often distinguishable by rising intonation (pitch)
        and certain emotional states. By conditioning on emotion and pitch,
        we can improve question detection accuracy.
    """

    def __init__(self, config: MultiHeadConfig):
        super().__init__()
        self.config = config

        # Calculate input dimension
        self.input_dim = config.d_model
        if config.punctuation_use_emotion:
            self.input_dim += config.num_emotions
        if config.punctuation_use_pitch:
            self.input_dim += 1

        # Layers
        self.ln_input = nn.LayerNorm(config.d_model) if config.use_layer_norm else None
        self.fc1 = nn.Linear(self.input_dim, config.punctuation_hidden_dim)
        self.ln_hidden = nn.LayerNorm(config.punctuation_hidden_dim) if config.use_layer_norm else None
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(config.punctuation_hidden_dim, config.num_punctuation_classes)

    def __call__(
        self,
        encoder_output: mx.array,
        emotion_probs: mx.array | None = None,
        pitch_values: mx.array | None = None,
        return_frame_logits: bool = True,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            emotion_probs: (batch, T, num_emotions) frame-level emotion probabilities
                          or (batch, num_emotions) for utterance-level (will be broadcast)
            pitch_values: (batch, T, 1) frame-level pitch values
                         or (batch, 1) for utterance-level (will be broadcast)
            return_frame_logits: If True, return per-frame logits (default)
                                If False, return mean-pooled logits

        Returns:
            logits: (batch, T, num_classes) if return_frame_logits
                   or (batch, num_classes) if not
        """
        batch, T, _ = encoder_output.shape
        x = encoder_output

        # Optional input layer norm
        if self.ln_input is not None:
            x = self.ln_input(x)

        # Concatenate additional features
        features = [x]

        if self.config.punctuation_use_emotion and emotion_probs is not None:
            # Broadcast utterance-level to frame-level if needed
            if emotion_probs.ndim == 2:
                emotion_probs = mx.broadcast_to(
                    emotion_probs[:, None, :], (batch, T, emotion_probs.shape[-1]),
                )
            features.append(emotion_probs)

        if self.config.punctuation_use_pitch and pitch_values is not None:
            # Broadcast utterance-level to frame-level if needed
            if pitch_values.ndim == 2 and pitch_values.shape[1] == 1:
                pitch_values = mx.broadcast_to(pitch_values[:, None, :], (batch, T, 1))
            elif pitch_values.ndim == 1:
                pitch_values = mx.broadcast_to(pitch_values[:, None, None], (batch, T, 1))
            features.append(pitch_values)

        # Concatenate all features
        if len(features) > 1:
            x = mx.concatenate(features, axis=-1)

        # Hidden layer
        x = self.fc1(x)
        x = nn.relu(x)
        if self.ln_hidden is not None:
            x = self.ln_hidden(x)
        x = self.dropout(x)

        # Output projection
        frame_logits = self.fc2(x)  # (batch, T, num_classes)

        if return_frame_logits:
            return frame_logits

        # Mean pooling for utterance-level
        return mx.mean(frame_logits, axis=1)

    def predict(
        self,
        encoder_output: mx.array,
        emotion_probs: mx.array | None = None,
        pitch_values: mx.array | None = None,
    ) -> mx.array:
        """
        Predict punctuation labels.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            emotion_probs: Optional emotion features
            pitch_values: Optional pitch features

        Returns:
            labels: (batch, T) predicted punctuation class indices
        """
        logits = self.__call__(encoder_output, emotion_probs, pitch_values)
        return mx.argmax(logits, axis=-1)

    def predict_probs(
        self,
        encoder_output: mx.array,
        emotion_probs: mx.array | None = None,
        pitch_values: mx.array | None = None,
    ) -> mx.array:
        """
        Get punctuation probabilities.

        Returns:
            probs: (batch, T, num_classes) softmax probabilities
        """
        logits = self.__call__(encoder_output, emotion_probs, pitch_values)
        return mx.softmax(logits, axis=-1)

    def predict_labels(
        self,
        encoder_output: mx.array,
        emotion_probs: mx.array | None = None,
        pitch_values: mx.array | None = None,
    ) -> list[list[str]]:
        """
        Predict punctuation as string labels.

        Returns:
            labels: List of lists of punctuation class names
        """
        predictions = self.predict(encoder_output, emotion_probs, pitch_values)
        batch_labels = []
        for batch_idx in range(predictions.shape[0]):
            frame_labels = [
                PUNCTUATION_CLASSES[int(predictions[batch_idx, t])]
                for t in range(predictions.shape[1])
            ]
            batch_labels.append(frame_labels)
        return batch_labels


def focal_loss(
    logits: mx.array,
    targets: mx.array,
    gamma: float = 2.0,
    alpha: mx.array | None = None,
    reduction: str = "mean",
) -> mx.array:
    """
    Focal loss for imbalanced classification.

    Focuses on hard examples by down-weighting easy (well-classified) examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (batch, num_classes) or (batch, T, num_classes) predicted logits
        targets: (batch,) or (batch, T) target class indices
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to cross-entropy. Typical: 2.0
        alpha: Optional (num_classes,) class weights for class imbalance
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Handle both frame-level and utterance-level inputs
    if logits.ndim == 3:
        batch, T, num_classes = logits.shape
        logits_flat = logits.reshape(batch * T, num_classes)
        targets_flat = targets.reshape(batch * T)
    else:
        batch, num_classes = logits.shape
        T = 1
        logits_flat = logits
        targets_flat = targets

    # Compute probabilities
    probs = mx.softmax(logits_flat, axis=-1)

    # Gather target probabilities
    n_samples = logits_flat.shape[0]
    target_probs = probs[mx.arange(n_samples), targets_flat.astype(mx.int32)]

    # Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
    focal_weight = (1 - target_probs) ** gamma
    log_probs = mx.log(target_probs + 1e-10)
    loss = -focal_weight * log_probs

    # Apply class weights if provided
    if alpha is not None:
        sample_weights = alpha[targets_flat.astype(mx.int32)]
        loss = loss * sample_weights

    # Reshape back if frame-level
    if T > 1:
        loss = loss.reshape(batch, T)

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def compute_class_weights_from_counts(
    class_counts: dict[int, int],
    num_classes: int = 18,
    smoothing: float = 0.1,
) -> mx.array:
    """
    Compute inverse-frequency class weights from sample counts.

    Uses inverse square root frequency with smoothing for stability.
    weight_i = 1 / sqrt(count_i + smoothing * max_count)

    Args:
        class_counts: Dict mapping class_id -> count
        num_classes: Total number of classes
        smoothing: Smoothing factor to prevent extreme weights

    Returns:
        (num_classes,) array of class weights, normalized to mean=1
    """
    import numpy as np

    counts = np.array([class_counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    max_count = counts.max()

    # Inverse square root frequency with smoothing
    weights = 1.0 / np.sqrt(counts + smoothing * max_count)

    # Normalize to mean=1
    weights = weights / weights.mean()

    return mx.array(weights, dtype=mx.float32)
