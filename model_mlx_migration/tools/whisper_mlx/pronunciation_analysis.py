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
Pronunciation Analysis API - Raw Factual Output.

This module provides raw phoneme-level analysis comparing audio to reference text.
It outputs ONLY factual data (phonemes, timestamps, alignments, error rates).
Interpretation of what deviations "mean" is handled separately.

Architecture:
    Audio -> Whisper Encoder -> Phoneme Head -> Predicted Phonemes
                                    |
    Text -> Misaki Phonemizer ------+-> Alignment -> Raw Analysis

Example:
    from tools.whisper_mlx.pronunciation_analysis import PronunciationAnalyzer

    analyzer = PronunciationAnalyzer.from_pretrained("checkpoints/phoneme_head")
    result = analyzer.analyze(audio, "hello world")

    # Raw output - no interpretation
    print(result.overall_per)           # 0.18
    print(result.substitutions)         # [("θ", "t", 1), ("ð", "d", 1)]
    print(result.word_alignments[0])    # Detailed per-word alignment

NOTE: Phoneme head weights are currently undertrained (450 samples).
      Full training on 132K+ samples is in progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


# =============================================================================
# Raw Data Classes (No Interpretation)
# =============================================================================


@dataclass
class PhonemeFrame:
    """Single phoneme prediction from audio with timing."""
    phoneme: str              # IPA symbol, e.g., "θ"
    token_id: int             # Model token ID
    confidence: float         # 0.0-1.0 model confidence (softmax prob)
    start_ms: int             # Start timestamp in audio
    end_ms: int               # End timestamp in audio
    frame_index: int          # Frame index in encoder output

    # Raw logits for this frame (optional, for detailed analysis)
    top_k_alternatives: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class PhonemeAlignment:
    """Alignment between a predicted and expected phoneme."""
    predicted: str | None   # What was said (None if deletion)
    expected: str | None    # What should be said (None if insertion)
    match_type: str            # "match" | "substitution" | "insertion" | "deletion"
    position_in_reference: int # Position in reference sequence
    position_in_predicted: int # Position in predicted sequence

    # For substitutions: quantitative measure of acoustic difference
    # Based on phoneme feature distance (place, manner, voicing)
    phonetic_distance: float = 0.0  # 0.0 = identical, 1.0 = maximally different


@dataclass
class WordAlignment:
    """Alignment for a single word."""
    word: str                           # The word text
    start_ms: int                       # Estimated start time
    end_ms: int                         # Estimated end time
    phonemes_expected: list[str]        # ["θ", "ɪ", "ŋ", "k"]
    phonemes_predicted: list[str]       # ["t", "ɪ", "ŋ", "k"]
    alignments: list[PhonemeAlignment]  # Per-phoneme alignment

    # Aggregate metrics (factual)
    phoneme_error_rate: float           # substitutions + insertions + deletions / reference length
    match_count: int = 0
    substitution_count: int = 0
    insertion_count: int = 0
    deletion_count: int = 0


@dataclass
class PronunciationAnalysis:
    """
    Raw pronunciation analysis output. No interpretation.

    All fields are factual measurements. Interpretation of what
    patterns "mean" (e.g., accent detection, feedback generation)
    should be done by a separate module.
    """

    # Input info
    reference_text: str
    audio_duration_ms: int
    language: str

    # Raw phoneme predictions from audio
    predicted_phonemes: list[str]         # Collapsed sequence (CTC decoded)

    # Reference phonemes from text
    expected_phonemes: list[str]          # From phonemizer

    # Word-level alignment
    word_alignments: list[WordAlignment]

    # Global metrics (factual)
    overall_per: float                    # Phoneme Error Rate
    total_phonemes_expected: int
    total_phonemes_predicted: int
    total_matches: int
    total_substitutions: int
    total_insertions: int
    total_deletions: int

    # Fields with defaults must come after fields without defaults
    model_version: str = "undertrained-v0"  # NOTE: Current weights are undertrained

    # Raw phoneme predictions with frames (optional)
    predicted_frames: list[PhonemeFrame] = field(default_factory=list)

    # Raw substitution data: (expected, predicted, count)
    # e.g., [("θ", "t", 3), ("ð", "d", 2)]
    substitutions: list[tuple[str, str, int]] = field(default_factory=list)

    # Raw insertion data: (inserted_phoneme, count)
    insertions: list[tuple[str, int]] = field(default_factory=list)

    # Raw deletion data: (deleted_phoneme, count)
    deletions: list[tuple[str, int]] = field(default_factory=list)

    # Confidence metrics
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    low_confidence_frames: int = 0  # Frames with confidence < 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "reference_text": self.reference_text,
            "audio_duration_ms": self.audio_duration_ms,
            "language": self.language,
            "model_version": self.model_version,
            "predicted_phonemes": self.predicted_phonemes,
            "expected_phonemes": self.expected_phonemes,
            "overall_per": self.overall_per,
            "total_phonemes_expected": self.total_phonemes_expected,
            "total_phonemes_predicted": self.total_phonemes_predicted,
            "total_matches": self.total_matches,
            "total_substitutions": self.total_substitutions,
            "total_insertions": self.total_insertions,
            "total_deletions": self.total_deletions,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "mean_confidence": self.mean_confidence,
            "word_alignments": [
                {
                    "word": wa.word,
                    "start_ms": wa.start_ms,
                    "end_ms": wa.end_ms,
                    "phonemes_expected": wa.phonemes_expected,
                    "phonemes_predicted": wa.phonemes_predicted,
                    "phoneme_error_rate": wa.phoneme_error_rate,
                    "match_count": wa.match_count,
                    "substitution_count": wa.substitution_count,
                    "insertion_count": wa.insertion_count,
                    "deletion_count": wa.deletion_count,
                }
                for wa in self.word_alignments
            ],
        }


# =============================================================================
# Phoneme Feature Distance (for phonetic_distance calculation)
# =============================================================================

# IPA phoneme features for distance calculation
# Format: phoneme -> (place, manner, voicing, vowel_height, vowel_backness)
# Consonants use (place, manner, voicing, 0, 0)
# Vowels use (0, 0, 0, height, backness)

PHONEME_FEATURES: dict[str, tuple[int, int, int, int, int]] = {
    # Stops
    "p": (1, 1, 0, 0, 0), "b": (1, 1, 1, 0, 0),  # bilabial
    "t": (3, 1, 0, 0, 0), "d": (3, 1, 1, 0, 0),  # alveolar
    "k": (5, 1, 0, 0, 0), "g": (5, 1, 1, 0, 0),  # velar

    # Fricatives
    "f": (2, 2, 0, 0, 0), "v": (2, 2, 1, 0, 0),  # labiodental
    "θ": (3, 2, 0, 0, 0), "ð": (3, 2, 1, 0, 0),  # dental
    "s": (3, 2, 0, 0, 0), "z": (3, 2, 1, 0, 0),  # alveolar
    "ʃ": (4, 2, 0, 0, 0), "ʒ": (4, 2, 1, 0, 0),  # postalveolar
    "h": (6, 2, 0, 0, 0),                         # glottal

    # Nasals
    "m": (1, 3, 1, 0, 0),  # bilabial
    "n": (3, 3, 1, 0, 0),  # alveolar
    "ŋ": (5, 3, 1, 0, 0),  # velar

    # Approximants
    "l": (3, 4, 1, 0, 0),  # alveolar lateral
    "ɹ": (3, 4, 1, 0, 0),  # alveolar
    "r": (3, 4, 1, 0, 0),  # alveolar (alternative)
    "w": (1, 4, 1, 0, 0),  # bilabial
    "j": (4, 4, 1, 0, 0),  # palatal

    # Vowels (height: 1=high, 2=mid, 3=low; backness: 1=front, 2=central, 3=back)
    "i": (0, 0, 1, 1, 1), "ɪ": (0, 0, 1, 1, 1),
    "e": (0, 0, 1, 2, 1), "ɛ": (0, 0, 1, 2, 1),
    "æ": (0, 0, 1, 3, 1),
    "ə": (0, 0, 1, 2, 2), "ʌ": (0, 0, 1, 2, 2),
    "ɑ": (0, 0, 1, 3, 3), "a": (0, 0, 1, 3, 2),
    "ɔ": (0, 0, 1, 2, 3), "o": (0, 0, 1, 2, 3),
    "u": (0, 0, 1, 1, 3), "ʊ": (0, 0, 1, 1, 3),

    # Diphthongs (approximate as first element)
    "aɪ": (0, 0, 1, 3, 2),
    "aʊ": (0, 0, 1, 3, 2),
    "eɪ": (0, 0, 1, 2, 1),
    "oʊ": (0, 0, 1, 2, 3),
    "ɔɪ": (0, 0, 1, 2, 3),
}


def phonetic_distance(p1: str, p2: str) -> float:
    """
    Compute phonetic distance between two phonemes.

    Returns 0.0 if identical, up to 1.0 for maximally different.
    Based on articulatory features.
    """
    if p1 == p2:
        return 0.0

    f1 = PHONEME_FEATURES.get(p1, (0, 0, 0, 0, 0))
    f2 = PHONEME_FEATURES.get(p2, (0, 0, 0, 0, 0))

    # Euclidean distance normalized to [0, 1]
    diff = sum((a - b) ** 2 for a, b in zip(f1, f2, strict=False))
    max_diff = 5 * 5  # Maximum possible difference per dimension
    return min(1.0, (diff / max_diff) ** 0.5)


# =============================================================================
# Alignment Algorithm
# =============================================================================


def align_sequences(
    predicted: list[str],
    expected: list[str],
) -> list[PhonemeAlignment]:
    """
    Align predicted and expected phoneme sequences using dynamic programming.

    Returns list of alignments with match_type for each position.
    """
    m, n = len(predicted), len(expected)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i-1] == expected[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion (predicted has extra)
                    dp[i][j-1],    # insertion (expected has extra)
                    dp[i-1][j-1],  # substitution
                )

    # Backtrack to get alignment
    alignments = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and predicted[i-1] == expected[j-1]:
            alignments.append(PhonemeAlignment(
                predicted=predicted[i-1],
                expected=expected[j-1],
                match_type="match",
                position_in_reference=j-1,
                position_in_predicted=i-1,
                phonetic_distance=0.0,
            ))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignments.append(PhonemeAlignment(
                predicted=predicted[i-1],
                expected=expected[j-1],
                match_type="substitution",
                position_in_reference=j-1,
                position_in_predicted=i-1,
                phonetic_distance=phonetic_distance(predicted[i-1], expected[j-1]),
            ))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignments.append(PhonemeAlignment(
                predicted=predicted[i-1],
                expected=None,
                match_type="insertion",
                position_in_reference=-1,
                position_in_predicted=i-1,
                phonetic_distance=1.0,
            ))
            i -= 1
        else:
            alignments.append(PhonemeAlignment(
                predicted=None,
                expected=expected[j-1],
                match_type="deletion",
                position_in_reference=j-1,
                position_in_predicted=-1,
                phonetic_distance=1.0,
            ))
            j -= 1

    alignments.reverse()
    return alignments


# =============================================================================
# Pronunciation Analyzer
# =============================================================================


class PronunciationAnalyzer:
    """
    Analyze pronunciation by comparing audio to reference text.

    Outputs raw factual data only - no interpretation.

    NOTE: Current phoneme head weights are undertrained (450 samples).
          Results will improve after full training on 132K+ samples.
    """

    def __init__(
        self,
        phoneme_head,
        phoneme_vocab: dict[int, str],
        frame_duration_ms: float = 20.0,  # Whisper uses 20ms frames
    ):
        """
        Initialize analyzer.

        Args:
            phoneme_head: Trained KokoroPhonemeHead
            phoneme_vocab: Token ID -> IPA string mapping
            frame_duration_ms: Duration of each encoder frame
        """
        self.head = phoneme_head
        self.phoneme_vocab = phoneme_vocab
        self.frame_duration_ms = frame_duration_ms

        # Reverse vocab for lookup
        self.id_to_phoneme = phoneme_vocab
        self.phoneme_to_id = {v: k for k, v in phoneme_vocab.items()}

    @classmethod
    def from_pretrained(
        cls,
        head_path: str,
        vocab_path: str | None = None,
    ) -> PronunciationAnalyzer:
        """
        Load analyzer from pretrained phoneme head.

        Args:
            head_path: Path to phoneme head weights
            vocab_path: Path to phoneme vocabulary (optional)
        """
        from .kokoro_phoneme_head import KokoroPhonemeHead

        head = KokoroPhonemeHead.from_pretrained(head_path)

        # Load or create default vocab
        if vocab_path:
            import json
            with open(vocab_path) as f:
                phoneme_vocab = json.load(f)
        else:
            # Use default Kokoro vocab
            phoneme_vocab = cls._get_default_vocab()

        return cls(head, phoneme_vocab)

    @staticmethod
    def _get_default_vocab() -> dict[int, str]:
        """Get default Kokoro phoneme vocabulary."""
        try:
            from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                load_vocab,
            )
            vocab = load_vocab()
            # Invert: char -> id becomes id -> char
            return {v: k for k, v in vocab.items()}
        except Exception:
            # Fallback: create minimal vocab
            return {i: f"<{i}>" for i in range(200)}

    def analyze(
        self,
        encoder_output: mx.array,
        reference_text: str,
        language: str = "en",
        include_frames: bool = False,
    ) -> PronunciationAnalysis:
        """
        Analyze pronunciation of audio against reference text.

        Args:
            encoder_output: Whisper encoder output [batch, T, d_model] or [T, d_model]
            reference_text: What should be said
            language: Language for phonemization
            include_frames: Include detailed per-frame data

        Returns:
            PronunciationAnalysis with raw factual data
        """
        # Ensure batch dimension
        if encoder_output.ndim == 2:
            encoder_output = encoder_output[None, :]

        # Get logits and predictions
        logits = self.head(encoder_output)
        mx.eval(logits)

        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)

        # Get predictions and confidence
        pred_ids = mx.argmax(logits, axis=-1)
        pred_confidence = mx.max(probs, axis=-1)
        mx.eval(pred_ids, pred_confidence)

        pred_ids_np = np.array(pred_ids).squeeze()
        confidence_np = np.array(pred_confidence).squeeze()

        # Build frame data
        frames = []
        for i, (token_id, conf) in enumerate(zip(pred_ids_np, confidence_np, strict=False)):
            token_id = int(token_id)
            phoneme = self.id_to_phoneme.get(token_id, f"<{token_id}>")

            frames.append(PhonemeFrame(
                phoneme=phoneme,
                token_id=token_id,
                confidence=float(conf),
                start_ms=int(i * self.frame_duration_ms),
                end_ms=int((i + 1) * self.frame_duration_ms),
                frame_index=i,
            ))

        # CTC collapse: remove blanks and repeats
        predicted_phonemes = []
        blank_id = self.head.blank_id
        prev = -1
        for frame in frames:
            if frame.token_id != blank_id and frame.token_id != prev:
                predicted_phonemes.append(frame.phoneme)
            prev = frame.token_id

        # Get expected phonemes from text
        try:
            from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                phonemize_text,
            )
            phoneme_str, token_ids = phonemize_text(reference_text, language=language)
            # Convert token IDs to phoneme strings
            expected_phonemes = [self.id_to_phoneme.get(tid, f"<{tid}>") for tid in token_ids if tid != 0]
        except Exception:
            expected_phonemes = list(reference_text)  # Fallback to characters

        # Align sequences
        alignments = align_sequences(predicted_phonemes, expected_phonemes)

        # Count operations
        matches = sum(1 for a in alignments if a.match_type == "match")
        substitutions = sum(1 for a in alignments if a.match_type == "substitution")
        insertions = sum(1 for a in alignments if a.match_type == "insertion")
        deletions = sum(1 for a in alignments if a.match_type == "deletion")

        # Calculate PER
        total_expected = len(expected_phonemes)
        per = (substitutions + insertions + deletions) / max(total_expected, 1)

        # Collect substitution/insertion/deletion patterns
        sub_counts: dict[tuple[str, str], int] = {}
        ins_counts: dict[str, int] = {}
        del_counts: dict[str, int] = {}

        for a in alignments:
            if a.match_type == "substitution":
                key = (a.expected, a.predicted)
                sub_counts[key] = sub_counts.get(key, 0) + 1
            elif a.match_type == "insertion":
                ins_counts[a.predicted] = ins_counts.get(a.predicted, 0) + 1
            elif a.match_type == "deletion":
                del_counts[a.expected] = del_counts.get(a.expected, 0) + 1

        # Build word alignments (simplified - split by space)
        words = reference_text.split()
        word_alignments = self._build_word_alignments(
            words, predicted_phonemes, expected_phonemes, alignments, language,
        )

        # Confidence stats
        confidences = [f.confidence for f in frames]
        mean_conf = np.mean(confidences) if confidences else 0.0
        min_conf = np.min(confidences) if confidences else 0.0
        low_conf_count = sum(1 for c in confidences if c < 0.5)

        return PronunciationAnalysis(
            reference_text=reference_text,
            audio_duration_ms=len(frames) * int(self.frame_duration_ms),
            language=language,
            model_version="undertrained-v0",
            predicted_frames=frames if include_frames else [],
            predicted_phonemes=predicted_phonemes,
            expected_phonemes=expected_phonemes,
            word_alignments=word_alignments,
            overall_per=per,
            total_phonemes_expected=total_expected,
            total_phonemes_predicted=len(predicted_phonemes),
            total_matches=matches,
            total_substitutions=substitutions,
            total_insertions=insertions,
            total_deletions=deletions,
            substitutions=[(e, p, c) for (e, p), c in sorted(sub_counts.items(), key=lambda x: -x[1])],
            insertions=[(p, c) for p, c in sorted(ins_counts.items(), key=lambda x: -x[1])],
            deletions=[(p, c) for p, c in sorted(del_counts.items(), key=lambda x: -x[1])],
            mean_confidence=float(mean_conf),
            min_confidence=float(min_conf),
            low_confidence_frames=low_conf_count,
        )

    def _build_word_alignments(
        self,
        words: list[str],
        predicted_phonemes: list[str],
        expected_phonemes: list[str],
        alignments: list[PhonemeAlignment],
        language: str,
    ) -> list[WordAlignment]:
        """Build per-word alignments (simplified implementation)."""
        word_alignments = []

        # Get phonemes per word
        try:
            from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                phonemize_text,
            )
        except ImportError:
            return []

        expected_idx = 0
        alignment_idx = 0

        for word in words:
            try:
                _, word_tokens = phonemize_text(word, language=language)
                word_phonemes = [self.id_to_phoneme.get(tid, f"<{tid}>") for tid in word_tokens if tid != 0]
            except Exception:
                word_phonemes = []

            word_len = len(word_phonemes)

            # Find alignments for this word
            word_aligns = []
            word_predicted = []

            for a in alignments[alignment_idx:]:
                if a.position_in_reference >= expected_idx + word_len:
                    break
                if a.position_in_reference >= expected_idx or a.match_type == "insertion":
                    word_aligns.append(a)
                    if a.predicted:
                        word_predicted.append(a.predicted)
                    alignment_idx += 1

            # Count operations for this word
            w_matches = sum(1 for a in word_aligns if a.match_type == "match")
            w_subs = sum(1 for a in word_aligns if a.match_type == "substitution")
            w_ins = sum(1 for a in word_aligns if a.match_type == "insertion")
            w_dels = sum(1 for a in word_aligns if a.match_type == "deletion")

            word_per = (w_subs + w_ins + w_dels) / max(word_len, 1)

            word_alignments.append(WordAlignment(
                word=word,
                start_ms=0,  # Would need frame tracking for accurate timing
                end_ms=0,
                phonemes_expected=word_phonemes,
                phonemes_predicted=word_predicted,
                alignments=word_aligns,
                phoneme_error_rate=word_per,
                match_count=w_matches,
                substitution_count=w_subs,
                insertion_count=w_ins,
                deletion_count=w_dels,
            ))

            expected_idx += word_len

        return word_alignments


# =============================================================================
# Test
# =============================================================================


def test_pronunciation_analysis():
    """Test the pronunciation analysis module."""
    print("Testing PronunciationAnalysis...")

    # Test alignment
    pred = ["t", "ɪ", "ŋ", "k"]
    exp = ["θ", "ɪ", "ŋ", "k"]
    aligns = align_sequences(pred, exp)

    print(f"  Alignment test: {len(aligns)} alignments")
    for a in aligns:
        print(f"    {a.expected} -> {a.predicted}: {a.match_type}")

    assert len(aligns) == 4
    assert aligns[0].match_type == "substitution"
    assert aligns[1].match_type == "match"

    # Test phonetic distance
    d1 = phonetic_distance("θ", "t")
    d2 = phonetic_distance("θ", "θ")
    d3 = phonetic_distance("p", "k")

    print(f"  Distance θ->t: {d1:.3f}")
    print(f"  Distance θ->θ: {d2:.3f}")
    print(f"  Distance p->k: {d3:.3f}")

    assert d2 == 0.0
    assert d1 > 0.0
    assert d3 > d1  # p and k are more different than θ and t

    print("PronunciationAnalysis tests PASSED")


if __name__ == "__main__":
    test_pronunciation_analysis()
