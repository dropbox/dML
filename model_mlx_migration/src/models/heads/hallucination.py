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
Hallucination detection head for Zipformer encoder.

Detects ASR hallucinations by comparing:
1. Phonemes predicted from audio (via phoneme head)
2. Phonemes expected from ASR text output (via G2P)

Key insight: When ASR hallucinates, the acoustic phoneme predictions
will not match the phonemes derived from the hallucinated text.

Zero trainable parameters - uses phoneme mismatch metrics directly.
Target: >90% hallucination detection rate

Hallucination Types:
- Insertion: Model generates words not present in audio
- Repetition: Model repeats phrases excessively
- Fabrication: Model generates plausible but incorrect content
- Silence hallucination: Model generates text during silence

Detection Signals:
1. Phoneme mismatch rate (primary signal)
2. Confidence entropy (uncertain predictions)
3. Audio energy correlation (text during silence)
4. Repetition patterns (n-gram analysis)

References:
- "Hallucinations in Neural Machine Translation" (Lee et al., 2018)
- "Detecting Hallucinated Content in ASR" (Huang et al., 2023)
- "WhisperX: Time-Accurate Speech Transcription" (Bain et al., 2023)
"""

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .phoneme import IPA_PHONEMES

# Common grapheme-to-phoneme mappings (English-centric for MVP)
# Full G2P would use a trained model like Phonemizer
BASIC_G2P: dict[str, list[str]] = {
    # Vowels
    "a": ["æ", "eɪ", "ɑ"],
    "e": ["ɛ", "i", "ə"],
    "i": ["ɪ", "aɪ", "i"],
    "o": ["ɑ", "oʊ", "ɔ"],
    "u": ["ʌ", "u", "ʊ"],
    # Consonants
    "b": ["b"],
    "c": ["k", "s"],
    "d": ["d"],
    "f": ["f"],
    "g": ["ɡ", "dʒ"],
    "h": ["h"],
    "j": ["dʒ"],
    "k": ["k"],
    "l": ["l"],
    "m": ["m"],
    "n": ["n"],
    "p": ["p"],
    "q": ["k"],
    "r": ["ɹ"],
    "s": ["s", "z"],
    "t": ["t"],
    "v": ["v"],
    "w": ["w"],
    "x": ["ks"],
    "y": ["j", "i"],
    "z": ["z"],
    # Digraphs
    "th": ["θ", "ð"],
    "sh": ["ʃ"],
    "ch": ["tʃ"],
    "ph": ["f"],
    "wh": ["w", "hw"],
    "ng": ["ŋ"],
    "ck": ["k"],
}


@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection head."""

    # Phoneme mismatch threshold (0-1)
    # Higher = more lenient (fewer hallucination detections)
    mismatch_threshold: float = 0.3

    # Minimum confidence for phoneme predictions
    min_phoneme_confidence: float = 0.5

    # Entropy threshold for uncertain predictions
    entropy_threshold: float = 2.0

    # Audio energy threshold for silence detection (dB)
    silence_threshold_db: float = -40.0

    # Repetition detection parameters
    max_ngram_repeat: int = 3
    ngram_sizes: tuple[int, ...] = (2, 3, 4)

    # Weights for combining signals
    weight_mismatch: float = 0.5
    weight_confidence: float = 0.2
    weight_energy: float = 0.2
    weight_repetition: float = 0.1

    # Detection threshold (0-1)
    # Score above this = hallucination detected
    detection_threshold: float = 0.5

    # Number of phoneme classes (from phoneme head)
    num_phonemes: int = len(IPA_PHONEMES)

    # Blank ID for phoneme decoding
    blank_id: int = 0


@dataclass
class HallucinationResult:
    """Result from hallucination detection."""

    # Overall hallucination score (0-1, higher = more likely hallucination)
    score: float

    # Whether hallucination is detected
    is_hallucination: bool

    # Individual signal scores
    mismatch_score: float
    confidence_score: float
    energy_score: float
    repetition_score: float

    # Additional info
    detected_phonemes: list[str] = field(default_factory=list)
    expected_phonemes: list[str] = field(default_factory=list)
    num_mismatches: int = 0
    repeated_ngrams: list[str] = field(default_factory=list)


class HallucinationHead(nn.Module):
    """
    Hallucination detection head for Zipformer encoder.

    This head does NOT have trainable parameters. It computes
    hallucination scores based on:
    1. Phoneme mismatch between acoustic predictions and text-derived phonemes
    2. Phoneme prediction confidence/entropy
    3. Audio energy during transcribed segments
    4. Text repetition patterns

    Args:
        config: HallucinationConfig instance with detection parameters.
    """

    def __init__(self, config: HallucinationConfig | None = None):
        super().__init__()
        if config is None:
            config = HallucinationConfig()

        self.config = config

        # Build phoneme to index mapping
        self._phoneme_to_idx = {p: i for i, p in enumerate(IPA_PHONEMES)}

        # No trainable parameters - this is a detection head only

    def __call__(
        self,
        phoneme_logits: mx.array,
        asr_text: list[str] | None = None,
        audio_energy: mx.array | None = None,
    ) -> mx.array:
        """
        Compute hallucination scores from phoneme logits.

        Args:
            phoneme_logits: Phoneme logits from phoneme head
                           Shape: (batch_size, seq_len, num_phonemes)
            asr_text: Optional ASR text output for each batch item
            audio_energy: Optional frame-level audio energy (dB)
                         Shape: (batch_size, seq_len)

        Returns:
            Hallucination scores of shape (batch_size,)
            Values 0-1, higher = more likely hallucination
        """
        phoneme_logits.shape[0]

        # Compute confidence-based score (entropy of predictions)
        confidence_scores = self._compute_confidence_score(phoneme_logits)

        # If no text provided, return confidence-based score only
        if asr_text is None:
            return confidence_scores

        # Compute all signals
        mismatch_scores = self._compute_mismatch_scores(phoneme_logits, asr_text)
        energy_scores = self._compute_energy_scores(phoneme_logits, audio_energy)
        repetition_scores = self._compute_repetition_scores(asr_text)

        # Combine signals
        scores = (
            self.config.weight_mismatch * mismatch_scores +
            self.config.weight_confidence * confidence_scores +
            self.config.weight_energy * energy_scores +
            self.config.weight_repetition * repetition_scores
        )

        return scores

    def _compute_confidence_score(self, phoneme_logits: mx.array) -> mx.array:
        """
        Compute confidence score based on prediction entropy.

        High entropy = uncertain predictions = potential hallucination
        """
        # Softmax to get probabilities
        probs = mx.softmax(phoneme_logits, axis=-1)  # (batch, seq, num_phonemes)

        # Compute entropy per frame
        # H = -sum(p * log(p))
        log_probs = mx.log(probs + 1e-8)
        entropy = -mx.sum(probs * log_probs, axis=-1)  # (batch, seq)

        # Average entropy across sequence
        mean_entropy = mx.mean(entropy, axis=-1)  # (batch,)

        # Normalize by max entropy (log(num_classes))
        max_entropy = mx.log(mx.array(self.config.num_phonemes, dtype=mx.float32))
        normalized_entropy = mean_entropy / max_entropy

        # High entropy = high score
        return mx.clip(normalized_entropy, 0.0, 1.0)

    def _compute_mismatch_scores(
        self,
        phoneme_logits: mx.array,
        asr_text: list[str],
    ) -> mx.array:
        """
        Compute phoneme mismatch scores.

        Compares predicted phonemes with expected phonemes from text.
        """
        batch_size = phoneme_logits.shape[0]
        scores = []

        # Get predicted phoneme sequences
        predictions = mx.argmax(phoneme_logits, axis=-1)  # (batch, seq)

        for b in range(batch_size):
            pred_indices = predictions[b].tolist()

            # Get expected phonemes from text
            if b < len(asr_text) and asr_text[b]:
                expected = self._text_to_phonemes(asr_text[b])
            else:
                expected = []

            # Compute mismatch rate
            if len(expected) == 0:
                # No text = potential hallucination if we have predictions
                non_blank = [p for p in pred_indices if p != self.config.blank_id]
                score = min(len(non_blank) / max(len(pred_indices), 1), 1.0)
            else:
                # Decode predictions (collapse repeated, remove blank)
                decoded = self._decode_phoneme_indices(pred_indices)

                # Compare sequences
                score = self._compute_sequence_mismatch(decoded, expected)

            scores.append(score)

        return mx.array(scores, dtype=mx.float32)

    def _compute_energy_scores(
        self,
        phoneme_logits: mx.array,
        audio_energy: mx.array | None,
    ) -> mx.array:
        """
        Compute energy-based hallucination score.

        Detects text generated during silence.
        """
        batch_size = phoneme_logits.shape[0]

        if audio_energy is None:
            # No energy info - return neutral score
            return mx.zeros((batch_size,), dtype=mx.float32)

        # Check if audio has low energy (silence) where predictions exist
        predictions = mx.argmax(phoneme_logits, axis=-1)  # (batch, seq)
        probs = mx.softmax(phoneme_logits, axis=-1)
        max_probs = mx.max(probs, axis=-1)  # (batch, seq)

        # Find frames with confident predictions
        confident_mask = max_probs > self.config.min_phoneme_confidence
        non_blank_mask = predictions != self.config.blank_id
        has_prediction = confident_mask & non_blank_mask

        # Find frames with low energy (silence)
        is_silence = audio_energy < self.config.silence_threshold_db

        # Hallucination signal: predictions during silence
        hallucination_frames = has_prediction & is_silence
        score = mx.sum(hallucination_frames.astype(mx.float32), axis=-1)
        score = score / (mx.sum(has_prediction.astype(mx.float32), axis=-1) + 1e-8)

        return mx.clip(score, 0.0, 1.0)

    def _compute_repetition_scores(self, asr_text: list[str]) -> mx.array:
        """
        Compute repetition-based hallucination score.

        Detects excessive n-gram repetitions in ASR output.
        """
        scores = []

        for text in asr_text:
            if not text:
                scores.append(0.0)
                continue

            words = text.lower().split()
            if len(words) < 2:
                scores.append(0.0)
                continue

            max_repeat_ratio = 0.0

            for n in self.config.ngram_sizes:
                if len(words) < n:
                    continue

                # Extract n-grams
                ngrams = []
                for i in range(len(words) - n + 1):
                    ngrams.append(tuple(words[i:i + n]))

                # Count repetitions
                from collections import Counter
                counts = Counter(ngrams)

                if counts:
                    max_count = max(counts.values())
                    # Ratio of most repeated n-gram to total
                    ratio = max_count / len(ngrams)

                    # Flag if repeated more than threshold
                    if max_count > self.config.max_ngram_repeat:
                        max_repeat_ratio = max(max_repeat_ratio, ratio)

            scores.append(max_repeat_ratio)

        return mx.array(scores, dtype=mx.float32)

    def _text_to_phonemes(self, text: str) -> list[str]:
        """
        Convert text to approximate phoneme sequence using basic G2P.

        This is a simplified G2P for demonstration.
        Production would use Phonemizer or similar.
        """
        text = text.lower().strip()
        phonemes = []

        i = 0
        while i < len(text):
            # Try digraphs first
            found = False
            for digraph in ["th", "sh", "ch", "ph", "wh", "ng", "ck"]:
                if text[i:i + 2] == digraph:
                    if digraph in BASIC_G2P:
                        phonemes.append(BASIC_G2P[digraph][0])
                    i += 2
                    found = True
                    break

            if not found:
                char = text[i]
                if char in BASIC_G2P:
                    phonemes.append(BASIC_G2P[char][0])
                elif char == " ":
                    phonemes.append("<wb>")
                # Skip other characters (punctuation, etc.)
                i += 1

        return phonemes

    def _decode_phoneme_indices(self, indices: list[int]) -> list[str]:
        """Decode phoneme indices to symbols, collapsing repeated."""
        decoded = []
        prev_idx = -1

        for idx in indices:
            # Skip blank
            if idx == self.config.blank_id:
                prev_idx = idx
                continue

            # Collapse repeated
            if idx == prev_idx:
                continue

            if 0 <= idx < len(IPA_PHONEMES):
                decoded.append(IPA_PHONEMES[idx])
            prev_idx = idx

        return decoded

    def _compute_sequence_mismatch(
        self,
        predicted: list[str],
        expected: list[str],
    ) -> float:
        """
        Compute mismatch between predicted and expected phoneme sequences.

        Uses simplified comparison (not full edit distance for efficiency).
        """
        if not predicted and not expected:
            return 0.0
        if not predicted or not expected:
            return 1.0

        # Count overlapping phonemes (order-independent for simplicity)
        pred_set = set(predicted)
        exp_set = set(expected)

        intersection = len(pred_set & exp_set)
        union = len(pred_set | exp_set)

        if union == 0:
            return 0.0

        # Jaccard distance
        jaccard_sim = intersection / union
        return 1.0 - jaccard_sim

    def detect(
        self,
        phoneme_logits: mx.array,
        asr_text: list[str] | None = None,
        audio_energy: mx.array | None = None,
    ) -> list[HallucinationResult]:
        """
        Detect hallucinations and return detailed results.

        Args:
            phoneme_logits: Phoneme logits from phoneme head
            asr_text: Optional ASR text output
            audio_energy: Optional frame-level audio energy

        Returns:
            List of HallucinationResult for each batch item
        """
        batch_size = phoneme_logits.shape[0]

        # Compute all scores
        confidence_scores = self._compute_confidence_score(phoneme_logits)

        if asr_text is None:
            asr_text = [""] * batch_size

        mismatch_scores = self._compute_mismatch_scores(phoneme_logits, asr_text)
        energy_scores = self._compute_energy_scores(phoneme_logits, audio_energy)
        repetition_scores = self._compute_repetition_scores(asr_text)

        # Combine scores
        overall_scores = (
            self.config.weight_mismatch * mismatch_scores +
            self.config.weight_confidence * confidence_scores +
            self.config.weight_energy * energy_scores +
            self.config.weight_repetition * repetition_scores
        )

        # Build results
        results = []
        predictions = mx.argmax(phoneme_logits, axis=-1)

        for b in range(batch_size):
            pred_indices = predictions[b].tolist()
            detected = self._decode_phoneme_indices(pred_indices)
            expected = self._text_to_phonemes(asr_text[b]) if asr_text[b] else []

            # Find repeated n-grams
            repeated = []
            if asr_text[b]:
                words = asr_text[b].lower().split()
                for n in self.config.ngram_sizes:
                    if len(words) >= n:
                        from collections import Counter
                        ngrams = [
                            " ".join(words[i:i + n])
                            for i in range(len(words) - n + 1)
                        ]
                        counts = Counter(ngrams)
                        for ngram, count in counts.items():
                            if count > self.config.max_ngram_repeat:
                                repeated.append(f"{ngram} (x{count})")

            score = float(overall_scores[b].item())
            result = HallucinationResult(
                score=score,
                is_hallucination=score > self.config.detection_threshold,
                mismatch_score=float(mismatch_scores[b].item()),
                confidence_score=float(confidence_scores[b].item()),
                energy_score=float(energy_scores[b].item()),
                repetition_score=float(repetition_scores[b].item()),
                detected_phonemes=detected,
                expected_phonemes=expected,
                num_mismatches=len(set(detected) - set(expected)),
                repeated_ngrams=repeated,
            )
            results.append(result)

        return results

    def predict(
        self,
        phoneme_logits: mx.array,
        asr_text: list[str] | None = None,
        audio_energy: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Get hallucination predictions and scores.

        Args:
            phoneme_logits: Phoneme logits from phoneme head
            asr_text: Optional ASR text output
            audio_energy: Optional frame-level audio energy

        Returns:
            Tuple of:
            - Hallucination predictions (batch_size,) - 1 if hallucination
            - Hallucination scores (batch_size,) - 0-1
        """
        scores = self(phoneme_logits, asr_text, audio_energy)
        predictions = (scores > self.config.detection_threshold).astype(mx.int32)
        return predictions, scores


def hallucination_loss(
    scores: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """
    Binary cross-entropy loss for hallucination detection.

    Used to supervise the overall detection system if labels are available.

    Args:
        scores: Predicted hallucination scores (batch,) in range [0, 1]
        targets: Target labels (batch,) - 1 if hallucination, 0 otherwise
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Clamp scores to avoid log(0)
    scores = mx.clip(scores, 1e-7, 1.0 - 1e-7)
    targets = targets.astype(mx.float32)

    # Binary cross entropy
    loss = -(targets * mx.log(scores) + (1 - targets) * mx.log(1 - scores))

    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


class HallucinationLoss(nn.Module):
    """Hallucination detection loss as nn.Module wrapper."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def __call__(
        self,
        scores: mx.array,
        targets: mx.array,
    ) -> mx.array:
        return hallucination_loss(
            scores=scores,
            targets=targets,
            reduction=self.reduction,
        )


def compute_hallucination_metrics(
    predictions: mx.array,
    targets: mx.array,
) -> dict[str, float]:
    """
    Compute hallucination detection metrics.

    Args:
        predictions: Predicted hallucination labels (batch,)
        targets: True hallucination labels (batch,)

    Returns:
        Dict with precision, recall, f1, accuracy
    """
    predictions = predictions.astype(mx.int32)
    targets = targets.astype(mx.int32)

    # True positives, etc.
    tp = mx.sum((predictions == 1) & (targets == 1)).astype(mx.float32)
    fp = mx.sum((predictions == 1) & (targets == 0)).astype(mx.float32)
    fn = mx.sum((predictions == 0) & (targets == 1)).astype(mx.float32)
    tn = mx.sum((predictions == 0) & (targets == 0)).astype(mx.float32)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
        "accuracy": float(accuracy.item()),
        "true_positives": int(tp.item()),
        "false_positives": int(fp.item()),
        "false_negatives": int(fn.item()),
        "true_negatives": int(tn.item()),
    }
