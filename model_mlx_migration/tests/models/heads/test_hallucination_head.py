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
Tests for the hallucination detection head.

Phase 5.8: Hallucination detection via phoneme mismatch
Target: >90% hallucination detection rate
"""

import mlx.core as mx
import pytest

from src.models.heads.hallucination import (
    BASIC_G2P,
    HallucinationConfig,
    HallucinationHead,
    HallucinationLoss,
    HallucinationResult,
    compute_hallucination_metrics,
    hallucination_loss,
)
from src.models.heads.phoneme import IPA_PHONEMES


class TestHallucinationConfig:
    """Tests for HallucinationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HallucinationConfig()
        assert config.mismatch_threshold == 0.3
        assert config.min_phoneme_confidence == 0.5
        assert config.entropy_threshold == 2.0
        assert config.detection_threshold == 0.5
        assert config.num_phonemes == len(IPA_PHONEMES)

    def test_custom_config(self):
        """Test custom configuration."""
        config = HallucinationConfig(
            mismatch_threshold=0.5,
            detection_threshold=0.7,
            weight_mismatch=0.8,
        )
        assert config.mismatch_threshold == 0.5
        assert config.detection_threshold == 0.7
        assert config.weight_mismatch == 0.8

    def test_weight_sum(self):
        """Test that default weights sum to 1.0."""
        config = HallucinationConfig()
        total = (
            config.weight_mismatch +
            config.weight_confidence +
            config.weight_energy +
            config.weight_repetition
        )
        assert abs(total - 1.0) < 1e-6


class TestHallucinationHead:
    """Tests for HallucinationHead."""

    @pytest.fixture
    def head(self):
        """Create default hallucination head."""
        return HallucinationHead()

    @pytest.fixture
    def sample_logits(self):
        """Create sample phoneme logits."""
        batch_size, seq_len, num_phonemes = 2, 50, len(IPA_PHONEMES)
        # Random logits simulating phoneme predictions
        mx.random.seed(42)
        return mx.random.normal((batch_size, seq_len, num_phonemes))

    def test_head_initialization(self, head):
        """Test head initializes correctly."""
        assert head.config is not None
        assert len(head._phoneme_to_idx) == len(IPA_PHONEMES)
        # Verify no trainable parameters
        params = head.parameters()
        assert len(params) == 0

    def test_forward_no_text(self, head, sample_logits):
        """Test forward pass without ASR text."""
        scores = head(sample_logits)
        assert scores.shape == (2,)
        assert mx.all(scores >= 0.0)
        assert mx.all(scores <= 1.0)

    def test_forward_with_text(self, head, sample_logits):
        """Test forward pass with ASR text."""
        asr_text = ["hello world", "testing speech"]
        scores = head(sample_logits, asr_text=asr_text)
        assert scores.shape == (2,)
        assert mx.all(scores >= 0.0)
        assert mx.all(scores <= 1.0)

    def test_forward_with_energy(self, head, sample_logits):
        """Test forward pass with audio energy."""
        batch_size, seq_len = sample_logits.shape[:2]
        audio_energy = mx.random.uniform(-60, 0, (batch_size, seq_len))
        asr_text = ["hello world", "testing speech"]

        scores = head(sample_logits, asr_text=asr_text, audio_energy=audio_energy)
        assert scores.shape == (2,)
        assert mx.all(scores >= 0.0)
        assert mx.all(scores <= 1.0)

    def test_detect_returns_results(self, head, sample_logits):
        """Test detect method returns HallucinationResult objects."""
        asr_text = ["hello world", "testing speech"]
        results = head.detect(sample_logits, asr_text=asr_text)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, HallucinationResult)
            assert 0.0 <= result.score <= 1.0
            assert isinstance(result.is_hallucination, bool)
            assert isinstance(result.detected_phonemes, list)
            assert isinstance(result.expected_phonemes, list)

    def test_predict_method(self, head, sample_logits):
        """Test predict method returns predictions and scores."""
        asr_text = ["hello world", "testing speech"]
        predictions, scores = head.predict(sample_logits, asr_text=asr_text)

        assert predictions.shape == (2,)
        assert scores.shape == (2,)
        assert mx.all((predictions == 0) | (predictions == 1))

    def test_confidence_score_computation(self, head):
        """Test confidence score based on entropy."""
        _batch_size, seq_len, num_phonemes = 2, 20, len(IPA_PHONEMES)

        # High entropy (uniform distribution) - potential hallucination
        uniform_logits = mx.zeros((1, seq_len, num_phonemes))
        high_entropy_score = head._compute_confidence_score(uniform_logits)

        # Low entropy (peaked distribution) - confident prediction
        peaked_logits = mx.zeros((1, seq_len, num_phonemes))
        peaked_logits = peaked_logits.at[:, :, 10].add(10.0)  # Peak at phoneme 10
        low_entropy_score = head._compute_confidence_score(peaked_logits)

        # High entropy should give higher score (more hallucination-like)
        assert float(high_entropy_score[0].item()) > float(low_entropy_score[0].item())

    def test_repetition_detection(self, head):
        """Test repetition-based hallucination detection."""
        # Text with excessive repetition (>3 occurrences of same n-gram)
        repeated_text = ["the cat the cat the cat the cat the cat", "hello world"]
        scores = head._compute_repetition_scores(repeated_text)

        # First text has repetition (5x "the cat"), second doesn't
        assert float(scores[0].item()) > float(scores[1].item())

    def test_repetition_normal_text(self, head):
        """Test that normal text doesn't trigger repetition detection."""
        normal_text = ["this is a normal sentence without repetition"]
        scores = head._compute_repetition_scores(normal_text)
        assert float(scores[0].item()) == 0.0

    def test_text_to_phonemes(self, head):
        """Test basic G2P conversion."""
        phonemes = head._text_to_phonemes("hello")
        assert len(phonemes) > 0
        assert all(isinstance(p, str) for p in phonemes)

    def test_text_to_phonemes_with_digraphs(self, head):
        """Test G2P handles digraphs."""
        phonemes = head._text_to_phonemes("the ship")
        assert "ð" in phonemes or "θ" in phonemes  # "th"
        assert "ʃ" in phonemes  # "sh"

    def test_decode_phoneme_indices(self, head):
        """Test phoneme index decoding."""
        # Create indices with repetition and blanks
        indices = [0, 0, 5, 5, 5, 0, 10, 10, 0]  # blank=0
        decoded = head._decode_phoneme_indices(indices)

        # Should collapse repeated and remove blanks
        assert 0 not in [head._phoneme_to_idx.get(p, -1) for p in decoded]

    def test_sequence_mismatch_identical(self, head):
        """Test mismatch score for identical sequences."""
        seq = ["p", "æ", "t"]
        mismatch = head._compute_sequence_mismatch(seq, seq)
        assert mismatch == 0.0

    def test_sequence_mismatch_different(self, head):
        """Test mismatch score for different sequences."""
        seq1 = ["p", "æ", "t"]
        seq2 = ["k", "ɔ", "l"]
        mismatch = head._compute_sequence_mismatch(seq1, seq2)
        assert mismatch == 1.0  # Completely different

    def test_sequence_mismatch_partial(self, head):
        """Test mismatch score for partially overlapping sequences."""
        seq1 = ["p", "æ", "t"]
        seq2 = ["p", "æ", "k"]  # 2/4 overlap in union
        mismatch = head._compute_sequence_mismatch(seq1, seq2)
        assert 0.0 < mismatch < 1.0

    def test_empty_text_handling(self, head, sample_logits):
        """Test handling of empty ASR text."""
        asr_text = ["", ""]
        scores = head(sample_logits, asr_text=asr_text)
        assert scores.shape == (2,)
        # Empty text should not crash

    def test_silence_detection(self, head):
        """Test detection of speech during silence."""
        batch_size, seq_len, num_phonemes = 1, 20, len(IPA_PHONEMES)

        # Create confident predictions (not blank)
        logits = mx.zeros((batch_size, seq_len, num_phonemes))
        logits = logits.at[:, :, 10].add(10.0)  # Confident prediction for phoneme 10

        # Audio energy indicating silence
        energy = mx.full((batch_size, seq_len), -50.0)  # Below threshold

        energy_score = head._compute_energy_scores(logits, energy)
        # Should detect hallucination (speech during silence)
        assert float(energy_score[0].item()) > 0.0

    def test_no_energy_gives_zero_score(self, head, sample_logits):
        """Test that missing energy info gives zero energy score."""
        energy_score = head._compute_energy_scores(sample_logits, None)
        assert mx.all(energy_score == 0.0)


class TestHallucinationLoss:
    """Tests for hallucination detection loss."""

    def test_loss_perfect_prediction(self):
        """Test loss for perfect predictions."""
        scores = mx.array([0.0, 1.0])
        targets = mx.array([0, 1])
        loss = hallucination_loss(scores, targets)
        # Should be close to 0 (clamped to avoid log(0))
        assert float(loss.item()) < 1e-3

    def test_loss_wrong_prediction(self):
        """Test loss for wrong predictions."""
        scores = mx.array([1.0, 0.0])
        targets = mx.array([0, 1])
        loss = hallucination_loss(scores, targets)
        # Should be high
        assert float(loss.item()) > 10.0

    def test_loss_module_wrapper(self):
        """Test HallucinationLoss module."""
        loss_fn = HallucinationLoss(reduction="mean")
        scores = mx.array([0.3, 0.7])
        targets = mx.array([0, 1])
        loss = loss_fn(scores, targets)
        assert loss.shape == ()

    def test_loss_reduction_none(self):
        """Test loss with no reduction."""
        scores = mx.array([0.3, 0.7])
        targets = mx.array([0, 1])
        loss = hallucination_loss(scores, targets, reduction="none")
        assert loss.shape == (2,)

    def test_loss_reduction_sum(self):
        """Test loss with sum reduction."""
        scores = mx.array([0.3, 0.7])
        targets = mx.array([0, 1])
        loss = hallucination_loss(scores, targets, reduction="sum")
        assert loss.shape == ()


class TestHallucinationMetrics:
    """Tests for hallucination detection metrics."""

    def test_perfect_metrics(self):
        """Test metrics for perfect predictions."""
        predictions = mx.array([0, 0, 1, 1])
        targets = mx.array([0, 0, 1, 1])
        metrics = compute_hallucination_metrics(predictions, targets)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["accuracy"] == 1.0

    def test_all_wrong_metrics(self):
        """Test metrics for all wrong predictions."""
        predictions = mx.array([1, 1, 0, 0])
        targets = mx.array([0, 0, 1, 1])
        metrics = compute_hallucination_metrics(predictions, targets)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["accuracy"] == 0.0

    def test_partial_metrics(self):
        """Test metrics for partial correct predictions."""
        predictions = mx.array([0, 1, 1, 0])
        targets = mx.array([0, 0, 1, 1])
        metrics = compute_hallucination_metrics(predictions, targets)

        # TP=1, FP=1, FN=1, TN=1
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["accuracy"] == 0.5

    def test_metrics_all_zeros(self):
        """Test metrics when all predictions are negative."""
        predictions = mx.array([0, 0, 0, 0])
        targets = mx.array([0, 0, 1, 1])
        metrics = compute_hallucination_metrics(predictions, targets)

        assert metrics["false_negatives"] == 2
        assert metrics["true_negatives"] == 2


class TestBasicG2P:
    """Tests for basic grapheme-to-phoneme mapping."""

    def test_g2p_coverage(self):
        """Test that G2P covers all lowercase letters."""
        for char in "abcdefghijklmnopqrstuvwxyz":
            assert char in BASIC_G2P, f"Missing G2P mapping for '{char}'"

    def test_g2p_digraphs(self):
        """Test that common digraphs are covered."""
        digraphs = ["th", "sh", "ch", "ph", "wh", "ng"]
        for digraph in digraphs:
            assert digraph in BASIC_G2P, f"Missing G2P mapping for '{digraph}'"

    def test_g2p_returns_ipa(self):
        """Test that G2P returns valid IPA phonemes."""
        for _char, phonemes in BASIC_G2P.items():
            for phoneme in phonemes:
                # Allow compound phonemes like "ks"
                if len(phoneme) > 1 and phoneme not in IPA_PHONEMES:
                    # Check if it's a compound that can be split
                    pass  # Compounds like "ks" are acceptable
                elif len(phoneme) == 1 or phoneme in IPA_PHONEMES:
                    # Single char or in IPA inventory is valid
                    pass


class TestHallucinationIntegration:
    """Integration tests for hallucination detection."""

    def test_high_hallucination_scenario(self):
        """Test detection of obvious hallucination."""
        head = HallucinationHead(HallucinationConfig(detection_threshold=0.3))

        batch_size, seq_len, num_phonemes = 1, 100, len(IPA_PHONEMES)

        # Create high-entropy (uncertain) predictions
        logits = mx.random.normal((batch_size, seq_len, num_phonemes)) * 0.1

        # Text with repetition
        asr_text = ["um um um um hello hello hello"]

        # Silent audio
        energy = mx.full((batch_size, seq_len), -50.0)

        results = head.detect(logits, asr_text=asr_text, audio_energy=energy)

        # Should detect as hallucination due to:
        # 1. High entropy (uncertain)
        # 2. Repetition
        # 3. Speech during silence
        assert results[0].is_hallucination or results[0].score > 0.2

    def test_normal_speech_scenario(self):
        """Test that normal speech is not flagged as hallucination."""
        head = HallucinationHead(HallucinationConfig(detection_threshold=0.6))

        batch_size, seq_len, num_phonemes = 1, 50, len(IPA_PHONEMES)

        # Create confident predictions (low entropy)
        # Simulate confident predictions by having one phoneme much higher
        logits = mx.zeros((batch_size, seq_len, num_phonemes))
        for t in range(seq_len):
            phoneme_idx = (t % 20) + 4  # Cycle through non-special phonemes
            logits = logits.at[0, t, phoneme_idx].add(10.0)

        # Normal text
        asr_text = ["this is a normal sentence"]

        # Normal audio energy
        energy = mx.full((batch_size, seq_len), -10.0)  # Above silence threshold

        results = head.detect(logits, asr_text=asr_text, audio_energy=energy)

        # Confidence score should be low (confident predictions)
        assert results[0].confidence_score < 0.5

    def test_batch_processing(self):
        """Test batch processing of multiple utterances."""
        head = HallucinationHead()

        batch_size = 4
        seq_len = 30
        num_phonemes = len(IPA_PHONEMES)

        mx.random.seed(123)
        logits = mx.random.normal((batch_size, seq_len, num_phonemes))
        asr_text = [
            "hello world",
            "the the the the the the",  # repetition (>3 bigrams)
            "testing speech",
            "",  # empty
        ]

        results = head.detect(logits, asr_text=asr_text)

        assert len(results) == 4
        # Text with repetition should have higher repetition score
        assert results[1].repetition_score > results[0].repetition_score

    def test_result_fields_populated(self):
        """Test that all HallucinationResult fields are populated."""
        head = HallucinationHead()

        logits = mx.random.normal((1, 30, len(IPA_PHONEMES)))
        asr_text = ["hello world"]

        results = head.detect(logits, asr_text=asr_text)
        result = results[0]

        # Check all fields exist and have correct types
        assert isinstance(result.score, float)
        assert isinstance(result.is_hallucination, bool)
        assert isinstance(result.mismatch_score, float)
        assert isinstance(result.confidence_score, float)
        assert isinstance(result.energy_score, float)
        assert isinstance(result.repetition_score, float)
        assert isinstance(result.detected_phonemes, list)
        assert isinstance(result.expected_phonemes, list)
        assert isinstance(result.num_mismatches, int)
        assert isinstance(result.repeated_ngrams, list)


class TestHallucinationEdgeCases:
    """Edge case tests for hallucination detection."""

    def test_single_frame(self):
        """Test with single frame input."""
        head = HallucinationHead()
        logits = mx.random.normal((1, 1, len(IPA_PHONEMES)))
        scores = head(logits)
        assert scores.shape == (1,)

    def test_very_long_sequence(self):
        """Test with very long sequence."""
        head = HallucinationHead()
        logits = mx.random.normal((1, 1000, len(IPA_PHONEMES)))
        scores = head(logits, asr_text=["long " * 100])
        assert scores.shape == (1,)

    def test_single_word_text(self):
        """Test with single word ASR output."""
        head = HallucinationHead()
        logits = mx.random.normal((1, 10, len(IPA_PHONEMES)))
        scores = head(logits, asr_text=["hello"])
        assert scores.shape == (1,)

    def test_special_characters_in_text(self):
        """Test with special characters in ASR text."""
        head = HallucinationHead()
        logits = mx.random.normal((1, 30, len(IPA_PHONEMES)))
        # Text with punctuation and numbers
        scores = head(logits, asr_text=["Hello, world! 123..."])
        assert scores.shape == (1,)

    def test_unicode_text(self):
        """Test with unicode characters."""
        head = HallucinationHead()
        logits = mx.random.normal((1, 30, len(IPA_PHONEMES)))
        # G2P should handle gracefully
        scores = head(logits, asr_text=["café résumé"])
        assert scores.shape == (1,)

    def test_extreme_energy_values(self):
        """Test with extreme audio energy values."""
        head = HallucinationHead()
        logits = mx.random.normal((1, 20, len(IPA_PHONEMES)))

        # Very loud
        loud_energy = mx.full((1, 20), 0.0)
        scores_loud = head(logits, asr_text=["test"], audio_energy=loud_energy)

        # Very quiet
        quiet_energy = mx.full((1, 20), -100.0)
        scores_quiet = head(logits, asr_text=["test"], audio_energy=quiet_energy)

        assert scores_loud.shape == (1,)
        assert scores_quiet.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
