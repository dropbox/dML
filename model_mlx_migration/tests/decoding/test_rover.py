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
Tests for ROVER voting algorithm.

Tests cover:
1. Configuration validation
2. Hypothesis alignment
3. Basic voting scenarios
4. Confidence weighting
5. Null word handling
6. Edge cases
7. Integration with multiple sources
"""

import pytest

from src.decoding.rover import (
    NULL_TOKEN,
    ROVER,
    AlignmentMethod,
    Hypothesis,
    ROVERConfig,
    ROVERResult,
    _dp_edit_distance,
    align_hypotheses,
    compute_oracle_wer,
    vote_rover,
)


class TestROVERConfig:
    """Test ROVER configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ROVERConfig()
        assert config.alignment_method == AlignmentMethod.DP
        assert config.use_confidence_weights is True
        assert config.null_weight == 0.5
        assert config.prefer_non_null is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ROVERConfig(
            alignment_method=AlignmentMethod.SIMPLE,
            use_confidence_weights=False,
            null_weight=0.3,
        )
        assert config.alignment_method == AlignmentMethod.SIMPLE
        assert config.use_confidence_weights is False
        assert config.null_weight == 0.3


class TestHypothesis:
    """Test Hypothesis dataclass."""

    def test_basic_hypothesis(self):
        """Test basic hypothesis creation."""
        hyp = Hypothesis(words=["hello", "world"])
        assert hyp.words == ["hello", "world"]
        assert hyp.confidences == [1.0, 1.0]  # Default

    def test_hypothesis_with_confidences(self):
        """Test hypothesis with explicit confidences."""
        hyp = Hypothesis(words=["hello", "world"], confidences=[0.9, 0.8])
        assert hyp.confidences == [0.9, 0.8]

    def test_hypothesis_length_mismatch(self):
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError):
            Hypothesis(words=["hello", "world"], confidences=[0.9])

    def test_hypothesis_source(self):
        """Test hypothesis source tracking."""
        hyp = Hypothesis(words=["hello"], source="whisper")
        assert hyp.source == "whisper"


class TestEditDistance:
    """Test edit distance computation."""

    def test_identical_sequences(self):
        """Test edit distance for identical sequences."""
        distance, alignment = _dp_edit_distance(["a", "b", "c"], ["a", "b", "c"])
        assert distance == 0

    def test_one_substitution(self):
        """Test edit distance with one substitution."""
        distance, alignment = _dp_edit_distance(["a", "b", "c"], ["a", "x", "c"])
        assert distance == 1

    def test_one_insertion(self):
        """Test edit distance with one insertion."""
        distance, alignment = _dp_edit_distance(["a", "c"], ["a", "b", "c"])
        assert distance == 1

    def test_one_deletion(self):
        """Test edit distance with one deletion."""
        distance, alignment = _dp_edit_distance(["a", "b", "c"], ["a", "c"])
        assert distance == 1

    def test_empty_sequences(self):
        """Test edit distance with empty sequences."""
        distance, _ = _dp_edit_distance([], [])
        assert distance == 0

        distance, _ = _dp_edit_distance(["a", "b"], [])
        assert distance == 2

        distance, _ = _dp_edit_distance([], ["a", "b"])
        assert distance == 2


class TestAlignment:
    """Test hypothesis alignment."""

    def test_simple_alignment_same_length(self):
        """Test simple alignment with same-length hypotheses."""
        config = ROVERConfig(alignment_method=AlignmentMethod.SIMPLE)
        hypotheses = [
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["halo", "word"]),
        ]

        aligned, wtn_length = align_hypotheses(hypotheses, config)

        assert wtn_length == 2
        assert len(aligned) == 2
        assert aligned[0].words == ["hello", "world"]
        assert aligned[1].words == ["halo", "word"]

    def test_simple_alignment_different_length(self):
        """Test simple alignment with different-length hypotheses."""
        config = ROVERConfig(alignment_method=AlignmentMethod.SIMPLE)
        hypotheses = [
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["hello"]),
        ]

        aligned, wtn_length = align_hypotheses(hypotheses, config)

        assert wtn_length == 2
        assert aligned[0].words == ["hello", "world"]
        assert aligned[1].words == ["hello", NULL_TOKEN]

    def test_dp_alignment_basic(self):
        """Test DP alignment with basic case."""
        config = ROVERConfig(alignment_method=AlignmentMethod.DP)
        hypotheses = [
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["hello", "world"]),
        ]

        aligned, wtn_length = align_hypotheses(hypotheses, config)

        assert wtn_length == 2
        assert len(aligned) == 2

    def test_empty_hypotheses(self):
        """Test alignment with empty input."""
        aligned, wtn_length = align_hypotheses([])
        assert aligned == []
        assert wtn_length == 0


class TestBasicVoting:
    """Test basic ROVER voting scenarios."""

    def test_unanimous_vote(self):
        """Test voting when all hypotheses agree."""
        hypotheses = [
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["hello", "world"]),
        ]

        result = vote_rover(hypotheses)

        assert result.words == ["hello", "world"]

    def test_majority_vote(self):
        """Test voting with majority decision."""
        hypotheses = [
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["halo", "word"]),
        ]

        result = vote_rover(hypotheses)

        assert result.words == ["hello", "world"]

    def test_single_hypothesis(self):
        """Test voting with single hypothesis."""
        hypotheses = [
            Hypothesis(words=["hello", "world"]),
        ]

        result = vote_rover(hypotheses)

        assert result.words == ["hello", "world"]

    def test_empty_hypotheses(self):
        """Test voting with empty input."""
        result = vote_rover([])
        assert result.words == []

    def test_phoneme_weighting_homophone_boost(self):
        """
        Phoneme weighting should let near-homophones reinforce each other.

        Without phoneme weighting, the single highest-confidence outlier can win.
        With phoneme weighting enabled, two homophone-like candidates should beat
        the outlier.
        """
        hypotheses = [
            Hypothesis(
                words=["their"],
                confidences=[0.90],
                phonemes=[["ð", "ɛ", "ɹ"]],
                source="a",
            ),
            Hypothesis(
                words=["there"],
                confidences=[0.85],
                phonemes=[["ð", "ɛ", "ɹ"]],
                source="b",
            ),
            Hypothesis(
                words=["dog"],
                confidences=[0.95],
                phonemes=[["d", "ɔ", "ɡ"]],
                source="c",
            ),
        ]

        result_no_ph = vote_rover(
            hypotheses,
            ROVERConfig(use_phoneme_weighting=False),
        )
        assert result_no_ph.words == ["dog"]

        result_ph = vote_rover(
            hypotheses,
            ROVERConfig(use_phoneme_weighting=True, phoneme_mismatch_penalty=0.4),
        )
        assert result_ph.words == ["their"]

    def test_empty_words(self):
        """Test voting with empty word list."""
        hypotheses = [
            Hypothesis(words=[]),
            Hypothesis(words=[]),
        ]

        result = vote_rover(hypotheses)
        assert result.words == []


class TestConfidenceWeighting:
    """Test confidence-weighted voting."""

    def test_high_confidence_wins(self):
        """Test that high confidence hypothesis wins."""
        config = ROVERConfig(use_confidence_weights=True)
        hypotheses = [
            Hypothesis(words=["hello"], confidences=[0.9]),
            Hypothesis(words=["halo"], confidences=[0.1]),
        ]

        result = vote_rover(hypotheses, config)

        assert result.words == ["hello"]

    def test_confidence_disabled(self):
        """Test voting without confidence weights (pure vote count)."""
        config = ROVERConfig(use_confidence_weights=False)
        hypotheses = [
            Hypothesis(words=["hello"], confidences=[0.9]),
            Hypothesis(words=["halo"], confidences=[0.1]),
        ]

        result = vote_rover(hypotheses, config)

        # With equal vote counts and no confidence weighting,
        # result depends on ordering/tie-breaking
        assert len(result.words) == 1

    def test_confidence_affects_result(self):
        """Test that confidence actually affects result."""
        config = ROVERConfig(use_confidence_weights=True)

        # Two hypotheses for "hello", one for "halo" with very high confidence
        hypotheses = [
            Hypothesis(words=["hello"], confidences=[0.5]),
            Hypothesis(words=["hello"], confidences=[0.5]),
            Hypothesis(words=["halo"], confidences=[0.99]),
        ]

        result = vote_rover(hypotheses, config)

        # "hello" has 2 votes with 0.5 each = 1.0 total
        # "halo" has 1 vote with 0.99 = 0.99 total
        # "hello" should win
        assert result.words == ["hello"]


class TestNullHandling:
    """Test null word (deletion) handling."""

    def test_null_weight_affects_votes(self):
        """Test that null weight affects voting."""
        config = ROVERConfig(null_weight=0.1)  # Low weight for nulls

        hypotheses = [
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["hello"]),  # Missing "world"
        ]

        result = vote_rover(hypotheses, config)

        # "world" should win over null because of low null weight
        assert "world" in result.words

    def test_prefer_non_null(self):
        """Test preference for non-null in ties."""
        config = ROVERConfig(prefer_non_null=True, use_confidence_weights=False)

        hypotheses = [
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["hello"]),  # Missing "world"
        ]

        result = vote_rover(hypotheses, config)

        # Should prefer "world" over null in tie
        assert "world" in result.words


class TestROVERClass:
    """Test ROVER class interface."""

    def test_rover_init(self):
        """Test ROVER initialization."""
        rover = ROVER()
        assert rover.config is not None

    def test_register_source(self):
        """Test source registration."""
        rover = ROVER()
        rover.register_source("whisper", weight=1.5)

        assert rover.sources["whisper"] == 1.5

    def test_combine_transducer_ctc(self):
        """Test transducer + CTC combination."""
        rover = ROVER()

        result = rover.combine_transducer_ctc(
            transducer_words=["hello", "world"],
            transducer_confs=[0.9, 0.85],
            ctc_words=["hello", "word"],
            ctc_confs=[0.8, 0.75],
        )

        assert "hello" in result.words
        assert len(result.words) == 2

    def test_combine_with_whisper(self):
        """Test combination with Whisper."""
        rover = ROVER()

        result = rover.combine_with_whisper(
            primary_words=["hello", "world"],
            primary_confs=[0.9, 0.85],
            primary_source="zipformer",
            whisper_words=["hello", "world"],
            whisper_confs=[0.95, 0.9],
        )

        assert result.words == ["hello", "world"]

    def test_combine_three_way(self):
        """Test three-way combination."""
        rover = ROVER()

        result = rover.combine_three_way(
            transducer_words=["hello", "world"],
            transducer_confs=[0.9, 0.85],
            ctc_words=["halo", "world"],
            ctc_confs=[0.8, 0.9],
            whisper_words=["hello", "world"],
            whisper_confs=[0.95, 0.88],
        )

        # "hello" has 2 votes (transducer, whisper) vs "halo" has 1 (ctc)
        # "world" is unanimous
        assert result.words == ["hello", "world"]

    def test_source_weights(self):
        """Test that source weights are applied."""
        rover = ROVER()
        rover.register_source("trusted", weight=2.0)
        rover.register_source("untrusted", weight=0.5)

        hypotheses = [
            Hypothesis(words=["hello"], confidences=[0.5], source="trusted"),
            Hypothesis(words=["halo"], confidences=[1.0], source="untrusted"),
        ]

        result = rover.vote(hypotheses)

        # "hello" gets 0.5 * 2.0 = 1.0
        # "halo" gets 1.0 * 0.5 = 0.5
        assert result.words == ["hello"]


class TestROVERResult:
    """Test ROVERResult dataclass."""

    def test_text_property(self):
        """Test text property."""
        result = ROVERResult(
            words=["hello", "world"],
            confidences=[0.9, 0.85],
            slots=[],
        )

        assert result.text == "hello world"

    def test_word_count(self):
        """Test word_count property."""
        result = ROVERResult(
            words=["hello", "world", "test"],
            confidences=[0.9, 0.85, 0.8],
            slots=[],
        )

        assert result.word_count == 3


class TestVotingSlot:
    """Test VotingSlot metadata."""

    def test_slot_info(self):
        """Test that voting slots contain useful info."""
        hypotheses = [
            Hypothesis(words=["hello"], source="trans"),
            Hypothesis(words=["halo"], source="ctc"),
            Hypothesis(words=["hello"], source="whisper"),
        ]

        result = vote_rover(hypotheses)

        assert len(result.slots) > 0
        slot = result.slots[0]

        assert "hello" in slot.candidates
        assert "halo" in slot.candidates
        assert slot.vote_counts["hello"] == 2
        assert slot.vote_counts["halo"] == 1
        assert "trans" in slot.sources["hello"]
        assert "whisper" in slot.sources["hello"]


class TestOracleWER:
    """Test oracle WER computation."""

    def test_oracle_wer_basic(self):
        """Test basic oracle WER."""
        hypotheses = [
            Hypothesis(words=["hello", "world"]),
            Hypothesis(words=["hello", "word"]),  # One error
        ]
        reference = ["hello", "world"]

        wer, best = compute_oracle_wer(hypotheses, reference)

        assert wer == 0.0  # First hypothesis is perfect
        assert best.words == ["hello", "world"]

    def test_oracle_wer_all_imperfect(self):
        """Test oracle WER when all hypotheses have errors."""
        hypotheses = [
            Hypothesis(words=["halo", "world"]),  # One error
            Hypothesis(words=["hello", "word"]),  # One error
        ]
        reference = ["hello", "world"]

        wer, best = compute_oracle_wer(hypotheses, reference)

        assert wer == 0.5  # 1/2 = 0.5

    def test_oracle_wer_empty(self):
        """Test oracle WER with empty hypotheses."""
        wer, _ = compute_oracle_wer([], ["hello"])
        assert wer == 1.0


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_very_different_lengths(self):
        """Test with very different hypothesis lengths."""
        hypotheses = [
            Hypothesis(words=["a"]),
            Hypothesis(words=["a", "b", "c", "d", "e"]),
        ]

        result = vote_rover(hypotheses)

        # Should handle gracefully
        assert len(result.words) >= 1

    def test_all_different_words(self):
        """Test when all hypotheses have different words."""
        hypotheses = [
            Hypothesis(words=["apple"]),
            Hypothesis(words=["banana"]),
            Hypothesis(words=["cherry"]),
        ]

        result = vote_rover(hypotheses)

        # Should select one of them
        assert len(result.words) == 1
        assert result.words[0] in ["apple", "banana", "cherry"]

    def test_repeated_words(self):
        """Test with repeated words in hypothesis."""
        hypotheses = [
            Hypothesis(words=["the", "the", "cat"]),
            Hypothesis(words=["the", "cat"]),
        ]

        result = vote_rover(hypotheses)

        # Should handle repeated words
        assert "cat" in result.words

    def test_special_characters(self):
        """Test with special characters in words."""
        hypotheses = [
            Hypothesis(words=["hello!", "@world"]),
            Hypothesis(words=["hello!", "@world"]),
        ]

        result = vote_rover(hypotheses)

        assert result.words == ["hello!", "@world"]


class TestIntegration:
    """Integration tests for ROVER."""

    def test_realistic_asr_scenario(self):
        """Test with realistic ASR output."""
        rover = ROVER()

        # Simulate Zipformer streaming output
        zipformer_words = ["the", "quick", "brown", "fox", "jumps"]
        zipformer_confs = [0.95, 0.88, 0.92, 0.85, 0.90]

        # Simulate CTC output (slightly different)
        ctc_words = ["the", "quick", "brown", "fox", "jump"]
        ctc_confs = [0.90, 0.85, 0.88, 0.82, 0.75]

        # Simulate Whisper output
        whisper_words = ["the", "quick", "brown", "fox", "jumps"]
        whisper_confs = [0.98, 0.96, 0.97, 0.94, 0.93]

        result = rover.combine_three_way(
            zipformer_words, zipformer_confs,
            ctc_words, ctc_confs,
            whisper_words, whisper_confs,
        )

        # "jumps" should win over "jump" (2 vs 1)
        assert result.words == ["the", "quick", "brown", "fox", "jumps"]

    def test_handles_insertion_deletion(self):
        """Test handling of insertions and deletions."""
        hypotheses = [
            Hypothesis(words=["I", "went", "to", "the", "store"]),
            Hypothesis(words=["I", "went", "the", "store"]),  # Missing "to"
            Hypothesis(words=["I", "went", "to", "a", "store"]),  # "a" instead of "the"
        ]

        result = vote_rover(hypotheses)

        # Majority should decide
        assert "I" in result.words
        assert "went" in result.words
        assert "store" in result.words


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
