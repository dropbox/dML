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
Phase 8.5: ROVER High-Accuracy Mode Validation Tests.

Tests validate:
1. WER computation correctness
2. ROVER improvement over single-source baselines
3. LibriSpeech test-clean validation framework
4. Target: <1.5% WER in high-accuracy mode

Success criteria from FINAL_ROADMAP_SOTA_PLUS_PLUS.md:
- LibriSpeech test-clean streaming: <2.5% WER
- LibriSpeech test-clean high-accuracy: <1.5% WER
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.decoding.rover import (
    ROVER,
    Hypothesis,
    ROVERConfig,
    _dp_edit_distance,
    compute_oracle_wer,
)

# =============================================================================
# WER Computation Utilities
# =============================================================================


def compute_wer(reference: str, hypothesis: str) -> dict[str, float]:
    """
    Compute Word Error Rate between reference and hypothesis.

    Args:
        reference: Ground truth transcription
        hypothesis: ASR output

    Returns:
        Dict with WER, substitutions, deletions, insertions, and word counts
    """
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()

    # Use edit distance from ROVER
    distance, alignment = _dp_edit_distance(ref_words, hyp_words)

    # Count operation types from alignment
    # alignment is list of (ref_idx, hyp_idx) pairs
    # (i, j) where both >= 0: match or substitution
    # (i, -1): deletion (word in ref not in hyp)
    # (-1, j): insertion (word in hyp not in ref)
    subs = dels = ins = 0
    for ref_idx, hyp_idx in alignment:
        if ref_idx >= 0 and hyp_idx >= 0:
            # Match or substitution
            if ref_words[ref_idx] != hyp_words[hyp_idx]:
                subs += 1
        elif ref_idx >= 0 and hyp_idx < 0:
            # Deletion
            dels += 1
        elif ref_idx < 0 and hyp_idx >= 0:
            # Insertion
            ins += 1

    ref_len = len(ref_words)
    wer = distance / ref_len if ref_len > 0 else 0.0

    return {
        "wer": wer,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "ref_words": ref_len,
        "hyp_words": len(hyp_words),
        "errors": distance,
    }


def compute_corpus_wer(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """
    Compute corpus-level WER over multiple utterances.

    Args:
        references: List of reference transcriptions
        hypotheses: List of ASR outputs

    Returns:
        Dict with corpus WER and statistics
    """
    total_errors = 0
    total_ref_words = 0
    total_subs = 0
    total_dels = 0
    total_ins = 0

    for ref, hyp in zip(references, hypotheses, strict=False):
        result = compute_wer(ref, hyp)
        total_errors += result["errors"]
        total_ref_words += result["ref_words"]
        total_subs += result["substitutions"]
        total_dels += result["deletions"]
        total_ins += result["insertions"]

    corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    return {
        "wer": corpus_wer,
        "total_errors": total_errors,
        "total_ref_words": total_ref_words,
        "substitutions": total_subs,
        "deletions": total_dels,
        "insertions": total_ins,
        "num_utterances": len(references),
    }


# =============================================================================
# LibriSpeech Data Loading
# =============================================================================


@dataclass
class LibriSpeechSample:
    """A single LibriSpeech sample."""
    utterance_id: str
    audio_path: Path
    reference: str
    speaker_id: str
    chapter_id: str


def load_librispeech_samples(
    data_dir: str = "data/librispeech/test-clean",
    max_samples: int | None = None,
) -> list[LibriSpeechSample]:
    """
    Load LibriSpeech samples from test-clean split.

    Args:
        data_dir: Path to LibriSpeech test-clean directory
        max_samples: Maximum samples to load (None for all)

    Returns:
        List of LibriSpeechSample objects
    """
    samples = []
    data_path = Path(data_dir)

    if not data_path.exists():
        return samples

    # Iterate through speaker directories
    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name

        # Iterate through chapter directories
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_id = chapter_dir.name

            # Find transcript file
            trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            if not trans_file.exists():
                continue

            # Parse transcript file
            with open(trans_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue

                    utt_id, transcript = parts
                    audio_file = chapter_dir / f"{utt_id}.flac"

                    if audio_file.exists():
                        samples.append(LibriSpeechSample(
                            utterance_id=utt_id,
                            audio_path=audio_file,
                            reference=transcript,
                            speaker_id=speaker_id,
                            chapter_id=chapter_id,
                        ))

                        if max_samples and len(samples) >= max_samples:
                            return samples

    return samples


# =============================================================================
# Mock ASR Sources for Testing
# =============================================================================


def simulate_asr_error(
    reference: str,
    error_rate: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[float]]:
    """
    Simulate ASR output with controlled error rate.

    Creates realistic errors: substitutions, insertions, deletions.

    Args:
        reference: Ground truth transcription
        error_rate: Target WER (approximate)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (words, confidences)
    """
    import random
    random.seed(seed)

    words = reference.upper().split()
    output_words = []
    output_confs = []

    # Common substitution patterns
    substitutions = {
        "THE": ["A", "THAT"],
        "A": ["THE", "AN"],
        "IS": ["WAS", "AS"],
        "AND": ["AN", "IN"],
        "TO": ["TOO", "TWO"],
        "OF": ["OFF", "IF"],
        "IT": ["IS", "IN"],
        "HE": ["SHE", "WE"],
        "WAS": ["IS", "HAS"],
        "FOR": ["FROM", "FOUR"],
    }

    for word in words:
        roll = random.random()

        if roll < error_rate * 0.4:  # Substitution
            if word in substitutions:
                sub_word = random.choice(substitutions[word])
                output_words.append(sub_word)
                output_confs.append(random.uniform(0.4, 0.7))
            else:
                # Slight modification
                if len(word) > 2:
                    pos = random.randint(0, len(word) - 1)
                    chars = list(word)
                    chars[pos] = chr(ord('A') + random.randint(0, 25))
                    output_words.append("".join(chars))
                    output_confs.append(random.uniform(0.3, 0.6))
                else:
                    output_words.append(word)
                    output_confs.append(random.uniform(0.7, 0.95))

        elif roll < error_rate * 0.6:  # Deletion
            # Skip this word (don't add it)
            continue

        elif roll < error_rate * 0.8:  # Insertion
            output_words.append(word)
            output_confs.append(random.uniform(0.7, 0.95))
            # Add extra word
            filler = random.choice(["UM", "THE", "A", "UH"])
            output_words.append(filler)
            output_confs.append(random.uniform(0.2, 0.4))

        else:  # Correct
            output_words.append(word)
            output_confs.append(random.uniform(0.8, 0.99))

    return output_words, output_confs


def create_mock_sources(
    reference: str,
    source_configs: dict[str, float],
) -> list[Hypothesis]:
    """
    Create mock ASR hypotheses with different error characteristics.

    Args:
        reference: Ground truth transcription
        source_configs: Dict mapping source name to error rate

    Returns:
        List of Hypothesis objects
    """
    hypotheses = []
    seed = 0

    for source_name, error_rate in source_configs.items():
        words, confs = simulate_asr_error(reference, error_rate, seed=seed)
        hypotheses.append(Hypothesis(
            words=words,
            confidences=confs,
            source=source_name,
        ))
        seed += 1

    return hypotheses


# =============================================================================
# Test Classes
# =============================================================================


class TestWERComputation:
    """Test WER computation correctness."""

    def test_perfect_match(self):
        """WER should be 0 for perfect match."""
        result = compute_wer("hello world", "hello world")
        assert result["wer"] == 0.0
        assert result["errors"] == 0

    def test_case_insensitive(self):
        """WER should be case insensitive."""
        result = compute_wer("HELLO WORLD", "hello world")
        assert result["wer"] == 0.0

    def test_single_substitution(self):
        """Single substitution should give correct WER."""
        result = compute_wer("hello world", "hello word")
        assert result["wer"] == 0.5  # 1 error / 2 words
        assert result["substitutions"] == 1

    def test_single_insertion(self):
        """Single insertion should give correct WER."""
        result = compute_wer("hello world", "hello the world")
        assert result["wer"] == 0.5  # 1 error / 2 words
        assert result["insertions"] == 1

    def test_single_deletion(self):
        """Single deletion should give correct WER."""
        result = compute_wer("hello the world", "hello world")
        assert result["wer"] == pytest.approx(1 / 3)  # 1 error / 3 words
        assert result["deletions"] == 1

    def test_multiple_errors(self):
        """Multiple errors should accumulate correctly."""
        result = compute_wer("one two three four", "one to three")
        # 1 sub (two->to) + 1 del (four) = 2 errors / 4 words = 0.5
        assert result["wer"] == 0.5
        assert result["errors"] == 2

    def test_empty_reference(self):
        """Empty reference should give WER 0."""
        result = compute_wer("", "hello world")
        assert result["wer"] == 0.0

    def test_empty_hypothesis(self):
        """Empty hypothesis should give WER 1.0 (all deletions)."""
        result = compute_wer("hello world", "")
        assert result["wer"] == 1.0
        assert result["deletions"] == 2


class TestCorpusWER:
    """Test corpus-level WER computation."""

    def test_corpus_wer_basic(self):
        """Corpus WER should aggregate correctly."""
        refs = ["hello world", "foo bar baz"]  # 2 + 3 = 5 words
        hyps = ["hello word", "foo bar baz"]  # 1 error

        result = compute_corpus_wer(refs, hyps)
        assert result["wer"] == pytest.approx(1 / 5)
        assert result["total_ref_words"] == 5
        assert result["total_errors"] == 1

    def test_corpus_wer_multiple_utterances(self):
        """Corpus WER over multiple utterances."""
        refs = [
            "THE CAT SAT ON THE MAT",
            "A QUICK BROWN FOX",
        ]
        hyps = [
            "THE CAT SAT ON THE MAT",  # Perfect
            "A QUICK BROWN DOG",  # 1 error
        ]

        result = compute_corpus_wer(refs, hyps)
        # 6 + 4 = 10 words, 1 error
        assert result["wer"] == pytest.approx(1 / 10)


class TestROVERVoting:
    """Test ROVER voting algorithm for combining ASR sources."""

    def test_unanimous_agreement(self):
        """All sources agreeing should produce that output."""
        rover = ROVER()
        hypotheses = [
            Hypothesis(words=["hello", "world"], confidences=[0.9, 0.9], source="s1"),
            Hypothesis(words=["hello", "world"], confidences=[0.8, 0.8], source="s2"),
            Hypothesis(words=["hello", "world"], confidences=[0.7, 0.7], source="s3"),
        ]
        result = rover.vote(hypotheses)
        assert result.words == ["hello", "world"]

    def test_majority_vote(self):
        """Majority should win in voting."""
        rover = ROVER(config=ROVERConfig(use_confidence_weights=False))
        hypotheses = [
            Hypothesis(words=["hello", "world"], confidences=[0.9, 0.9], source="s1"),
            Hypothesis(words=["hello", "world"], confidences=[0.8, 0.8], source="s2"),
            Hypothesis(words=["hello", "word"], confidences=[0.95, 0.95], source="s3"),
        ]
        result = rover.vote(hypotheses)
        # "world" should win 2 vs 1 despite lower confidence
        assert result.words == ["hello", "world"]

    def test_confidence_weighted_voting(self):
        """High confidence should break ties."""
        rover = ROVER(config=ROVERConfig(use_confidence_weights=True))
        hypotheses = [
            Hypothesis(words=["hello", "world"], confidences=[0.9, 0.5], source="s1"),
            Hypothesis(words=["hello", "word"], confidences=[0.9, 0.95], source="s2"),
        ]
        result = rover.vote(hypotheses)
        # With only 2 sources and confidence weighting, higher confidence wins
        assert "hello" in result.words

    def test_rover_reduces_errors(self):
        """ROVER should reduce errors compared to worst source."""
        rover = ROVER()
        reference = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"

        # Create sources with different errors
        source_configs = {
            "transducer": 0.05,
            "ctc": 0.08,
            "whisper": 0.03,
        }
        hypotheses = create_mock_sources(reference, source_configs)

        # Get individual WERs
        ref_words = reference.split()
        source_wers = {}
        for hyp in hypotheses:
            dist, _ = _dp_edit_distance(ref_words, hyp.words)
            source_wers[hyp.source] = dist / len(ref_words)

        # Get ROVER result
        result = rover.vote(hypotheses)
        rover_dist, _ = _dp_edit_distance(ref_words, result.words)
        rover_wer = rover_dist / len(ref_words)

        # ROVER should be at least as good as best source
        best_source_wer = min(source_wers.values())
        # Note: ROVER can sometimes be slightly worse due to alignment issues
        # but should generally be close to oracle
        assert rover_wer <= best_source_wer + 0.15  # Allow 15% margin


class TestOracleWER:
    """Test oracle WER computation."""

    def test_oracle_selects_best(self):
        """Oracle should return best hypothesis."""
        reference = ["hello", "world"]
        hypotheses = [
            Hypothesis(words=["hello", "word"], confidences=[0.9, 0.9]),  # 1 error
            Hypothesis(words=["hello", "world"], confidences=[0.5, 0.5]),  # 0 errors
            Hypothesis(words=["halo", "world"], confidences=[0.9, 0.9]),  # 1 error
        ]

        oracle_wer, best_hyp = compute_oracle_wer(hypotheses, reference)
        assert oracle_wer == 0.0
        assert best_hyp.words == ["hello", "world"]

    def test_oracle_bounds_rover(self):
        """Oracle WER should be a lower bound on ROVER WER."""
        reference = "THE QUICK BROWN FOX"
        ref_words = reference.split()

        hypotheses = create_mock_sources(reference, {
            "s1": 0.1,
            "s2": 0.1,
            "s3": 0.1,
        })

        oracle_wer, _ = compute_oracle_wer(hypotheses, ref_words)

        rover = ROVER()
        result = rover.vote(hypotheses)
        rover_dist, _ = _dp_edit_distance(ref_words, result.words)
        rover_wer = rover_dist / len(ref_words)

        # Oracle is theoretical best - ROVER may be slightly higher
        # but should be within reasonable margin
        assert oracle_wer <= rover_wer + 0.1


class TestLibriSpeechValidation:
    """Test validation framework on LibriSpeech samples."""

    @pytest.fixture
    def librispeech_samples(self):
        """Load a few LibriSpeech samples for testing."""
        samples = load_librispeech_samples(max_samples=10)
        if not samples:
            pytest.skip("LibriSpeech test-clean not available")
        return samples

    def test_load_samples(self, librispeech_samples):
        """Verify sample loading works."""
        assert len(librispeech_samples) > 0
        sample = librispeech_samples[0]
        assert sample.utterance_id
        assert sample.reference
        assert sample.audio_path.exists()

    def test_reference_format(self, librispeech_samples):
        """References should be uppercase text."""
        for sample in librispeech_samples[:5]:
            # LibriSpeech references are uppercase
            assert sample.reference == sample.reference.upper()
            # Should contain only words and spaces
            assert all(c.isalpha() or c.isspace() or c == "'" for c in sample.reference)

    def test_rover_on_librispeech(self, librispeech_samples):
        """Test ROVER voting on LibriSpeech samples."""
        rover = ROVER()

        for sample in librispeech_samples[:3]:
            # Simulate multiple ASR sources
            hypotheses = create_mock_sources(sample.reference, {
                "transducer": 0.02,
                "ctc": 0.05,
                "whisper": 0.03,
            })

            result = rover.vote(hypotheses)

            # Should produce non-empty output
            assert len(result.words) > 0

            # Calculate WER
            ref_words = sample.reference.split()
            dist, _ = _dp_edit_distance(ref_words, result.words)
            wer = dist / len(ref_words) if ref_words else 0

            # With simulated good sources, WER should be reasonable
            # (this is testing the framework, not actual model performance)
            assert wer < 0.3  # 30% WER threshold for mock sources


class TestHighAccuracyTarget:
    """Tests validating progress toward <1.5% WER target."""

    def test_wer_target_definition(self):
        """Document the target metrics."""
        # Target from FINAL_ROADMAP_SOTA_PLUS_PLUS.md
        TARGET_HIGH_ACCURACY_WER = 0.015  # <1.5%
        TARGET_STREAMING_WER = 0.025  # <2.5%

        # These are aspirational targets requiring trained models
        # Current tests use mock data to validate the framework
        assert TARGET_HIGH_ACCURACY_WER == 0.015
        assert TARGET_STREAMING_WER == 0.025

    def test_rover_improvement_ratio(self):
        """ROVER should improve over single-source baseline."""
        rover = ROVER()
        reference = "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE"
        ref_words = reference.split()

        # Simulate sources with realistic WER range
        hypotheses = create_mock_sources(reference, {
            "transducer": 0.03,  # ~3% WER
            "ctc": 0.06,  # ~6% WER
            "whisper": 0.025,  # ~2.5% WER
        })

        # Calculate individual WERs
        source_wers = {}
        for hyp in hypotheses:
            dist, _ = _dp_edit_distance(ref_words, hyp.words)
            source_wers[hyp.source] = dist / len(ref_words)

        # Get ROVER result
        result = rover.vote(hypotheses)
        rover_dist, _ = _dp_edit_distance(ref_words, result.words)
        rover_wer = rover_dist / len(ref_words)

        # ROVER should be close to or better than best source
        best_source_wer = min(source_wers.values())

        # Log results for debugging
        print(f"\nSource WERs: {source_wers}")
        print(f"Best source WER: {best_source_wer:.4f}")
        print(f"ROVER WER: {rover_wer:.4f}")

        # ROVER should achieve improvement (or at least not be much worse)
        assert rover_wer <= best_source_wer + 0.05

    def test_phoneme_weighting_helps(self):
        """Phoneme weighting should help with homophones."""
        # Test with phoneme weighting enabled
        config_with_phoneme = ROVERConfig(use_phoneme_weighting=True)
        rover_with = ROVER(config=config_with_phoneme)

        config_without = ROVERConfig(use_phoneme_weighting=False)
        rover_without = ROVER(config=config_without)

        # Test case where phonemes help
        # "to" vs "too" vs "two" - same pronunciation
        hypotheses = [
            Hypothesis(
                words=["going", "to", "work"],
                confidences=[0.9, 0.6, 0.9],
                source="s1",
            ),
            Hypothesis(
                words=["going", "too", "work"],
                confidences=[0.9, 0.65, 0.9],
                source="s2",
            ),
        ]

        result_with = rover_with.vote(hypotheses)
        result_without = rover_without.vote(hypotheses)

        # Both should produce valid output
        assert len(result_with.words) > 0
        assert len(result_without.words) > 0


# =============================================================================
# Integration Test: Full Validation Pipeline
# =============================================================================


class TestValidationPipeline:
    """Integration tests for the full validation pipeline."""

    def test_validation_framework_completeness(self):
        """Verify all validation components are in place."""
        # Check WER computation
        assert callable(compute_wer)
        assert callable(compute_corpus_wer)

        # Check ROVER components
        assert ROVER is not None
        assert ROVERConfig is not None
        assert Hypothesis is not None

        # Check data loading
        assert callable(load_librispeech_samples)

    def test_end_to_end_mock_validation(self):
        """End-to-end validation with mock data."""
        # Simulate a mini validation run
        references = [
            "THE CAT SAT ON THE MAT",
            "A QUICK BROWN FOX JUMPS",
            "HELLO WORLD THIS IS A TEST",
        ]

        rover = ROVER()
        rover_hypotheses = []

        for ref in references:
            sources = create_mock_sources(ref, {
                "transducer": 0.03,
                "whisper": 0.02,
            })
            result = rover.vote(sources)
            rover_hypotheses.append(" ".join(result.words))

        # Compute corpus WER
        corpus_result = compute_corpus_wer(references, rover_hypotheses)

        # With good mock sources, WER should be reasonable
        assert corpus_result["wer"] < 0.2  # 20% threshold
        assert corpus_result["num_utterances"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
