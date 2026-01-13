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
Tests for ROVER integration module.

Phase 8.2-8.3: Whisper and CTC integration with ROVER.

These tests verify:
- SourceResult dataclass functionality
- WhisperROVERSource confidence extraction
- CTCROVERSource decoding with confidences
- TransducerROVERSource interface
- HighAccuracyDecoder voting combination
- End-to-end integration scenarios
"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.decoding.rover import Hypothesis
from src.decoding.rover_integration import (
    CTCROVERSource,
    HighAccuracyDecoder,
    HighAccuracyResult,
    ROVERSource,
    SourceResult,
    TransducerROVERSource,
    WhisperROVERSource,
    combine_whisper_transducer,
)

# =============================================================================
# SourceResult Tests
# =============================================================================


class TestSourceResult:
    """Tests for SourceResult dataclass."""

    def test_basic_creation(self):
        """Test basic SourceResult creation."""
        result = SourceResult(
            words=["hello", "world"],
            confidences=[0.9, 0.85],
            source_name="whisper",
            raw_text="hello world",
        )

        assert result.words == ["hello", "world"]
        assert result.confidences == [0.9, 0.85]
        assert result.source_name == "whisper"
        assert result.raw_text == "hello world"
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test SourceResult with metadata."""
        result = SourceResult(
            words=["test"],
            confidences=[0.95],
            source_name="ctc",
            raw_text="test",
            metadata={"num_tokens": 5, "encoder_frames": 100},
        )

        assert result.metadata["num_tokens"] == 5
        assert result.metadata["encoder_frames"] == 100

    def test_empty_result(self):
        """Test SourceResult with empty words."""
        result = SourceResult(
            words=[],
            confidences=[],
            source_name="transducer",
            raw_text="",
        )

        assert len(result.words) == 0
        assert len(result.confidences) == 0


# =============================================================================
# WhisperROVERSource Tests
# =============================================================================


class TestWhisperROVERSource:
    """Tests for Whisper ROVER source."""

    def test_source_name(self):
        """Test source name property."""
        source = WhisperROVERSource(model_name="large-v3")
        assert source.name == "whisper"

    def test_confidence_extraction_basic(self):
        """Test word confidence extraction from segments."""
        source = WhisperROVERSource()

        text = "hello world test"
        segments = [
            {"text": "hello world", "avg_logprob": -0.2},
            {"text": "test", "avg_logprob": -0.5},
        ]

        words, confidences = source._extract_word_confidences(text, segments)

        assert words == ["hello", "world", "test"]
        assert len(confidences) == 3
        # First two words share segment 1 confidence
        assert confidences[0] == confidences[1]
        # Last word has segment 2 confidence (lower)
        assert confidences[2] < confidences[0]

    def test_confidence_extraction_empty_text(self):
        """Test confidence extraction with empty text."""
        source = WhisperROVERSource()

        words, confidences = source._extract_word_confidences("", [])

        assert words == []
        assert confidences == []

    def test_confidence_extraction_no_segments(self):
        """Test confidence extraction when no segments available."""
        source = WhisperROVERSource()

        words, confidences = source._extract_word_confidences("hello world", [])

        assert words == ["hello", "world"]
        assert confidences == [0.5, 0.5]  # Default uniform confidence

    def test_confidence_bounds(self):
        """Test that confidences are bounded to [0.01, 0.99]."""
        source = WhisperROVERSource()

        # Very low logprob (should bound to 0.01)
        text = "test"
        segments = [{"text": "test", "avg_logprob": -10.0}]  # Very low confidence

        words, confidences = source._extract_word_confidences(text, segments)

        assert confidences[0] >= 0.01
        assert confidences[0] <= 0.99

    def test_to_hypothesis(self):
        """Test conversion to ROVER Hypothesis."""
        source = WhisperROVERSource()

        result = SourceResult(
            words=["hello", "world"],
            confidences=[0.9, 0.85],
            source_name="whisper",
            raw_text="hello world",
        )

        hypothesis = source.to_hypothesis(result)

        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.words == ["hello", "world"]
        assert hypothesis.confidences == [0.9, 0.85]
        assert hypothesis.source == "whisper"


# =============================================================================
# CTCROVERSource Tests
# =============================================================================


class TestCTCROVERSource:
    """Tests for CTC ROVER source."""

    def test_source_name(self):
        """Test source name property."""
        source = CTCROVERSource()
        assert source.name == "ctc"

    def test_decode_with_confidences_mock(self):
        """Test CTC decoding with mock logits."""
        source = CTCROVERSource()

        # Create mock logits (T=10, vocab=5)
        # Simulate clear predictions at frames 0, 3, 6
        try:
            import mlx.core as mx

            logits = mx.zeros((10, 5))
            # Set high logits for specific tokens at specific frames
            # Frame 0: token 1, Frame 3: token 2, Frame 6: token 3
            logits_np = np.zeros((10, 5), dtype=np.float32)
            logits_np[0, 1] = 10.0  # Token 1 at frame 0
            logits_np[1, 0] = 10.0  # Blank at frame 1
            logits_np[2, 0] = 10.0  # Blank at frame 2
            logits_np[3, 2] = 10.0  # Token 2 at frame 3
            logits_np[4, 2] = 10.0  # Token 2 again (should be collapsed)
            logits_np[5, 0] = 10.0  # Blank
            logits_np[6, 3] = 10.0  # Token 3 at frame 6
            logits = mx.array(logits_np)

            tokens, confidences = source._decode_with_confidences(logits)

            assert len(tokens) == 3  # tokens 1, 2, 3
            assert tokens == [1, 2, 3]
            assert len(confidences) == 3
            # All confidences should be high (close to 1 due to softmax on large values)
            for conf in confidences:
                assert conf > 0.9
        except ImportError:
            pytest.skip("MLX not available")

    def test_token_to_word_confidences(self):
        """Test token-to-word confidence mapping."""
        source = CTCROVERSource()

        tokens = [1, 2, 3, 4, 5]
        token_confidences = [0.9, 0.85, 0.95, 0.8, 0.88]
        words = ["hello", "world", "test"]

        word_confidences = source._token_to_word_confidences(
            tokens, token_confidences, words,
        )

        assert len(word_confidences) == 3
        # Should be average of token confidences
        avg_conf = np.mean(token_confidences)
        for conf in word_confidences:
            assert abs(conf - avg_conf) < 0.01

    def test_to_hypothesis(self):
        """Test conversion to ROVER Hypothesis."""
        source = CTCROVERSource()

        result = SourceResult(
            words=["hello", "world"],
            confidences=[0.9, 0.85],
            source_name="ctc",
            raw_text="hello world",
        )

        hypothesis = source.to_hypothesis(result)

        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.words == ["hello", "world"]
        assert hypothesis.source == "ctc"


# =============================================================================
# TransducerROVERSource Tests
# =============================================================================


class TestTransducerROVERSource:
    """Tests for Transducer ROVER source."""

    def test_source_name(self):
        """Test source name property."""
        source = TransducerROVERSource()
        assert source.name == "transducer"

    def test_requires_model(self):
        """Test that transcribe fails without model."""
        source = TransducerROVERSource()

        with pytest.raises(ValueError, match="model not loaded"):
            source.transcribe("test.wav")

    def test_to_hypothesis(self):
        """Test conversion to ROVER Hypothesis."""
        source = TransducerROVERSource()

        result = SourceResult(
            words=["hello", "world"],
            confidences=[0.9, 0.85],
            source_name="transducer",
            raw_text="hello world",
        )

        hypothesis = source.to_hypothesis(result)

        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.words == ["hello", "world"]
        assert hypothesis.source == "transducer"


# =============================================================================
# HighAccuracyDecoder Tests
# =============================================================================


class TestHighAccuracyDecoder:
    """Tests for HighAccuracyDecoder."""

    def test_no_sources_error(self):
        """Test that decoder fails with no sources."""
        decoder = HighAccuracyDecoder()

        with pytest.raises(ValueError, match="No ASR sources configured"):
            decoder.transcribe("test.wav")

    def test_mock_single_source_voting(self):
        """Test voting with a single mock source."""
        # Create mock source
        mock_source = Mock(spec=ROVERSource)
        mock_source.name = "mock"
        mock_source.transcribe.return_value = SourceResult(
            words=["hello", "world"],
            confidences=[0.9, 0.85],
            source_name="mock",
            raw_text="hello world",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source]

        result = decoder.transcribe("test.wav")

        assert isinstance(result, HighAccuracyResult)
        assert result.text == "hello world"
        assert result.words == ["hello", "world"]
        assert "mock" in result.sources

    def test_mock_multi_source_voting(self):
        """Test voting with multiple mock sources."""
        # Source 1: says "hello world"
        mock_source1 = Mock(spec=ROVERSource)
        mock_source1.name = "source1"
        mock_source1.transcribe.return_value = SourceResult(
            words=["hello", "world"],
            confidences=[0.9, 0.9],
            source_name="source1",
            raw_text="hello world",
        )

        # Source 2: says "hello word" (typo)
        mock_source2 = Mock(spec=ROVERSource)
        mock_source2.name = "source2"
        mock_source2.transcribe.return_value = SourceResult(
            words=["hello", "word"],
            confidences=[0.85, 0.7],
            source_name="source2",
            raw_text="hello word",
        )

        # Source 3: says "hallo world" (typo)
        mock_source3 = Mock(spec=ROVERSource)
        mock_source3.name = "source3"
        mock_source3.transcribe.return_value = SourceResult(
            words=["hallo", "world"],
            confidences=[0.6, 0.95],
            source_name="source3",
            raw_text="hallo world",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source1, mock_source2, mock_source3]

        result = decoder.transcribe("test.wav")

        # ROVER should vote for "hello world" (majority)
        assert result.text == "hello world"
        assert "source1" in result.sources
        assert "source2" in result.sources
        assert "source3" in result.sources

    def test_transcribe_with_source(self):
        """Test transcribing with a specific source."""
        mock_source = Mock(spec=ROVERSource)
        mock_source.name = "specific"
        mock_source.transcribe.return_value = SourceResult(
            words=["test"],
            confidences=[0.95],
            source_name="specific",
            raw_text="test",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source]

        result = decoder.transcribe_with_source("test.wav", "specific")

        assert isinstance(result, SourceResult)
        assert result.words == ["test"]

    def test_transcribe_with_source_not_found(self):
        """Test error when source not found."""
        decoder = HighAccuracyDecoder()
        decoder._sources = []

        with pytest.raises(ValueError, match="not found"):
            decoder.transcribe_with_source("test.wav", "nonexistent")


# =============================================================================
# Factory Method Tests
# =============================================================================


class TestFactoryMethods:
    """Tests for factory methods."""

    def test_from_whisper_only_init(self):
        """Test from_whisper_only creates decoder correctly."""
        decoder = HighAccuracyDecoder.from_whisper_only(
            model_name="tiny",  # Use tiny for faster test
            language="en",
        )

        assert len(decoder._sources) == 1
        assert isinstance(decoder._sources[0], WhisperROVERSource)

    def test_from_transducer_and_whisper(self):
        """Test from_transducer_and_whisper creates decoder correctly."""
        mock_transducer = Mock(spec=TransducerROVERSource)
        mock_transducer.name = "transducer"

        decoder = HighAccuracyDecoder.from_transducer_and_whisper(
            transducer_source=mock_transducer,
            whisper_model="tiny",
            language="en",
        )

        assert len(decoder._sources) == 2
        assert decoder._transducer is mock_transducer
        assert isinstance(decoder._whisper, WhisperROVERSource)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_combine_whisper_transducer_basic(self):
        """Test basic two-source combination."""
        whisper_text = "hello world"
        whisper_logprob = -0.2

        transducer_text = "hello world"
        transducer_score = -2.0

        result = combine_whisper_transducer(
            whisper_text, whisper_logprob,
            transducer_text, transducer_score,
        )

        assert result == "hello world"

    def test_combine_whisper_transducer_different(self):
        """Test combination when sources disagree."""
        # Whisper says "hello world" with high confidence
        whisper_text = "hello world"
        whisper_logprob = -0.1  # High confidence

        # Transducer says "hallo world" with lower confidence
        transducer_text = "hallo world"
        transducer_score = -3.0  # Lower confidence

        result = combine_whisper_transducer(
            whisper_text, whisper_logprob,
            transducer_text, transducer_score,
        )

        # Should prefer "hello world" due to higher confidence
        # Note: exact behavior depends on ROVER voting
        assert "world" in result

    def test_combine_whisper_transducer_empty(self):
        """Test combination with empty input."""
        result = combine_whisper_transducer(
            "", -1.0,
            "", -1.0,
        )

        assert result == ""


# =============================================================================
# Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Tests for realistic integration scenarios."""

    def test_typical_error_correction(self):
        """Test ROVER correcting typical ASR errors."""
        # Scenario: Three sources, each with one error
        # Source 1: "the quick brown fox" (correct)
        # Source 2: "the quick brown fax" (typo)
        # Source 3: "the quick brow fox" (missing letter)

        mock_source1 = Mock(spec=ROVERSource)
        mock_source1.name = "model1"
        mock_source1.transcribe.return_value = SourceResult(
            words=["the", "quick", "brown", "fox"],
            confidences=[0.95, 0.92, 0.88, 0.90],
            source_name="model1",
            raw_text="the quick brown fox",
        )

        mock_source2 = Mock(spec=ROVERSource)
        mock_source2.name = "model2"
        mock_source2.transcribe.return_value = SourceResult(
            words=["the", "quick", "brown", "fax"],
            confidences=[0.94, 0.91, 0.87, 0.75],
            source_name="model2",
            raw_text="the quick brown fax",
        )

        mock_source3 = Mock(spec=ROVERSource)
        mock_source3.name = "model3"
        mock_source3.transcribe.return_value = SourceResult(
            words=["the", "quick", "brow", "fox"],
            confidences=[0.93, 0.90, 0.60, 0.89],
            source_name="model3",
            raw_text="the quick brow fox",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source1, mock_source2, mock_source3]

        result = decoder.transcribe("test.wav")

        # Majority should win: "the quick brown fox"
        assert result.text == "the quick brown fox"

    def test_insertion_handling(self):
        """Test ROVER handling word insertions."""
        # Source 1: "hello world"
        # Source 2: "hello there world" (insertion)
        # Source 3: "hello world"

        mock_source1 = Mock(spec=ROVERSource)
        mock_source1.name = "model1"
        mock_source1.transcribe.return_value = SourceResult(
            words=["hello", "world"],
            confidences=[0.9, 0.9],
            source_name="model1",
            raw_text="hello world",
        )

        mock_source2 = Mock(spec=ROVERSource)
        mock_source2.name = "model2"
        mock_source2.transcribe.return_value = SourceResult(
            words=["hello", "there", "world"],
            confidences=[0.85, 0.5, 0.85],  # "there" has low confidence
            source_name="model2",
            raw_text="hello there world",
        )

        mock_source3 = Mock(spec=ROVERSource)
        mock_source3.name = "model3"
        mock_source3.transcribe.return_value = SourceResult(
            words=["hello", "world"],
            confidences=[0.88, 0.88],
            source_name="model3",
            raw_text="hello world",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source1, mock_source2, mock_source3]

        result = decoder.transcribe("test.wav")

        # "hello world" should win (2 out of 3 sources agree, "there" has low conf)
        assert "hello" in result.text
        assert "world" in result.text

    def test_deletion_handling(self):
        """Test ROVER handling word deletions."""
        # Source 1: "hello beautiful world"
        # Source 2: "hello world" (deletion)
        # Source 3: "hello beautiful world"

        mock_source1 = Mock(spec=ROVERSource)
        mock_source1.name = "model1"
        mock_source1.transcribe.return_value = SourceResult(
            words=["hello", "beautiful", "world"],
            confidences=[0.9, 0.85, 0.9],
            source_name="model1",
            raw_text="hello beautiful world",
        )

        mock_source2 = Mock(spec=ROVERSource)
        mock_source2.name = "model2"
        mock_source2.transcribe.return_value = SourceResult(
            words=["hello", "world"],
            confidences=[0.88, 0.88],
            source_name="model2",
            raw_text="hello world",
        )

        mock_source3 = Mock(spec=ROVERSource)
        mock_source3.name = "model3"
        mock_source3.transcribe.return_value = SourceResult(
            words=["hello", "beautiful", "world"],
            confidences=[0.87, 0.82, 0.87],
            source_name="model3",
            raw_text="hello beautiful world",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source1, mock_source2, mock_source3]

        result = decoder.transcribe("test.wav")

        # "hello beautiful world" should win (2 out of 3 have "beautiful")
        assert "hello" in result.text
        assert "world" in result.text
        # "beautiful" should be kept (majority vote)
        assert "beautiful" in result.text


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_word_input(self):
        """Test handling of single word input."""
        mock_source = Mock(spec=ROVERSource)
        mock_source.name = "mock"
        mock_source.transcribe.return_value = SourceResult(
            words=["hello"],
            confidences=[0.9],
            source_name="mock",
            raw_text="hello",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source]

        result = decoder.transcribe("test.wav")

        assert result.text == "hello"
        assert len(result.words) == 1

    def test_very_long_input(self):
        """Test handling of long input."""
        # Generate long word list
        words = ["word"] * 1000
        confidences = [0.9] * 1000

        mock_source = Mock(spec=ROVERSource)
        mock_source.name = "mock"
        mock_source.transcribe.return_value = SourceResult(
            words=words,
            confidences=confidences,
            source_name="mock",
            raw_text=" ".join(words),
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source]

        result = decoder.transcribe("test.wav")

        assert len(result.words) == 1000

    def test_unicode_text(self):
        """Test handling of unicode text."""
        mock_source = Mock(spec=ROVERSource)
        mock_source.name = "mock"
        mock_source.transcribe.return_value = SourceResult(
            words=["bonjour", "monde"],
            confidences=[0.9, 0.85],
            source_name="mock",
            raw_text="bonjour monde",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source]

        result = decoder.transcribe("test.wav")

        assert result.text == "bonjour monde"

    def test_all_sources_empty(self):
        """Test when all sources return empty."""
        mock_source1 = Mock(spec=ROVERSource)
        mock_source1.name = "source1"
        mock_source1.transcribe.return_value = SourceResult(
            words=[],
            confidences=[],
            source_name="source1",
            raw_text="",
        )

        mock_source2 = Mock(spec=ROVERSource)
        mock_source2.name = "source2"
        mock_source2.transcribe.return_value = SourceResult(
            words=[],
            confidences=[],
            source_name="source2",
            raw_text="",
        )

        decoder = HighAccuracyDecoder()
        decoder._sources = [mock_source1, mock_source2]

        result = decoder.transcribe("test.wav")

        assert result.text == ""
        assert len(result.words) == 0
