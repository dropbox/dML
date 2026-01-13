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
Tests for WhisperMLX streaming evaluation framework.

Tests:
1. WER/CER computation utilities
2. StreamingMetrics dataclass
3. AggregatedMetrics calculation
4. StreamingReplayHarness
5. StreamingEvaluator
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, ".")

from tools.whisper_mlx.streaming_eval import (
    AggregatedMetrics,
    PartialResult,
    StreamingEvaluator,
    StreamingMetrics,
    StreamingReplayHarness,
    compute_cer,
    compute_wer,
    levenshtein_distance,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestLevenshteinDistance:
    """Test Levenshtein distance computation."""

    def test_identical_sequences(self):
        """Test distance between identical sequences is 0."""
        ref = ["hello", "world"]
        hyp = ["hello", "world"]
        distance, subs, ins, dels = levenshtein_distance(ref, hyp)
        assert distance == 0
        assert subs == 0
        assert ins == 0
        assert dels == 0

    def test_single_substitution(self):
        """Test single word substitution."""
        ref = ["hello", "world"]
        hyp = ["hello", "there"]
        distance, subs, ins, dels = levenshtein_distance(ref, hyp)
        assert distance == 1
        assert subs == 1
        assert ins == 0
        assert dels == 0

    def test_single_insertion(self):
        """Test single word insertion."""
        ref = ["hello", "world"]
        hyp = ["hello", "big", "world"]
        distance, subs, ins, dels = levenshtein_distance(ref, hyp)
        assert distance == 1
        assert ins == 1

    def test_single_deletion(self):
        """Test single word deletion."""
        ref = ["hello", "big", "world"]
        hyp = ["hello", "world"]
        distance, subs, ins, dels = levenshtein_distance(ref, hyp)
        assert distance == 1
        assert dels == 1

    def test_empty_reference(self):
        """Test with empty reference."""
        ref = []
        hyp = ["hello", "world"]
        distance, subs, ins, dels = levenshtein_distance(ref, hyp)
        assert distance == 2
        assert ins == 2

    def test_empty_hypothesis(self):
        """Test with empty hypothesis."""
        ref = ["hello", "world"]
        hyp = []
        distance, subs, ins, dels = levenshtein_distance(ref, hyp)
        assert distance == 2
        assert dels == 2

    def test_both_empty(self):
        """Test with both empty."""
        ref = []
        hyp = []
        distance, subs, ins, dels = levenshtein_distance(ref, hyp)
        assert distance == 0

    def test_complex_edit_sequence(self):
        """Test complex sequence of edits."""
        ref = ["the", "quick", "brown", "fox"]
        hyp = ["a", "quick", "red", "fox", "jumps"]
        # 1 sub (the->a), 1 sub (brown->red), 1 ins (jumps)
        distance, subs, ins, dels = levenshtein_distance(ref, hyp)
        assert distance == 3


class TestComputeWER:
    """Test Word Error Rate computation."""

    def test_perfect_match(self):
        """Test WER is 0 for perfect match."""
        result = compute_wer("Hello world", "Hello world")
        assert result["wer"] == 0.0
        assert result["ref_words"] == 2
        assert result["hyp_words"] == 2

    def test_case_insensitive(self):
        """Test WER is case insensitive."""
        result = compute_wer("Hello World", "hello world")
        assert result["wer"] == 0.0

    def test_single_word_error(self):
        """Test WER with single word error."""
        result = compute_wer("Hello world", "Hello there")
        assert result["wer"] == 0.5  # 1 error / 2 words

    def test_empty_reference(self):
        """Test WER with empty reference."""
        result = compute_wer("", "hello")
        assert result["wer"] == 1.0

    def test_empty_hypothesis(self):
        """Test WER with empty hypothesis."""
        result = compute_wer("hello world", "")
        assert result["wer"] == 1.0

    def test_both_empty(self):
        """Test WER with both empty."""
        result = compute_wer("", "")
        assert result["wer"] == 0.0

    def test_wer_can_exceed_one(self):
        """Test WER can exceed 1.0 (many insertions)."""
        result = compute_wer("hello", "hello world how are you today")
        # 1 word correct, 5 insertions
        assert result["wer"] > 1.0


class TestComputeCER:
    """Test Character Error Rate computation."""

    def test_perfect_match(self):
        """Test CER is 0 for perfect match."""
        cer = compute_cer("Hello", "Hello")
        assert cer == 0.0

    def test_case_insensitive(self):
        """Test CER is case insensitive."""
        cer = compute_cer("Hello", "hello")
        assert cer == 0.0

    def test_single_char_error(self):
        """Test CER with single character error."""
        cer = compute_cer("hello", "hallo")
        assert cer == pytest.approx(0.2)  # 1 error / 5 chars

    def test_ignores_spaces(self):
        """Test CER ignores spaces."""
        cer = compute_cer("hello world", "helloworld")
        assert cer == 0.0

    def test_empty_reference(self):
        """Test CER with empty reference."""
        cer = compute_cer("", "hello")
        assert cer == 1.0

    def test_both_empty(self):
        """Test CER with both empty."""
        cer = compute_cer("", "")
        assert cer == 0.0


class TestPartialResult:
    """Test PartialResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = PartialResult(
            timestamp_ms=100.0,
            audio_time_ms=500.0,
            text="Hello",
            is_final=False,
            is_confirmed=False,
        )
        assert result.timestamp_ms == 100.0
        assert result.audio_time_ms == 500.0
        assert result.text == "Hello"
        assert result.is_final is False
        assert result.is_confirmed is False

    def test_final_result(self):
        """Test final result creation."""
        result = PartialResult(
            timestamp_ms=500.0,
            audio_time_ms=1000.0,
            text="Hello world",
            is_final=True,
            is_confirmed=True,
        )
        assert result.is_final is True
        assert result.is_confirmed is True


class TestStreamingMetrics:
    """Test StreamingMetrics dataclass."""

    def test_default_values(self):
        """Test default values are initialized correctly."""
        metrics = StreamingMetrics()
        assert metrics.sample_id == ""
        assert metrics.reference == ""
        assert metrics.final_hypothesis == ""
        assert metrics.wer == 0.0
        assert metrics.cer == 0.0
        assert metrics.first_partial_latency_ms == 0.0
        assert metrics.finalization_latency_ms == 0.0
        assert metrics.edit_count == 0
        assert metrics.edit_rate == 0.0
        assert metrics.committed_retractions == 0
        assert metrics.time_to_commit_ms == []
        assert metrics.rtf == 0.0
        assert metrics.partial_trace == []

    def test_custom_values(self):
        """Test custom values."""
        metrics = StreamingMetrics(
            sample_id="test_001",
            reference="Hello world",
            final_hypothesis="Hello there",
            wer=0.5,
            cer=0.2,
            first_partial_latency_ms=150.0,
            finalization_latency_ms=250.0,
            edit_count=3,
            edit_rate=1.5,
            rtf=0.3,
        )
        assert metrics.sample_id == "test_001"
        assert metrics.wer == 0.5
        assert metrics.rtf == 0.3


class TestAggregatedMetrics:
    """Test AggregatedMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        agg = AggregatedMetrics()
        assert agg.mean_wer == 0.0
        assert agg.mean_cer == 0.0
        assert agg.first_partial_latency_median == 0.0
        assert agg.first_partial_latency_p95 == 0.0
        assert agg.finalization_latency_median == 0.0
        assert agg.finalization_latency_p95 == 0.0
        assert agg.mean_edit_rate == 0.0
        assert agg.committed_retraction_rate == 0.0
        assert agg.mean_rtf == 0.0
        assert agg.num_samples == 0
        assert agg.num_failed == 0


class TestStreamingReplayHarness:
    """Test StreamingReplayHarness."""

    def test_init(self):
        """Test harness initialization."""
        mock_streamer = MagicMock()
        harness = StreamingReplayHarness(
            mock_streamer,
            chunk_duration_ms=100.0,
            speed_factor=10.0,
        )
        assert harness.streamer is mock_streamer
        assert harness.chunk_duration_ms == 100.0
        assert harness.speed_factor == 10.0

    def test_replay_audio_basic(self):
        """Test basic audio replay."""
        from tools.whisper_mlx.streaming import StreamingResult

        # Create mock streamer
        mock_streamer = MagicMock()
        mock_streamer.reset = MagicMock()

        # Create mock async transcribe_stream
        async def mock_transcribe_stream(audio_gen):
            yield StreamingResult(
                text="Hello world",
                is_final=True,
                is_partial=False,
                segment_start=0.0,
                segment_end=1.0,
            )

        mock_streamer.transcribe_stream = mock_transcribe_stream

        harness = StreamingReplayHarness(
            mock_streamer,
            chunk_duration_ms=100.0,
            speed_factor=100.0,  # Run fast
        )

        # Create test audio (1 second)
        audio = _rng.standard_normal(16000).astype(np.float32) * 0.3

        async def run():
            return await harness.replay_audio(audio, reference="Hello world")

        metrics = asyncio.run(run())

        assert metrics.audio_duration_ms == pytest.approx(1000.0, rel=0.01)
        assert metrics.final_hypothesis == "Hello world"
        assert metrics.wer == 0.0
        assert mock_streamer.reset.called

    def test_replay_audio_with_edits(self):
        """Test audio replay tracks edit count."""
        from tools.whisper_mlx.streaming import StreamingResult

        mock_streamer = MagicMock()
        mock_streamer.reset = MagicMock()

        # Simulate multiple partial results with edits
        async def mock_transcribe_stream(audio_gen):
            yield StreamingResult(
                text="Hello",
                is_final=False,
                is_partial=True,
                segment_start=0.0,
                segment_end=0.5,
            )
            yield StreamingResult(
                text="Hello wor",
                is_final=False,
                is_partial=True,
                segment_start=0.0,
                segment_end=0.7,
            )
            yield StreamingResult(
                text="Hello world",
                is_final=True,
                is_partial=False,
                segment_start=0.0,
                segment_end=1.0,
            )

        mock_streamer.transcribe_stream = mock_transcribe_stream

        harness = StreamingReplayHarness(mock_streamer, speed_factor=100.0)
        audio = _rng.standard_normal(16000).astype(np.float32)

        async def run():
            return await harness.replay_audio(audio, reference="Hello world")

        metrics = asyncio.run(run())

        # Should have 2 edits (Hello -> Hello wor, Hello wor -> Hello world)
        assert metrics.edit_count == 2
        assert metrics.wer == 0.0

    def test_replay_audio_calculates_rtf(self):
        """Test RTF calculation."""
        from tools.whisper_mlx.streaming import StreamingResult

        mock_streamer = MagicMock()
        mock_streamer.reset = MagicMock()

        async def mock_transcribe_stream(audio_gen):
            yield StreamingResult(
                text="Test",
                is_final=True,
                is_partial=False,
                segment_start=0.0,
                segment_end=1.0,
            )

        mock_streamer.transcribe_stream = mock_transcribe_stream

        harness = StreamingReplayHarness(mock_streamer, speed_factor=100.0)
        audio = _rng.standard_normal(16000).astype(np.float32)

        async def run():
            return await harness.replay_audio(audio)

        metrics = asyncio.run(run())

        # RTF should be calculated (processing_time / audio_duration)
        assert metrics.rtf >= 0.0
        assert metrics.total_processing_ms > 0


class TestStreamingEvaluator:
    """Test StreamingEvaluator."""

    def test_init_default_config(self):
        """Test evaluator initialization with default config."""
        mock_model = MagicMock()

        # Patch the StreamingWhisper to avoid VAD dependency
        with patch("tools.whisper_mlx.streaming.StreamingWhisper") as mock_streamer_cls:
            evaluator = StreamingEvaluator(mock_model)

            assert evaluator.model is mock_model
            assert evaluator.sample_metrics == []
            mock_streamer_cls.assert_called_once()

    def test_aggregate_metrics_empty(self):
        """Test aggregate metrics with no samples."""
        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper"):
            evaluator = StreamingEvaluator(mock_model)
            agg = evaluator.aggregate_metrics()

            assert agg.num_samples == 0

    def test_aggregate_metrics_single_sample(self):
        """Test aggregate metrics with single sample."""
        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper"):
            evaluator = StreamingEvaluator(mock_model)

            # Add a sample manually
            evaluator.sample_metrics.append(StreamingMetrics(
                sample_id="test",
                wer=0.1,
                cer=0.05,
                first_partial_latency_ms=150.0,
                finalization_latency_ms=200.0,
                edit_rate=0.5,
                rtf=0.3,
            ))

            agg = evaluator.aggregate_metrics()

            assert agg.num_samples == 1
            assert agg.mean_wer == 0.1
            assert agg.mean_cer == 0.05
            assert agg.first_partial_latency_median == 150.0
            assert agg.finalization_latency_median == 200.0
            assert agg.mean_edit_rate == 0.5
            assert agg.mean_rtf == 0.3

    def test_aggregate_metrics_multiple_samples(self):
        """Test aggregate metrics with multiple samples."""
        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper"):
            evaluator = StreamingEvaluator(mock_model)

            # Add multiple samples
            evaluator.sample_metrics.append(StreamingMetrics(
                sample_id="test1",
                wer=0.1,
                first_partial_latency_ms=100.0,
                rtf=0.2,
            ))
            evaluator.sample_metrics.append(StreamingMetrics(
                sample_id="test2",
                wer=0.2,
                first_partial_latency_ms=200.0,
                rtf=0.4,
            ))
            evaluator.sample_metrics.append(StreamingMetrics(
                sample_id="test3",
                wer=0.3,
                first_partial_latency_ms=300.0,
                rtf=0.6,
            ))

            agg = evaluator.aggregate_metrics()

            assert agg.num_samples == 3
            assert agg.mean_wer == pytest.approx(0.2, rel=0.01)
            assert agg.first_partial_latency_median == 200.0  # Middle value
            assert agg.mean_rtf == pytest.approx(0.4, rel=0.01)

    def test_aggregate_metrics_percentiles(self):
        """Test p95 percentile calculation."""
        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper"):
            evaluator = StreamingEvaluator(mock_model)

            # Add 100 samples with varying latencies
            for i in range(100):
                evaluator.sample_metrics.append(StreamingMetrics(
                    sample_id=f"test{i}",
                    wer=0.0,
                    first_partial_latency_ms=float(i + 1),  # 1 to 100
                    finalization_latency_ms=float(i + 1),
                ))

            agg = evaluator.aggregate_metrics()

            # Median should be around 50.5
            assert agg.first_partial_latency_median == pytest.approx(50.5, rel=0.01)
            # P95 should be around 95
            assert agg.first_partial_latency_p95 == pytest.approx(95.05, rel=0.02)

    def test_failed_sample_count(self):
        """Test failed sample counting."""
        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper"):
            evaluator = StreamingEvaluator(mock_model)

            # Add samples with various WER values
            evaluator.sample_metrics.append(StreamingMetrics(wer=0.1))
            evaluator.sample_metrics.append(StreamingMetrics(wer=0.5))
            evaluator.sample_metrics.append(StreamingMetrics(wer=1.0))  # Failed
            evaluator.sample_metrics.append(StreamingMetrics(wer=1.5))  # Failed

            agg = evaluator.aggregate_metrics()

            assert agg.num_samples == 4
            assert agg.num_failed == 2  # WER >= 1.0


class TestStreamingEvaluatorSaveResults:
    """Test StreamingEvaluator result saving."""

    def test_save_results_creates_file(self, tmp_path):
        """Test save_results creates JSON file."""
        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper"):
            evaluator = StreamingEvaluator(mock_model)

            # Add a sample
            evaluator.sample_metrics.append(StreamingMetrics(
                sample_id="test",
                wer=0.1,
                reference="Hello",
                final_hypothesis="Hello",
            ))

            output_path = tmp_path / "results.json"
            evaluator.save_results(str(output_path))

            assert output_path.exists()

            # Verify JSON structure
            import json
            with open(output_path) as f:
                data = json.load(f)

            assert "aggregated" in data
            assert "config" in data
            assert "samples" in data
            assert len(data["samples"]) == 1
            assert data["samples"][0]["sample_id"] == "test"

    def test_save_results_excludes_partial_trace(self, tmp_path):
        """Test partial_trace is excluded from saved results."""
        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper"):
            evaluator = StreamingEvaluator(mock_model)

            # Add a sample with partial trace
            metrics = StreamingMetrics(sample_id="test")
            metrics.partial_trace = [
                PartialResult(100.0, 500.0, "Hello", False, False),
                PartialResult(200.0, 1000.0, "Hello world", True, True),
            ]
            evaluator.sample_metrics.append(metrics)

            output_path = tmp_path / "results.json"
            evaluator.save_results(str(output_path))

            import json
            with open(output_path) as f:
                data = json.load(f)

            # partial_trace should be excluded
            assert "partial_trace" not in data["samples"][0]


class TestStreamingEvaluatorPrintSummary:
    """Test StreamingEvaluator summary printing."""

    def test_print_summary_no_error(self, capsys):
        """Test print_summary runs without error."""
        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper"):
            evaluator = StreamingEvaluator(mock_model)

            # Add samples
            evaluator.sample_metrics.append(StreamingMetrics(
                wer=0.1,
                cer=0.05,
                first_partial_latency_ms=150.0,
                finalization_latency_ms=200.0,
                edit_rate=0.5,
                rtf=0.3,
            ))

            # Should not raise
            evaluator.print_summary()

            captured = capsys.readouterr()
            assert "STREAMING EVALUATION RESULTS" in captured.out
            assert "Mean WER" in captured.out
            assert "Mean RTF" in captured.out


class TestStreamingEvaluatorIntegration:
    """Integration tests for StreamingEvaluator."""

    def test_evaluate_sample_mocked(self):
        """Test evaluate_sample with mocked streaming."""
        from tools.whisper_mlx.streaming import StreamingResult

        mock_model = MagicMock()

        with patch("tools.whisper_mlx.streaming.StreamingWhisper") as mock_streamer_cls:
            # Setup mock streamer
            mock_streamer = MagicMock()
            mock_streamer.reset = MagicMock()

            async def mock_transcribe_stream(audio_gen):
                yield StreamingResult(
                    text="Hello world",
                    is_final=True,
                    is_partial=False,
                    segment_start=0.0,
                    segment_end=1.0,
                )

            mock_streamer.transcribe_stream = mock_transcribe_stream
            mock_streamer_cls.return_value = mock_streamer

            evaluator = StreamingEvaluator(mock_model)

            # Test audio
            audio = _rng.standard_normal(16000).astype(np.float32)

            async def run():
                return await evaluator.evaluate_sample(
                    audio,
                    reference="Hello world",
                    sample_id="test_001",
                )

            metrics = asyncio.run(run())

            assert metrics.sample_id == "test_001"
            assert metrics.final_hypothesis == "Hello world"
            assert metrics.wer == 0.0
            assert len(evaluator.sample_metrics) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
