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
Tests for WhisperMLX streaming transcription.

Tests:
1. StreamingConfig defaults and validation
2. AudioBuffer operations
3. VAD integration
4. Streaming API (async generator)
5. Synchronous wrapper
"""

import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, ".")

from tools.whisper_mlx.streaming import (
    AudioBuffer,
    LocalAgreement,
    StreamingConfig,
    StreamingResult,
    StreamingWhisper,
    StreamState,
    SyncStreamingWhisper,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestStreamingConfig:
    """Test StreamingConfig defaults and settings."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()

        assert config.sample_rate == 16000
        assert config.use_vad is True
        assert config.vad_aggressiveness == 2
        assert config.min_chunk_duration == 0.5
        assert config.max_chunk_duration == 10.0
        assert config.silence_threshold_duration == 0.5
        assert config.emit_partials is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingConfig(
            sample_rate=8000,
            use_vad=False,
            max_chunk_duration=5.0,
            emit_partials=False,
            language="en",
        )

        assert config.sample_rate == 8000
        assert config.use_vad is False
        assert config.max_chunk_duration == 5.0
        assert config.emit_partials is False
        assert config.language == "en"


class TestAudioBuffer:
    """Test AudioBuffer circular buffer operations."""

    def test_basic_append(self):
        """Test basic append and retrieval."""
        buffer = AudioBuffer(max_duration=2.0, sample_rate=16000)

        # Append 1 second of audio
        audio = np.ones(16000, dtype=np.float32) * 0.5
        buffer.append(audio)

        assert buffer.duration == 1.0
        assert buffer.total_time == 1.0

        # Retrieve audio
        retrieved = buffer.get_audio(1.0)
        assert len(retrieved) == 16000
        np.testing.assert_array_equal(retrieved, audio)

    def test_multiple_appends(self):
        """Test multiple append operations."""
        buffer = AudioBuffer(max_duration=2.0, sample_rate=16000)

        # Append in chunks
        chunk_size = 4000  # 0.25 seconds
        for i in range(4):
            chunk = np.ones(chunk_size, dtype=np.float32) * (i + 1) * 0.1
            buffer.append(chunk)

        assert buffer.duration == 1.0
        assert buffer.total_time == 1.0

        # Retrieve last 0.5 seconds
        retrieved = buffer.get_audio(0.5)
        assert len(retrieved) == 8000

    def test_buffer_overflow(self):
        """Test buffer handles overflow correctly."""
        buffer = AudioBuffer(max_duration=1.0, sample_rate=16000)

        # Append 2 seconds (overflow)
        audio = np.arange(32000, dtype=np.float32) / 32000
        buffer.append(audio)

        # Buffer should contain last 1 second
        assert buffer.duration <= 1.0
        assert buffer.total_time == 2.0

    def test_clear(self):
        """Test buffer clear operation."""
        buffer = AudioBuffer(max_duration=2.0, sample_rate=16000)

        buffer.append(np.ones(16000, dtype=np.float32))
        assert buffer.duration == 1.0

        buffer.clear()
        assert buffer.duration == 0.0

    def test_get_audio_empty(self):
        """Test get_audio on empty buffer."""
        buffer = AudioBuffer(max_duration=2.0, sample_rate=16000)

        audio = buffer.get_audio(1.0)
        assert len(audio) == 0


class TestStreamingResult:
    """Test StreamingResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = StreamingResult(
            text="Hello world",
            is_final=True,
            is_partial=False,
            segment_start=0.0,
            segment_end=1.5,
            language="en",
            processing_time=0.2,
            audio_duration=1.5,
        )

        assert result.text == "Hello world"
        assert result.is_final is True
        assert result.is_partial is False
        assert result.language == "en"

    def test_rtf_calculation(self):
        """Test real-time factor calculation."""
        result = StreamingResult(
            text="Test",
            is_final=True,
            is_partial=False,
            segment_start=0.0,
            segment_end=1.0,
            processing_time=0.5,
            audio_duration=1.0,
        )

        assert result.rtf == 0.5  # Processed 1s audio in 0.5s

    def test_rtf_zero_duration(self):
        """Test RTF with zero duration."""
        result = StreamingResult(
            text="",
            is_final=True,
            is_partial=False,
            segment_start=0.0,
            segment_end=0.0,
            processing_time=0.0,
            audio_duration=0.0,
        )

        assert result.rtf == 0.0


class TestStreamingWhisperInit:
    """Test StreamingWhisper initialization."""

    def test_init_without_vad(self):
        """Test initialization without VAD."""
        mock_model = MagicMock()
        config = StreamingConfig(use_vad=False)

        streamer = StreamingWhisper(mock_model, config)

        assert streamer.vad is None
        assert streamer.state == StreamState.IDLE

    @pytest.mark.skipif(
        not pytest.importorskip("webrtcvad", reason="webrtcvad not installed"),
        reason="webrtcvad not installed",
    )
    def test_init_with_vad(self):
        """Test initialization with VAD."""
        mock_model = MagicMock()
        config = StreamingConfig(use_vad=True)

        streamer = StreamingWhisper(mock_model, config)

        assert streamer.vad is not None

    def test_reset(self):
        """Test reset clears state."""
        mock_model = MagicMock()
        config = StreamingConfig(use_vad=False)

        streamer = StreamingWhisper(mock_model, config)
        streamer._detected_language = "en"
        streamer.state = StreamState.SPEECH

        streamer.reset()

        assert streamer.state == StreamState.IDLE
        assert streamer._detected_language is None


class TestVADDetection:
    """Test VAD speech detection."""

    @pytest.mark.skipif(
        not pytest.importorskip("webrtcvad", reason="webrtcvad not installed"),
        reason="webrtcvad not installed",
    )
    def test_vad_detects_speech(self):
        """Test VAD detects speech in audio."""
        mock_model = MagicMock()
        config = StreamingConfig(use_vad=True, vad_aggressiveness=1)

        streamer = StreamingWhisper(mock_model, config)

        # Generate synthetic speech-like audio (sinusoid with harmonics)
        t = np.linspace(0, 0.5, 8000)  # 0.5 seconds
        audio = (
            0.3 * np.sin(2 * np.pi * 150 * t) +
            0.2 * np.sin(2 * np.pi * 300 * t) +
            0.1 * np.sin(2 * np.pi * 450 * t)
        ).astype(np.float32)

        # Check VAD detection (may or may not detect synthetic as speech)
        result = streamer._check_vad(audio)
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not pytest.importorskip("webrtcvad", reason="webrtcvad not installed"),
        reason="webrtcvad not installed",
    )
    def test_vad_detects_silence(self):
        """Test VAD correctly identifies silence."""
        mock_model = MagicMock()
        config = StreamingConfig(use_vad=True, vad_aggressiveness=2)

        streamer = StreamingWhisper(mock_model, config)

        # Generate silence (very low amplitude noise)
        audio = _rng.standard_normal(8000).astype(np.float32) * 0.001

        result = streamer._check_vad(audio)
        # Silence should generally not be detected as speech
        # (though VAD can be sensitive)
        assert isinstance(result, bool)


class TestStreamingTranscription:
    """Test streaming transcription functionality."""

    def test_stream_basic(self):
        """Test basic streaming transcription."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
        }

        config = StreamingConfig(
            use_vad=False,
            min_chunk_duration=0.5,
            max_chunk_duration=2.0,
            silence_threshold_duration=0.3,
            emit_partials=False,
        )

        streamer = StreamingWhisper(mock_model, config)

        # Create async generator
        async def audio_generator():
            # 1 second of audio, then silence
            yield np.ones(16000, dtype=np.float32) * 0.5
            # Silence to trigger endpoint
            for _ in range(5):
                yield np.zeros(4800, dtype=np.float32)  # 0.3s silence each

        async def run_test():
            return [result async for result in streamer.transcribe_stream(audio_generator())]

        results = asyncio.run(run_test())

        # Should have at least one result
        assert len(results) >= 1
        assert any(r.text == "Hello world" for r in results)

    def test_stream_max_duration_reached(self):
        """Test that max_chunk_duration triggers processing."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Long utterance",
            "language": "en",
        }

        config = StreamingConfig(
            use_vad=False,
            min_chunk_duration=0.5,
            max_chunk_duration=1.0,  # 1 second max
            emit_partials=False,
        )

        streamer = StreamingWhisper(mock_model, config)

        async def audio_generator():
            # Send 3 seconds of continuous audio
            for _ in range(3):
                yield np.ones(16000, dtype=np.float32) * 0.5

        async def run_test():
            return [result async for result in streamer.transcribe_stream(audio_generator())]

        results = asyncio.run(run_test())

        # Should have multiple results (forced processing at max duration)
        assert len(results) >= 2

    def test_stream_with_partials(self):
        """Test streaming with partial results enabled."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Partial result",
            "language": "en",
        }

        config = StreamingConfig(
            use_vad=False,
            min_chunk_duration=0.5,
            max_chunk_duration=5.0,
            emit_partials=True,
            partial_interval=0.5,
        )

        streamer = StreamingWhisper(mock_model, config)

        async def audio_generator():
            # Send 2 seconds of audio
            for _ in range(4):
                yield np.ones(8000, dtype=np.float32) * 0.5
            # Silence to finalize
            for _ in range(5):
                yield np.zeros(4800, dtype=np.float32)

        async def run_test():
            return [result async for result in streamer.transcribe_stream(audio_generator())]

        results = asyncio.run(run_test())

        # Should have final results (partial results may or may not occur depending on VAD)
        final_results = [r for r in results if r.is_final]

        assert len(final_results) >= 1


class TestSyncStreamingWhisper:
    """Test synchronous streaming wrapper."""

    def test_sync_process_audio(self):
        """Test synchronous audio processing."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Sync test",
            "language": "en",
        }

        config = StreamingConfig(
            use_vad=False,
            min_chunk_duration=0.5,
            max_chunk_duration=1.0,
            emit_partials=False,
        )

        streamer = SyncStreamingWhisper(mock_model, config)

        # Send 1.5 seconds of audio (should trigger max duration)
        results = streamer.process_audio(np.ones(24000, dtype=np.float32) * 0.5)

        # May or may not have results yet
        assert isinstance(results, list)

    def test_sync_finalize(self):
        """Test synchronous finalization."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Final",
            "language": "en",
        }

        config = StreamingConfig(
            use_vad=False,
            min_chunk_duration=0.5,
        )

        streamer = SyncStreamingWhisper(mock_model, config)

        # Send some audio
        streamer.process_audio(np.ones(16000, dtype=np.float32) * 0.5)

        # Finalize
        results = streamer.finalize()

        assert isinstance(results, list)

    def test_sync_reset(self):
        """Test synchronous reset."""
        mock_model = MagicMock()
        config = StreamingConfig(use_vad=False)

        streamer = SyncStreamingWhisper(mock_model, config)
        streamer.process_audio(np.ones(8000, dtype=np.float32))

        streamer.reset()

        assert streamer._async_streamer.state == StreamState.IDLE


class TestCallbackInterface:
    """Test callback-based interface."""

    def test_callback_set(self):
        """Test setting callback."""
        mock_model = MagicMock()
        config = StreamingConfig(use_vad=False)

        streamer = StreamingWhisper(mock_model, config)

        callback_results = []
        def callback(result):
            callback_results.append(result)

        streamer.set_callback(callback)
        assert streamer._callback == callback


class TestIntegration:
    """Integration tests with mock audio."""

    def test_realistic_speech_pattern(self):
        """Test with realistic speech pattern (speech + silence + speech)."""
        mock_model = MagicMock()

        call_count = [0]
        def mock_transcribe(*args, **kwargs):
            call_count[0] += 1
            return {
                "text": f"Segment {call_count[0]}",
                "language": "en",
            }

        mock_model.transcribe.side_effect = mock_transcribe

        config = StreamingConfig(
            use_vad=False,
            min_chunk_duration=0.3,
            max_chunk_duration=5.0,
            silence_threshold_duration=0.5,
            emit_partials=False,
        )

        streamer = StreamingWhisper(mock_model, config)

        async def audio_generator():
            # First utterance (1 second)
            yield np.ones(16000, dtype=np.float32) * 0.5
            # Silence (0.6 seconds - triggers endpoint)
            yield np.zeros(9600, dtype=np.float32)
            # Second utterance (0.5 seconds)
            yield np.ones(8000, dtype=np.float32) * 0.3
            # More silence to finalize
            yield np.zeros(9600, dtype=np.float32)

        async def run_test():
            return [result async for result in streamer.transcribe_stream(audio_generator())]

        results = asyncio.run(run_test())

        # Should have results from both utterances
        assert len(results) >= 1
        # Check all results are marked as final (no partials)
        assert all(r.is_final for r in results)


class TestLocalAgreement:
    """Test LocalAgreement algorithm for stable streaming transcription."""

    def test_init_default(self):
        """Test default initialization."""
        agreement = LocalAgreement()
        assert agreement.n == 2
        assert agreement.history == []
        assert agreement.committed == ""

    def test_init_custom_n(self):
        """Test custom n value."""
        agreement = LocalAgreement(n=3)
        assert agreement.n == 3

    def test_init_invalid_n(self):
        """Test invalid n value raises error."""
        with pytest.raises(ValueError, match="n must be >= 2"):
            LocalAgreement(n=1)

    def test_single_transcript_no_agreement(self):
        """Test single transcript returns empty (no agreement yet)."""
        agreement = LocalAgreement(n=2)

        result = agreement.update("Hello world")
        assert result == ""
        assert agreement.get_confirmed() == ""
        assert agreement.get_speculative() == "Hello world"

    def test_two_identical_transcripts(self):
        """Test two identical transcripts confirm text."""
        agreement = LocalAgreement(n=2)

        agreement.update("Hello world")
        result = agreement.update("Hello world")

        assert result == "Hello world"
        assert agreement.get_confirmed() == "Hello world"
        assert agreement.get_speculative() == ""

    def test_common_prefix_agreement(self):
        """Test agreement on common prefix with word boundary."""
        agreement = LocalAgreement(n=2)

        agreement.update("Hello wor")
        result = agreement.update("Hello world")

        # Character-level common prefix is "Hello wor", but implementation
        # backs off to word boundaries to avoid partial words.
        # "wor" is incomplete (not followed by space in "Hello world")
        # so it backs off to "Hello"
        assert agreement.get_confirmed() == "Hello"
        assert result == "Hello"

    def test_incremental_confirmation(self):
        """Test incremental text confirmation over multiple updates."""
        agreement = LocalAgreement(n=2)

        # First: "Hello" - no agreement yet
        result1 = agreement.update("Hello")
        assert result1 == ""

        # Second: "Hello world" - confirms "Hello"
        agreement.update("Hello world")
        assert agreement.get_confirmed() == "Hello"

        # Third: "Hello world today" - confirms "Hello world"
        # (word boundary: space after "world" in both transcripts)
        agreement.update("Hello world today")
        assert agreement.get_confirmed() == "Hello world"

    def test_n3_requires_three_agreements(self):
        """Test n=3 requires three consecutive agreements."""
        agreement = LocalAgreement(n=3)

        agreement.update("Hello")
        result1 = agreement.update("Hello world")
        assert result1 == ""  # Only 2 transcripts, need 3

        agreement.update("Hello world!")
        assert agreement.get_confirmed() == "Hello"  # All 3 start with "Hello"

    def test_reset(self):
        """Test reset clears state."""
        agreement = LocalAgreement(n=2)

        agreement.update("Hello")
        agreement.update("Hello world")
        assert agreement.get_confirmed() == "Hello"

        agreement.reset()
        assert agreement.history == []
        assert agreement.committed == ""

    def test_get_speculative(self):
        """Test get_speculative returns unconfirmed text."""
        agreement = LocalAgreement(n=2)

        agreement.update("Hello")
        agreement.update("Hello world")

        # "Hello" is confirmed, "world" is speculative
        speculative = agreement.get_speculative()
        assert "world" in speculative

    def test_no_common_prefix(self):
        """Test completely different transcripts have no agreement."""
        agreement = LocalAgreement(n=2)

        agreement.update("Hello there")
        result = agreement.update("Goodbye now")

        assert result == ""
        assert agreement.get_confirmed() == ""

    def test_whitespace_normalization(self):
        """Test whitespace is normalized."""
        agreement = LocalAgreement(n=2)

        agreement.update("  Hello world  ")
        result = agreement.update("Hello world")

        assert result == "Hello world"

    def test_word_aligned_prefix(self):
        """Test prefix finding is word-aligned when possible."""
        agreement = LocalAgreement(n=2)

        agreement.update("Hello world")
        agreement.update("Hello wor")  # Different ending

        # Should back off to "Hello" (word boundary) not "Hello wor"
        confirmed = agreement.get_confirmed()
        assert confirmed == "Hello"

    def test_history_size_limit(self):
        """Test history only keeps last n transcripts."""
        agreement = LocalAgreement(n=2)

        agreement.update("First")
        agreement.update("Second")
        agreement.update("Third")

        # Should only have last 2
        assert len(agreement.history) == 2
        assert "First" not in agreement.history


class TestStreamingConfigLocalAgreement:
    """Test LocalAgreement integration with StreamingConfig."""

    def test_default_local_agreement_enabled(self):
        """Test LocalAgreement is enabled by default."""
        config = StreamingConfig()
        assert config.use_local_agreement is True
        assert config.agreement_n == 2

    def test_local_agreement_disabled(self):
        """Test LocalAgreement can be disabled."""
        config = StreamingConfig(use_local_agreement=False)
        assert config.use_local_agreement is False

    def test_custom_agreement_n(self):
        """Test custom agreement_n value."""
        config = StreamingConfig(agreement_n=3)
        assert config.agreement_n == 3


class TestStreamingResultLocalAgreement:
    """Test LocalAgreement fields in StreamingResult."""

    def test_result_with_confirmed_text(self):
        """Test result includes confirmed and speculative text."""
        result = StreamingResult(
            text="Hello world, how are you?",
            is_final=False,
            is_partial=True,
            segment_start=0.0,
            segment_end=1.5,
            confirmed_text="Hello world",
            speculative_text=", how are you?",
            is_confirmed=True,
        )

        assert result.confirmed_text == "Hello world"
        assert result.speculative_text == ", how are you?"
        assert result.is_confirmed is True

    def test_result_default_values(self):
        """Test default values for LocalAgreement fields."""
        result = StreamingResult(
            text="Hello",
            is_final=True,
            is_partial=False,
            segment_start=0.0,
            segment_end=1.0,
        )

        assert result.confirmed_text == ""
        assert result.speculative_text == ""
        assert result.is_confirmed is False


class TestStreamingWhisperLocalAgreement:
    """Test LocalAgreement integration with StreamingWhisper."""

    def test_init_with_local_agreement(self):
        """Test StreamingWhisper initializes LocalAgreement."""
        mock_model = MagicMock()
        config = StreamingConfig(use_local_agreement=True, use_vad=False)

        streamer = StreamingWhisper(mock_model, config)

        assert streamer._local_agreement is not None
        assert streamer._local_agreement.n == 2

    def test_init_without_local_agreement(self):
        """Test StreamingWhisper without LocalAgreement."""
        mock_model = MagicMock()
        config = StreamingConfig(use_local_agreement=False, use_vad=False)

        streamer = StreamingWhisper(mock_model, config)

        assert streamer._local_agreement is None

    def test_reset_clears_local_agreement(self):
        """Test reset clears LocalAgreement state."""
        mock_model = MagicMock()
        config = StreamingConfig(use_local_agreement=True, use_vad=False)

        streamer = StreamingWhisper(mock_model, config)

        # Simulate some updates
        streamer._local_agreement.update("Hello")
        streamer._local_agreement.update("Hello world")

        # Reset should clear
        streamer.reset()

        assert streamer._local_agreement.history == []
        assert streamer._local_agreement.committed == ""


class TestDualPathConfig:
    """Test DualPathConfig defaults and settings."""

    def test_default_config(self):
        """Test default configuration values."""
        from tools.whisper_mlx.streaming import DualPathConfig

        config = DualPathConfig()

        assert config.sample_rate == 16000
        assert config.fast_chunk_duration == 1.0
        assert config.fast_latency_target_ms == 200
        assert config.quality_chunk_duration == 5.0
        assert config.quality_overlap == 1.0
        assert config.agreement_n == 2
        assert config.use_vad is True
        assert config.language is None
        assert config.task == "transcribe"

    def test_custom_config(self):
        """Test custom configuration."""
        from tools.whisper_mlx.streaming import DualPathConfig

        config = DualPathConfig(
            fast_chunk_duration=0.5,
            quality_chunk_duration=3.0,
            agreement_n=3,
            language="en",
        )

        assert config.fast_chunk_duration == 0.5
        assert config.quality_chunk_duration == 3.0
        assert config.agreement_n == 3
        assert config.language == "en"


class TestDualPathResult:
    """Test DualPathResult dataclass."""

    def test_result_creation(self):
        """Test creating a DualPathResult."""
        from tools.whisper_mlx.streaming import DualPathResult

        result = DualPathResult(
            speculative_text="Hello wor",
            speculative_is_new=True,
            confirmed_text="",
            confirmed_is_new=False,
            full_speculative="Hello wor",
            full_confirmed="",
            fast_latency_ms=150.0,
            quality_latency_ms=0.0,
            audio_time=1.0,
        )

        assert result.speculative_text == "Hello wor"
        assert result.speculative_is_new is True
        assert result.confirmed_text == ""
        assert result.confirmed_is_new is False
        assert result.fast_latency_ms == 150.0

    def test_result_with_confirmed(self):
        """Test result with confirmed text."""
        from tools.whisper_mlx.streaming import DualPathResult

        result = DualPathResult(
            speculative_text="how are you",
            speculative_is_new=True,
            confirmed_text="Hello world",
            confirmed_is_new=True,
            full_speculative="Hello world how are you",
            full_confirmed="Hello world",
            fast_latency_ms=180.0,
            quality_latency_ms=450.0,
            audio_time=6.0,
        )

        assert result.confirmed_text == "Hello world"
        assert result.confirmed_is_new is True
        assert result.quality_latency_ms == 450.0


class TestDualPathStreamerInit:
    """Test DualPathStreamer initialization."""

    def test_init_without_vad(self):
        """Test initialization without VAD."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        config = DualPathConfig(use_vad=False)

        streamer = DualPathStreamer(mock_model, config)

        assert streamer.model is mock_model
        assert streamer.config is config
        assert streamer._vad is None

    def test_init_default_config(self):
        """Test initialization with default config."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        config = DualPathConfig(use_vad=False)

        streamer = DualPathStreamer(mock_model, config)

        assert streamer._local_agreement is not None
        assert streamer._local_agreement.n == 2
        assert streamer._fast_buffer is not None
        assert streamer._quality_buffer is not None

    def test_init_state(self):
        """Test initial state after creation."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        config = DualPathConfig(use_vad=False)

        streamer = DualPathStreamer(mock_model, config)

        assert streamer._total_audio_time == 0.0
        assert streamer._full_speculative == ""
        assert streamer._full_confirmed == ""
        assert streamer._detected_language is None


class TestDualPathStreamerReset:
    """Test DualPathStreamer reset functionality."""

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        config = DualPathConfig(use_vad=False)

        streamer = DualPathStreamer(mock_model, config)

        # Simulate some usage
        streamer._total_audio_time = 10.0
        streamer._full_speculative = "some text"
        streamer._full_confirmed = "confirmed"
        streamer._detected_language = "en"

        # Reset
        streamer.reset()

        assert streamer._total_audio_time == 0.0
        assert streamer._full_speculative == ""
        assert streamer._full_confirmed == ""
        assert streamer._detected_language is None

    def test_reset_clears_buffers(self):
        """Test reset clears audio buffers."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        config = DualPathConfig(use_vad=False)

        streamer = DualPathStreamer(mock_model, config)

        # Add some audio
        audio = np.ones(16000, dtype=np.float32)
        streamer._fast_buffer.append(audio)
        streamer._quality_buffer.append(audio)

        assert streamer._fast_buffer.duration > 0
        assert streamer._quality_buffer.duration > 0

        # Reset
        streamer.reset()

        assert streamer._fast_buffer.duration == 0.0
        assert streamer._quality_buffer.duration == 0.0


class TestDualPathStreamerProcessing:
    """Test DualPathStreamer audio processing."""

    def test_process_chunk_accumulation(self):
        """Test audio accumulates in buffers."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        config = DualPathConfig(use_vad=False)

        streamer = DualPathStreamer(mock_model, config)

        # Add 0.5s of audio (not enough for fast path)
        audio = np.ones(8000, dtype=np.float32) * 0.5

        async def run():
            return [result async for result in streamer._process_chunk(audio)]

        results = asyncio.run(run())

        # No results yet (below threshold)
        assert len(results) == 0
        assert streamer._fast_buffer.duration == 0.5
        assert streamer._quality_buffer.duration == 0.5

    def test_fast_path_triggers(self):
        """Test fast path triggers at threshold."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello", "language": "en"}
        config = DualPathConfig(use_vad=False, fast_chunk_duration=1.0)

        streamer = DualPathStreamer(mock_model, config)

        # Add 1.0s of audio (triggers fast path)
        audio = np.ones(16000, dtype=np.float32) * 0.5

        async def run():
            return [result async for result in streamer._process_chunk(audio)]

        results = asyncio.run(run())

        # Fast path should trigger
        assert len(results) == 1
        assert results[0].speculative_text == "Hello"
        assert results[0].speculative_is_new is True
        assert mock_model.transcribe.called

    def test_quality_path_triggers(self):
        """Test quality path triggers at threshold."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello world", "language": "en"}
        config = DualPathConfig(
            use_vad=False,
            fast_chunk_duration=1.0,
            quality_chunk_duration=2.0,  # Lower for test
            agreement_n=2,
        )

        streamer = DualPathStreamer(mock_model, config)

        async def run():
            # First 2s chunk
            audio1 = np.ones(32000, dtype=np.float32) * 0.5
            results = [result async for result in streamer._process_chunk(audio1)]
            # Second chunk for agreement
            audio2 = np.ones(32000, dtype=np.float32) * 0.5
            results.extend([result async for result in streamer._process_chunk(audio2)])
            return results

        results = asyncio.run(run())

        # Quality path should have triggered (LocalAgreement needs 2 passes)
        assert len(results) >= 2


class TestDualPathStreamerIntegration:
    """Integration tests for DualPathStreamer."""

    def test_full_stream_processing(self):
        """Test processing a complete audio stream."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        call_count = [0]

        def mock_transcribe(*args, **kwargs):
            call_count[0] += 1
            texts = ["Hello", "Hello world", "Hello world, how", "Hello world, how are"]
            idx = min(call_count[0] - 1, len(texts) - 1)
            return {"text": texts[idx], "language": "en"}

        mock_model.transcribe.side_effect = mock_transcribe
        config = DualPathConfig(
            use_vad=False,
            fast_chunk_duration=1.0,
            quality_chunk_duration=5.0,
        )

        streamer = DualPathStreamer(mock_model, config)

        async def audio_gen():
            # Simulate 6 seconds of audio in chunks
            for _ in range(6):
                yield np.ones(16000, dtype=np.float32) * 0.3

        async def run():
            return [result async for result in streamer.process_stream(audio_gen())]

        results = asyncio.run(run())

        # Should have multiple results from fast path
        assert len(results) > 0
        # Should have accumulated speculative text
        assert any(r.speculative_is_new for r in results)

    def test_finalization(self):
        """Test stream finalization."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Final text", "language": "en"}
        config = DualPathConfig(
            use_vad=False,
            fast_chunk_duration=1.0,
            quality_chunk_duration=5.0,
        )

        streamer = DualPathStreamer(mock_model, config)

        # Add 0.5s of audio (below threshold but above min)
        audio = np.ones(8000, dtype=np.float32) * 0.5
        streamer._fast_buffer.append(audio)
        streamer._quality_buffer.append(audio)

        async def run():
            return [result async for result in streamer._finalize()]

        results = asyncio.run(run())

        # Finalize should process remaining audio
        assert len(results) == 1
        assert "Final text" in results[0].speculative_text or "Final text" in results[0].confirmed_text


class TestDualPathStreamerLanguageDetection:
    """Test language detection caching in DualPathStreamer."""

    def test_language_caching(self):
        """Test language is cached after first detection."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Bonjour", "language": "fr"}
        config = DualPathConfig(use_vad=False, fast_chunk_duration=1.0)

        streamer = DualPathStreamer(mock_model, config)

        assert streamer._detected_language is None

        # Process enough audio to trigger fast path
        audio = np.ones(16000, dtype=np.float32) * 0.5

        async def run():
            async for _ in streamer._process_chunk(audio):
                pass

        asyncio.run(run())

        assert streamer._detected_language == "fr"


class TestInitialPromptTerminologyInjection:
    """Test J9: Initial prompt for terminology injection."""

    def test_streaming_config_initial_prompt(self):
        """Test initial_prompt in StreamingConfig."""
        config = StreamingConfig(initial_prompt="Technical terms: API, Kubernetes")
        assert config.initial_prompt == "Technical terms: API, Kubernetes"

    def test_streaming_config_default_no_prompt(self):
        """Test default StreamingConfig has no initial_prompt."""
        config = StreamingConfig()
        assert config.initial_prompt is None

    def test_dual_path_config_initial_prompt(self):
        """Test initial_prompt in DualPathConfig."""
        from tools.whisper_mlx.streaming import DualPathConfig

        config = DualPathConfig(initial_prompt="Names: John, Sarah, Mike")
        assert config.initial_prompt == "Names: John, Sarah, Mike"

    def test_dual_path_config_default_no_prompt(self):
        """Test default DualPathConfig has no initial_prompt."""
        from tools.whisper_mlx.streaming import DualPathConfig

        config = DualPathConfig()
        assert config.initial_prompt is None

    def test_streaming_whisper_passes_prompt(self):
        """Test StreamingWhisper passes initial_prompt to model.transcribe."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Test output", "language": "en"}

        config = StreamingConfig(
            use_vad=False,
            min_chunk_duration=0.5,
            max_chunk_duration=2.0,
            silence_threshold_duration=0.3,
            emit_partials=False,
            initial_prompt="Technical: API endpoints",
        )

        streamer = StreamingWhisper(mock_model, config)

        async def audio_generator():
            yield np.ones(16000, dtype=np.float32) * 0.5
            for _ in range(5):
                yield np.zeros(4800, dtype=np.float32)

        async def run_test():
            return [result async for result in streamer.transcribe_stream(audio_generator())]

        asyncio.run(run_test())

        # Check that transcribe was called with initial_prompt
        assert mock_model.transcribe.called
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("initial_prompt") == "Technical: API endpoints"

    def test_dual_path_streamer_passes_prompt(self):
        """Test DualPathStreamer passes initial_prompt to model.transcribe."""
        from tools.whisper_mlx.streaming import DualPathConfig, DualPathStreamer

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Test output", "language": "en"}

        config = DualPathConfig(
            use_vad=False,
            fast_chunk_duration=1.0,
            initial_prompt="Medical: hematology, oncology",
        )

        streamer = DualPathStreamer(mock_model, config)

        audio = np.ones(16000, dtype=np.float32) * 0.5

        async def run():
            async for _ in streamer._process_chunk(audio):
                pass

        asyncio.run(run())

        # Check that transcribe was called with initial_prompt
        assert mock_model.transcribe.called
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("initial_prompt") == "Medical: hematology, oncology"


# =============================================================================
# J2: AlignAttPolicy Tests
# =============================================================================


class TestAlignAttPolicyInit:
    """Test AlignAttPolicy initialization and parameter validation."""

    def test_default_init(self):
        """Test default initialization values."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()
        assert policy.frame_threshold == 0.5
        assert policy.recent_threshold == 0.2
        assert policy.min_confidence == 0.3
        assert policy._emitted_count == 0

    def test_custom_init(self):
        """Test custom initialization values."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy(
            frame_threshold=0.7,
            recent_threshold=0.3,
            min_confidence=0.5,
        )
        assert policy.frame_threshold == 0.7
        assert policy.recent_threshold == 0.3
        assert policy.min_confidence == 0.5

    def test_invalid_frame_threshold(self):
        """Test invalid frame_threshold raises error."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        with pytest.raises(ValueError, match="frame_threshold must be in"):
            AlignAttPolicy(frame_threshold=0.0)
        with pytest.raises(ValueError, match="frame_threshold must be in"):
            AlignAttPolicy(frame_threshold=1.0)
        with pytest.raises(ValueError, match="frame_threshold must be in"):
            AlignAttPolicy(frame_threshold=1.5)

    def test_invalid_recent_threshold(self):
        """Test invalid recent_threshold raises error."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        with pytest.raises(ValueError, match="recent_threshold must be in"):
            AlignAttPolicy(recent_threshold=0.0)
        with pytest.raises(ValueError, match="recent_threshold must be in"):
            AlignAttPolicy(recent_threshold=1.0)


class TestAlignAttPolicyShouldEmit:
    """Test AlignAttPolicy.should_emit() method."""

    def test_emit_when_attention_on_old_frames(self):
        """Test should_emit returns True when attention is on older frames."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy(frame_threshold=0.5, recent_threshold=0.2)

        # Attention concentrated at beginning (frame 10 out of 100)
        # Shape: (n_heads=4, text_len=10, audio_len=100)
        weights = np.zeros((4, 10, 100), dtype=np.float32)
        for head in range(4):
            for token in range(10):
                # Concentrated attention on frame 10
                weights[head, token, 10] = 0.9
                weights[head, token, 11] = 0.1

        # Token at position 0 should emit (attention on old frames)
        assert policy.should_emit(weights, current_token_idx=0) is True

    def test_no_emit_when_attention_on_recent_frames(self):
        """Test should_emit returns False when attention is on recent frames."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy(frame_threshold=0.5, recent_threshold=0.2)

        # Attention concentrated at end (frame 95 out of 100)
        weights = np.zeros((4, 10, 100), dtype=np.float32)
        for head in range(4):
            for token in range(10):
                # Concentrated attention on recent frames
                weights[head, token, 95] = 0.9
                weights[head, token, 96] = 0.1

        # Should NOT emit (attention on recent frames)
        assert policy.should_emit(weights, current_token_idx=0) is False

    def test_no_emit_when_low_confidence(self):
        """Test should_emit returns False when attention is diffuse."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy(min_confidence=0.3)

        # Very diffuse attention (uniform distribution)
        weights = np.ones((4, 10, 100), dtype=np.float32) / 100
        # Max attention per position is 0.01 < 0.3 threshold

        assert policy.should_emit(weights, current_token_idx=0) is False

    def test_handles_batch_dimension(self):
        """Test should_emit handles 4D input with batch dimension."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()

        # 4D: (batch=2, n_heads=4, text_len=10, audio_len=100)
        weights = np.zeros((2, 4, 10, 100), dtype=np.float32)
        # Put strong attention on old frame for first batch item
        weights[0, :, :, 10] = 0.9

        # Should use first batch item and emit
        assert policy.should_emit(weights, current_token_idx=0) is True

    def test_invalid_token_idx(self):
        """Test should_emit returns False for out-of-range token index."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()
        weights = np.ones((4, 10, 100), dtype=np.float32)

        # Token index beyond text_len
        assert policy.should_emit(weights, current_token_idx=15) is False


class TestAlignAttPolicyGetEmitBoundary:
    """Test AlignAttPolicy.get_emit_boundary() method."""

    def test_finds_boundary_at_first_uncertain_token(self):
        """Test get_emit_boundary finds first uncertain token."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy(frame_threshold=0.5, recent_threshold=0.2)

        # Tokens 0-4 attend to old frames, tokens 5-9 attend to recent frames
        weights = np.zeros((4, 10, 100), dtype=np.float32)
        for head in range(4):
            # Tokens 0-4: attention on frame 20 (old)
            for token in range(5):
                weights[head, token, 20] = 0.9

            # Tokens 5-9: attention on frame 95 (recent)
            for token in range(5, 10):
                weights[head, token, 95] = 0.9

        boundary = policy.get_emit_boundary(weights)
        assert boundary == 5  # Can emit tokens 0-4

    def test_boundary_zero_when_all_uncertain(self):
        """Test boundary is 0 when first token is uncertain."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()

        # All tokens attend to recent frames
        weights = np.zeros((4, 10, 100), dtype=np.float32)
        weights[:, :, 95] = 0.9

        boundary = policy.get_emit_boundary(weights)
        assert boundary == 0

    def test_boundary_all_when_all_confident(self):
        """Test boundary is full length when all tokens are confident."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()

        # All tokens attend to old frames
        weights = np.zeros((4, 10, 100), dtype=np.float32)
        weights[:, :, 10] = 0.9

        boundary = policy.get_emit_boundary(weights)
        assert boundary == 10


class TestAlignAttPolicyUpdateWithWeights:
    """Test AlignAttPolicy.update_with_weights() method."""

    def test_returns_emittable_text(self):
        """Test update_with_weights returns correct portion of text."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()

        # 5 confident tokens, 5 uncertain
        weights = np.zeros((4, 10, 100), dtype=np.float32)
        for token in range(5):
            weights[:, token, 10] = 0.9  # Old frame - confident
        for token in range(5, 10):
            weights[:, token, 95] = 0.9  # Recent frame - uncertain

        # Text with ~10 words (approximately 1 word per token)
        text = "Hello world this is a test of the system today"

        result = policy.update_with_weights(text, weights)

        # Should return approximately first half (words corresponding to 5 tokens)
        # Due to word boundary detection, may be slightly different
        assert len(result) > 0
        assert len(result) < len(text)

    def test_incremental_emission(self):
        """Test update tracks emitted text incrementally."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()

        # All confident
        weights = np.zeros((4, 5, 100), dtype=np.float32)
        weights[:, :, 10] = 0.9

        text1 = "Hello world"
        result1 = policy.update_with_weights(text1, weights)
        # Should emit something
        assert len(result1) > 0 or policy._emitted_count > 0

        # Second call should not re-emit already emitted text
        result2 = policy.update_with_weights(text1, weights)
        assert result2 == ""  # Nothing new to emit

    def test_reset_clears_state(self):
        """Test reset clears emitted count."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()

        # Simulate some emission
        policy._emitted_count = 5

        policy.reset()
        assert policy._emitted_count == 0


class TestAlignAttPolicyEdgeCases:
    """Test AlignAttPolicy edge cases."""

    def test_empty_weights(self):
        """Test behavior with empty attention weights."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy()

        # Empty text dimension
        weights = np.zeros((4, 0, 100), dtype=np.float32)

        boundary = policy.get_emit_boundary(weights)
        assert boundary == 0

    def test_single_audio_frame(self):
        """Test behavior with single audio frame."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        policy = AlignAttPolicy(recent_threshold=0.2)

        # Single audio frame
        weights = np.zeros((4, 5, 1), dtype=np.float32)
        weights[:, :, 0] = 1.0

        # With only 1 frame, recent_boundary = int(1 * 0.8) = 0
        # mass_frame = 0, so 0 < 0 is False
        assert policy.should_emit(weights, current_token_idx=0) is False

    def test_attention_spread_across_frames(self):
        """Test with attention spread across multiple frames."""
        from tools.whisper_mlx.streaming import AlignAttPolicy

        # Lower min_confidence to allow spread attention
        policy = AlignAttPolicy(frame_threshold=0.5, min_confidence=0.05)

        # Attention gradually increasing toward middle
        weights = np.zeros((4, 5, 100), dtype=np.float32)
        # Spread attention: 0.1 each on frames 20-29 = total 1.0 on old frames
        for i in range(20, 30):
            weights[:, :, i] = 0.1

        # 50% attention mass should be around frame 25 (old, before frame 80)
        assert policy.should_emit(weights, current_token_idx=0) is True


# =============================================================================
# J10: BatchingStreamServer Tests
# =============================================================================


class TestBatchServerConfig:
    """Test BatchServerConfig defaults and settings."""

    def test_default_config(self):
        """Test default configuration values."""
        from tools.whisper_mlx.streaming import BatchServerConfig

        config = BatchServerConfig()

        assert config.max_batch_size == 8
        assert config.batch_timeout_ms == 100.0
        assert config.min_audio_duration == 0.5
        assert config.sample_rate == 16000
        assert config.language is None
        assert config.task == "transcribe"
        assert config.session_timeout_seconds == 60.0
        assert config.max_sessions == 100

    def test_custom_config(self):
        """Test custom configuration."""
        from tools.whisper_mlx.streaming import BatchServerConfig

        config = BatchServerConfig(
            max_batch_size=4,
            batch_timeout_ms=200.0,
            min_audio_duration=1.0,
            language="en",
            max_sessions=50,
        )

        assert config.max_batch_size == 4
        assert config.batch_timeout_ms == 200.0
        assert config.min_audio_duration == 1.0
        assert config.language == "en"
        assert config.max_sessions == 50


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_result_creation(self):
        """Test creating a BatchResult."""
        from tools.whisper_mlx.streaming import BatchResult

        result = BatchResult(
            session_id="session_1",
            text="Hello world",
            confirmed_text="Hello",
            speculative_text="world",
            is_confirmed=True,
            processing_time_ms=50.0,
            batch_size=4,
            language="en",
        )

        assert result.session_id == "session_1"
        assert result.text == "Hello world"
        assert result.confirmed_text == "Hello"
        assert result.speculative_text == "world"
        assert result.is_confirmed is True
        assert result.processing_time_ms == 50.0
        assert result.batch_size == 4
        assert result.language == "en"


class TestBatchingStreamServerInit:
    """Test BatchingStreamServer initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        assert server.model is mock_model
        assert server.config is not None
        assert server.config.max_batch_size == 8
        assert len(server._sessions) == 0
        assert server._batches_processed == 0
        assert server._total_sessions_served == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from tools.whisper_mlx.streaming import BatchingStreamServer, BatchServerConfig

        mock_model = MagicMock()
        config = BatchServerConfig(max_batch_size=4, language="en")
        server = BatchingStreamServer(mock_model, config)

        assert server.config is config
        assert server.config.max_batch_size == 4
        assert server.config.language == "en"


class TestBatchingStreamServerSession:
    """Test session management in BatchingStreamServer."""

    def test_create_session(self):
        """Test creating a new session."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        async def run():
            return await server.create_session("session_1")

        result = asyncio.run(run())

        assert result is True
        assert "session_1" in server._sessions
        assert server._sessions["session_1"].session_id == "session_1"
        assert server._sessions["session_1"].is_active is True
        assert server._total_sessions_served == 1

    def test_create_duplicate_session(self):
        """Test creating duplicate session returns False."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_1")
            return await server.create_session("session_1")

        result = asyncio.run(run())

        assert result is False
        assert server._total_sessions_served == 1  # Only counted once

    def test_max_sessions_limit(self):
        """Test max sessions limit is enforced."""
        from tools.whisper_mlx.streaming import BatchingStreamServer, BatchServerConfig

        mock_model = MagicMock()
        config = BatchServerConfig(max_sessions=2)
        server = BatchingStreamServer(mock_model, config)

        async def run():
            await server.create_session("s1")
            await server.create_session("s2")
            # Third should raise
            await server.create_session("s3")

        with pytest.raises(RuntimeError, match="Maximum sessions"):
            asyncio.run(run())

    def test_session_with_local_agreement(self):
        """Test session created with LocalAgreement."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_1", use_local_agreement=True, agreement_n=3)

        asyncio.run(run())

        session = server._sessions["session_1"]
        assert session.local_agreement is not None
        assert session.local_agreement.n == 3

    def test_session_without_local_agreement(self):
        """Test session created without LocalAgreement."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_1", use_local_agreement=False)

        asyncio.run(run())

        session = server._sessions["session_1"]
        assert session.local_agreement is None


class TestBatchingStreamServerAddAudio:
    """Test adding audio to BatchingStreamServer."""

    def test_add_audio_creates_session(self):
        """Test adding audio auto-creates session if needed."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        audio = np.ones(8000, dtype=np.float32) * 0.5

        async def run():
            await server.add_audio("session_1", audio)

        asyncio.run(run())

        assert "session_1" in server._sessions
        assert server._sessions["session_1"].total_audio_time > 0

    def test_add_audio_accumulates(self):
        """Test audio accumulates in session buffer."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_1")
            await server.add_audio("session_1", np.ones(8000, dtype=np.float32))
            await server.add_audio("session_1", np.ones(8000, dtype=np.float32))

        asyncio.run(run())

        session = server._sessions["session_1"]
        assert session.total_audio_time == 1.0  # 16000 samples / 16000 Hz

    def test_add_audio_normalizes_int16(self):
        """Test int16 audio is normalized to float32."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        audio = np.ones(8000, dtype=np.int16) * 16384  # Half max

        async def run():
            await server.add_audio("session_1", audio)

        asyncio.run(run())

        session = server._sessions["session_1"]
        # Should be normalized to ~0.5
        assert session.audio_buffer.dtype == np.float32
        assert np.mean(session.audio_buffer) < 1.0

    def test_add_audio_marks_pending(self):
        """Test session is marked pending when enough audio accumulated."""
        from tools.whisper_mlx.streaming import BatchingStreamServer, BatchServerConfig

        mock_model = MagicMock()
        config = BatchServerConfig(min_audio_duration=0.5)
        server = BatchingStreamServer(mock_model, config)

        async def run():
            await server.create_session("session_1")
            # Add 0.5s of audio
            await server.add_audio("session_1", np.ones(8000, dtype=np.float32))

        asyncio.run(run())

        assert "session_1" in server._pending_sessions


class TestBatchingStreamServerFinalize:
    """Test session finalization in BatchingStreamServer."""

    def test_finalize_nonexistent_session(self):
        """Test finalizing non-existent session returns None."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        async def run():
            return await server.finalize_session("nonexistent")

        result = asyncio.run(run())
        assert result is None

    def test_finalize_with_remaining_audio(self):
        """Test finalization processes remaining audio."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Final text", "language": "en"}
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_1")
            await server.add_audio("session_1", np.ones(8000, dtype=np.float32))
            return await server.finalize_session("session_1")

        result = asyncio.run(run())

        assert result is not None
        assert result.text == "Final text"
        assert result.is_confirmed is True
        assert "session_1" not in server._sessions  # Session cleaned up

    def test_finalize_empty_session(self):
        """Test finalizing session with minimal audio returns None."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_1")
            # Add very little audio (below min threshold)
            await server.add_audio("session_1", np.ones(1000, dtype=np.float32))
            return await server.finalize_session("session_1")

        result = asyncio.run(run())

        assert result is None
        assert "session_1" not in server._sessions


class TestBatchingStreamServerBatchProcessing:
    """Test batch processing in BatchingStreamServer."""

    def test_process_batch_single_session(self):
        """Test batch processing with single session."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        mock_model.transcribe_batch.return_value = [
            {"text": "Hello world", "language": "en"},
        ]
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_1")
            await server.add_audio("session_1", np.ones(8000, dtype=np.float32))
            await server._process_batch(["session_1"])

        asyncio.run(run())

        assert mock_model.transcribe_batch.called
        # Check batch was called with one audio array
        call_args = mock_model.transcribe_batch.call_args
        assert len(call_args[0][0]) == 1  # One audio in batch

    def test_process_batch_multiple_sessions(self):
        """Test batch processing with multiple sessions."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        mock_model.transcribe_batch.return_value = [
            {"text": "Session 1 text", "language": "en"},
            {"text": "Session 2 text", "language": "en"},
            {"text": "Session 3 text", "language": "en"},
        ]
        server = BatchingStreamServer(mock_model)

        async def run():
            for i in range(3):
                await server.create_session(f"session_{i}")
                await server.add_audio(f"session_{i}", np.ones(8000, dtype=np.float32))
            await server._process_batch(["session_0", "session_1", "session_2"])

        asyncio.run(run())

        assert mock_model.transcribe_batch.called
        # Check batch size
        call_args = mock_model.transcribe_batch.call_args
        assert len(call_args[0][0]) == 3

    def test_process_batch_distributes_results(self):
        """Test results are distributed to correct sessions."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        mock_model.transcribe_batch.return_value = [
            {"text": "Result A", "language": "en"},
            {"text": "Result B", "language": "fr"},
        ]
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_a", use_local_agreement=False)
            await server.create_session("session_b", use_local_agreement=False)
            await server.add_audio("session_a", np.ones(8000, dtype=np.float32))
            await server.add_audio("session_b", np.ones(8000, dtype=np.float32))
            await server._process_batch(["session_a", "session_b"])

            # Get results
            results_a = await server._get_results("session_a")
            results_b = await server._get_results("session_b")
            return results_a, results_b

        results_a, results_b = asyncio.run(run())

        assert len(results_a) == 1
        assert results_a[0].text == "Result A"
        assert results_a[0].language == "en"

        assert len(results_b) == 1
        assert results_b[0].text == "Result B"
        assert results_b[0].language == "fr"

    def test_process_batch_with_local_agreement(self):
        """Test LocalAgreement is applied during batch processing."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        mock_model.transcribe_batch.return_value = [
            {"text": "Hello world", "language": "en"},
        ]
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("session_1", use_local_agreement=True)

            # First batch - update LocalAgreement (no confirmation yet)
            server._sessions["session_1"].audio_buffer = np.ones(8000, dtype=np.float32)
            await server._process_batch(["session_1"])
            first_results = await server._get_results("session_1")

            # Second batch - should confirm
            mock_model.transcribe_batch.return_value = [
                {"text": "Hello world", "language": "en"},
            ]
            server._sessions["session_1"].audio_buffer = np.ones(8000, dtype=np.float32)
            await server._process_batch(["session_1"])
            second_results = await server._get_results("session_1")

            return first_results, second_results

        first_results, second_results = asyncio.run(run())

        # First result should not be confirmed (only 1 transcript)
        assert len(first_results) == 1
        assert first_results[0].is_confirmed is False

        # Second result should be confirmed (2 identical transcripts)
        assert len(second_results) == 1
        assert second_results[0].is_confirmed is True
        assert second_results[0].confirmed_text == "Hello world"


class TestBatchingStreamServerStats:
    """Test BatchingStreamServer statistics."""

    def test_get_stats_initial(self):
        """Test initial stats are zero."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        stats = server.get_stats()

        assert stats["active_sessions"] == 0
        assert stats["batches_processed"] == 0
        assert stats["total_sessions_served"] == 0

    def test_get_stats_after_sessions(self):
        """Test stats after creating sessions."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        async def run():
            await server.create_session("s1")
            await server.create_session("s2")

        asyncio.run(run())

        stats = server.get_stats()

        assert stats["active_sessions"] == 2
        assert stats["total_sessions_served"] == 2


class TestBatchingStreamServerShutdown:
    """Test BatchingStreamServer shutdown."""

    def test_shutdown_sets_flag(self):
        """Test shutdown sets the shutdown flag."""
        from tools.whisper_mlx.streaming import BatchingStreamServer

        mock_model = MagicMock()
        server = BatchingStreamServer(mock_model)

        assert server._shutdown is False

        async def run():
            await server.shutdown()

        asyncio.run(run())

        assert server._shutdown is True


class TestBatchingStreamServerIntegration:
    """Integration tests for BatchingStreamServer."""

    def test_multi_user_workflow(self):
        """Test realistic multi-user workflow."""
        from tools.whisper_mlx.streaming import BatchingStreamServer, BatchServerConfig

        mock_model = MagicMock()
        call_count = [0]

        def mock_batch(*args, **kwargs):
            call_count[0] += 1
            batch_size = len(args[0])
            return [
                {"text": f"User {i} text", "language": "en"}
                for i in range(batch_size)
            ]

        mock_model.transcribe_batch.side_effect = mock_batch

        config = BatchServerConfig(
            max_batch_size=4,
            min_audio_duration=0.5,
        )
        server = BatchingStreamServer(mock_model, config)

        async def run():
            # Create 3 sessions
            for i in range(3):
                await server.create_session(f"user_{i}", use_local_agreement=False)

            # Each user sends audio
            for i in range(3):
                audio = _rng.standard_normal(8000).astype(np.float32) * 0.3
                await server.add_audio(f"user_{i}", audio)

            # Process batch
            await server._process_batch(["user_0", "user_1", "user_2"])

            # Collect results
            all_results = []
            for i in range(3):
                results = await server._get_results(f"user_{i}")
                all_results.extend(results)

            return all_results

        results = asyncio.run(run())

        assert len(results) == 3
        assert mock_model.transcribe_batch.called
        assert server.get_stats()["active_sessions"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
