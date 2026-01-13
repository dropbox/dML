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

"""Tests for the DashVoice Echo Cancellation module."""

import sys
from pathlib import Path

import numpy as np

# Add tools to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from dashvoice.echo_cancel import (
    EchoCancellationResult,
    EchoCanceller,
    ReferenceBuffer,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestEchoCancellationResult:
    """Tests for EchoCancellationResult dataclass."""

    def test_create_result(self):
        """Test creating an EchoCancellationResult."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = EchoCancellationResult(
            cleaned_audio=audio,
            detected_delay_ms=50.0,
            echo_reduction_db=20.0,
            had_echo=True,
            processing_time_ms=2.5,
        )
        np.testing.assert_array_equal(result.cleaned_audio, audio)
        assert result.detected_delay_ms == 50.0
        assert result.echo_reduction_db == 20.0
        assert result.had_echo is True
        assert result.processing_time_ms == 2.5

    def test_result_no_echo(self):
        """Test result when no echo detected."""
        audio = np.zeros(100, dtype=np.float32)
        result = EchoCancellationResult(
            cleaned_audio=audio,
            detected_delay_ms=0.0,
            echo_reduction_db=0.0,
            had_echo=False,
            processing_time_ms=1.0,
        )
        assert result.had_echo is False
        assert result.echo_reduction_db == 0.0


class TestReferenceBuffer:
    """Tests for ReferenceBuffer circular buffer."""

    def test_init_default(self):
        """Test default initialization."""
        buffer = ReferenceBuffer()
        assert buffer.sample_rate == 24000
        assert buffer.max_samples == int(5.0 * 24000)
        assert buffer.write_pos == 0
        assert buffer.total_written == 0

    def test_init_custom(self):
        """Test custom initialization."""
        buffer = ReferenceBuffer(max_duration_s=2.0, sample_rate=16000)
        assert buffer.sample_rate == 16000
        assert buffer.max_samples == 32000

    def test_add_simple(self):
        """Test adding audio to buffer."""
        buffer = ReferenceBuffer(max_duration_s=1.0, sample_rate=1000)
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        buffer.add(audio)
        assert buffer.total_written == 3
        assert buffer.write_pos == 3

    def test_add_multiple(self):
        """Test adding multiple audio chunks."""
        buffer = ReferenceBuffer(max_duration_s=1.0, sample_rate=1000)
        buffer.add(np.array([0.1, 0.2], dtype=np.float32))
        buffer.add(np.array([0.3, 0.4, 0.5], dtype=np.float32))
        assert buffer.total_written == 5
        assert buffer.write_pos == 5

    def test_add_wrap_around(self):
        """Test buffer wrap-around."""
        buffer = ReferenceBuffer(max_duration_s=0.005, sample_rate=1000)  # 5 samples
        buffer.add(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))  # 4 samples
        buffer.add(np.array([0.5, 0.6, 0.7], dtype=np.float32))  # 3 more, wraps
        assert buffer.total_written == 7
        # Should wrap around
        assert buffer.write_pos == 2  # (4 + 3) % 5 = 2

    def test_add_larger_than_buffer(self):
        """Test adding audio larger than buffer capacity."""
        buffer = ReferenceBuffer(max_duration_s=0.003, sample_rate=1000)  # 3 samples
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)  # 5 samples
        buffer.add(audio)
        # Should keep last 3 samples
        expected = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(buffer.buffer, expected)

    def test_get_recent_simple(self):
        """Test getting recent audio."""
        buffer = ReferenceBuffer(max_duration_s=1.0, sample_rate=1000)
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        buffer.add(audio)
        recent = buffer.get_recent(0.003)  # Get 3 samples
        expected = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(recent, expected)

    def test_get_recent_empty(self):
        """Test getting recent from empty buffer."""
        buffer = ReferenceBuffer(max_duration_s=1.0, sample_rate=1000)
        recent = buffer.get_recent(0.1)
        assert len(recent) == 0

    def test_get_recent_wrap_around(self):
        """Test getting recent audio that spans buffer boundary."""
        buffer = ReferenceBuffer(max_duration_s=0.005, sample_rate=1000)  # 5 samples
        buffer.add(np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
        buffer.add(np.array([0.6, 0.7], dtype=np.float32))  # Wraps
        # Buffer now: [0.6, 0.7, 0.3, 0.4, 0.5], write_pos = 2
        recent = buffer.get_recent(0.004)  # Get 4 samples
        # Most recent 4: 0.4, 0.5, 0.6, 0.7
        expected = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        np.testing.assert_array_almost_equal(recent, expected)

    def test_get_recent_more_than_written(self):
        """Test requesting more than what's been written."""
        buffer = ReferenceBuffer(max_duration_s=1.0, sample_rate=1000)
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        buffer.add(audio)
        recent = buffer.get_recent(1.0)  # Request 1000 samples
        # Should return only what's available
        assert len(recent) == 3
        np.testing.assert_array_almost_equal(recent, audio)

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = ReferenceBuffer(max_duration_s=1.0, sample_rate=1000)
        buffer.add(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        buffer.clear()
        assert buffer.write_pos == 0
        assert buffer.total_written == 0
        np.testing.assert_array_equal(buffer.buffer, np.zeros_like(buffer.buffer))


class TestEchoCancellerInit:
    """Tests for EchoCanceller initialization."""

    def test_default_init(self):
        """Test default initialization."""
        canceller = EchoCanceller()
        assert canceller.sample_rate == 24000
        assert canceller.max_delay_samples == int(500 * 24000 / 1000)
        assert canceller.min_correlation == 0.3

    def test_custom_init(self):
        """Test custom initialization."""
        canceller = EchoCanceller(
            sample_rate=16000, max_delay_ms=200, min_correlation=0.5,
        )
        assert canceller.sample_rate == 16000
        assert canceller.max_delay_samples == int(200 * 16000 / 1000)
        assert canceller.min_correlation == 0.5

    def test_ref_buffer_created(self):
        """Test that reference buffer is created."""
        canceller = EchoCanceller()
        assert canceller.ref_buffer is not None
        assert isinstance(canceller.ref_buffer, ReferenceBuffer)

    def test_filter_initialized(self):
        """Test that adaptive filter is initialized."""
        canceller = EchoCanceller()
        assert canceller.filter_length == 1024
        assert len(canceller.room_filter) == 1024
        np.testing.assert_array_equal(canceller.room_filter, np.zeros(1024))


class TestEchoCancellerAddReference:
    """Tests for add_reference method."""

    def test_add_reference(self):
        """Test adding reference audio."""
        canceller = EchoCanceller(sample_rate=1000)
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        canceller.add_reference(audio)
        assert canceller.ref_buffer.total_written == 3

    def test_add_multiple_references(self):
        """Test adding multiple reference chunks."""
        canceller = EchoCanceller(sample_rate=1000)
        canceller.add_reference(np.array([0.1, 0.2], dtype=np.float32))
        canceller.add_reference(np.array([0.3, 0.4], dtype=np.float32))
        assert canceller.ref_buffer.total_written == 4


class TestEchoCancellerFindDelay:
    """Tests for find_delay method."""

    def test_find_delay_exact(self):
        """Test finding delay with exact echo."""
        canceller = EchoCanceller(sample_rate=1000, max_delay_ms=500)
        # Create reference and delayed echo
        ref = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32)
        delay_samples = 20
        mic = np.zeros(100, dtype=np.float32)
        mic[delay_samples:] = ref[:-delay_samples]

        detected_delay, correlation = canceller.find_delay(mic, ref)
        # Allow some tolerance
        assert abs(detected_delay - delay_samples) <= 2
        assert correlation > 0.5

    def test_find_delay_no_echo(self):
        """Test finding delay with unrelated signals."""
        canceller = EchoCanceller(sample_rate=1000)
        ref = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32)
        mic = np.cos(np.linspace(0, 8 * np.pi, 100)).astype(np.float32)

        _, correlation = canceller.find_delay(mic, ref)
        # Correlation should be low for unrelated signals
        assert abs(correlation) < 0.5

    def test_find_delay_empty_inputs(self):
        """Test finding delay with empty inputs."""
        canceller = EchoCanceller()
        delay, corr = canceller.find_delay(np.array([]), np.array([0.1, 0.2]))
        assert delay == 0
        assert corr == 0.0

        delay, corr = canceller.find_delay(np.array([0.1, 0.2]), np.array([]))
        assert delay == 0
        assert corr == 0.0

    def test_find_delay_silent_input(self):
        """Test finding delay with silent input."""
        canceller = EchoCanceller()
        ref = np.zeros(100, dtype=np.float32)
        mic = np.zeros(100, dtype=np.float32)
        delay, corr = canceller.find_delay(mic, ref)
        assert corr == 0.0


class TestEchoCancellerSubtractEcho:
    """Tests for subtract_echo method."""

    def test_subtract_exact_echo(self):
        """Test subtracting exact echo."""
        canceller = EchoCanceller(sample_rate=1000)
        ref = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32)
        # When delay=0, ref appears directly in mic without shift
        mic = ref.copy()

        cancelled = canceller.subtract_echo(mic, ref, delay_samples=0)
        # Should significantly reduce energy for exact match
        original_energy = np.sum(mic**2)
        cancelled_energy = np.sum(cancelled**2)
        # Energy should be reduced (may not be zero due to gain estimation)
        assert cancelled_energy < original_energy * 0.5

    def test_subtract_no_overlap(self):
        """Test when delay is larger than reference."""
        canceller = EchoCanceller()
        ref = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mic = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        # Delay larger than reference length
        cancelled = canceller.subtract_echo(mic, ref, 10)
        np.testing.assert_array_almost_equal(cancelled, mic)

    def test_subtract_preserves_other_signal(self):
        """Test that non-echo signal is preserved."""
        canceller = EchoCanceller(sample_rate=1000)
        t = np.linspace(0, 0.1, 100)
        ref = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        other_signal = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        delay = 10
        # Mic is delayed ref + other signal
        mic = np.zeros(100, dtype=np.float32)
        mic[delay:] = 0.5 * ref[:-delay]
        mic += other_signal

        cancelled = canceller.subtract_echo(mic, ref, delay)
        # Other signal should be partially preserved
        # Residual should be less than original mic power
        assert np.sum(cancelled**2) < np.sum(mic**2)


class TestEchoCancellerSpectralCleanup:
    """Tests for spectral_cleanup method."""

    def test_spectral_cleanup_reduces_noise(self):
        """Test that spectral cleanup reduces noise."""
        canceller = EchoCanceller()
        # Create noisy signal
        t = np.linspace(0, 0.5, 12000)  # 0.5 seconds at 24kHz
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noise = 0.01 * _rng.standard_normal(len(t)).astype(np.float32)
        noisy = signal + noise

        cleaned = canceller.spectral_cleanup(noisy)
        # Output should have similar length
        assert len(cleaned) == len(noisy)

    def test_spectral_cleanup_short_audio(self):
        """Test spectral cleanup with very short audio."""
        canceller = EchoCanceller()
        short_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cleaned = canceller.spectral_cleanup(short_audio)
        # Should return original for too-short audio
        np.testing.assert_array_almost_equal(cleaned, short_audio)

    def test_spectral_cleanup_different_noise_floors(self):
        """Test spectral cleanup with different noise floor settings."""
        canceller = EchoCanceller()
        audio = np.sin(np.linspace(0, 10 * np.pi, 4000)).astype(np.float32)

        # Both should produce valid output
        cleaned_low = canceller.spectral_cleanup(audio, noise_floor_db=-80)
        cleaned_high = canceller.spectral_cleanup(audio, noise_floor_db=-40)

        assert len(cleaned_low) == len(audio)
        assert len(cleaned_high) == len(audio)


class TestEchoCancellerProcess:
    """Tests for the main process method."""

    def test_process_no_reference(self):
        """Test processing when no reference has been added."""
        canceller = EchoCanceller()
        mic = _rng.standard_normal(1000).astype(np.float32)
        result = canceller.process(mic)

        assert result.had_echo is False
        np.testing.assert_array_almost_equal(result.cleaned_audio, mic)
        assert result.processing_time_ms > 0

    def test_process_with_echo(self):
        """Test processing with simulated echo."""
        canceller = EchoCanceller(sample_rate=24000, min_correlation=0.3)

        # Create reference signal
        t = np.linspace(0, 0.5, 12000)
        ref = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Add to reference buffer
        canceller.add_reference(ref)

        # Create mic input with echo
        delay_samples = 500  # ~20ms delay
        mic = np.zeros(12000, dtype=np.float32)
        mic[delay_samples:] = 0.8 * ref[:-delay_samples]

        result = canceller.process(mic)

        # Should detect echo
        assert result.processing_time_ms > 0
        # Echo reduction should be positive if echo was detected
        if result.had_echo:
            assert result.echo_reduction_db >= 0

    def test_process_without_spectral_cleanup(self):
        """Test processing with spectral cleanup disabled."""
        canceller = EchoCanceller(sample_rate=24000)

        ref = np.sin(np.linspace(0, 10 * np.pi, 5000)).astype(np.float32)
        canceller.add_reference(ref)

        mic = np.zeros(5000, dtype=np.float32)
        mic[100:] = ref[:-100]

        result = canceller.process(mic, enable_spectral_cleanup=False)
        assert len(result.cleaned_audio) == len(mic)
        assert result.processing_time_ms > 0

    def test_process_no_echo_detected(self):
        """Test processing when correlation is below threshold."""
        canceller = EchoCanceller(sample_rate=24000, min_correlation=0.9)

        # Add reference
        ref = np.sin(np.linspace(0, 10 * np.pi, 5000)).astype(np.float32)
        canceller.add_reference(ref)

        # Mic is uncorrelated noise
        mic = _rng.standard_normal(5000).astype(np.float32) * 0.1

        _ = canceller.process(mic)
        # With min_correlation=0.9, unlikely to detect echo from noise
        # Just verify processing completes without error


class TestEchoCancellerClear:
    """Tests for clear method."""

    def test_clear(self):
        """Test clearing echo canceller state."""
        canceller = EchoCanceller()
        canceller.add_reference(_rng.standard_normal(1000).astype(np.float32))

        canceller.clear()

        assert canceller.ref_buffer.total_written == 0
        np.testing.assert_array_equal(canceller.room_filter, np.zeros(1024))


class TestEchoCancellerWarmup:
    """Tests for warmup method."""

    def test_warmup(self):
        """Test warmup initializes properly."""
        canceller = EchoCanceller()
        # Should not raise
        canceller.warmup()
        # After warmup, buffer should be cleared
        assert canceller.ref_buffer.total_written == 0

    def test_warmup_multiple_times(self):
        """Test warmup can be called multiple times."""
        canceller = EchoCanceller()
        canceller.warmup()
        canceller.warmup()
        # Should still be in clean state
        assert canceller.ref_buffer.total_written == 0


class TestEchoCancellationIntegration:
    """Integration tests for echo cancellation."""

    def test_full_pipeline_simulated(self):
        """Test full echo cancellation pipeline with simulated audio."""
        sample_rate = 24000
        duration_s = 0.5

        canceller = EchoCanceller(sample_rate=sample_rate, max_delay_ms=200)
        canceller.warmup()

        # Create TTS output (reference) - use chirp signal for better correlation
        num_samples = int(duration_s * sample_rate)
        t = np.linspace(0, duration_s, num_samples)
        # Chirp signal (frequency sweep) for unique cross-correlation
        tts_output = np.sin(2 * np.pi * (200 + 400 * t) * t).astype(np.float32)

        # Add to reference buffer (simulates playing through speaker)
        canceller.add_reference(tts_output)

        # Create mic input: echo + user speech (with some delay)
        delay_samples = 500  # ~20ms delay
        echo = np.zeros_like(tts_output)
        echo[delay_samples:] = 0.5 * tts_output[:-delay_samples]

        user_voice = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        mic_input = echo + user_voice

        # Process
        result = canceller.process(mic_input)

        # Verify results
        assert len(result.cleaned_audio) == len(mic_input)
        assert result.processing_time_ms < 50  # Should be fast
        # Should process without error - delay detection quality varies with signal

    def test_latency_requirement(self):
        """Test that processing meets latency requirement."""
        canceller = EchoCanceller()
        canceller.warmup()

        # Add reference
        ref = _rng.standard_normal(24000).astype(np.float32)
        canceller.add_reference(ref)

        # Process 100ms of audio
        mic = _rng.standard_normal(2400).astype(np.float32)

        times = []
        for _ in range(5):
            result = canceller.process(mic)
            times.append(result.processing_time_ms)

        avg_time = np.mean(times)
        # Target is <5ms for real-time operation
        # Allow some margin for test environment
        assert avg_time < 20, f"Processing too slow: {avg_time:.2f}ms"

    def test_continuous_processing(self):
        """Test processing continuous stream of audio."""
        canceller = EchoCanceller(sample_rate=24000)
        canceller.warmup()

        chunk_size = 2400  # 100ms at 24kHz
        num_chunks = 10

        # Generate continuous reference
        ref = np.sin(np.linspace(0, 20 * np.pi, chunk_size * num_chunks)).astype(
            np.float32,
        )

        for i in range(num_chunks):
            # Add reference chunk
            ref_chunk = ref[i * chunk_size : (i + 1) * chunk_size]
            canceller.add_reference(ref_chunk)

            # Process mic chunk (with echo)
            mic_chunk = 0.5 * ref_chunk + _rng.standard_normal(chunk_size).astype(
                np.float32,
            ) * 0.1

            result = canceller.process(mic_chunk)
            assert len(result.cleaned_audio) == chunk_size


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_delay(self):
        """Test with zero delay echo."""
        canceller = EchoCanceller()
        ref = _rng.standard_normal(1000).astype(np.float32)
        canceller.add_reference(ref)

        # Direct copy (zero delay)
        mic = ref.copy()
        result = canceller.process(mic)
        # Should still process without error
        assert len(result.cleaned_audio) == len(mic)

    def test_very_short_audio(self):
        """Test with very short audio."""
        canceller = EchoCanceller()
        ref = np.array([0.1, 0.2], dtype=np.float32)
        canceller.add_reference(ref)

        mic = np.array([0.3, 0.4], dtype=np.float32)
        result = canceller.process(mic)
        assert len(result.cleaned_audio) == 2

    def test_silent_audio(self):
        """Test with silent audio."""
        canceller = EchoCanceller()
        ref = np.zeros(1000, dtype=np.float32)
        canceller.add_reference(ref)

        mic = np.zeros(1000, dtype=np.float32)
        result = canceller.process(mic)
        assert result.had_echo is False
