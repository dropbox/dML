# Copyright 2024-2026 Andrew Yates
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
Tests for audio preprocessing pipeline.

Target: <15ms total preprocessing latency.
"""

import numpy as np
import pytest

from src.models.preprocessing import (
    AudioChunk,
    PreprocessingConfig,
    PreprocessingPipeline,
    VADResult,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


# Helper functions (must be defined before use in skipif decorators)
def _has_torchaudio() -> bool:
    """Check if torchaudio is available."""
    try:
        import torch  # noqa: F401 - checking availability
        import torchaudio  # noqa: F401 - checking availability
        return True
    except ImportError:
        return False


def _has_silero_vad() -> bool:
    """Check if silero-vad is available."""
    try:
        import silero_vad  # noqa: F401 - checking availability
        return True
    except ImportError:
        try:
            import torch
            torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
            return True
        except Exception:
            return False


class TestDCRemoval:
    """Test DC offset removal."""

    def test_removes_dc_offset(self):
        """DC removal should center audio around zero."""
        pipeline = PreprocessingPipeline()

        # Create audio with DC offset
        audio = np.sin(np.linspace(0, 10 * np.pi, 16000)) + 0.5
        assert np.abs(np.mean(audio)) > 0.4  # Has DC offset

        # Remove DC
        cleaned = pipeline.remove_dc(audio)
        assert np.abs(np.mean(cleaned)) < 1e-10  # No DC offset

    def test_preserves_signal(self):
        """DC removal should preserve signal shape."""
        pipeline = PreprocessingPipeline()

        audio = np.sin(np.linspace(0, 10 * np.pi, 16000))
        cleaned = pipeline.remove_dc(audio)

        # Shape should be preserved
        assert cleaned.shape == audio.shape

        # Signal should correlate highly
        correlation = np.corrcoef(audio, cleaned)[0, 1]
        assert correlation > 0.99

    def test_disabled_when_configured(self):
        """DC removal can be disabled."""
        config = PreprocessingConfig(enable_dc_removal=False)
        pipeline = PreprocessingPipeline(config)

        audio = np.sin(np.linspace(0, 10 * np.pi, 16000)) + 0.5
        result = pipeline.remove_dc(audio)

        # Should be unchanged
        np.testing.assert_array_equal(result, audio)


class TestLUFSNormalization:
    """Test LUFS loudness normalization."""

    def test_computes_reasonable_lufs(self):
        """LUFS calculation should return reasonable values."""
        pipeline = PreprocessingPipeline()

        # Generate 1 second of -6dBFS sine wave
        t = np.linspace(0, 1, 16000)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # -6dB amplitude

        lufs = pipeline.compute_lufs(audio, 16000)

        # Should be between -30 and 0 LUFS for this signal
        # (Simplified LUFS calculation may vary from ITU-R BS.1770-4)
        assert -30 < lufs < 0

    def test_silence_returns_low_lufs(self):
        """Silence should return very low LUFS."""
        pipeline = PreprocessingPipeline()

        silence = np.zeros(16000)
        lufs = pipeline.compute_lufs(silence, 16000)

        assert lufs <= -60

    def test_normalization_adjusts_loudness(self):
        """LUFS normalization should adjust loudness."""
        pipeline = PreprocessingPipeline()

        # Quiet audio
        t = np.linspace(0, 1, 16000)
        audio = 0.01 * np.sin(2 * np.pi * 440 * t)

        original_lufs = pipeline.compute_lufs(audio, 16000)
        normalized = pipeline.normalize_lufs(audio, 16000, target_lufs=-23.0)
        new_lufs = pipeline.compute_lufs(normalized, 16000)

        # Should be closer to target
        assert abs(new_lufs - (-23.0)) < abs(original_lufs - (-23.0))


class TestResampling:
    """Test audio resampling."""

    @pytest.mark.skipif(
        not _has_torchaudio(),
        reason="torchaudio required for resampling",
    )
    def test_48k_to_16k(self):
        """Resample 48kHz to 16kHz."""
        pipeline = PreprocessingPipeline()

        # 1 second at 48kHz
        audio_48k = np.sin(np.linspace(0, 10 * np.pi, 48000)).astype(np.float32)

        resampled = pipeline.resample(audio_48k, 48000, 16000)

        # Should be 1 second at 16kHz
        assert len(resampled) == 16000

    @pytest.mark.skipif(
        not _has_torchaudio(),
        reason="torchaudio required for resampling",
    )
    def test_same_rate_passthrough(self):
        """Same sample rate should return original."""
        pipeline = PreprocessingPipeline()

        audio = np.sin(np.linspace(0, 10 * np.pi, 16000)).astype(np.float32)
        result = pipeline.resample(audio, 16000, 16000)

        np.testing.assert_array_equal(result, audio)


class TestChunking:
    """Test audio chunking for streaming."""

    def test_chunks_correct_size(self):
        """Chunks should have correct size."""
        config = PreprocessingConfig(chunk_size_ms=320)
        pipeline = PreprocessingPipeline(config)

        # 1 second audio
        audio = _rng.standard_normal(16000).astype(np.float32)

        chunks = list(pipeline.chunk_audio(audio, 16000))

        # 320ms = 5120 samples at 16kHz
        expected_samples = int(320 * 16000 / 1000)

        for chunk in chunks[:-1]:  # All but last
            assert len(chunk.samples) == expected_samples

    def test_chunks_cover_all_audio(self):
        """Chunks should cover all audio samples."""
        config = PreprocessingConfig(chunk_size_ms=320, chunk_overlap_ms=0)
        pipeline = PreprocessingPipeline(config)

        audio = _rng.standard_normal(16000).astype(np.float32)
        chunks = list(pipeline.chunk_audio(audio, 16000))

        # Reconstruct audio from chunks
        total_samples = sum(len(c.samples) for c in chunks)
        # May have padding on last chunk
        assert total_samples >= len(audio)

    def test_final_chunk_marked(self):
        """Final chunk should be marked is_final."""
        pipeline = PreprocessingPipeline()

        audio = _rng.standard_normal(16000).astype(np.float32)
        chunks = list(pipeline.chunk_audio(audio, 16000))

        assert chunks[-1].is_final is True
        for chunk in chunks[:-1]:
            assert chunk.is_final is False

    def test_chunk_timing_correct(self):
        """Chunk timing should be correct."""
        config = PreprocessingConfig(chunk_size_ms=320)
        pipeline = PreprocessingPipeline(config)

        audio = _rng.standard_normal(16000).astype(np.float32)
        chunks = list(pipeline.chunk_audio(audio, 16000))

        assert chunks[0].start_time_ms == 0.0
        assert abs(chunks[0].end_time_ms - 320.0) < 1.0

        if len(chunks) > 1:
            assert abs(chunks[1].start_time_ms - 320.0) < 1.0


class TestVAD:
    """Test Voice Activity Detection."""

    @pytest.mark.skipif(
        not _has_silero_vad(),
        reason="silero-vad required for VAD tests",
    )
    def test_vad_returns_valid_result(self):
        """VAD should return valid result structure."""
        pipeline = PreprocessingPipeline()

        # Generate random audio (may or may not be detected as speech)
        t = np.linspace(0, 2, 32000)
        envelope = np.abs(np.sin(2 * np.pi * 2 * t))
        audio = envelope * _rng.standard_normal(32000) * 0.5

        result = pipeline.detect_speech(audio.astype(np.float32), 16000)

        # Should return valid VADResult structure
        assert isinstance(result, VADResult)
        assert isinstance(result.speech_segments, list)
        assert result.total_speech_ms >= 0
        assert result.total_silence_ms >= 0
        # Total should equal audio duration
        assert abs(result.total_speech_ms + result.total_silence_ms - 2000.0) < 1.0

    @pytest.mark.skipif(
        not _has_silero_vad(),
        reason="silero-vad required for VAD tests",
    )
    def test_detects_silence_in_silence(self):
        """VAD should detect silence in silent audio."""
        pipeline = PreprocessingPipeline()

        # Pure silence with very small noise
        audio = _rng.standard_normal(32000) * 0.001

        result = pipeline.detect_speech(audio.astype(np.float32), 16000)

        # Should detect mostly silence
        assert result.total_silence_ms > result.total_speech_ms

    def test_vad_disabled(self):
        """VAD can be disabled."""
        config = PreprocessingConfig(enable_vad=False)
        pipeline = PreprocessingPipeline(config)

        audio = np.zeros(16000)
        result = pipeline.detect_speech(audio.astype(np.float32), 16000)

        # All audio treated as speech when VAD disabled
        assert len(result.speech_segments) == 1
        assert result.total_speech_ms > 0


class TestDenosingDisabled:
    """Test that denoising is disabled (critical requirement)."""

    def test_denoising_disabled_by_default(self):
        """Denoising should be disabled by default."""
        config = PreprocessingConfig()
        assert config.enable_denoising is False

    def test_enabling_denoising_raises_error(self):
        """Enabling denoising should raise error."""
        config = PreprocessingConfig(enable_denoising=True)

        with pytest.raises(ValueError, match="Denoising MUST be disabled"):
            PreprocessingPipeline(config)


class TestFullPipeline:
    """Test full preprocessing pipeline."""

    @pytest.mark.skipif(
        not _has_torchaudio(),
        reason="torchaudio required for full pipeline",
    )
    def test_preprocess_full(self):
        """Full preprocessing should work."""
        pipeline = PreprocessingPipeline()

        # 48kHz input
        audio = np.sin(np.linspace(0, 10 * np.pi, 48000)).astype(np.float32)
        audio += 0.3  # DC offset

        result = pipeline.preprocess(audio, 48000)

        # Should be 16kHz
        assert len(result) == 16000

        # Should have no DC offset
        assert np.abs(np.mean(result)) < 0.01

    @pytest.mark.skipif(
        not _has_torchaudio(),
        reason="torchaudio required for streaming pipeline",
    )
    def test_process_streaming(self):
        """Streaming processing should work."""
        config = PreprocessingConfig(
            enable_vad=False,  # Skip VAD for simpler test
            chunk_size_ms=320,
        )
        pipeline = PreprocessingPipeline(config)

        # 1 second at 48kHz
        audio = np.sin(np.linspace(0, 10 * np.pi, 48000)).astype(np.float32)

        chunks = list(pipeline.process_streaming(audio, 48000))

        # Should produce chunks
        assert len(chunks) > 0

        # All chunks should be AudioChunk objects
        for chunk in chunks:
            assert isinstance(chunk, AudioChunk)
            assert chunk.sample_rate == 16000


class TestLatencyRequirements:
    """Test latency requirements (<15ms total)."""

    @pytest.mark.skipif(
        not _has_torchaudio(),
        reason="torchaudio required for benchmark",
    )
    def test_preprocessing_latency(self):
        """Preprocessing should be <15ms for 1 second audio."""
        pipeline = PreprocessingPipeline(
            PreprocessingConfig(enable_vad=False),  # VAD adds latency
        )

        # 1 second at 16kHz (typical ASR input)
        audio = np.sin(np.linspace(0, 100 * np.pi, 16000)).astype(np.float32)

        results = pipeline.benchmark(audio, 16000, iterations=5)

        # Core preprocessing (without VAD) should be <5ms
        core_latency = (
            results['resample_ms']
            + results['dc_removal_ms']
            + results['lufs_normalize_ms']
        )

        print("\nPreprocessing latency breakdown:")
        print(f"  Resample: {results['resample_ms']:.2f}ms")
        print(f"  DC removal: {results['dc_removal_ms']:.2f}ms")
        print(f"  LUFS normalize: {results['lufs_normalize_ms']:.2f}ms")
        print(f"  Total (no VAD): {core_latency:.2f}ms")

        # Core preprocessing should be very fast
        assert core_latency < 10.0, f"Core preprocessing too slow: {core_latency:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
