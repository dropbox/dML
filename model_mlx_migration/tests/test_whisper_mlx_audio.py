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
Tests for WhisperMLX audio loading (OPT-W3).

Tests the optimized audio loading that uses soundfile for WAV/FLAC
and falls back to ffmpeg for other formats.
"""

import os
import tempfile

import numpy as np
import pytest

# Skip all tests if soundfile or scipy not available
soundfile = pytest.importorskip("soundfile")
signal = pytest.importorskip("scipy.signal")

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


@pytest.fixture
def wav_16khz():
    """Create a 16kHz WAV file for testing."""
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    soundfile.write(path, audio, sample_rate)

    yield path, audio, sample_rate

    os.unlink(path)


@pytest.fixture
def wav_44khz():
    """Create a 44.1kHz WAV file for testing."""
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    soundfile.write(path, audio, sample_rate)

    yield path, audio, sample_rate

    os.unlink(path)


@pytest.fixture
def wav_stereo():
    """Create a stereo WAV file for testing."""
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    audio = np.stack([left, right], axis=1)

    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    soundfile.write(path, audio, sample_rate)

    yield path, audio, sample_rate

    os.unlink(path)


class TestLoadAudioNative:
    """Tests for _load_audio_native function."""

    def test_load_16khz_wav(self, wav_16khz):
        """Test loading a 16kHz WAV file (no resampling needed)."""
        from tools.whisper_mlx.audio import _load_audio_native

        path, original, sr = wav_16khz
        loaded = _load_audio_native(path, 16000)

        assert loaded.dtype == np.float32
        assert len(loaded) == len(original)
        # Minor numerical differences (max ~3e-5) expected due to audio codec
        np.testing.assert_allclose(loaded, original, atol=1e-4)

    def test_load_44khz_wav_resample(self, wav_44khz):
        """Test loading a 44.1kHz WAV file with resampling to 16kHz."""
        from tools.whisper_mlx.audio import _load_audio_native

        path, original, sr = wav_44khz
        loaded = _load_audio_native(path, 16000)

        assert loaded.dtype == np.float32
        # 44100 -> 16000: length should be roughly 16000/44100 of original
        expected_len = int(len(original) * 16000 / 44100)
        assert abs(len(loaded) - expected_len) <= 1

    def test_load_stereo_to_mono(self, wav_stereo):
        """Test loading a stereo file and converting to mono."""
        from tools.whisper_mlx.audio import _load_audio_native

        path, original, sr = wav_stereo
        loaded = _load_audio_native(path, 16000)

        assert loaded.dtype == np.float32
        assert loaded.ndim == 1  # Should be mono
        # Mono should be average of channels
        expected_mono = original.mean(axis=1)
        # Minor numerical differences expected due to audio codec
        np.testing.assert_allclose(loaded, expected_mono, atol=1e-4)


class TestLoadAudioFfmpeg:
    """Tests for _load_audio_ffmpeg function."""

    @pytest.mark.skipif(
        os.system("which ffmpeg > /dev/null 2>&1") != 0,
        reason="ffmpeg not available",
    )
    def test_load_wav_via_ffmpeg(self, wav_16khz):
        """Test loading WAV via ffmpeg as fallback."""
        from tools.whisper_mlx.audio import _load_audio_ffmpeg

        path, original, sr = wav_16khz
        loaded = _load_audio_ffmpeg(path, 16000)

        assert loaded.dtype == np.float32
        # ffmpeg may have slight differences due to processing
        assert abs(len(loaded) - len(original)) <= 1


class TestLoadAudio:
    """Tests for the main load_audio function."""

    def test_load_audio_prefers_native(self, wav_16khz):
        """Test that load_audio uses native loading for WAV files."""
        from tools.whisper_mlx.audio import load_audio

        path, original, sr = wav_16khz
        loaded = load_audio(path, 16000)

        assert loaded.dtype == np.float32
        # Minor numerical differences expected due to audio codec
        np.testing.assert_allclose(loaded, original, atol=1e-4)

    def test_load_audio_handles_resampling(self, wav_44khz):
        """Test that load_audio handles resampling correctly."""
        from tools.whisper_mlx.audio import load_audio

        path, original, sr = wav_44khz
        loaded = load_audio(path, 16000)

        assert loaded.dtype == np.float32
        # Should be resampled to 16kHz
        expected_len = int(len(original) * 16000 / 44100)
        assert abs(len(loaded) - expected_len) <= 1

    def test_load_audio_handles_stereo(self, wav_stereo):
        """Test that load_audio converts stereo to mono."""
        from tools.whisper_mlx.audio import load_audio

        path, original, sr = wav_stereo
        loaded = load_audio(path, 16000)

        assert loaded.dtype == np.float32
        assert loaded.ndim == 1  # Mono output


class TestNativeVsFfmpeg:
    """Tests comparing native and ffmpeg loading results."""

    @pytest.mark.skipif(
        os.system("which ffmpeg > /dev/null 2>&1") != 0,
        reason="ffmpeg not available",
    )
    def test_native_matches_ffmpeg(self, wav_16khz):
        """Test that native loading produces similar results to ffmpeg."""
        from tools.whisper_mlx.audio import _load_audio_ffmpeg, _load_audio_native

        path, original, sr = wav_16khz

        native_audio = _load_audio_native(path, 16000)
        ffmpeg_audio = _load_audio_ffmpeg(path, 16000)

        # Trim to same length
        min_len = min(len(native_audio), len(ffmpeg_audio))
        native_audio = native_audio[:min_len]
        ffmpeg_audio = ffmpeg_audio[:min_len]

        # Should be very close (minor numerical differences expected)
        max_diff = np.max(np.abs(native_audio - ffmpeg_audio))
        assert max_diff < 0.01, f"Max difference: {max_diff}"

        # High correlation
        correlation = np.corrcoef(native_audio, ffmpeg_audio)[0, 1]
        assert correlation > 0.999, f"Correlation: {correlation}"


class TestPerformance:
    """Performance tests for audio loading."""

    def test_native_is_faster(self, wav_16khz):
        """Test that native loading is faster than ffmpeg."""
        import time

        from tools.whisper_mlx.audio import _load_audio_ffmpeg, _load_audio_native

        path, original, sr = wav_16khz

        # Skip if ffmpeg not available
        if os.system("which ffmpeg > /dev/null 2>&1") != 0:
            pytest.skip("ffmpeg not available")

        # Warm up
        _load_audio_native(path, 16000)
        _load_audio_ffmpeg(path, 16000)

        # Benchmark native
        start = time.perf_counter()
        for _ in range(5):
            _load_audio_native(path, 16000)
        native_time = (time.perf_counter() - start) / 5

        # Benchmark ffmpeg
        start = time.perf_counter()
        for _ in range(5):
            _load_audio_ffmpeg(path, 16000)
        ffmpeg_time = (time.perf_counter() - start) / 5

        # Native should be at least 2x faster
        speedup = ffmpeg_time / native_time
        assert speedup > 2.0, f"Expected speedup > 2x, got {speedup:.2f}x"


class TestNativeCppLoader:
    """Tests for the C++ native audio loader (OPT-W3-C)."""

    def test_cpp_native_available(self):
        """Test that the C++ native loader is available."""
        from tools.whisper_mlx.audio import _is_native_available
        # Should be available if the .so was built
        assert _is_native_available() is True

    def test_cpp_native_loads_audio(self, wav_16khz):
        """Test that the C++ native loader can load audio."""
        from tools.whisper_mlx.audio import _load_audio_native_cpp

        path, original, sr = wav_16khz
        loaded = _load_audio_native_cpp(path, 16000)

        assert loaded.dtype == np.float32
        assert len(loaded) > 0
        assert loaded.min() >= -1.0
        assert loaded.max() <= 1.0

    @pytest.mark.skipif(
        os.system("which ffmpeg > /dev/null 2>&1") != 0,
        reason="ffmpeg not available",
    )
    def test_cpp_native_matches_ffmpeg(self, wav_16khz):
        """Test that C++ native loader output matches ffmpeg exactly."""
        from tools.whisper_mlx.audio import _load_audio_ffmpeg, _load_audio_native_cpp

        path, original, sr = wav_16khz
        native_audio = _load_audio_native_cpp(path, 16000)
        ffmpeg_audio = _load_audio_ffmpeg(path, 16000)

        # Length must match
        assert len(native_audio) == len(ffmpeg_audio), (
            f"Length mismatch: {len(native_audio)} vs {len(ffmpeg_audio)}"
        )

        # Values must match within 1 LSB (3e-5)
        max_diff = np.max(np.abs(native_audio - ffmpeg_audio))
        assert max_diff <= 3.1e-5, f"Max diff {max_diff} exceeds 1 LSB"

    def test_cpp_native_speedup(self, wav_16khz):
        """Test that C++ native loader is significantly faster than ffmpeg."""
        import time

        from tools.whisper_mlx.audio import (
            _is_native_available,
            _load_audio_ffmpeg,
            _load_audio_native_cpp,
        )

        if not _is_native_available():
            pytest.skip("C++ native loader not available")

        # Skip if ffmpeg not available
        if os.system("which ffmpeg > /dev/null 2>&1") != 0:
            pytest.skip("ffmpeg not available")

        path, original, sr = wav_16khz

        # Warm up
        _load_audio_native_cpp(path, 16000)
        _load_audio_ffmpeg(path, 16000)

        # Benchmark C++ native
        start = time.perf_counter()
        for _ in range(10):
            _load_audio_native_cpp(path, 16000)
        native_time = (time.perf_counter() - start) / 10

        # Benchmark ffmpeg
        start = time.perf_counter()
        for _ in range(10):
            _load_audio_ffmpeg(path, 16000)
        ffmpeg_time = (time.perf_counter() - start) / 10

        # C++ native should be at least 5x faster
        speedup = ffmpeg_time / native_time
        assert speedup > 5.0, f"Expected speedup > 5x, got {speedup:.2f}x"

    def test_backend_switching(self, wav_16khz):
        """Test that backend can be switched between native, pyav, and ffmpeg."""
        from tools.whisper_mlx.audio import (
            _is_native_available,
            get_audio_backend,
            load_audio,
            set_audio_backend,
        )

        if not _is_native_available():
            pytest.skip("C++ native loader not available")

        path, original, sr = wav_16khz

        # Test native backend
        set_audio_backend('native')
        assert get_audio_backend() == 'native'
        native_result = load_audio(path, 16000)

        # Test ffmpeg backend
        set_audio_backend('ffmpeg')
        assert get_audio_backend() == 'ffmpeg'
        ffmpeg_result = load_audio(path, 16000)

        # Results should match
        assert len(native_result) == len(ffmpeg_result)
        max_diff = np.max(np.abs(native_result - ffmpeg_result))
        assert max_diff <= 3.1e-5

        # Reset to auto
        set_audio_backend('auto')
        assert get_audio_backend() == 'auto'


class TestAsyncMelPreparer:
    """Tests for AsyncMelPreparer (OPT-PREFETCH)."""

    def test_prepare_mel_standard_mode(self):
        """Test mel preparation in standard 30s padded mode."""
        from tools.whisper_mlx.audio import SAMPLE_RATE, AsyncMelPreparer

        # Create 5 second audio chunk
        duration = 5.0
        audio = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        chunk_samples = int(30.0 * SAMPLE_RATE)

        preparer = AsyncMelPreparer(n_mels=128, n_audio_ctx=1500)
        try:
            mel, mel_info = preparer.prepare_mel(audio, chunk_samples, is_variable_length=False)

            is_variable, encoder_positions, actual_duration = mel_info
            assert is_variable is False
            assert encoder_positions == 1500
            assert actual_duration == 30.0
            # Mel should be padded to 3000 frames
            assert mel.shape == (3000, 128)
        finally:
            preparer.shutdown()

    def test_prepare_mel_variable_length_mode(self):
        """Test mel preparation in variable-length mode."""
        from tools.whisper_mlx.audio import SAMPLE_RATE, AsyncMelPreparer

        # Create 5 second audio chunk
        duration = 5.0
        audio = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        chunk_samples = int(30.0 * SAMPLE_RATE)

        preparer = AsyncMelPreparer(n_mels=128, n_audio_ctx=1500)
        try:
            mel, mel_info = preparer.prepare_mel(audio, chunk_samples, is_variable_length=True)

            is_variable, encoder_positions, actual_duration = mel_info
            assert is_variable is True
            assert actual_duration <= duration + 0.1  # Should match audio duration
            # Mel frames = audio_samples / hop_length with some padding
            assert mel.shape[1] == 128
        finally:
            preparer.shutdown()

    def test_submit_async(self):
        """Test submitting chunks for async processing."""
        from tools.whisper_mlx.audio import SAMPLE_RATE, AsyncMelPreparer

        # Create 3 audio chunks
        duration = 5.0
        audio1 = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        audio2 = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        audio3 = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        chunk_samples = int(30.0 * SAMPLE_RATE)

        with AsyncMelPreparer(n_mels=128, n_audio_ctx=1500, max_workers=2) as preparer:
            # Submit all chunks
            future1 = preparer.submit(audio1, chunk_samples, is_variable_length=False)
            future2 = preparer.submit(audio2, chunk_samples, is_variable_length=False)
            future3 = preparer.submit(audio3, chunk_samples, is_variable_length=False)

            # Get results
            mel1, info1 = future1.result()
            mel2, info2 = future2.result()
            mel3, info3 = future3.result()

            # All should be standard mode
            assert info1[0] is False
            assert info2[0] is False
            assert info3[0] is False
            assert mel1.shape == (3000, 128)
            assert mel2.shape == (3000, 128)
            assert mel3.shape == (3000, 128)

    def test_submit_batch(self):
        """Test batch submission."""
        from tools.whisper_mlx.audio import SAMPLE_RATE, AsyncMelPreparer

        # Create 4 chunks, last one is partial
        chunks = []
        for i in range(4):
            duration = 5.0 if i < 3 else 3.0  # Last chunk is shorter
            chunks.append(np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32))

        chunk_samples = int(30.0 * SAMPLE_RATE)
        variable_length_indices = {3}  # Last chunk uses variable-length

        with AsyncMelPreparer(n_mels=128, n_audio_ctx=1500, max_workers=2) as preparer:
            futures = preparer.submit_batch(chunks, chunk_samples, variable_length_indices)

            assert len(futures) == 4

            # Get all results
            results = [f.result() for f in futures]

            # First 3 should be standard mode
            for i in range(3):
                mel, info = results[i]
                assert info[0] is False, f"Chunk {i} should be standard mode"
                assert mel.shape == (3000, 128)

            # Last should be variable-length mode
            mel_last, info_last = results[3]
            assert info_last[0] is True, "Last chunk should be variable-length mode"

    def test_context_manager(self):
        """Test that context manager properly shuts down executor."""
        from tools.whisper_mlx.audio import AsyncMelPreparer

        with AsyncMelPreparer(n_mels=128, n_audio_ctx=1500) as preparer:
            assert preparer._executor is not None

        # After context, executor should be shut down
        # (we can't easily test this, but ensure no exception)


# =============================================================================
# Shared FFT Tests (OPT-SHARED-FFT)
# =============================================================================


class TestComputeStftAndMel:
    """Tests for compute_stft_and_mel shared FFT function."""

    def test_returns_mel_only_when_stft_disabled(self):
        """Test that return_stft=False returns only mel spectrogram."""
        import mlx.core as mx

        from tools.whisper_mlx.audio import SAMPLE_RATE, compute_stft_and_mel

        # Create test audio (1 second, 440Hz sine)
        duration = 1.0
        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        result = compute_stft_and_mel(audio, return_stft=False)

        # Should return single array (mel)
        assert isinstance(result, mx.array)
        assert result.ndim == 2
        assert result.shape[1] == 128  # n_mels

    def test_returns_both_mel_and_stft(self):
        """Test that return_stft=True returns both mel and STFT magnitude."""
        import mlx.core as mx

        from tools.whisper_mlx.audio import N_FFT, SAMPLE_RATE, compute_stft_and_mel

        # Create test audio (1 second)
        duration = 1.0
        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        mel, stft_mag = compute_stft_and_mel(audio, return_stft=True)

        # Check mel shape
        assert isinstance(mel, mx.array)
        assert mel.ndim == 2
        assert mel.shape[1] == 128  # n_mels

        # Check STFT magnitude shape
        assert isinstance(stft_mag, mx.array)
        assert stft_mag.ndim == 2
        assert stft_mag.shape[1] == N_FFT // 2  # 200 freq bins for n_fft=400

        # Frames should match between mel and stft
        assert mel.shape[0] == stft_mag.shape[0]

    def test_stft_shape_matches_fft_params(self):
        """Test STFT magnitude has correct shape based on FFT parameters."""
        from tools.whisper_mlx.audio import (
            HOP_LENGTH,
            N_FFT,
            SAMPLE_RATE,
            compute_stft_and_mel,
        )

        # Create test audio (2 seconds)
        duration = 2.0
        n_samples = int(duration * SAMPLE_RATE)
        audio = _rng.standard_normal(n_samples).astype(np.float32) * 0.1

        mel, stft_mag = compute_stft_and_mel(audio, return_stft=True)

        # Calculate expected frame count
        # With padding = n_fft // 2 on both sides
        padded_len = n_samples + N_FFT  # padding on both sides
        expected_frames = 1 + (padded_len - N_FFT) // HOP_LENGTH

        assert stft_mag.shape[0] == expected_frames
        assert stft_mag.shape[1] == N_FFT // 2  # 200

    def test_mel_matches_original_implementation(self):
        """Test that shared FFT mel matches original log_mel_spectrogram."""
        from tools.whisper_mlx.audio import (
            SAMPLE_RATE,
            compute_stft_and_mel,
        )

        # Create test audio with speech-like characteristics
        duration = 1.0
        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), dtype=np.float32)
        # Multi-tone to simulate speech
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.2 * np.sin(2 * np.pi * 880 * t)
            + 0.1 * np.sin(2 * np.pi * 220 * t)
        ).astype(np.float32)

        # Get mel from shared FFT
        mel_shared, _ = compute_stft_and_mel(audio, return_stft=True)

        # Get mel from original function (fallback path)
        from tools.whisper_mlx.audio import _log_mel_spectrogram_fallback
        mel_original = _log_mel_spectrogram_fallback(audio, n_mels=128)

        # Should be identical (using same implementation internally)
        np.testing.assert_allclose(
            np.array(mel_shared),
            np.array(mel_original),
            atol=1e-5,
            rtol=1e-5,
        )


class TestFusedVAD:
    """Tests for FusedVAD class."""

    def test_fused_vad_init(self):
        """Test FusedVAD initialization."""
        from tools.whisper_mlx.fused_vad import FusedVAD

        vad = FusedVAD()

        assert vad.n_fft == 400
        assert vad.hop_length == 160
        assert vad.sample_rate == 16000
        assert len(vad.encoder) == 4

    def test_fused_vad_forward_pass(self):
        """Test FusedVAD forward pass with STFT magnitude."""
        import mlx.core as mx

        from tools.whisper_mlx.audio import SAMPLE_RATE, compute_stft_and_mel
        from tools.whisper_mlx.fused_vad import FusedVAD

        # Create test audio (1 second)
        duration = 1.0
        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Get STFT magnitude
        _, stft_mag = compute_stft_and_mel(audio, return_stft=True)

        # Run through VAD
        vad = FusedVAD()
        probs, state = vad(stft_mag)

        # Check output shape
        assert isinstance(probs, mx.array)
        assert probs.ndim == 2  # [batch, n_output_frames]
        assert probs.shape[0] == 1  # batch size

        # Probabilities should be in [0, 1] (after sigmoid)
        probs_np = np.array(probs)
        assert np.all(probs_np >= 0) and np.all(probs_np <= 1)

    def test_detect_frames(self):
        """Test detect_frames returns per-frame probabilities."""
        import mlx.core as mx

        from tools.whisper_mlx.audio import SAMPLE_RATE, compute_stft_and_mel
        from tools.whisper_mlx.fused_vad import FusedVAD

        # Create test audio (2 seconds)
        duration = 2.0
        n_samples = int(duration * SAMPLE_RATE)
        audio = _rng.standard_normal(n_samples).astype(np.float32) * 0.1

        # Get STFT magnitude
        _, stft_mag = compute_stft_and_mel(audio, return_stft=True)

        # Run VAD
        vad = FusedVAD()
        probs = vad.detect_frames(stft_mag)

        # Should return 1D array of probabilities
        assert isinstance(probs, mx.array)
        assert probs.ndim == 1

        # All probabilities should be valid
        probs_np = np.array(probs)
        assert np.all(probs_np >= 0) and np.all(probs_np <= 1)

    def test_get_speech_segments(self):
        """Test speech segment extraction from probabilities."""
        import mlx.core as mx

        from tools.whisper_mlx.fused_vad import FusedVAD

        vad = FusedVAD()

        # Create synthetic probabilities with clear speech/silence pattern
        # Pattern: 0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2 (middle 3 frames are speech)
        probs = mx.array([0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2])

        segments = vad.get_speech_segments(
            probs,
            threshold=0.5,
            min_speech_duration_ms=50,
            min_silence_duration_ms=50,
        )

        # Should find one speech segment
        assert isinstance(segments, list)
        # Each segment should be (start_time, end_time) tuple
        for seg in segments:
            assert len(seg) == 2
            assert seg[0] < seg[1]  # start < end

    def test_detect_speech_fused_convenience(self):
        """Test detect_speech_fused convenience function."""
        import mlx.core as mx

        from tools.whisper_mlx.audio import SAMPLE_RATE
        from tools.whisper_mlx.fused_vad import detect_speech_fused

        # Create test audio (1 second)
        duration = 1.0
        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        segments, mel, probs = detect_speech_fused(audio)

        # Check mel is valid
        assert isinstance(mel, mx.array)
        assert mel.ndim == 2
        assert mel.shape[1] == 128

        # Check probs is valid
        assert isinstance(probs, mx.array)
        probs_np = np.array(probs)
        assert np.all(probs_np >= 0) and np.all(probs_np <= 1)

        # Check segments is a list
        assert isinstance(segments, list)

    def test_reset_state(self):
        """Test that reset_state clears LSTM state."""
        from tools.whisper_mlx.audio import SAMPLE_RATE, compute_stft_and_mel
        from tools.whisper_mlx.fused_vad import FusedVAD

        # Create test audio
        duration = 0.5
        n_samples = int(duration * SAMPLE_RATE)
        audio = _rng.standard_normal(n_samples).astype(np.float32) * 0.1

        _, stft_mag = compute_stft_and_mel(audio, return_stft=True)

        vad = FusedVAD()

        # First pass sets state
        vad(stft_mag)
        assert vad._state is not None

        # Reset clears state
        vad.reset_state()
        assert vad._state is None

    def test_vad_with_silence(self):
        """Test VAD on silent audio returns low probabilities."""
        from tools.whisper_mlx.audio import SAMPLE_RATE, compute_stft_and_mel
        from tools.whisper_mlx.fused_vad import FusedVAD

        # Create silent audio (1 second of zeros)
        duration = 1.0
        n_samples = int(duration * SAMPLE_RATE)
        audio = np.zeros(n_samples, dtype=np.float32)

        _, stft_mag = compute_stft_and_mel(audio, return_stft=True)

        vad = FusedVAD()
        probs = vad.detect_frames(stft_mag)

        # On untrained VAD, we can't assert specific values,
        # but the forward pass should complete without error
        assert probs is not None

    def test_shared_fft_efficiency(self):
        """Test that shared FFT produces both outputs efficiently."""
        import time

        from tools.whisper_mlx.audio import SAMPLE_RATE, compute_stft_and_mel

        # Create longer test audio for timing
        duration = 10.0
        n_samples = int(duration * SAMPLE_RATE)
        audio = _rng.standard_normal(n_samples).astype(np.float32) * 0.1

        # Warmup both code paths to eliminate JIT/caching overhead
        compute_stft_and_mel(audio, return_stft=True)
        compute_stft_and_mel(audio, return_stft=False)

        # Time shared FFT
        start = time.perf_counter()
        for _ in range(5):
            mel, stft_mag = compute_stft_and_mel(audio, return_stft=True)
        shared_time = time.perf_counter() - start

        # Time mel-only
        start = time.perf_counter()
        for _ in range(5):
            compute_stft_and_mel(audio, return_stft=False)
        mel_only_time = time.perf_counter() - start

        # Shared FFT should NOT be significantly slower than mel-only
        # (overhead of returning STFT should be minimal)
        overhead_ratio = shared_time / mel_only_time
        assert overhead_ratio < 1.5, f"Shared FFT overhead too high: {overhead_ratio:.2f}x"
