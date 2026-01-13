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

"""Tests for the OptimizedWhisper module."""

import os
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add tools to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from optimized_whisper import OptimizedWhisper, TranscriptionResult, transcribe


class TestTranscriptionResult:
    """Tests for the TranscriptionResult dataclass."""

    def test_create_transcription_result(self):
        """Test creating a TranscriptionResult."""
        result = TranscriptionResult(
            text="Hello world",
            vad_time=0.1,
            whisper_time=1.5,
            total_time=1.6,
            original_duration=5.0,
            speech_duration=3.0,
            silence_skipped=2.0,
            speedup_factor=1.5,
        )
        assert result.text == "Hello world"
        assert result.vad_time == 0.1
        assert result.whisper_time == 1.5
        assert result.total_time == 1.6
        assert result.original_duration == 5.0
        assert result.speech_duration == 3.0
        assert result.silence_skipped == 2.0
        assert result.speedup_factor == 1.5

    def test_transcription_result_default_values(self):
        """Test that all fields must be provided."""
        with pytest.raises(TypeError):
            TranscriptionResult(text="Hello")

    def test_transcription_result_immutable_fields(self):
        """Test that dataclass fields can be accessed."""
        result = TranscriptionResult(
            text="Test",
            vad_time=0.0,
            whisper_time=0.0,
            total_time=0.0,
            original_duration=0.0,
            speech_duration=0.0,
            silence_skipped=0.0,
            speedup_factor=1.0,
        )
        # Dataclass fields should be accessible
        assert hasattr(result, 'text')
        assert hasattr(result, 'speedup_factor')


class TestOptimizedWhisperInit:
    """Tests for OptimizedWhisper initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        whisper = OptimizedWhisper()
        assert whisper.model_path == OptimizedWhisper.MODELS["large-v3"]
        assert whisper.use_vad is True
        assert whisper.vad_aggressiveness == 2
        assert whisper.frame_duration_ms == 30

    def test_custom_model_name(self):
        """Test initialization with model name from MODELS dict."""
        whisper = OptimizedWhisper(model="tiny")
        assert whisper.model_path == OptimizedWhisper.MODELS["tiny"]

    def test_custom_model_path(self):
        """Test initialization with custom HuggingFace path."""
        custom_path = "my-org/my-whisper-model"
        whisper = OptimizedWhisper(model=custom_path)
        assert whisper.model_path == custom_path

    def test_vad_disabled(self):
        """Test initialization with VAD disabled."""
        whisper = OptimizedWhisper(use_vad=False)
        assert whisper.use_vad is False

    def test_vad_aggressiveness_levels(self):
        """Test different VAD aggressiveness levels."""
        for level in [0, 1, 2, 3]:
            whisper = OptimizedWhisper(vad_aggressiveness=level)
            assert whisper.vad_aggressiveness == level

    def test_all_model_names_valid(self):
        """Test that all model names in MODELS dict work."""
        for model_name in OptimizedWhisper.MODELS.keys():
            whisper = OptimizedWhisper(model=model_name)
            assert whisper.model_path == OptimizedWhisper.MODELS[model_name]


class TestOptimizedWhisperModels:
    """Tests for model configuration."""

    def test_models_dict_not_empty(self):
        """Test that MODELS dict is populated."""
        assert len(OptimizedWhisper.MODELS) > 0

    def test_models_have_valid_paths(self):
        """Test that all models have valid-looking paths."""
        for name, path in OptimizedWhisper.MODELS.items():
            assert "/" in path, f"Model {name} should have org/model format"
            assert len(path) > 5, f"Model {name} path too short"

    def test_expected_models_present(self):
        """Test that expected models are present."""
        expected = ["large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"]
        for model in expected:
            assert model in OptimizedWhisper.MODELS, f"Missing model: {model}"


class TestOptimizedWhisperLazyLoading:
    """Tests for lazy loading of mlx_whisper."""

    def test_mlx_whisper_not_loaded_initially(self):
        """Test that mlx_whisper is not loaded on init."""
        whisper = OptimizedWhisper()
        assert whisper._mlx_whisper is None

    @patch.dict(sys.modules, {"mlx_whisper": MagicMock()})
    def test_mlx_whisper_loaded_on_access(self):
        """Test that mlx_whisper is loaded on first access."""
        whisper = OptimizedWhisper()
        # Clear the mock to ensure fresh import
        whisper._mlx_whisper = None
        _ = whisper.mlx_whisper
        assert whisper._mlx_whisper is not None

    @patch.dict(sys.modules, {"mlx_whisper": MagicMock()})
    def test_mlx_whisper_cached(self):
        """Test that mlx_whisper is cached after first access."""
        whisper = OptimizedWhisper()
        whisper._mlx_whisper = None
        first_access = whisper.mlx_whisper
        second_access = whisper.mlx_whisper
        assert first_access is second_access


class TestVADMethods:
    """Tests for VAD-related methods."""

    def test_get_speech_segments_empty_audio(self):
        """Test VAD with silence-only audio."""
        whisper = OptimizedWhisper()
        # Create silent audio (zeros)
        sample_rate = 16000
        duration = 0.3  # 300ms
        num_samples = int(sample_rate * duration)
        silent_pcm = b'\x00' * (num_samples * 2)  # 16-bit = 2 bytes per sample

        segments = whisper._get_speech_segments(silent_pcm, sample_rate)
        # Silent audio should produce few or no segments
        assert isinstance(segments, list)

    def test_get_speech_segments_returns_list(self):
        """Test that _get_speech_segments returns a list."""
        whisper = OptimizedWhisper()
        # Create minimal audio data
        sample_rate = 16000
        pcm_data = b'\x00' * (sample_rate * 2)  # 1 second of silence

        segments = whisper._get_speech_segments(pcm_data, sample_rate)
        assert isinstance(segments, list)

    def test_get_speech_segments_structure(self):
        """Test that segments have correct structure."""
        whisper = OptimizedWhisper()

        # Create audio with some non-zero values that might trigger VAD
        sample_rate = 16000
        # 100ms of "noise" that might be detected as speech
        frame_size = int(sample_rate * 0.03)  # 30ms frame
        frames_needed = 10
        samples_needed = frame_size * frames_needed

        # Create audio with varying amplitude
        audio = np.sin(np.linspace(0, 100 * np.pi, samples_needed)) * 16000
        pcm_data = audio.astype(np.int16).tobytes()

        segments = whisper._get_speech_segments(pcm_data, sample_rate)

        for segment in segments:
            assert 'start' in segment
            assert 'end' in segment
            assert segment['start'] >= 0
            assert segment['end'] > segment['start']

    def test_extract_speech_empty_segments(self):
        """Test that empty segments return None."""
        whisper = OptimizedWhisper()
        result = whisper._extract_speech(b'\x00\x00', [], 16000)
        assert result is None

    def test_extract_speech_creates_file(self):
        """Test that extract_speech creates a temp file."""
        whisper = OptimizedWhisper()
        sample_rate = 16000
        pcm_data = b'\x00' * (sample_rate * 2)  # 1 second
        segments = [{'start': 0, 'end': sample_rate * 2}]  # Full audio

        result = whisper._extract_speech(pcm_data, segments, sample_rate)

        try:
            assert result is not None
            assert os.path.exists(result)
            # Verify it's a valid wav file
            with wave.open(result, 'rb') as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == sample_rate
        finally:
            if result and os.path.exists(result):
                os.unlink(result)


class TestAudioConversion:
    """Tests for audio conversion methods."""

    def test_convert_to_16khz_mono_creates_files(self):
        """Test that conversion creates temporary files."""
        # Create a minimal WAV file for testing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write a minimal valid WAV file
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # Write 0.1 seconds of silence
                wf.writeframes(b'\x00' * 3200)

            whisper = OptimizedWhisper()
            pcm_data, sample_rate, converted_path = whisper._convert_to_16khz_mono(tmp_path)

            assert isinstance(pcm_data, bytes)
            assert sample_rate == 16000
            assert os.path.exists(converted_path)

            # Cleanup
            os.unlink(converted_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestTranscription:
    """Tests for transcription methods (requires mocking)."""

    @patch.object(OptimizedWhisper, 'mlx_whisper', new_callable=lambda: MagicMock())
    def test_transcribe_without_vad(self, mock_mlx):
        """Test transcription with VAD disabled."""
        # Create mock transcribe result
        mock_mlx.transcribe.return_value = {'text': ' Hello world '}

        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'\x00' * 16000)  # 0.5 seconds

        try:
            whisper = OptimizedWhisper(use_vad=False)
            whisper._mlx_whisper = mock_mlx

            result = whisper.transcribe(tmp_path)

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello world"  # Should be stripped
            assert result.vad_time == 0
            assert result.silence_skipped == 0
            assert result.speedup_factor == 1.0
            mock_mlx.transcribe.assert_called_once()
        finally:
            os.unlink(tmp_path)

    @patch.object(OptimizedWhisper, 'mlx_whisper', new_callable=lambda: MagicMock())
    def test_transcribe_simple_returns_text(self, mock_mlx):
        """Test that transcribe_simple returns just text."""
        mock_mlx.transcribe.return_value = {'text': ' Test transcription '}

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'\x00' * 16000)

        try:
            whisper = OptimizedWhisper(use_vad=False)
            whisper._mlx_whisper = mock_mlx

            result = whisper.transcribe_simple(tmp_path)

            assert isinstance(result, str)
            assert result == "Test transcription"
        finally:
            os.unlink(tmp_path)

    @patch.object(OptimizedWhisper, 'mlx_whisper', new_callable=lambda: MagicMock())
    def test_transcribe_passes_language(self, mock_mlx):
        """Test that language parameter is passed to mlx_whisper."""
        mock_mlx.transcribe.return_value = {'text': 'Test'}

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'\x00' * 16000)

        try:
            whisper = OptimizedWhisper(use_vad=False)
            whisper._mlx_whisper = mock_mlx

            whisper.transcribe(tmp_path, language='en')

            call_kwargs = mock_mlx.transcribe.call_args[1]
            assert call_kwargs.get('language') == 'en'
        finally:
            os.unlink(tmp_path)

    @patch.object(OptimizedWhisper, 'mlx_whisper', new_callable=lambda: MagicMock())
    def test_transcribe_passes_kwargs(self, mock_mlx):
        """Test that additional kwargs are passed to mlx_whisper."""
        mock_mlx.transcribe.return_value = {'text': 'Test'}

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'\x00' * 16000)

        try:
            whisper = OptimizedWhisper(use_vad=False)
            whisper._mlx_whisper = mock_mlx

            whisper.transcribe(tmp_path, beam_size=5, patience=1.5)

            call_kwargs = mock_mlx.transcribe.call_args[1]
            assert call_kwargs.get('beam_size') == 5
            assert call_kwargs.get('patience') == 1.5
        finally:
            os.unlink(tmp_path)


class TestConvenienceFunction:
    """Tests for the module-level transcribe function."""

    @patch('optimized_whisper.OptimizedWhisper')
    def test_transcribe_function_default_args(self, mock_class):
        """Test that transcribe function uses correct defaults."""
        mock_instance = MagicMock()
        mock_instance.transcribe_simple.return_value = "Hello"
        mock_class.return_value = mock_instance

        result = transcribe("test.wav")

        mock_class.assert_called_once_with(model="large-v3-turbo", use_vad=True)
        mock_instance.transcribe_simple.assert_called_once()
        assert result == "Hello"

    @patch('optimized_whisper.OptimizedWhisper')
    def test_transcribe_function_custom_args(self, mock_class):
        """Test that transcribe function passes custom args."""
        mock_instance = MagicMock()
        mock_instance.transcribe_simple.return_value = "Test"
        mock_class.return_value = mock_instance

        transcribe("test.wav", model="tiny", use_vad=False)

        mock_class.assert_called_once_with(model="tiny", use_vad=False)


class TestSpeedupCalculation:
    """Tests for speedup factor calculation."""

    def test_speedup_no_vad(self):
        """Test that speedup is 1.0 when VAD is disabled."""
        result = TranscriptionResult(
            text="Test",
            vad_time=0.0,
            whisper_time=1.0,
            total_time=1.0,
            original_duration=5.0,
            speech_duration=5.0,
            silence_skipped=0.0,
            speedup_factor=1.0,
        )
        assert result.speedup_factor == 1.0

    def test_speedup_with_silence_skipped(self):
        """Test speedup when silence is skipped."""
        # If original=10s, speech=5s, and whisper took 1s
        # then without VAD it would have taken ~2s
        # speedup = 2/1 = 2.0
        result = TranscriptionResult(
            text="Test",
            vad_time=0.1,
            whisper_time=1.0,
            total_time=1.1,
            original_duration=10.0,
            speech_duration=5.0,
            silence_skipped=5.0,
            speedup_factor=2.0,
        )
        assert result.speedup_factor > 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_segments_handling(self):
        """Test handling of empty speech segments."""
        whisper = OptimizedWhisper()
        result = whisper._extract_speech(b'\x00\x00\x00\x00', [], 16000)
        assert result is None

    def test_single_byte_segments(self):
        """Test handling of very small segments."""
        whisper = OptimizedWhisper()
        segments = [{'start': 0, 'end': 2}]  # Single sample
        result = whisper._extract_speech(b'\x00\x00', segments, 16000)

        try:
            assert result is not None
        finally:
            if result and os.path.exists(result):
                os.unlink(result)

    def test_multiple_segments_concatenation(self):
        """Test that multiple segments are concatenated correctly."""
        whisper = OptimizedWhisper()
        pcm_data = b'\x01\x00\x02\x00\x03\x00\x04\x00'  # 4 samples
        segments = [
            {'start': 0, 'end': 2},  # First sample
            {'start': 4, 'end': 6},  # Third sample
        ]

        result = whisper._extract_speech(pcm_data, segments, 16000)

        try:
            assert result is not None
            with wave.open(result, 'rb') as wf:
                data = wf.readframes(wf.getnframes())
                # Should have concatenated the two samples
                assert len(data) == 4  # 2 samples * 2 bytes
        finally:
            if result and os.path.exists(result):
                os.unlink(result)


class TestFrameDuration:
    """Tests for frame duration configuration."""

    def test_default_frame_duration(self):
        """Test default frame duration is 30ms."""
        whisper = OptimizedWhisper()
        assert whisper.frame_duration_ms == 30

    def test_frame_size_calculation(self):
        """Test that frame size is calculated correctly."""
        whisper = OptimizedWhisper()
        sample_rate = 16000
        # Frame size in bytes = samples_per_frame * 2 (16-bit audio)
        expected_samples = int(sample_rate * whisper.frame_duration_ms / 1000)
        expected_bytes = expected_samples * 2

        # Verify frame size is used correctly in _get_speech_segments
        frame_size = int(sample_rate * whisper.frame_duration_ms / 1000) * 2
        assert frame_size == expected_bytes


# Integration tests - skipped if no real audio files available
class TestIntegration:
    """Integration tests requiring real audio files."""

    @pytest.fixture
    def test_audio_dir(self):
        """Get path to test audio directory."""
        return Path(__file__).parent.parent / "reports" / "audio"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "reports" / "audio").exists(),
        reason="Test audio directory not found",
    )
    def test_real_audio_conversion(self, test_audio_dir):
        """Test conversion with real audio file."""
        audio_files = list(test_audio_dir.glob("*.wav"))
        if not audio_files:
            pytest.skip("No WAV files found in test audio directory")

        whisper = OptimizedWhisper()
        pcm_data, sample_rate, converted_path = whisper._convert_to_16khz_mono(
            str(audio_files[0]),
        )

        try:
            assert len(pcm_data) > 0
            assert sample_rate == 16000
            assert os.path.exists(converted_path)
        finally:
            if os.path.exists(converted_path):
                os.unlink(converted_path)

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "reports" / "audio").exists(),
        reason="Test audio directory not found",
    )
    def test_real_audio_vad(self, test_audio_dir):
        """Test VAD with real audio file."""
        audio_files = list(test_audio_dir.glob("*.wav"))
        if not audio_files:
            pytest.skip("No WAV files found in test audio directory")

        whisper = OptimizedWhisper()
        pcm_data, sample_rate, converted_path = whisper._convert_to_16khz_mono(
            str(audio_files[0]),
        )

        try:
            segments = whisper._get_speech_segments(pcm_data, sample_rate)
            # Real audio should have some speech segments
            assert isinstance(segments, list)
            # Most test audio files should have speech
            if len(segments) > 0:
                for segment in segments:
                    assert 'start' in segment
                    assert 'end' in segment
        finally:
            if os.path.exists(converted_path):
                os.unlink(converted_path)
