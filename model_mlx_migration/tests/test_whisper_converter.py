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
Tests for Whisper STT Converter

Tests the WhisperConverter wrapper around mlx-whisper.
"""

import importlib

import pytest


@pytest.fixture
def whisper_converter_class():
    """Fixture that returns a fresh WhisperConverter class."""
    from tools.pytorch_to_mlx.converters import whisper_converter

    importlib.reload(whisper_converter)
    return whisper_converter.WhisperConverter


@pytest.fixture
def whisper_converter(whisper_converter_class):
    """Fixture that returns a WhisperConverter instance."""
    return whisper_converter_class()


class TestWhisperConverterImport:
    """Test WhisperConverter import and initialization."""

    def test_import_whisper_converter(self):
        """Should be able to import WhisperConverter."""
        from tools.pytorch_to_mlx.converters import WhisperConverter

        assert WhisperConverter is not None

    def test_init_with_mlx_whisper(self):
        """Should initialize successfully with mlx-whisper installed."""
        # Reimport module to ensure we have fresh state
        import importlib

        from tools.pytorch_to_mlx.converters import whisper_converter

        importlib.reload(whisper_converter)
        from tools.pytorch_to_mlx.converters.whisper_converter import WhisperConverter

        converter = WhisperConverter()
        assert converter is not None


class TestWhisperConverterListModels:
    """Test model listing functionality."""

    def test_list_models_returns_list(self):
        """list_models() should return a list of model paths."""
        from tools.pytorch_to_mlx.converters import WhisperConverter

        models = WhisperConverter.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_models_contains_turbo(self):
        """list_models() should include whisper-large-v3-turbo."""
        from tools.pytorch_to_mlx.converters import WhisperConverter

        models = WhisperConverter.list_models()
        turbo_models = [m for m in models if "turbo" in m]
        assert len(turbo_models) > 0

    def test_list_models_contains_common_variants(self):
        """list_models() should include common model variants."""
        from tools.pytorch_to_mlx.converters import WhisperConverter

        models = WhisperConverter.list_models()
        model_str = str(models)
        assert "tiny" in model_str
        assert "base" in model_str
        assert "small" in model_str
        assert "large" in model_str


class TestWhisperConverterGetModelInfo:
    """Test model info retrieval."""

    def test_get_model_info_returns_dict(self):
        """get_model_info() should return a dictionary."""
        from tools.pytorch_to_mlx.converters import WhisperConverter

        info = WhisperConverter.get_model_info("mlx-community/whisper-large-v3-turbo")
        assert isinstance(info, dict)
        assert "path" in info
        assert "variant" in info
        assert "size" in info
        assert "multilingual" in info

    def test_get_model_info_turbo(self):
        """get_model_info() should correctly identify turbo model."""
        from tools.pytorch_to_mlx.converters import WhisperConverter

        info = WhisperConverter.get_model_info("mlx-community/whisper-large-v3-turbo")
        assert info["variant"] == "large-v3-turbo"
        assert info["multilingual"] is True

    def test_get_model_info_english_only(self):
        """get_model_info() should correctly identify English-only models."""
        from tools.pytorch_to_mlx.converters import WhisperConverter

        info = WhisperConverter.get_model_info("mlx-community/whisper-small.en")
        assert info["multilingual"] is False


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_transcription_result_creation(self):
        """Should create TranscriptionResult with required fields."""
        from tools.pytorch_to_mlx.converters.whisper_converter import (
            TranscriptionResult,
        )

        result = TranscriptionResult(
            success=True,
            text="Hello world",
            language="en",
            duration_seconds=5.0,
            transcription_time_seconds=0.5,
            real_time_factor=0.1,
        )
        assert result.success is True
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.real_time_factor == 0.1

    def test_transcription_result_failure(self):
        """Should create failed TranscriptionResult with error."""
        from tools.pytorch_to_mlx.converters.whisper_converter import (
            TranscriptionResult,
        )

        result = TranscriptionResult(success=False, text="", error="File not found")
        assert result.success is False
        assert result.error == "File not found"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Should create BenchmarkResult with all fields."""
        from tools.pytorch_to_mlx.converters.whisper_converter import BenchmarkResult

        result = BenchmarkResult(
            audio_duration_seconds=10.0,
            transcription_time_seconds=1.0,
            real_time_factor=0.1,
            words_per_second=50.0,
            model="mlx-community/whisper-large-v3-turbo",
        )
        assert result.audio_duration_seconds == 10.0
        assert result.real_time_factor == 0.1
        assert result.words_per_second == 50.0


class TestFormatOutput:
    """Test output formatting functions."""

    def test_format_output_text(self, whisper_converter):
        """format_output() should return plain text."""
        from tools.pytorch_to_mlx.converters.whisper_converter import (
            TranscriptionResult,
        )

        result = TranscriptionResult(
            success=True,
            text="Hello world",
        )
        output = whisper_converter.format_output(result, "text")
        assert output == "Hello world"

    def test_format_output_json(self, whisper_converter):
        """format_output() should return valid JSON."""
        import json

        from tools.pytorch_to_mlx.converters.whisper_converter import (
            TranscriptionResult,
        )

        result = TranscriptionResult(
            success=True,
            text="Hello world",
            language="en",
            duration_seconds=5.0,
        )
        output = whisper_converter.format_output(result, "json")
        data = json.loads(output)
        assert data["text"] == "Hello world"
        assert data["language"] == "en"

    def test_format_output_srt(self, whisper_converter):
        """format_output() should return SRT format."""
        from tools.pytorch_to_mlx.converters.whisper_converter import (
            TranscriptionResult,
        )

        result = TranscriptionResult(
            success=True,
            text="Hello world",
            segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.0, "end": 4.0, "text": "world"},
            ],
        )
        output = whisper_converter.format_output(result, "srt")
        assert "1" in output
        assert "00:00:00,000 --> 00:00:02,000" in output
        assert "Hello" in output

    def test_format_output_vtt(self, whisper_converter):
        """format_output() should return WebVTT format."""
        from tools.pytorch_to_mlx.converters.whisper_converter import (
            TranscriptionResult,
        )

        result = TranscriptionResult(
            success=True,
            text="Hello world",
            segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello"},
            ],
        )
        output = whisper_converter.format_output(result, "vtt")
        assert "WEBVTT" in output
        assert "00:00:00.000 --> 00:00:02.000" in output


class TestTimestampFormatting:
    """Test timestamp formatting functions."""

    def test_format_timestamp_srt(self, whisper_converter):
        """SRT timestamps should use comma separator."""
        ts = whisper_converter._format_timestamp_srt(65.123)
        assert ts == "00:01:05,123"

    def test_format_timestamp_vtt(self, whisper_converter):
        """VTT timestamps should use period separator."""
        ts = whisper_converter._format_timestamp_vtt(65.123)
        assert ts == "00:01:05.123"

    def test_format_timestamp_hours(self, whisper_converter):
        """Should correctly format timestamps with hours."""
        ts = whisper_converter._format_timestamp_srt(3723.456)  # 1:02:03.456
        assert ts == "01:02:03,456"


class TestTranscribeFileNotFound:
    """Test transcription with missing file."""

    def test_transcribe_file_not_found(self, whisper_converter):
        """transcribe() should return error for missing file."""
        result = whisper_converter.transcribe("/nonexistent/audio.mp3")
        assert result.success is False
        assert "not found" in result.error.lower()


class TestTranscribeArrayBasic:
    """Test transcribe_array function basics."""

    def test_transcribe_array_exists(self, whisper_converter):
        """transcribe_array() method should exist."""
        assert hasattr(whisper_converter, "transcribe_array")
        assert callable(whisper_converter.transcribe_array)


class TestDefaultModel:
    """Test default model configuration."""

    def test_default_model_is_turbo(self):
        """DEFAULT_MODEL should be whisper-large-v3-turbo."""
        from tools.pytorch_to_mlx.converters import WhisperConverter

        assert "turbo" in WhisperConverter.DEFAULT_MODEL
        assert "large-v3" in WhisperConverter.DEFAULT_MODEL


@pytest.mark.e2e
class TestWhisperE2E:
    """End-to-end tests for Whisper transcription with real speech audio.

    These tests require a real audio file with speech.
    The test fixture is generated using macOS 'say' command.

    Run with: pytest -m e2e tests/test_whisper_converter.py
    """

    TEST_AUDIO_PATH = "tests/fixtures/audio/test_speech.wav"
    # Expected text (generated with macOS 'say'):
    # "Hello, this is a test of the Whisper speech recognition system.
    #  The quick brown fox jumps over the lazy dog."
    EXPECTED_WORDS = [
        "hello",
        "test",
        "whisper",
        "speech",
        "recognition",
        "quick",
        "brown",
        "fox",
        "jumps",
        "lazy",
        "dog",
    ]

    @pytest.fixture
    def audio_path(self):
        """Check test audio file exists."""
        import os

        if not os.path.exists(self.TEST_AUDIO_PATH):
            pytest.skip(f"Test audio file not found: {self.TEST_AUDIO_PATH}")
        return self.TEST_AUDIO_PATH

    def test_e2e_transcription_quality(self, whisper_converter, audio_path):
        """E2E: Transcription should correctly recognize speech content."""
        result = whisper_converter.transcribe(
            audio_path,
            model="mlx-community/whisper-tiny",  # Use tiny for fast E2E tests
            language="en",
        )

        assert result.success is True, f"Transcription failed: {result.error}"
        assert result.text != "", "Transcription returned empty text"

        # Check key words are present (case-insensitive)
        text_lower = result.text.lower()
        recognized_words = [w for w in self.EXPECTED_WORDS if w in text_lower]

        # Should recognize at least 80% of expected words
        recognition_rate = len(recognized_words) / len(self.EXPECTED_WORDS)
        assert recognition_rate >= 0.8, (
            f"Recognition rate too low: {recognition_rate:.1%}. "
            f"Expected words: {self.EXPECTED_WORDS}, "
            f"Recognized: {recognized_words}, "
            f"Transcription: '{result.text}'"
        )

    def test_e2e_detects_english(self, whisper_converter, audio_path):
        """E2E: Should correctly detect English language."""
        result = whisper_converter.transcribe(
            audio_path, model="mlx-community/whisper-tiny",
        )

        assert result.success is True
        assert result.language == "en", f"Expected 'en', got '{result.language}'"

    def test_e2e_realtime_factor(self, whisper_converter, audio_path):
        """E2E: Transcription should be faster than real-time."""
        result = whisper_converter.transcribe(
            audio_path, model="mlx-community/whisper-tiny",
        )

        assert result.success is True
        assert result.duration_seconds > 0, "Audio duration not detected"
        assert result.transcription_time_seconds > 0, "Transcription time not recorded"

        # RTF < 1.0 means faster than real-time
        assert result.real_time_factor < 1.0, (
            f"Transcription slower than real-time: RTF={result.real_time_factor:.3f}"
        )
        # Expect at least 5x real-time for tiny model
        assert result.real_time_factor < 0.2, (
            f"Expected RTF < 0.2 (5x real-time), got {result.real_time_factor:.3f}"
        )

    def test_e2e_segments_have_timestamps(self, whisper_converter, audio_path):
        """E2E: Transcription should include timestamped segments."""
        result = whisper_converter.transcribe(
            audio_path, model="mlx-community/whisper-tiny",
        )

        assert result.success is True
        assert result.segments is not None, "No segments returned"
        assert len(result.segments) > 0, "Empty segments list"

        # Check segment structure
        for segment in result.segments:
            assert "start" in segment, "Segment missing 'start' timestamp"
            assert "end" in segment, "Segment missing 'end' timestamp"
            assert "text" in segment, "Segment missing 'text'"
            assert segment["end"] > segment["start"], "Invalid timestamp order"

    def test_e2e_benchmark(self, whisper_converter, audio_path):
        """E2E: Benchmark should return valid performance metrics."""
        bench = whisper_converter.benchmark(
            audio_path,
            model="mlx-community/whisper-tiny",
            runs=2,  # Keep low for fast tests
        )

        assert bench.audio_duration_seconds > 0
        assert bench.transcription_time_seconds > 0
        assert bench.real_time_factor > 0
        assert bench.real_time_factor < 1.0  # Faster than real-time
        assert bench.words_per_second > 0
