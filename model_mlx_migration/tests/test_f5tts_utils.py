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
Tests for F5-TTS utility functions.

DEPRECATED: F5-TTS is deprecated in favor of CosyVoice2.
CosyVoice2 is 18x faster (35x vs 2x RTF) with equal/better quality.
These tests are kept for historical compatibility only.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.f5tts_utils import (
    F5TTS_SAMPLE_RATE,
    ensure_24khz,
    get_audio_info,
)


def create_test_audio(sample_rate: int, duration: float = 1.0) -> str:
    """Create a test audio file at the specified sample rate."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz sine wave

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        return f.name


class TestGetAudioInfo:
    """Tests for get_audio_info function."""

    def test_24khz_audio(self):
        """Audio at 24kHz should not need resampling."""
        path = create_test_audio(24000)
        try:
            info = get_audio_info(path)
            assert info["sample_rate"] == 24000
            assert info["needs_resample"] is False
            assert info["duration"] == pytest.approx(1.0, abs=0.01)
        finally:
            Path(path).unlink()

    def test_48khz_audio(self):
        """Audio at 48kHz should need resampling."""
        path = create_test_audio(48000)
        try:
            info = get_audio_info(path)
            assert info["sample_rate"] == 48000
            assert info["needs_resample"] is True
        finally:
            Path(path).unlink()

    def test_16khz_audio(self):
        """Audio at 16kHz should need resampling."""
        path = create_test_audio(16000)
        try:
            info = get_audio_info(path)
            assert info["sample_rate"] == 16000
            assert info["needs_resample"] is True
        finally:
            Path(path).unlink()


class TestEnsure24khz:
    """Tests for ensure_24khz function."""

    def test_already_24khz(self):
        """Audio already at 24kHz should return original path."""
        path = create_test_audio(24000)
        try:
            result = ensure_24khz(path, verbose=False)
            assert result == path  # Should return same path
        finally:
            Path(path).unlink()

    def test_resample_48khz(self):
        """Audio at 48kHz should be resampled to 24kHz."""
        path = create_test_audio(48000, duration=1.0)
        try:
            result = ensure_24khz(path, verbose=False)
            assert result != path  # Should be a new file

            info = get_audio_info(result)
            assert info["sample_rate"] == F5TTS_SAMPLE_RATE
            assert info["duration"] == pytest.approx(1.0, abs=0.02)

            # Clean up resampled file
            Path(result).unlink()
        finally:
            Path(path).unlink()

    def test_resample_16khz(self):
        """Audio at 16kHz should be upsampled to 24kHz."""
        path = create_test_audio(16000, duration=1.0)
        try:
            result = ensure_24khz(path, verbose=False)
            assert result != path

            info = get_audio_info(result)
            assert info["sample_rate"] == F5TTS_SAMPLE_RATE
            assert info["duration"] == pytest.approx(1.0, abs=0.02)

            Path(result).unlink()
        finally:
            Path(path).unlink()

    def test_resample_44100hz(self):
        """Audio at 44.1kHz (common CD quality) should be resampled."""
        path = create_test_audio(44100, duration=1.0)
        try:
            result = ensure_24khz(path, verbose=False)
            assert result != path

            info = get_audio_info(result)
            assert info["sample_rate"] == F5TTS_SAMPLE_RATE

            Path(result).unlink()
        finally:
            Path(path).unlink()

    def test_custom_output_path(self):
        """Should save to custom output path when specified."""
        path = create_test_audio(48000)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

            result = ensure_24khz(path, output_path=output_path, verbose=False)
            assert result == output_path

            info = get_audio_info(result)
            assert info["sample_rate"] == F5TTS_SAMPLE_RATE

            Path(output_path).unlink()
        finally:
            Path(path).unlink()


class TestF5TTSSampleRate:
    """Tests for F5TTS_SAMPLE_RATE constant."""

    def test_sample_rate_is_24000(self):
        """F5-TTS requires 24kHz audio."""
        assert F5TTS_SAMPLE_RATE == 24000
