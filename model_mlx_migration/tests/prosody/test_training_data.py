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
Tests for prosody training data preparation pipeline.

Tests:
1. Feature extraction (F0, duration, energy)
2. Auto-annotation logic
3. Dataset save/load roundtrip
4. Prosody type mappings
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.prepare_prosody_training_data import (
    EMOTION_TO_PROSODY,
    PROSODY_TYPES,
    ProsodySample,
    auto_annotate_prosody,
    compute_stats,
    extract_prosody_features,
    load_dataset,
    save_dataset,
    verify_dataset,
)


class TestProsodyTypes:
    """Test prosody type definitions."""

    def test_prosody_types_valid_range(self):
        """Prosody types should be in valid range."""
        for name, value in PROSODY_TYPES.items():
            assert 0 <= value < 70, f"Prosody type {name} out of range: {value}"

    def test_emotion_mapping_coverage(self):
        """All emotion mappings should point to valid prosody types."""
        valid_types = set(PROSODY_TYPES.values())
        for emotion, prosody_type in EMOTION_TO_PROSODY.items():
            assert prosody_type in valid_types, f"Invalid prosody type for {emotion}"

    def test_core_emotions_mapped(self):
        """Core emotions should be mapped."""
        core_emotions = ["angry", "sad", "excited", "calm", "neutral"]
        for emotion in core_emotions:
            assert emotion in EMOTION_TO_PROSODY, f"Missing mapping for {emotion}"


class TestFeatureExtraction:
    """Test audio feature extraction."""

    @pytest.fixture
    def sine_wave(self):
        """Generate a simple sine wave for testing."""
        sr = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        freq = 220  # Hz (A3)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        return audio, sr

    @pytest.fixture
    def varying_sine(self):
        """Generate a sine wave with varying frequency."""
        sr = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # Frequency varies from 200 to 300 Hz
        freq = 200 + 100 * t
        phase = np.cumsum(2 * np.pi * freq / sr)
        audio = np.sin(phase).astype(np.float32)
        return audio, sr

    def test_extract_prosody_features_returns_dict(self, sine_wave):
        """extract_prosody_features should return a dict with expected keys."""
        audio, sr = sine_wave
        features = extract_prosody_features(audio, sr)

        expected_keys = ["f0_mean", "f0_std", "f0_range", "duration_s", "energy_rms"]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_duration_calculation(self, sine_wave):
        """Duration should be calculated correctly."""
        audio, sr = sine_wave
        features = extract_prosody_features(audio, sr)

        expected_duration = len(audio) / sr
        assert abs(features["duration_s"] - expected_duration) < 0.01

    def test_energy_rms_positive(self, sine_wave):
        """Energy RMS should be positive for non-silent audio."""
        audio, sr = sine_wave
        features = extract_prosody_features(audio, sr)

        assert features["energy_rms"] > 0

    def test_silent_audio(self):
        """Silent audio should have zero energy."""
        sr = 24000
        audio = np.zeros(sr, dtype=np.float32)  # 1 second of silence
        features = extract_prosody_features(audio, sr)

        assert features["energy_rms"] == 0
        assert features["duration_s"] == 1.0


class TestAutoAnnotation:
    """Test automatic prosody annotation."""

    def test_neutral_annotation(self):
        """Low variance features should be annotated as neutral."""
        features = {
            "f0_mean": 180,  # Normal pitch
            "f0_std": 20,    # Low variance
            "energy_rms": 0.05,
        }
        text = "Hello world"

        prosody_type, annotated = auto_annotate_prosody(features, text)

        # Should be neutral or calm due to moderate values
        assert prosody_type in [PROSODY_TYPES["NEUTRAL"], PROSODY_TYPES["EMOTION_CALM"]]

    def test_excited_annotation(self):
        """High pitch + high variance should be annotated as excited."""
        features = {
            "f0_mean": 280,  # High pitch
            "f0_std": 50,    # High variance
            "energy_rms": 0.1,
        }
        text = "Wow amazing"

        prosody_type, annotated = auto_annotate_prosody(features, text)

        assert prosody_type == PROSODY_TYPES["EMOTION_EXCITED"]
        assert "excited" in annotated.lower()

    def test_sad_annotation(self):
        """Low energy should be annotated as sad."""
        features = {
            "f0_mean": 150,
            "f0_std": 15,
            "energy_rms": 0.02,  # Very low energy
        }
        text = "I feel down"

        prosody_type, annotated = auto_annotate_prosody(features, text)

        assert prosody_type == PROSODY_TYPES["EMOTION_SAD"]
        assert "sad" in annotated.lower()

    def test_annotated_text_format(self):
        """Annotated text should use SSML-style emotion tags."""
        features = {
            "f0_mean": 280,
            "f0_std": 50,
            "energy_rms": 0.1,
        }
        text = "Hello"

        _, annotated = auto_annotate_prosody(features, text)

        # Should contain emotion tag
        assert "<emotion type=" in annotated or annotated == text


class TestDatasetManagement:
    """Test dataset save/load functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample prosody data."""
        return [
            ProsodySample(
                text="Hello world",
                annotated_text="<emotion type='excited'>Hello world</emotion>",
                prosody_type=PROSODY_TYPES["EMOTION_EXCITED"],
                audio_path="/tmp/test1.wav",
                f0_mean=220.5,
                f0_std=30.2,
                f0_range=85.0,
                duration_s=1.2,
                energy_rms=0.08,
                source="synthetic",
            ),
            ProsodySample(
                text="Goodbye",
                annotated_text="Goodbye",
                prosody_type=PROSODY_TYPES["NEUTRAL"],
                audio_path="/tmp/test2.wav",
                f0_mean=180.0,
                f0_std=15.5,
                f0_range=40.0,
                duration_s=0.8,
                energy_rms=0.05,
                source="synthetic",
            ),
        ]

    def test_save_load_roundtrip(self, sample_data, tmp_path):
        """Data should survive save/load roundtrip."""
        output_path = tmp_path / "test_data.json"

        save_dataset(sample_data, output_path)
        loaded = load_dataset(output_path)

        assert len(loaded) == len(sample_data)
        for original, loaded_sample in zip(sample_data, loaded, strict=False):
            assert loaded_sample.text == original.text
            assert loaded_sample.annotated_text == original.annotated_text
            assert loaded_sample.prosody_type == original.prosody_type
            assert loaded_sample.f0_mean == original.f0_mean

    def test_compute_stats(self, sample_data):
        """compute_stats should return correct statistics."""
        stats = compute_stats(sample_data)

        assert stats.total_samples == 2
        assert stats.total_duration_s == pytest.approx(2.0, rel=0.01)
        assert stats.mean_duration == pytest.approx(1.0, rel=0.01)
        assert stats.mean_f0 == pytest.approx(200.25, rel=0.01)

        # Check samples by type
        assert PROSODY_TYPES["EMOTION_EXCITED"] in stats.samples_by_type
        assert PROSODY_TYPES["NEUTRAL"] in stats.samples_by_type


class TestDatasetVerification:
    """Test dataset verification."""

    def test_valid_dataset(self, tmp_path):
        """Valid dataset should pass verification."""
        # Create a valid audio file
        import soundfile as sf

        rng = np.random.default_rng(42)
        audio = rng.standard_normal(24000).astype(np.float32) * 0.1
        audio_path = tmp_path / "valid.wav"
        sf.write(audio_path, audio, 24000)

        samples = [
            ProsodySample(
                text="Test",
                annotated_text="Test",
                prosody_type=PROSODY_TYPES["NEUTRAL"],
                audio_path=str(audio_path),
                f0_mean=180.0,
                f0_std=20.0,
                f0_range=60.0,
                duration_s=1.0,
                energy_rms=0.05,
                source="test",
            ),
        ]

        result = verify_dataset(samples)
        assert result is True

    def test_missing_audio_detection(self, tmp_path):
        """Should detect missing audio files."""
        samples = [
            ProsodySample(
                text="Test",
                annotated_text="Test",
                prosody_type=PROSODY_TYPES["NEUTRAL"],
                audio_path="/nonexistent/path.wav",
                f0_mean=180.0,
                f0_std=20.0,
                f0_range=60.0,
                duration_s=1.0,
                energy_rms=0.05,
                source="test",
            ),
        ]

        result = verify_dataset(samples)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
