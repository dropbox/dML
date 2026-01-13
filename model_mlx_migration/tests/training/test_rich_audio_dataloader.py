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
Tests for rich audio dataloaders.

Tests loading from emotion (CREMA-D) and pitch (VocalSet) datasets.
"""

import sys
from pathlib import Path

import mlx.core as mx
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training import (
    EMOTION_LABELS,
    MELD_EMOTION_TO_INT,
    PARALINGUISTIC_LABELS,
    CREMADDataset,
    MELDDataset,
    RichAudioDataLoader,
    RichAudioSample,
    VocalSetDataset,
    VocalSoundDataset,
    create_emotion_loader,
    create_meld_loader,
    create_paralinguistics_loader,
    create_pitch_loader,
)


class TestRichAudioSample:
    """Tests for RichAudioSample dataclass."""

    def test_create_sample(self):
        """Test creating a basic sample."""
        sample = RichAudioSample(
            audio_path="/path/to/audio.wav",
            emotion_label=3,
        )
        assert sample.audio_path == "/path/to/audio.wav"
        assert sample.emotion_label == 3
        assert sample.text is None
        assert sample.pitch_hz is None

    def test_sample_with_all_fields(self):
        """Test creating sample with all fields."""
        sample = RichAudioSample(
            audio_path="/path/to/audio.wav",
            text="hello world",
            emotion_label=3,
            pitch_hz=440.0,
            phonemes=[1, 2, 3],
            language="en",
            paralinguistic_label=5,
            duration=2.5,
            speaker_id="SPK001",
        )
        assert sample.audio_path == "/path/to/audio.wav"
        assert sample.text == "hello world"
        assert sample.emotion_label == 3
        assert sample.pitch_hz == 440.0
        assert sample.phonemes == [1, 2, 3]
        assert sample.language == "en"
        assert sample.paralinguistic_label == 5
        assert sample.duration == 2.5
        assert sample.speaker_id == "SPK001"


class TestEmotionLabels:
    """Tests for emotion label mappings."""

    def test_emotion_labels_count(self):
        """Test correct number of emotion labels."""
        assert len(EMOTION_LABELS) == 8

    def test_emotion_labels_content(self):
        """Test emotion label names."""
        expected = [
            "anger",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
            "other",
        ]
        assert EMOTION_LABELS == expected


class TestParalinguisticLabels:
    """Tests for paralinguistic label mappings."""

    def test_paralinguistic_labels_count(self):
        """Test correct number of paralinguistic labels."""
        assert len(PARALINGUISTIC_LABELS) == 6

    def test_paralinguistic_labels_content(self):
        """Test paralinguistic label names."""
        expected = [
            "laughter",
            "sigh",
            "cough",
            "throat_clearing",
            "sneeze",
            "sniff",
        ]
        assert PARALINGUISTIC_LABELS == expected


class TestCREMADFilenameParser:
    """Tests for CREMA-D filename parsing."""

    def test_parse_valid_filename(self):
        """Test parsing valid CREMA-D filename."""
        # Create mock dataset and test parsing
        dataset = CREMADDataset.__new__(CREMADDataset)
        dataset.root_dir = Path("/tmp")
        dataset.split = "train"
        dataset.val_ratio = 0.1
        dataset.samples = []

        result = dataset._parse_filename("1001_DFA_ANG_XX.wav")
        assert result is not None
        assert result["actor_id"] == "1001"
        assert result["statement"] == "DFA"
        assert result["emotion"] == "ANG"
        assert result["level"] == "XX"
        assert result["emotion_label"] == 0  # ANG = anger = 0

    def test_parse_all_emotions(self):
        """Test parsing all CREMA-D emotion codes."""
        dataset = CREMADDataset.__new__(CREMADDataset)
        dataset.root_dir = Path("/tmp")

        emotions = {
            "ANG": 0,  # Anger
            "DIS": 1,  # Disgust
            "FEA": 2,  # Fear
            "HAP": 3,  # Happy
            "NEU": 4,  # Neutral
            "SAD": 5,  # Sad
        }

        for code, expected_label in emotions.items():
            result = dataset._parse_filename(f"1001_IEO_{code}_HI.wav")
            assert result is not None
            assert result["emotion_label"] == expected_label

    def test_parse_invalid_filename(self):
        """Test parsing invalid filename."""
        dataset = CREMADDataset.__new__(CREMADDataset)
        dataset.root_dir = Path("/tmp")

        # Unknown emotion
        result = dataset._parse_filename("1001_DFA_XYZ_XX.wav")
        assert result is None

        # Wrong format
        result = dataset._parse_filename("invalid.wav")
        assert result is None


class TestCREMADDataset:
    """Tests for CREMA-D dataset loading."""

    @pytest.fixture
    def crema_d_dir(self):
        """Path to CREMA-D data."""
        return Path("data/emotion/crema-d")

    def test_load_crema_d_if_available(self, crema_d_dir):
        """Test loading CREMA-D dataset if available."""
        if not crema_d_dir.exists():
            pytest.skip("CREMA-D data not available")

        dataset = CREMADDataset(str(crema_d_dir), split="train")
        assert len(dataset) > 0
        print(f"Loaded {len(dataset)} CREMA-D samples")

    def test_train_val_split(self, crema_d_dir):
        """Test train/val split."""
        if not crema_d_dir.exists():
            pytest.skip("CREMA-D data not available")

        train_dataset = CREMADDataset(str(crema_d_dir), split="train", val_ratio=0.1)
        val_dataset = CREMADDataset(str(crema_d_dir), split="val", val_ratio=0.1)

        # Train should be larger than val
        assert len(train_dataset) > len(val_dataset)

        # Total should be close to expected
        total = len(train_dataset) + len(val_dataset)
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Total: {total}")


class TestVocalSetDataset:
    """Tests for VocalSet dataset loading."""

    @pytest.fixture
    def vocalset_dir(self):
        """Path to VocalSet data."""
        return Path("data/vocalset/FULL")

    def test_load_vocalset_if_available(self, vocalset_dir):
        """Test loading VocalSet dataset if available."""
        if not vocalset_dir.exists():
            pytest.skip("VocalSet data not available")

        dataset = VocalSetDataset(str(vocalset_dir), split="train")
        assert len(dataset) > 0
        print(f"Loaded {len(dataset)} VocalSet samples")

    def test_pitch_estimation(self, vocalset_dir):
        """Test pitch estimation function."""
        if not vocalset_dir.exists():
            pytest.skip("VocalSet data not available")

        dataset = VocalSetDataset(str(vocalset_dir), split="train")

        # Test pitch estimation on a sample
        if len(dataset) > 0:
            sample = dataset[0]
            assert "pitch_hz" in sample
            pitch = float(sample["pitch_hz"].item())
            # Pitch should be either 0 (unvoiced) or in reasonable range
            assert pitch == 0 or (80 <= pitch <= 600)
            print(f"Estimated pitch: {pitch:.1f} Hz")


class TestVocalSoundDataset:
    """Tests for VocalSound paralinguistics dataset loading."""

    @pytest.fixture
    def vocalsound_dir(self):
        """Path to VocalSound data."""
        return Path("data/paralinguistics/vocalsound_labeled")

    def test_load_vocalsound_if_available(self, vocalsound_dir):
        """Test loading VocalSound dataset if available."""
        if not vocalsound_dir.exists():
            pytest.skip("VocalSound data not available")

        dataset = VocalSoundDataset(str(vocalsound_dir), split="train")
        assert len(dataset) > 0
        print(f"Loaded {len(dataset)} VocalSound train samples")

    def test_train_test_split(self, vocalsound_dir):
        """Test train/test split."""
        if not vocalsound_dir.exists():
            pytest.skip("VocalSound data not available")

        train_dataset = VocalSoundDataset(str(vocalsound_dir), split="train")
        test_dataset = VocalSoundDataset(str(vocalsound_dir), split="test")

        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    def test_getitem_returns_paralinguistic_label(self, vocalsound_dir):
        """Test that __getitem__ returns paralinguistic labels."""
        if not vocalsound_dir.exists():
            pytest.skip("VocalSound data not available")

        dataset = VocalSoundDataset(str(vocalsound_dir), split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "features" in sample
            assert "feature_lengths" in sample
            assert "paralinguistic_label" in sample
            # Label should be in range 0-5
            label = int(sample["paralinguistic_label"].item())
            assert 0 <= label <= 5


class TestRichAudioDataLoader:
    """Tests for rich audio data loader."""

    def test_collate_batch_emotion(self):
        """Test batch collation with emotion labels."""
        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.samples = [
                    {"features": mx.zeros((100, 80)), "feature_lengths": mx.array([100]), "emotion_label": mx.array([0])},
                    {"features": mx.zeros((150, 80)), "feature_lengths": mx.array([150]), "emotion_label": mx.array([3])},
                    {"features": mx.zeros((120, 80)), "feature_lengths": mx.array([120]), "emotion_label": mx.array([5])},
                ]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        dataset = MockDataset()
        loader = RichAudioDataLoader(dataset, batch_size=2, shuffle=False)

        batch = loader._collate_batch(dataset.samples[:2])

        # Check shapes
        assert batch["features"].shape == (2, 150, 80)  # Padded to max length
        assert batch["feature_lengths"].shape == (2,)
        assert batch["emotion_labels"].shape == (2,)

        # Check values
        assert int(batch["feature_lengths"][0]) == 100
        assert int(batch["feature_lengths"][1]) == 150

    def test_collate_batch_pitch(self):
        """Test batch collation with pitch labels."""
        class MockDataset:
            def __init__(self):
                self.samples = [
                    {"features": mx.zeros((100, 80)), "feature_lengths": mx.array([100]), "pitch_hz": mx.array([220.0])},
                    {"features": mx.zeros((80, 80)), "feature_lengths": mx.array([80]), "pitch_hz": mx.array([440.0])},
                ]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        dataset = MockDataset()
        loader = RichAudioDataLoader(dataset, batch_size=2, shuffle=False)

        batch = loader._collate_batch(dataset.samples)

        assert "pitch_hz" in batch
        assert batch["pitch_hz"].shape == (2,)

    def test_collate_batch_paralinguistic(self):
        """Test batch collation with paralinguistic labels."""
        class MockDataset:
            def __init__(self):
                self.samples = [
                    {"features": mx.zeros((100, 80)), "feature_lengths": mx.array([100]), "paralinguistic_label": mx.array([0])},
                    {"features": mx.zeros((80, 80)), "feature_lengths": mx.array([80]), "paralinguistic_label": mx.array([2])},
                    {"features": mx.zeros((120, 80)), "feature_lengths": mx.array([120]), "paralinguistic_label": mx.array([5])},
                ]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        dataset = MockDataset()
        loader = RichAudioDataLoader(dataset, batch_size=3, shuffle=False)

        batch = loader._collate_batch(dataset.samples)

        assert "paralinguistic_labels" in batch
        assert batch["paralinguistic_labels"].shape == (3,)
        # Check values
        assert int(batch["paralinguistic_labels"][0]) == 0  # laughter
        assert int(batch["paralinguistic_labels"][1]) == 2  # cough
        assert int(batch["paralinguistic_labels"][2]) == 5  # sniff

    def test_loader_length(self):
        """Test loader length calculation."""
        class MockDataset:
            def __len__(self):
                return 100

        dataset = MockDataset()
        loader = RichAudioDataLoader(dataset, batch_size=16)

        assert len(loader) == 100 // 16  # 6 batches


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_emotion_loader(self):
        """Test emotion loader creation."""
        crema_d_dir = Path("data/emotion/crema-d")
        if not crema_d_dir.exists():
            pytest.skip("CREMA-D data not available")

        loader = create_emotion_loader(
            data_dir=str(crema_d_dir),
            split="train",
            batch_size=8,
        )

        assert len(loader) > 0
        print(f"Created emotion loader with {len(loader)} batches")

    def test_create_pitch_loader(self):
        """Test pitch loader creation."""
        vocalset_dir = Path("data/vocalset/FULL")
        if not vocalset_dir.exists():
            pytest.skip("VocalSet data not available")

        loader = create_pitch_loader(
            data_dir=str(vocalset_dir),
            split="train",
            batch_size=8,
        )

        assert len(loader) > 0
        print(f"Created pitch loader with {len(loader)} batches")

    def test_create_paralinguistics_loader(self):
        """Test paralinguistics loader creation."""
        vocalsound_dir = Path("data/paralinguistics/vocalsound_labeled")
        if not vocalsound_dir.exists():
            pytest.skip("VocalSound data not available")

        loader = create_paralinguistics_loader(
            data_dir=str(vocalsound_dir),
            split="train",
            batch_size=8,
        )

        assert len(loader) > 0
        print(f"Created paralinguistics loader with {len(loader)} batches")


class TestIntegration:
    """Integration tests with real data."""

    def test_load_and_iterate_crema_d(self):
        """Test loading and iterating CREMA-D."""
        crema_d_dir = Path("data/emotion/crema-d")
        if not crema_d_dir.exists():
            pytest.skip("CREMA-D data not available")

        loader = create_emotion_loader(
            data_dir=str(crema_d_dir),
            split="train",
            batch_size=4,
        )

        # Iterate one batch
        for batch in loader:
            assert "features" in batch
            assert "feature_lengths" in batch
            assert "emotion_labels" in batch

            assert batch["features"].ndim == 3
            assert batch["emotion_labels"].ndim == 1

            # All emotion labels should be valid (0-5 for CREMA-D)
            for label in batch["emotion_labels"].tolist():
                assert 0 <= label <= 5

            print(f"Batch shape: {batch['features'].shape}")
            print(f"Emotion labels: {batch['emotion_labels'].tolist()}")
            break  # Just check first batch


class TestMELDDataset:
    """Tests for MELD emotion dataset."""

    @pytest.fixture
    def meld_dir(self):
        """Path to MELD data."""
        return Path("data/emotion_punctuation/MELD.Raw")

    def test_meld_emotion_mapping(self):
        """Test MELD emotion label mapping."""
        # Check all 7 MELD emotions are mapped
        assert len(MELD_EMOTION_TO_INT) == 7
        expected_emotions = {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
        assert set(MELD_EMOTION_TO_INT.keys()) == expected_emotions

        # Check labels are in valid range
        for emotion, label in MELD_EMOTION_TO_INT.items():
            assert 0 <= label <= 6, f"Invalid label {label} for {emotion}"

    def test_load_meld_if_available(self, meld_dir):
        """Test loading MELD dataset if available."""
        if not meld_dir.exists():
            pytest.skip("MELD data not available")

        dataset = MELDDataset(str(meld_dir), split="train")
        assert len(dataset) > 0
        print(f"Loaded {len(dataset)} MELD train samples")

    def test_meld_splits(self, meld_dir):
        """Test MELD train/dev/test splits."""
        if not meld_dir.exists():
            pytest.skip("MELD data not available")

        train_dataset = MELDDataset(str(meld_dir), split="train")
        dev_dataset = MELDDataset(str(meld_dir), split="dev")
        test_dataset = MELDDataset(str(meld_dir), split="test")

        assert len(train_dataset) > 0
        assert len(dev_dataset) > 0
        assert len(test_dataset) > 0
        print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

    def test_meld_getitem_returns_emotion_label(self, meld_dir):
        """Test that __getitem__ returns emotion labels."""
        if not meld_dir.exists():
            pytest.skip("MELD data not available")

        dataset = MELDDataset(str(meld_dir), split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "features" in sample
            assert "feature_lengths" in sample
            assert "emotion_label" in sample
            # MELD has 7 classes (0-6)
            label = int(sample["emotion_label"].item())
            assert 0 <= label <= 6

    def test_create_meld_loader(self, meld_dir):
        """Test MELD loader creation."""
        if not meld_dir.exists():
            pytest.skip("MELD data not available")

        loader = create_meld_loader(
            data_dir=str(meld_dir),
            split="train",
            batch_size=8,
        )

        assert len(loader) > 0
        print(f"Created MELD loader with {len(loader)} batches")

    def test_load_and_iterate_meld(self, meld_dir):
        """Test loading and iterating MELD."""
        if not meld_dir.exists():
            pytest.skip("MELD data not available")

        loader = create_meld_loader(
            data_dir=str(meld_dir),
            split="train",
            batch_size=4,
        )

        # Iterate one batch
        for batch in loader:
            assert "features" in batch
            assert "feature_lengths" in batch
            assert "emotion_labels" in batch

            assert batch["features"].ndim == 3
            assert batch["emotion_labels"].ndim == 1

            # All emotion labels should be valid (0-6 for MELD)
            for label in batch["emotion_labels"].tolist():
                assert 0 <= label <= 6

            print(f"Batch shape: {batch['features'].shape}")
            print(f"Emotion labels: {batch['emotion_labels'].tolist()}")
            break  # Just check first batch
