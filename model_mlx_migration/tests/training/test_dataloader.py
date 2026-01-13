# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for ASR data loading."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from src.training.dataloader import (
    ASRDataLoader,
    AudioSample,
    LibriSpeechDataset,
    _create_mel_filterbank,
    compute_fbank_features,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestAudioSample:
    """Tests for AudioSample dataclass."""

    def test_create_sample(self):
        """Test creating an audio sample."""
        sample = AudioSample(
            audio_path="/path/to/audio.flac",
            text="hello world",
            speaker_id="spk001",
            duration=2.5,
        )

        assert sample.audio_path == "/path/to/audio.flac"
        assert sample.text == "hello world"
        assert sample.speaker_id == "spk001"
        assert sample.duration == 2.5

    def test_optional_fields(self):
        """Test optional fields default to None."""
        sample = AudioSample(
            audio_path="/path/to/audio.flac",
            text="test",
        )

        assert sample.speaker_id is None
        assert sample.duration is None


class TestMelFilterbank:
    """Tests for mel filterbank computation."""

    def test_filterbank_shape(self):
        """Test filterbank has correct shape."""
        n_fft = 512
        n_mels = 80
        sr = 16000

        fb = _create_mel_filterbank(n_fft, sr, n_mels)

        assert fb.shape == (n_mels, n_fft // 2 + 1)

    def test_filterbank_values(self):
        """Test filterbank has valid values."""
        fb = _create_mel_filterbank(512, 16000, 80)

        # All values should be non-negative
        assert np.all(fb >= 0)

        # Most filters should have non-zero values (some edge filters may be empty)
        non_empty_filters = sum(1 for i in range(fb.shape[0]) if np.sum(fb[i] > 0) > 0)
        assert non_empty_filters >= fb.shape[0] * 0.9  # At least 90% non-empty


class TestFbankFeatures:
    """Tests for filterbank feature computation."""

    def test_compute_features(self):
        """Test computing filterbank features."""
        # Create synthetic audio (1 second)
        sr = 16000
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.arange(int(sr * duration)) / sr)
        audio = audio.astype(np.float32)

        features = compute_fbank_features(audio, sr, n_mels=80)

        # Check output shape
        # 1 second at 10ms frame shift = ~100 frames
        assert features.shape[1] == 80
        assert 90 <= features.shape[0] <= 110  # ~100 frames with tolerance

    def test_feature_dtype(self):
        """Test features have correct dtype."""
        audio = _rng.standard_normal(16000).astype(np.float32)
        features = compute_fbank_features(audio)

        assert features.dtype == np.float32


class TestLibriSpeechDataset:
    """Tests for LibriSpeech dataset loading."""

    def test_default_tokenizer(self):
        """Test the default character tokenizer."""
        from src.training.dataloader import LibriSpeechDataset

        # Test tokenizer directly without needing a full dataset
        ds = object.__new__(LibriSpeechDataset)
        tokens = ds._default_tokenizer("hello")

        # h=8, e=5, l=12, l=12, o=15
        assert len(tokens) == 5
        assert all(isinstance(t, int) for t in tokens)

    def test_tokenizer_handles_space(self):
        """Test tokenizer handles spaces."""
        from src.training.dataloader import LibriSpeechDataset

        ds = object.__new__(LibriSpeechDataset)
        tokens = ds._default_tokenizer("hi there")

        # Should include space token (27)
        assert 27 in tokens


class TestASRDataLoader:
    """Tests for ASR data loader."""

    def test_collate_batch(self):
        """Test batch collation with padding."""

        class MockDataset:
            def __init__(self):
                self.samples = [
                    {
                        "features": mx.ones((10, 80)),
                        "feature_lengths": mx.array([10]),
                        "targets": mx.array([1, 2, 3]),
                        "target_lengths": mx.array([3]),
                    },
                    {
                        "features": mx.ones((15, 80)),
                        "feature_lengths": mx.array([15]),
                        "targets": mx.array([4, 5]),
                        "target_lengths": mx.array([2]),
                    },
                ]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        dataset = MockDataset()
        loader = ASRDataLoader(dataset, batch_size=2, shuffle=False)

        batch = loader._collate_batch(dataset.samples)

        # Check shapes
        assert batch["features"].shape == (2, 15, 80)  # Padded to max length
        assert batch["targets"].shape == (2, 3)  # Padded to max length
        assert batch["feature_lengths"].shape == (2,)
        assert batch["target_lengths"].shape == (2,)

        # Check lengths are preserved
        assert batch["feature_lengths"][0].item() == 10
        assert batch["feature_lengths"][1].item() == 15

    def test_loader_iteration(self):
        """Test iterating over data loader."""

        class MockDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "features": mx.ones((20, 80)),
                    "feature_lengths": mx.array([20]),
                    "targets": mx.array([1, 2, 3]),
                    "target_lengths": mx.array([3]),
                }

        dataset = MockDataset()
        loader = ASRDataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)

        batches = list(loader)

        # Should have 2 batches (10 samples / 4 = 2, dropping last)
        assert len(batches) == 2

        # Each batch should have batch_size samples
        assert batches[0]["features"].shape[0] == 4

    def test_max_duration_filter(self):
        """Test filtering samples by max duration."""

        class MockDataset:
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                # Return progressively longer samples
                length = (idx + 1) * 500  # 500, 1000, 1500 frames
                return {
                    "features": mx.ones((length, 80)),
                    "feature_lengths": mx.array([length]),
                    "targets": mx.array([1, 2]),
                    "target_lengths": mx.array([2]),
                }

        dataset = MockDataset()
        # 10 seconds = 1000 frames at 100 fps
        loader = ASRDataLoader(
            dataset, batch_size=3, shuffle=False, max_duration=10.0, drop_last=False,
        )

        batches = list(loader)

        # Only samples with <= 1000 frames should be included
        # First two samples have 500 and 1000 frames
        assert len(batches) == 1
        assert batches[0]["features"].shape[0] == 2


class TestDataLoaderIntegration:
    """Integration tests with actual LibriSpeech data (if available)."""

    @pytest.fixture
    def librispeech_path(self):
        """Get LibriSpeech path if available."""
        path = Path("/Users/ayates/model_mlx_migration/data/LibriSpeech")
        if not path.exists():
            pytest.skip("LibriSpeech not available")
        return path

    def test_load_dev_clean(self, librispeech_path):
        """Test loading dev-clean split."""
        dataset = LibriSpeechDataset(str(librispeech_path), "dev-clean")

        # Should have loaded some samples
        assert len(dataset) > 0

        # Check first sample
        sample = dataset[0]
        assert "features" in sample
        assert "targets" in sample
        assert sample["features"].shape[1] == 80

    def test_iterate_batches(self, librispeech_path):
        """Test iterating over batches."""
        dataset = LibriSpeechDataset(str(librispeech_path), "dev-clean")
        loader = ASRDataLoader(dataset, batch_size=2, shuffle=False)

        # Get first batch
        batch = next(iter(loader))

        assert batch["features"].shape[0] == 2
        assert batch["features"].shape[2] == 80
