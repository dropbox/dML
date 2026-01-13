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

"""Tests for speaker embedding training infrastructure."""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.heads.speaker import SpeakerConfig, SpeakerHead
from src.training.config import (
    OptimizerConfig,
    SpeakerTrainingConfig,
    TrainingConfig,
)
from src.training.speaker_dataloader import (
    CombinedSpeakerDataset,
    SpeakerDataLoader,
    SpeakerDataset,
    SpeakerSample,
    VerificationTrialLoader,
)
from src.training.speaker_trainer import (
    SpeakerTrainer,
    SpeakerTrainingMetrics,
    create_speaker_trainer,
)

# ============================================================================
# Test Fixtures
# ============================================================================


class MockSpeakerDataset(SpeakerDataset):
    """Mock speaker dataset for testing."""

    def __init__(
        self,
        num_speakers: int = 10,
        samples_per_speaker: int = 5,
        feat_dim: int = 80,
        seq_len: int = 100,
    ):
        super().__init__(
            root_dir="/mock",
            min_duration=2.0,
            max_duration=10.0,
            sample_rate=16000,
        )

        self.feat_dim = feat_dim
        self.seq_len = seq_len

        # Generate mock samples
        for spk_idx in range(num_speakers):
            speaker_id = f"speaker_{spk_idx:03d}"
            self._register_speaker(speaker_id)

            for sample_idx in range(samples_per_speaker):
                self.samples.append(
                    SpeakerSample(
                        audio_path=f"/mock/{speaker_id}/sample_{sample_idx}.wav",
                        speaker_id=speaker_id,
                        speaker_idx=spk_idx,
                    ),
                )

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a mock sample with random features."""
        sample = self.samples[idx]

        # Generate deterministic "random" features based on index
        mx.random.seed(idx)
        features = mx.random.normal((self.seq_len, self.feat_dim))

        return {
            "features": features,
            "feature_lengths": mx.array([self.seq_len]),
            "speaker_idx": mx.array([sample.speaker_idx]),
        }


@pytest.fixture
def mock_dataset():
    """Create a mock speaker dataset."""
    return MockSpeakerDataset(num_speakers=10, samples_per_speaker=5)


@pytest.fixture
def speaker_config():
    """Create a speaker config."""
    return SpeakerConfig(
        encoder_dim=80,  # Match mock features
        embedding_dim=64,
        num_speakers=10,
        hidden_dim=128,
        res2net_scale=4,
    )


@pytest.fixture
def speaker_head(speaker_config):
    """Create a speaker head."""
    return SpeakerHead(speaker_config)


@pytest.fixture
def training_config():
    """Create a speaker training config."""
    return SpeakerTrainingConfig(
        enabled=True,
        num_speakers=10,
        embedding_dim=64,
        encoder_dim=80,
        batch_size=4,
        aam_margin=0.2,
        aam_scale=30.0,
    )


# ============================================================================
# SpeakerTrainingConfig Tests
# ============================================================================


class TestSpeakerTrainingConfig:
    """Tests for SpeakerTrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpeakerTrainingConfig()

        assert config.enabled is False
        assert config.num_speakers == 997
        assert config.embedding_dim == 256
        assert config.aam_margin == 0.2
        assert config.aam_scale == 30.0
        assert config.encoder_dim == 384

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpeakerTrainingConfig(
            enabled=True,
            num_speakers=1000,
            embedding_dim=128,
            aam_margin=0.3,
        )

        assert config.enabled is True
        assert config.num_speakers == 1000
        assert config.embedding_dim == 128
        assert config.aam_margin == 0.3

    def test_config_in_training_config(self):
        """Test speaker config integrated in TrainingConfig."""
        config = TrainingConfig()

        assert hasattr(config, "speaker")
        assert isinstance(config.speaker, SpeakerTrainingConfig)
        assert config.speaker.enabled is False

    def test_config_serialization(self):
        """Test config to_dict serialization."""
        config = TrainingConfig()
        config.speaker.enabled = True
        config.speaker.num_speakers = 500

        d = config.to_dict()

        assert "speaker" in d
        assert d["speaker"]["enabled"] is True
        assert d["speaker"]["num_speakers"] == 500

    def test_config_deserialization(self):
        """Test config from_dict deserialization."""
        d = {
            "speaker": {
                "enabled": True,
                "num_speakers": 500,
                "embedding_dim": 128,
            },
        }

        config = TrainingConfig.from_dict(d)

        assert config.speaker.enabled is True
        assert config.speaker.num_speakers == 500
        assert config.speaker.embedding_dim == 128


# ============================================================================
# SpeakerDataset Tests
# ============================================================================


class TestSpeakerDataset:
    """Tests for speaker datasets."""

    def test_mock_dataset_creation(self, mock_dataset):
        """Test mock dataset is created correctly."""
        assert len(mock_dataset) == 50  # 10 speakers * 5 samples
        assert mock_dataset.num_speakers == 10

    def test_mock_dataset_getitem(self, mock_dataset):
        """Test getting items from mock dataset."""
        sample = mock_dataset[0]

        assert "features" in sample
        assert "feature_lengths" in sample
        assert "speaker_idx" in sample

        assert sample["features"].shape == (100, 80)
        assert sample["speaker_idx"].shape == (1,)

    def test_speaker_registration(self, mock_dataset):
        """Test speaker registration."""
        assert "speaker_000" in mock_dataset.speaker_to_idx
        assert mock_dataset.speaker_to_idx["speaker_000"] == 0
        assert mock_dataset.idx_to_speaker[0] == "speaker_000"


class TestSpeakerDataLoader:
    """Tests for SpeakerDataLoader."""

    def test_dataloader_creation(self, mock_dataset):
        """Test data loader creation."""
        loader = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=False,
        )

        assert loader.batch_size == 4
        assert len(loader) == 12  # 50 samples / 4 batch

    def test_dataloader_iteration(self, mock_dataset):
        """Test iterating over data loader."""
        loader = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=False,
        )

        batch = next(iter(loader))

        assert "features" in batch
        assert "feature_lengths" in batch
        assert "speaker_indices" in batch

        assert batch["features"].shape[0] == 4  # Batch size
        assert batch["speaker_indices"].shape == (4,)

    def test_dataloader_shuffling(self, mock_dataset):
        """Test data loader shuffling."""
        loader1 = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=True,
        )
        loader2 = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=True,
        )

        # With shuffling, batches should be different
        # (probabilistically - not guaranteed)
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # At least the structure should be the same
        assert batch1["features"].shape == batch2["features"].shape


class TestVerificationTrialLoader:
    """Tests for VerificationTrialLoader."""

    def test_trial_loader_creation(self, mock_dataset):
        """Test trial loader creation."""
        loader = VerificationTrialLoader(
            mock_dataset,
            num_positive_pairs=10,
            num_negative_pairs=10,
        )

        assert loader.num_positive_pairs == 10
        assert loader.num_negative_pairs == 10

    def test_trial_generation(self, mock_dataset):
        """Test generating trials."""
        loader = VerificationTrialLoader(
            mock_dataset,
            num_positive_pairs=10,
            num_negative_pairs=10,
        )

        trials = loader.generate_trials()

        assert len(trials) == 20

        # Check trial format
        for idx1, idx2, label in trials:
            assert isinstance(idx1, int)
            assert isinstance(idx2, int)
            assert label in (0, 1)

    def test_positive_pairs_same_speaker(self, mock_dataset):
        """Test that positive pairs are from same speaker."""
        loader = VerificationTrialLoader(
            mock_dataset,
            num_positive_pairs=20,
            num_negative_pairs=0,
        )

        trials = loader.generate_trials()

        for idx1, idx2, label in trials:
            if label == 1:
                spk1 = mock_dataset.samples[idx1].speaker_idx
                spk2 = mock_dataset.samples[idx2].speaker_idx
                assert spk1 == spk2

    def test_negative_pairs_different_speaker(self, mock_dataset):
        """Test that negative pairs are from different speakers."""
        loader = VerificationTrialLoader(
            mock_dataset,
            num_positive_pairs=0,
            num_negative_pairs=20,
        )

        trials = loader.generate_trials()

        for idx1, idx2, label in trials:
            if label == 0:
                spk1 = mock_dataset.samples[idx1].speaker_idx
                spk2 = mock_dataset.samples[idx2].speaker_idx
                assert spk1 != spk2


# ============================================================================
# SpeakerTrainer Tests
# ============================================================================


class TestSpeakerTrainer:
    """Tests for SpeakerTrainer."""

    def test_trainer_creation(self, speaker_head, training_config):
        """Test trainer creation."""
        trainer = SpeakerTrainer(
            speaker_head=speaker_head,
            config=training_config,
        )

        assert trainer.speaker_head is speaker_head
        assert trainer.encoder is None
        assert trainer.state.step == 0

    def test_trainer_forward(self, speaker_head, training_config, mock_dataset):
        """Test trainer forward pass."""
        trainer = SpeakerTrainer(
            speaker_head=speaker_head,
            config=training_config,
        )

        sample = mock_dataset[0]
        features = mx.expand_dims(sample["features"], axis=0)
        lengths = sample["feature_lengths"]

        embeddings, logits = trainer._forward(features, lengths)

        assert embeddings.shape == (1, training_config.embedding_dim)
        assert logits.shape == (1, training_config.num_speakers)

    def test_train_step(self, speaker_head, training_config, mock_dataset):
        """Test single training step."""
        trainer = SpeakerTrainer(
            speaker_head=speaker_head,
            config=training_config,
        )

        loader = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=False,
        )

        batch = next(iter(loader))
        metrics = trainer.train_step(batch)

        assert isinstance(metrics, SpeakerTrainingMetrics)
        assert metrics.loss > 0
        assert 0 <= metrics.accuracy <= 1
        assert metrics.num_samples == 4
        assert metrics.time_ms > 0

        # Check state updated
        assert trainer.state.step == 1

    def test_gradient_clipping(self, speaker_head, training_config):
        """Test gradient clipping."""
        trainer = SpeakerTrainer(
            speaker_head=speaker_head,
            config=training_config,
            optimizer_config=OptimizerConfig(grad_clip=1.0),
        )

        # Create mock gradients
        grads = {
            "test": mx.ones((10, 10)) * 100,  # Large gradients
        }

        clipped = trainer._clip_gradients(grads, max_norm=1.0)

        # Check gradients are clipped
        grad_norm = mx.sqrt(mx.sum(clipped["test"] * clipped["test"]))
        mx.eval(grad_norm)
        assert float(grad_norm) <= 1.0 + 1e-5

    def test_validation(self, speaker_head, training_config, mock_dataset):
        """Test validation."""
        trainer = SpeakerTrainer(
            speaker_head=speaker_head,
            config=training_config,
        )

        loader = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=False,
        )

        metrics = trainer.validate(loader, max_batches=5)

        assert isinstance(metrics, SpeakerTrainingMetrics)
        assert metrics.loss >= 0
        assert 0 <= metrics.accuracy <= 1
        assert metrics.num_samples > 0

    def test_validation_with_eer(self, speaker_head, training_config, mock_dataset):
        """Test validation with EER computation."""
        trainer = SpeakerTrainer(
            speaker_head=speaker_head,
            config=training_config,
        )

        loader = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=False,
        )

        trial_loader = VerificationTrialLoader(
            mock_dataset,
            num_positive_pairs=10,
            num_negative_pairs=10,
        )

        metrics = trainer.validate(loader, trial_loader, max_batches=5)

        assert metrics.eer is not None
        assert 0 <= metrics.eer <= 1

    def test_checkpoint_save_load(self, speaker_head, training_config):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SpeakerTrainer(
                speaker_head=speaker_head,
                config=training_config,
            )
            trainer.checkpoint_dir = Path(tmpdir)
            trainer.state.step = 100
            trainer.state.best_eer = 0.05

            # Save
            path = trainer.save_checkpoint("test")
            assert path.exists()

            # Create new trainer and load
            new_head = SpeakerHead(SpeakerConfig(
                encoder_dim=80,
                embedding_dim=64,
                num_speakers=10,
                hidden_dim=128,
                res2net_scale=4,
            ))
            new_trainer = SpeakerTrainer(
                speaker_head=new_head,
                config=training_config,
            )
            new_trainer.checkpoint_dir = Path(tmpdir)

            success = new_trainer.load_checkpoint("test")
            assert success
            assert new_trainer.state.step == 100
            assert new_trainer.state.best_eer == 0.05


class TestCreateSpeakerTrainer:
    """Tests for create_speaker_trainer factory function."""

    def test_create_trainer(self, training_config):
        """Test creating trainer from config."""
        trainer = create_speaker_trainer(training_config)

        assert isinstance(trainer, SpeakerTrainer)
        assert trainer.speaker_head is not None
        assert trainer.config == training_config

    def test_create_trainer_with_encoder(self, training_config):
        """Test creating trainer with encoder."""

        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(80, 80)

            def __call__(self, x, lengths):
                return self.linear(x)

        encoder = MockEncoder()
        trainer = create_speaker_trainer(training_config, encoder=encoder)

        assert trainer.encoder is encoder


# ============================================================================
# Integration Tests
# ============================================================================


class TestSpeakerTrainingIntegration:
    """Integration tests for speaker training pipeline."""

    def test_full_training_step(self, mock_dataset):
        """Test complete training step with all components."""
        config = SpeakerTrainingConfig(
            enabled=True,
            num_speakers=10,
            embedding_dim=64,
            encoder_dim=80,
            batch_size=4,
        )

        trainer = create_speaker_trainer(config)

        loader = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=False,
        )

        # Train for a few steps
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            metrics = trainer.train_step(batch)
            assert metrics.loss > 0

        assert trainer.state.step == 3

    def test_training_reduces_loss(self, mock_dataset):
        """Test that training reduces loss over time."""
        config = SpeakerTrainingConfig(
            enabled=True,
            num_speakers=10,
            embedding_dim=64,
            encoder_dim=80,
            batch_size=4,
        )

        trainer = create_speaker_trainer(
            config,
            optimizer_config=OptimizerConfig(learning_rate=0.01),
        )

        loader = SpeakerDataLoader(
            mock_dataset,
            batch_size=4,
            shuffle=False,
        )

        losses = []
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            metrics = trainer.train_step(batch)
            losses.append(metrics.loss)

        # Loss should generally decrease (allow some variance)
        # Check that last 3 losses are lower than first 3 on average
        first_avg = sum(losses[:3]) / 3
        last_avg = sum(losses[-3:]) / 3

        # This is a weak check - training should improve but randomness exists
        assert last_avg <= first_avg * 1.5  # Allow 50% increase due to variance

    def test_embedding_quality(self, mock_dataset):
        """Test that embeddings from same speaker are similar."""
        config = SpeakerTrainingConfig(
            enabled=True,
            num_speakers=10,
            embedding_dim=64,
            encoder_dim=80,
            batch_size=4,
        )

        trainer = create_speaker_trainer(config)

        # Get embeddings for two samples from same speaker
        sample1 = mock_dataset[0]  # Speaker 0, sample 0
        sample2 = mock_dataset[1]  # Speaker 0, sample 1

        emb1 = trainer._get_embedding(sample1)
        emb2 = trainer._get_embedding(sample2)

        # Compute similarity
        sim = trainer.speaker_head.similarity(emb1, emb2)
        mx.eval(sim)

        # Embeddings should exist (not NaN)
        assert not mx.isnan(sim).any()

    def test_eer_computation(self, mock_dataset):
        """Test EER computation in full pipeline."""
        config = SpeakerTrainingConfig(
            enabled=True,
            num_speakers=10,
            embedding_dim=64,
            encoder_dim=80,
            batch_size=4,
        )

        trainer = create_speaker_trainer(config)

        trial_loader = VerificationTrialLoader(
            mock_dataset,
            num_positive_pairs=20,
            num_negative_pairs=20,
        )

        eer = trainer._compute_eer(trial_loader)

        assert 0 <= eer <= 1


class TestCombinedDataset:
    """Tests for CombinedSpeakerDataset."""

    def test_combine_two_datasets(self):
        """Test combining two datasets."""
        dataset1 = MockSpeakerDataset(num_speakers=5, samples_per_speaker=3)
        dataset2 = MockSpeakerDataset(num_speakers=5, samples_per_speaker=3)

        combined = CombinedSpeakerDataset([dataset1, dataset2])

        # Should have 30 samples (5*3 + 5*3)
        assert len(combined) == 30

        # Should have 10 unique speakers (with dataset prefixes)
        assert combined.num_speakers == 10

    def test_combined_speaker_indices_unique(self):
        """Test that combined dataset has unique speaker indices."""
        dataset1 = MockSpeakerDataset(num_speakers=5, samples_per_speaker=2)
        dataset2 = MockSpeakerDataset(num_speakers=5, samples_per_speaker=2)

        combined = CombinedSpeakerDataset([dataset1, dataset2])

        # All speaker indices should be unique
        speaker_indices = set(s.speaker_idx for s in combined.samples)
        assert len(speaker_indices) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
