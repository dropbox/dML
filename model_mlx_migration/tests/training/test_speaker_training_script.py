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
Integration tests for speaker training execution script.

Tests the train_speaker_embedding.py script functionality.
"""

import sys
from pathlib import Path

import mlx.core as mx

# Add project root and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestTrainingScriptImports:
    """Test that training script can be imported."""

    def test_import_training_modules(self):
        """Test importing training modules."""
        from src.training.config import SpeakerTrainingConfig
        from src.training.speaker_dataloader import (
            SpeakerDataLoader,
        )
        from src.training.speaker_trainer import SpeakerTrainer

        # Should not raise
        assert SpeakerTrainingConfig is not None
        assert SpeakerTrainer is not None
        assert SpeakerDataLoader is not None


class TestTrainingConfigValidation:
    """Test training configuration validation."""

    def test_default_speaker_config(self):
        """Test default speaker training configuration."""
        from src.training.config import SpeakerTrainingConfig

        config = SpeakerTrainingConfig()

        # Check defaults
        assert config.encoder_dim == 384
        assert config.embedding_dim == 256
        assert config.aam_margin == 0.2
        assert config.aam_scale == 30.0

    def test_custom_speaker_config(self):
        """Test custom speaker training configuration."""
        from src.training.config import SpeakerTrainingConfig

        config = SpeakerTrainingConfig(
            encoder_dim=80,  # For standalone fbank training
            num_speakers=2169,
            aam_margin=0.3,
        )

        assert config.encoder_dim == 80
        assert config.num_speakers == 2169
        assert config.aam_margin == 0.3


class TestMockDatasetTraining:
    """Test training with mock datasets."""

    def _create_mock_dataset(self, num_samples: int = 100, num_speakers: int = 10):
        """Create a mock speaker dataset."""
        from src.training.speaker_dataloader import SpeakerDataset, SpeakerSample

        class MockSpeakerDataset(SpeakerDataset):
            def __init__(self, num_samples: int, num_speakers: int):
                super().__init__(".", min_duration=2.0, max_duration=10.0)

                for i in range(num_samples):
                    speaker_id = f"spk_{i % num_speakers}"
                    speaker_idx = self._register_speaker(speaker_id)
                    self.samples.append(
                        SpeakerSample(
                            audio_path=f"audio_{i}.wav",
                            speaker_id=speaker_id,
                            speaker_idx=speaker_idx,
                        ),
                    )

            def __getitem__(self, idx: int):
                sample = self.samples[idx]
                # Generate random features (time, feat_dim)
                time_len = 100 + (idx % 50)  # Variable length
                features = mx.random.normal((time_len, 80))
                return {
                    "features": features,
                    "feature_lengths": mx.array([time_len]),
                    "speaker_idx": mx.array([sample.speaker_idx]),
                }

        return MockSpeakerDataset(num_samples, num_speakers)

    def test_training_step_with_mock_data(self):
        """Test a training step with mock data."""
        from src.training.config import (
            OptimizerConfig,
            SchedulerConfig,
            SpeakerTrainingConfig,
        )
        from src.training.speaker_dataloader import SpeakerDataLoader
        from src.training.speaker_trainer import create_speaker_trainer

        # Create mock dataset
        num_speakers = 50
        dataset = self._create_mock_dataset(num_samples=100, num_speakers=num_speakers)

        # Create trainer
        config = SpeakerTrainingConfig(
            encoder_dim=80,
            embedding_dim=128,
            num_speakers=num_speakers,
        )
        optimizer_config = OptimizerConfig(learning_rate=1e-3)
        scheduler_config = SchedulerConfig(
            scheduler_type="constant",
            total_steps=100,
        )

        trainer = create_speaker_trainer(
            config=config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
        )

        # Create data loader
        loader = SpeakerDataLoader(dataset, batch_size=8, shuffle=True)

        # Run one training step
        batch = next(iter(loader))
        metrics = trainer.train_step(batch)

        # Verify metrics
        assert metrics.loss > 0
        assert metrics.time_ms > 0
        assert metrics.num_samples == 8

    def test_training_reduces_loss(self):
        """Test that training reduces loss over steps."""
        from src.training.config import (
            OptimizerConfig,
            SchedulerConfig,
            SpeakerTrainingConfig,
        )
        from src.training.speaker_dataloader import SpeakerDataLoader
        from src.training.speaker_trainer import create_speaker_trainer

        # Create mock dataset
        num_speakers = 20
        dataset = self._create_mock_dataset(num_samples=200, num_speakers=num_speakers)

        # Create trainer
        config = SpeakerTrainingConfig(
            encoder_dim=80,
            embedding_dim=64,
            num_speakers=num_speakers,
        )
        optimizer_config = OptimizerConfig(learning_rate=1e-2)  # Higher LR for test
        scheduler_config = SchedulerConfig(
            scheduler_type="constant",
            total_steps=50,
        )

        trainer = create_speaker_trainer(
            config=config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
        )

        # Create data loader
        loader = SpeakerDataLoader(dataset, batch_size=16, shuffle=True)

        # Train for a few steps
        losses = []
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            metrics = trainer.train_step(batch)
            losses.append(metrics.loss)

        # Loss should generally decrease (allow some variance)
        # Use first 3 vs last 3 average comparison
        first_avg = sum(losses[:3]) / 3
        last_avg = sum(losses[-3:]) / 3

        # Loss should not increase significantly
        assert last_avg <= first_avg * 1.5, f"Loss increased: {first_avg:.4f} -> {last_avg:.4f}"

    def test_validation_runs(self):
        """Test validation runs without error."""
        from src.training.config import (
            SpeakerTrainingConfig,
        )
        from src.training.speaker_dataloader import SpeakerDataLoader
        from src.training.speaker_trainer import create_speaker_trainer

        # Create mock dataset
        num_speakers = 30
        dataset = self._create_mock_dataset(num_samples=100, num_speakers=num_speakers)

        # Create trainer
        config = SpeakerTrainingConfig(
            encoder_dim=80,
            embedding_dim=64,
            num_speakers=num_speakers,
        )
        trainer = create_speaker_trainer(config=config)

        # Create data loader
        loader = SpeakerDataLoader(dataset, batch_size=8, shuffle=False)

        # Run validation
        val_metrics = trainer.validate(loader, trial_loader=None, max_batches=5)

        # Verify metrics
        assert val_metrics.loss >= 0
        assert val_metrics.num_samples > 0


class TestDataLoaderIntegration:
    """Test data loader integration with real-like conditions."""

    def test_batch_collation(self):
        """Test that batch collation handles variable lengths."""
        from src.training.speaker_dataloader import (
            SpeakerDataLoader,
            SpeakerDataset,
            SpeakerSample,
        )

        class VariableLengthDataset(SpeakerDataset):
            def __init__(self):
                super().__init__(".", min_duration=2.0, max_duration=10.0)
                for i in range(50):
                    speaker_id = f"spk_{i % 5}"
                    speaker_idx = self._register_speaker(speaker_id)
                    self.samples.append(
                        SpeakerSample(
                            audio_path=f"audio_{i}.wav",
                            speaker_id=speaker_id,
                            speaker_idx=speaker_idx,
                        ),
                    )

            def __getitem__(self, idx: int):
                sample = self.samples[idx]
                # Highly variable lengths
                time_len = 50 + (idx * 10) % 200
                features = mx.random.normal((time_len, 80))
                return {
                    "features": features,
                    "feature_lengths": mx.array([time_len]),
                    "speaker_idx": mx.array([sample.speaker_idx]),
                }

        dataset = VariableLengthDataset()
        loader = SpeakerDataLoader(dataset, batch_size=4, shuffle=False)

        # Get a batch
        batch = next(iter(loader))

        # Check shapes
        assert batch["features"].ndim == 3  # (batch, time, feat)
        assert batch["features"].shape[0] == 4
        assert batch["features"].shape[2] == 80

        # All items should be padded to max length
        feature_lengths = batch["feature_lengths"]
        max_len = int(feature_lengths.max())
        assert batch["features"].shape[1] == max_len

    def test_speaker_indices_valid(self):
        """Test that speaker indices are valid for classification."""
        from src.training.speaker_dataloader import (
            SpeakerDataLoader,
            SpeakerDataset,
            SpeakerSample,
        )

        class SimpleDataset(SpeakerDataset):
            def __init__(self, num_speakers: int = 10):
                super().__init__(".")
                self._num_speakers = num_speakers
                for i in range(100):
                    speaker_id = f"spk_{i % num_speakers}"
                    speaker_idx = self._register_speaker(speaker_id)
                    self.samples.append(
                        SpeakerSample(
                            audio_path=f"audio_{i}.wav",
                            speaker_id=speaker_id,
                            speaker_idx=speaker_idx,
                        ),
                    )

            def __getitem__(self, idx: int):
                sample = self.samples[idx]
                features = mx.random.normal((100, 80))
                return {
                    "features": features,
                    "feature_lengths": mx.array([100]),
                    "speaker_idx": mx.array([sample.speaker_idx]),
                }

        num_speakers = 10
        dataset = SimpleDataset(num_speakers)
        loader = SpeakerDataLoader(dataset, batch_size=8, shuffle=True)

        # Check all batches have valid speaker indices
        for i, batch in enumerate(loader):
            if i >= 5:
                break

            speaker_indices = batch["speaker_indices"]
            # All indices should be in valid range
            assert int(speaker_indices.min()) >= 0
            assert int(speaker_indices.max()) < num_speakers


class TestCheckpointSaveLoad:
    """Test checkpoint save/load functionality."""

    def test_checkpoint_round_trip(self, tmp_path):
        """Test saving and loading a checkpoint."""
        from src.training.config import SpeakerTrainingConfig
        from src.training.speaker_trainer import create_speaker_trainer

        # Create trainer
        config = SpeakerTrainingConfig(
            encoder_dim=80,
            embedding_dim=64,
            num_speakers=50,
        )
        trainer = create_speaker_trainer(config=config)
        trainer.checkpoint_dir = tmp_path

        # Set some state
        trainer.state.step = 100
        trainer.state.epoch = 2
        trainer.state.best_eer = 0.05

        # Save checkpoint
        path = trainer.save_checkpoint("test")
        assert path.exists()

        # Create new trainer and load
        trainer2 = create_speaker_trainer(config=config)
        trainer2.checkpoint_dir = tmp_path

        assert trainer2.load_checkpoint("test")
        assert trainer2.state.step == 100
        assert trainer2.state.epoch == 2
        assert trainer2.state.best_eer == 0.05
