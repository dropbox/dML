# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for Trainer class."""

import tempfile

import mlx.core as mx
import mlx.nn as nn

from src.training.config import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.training.trainer import Trainer, TrainingState


class DummyModel(nn.Module):
    """Simple model for testing trainer."""

    def __init__(self, vocab_size: int = 100, encoder_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim

        # Simple encoder
        self.encoder = DummyEncoder(encoder_dim)

        # Simple decoder
        self.decoder = DummyDecoder(vocab_size, encoder_dim)

        # Simple joiner
        self.joiner = DummyJoiner(encoder_dim, vocab_size)

    def __call__(self, x, x_lens=None, y=None):
        encoder_out = self.encoder(x)
        encoder_out_lens = x_lens if x_lens is not None else mx.array([x.shape[1]])
        return encoder_out, encoder_out_lens


class DummyEncoder(nn.Module):
    """Dummy encoder for testing."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(80, dim)  # 80 mel features

    def __call__(self, x, x_lens=None):
        out = self.proj(x)
        out_lens = x_lens if x_lens is not None else mx.array([x.shape[1]])
        return out, out_lens


class DummyDecoder(nn.Module):
    """Dummy decoder for testing."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

    def __call__(self, y):
        return self.embedding(y)


class DummyJoiner(nn.Module):
    """Dummy joiner for testing."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.output_linear = nn.Linear(dim, vocab_size)

    def __call__(self, encoder_out, decoder_out):
        # Broadcast and combine (T=encoder time, U=decoder time)
        enc = mx.expand_dims(encoder_out, axis=2)  # (B, T, 1, D)
        dec = mx.expand_dims(decoder_out, axis=1)  # (B, 1, U, D)
        combined = enc + dec  # (B, T, U, D)
        return self.output_linear(combined)

    def forward_lattice(self, enc, dec):
        # For loss computation
        combined = enc + dec
        return self.output_linear(combined)


def create_dummy_batch(batch_size: int = 2, time: int = 10, target_len: int = 3):
    """Create dummy batch for testing."""
    return {
        "features": mx.random.normal((batch_size, time, 80)),
        "feature_lengths": mx.array([time] * batch_size),
        "targets": mx.random.randint(1, 50, (batch_size, target_len)),
        "target_lengths": mx.array([target_len] * batch_size),
    }


def create_dummy_dataloader(num_batches: int = 10, batch_size: int = 2):
    """Create dummy data iterator."""
    for _ in range(num_batches):
        yield create_dummy_batch(batch_size=batch_size)


class TestTrainingState:
    """Tests for TrainingState."""

    def test_default_values(self):
        """Test default state values."""
        state = TrainingState()

        assert state.step == 0
        assert state.epoch == 0
        assert state.best_loss == float("inf")
        assert len(state.train_losses) == 0  # May be list or deque

    def test_to_dict(self):
        """Test state serialization."""
        state = TrainingState(
            step=100,
            epoch=2,
            best_loss=0.5,
        )
        state.train_losses = [1.0, 0.8, 0.6]

        d = state.to_dict()

        assert d["step"] == 100
        assert d["epoch"] == 2
        assert d["best_loss"] == 0.5
        assert len(d["train_losses"]) == 3

    def test_from_dict(self):
        """Test state deserialization."""
        d = {
            "step": 100,
            "epoch": 2,
            "best_loss": 0.5,
            "best_val_wer": float("inf"),
            "train_losses": [1.0, 0.8],
            "val_losses": [0.9],
            "learning_rates": [1e-4, 1e-4],
            "total_train_time": 100.0,
            "samples_processed": 1000,
        }

        state = TrainingState.from_dict(d)

        assert state.step == 100
        assert state.epoch == 2
        assert state.best_loss == 0.5


class TestTrainer:
    """Tests for Trainer class."""

    def test_initialization(self):
        """Test trainer initialization."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",  # Use CTC for simpler testing
            batch_size=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir

            trainer = Trainer(model, config)

            assert trainer.model is model
            assert trainer.config is config
            assert trainer.state.step == 0

    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = DummyModel()

        # AdamW optimizer
        config_adam = TrainingConfig(
            loss_type="ctc",
            optimizer=OptimizerConfig(optimizer_type="adamw"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_adam.checkpoint_dir = tmpdir
            trainer_adam = Trainer(model, config_adam)
            assert trainer_adam.optimizer is not None

        # SGD optimizer
        config_sgd = TrainingConfig(
            loss_type="ctc",
            optimizer=OptimizerConfig(optimizer_type="sgd"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_sgd.checkpoint_dir = tmpdir
            trainer_sgd = Trainer(model, config_sgd)
            assert trainer_sgd.optimizer is not None

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",
            optimizer=OptimizerConfig(grad_clip=1.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            # Create large gradients
            grads = {"layer": mx.ones((100,)) * 100}

            clipped, norm = trainer._clip_gradients(grads, max_norm=1.0)

            # Norm should have been large
            assert norm > 1.0

            # Clipped gradients should have smaller norm
            clipped_flat = mx.reshape(clipped["layer"], (-1,))
            clipped_norm = mx.sqrt(mx.sum(clipped_flat * clipped_flat)).item()
            assert clipped_norm <= 1.1  # Small tolerance

    def test_check_loss_valid(self):
        """Test loss validity checking."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",
            max_loss=100.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            # Valid loss
            assert trainer._check_loss_valid(mx.array(1.0))

            # NaN loss
            assert not trainer._check_loss_valid(mx.array(float("nan")))

            # Too large loss
            assert not trainer._check_loss_valid(mx.array(200.0))


class TestTrainerCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        model = DummyModel()
        config = TrainingConfig(loss_type="ctc")

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            # Modify state
            trainer.state.step = 100

            # Save checkpoint
            path = trainer.save_checkpoint("test_ckpt")

            assert path.exists()
            assert (path / "model.safetensors").exists()
            assert (path / "training_state.json").exists()
            assert (path / "config.json").exists()

    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        model = DummyModel()
        config = TrainingConfig(loss_type="ctc")

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir

            # Save checkpoint
            trainer1 = Trainer(model, config)
            trainer1.state.step = 100
            trainer1.state.best_loss = 0.5
            path = trainer1.save_checkpoint("test_ckpt")

            # Load checkpoint
            model2 = DummyModel()
            trainer2 = Trainer(model2, config)
            trainer2.load_checkpoint(str(path))

            assert trainer2.state.step == 100
            assert trainer2.state.best_loss == 0.5


class TestTrainerStability:
    """Tests for training stability features."""

    def test_nan_loss_skipping(self):
        """Test that NaN losses are skipped when configured."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",
            skip_nan_loss=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            # Invalid loss should not raise
            valid = trainer._check_loss_valid(mx.array(float("nan")))
            assert not valid

    def test_max_loss_threshold(self):
        """Test max loss threshold."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",
            max_loss=50.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            # Loss below threshold
            assert trainer._check_loss_valid(mx.array(10.0))

            # Loss above threshold
            assert not trainer._check_loss_valid(mx.array(100.0))


class TestSchedulerIntegration:
    """Tests for scheduler integration with trainer."""

    def test_scheduler_initialization(self):
        """Test scheduler is initialized from config."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",
            scheduler=SchedulerConfig(
                warmup_steps=1000,
                total_steps=10000,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            assert trainer.scheduler.warmup_steps == 1000
            assert trainer.scheduler.total_steps == 10000

    def test_lr_update_during_training(self):
        """Test learning rate updates during training."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",
            scheduler=SchedulerConfig(
                warmup_steps=10,
                total_steps=100,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            # Get initial LR
            initial_lr = trainer.scheduler.current_lr

            # Simulate some steps
            for _ in range(5):
                trainer.scheduler.step()

            # LR should have increased during warmup
            assert trainer.scheduler.current_lr > initial_lr
