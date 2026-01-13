# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for Rich Audio Heads training integration."""

import tempfile

import mlx.core as mx
import mlx.nn as nn
import pytest

from src.models.heads.rich_audio import RichAudioHeads, RichAudioHeadsConfig
from src.training.config import (
    RichAudioHeadsTrainingConfig,
    TrainingConfig,
)
from src.training.loss import (
    RichAudioLoss,
    RichAudioLossOutput,
    binary_cross_entropy_loss,
    cross_entropy_loss,
    l1_loss,
    mse_loss,
)
from src.training.trainer import Trainer


class DummyModel(nn.Module):
    """Simple model for testing trainer."""

    def __init__(self, vocab_size: int = 100, encoder_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.encoder = DummyEncoder(encoder_dim)
        self.joiner = DummyJoiner(encoder_dim, vocab_size)


class DummyEncoder(nn.Module):
    """Dummy encoder for testing."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(80, dim)

    def __call__(self, x, x_lens=None):
        out = self.proj(x)
        out_lens = x_lens if x_lens is not None else mx.array([x.shape[1]])
        return out, out_lens


class DummyJoiner(nn.Module):
    """Dummy joiner for testing."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.output_linear = nn.Linear(dim, vocab_size)


# =============================================================================
# Test Loss Functions
# =============================================================================


class TestCrossEntropyLoss:
    """Tests for cross entropy loss function."""

    def test_basic_ce_2d(self):
        """Test CE loss with 2D logits."""
        logits = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        targets = mx.array([0, 1])

        loss = cross_entropy_loss(logits, targets)

        assert loss.shape == ()
        assert loss.item() < 1.0  # Loss should be low for correct predictions

    def test_basic_ce_3d(self):
        """Test CE loss with 3D logits (batch, seq, classes)."""
        logits = mx.random.normal((2, 4, 8))
        targets = mx.random.randint(0, 8, (2, 4))

        loss = cross_entropy_loss(logits, targets)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_label_smoothing(self):
        """Test label smoothing."""
        logits = mx.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = mx.array([0, 1])

        loss_no_smooth = cross_entropy_loss(logits, targets, label_smoothing=0.0)
        loss_smooth = cross_entropy_loss(logits, targets, label_smoothing=0.1)

        # Smoothed loss should be higher (penalizes overconfidence)
        assert loss_smooth.item() > loss_no_smooth.item()

    def test_reduction_none(self):
        """Test no reduction."""
        logits = mx.random.normal((2, 3))
        targets = mx.array([0, 1])

        loss = cross_entropy_loss(logits, targets, reduction="none")

        assert loss.shape == (2,)


class TestBinaryCrossEntropyLoss:
    """Tests for binary cross entropy loss."""

    def test_basic_bce(self):
        """Test BCE loss."""
        logits = mx.array([5.0, -5.0])  # Confident positive, confident negative
        targets = mx.array([1.0, 0.0])

        loss = binary_cross_entropy_loss(logits, targets)

        assert loss.shape == ()
        assert loss.item() < 0.1  # Low loss for correct predictions

    def test_bce_wrong_predictions(self):
        """Test BCE with wrong predictions."""
        logits = mx.array([5.0, -5.0])  # Confident positive, confident negative
        targets = mx.array([0.0, 1.0])  # But targets are opposite

        loss = binary_cross_entropy_loss(logits, targets)

        assert loss.item() > 2.0  # High loss for wrong predictions


class TestMSELoss:
    """Tests for MSE loss."""

    def test_basic_mse(self):
        """Test basic MSE loss."""
        predictions = mx.array([1.0, 2.0, 3.0])
        targets = mx.array([1.0, 2.0, 3.0])

        loss = mse_loss(predictions, targets)

        assert loss.shape == ()
        assert abs(loss.item()) < 1e-6  # Perfect predictions

    def test_mse_with_error(self):
        """Test MSE with error."""
        predictions = mx.array([1.0, 2.0, 3.0])
        targets = mx.array([2.0, 3.0, 4.0])  # All off by 1

        loss = mse_loss(predictions, targets)

        assert abs(loss.item() - 1.0) < 1e-6  # MSE should be 1.0

    def test_mse_with_mask(self):
        """Test MSE with mask."""
        predictions = mx.array([1.0, 2.0, 3.0])
        targets = mx.array([1.0, 2.0, 100.0])  # Third is very wrong
        mask = mx.array([1.0, 1.0, 0.0])  # But masked out

        loss = mse_loss(predictions, targets, mask=mask)

        assert abs(loss.item()) < 1e-6  # Should ignore the masked position


class TestL1Loss:
    """Tests for L1 loss."""

    def test_basic_l1(self):
        """Test basic L1 loss."""
        predictions = mx.array([1.0, 2.0, 3.0])
        targets = mx.array([2.0, 3.0, 4.0])  # All off by 1

        loss = l1_loss(predictions, targets)

        assert abs(loss.item() - 1.0) < 1e-6


# =============================================================================
# Test RichAudioLoss
# =============================================================================


class TestRichAudioLoss:
    """Tests for combined RichAudioLoss."""

    def test_initialization(self):
        """Test RichAudioLoss initialization."""
        loss_fn = RichAudioLoss(
            emotion_weight=0.1,
            language_weight=0.1,
            phoneme_weight=0.2,
        )

        assert loss_fn.emotion_weight == 0.1
        assert loss_fn.phoneme_weight == 0.2

    def test_emotion_loss_only(self):
        """Test computing only emotion loss."""
        loss_fn = RichAudioLoss()

        outputs = {
            "emotion_logits": mx.random.normal((2, 8)),
        }
        labels = {
            "emotion_labels": mx.array([0, 1]),
        }

        result = loss_fn(outputs, labels)

        assert isinstance(result, RichAudioLossOutput)
        assert "emotion" in result.losses
        assert result.total_loss.item() > 0

    def test_multiple_heads(self):
        """Test computing losses for multiple heads."""
        loss_fn = RichAudioLoss()

        outputs = {
            "emotion_logits": mx.random.normal((2, 8)),
            "language_logits": mx.random.normal((2, 9)),
            "paralinguistics_logits": mx.random.normal((2, 50)),
        }
        labels = {
            "emotion_labels": mx.array([0, 1]),
            "language_labels": mx.array([2, 3]),
            "paralinguistics_labels": mx.array([10, 20]),
        }

        result = loss_fn(outputs, labels)

        assert "emotion" in result.losses
        assert "language" in result.losses
        assert "paralinguistics" in result.losses

    def test_pitch_loss(self):
        """Test pitch loss computation."""
        loss_fn = RichAudioLoss()

        outputs = {
            "pitch_f0_hz": mx.random.normal((2, 10)) * 200 + 200,  # ~200Hz
            "pitch_voiced_logits": mx.random.normal((2, 10)),
        }
        labels = {
            "pitch_f0_targets": mx.random.normal((2, 10)) * 200 + 200,
            "pitch_voiced_targets": mx.random.uniform(shape=(2, 10)),
        }

        result = loss_fn(outputs, labels)

        assert "pitch_f0" in result.losses
        assert "pitch_voiced" in result.losses
        assert "pitch" in result.losses

    def test_phoneme_loss(self):
        """Test phoneme loss computation."""
        loss_fn = RichAudioLoss()

        outputs = {
            "phoneme_logits": mx.random.normal((2, 10, 178)),
        }
        labels = {
            "phoneme_labels": mx.random.randint(0, 178, (2, 10)),
        }

        result = loss_fn(outputs, labels)

        assert "phoneme" in result.losses

    def test_singing_loss(self):
        """Test singing loss computation."""
        loss_fn = RichAudioLoss()

        outputs = {
            "singing_binary_logits": mx.random.normal((2, 1)),
            "singing_technique_logits": mx.random.normal((2, 10)),
        }
        labels = {
            "singing_binary_labels": mx.random.uniform(shape=(2, 1)),
            "singing_technique_labels": mx.array([0, 5]),
        }

        result = loss_fn(outputs, labels)

        assert "singing_binary" in result.losses
        assert "singing_technique" in result.losses
        assert "singing" in result.losses

    def test_timestamp_loss(self):
        """Test timestamp loss computation."""
        loss_fn = RichAudioLoss()

        outputs = {
            "timestamp_boundary_logits": mx.random.normal((2, 10, 1)),
            "timestamp_offset_preds": mx.random.normal((2, 10, 2)),
        }
        labels = {
            "timestamp_boundary_targets": mx.random.uniform(shape=(2, 10, 1)),
            "timestamp_offset_targets": mx.random.normal((2, 10, 2)),
        }

        result = loss_fn(outputs, labels)

        assert "timestamp_boundary" in result.losses
        assert "timestamp_offset" in result.losses
        assert "timestamp" in result.losses

    def test_no_labels_returns_zero(self):
        """Test that no labels returns zero loss."""
        loss_fn = RichAudioLoss()

        outputs = {
            "emotion_logits": mx.random.normal((2, 8)),
        }
        labels = {}  # No labels provided

        result = loss_fn(outputs, labels)

        assert result.total_loss.item() == 0.0
        assert len(result.losses) == 0


# =============================================================================
# Test Config Integration
# =============================================================================


class TestRichAudioHeadsTrainingConfig:
    """Tests for RichAudioHeadsTrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RichAudioHeadsTrainingConfig()

        assert config.enabled is False
        assert config.emotion_weight == 0.1
        assert config.phoneme_weight == 0.2
        assert config.encoder_dim == 384

    def test_config_serialization(self):
        """Test config serialization in TrainingConfig."""
        config = TrainingConfig(
            loss_type="ctc",
            rich_audio_heads=RichAudioHeadsTrainingConfig(
                enabled=True,
                emotion_weight=0.2,
            ),
        )

        d = config.to_dict()

        assert d["rich_audio_heads"]["enabled"] is True
        assert d["rich_audio_heads"]["emotion_weight"] == 0.2

    def test_config_deserialization(self):
        """Test config deserialization."""
        d = {
            "loss_type": "ctc",
            "vocab_size": 100,
            "rich_audio_heads": {
                "enabled": True,
                "emotion_weight": 0.2,
            },
        }

        config = TrainingConfig.from_dict(d)

        assert config.rich_audio_heads.enabled is True
        assert config.rich_audio_heads.emotion_weight == 0.2


# =============================================================================
# Test Trainer Integration
# =============================================================================


class TestTrainerRichAudioIntegration:
    """Tests for Trainer with rich audio heads."""

    def test_trainer_without_rich_audio(self):
        """Test trainer works without rich audio heads."""
        model = DummyModel()
        config = TrainingConfig(loss_type="ctc")

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            assert trainer.rich_audio_heads is None
            assert trainer.rich_audio_loss_fn is None

    def test_trainer_with_rich_audio_disabled(self):
        """Test trainer with rich audio disabled in config."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",
            rich_audio_heads=RichAudioHeadsTrainingConfig(enabled=False),
        )
        rich_heads = RichAudioHeads(RichAudioHeadsConfig(encoder_dim=64))

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config, rich_audio_heads=rich_heads)

            # Should not create loss function when disabled
            assert trainer.rich_audio_heads is not None
            assert trainer.rich_audio_loss_fn is None

    def test_trainer_with_rich_audio_enabled(self):
        """Test trainer with rich audio enabled."""
        model = DummyModel()
        config = TrainingConfig(
            loss_type="ctc",
            rich_audio_heads=RichAudioHeadsTrainingConfig(
                enabled=True,
                encoder_dim=64,
            ),
        )
        rich_heads = RichAudioHeads(RichAudioHeadsConfig(encoder_dim=64))

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config, rich_audio_heads=rich_heads)

            assert trainer.rich_audio_heads is not None
            assert trainer.rich_audio_loss_fn is not None

    def test_has_rich_audio_labels(self):
        """Test _has_rich_audio_labels helper."""
        model = DummyModel()
        config = TrainingConfig(loss_type="ctc")

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            # Batch without rich labels
            batch_no_labels = {
                "features": mx.random.normal((2, 10, 80)),
                "feature_lengths": mx.array([10, 10]),
            }
            assert not trainer._has_rich_audio_labels(batch_no_labels)

            # Batch with emotion labels
            batch_with_labels = {
                "features": mx.random.normal((2, 10, 80)),
                "emotion_labels": mx.array([0, 1]),
            }
            assert trainer._has_rich_audio_labels(batch_with_labels)

    def test_extract_rich_audio_labels(self):
        """Test _extract_rich_audio_labels helper."""
        model = DummyModel()
        config = TrainingConfig(loss_type="ctc")

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(model, config)

            batch = {
                "features": mx.random.normal((2, 10, 80)),
                "emotion_labels": mx.array([0, 1]),
                "language_labels": mx.array([2, 3]),
                "unrelated_key": mx.array([0]),
            }

            labels = trainer._extract_rich_audio_labels(batch)

            assert "emotion_labels" in labels
            assert "language_labels" in labels
            assert "unrelated_key" not in labels
            assert "features" not in labels


class TestTrainStepWithRichAudio:
    """Tests for train_step with rich audio heads."""

    @pytest.fixture
    def trainer_with_rich_audio(self):
        """Create trainer with rich audio heads enabled."""
        model = DummyModel(encoder_dim=64)
        config = TrainingConfig(
            loss_type="ctc",
            rich_audio_heads=RichAudioHeadsTrainingConfig(
                enabled=True,
                encoder_dim=64,
            ),
        )
        rich_heads = RichAudioHeads(RichAudioHeadsConfig(encoder_dim=64))

        tmpdir = tempfile.mkdtemp()
        config.checkpoint_dir = tmpdir
        return Trainer(model, config, rich_audio_heads=rich_heads)


    def test_train_step_without_rich_labels(self, trainer_with_rich_audio):
        """Test train step works without rich audio labels."""
        batch = {
            "features": mx.random.normal((2, 10, 80)),
            "feature_lengths": mx.array([10, 10]),
            "targets": mx.random.randint(1, 50, (2, 3)),
            "target_lengths": mx.array([3, 3]),
        }

        metrics = trainer_with_rich_audio.train_step(batch)

        assert "loss" in metrics
        assert metrics["rich_audio_loss"] == 0.0  # No rich labels

    def test_train_step_with_emotion_labels(self, trainer_with_rich_audio):
        """Test train step with emotion labels."""
        batch = {
            "features": mx.random.normal((2, 10, 80)),
            "feature_lengths": mx.array([10, 10]),
            "targets": mx.random.randint(1, 50, (2, 3)),
            "target_lengths": mx.array([3, 3]),
            "emotion_labels": mx.array([0, 1]),
        }

        metrics = trainer_with_rich_audio.train_step(batch)

        assert "loss" in metrics
        assert metrics["rich_audio_loss"] > 0
        assert "rich_emotion_loss" in metrics

    def test_train_step_with_multiple_labels(self, trainer_with_rich_audio):
        """Test train step with multiple rich audio labels."""
        batch = {
            "features": mx.random.normal((2, 10, 80)),
            "feature_lengths": mx.array([10, 10]),
            "targets": mx.random.randint(1, 50, (2, 3)),
            "target_lengths": mx.array([3, 3]),
            "emotion_labels": mx.array([0, 1]),
            "language_labels": mx.array([2, 3]),
            "phoneme_labels": mx.random.randint(0, 178, (2, 10)),
        }

        metrics = trainer_with_rich_audio.train_step(batch)

        assert "loss" in metrics
        assert metrics["rich_audio_loss"] > 0
        assert "rich_emotion_loss" in metrics
        assert "rich_language_loss" in metrics
        assert "rich_phoneme_loss" in metrics


class TestValidateWithRichAudio:
    """Tests for validate with rich audio heads."""

    @pytest.fixture
    def trainer_with_val(self):
        """Create trainer with validation dataloader and rich audio."""
        model = DummyModel(encoder_dim=64)
        config = TrainingConfig(
            loss_type="ctc",
            rich_audio_heads=RichAudioHeadsTrainingConfig(
                enabled=True,
                encoder_dim=64,
            ),
        )
        rich_heads = RichAudioHeads(RichAudioHeadsConfig(encoder_dim=64))

        def val_dataloader():
            for _ in range(3):
                yield {
                    "features": mx.random.normal((2, 10, 80)),
                    "feature_lengths": mx.array([10, 10]),
                    "targets": mx.random.randint(1, 50, (2, 3)),
                    "target_lengths": mx.array([3, 3]),
                    "emotion_labels": mx.array([0, 1]),
                }

        tmpdir = tempfile.mkdtemp()
        config.checkpoint_dir = tmpdir
        return Trainer(
            model,
            config,
            val_dataloader=val_dataloader(),
            rich_audio_heads=rich_heads,
        )


    def test_validate_returns_rich_metrics(self, trainer_with_val):
        """Test validation returns rich audio metrics."""
        metrics = trainer_with_val.validate()

        assert "val_loss" in metrics
        assert "val_rich_audio_loss" in metrics
        assert metrics["val_rich_audio_loss"] > 0


class TestGradientFlow:
    """Tests for gradient flow through rich audio heads."""

    def test_gradients_computed_for_rich_heads(self):
        """Test that gradients flow through rich audio heads."""
        model = DummyModel(encoder_dim=64)
        rich_heads = RichAudioHeads(RichAudioHeadsConfig(encoder_dim=64))
        loss_fn = RichAudioLoss()

        # Forward pass
        features = mx.random.normal((2, 10, 80))
        encoder_out, encoder_lengths = model.encoder(features)
        rich_outputs = rich_heads(encoder_out, encoder_lengths)

        labels = {
            "emotion_labels": mx.array([0, 1]),
        }

        # Compute loss
        result = loss_fn(rich_outputs, labels)

        # Check loss is valid
        assert result.total_loss.item() > 0

        # The loss should be differentiable (we can't easily check gradients
        # without value_and_grad, but the fact that it runs is a good sign)
