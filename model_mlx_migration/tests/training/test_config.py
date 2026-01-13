# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for training configuration."""

import tempfile
from pathlib import Path

import pytest

from src.training.config import (
    STREAMING_LARGE_CONFIG,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    get_config,
)


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_values(self):
        """Test default optimizer config values."""
        cfg = OptimizerConfig()

        assert cfg.optimizer_type == "adamw"
        assert cfg.learning_rate == 1e-4
        assert cfg.weight_decay == 0.01
        assert cfg.beta1 == 0.9
        assert cfg.beta2 == 0.98
        assert cfg.grad_clip == 1.0
        assert cfg.grad_accumulation_steps == 1

    def test_custom_values(self):
        """Test custom optimizer config."""
        cfg = OptimizerConfig(
            optimizer_type="sgd",
            learning_rate=0.01,
            grad_clip=0.5,
        )

        assert cfg.optimizer_type == "sgd"
        assert cfg.learning_rate == 0.01
        assert cfg.grad_clip == 0.5


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_values(self):
        """Test default scheduler config values."""
        cfg = SchedulerConfig()

        assert cfg.scheduler_type == "warmup_cosine"
        assert cfg.warmup_steps == 10000
        assert cfg.total_steps == 500000
        assert cfg.min_lr == 1e-6
        assert cfg.cooldown_steps == 50000

    def test_custom_values(self):
        """Test custom scheduler config."""
        cfg = SchedulerConfig(
            scheduler_type="warmup_linear",
            warmup_steps=5000,
            total_steps=100000,
        )

        assert cfg.scheduler_type == "warmup_linear"
        assert cfg.warmup_steps == 5000
        assert cfg.total_steps == 100000


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training config values."""
        cfg = TrainingConfig()

        assert cfg.model_variant == "streaming-large"
        assert cfg.vocab_size == 51865
        assert cfg.blank_id == 0
        assert cfg.chunk_sizes == (32, 64)
        assert cfg.causal is True
        assert cfg.loss_type == "cr_ctc"
        assert cfg.ctc_weight == 0.3
        assert cfg.batch_size == 8
        assert cfg.num_epochs == 100

    def test_optimizer_and_scheduler(self):
        """Test nested optimizer and scheduler configs."""
        cfg = TrainingConfig()

        assert isinstance(cfg.optimizer, OptimizerConfig)
        assert isinstance(cfg.scheduler, SchedulerConfig)
        assert cfg.optimizer.learning_rate == 1e-4
        assert cfg.scheduler.warmup_steps == 10000

    def test_to_dict(self):
        """Test config serialization to dict."""
        cfg = TrainingConfig(
            model_variant="test",
            batch_size=4,
        )

        d = cfg.to_dict()

        assert d["model_variant"] == "test"
        assert d["batch_size"] == 4
        assert "optimizer" in d
        assert "scheduler" in d
        assert d["optimizer"]["learning_rate"] == 1e-4

    def test_from_dict(self):
        """Test config deserialization from dict."""
        d = {
            "model_variant": "test",
            "batch_size": 4,
            "optimizer": {"learning_rate": 0.001},
            "scheduler": {"warmup_steps": 1000},
        }

        cfg = TrainingConfig.from_dict(d)

        assert cfg.model_variant == "test"
        assert cfg.batch_size == 4
        assert cfg.optimizer.learning_rate == 0.001
        assert cfg.scheduler.warmup_steps == 1000

    def test_save_and_load(self):
        """Test config save and load."""
        cfg = TrainingConfig(
            model_variant="test-save",
            batch_size=16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            cfg.save(str(path))

            assert path.exists()

            loaded = TrainingConfig.load(str(path))

            assert loaded.model_variant == "test-save"
            assert loaded.batch_size == 16


class TestGetConfig:
    """Tests for get_config factory function."""

    def test_streaming_large(self):
        """Test getting streaming-large config."""
        cfg = get_config("streaming-large")

        assert cfg.model_variant == "streaming-large"
        assert cfg.optimizer.learning_rate == 1e-4
        assert cfg.optimizer.grad_clip == 1.0
        assert cfg.scheduler.warmup_steps == 10000
        assert cfg.gradient_checkpointing is True

    def test_streaming_small(self):
        """Test getting streaming-small config."""
        cfg = get_config("streaming-small")

        assert cfg.model_variant == "streaming-small"
        assert cfg.optimizer.learning_rate == 4e-3
        assert cfg.batch_size == 16

    def test_streaming_medium(self):
        """Test getting streaming-medium config."""
        cfg = get_config("streaming-medium")

        assert cfg.model_variant == "streaming-medium"
        assert cfg.optimizer.learning_rate == 2e-3
        assert cfg.batch_size == 12

    def test_unknown_variant(self):
        """Test unknown variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            get_config("nonexistent")


class TestStabilityConfig:
    """Test stability-related config settings."""

    def test_streaming_large_stability(self):
        """Verify streaming-large has stability mitigations."""
        cfg = STREAMING_LARGE_CONFIG

        # Lower learning rate for stability
        assert cfg.optimizer.learning_rate <= 1e-4

        # Gradient clipping enabled
        assert cfg.optimizer.grad_clip > 0

        # Longer warmup
        assert cfg.scheduler.warmup_steps >= 10000

        # Gradient checkpointing for memory
        assert cfg.gradient_checkpointing is True

    def test_loss_stability_settings(self):
        """Test loss-related stability settings."""
        cfg = TrainingConfig()

        assert cfg.skip_nan_loss is True
        assert cfg.max_loss > 0
        assert cfg.max_loss < 1000  # Reasonable upper bound
