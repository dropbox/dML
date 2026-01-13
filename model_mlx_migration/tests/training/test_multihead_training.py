"""
Tests for multihead Zipformer training infrastructure.

These tests verify:
1. Training config creation
2. Rich audio heads creation
3. Data loader integration
4. Training step execution
"""

import sys
from pathlib import Path

import mlx.core as mx
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.heads import (
    EmotionConfig,
    EmotionHead,
    LanguageConfig,
    LanguageHead,
    PhonemeConfig,
    PhonemeHead,
    RichAudioHeads,
    RichAudioHeadsConfig,
)
from src.training import (
    OptimizerConfig,
    RichAudioHeadsTrainingConfig,
    RichAudioLoss,
    SchedulerConfig,
    TrainingConfig,
)


class TestRichAudioHeadsCreation:
    """Test creating rich audio heads."""

    def test_create_emotion_head(self):
        """Test creating emotion head."""
        config = EmotionConfig(encoder_dim=384)
        head = EmotionHead(config)

        # Test forward pass
        encoder_out = mx.random.normal((2, 100, 384))
        output = head(encoder_out)

        assert output.shape == (2, 8)  # 8 emotion classes
        mx.eval(output)

    def test_create_phoneme_head(self):
        """Test creating phoneme head."""
        config = PhonemeConfig(encoder_dim=384)
        head = PhonemeHead(config)

        # Test forward pass
        encoder_out = mx.random.normal((2, 100, 384))
        output = head(encoder_out)

        assert output.shape[0] == 2
        # Number of phonemes varies by config
        assert output.shape[2] > 0
        mx.eval(output)

    def test_create_language_head(self):
        """Test creating language head."""
        config = LanguageConfig(encoder_dim=384)
        head = LanguageHead(config)

        # Test forward pass
        encoder_out = mx.random.normal((2, 100, 384))
        output = head(encoder_out)

        assert output.shape[0] == 2
        mx.eval(output)

    def test_create_rich_audio_heads(self):
        """Test creating full RichAudioHeads module."""
        config = RichAudioHeadsConfig(
            encoder_dim=384,
            speaker_enabled=False,  # Disable speaker head
        )
        heads = RichAudioHeads(config)

        # Test forward pass
        encoder_out = mx.random.normal((2, 100, 384))
        outputs = heads(encoder_out)

        # All heads should be present (RichAudioHeads always creates all)
        assert "emotion_logits" in outputs
        assert "phoneme_logits" in outputs
        assert "language_logits" in outputs
        assert "pitch_f0_hz" in outputs

        mx.eval(outputs)


class TestTrainingConfig:
    """Test training configuration."""

    def test_create_training_config(self):
        """Test creating training config."""
        rich_audio_config = RichAudioHeadsTrainingConfig(
            enabled=True,
            emotion_weight=0.1,
            phoneme_weight=0.2,
            language_weight=0.1,
        )

        optimizer_config = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=1e-4,
        )

        scheduler_config = SchedulerConfig(
            scheduler_type="warmup_cosine",
            warmup_steps=1000,
        )

        config = TrainingConfig(
            num_epochs=10,
            batch_size=8,
            loss_type="ctc",
            checkpoint_dir="checkpoints/test",
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            rich_audio_heads=rich_audio_config,
        )

        assert config.num_epochs == 10
        assert config.batch_size == 8
        assert config.rich_audio_heads.enabled
        assert config.rich_audio_heads.emotion_weight == 0.1


class TestRichAudioLoss:
    """Test rich audio loss computation."""

    def test_emotion_loss(self):
        """Test emotion classification loss."""
        loss_fn = RichAudioLoss(
            emotion_weight=1.0,
            language_weight=0.0,
            paralinguistics_weight=0.0,
            pitch_weight=0.0,
            phoneme_weight=0.0,
            singing_weight=0.0,
            timestamp_weight=0.0,
        )

        # Create predictions and targets
        predictions = {
            "emotion_logits": mx.random.normal((2, 8)),
        }
        targets = {
            "emotion_labels": mx.array([0, 3]),  # Class indices
        }

        loss_output = loss_fn(predictions, targets)

        # RichAudioLossOutput has total_loss and losses attributes
        assert loss_output.total_loss.item() > 0
        assert "emotion" in loss_output.losses

    def test_phoneme_loss(self):
        """Test phoneme frame loss."""
        loss_fn = RichAudioLoss(
            emotion_weight=0.0,
            language_weight=0.0,
            paralinguistics_weight=0.0,
            pitch_weight=0.0,
            phoneme_weight=1.0,
            singing_weight=0.0,
            timestamp_weight=0.0,
        )

        # Create predictions and targets
        predictions = {
            "phoneme_logits": mx.random.normal((2, 100, 66)),  # PhonemeHead output
        }
        targets = {
            "phoneme_labels": mx.zeros((2, 100), dtype=mx.int32),
            "phoneme_mask": mx.ones((2, 100)),
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() > 0
        assert "phoneme" in loss_output.losses


class TestIntegration:
    """Integration tests for training pipeline."""

    def test_heads_with_dummy_encoder(self):
        """Test heads work with dummy encoder output."""
        # Create dummy encoder output
        encoder_out = mx.random.normal((2, 100, 384))

        # Create heads
        config = RichAudioHeadsConfig(
            encoder_dim=384,
            speaker_enabled=False,
        )
        heads = RichAudioHeads(config)

        # Forward pass
        outputs = heads(encoder_out)

        # Create loss function
        loss_fn = RichAudioLoss(
            emotion_weight=0.1,
            phoneme_weight=0.2,
            language_weight=0.1,
        )

        # Create targets
        targets = {
            "emotion_labels": mx.array([0, 3]),
            "language_labels": mx.array([0, 1]),
            "phoneme_labels": mx.zeros((2, 100), dtype=mx.int32),
            "phoneme_mask": mx.ones((2, 100)),
        }

        # Compute loss
        loss_output = loss_fn(outputs, targets)

        assert loss_output.total_loss.item() > 0
        mx.eval(loss_output.total_loss)

    def test_training_script_imports(self):
        """Test that training script imports work."""
        # This tests that all imports in the training script are valid
        import importlib.util

        script_path = Path(__file__).parent.parent.parent / "scripts" / "train_zipformer_multihead.py"

        spec = importlib.util.spec_from_file_location("train_multihead", script_path)
        module = importlib.util.module_from_spec(spec)

        # Load module - this will fail if imports are broken
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass  # argparse may call sys.exit

        # Check key functions exist
        assert hasattr(module, "create_rich_audio_heads")
        assert hasattr(module, "create_training_config")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
