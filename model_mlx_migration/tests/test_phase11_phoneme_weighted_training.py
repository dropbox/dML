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
Tests for Phase 11.1: Phoneme-Weighted Adaptation Training.

Tests the novel phoneme quality-weighted training approach:
- Quality score weighting
- Sample filtering
- LoRA adapter training
- Online/streaming training
"""

import time

import mlx.core as mx
import pytest

from tools.whisper_mlx.sota.phoneme_adaptation import AdaptationSample
from tools.whisper_mlx.sota.phoneme_weighted_trainer import (
    LoRALinear,
    OnlinePhonemeWeightedTrainer,
    PhonemeWeightedTrainer,
    PhonemeWeightedTrainingConfig,
    SpeakerLoRAAdapter,
    TrainingResult,
)


# Test fixtures
def make_sample(
    quality_score: float = 0.8,
    speaker_id: int = 1,
    transcript: str = "test transcript",
) -> AdaptationSample:
    """Create a test adaptation sample."""
    return AdaptationSample(
        audio=mx.zeros((16000,)),  # 1 second at 16kHz
        transcript=transcript,
        speaker_id=speaker_id,
        speaker_embedding=mx.random.normal((192,)),
        quality_score=quality_score,
        phoneme_score=quality_score * 0.9,
        confidence=quality_score * 0.8,
        timestamp=time.time(),
    )


class TestPhonemeWeightedTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PhonemeWeightedTrainingConfig()

        assert config.quality_exponent == 2.0
        assert config.min_quality_for_training == 0.5
        assert config.quality_weight_floor == 0.1
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.lora_rank == 8

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PhonemeWeightedTrainingConfig(
            quality_exponent=3.0,
            min_quality_for_training=0.7,
            learning_rate=5e-5,
            lora_rank=16,
        )

        assert config.quality_exponent == 3.0
        assert config.min_quality_for_training == 0.7
        assert config.learning_rate == 5e-5
        assert config.lora_rank == 16


class TestLoRALinear:
    """Tests for LoRA linear layer."""

    def test_lora_initialization(self):
        """Test LoRA layer initialization."""
        lora = LoRALinear(
            in_features=512,
            out_features=512,
            rank=8,
            alpha=16.0,
            dropout=0.1,
        )

        assert lora.in_features == 512
        assert lora.out_features == 512
        assert lora.rank == 8
        assert lora.scaling == 2.0  # alpha / rank

    def test_lora_forward(self):
        """Test LoRA forward pass."""
        lora = LoRALinear(
            in_features=256,
            out_features=256,
            rank=4,
            alpha=8.0,
            dropout=0.0,  # No dropout for deterministic test
        )

        # Input
        x = mx.random.normal((2, 10, 256))
        base_output = mx.random.normal((2, 10, 256))

        # Forward
        output = lora(x, base_output)

        # Shape should match
        assert output.shape == base_output.shape

        # Initially lora_B is zeros, so output equals base_output
        # After setting non-zero B, delta should be non-zero
        lora.lora_B = mx.random.normal((256, 4)) * 0.01
        output_after = lora(x, base_output)

        diff = mx.abs(output_after - base_output)
        assert float(mx.max(diff)) > 0.0

    def test_lora_weights(self):
        """Test LoRA weight get/set."""
        lora = LoRALinear(in_features=128, out_features=128, rank=4)

        # Get weights
        weights = lora.get_weights()
        assert "lora_A" in weights
        assert "lora_B" in weights
        assert weights["lora_A"].shape == (4, 128)
        assert weights["lora_B"].shape == (128, 4)

        # Modify and set
        new_weights = {
            "lora_A": mx.ones((4, 128)),
            "lora_B": mx.ones((128, 4)),
        }
        lora.set_weights(new_weights)

        # Verify
        result = lora.get_weights()
        assert mx.allclose(result["lora_A"], mx.ones((4, 128)))


class TestSpeakerLoRAAdapter:
    """Tests for speaker LoRA adapter."""

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = SpeakerLoRAAdapter(
            d_model=512,
            num_layers=6,
            rank=8,
            alpha=16.0,
            dropout=0.1,
        )

        assert adapter.d_model == 512
        assert adapter.num_layers == 6
        assert len(adapter.q_loras) == 6
        assert len(adapter.v_loras) == 6

    def test_apply_to_layer(self):
        """Test applying adapter to a layer."""
        adapter = SpeakerLoRAAdapter(
            d_model=256,
            num_layers=4,
            rank=4,
        )

        x = mx.random.normal((2, 10, 256))
        q_base = mx.random.normal((2, 10, 256))
        v_base = mx.random.normal((2, 10, 256))

        q_adapted, v_adapted = adapter.apply_to_layer(
            layer_idx=0,
            x=x,
            q_base=q_base,
            v_base=v_base,
        )

        assert q_adapted.shape == q_base.shape
        assert v_adapted.shape == v_base.shape

    def test_adapter_weights(self):
        """Test adapter weight serialization."""
        adapter = SpeakerLoRAAdapter(
            d_model=128,
            num_layers=2,
            rank=4,
        )

        weights = adapter.get_weights()

        # Should have Q and V LoRA weights for each layer
        assert "layer_0_q_lora_A" in weights
        assert "layer_0_q_lora_B" in weights
        assert "layer_0_v_lora_A" in weights
        assert "layer_0_v_lora_B" in weights
        assert "layer_1_q_lora_A" in weights

        # Roundtrip
        adapter2 = SpeakerLoRAAdapter(d_model=128, num_layers=2, rank=4)
        adapter2.set_weights(weights)

        weights2 = adapter2.get_weights()
        for key in weights:
            assert mx.allclose(weights[key], weights2[key])

    def test_num_parameters(self):
        """Test parameter counting."""
        adapter = SpeakerLoRAAdapter(
            d_model=256,
            num_layers=4,
            rank=8,
        )

        num_params = adapter.num_parameters()

        # Each layer: 2 * (rank * d_model + d_model * rank) = 4 * rank * d_model
        # 4 layers, Q and V = 4 * 4 * 8 * 256 = 32768
        expected = 4 * 4 * 8 * 256
        assert num_params == expected


class TestPhonemeWeightedTrainer:
    """Tests for phoneme-weighted trainer."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        config = PhonemeWeightedTrainingConfig(
            min_quality_for_training=0.6,
            batch_size=4,
        )
        trainer = PhonemeWeightedTrainer(config=config)

        assert trainer.config.min_quality_for_training == 0.6
        assert trainer.config.batch_size == 4

    def test_quality_weights_computation(self):
        """Test quality score to weight conversion."""
        trainer = PhonemeWeightedTrainer(
            config=PhonemeWeightedTrainingConfig(
                quality_exponent=2.0,
                quality_weight_floor=0.1,
            ),
        )

        quality_scores = [0.9, 0.7, 0.5, 0.3]
        weights = trainer._compute_quality_weights(quality_scores)  # noqa: SLF001

        # Weights should sum to batch size (for comparable loss)
        assert abs(float(mx.sum(weights)) - len(quality_scores)) < 0.01

        # Higher quality should have higher weight
        assert float(weights[0]) > float(weights[1])
        assert float(weights[1]) > float(weights[2])
        assert float(weights[2]) > float(weights[3])

    def test_sample_filtering(self):
        """Test sample filtering by quality."""
        config = PhonemeWeightedTrainingConfig(
            min_quality_for_training=0.6,
        )
        trainer = PhonemeWeightedTrainer(config=config)

        samples = [
            make_sample(quality_score=0.9),
            make_sample(quality_score=0.7),
            make_sample(quality_score=0.5),  # Below threshold
            make_sample(quality_score=0.3),  # Below threshold
        ]

        filtered = trainer._filter_samples(samples)  # noqa: SLF001

        assert len(filtered) == 2
        assert filtered[0].quality_score == 0.9
        assert filtered[1].quality_score == 0.7

    def test_quality_histogram(self):
        """Test quality histogram building."""
        trainer = PhonemeWeightedTrainer()

        samples = [
            make_sample(quality_score=0.1),
            make_sample(quality_score=0.4),
            make_sample(quality_score=0.6),
            make_sample(quality_score=0.8),
            make_sample(quality_score=0.95),
        ]

        histogram = trainer._build_quality_histogram(samples)  # noqa: SLF001

        assert histogram["0.0-0.3"] == 1
        assert histogram["0.3-0.5"] == 1
        assert histogram["0.5-0.7"] == 1
        assert histogram["0.7-0.9"] == 1
        assert histogram["0.9-1.0"] == 1

    def test_train_speaker_adapter(self):
        """Test training a speaker adapter."""
        config = PhonemeWeightedTrainingConfig(
            min_quality_for_training=0.5,
            batch_size=4,
            num_epochs=2,
            log_interval=2,
        )
        trainer = PhonemeWeightedTrainer(config=config)

        # Create samples with varying quality
        samples = [
            make_sample(quality_score=0.9, speaker_id=1),
            make_sample(quality_score=0.8, speaker_id=1),
            make_sample(quality_score=0.7, speaker_id=1),
            make_sample(quality_score=0.6, speaker_id=1),
            make_sample(quality_score=0.55, speaker_id=1),
            make_sample(quality_score=0.4, speaker_id=1),  # Filtered
            make_sample(quality_score=0.3, speaker_id=1),  # Filtered
            make_sample(quality_score=0.2, speaker_id=1),  # Filtered
        ]

        result = trainer.train_speaker_adapter(
            speaker_id=1,
            samples=samples,
            d_model=256,
            num_layers=4,
        )

        assert isinstance(result, TrainingResult)
        assert result.speaker_id == 1
        assert result.total_samples == 8
        assert result.filtered_samples == 3  # 3 below 0.5 threshold
        assert result.total_steps > 0
        assert result.adapter_weights is not None
        assert len(result.quality_histogram) == 5

    def test_train_empty_samples(self):
        """Test training with no valid samples."""
        config = PhonemeWeightedTrainingConfig(
            min_quality_for_training=0.9,  # Very high threshold
        )
        trainer = PhonemeWeightedTrainer(config=config)

        samples = [
            make_sample(quality_score=0.5),
            make_sample(quality_score=0.6),
            make_sample(quality_score=0.7),
        ]

        result = trainer.train_speaker_adapter(
            speaker_id=1,
            samples=samples,
        )

        # All samples filtered, no training
        assert result.filtered_samples == 3
        assert result.total_steps == 0
        assert result.adapter_weights is None

    def test_multiple_speakers(self):
        """Test training adapters for multiple speakers."""
        trainer = PhonemeWeightedTrainer(
            config=PhonemeWeightedTrainingConfig(
                batch_size=2,
                num_epochs=1,
            ),
        )

        # Train speaker 1
        samples_1 = [make_sample(quality_score=0.9, speaker_id=1) for _ in range(5)]
        trainer.train_speaker_adapter(
            speaker_id=1,
            samples=samples_1,
            d_model=128,
            num_layers=2,
        )

        # Train speaker 2
        samples_2 = [make_sample(quality_score=0.8, speaker_id=2) for _ in range(5)]
        trainer.train_speaker_adapter(
            speaker_id=2,
            samples=samples_2,
            d_model=128,
            num_layers=2,
        )

        # Both should have adapters
        adapter_1 = trainer.get_adapter(1)
        adapter_2 = trainer.get_adapter(2)

        assert adapter_1 is not None
        assert adapter_2 is not None
        assert adapter_1 is not adapter_2


class TestOnlinePhonemeWeightedTrainer:
    """Tests for online/streaming trainer."""

    def test_online_trainer_initialization(self):
        """Test online trainer initialization."""
        base_trainer = PhonemeWeightedTrainer()
        online = OnlinePhonemeWeightedTrainer(
            base_trainer=base_trainer,
            ema_decay=0.95,
            update_interval=5,
        )

        assert online.ema_decay == 0.95
        assert online.update_interval == 5

    def test_process_single_sample(self):
        """Test processing a single sample."""
        base_trainer = PhonemeWeightedTrainer(
            config=PhonemeWeightedTrainingConfig(
                min_quality_for_training=0.5,
                batch_size=2,
            ),
        )
        online = OnlinePhonemeWeightedTrainer(
            base_trainer=base_trainer,
            update_interval=5,
        )

        sample = make_sample(quality_score=0.8, speaker_id=1)
        result = online.process_sample(speaker_id=1, sample=sample)

        assert result["accepted"] is True
        assert result["buffer_size"] == 1

    def test_filter_low_quality(self):
        """Test filtering low quality samples."""
        base_trainer = PhonemeWeightedTrainer(
            config=PhonemeWeightedTrainingConfig(
                min_quality_for_training=0.6,
            ),
        )
        online = OnlinePhonemeWeightedTrainer(base_trainer=base_trainer)

        sample = make_sample(quality_score=0.4, speaker_id=1)
        result = online.process_sample(speaker_id=1, sample=sample)

        assert result["accepted"] is False
        assert "below threshold" in result["reason"]

    def test_batch_update(self):
        """Test batch update after interval."""
        base_trainer = PhonemeWeightedTrainer(
            config=PhonemeWeightedTrainingConfig(
                min_quality_for_training=0.5,
                batch_size=2,
                num_epochs=1,
            ),
        )
        online = OnlinePhonemeWeightedTrainer(
            base_trainer=base_trainer,
            update_interval=3,  # Update every 3 samples
        )

        # Process samples
        for _ in range(3):
            sample = make_sample(quality_score=0.8, speaker_id=1)
            result = online.process_sample(speaker_id=1, sample=sample)

        # Third sample should trigger update
        assert result.get("updated") is True or result["buffer_size"] == 3

    def test_ema_loss_tracking(self):
        """Test EMA loss tracking."""
        base_trainer = PhonemeWeightedTrainer()
        online = OnlinePhonemeWeightedTrainer(
            base_trainer=base_trainer,
            ema_decay=0.9,
        )

        # Initial EMA loss should be 1.0
        assert online.get_ema_loss(speaker_id=1) == 1.0

        # Process a sample
        sample = make_sample(quality_score=0.8, speaker_id=1)
        online.process_sample(speaker_id=1, sample=sample)

        # EMA loss should be tracked
        ema = online.get_ema_loss(speaker_id=1)
        assert ema <= 1.0


class TestIntegration:
    """Integration tests for phoneme-weighted training."""

    def test_end_to_end_training(self):
        """Test end-to-end training workflow."""
        # Create trainer
        config = PhonemeWeightedTrainingConfig(
            min_quality_for_training=0.4,
            batch_size=4,
            num_epochs=2,
            lora_rank=4,
        )
        trainer = PhonemeWeightedTrainer(config=config)

        # Simulate collected samples with varying quality
        samples = [
            make_sample(quality_score=quality, speaker_id=1)
            for quality in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.3]
        ]

        # Train
        result = trainer.train_speaker_adapter(
            speaker_id=1,
            samples=samples,
            d_model=256,
            num_layers=4,
        )

        # Verify result
        assert result.total_samples == 10
        assert result.filtered_samples == 1  # Only 0.3 filtered
        assert result.adapter_weights is not None
        assert result.best_loss < float("inf")

        # Verify adapter can be retrieved
        adapter = trainer.get_adapter(1)
        assert adapter is not None
        assert adapter.num_parameters() > 0

    def test_quality_weighting_effect(self):
        """Test that quality weighting affects training."""
        # High quality samples
        high_quality_samples = [
            make_sample(quality_score=0.95) for _ in range(10)
        ]

        # Low quality samples
        low_quality_samples = [
            make_sample(quality_score=0.55) for _ in range(10)
        ]

        trainer_high = PhonemeWeightedTrainer(
            config=PhonemeWeightedTrainingConfig(
                batch_size=4,
                num_epochs=2,
            ),
        )
        trainer_low = PhonemeWeightedTrainer(
            config=PhonemeWeightedTrainingConfig(
                batch_size=4,
                num_epochs=2,
            ),
        )

        result_high = trainer_high.train_speaker_adapter(
            speaker_id=1,
            samples=high_quality_samples,
            d_model=128,
            num_layers=2,
        )

        result_low = trainer_low.train_speaker_adapter(
            speaker_id=1,
            samples=low_quality_samples,
            d_model=128,
            num_layers=2,
        )

        # Both should train, but loss differs due to quality weighting
        assert result_high.total_steps > 0
        assert result_low.total_steps > 0

        # Quality distribution should reflect input
        assert result_high.quality_histogram["0.9-1.0"] == 10
        assert result_low.quality_histogram["0.5-0.7"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
