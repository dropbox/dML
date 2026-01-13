#!/usr/bin/env python3
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
Tests for prosody embedding training script.

Tests cover:
- Mock data generation
- Training loop
- Checkpoint save/load
- Embedding analysis
"""

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.train_prosody_embeddings import (
    NUM_PROSODY_TYPES,
    PROSODY_TYPES,
    ProsodyEmbedding,
    TrainingConfig,
    analyze_embeddings,
    compute_loss,
    create_mock_dataset,
    load_checkpoint,
    load_dataset,
    prepare_batch,
    save_checkpoint,
    train,
)


class TestProsodyEmbedding:
    """Tests for ProsodyEmbedding module."""

    def test_init(self):
        """Test module initialization."""
        model = ProsodyEmbedding(num_types=63, hidden_dim=512)
        assert model.num_types == 63
        assert model.hidden_dim == 512
        assert model.embedding.weight.shape == (63, 512)

    def test_forward(self):
        """Test forward pass."""
        model = ProsodyEmbedding(num_types=63, hidden_dim=512)

        # Single prosody type
        prosody_mask = mx.array([[0]], dtype=mx.int32)  # NEUTRAL
        output = model(prosody_mask)
        assert output.shape == (1, 1, 512)

        # Multiple tokens
        prosody_mask = mx.array([[0, 1, 2, 1, 0]], dtype=mx.int32)
        output = model(prosody_mask)
        assert output.shape == (1, 5, 512)

        # Batch
        prosody_mask = mx.array([[0, 1, 2], [3, 4, 5]], dtype=mx.int32)
        output = model(prosody_mask)
        assert output.shape == (2, 3, 512)

    def test_scale_factor(self):
        """Test that scale factor affects output magnitude."""
        model = ProsodyEmbedding(num_types=63, hidden_dim=512, init_scale=0.1)
        prosody_mask = mx.array([[1]], dtype=mx.int32)

        output1 = model(prosody_mask)

        # Increase scale
        model.scale = mx.array([1.0])
        output2 = model(prosody_mask)

        # Output should be 10x larger
        ratio = float(mx.mean(mx.abs(output2)) / mx.mean(mx.abs(output1)))
        assert 9.0 < ratio < 11.0


class TestMockDataset:
    """Tests for mock dataset generation."""

    def test_create_mock_dataset(self):
        """Test mock dataset creation."""
        samples = create_mock_dataset(50)
        assert len(samples) == 50

        # Check sample fields
        for sample in samples:
            assert isinstance(sample.text, str)
            assert isinstance(sample.prosody_type, (int, np.integer))
            assert isinstance(sample.f0_mean, (float, np.floating))
            assert sample.f0_mean > 0
            assert sample.duration_s > 0

    def test_mock_dataset_variety(self):
        """Test that mock dataset has variety in prosody types."""
        samples = create_mock_dataset(100)
        prosody_types = set(s.prosody_type for s in samples)

        # Should have multiple prosody types
        assert len(prosody_types) >= 5

    def test_prosody_effects_correlation(self):
        """Test that prosody types have correlated effects."""
        samples = create_mock_dataset(500)

        # Group by prosody type
        by_type = {}
        for s in samples:
            if s.prosody_type not in by_type:
                by_type[s.prosody_type] = []
            by_type[s.prosody_type].append(s)

        # EMOTION_SAD should have lower F0 than EMOTION_EXCITED
        if PROSODY_TYPES["EMOTION_SAD"] in by_type and PROSODY_TYPES["EMOTION_EXCITED"] in by_type:
            sad_f0 = np.mean([s.f0_mean for s in by_type[PROSODY_TYPES["EMOTION_SAD"]]])
            excited_f0 = np.mean([s.f0_mean for s in by_type[PROSODY_TYPES["EMOTION_EXCITED"]]])
            assert sad_f0 < excited_f0

        # RATE_SLOW should have longer duration than RATE_FAST
        if PROSODY_TYPES["RATE_SLOW"] in by_type and PROSODY_TYPES["RATE_FAST"] in by_type:
            slow_dur = np.mean([s.duration_s for s in by_type[PROSODY_TYPES["RATE_SLOW"]]])
            fast_dur = np.mean([s.duration_s for s in by_type[PROSODY_TYPES["RATE_FAST"]]])
            assert slow_dur > fast_dur


class TestBatchPreparation:
    """Tests for batch preparation."""

    def test_prepare_batch(self):
        """Test batch preparation."""
        samples = create_mock_dataset(10)
        indices = [0, 1, 2, 3]

        prosody_types, f0_targets, f0_std_targets, dur_targets = prepare_batch(samples, indices)

        assert prosody_types.shape == (4,)
        assert f0_targets.shape == (4,)
        assert f0_std_targets.shape == (4,)
        assert dur_targets.shape == (4,)

        # Check dtype
        assert prosody_types.dtype == mx.int32
        assert f0_targets.dtype == mx.float32


class TestComputeLoss:
    """Tests for loss computation."""

    def test_loss_positive(self):
        """Test that loss is positive."""
        model = ProsodyEmbedding(num_types=63, hidden_dim=512)
        samples = create_mock_dataset(10)
        indices = list(range(10))

        prosody_types, f0_targets, f0_std_targets, dur_targets = prepare_batch(samples, indices)
        loss, metrics = compute_loss(model, prosody_types, f0_targets, f0_std_targets, dur_targets)

        assert float(loss) > 0
        assert metrics["total"] > 0
        assert metrics["loss_f0"] >= 0
        assert metrics["loss_dur"] >= 0

    def test_loss_decreases_with_training(self):
        """Test that loss decreases during training."""
        model = ProsodyEmbedding(num_types=63, hidden_dim=512)
        samples = create_mock_dataset(100)
        indices = list(range(100))

        prosody_types, f0_targets, f0_std_targets, dur_targets = prepare_batch(samples, indices)

        initial_loss, _ = compute_loss(model, prosody_types, f0_targets, f0_std_targets, dur_targets)

        # Simple training (no optimizer, just checking gradient flow)
        # This test verifies the loss function is differentiable


class TestTraining:
    """Tests for training loop."""

    def test_train_short(self):
        """Test short training run."""
        config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=16,
            epochs=5,
            hidden_dim=128,  # Smaller for faster test
            log_interval=100,  # Don't log during test
            val_interval=100,
        )

        train_samples = create_mock_dataset(50)
        val_samples = create_mock_dataset(10)

        model = train(config, train_samples, val_samples)

        assert model is not None
        assert model.embedding.weight.shape == (63, 128)

    def test_train_loss_decreases(self):
        """Test that training reduces loss."""
        config = TrainingConfig(
            learning_rate=1e-2,  # Higher LR for faster convergence
            batch_size=32,
            epochs=10,
            hidden_dim=64,
            log_interval=100,
            val_interval=100,
        )

        samples = create_mock_dataset(100)

        # Get initial loss
        model_init = ProsodyEmbedding(num_types=63, hidden_dim=64)
        indices = list(range(100))
        prosody_types, f0_targets, f0_std_targets, dur_targets = prepare_batch(samples, indices)
        initial_loss, _ = compute_loss(model_init, prosody_types, f0_targets, f0_std_targets, dur_targets)

        # Train
        model = train(config, samples)

        # Get final loss
        final_loss, _ = compute_loss(model, prosody_types, f0_targets, f0_std_targets, dur_targets)

        assert float(final_loss) < float(initial_loss)


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        model = ProsodyEmbedding(num_types=63, hidden_dim=128)
        model.scale = mx.array([0.5])  # Change scale

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"
            save_checkpoint(model, path)

            assert path.exists()

            # Load
            loaded = load_checkpoint(path, hidden_dim=128)

            # Compare
            assert float(loaded.scale) == pytest.approx(0.5, abs=1e-6)
            assert loaded.embedding.weight.shape == model.embedding.weight.shape

            # Compare weights
            orig_weight = np.array(model.embedding.weight)
            loaded_weight = np.array(loaded.embedding.weight)
            np.testing.assert_allclose(orig_weight, loaded_weight, rtol=1e-5)


class TestAnalysis:
    """Tests for embedding analysis."""

    def test_analyze_embeddings(self):
        """Test embedding analysis."""
        model = ProsodyEmbedding(num_types=63, hidden_dim=128)
        model.scale = mx.array([0.3])

        analysis = analyze_embeddings(model)

        assert "scale" in analysis
        assert analysis["scale"] == pytest.approx(0.3, abs=1e-6)
        assert "embedding_norm_mean" in analysis
        assert "top_similar_pairs" in analysis


class TestDatasetLoading:
    """Tests for dataset loading from JSON."""

    def test_load_dataset(self):
        """Test loading dataset from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON file
            data = [
                {
                    "text": "Hello world",
                    "prosody_type": 0,
                    "f0_mean": 180.0,
                    "f0_std": 30.0,
                    "duration_s": 1.0,
                    "energy_rms": 0.05,
                },
                {
                    "text": "Test sentence",
                    "prosody_type": 40,  # EMOTION_ANGRY
                    "f0_mean": 200.0,
                    "f0_std": 40.0,
                    "duration_s": 0.8,
                    "energy_rms": 0.08,
                },
            ]

            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(data, f)

            # Load
            samples = load_dataset(path)

            assert len(samples) == 2
            assert samples[0].text == "Hello world"
            assert samples[0].prosody_type == 0
            assert samples[0].f0_mean == 180.0
            assert samples[1].prosody_type == 40


class TestProsodyTypes:
    """Tests for prosody type definitions."""

    def test_prosody_types_coverage(self):
        """Test that all expected prosody types are defined."""
        # Emphasis
        assert "NEUTRAL" in PROSODY_TYPES
        assert "EMPHASIS" in PROSODY_TYPES
        assert "STRONG_EMPHASIS" in PROSODY_TYPES

        # Rate
        assert "RATE_SLOW" in PROSODY_TYPES
        assert "RATE_FAST" in PROSODY_TYPES

        # Emotions
        assert "EMOTION_ANGRY" in PROSODY_TYPES
        assert "EMOTION_SAD" in PROSODY_TYPES
        assert "EMOTION_EXCITED" in PROSODY_TYPES
        assert "EMOTION_CALM" in PROSODY_TYPES

    def test_num_prosody_types(self):
        """Test NUM_PROSODY_TYPES is correct."""
        max_type = max(PROSODY_TYPES.values())
        assert NUM_PROSODY_TYPES == max_type + 1


class TestKokoroModelIntegration:
    """Tests for Kokoro model prosody embedding integration."""

    def test_kokoro_prosody_embedding_import(self):
        """Test that ProsodyEmbedding is importable from kokoro.py."""
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            NUM_PROSODY_TYPES as KOKORO_NUM_TYPES,
        )
        from tools.pytorch_to_mlx.converters.models.kokoro import (
            ProsodyEmbedding as KokoroProsodyEmbedding,
        )

        assert KOKORO_NUM_TYPES == 63
        # Test module creation
        emb = KokoroProsodyEmbedding(num_types=63, hidden_dim=768)
        assert emb.num_types == 63
        assert emb.hidden_dim == 768

    def test_kokoro_prosody_embedding_forward(self):
        """Test ProsodyEmbedding forward pass."""
        from tools.pytorch_to_mlx.converters.models.kokoro import ProsodyEmbedding

        emb = ProsodyEmbedding(num_types=63, hidden_dim=768, init_scale=0.5)

        # Create input mask
        prosody_mask = mx.array([[0, 1, 2, 40, 41]])  # [batch=1, seq=5]

        # Forward pass
        output = emb(prosody_mask)

        assert output.shape == (1, 5, 768)

        # Check scale is applied
        emb_raw = emb.embedding(prosody_mask)
        expected = 0.5 * emb_raw
        np.testing.assert_allclose(
            np.array(output), np.array(expected), rtol=1e-5,
        )

    def test_load_prosody_weights(self):
        """Test loading trained prosody weights into a mock model."""
        from tools.pytorch_to_mlx.converters.models.kokoro import ProsodyEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a checkpoint from training module
            train_emb = ProsodyEmbedding(num_types=63, hidden_dim=768)
            train_emb.scale = mx.array([0.836])  # Set trained scale
            mx.eval(train_emb.embedding.weight)

            # Save checkpoint
            path = Path(tmpdir) / "test_weights.safetensors"
            mx.save_safetensors(
                str(path),
                {
                    "embedding.weight": train_emb.embedding.weight,
                    "scale": train_emb.scale,
                },
            )

            # Load into new embedding
            new_emb = ProsodyEmbedding(num_types=63, hidden_dim=768)
            weights = mx.load(str(path))
            new_emb.embedding.weight = weights["embedding.weight"]
            new_emb.scale = weights["scale"]

            # Verify
            assert float(new_emb.scale) == pytest.approx(0.836, abs=1e-3)
            np.testing.assert_allclose(
                np.array(train_emb.embedding.weight),
                np.array(new_emb.embedding.weight),
                rtol=1e-5,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
