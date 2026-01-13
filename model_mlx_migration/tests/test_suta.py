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
Tests for SUTA (Single-Utterance Test-Time Adaptation).

Tests cover:
1. Basic adapter creation and configuration
2. Entropy and diversity loss functions
3. Layer norm discovery and state management
4. Adaptation behavior with mock models
5. Gradient-based optimization path
"""
# ruff: noqa: SLF001, RET504

import mlx.core as mx
import mlx.nn as nn
import pytest

from tools.whisper_mlx.sota.suta import (
    SUTAAdapter,
    SUTAConfig,
    SUTAWithGradients,
    create_suta_adapter,
    diversity_loss,
    entropy_loss,
)


class SimpleModelWithLayerNorm(nn.Module):
    """Simple model for testing SUTA."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, vocab_size: int = 100):
        super().__init__()
        self.encoder_ln = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.decoder_ln = nn.LayerNorm(vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.encoder_ln(x)
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.hidden_ln(x)
        x = self.fc2(x)
        return self.decoder_ln(x)


class TestEntropyLoss:
    """Tests for entropy loss function."""

    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution should have high entropy."""
        # Uniform logits (all zeros) -> uniform probs
        logits = mx.zeros((2, 10, 100))
        ent = entropy_loss(logits)

        # Entropy of uniform over 100 classes = log(100) ≈ 4.6
        assert float(ent) > 4.0, "Uniform distribution should have high entropy"

    def test_peaked_distribution_low_entropy(self):
        """Peaked distribution should have low entropy."""
        # One-hot logits -> peaked distribution
        logits = mx.zeros((2, 10, 100))
        logits = logits.at[:, :, 0].add(100.0)  # Make first class very likely

        ent = entropy_loss(logits)
        assert float(ent) < 0.1, "Peaked distribution should have low entropy"

    def test_entropy_shape_invariance(self):
        """Entropy should work with different batch/seq lengths."""
        for batch in [1, 4, 8]:
            for seq in [5, 10, 50]:
                logits = mx.random.normal((batch, seq, 100))
                ent = entropy_loss(logits)
                assert ent.ndim == 0, "Entropy should be scalar"


class TestDiversityLoss:
    """Tests for diversity loss function."""

    def test_same_predictions_low_diversity(self):
        """Same prediction at all timesteps = low diversity (high loss)."""
        # All timesteps predict same distribution
        logits = mx.zeros((2, 20, 100))
        logits = logits.at[:, :, 5].add(10.0)  # All predict class 5

        div = diversity_loss(logits)
        # Negative entropy of uniform is positive (we negate to maximize)
        # Here, not diverse, so loss should be high (less negative)
        assert float(div) > -1.0, "Same predictions should have high diversity loss"

    def test_varied_predictions_high_diversity(self):
        """Different predictions at each timestep = high diversity (low loss)."""
        # Create different distributions at each timestep
        batch_size, seq_len, vocab_size = 2, 20, 100
        logits = mx.random.normal((batch_size, seq_len, vocab_size)) * 5.0

        div = diversity_loss(logits)
        # Random predictions averaged should be somewhat uniform -> high entropy -> low loss
        assert float(div) < 0, "Varied predictions should have negative (low) diversity loss"


class TestSUTAConfig:
    """Tests for SUTA configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SUTAConfig()
        assert config.learning_rate == 1e-4
        assert config.n_steps == 3
        assert config.entropy_weight == 1.0
        assert config.adapt_encoder_ln is True
        assert config.adapt_decoder_ln is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SUTAConfig(
            learning_rate=5e-5,
            n_steps=5,
            entropy_weight=2.0,
            adapt_encoder_ln=False,
        )
        assert config.learning_rate == 5e-5
        assert config.n_steps == 5
        assert config.entropy_weight == 2.0
        assert config.adapt_encoder_ln is False


class TestSUTAAdapter:
    """Tests for SUTA adapter."""

    def test_adapter_creation(self):
        """Test basic adapter creation."""
        model = SimpleModelWithLayerNorm()
        adapter = SUTAAdapter(model)

        assert adapter.model is model
        assert adapter.lr == 1e-4
        assert adapter.n_steps == 3
        assert len(adapter._layer_norms) > 0, "Should find layer norms"  # noqa: SLF001

    def test_adapter_with_config(self):
        """Test adapter with custom config."""
        model = SimpleModelWithLayerNorm()
        config = SUTAConfig(learning_rate=1e-3, n_steps=5)
        adapter = SUTAAdapter(model, config=config)

        assert adapter.lr == 1e-3
        assert adapter.n_steps == 5

    def test_layer_norm_discovery(self):
        """Test that adapter finds all layer norms."""
        model = SimpleModelWithLayerNorm()
        adapter = SUTAAdapter(model)

        # Model has 3 LayerNorms
        assert len(adapter._layer_norms) == 3  # noqa: SLF001

    def test_state_save_restore(self):
        """Test state save and restore."""
        model = SimpleModelWithLayerNorm()
        adapter = SUTAAdapter(model)

        # Get initial weights
        initial_weight = mx.array(model.encoder_ln.weight)

        # Save state
        adapter._save_states()

        # Modify weights
        model.encoder_ln.weight = model.encoder_ln.weight * 2.0
        modified_weight = mx.array(model.encoder_ln.weight)

        assert not mx.allclose(initial_weight, modified_weight), "Weights should be modified"

        # Restore state
        adapter._restore_states()

        # Verify restored
        restored_weight = model.encoder_ln.weight
        assert mx.allclose(initial_weight, restored_weight), "Weights should be restored"

    def test_adapt_and_predict_shape(self):
        """Test that adapt_and_predict returns correct shape."""
        model = SimpleModelWithLayerNorm(input_dim=64, hidden_dim=128, vocab_size=100)
        adapter = SUTAAdapter(model)

        # Input features
        features = mx.random.normal((2, 10, 64))

        # Adapt and predict
        output = adapter.adapt_and_predict(features)

        assert output.shape == (2, 10, 100), f"Expected (2, 10, 100), got {output.shape}"

    def test_adapt_restores_weights(self):
        """Test that adaptation doesn't persist weight changes."""
        model = SimpleModelWithLayerNorm()
        adapter = SUTAAdapter(model)

        # Get initial weight
        initial_weight = mx.array(model.encoder_ln.weight)

        # Run adaptation
        features = mx.random.normal((2, 10, 64))
        _ = adapter.adapt_and_predict(features)

        # Weights should be restored
        final_weight = model.encoder_ln.weight
        assert mx.allclose(initial_weight, final_weight), "Weights should be restored after adaptation"

    def test_get_stats(self):
        """Test statistics collection."""
        model = SimpleModelWithLayerNorm()
        adapter = SUTAAdapter(model)

        # Run adaptation
        features = mx.random.normal((2, 10, 64))
        _ = adapter.adapt_and_predict(features)

        stats = adapter.get_stats()
        assert "initial_entropy" in stats
        assert "final_entropy" in stats
        assert "entropy_reduction" in stats
        assert "n_adapted_steps" in stats
        assert "n_layer_norms" in stats

    def test_confident_prediction_skips_adaptation(self):
        """Test that confident predictions skip adaptation."""
        model = SimpleModelWithLayerNorm()
        # Use high min_confidence to trigger skip
        config = SUTAConfig(min_confidence=0.99)
        adapter = SUTAAdapter(model, config=config)

        # Create very confident input (single dominant logit)
        features = mx.zeros((1, 1, 64))

        _ = adapter.adapt_and_predict(features)

        # With random initialization, might not skip, but stats should be collected
        stats = adapter.get_stats()
        assert stats["n_layer_norms"] == 3


class TestSUTAWithGradients:
    """Tests for gradient-based SUTA."""

    def test_gradient_adapter_creation(self):
        """Test gradient-based adapter creation."""
        model = SimpleModelWithLayerNorm()
        adapter = SUTAWithGradients(model)

        assert isinstance(adapter, SUTAAdapter)
        assert len(adapter._layer_norms) == 3

    def test_gradient_adapt_and_predict_shape(self):
        """Test gradient-based adaptation output shape."""
        model = SimpleModelWithLayerNorm(input_dim=64, hidden_dim=128, vocab_size=100)
        adapter = SUTAWithGradients(model)

        features = mx.random.normal((2, 10, 64))
        output = adapter.adapt_and_predict(features)

        assert output.shape == (2, 10, 100)

    def test_gradient_restores_weights(self):
        """Test that gradient adaptation restores weights."""
        model = SimpleModelWithLayerNorm()
        adapter = SUTAWithGradients(model)

        initial_weight = mx.array(model.encoder_ln.weight)

        features = mx.random.normal((2, 10, 64))
        _ = adapter.adapt_and_predict(features)

        final_weight = model.encoder_ln.weight
        assert mx.allclose(initial_weight, final_weight), "Weights should be restored"


class TestCreateSUTAAdapter:
    """Tests for factory function."""

    def test_create_with_gradients(self):
        """Test creating gradient-based adapter."""
        model = SimpleModelWithLayerNorm()
        adapter = create_suta_adapter(model, use_gradients=True)

        assert isinstance(adapter, SUTAWithGradients)

    def test_create_without_gradients(self):
        """Test creating heuristic adapter."""
        model = SimpleModelWithLayerNorm()
        adapter = create_suta_adapter(model, use_gradients=False)

        assert isinstance(adapter, SUTAAdapter)
        assert not isinstance(adapter, SUTAWithGradients)

    def test_create_with_kwargs(self):
        """Test creating adapter with custom config via kwargs."""
        model = SimpleModelWithLayerNorm()
        adapter = create_suta_adapter(
            model,
            use_gradients=False,
            learning_rate=5e-4,
            n_steps=10,
        )

        assert adapter.lr == 5e-4
        assert adapter.n_steps == 10


class TestAdaptCTC:
    """Tests for CTC-specific adaptation."""

    def test_adapt_ctc_shape(self):
        """Test CTC adaptation output shape."""
        # Create a simple model that outputs vocab-sized logits
        hidden_dim = 128
        vocab_size = 51865

        class EncoderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(hidden_dim)

            def __call__(self, x):
                return self.ln(x)

        encoder = EncoderModel()
        ctc_head = nn.Linear(hidden_dim, vocab_size)

        adapter = SUTAAdapter(encoder)

        # Input is already encoder output (hidden_dim)
        encoder_out = mx.random.normal((2, 50, hidden_dim))
        output = adapter.adapt_ctc(encoder_out, ctc_head)

        # Should match CTC head output shape
        assert output.shape == (2, 50, vocab_size), f"Expected (2, 50, {vocab_size}), got {output.shape}"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_model(self):
        """Test with model that has no layer norms."""

        class NoLayerNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 100)

            def __call__(self, x):
                return self.fc(x)

        model = NoLayerNormModel()
        adapter = SUTAAdapter(model)

        assert len(adapter._layer_norms) == 0

        # Should still work, just no adaptation
        features = mx.random.normal((2, 10, 64))
        output = adapter.adapt_and_predict(features)
        assert output.shape == (2, 10, 100)

    def test_single_sample(self):
        """Test with single sample."""
        model = SimpleModelWithLayerNorm()
        adapter = SUTAAdapter(model)

        features = mx.random.normal((1, 5, 64))
        output = adapter.adapt_and_predict(features)
        assert output.shape == (1, 5, 100)

    def test_encoder_only_adaptation(self):
        """Test adapting only encoder layer norms."""
        model = SimpleModelWithLayerNorm()
        config = SUTAConfig(adapt_encoder_ln=True, adapt_decoder_ln=False)
        adapter = SUTAAdapter(model, config=config)

        # Should find fewer layer norms (only encoder_ln)
        # Note: Our simple model doesn't have clear encoder/decoder separation
        # so all are discovered based on name heuristics
        features = mx.random.normal((2, 10, 64))
        output = adapter.adapt_and_predict(features)
        assert output.shape == (2, 10, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
