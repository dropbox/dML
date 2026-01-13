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
Tests for pitch (F0) prediction head.

Tests cover:
- Forward pass with different input shapes
- F0 prediction range and scaling
- Voicing prediction
- Loss function with voicing mask
- Gradient flow for training
- Utility functions (Hz to cents conversion)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from src.models.heads.pitch import (
    PitchConfig,
    PitchHead,
    PitchLoss,
    cents_to_hz,
    compute_pitch_mae,
    compute_voicing_accuracy,
    hz_to_cents,
    pitch_loss,
)


class TestPitchConfig:
    """Tests for PitchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PitchConfig()
        assert config.encoder_dim == 384
        assert config.hidden_dim == 256
        assert config.num_layers == 2
        assert config.f0_min_hz == 50.0
        assert config.f0_max_hz == 800.0
        assert config.output_mode == "log_hz"
        assert config.predict_voicing is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PitchConfig(
            encoder_dim=512,
            f0_max_hz=1000.0,
            output_mode="hz",
        )
        assert config.encoder_dim == 512
        assert config.f0_max_hz == 1000.0
        assert config.output_mode == "hz"


class TestPitchHead:
    """Tests for PitchHead module."""

    def test_forward_shape(self):
        """Test output shape of pitch head."""
        config = PitchConfig(encoder_dim=384)
        head = PitchHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        f0_hz, voicing_prob = head(encoder_out)

        assert f0_hz.shape == (batch_size, seq_len)
        assert voicing_prob.shape == (batch_size, seq_len)

    def test_f0_range(self):
        """Test that F0 predictions are within expected range."""
        config = PitchConfig(
            encoder_dim=256,
            f0_min_hz=50.0,
            f0_max_hz=800.0,
        )
        head = PitchHead(config)

        batch_size = 4
        seq_len = 50
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 256))

        f0_hz, _ = head(encoder_out)

        # F0 should be within configured range
        assert float(mx.min(f0_hz)) >= config.f0_min_hz - 1e-3
        assert float(mx.max(f0_hz)) <= config.f0_max_hz + 1e-3

    def test_voicing_range(self):
        """Test that voicing predictions are probabilities."""
        config = PitchConfig(encoder_dim=256)
        head = PitchHead(config)

        encoder_out = mx.random.normal(shape=(4, 50, 256))
        _, voicing_prob = head(encoder_out)

        # Voicing should be in [0, 1]
        assert float(mx.min(voicing_prob)) >= 0.0
        assert float(mx.max(voicing_prob)) <= 1.0

    def test_no_voicing_prediction(self):
        """Test with voicing prediction disabled."""
        config = PitchConfig(encoder_dim=256, predict_voicing=False)
        head = PitchHead(config)

        encoder_out = mx.random.normal(shape=(4, 50, 256))
        f0_hz, voicing_prob = head(encoder_out)

        assert f0_hz.shape == (4, 50)
        assert voicing_prob is None

    def test_predict_method(self):
        """Test predict method with voicing threshold."""
        config = PitchConfig(encoder_dim=256)
        head = PitchHead(config)

        encoder_out = mx.random.normal(shape=(4, 50, 256))
        f0_hz, voiced = head.predict(encoder_out, voicing_threshold=0.5)

        assert f0_hz.shape == (4, 50)
        assert voiced.shape == (4, 50)
        assert voiced.dtype == mx.bool_

    def test_output_mode_hz(self):
        """Test direct Hz output mode."""
        config = PitchConfig(encoder_dim=256, output_mode="hz")
        head = PitchHead(config)

        encoder_out = mx.random.normal(shape=(4, 50, 256))
        f0_hz, _ = head(encoder_out)

        assert float(mx.min(f0_hz)) >= config.f0_min_hz - 1e-3
        assert float(mx.max(f0_hz)) <= config.f0_max_hz + 1e-3

    def test_output_mode_log_hz(self):
        """Test log Hz output mode."""
        config = PitchConfig(encoder_dim=256, output_mode="log_hz")
        head = PitchHead(config)

        encoder_out = mx.random.normal(shape=(4, 50, 256))
        f0_hz, _ = head(encoder_out)

        assert float(mx.min(f0_hz)) >= config.f0_min_hz - 1e-3
        assert float(mx.max(f0_hz)) <= config.f0_max_hz + 1e-3

    def test_output_mode_cents(self):
        """Test cents output mode."""
        config = PitchConfig(encoder_dim=256, output_mode="cents")
        head = PitchHead(config)

        encoder_out = mx.random.normal(shape=(4, 50, 256))
        f0_hz, _ = head(encoder_out)

        # Should still be valid Hz values
        assert mx.all(mx.isfinite(f0_hz))

    def test_default_config_creation(self):
        """Test head creation with default config."""
        head = PitchHead()
        assert head.config.encoder_dim == 384


class TestPitchLoss:
    """Tests for pitch loss function."""

    def test_loss_shape(self):
        """Test loss output shape."""
        batch_size = 4
        seq_len = 50

        f0_pred = mx.random.uniform(low=100, high=400, shape=(batch_size, seq_len))
        f0_target = mx.random.uniform(low=100, high=400, shape=(batch_size, seq_len))
        voicing_pred = mx.random.uniform(shape=(batch_size, seq_len))
        voicing_target = mx.random.uniform(shape=(batch_size, seq_len))

        total, f0_loss, voicing_loss = pitch_loss(
            f0_pred, f0_target, voicing_pred, voicing_target,
        )

        assert total.shape == ()
        assert f0_loss.shape == ()
        assert voicing_loss.shape == ()

    def test_loss_none_reduction(self):
        """Test loss with no reduction."""
        batch_size = 4
        seq_len = 50

        f0_pred = mx.random.uniform(low=100, high=400, shape=(batch_size, seq_len))
        f0_target = mx.random.uniform(low=100, high=400, shape=(batch_size, seq_len))

        total, f0_loss, _ = pitch_loss(
            f0_pred, f0_target, None, None, reduction="none",
        )

        assert f0_loss.shape == (batch_size, seq_len)

    def test_voiced_only_f0_loss(self):
        """Test that F0 loss is only computed on voiced frames."""
        batch_size = 2
        seq_len = 10

        f0_pred = mx.ones((batch_size, seq_len)) * 200
        f0_target = mx.ones((batch_size, seq_len)) * 300

        # All unvoiced - F0 loss should be ~0
        voicing_target = mx.zeros((batch_size, seq_len))
        _, f0_loss_unvoiced, _ = pitch_loss(
            f0_pred, f0_target, None, voicing_target,
        )

        # All voiced - F0 loss should be 100 (MAE)
        voicing_target = mx.ones((batch_size, seq_len))
        _, f0_loss_voiced, _ = pitch_loss(
            f0_pred, f0_target, None, voicing_target,
        )

        assert float(f0_loss_unvoiced) < 1e-5  # Nearly 0 for unvoiced
        assert float(f0_loss_voiced) > 90  # ~100 for voiced

    def test_l1_vs_l2_loss(self):
        """Test L1 vs L2 loss types."""
        f0_pred = mx.array([[100.0, 200.0]])
        f0_target = mx.array([[150.0, 250.0]])  # 50 Hz error each

        _, f0_loss_l1, _ = pitch_loss(
            f0_pred, f0_target, None, None, f0_loss_type="l1",
        )
        _, f0_loss_l2, _ = pitch_loss(
            f0_pred, f0_target, None, None, f0_loss_type="l2",
        )

        # L1 loss = mean(50) = 50
        # L2 loss = mean(2500) = 2500
        assert abs(float(f0_loss_l1) - 50) < 1
        assert abs(float(f0_loss_l2) - 2500) < 1

    def test_loss_module(self):
        """Test PitchLoss module wrapper."""
        loss_fn = PitchLoss(f0_loss_type="l1", voicing_weight=0.5)

        f0_pred = mx.random.uniform(low=100, high=400, shape=(4, 50))
        f0_target = mx.random.uniform(low=100, high=400, shape=(4, 50))
        voicing_pred = mx.random.uniform(shape=(4, 50))
        voicing_target = mx.random.uniform(shape=(4, 50))

        total, f0_loss, voicing_loss = loss_fn(
            f0_pred, f0_target, voicing_pred, voicing_target,
        )

        assert total.shape == ()
        assert float(total) > 0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_hz_to_cents(self):
        """Test Hz to cents conversion."""
        # A4 = 440 Hz should be 0 cents relative to 440
        cents = hz_to_cents(mx.array([440.0]), reference_hz=440.0)
        assert abs(float(cents[0])) < 1e-5

        # A5 = 880 Hz should be 1200 cents (1 octave)
        cents = hz_to_cents(mx.array([880.0]), reference_hz=440.0)
        assert abs(float(cents[0]) - 1200) < 1e-3

        # A3 = 220 Hz should be -1200 cents
        cents = hz_to_cents(mx.array([220.0]), reference_hz=440.0)
        assert abs(float(cents[0]) + 1200) < 1e-3

    def test_cents_to_hz(self):
        """Test cents to Hz conversion."""
        # 0 cents at 440 ref should be 440 Hz
        hz = cents_to_hz(mx.array([0.0]), reference_hz=440.0)
        assert abs(float(hz[0]) - 440) < 1e-3

        # 1200 cents should be 880 Hz
        hz = cents_to_hz(mx.array([1200.0]), reference_hz=440.0)
        assert abs(float(hz[0]) - 880) < 1e-3

    def test_hz_cents_roundtrip(self):
        """Test Hz -> cents -> Hz roundtrip."""
        original_hz = mx.array([220.0, 440.0, 880.0])
        cents = hz_to_cents(original_hz, reference_hz=440.0)
        recovered_hz = cents_to_hz(cents, reference_hz=440.0)

        assert mx.all(mx.abs(original_hz - recovered_hz) < 1e-3)

    def test_compute_pitch_mae(self):
        """Test MAE computation."""
        f0_pred = mx.array([[100.0, 200.0, 300.0]])
        f0_target = mx.array([[150.0, 250.0, 350.0]])

        mae = compute_pitch_mae(f0_pred, f0_target)
        assert abs(float(mae) - 50) < 1e-3

    def test_compute_pitch_mae_with_voicing(self):
        """Test MAE with voicing mask."""
        f0_pred = mx.array([[100.0, 200.0, 300.0]])
        f0_target = mx.array([[150.0, 250.0, 350.0]])
        # Only first two frames voiced
        voicing_target = mx.array([[1.0, 1.0, 0.0]])

        mae = compute_pitch_mae(f0_pred, f0_target, voicing_target)
        # Only first two frames count, both have 50 Hz error
        assert abs(float(mae) - 50) < 1e-3

    def test_compute_voicing_accuracy(self):
        """Test voicing accuracy computation."""
        voicing_pred = mx.array([[0.8, 0.2, 0.6, 0.4]])
        voicing_target = mx.array([[1.0, 0.0, 1.0, 0.0]])

        accuracy = compute_voicing_accuracy(voicing_pred, voicing_target, threshold=0.5)
        # All 4 correct: 0.8>0.5=voiced(correct), 0.2<0.5=unvoiced(correct), etc.
        assert float(accuracy) == 1.0


class TestGradientFlow:
    """Tests for gradient flow through pitch head."""

    def test_gradient_exists(self):
        """Test that gradients flow through the head."""
        config = PitchConfig(encoder_dim=128)
        head = PitchHead(config)

        batch_size = 4
        seq_len = 20
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 128))
        f0_target = mx.random.uniform(low=100, high=400, shape=(batch_size, seq_len))
        voicing_target = (mx.random.uniform(shape=(batch_size, seq_len)) > 0.5).astype(mx.float32)

        def loss_fn(model, encoder_out, f0_target, voicing_target):
            f0_pred, voicing_pred = model(encoder_out)
            total, _, _ = pitch_loss(f0_pred, f0_target, voicing_pred, voicing_target)
            return total

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, encoder_out, f0_target, voicing_target,
        )

        # Check that loss is finite
        assert mx.isfinite(loss)

        # Check that gradients exist and are finite
        def check_grads(g):
            if isinstance(g, dict):
                for v in g.values():
                    check_grads(v)
            elif isinstance(g, list):
                for v in g:
                    check_grads(v)
            elif isinstance(g, mx.array):
                assert mx.all(mx.isfinite(g)), "Found non-finite gradient"

        check_grads(grads)

    def test_learning_occurs(self):
        """Test that the model can learn from training."""
        config = PitchConfig(encoder_dim=64, hidden_dim=32, num_layers=1)
        head = PitchHead(config)

        # Create training data with pattern
        batch_size = 8
        seq_len = 20
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 64))

        # Target: F0 correlates with mean of encoder output
        encoder_mean = mx.mean(encoder_out, axis=-1)
        f0_target = 200 + encoder_mean * 50  # Range roughly 150-250 Hz
        voicing_target = mx.ones((batch_size, seq_len))

        def loss_fn(model, encoder_out, f0_target, voicing_target):
            f0_pred, voicing_pred = model(encoder_out)
            total, _, _ = pitch_loss(f0_pred, f0_target, voicing_pred, voicing_target)
            return total

        # Initial loss
        initial_loss = loss_fn(head, encoder_out, f0_target, voicing_target)

        # Train
        optimizer = optim.Adam(learning_rate=0.01)
        for _ in range(100):
            loss, grads = nn.value_and_grad(head, loss_fn)(
                head, encoder_out, f0_target, voicing_target,
            )
            optimizer.update(head, grads)
            mx.eval(head.parameters())

        # Final loss
        final_loss = loss_fn(head, encoder_out, f0_target, voicing_target)

        # Loss should decrease
        assert float(final_loss) < float(initial_loss) * 0.5


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_frame(self):
        """Test with single frame input."""
        config = PitchConfig(encoder_dim=256)
        head = PitchHead(config)

        encoder_out = mx.random.normal(shape=(4, 1, 256))
        f0_hz, voicing_prob = head(encoder_out)

        assert f0_hz.shape == (4, 1)
        assert voicing_prob.shape == (4, 1)

    def test_large_batch(self):
        """Test with large batch size."""
        config = PitchConfig(encoder_dim=256)
        head = PitchHead(config)

        encoder_out = mx.random.normal(shape=(64, 100, 256))
        f0_hz, voicing_prob = head(encoder_out)

        assert f0_hz.shape == (64, 100)
        assert voicing_prob.shape == (64, 100)

    def test_different_encoder_dims(self):
        """Test with various encoder dimensions."""
        for encoder_dim in [128, 256, 384, 512]:
            config = PitchConfig(encoder_dim=encoder_dim)
            head = PitchHead(config)

            encoder_out = mx.random.normal(shape=(4, 50, encoder_dim))
            f0_hz, voicing_prob = head(encoder_out)

            assert f0_hz.shape == (4, 50)
            assert voicing_prob.shape == (4, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
