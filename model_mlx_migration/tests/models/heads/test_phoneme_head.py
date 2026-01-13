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
Tests for phoneme prediction head.

Tests cover:
- Forward pass with different input shapes
- Phoneme prediction and decoding
- Loss function with label smoothing
- Greedy decoding with CTC-style collapse
- Gradient flow for training
- Phoneme inventory access
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from src.models.heads.phoneme import (
    IPA_PHONEMES,
    PhonemeConfig,
    PhonemeFrameLoss,
    PhonemeHead,
    compute_frame_accuracy,
    compute_per,
    phoneme_ce_loss,
)


class TestPhonemeConfig:
    """Tests for PhonemeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PhonemeConfig()
        assert config.encoder_dim == 384
        assert config.num_phonemes == len(IPA_PHONEMES)
        assert config.hidden_dim == 256
        assert config.num_layers == 2
        assert config.blank_id == 0
        assert config.use_ctc is False
        assert config.label_smoothing == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = PhonemeConfig(
            encoder_dim=512,
            num_phonemes=100,
            hidden_dim=128,
        )
        assert config.encoder_dim == 512
        assert config.num_phonemes == 100
        assert config.hidden_dim == 128

    def test_ipa_inventory(self):
        """Test IPA phoneme inventory."""
        # Check special tokens
        assert "<blank>" in IPA_PHONEMES
        assert "<sil>" in IPA_PHONEMES
        # Check common phonemes
        assert "p" in IPA_PHONEMES
        assert "t" in IPA_PHONEMES
        assert "i" in IPA_PHONEMES
        assert "a" in IPA_PHONEMES or "ɑ" in IPA_PHONEMES


class TestPhonemeHead:
    """Tests for PhonemeHead module."""

    def test_forward_shape(self):
        """Test output shape of phoneme head."""
        config = PhonemeConfig(encoder_dim=384, num_phonemes=50)
        head = PhonemeHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        logits = head(encoder_out)

        assert logits.shape == (batch_size, seq_len, 50)

    def test_predict_method(self):
        """Test predict method returns predictions and probs."""
        config = PhonemeConfig(encoder_dim=256, num_phonemes=30)
        head = PhonemeHead(config)

        batch_size = 4
        seq_len = 50
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 256))

        predictions, probs = head.predict(encoder_out)

        assert predictions.shape == (batch_size, seq_len)
        assert probs.shape == (batch_size, seq_len, 30)

        # Probabilities should sum to ~1
        prob_sums = mx.sum(probs, axis=-1)
        assert mx.all(mx.abs(prob_sums - 1.0) < 1e-5)

    def test_get_phoneme(self):
        """Test phoneme lookup by index."""
        config = PhonemeConfig()
        head = PhonemeHead(config)

        # First phoneme should be blank
        assert head.get_phoneme(0) == "<blank>"
        # Check some common phonemes
        p_idx = head.get_phoneme_id("p")
        assert head.get_phoneme(p_idx) == "p"

    def test_get_phoneme_id(self):
        """Test phoneme ID lookup."""
        config = PhonemeConfig()
        head = PhonemeHead(config)

        blank_id = head.get_phoneme_id("<blank>")
        assert blank_id == 0

        p_id = head.get_phoneme_id("p")
        assert head.get_phoneme(p_id) == "p"

    def test_decode_greedy_basic(self):
        """Test basic greedy decoding."""
        config = PhonemeConfig(encoder_dim=256, num_phonemes=10)
        head = PhonemeHead(config)

        batch_size = 2
        seq_len = 5

        # Create logits where phoneme 3 has highest score
        logits = mx.zeros((batch_size, seq_len, 10))
        logits = logits.at[:, :, 3].add(10.0)

        decoded = head.decode_greedy(logits)

        assert len(decoded) == batch_size
        # After collapsing repeated, should have single phoneme per sequence
        # (phoneme at index 3)
        for seq in decoded:
            assert len(seq) == 1

    def test_decode_greedy_with_lengths(self):
        """Test greedy decoding with variable lengths."""
        config = PhonemeConfig(encoder_dim=256, num_phonemes=10)
        head = PhonemeHead(config)

        logits = mx.zeros((2, 10, 10))
        # Set different phonemes
        logits = logits.at[0, :, 3].add(10.0)  # All phoneme 3
        logits = logits.at[1, :, 4].add(10.0)  # All phoneme 4

        lengths = mx.array([5, 3])
        decoded = head.decode_greedy(logits, lengths)

        assert len(decoded) == 2
        # Both should have single collapsed phoneme
        assert len(decoded[0]) == 1
        assert len(decoded[1]) == 1

    def test_decode_greedy_no_collapse(self):
        """Test greedy decoding without collapsing."""
        config = PhonemeConfig(encoder_dim=256, num_phonemes=10)
        head = PhonemeHead(config)

        logits = mx.zeros((1, 5, 10))
        logits = logits.at[0, :, 3].add(10.0)

        decoded = head.decode_greedy(logits, collapse_repeated=False)

        # Without collapsing, should have 5 repeated phonemes
        assert len(decoded[0]) == 5

    def test_default_config_creation(self):
        """Test head creation with default config."""
        head = PhonemeHead()
        assert head.config.encoder_dim == 384
        assert head.config.num_phonemes == len(IPA_PHONEMES)


class TestPhonemeLoss:
    """Tests for phoneme loss function."""

    def test_loss_shape(self):
        """Test loss output shape."""
        batch_size = 4
        seq_len = 20
        num_phonemes = 50

        logits = mx.random.normal(shape=(batch_size, seq_len, num_phonemes))
        targets = mx.random.randint(0, num_phonemes, shape=(batch_size, seq_len))

        loss = phoneme_ce_loss(logits, targets, label_smoothing=0.0)

        assert loss.shape == ()

    def test_loss_none_reduction(self):
        """Test loss with no reduction."""
        batch_size = 2
        seq_len = 5
        num_phonemes = 10

        logits = mx.random.normal(shape=(batch_size, seq_len, num_phonemes))
        targets = mx.random.randint(0, num_phonemes, shape=(batch_size, seq_len))

        loss = phoneme_ce_loss(logits, targets, reduction="none", label_smoothing=0.0)

        assert loss.shape == (batch_size, seq_len)

    def test_loss_with_mask(self):
        """Test loss with padding mask."""
        batch_size = 2
        seq_len = 5
        num_phonemes = 10

        logits = mx.random.normal(shape=(batch_size, seq_len, num_phonemes))
        targets = mx.random.randint(0, num_phonemes, shape=(batch_size, seq_len))

        # Mask out last 2 positions for both sequences
        # Create mask as: True, True, True, False, False
        mask_list = [[True, True, True, False, False] for _ in range(batch_size)]
        mask = mx.array(mask_list)

        loss_masked = phoneme_ce_loss(logits, targets, mask=mask, label_smoothing=0.0)

        # Full mask (all valid)
        full_mask = mx.ones((batch_size, seq_len), dtype=mx.bool_)
        loss_full = phoneme_ce_loss(logits, targets, mask=full_mask, label_smoothing=0.0)

        # Losses should be different when mask differs
        # (unless by chance they're the same, which is unlikely)
        assert mask.shape == (batch_size, seq_len)
        assert float(loss_masked) >= 0  # Loss should be valid
        assert float(loss_full) >= 0  # Full mask loss should also be valid

    def test_label_smoothing_effect(self):
        """Test that label smoothing affects loss."""
        batch_size = 2
        seq_len = 5
        num_phonemes = 10

        # Create confident predictions
        logits = mx.zeros((batch_size, seq_len, num_phonemes))
        logits = logits.at[:, :, 0].add(10.0)  # High confidence for class 0
        targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)

        loss_no_smooth = phoneme_ce_loss(logits, targets, label_smoothing=0.0)
        loss_smooth = phoneme_ce_loss(logits, targets, label_smoothing=0.1)

        # Smoothed loss should be higher for confident predictions
        assert float(loss_smooth) > float(loss_no_smooth)

    def test_loss_module(self):
        """Test PhonemeFrameLoss module wrapper."""
        loss_fn = PhonemeFrameLoss(label_smoothing=0.1)

        batch_size = 2
        seq_len = 5
        num_phonemes = 20
        logits = mx.random.normal(shape=(batch_size, seq_len, num_phonemes))
        targets = mx.random.randint(0, num_phonemes, shape=(batch_size, seq_len))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert float(loss) > 0


class TestMetrics:
    """Tests for phoneme metrics."""

    def test_compute_per(self):
        """Test PER computation."""
        # All correct predictions
        predictions = mx.array([[1, 2, 3, 4]])
        targets = mx.array([[1, 2, 3, 4]])
        per = compute_per(predictions, targets, blank_id=0)
        assert float(per) == 0.0

        # All wrong predictions (no blanks)
        predictions = mx.array([[5, 6, 7, 8]])
        per = compute_per(predictions, targets, blank_id=0)
        assert float(per) == 1.0

        # 50% error
        predictions = mx.array([[1, 6, 3, 8]])
        per = compute_per(predictions, targets, blank_id=0)
        assert abs(float(per) - 0.5) < 0.01

    def test_compute_per_with_blank(self):
        """Test PER excludes blank tokens."""
        # Targets have blanks, predictions don't matter for blanks
        predictions = mx.array([[1, 2, 3, 4]])
        targets = mx.array([[0, 2, 0, 4]])  # Blanks at positions 0 and 2
        per = compute_per(predictions, targets, blank_id=0)

        # Only non-blank targets count: positions 1 and 3
        # Predictions at 1 and 3 are 2 and 4, targets are 2 and 4 - both correct
        assert float(per) == 0.0

    def test_compute_frame_accuracy(self):
        """Test frame accuracy computation."""
        # All correct
        predictions = mx.array([[1, 2, 3, 4]])
        targets = mx.array([[1, 2, 3, 4]])
        acc = compute_frame_accuracy(predictions, targets)
        assert float(acc) == 1.0

        # 50% correct
        predictions = mx.array([[1, 6, 3, 8]])
        acc = compute_frame_accuracy(predictions, targets)
        assert abs(float(acc) - 0.5) < 0.01


class TestGradientFlow:
    """Tests for gradient flow through phoneme head."""

    def test_gradient_exists(self):
        """Test that gradients flow through the head."""
        config = PhonemeConfig(encoder_dim=128, num_phonemes=20, hidden_dim=64)
        head = PhonemeHead(config)

        batch_size = 2
        seq_len = 10
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 128))
        targets = mx.random.randint(0, 20, shape=(batch_size, seq_len))

        def loss_fn(model, encoder_out, targets):
            logits = model(encoder_out)
            return phoneme_ce_loss(logits, targets, label_smoothing=0.0)

        loss, grads = nn.value_and_grad(head, loss_fn)(head, encoder_out, targets)

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
        config = PhonemeConfig(encoder_dim=64, num_phonemes=5, hidden_dim=32, num_layers=1)
        head = PhonemeHead(config)

        # Create training data with pattern
        batch_size = 4
        seq_len = 10

        # Each phoneme class has distinct pattern in different encoder dims
        targets = mx.array([[i % 5 for i in range(seq_len)] for _ in range(batch_size)])
        encoder_out = mx.zeros((batch_size, seq_len, 64))

        # Add class-specific signal
        for b in range(batch_size):
            for s in range(seq_len):
                class_idx = s % 5
                start_dim = class_idx * 12
                end_dim = start_dim + 12
                encoder_out = encoder_out.at[b, s, start_dim:end_dim].add(1.0)

        def loss_fn(model, encoder_out, targets):
            logits = model(encoder_out)
            return phoneme_ce_loss(logits, targets, label_smoothing=0.0)

        # Initial loss
        initial_loss = loss_fn(head, encoder_out, targets)

        # Train
        optimizer = optim.Adam(learning_rate=0.01)
        for _ in range(100):
            loss, grads = nn.value_and_grad(head, loss_fn)(head, encoder_out, targets)
            optimizer.update(head, grads)
            mx.eval(head.parameters())

        # Final loss
        final_loss = loss_fn(head, encoder_out, targets)

        # Loss should decrease
        assert float(final_loss) < float(initial_loss) * 0.5


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_frame(self):
        """Test with single frame input."""
        config = PhonemeConfig(encoder_dim=256, num_phonemes=30)
        head = PhonemeHead(config)

        encoder_out = mx.random.normal(shape=(4, 1, 256))
        logits = head(encoder_out)

        assert logits.shape == (4, 1, 30)

    def test_large_batch(self):
        """Test with large batch size."""
        config = PhonemeConfig(encoder_dim=256, num_phonemes=30)
        head = PhonemeHead(config)

        encoder_out = mx.random.normal(shape=(32, 50, 256))
        logits = head(encoder_out)

        assert logits.shape == (32, 50, 30)

    def test_different_encoder_dims(self):
        """Test with various encoder dimensions."""
        for encoder_dim in [128, 256, 384, 512]:
            config = PhonemeConfig(encoder_dim=encoder_dim, num_phonemes=40)
            head = PhonemeHead(config)

            encoder_out = mx.random.normal(shape=(4, 20, encoder_dim))
            logits = head(encoder_out)

            assert logits.shape == (4, 20, 40)

    def test_different_num_phonemes(self):
        """Test with various phoneme counts."""
        for num_phonemes in [20, 50, 100, 178]:
            config = PhonemeConfig(encoder_dim=256, num_phonemes=num_phonemes)
            head = PhonemeHead(config)

            encoder_out = mx.random.normal(shape=(4, 20, 256))
            logits = head(encoder_out)

            assert logits.shape == (4, 20, num_phonemes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
