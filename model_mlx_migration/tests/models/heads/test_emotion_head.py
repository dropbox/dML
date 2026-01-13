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
Tests for emotion classification head.

Tests cover:
- Forward pass with different input shapes
- Attention pooling mechanism
- Loss function with label smoothing
- Gradient flow for training
- Batch processing with variable lengths
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from src.models.heads.emotion import (
    AttentionPooling,
    EmotionConfig,
    EmotionHead,
    EmotionLoss,
    emotion_loss,
)


class TestEmotionConfig:
    """Tests for EmotionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmotionConfig()
        assert config.encoder_dim == 384
        assert config.num_classes == 8
        assert config.hidden_dim == 256
        assert config.num_attention_heads == 4
        assert config.dropout_rate == 0.1
        assert config.use_attention_pooling is True
        assert config.label_smoothing == 0.1
        assert len(config.class_names) == 8

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmotionConfig(
            encoder_dim=512,
            num_classes=6,
            hidden_dim=128,
        )
        assert config.encoder_dim == 512
        assert config.num_classes == 6
        assert config.hidden_dim == 128

    def test_class_names(self):
        """Test emotion class names."""
        config = EmotionConfig()
        assert "neutral" in config.class_names
        assert "happy" in config.class_names
        assert "angry" in config.class_names
        assert "sad" in config.class_names


class TestAttentionPooling:
    """Tests for AttentionPooling module."""

    def test_forward_shape(self):
        """Test output shape of attention pooling."""
        embed_dim = 384
        batch_size = 4
        seq_len = 100

        pooling = AttentionPooling(embed_dim, num_heads=4)
        x = mx.random.normal(shape=(batch_size, seq_len, embed_dim))

        out = pooling(x)

        assert out.shape == (batch_size, embed_dim)

    def test_forward_with_mask(self):
        """Test attention pooling with padding mask."""
        embed_dim = 384
        batch_size = 4
        seq_len = 100

        pooling = AttentionPooling(embed_dim, num_heads=4)
        x = mx.random.normal(shape=(batch_size, seq_len, embed_dim))

        # Create mask (True = masked)
        mask = mx.zeros((batch_size, seq_len), dtype=mx.bool_)
        # Mask last 20 positions for all samples
        mask = mask.at[:, 80:].add(True)

        out = pooling(x, mask=mask)

        assert out.shape == (batch_size, embed_dim)

    def test_different_seq_lengths(self):
        """Test with different sequence lengths."""
        embed_dim = 384
        batch_size = 4

        pooling = AttentionPooling(embed_dim, num_heads=4)

        for seq_len in [10, 50, 100, 200]:
            x = mx.random.normal(shape=(batch_size, seq_len, embed_dim))
            out = pooling(x)
            assert out.shape == (batch_size, embed_dim)

    def test_single_sequence(self):
        """Test with batch size 1."""
        embed_dim = 256
        seq_len = 50

        pooling = AttentionPooling(embed_dim, num_heads=4)
        x = mx.random.normal(shape=(1, seq_len, embed_dim))

        out = pooling(x)

        assert out.shape == (1, embed_dim)


class TestEmotionHead:
    """Tests for EmotionHead module."""

    def test_forward_shape(self):
        """Test output shape of emotion head."""
        config = EmotionConfig(encoder_dim=384, num_classes=8)
        head = EmotionHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        logits = head(encoder_out)

        assert logits.shape == (batch_size, 8)

    def test_forward_with_transpose(self):
        """Test with encoder output transposed from (seq, batch, dim) format."""
        config = EmotionConfig(encoder_dim=384, num_classes=8)
        head = EmotionHead(config)

        batch_size = 4
        seq_len = 100
        # (seq, batch, dim) format from encoder
        encoder_out_seq_batch = mx.random.normal(shape=(seq_len, batch_size, 384))
        # Transpose to (batch, seq, dim) as required
        encoder_out = mx.transpose(encoder_out_seq_batch, (1, 0, 2))

        logits = head(encoder_out)

        assert logits.shape == (batch_size, 8)

    def test_forward_with_lengths(self):
        """Test with variable sequence lengths."""
        config = EmotionConfig(encoder_dim=384, num_classes=8)
        head = EmotionHead(config)

        batch_size = 4
        max_seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, max_seq_len, 384))
        encoder_lengths = mx.array([80, 100, 60, 90])

        logits = head(encoder_out, encoder_lengths)

        assert logits.shape == (batch_size, 8)

    def test_mean_pooling_mode(self):
        """Test with mean pooling instead of attention."""
        config = EmotionConfig(
            encoder_dim=384,
            num_classes=8,
            use_attention_pooling=False,
        )
        head = EmotionHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        logits = head(encoder_out)

        assert logits.shape == (batch_size, 8)
        assert head.pooling is None

    def test_predict_method(self):
        """Test predict method returns classes and probs."""
        config = EmotionConfig(encoder_dim=384, num_classes=8)
        head = EmotionHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        predictions, probs = head.predict(encoder_out)

        assert predictions.shape == (batch_size,)
        assert probs.shape == (batch_size, 8)
        # Probabilities should sum to ~1
        prob_sums = mx.sum(probs, axis=-1)
        assert mx.all(mx.abs(prob_sums - 1.0) < 1e-5)

    def test_get_class_name(self):
        """Test class name lookup."""
        config = EmotionConfig()
        head = EmotionHead(config)

        assert head.get_class_name(0) == "neutral"
        assert head.get_class_name(1) == "happy"
        assert head.get_class_name(3) == "angry"

    def test_default_config(self):
        """Test head creation with default config."""
        head = EmotionHead()
        assert head.config.encoder_dim == 384
        assert head.config.num_classes == 8


class TestEmotionLoss:
    """Tests for emotion loss function."""

    def test_loss_shape(self):
        """Test loss output shape."""
        batch_size = 8
        num_classes = 8

        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([0, 1, 2, 3, 4, 5, 6, 7])

        loss = emotion_loss(logits, targets, label_smoothing=0.0)

        assert loss.shape == ()  # Scalar

    def test_loss_none_reduction(self):
        """Test loss with no reduction."""
        batch_size = 8
        num_classes = 8

        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([0, 1, 2, 3, 4, 5, 6, 7])

        loss = emotion_loss(logits, targets, reduction="none")

        assert loss.shape == (batch_size,)

    def test_loss_sum_reduction(self):
        """Test loss with sum reduction."""
        batch_size = 8
        num_classes = 8

        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([0, 1, 2, 3, 4, 5, 6, 7])

        loss = emotion_loss(logits, targets, reduction="sum")

        assert loss.shape == ()

    def test_label_smoothing_effect(self):
        """Test that label smoothing reduces confidence."""
        batch_size = 4
        num_classes = 8

        # Create confident predictions
        logits = mx.zeros((batch_size, num_classes))
        # Set high logit for correct class
        logits = logits.at[:, 0].add(10.0)
        targets = mx.zeros(batch_size, dtype=mx.int32)

        loss_no_smooth = emotion_loss(logits, targets, label_smoothing=0.0)
        loss_smooth = emotion_loss(logits, targets, label_smoothing=0.1)

        # Loss with smoothing should be higher for confident predictions
        assert float(loss_smooth) > float(loss_no_smooth)

    def test_loss_module(self):
        """Test EmotionLoss module wrapper."""
        loss_fn = EmotionLoss(label_smoothing=0.1)

        batch_size = 8
        num_classes = 8
        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([0, 1, 2, 3, 4, 5, 6, 7])

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert float(loss) > 0

    def test_loss_decreases_with_correct_predictions(self):
        """Test that loss is lower for correct predictions."""
        batch_size = 4
        num_classes = 8

        # Correct predictions: high logit for target class
        logits_correct = mx.zeros((batch_size, num_classes))
        logits_correct = logits_correct.at[0, 0].add(5.0)
        logits_correct = logits_correct.at[1, 1].add(5.0)
        logits_correct = logits_correct.at[2, 2].add(5.0)
        logits_correct = logits_correct.at[3, 3].add(5.0)
        targets = mx.array([0, 1, 2, 3])

        # Wrong predictions: high logit for wrong class
        logits_wrong = mx.zeros((batch_size, num_classes))
        logits_wrong = logits_wrong.at[0, 7].add(5.0)
        logits_wrong = logits_wrong.at[1, 6].add(5.0)
        logits_wrong = logits_wrong.at[2, 5].add(5.0)
        logits_wrong = logits_wrong.at[3, 4].add(5.0)

        loss_correct = emotion_loss(logits_correct, targets, label_smoothing=0.0)
        loss_wrong = emotion_loss(logits_wrong, targets, label_smoothing=0.0)

        assert float(loss_correct) < float(loss_wrong)


class TestGradientFlow:
    """Tests for gradient flow through emotion head."""

    def test_gradient_exists(self):
        """Test that gradients flow through the head."""
        config = EmotionConfig(encoder_dim=256, num_classes=8)
        head = EmotionHead(config)

        batch_size = 4
        seq_len = 50
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 256))
        targets = mx.array([0, 1, 2, 3])

        def loss_fn(model, encoder_out, targets):
            logits = model(encoder_out)
            return emotion_loss(logits, targets)

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
        """Test that the model can learn from training (overfit on one batch)."""
        config = EmotionConfig(encoder_dim=128, num_classes=4, hidden_dim=64)
        head = EmotionHead(config)

        # Create simple training data with class-specific patterns
        batch_size = 8
        seq_len = 20
        mx.random.seed(42)

        # Each class has a distinct pattern in different encoder dimensions
        targets = mx.array([0, 1, 2, 3, 0, 1, 2, 3])
        encoder_out = mx.zeros((batch_size, seq_len, 128))
        # Class 0: high values in first 32 dims
        encoder_out = encoder_out.at[0, :, :32].add(1.0)
        encoder_out = encoder_out.at[4, :, :32].add(1.0)
        # Class 1: high values in dims 32-64
        encoder_out = encoder_out.at[1, :, 32:64].add(1.0)
        encoder_out = encoder_out.at[5, :, 32:64].add(1.0)
        # Class 2: high values in dims 64-96
        encoder_out = encoder_out.at[2, :, 64:96].add(1.0)
        encoder_out = encoder_out.at[6, :, 64:96].add(1.0)
        # Class 3: high values in dims 96-128
        encoder_out = encoder_out.at[3, :, 96:128].add(1.0)
        encoder_out = encoder_out.at[7, :, 96:128].add(1.0)

        def loss_fn(model, encoder_out, targets):
            logits = model(encoder_out)
            return emotion_loss(logits, targets, label_smoothing=0.0)

        # Initial loss
        initial_loss = loss_fn(head, encoder_out, targets)

        # Train for multiple steps on the same batch (overfit)
        optimizer = optim.Adam(learning_rate=0.001)

        for _ in range(100):
            loss, grads = nn.value_and_grad(head, loss_fn)(head, encoder_out, targets)
            optimizer.update(head, grads)
            mx.eval(head.parameters())

        # Final loss
        final_loss = loss_fn(head, encoder_out, targets)

        # Loss should decrease significantly (should reach near 0 on this simple task)
        assert float(final_loss) < float(initial_loss) * 0.1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_frame_sequence(self):
        """Test with single frame input."""
        config = EmotionConfig(encoder_dim=256, num_classes=8)
        head = EmotionHead(config)

        batch_size = 4
        seq_len = 1
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 256))

        logits = head(encoder_out)

        assert logits.shape == (batch_size, 8)

    def test_large_batch(self):
        """Test with large batch size."""
        config = EmotionConfig(encoder_dim=256, num_classes=8)
        head = EmotionHead(config)

        batch_size = 64
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 256))

        logits = head(encoder_out)

        assert logits.shape == (batch_size, 8)

    def test_all_masked_sequence(self):
        """Test behavior when entire sequence is masked."""
        config = EmotionConfig(encoder_dim=256, num_classes=8)
        head = EmotionHead(config)

        batch_size = 2
        seq_len = 10
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 256))
        # All frames masked for first sample, none for second
        encoder_lengths = mx.array([0, 10])

        logits = head(encoder_out, encoder_lengths)

        assert logits.shape == (batch_size, 8)
        # Output should still be finite
        assert mx.all(mx.isfinite(logits))

    def test_different_encoder_dims(self):
        """Test with various encoder dimensions."""
        for encoder_dim in [128, 256, 384, 512]:
            config = EmotionConfig(encoder_dim=encoder_dim, num_classes=8)
            head = EmotionHead(config)

            encoder_out = mx.random.normal(shape=(4, 50, encoder_dim))
            logits = head(encoder_out)

            assert logits.shape == (4, 8)

    def test_different_num_classes(self):
        """Test with various number of classes."""
        for num_classes in [4, 6, 8, 10]:
            config = EmotionConfig(encoder_dim=256, num_classes=num_classes)
            head = EmotionHead(config)

            encoder_out = mx.random.normal(shape=(4, 50, 256))
            logits = head(encoder_out)

            assert logits.shape == (4, num_classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
