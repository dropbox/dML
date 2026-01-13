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
Tests for paralinguistics classification head.

Tests cover:
- Forward pass with different input shapes
- Multi-label classification
- Single-label classification
- Loss function (BCE and CE)
- Gradient flow for training
- Batch processing with variable lengths
"""

import mlx.core as mx
import mlx.optimizers as optim

from src.models.heads.paralinguistics import (
    PARALINGUISTIC_CLASSES,
    AttentionPooling,
    ParalinguisticsConfig,
    ParalinguisticsHead,
    ParalinguisticsLoss,
    compute_paralinguistics_accuracy,
    paralinguistics_loss,
)


class TestParalinguisticsConfig:
    """Tests for ParalinguisticsConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParalinguisticsConfig()
        assert config.encoder_dim == 384
        assert config.num_classes == len(PARALINGUISTIC_CLASSES)
        assert config.hidden_dim == 256
        assert config.num_attention_heads == 4
        assert config.dropout_rate == 0.1
        assert config.use_attention_pooling is True
        assert config.label_smoothing == 0.1
        assert config.multi_label is True
        assert config.detection_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = ParalinguisticsConfig(
            encoder_dim=512,
            num_classes=30,
            hidden_dim=128,
            multi_label=False,
        )
        assert config.encoder_dim == 512
        assert config.num_classes == 30
        assert config.hidden_dim == 128
        assert config.multi_label is False

    def test_class_names(self):
        """Test paralinguistic class names."""
        config = ParalinguisticsConfig()
        assert "laughter" in config.class_names
        assert "cough" in config.class_names
        assert "um" in config.class_names
        assert "<silence>" in config.class_names


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
        # Mask last 20 positions for each batch
        mask = mask.at[:, 80:].add(True)

        out = pooling(x, mask=mask)

        assert out.shape == (batch_size, embed_dim)


class TestParalinguisticsHead:
    """Tests for ParalinguisticsHead module."""

    def test_forward_default_config(self):
        """Test forward pass with default config."""
        head = ParalinguisticsHead()
        batch_size = 4
        seq_len = 100
        encoder_dim = 384

        x = mx.random.normal(shape=(batch_size, seq_len, encoder_dim))
        logits = head(x)

        assert logits.shape == (batch_size, head.config.num_classes)

    def test_forward_custom_config(self):
        """Test forward pass with custom config."""
        config = ParalinguisticsConfig(
            encoder_dim=256,
            num_classes=20,
            hidden_dim=128,
        )
        head = ParalinguisticsHead(config)
        batch_size = 2
        seq_len = 50

        x = mx.random.normal(shape=(batch_size, seq_len, 256))
        logits = head(x)

        assert logits.shape == (batch_size, 20)

    def test_forward_with_lengths(self):
        """Test forward pass with variable sequence lengths."""
        head = ParalinguisticsHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        lengths = mx.array([100, 80, 60, 40])

        logits = head(x, encoder_lengths=lengths)

        assert logits.shape == (batch_size, head.config.num_classes)

    def test_forward_mean_pooling(self):
        """Test forward pass with mean pooling (no attention)."""
        config = ParalinguisticsConfig(use_attention_pooling=False)
        head = ParalinguisticsHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        logits = head(x)

        assert logits.shape == (batch_size, head.config.num_classes)

    def test_predict_multi_label(self):
        """Test prediction with multi-label mode."""
        config = ParalinguisticsConfig(multi_label=True)
        head = ParalinguisticsHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        predictions, probs = head.predict(x)

        # Multi-label returns binary mask
        assert predictions.shape == (batch_size, head.config.num_classes)
        assert probs.shape == (batch_size, head.config.num_classes)
        # Probs should be sigmoid outputs (0-1)
        assert mx.all(probs >= 0) and mx.all(probs <= 1)

    def test_predict_single_label(self):
        """Test prediction with single-label mode."""
        config = ParalinguisticsConfig(multi_label=False)
        head = ParalinguisticsHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        predictions, probs = head.predict(x)

        # Single-label returns class indices
        assert predictions.shape == (batch_size,)
        assert probs.shape == (batch_size, head.config.num_classes)
        # Probs should sum to 1 (softmax)
        prob_sums = mx.sum(probs, axis=-1)
        assert mx.allclose(prob_sums, mx.ones(batch_size), atol=1e-5)

    def test_get_class_name(self):
        """Test class name retrieval."""
        head = ParalinguisticsHead()
        name = head.get_class_name(0)
        assert name == "laughter"

    def test_decode_predictions_multi_label(self):
        """Test decoding predictions in multi-label mode."""
        config = ParalinguisticsConfig(multi_label=True)
        head = ParalinguisticsHead(config)

        # Create mock predictions
        predictions = mx.zeros((2, config.num_classes), dtype=mx.int32)
        predictions = predictions.at[0, 0].add(1)  # laughter
        predictions = predictions.at[0, 2].add(1)  # cough
        predictions = predictions.at[1, 5].add(1)  # sneeze

        probs = mx.full((2, config.num_classes), 0.3)
        probs = probs.at[0, 0].add(0.9)
        probs = probs.at[0, 2].add(0.7)
        probs = probs.at[1, 5].add(0.8)

        results = head.decode_predictions(predictions, probs)

        assert len(results) == 2
        assert len(results[0]) == 2  # laughter, cough
        assert results[0][0][0] == "laughter"
        assert results[0][0][1] > 0.8


class TestParalinguisticsLoss:
    """Tests for paralinguistics loss functions."""

    def test_multi_label_loss(self):
        """Test BCE loss for multi-label classification."""
        batch_size = 4
        num_classes = 50

        logits = mx.random.normal(shape=(batch_size, num_classes))
        # Multi-label targets (binary mask)
        targets = mx.zeros((batch_size, num_classes), dtype=mx.int32)
        targets = targets.at[0, 0].add(1)
        targets = targets.at[0, 5].add(1)
        targets = targets.at[1, 10].add(1)

        loss = paralinguistics_loss(logits, targets, multi_label=True)

        assert loss.shape == ()
        assert loss > 0

    def test_single_label_loss(self):
        """Test CE loss for single-label classification."""
        batch_size = 4
        num_classes = 50

        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([0, 5, 10, 15])

        loss = paralinguistics_loss(
            logits, targets, multi_label=False, label_smoothing=0.1,
        )

        assert loss.shape == ()
        assert loss > 0

    def test_loss_module_wrapper(self):
        """Test ParalinguisticsLoss module."""
        loss_fn = ParalinguisticsLoss(multi_label=True)
        batch_size = 4
        num_classes = 50

        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.zeros((batch_size, num_classes), dtype=mx.int32)
        targets = targets.at[0, 0].add(1)

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss > 0

    def test_loss_reduction_options(self):
        """Test different reduction options."""
        batch_size = 4
        num_classes = 50

        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([0, 5, 10, 15])

        loss_mean = paralinguistics_loss(
            logits, targets, multi_label=False, reduction="mean",
        )
        loss_sum = paralinguistics_loss(
            logits, targets, multi_label=False, reduction="sum",
        )
        loss_none = paralinguistics_loss(
            logits, targets, multi_label=False, reduction="none",
        )

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (batch_size,)


class TestParalinguisticsAccuracy:
    """Tests for accuracy computation."""

    def test_multi_label_f1(self):
        """Test F1 score for multi-label."""
        predictions = mx.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=mx.int32)
        targets = mx.array([
            [1, 0, 1, 0],  # Perfect match
            [0, 1, 1, 0],  # One FP, one FN
        ], dtype=mx.int32)

        f1 = compute_paralinguistics_accuracy(
            predictions, targets, multi_label=True,
        )

        assert f1.shape == ()
        assert f1 > 0 and f1 <= 1

    def test_single_label_accuracy(self):
        """Test accuracy for single-label."""
        predictions = mx.array([0, 1, 2, 3])
        targets = mx.array([0, 1, 2, 2])  # 3/4 correct

        acc = compute_paralinguistics_accuracy(
            predictions, targets, multi_label=False,
        )

        assert mx.allclose(acc, mx.array(0.75), atol=1e-5)


class TestParalinguisticsGradients:
    """Tests for gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        head = ParalinguisticsHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        targets = mx.zeros((batch_size, head.config.num_classes), dtype=mx.int32)
        targets = targets.at[0, 0].add(1)

        def loss_fn(model, x, targets):
            logits = model(x)
            return paralinguistics_loss(logits, targets, multi_label=True)

        loss, grads = mx.value_and_grad(loss_fn)(head, x, targets)
        mx.eval(loss, grads)

        assert loss > 0
        # Check that gradients exist for classifier
        assert "classifier" in grads

    def test_training_step(self):
        """Test a complete training step."""
        config = ParalinguisticsConfig(multi_label=False)
        head = ParalinguisticsHead(config)
        optimizer = optim.Adam(learning_rate=1e-4)

        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        targets = mx.array([0, 5, 10, 15])

        def loss_fn(model, x, targets):
            logits = model(x)
            return paralinguistics_loss(logits, targets, multi_label=False)

        # Get initial loss
        initial_loss = loss_fn(head, x, targets)

        # Training step
        loss, grads = mx.value_and_grad(loss_fn)(head, x, targets)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)

        # Get updated loss
        updated_loss = loss_fn(head, x, targets)
        mx.eval(updated_loss)

        # Loss should decrease or stay similar
        assert updated_loss <= initial_loss * 1.1  # Allow small increase due to stochasticity


class TestParalinguisticsConstants:
    """Tests for constants and class lists."""

    def test_class_count(self):
        """Test that we have 50 classes."""
        assert len(PARALINGUISTIC_CLASSES) == 50

    def test_special_tokens(self):
        """Test special tokens are present."""
        assert "<silence>" in PARALINGUISTIC_CLASSES
        assert "<speech>" in PARALINGUISTIC_CLASSES
        assert "<noise>" in PARALINGUISTIC_CLASSES
        assert "<unknown>" in PARALINGUISTIC_CLASSES

    def test_vocalsound_classes(self):
        """Test VocalSound core classes."""
        vocalsound = ["laughter", "sigh", "cough", "throat_clear", "sneeze"]
        for cls in vocalsound:
            assert cls in PARALINGUISTIC_CLASSES

    def test_filler_classes(self):
        """Test filler word classes."""
        fillers = ["um", "uh", "hmm", "ah", "er"]
        for cls in fillers:
            assert cls in PARALINGUISTIC_CLASSES
