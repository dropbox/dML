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
Tests for singing detection and technique classification head.

Tests cover:
- Binary singing detection
- Multi-technique classification
- Single-technique classification
- Combined loss function
- Gradient flow for training
"""

import mlx.core as mx
import mlx.optimizers as optim

from src.models.heads.singing import (
    SINGING_TECHNIQUES,
    AttentionPooling,
    SingingConfig,
    SingingHead,
    SingingLoss,
    compute_singing_accuracy,
    compute_technique_accuracy,
    singing_loss,
)


class TestSingingConfig:
    """Tests for SingingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SingingConfig()
        assert config.encoder_dim == 384
        assert config.num_techniques == len(SINGING_TECHNIQUES)
        assert config.hidden_dim == 256
        assert config.num_attention_heads == 4
        assert config.dropout_rate == 0.1
        assert config.use_attention_pooling is True
        assert config.label_smoothing == 0.1
        assert config.singing_threshold == 0.5
        assert config.technique_threshold == 0.3
        assert config.multi_technique is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SingingConfig(
            encoder_dim=512,
            num_techniques=8,
            hidden_dim=128,
            multi_technique=False,
        )
        assert config.encoder_dim == 512
        assert config.num_techniques == 8
        assert config.hidden_dim == 128
        assert config.multi_technique is False

    def test_technique_names(self):
        """Test technique names are defined."""
        config = SingingConfig()
        assert len(config.technique_names) == len(SINGING_TECHNIQUES)
        assert "belt" in config.technique_names
        assert "falsetto" in config.technique_names
        assert "vibrato" in config.technique_names


class TestSingingConstants:
    """Tests for singing constants."""

    def test_technique_count(self):
        """Test that we have 10 techniques."""
        assert len(SINGING_TECHNIQUES) == 10

    def test_technique_names(self):
        """Test technique names."""
        techniques = ["belt", "falsetto", "head_voice", "chest_voice", "mixed_voice"]
        for tech in techniques:
            assert tech in SINGING_TECHNIQUES

        techniques2 = ["vibrato", "straight_tone", "breathy", "twang", "opera"]
        for tech in techniques2:
            assert tech in SINGING_TECHNIQUES


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


class TestSingingHead:
    """Tests for SingingHead module."""

    def test_forward_default_config(self):
        """Test forward pass with default config."""
        head = SingingHead()
        batch_size = 4
        seq_len = 100
        encoder_dim = 384

        x = mx.random.normal(shape=(batch_size, seq_len, encoder_dim))
        singing_logits, technique_logits = head(x)

        assert singing_logits.shape == (batch_size, 1)
        assert technique_logits.shape == (batch_size, head.config.num_techniques)

    def test_forward_custom_config(self):
        """Test forward pass with custom config."""
        config = SingingConfig(
            encoder_dim=256,
            num_techniques=8,
            hidden_dim=128,
        )
        head = SingingHead(config)
        batch_size = 2
        seq_len = 50

        x = mx.random.normal(shape=(batch_size, seq_len, 256))
        singing_logits, technique_logits = head(x)

        assert singing_logits.shape == (batch_size, 1)
        assert technique_logits.shape == (batch_size, 8)

    def test_forward_with_lengths(self):
        """Test forward pass with variable sequence lengths."""
        head = SingingHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        lengths = mx.array([100, 80, 60, 40])

        singing_logits, technique_logits = head(x, encoder_lengths=lengths)

        assert singing_logits.shape == (batch_size, 1)
        assert technique_logits.shape == (batch_size, head.config.num_techniques)

    def test_forward_mean_pooling(self):
        """Test forward pass with mean pooling."""
        config = SingingConfig(use_attention_pooling=False)
        head = SingingHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        singing_logits, technique_logits = head(x)

        assert singing_logits.shape == (batch_size, 1)
        assert technique_logits.shape == (batch_size, head.config.num_techniques)

    def test_predict_multi_technique(self):
        """Test prediction with multi-technique mode."""
        config = SingingConfig(multi_technique=True)
        head = SingingHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        is_singing, singing_prob, technique_preds, technique_probs = head.predict(x)

        assert is_singing.shape == (batch_size,)
        assert singing_prob.shape == (batch_size,)
        # Multi-technique returns binary mask
        assert technique_preds.shape == (batch_size, head.config.num_techniques)
        assert technique_probs.shape == (batch_size, head.config.num_techniques)
        # Singing prob should be sigmoid output
        assert mx.all(singing_prob >= 0) and mx.all(singing_prob <= 1)
        # Technique probs should be sigmoid outputs
        assert mx.all(technique_probs >= 0) and mx.all(technique_probs <= 1)

    def test_predict_single_technique(self):
        """Test prediction with single-technique mode."""
        config = SingingConfig(multi_technique=False)
        head = SingingHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        is_singing, singing_prob, technique_preds, technique_probs = head.predict(x)

        assert is_singing.shape == (batch_size,)
        assert singing_prob.shape == (batch_size,)
        # Single-technique returns class indices
        assert technique_preds.shape == (batch_size,)
        assert technique_probs.shape == (batch_size, head.config.num_techniques)
        # Probs should sum to 1 (softmax)
        prob_sums = mx.sum(technique_probs, axis=-1)
        assert mx.allclose(prob_sums, mx.ones(batch_size), atol=1e-5)

    def test_get_technique_name(self):
        """Test technique name retrieval."""
        head = SingingHead()
        name = head.get_technique_name(0)
        assert name == "belt"

    def test_decode_predictions_multi_technique(self):
        """Test decoding predictions in multi-technique mode."""
        config = SingingConfig(multi_technique=True)
        head = SingingHead(config)
        batch_size = 2

        is_singing = mx.array([1, 0])
        singing_prob = mx.array([0.9, 0.3])

        # Technique predictions
        technique_preds = mx.zeros((batch_size, config.num_techniques), dtype=mx.int32)
        technique_preds = technique_preds.at[0, 0].add(1)  # belt
        technique_preds = technique_preds.at[0, 5].add(1)  # vibrato

        technique_probs = mx.full((batch_size, config.num_techniques), 0.2)
        technique_probs = technique_probs.at[0, 0].add(0.85)
        technique_probs = technique_probs.at[0, 5].add(0.7)

        results = head.decode_predictions(
            is_singing, singing_prob, technique_preds, technique_probs,
        )

        assert len(results) == 2
        assert results[0]["is_singing"] is True
        assert results[0]["singing_confidence"] > 0.8
        assert len(results[0]["techniques"]) == 2
        assert results[1]["is_singing"] is False
        assert len(results[1]["techniques"]) == 0


class TestSingingLoss:
    """Tests for singing loss functions."""

    def test_multi_technique_loss(self):
        """Test loss for multi-technique mode."""
        batch_size = 4
        num_techniques = 10

        singing_logits = mx.random.normal(shape=(batch_size, 1))
        technique_logits = mx.random.normal(shape=(batch_size, num_techniques))

        # Targets
        singing_targets = mx.array([1, 1, 0, 0])
        technique_targets = mx.zeros((batch_size, num_techniques), dtype=mx.int32)
        technique_targets = technique_targets.at[0, 0].add(1)
        technique_targets = technique_targets.at[1, 5].add(1)

        total, sing_loss, tech_loss = singing_loss(
            singing_logits, technique_logits,
            singing_targets, technique_targets,
            multi_technique=True,
        )

        assert total.shape == ()
        assert sing_loss.shape == ()
        assert tech_loss.shape == ()
        assert total > 0
        assert sing_loss > 0
        assert tech_loss > 0

    def test_single_technique_loss(self):
        """Test loss for single-technique mode."""
        batch_size = 4
        num_techniques = 10

        singing_logits = mx.random.normal(shape=(batch_size, 1))
        technique_logits = mx.random.normal(shape=(batch_size, num_techniques))

        singing_targets = mx.array([1, 1, 0, 0])
        technique_targets = mx.array([0, 5, 2, 3])  # Class indices

        total, sing_loss, tech_loss = singing_loss(
            singing_logits, technique_logits,
            singing_targets, technique_targets,
            multi_technique=False,
        )

        assert total.shape == ()
        assert sing_loss.shape == ()
        assert tech_loss.shape == ()
        assert total > 0

    def test_technique_loss_masked_for_speech(self):
        """Test that technique loss is masked for non-singing samples."""
        batch_size = 4
        num_techniques = 10

        singing_logits = mx.random.normal(shape=(batch_size, 1))
        technique_logits = mx.random.normal(shape=(batch_size, num_techniques))

        # All speech (no singing)
        singing_targets = mx.zeros(batch_size, dtype=mx.int32)
        technique_targets = mx.array([0, 5, 2, 3])

        total, sing_loss, tech_loss = singing_loss(
            singing_logits, technique_logits,
            singing_targets, technique_targets,
            multi_technique=False,
        )

        # Technique loss should be very small (masked)
        assert tech_loss < 1e-5

    def test_loss_module_wrapper(self):
        """Test SingingLoss module."""
        loss_fn = SingingLoss(multi_technique=True)
        batch_size = 4
        num_techniques = 10

        singing_logits = mx.random.normal(shape=(batch_size, 1))
        technique_logits = mx.random.normal(shape=(batch_size, num_techniques))
        singing_targets = mx.array([1, 1, 0, 0])
        technique_targets = mx.zeros((batch_size, num_techniques), dtype=mx.int32)
        technique_targets = technique_targets.at[0, 0].add(1)

        total, sing_loss, tech_loss = loss_fn(
            singing_logits, technique_logits,
            singing_targets, technique_targets,
        )

        assert total.shape == ()
        assert total > 0


class TestSingingAccuracy:
    """Tests for accuracy computation."""

    def test_singing_accuracy_perfect(self):
        """Test 100% singing detection accuracy."""
        predictions = mx.array([1, 1, 0, 0])
        targets = mx.array([1, 1, 0, 0])

        acc = compute_singing_accuracy(predictions, targets)

        assert mx.allclose(acc, mx.array(1.0), atol=1e-5)

    def test_singing_accuracy_partial(self):
        """Test partial singing detection accuracy."""
        predictions = mx.array([1, 1, 0, 0])
        targets = mx.array([1, 0, 0, 1])  # 2/4 correct

        acc = compute_singing_accuracy(predictions, targets)

        assert mx.allclose(acc, mx.array(0.5), atol=1e-5)

    def test_technique_accuracy_multi(self):
        """Test technique F1 for multi-label."""
        predictions = mx.array([
            [1, 0, 1, 0],
            [0, 1, 0, 0],
        ], dtype=mx.int32)
        targets = mx.array([
            [1, 0, 1, 0],  # Perfect match
            [0, 1, 1, 0],  # One FN
        ], dtype=mx.int32)
        singing_mask = mx.array([1, 1])

        f1 = compute_technique_accuracy(
            predictions, targets, singing_mask, multi_technique=True,
        )

        assert f1.shape == ()
        assert f1 > 0 and f1 <= 1

    def test_technique_accuracy_single(self):
        """Test technique accuracy for single-label."""
        predictions = mx.array([0, 1, 2, 3])
        targets = mx.array([0, 1, 2, 2])  # 3/4 correct
        singing_mask = mx.array([1, 1, 1, 1])

        acc = compute_technique_accuracy(
            predictions, targets, singing_mask, multi_technique=False,
        )

        assert mx.allclose(acc, mx.array(0.75), atol=1e-5)

    def test_technique_accuracy_masked(self):
        """Test that accuracy is computed only for singing samples."""
        predictions = mx.array([0, 1, 2, 3])
        targets = mx.array([0, 0, 0, 0])  # All would be wrong
        singing_mask = mx.array([1, 0, 0, 0])  # Only first is singing

        acc = compute_technique_accuracy(
            predictions, targets, singing_mask, multi_technique=False,
        )

        # First sample is correct (0 == 0)
        assert mx.allclose(acc, mx.array(1.0), atol=1e-5)


class TestSingingGradients:
    """Tests for gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        head = SingingHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        singing_targets = mx.array([1, 1, 0, 0])
        technique_targets = mx.zeros((batch_size, head.config.num_techniques), dtype=mx.int32)
        technique_targets = technique_targets.at[0, 0].add(1)

        def loss_fn(model, x, singing_targets, technique_targets):
            singing_logits, technique_logits = model(x)
            total, _, _ = singing_loss(
                singing_logits, technique_logits,
                singing_targets, technique_targets,
                multi_technique=True,
            )
            return total

        loss, grads = mx.value_and_grad(loss_fn)(head, x, singing_targets, technique_targets)
        mx.eval(loss, grads)

        assert loss > 0
        assert "shared_layers" in grads
        assert "singing_classifier" in grads
        assert "technique_classifier" in grads

    def test_training_step(self):
        """Test a complete training step."""
        config = SingingConfig(multi_technique=False)
        head = SingingHead(config)
        optimizer = optim.Adam(learning_rate=1e-4)

        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        singing_targets = mx.array([1, 1, 0, 0])
        technique_targets = mx.array([0, 5, 2, 3])

        def loss_fn(model, x, singing_targets, technique_targets):
            singing_logits, technique_logits = model(x)
            total, _, _ = singing_loss(
                singing_logits, technique_logits,
                singing_targets, technique_targets,
                multi_technique=False,
            )
            return total

        # Get initial loss
        initial_loss = loss_fn(head, x, singing_targets, technique_targets)

        # Training step
        loss, grads = mx.value_and_grad(loss_fn)(head, x, singing_targets, technique_targets)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)

        # Get updated loss
        updated_loss = loss_fn(head, x, singing_targets, technique_targets)
        mx.eval(updated_loss)

        # Loss should decrease or stay similar
        assert updated_loss <= initial_loss * 1.1


class TestSingingIntegration:
    """Integration tests for singing head."""

    def test_speech_only_inference(self):
        """Test inference on speech-only input."""
        head = SingingHead()
        batch_size = 4
        seq_len = 100

        # Random input (likely classified as speech)
        x = mx.random.normal(shape=(batch_size, seq_len, 384))

        is_singing, singing_prob, technique_preds, technique_probs = head.predict(x)

        # Should have valid outputs regardless of classification
        assert is_singing.shape == (batch_size,)
        assert singing_prob.shape == (batch_size,)

    def test_batch_consistency(self):
        """Test that batch processing is consistent with individual processing."""
        head = SingingHead()

        # Process individually
        x1 = mx.random.normal(shape=(1, 100, 384))
        x2 = mx.random.normal(shape=(1, 100, 384))

        s1, p1 = head(x1)
        s2, p2 = head(x2)

        # Process as batch
        x_batch = mx.concatenate([x1, x2], axis=0)
        s_batch, p_batch = head(x_batch)

        # Should match
        assert mx.allclose(s1.squeeze(0), s_batch[0], atol=1e-5)
        assert mx.allclose(s2.squeeze(0), s_batch[1], atol=1e-5)
