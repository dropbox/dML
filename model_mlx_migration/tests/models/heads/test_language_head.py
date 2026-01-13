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
Tests for language identification head.

Tests cover:
- Forward pass with different input shapes
- Core and extended language sets
- Loss function with label smoothing
- Gradient flow for training
- Language code and name mapping
"""

import mlx.core as mx
import mlx.optimizers as optim

from src.models.heads.language import (
    CORE_LANGUAGES,
    EXTENDED_LANGUAGES,
    LANGUAGE_NAMES,
    AttentionPooling,
    LanguageConfig,
    LanguageHead,
    LanguageLoss,
    compute_language_accuracy,
    language_loss,
)


class TestLanguageConfig:
    """Tests for LanguageConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LanguageConfig()
        assert config.encoder_dim == 384
        assert config.num_languages == len(CORE_LANGUAGES)
        assert config.hidden_dim == 256
        assert config.num_attention_heads == 4
        assert config.dropout_rate == 0.1
        assert config.use_attention_pooling is True
        assert config.label_smoothing == 0.1
        assert config.confidence_threshold == 0.7
        assert config.min_duration_sec == 0.5
        assert config.use_extended is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = LanguageConfig(
            encoder_dim=512,
            num_languages=20,
            hidden_dim=128,
            use_extended=True,
        )
        assert config.encoder_dim == 512
        assert config.num_languages == 20
        assert config.hidden_dim == 128
        assert config.use_extended is True

    def test_language_codes(self):
        """Test language codes are valid ISO 639-1."""
        config = LanguageConfig()
        for code in config.language_codes:
            assert len(code) == 2  # ISO 639-1 codes are 2 chars


class TestLanguageConstants:
    """Tests for language constants."""

    def test_core_languages(self):
        """Test core language set."""
        assert len(CORE_LANGUAGES) == 9
        assert "en" in CORE_LANGUAGES  # English
        assert "es" in CORE_LANGUAGES  # Spanish
        assert "fr" in CORE_LANGUAGES  # French
        assert "de" in CORE_LANGUAGES  # German
        assert "zh" in CORE_LANGUAGES  # Chinese

    def test_extended_languages(self):
        """Test extended language set."""
        assert len(EXTENDED_LANGUAGES) > len(CORE_LANGUAGES)
        # Should include all core languages
        for lang in CORE_LANGUAGES:
            assert lang in EXTENDED_LANGUAGES
        # Should include additional languages
        assert "ja" in EXTENDED_LANGUAGES  # Japanese
        assert "ko" in EXTENDED_LANGUAGES  # Korean
        assert "ar" in EXTENDED_LANGUAGES  # Arabic
        # Should include special tokens
        assert "<unknown>" in EXTENDED_LANGUAGES
        assert "<mixed>" in EXTENDED_LANGUAGES

    def test_language_names(self):
        """Test language name mapping."""
        assert LANGUAGE_NAMES["en"] == "English"
        assert LANGUAGE_NAMES["es"] == "Spanish"
        assert LANGUAGE_NAMES["zh"] == "Chinese"
        assert LANGUAGE_NAMES["<unknown>"] == "Unknown"


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

        mask = mx.zeros((batch_size, seq_len), dtype=mx.bool_)
        mask = mask.at[:, 80:].add(True)

        out = pooling(x, mask=mask)

        assert out.shape == (batch_size, embed_dim)


class TestLanguageHead:
    """Tests for LanguageHead module."""

    def test_forward_default_config(self):
        """Test forward pass with default config."""
        head = LanguageHead()
        batch_size = 4
        seq_len = 100
        encoder_dim = 384

        x = mx.random.normal(shape=(batch_size, seq_len, encoder_dim))
        logits = head(x)

        assert logits.shape == (batch_size, head.config.num_languages)

    def test_forward_custom_config(self):
        """Test forward pass with custom config."""
        config = LanguageConfig(
            encoder_dim=256,
            num_languages=15,
            hidden_dim=128,
        )
        head = LanguageHead(config)
        batch_size = 2
        seq_len = 50

        x = mx.random.normal(shape=(batch_size, seq_len, 256))
        logits = head(x)

        assert logits.shape == (batch_size, 15)

    def test_forward_extended_languages(self):
        """Test forward pass with extended language set."""
        config = LanguageConfig(use_extended=True)
        head = LanguageHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        logits = head(x)

        assert logits.shape == (batch_size, len(EXTENDED_LANGUAGES))

    def test_forward_with_lengths(self):
        """Test forward pass with variable sequence lengths."""
        head = LanguageHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        lengths = mx.array([100, 80, 60, 40])

        logits = head(x, encoder_lengths=lengths)

        assert logits.shape == (batch_size, head.config.num_languages)

    def test_forward_mean_pooling(self):
        """Test forward pass with mean pooling."""
        config = LanguageConfig(use_attention_pooling=False)
        head = LanguageHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        logits = head(x)

        assert logits.shape == (batch_size, head.config.num_languages)

    def test_predict(self):
        """Test prediction method."""
        head = LanguageHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        predictions, probs = head.predict(x)

        assert predictions.shape == (batch_size,)
        assert probs.shape == (batch_size, head.config.num_languages)
        # Probs should sum to 1
        prob_sums = mx.sum(probs, axis=-1)
        assert mx.allclose(prob_sums, mx.ones(batch_size), atol=1e-5)

    def test_get_language_code(self):
        """Test language code retrieval."""
        head = LanguageHead()
        code = head.get_language_code(0)
        assert code == CORE_LANGUAGES[0]

    def test_get_language_name(self):
        """Test language name retrieval."""
        head = LanguageHead()
        name = head.get_language_name(0)
        assert name == LANGUAGE_NAMES[CORE_LANGUAGES[0]]

    def test_get_language_id(self):
        """Test language ID retrieval."""
        head = LanguageHead()
        lang_id = head.get_language_id("en")
        code = head.get_language_code(lang_id)
        assert code == "en"

    def test_decode_predictions(self):
        """Test decoding predictions to readable format."""
        head = LanguageHead()
        batch_size = 2

        predictions = mx.array([0, 1])
        probs = mx.zeros((batch_size, head.config.num_languages))
        probs = probs.at[0, 0].add(0.9)
        probs = probs.at[0, 1].add(0.05)
        probs = probs.at[0, 2].add(0.05)
        probs = probs.at[1, 1].add(0.8)
        probs = probs.at[1, 0].add(0.15)
        probs = probs.at[1, 2].add(0.05)

        results = head.decode_predictions(predictions, probs, top_k=3)

        assert len(results) == 2
        assert len(results[0]) == 3  # top-3
        assert results[0][0][0] == CORE_LANGUAGES[0]  # code
        assert results[0][0][2] > 0.8  # confidence


class TestLanguageLoss:
    """Tests for language loss functions."""

    def test_basic_loss(self):
        """Test basic cross-entropy loss."""
        batch_size = 4
        num_languages = 9

        logits = mx.random.normal(shape=(batch_size, num_languages))
        targets = mx.array([0, 1, 2, 3])

        loss = language_loss(logits, targets)

        assert loss.shape == ()
        assert loss > 0

    def test_loss_with_label_smoothing(self):
        """Test loss with label smoothing."""
        batch_size = 4
        num_languages = 9

        logits = mx.random.normal(shape=(batch_size, num_languages))
        targets = mx.array([0, 1, 2, 3])

        loss_smooth = language_loss(logits, targets, label_smoothing=0.1)
        loss_hard = language_loss(logits, targets, label_smoothing=0.0)

        # Smoothed loss should generally be different
        assert not mx.allclose(loss_smooth, loss_hard, atol=1e-6)

    def test_loss_module_wrapper(self):
        """Test LanguageLoss module."""
        loss_fn = LanguageLoss(label_smoothing=0.1)
        batch_size = 4
        num_languages = 9

        logits = mx.random.normal(shape=(batch_size, num_languages))
        targets = mx.array([0, 1, 2, 3])

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss > 0

    def test_loss_reduction_options(self):
        """Test different reduction options."""
        batch_size = 4
        num_languages = 9

        logits = mx.random.normal(shape=(batch_size, num_languages))
        targets = mx.array([0, 1, 2, 3])

        loss_mean = language_loss(logits, targets, reduction="mean")
        loss_sum = language_loss(logits, targets, reduction="sum")
        loss_none = language_loss(logits, targets, reduction="none")

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (batch_size,)


class TestLanguageAccuracy:
    """Tests for accuracy computation."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        predictions = mx.array([0, 1, 2, 3])
        targets = mx.array([0, 1, 2, 3])

        acc = compute_language_accuracy(predictions, targets)

        assert mx.allclose(acc, mx.array(1.0), atol=1e-5)

    def test_partial_accuracy(self):
        """Test partial accuracy."""
        predictions = mx.array([0, 1, 2, 3])
        targets = mx.array([0, 1, 2, 2])  # 3/4 correct

        acc = compute_language_accuracy(predictions, targets)

        assert mx.allclose(acc, mx.array(0.75), atol=1e-5)

    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        predictions = mx.array([0, 1, 2, 3])
        targets = mx.array([1, 2, 3, 0])  # All wrong

        acc = compute_language_accuracy(predictions, targets)

        assert mx.allclose(acc, mx.array(0.0), atol=1e-5)


class TestLanguageGradients:
    """Tests for gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        head = LanguageHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        targets = mx.array([0, 1, 2, 3])

        def loss_fn(model, x, targets):
            logits = model(x)
            return language_loss(logits, targets)

        loss, grads = mx.value_and_grad(loss_fn)(head, x, targets)
        mx.eval(loss, grads)

        assert loss > 0
        assert "classifier" in grads

    def test_training_step(self):
        """Test a complete training step."""
        head = LanguageHead()
        optimizer = optim.Adam(learning_rate=1e-4)

        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        targets = mx.array([0, 1, 2, 3])

        def loss_fn(model, x, targets):
            logits = model(x)
            return language_loss(logits, targets)

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
        assert updated_loss <= initial_loss * 1.1


class TestLanguageHeadIntegration:
    """Integration tests for language head."""

    def test_batch_processing(self):
        """Test processing multiple batches."""
        head = LanguageHead()

        # Process several batches
        for batch_idx in range(3):
            batch_size = 4
            seq_len = 100 + batch_idx * 20  # Varying lengths

            x = mx.random.normal(shape=(batch_size, seq_len, 384))
            predictions, probs = head.predict(x)

            assert predictions.shape == (batch_size,)
            assert probs.shape == (batch_size, head.config.num_languages)

    def test_deterministic_inference(self):
        """Test that inference is deterministic."""
        head = LanguageHead()

        x = mx.random.normal(shape=(4, 100, 384))

        pred1, probs1 = head.predict(x)
        pred2, probs2 = head.predict(x)

        assert mx.array_equal(pred1, pred2)
        assert mx.allclose(probs1, probs2, atol=1e-6)
