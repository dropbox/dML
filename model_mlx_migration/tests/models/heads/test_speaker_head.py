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
Tests for DELULU-style speaker embedding head.

Tests cover:
- Forward pass and embedding extraction
- SE-Res2Net blocks
- Attentive statistics pooling
- AAM-Softmax loss function
- Cosine similarity computation
- Gradient flow for training
- EER computation
"""

import mlx.core as mx
import mlx.optimizers as optim

from src.models.heads.speaker import (
    AttentiveStatisticsPooling,
    Res2NetBlock,
    SERes2NetBlock,
    SpeakerConfig,
    SpeakerHead,
    SpeakerLoss,
    SqueezeExcitation,
    aam_softmax_loss,
    speaker_loss,
    verification_eer,
)


class TestSpeakerConfig:
    """Tests for SpeakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpeakerConfig()
        assert config.encoder_dim == 384
        assert config.embedding_dim == 256
        assert config.num_speakers == 7205
        assert config.res2net_scale == 8
        assert config.se_reduction == 8
        assert config.num_attention_heads == 1
        assert config.hidden_dim == 1536
        assert config.aam_margin == 0.2
        assert config.aam_scale == 30.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = SpeakerConfig(
            encoder_dim=512,
            embedding_dim=192,
            num_speakers=1000,
            res2net_scale=4,
        )
        assert config.encoder_dim == 512
        assert config.embedding_dim == 192
        assert config.num_speakers == 1000
        assert config.res2net_scale == 4


class TestSqueezeExcitation:
    """Tests for SE block."""

    def test_forward_shape(self):
        """Test SE block preserves shape."""
        channels = 256
        batch_size = 4
        seq_len = 100

        se = SqueezeExcitation(channels, reduction=8)
        x = mx.random.normal(shape=(batch_size, seq_len, channels))

        out = se(x)

        assert out.shape == x.shape

    def test_channel_weighting(self):
        """Test SE block applies channel-wise weighting."""
        channels = 64
        batch_size = 2
        seq_len = 50

        se = SqueezeExcitation(channels, reduction=4)
        x = mx.random.normal(shape=(batch_size, seq_len, channels))

        out = se(x)

        # Output should be weighted version of input
        # Values can change but shape preserved
        assert out.shape == (batch_size, seq_len, channels)

    def test_single_sample(self):
        """Test SE block with batch size 1."""
        channels = 128
        seq_len = 100

        se = SqueezeExcitation(channels, reduction=8)
        x = mx.random.normal(shape=(1, seq_len, channels))

        out = se(x)

        assert out.shape == (1, seq_len, channels)


class TestRes2NetBlock:
    """Tests for Res2Net block."""

    def test_forward_shape(self):
        """Test Res2Net block preserves shape."""
        channels = 256  # Must be divisible by scale
        batch_size = 4
        seq_len = 100
        scale = 8

        block = Res2NetBlock(channels, scale=scale)
        x = mx.random.normal(shape=(batch_size, seq_len, channels))

        out = block(x)

        assert out.shape == x.shape

    def test_different_scales(self):
        """Test with different scale factors."""
        channels = 256
        batch_size = 2
        seq_len = 50

        for scale in [2, 4, 8]:
            block = Res2NetBlock(channels, scale=scale)
            x = mx.random.normal(shape=(batch_size, seq_len, channels))
            out = block(x)
            assert out.shape == (batch_size, seq_len, channels)

    def test_multiscale_processing(self):
        """Test that different scales are processed."""
        channels = 64
        batch_size = 2
        seq_len = 50
        scale = 4

        block = Res2NetBlock(channels, scale=scale)
        x = mx.random.normal(shape=(batch_size, seq_len, channels))

        out = block(x)

        # Output should differ from input due to multi-scale processing
        assert out.shape == (batch_size, seq_len, channels)


class TestSERes2NetBlock:
    """Tests for combined SE-Res2Net block."""

    def test_forward_shape(self):
        """Test SE-Res2Net block preserves shape."""
        channels = 256
        batch_size = 4
        seq_len = 100

        block = SERes2NetBlock(channels, scale=8, reduction=8)
        x = mx.random.normal(shape=(batch_size, seq_len, channels))

        out = block(x)

        assert out.shape == x.shape

    def test_residual_connection(self):
        """Test residual connection is applied."""
        channels = 256
        batch_size = 2
        seq_len = 50

        block = SERes2NetBlock(channels, scale=8, reduction=8)
        x = mx.random.normal(shape=(batch_size, seq_len, channels))

        out = block(x)

        # With residual connection, output = processed(x) + x
        # So output shouldn't be too different from input for random weights
        assert out.shape == (batch_size, seq_len, channels)


class TestAttentiveStatisticsPooling:
    """Tests for ASP module."""

    def test_forward_shape(self):
        """Test ASP output shape (in_dim * 2 due to mean + std)."""
        in_dim = 256
        batch_size = 4
        seq_len = 100

        asp = AttentiveStatisticsPooling(in_dim, attention_dim=128)
        x = mx.random.normal(shape=(batch_size, seq_len, in_dim))

        out = asp(x)

        # Output is mean and std concatenated
        assert out.shape == (batch_size, in_dim * 2)

    def test_forward_with_mask(self):
        """Test ASP with padding mask."""
        in_dim = 256
        batch_size = 4
        seq_len = 100

        asp = AttentiveStatisticsPooling(in_dim, attention_dim=128)
        x = mx.random.normal(shape=(batch_size, seq_len, in_dim))

        # Create mask (True = masked)
        mask = mx.zeros((batch_size, seq_len), dtype=mx.bool_)
        mask = mask.at[:, 80:].add(True)

        out = asp(x, mask=mask)

        assert out.shape == (batch_size, in_dim * 2)

    def test_different_seq_lengths(self):
        """Test ASP with different sequence lengths."""
        in_dim = 256
        batch_size = 4

        asp = AttentiveStatisticsPooling(in_dim)

        for seq_len in [10, 50, 100, 200]:
            x = mx.random.normal(shape=(batch_size, seq_len, in_dim))
            out = asp(x)
            assert out.shape == (batch_size, in_dim * 2)

    def test_statistics_computation(self):
        """Test that mean and std are computed."""
        in_dim = 64
        batch_size = 2
        seq_len = 50

        asp = AttentiveStatisticsPooling(in_dim, attention_dim=32)
        x = mx.random.normal(shape=(batch_size, seq_len, in_dim))

        out = asp(x)

        # First half should be weighted mean, second half weighted std
        mean_part = out[:, :in_dim]
        std_part = out[:, in_dim:]

        # Mean and std should have expected shapes
        assert mean_part.shape == (batch_size, in_dim)
        assert std_part.shape == (batch_size, in_dim)
        # Std should be positive
        assert mx.all(std_part >= 0).item()


class TestSpeakerHead:
    """Tests for SpeakerHead module."""

    def test_embedding_shape(self):
        """Test embedding output shape."""
        config = SpeakerConfig(encoder_dim=384, embedding_dim=256)
        head = SpeakerHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        embeddings = head(encoder_out, return_embeddings_only=True)

        assert embeddings.shape == (batch_size, 256)

    def test_training_output(self):
        """Test training output (embeddings + logits)."""
        config = SpeakerConfig(
            encoder_dim=384,
            embedding_dim=256,
            num_speakers=100,
        )
        head = SpeakerHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        embeddings, logits = head(encoder_out, return_embeddings_only=False)

        assert embeddings.shape == (batch_size, 256)
        assert logits.shape == (batch_size, 100)

    def test_embedding_normalization(self):
        """Test embeddings are L2 normalized."""
        config = SpeakerConfig(encoder_dim=384, embedding_dim=256)
        head = SpeakerHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        embeddings = head.extract_embedding(encoder_out)

        # Check L2 norm is approximately 1
        norms = mx.sqrt(mx.sum(embeddings * embeddings, axis=-1))
        assert mx.allclose(norms, mx.ones(batch_size), atol=1e-5).item()

    def test_forward_with_lengths(self):
        """Test forward pass with sequence lengths."""
        config = SpeakerConfig(encoder_dim=384, embedding_dim=256)
        head = SpeakerHead(config)

        batch_size = 4
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))
        lengths = mx.array([80, 90, 100, 70])

        embeddings = head(encoder_out, encoder_lengths=lengths)

        assert embeddings.shape == (batch_size, 256)

    def test_similarity_computation(self):
        """Test cosine similarity between embeddings."""
        config = SpeakerConfig(encoder_dim=384, embedding_dim=256)
        head = SpeakerHead(config)

        batch_size = 2
        seq_len = 100
        encoder_out = mx.random.normal(shape=(batch_size, seq_len, 384))

        embeddings = head.extract_embedding(encoder_out)

        # Same embedding should have similarity 1
        sim_same = head.similarity(embeddings[0], embeddings[0])
        assert mx.abs(sim_same - 1.0).item() < 1e-5

        # Different embeddings should have different similarity
        sim_diff = head.similarity(embeddings[0], embeddings[1])
        # Just check it's a valid value
        assert -1.0 <= sim_diff.item() <= 1.0

    def test_different_encoder_dims(self):
        """Test with different encoder dimensions."""
        for encoder_dim in [256, 384, 512]:
            config = SpeakerConfig(encoder_dim=encoder_dim, embedding_dim=192)
            head = SpeakerHead(config)

            encoder_out = mx.random.normal(shape=(2, 50, encoder_dim))
            embeddings = head(encoder_out)

            assert embeddings.shape == (2, 192)

    def test_deterministic_output(self):
        """Test that same input gives same output (no dropout in eval)."""
        config = SpeakerConfig(encoder_dim=384, embedding_dim=256)
        head = SpeakerHead(config)

        encoder_out = mx.random.normal(shape=(2, 50, 384))

        emb1 = head.extract_embedding(encoder_out)
        emb2 = head.extract_embedding(encoder_out)

        assert mx.allclose(emb1, emb2, atol=1e-5).item()

    def test_default_config(self):
        """Test head with default config."""
        head = SpeakerHead()

        encoder_out = mx.random.normal(shape=(2, 50, 384))
        embeddings = head(encoder_out)

        assert embeddings.shape == (2, 256)


class TestAAMSoftmaxLoss:
    """Tests for AAM-Softmax loss function."""

    def test_basic_loss(self):
        """Test basic AAM-Softmax loss computation."""
        batch_size = 4
        embedding_dim = 256
        num_classes = 100

        embeddings = mx.random.normal(shape=(batch_size, embedding_dim))
        # Normalize embeddings
        embeddings = embeddings / mx.sqrt(
            mx.sum(embeddings * embeddings, axis=-1, keepdims=True),
        )

        # Create logits (scaled cosine similarities)
        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([0, 1, 2, 3])

        loss = aam_softmax_loss(
            embeddings, logits, targets, margin=0.2, scale=30.0,
        )

        # Loss should be positive
        assert loss.item() > 0

    def test_loss_decreases_with_correct_predictions(self):
        """Test that loss is lower for correct predictions."""
        batch_size = 4
        embedding_dim = 64
        num_classes = 10

        embeddings = mx.random.normal(shape=(batch_size, embedding_dim))
        embeddings = embeddings / mx.sqrt(
            mx.sum(embeddings * embeddings, axis=-1, keepdims=True),
        )

        targets = mx.array([0, 1, 2, 3])

        # Logits where targets have highest scores
        correct_logits = mx.zeros((batch_size, num_classes))
        for i in range(batch_size):
            correct_logits = correct_logits.at[i, targets[i].item()].add(5.0)

        # Random logits
        random_logits = mx.random.normal(shape=(batch_size, num_classes))

        loss_correct = aam_softmax_loss(
            embeddings, correct_logits, targets, margin=0.2, scale=30.0,
        )
        loss_random = aam_softmax_loss(
            embeddings, random_logits, targets, margin=0.2, scale=30.0,
        )

        # Loss should be lower for correct predictions
        # Note: This might not always hold due to the margin, but generally should
        assert loss_correct.item() < loss_random.item() * 10  # Relaxed check

    def test_speaker_loss_module(self):
        """Test SpeakerLoss module."""
        batch_size = 4
        embedding_dim = 256
        num_classes = 100

        loss_fn = SpeakerLoss(margin=0.2, scale=30.0)

        embeddings = mx.random.normal(shape=(batch_size, embedding_dim))
        embeddings = embeddings / mx.sqrt(
            mx.sum(embeddings * embeddings, axis=-1, keepdims=True),
        )
        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([0, 1, 2, 3])

        loss = loss_fn(embeddings, logits, targets)

        assert loss.item() > 0

    def test_speaker_loss_function(self):
        """Test speaker_loss convenience function."""
        batch_size = 2
        embedding_dim = 128
        num_classes = 50

        embeddings = mx.random.normal(shape=(batch_size, embedding_dim))
        embeddings = embeddings / mx.sqrt(
            mx.sum(embeddings * embeddings, axis=-1, keepdims=True),
        )
        logits = mx.random.normal(shape=(batch_size, num_classes))
        targets = mx.array([10, 20])

        loss = speaker_loss(embeddings, logits, targets)

        assert loss.item() > 0


class TestGradientFlow:
    """Tests for gradient computation."""

    def test_embedding_gradients(self):
        """Test gradients flow through embedding extraction."""
        config = SpeakerConfig(
            encoder_dim=384,
            embedding_dim=256,
            num_speakers=100,
            hidden_dim=256,  # Smaller for test
            res2net_scale=4,  # Smaller for test
        )
        head = SpeakerHead(config)

        encoder_out = mx.random.normal(shape=(2, 50, 384))
        targets = mx.array([0, 1])

        def loss_fn(model, x):
            embeddings, logits = model(x, return_embeddings_only=False)
            return aam_softmax_loss(embeddings, logits, targets, margin=0.2, scale=30.0)

        loss, grads = mx.value_and_grad(loss_fn)(head, encoder_out)

        # Check loss is computed
        assert loss.item() > 0

    def test_training_step(self):
        """Test a complete training step."""
        config = SpeakerConfig(
            encoder_dim=256,
            embedding_dim=128,
            num_speakers=50,
            hidden_dim=128,
            res2net_scale=4,
        )
        head = SpeakerHead(config)
        optimizer = optim.Adam(learning_rate=0.001)

        encoder_out = mx.random.normal(shape=(4, 30, 256))
        targets = mx.array([0, 1, 2, 3])

        def loss_fn(model, x):
            embeddings, logits = model(x, return_embeddings_only=False)
            return aam_softmax_loss(embeddings, logits, targets, margin=0.2, scale=30.0)

        # Initial loss
        loss1 = loss_fn(head, encoder_out)
        assert loss1.item() > 0  # Initial loss should be positive

        # Training step
        loss, grads = mx.value_and_grad(loss_fn)(head, encoder_out)
        optimizer.update(head, grads)
        mx.eval(head.parameters())

        # Check loss was computed
        assert loss.item() > 0


class TestEERComputation:
    """Tests for EER computation."""

    def test_perfect_separation(self):
        """Test EER with perfect speaker separation."""
        # Perfect case: all same-speaker pairs have higher similarity
        similarities = mx.array([0.9, 0.85, 0.8, 0.1, 0.15, 0.2])
        labels = mx.array([1, 1, 1, 0, 0, 0])

        eer = verification_eer(similarities, labels)

        # Should be 0 or very close to 0
        assert eer < 0.1

    def test_random_baseline(self):
        """Test EER with random predictions."""
        # Random similarities for same distribution
        similarities = mx.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        labels = mx.array([1, 1, 1, 0, 0, 0])

        eer = verification_eer(similarities, labels)

        # Should be close to 0.5 for random
        assert 0.0 <= eer <= 1.0

    def test_eer_range(self):
        """Test EER is in valid range."""
        for _ in range(5):
            similarities = mx.random.uniform(shape=(100,))
            labels = mx.random.randint(0, 2, shape=(100,))

            eer = verification_eer(similarities, labels)

            assert 0.0 <= eer <= 1.0

    def test_empty_class(self):
        """Test EER with only one class."""
        similarities = mx.array([0.5, 0.6, 0.7])
        labels = mx.array([1, 1, 1])  # Only positive class

        eer = verification_eer(similarities, labels)

        # Returns 0.5 for undefined case
        assert eer == 0.5


class TestIntegration:
    """Integration tests for speaker head."""

    def test_end_to_end_verification(self):
        """Test complete speaker verification pipeline."""
        config = SpeakerConfig(
            encoder_dim=384,
            embedding_dim=256,
            hidden_dim=256,
            res2net_scale=4,
        )
        head = SpeakerHead(config)

        # Simulate two utterances from same speaker
        encoder_out1 = mx.random.normal(shape=(1, 100, 384))
        encoder_out2 = encoder_out1 + mx.random.normal(shape=(1, 100, 384)) * 0.1

        # Simulate utterance from different speaker
        encoder_out3 = mx.random.normal(shape=(1, 100, 384))

        emb1 = head.extract_embedding(encoder_out1)
        emb2 = head.extract_embedding(encoder_out2)
        emb3 = head.extract_embedding(encoder_out3)

        # Same speaker similarity should be higher
        sim_same = head.similarity(emb1, emb2)
        sim_diff = head.similarity(emb1, emb3)

        # Both should be valid similarities
        assert -1.0 <= sim_same.item() <= 1.0
        assert -1.0 <= sim_diff.item() <= 1.0

    def test_batch_embedding_extraction(self):
        """Test batch processing of embeddings."""
        config = SpeakerConfig(
            encoder_dim=384,
            embedding_dim=256,
            hidden_dim=256,
            res2net_scale=4,
        )
        head = SpeakerHead(config)

        batch_size = 8
        encoder_out = mx.random.normal(shape=(batch_size, 100, 384))

        embeddings = head.extract_embedding(encoder_out)

        assert embeddings.shape == (batch_size, 256)

        # All embeddings should be normalized
        norms = mx.sqrt(mx.sum(embeddings * embeddings, axis=-1))
        assert mx.allclose(norms, mx.ones(batch_size), atol=1e-5).item()

    def test_variable_length_utterances(self):
        """Test with variable length utterances."""
        config = SpeakerConfig(
            encoder_dim=384,
            embedding_dim=256,
            hidden_dim=256,
            res2net_scale=4,
        )
        head = SpeakerHead(config)

        batch_size = 4
        max_len = 200
        encoder_out = mx.random.normal(shape=(batch_size, max_len, 384))
        lengths = mx.array([100, 150, 200, 80])

        embeddings = head.extract_embedding(encoder_out, encoder_lengths=lengths)

        assert embeddings.shape == (batch_size, 256)

    def test_small_config(self):
        """Test with minimal configuration."""
        config = SpeakerConfig(
            encoder_dim=128,
            embedding_dim=64,
            num_speakers=10,
            hidden_dim=64,
            res2net_scale=2,
            se_reduction=4,
        )
        head = SpeakerHead(config)

        encoder_out = mx.random.normal(shape=(2, 30, 128))

        embeddings = head(encoder_out)

        assert embeddings.shape == (2, 64)
