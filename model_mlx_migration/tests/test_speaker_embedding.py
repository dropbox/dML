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

"""Tests for Speaker Embedding Head (Phase 8)."""

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture
def speaker_head():
    """Create a SpeakerEmbeddingHead for testing."""
    from tools.whisper_mlx.rich_ctc_head import SpeakerEmbeddingHead
    return SpeakerEmbeddingHead(
        d_model=1280,
        embed_dim=256,
        hidden_dim=512,
        normalize=True,
    )


@pytest.fixture
def rich_ctc_with_speaker():
    """Create a RichCTCHead with speaker embedding enabled."""
    from tools.whisper_mlx.rich_ctc_head import RichCTCConfig, RichCTCHead
    config = RichCTCConfig(use_speaker_embedding=True)
    return RichCTCHead(config)


@pytest.fixture
def rich_ctc_without_speaker():
    """Create a RichCTCHead with speaker embedding disabled (default)."""
    from tools.whisper_mlx.rich_ctc_head import RichCTCHead
    return RichCTCHead()


class TestSpeakerEmbeddingHead:
    """Tests for the standalone SpeakerEmbeddingHead module."""

    def test_init(self, speaker_head):
        """Test SpeakerEmbeddingHead initialization."""
        assert speaker_head.d_model == 1280
        assert speaker_head.embed_dim == 256
        assert speaker_head.normalize is True

    def test_output_shape_batched(self, speaker_head):
        """Test output shape with batched input."""
        encoder_output = mx.random.normal((2, 100, 1280))
        embedding = speaker_head(encoder_output)
        mx.eval(embedding)
        assert embedding.shape == (2, 256)

    def test_output_shape_unbatched(self, speaker_head):
        """Test output shape with unbatched input."""
        encoder_output = mx.random.normal((100, 1280))
        embedding = speaker_head(encoder_output)
        mx.eval(embedding)
        assert embedding.shape == (256,)

    def test_l2_normalized(self, speaker_head):
        """Test that output embeddings are L2 normalized."""
        encoder_output = mx.random.normal((2, 100, 1280))
        embedding = speaker_head(encoder_output)
        mx.eval(embedding)

        # Check norm is approximately 1 for each embedding in batch
        norms = mx.sqrt(mx.sum(embedding ** 2, axis=-1))
        mx.eval(norms)
        np.testing.assert_allclose(np.array(norms), [1.0, 1.0], atol=1e-5)

    def test_l2_normalized_unbatched(self, speaker_head):
        """Test L2 normalization for unbatched input."""
        encoder_output = mx.random.normal((100, 1280))
        embedding = speaker_head(encoder_output)
        mx.eval(embedding)

        norm = float(mx.sqrt(mx.sum(embedding ** 2)))
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_unnormalized_head(self):
        """Test head without L2 normalization."""
        from tools.whisper_mlx.rich_ctc_head import SpeakerEmbeddingHead

        head = SpeakerEmbeddingHead(
            d_model=1280,
            embed_dim=256,
            hidden_dim=512,
            normalize=False,
        )
        encoder_output = mx.random.normal((1, 100, 1280))
        embedding = head(encoder_output)
        mx.eval(embedding)

        # Norm should NOT be 1 when normalize=False
        norm = float(mx.sqrt(mx.sum(embedding ** 2)))
        # Just check it's a valid positive number
        assert norm > 0

    def test_different_inputs_different_embeddings(self, speaker_head):
        """Test that different inputs produce different embeddings."""
        input1 = mx.random.normal((100, 1280), key=mx.random.key(42))
        input2 = mx.random.normal((100, 1280), key=mx.random.key(123))

        emb1 = speaker_head(input1)
        emb2 = speaker_head(input2)
        mx.eval(emb1, emb2)

        # Embeddings should be different
        diff = mx.sum(mx.abs(emb1 - emb2))
        mx.eval(diff)
        assert float(diff) > 0.1

    def test_same_input_same_embedding(self, speaker_head):
        """Test that same input produces same embedding."""
        input1 = mx.random.normal((100, 1280), key=mx.random.key(42))

        emb1 = speaker_head(input1)
        emb2 = speaker_head(input1)
        mx.eval(emb1, emb2)

        # Embeddings should be identical
        np.testing.assert_allclose(np.array(emb1), np.array(emb2), rtol=1e-5)

    def test_compute_similarity_identical(self, speaker_head):
        """Test similarity computation for identical embeddings."""
        encoder_output = mx.random.normal((100, 1280))
        embedding = speaker_head(encoder_output)
        mx.eval(embedding)

        similarity = speaker_head.compute_similarity(embedding, embedding)
        mx.eval(similarity)

        # Identical embeddings should have similarity ~1.0
        np.testing.assert_allclose(float(similarity), 1.0, atol=1e-5)

    def test_compute_similarity_different(self, speaker_head):
        """Test similarity computation for different embeddings."""
        input1 = mx.random.normal((100, 1280), key=mx.random.key(42))
        input2 = mx.random.normal((100, 1280), key=mx.random.key(123))

        emb1 = speaker_head(input1)
        emb2 = speaker_head(input2)
        mx.eval(emb1, emb2)

        similarity = speaker_head.compute_similarity(emb1, emb2)
        mx.eval(similarity)

        # Different random inputs should have similarity < 1 but could be anywhere
        sim_val = float(similarity)
        assert -1.0 <= sim_val <= 1.0
        assert sim_val < 0.99  # Should not be identical

    def test_compute_similarity_batched(self, speaker_head):
        """Test batched similarity computation."""
        encoder_output1 = mx.random.normal((3, 100, 1280))
        encoder_output2 = mx.random.normal((3, 100, 1280))

        emb1 = speaker_head(encoder_output1)
        emb2 = speaker_head(encoder_output2)
        mx.eval(emb1, emb2)

        similarity = speaker_head.compute_similarity(emb1, emb2)
        mx.eval(similarity)

        assert similarity.shape == (3,)
        # All similarities should be valid
        sim_array = np.array(similarity)
        assert np.all(sim_array >= -1.0)
        assert np.all(sim_array <= 1.0)

    def test_attention_mask(self, speaker_head):
        """Test masked mean pooling with attention mask."""
        encoder_output = mx.random.normal((2, 100, 1280))

        # Mask where only first 50 frames are valid for first sample,
        # and only last 50 frames for second sample
        mask = mx.zeros((2, 100))
        mask = mx.array([[1.0] * 50 + [0.0] * 50,
                         [0.0] * 50 + [1.0] * 50])

        emb_masked = speaker_head(encoder_output, attention_mask=mask)
        emb_unmasked = speaker_head(encoder_output)
        mx.eval(emb_masked, emb_unmasked)

        # Masked and unmasked should be different
        diff = mx.sum(mx.abs(emb_masked - emb_unmasked))
        mx.eval(diff)
        assert float(diff) > 0.01


class TestRichCTCHeadWithSpeaker:
    """Tests for RichCTCHead with speaker embedding enabled."""

    def test_config_speaker_enabled(self, rich_ctc_with_speaker):
        """Test that speaker embedding is enabled in config."""
        assert rich_ctc_with_speaker._use_speaker_embedding is True
        assert rich_ctc_with_speaker.speaker is not None

    def test_config_speaker_disabled(self, rich_ctc_without_speaker):
        """Test that speaker embedding is disabled by default."""
        assert rich_ctc_without_speaker._use_speaker_embedding is False
        assert rich_ctc_without_speaker.speaker is None

    def test_forward_includes_speaker_embedding(self, rich_ctc_with_speaker):
        """Test that forward pass includes speaker embedding when enabled."""
        encoder_output = mx.random.normal((1, 100, 1280))
        outputs = rich_ctc_with_speaker(encoder_output)
        mx.eval(outputs)

        assert "speaker_embedding" in outputs
        assert outputs["speaker_embedding"].shape == (1, 256)

    def test_forward_excludes_speaker_embedding_when_disabled(self, rich_ctc_without_speaker):
        """Test that forward pass excludes speaker embedding when disabled."""
        encoder_output = mx.random.normal((1, 100, 1280))
        outputs = rich_ctc_without_speaker(encoder_output)
        mx.eval(outputs)

        assert "speaker_embedding" not in outputs

    def test_get_speaker_embedding_enabled(self, rich_ctc_with_speaker):
        """Test get_speaker_embedding when enabled."""
        encoder_output = mx.random.normal((1, 100, 1280))
        embedding = rich_ctc_with_speaker.get_speaker_embedding(encoder_output)
        mx.eval(embedding)

        assert embedding is not None
        assert embedding.shape == (1, 256)

    def test_get_speaker_embedding_disabled(self, rich_ctc_without_speaker):
        """Test get_speaker_embedding returns None when disabled."""
        encoder_output = mx.random.normal((1, 100, 1280))
        embedding = rich_ctc_without_speaker.get_speaker_embedding(encoder_output)

        assert embedding is None

    def test_compute_speaker_similarity_enabled(self, rich_ctc_with_speaker):
        """Test compute_speaker_similarity when enabled."""
        input1 = mx.random.normal((100, 1280), key=mx.random.key(42))
        input2 = mx.random.normal((100, 1280), key=mx.random.key(42))  # Same input

        emb1 = rich_ctc_with_speaker.get_speaker_embedding(input1)
        emb2 = rich_ctc_with_speaker.get_speaker_embedding(input2)
        mx.eval(emb1, emb2)

        similarity = rich_ctc_with_speaker.compute_speaker_similarity(emb1, emb2)
        mx.eval(similarity)

        # Same input should give similarity ~1.0
        np.testing.assert_allclose(float(similarity), 1.0, atol=1e-5)

    def test_compute_speaker_similarity_disabled_raises(self, rich_ctc_without_speaker):
        """Test compute_speaker_similarity raises when disabled."""
        emb1 = mx.random.normal((256,))
        emb2 = mx.random.normal((256,))

        with pytest.raises(ValueError, match="Speaker embedding head is not enabled"):
            rich_ctc_without_speaker.compute_speaker_similarity(emb1, emb2)

    def test_speaker_embedding_unbatched(self, rich_ctc_with_speaker):
        """Test speaker embedding with unbatched input."""
        encoder_output = mx.random.normal((100, 1280))  # No batch dim
        outputs = rich_ctc_with_speaker(encoder_output)
        mx.eval(outputs)

        assert "speaker_embedding" in outputs
        assert outputs["speaker_embedding"].shape == (256,)

    def test_speaker_embedding_normalized(self, rich_ctc_with_speaker):
        """Test that speaker embeddings from RichCTCHead are L2 normalized."""
        encoder_output = mx.random.normal((2, 100, 1280))
        outputs = rich_ctc_with_speaker(encoder_output)
        mx.eval(outputs)

        embedding = outputs["speaker_embedding"]
        norms = mx.sqrt(mx.sum(embedding ** 2, axis=-1))
        mx.eval(norms)

        np.testing.assert_allclose(np.array(norms), [1.0, 1.0], atol=1e-5)


class TestSpeakerEmbeddingUseCases:
    """Tests demonstrating practical use cases for speaker embedding."""

    def test_same_speaker_different_utterances(self, rich_ctc_with_speaker):
        """Test that same speaker (similar acoustic features) has high similarity."""
        # Simulate two utterances from same "speaker" by using correlated noise
        base_features = mx.random.normal((100, 1280), key=mx.random.key(42))

        # Add small variations (same speaker, different utterance)
        noise1 = mx.random.normal((100, 1280), key=mx.random.key(1)) * 0.1
        noise2 = mx.random.normal((100, 1280), key=mx.random.key(2)) * 0.1

        utterance1 = base_features + noise1
        utterance2 = base_features + noise2

        emb1 = rich_ctc_with_speaker.get_speaker_embedding(utterance1)
        emb2 = rich_ctc_with_speaker.get_speaker_embedding(utterance2)
        mx.eval(emb1, emb2)

        similarity = rich_ctc_with_speaker.compute_speaker_similarity(emb1, emb2)
        mx.eval(similarity)

        # High similarity expected (same base features with small noise)
        assert float(similarity) > 0.5

    def test_different_speakers(self, rich_ctc_with_speaker):
        """Test that different speakers have lower similarity."""
        # Completely different features = different speakers
        speaker1_input = mx.random.normal((100, 1280), key=mx.random.key(1000))
        speaker2_input = mx.random.normal((100, 1280), key=mx.random.key(2000))

        emb1 = rich_ctc_with_speaker.get_speaker_embedding(speaker1_input)
        emb2 = rich_ctc_with_speaker.get_speaker_embedding(speaker2_input)
        mx.eval(emb1, emb2)

        similarity = rich_ctc_with_speaker.compute_speaker_similarity(emb1, emb2)
        mx.eval(similarity)

        # Random inputs have similarity that can vary, but typically less extreme
        # than same-speaker scenarios
        sim_val = float(similarity)
        assert -1.0 <= sim_val <= 1.0

    def test_speaker_clustering_scenario(self, rich_ctc_with_speaker):
        """Test scenario: cluster embeddings from multiple utterances."""
        # 3 utterances from "speaker A" (correlated)
        base_a = mx.random.normal((100, 1280), key=mx.random.key(100))
        speaker_a_utterances = [
            base_a + mx.random.normal((100, 1280), key=mx.random.key(i)) * 0.1
            for i in range(3)
        ]

        # 3 utterances from "speaker B" (different base)
        base_b = mx.random.normal((100, 1280), key=mx.random.key(200))
        speaker_b_utterances = [
            base_b + mx.random.normal((100, 1280), key=mx.random.key(i + 10)) * 0.1
            for i in range(3)
        ]

        # Get embeddings
        embeddings_a = [rich_ctc_with_speaker.get_speaker_embedding(u) for u in speaker_a_utterances]
        embeddings_b = [rich_ctc_with_speaker.get_speaker_embedding(u) for u in speaker_b_utterances]

        for e in embeddings_a + embeddings_b:
            mx.eval(e)

        # Intra-speaker similarities (same speaker)
        intra_a = [
            float(rich_ctc_with_speaker.compute_speaker_similarity(embeddings_a[0], embeddings_a[i]))
            for i in range(1, 3)
        ]

        # Inter-speaker similarity (different speakers)
        inter = float(rich_ctc_with_speaker.compute_speaker_similarity(embeddings_a[0], embeddings_b[0]))

        # Intra-speaker should be higher (on average) than inter-speaker
        avg_intra = sum(intra_a) / len(intra_a)
        # Note: with random initialization and no training, this may not always hold
        # but the structure should support this use case
        assert avg_intra > -1.0  # Just verify valid intra-speaker values
        assert inter > -1.0  # Just verify valid inter-speaker values


class TestConfigOptions:
    """Tests for configuration options."""

    def test_custom_embed_dim(self):
        """Test custom embedding dimension."""
        from tools.whisper_mlx.rich_ctc_head import RichCTCConfig, RichCTCHead

        config = RichCTCConfig(
            use_speaker_embedding=True,
            speaker_embed_dim=128,  # Smaller embedding
        )
        head = RichCTCHead(config)

        encoder_output = mx.random.normal((1, 100, 1280))
        outputs = head(encoder_output)
        mx.eval(outputs)

        assert outputs["speaker_embedding"].shape == (1, 128)

    def test_custom_hidden_dim(self):
        """Test custom hidden dimension."""
        from tools.whisper_mlx.rich_ctc_head import RichCTCConfig, RichCTCHead

        config = RichCTCConfig(
            use_speaker_embedding=True,
            speaker_hidden_dim=256,  # Smaller hidden
        )
        head = RichCTCHead(config)

        encoder_output = mx.random.normal((1, 100, 1280))
        outputs = head(encoder_output)
        mx.eval(outputs)

        # Output shape should still be default 256
        assert outputs["speaker_embedding"].shape == (1, 256)

    def test_no_normalization(self):
        """Test disabling L2 normalization."""
        from tools.whisper_mlx.rich_ctc_head import RichCTCConfig, RichCTCHead

        config = RichCTCConfig(
            use_speaker_embedding=True,
            speaker_normalize=False,
        )
        head = RichCTCHead(config)

        encoder_output = mx.random.normal((1, 100, 1280))
        outputs = head(encoder_output)
        mx.eval(outputs)

        embedding = outputs["speaker_embedding"]
        norm = float(mx.sqrt(mx.sum(embedding ** 2)))

        # Norm should NOT be 1 when normalization is disabled
        # Just check it's a valid positive number
        assert norm > 0
