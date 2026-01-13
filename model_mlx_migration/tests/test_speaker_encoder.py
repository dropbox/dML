"""Tests for Speaker Encoder (Phase 10.1).

Tests the ECAPA-TDNN speaker verification model for:
- Model loading from pretrained weights
- Embedding extraction
- Cosine similarity computation
- Speaker database functionality
"""

from pathlib import Path

import mlx.core as mx
import pytest

# Skip if model weights not available
SPKVER_MODEL_PATH = Path("models/ecapa-spkver-mlx")
SKIP_NO_MODEL = pytest.mark.skipif(
    not SPKVER_MODEL_PATH.exists(),
    reason="Speaker verification model not found",
)


class TestSpeakerEncoder:
    """Test SpeakerEncoder class."""

    @SKIP_NO_MODEL
    def test_load_pretrained(self):
        """Test loading from pretrained weights."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder.from_pretrained(str(SPKVER_MODEL_PATH))
        assert encoder is not None
        assert encoder.embedding_dim == 192

    @SKIP_NO_MODEL
    def test_encode_basic(self):
        """Test basic encoding."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder.from_pretrained(str(SPKVER_MODEL_PATH))

        # Create random mel features (batch, time, n_mels)
        mel = mx.random.normal((1, 100, 80))
        emb = encoder.encode(mel)

        assert emb.shape == (1, 192)
        # Check normalized
        norm = mx.linalg.norm(emb, axis=-1)
        assert abs(float(norm[0]) - 1.0) < 0.01

    @SKIP_NO_MODEL
    def test_encode_batch(self):
        """Test batch encoding."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder.from_pretrained(str(SPKVER_MODEL_PATH))

        # Batch of 4
        mel = mx.random.normal((4, 100, 80))
        emb = encoder.encode(mel)

        assert emb.shape == (4, 192)

    @SKIP_NO_MODEL
    def test_encode_variable_length(self):
        """Test encoding different length inputs."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder.from_pretrained(str(SPKVER_MODEL_PATH))

        # Different lengths
        for length in [50, 100, 200, 400]:
            mel = mx.random.normal((1, length, 80))
            emb = encoder.encode(mel)
            assert emb.shape == (1, 192)

    @SKIP_NO_MODEL
    def test_similarity(self):
        """Test similarity computation."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder.from_pretrained(str(SPKVER_MODEL_PATH))

        # Same input should have similarity ~1
        mel = mx.random.normal((1, 100, 80))
        emb1 = encoder.encode(mel)
        emb2 = encoder.encode(mel)  # Same input

        sim = encoder.similarity(emb1[0], emb2[0])
        assert float(sim) > 0.99

    @SKIP_NO_MODEL
    def test_different_speakers(self):
        """Test that different random inputs produce different embeddings.

        Note: Random noise inputs may still produce similar embeddings
        because the model extracts speech-like features. We test that
        the embeddings are at least not identical.
        """
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder.from_pretrained(str(SPKVER_MODEL_PATH))

        # Different random inputs
        mel1 = mx.random.normal((1, 100, 80), key=mx.random.key(42))
        mel2 = mx.random.normal((1, 100, 80), key=mx.random.key(123))

        emb1 = encoder.encode(mel1)
        emb2 = encoder.encode(mel2)

        # Embeddings should be different (not identical)
        diff = mx.max(mx.abs(emb1 - emb2))
        assert float(diff) > 1e-6, "Embeddings should differ for different inputs"


class TestSpeakerDatabase:
    """Test SpeakerDatabase class."""

    def test_add_speaker(self):
        """Test adding speakers."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerDatabase

        db = SpeakerDatabase()

        emb = mx.random.normal((192,))
        spk_id = db.add_speaker(emb)

        assert spk_id == 0
        assert len(db) == 1

    def test_identify(self):
        """Test speaker identification."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerDatabase

        db = SpeakerDatabase(similarity_threshold=0.7)

        # Add speaker
        emb = mx.random.normal((192,))
        emb = emb / mx.linalg.norm(emb)  # Normalize
        spk_id = db.add_speaker(emb)

        # Same embedding should match
        match_id, sim = db.identify(emb, update=False)
        assert match_id == spk_id
        assert sim > 0.99

    def test_identify_different(self):
        """Test that different embeddings don't match."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerDatabase

        db = SpeakerDatabase(similarity_threshold=0.7)

        # Add speaker
        emb1 = mx.random.normal((192,), key=mx.random.key(42))
        db.add_speaker(emb1)

        # Very different embedding
        emb2 = -emb1  # Opposite direction
        match_id, sim = db.identify(emb2, update=False)
        assert match_id == -1  # No match

    def test_identify_or_add(self):
        """Test identify or add functionality."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerDatabase

        db = SpeakerDatabase(similarity_threshold=0.7)

        # First embedding - should be added
        emb1 = mx.random.normal((192,), key=mx.random.key(42))
        emb1 = emb1 / mx.linalg.norm(emb1)
        spk_id1, sim1, is_new1 = db.identify_or_add(emb1)
        assert is_new1 is True
        assert len(db) == 1

        # Same embedding - should match
        spk_id2, sim2, is_new2 = db.identify_or_add(emb1)
        assert is_new2 is False
        assert spk_id2 == spk_id1
        assert sim2 > 0.99

    def test_ema_update(self):
        """Test EMA update of embeddings."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerDatabase

        db = SpeakerDatabase(similarity_threshold=0.5, ema_decay=0.9)

        # Add speaker
        emb1 = mx.random.normal((192,), key=mx.random.key(42))
        emb1 = emb1 / mx.linalg.norm(emb1)
        spk_id = db.add_speaker(emb1)
        old_emb = mx.array(db.embeddings[spk_id])  # Create a copy

        # Slightly different embedding that still matches
        noise = mx.random.normal((192,), key=mx.random.key(123)) * 0.1
        emb2 = emb1 + noise
        emb2 = emb2 / mx.linalg.norm(emb2)

        # Identify with update
        db.identify(emb2, update=True)

        # Embedding should have changed
        new_emb = db.embeddings[spk_id]
        diff = mx.sum(mx.abs(old_emb - new_emb))
        assert float(diff) > 0.01


class TestECAPAIntegration:
    """Integration tests for ECAPA-TDNN components."""

    @SKIP_NO_MODEL
    def test_full_pipeline(self):
        """Test full speaker enrollment and verification pipeline."""
        from tools.whisper_mlx.sota.speaker_encoder import (
            SpeakerDatabase,
            SpeakerEncoder,
        )

        encoder = SpeakerEncoder.from_pretrained(str(SPKVER_MODEL_PATH))
        db = SpeakerDatabase()

        # Enroll "speaker 1"
        mel1 = mx.random.normal((1, 100, 80), key=mx.random.key(42))
        emb1 = encoder.encode(mel1)
        db.add_speaker(emb1[0])

        # Verify same speaker
        mel1_again = mx.random.normal((1, 100, 80), key=mx.random.key(42))
        emb1_again = encoder.encode(mel1_again)
        spk_id, sim = db.identify(emb1_again[0])
        assert spk_id == 0
        assert sim > 0.99

    @SKIP_NO_MODEL
    def test_deterministic(self):
        """Test that encoding is deterministic."""
        from tools.whisper_mlx.sota.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder.from_pretrained(str(SPKVER_MODEL_PATH))

        mel = mx.random.normal((1, 100, 80), key=mx.random.key(42))

        emb1 = encoder.encode(mel)
        mx.eval(emb1)
        emb2 = encoder.encode(mel)
        mx.eval(emb2)

        diff = mx.max(mx.abs(emb1 - emb2))
        assert float(diff) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
