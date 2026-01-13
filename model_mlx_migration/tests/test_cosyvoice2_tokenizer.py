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
Tests for CosyVoice2 Tokenizer

Tests text tokenization and speaker embedding functionality.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from tools.pytorch_to_mlx.converters.models.cosyvoice2_tokenizer import (
    CosyVoice2Tokenizer,
    CosyVoice2TokenizerConfig,
)


class TestCosyVoice2TokenizerConfig:
    """Tests for tokenizer configuration."""

    def test_default_config(self):
        """Test default config values."""
        config = CosyVoice2TokenizerConfig()
        assert config.speaker_dim == 192
        assert config.sample_rate == 16000
        assert config.max_length == 2048

    def test_custom_config(self):
        """Test custom config values."""
        config = CosyVoice2TokenizerConfig(
            speaker_dim=256,
            sample_rate=22050,
            max_length=4096,
        )
        assert config.speaker_dim == 256
        assert config.sample_rate == 22050
        assert config.max_length == 4096


class TestSpeakerEmbedding:
    """Tests for speaker embedding generation."""

    def test_random_speaker_embedding_shape(self):
        """Test random speaker embedding has correct shape."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)

        embedding = tokenizer.random_speaker_embedding()
        assert embedding.shape == (192,)

    def test_random_speaker_embedding_normalized(self):
        """Test random speaker embedding is L2 normalized."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)

        embedding = tokenizer.random_speaker_embedding()
        norm = mx.sqrt(mx.sum(embedding * embedding))
        # Should be approximately 1.0
        assert abs(float(norm) - 1.0) < 1e-5

    def test_random_speaker_embedding_reproducible(self):
        """Test random speaker embedding is reproducible with same seed."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)

        emb1 = tokenizer.random_speaker_embedding(seed=123)
        emb2 = tokenizer.random_speaker_embedding(seed=123)

        # Same seed should give same embedding
        assert mx.allclose(emb1, emb2)

    def test_random_speaker_embedding_different_seeds(self):
        """Test different seeds give different embeddings."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)

        emb1 = tokenizer.random_speaker_embedding(seed=1)
        emb2 = tokenizer.random_speaker_embedding(seed=2)

        # Different seeds should give different embeddings
        assert not mx.allclose(emb1, emb2)

    def test_zero_speaker_embedding(self):
        """Test zero speaker embedding."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)

        embedding = tokenizer.zero_speaker_embedding()
        assert embedding.shape == (192,)
        assert mx.all(embedding == 0)


class TestTokenizerProperties:
    """Tests for tokenizer properties."""

    def test_vocab_size_uninitialized(self):
        """Test vocab_size returns 0 when tokenizer not loaded."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)
        assert tokenizer.vocab_size == 0

    def test_pad_token_id_uninitialized(self):
        """Test pad_token_id returns 0 when tokenizer not loaded."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)
        assert tokenizer.pad_token_id == 0

    def test_eos_token_id_uninitialized(self):
        """Test eos_token_id returns 0 when tokenizer not loaded."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)
        assert tokenizer.eos_token_id == 0


class TestTokenizerEncode:
    """Tests for text encoding (requires model files)."""

    @pytest.fixture
    def model_path(self):
        """Get model path, skip if not available."""
        path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"
        if not path.exists():
            pytest.skip("CosyVoice2 model not downloaded")
        return path

    def test_encode_single_string(self, model_path):
        """Test encoding a single string."""
        tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        text = "Hello, world!"
        tokens = tokenizer.encode(text)

        assert isinstance(tokens, mx.array)
        assert tokens.ndim == 1
        assert tokens.shape[0] > 0

    def test_encode_returns_mlx_array(self, model_path):
        """Test encoding returns MLX array by default."""
        tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        tokens = tokenizer.encode("Test")
        assert isinstance(tokens, mx.array)

    def test_encode_returns_numpy(self, model_path):
        """Test encoding can return numpy array."""
        tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        tokens = tokenizer.encode("Test", return_tensors="np")
        assert isinstance(tokens, np.ndarray)

    def test_encode_batch(self, model_path):
        """Test encoding a batch of strings."""
        tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        texts = ["Hello", "World"]
        tokens = tokenizer.encode(texts)

        assert isinstance(tokens, mx.array)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 2

    def test_decode_roundtrip(self, model_path):
        """Test encoding and decoding roundtrip."""
        tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # Should get back similar text (may have minor whitespace differences)
        assert "Hello" in decoded
        assert "world" in decoded

    def test_vocab_size_loaded(self, model_path):
        """Test vocab_size when tokenizer is loaded."""
        tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        # Qwen2 tokenizer has ~151K tokens
        assert tokenizer.vocab_size > 100000


class TestFromPretrained:
    """Tests for loading from pretrained."""

    @pytest.fixture
    def model_path(self):
        """Get model path, skip if not available."""
        path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"
        if not path.exists():
            pytest.skip("CosyVoice2 model not downloaded")
        return path

    def test_from_pretrained_loads_tokenizer(self, model_path):
        """Test from_pretrained loads the text tokenizer."""
        tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        assert tokenizer._text_tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_get_default_model_path(self):
        """Test default model path."""
        path = CosyVoice2Tokenizer.get_default_model_path()
        assert path == Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"


class TestSpeakerEmbeddingExtraction:
    """Tests for speaker embedding extraction (requires onnxruntime)."""

    def test_extract_speaker_embedding_no_onnx(self):
        """Test extract_speaker_embedding raises when onnx not available."""
        config = CosyVoice2TokenizerConfig()
        tokenizer = CosyVoice2Tokenizer(config)

        # If onnxruntime not available, should raise
        if not tokenizer._onnx_available:
            audio = mx.zeros((16000,))  # 1 second of silence
            with pytest.raises(RuntimeError, match="onnxruntime not available"):
                tokenizer.extract_speaker_embedding(audio)
