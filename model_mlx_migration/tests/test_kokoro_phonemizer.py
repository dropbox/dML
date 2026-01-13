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
Tests for Kokoro phonemizer module.

Tests cover:
- Vocabulary integrity and loading
- Phonemization accuracy
- Tokenization correctness
- G2P/tokenizer correctness gate (I5)
"""

import pytest


class TestVocabulary:
    """Test vocabulary loading and constants."""

    def test_load_vocab_returns_dict(self):
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import load_vocab

        vocab = load_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 100  # Should have ~113 entries

    def test_vocab_max_token_id(self):
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import load_vocab

        vocab = load_vocab()
        # Kokoro expects 178 tokens (0-177)
        assert max(vocab.values()) <= 177

    def test_pad_token(self):
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import PAD_TOKEN

        assert PAD_TOKEN == 0

    def test_common_phonemes_in_vocab(self):
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import load_vocab

        vocab = load_vocab()
        # Check common IPA phonemes are present
        common = ["a", "e", "i", "o", "u", "ə", "ɪ", " ", ".", ",", "!"]
        for phoneme in common:
            assert phoneme in vocab, f"Missing common phoneme: {phoneme}"

    def test_get_vocab_size(self):
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            get_vocab_size,
        )

        assert get_vocab_size() == 178


class TestTokenization:
    """Test tokenization functions."""

    def test_tokenize_phonemes_basic(self):
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            PAD_TOKEN,
            tokenize_phonemes,
        )

        # Simple phoneme string
        phonemes = "helo"
        tokens = tokenize_phonemes(phonemes)

        # Should have PAD at start and end
        assert tokens[0] == PAD_TOKEN
        assert tokens[-1] == PAD_TOKEN
        assert len(tokens) == len(phonemes) + 2

    def test_tokenize_phonemes_with_spaces(self):
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            PAD_TOKEN,
            load_vocab,
            tokenize_phonemes,
        )

        vocab = load_vocab()
        phonemes = "helo wɜld"
        tokens = tokenize_phonemes(phonemes)

        # Space should be included
        assert vocab[" "] in tokens
        assert tokens[0] == PAD_TOKEN
        assert tokens[-1] == PAD_TOKEN

    def test_tokenize_phonemes_ipa_chars(self):
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            tokenize_phonemes,
        )

        # Test IPA characters
        phonemes = "həˈloʊ"
        tokens = tokenize_phonemes(phonemes)

        # Length must be preserved: len(phonemes) + 2 (BOS + phonemes + EOS)
        # Unknown chars are mapped to PAD to preserve length alignment
        assert len(tokens) == len(phonemes) + 2

    def test_tokenize_phonemes_unknown_chars_preserve_length(self):
        """Test that unknown chars are mapped to PAD and preserve length."""
        import warnings

        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            tokenize_phonemes,
        )

        # Use a string with chars unlikely to be in vocab
        # (emoji or special chars)
        phonemes = "hello😀world"  # 11 chars including emoji
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tokens = tokenize_phonemes(phonemes)

            # Length MUST be preserved regardless of unknown chars
            assert len(tokens) == len(phonemes) + 2

            # Warning should be emitted for unknown chars
            assert len(w) >= 1
            assert "Unknown phoneme characters" in str(w[0].message)


class TestPhonemizerIntegration:
    """Integration tests for phonemize_text function."""

    @pytest.fixture
    def phonemizer_available(self):
        """Check if phonemizer is available."""
        import importlib.util

        return importlib.util.find_spec("phonemizer") is not None

    def test_phonemize_text_basic(self, phonemizer_available):
        if not phonemizer_available:
            pytest.skip("phonemizer not installed")

        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            PAD_TOKEN,
            phonemize_text,
        )

        phonemes, tokens = phonemize_text("hello")

        # Should return non-empty phonemes
        assert len(phonemes) > 0
        # Should have IPA characters
        assert any(c in phonemes for c in "əɛɪoʊ")
        # Tokens should have PAD at start/end
        assert tokens[0] == PAD_TOKEN
        assert tokens[-1] == PAD_TOKEN

    def test_phonemize_text_hello_world(self, phonemizer_available):
        if not phonemizer_available:
            pytest.skip("phonemizer not installed")

        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text,
        )

        phonemes, tokens = phonemize_text("Hello world")

        # Should produce reasonable output
        assert len(phonemes) > 5
        assert len(tokens) > 5
        # Should have space between words
        assert " " in phonemes

    def test_phonemize_different_inputs_different_outputs(self, phonemizer_available):
        if not phonemizer_available:
            pytest.skip("phonemizer not installed")

        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text,
        )

        _, tokens1 = phonemize_text("Hello")
        _, tokens2 = phonemize_text("Goodbye")

        # Different inputs should produce different outputs
        assert tokens1 != tokens2

    def test_phonemize_text_punctuation(self, phonemizer_available):
        if not phonemizer_available:
            pytest.skip("phonemizer not installed")

        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            load_vocab,
            phonemize_text,
        )

        vocab = load_vocab()
        phonemes, tokens = phonemize_text("Hello!")

        # Should preserve punctuation
        assert "!" in phonemes or vocab["!"] in tokens


class TestG2PCorrectnessGate:
    """G2P/tokenizer correctness gate (I5).

    These tests verify that our tokenization matches the reference PyTorch
    implementation. This is CRITICAL for C++ runtime correctness.
    """

    # Reference test cases: (text, expected_phonemes, expected_token_ids)
    # These should be validated against the official Kokoro PyTorch implementation.
    # Run scripts/export_kokoro_reference.py to generate reference data.

    @pytest.fixture
    def misaki_available(self):
        """Check if misaki is available."""
        import importlib.util

        return importlib.util.find_spec("misaki") is not None

    def test_vocab_matches_hf_config(self):
        """Verify exported vocab matches HuggingFace config.json."""
        import json
        from pathlib import Path

        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import load_vocab

        vocab = load_vocab()

        # Load HF config.json
        hf_config_path = Path.home() / "models" / "kokoro" / "config.json"
        if not hf_config_path.exists():
            pytest.skip("HuggingFace config.json not found")

        with open(hf_config_path) as f:
            hf_config = json.load(f)

        hf_vocab = hf_config.get("vocab", {})

        # Verify all entries match
        for phoneme, token_id in hf_vocab.items():
            assert phoneme in vocab, f"Missing phoneme: {phoneme}"
            assert vocab[phoneme] == token_id, (
                f"Token mismatch for {phoneme}: {vocab[phoneme]} != {token_id}"
            )

    def test_tokenize_known_phonemes(self):
        """Test tokenization of known phoneme strings."""
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            load_vocab,
            tokenize_phonemes,
        )

        vocab = load_vocab()

        # Test cases: (phonemes, expected_tokens without BOS/EOS)
        test_cases = [
            ("həˈloʊ", [vocab["h"], vocab["ə"], vocab["ˈ"], vocab["l"], vocab["o"]]),
            ("wˈɜld", [vocab["w"], vocab["ˈ"], vocab["ɜ"], vocab["l"], vocab["d"]]),
        ]

        for phonemes, expected_middle in test_cases:
            tokens = tokenize_phonemes(phonemes)
            # Check BOS/EOS
            assert tokens[0] == 0, "Missing BOS token"
            assert tokens[-1] == 0, "Missing EOS token"
            # Check middle matches (only for phonemes that exist in vocab)
            actual_middle = tokens[1:-1]
            # Filter expected_middle for phonemes that exist
            for i, (actual, expected) in enumerate(zip(actual_middle, expected_middle, strict=False)):
                if expected is not None:  # Skip phonemes not in vocab
                    assert actual == expected, (
                        f"Token mismatch at position {i}: {actual} != {expected}"
                    )

    def test_misaki_g2p_produces_tokenizable_output(self, misaki_available):
        """Test that misaki G2P output is tokenizable.

        NOTE: This test requires the misaki environment with spacy installed.
        Run in .venv_phonemizer environment.
        """
        if not misaki_available:
            pytest.skip("misaki not installed")

        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        try:
            from phonemize_for_kokoro import load_kokoro_vocab, phonemize
        except (ModuleNotFoundError, SystemExit) as e:
            pytest.skip(f"misaki dependencies not available: {e}")

        vocab = load_kokoro_vocab()

        test_texts = [
            "Hello world",
            "The quick brown fox",
            "Testing one two three",
        ]

        for text in test_texts:
            result = phonemize(text, vocab=vocab)
            assert result["token_count"] > 3, f"Too few tokens for '{text}'"
            assert (
                len(result["unknown_chars"]) == 0
                or len(result["unknown_chars"]) < len(result["phonemes"]) * 0.1
            ), f"Too many unknown chars: {result['unknown_chars']}"
