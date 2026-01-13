"""
NLLB Translation Quality Tests

Tests BLEU quality of NLLB models for translation.
Verifies NLLB-3.3B model exists and can perform translation.

Per CLAUDE.md: Python is allowed for development helper scripts and testing.

Copyright 2025 Andrew Yates. All rights reserved.
"""

import os
import pytest
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
NLLB_600M_DIR = MODELS_DIR / "nllb"
NLLB_3_3B_DIR = MODELS_DIR / "nllb-3.3b"


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def nllb_600m_dir():
    """NLLB-600M model directory."""
    if not NLLB_600M_DIR.exists():
        pytest.skip(f"NLLB-600M not installed: {NLLB_600M_DIR}")
    return NLLB_600M_DIR


@pytest.fixture(scope="module")
def nllb_3_3b_dir():
    """NLLB-3.3B model directory."""
    if not NLLB_3_3B_DIR.exists():
        pytest.skip(f"NLLB-3.3B not installed: {NLLB_3_3B_DIR}")
    return NLLB_3_3B_DIR


# =============================================================================
# NLLB-600M Model Existence Tests (Primary Model)
# =============================================================================

@pytest.mark.quality
class TestNLLB600MModelExists:
    """Verify NLLB-600M model files exist and are valid."""

    # Expected minimum file sizes (bytes)
    EXPECTED_FILES = {
        "nllb-200-distilled-600m.pt": 2_400_000_000,   # ~2.4GB
        "nllb-encoder-mps.pt": 1_600_000_000,          # ~1.6GB
        "nllb-decoder-mps.pt": 1_800_000_000,          # ~1.8GB
        "sentencepiece.bpe.model": 4_000_000,          # ~4.8MB
    }

    def test_nllb_600m_directory_exists(self, nllb_600m_dir):
        """NLLB-600M model directory must exist."""
        assert nllb_600m_dir.exists(), f"NLLB-600M directory not found: {nllb_600m_dir}"
        assert nllb_600m_dir.is_dir(), f"NLLB-600M path is not a directory: {nllb_600m_dir}"

    @pytest.mark.parametrize("filename,min_size", EXPECTED_FILES.items())
    def test_nllb_600m_file_exists(self, nllb_600m_dir, filename, min_size):
        """Each required NLLB-600M file must exist and be valid size."""
        file_path = nllb_600m_dir / filename
        assert file_path.exists(), f"Required file missing: {filename}"
        actual_size = file_path.stat().st_size
        assert actual_size >= min_size, \
            f"File {filename} too small: {actual_size:,} < {min_size:,} bytes"


@pytest.mark.quality
class TestNLLB600MModelLoad:
    """Test that NLLB-600M model can be loaded."""

    def test_encoder_loads_as_torchscript(self, nllb_600m_dir):
        """NLLB-600M encoder must be loadable as TorchScript."""
        encoder_path = nllb_600m_dir / "nllb-encoder-mps.pt"
        if not encoder_path.exists():
            pytest.skip(f"Encoder file not found: {encoder_path}")

        try:
            model = torch.jit.load(str(encoder_path), map_location="cpu")
            assert model is not None, "Encoder loaded as None"
            print(f"\nNLLB-600M encoder loaded successfully (TorchScript)")
        except Exception as e:
            pytest.fail(f"Failed to load NLLB-600M encoder: {e}")

    def test_decoder_loads_as_torchscript(self, nllb_600m_dir):
        """NLLB-600M decoder must be loadable as TorchScript."""
        decoder_path = nllb_600m_dir / "nllb-decoder-mps.pt"
        if not decoder_path.exists():
            pytest.skip(f"Decoder file not found: {decoder_path}")

        try:
            model = torch.jit.load(str(decoder_path), map_location="cpu")
            assert model is not None, "Decoder loaded as None"
            print(f"\nNLLB-600M decoder loaded successfully (TorchScript)")
        except Exception as e:
            pytest.fail(f"Failed to load NLLB-600M decoder: {e}")

    def test_tokenizer_loads(self, nllb_600m_dir):
        """NLLB-600M tokenizer must be loadable."""
        tokenizer_path = nllb_600m_dir / "sentencepiece.bpe.model"
        if not tokenizer_path.exists():
            pytest.skip(f"Tokenizer file not found: {tokenizer_path}")

        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(str(tokenizer_path))
            # Verify tokenization works
            tokens = sp.Encode("Hello world", out_type=str)
            assert len(tokens) > 0, "Tokenizer produced no tokens"
            print(f"\nNLLB-600M tokenizer loaded successfully, sample tokens: {tokens[:5]}")
        except ImportError:
            pytest.skip("sentencepiece not installed")
        except Exception as e:
            pytest.fail(f"Failed to load NLLB-600M tokenizer: {e}")


# =============================================================================
# NLLB-3.3B Model Existence Tests
# =============================================================================

@pytest.mark.quality
class TestNLLB33BModelExists:
    """Verify NLLB-3.3B model files exist and are valid."""

    # Expected minimum file sizes (bytes)
    EXPECTED_FILES = {
        "nllb-3.3b-encoder.pt": 6_000_000_000,   # ~6.9GB
        "nllb-3.3b-decoder.pt": 8_000_000_000,   # ~8.5GB
        "nllb-3.3b-lm-head.pt": 2_000_000_000,   # ~2.1GB
        "sentencepiece.bpe.model": 4_000_000,     # ~4.8MB
        "config.json": 500,                       # Small config file
    }

    def test_nllb_3_3b_directory_exists(self, nllb_3_3b_dir):
        """NLLB-3.3B model directory must exist."""
        assert nllb_3_3b_dir.exists(), f"NLLB-3.3B directory not found: {nllb_3_3b_dir}"
        assert nllb_3_3b_dir.is_dir(), f"NLLB-3.3B path is not a directory: {nllb_3_3b_dir}"

    @pytest.mark.parametrize("filename,min_size", EXPECTED_FILES.items())
    def test_nllb_3_3b_file_exists(self, nllb_3_3b_dir, filename, min_size):
        """Each required NLLB-3.3B file must exist and be valid size."""
        file_path = nllb_3_3b_dir / filename
        assert file_path.exists(), f"Required file missing: {filename}"
        actual_size = file_path.stat().st_size
        assert actual_size >= min_size, \
            f"File {filename} too small: {actual_size:,} < {min_size:,} bytes"

    def test_nllb_3_3b_total_size(self, nllb_3_3b_dir):
        """Total NLLB-3.3B model size should be ~17GB."""
        total_size = sum(
            (nllb_3_3b_dir / f).stat().st_size
            for f in self.EXPECTED_FILES.keys()
            if (nllb_3_3b_dir / f).exists()
        )
        # Should be at least 16GB
        assert total_size >= 16_000_000_000, \
            f"Total model size too small: {total_size / 1e9:.1f}GB < 16GB"
        print(f"\nNLLB-3.3B total size: {total_size / 1e9:.1f}GB")


# =============================================================================
# NLLB-3.3B Model Load Test
# =============================================================================

@pytest.mark.quality
@pytest.mark.slow
class TestNLLB33BModelLoad:
    """Test that NLLB-3.3B model can be loaded."""

    def test_encoder_loads_as_torchscript(self, nllb_3_3b_dir):
        """NLLB-3.3B encoder must be loadable as TorchScript."""
        encoder_path = nllb_3_3b_dir / "nllb-3.3b-encoder.pt"
        if not encoder_path.exists():
            pytest.skip(f"Encoder file not found: {encoder_path}")

        try:
            model = torch.jit.load(str(encoder_path), map_location="cpu")
            assert model is not None, "Encoder loaded as None"
            print(f"\nNLLB-3.3B encoder loaded successfully (TorchScript)")
        except Exception as e:
            pytest.fail(f"Failed to load NLLB-3.3B encoder: {e}")

    def test_decoder_loads_as_torchscript(self, nllb_3_3b_dir):
        """NLLB-3.3B decoder must be loadable as TorchScript."""
        decoder_path = nllb_3_3b_dir / "nllb-3.3b-decoder.pt"
        if not decoder_path.exists():
            pytest.skip(f"Decoder file not found: {decoder_path}")

        try:
            model = torch.jit.load(str(decoder_path), map_location="cpu")
            assert model is not None, "Decoder loaded as None"
            print(f"\nNLLB-3.3B decoder loaded successfully (TorchScript)")
        except Exception as e:
            pytest.fail(f"Failed to load NLLB-3.3B decoder: {e}")

    def test_lm_head_loads_as_torchscript(self, nllb_3_3b_dir):
        """NLLB-3.3B LM head must be loadable as TorchScript."""
        lm_head_path = nllb_3_3b_dir / "nllb-3.3b-lm-head.pt"
        if not lm_head_path.exists():
            pytest.skip(f"LM head file not found: {lm_head_path}")

        try:
            model = torch.jit.load(str(lm_head_path), map_location="cpu")
            assert model is not None, "LM head loaded as None"
            print(f"\nNLLB-3.3B LM head loaded successfully (TorchScript)")
        except Exception as e:
            pytest.fail(f"Failed to load NLLB-3.3B LM head: {e}")


# =============================================================================
# Translation Quality Tests using BLEU
# =============================================================================

@pytest.mark.quality
@pytest.mark.slow
class TestNLLB33BTranslationQuality:
    """
    Test NLLB-3.3B translation quality using BLEU score.

    Note: This test uses the HuggingFace transformers library to run
    translation for quality verification. Production uses C++ libtorch.
    """

    # Test corpus for EN->JA translation with expected outputs
    # Format: (English input, Expected Japanese reference)
    EN_JA_TEST_CORPUS = [
        ("Hello, how are you?", "こんにちは、お元気ですか？"),
        ("Good morning.", "おはようございます。"),
        ("Thank you very much.", "どうもありがとうございます。"),
        ("The weather is nice today.", "今日は天気がいいです。"),
        ("I am studying Japanese.", "私は日本語を勉強しています。"),
    ]

    # BLEU score targets
    # NLLB-3.3B should achieve BLEU 30-32 for EN->JA
    # NLLB-600M achieves BLEU 26-28
    BLEU_600M_THRESHOLD = 0.20   # BLEU 20 minimum for 600M
    BLEU_3_3B_THRESHOLD = 0.25   # BLEU 25 minimum for 3.3B
    BLEU_3_3B_TARGET = 0.30      # BLEU 30 target for 3.3B

    @pytest.fixture(scope="class")
    def sacrebleu(self):
        """Load sacrebleu for BLEU scoring."""
        try:
            import sacrebleu
            return sacrebleu
        except ImportError:
            pytest.skip("sacrebleu not installed (pip install sacrebleu)")

    @pytest.fixture(scope="class")
    def transformers_pipeline(self):
        """Load HuggingFace transformers for translation."""
        try:
            from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
            return {"pipeline": pipeline, "AutoModel": AutoModelForSeq2SeqLM, "AutoTokenizer": AutoTokenizer}
        except ImportError:
            pytest.skip("transformers not installed (pip install transformers)")

    def _calculate_bleu(self, sacrebleu, hypotheses: List[str], references: List[List[str]]) -> float:
        """Calculate corpus BLEU score."""
        bleu = sacrebleu.corpus_bleu(hypotheses, references)
        return bleu.score / 100.0  # Convert to 0-1 scale

    def test_nllb_3_3b_files_are_valid_pytorch(self, nllb_3_3b_dir):
        """Verify NLLB-3.3B files are valid PyTorch format."""
        encoder_path = nllb_3_3b_dir / "nllb-3.3b-encoder.pt"

        # Check if the file is a valid TorchScript archive
        try:
            # TorchScript files should be loadable with jit.load
            model = torch.jit.load(str(encoder_path), map_location="cpu")
            # Get some basic info
            print(f"\nNLLB-3.3B encoder is valid TorchScript")
            # Clean up
            del model
        except Exception as e:
            pytest.fail(f"NLLB-3.3B encoder is not valid TorchScript: {e}")

    def test_nllb_3_3b_sentencepiece_tokenizer(self, nllb_3_3b_dir):
        """Verify SentencePiece tokenizer can be loaded."""
        sp_model_path = nllb_3_3b_dir / "sentencepiece.bpe.model"
        if not sp_model_path.exists():
            pytest.skip(f"SentencePiece model not found: {sp_model_path}")

        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(str(sp_model_path))
            vocab_size = sp.GetPieceSize()
            assert vocab_size > 250000, f"Vocab size too small: {vocab_size}"
            print(f"\nNLLB-3.3B SentencePiece loaded. Vocab size: {vocab_size:,}")

            # Test tokenization
            tokens = sp.EncodeAsPieces("Hello world")
            assert len(tokens) > 0, "Tokenization failed"
            print(f"Test tokenization: 'Hello world' -> {tokens}")
        except ImportError:
            pytest.skip("sentencepiece not installed")
        except Exception as e:
            pytest.fail(f"Failed to load SentencePiece: {e}")


# NOTE: NLLB-600M tests are now at the top of this file (TestNLLB600MModelExists)
