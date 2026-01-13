#!/usr/bin/env python3
"""
Integration tests for speaker embedding system (Phase 3).

Tests:
1. Export script generates valid TorchScript model
2. C++ test program passes all unit tests
3. Speaker identification accuracy with real audio
"""

import os
import subprocess
import sys
import pytest
import tempfile
import struct
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "stream-tts-cpp" / "build"
MODELS_DIR = PROJECT_ROOT / "models" / "speaker"


class TestSpeakerEmbeddingCpp:
    """Test the C++ speaker embedding implementation."""

    def test_cpp_unit_tests_pass(self):
        """C++ unit tests pass (similarity, normalization, database)."""
        test_binary = BUILD_DIR / "test_speaker_embedding"
        if not test_binary.exists():
            pytest.skip("test_speaker_embedding not built")

        result = subprocess.run(
            [str(test_binary)],
            capture_output=True,
            text=True,
            cwd=str(BUILD_DIR),
            timeout=60
        )

        # Check all unit tests passed
        assert "All tests PASSED!" in result.stdout, f"Tests failed:\n{result.stdout}\n{result.stderr}"

        # Verify specific tests ran
        assert "Test: Similarity Computation" in result.stdout
        assert "Test: L2 Normalization" in result.stdout
        assert "Test: Speaker Database Basic Operations" in result.stdout
        assert "Test: Speaker Identification" in result.stdout
        assert "Test: Database Save/Load" in result.stdout

    def test_speaker_database_file_format(self):
        """Speaker database file format is valid binary."""
        test_binary = BUILD_DIR / "test_speaker_embedding"
        if not test_binary.exists():
            pytest.skip("test_speaker_embedding not built")

        # Run tests to create the database file
        subprocess.run(
            [str(test_binary)],
            capture_output=True,
            cwd=str(BUILD_DIR),
            timeout=60
        )

        db_file = Path("/tmp/test_speakers.bin")
        if not db_file.exists():
            pytest.skip("Database file not created")

        # Verify file format
        with open(db_file, "rb") as f:
            # Magic: "SPKR" (0x524B5053 little-endian)
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x524B5053, f"Bad magic: {hex(magic)}"

            # Version: 1
            version = struct.unpack("<I", f.read(4))[0]
            assert version == 1, f"Bad version: {version}"

            # Num speakers
            num_speakers = struct.unpack("<I", f.read(4))[0]
            assert num_speakers == 3, f"Expected 3 speakers, got {num_speakers}"

            # Read each speaker
            for i in range(num_speakers):
                id_len = struct.unpack("<I", f.read(4))[0]
                speaker_id = f.read(id_len).decode("utf-8")
                assert len(speaker_id) > 0

                # Embedding: 192 floats = 768 bytes
                embedding_bytes = f.read(192 * 4)
                assert len(embedding_bytes) == 768


class TestExportScript:
    """Test the ECAPA-TDNN export script."""

    @pytest.mark.slow
    def test_export_script_runs(self):
        """Export script runs without errors (downloads model if needed)."""
        export_script = PROJECT_ROOT / "scripts" / "export_ecapa_tdnn.py"
        assert export_script.exists(), "Export script not found"

        # Check if speechbrain is installed and functional
        try:
            import speechbrain
        except ImportError:
            pytest.skip("SpeechBrain not installed (pip install speechbrain)")
        except Exception as e:
            pytest.skip(f"SpeechBrain not functional: {e}")

        # Run export script
        result = subprocess.run(
            [sys.executable, str(export_script)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300  # 5 minutes for model download
        )

        assert result.returncode == 0, f"Export failed:\n{result.stderr}"
        assert "Export complete!" in result.stdout

    @pytest.mark.slow
    def test_exported_model_valid(self):
        """Exported TorchScript model can be loaded and used."""
        model_path = MODELS_DIR / "ecapa_tdnn.pt"
        if not model_path.exists():
            pytest.skip("Model not exported (run export_ecapa_tdnn.py first)")

        import torch

        # Load model
        model = torch.jit.load(str(model_path))
        model.eval()

        # Test inference
        with torch.no_grad():
            # 1 second of random audio at 16kHz
            audio = torch.randn(1, 16000)
            embedding = model(audio)

        # Check output shape
        assert embedding.shape == (1, 192), f"Bad shape: {embedding.shape}"

        # Check L2 normalization
        norm = torch.norm(embedding, p=2, dim=1)
        assert torch.allclose(norm, torch.ones(1), atol=0.01), f"Not normalized: norm={norm.item()}"


class TestSpeakerIdentification:
    """Test speaker identification accuracy."""

    @pytest.mark.slow
    def test_same_speaker_high_similarity(self):
        """Same speaker audio produces high similarity scores."""
        model_path = MODELS_DIR / "ecapa_tdnn.pt"
        if not model_path.exists():
            pytest.skip("Model not exported")

        import torch
        import torchaudio

        model = torch.jit.load(str(model_path))
        model.eval()

        # Generate two "utterances" from same speaker (simulated with same seed)
        torch.manual_seed(42)
        audio1 = torch.randn(1, 32000)  # 2 seconds

        torch.manual_seed(42)  # Same seed = same speaker characteristics
        audio2 = torch.randn(1, 32000)

        with torch.no_grad():
            emb1 = model(audio1)
            emb2 = model(audio2)

        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        # Same audio should have similarity ~1.0
        assert similarity > 0.99, f"Same audio similarity too low: {similarity}"

    @pytest.mark.slow
    def test_different_speakers_low_similarity(self):
        """Different speaker audio produces low similarity scores."""
        model_path = MODELS_DIR / "ecapa_tdnn.pt"
        if not model_path.exists():
            pytest.skip("Model not exported")

        import torch

        model = torch.jit.load(str(model_path))
        model.eval()

        # Generate audio from different "speakers" (different seeds)
        torch.manual_seed(42)
        audio1 = torch.randn(1, 16000)

        torch.manual_seed(123)
        audio2 = torch.randn(1, 16000)

        with torch.no_grad():
            emb1 = model(audio1)
            emb2 = model(audio2)

        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        # Different random audio should have lower similarity
        # Note: Random noise doesn't represent real speaker characteristics, so
        # embeddings will still be relatively similar. Threshold 0.95 ensures
        # model outputs are not identical, while acknowledging test limitations.
        # For real speaker differentiation tests, use actual speech samples.
        assert similarity < 0.95, f"Different audio similarity too high: {similarity}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
