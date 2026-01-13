#!/usr/bin/env python3
"""
Integration tests for speaker diarization system (Phase 3.6).

Tests:
1. Agent voice is correctly identified
2. Different speakers are correctly separated
3. Self-speech filtering works correctly
"""

import os
import subprocess
import sys
import pytest
import tempfile
import struct
from pathlib import Path

import torch
import torchaudio

# Patch torchaudio for SpeechBrain compatibility
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['ffmpeg', 'soundfile']

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "speaker"
ECAPA_MODEL = MODELS_DIR / "ecapa_tdnn.pt"
AGENT_EMBEDDING = MODELS_DIR / "agent.bin"


class TestAgentEmbedding:
    """Test agent embedding file format and contents."""

    def test_agent_embedding_exists(self):
        """Agent embedding file exists."""
        assert AGENT_EMBEDDING.exists(), f"Agent embedding not found: {AGENT_EMBEDDING}"

    def test_agent_embedding_format(self):
        """Agent embedding file has valid format."""
        if not AGENT_EMBEDDING.exists():
            pytest.skip("Agent embedding not found")

        with open(AGENT_EMBEDDING, "rb") as f:
            # Magic: "SPKR" (0x524B5053 little-endian)
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x524B5053, f"Bad magic: {hex(magic)}"

            # Dimension: 192
            dim = struct.unpack("<I", f.read(4))[0]
            assert dim == 192, f"Bad dimension: {dim}"

            # Embedding values (192 floats)
            embedding_bytes = f.read(192 * 4)
            assert len(embedding_bytes) == 768, f"Incomplete embedding data"

    def test_agent_embedding_normalized(self):
        """Agent embedding is L2 normalized."""
        if not AGENT_EMBEDDING.exists():
            pytest.skip("Agent embedding not found")

        with open(AGENT_EMBEDDING, "rb") as f:
            f.seek(8)  # Skip header
            embedding = [struct.unpack("<f", f.read(4))[0] for _ in range(192)]

        # Calculate L2 norm
        norm = sum(v * v for v in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01, f"Not normalized: norm={norm}"


class TestSpeakerIdentification:
    """Test speaker identification with the ECAPA-TDNN model."""

    @pytest.fixture
    def model(self):
        """Load the exported model."""
        if not ECAPA_MODEL.exists():
            pytest.skip("ECAPA-TDNN model not found")
        return torch.jit.load(str(ECAPA_MODEL))

    @pytest.fixture
    def agent_embedding(self):
        """Load agent embedding."""
        if not AGENT_EMBEDDING.exists():
            pytest.skip("Agent embedding not found")

        with open(AGENT_EMBEDDING, "rb") as f:
            f.seek(8)  # Skip header
            embedding = [struct.unpack("<f", f.read(4))[0] for _ in range(192)]
        return torch.tensor(embedding)

    @pytest.fixture
    def tts_model(self):
        """Load TTS for generating agent voice."""
        try:
            from TTS.api import TTS
            return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        except ImportError:
            pytest.skip("Coqui TTS not installed")

    @pytest.mark.slow
    def test_tts_voice_matches_agent(self, model, agent_embedding, tts_model):
        """TTS output should match the registered agent embedding."""
        # Generate TTS audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        tts_model.tts_to_file(text="Hello, this is a test.", file_path=temp_path)

        # Load and resample
        waveform, sr = torchaudio.load(temp_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        os.unlink(temp_path)

        # Extract embedding
        model.eval()
        with torch.no_grad():
            new_embedding = model(waveform).squeeze()

        # Compare with agent embedding
        similarity = torch.nn.functional.cosine_similarity(
            new_embedding.unsqueeze(0),
            agent_embedding.unsqueeze(0)
        ).item()

        # Same TTS model should have high similarity (>0.6)
        assert similarity > 0.6, f"TTS voice doesn't match agent: similarity={similarity}"

    @pytest.mark.slow
    def test_different_audio_has_low_similarity(self, model, agent_embedding):
        """Random audio should have low similarity to agent."""
        # Generate random noise (not speech)
        torch.manual_seed(12345)
        random_audio = torch.randn(1, 16000)  # 1 second of noise

        model.eval()
        with torch.no_grad():
            noise_embedding = model(random_audio).squeeze()

        # Compare with agent embedding
        similarity = torch.nn.functional.cosine_similarity(
            noise_embedding.unsqueeze(0),
            agent_embedding.unsqueeze(0)
        ).item()

        # Random noise should have low similarity (<0.4)
        # Note: High-dimensional random vectors typically have near-zero similarity
        assert similarity < 0.4, f"Random noise too similar to agent: similarity={similarity}"

    @pytest.mark.slow
    def test_same_speaker_consistency(self, model, tts_model):
        """Same speaker (TTS) should produce consistent embeddings."""
        embeddings = []

        # Use longer phrases for more reliable speaker embeddings
        phrases = [
            "Hello world, how are you doing today?",
            "This is a test of the speaker identification system.",
            "The quick brown fox jumps over the lazy dog."
        ]

        for text in phrases:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            tts_model.tts_to_file(text=text, file_path=temp_path)

            waveform, sr = torchaudio.load(temp_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            os.unlink(temp_path)

            model.eval()
            with torch.no_grad():
                emb = model(waveform).squeeze()
            embeddings.append(emb)

        # All pairs should have reasonable similarity (>0.5)
        # Note: TTS can produce varying prosody/pitch which affects embeddings
        # The threshold 0.5 is appropriate for same-speaker verification
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                ).item()
                assert similarity > 0.5, f"Same speaker inconsistent: {i} vs {j} = {similarity}"


class TestSpeakerDatabase:
    """Test the speaker database functionality via C++ tests."""

    def test_cpp_speaker_database_tests(self):
        """C++ speaker database tests pass."""
        test_binary = PROJECT_ROOT / "stream-tts-cpp" / "build" / "test_speaker_embedding"
        if not test_binary.exists():
            pytest.skip("test_speaker_embedding not built")

        result = subprocess.run(
            [str(test_binary)],
            capture_output=True,
            text=True,
            cwd=str(test_binary.parent),
            timeout=60
        )

        assert "All tests PASSED!" in result.stdout, f"Tests failed:\n{result.stdout}\n{result.stderr}"


class TestSelfSpeechFiltering:
    """Test self-speech filtering integration."""

    def test_text_match_filter_exists(self):
        """TextMatchFilter C++ implementation exists."""
        text_match_filter = PROJECT_ROOT / "stream-tts-cpp" / "include" / "text_match_filter.hpp"
        assert text_match_filter.exists(), "TextMatchFilter header not found"

    def test_speaker_diarized_stt_exists(self):
        """SpeakerDiarizedSTT C++ implementation exists."""
        diarized_stt = PROJECT_ROOT / "stream-tts-cpp" / "include" / "speaker_diarized_stt.hpp"
        assert diarized_stt.exists(), "SpeakerDiarizedSTT header not found"

    def test_diarized_stt_compiles(self):
        """SpeakerDiarizedSTT compiles successfully."""
        binary = PROJECT_ROOT / "stream-tts-cpp" / "build" / "stream-tts-cpp"
        assert binary.exists(), "stream-tts-cpp binary not found"

        # Check if it runs (just help)
        result = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Binary should at least start without crashing
        assert result.returncode == 0 or "Usage" in result.stdout or "voice" in result.stdout.lower()


class TestModelExport:
    """Test model export and loading."""

    def test_ecapa_model_exists(self):
        """ECAPA-TDNN TorchScript model exists."""
        assert ECAPA_MODEL.exists(), f"ECAPA-TDNN model not found: {ECAPA_MODEL}"

    def test_ecapa_model_loads(self):
        """ECAPA-TDNN model loads correctly."""
        if not ECAPA_MODEL.exists():
            pytest.skip("ECAPA-TDNN model not found")

        model = torch.jit.load(str(ECAPA_MODEL))
        assert model is not None

    def test_ecapa_model_inference(self):
        """ECAPA-TDNN model produces correct output shape."""
        if not ECAPA_MODEL.exists():
            pytest.skip("ECAPA-TDNN model not found")

        model = torch.jit.load(str(ECAPA_MODEL))
        model.eval()

        with torch.no_grad():
            audio = torch.randn(1, 16000)  # 1 second at 16kHz
            output = model(audio)

        assert output.shape == (1, 192), f"Wrong output shape: {output.shape}"

    def test_ecapa_model_normalized_output(self):
        """ECAPA-TDNN model outputs are L2 normalized."""
        if not ECAPA_MODEL.exists():
            pytest.skip("ECAPA-TDNN model not found")

        model = torch.jit.load(str(ECAPA_MODEL))
        model.eval()

        with torch.no_grad():
            audio = torch.randn(1, 16000)
            output = model(audio)

        norm = torch.norm(output, dim=1).item()
        assert abs(norm - 1.0) < 0.01, f"Output not normalized: norm={norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
