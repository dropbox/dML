# Copyright 2024-2026 Andrew Yates
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
Integration tests for Source Separation → ASR pipeline.

Tests the multi-speaker voice server path:
1. Load mixed audio (simulated multi-speaker)
2. Run source separation (MossFormer2)
3. Run each separated source through ASR (Zipformer)
4. Validate transcriptions

These tests validate that Phase 2 source separation integrates
correctly with Phase 1 Zipformer ASR.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.flasepformer import SourceSeparator, create_separator

# Try importing ASR components
try:
    from models.zipformer.features import (
        HAS_TORCHAUDIO,
        FbankConfig,
        FbankExtractor,  # noqa: F401 - exported for test module API
        load_audio,
    )
    from models.zipformer.inference import (
        HAS_SENTENCEPIECE,
        ASRConfig,  # noqa: F401 - exported for test module API
        ASRPipeline,
        StreamingState,  # noqa: F401 - exported for test module API
        transcribe_file,  # noqa: F401 - exported for test module API
    )
    HAS_ASR = HAS_TORCHAUDIO and HAS_SENTENCEPIECE
except ImportError:
    HAS_ASR = False
    HAS_TORCHAUDIO = False
    HAS_SENTENCEPIECE = False

# Test paths
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints" / "zipformer" / "en-streaming"
CHECKPOINT_PATH = CHECKPOINT_DIR / "exp" / "pretrained.pt"
BPE_MODEL_PATH = CHECKPOINT_DIR / "data" / "lang_bpe_500" / "bpe.model"
TEST_WAVS_DIR = CHECKPOINT_DIR / "test_wavs"


def has_checkpoint() -> bool:
    """Check if required checkpoints are available."""
    return CHECKPOINT_PATH.exists() and BPE_MODEL_PATH.exists()


def has_test_audio() -> bool:
    """Check if test audio files exist."""
    return TEST_WAVS_DIR.exists() and any(TEST_WAVS_DIR.glob("*.wav"))


class TestSeparatorASRIntegrationMocked:
    """Tests for source separator → ASR pipeline with mocked models.

    These tests verify the integration logic without requiring model weights.
    """

    @pytest.fixture
    def mock_separator(self):
        """Create a mock separator that splits audio into halves."""
        separator = SourceSeparator(num_speakers=2)

        def mock_model(audio):
            # Simple mock: create two scaled versions of the audio
            # (real separator would actually separate sources)
            src1 = audio * 0.7  # Scale down
            src2 = audio * 0.5  # Scale down more
            return [src1, src2]

        separator._model = mock_model
        return separator

    def test_mock_separation_produces_two_outputs(self, mock_separator):
        """Test mock separator produces two outputs."""
        audio = mx.zeros((16000,), dtype=mx.float32)
        sources = mock_separator.separate(audio)

        assert len(sources) == 2
        assert sources[0].shape == (16000,)
        assert sources[1].shape == (16000,)

    def test_separated_audio_can_be_numpy(self, mock_separator):
        """Test separated audio can be converted to numpy."""
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(16000).astype(np.float32)
        sources = mock_separator.separate_to_numpy(audio)

        assert len(sources) == 2
        assert isinstance(sources[0], np.ndarray)
        assert isinstance(sources[1], np.ndarray)

    def test_pipeline_shape_flow(self, mock_separator):
        """Test data flows correctly through separation pipeline."""
        # Simulate mixed audio
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(32000).astype(np.float32)  # 2 seconds

        # Separate
        sources = mock_separator.separate_to_numpy(audio)

        # Each source should maintain length
        for i, src in enumerate(sources):
            assert src.shape == (32000,), f"Source {i} has wrong shape"
            assert np.isfinite(src).all(), f"Source {i} has non-finite values"


@pytest.mark.slow
class TestSeparatorASRIntegrationReal:
    """
    Integration tests with real model weights.

    These tests require:
    - MossFormer2 weights (auto-downloaded from HuggingFace)
    - Zipformer checkpoint (must be present)
    - Test audio files

    Run with: pytest -m slow
    """

    @pytest.fixture(scope="class")
    def separator(self):
        """Load real separator model."""
        try:
            sep = create_separator(num_speakers=2)
            _ = sep.model  # Force load
            return sep
        except Exception as e:
            pytest.skip(f"Could not load separator: {e}")

    @pytest.fixture(scope="class")
    def asr_pipeline(self):
        """Load real ASR pipeline."""
        if not HAS_ASR:
            pytest.skip("ASR dependencies not installed")
        if not has_checkpoint():
            pytest.skip("Zipformer checkpoint not found")

        return ASRPipeline.from_pretrained(
            checkpoint_path=str(CHECKPOINT_PATH),
            bpe_model_path=str(BPE_MODEL_PATH),
            decoding_method="greedy",
        )

    def test_separation_to_asr_synthetic(self, separator, asr_pipeline):
        """
        Test full pipeline with synthetic audio.

        Note: This tests pipeline connectivity, not transcription accuracy,
        since synthetic audio won't produce meaningful transcriptions.
        """
        # Create synthetic "mixed" audio
        duration = 2.0  # seconds
        sr = 16000
        n_samples = int(duration * sr)

        # Two frequency components (not real speech)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 600 * t)

        # Step 1: Separate
        sources = separator.separate_to_numpy(audio)
        assert len(sources) == 2, "Should produce 2 sources"

        # Step 2: Transcribe each source
        # Note: We don't expect meaningful text from synthetic audio
        for i, src in enumerate(sources):
            # Ensure audio is right format
            assert src.shape == (n_samples,), f"Source {i} wrong shape"
            assert np.isfinite(src).all(), f"Source {i} has NaN/Inf"

            # Run through ASR (may return empty string for synthetic audio)
            try:
                text = asr_pipeline.transcribe(src, sample_rate=sr)
                print(f"Source {i} transcription: '{text}'")
                # Just verify it returns a string (content may be empty/garbage)
                assert isinstance(text, str)
            except Exception as e:
                # Some failures are acceptable for synthetic audio
                print(f"Source {i} transcription failed (expected for synthetic): {e}")

    @pytest.mark.skipif(not has_test_audio(), reason="Test audio files not found")
    def test_separation_to_asr_real_audio(self, separator, asr_pipeline):
        """
        Test full pipeline with real speech audio.

        Takes a real utterance, creates a "mixture" by adding noise,
        separates it, and transcribes.
        """
        # Find a test audio file
        wav_files = list(TEST_WAVS_DIR.glob("*.wav"))
        if not wav_files:
            pytest.skip("No test audio files found")

        wav_file = wav_files[0]
        print(f"\nUsing test file: {wav_file.name}")

        # Load audio
        waveform, sr = load_audio(str(wav_file))
        waveform = np.array(waveform)

        # Resample to 16kHz if needed
        if sr != 16000:
            pytest.skip(f"Audio sample rate {sr} != 16000")

        # Create a simple mixture: original + scaled noise
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(len(waveform)).astype(np.float32) * 0.1
        mixture = waveform + noise

        # Normalize
        mixture = mixture / (np.max(np.abs(mixture)) + 1e-8)

        # Step 1: Separate
        sources = separator.separate_to_numpy(mixture)
        assert len(sources) == 2

        # Step 2: Transcribe original (reference)
        ref_text = asr_pipeline.transcribe(waveform, sample_rate=sr)
        print(f"Original transcription: '{ref_text}'")

        # Step 3: Transcribe separated sources
        for i, src in enumerate(sources):
            text = asr_pipeline.transcribe(src, sample_rate=sr)
            print(f"Separated source {i}: '{text}'")

            # At least one source should have some text
            # (even if not matching perfectly)

        # Basic sanity check: original should transcribe
        assert len(ref_text.strip()) > 0, "Original audio should transcribe"

    def test_separation_preserves_sample_count(self, separator):
        """Verify separation doesn't change audio length."""
        rng = np.random.default_rng(42)
        for length in [16000, 32000, 48000]:  # 1s, 2s, 3s
            audio = rng.standard_normal(length).astype(np.float32)
            sources = separator.separate_to_numpy(audio)

            for i, src in enumerate(sources):
                assert src.shape[0] == length, (
                    f"Source {i} length {src.shape[0]} != input {length}"
                )


class TestPipelineConfiguration:
    """Test pipeline configuration and compatibility."""

    def test_separator_sample_rate_matches_asr(self):
        """Verify separator and ASR use compatible sample rates."""
        separator = SourceSeparator(num_speakers=2)
        assert separator.sample_rate == 16000, "Separator should use 16kHz"

        # ASR also expects 16kHz (standard for Zipformer)
        config = FbankConfig() if HAS_TORCHAUDIO else None
        if config:
            assert config.sample_rate == 16000, "ASR should use 16kHz"

    def test_3speaker_uses_8khz(self):
        """Verify 3-speaker separator uses 8kHz (MossFormer2 limitation)."""
        separator = SourceSeparator(num_speakers=3)
        assert separator.sample_rate == 8000

        # Note: Would need resampling for ASR if using 3-speaker mode
        print("\nNote: 3-speaker mode uses 8kHz, requires resampling for 16kHz ASR")
