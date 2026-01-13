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
Integration tests for Preprocessing Pipeline → Zipformer ASR.

Tests the full voice server audio path:
1. Load audio file
2. Preprocess (resample, DC removal, LUFS normalize)
3. Chunk for streaming
4. Run through Zipformer encoder/decoder
5. Validate transcription output

These tests validate the Phase 3 preprocessing works correctly
with the Phase 1 Zipformer implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipeline,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)

# Checkpoint paths
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "zipformer" / "en-streaming"
CHECKPOINT_PATH = CHECKPOINT_DIR / "exp" / "pretrained.pt"
BPE_MODEL_PATH = CHECKPOINT_DIR / "data" / "lang_bpe_500" / "bpe.model"
TEST_WAVS_DIR = CHECKPOINT_DIR / "test_wavs"


def _has_zipformer_checkpoint() -> bool:
    """Check if Zipformer checkpoint is available."""
    return CHECKPOINT_PATH.exists() and BPE_MODEL_PATH.exists()


def _has_test_audio() -> bool:
    """Check if test audio files exist."""
    if not TEST_WAVS_DIR.exists():
        return False
    wav_files = list(TEST_WAVS_DIR.glob("*.wav"))
    return len(wav_files) > 0


def _has_dependencies() -> bool:
    """Check if all dependencies are available."""
    try:
        import mlx.core as mx  # noqa: F401 - checking availability
        import sentencepiece  # noqa: F401 - checking availability
        import torch  # noqa: F401 - checking availability
        import torchaudio  # noqa: F401 - checking availability
        return True
    except ImportError:
        return False


class TestPreprocessingZipformerIntegration:
    """Test preprocessing pipeline integration with Zipformer ASR."""

    @pytest.fixture
    def pipeline(self):
        """Create preprocessing pipeline."""
        config = PreprocessingConfig(
            target_sample_rate=16000,
            enable_vad=False,  # Disable VAD for simpler testing
            chunk_size_ms=320,  # Match Zipformer streaming config
        )
        return PreprocessingPipeline(config)

    @pytest.mark.skipif(
        not _has_dependencies(),
        reason="Dependencies not available",
    )
    @pytest.mark.skipif(
        not _has_zipformer_checkpoint(),
        reason="Zipformer checkpoint not available",
    )
    @pytest.mark.skipif(
        not _has_test_audio(),
        reason="Test audio files not available",
    )
    def test_preprocessed_audio_works_with_zipformer(self, pipeline):
        """Preprocessed audio should work with Zipformer ASR."""
        from src.models.preprocessing.pipeline import load_audio
        from src.models.zipformer.inference import ASRPipeline

        # Load ASR model
        asr = ASRPipeline.from_pretrained(
            checkpoint_path=str(CHECKPOINT_PATH),
            bpe_model_path=str(BPE_MODEL_PATH),
        )

        # Get first test file
        wav_files = list(TEST_WAVS_DIR.glob("*.wav"))
        audio_path = wav_files[0]

        # Load and preprocess audio
        audio, sr = load_audio(str(audio_path))
        preprocessed = pipeline.preprocess(audio, sr)

        # Run ASR on preprocessed audio (via internal pipeline that extracts features)
        text = asr.transcribe(preprocessed, sample_rate=16000)

        # Should produce non-empty output
        assert text is not None
        assert len(text) > 0
        print(f"\nTranscribed: '{text}'")

    @pytest.mark.skipif(
        not _has_dependencies(),
        reason="Dependencies not available",
    )
    @pytest.mark.skipif(
        not _has_zipformer_checkpoint(),
        reason="Zipformer checkpoint not available",
    )
    @pytest.mark.skipif(
        not _has_test_audio(),
        reason="Test audio files not available",
    )
    def test_streaming_preprocessing_with_zipformer(self, pipeline):
        """Streaming preprocessed chunks should work with streaming ASR."""
        from src.models.preprocessing.pipeline import load_audio
        from src.models.zipformer.inference import ASRPipeline

        # Load ASR model
        asr = ASRPipeline.from_pretrained(
            checkpoint_path=str(CHECKPOINT_PATH),
            bpe_model_path=str(BPE_MODEL_PATH),
        )

        # Get first test file
        wav_files = list(TEST_WAVS_DIR.glob("*.wav"))
        audio_path = wav_files[0]

        # Load audio
        audio, sr = load_audio(str(audio_path))

        # Process through streaming preprocessing
        chunks = list(pipeline.process_streaming(audio, sr))

        # Should produce chunks
        assert len(chunks) > 0
        print(f"\nGenerated {len(chunks)} chunks")

        # Each chunk should have correct sample rate
        for chunk in chunks:
            assert chunk.sample_rate == 16000

        # Transcribe using streaming ASR
        state = asr.init_streaming()
        accumulated_text = ""

        for chunk in chunks:
            # Extract features for this chunk
            features = asr.fbank_extractor.extract(chunk.samples, chunk.sample_rate)

            # Run streaming transcription
            partial_text, state = asr.transcribe_chunk(
                features,
                state,
                valid_fbank_frames=features.shape[0],
            )
            accumulated_text += partial_text

        # Decode final accumulated tokens
        final_text = asr.sp.DecodeIds(state.accumulated_tokens)

        assert final_text is not None
        print(f"Streaming transcription: '{final_text}'")

    @pytest.mark.skipif(
        not _has_dependencies(),
        reason="Dependencies not available",
    )
    def test_preprocessing_output_format(self, pipeline):
        """Preprocessing output should be in correct format for Zipformer."""
        # Generate synthetic audio at 48kHz
        duration = 1.0
        sr = 48000
        t = np.linspace(0, duration, int(sr * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Preprocess
        preprocessed = pipeline.preprocess(audio, sr)

        # Should be at 16kHz
        assert len(preprocessed) == 16000

        # Should be floating point (float32 or float64 both work)
        assert np.issubdtype(preprocessed.dtype, np.floating)

        # Should be 1D
        assert preprocessed.ndim == 1

        # Should be normalized (not clipped)
        assert np.max(np.abs(preprocessed)) < 10.0  # Reasonable range

    @pytest.mark.skipif(
        not _has_dependencies(),
        reason="Dependencies not available",
    )
    def test_chunk_sizes_match_zipformer_config(self, pipeline):
        """Chunk sizes should match Zipformer streaming configuration."""
        # 1 second of audio at 16kHz
        audio = _rng.standard_normal(16000).astype(np.float32)

        chunks = list(pipeline.chunk_audio(audio, 16000))

        # Each chunk should be 320ms = 5120 samples
        expected_chunk_samples = int(320 * 16000 / 1000)

        for chunk in chunks[:-1]:  # All but last
            assert len(chunk.samples) == expected_chunk_samples, (
                f"Chunk size {len(chunk.samples)} != expected {expected_chunk_samples}"
            )


class TestLatencyTargets:
    """Test that preprocessing meets latency requirements."""

    @pytest.fixture
    def pipeline(self):
        """Create preprocessing pipeline with VAD disabled for speed."""
        config = PreprocessingConfig(
            target_sample_rate=16000,
            enable_vad=False,
        )
        return PreprocessingPipeline(config)

    @pytest.mark.skipif(
        not _has_dependencies(),
        reason="Dependencies not available",
    )
    def test_preprocessing_under_15ms(self, pipeline):
        """Preprocessing should complete in <15ms for 1 second audio."""
        # 1 second audio at 48kHz (typical input)
        audio = _rng.standard_normal(48000).astype(np.float32)

        # Benchmark
        results = pipeline.benchmark(audio, 48000, iterations=5)

        print("\nPreprocessing benchmark:")
        print(f"  Resample: {results['resample_ms']:.2f}ms")
        print(f"  DC removal: {results['dc_removal_ms']:.2f}ms")
        print(f"  LUFS: {results['lufs_normalize_ms']:.2f}ms")
        print(f"  Total: {results['total_preprocess_ms']:.2f}ms")
        print(f"  RTF: {results['real_time_factor']:.4f}x")

        # Target: <15ms total
        assert results['total_preprocess_ms'] < 15.0, (
            f"Preprocessing latency {results['total_preprocess_ms']:.2f}ms exceeds 15ms target"
        )


class TestDenoisingDisabled:
    """Verify denoising is disabled (critical for ASR quality)."""

    def test_denoising_cannot_be_enabled(self):
        """Enabling denoising should raise an error."""
        config = PreprocessingConfig(enable_denoising=True)

        with pytest.raises(ValueError, match="Denoising MUST be disabled"):
            PreprocessingPipeline(config)

    def test_default_config_has_denoising_disabled(self):
        """Default config should have denoising disabled."""
        config = PreprocessingConfig()
        assert config.enable_denoising is False

    def test_pipeline_created_without_denoising(self):
        """Pipeline should be created with denoising disabled."""
        pipeline = PreprocessingPipeline()
        assert pipeline.config.enable_denoising is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
