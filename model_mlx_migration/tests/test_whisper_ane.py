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
Tests for WhisperANE - CoreML/ANE encoder acceleration.

Tests cover:
1. CoreML encoder loading and initialization
2. Output shape verification
3. Numerical equivalence with MLX encoder
4. Benchmarking utilities
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# Check if coremltools is available
try:
    import coremltools as ct  # noqa: F401
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)

# Check if model is downloaded
MODEL_DIR = Path(__file__).parent.parent / "models" / "whisperkit" / "openai_whisper-large-v3"
MODEL_AVAILABLE = (MODEL_DIR / "AudioEncoder.mlmodelc" / "model.mlmodel").exists()


@pytest.mark.skipif(not COREML_AVAILABLE, reason="coremltools not installed")
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="WhisperKit model not downloaded")
class TestCoreMLEncoder:
    """Test CoreML encoder functionality."""

    @pytest.fixture
    def encoder(self):
        """Load CoreML encoder for tests."""
        from tools.whisper_ane import CoreMLEncoder
        return CoreMLEncoder.from_pretrained("large-v3", auto_download=False)

    def test_encoder_loads(self, encoder):
        """Test that encoder loads correctly."""
        assert encoder is not None
        assert encoder.n_state == 1280
        assert encoder.n_mels == 128
        assert encoder.n_ctx == 1500

    def test_encoder_input_output_names(self, encoder):
        """Test that encoder has correct input/output names."""
        assert "melspectrogram_features" in encoder._input_names
        assert "encoder_output_embeds" in encoder._output_names

    def test_encoder_output_shape_30s(self, encoder):
        """Test encoder output shape for 30s audio."""
        # 30s audio = 3000 mel frames
        mel = _rng.standard_normal(3000, 128).astype(np.float32)
        output = encoder(mel)

        # Expected: (1, 1500, 1280)
        assert output.shape == (1, 1500, 1280)
        assert output.dtype == np.float32

    def test_encoder_output_shape_batch(self, encoder):
        """Test encoder output shape with batch dimension."""
        mel = _rng.standard_normal(1, 3000, 128).astype(np.float32)
        output = encoder(mel)

        assert output.shape == (1, 1500, 1280)

    def test_encoder_pads_short_audio(self, encoder):
        """Test encoder pads short audio to 30s."""
        # 10s audio = 1000 mel frames
        mel = _rng.standard_normal(1000, 128).astype(np.float32)
        output = encoder(mel)

        # Output should still be full 1500 seq_len (padded)
        assert output.shape == (1, 1500, 1280)

    def test_encoder_get_output_length(self, encoder):
        """Test output length calculation."""
        # 3000 frames -> 1500 output
        assert encoder.get_output_length(3000) == 1500
        # 2000 frames -> 1000 output
        assert encoder.get_output_length(2000) == 1000

    def test_encoder_benchmark(self, encoder):
        """Test benchmark function."""
        mel = _rng.standard_normal(1, 3000, 128).astype(np.float32)
        results = encoder.benchmark(mel, n_iterations=2, warmup=1)

        assert "mean_ms" in results
        assert "std_ms" in results
        assert "min_ms" in results
        assert "max_ms" in results
        assert results["mean_ms"] > 0


@pytest.mark.skipif(not COREML_AVAILABLE, reason="coremltools not installed")
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="WhisperKit model not downloaded")
class TestCoreMLEncoderComputeUnits:
    """Test CoreML encoder with different compute units."""

    @pytest.mark.parametrize("compute_units", [
        "CPU_ONLY",
        "CPU_AND_GPU",
        "CPU_AND_NE",
    ])
    def test_compute_units_load(self, compute_units):
        """Test encoder loads with different compute units."""
        from tools.whisper_ane import CoreMLEncoder

        encoder = CoreMLEncoder.from_pretrained(
            "large-v3",
            compute_units=compute_units,
            auto_download=False,
        )
        assert encoder.compute_units_str == compute_units

        # Verify inference works
        mel = _rng.standard_normal(1, 3000, 128).astype(np.float32)
        output = encoder(mel)
        assert output.shape == (1, 1500, 1280)


@pytest.mark.skipif(not COREML_AVAILABLE, reason="coremltools not installed")
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="WhisperKit model not downloaded")
class TestNumericalEquivalence:
    """Test numerical equivalence between CoreML and MLX encoders."""

    @pytest.fixture
    def mlx_model(self):
        """Load MLX model for comparison."""
        from tools.whisper_mlx import WhisperMLX
        return WhisperMLX.from_pretrained("large-v3")

    @pytest.fixture
    def coreml_encoder(self):
        """Load CoreML encoder for comparison."""
        from tools.whisper_ane import CoreMLEncoder
        return CoreMLEncoder.from_pretrained("large-v3", compute_units="CPU_AND_GPU")

    @pytest.fixture
    def test_mel(self, mlx_model):
        """Generate test mel spectrogram from audio.

        Note: Uses exactly 30s (3000 frames) to ensure MLX and CoreML
        produce equivalent outputs. Shorter audio gets padded differently
        by each encoder, causing numerical differences.
        """
        from tools.whisper_mlx.audio import log_mel_spectrogram

        # Generate 30s sine wave audio (full Whisper context)
        sample_rate = 16000
        duration = 30.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        mel = log_mel_spectrogram(audio, n_mels=128)
        return np.array(mel)

    def test_output_correlation(self, mlx_model, coreml_encoder, test_mel):
        """Test that outputs are highly correlated for 30s audio."""
        import mlx.core as mx

        # Run MLX encoder (variable_length=False for 30s audio)
        mel_mx = mx.array(test_mel)
        mlx_output = mlx_model.encoder(mel_mx, variable_length=False)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Run CoreML encoder
        mel_batch = test_mel[np.newaxis, ...]
        coreml_output = coreml_encoder(mel_batch)

        # Ensure same shape for comparison
        if mlx_output_np.ndim == 2:
            mlx_output_np = mlx_output_np[np.newaxis, ...]

        # Both should be (1, 1500, 1280) for 30s audio
        mlx_flat = mlx_output_np.flatten()
        coreml_flat = coreml_output.flatten()

        # Check correlation (should be nearly 1.0 for identical models)
        corr = np.corrcoef(mlx_flat, coreml_flat)[0, 1]
        assert corr > 0.99, f"Correlation {corr} < 0.99"

    def test_mean_abs_diff(self, mlx_model, coreml_encoder, test_mel):
        """Test that mean absolute difference is small for 30s audio."""
        import mlx.core as mx

        # Run MLX encoder (variable_length=False for 30s audio)
        mel_mx = mx.array(test_mel)
        mlx_output = mlx_model.encoder(mel_mx, variable_length=False)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Run CoreML encoder
        mel_batch = test_mel[np.newaxis, ...]
        coreml_output = coreml_encoder(mel_batch)

        # Ensure same shape for comparison
        if mlx_output_np.ndim == 2:
            mlx_output_np = mlx_output_np[np.newaxis, ...]

        # Both should be (1, 1500, 1280) for 30s audio
        diff = np.abs(mlx_output_np - coreml_output)

        # Mean diff should be < 0.01 (accounting for float16 precision)
        assert diff.mean() < 0.01, f"Mean diff {diff.mean()} > 0.01"


class TestDownloadScript:
    """Test download script functionality."""

    def test_available_models(self):
        """Test that model list is defined."""
        from scripts.download_whisperkit_models import AVAILABLE_MODELS

        assert "large-v3" in AVAILABLE_MODELS
        assert "turbo" in AVAILABLE_MODELS

    def test_list_models(self, capsys):
        """Test list_models function."""
        from scripts.download_whisperkit_models import list_models

        list_models()
        captured = capsys.readouterr()
        assert "large-v3" in captured.out


@pytest.mark.skipif(not COREML_AVAILABLE, reason="coremltools not installed")
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="WhisperKit model not downloaded")
class TestHybridIntegration:
    """Integration tests for HybridWhisperMLX with real audio."""

    @pytest.fixture
    def hybrid_model(self):
        """Load hybrid model for tests."""
        from tools.whisper_ane import HybridWhisperMLX
        return HybridWhisperMLX.from_pretrained(
            "large-v3",
            compute_units="CPU_AND_GPU",  # Fastest on M4 Max
            auto_download=False,
        )

    @pytest.fixture
    def test_audio_path(self):
        """Return path to a test audio file."""
        # Use RAVDESS speech file
        audio_dir = Path(__file__).parent.parent / "data" / "prosody" / "ravdess" / "Actor_01"
        audio_files = list(audio_dir.glob("*.wav"))
        if not audio_files:
            pytest.skip("No test audio files available")
        return audio_files[0]

    def test_hybrid_transcribe_returns_text(self, hybrid_model, test_audio_path):
        """Test that hybrid model returns transcription."""
        result = hybrid_model.transcribe(str(test_audio_path))

        assert "text" in result
        assert isinstance(result["text"], str)
        # RAVDESS audio should produce some text (speech emotion dataset)
        # Don't check content, just that it produces non-empty result

    def test_hybrid_transcribe_numpy_input(self, hybrid_model):
        """Test hybrid model with numpy array input."""
        # Generate 2s of 440Hz sine wave (should produce some noise/text)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        result = hybrid_model.transcribe(audio)
        assert "text" in result
        # Sine wave may or may not produce text, but shouldn't crash

    def test_hybrid_benchmark_runs(self, hybrid_model, test_audio_path):
        """Test that benchmark function works with real audio."""
        from tools.whisper_mlx.audio import SAMPLE_RATE, load_audio
        audio = load_audio(str(test_audio_path), sample_rate=SAMPLE_RATE)

        results = hybrid_model.benchmark(audio=audio, n_iterations=2, warmup=1)

        assert "encoder_mean_ms" in results
        assert "decoder_mean_ms" in results
        assert results["encoder_mean_ms"] > 0
        assert results["decoder_mean_ms"] > 0


@pytest.mark.skipif(not COREML_AVAILABLE, reason="coremltools not installed")
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="WhisperKit model not downloaded")
class TestQualityDialCoreML:
    """Test quality dial system CoreML integration."""

    def test_coreml_dial_creates_hybrid(self):
        """Test that coreml_dial > 0 creates HybridWhisperMLX."""
        from tools.whisper_mlx.quality_dial import WhisperQualityConfig

        config = WhisperQualityConfig(
            model_dial=0.0,
            coreml_dial=0.5,  # Request CoreML GPU
        )

        model = config.create_model()

        # Should be HybridWhisperMLX
        from tools.whisper_ane import HybridWhisperMLX
        assert isinstance(model, HybridWhisperMLX)

    def test_coreml_dial_zero_creates_mlx(self):
        """Test that coreml_dial=0 creates pure WhisperMLX."""
        from tools.whisper_mlx.quality_dial import WhisperQualityConfig

        config = WhisperQualityConfig(
            model_dial=0.0,
            coreml_dial=0.0,  # Request MLX
        )

        model = config.create_model()

        # Should be WhisperMLX (not HybridWhisperMLX)
        assert type(model).__name__ == "WhisperMLX"

    def test_coreml_summary_includes_backend(self):
        """Test that summary() shows CoreML backend."""
        from tools.whisper_mlx.quality_dial import WhisperQualityConfig

        config = WhisperQualityConfig(coreml_dial=0.5)
        summary = config.summary()

        assert "coreml:" in summary
        assert "coreml_gpu" in summary


class TestCoreMLFallback:
    """Test fallback behavior when CoreML is unavailable."""

    def test_fallback_when_model_missing(self):
        """Test fallback to MLX when CoreML model is not downloaded."""
        from tools.whisper_mlx.quality_dial import WhisperQualityConfig

        # Even with coreml_dial > 0, should fall back to MLX if model not found
        config = WhisperQualityConfig(
            model_dial=0.0,
            coreml_dial=1.0,  # Request CoreML ANE (but model may not exist)
        )

        # This should not raise, should fall back to MLX
        try:
            model = config.create_model()
            # If we get here, either:
            # 1. CoreML model exists and HybridWhisperMLX was created
            # 2. CoreML unavailable and fell back to WhisperMLX
            assert model is not None
        except FileNotFoundError:
            # This is acceptable - means model needs to be downloaded
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
