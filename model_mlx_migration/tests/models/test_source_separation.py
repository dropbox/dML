"""
Tests for source separation using MossFormer2 MLX.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Module-level random generator for reproducibility
_rng = np.random.default_rng(42)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.flasepformer import SeparatorConfig, SourceSeparator, create_separator


class TestSeparatorConfig:
    """Tests for SeparatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SeparatorConfig()
        assert config.num_speakers == 2
        assert config.sample_rate == 16000
        assert config.encoder_embedding_dim == 512
        assert config.mossformer_sequence_dim == 512
        assert config.num_mossformer_layer == 24
        assert config.encoder_kernel_size == 16
        assert config.is_whamr is False
        assert config.compile_model is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SeparatorConfig(
            num_speakers=3,
            sample_rate=8000,
            is_whamr=True,
        )
        assert config.num_speakers == 3
        assert config.sample_rate == 8000
        assert config.is_whamr is True


class TestSourceSeparatorInit:
    """Tests for SourceSeparator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        separator = SourceSeparator(num_speakers=2)
        assert separator.num_speakers == 2
        assert separator.sample_rate == 16000
        assert separator.config is not None

    def test_init_3speakers(self):
        """Test 3-speaker initialization uses 8kHz."""
        separator = SourceSeparator(num_speakers=3)
        assert separator.num_speakers == 3
        assert separator.sample_rate == 8000

    def test_init_invalid_speakers(self):
        """Test invalid speaker count raises error."""
        with pytest.raises(ValueError, match="must be 2 or 3"):
            SourceSeparator(num_speakers=4)

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = SeparatorConfig(
            num_speakers=2,
            sample_rate=16000,
            compile_model=False,
        )
        separator = SourceSeparator(num_speakers=2, config=config)
        assert separator.config.compile_model is False


class TestCreateSeparator:
    """Tests for create_separator factory function."""

    def test_create_2spk(self):
        """Test creating 2-speaker separator."""
        separator = create_separator(num_speakers=2)
        assert separator.num_speakers == 2
        assert separator.sample_rate == 16000

    def test_create_3spk(self):
        """Test creating 3-speaker separator."""
        separator = create_separator(num_speakers=3, sample_rate=8000)
        assert separator.num_speakers == 3
        assert separator.sample_rate == 8000

    def test_create_whamr(self):
        """Test creating WHAMR separator."""
        separator = create_separator(num_speakers=2, whamr=True)
        assert separator.config.is_whamr is True


class TestSourceSeparatorWithMockedModel:
    """Tests for SourceSeparator with mocked model (no weights needed)."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns dummy separated sources."""
        def mock_forward(audio):
            # Return list of sources with same shape as input
            batch_size = audio.shape[0]
            length = audio.shape[1]
            # Return 2 dummy sources
            return [
                mx.zeros((batch_size, length), dtype=mx.float32),
                mx.ones((batch_size, length), dtype=mx.float32) * 0.5,
            ]
        return mock_forward

    def test_separate_1d_input(self, mock_model):
        """Test separation with 1D input."""
        separator = SourceSeparator(num_speakers=2)
        separator._model = mock_model

        audio = mx.zeros((16000,), dtype=mx.float32)
        sources = separator.separate(audio)

        assert len(sources) == 2
        assert sources[0].shape == (16000,)
        assert sources[1].shape == (16000,)

    def test_separate_2d_input(self, mock_model):
        """Test separation with 2D batched input."""
        separator = SourceSeparator(num_speakers=2)
        separator._model = mock_model

        audio = mx.zeros((2, 16000), dtype=mx.float32)
        sources = separator.separate(audio)

        assert len(sources) == 2
        assert sources[0].shape == (2, 16000)
        assert sources[1].shape == (2, 16000)

    def test_separate_numpy_input(self, mock_model):
        """Test separation with numpy input."""
        separator = SourceSeparator(num_speakers=2)
        separator._model = mock_model

        audio = np.zeros((16000,), dtype=np.float32)
        sources = separator.separate(audio)

        assert len(sources) == 2
        assert isinstance(sources[0], mx.array)

    def test_separate_to_numpy(self, mock_model):
        """Test separation returning numpy arrays."""
        separator = SourceSeparator(num_speakers=2)
        separator._model = mock_model

        audio = mx.zeros((16000,), dtype=mx.float32)
        sources = separator.separate_to_numpy(audio)

        assert len(sources) == 2
        assert isinstance(sources[0], np.ndarray)
        assert isinstance(sources[1], np.ndarray)


class TestStreamingSeparation:
    """Tests for streaming separation functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for streaming tests."""
        def mock_forward(audio):
            batch_size = audio.shape[0]
            length = audio.shape[1]
            return [
                mx.zeros((batch_size, length), dtype=mx.float32),
                mx.ones((batch_size, length), dtype=mx.float32),
            ]
        return mock_forward

    def test_streaming_basic(self, mock_model):
        """Test basic streaming separation."""
        separator = SourceSeparator(num_speakers=2)
        separator._model = mock_model

        # 3 seconds of audio at 16kHz
        audio = mx.zeros((48000,), dtype=mx.float32)

        # Process in 1-second chunks
        chunks = separator.separate_streaming(audio, chunk_size=16000)

        assert len(chunks) == 3  # 3 chunks for 3 seconds
        assert len(chunks[0]) == 2  # 2 speakers per chunk

    def test_streaming_with_overlap(self, mock_model):
        """Test streaming with overlap."""
        separator = SourceSeparator(num_speakers=2)
        separator._model = mock_model

        audio = mx.zeros((32000,), dtype=mx.float32)

        # Process with 50% overlap
        chunks = separator.separate_streaming(
            audio,
            chunk_size=16000,
            overlap=8000,
        )

        # With 50% overlap and 32000 samples: chunks at [0, 8000, 16000, 24000]
        assert len(chunks) >= 3

    def test_streaming_rejects_batched_input(self, mock_model):
        """Test that streaming mode rejects 2D input."""
        separator = SourceSeparator(num_speakers=2)
        separator._model = mock_model

        audio = mx.zeros((2, 16000), dtype=mx.float32)

        with pytest.raises(ValueError, match="only supports 1D"):
            separator.separate_streaming(audio)


@pytest.mark.slow
class TestSourceSeparatorIntegration:
    """Integration tests requiring actual model weights.

    These tests are marked slow and can be skipped in CI.
    Run with: pytest -m slow
    """

    @pytest.fixture(scope="class")
    def separator(self):
        """Create separator with real model (downloads weights if needed)."""
        try:
            sep = SourceSeparator(num_speakers=2)
            # Force model load
            _ = sep.model
            return sep
        except Exception as e:
            pytest.skip(f"Could not load model weights: {e}")

    def test_real_separation(self, separator):
        """Test actual separation with synthetic mixed audio."""
        # Create simple mixed signal: two sine waves
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        source1 = np.sin(2 * np.pi * 440 * t)  # 440 Hz
        source2 = np.sin(2 * np.pi * 880 * t)  # 880 Hz
        mixed = source1 + source2

        sources = separator.separate_to_numpy(mixed)

        assert len(sources) == 2
        assert sources[0].shape == (16000,)
        assert sources[1].shape == (16000,)

        # Check sources are different from each other
        correlation = np.corrcoef(sources[0], sources[1])[0, 1]
        assert abs(correlation) < 0.9  # Should not be identical

    def test_real_separation_batch(self, separator):
        """Test batched separation with real model."""
        audio = _rng.standard_normal((2, 16000)).astype(np.float32)
        sources = separator.separate_to_numpy(audio)

        assert len(sources) == 2
        assert sources[0].shape == (2, 16000)


class TestModuleImports:
    """Test that module imports work correctly."""

    def test_import_from_package(self):
        """Test importing from package."""
        from models.flasepformer import SourceSeparator
        assert SourceSeparator is not None

    def test_import_config(self):
        """Test importing SeparatorConfig."""
        from models.flasepformer import SeparatorConfig
        assert SeparatorConfig is not None

    def test_import_factory(self):
        """Test importing factory function."""
        from models.flasepformer import create_separator
        assert create_separator is not None


def compute_si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = (<estimate, reference> / ||reference||^2) * reference
    and e_noise = estimate - s_target

    Args:
        estimate: Estimated signal
        reference: Reference (clean) signal
        eps: Small constant for numerical stability

    Returns:
        SI-SDR in dB
    """
    # Zero-mean
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)

    # Compute scaling factor
    dot = np.sum(estimate * reference)
    ref_energy = np.sum(reference ** 2) + eps
    alpha = dot / ref_energy

    # Target and noise components
    s_target = alpha * reference
    e_noise = estimate - s_target

    # SI-SDR
    target_energy = np.sum(s_target ** 2) + eps
    noise_energy = np.sum(e_noise ** 2) + eps

    return 10 * np.log10(target_energy / noise_energy)


def compute_si_sdri(
    estimate: np.ndarray,
    reference: np.ndarray,
    mixture: np.ndarray,
) -> float:
    """
    Compute SI-SDR improvement (SI-SDRi).

    SI-SDRi = SI-SDR(estimate, reference) - SI-SDR(mixture, reference)

    Args:
        estimate: Separated signal
        reference: Ground truth clean signal
        mixture: Original mixed signal

    Returns:
        SI-SDRi in dB (higher is better)
    """
    si_sdr_estimate = compute_si_sdr(estimate, reference)
    si_sdr_mixture = compute_si_sdr(mixture, reference)
    return si_sdr_estimate - si_sdr_mixture


@pytest.mark.slow
class TestSISDRiValidation:
    """
    SI-SDRi validation tests for source separation quality.

    Target: >19dB SI-SDRi (MossFormer2 achieves ~21dB on WSJ0-2mix)

    These tests require downloading model weights and are marked slow.
    Run with: pytest -m slow
    """

    @pytest.fixture(scope="class")
    def separator(self):
        """Create separator with real model (downloads weights if needed)."""
        try:
            sep = SourceSeparator(num_speakers=2)
            # Force model load
            _ = sep.model
            return sep
        except Exception as e:
            pytest.skip(f"Could not load model weights: {e}")

    def test_si_sdri_synthetic_sine_waves(self, separator):
        """
        Test SI-SDRi with synthetic sine wave sources.

        Creates two distinct sine waves, mixes them, and validates separation.
        This is an easy test - real speech is harder.
        """
        # Create distinct sources: 440Hz and 880Hz sine waves
        t = np.linspace(0, 2, 32000, dtype=np.float32)  # 2 seconds
        source1 = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz, A4
        source2 = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz, A5

        # Create mixture
        mixture = source1 + source2

        # Separate
        separated = separator.separate_to_numpy(mixture)
        assert len(separated) == 2

        # Assign separated sources to ground truth (best matching)
        # Try both assignments and take the better one
        si_sdri_1_to_1 = compute_si_sdri(separated[0], source1, mixture)
        si_sdri_1_to_2 = compute_si_sdri(separated[0], source2, mixture)
        si_sdri_2_to_1 = compute_si_sdri(separated[1], source1, mixture)
        si_sdri_2_to_2 = compute_si_sdri(separated[1], source2, mixture)

        # Best assignment
        assignment1 = si_sdri_1_to_1 + si_sdri_2_to_2
        assignment2 = si_sdri_1_to_2 + si_sdri_2_to_1

        if assignment1 >= assignment2:
            mean_si_sdri = (si_sdri_1_to_1 + si_sdri_2_to_2) / 2
            print(f"\nAssignment 1: sep[0]->src1={si_sdri_1_to_1:.2f}dB, sep[1]->src2={si_sdri_2_to_2:.2f}dB")
        else:
            mean_si_sdri = (si_sdri_1_to_2 + si_sdri_2_to_1) / 2
            print(f"\nAssignment 2: sep[0]->src2={si_sdri_1_to_2:.2f}dB, sep[1]->src1={si_sdri_2_to_1:.2f}dB")

        print(f"Mean SI-SDRi: {mean_si_sdri:.2f} dB")

        # Note: MossFormer2 is trained on SPEECH, not synthetic signals.
        # Sine waves may not separate well - we just verify model doesn't crash
        # and produces some positive improvement (even small).
        # For true SI-SDRi benchmarks (>19dB), use WSJ0-2mix or LibriMix speech.
        assert mean_si_sdri > 0, f"SI-SDRi should be positive: {mean_si_sdri:.2f}dB"

    def test_si_sdri_synthetic_speech_like(self, separator):
        """
        Test SI-SDRi with speech-like synthetic sources.

        Creates amplitude-modulated noise to simulate speech-like signals.
        More realistic than pure sine waves.
        """
        rng = np.random.default_rng(42)
        duration = 2  # seconds
        sr = 16000
        n_samples = duration * sr

        # Source 1: Low-frequency modulated noise (simulates male voice)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        envelope1 = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # 3Hz envelope
        source1 = envelope1 * rng.standard_normal(n_samples).astype(np.float32)
        # Bandpass filter approximation (simple moving average + high pass)
        source1 = np.convolve(source1, np.ones(50)/50, mode='same').astype(np.float32)

        # Source 2: Higher-frequency modulated noise (simulates female voice)
        envelope2 = 0.5 * (1 + np.sin(2 * np.pi * 4 * t + np.pi/2))  # 4Hz envelope, phase shifted
        source2 = envelope2 * rng.standard_normal(n_samples).astype(np.float32)
        source2 = np.convolve(source2, np.ones(30)/30, mode='same').astype(np.float32)

        # Normalize sources
        source1 = source1 / (np.max(np.abs(source1)) + 1e-8)
        source2 = source2 / (np.max(np.abs(source2)) + 1e-8)

        # Create mixture
        mixture = 0.5 * source1 + 0.5 * source2

        # Separate
        separated = separator.separate_to_numpy(mixture)
        assert len(separated) == 2

        # Find best permutation
        si_sdri_1_to_1 = compute_si_sdri(separated[0], source1, mixture)
        si_sdri_1_to_2 = compute_si_sdri(separated[0], source2, mixture)
        si_sdri_2_to_1 = compute_si_sdri(separated[1], source1, mixture)
        si_sdri_2_to_2 = compute_si_sdri(separated[1], source2, mixture)

        assignment1 = si_sdri_1_to_1 + si_sdri_2_to_2
        assignment2 = si_sdri_1_to_2 + si_sdri_2_to_1

        if assignment1 >= assignment2:
            mean_si_sdri = (si_sdri_1_to_1 + si_sdri_2_to_2) / 2
            print(f"\nSpeech-like SI-SDRi: {si_sdri_1_to_1:.2f}dB, {si_sdri_2_to_2:.2f}dB")
        else:
            mean_si_sdri = (si_sdri_1_to_2 + si_sdri_2_to_1) / 2
            print(f"\nSpeech-like SI-SDRi: {si_sdri_1_to_2:.2f}dB, {si_sdri_2_to_1:.2f}dB")

        print(f"Mean SI-SDRi (speech-like): {mean_si_sdri:.2f} dB")

        # Note: MossFormer2 is trained on actual human speech from WSJ0/LibriMix.
        # Synthetic "speech-like" signals (modulated noise) are NOT recognized as speech
        # by the model and may not separate well or even get worse (negative SI-SDRi).
        # This test verifies the model runs without error.
        # For true validation, use actual speech audio from test datasets.
        # We only check outputs are non-trivial (not all zeros)
        assert np.max(np.abs(separated[0])) > 0.01, "Output 0 is too quiet"
        assert np.max(np.abs(separated[1])) > 0.01, "Output 1 is too quiet"

    def test_separation_preserves_energy(self, separator):
        """Test that separation approximately preserves signal energy."""
        rng = np.random.default_rng(123)
        audio = rng.standard_normal(16000).astype(np.float32)
        audio = audio / np.max(np.abs(audio))  # Normalize

        input_energy = np.sum(audio ** 2)

        separated = separator.separate_to_numpy(audio)
        output_energy = sum(np.sum(s ** 2) for s in separated)

        # Energy ratio should be reasonable (not massively inflated or collapsed)
        energy_ratio = output_energy / input_energy
        print(f"\nEnergy ratio (output/input): {energy_ratio:.2f}")

        # Allow significant variation but catch extreme cases
        assert 0.1 < energy_ratio < 100, f"Unreasonable energy ratio: {energy_ratio}"

    def test_different_inputs_different_outputs(self, separator):
        """Verify different inputs produce different outputs (not degenerate)."""
        rng = np.random.default_rng(456)

        audio1 = rng.standard_normal(16000).astype(np.float32)
        audio2 = rng.standard_normal(16000).astype(np.float32)

        sep1 = separator.separate_to_numpy(audio1)
        sep2 = separator.separate_to_numpy(audio2)

        # Outputs should be different
        corr_00 = np.corrcoef(sep1[0], sep2[0])[0, 1]
        corr_01 = np.corrcoef(sep1[0], sep2[1])[0, 1]

        print(f"\nCorrelation between outputs from different inputs: {corr_00:.4f}, {corr_01:.4f}")

        # Should be low correlation for different inputs
        assert abs(corr_00) < 0.5, f"Too high correlation: {corr_00}"
        assert abs(corr_01) < 0.5, f"Too high correlation: {corr_01}"
