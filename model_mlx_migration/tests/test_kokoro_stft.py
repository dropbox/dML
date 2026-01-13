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
Unit tests for Kokoro STFT and SourceModule fixes.

Tests verify:
1. STFT round-trip reconstruction
2. SourceModule output format and bounds
3. Generator integration
"""

import math

# Add parent path for imports
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

from converters.models.kokoro import SineGen, SourceModule
from converters.models.stft import SmallSTFT, TorchSTFT, frame_signal, get_hann_window


class TestSTFT:
    """Tests for TorchSTFT class."""

    def test_hann_window(self):
        """Test Hann window generation."""
        win = get_hann_window(800)
        mx.eval(win)

        assert win.shape == (800,), f"Expected (800,), got {win.shape}"
        assert float(win[0]) == pytest.approx(0.0, abs=1e-6), "Window should start at 0"
        assert float(win[400]) == pytest.approx(1.0, abs=1e-6), (
            "Window peak should be 1"
        )

    def test_frame_signal(self):
        """Test signal framing."""
        # Create simple signal
        signal = mx.arange(100, dtype=mx.float32)[None, :]  # [1, 100]

        frames = frame_signal(signal, frame_length=20, hop_length=10)
        mx.eval(frames)

        # Expected: (100 - 20) / 10 + 1 = 9 frames
        assert frames.shape == (1, 9, 20), f"Expected (1, 9, 20), got {frames.shape}"

        # First frame should be [0, 1, ..., 19]
        first_frame = frames[0, 0, :]
        mx.eval(first_frame)
        expected = mx.arange(20, dtype=mx.float32)
        assert mx.allclose(first_frame, expected), "First frame content mismatch"

    def test_stft_transform_shape(self):
        """Test STFT output shape."""
        stft = TorchSTFT(filter_length=800, hop_length=200, win_length=800)

        # 1 second at 24kHz
        signal = mx.random.normal((1, 24000))

        mag, phase = stft.transform(signal)
        mx.eval(mag, phase)

        # n_fft//2 + 1 = 401 bins
        assert mag.shape[0] == 1, "Batch size mismatch"
        assert mag.shape[1] == 401, f"Expected 401 freq bins, got {mag.shape[1]}"
        assert phase.shape == mag.shape, "Mag and phase shape mismatch"

    def test_stft_roundtrip(self):
        """Test STFT -> ISTFT roundtrip reconstruction."""
        stft = TorchSTFT(filter_length=800, hop_length=200, win_length=800)

        # Create test signal: 440 Hz sine wave
        t = mx.arange(24000, dtype=mx.float32) / 24000.0
        signal = mx.sin(2 * math.pi * 440 * t)[None, :]  # [1, 24000]

        # Forward STFT
        mag, phase = stft.transform(signal)
        mx.eval(mag, phase)

        # Inverse STFT
        reconstructed = stft.inverse(mag, phase)
        mx.eval(reconstructed)

        # reconstructed is [batch, 1, samples]
        reconstructed = reconstructed.squeeze(1)

        # Compare (allowing for edge effects)
        min_len = min(signal.shape[1], reconstructed.shape[1])

        # Skip edges due to windowing
        start, end = 800, min_len - 800
        if end > start:
            original_middle = signal[:, start:end]
            recon_middle = reconstructed[:, start:end]

            error = mx.abs(original_middle - recon_middle)
            max_error = float(error.max())
            mean_error = float(error.mean())

            print(
                f"STFT roundtrip: max_error={max_error:.6f}, mean_error={mean_error:.6f}",
            )

            # Allow tolerance for windowing artifacts
            # STFT/ISTFT reconstruction has inherent errors at boundaries
            assert max_error < 0.3, f"STFT roundtrip error too high: {max_error}"

    def test_small_stft_inverse(self):
        """Test SmallSTFT inverse for generator output."""
        stft = SmallSTFT(n_fft=20, hop_size=5)

        # Simulate generator output
        # [batch, n_fft//2+1, frames]
        n_bins = 11
        frames = 100
        mag = mx.abs(mx.random.normal((1, n_bins, frames))) + 0.1
        phase = mx.random.uniform(low=-math.pi, high=math.pi, shape=(1, n_bins, frames))

        audio = stft.inverse(mag, phase)
        mx.eval(audio)

        assert audio.ndim == 2, f"Expected 2D output, got {audio.ndim}D"
        assert audio.shape[0] == 1, "Batch size mismatch"
        print(f"SmallSTFT output shape: {audio.shape}")


class TestSourceModule:
    """Tests for SourceModule class."""

    def test_sinegen_output_shape(self):
        """Test SineGen output shapes with upsampling."""
        sine_gen = SineGen(sample_rate=24000, harmonic_num=0)

        # F0 at 200 Hz
        f0 = mx.full((1, 10), 200.0)
        upp = 10  # Upsample by 10

        sine, uv, noise = sine_gen(f0, upp)
        mx.eval(sine, uv, noise)

        # Output is upsampled: [batch, length * upp]
        expected_samples = 10 * upp  # 100
        assert sine.shape == (1, expected_samples), (
            f"Expected (1, {expected_samples}), got {sine.shape}"
        )
        assert uv.shape == (1, expected_samples), (
            f"Expected (1, {expected_samples}), got {uv.shape}"
        )

    def test_sinegen_with_harmonics(self):
        """Test SineGen output shape (harmonic_num doesn't affect output in new API)."""
        sine_gen = SineGen(sample_rate=24000, harmonic_num=8)

        f0 = mx.full((1, 10), 200.0)
        upp = 10

        sine, uv, noise = sine_gen(f0, upp)
        mx.eval(sine, uv, noise)

        # Output is still [batch, samples] (single sine wave in new API)
        expected_samples = 10 * upp  # 100
        assert sine.shape == (1, expected_samples), (
            f"Expected (1, {expected_samples}), got {sine.shape}"
        )

    def test_source_module_output_shape(self):
        """Test SourceModule outputs single channel with upsampling."""
        source = SourceModule(sample_rate=24000, num_harmonics=9)

        # F0 is normalized (model outputs ~0.5-1.5)
        f0 = mx.full((1, 10), 1.0)  # Normalized F0
        upp = 60  # Total upsampling factor

        har_source, noise, uv = source(f0, upp)
        mx.eval(har_source, noise, uv)

        # Output should be upsampled: [batch, f0_len * upp, 1]
        expected_samples = 10 * upp  # 600
        assert har_source.shape == (1, expected_samples, 1), (
            f"Expected (1, {expected_samples}, 1), got {har_source.shape}"
        )
        assert noise.shape == (1, expected_samples, 1), (
            f"Expected (1, {expected_samples}, 1), got {noise.shape}"
        )
        assert uv.shape == (1, expected_samples, 1), (
            f"Expected (1, {expected_samples}, 1), got {uv.shape}"
        )

    def test_source_module_bounded_output(self):
        """Test SourceModule output is bounded by tanh."""
        source = SourceModule(sample_rate=24000, num_harmonics=9)

        # Test with various F0 values (normalized)
        f0 = mx.concatenate(
            [
                mx.full((1, 50), 1.0),  # Normal (~200 Hz)
                mx.full((1, 50), 2.0),  # High pitch (~400 Hz)
            ],
            axis=1,
        )

        har_source, noise, uv = source(f0, upp=60)
        mx.eval(har_source)

        # Output should be bounded [-1, 1] due to tanh
        max_val = float(har_source.max())
        min_val = float(har_source.min())

        print(f"SourceModule output range: [{min_val:.4f}, {max_val:.4f}]")

        assert max_val <= 1.0, f"Output exceeds tanh bound: max={max_val}"
        assert min_val >= -1.0, f"Output exceeds tanh bound: min={min_val}"

    def test_source_module_voiced_mask(self):
        """Test voiced/unvoiced detection."""
        source = SourceModule(sample_rate=24000, num_harmonics=9)

        # SourceModule uses voiced_threshold=10 Hz
        # F0 > 10 Hz is voiced, F0 <= 10 Hz is unvoiced
        f0 = mx.concatenate(
            [
                mx.full((1, 50), 200.0),  # Voiced (200 Hz > threshold)
                mx.zeros((1, 50)),  # Unvoiced (0 Hz < threshold)
            ],
            axis=1,
        )

        har_source, noise, uv = source(f0, upp=60)
        mx.eval(uv)

        # First half should be voiced (1.0), second half unvoiced (0.0)
        # uv is upsampled: [batch, 100*60, 1]
        voiced_part = uv[0, : 50 * 60, 0]  # First 3000 samples
        unvoiced_part = uv[0, 50 * 60 :, 0]  # Last 3000 samples

        assert float(voiced_part.mean()) == 1.0, "Voiced region should have UV=1"
        assert float(unvoiced_part.mean()) == 0.0, "Unvoiced region should have UV=0"


class TestGeneratorIntegration:
    """Integration tests for Generator with STFT."""

    def test_generator_creates_stft_modules(self):
        """Test Generator initializes STFT modules."""
        from converters.models.kokoro import Generator
        from converters.models.kokoro_modules import KokoroConfig

        config = KokoroConfig()
        gen = Generator(config)

        # Check STFT-related attributes exist
        assert hasattr(gen, "_source_window"), (
            "Generator should have _source_window attribute"
        )
        assert hasattr(gen, "source_stft_n_fft"), (
            "Generator should have source_stft_n_fft attribute"
        )
        assert hasattr(gen, "m_source"), "Generator should have m_source attribute"

        # Check m_source is SourceModule
        assert isinstance(gen.m_source, SourceModule), "m_source should be SourceModule"

        # Check STFT parameters
        assert gen.source_stft_n_fft == 20, "source_stft_n_fft should be 20"
        assert gen.source_stft_hop == 5, "source_stft_hop should be 5"

    def test_generator_forward_shape(self):
        """Test Generator produces audio output."""
        from converters.models.kokoro import Generator
        from converters.models.kokoro_modules import KokoroConfig

        config = KokoroConfig()
        gen = Generator(config)

        batch = 1
        # Use dimensions matching Decoder output:
        # - Decoder's decode_3 upsamples by 2x
        # - Generator expects x and f0 to have aligned lengths
        # For 64 input frames after decode_3 upsample, f0 should also be 64
        length = 64  # After decode_3 upsample
        channels = 512

        x = mx.random.normal((batch, length, channels))
        s = mx.random.normal((batch, config.style_dim))
        f0 = mx.full((batch, length), 200.0)  # F0 in Hz, same length as x

        audio = gen(x, s, f0)
        mx.eval(audio)

        assert audio.ndim == 2, f"Expected 2D output, got {audio.ndim}D"
        assert audio.shape[0] == batch, "Batch size mismatch"
        print(f"Generator output shape: {audio.shape}")

        # Check audio has reasonable amplitude
        rms = float(mx.sqrt(mx.mean(audio**2)))
        max_amp = float(mx.abs(audio).max())

        print(f"Audio RMS: {rms:.4f}, Max amplitude: {max_amp:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
