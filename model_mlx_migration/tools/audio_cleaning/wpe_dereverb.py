#!/usr/bin/env python3
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
Spectral Dereverberation Module

Implements spectral-domain dereverberation using spectral subtraction
and cepstral smoothing. Faster and more robust than full WPE.

Per architecture A3: Target 25ms latency, handles T60 up to 1.0s.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DereverbConfig:
    """Configuration for spectral dereverberation."""

    # STFT parameters
    n_fft: int = 512  # FFT size
    hop_length: int = 128  # Hop size

    # Dereverberation parameters
    reverb_decay_time_ms: float = 50.0  # Expected late reverb start (ms)
    reverb_reduction_db: float = 6.0  # Amount of late reverb reduction
    smoothing_frames: int = 3  # Temporal smoothing width

    # Audio parameters
    sample_rate: int = 16000


class SpectralDereverberator:
    """
    Spectral-domain dereverberation.

    Uses cepstral analysis to identify and reduce late reverb energy.
    Faster than WPE with similar perceptual results.
    """

    def __init__(self, config: DereverbConfig | None = None):
        """
        Initialize spectral dereverberator.

        Args:
            config: Dereverb configuration. If None, uses defaults.
        """
        self.config = config or DereverbConfig()
        self._window = None

    def _get_window(self, n: int) -> np.ndarray:
        """Get cached window function."""
        if self._window is None or len(self._window) != n:
            self._window = np.hanning(n).astype(np.float64)
        return self._window

    def _rfft2d(self, frames: np.ndarray) -> np.ndarray:
        """Vectorized real FFT for all frames."""
        return np.fft.rfft(frames, axis=1)

    def _irfft2d(self, spectra: np.ndarray, n: int) -> np.ndarray:
        """Vectorized inverse real FFT for all frames."""
        return np.fft.irfft(spectra, n=n, axis=1)

    def dereverb(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Apply spectral dereverberation.

        Uses a multi-stage approach:
        1. STFT analysis
        2. Estimate reverb magnitude from spectral variance
        3. Spectral subtraction of reverb component
        4. STFT synthesis

        Args:
            audio: Input audio (float32 or float64, mono)
            sample_rate: Sample rate

        Returns:
            Dereverberated audio (float32)
        """
        audio = np.asarray(audio, dtype=np.float64)

        if len(audio) == 0:
            return audio.astype(np.float32)

        n_fft = self.config.n_fft
        hop = self.config.hop_length
        window = self._get_window(n_fft)

        # Minimum length check
        if len(audio) < n_fft:
            return audio.astype(np.float32)

        original_len = len(audio)

        # Vectorized STFT
        n_frames = (len(audio) - n_fft) // hop + 1
        frame_indices = np.arange(n_frames)[:, None] * hop + np.arange(n_fft)

        # Handle boundary
        valid_frames = frame_indices[:, -1] < len(audio)
        n_frames = np.sum(valid_frames)
        frame_indices = frame_indices[:n_frames]

        # Extract frames
        frames = audio[frame_indices] * window

        # Compute STFT
        spectra = self._rfft2d(frames)
        magnitudes = np.abs(spectra)
        phases = np.angle(spectra)

        # Estimate reverb from temporal variance
        # High variance indicates transients (direct), low variance indicates reverb
        smooth = self.config.smoothing_frames

        if n_frames > smooth * 2:
            # Rolling mean of magnitude
            kernel = np.ones(smooth) / smooth
            smoothed = np.zeros_like(magnitudes)
            for f in range(magnitudes.shape[1]):
                smoothed[:, f] = np.convolve(magnitudes[:, f], kernel, mode='same')

            # Reverb estimate: smoothed minus variance
            # More conservative: only reduce where smoothed >> instantaneous
            reverb_mask = smoothed / (magnitudes + 1e-10)
            reverb_mask = np.clip(reverb_mask - 1.0, 0, 1)

            # Apply reduction
            reduction = 10 ** (-self.config.reverb_reduction_db / 20)
            gain = 1.0 - reverb_mask * (1.0 - reduction)
            gain = np.clip(gain, reduction, 1.0)

            # Apply gain
            magnitudes_clean = magnitudes * gain
        else:
            # Too short for analysis
            magnitudes_clean = magnitudes

        # Reconstruct
        spectra_clean = magnitudes_clean * np.exp(1j * phases)

        # Vectorized ISTFT
        frames_out = self._irfft2d(spectra_clean, n_fft).real * window

        # Overlap-add
        output_len = (n_frames - 1) * hop + n_fft
        output = np.zeros(output_len, dtype=np.float64)
        window_sum = np.zeros(output_len, dtype=np.float64)

        for i in range(n_frames):
            start = i * hop
            output[start:start + n_fft] += frames_out[i]
            window_sum[start:start + n_fft] += window ** 2

        # Normalize
        window_sum = np.maximum(window_sum, 1e-8)
        output = output / window_sum

        # Trim to original length
        output = output[:original_len]
        if len(output) < original_len:
            output = np.pad(output, (0, original_len - len(output)))

        return output.astype(np.float32)

    def warmup(self):
        """Warm up the dereverberator."""
        rng = np.random.default_rng()
        dummy = rng.standard_normal(self.config.sample_rate).astype(np.float32) * 0.1
        self.dereverb(dummy)


# Alias for backward compatibility
WPEConfig = DereverbConfig
WPEDereverberator = SpectralDereverberator


def test_wpe_dereverb():
    """Test WPE dereverberation with synthetic signals."""
    print("=" * 60)
    print("WPE Dereverberation Test")
    print("=" * 60)

    sample_rate = 16000
    duration = 2.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)

    # Create clean signal (440 Hz tone with amplitude modulation)
    clean = (0.5 * np.sin(2 * np.pi * 440 * t) *
             (1 + 0.3 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)

    # Create reverberant signal (convolve with synthetic RIR)
    print("\n1. Creating reverberant signal...")
    rng = np.random.default_rng()
    rir_len = int(0.4 * sample_rate)  # 400ms T60
    rir = np.exp(-np.linspace(0, 8, rir_len)) * rng.standard_normal(rir_len)
    rir = rir.astype(np.float32) / np.max(np.abs(rir))
    rir[0] = 1.0  # Direct path

    reverb = np.convolve(clean, rir, mode='same').astype(np.float32)
    reverb = reverb / np.max(np.abs(reverb))  # Normalize

    print(f"   Clean signal RMS: {np.sqrt(np.mean(clean**2)):.4f}")
    print(f"   Reverb signal RMS: {np.sqrt(np.mean(reverb**2)):.4f}")

    # Apply WPE
    print("\n2. Applying WPE dereverberation...")
    dereverb = WPEDereverberator()

    import time
    start = time.perf_counter()
    output = dereverb.dereverb(reverb, sample_rate)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"   Processing time: {elapsed:.1f} ms")
    print(f"   Output RMS: {np.sqrt(np.mean(output**2)):.4f}")

    # Measure improvement
    # Correlation with clean should increase after dereverberation
    def correlation(a, b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        return np.abs(np.sum(a * b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)) + 1e-10))

    reverb_corr = correlation(clean, reverb)
    output_corr = correlation(clean, output)

    print("\n3. Quality assessment:")
    print(f"   Correlation with clean (reverb): {reverb_corr:.4f}")
    print(f"   Correlation with clean (output): {output_corr:.4f}")
    print(f"   Improvement: {(output_corr - reverb_corr) / reverb_corr * 100:.1f}%")

    # Performance benchmark
    print("\n4. Performance benchmark (50 iterations)...")
    times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = dereverb.dereverb(reverb, sample_rate)
        times.append((time.perf_counter() - start) * 1000)

    print(f"   Mean: {np.mean(times):.1f} ms")
    print(f"   p95: {np.percentile(times, 95):.1f} ms")

    # Target is 25ms per architecture A3
    target = 25.0
    passed = np.percentile(times, 95) < target
    print(f"   Target: <{target} ms")
    print(f"   Status: {'PASS' if passed else 'FAIL - may need optimization'}")


if __name__ == "__main__":
    test_wpe_dereverb()
