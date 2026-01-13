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
Tests for AdaptiveRouter - ASR-informed audio enhancement.

Run with: pytest tests/test_adaptive_router.py -v
"""

import numpy as np
import pytest

from tools.audio_cleaning.adaptive_router import (
    AdaptiveRouter,
    EnhancementResult,
    SpectralSubtractionDenoiser,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestSpectralSubtractionDenoiser:
    """Test spectral subtraction denoiser."""

    @pytest.fixture
    def denoiser(self):
        return SpectralSubtractionDenoiser()

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def test_denoise_reduces_noise(self, denoiser, sample_rate):
        """Denoising should reduce noise energy."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        # Create noisy signal
        signal = 0.3 * np.sin(2 * np.pi * 440 * t)
        noise = _rng.standard_normal(len(t)) * 0.15
        noisy = (signal + noise).astype(np.float32)

        # Denoise
        denoised = denoiser.denoise(noisy, sample_rate)

        # Check that denoised is same shape
        assert denoised.shape == noisy.shape

        # Check that high-freq noise is reduced
        # Compute energy in 4000-8000 Hz band
        from numpy.fft import rfft, rfftfreq

        freqs = rfftfreq(len(noisy), 1 / sample_rate)
        hf_mask = (freqs > 4000) & (freqs < 8000)

        noisy_spectrum = np.abs(rfft(noisy))
        denoised_spectrum = np.abs(rfft(denoised))

        noisy_hf_energy = np.sum(noisy_spectrum[hf_mask] ** 2)
        denoised_hf_energy = np.sum(denoised_spectrum[hf_mask] ** 2)

        # HF energy should be reduced
        assert denoised_hf_energy < noisy_hf_energy

    def test_denoise_output_valid(self, denoiser, sample_rate):
        """Denoiser output should be valid array of same length."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        # Speech-like signal with amplitude modulation
        envelope = 0.3 + 0.2 * np.sin(2 * np.pi * 3 * t)
        carrier = np.sin(2 * np.pi * 440 * t)
        signal = (envelope * carrier).astype(np.float32)

        # Denoise
        denoised = denoiser.denoise(signal, sample_rate)

        # Should be same length
        assert len(denoised) == len(signal)

        # Should be finite values
        assert np.all(np.isfinite(denoised))

        # Should not be all zeros
        assert np.any(denoised != 0)

    def test_empty_audio(self, denoiser, sample_rate):
        """Empty audio should return empty."""
        empty = np.array([], dtype=np.float32)
        result = denoiser.denoise(empty, sample_rate)
        assert len(result) == 0


class TestAdaptiveRouter:
    """Test adaptive router."""

    @pytest.fixture
    def router(self):
        return AdaptiveRouter(sample_rate=16000)

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def test_clean_signal_bypassed(self, router, sample_rate):
        """Clean signal should bypass enhancement."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = router.route(clean)

        assert result.was_enhanced is False
        assert result.enhancement_type == "none"

    def test_noisy_signal_enhanced(self, router, sample_rate):
        """Noisy signal should be enhanced."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        signal = 0.2 * np.sin(2 * np.pi * 440 * t)
        noise = _rng.standard_normal(len(t)) * 0.2
        noisy = (signal + noise).astype(np.float32)

        result = router.route(noisy)

        # Should attempt enhancement
        assert result.enhancement_type == "denoise"

    def test_force_enhance(self, router, sample_rate):
        """Force enhance should run enhancement even on clean."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = router.route(clean, force_enhance=True)

        # Should have attempted enhancement (but may not use it)
        assert result.enhancement_time_ms > 0

    def test_result_dataclass(self, router, sample_rate):
        """Result should have all expected fields."""
        signal = _rng.standard_normal(sample_rate).astype(np.float32)
        result = router.route(signal)

        assert isinstance(result, EnhancementResult)
        assert isinstance(result.audio, np.ndarray)
        assert isinstance(result.was_enhanced, bool)
        assert isinstance(result.enhancement_type, str)
        assert result.routing_time_ms >= 0
        assert result.enhancement_time_ms >= 0

    def test_harm_tracking(self, router, sample_rate):
        """Harm rate should be tracked."""
        # Process several signals
        for _ in range(5):
            signal = _rng.standard_normal(sample_rate).astype(np.float32)
            router.route(signal)

        # Should have processed something
        assert router.total_processed > 0

    def test_warmup(self, router):
        """Warmup should reset counters."""
        # Process some signals
        signal = _rng.standard_normal(16000).astype(np.float32)
        router.route(signal)
        router.route(signal)

        # Warmup
        router.warmup()

        # Counters should be reset
        assert router.total_processed == 0
        assert router.harm_count == 0

    def test_performance(self, router, sample_rate):
        """Routing should be fast."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))
        signal = _rng.standard_normal(len(t)).astype(np.float32)

        # Warmup
        router.warmup()

        # Time multiple iterations
        times = []
        for _ in range(20):
            result = router.route(signal)
            times.append(result.routing_time_ms + result.enhancement_time_ms)

        # Total time should be reasonable (<50ms for clean, <100ms for noisy)
        p95 = np.percentile(times, 95)
        assert p95 < 100, f"p95 total time should be <100ms, got {p95:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
