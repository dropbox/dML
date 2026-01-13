#!/usr/bin/env python3
"""Tests for WPE dereverberation."""

import numpy as np
import pytest

from tools.audio_cleaning.wpe_dereverb import (
    DereverbConfig,
    WPEConfig,
    WPEDereverberator,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestDereverbConfig:
    """Test dereverberation configuration."""

    def test_default_config(self):
        """Test default config values."""
        config = DereverbConfig()
        assert config.n_fft == 512
        assert config.hop_length == 128
        assert config.reverb_decay_time_ms == 50.0
        assert config.reverb_reduction_db == 6.0
        assert config.smoothing_frames == 3
        assert config.sample_rate == 16000

    def test_custom_config(self):
        """Test custom config values."""
        config = DereverbConfig(n_fft=1024, reverb_reduction_db=10.0)
        assert config.n_fft == 1024
        assert config.reverb_reduction_db == 10.0

    def test_backward_compatibility_alias(self):
        """Test WPEConfig alias for backward compatibility."""
        assert WPEConfig is DereverbConfig


class TestWPEDereverberator:
    """Test WPE dereverberator."""

    @pytest.fixture
    def dereverb(self):
        """Create dereverberator instance."""
        return WPEDereverberator()

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Amplitude-modulated tone
        return (0.5 * np.sin(2 * np.pi * 440 * t) *
                (1 + 0.3 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)

    def test_dereverb_clean_audio(self, dereverb, sample_audio):
        """Test dereverb on clean audio produces similar output."""
        output = dereverb.dereverb(sample_audio, 16000)
        assert len(output) == len(sample_audio)
        assert output.dtype == np.float32
        # Clean audio shouldn't be dramatically changed
        # Check RMS is preserved within reasonable bounds
        input_rms = np.sqrt(np.mean(sample_audio ** 2))
        output_rms = np.sqrt(np.mean(output ** 2))
        assert 0.3 * input_rms < output_rms < 3.0 * input_rms

    def test_dereverb_reverberant_audio(self, dereverb, sample_audio):
        """Test dereverb on reverberant audio."""
        sample_rate = 16000
        # Create synthetic room impulse response
        rir_len = int(0.4 * sample_rate)  # 400ms T60
        rir = np.exp(-np.linspace(0, 8, rir_len)) * _rng.standard_normal(rir_len)
        rir = rir.astype(np.float32) / np.max(np.abs(rir))
        rir[0] = 1.0  # Direct path

        # Create reverberant audio
        reverb = np.convolve(sample_audio, rir, mode="same").astype(np.float32)

        # Apply dereverberation
        output = dereverb.dereverb(reverb, sample_rate)

        assert len(output) == len(reverb)
        assert output.dtype == np.float32
        # Output should be valid (no NaN/Inf)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_dereverb_empty_audio(self, dereverb):
        """Test dereverb handles empty audio."""
        empty = np.array([], dtype=np.float32)
        output = dereverb.dereverb(empty, 16000)
        assert len(output) == 0

    def test_dereverb_short_audio(self, dereverb):
        """Test dereverb handles audio shorter than FFT size."""
        short = _rng.standard_normal(256).astype(np.float32)
        output = dereverb.dereverb(short, 16000)
        # Should return original if too short
        assert len(output) == len(short)

    def test_performance_under_25ms(self, dereverb, sample_audio):
        """Test performance target: <25ms p95 for 2s audio."""
        import time
        times = []
        for _ in range(20):
            start = time.perf_counter()
            dereverb.dereverb(sample_audio, 16000)
            times.append((time.perf_counter() - start) * 1000)

        p95 = np.percentile(times, 95)
        # Architecture A3 target: <25ms
        assert p95 < 25.0, f"p95 latency {p95:.1f}ms exceeds 25ms target"

    def test_warmup(self, dereverb):
        """Test warmup runs without error."""
        dereverb.warmup()  # Should not raise


class TestAdaptiveRouterIntegration:
    """Test AdaptiveRouter with WPE integration."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        from tools.audio_cleaning import AdaptiveRouter
        return AdaptiveRouter(sample_rate=16000)

    def test_router_has_dereverberator(self, router):
        """Test router initializes with dereverberator."""
        assert hasattr(router, "dereverberator")
        assert hasattr(router, "dereverberator_name")
        assert router.dereverberator_name == "WPE"

    def test_router_clean_audio_skips_enhancement(self, router):
        """Test clean audio skips enhancement."""
        sample_rate = 16000
        t = np.linspace(0, 2.0, 2 * sample_rate)
        clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = router.route(clean)
        assert not result.was_enhanced
        assert result.enhancement_type == "none"

    def test_router_noisy_audio_denoises(self, router):
        """Test noisy audio gets denoised."""
        sample_rate = 16000
        t = np.linspace(0, 2.0, 2 * sample_rate)
        clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noise = _rng.standard_normal(len(t)).astype(np.float32) * 0.3
        noisy = clean + noise

        result = router.route(noisy)
        # Should attempt denoising
        assert result.condition.needs_denoising
