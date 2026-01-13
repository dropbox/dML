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
Tests for ConditionEstimator - Audio condition analysis.

Run with: pytest tests/test_condition_estimator.py -v
"""

import numpy as np
import pytest

from tools.audio_cleaning.condition_estimator import (
    AudioCondition,
    ConditionEstimator,
    ContentType,
    classify_content_type,
    estimate_reverb_t60,
    estimate_snr_wada,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestSNREstimation:
    """Test SNR estimation accuracy."""

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def test_clean_signal_high_snr(self, sample_rate):
        """Clean signal should report high SNR."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))
        # Speech-like signal: fundamental + harmonics
        signal = (
            0.5 * np.sin(2 * np.pi * 150 * t)
            + 0.3 * np.sin(2 * np.pi * 300 * t)
            + 0.1 * np.sin(2 * np.pi * 450 * t)
        ).astype(np.float32)

        snr = estimate_snr_wada(signal, sample_rate)
        assert snr > 30, f"Clean signal should have SNR > 30 dB, got {snr:.1f}"

    def test_noisy_signal_correct_snr(self, sample_rate):
        """Signal with known noise level should report correct SNR."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        # Create signal with approximately 10 dB SNR
        signal = 0.3 * np.sin(2 * np.pi * 150 * t)
        # Noise power should be ~10 dB below signal power
        # SNR = 10 * log10(signal_power / noise_power)
        # 10 = 10 * log10(0.3^2 / noise_power^2)
        # noise_power = 0.3 / sqrt(10) ≈ 0.095
        noise = _rng.standard_normal(len(t)).astype(np.float32) * 0.1

        noisy = (signal + noise).astype(np.float32)
        snr = estimate_snr_wada(noisy, sample_rate)

        # Should be approximately 10 dB (allowing ±5 dB tolerance)
        assert 5 < snr < 20, f"10dB SNR signal should be ~10dB, got {snr:.1f}"

    def test_pure_noise_low_snr(self, sample_rate):
        """Pure noise should report very low SNR."""
        duration_s = 2.0
        noise = _rng.standard_normal(int(duration_s * sample_rate)).astype(np.float32) * 0.5

        snr = estimate_snr_wada(noise, sample_rate)
        assert snr < 10, f"Pure noise should have SNR < 10 dB, got {snr:.1f}"

    def test_empty_audio_returns_zero(self, sample_rate):
        """Empty audio should return 0 SNR."""
        empty = np.array([], dtype=np.float32)
        snr = estimate_snr_wada(empty, sample_rate)
        assert snr == 0.0


class TestT60Estimation:
    """Test reverb T60 estimation."""

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def test_dry_impulse_low_t60(self, sample_rate):
        """Dry signal with sharp transients should have low T60."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        # Create impulse train
        signal = np.zeros_like(t)
        for i in range(10):
            idx = int((i + 0.5) * len(t) / 10)
            signal[idx : idx + 100] = 0.5 * _rng.standard_normal(100)

        t60 = estimate_reverb_t60(signal.astype(np.float32), sample_rate)
        assert t60 < 0.5, f"Dry impulsive signal should have T60 < 0.5s, got {t60:.3f}"

    def test_sustained_tone_low_t60(self, sample_rate):
        """Sustained pure tone should NOT be classified as reverberant."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        t60 = estimate_reverb_t60(tone, sample_rate)
        # Sustained tone is NOT reverb - should be low
        assert t60 < 0.3, f"Sustained tone should have T60 < 0.3s, got {t60:.3f}"

    def test_short_audio_default_t60(self, sample_rate):
        """Very short audio should return default T60."""
        short = _rng.standard_normal(sample_rate // 10).astype(np.float32)  # 100ms
        t60 = estimate_reverb_t60(short, sample_rate)
        assert t60 == 0.2  # Default value


class TestContentClassification:
    """Test content type classification."""

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def test_noise_classification(self, sample_rate):
        """White noise should be classified as NOISE."""
        duration_s = 2.0
        noise = _rng.standard_normal(int(duration_s * sample_rate)).astype(np.float32)

        content = classify_content_type(noise, sample_rate)
        assert content == ContentType.NOISE

    def test_speech_like_signal(self, sample_rate):
        """Speech-like signal (varying energy, harmonics) should be SPEECH."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        # Create speech-like signal with amplitude modulation
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # 3 Hz syllable rate
        carrier = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
        signal = (envelope * carrier).astype(np.float32)

        content = classify_content_type(signal, sample_rate)
        assert content == ContentType.SPEECH

    def test_very_short_audio_default(self, sample_rate):
        """Very short audio should return SPEECH (default)."""
        short = _rng.standard_normal(sample_rate // 20).astype(np.float32)  # 50ms
        content = classify_content_type(short, sample_rate)
        assert content == ContentType.SPEECH


class TestConditionEstimator:
    """Test complete ConditionEstimator class."""

    @pytest.fixture
    def estimator(self):
        return ConditionEstimator(sample_rate=16000)

    def test_clean_signal_detected(self, estimator):
        """Clean signal should be detected as clean."""
        duration_s = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        # Clean speech-like signal
        signal = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

        result = estimator.estimate(signal)

        assert result.snr_db > 15, f"SNR should be > 15, got {result.snr_db:.1f}"
        assert result.reverb_t60 < 0.3, f"T60 should be < 0.3, got {result.reverb_t60:.3f}"
        assert result.is_clean is True

    def test_noisy_signal_needs_denoising(self, estimator):
        """Noisy signal should be flagged for denoising."""
        duration_s = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        signal = 0.2 * np.sin(2 * np.pi * 150 * t)
        noise = _rng.standard_normal(len(t)) * 0.15
        noisy = (signal + noise).astype(np.float32)

        result = estimator.estimate(noisy)

        assert result.needs_denoising is True

    def test_performance_under_5ms(self, estimator):
        """Estimation should complete in under 5ms p95."""
        duration_s = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))

        signal = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

        times = []
        for _ in range(50):
            result = estimator.estimate(signal)
            times.append(result.estimation_time_ms)

        p95 = np.percentile(times, 95)
        assert p95 < 5.0, f"p95 latency should be < 5ms, got {p95:.2f}ms"

    def test_warmup_runs_without_error(self, estimator):
        """Warmup should complete without error."""
        estimator.warmup()  # Should not raise

    def test_result_dataclass_fields(self, estimator):
        """Result should have all expected fields."""
        signal = _rng.standard_normal(16000).astype(np.float32)
        result = estimator.estimate(signal)

        assert isinstance(result, AudioCondition)
        assert isinstance(result.snr_db, float)
        assert isinstance(result.reverb_t60, float)
        assert isinstance(result.content_type, ContentType)
        assert isinstance(result.is_clean, bool)
        assert isinstance(result.needs_denoising, bool)
        assert isinstance(result.needs_dereverb, bool)
        assert isinstance(result.estimation_time_ms, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
