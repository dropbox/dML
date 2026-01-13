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
Tests for audio post-processing utilities.

Tests the audio quality enhancement functions including:
- Click/pop removal
- Peak limiting
- Loudness normalization
- DC offset removal
- Edge fading
- De-essing (sibilant reduction)
- Audio quality metrics
- Full quality pipeline
"""

import numpy as np

# Use modern numpy RNG
_rng = np.random.default_rng(42)

from tools.pytorch_to_mlx.converters.models.audio_postprocess import (
    AudioQualityConfig,
    AudioQualityMetrics,
    AudioQualityPipeline,
    apply_limiter,
    dc_offset_removal,
    deess,
    fade_edges,
    normalize_loudness,
    remove_clicks,
)


class TestRemoveClicks:
    """Tests for click removal function."""

    def test_no_clicks(self):
        """Audio without clicks should pass through unchanged."""
        # Smooth sine wave has no sudden jumps
        t = np.linspace(0, 0.1, 2400)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = remove_clicks(audio, threshold=0.3)

        # Should be very similar (small floating point differences allowed)
        assert np.allclose(audio, result, atol=1e-5)

    def test_removes_large_jump(self):
        """Large amplitude jumps should be smoothed."""
        audio = np.zeros(1000, dtype=np.float32)
        audio[500] = 0.8  # Single sample spike

        result = remove_clicks(audio, threshold=0.3)

        # The spike should be significantly reduced
        assert np.max(np.abs(result)) < 0.5

    def test_empty_audio(self):
        """Empty audio should return empty."""
        audio = np.array([], dtype=np.float32)
        result = remove_clicks(audio)
        assert result.size == 0

    def test_preserves_dtype(self):
        """Output should be float32."""
        audio = _rng.standard_normal(1000).astype(np.float32)
        result = remove_clicks(audio)
        assert result.dtype == np.float32


class TestApplyLimiter:
    """Tests for peak limiter function."""

    def test_below_threshold_unchanged(self):
        """Audio below threshold should pass through."""
        audio = np.ones(100, dtype=np.float32) * 0.5  # -6 dB
        result = apply_limiter(audio, threshold_db=-1.0)

        # Should be unchanged (within tolerance)
        assert np.allclose(audio, result, atol=1e-5)

    def test_limits_peaks(self):
        """Peaks above threshold should be reduced."""
        audio = np.ones(100, dtype=np.float32) * 2.0  # Way over 0 dB
        result = apply_limiter(audio, threshold_db=-1.0)

        # Threshold -1 dB = 10^(-1/20) ≈ 0.891
        threshold_linear = 10 ** (-1.0 / 20)
        assert np.all(np.abs(result) <= threshold_linear + 0.01)

    def test_empty_audio(self):
        """Empty audio should return empty."""
        audio = np.array([], dtype=np.float32)
        result = apply_limiter(audio)
        assert result.size == 0

    def test_preserves_shape(self):
        """Output shape should match input."""
        audio = _rng.standard_normal(1000).astype(np.float32)
        result = apply_limiter(audio)
        assert result.shape == audio.shape

    def test_smooth_release(self):
        """Limiter should have smooth release, not pumping."""
        # Impulse followed by quiet
        audio = np.zeros(1000, dtype=np.float32)
        audio[100] = 2.0  # Peak

        result = apply_limiter(audio, threshold_db=-6.0, release_ms=50.0)

        # Check that the gain returns smoothly (no sudden jumps after peak)
        post_peak = result[101:200]
        diff = np.abs(np.diff(post_peak))
        max_jump = np.max(diff)
        assert max_jump < 0.1  # Smooth release


class TestNormalizeLoudness:
    """Tests for loudness normalization function."""

    def test_normalizes_quiet_audio(self):
        """Quiet audio should be boosted to target level."""
        # Moderately quiet audio at -40 dB (amplitude 0.01)
        audio = np.ones(1000, dtype=np.float32) * 0.01

        result = normalize_loudness(audio, target_rms_db=-20.0)

        # Calculate output RMS
        rms = np.sqrt(np.mean(result**2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Should be close to target (within 1 dB)
        assert abs(rms_db - (-20.0)) < 1.0

    def test_normalizes_loud_audio(self):
        """Loud audio should be reduced to target level."""
        # Loud audio
        audio = np.ones(1000, dtype=np.float32) * 0.9

        result = normalize_loudness(audio, target_rms_db=-20.0)

        # Should be quieter than input
        assert np.sqrt(np.mean(result**2)) < np.sqrt(np.mean(audio**2))

    def test_max_gain_limit(self):
        """Should not apply more than max_gain_db."""
        # Extremely quiet audio
        audio = np.ones(1000, dtype=np.float32) * 1e-6

        result = normalize_loudness(audio, target_rms_db=-20.0, max_gain_db=20.0)

        # Gain should be limited
        gain = np.max(np.abs(result)) / 1e-6
        gain_db = 20 * np.log10(gain + 1e-10)
        assert gain_db <= 21.0  # Allow small tolerance

    def test_clips_to_valid_range(self):
        """Output should be clipped to [-1, 1]."""
        audio = np.ones(1000, dtype=np.float32) * 0.1

        result = normalize_loudness(audio, target_rms_db=0.0)  # Very loud target

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_empty_audio(self):
        """Empty audio should return empty."""
        audio = np.array([], dtype=np.float32)
        result = normalize_loudness(audio)
        assert result.size == 0


class TestDCOffsetRemoval:
    """Tests for DC offset removal function."""

    def test_removes_dc_offset(self):
        """DC offset should be removed."""
        audio = _rng.standard_normal(1000).astype(np.float32) + 0.5  # 0.5 DC offset

        result = dc_offset_removal(audio)

        # Mean should be close to zero
        assert abs(np.mean(result)) < 1e-5

    def test_preserves_zero_mean(self):
        """Zero-mean audio should pass through."""
        audio = _rng.standard_normal(1000).astype(np.float32)
        audio = audio - np.mean(audio)  # Ensure zero mean

        result = dc_offset_removal(audio)

        assert np.allclose(audio, result, atol=1e-5)

    def test_empty_audio(self):
        """Empty audio should return empty."""
        audio = np.array([], dtype=np.float32)
        result = dc_offset_removal(audio)
        assert result.size == 0


class TestFadeEdges:
    """Tests for edge fading function."""

    def test_fades_edges(self):
        """Audio should have faded edges."""
        audio = np.ones(2400, dtype=np.float32)  # 100ms at 24kHz

        result = fade_edges(audio, fade_ms=10.0, sample_rate=24000)

        # First samples should be faded (near zero)
        assert result[0] < 0.1
        # Last samples should be faded
        assert result[-1] < 0.1
        # Middle should be unchanged
        assert abs(result[1200] - 1.0) < 0.01

    def test_short_audio(self):
        """Short audio should still work."""
        audio = np.ones(100, dtype=np.float32)

        result = fade_edges(audio, fade_ms=10.0, sample_rate=24000)

        # Should not crash and should have some fading
        assert result[0] < result[50]

    def test_empty_audio(self):
        """Empty audio should return empty."""
        audio = np.array([], dtype=np.float32)
        result = fade_edges(audio)
        assert result.size == 0

    def test_cosine_fade(self):
        """Fade should be smooth (cosine curve)."""
        audio = np.ones(2400, dtype=np.float32)

        result = fade_edges(audio, fade_ms=10.0, sample_rate=24000)

        # Fade should be monotonically increasing at start
        fade_samples = int(10.0 * 24000 / 1000)
        start_fade = result[:fade_samples]
        assert np.all(np.diff(start_fade) >= 0)


class TestAudioQualityPipeline:
    """Tests for the full audio quality pipeline."""

    def test_basic_processing(self):
        """Pipeline should process audio without errors."""
        audio = _rng.standard_normal(24000).astype(np.float32) * 0.5
        pipeline = AudioQualityPipeline(sample_rate=24000)

        result = pipeline.process(audio)

        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_empty_audio(self):
        """Empty audio should return empty."""
        pipeline = AudioQualityPipeline()
        result = pipeline.process(np.array([], dtype=np.float32))
        assert result.size == 0

    def test_custom_config(self):
        """Custom configuration should be respected."""
        config = AudioQualityConfig(
            sample_rate=48000,
            remove_dc=False,
            normalize=True,
            target_rms_db=-16.0,
        )
        pipeline = AudioQualityPipeline(config=config)

        assert pipeline.config.sample_rate == 48000
        assert pipeline.config.remove_dc is False
        assert pipeline.config.target_rms_db == -16.0

    def test_output_range(self):
        """Output should be in valid range [-1, 1]."""
        # Input with various issues
        audio = _rng.standard_normal(24000).astype(np.float32) * 2.0  # Too loud
        audio += 0.5  # DC offset
        audio[1000] = 5.0  # Spike

        pipeline = AudioQualityPipeline()
        result = pipeline.process(audio)

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_preserves_intelligibility(self):
        """Processing should not destroy the signal."""
        # Create a recognizable pattern
        t = np.linspace(0, 1, 24000)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        pipeline = AudioQualityPipeline()
        result = pipeline.process(audio)

        # Check correlation (should be high for similar signals)
        correlation = np.corrcoef(audio[1000:-1000], result[1000:-1000])[0, 1]
        assert correlation > 0.9

    def test_disabled_processing(self):
        """All processing can be disabled."""
        config = AudioQualityConfig(
            remove_dc=False,
            remove_clicks=False,
            fade_edges=False,
            normalize=False,
            apply_limiter=False,
        )
        pipeline = AudioQualityPipeline(config=config)

        audio = _rng.standard_normal(1000).astype(np.float32)
        result = pipeline.process(audio)

        # Should be nearly identical
        assert np.allclose(audio, result, atol=1e-6)


class TestIntegration:
    """Integration tests with realistic audio scenarios."""

    def test_tts_output_processing(self):
        """Simulate processing TTS output."""
        # Simulate typical TTS output: varying amplitude with some artifacts
        duration = 2.0  # 2 seconds
        sample_rate = 24000
        n_samples = int(duration * sample_rate)

        # Base signal: speech-like with varying amplitude
        t = np.linspace(0, duration, n_samples)
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t)
            + 0.2 * np.sin(2 * np.pi * 400 * t)
            + 0.1 * np.sin(2 * np.pi * 800 * t)
        )
        audio = audio.astype(np.float32)

        # Add some typical artifacts
        audio += 0.02  # Small DC offset
        audio[10000] = 0.9  # Click artifact

        pipeline = AudioQualityPipeline(sample_rate=sample_rate)
        result = pipeline.process(audio)

        # Verify improvements
        assert abs(np.mean(result)) < abs(np.mean(audio))  # DC reduced
        assert np.max(result) < np.max(audio)  # Peak reduced

    def test_batch_processing(self):
        """Process multiple audio segments."""
        pipeline = AudioQualityPipeline()

        segments = [
            _rng.standard_normal(24000).astype(np.float32) * 0.3,
            _rng.standard_normal(48000).astype(np.float32) * 0.5,
            _rng.standard_normal(12000).astype(np.float32) * 0.1,
        ]

        results = [pipeline.process(seg) for seg in segments]

        # All should have similar loudness after normalization
        rms_values = [np.sqrt(np.mean(r**2)) for r in results]
        rms_range = max(rms_values) - min(rms_values)
        assert rms_range < 0.1  # Should be relatively consistent


class TestDeess:
    """Tests for de-essing function."""

    def test_empty_audio(self):
        """Empty audio should return empty."""
        audio = np.array([], dtype=np.float32)
        result = deess(audio)
        assert result.size == 0

    def test_preserves_low_frequencies(self):
        """Low frequency content should be preserved."""
        # Pure low frequency tone (well below sibilant range)
        t = np.linspace(0, 0.5, 12000)
        audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        result = deess(audio)

        # Should be very similar (de-essing shouldn't affect 200Hz)
        assert np.allclose(audio, result, atol=0.05)

    def test_reduces_high_frequencies(self):
        """High frequency content (sibilant range) should be compressed."""
        # Generate sibilant-range tone
        sample_rate = 24000

        # Create a tone in the sibilant range
        t = np.linspace(0, 0.5, 12000)
        # 6000 Hz is in the sibilant range (4-10kHz)
        audio = (0.5 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)

        result = deess(audio, threshold_db=-30.0, ratio=4.0, sample_rate=sample_rate)

        # Energy in sibilant band should be reduced
        # (Note: if scipy is not available, function returns unchanged)
        try:
            import scipy.signal  # noqa: F401

            # Sibilant energy should be reduced
            original_energy = np.mean(audio**2)
            result_energy = np.mean(result**2)
            assert result_energy <= original_energy * 1.1  # Allow small tolerance
        except ImportError:
            # Without scipy, deess returns unchanged
            assert np.allclose(audio, result)

    def test_preserves_dtype(self):
        """Output should be float32."""
        audio = _rng.standard_normal(1000).astype(np.float32)
        result = deess(audio)
        assert result.dtype == np.float32

    def test_handles_various_sample_rates(self):
        """Should work with different sample rates."""
        audio = _rng.standard_normal(24000).astype(np.float32) * 0.5

        # Test different sample rates
        for sr in [16000, 24000, 44100, 48000]:
            result = deess(audio, sample_rate=sr)
            assert result.shape == audio.shape

    def test_pipeline_integration(self):
        """De-essing should integrate with the pipeline."""
        config = AudioQualityConfig(
            deess=True,
            deess_threshold_db=-20.0,
            deess_ratio=4.0,
        )
        pipeline = AudioQualityPipeline(config=config)

        audio = _rng.standard_normal(12000).astype(np.float32) * 0.5
        result = pipeline.process(audio)

        assert result.shape == audio.shape
        assert result.dtype == np.float32


class TestAudioQualityMetrics:
    """Tests for audio quality metrics."""

    def test_empty_audio(self):
        """Empty audio should return default values."""
        metrics = AudioQualityMetrics()
        result = metrics.compute(np.array([]))

        assert result["rms_db"] == float("-inf")
        assert result["peak_db"] == float("-inf")
        assert result["duration_seconds"] == 0.0

    def test_computes_rms(self):
        """RMS calculation should be accurate."""
        # Unit amplitude sine wave: RMS = 1/sqrt(2) ≈ 0.707
        t = np.linspace(0, 1, 24000)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        metrics = AudioQualityMetrics()
        result = metrics.compute(audio)

        # Expected RMS in dB: 20*log10(0.707) ≈ -3.01 dB
        expected_rms_db = 20 * np.log10(1 / np.sqrt(2))
        assert abs(result["rms_db"] - expected_rms_db) < 0.5

    def test_computes_peak(self):
        """Peak calculation should be accurate."""
        # Unit amplitude signal
        audio = np.zeros(1000, dtype=np.float32)
        audio[500] = 1.0  # Peak at 1.0

        metrics = AudioQualityMetrics()
        result = metrics.compute(audio)

        # Expected peak in dB: 20*log10(1.0) = 0 dB
        assert abs(result["peak_db"]) < 0.1

    def test_computes_crest_factor(self):
        """Crest factor should be computed correctly."""
        # Impulse-like signal has high crest factor
        audio = np.zeros(1000, dtype=np.float32)
        audio[500] = 1.0

        metrics = AudioQualityMetrics()
        result = metrics.compute(audio)

        # Crest factor should be high (peak much higher than RMS)
        assert result["crest_factor_db"] > 20

        # Constant signal has crest factor of 0 dB
        constant_audio = np.ones(1000, dtype=np.float32) * 0.5
        constant_result = metrics.compute(constant_audio)
        assert abs(constant_result["crest_factor_db"]) < 1.0

    def test_computes_zcr(self):
        """Zero crossing rate should be computed correctly."""
        # Low frequency: fewer zero crossings
        t = np.linspace(0, 1, 24000)
        low_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)

        # High frequency: more zero crossings
        high_freq = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        metrics = AudioQualityMetrics()
        low_result = metrics.compute(low_freq)
        high_result = metrics.compute(high_freq)

        # High frequency should have higher ZCR
        assert high_result["zero_crossing_rate"] > low_result["zero_crossing_rate"]

    def test_computes_spectral_flatness(self):
        """Spectral flatness should differentiate tonal from noise."""
        # Tonal signal (sine wave): low spectral flatness
        t = np.linspace(0, 1, 24000)
        tonal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Noise-like signal: high spectral flatness
        noise = _rng.standard_normal(24000).astype(np.float32)

        metrics = AudioQualityMetrics()
        tonal_result = metrics.compute(tonal)
        noise_result = metrics.compute(noise)

        # Noise should have higher spectral flatness
        assert noise_result["spectral_flatness"] > tonal_result["spectral_flatness"]

    def test_computes_snr_estimate(self):
        """SNR estimation should give reasonable values."""
        # Signal with clear silence regions
        audio = np.zeros(24000, dtype=np.float32)
        audio[5000:15000] = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 10000))

        metrics = AudioQualityMetrics()
        result = metrics.compute(audio)

        # Should estimate positive SNR
        assert result["snr_estimate_db"] > 0

    def test_computes_duration(self):
        """Duration should be computed correctly."""
        sample_rate = 24000
        duration_sec = 2.5
        audio = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)

        metrics = AudioQualityMetrics(sample_rate=sample_rate)
        result = metrics.compute(audio)

        assert abs(result["duration_seconds"] - duration_sec) < 0.01

    def test_short_audio(self):
        """Should handle very short audio."""
        metrics = AudioQualityMetrics()

        # Very short audio (less than one frame)
        short_audio = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        result = metrics.compute(short_audio)

        # Should still compute basic metrics
        assert result["rms_db"] > float("-inf")
        assert result["peak_db"] > float("-inf")
