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
ConditionEstimator - Audio condition analysis for adaptive cleaning

Estimates:
- SNR (Signal-to-Noise Ratio) using WADA-SNR algorithm
- Reverb T60 using energy decay analysis
- Content type (speech/singing/music/noise)

Per architecture spec A1: <5ms overhead on clean audio.

References:
- WADA-SNR: Kim & Stern (2008), "Robust Signal-to-Noise Ratio Estimation
  Based on Waveform Amplitude Distribution Analysis"
- T60 estimation: Schroeder method (backward integration)
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class ContentType(IntEnum):
    """Audio content type classification."""

    SPEECH = 0
    SINGING = 1
    MUSIC = 2
    NOISE = 3
    UNKNOWN = 4


@dataclass
class AudioCondition:
    """Result of audio condition estimation."""

    snr_db: float
    reverb_t60: float
    content_type: ContentType
    is_clean: bool  # SNR > 15 and T60 < 0.3
    needs_denoising: bool  # SNR < 15
    needs_dereverb: bool  # T60 > 0.3
    estimation_time_ms: float


def estimate_snr_wada(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_size_ms: float = 25.0,
) -> float:
    """
    Estimate SNR using spectral noise floor analysis.

    Uses high-frequency noise floor estimation - assumes speech energy
    is concentrated below 4kHz while noise is more broadband.

    Args:
        audio: Audio signal (float32, mono)
        sample_rate: Sample rate in Hz
        frame_size_ms: Frame size for analysis

    Returns:
        Estimated SNR in dB
    """
    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        return 0.0

    # Compute spectrum
    n_fft = 2048
    if len(audio) < n_fft:
        n_fft = len(audio)

    # Use multiple frames for robustness
    frame_size = n_fft
    hop = frame_size // 2
    n_frames = max(1, (len(audio) - frame_size) // hop + 1)

    signal_power_total = 0.0
    noise_power_total = 0.0

    freq_bins = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    # Speech band: 100-4000 Hz
    speech_mask = (freq_bins >= 100) & (freq_bins <= 4000)
    # High-freq noise band: 6000-8000 Hz (if available)
    hf_mask = (freq_bins >= 6000) & (freq_bins <= min(8000, sample_rate // 2))

    for i in range(min(n_frames, 10)):  # Limit to 10 frames for speed
        start = i * hop
        frame = audio[start : start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        # Window
        window = np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2

        # Sum speech band power
        signal_power_total += np.sum(spectrum[speech_mask])

        # Estimate noise from high-frequency band if available
        if np.sum(hf_mask) > 0:
            # Scale HF noise to full spectrum
            hf_power = np.mean(spectrum[hf_mask])
            estimated_noise_power = hf_power * np.sum(speech_mask)
            noise_power_total += estimated_noise_power
        else:
            # Fallback: use spectral floor
            spectral_floor = np.percentile(spectrum, 10)
            noise_power_total += spectral_floor * np.sum(speech_mask)

    if noise_power_total < 1e-15:
        # Very clean signal
        return 40.0

    if signal_power_total < 1e-15:
        return -10.0

    snr_db = 10 * np.log10(signal_power_total / noise_power_total)

    return float(np.clip(snr_db, -20, 60))


def estimate_reverb_t60(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> float:
    """
    Estimate reverberation time T60 using spectral correlation.

    Reverberant audio shows high correlation between adjacent spectral frames
    due to the "smearing" effect of room reflections. Dry audio has more
    independent frames.

    Args:
        audio: Audio signal (float32, mono)
        sample_rate: Sample rate in Hz

    Returns:
        Estimated T60 in seconds (0.1 = very dry, 1.0+ = very reverberant)
    """
    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) < sample_rate // 4:  # Need at least 250ms
        return 0.2  # Default dry

    # Compute short-time spectra
    n_fft = 1024
    hop_size = n_fft // 4  # 75% overlap
    n_frames = (len(audio) - n_fft) // hop_size + 1

    if n_frames < 4:
        return 0.2

    # Limit frames for speed
    n_frames = min(n_frames, 32)

    window = np.hanning(n_fft)
    spectra = np.zeros((n_frames, n_fft // 2 + 1))

    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectra[i] = np.abs(np.fft.rfft(frame * window))

    # Normalize spectra
    spectra = spectra / (np.max(spectra) + 1e-10)

    # Compute inter-frame correlation
    if n_frames < 2:
        return 0.2

    correlations = []
    for i in range(n_frames - 1):
        corr = np.corrcoef(spectra[i], spectra[i + 1])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    if len(correlations) == 0:
        return 0.2

    mean_corr = np.mean(correlations)

    # Map correlation to T60
    # High correlation (>0.9) = reverberant (long decay)
    # Low correlation (<0.5) = dry (short decay)
    # Sustained tones will have high correlation but are not "reverberant"

    # Check if this is a sustained tone (very low spectral variance)
    spectral_var = np.var(spectra, axis=0)
    mean_spectral_var = np.mean(spectral_var)

    if mean_spectral_var < 0.01:
        # Sustained tone - not reverberant
        return 0.15

    # Map correlation to T60
    if mean_corr < 0.5:
        t60 = 0.15
    elif mean_corr < 0.7:
        t60 = 0.25
    elif mean_corr < 0.85:
        t60 = 0.4
    elif mean_corr < 0.92:
        t60 = 0.6
    else:
        t60 = 0.8 + (mean_corr - 0.92) * 5.0  # Scale up for very high corr

    return float(np.clip(t60, 0.1, 2.0))


def classify_content_type(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> ContentType:
    """
    Classify audio content type (speech/singing/music/noise).

    Fast classification using:
    - Zero-crossing rate: high for noise
    - Spectral flatness: high for noise
    - Energy variation: high for speech, low for sustained tones

    Args:
        audio: Audio signal (float32, mono)
        sample_rate: Sample rate in Hz

    Returns:
        Classified content type
    """
    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) < sample_rate // 10:  # Need at least 100ms
        return ContentType.SPEECH  # Default

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val < 1e-8:
        return ContentType.NOISE
    audio = audio / max_val

    # Fast features
    # 1. Zero-crossing rate
    zcr = np.mean(np.abs(np.diff(np.signbit(audio))))

    # 2. Quick spectral flatness using single FFT
    n_fft = min(2048, len(audio))
    spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
    spectrum = np.maximum(spectrum, 1e-10)

    # Spectral flatness (geometric mean / arithmetic mean)
    log_spectrum = np.log(spectrum)
    spectral_flatness = np.exp(np.mean(log_spectrum)) / np.mean(spectrum)

    # 3. Energy variation
    frame_size = int(0.025 * sample_rate)  # 25ms
    n_frames = len(audio) // frame_size
    if n_frames > 1:
        frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
        frame_energy = np.sum(frames**2, axis=1)
        energy_var = np.std(frame_energy) / (np.mean(frame_energy) + 1e-10)
    else:
        energy_var = 0.5

    # Decision logic
    # Noise: high flatness (>0.3), high ZCR (>0.1)
    if spectral_flatness > 0.25 and zcr > 0.08:
        return ContentType.NOISE

    # Music/Singing: low flatness, low energy variation (sustained)
    if spectral_flatness < 0.1 and energy_var < 0.3:
        # Check for pitch stability (simple autocorrelation peak)
        if len(audio) >= sample_rate // 4:
            segment = audio[: sample_rate // 4]
            # Check for periodicity in voice range
            autocorr = np.correlate(segment[:1024], segment[:1024], mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            min_lag = int(sample_rate / 400)  # 400 Hz
            max_lag = min(int(sample_rate / 80), len(autocorr) - 1)  # 80 Hz
            if max_lag > min_lag:
                pitch_region = autocorr[min_lag : max_lag + 1]
                periodicity = np.max(pitch_region) / (autocorr[0] + 1e-10)
                if periodicity > 0.5:
                    return ContentType.SINGING
        return ContentType.MUSIC

    # Default: speech
    return ContentType.SPEECH


class ConditionEstimator:
    """
    Complete audio condition estimator.

    Combines SNR, reverb, and content type estimation into a single
    analysis pass for efficiency.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        snr_clean_threshold: float = 15.0,
        reverb_clean_threshold: float = 0.3,
    ):
        """
        Initialize condition estimator.

        Args:
            sample_rate: Expected sample rate
            snr_clean_threshold: SNR above this is considered clean (dB)
            reverb_clean_threshold: T60 below this is considered dry (seconds)
        """
        self.sample_rate = sample_rate
        self.snr_clean_threshold = snr_clean_threshold
        self.reverb_clean_threshold = reverb_clean_threshold

    def estimate(self, audio: np.ndarray) -> AudioCondition:
        """
        Estimate audio condition.

        Args:
            audio: Audio signal (float32, mono)

        Returns:
            AudioCondition with all estimates
        """
        import time

        start_time = time.perf_counter()

        audio = np.asarray(audio, dtype=np.float32)

        # Estimate SNR
        snr_db = estimate_snr_wada(audio, self.sample_rate)

        # Estimate reverb
        reverb_t60 = estimate_reverb_t60(audio, self.sample_rate)

        # Classify content
        content_type = classify_content_type(audio, self.sample_rate)

        # Determine cleaning needs
        is_clean = snr_db > self.snr_clean_threshold and reverb_t60 < self.reverb_clean_threshold
        needs_denoising = snr_db < self.snr_clean_threshold
        needs_dereverb = reverb_t60 > self.reverb_clean_threshold

        estimation_time_ms = (time.perf_counter() - start_time) * 1000

        return AudioCondition(
            snr_db=snr_db,
            reverb_t60=reverb_t60,
            content_type=content_type,
            is_clean=is_clean,
            needs_denoising=needs_denoising,
            needs_dereverb=needs_dereverb,
            estimation_time_ms=estimation_time_ms,
        )

    def warmup(self):
        """
        Warm up the estimator to avoid cold-start latency.
        """
        # Create dummy signal
        rng = np.random.default_rng()
        dummy = rng.standard_normal(self.sample_rate).astype(np.float32) * 0.1
        self.estimate(dummy)


def test_condition_estimator():
    """Test the condition estimator with synthetic signals."""

    print("=" * 60)
    print("ConditionEstimator Test")
    print("=" * 60)

    sample_rate = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))

    estimator = ConditionEstimator(sample_rate=sample_rate)

    # Warmup
    print("Warming up...")
    estimator.warmup()

    # Test 1: Clean speech (440 Hz tone as proxy)
    print("\n1. Clean speech proxy (440 Hz tone):")
    clean_speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = estimator.estimate(clean_speech)
    print(f"   SNR: {result.snr_db:.1f} dB")
    print(f"   T60: {result.reverb_t60:.3f} s")
    print(f"   Content: {result.content_type.name}")
    print(f"   Is clean: {result.is_clean}")
    print(f"   Time: {result.estimation_time_ms:.2f} ms")

    # Test 2: Noisy signal
    print("\n2. Noisy signal (SNR ~10dB):")
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(t)).astype(np.float32) * 0.15
    noisy_speech = clean_speech + noise
    result = estimator.estimate(noisy_speech)
    print(f"   SNR: {result.snr_db:.1f} dB")
    print(f"   T60: {result.reverb_t60:.3f} s")
    print(f"   Content: {result.content_type.name}")
    print(f"   Needs denoising: {result.needs_denoising}")
    print(f"   Time: {result.estimation_time_ms:.2f} ms")

    # Test 3: Pure noise
    print("\n3. Pure noise:")
    pure_noise = rng.standard_normal(len(t)).astype(np.float32) * 0.3
    result = estimator.estimate(pure_noise)
    print(f"   SNR: {result.snr_db:.1f} dB")
    print(f"   T60: {result.reverb_t60:.3f} s")
    print(f"   Content: {result.content_type.name}")
    print(f"   Time: {result.estimation_time_ms:.2f} ms")

    # Test 4: Reverberant signal (simulated with convolution)
    print("\n4. Reverberant signal (simulated):")
    # Create simple room impulse response
    rir_len = int(0.5 * sample_rate)  # 500ms T60
    rir = np.exp(-np.linspace(0, 6, rir_len)) * rng.standard_normal(rir_len)
    rir = rir.astype(np.float32) / np.max(np.abs(rir))
    reverb_speech = np.convolve(clean_speech, rir, mode="same").astype(np.float32)
    result = estimator.estimate(reverb_speech)
    print(f"   SNR: {result.snr_db:.1f} dB")
    print(f"   T60: {result.reverb_t60:.3f} s")
    print(f"   Content: {result.content_type.name}")
    print(f"   Needs dereverb: {result.needs_dereverb}")
    print(f"   Time: {result.estimation_time_ms:.2f} ms")

    # Performance test
    print("\n5. Performance test (100 iterations):")
    times = []
    for _ in range(100):
        result = estimator.estimate(noisy_speech)
        times.append(result.estimation_time_ms)

    print(f"   Mean: {np.mean(times):.2f} ms")
    print(f"   Std: {np.std(times):.2f} ms")
    print(f"   p95: {np.percentile(times, 95):.2f} ms")

    # Check against target (<5ms)
    target_ms = 5.0
    p95 = np.percentile(times, 95)
    passed = p95 < target_ms
    print(f"\n   Target: <{target_ms} ms p95")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    test_condition_estimator()
