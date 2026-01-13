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
Audio Post-Processing Utilities for TTS Output

Provides audio quality enhancement functions that can be applied to TTS output:
- Click/pop removal: Remove sudden amplitude discontinuities
- Peak limiter: Prevent clipping while preserving dynamics
- Loudness normalization: Consistent output levels
- Full pipeline: Apply all enhancements in optimal order

All functions work with numpy arrays and are designed to be lightweight
and suitable for real-time post-processing.

Usage:
    from tools.pytorch_to_mlx.converters.models.audio_postprocess import (
        AudioQualityPipeline,
        remove_clicks,
        normalize_loudness,
        apply_limiter,
    )

    # Full pipeline
    pipeline = AudioQualityPipeline(sample_rate=24000)
    processed = pipeline.process(audio)

    # Individual functions
    audio = remove_clicks(audio)
    audio = normalize_loudness(audio, target_rms_db=-20.0)
    audio = apply_limiter(audio, threshold_db=-1.0)
"""

from dataclasses import dataclass

import numpy as np


def remove_clicks(
    audio: np.ndarray,
    threshold: float = 0.3,
    window_ms: float = 2.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Remove click artifacts from synthesized audio.

    Detects sudden amplitude jumps that are characteristic of synthesis
    artifacts at chunk boundaries and interpolates over them.

    Args:
        audio: Input audio as 1D numpy array
        threshold: Jump detection threshold (0.0 to 1.0)
        window_ms: Interpolation window size in milliseconds
        sample_rate: Audio sample rate

    Returns:
        Audio with clicks removed
    """
    if audio.size == 0:
        return audio

    # Work on a copy
    audio = audio.copy().astype(np.float64)

    # Calculate window size in samples
    window_samples = max(2, int(window_ms * sample_rate / 1000))

    # Find sudden amplitude jumps (potential clicks)
    diff = np.abs(np.diff(audio))
    click_indices = np.where(diff > threshold)[0]

    # Interpolate over each detected click
    for idx in click_indices:
        start = max(0, idx - window_samples)
        end = min(len(audio), idx + window_samples + 1)

        if end - start < 2:
            continue

        # Linear interpolation
        audio[start:end] = np.interp(
            np.arange(start, end), [start, end - 1], [audio[start], audio[end - 1]],
        )

    return audio.astype(np.float32)


def apply_limiter(
    audio: np.ndarray,
    threshold_db: float = -1.0,
    release_ms: float = 50.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Apply a transparent peak limiter to prevent clipping.

    Uses an envelope follower with instant attack and configurable release
    to smoothly reduce gain when peaks exceed threshold.

    Args:
        audio: Input audio as 1D numpy array
        threshold_db: Limiting threshold in dB (e.g., -1.0 dB)
        release_ms: Release time in milliseconds
        sample_rate: Audio sample rate

    Returns:
        Limited audio that won't exceed threshold
    """
    if audio.size == 0:
        return audio

    # Work with float64 for precision
    audio = audio.astype(np.float64)

    # Convert threshold to linear
    threshold_linear = 10 ** (threshold_db / 20)

    # Release coefficient (time constant)
    release_samples = max(1, int(release_ms * sample_rate / 1000))
    release_coeff = np.exp(-1.0 / release_samples)

    # Envelope follower with instant attack, smooth release
    envelope = np.zeros_like(audio)
    abs_audio = np.abs(audio)

    current_env = 0.0
    for i in range(len(audio)):
        if abs_audio[i] > current_env:
            current_env = abs_audio[i]  # Instant attack
        else:
            current_env = current_env * release_coeff  # Smooth release
        envelope[i] = current_env

    # Calculate gain reduction
    gain = np.ones_like(audio)
    over_threshold = envelope > threshold_linear
    gain[over_threshold] = threshold_linear / (envelope[over_threshold] + 1e-10)

    # Apply gain
    return (audio * gain).astype(np.float32)  # type: ignore[no-any-return]


def normalize_loudness(
    audio: np.ndarray,
    target_rms_db: float = -20.0,
    max_gain_db: float = 30.0,
) -> np.ndarray:
    """
    Normalize audio to target RMS loudness.

    Uses RMS (root mean square) as a proxy for perceived loudness.
    More accurate loudness measurement (LUFS) requires external libraries.

    Args:
        audio: Input audio as 1D numpy array
        target_rms_db: Target RMS level in dB (default -20 dB)
        max_gain_db: Maximum gain to apply (prevents excessive boosting
                     of very quiet signals)

    Returns:
        Normalized audio
    """
    if audio.size == 0:
        return audio

    # Calculate current RMS (avoid log of zero)
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-10)
    current_rms_db = 20 * np.log10(rms + 1e-10)

    # Calculate required gain
    gain_db = target_rms_db - current_rms_db
    gain_db = min(gain_db, max_gain_db)  # Limit maximum gain
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain and ensure within [-1, 1]
    normalized = audio.astype(np.float64) * gain_linear
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def deess(
    audio: np.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    frequency_low: float = 4000.0,
    frequency_high: float = 10000.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Apply dynamic de-essing to reduce harsh sibilants.

    De-essing targets the frequency range where sibilant sounds (s, sh, ch)
    are most prominent (typically 4-10kHz) and applies compression only
    when those frequencies exceed the threshold.

    Args:
        audio: Input audio as 1D numpy array
        threshold_db: Compression threshold in dB (default -20 dB)
        ratio: Compression ratio (e.g., 4.0 = 4:1 compression)
        frequency_low: Lower bound of sibilant frequency range (Hz)
        frequency_high: Upper bound of sibilant frequency range (Hz)
        sample_rate: Audio sample rate

    Returns:
        De-essed audio
    """
    if audio.size == 0:
        return audio

    # Check if scipy is available for filtering
    try:
        from scipy import signal
        from scipy.signal import hilbert
    except ImportError:
        # No scipy - return unchanged
        return audio.astype(np.float32)

    audio = audio.astype(np.float64)

    # Ensure frequencies are within Nyquist limit
    nyquist = sample_rate / 2
    freq_low = min(frequency_low, nyquist * 0.95)
    freq_high = min(frequency_high, nyquist * 0.95)

    if freq_low >= freq_high:
        return audio.astype(np.float32)

    # Design bandpass filter for sibilant frequencies
    try:
        # Normalize frequencies to Nyquist
        low_norm = freq_low / nyquist
        high_norm = freq_high / nyquist

        # Butterworth bandpass filter (order 4)
        b, a = signal.butter(4, [low_norm, high_norm], btype="band")

        # Extract sibilant band
        sibilants = signal.filtfilt(b, a, audio)
    except (ValueError, np.linalg.LinAlgError):
        # Filter design failed - return unchanged
        return audio.astype(np.float32)

    # Compute envelope using Hilbert transform
    try:
        envelope = np.abs(hilbert(sibilants))
    except ValueError:
        # Hilbert transform failed
        return audio.astype(np.float32)

    # Convert threshold to linear scale
    threshold_linear = 10 ** (threshold_db / 20)

    # Calculate gain reduction
    # When envelope > threshold: compress by ratio
    gain = np.ones_like(audio)
    over_threshold = envelope > threshold_linear

    if np.any(over_threshold):
        # Soft-knee compression
        excess = envelope[over_threshold] - threshold_linear
        compressed = threshold_linear + excess / ratio
        gain[over_threshold] = compressed / (envelope[over_threshold] + 1e-10)

    # Apply gain only to sibilant band, preserve rest
    processed_sibilants = sibilants * gain
    non_sibilants = audio - sibilants

    return (non_sibilants + processed_sibilants).astype(np.float32)  # type: ignore[no-any-return]


def dc_offset_removal(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from audio.

    Args:
        audio: Input audio as 1D numpy array

    Returns:
        Audio with DC offset removed
    """
    if audio.size == 0:
        return audio

    return (audio - np.mean(audio)).astype(np.float32)  # type: ignore[no-any-return]


def fade_edges(
    audio: np.ndarray,
    fade_ms: float = 5.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Apply fade-in and fade-out to avoid edge clicks.

    Args:
        audio: Input audio as 1D numpy array
        fade_ms: Fade duration in milliseconds
        sample_rate: Audio sample rate

    Returns:
        Audio with faded edges
    """
    if audio.size == 0:
        return audio

    audio = audio.copy().astype(np.float64)
    fade_samples = max(1, int(fade_ms * sample_rate / 1000))
    fade_samples = min(fade_samples, len(audio) // 2)

    # Create fade curves (cosine for smooth transition)
    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
    fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))

    # Apply fades
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out

    return audio.astype(np.float32)


class AudioQualityMetrics:
    """
    Compute objective audio quality metrics.

    Provides a set of metrics for evaluating TTS audio quality:
    - RMS level: Average loudness
    - Peak level: Maximum amplitude
    - Crest factor: Dynamic range indicator
    - Zero crossing rate: Speech vs noise indicator
    - Spectral flatness: How noise-like the signal is

    Example:
        metrics = AudioQualityMetrics(sample_rate=24000)
        results = metrics.compute(audio)
        print(f"RMS: {results['rms_db']:.1f} dB")
        print(f"SNR estimate: {results['snr_estimate_db']:.1f} dB")
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize metrics calculator.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def compute(self, audio: np.ndarray) -> dict:
        """
        Compute all audio quality metrics.

        Args:
            audio: Input audio as 1D numpy array

        Returns:
            Dictionary with computed metrics
        """
        if audio.size == 0:
            return {
                "rms_db": float("-inf"),
                "peak_db": float("-inf"),
                "crest_factor_db": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_flatness": 0.0,
                "snr_estimate_db": 0.0,
                "duration_seconds": 0.0,
            }

        audio = audio.astype(np.float64)

        return {
            "rms_db": self._compute_rms_db(audio),
            "peak_db": self._compute_peak_db(audio),
            "crest_factor_db": self._compute_crest_factor_db(audio),
            "zero_crossing_rate": self._compute_zcr(audio),
            "spectral_flatness": self._compute_spectral_flatness(audio),
            "snr_estimate_db": self._estimate_snr_db(audio),
            "duration_seconds": len(audio) / self.sample_rate,
        }

    def _compute_rms_db(self, audio: np.ndarray) -> float:
        """Compute RMS level in dB."""
        rms = np.sqrt(np.mean(audio**2) + 1e-10)
        return float(20 * np.log10(rms + 1e-10))

    def _compute_peak_db(self, audio: np.ndarray) -> float:
        """Compute peak level in dB."""
        peak = np.max(np.abs(audio))
        return float(20 * np.log10(peak + 1e-10))

    def _compute_crest_factor_db(self, audio: np.ndarray) -> float:
        """
        Compute crest factor (peak/RMS ratio) in dB.

        Higher values indicate more dynamic range / transients.
        Typical speech: 12-18 dB
        """
        rms = np.sqrt(np.mean(audio**2) + 1e-10)
        peak = np.max(np.abs(audio))
        if rms < 1e-10:
            return 0.0
        return float(20 * np.log10((peak + 1e-10) / (rms + 1e-10)))

    def _compute_zcr(self, audio: np.ndarray) -> float:
        """
        Compute zero crossing rate.

        Higher values indicate more high-frequency content or noise.
        """
        if len(audio) < 2:
            return 0.0
        signs = np.sign(audio)
        signs[signs == 0] = 1
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings / (len(audio) - 1))

    def _compute_spectral_flatness(self, audio: np.ndarray) -> float:
        """
        Compute spectral flatness (Wiener entropy).

        Values closer to 1.0 indicate noise-like spectrum.
        Values closer to 0.0 indicate tonal/harmonic content.
        Speech typically: 0.01 - 0.3
        """
        # Simple FFT-based spectral flatness
        n_fft = min(2048, len(audio))
        if n_fft < 32:
            return 0.0

        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
        spectrum = spectrum + 1e-10  # Avoid log(0)

        # Spectral flatness = geometric mean / arithmetic mean
        log_spectrum = np.log(spectrum)
        geometric_mean = np.exp(np.mean(log_spectrum))
        arithmetic_mean = np.mean(spectrum)

        return float(geometric_mean / (arithmetic_mean + 1e-10))

    def _estimate_snr_db(self, audio: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio from signal statistics.

        Uses a heuristic based on frame energy distribution:
        high-energy frames are assumed to be signal,
        low-energy frames are assumed to be noise/silence.

        This is an approximation - true SNR requires a reference.
        """
        if len(audio) < 256:
            return 0.0

        # Compute frame energies (32ms frames, 16ms hop)
        frame_size = int(0.032 * self.sample_rate)
        hop_size = frame_size // 2

        n_frames = (len(audio) - frame_size) // hop_size + 1
        if n_frames < 3:
            return 0.0

        frame_energies_list = []
        for i in range(n_frames):
            start = i * hop_size
            frame = audio[start : start + frame_size]
            energy = np.mean(frame**2)
            frame_energies_list.append(energy)

        frame_energies = np.array(frame_energies_list)

        # Sort energies and use bottom 10% as noise estimate
        sorted_energies = np.sort(frame_energies)
        noise_percentile = int(max(1, 0.1 * len(sorted_energies)))
        noise_energy = np.mean(sorted_energies[:noise_percentile]) + 1e-10

        # Use top 50% as signal estimate
        signal_percentile = int(max(1, 0.5 * len(sorted_energies)))
        signal_energy = np.mean(sorted_energies[-signal_percentile:]) + 1e-10

        return float(10 * np.log10(signal_energy / noise_energy))


@dataclass
class AudioQualityConfig:
    """Configuration for the audio quality pipeline."""

    sample_rate: int = 24000
    remove_dc: bool = True
    remove_clicks: bool = True
    click_threshold: float = 0.3
    deess: bool = False  # Off by default - requires scipy
    deess_threshold_db: float = -20.0
    deess_ratio: float = 4.0
    fade_edges: bool = True
    fade_ms: float = 5.0
    normalize: bool = True
    target_rms_db: float = -20.0
    apply_limiter: bool = True
    limiter_threshold_db: float = -1.0


class AudioQualityPipeline:
    """
    Complete audio quality enhancement pipeline.

    Applies a sequence of post-processing steps in optimal order:
    1. DC offset removal
    2. Click/pop removal
    3. Edge fading
    4. Loudness normalization
    5. Peak limiting

    Example:
        pipeline = AudioQualityPipeline(sample_rate=24000)
        processed = pipeline.process(tts_output)

        # Or with custom config
        config = AudioQualityConfig(
            target_rms_db=-16.0,
            click_threshold=0.2,
        )
        pipeline = AudioQualityPipeline(config=config)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        config: AudioQualityConfig | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            sample_rate: Audio sample rate (ignored if config provided)
            config: Optional full configuration object
        """
        if config is not None:
            self.config = config
        else:
            self.config = AudioQualityConfig(sample_rate=sample_rate)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through the quality pipeline.

        Args:
            audio: Input audio as 1D numpy array (expected range [-1, 1])

        Returns:
            Processed audio as float32 numpy array
        """
        if audio.size == 0:
            return audio.astype(np.float32)

        # Convert to float32 if needed
        audio = np.asarray(audio, dtype=np.float32)

        # 1. DC offset removal (before other processing)
        if self.config.remove_dc:
            audio = dc_offset_removal(audio)

        # 2. Click removal (before normalization to avoid amplifying artifacts)
        if self.config.remove_clicks:
            audio = remove_clicks(
                audio,
                threshold=self.config.click_threshold,
                sample_rate=self.config.sample_rate,
            )

        # 3. De-essing (reduce harsh sibilants)
        if self.config.deess:
            audio = deess(
                audio,
                threshold_db=self.config.deess_threshold_db,
                ratio=self.config.deess_ratio,
                sample_rate=self.config.sample_rate,
            )

        # 4. Edge fading (prevent clicks at boundaries)
        if self.config.fade_edges:
            audio = fade_edges(
                audio,
                fade_ms=self.config.fade_ms,
                sample_rate=self.config.sample_rate,
            )

        # 5. Loudness normalization
        if self.config.normalize:
            audio = normalize_loudness(
                audio,
                target_rms_db=self.config.target_rms_db,
            )

        # 6. Peak limiting (always last to catch any peaks)
        if self.config.apply_limiter:
            audio = apply_limiter(
                audio,
                threshold_db=self.config.limiter_threshold_db,
                sample_rate=self.config.sample_rate,
            )

        return audio
