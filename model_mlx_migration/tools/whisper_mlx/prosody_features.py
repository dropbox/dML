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
Prosody Feature Extraction for RichDecoder V4.

Extracts F0 (pitch), energy, and delta features from audio for emotion recognition.
Features are frame-aligned with Whisper encoder output (50 fps after 2x conv stride).

Key Properties:
--------------
- **F0 Extraction**: YIN algorithm (robust, fast, no external dependencies)
- **Energy**: RMS energy per frame
- **Delta Features**: First derivative for rate-of-change information
- **Frame Alignment**: Matches Whisper's 20ms frame rate (50 fps)

Why Prosody Matters:
------------------
Emotion is ~80% prosody. Explicit F0/energy features are more reliable than
implicit learning from spectrograms because:
1. Mel spectrograms encode pitch implicitly across many channels
2. Neural networks may not learn to extract pitch efficiently
3. Explicit features provide a shortcut for emotion-relevant information

Architecture Integration:
-----------------------
    audio -> prosody_features() -> [T, 4]  (f0, energy, f0_delta, energy_delta)
    mel   -> whisper_encoder   -> [T, 1280]

    # Concatenate and project back
    combined = concat(encoder_out, prosody)  # [T, 1284]
    projected = linear(1284, 1280)(combined)  # [T, 1280]

Usage:
------
    from tools.whisper_mlx.prosody_features import extract_prosody_features

    # From audio array (16kHz)
    audio = load_audio("speech.wav")
    prosody = extract_prosody_features(audio, sr=16000)  # [T, 4]

    # For batch processing
    features = extract_prosody_batch(audio_batch, sr=16000)  # [B, T, 4]

References:
----------
- RICHDECODER_V4_ROADMAP.md Phase 2: Prosody Features
- YIN: A fundamental frequency estimator (de Cheveigné & Kawahara, 2002)
- Whisper uses 20ms frames (50 fps) after conv downsampling
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProsodyConfig:
    """Configuration for prosody feature extraction."""

    # Audio parameters
    sample_rate: int = 16000

    # Frame parameters (match Whisper's output rate)
    # Whisper: 10ms mel hop -> 2x conv stride -> 20ms output frames (50 fps)
    hop_length: int = 320  # 20ms at 16kHz
    frame_length: int = 640  # 40ms window for better pitch resolution

    # F0 parameters
    f0_min: float = 50.0   # Hz (low end for bass voices)
    f0_max: float = 500.0  # Hz (high end for soprano voices)

    # YIN parameters
    yin_threshold: float = 0.1  # Lower = stricter pitch detection

    # Normalization
    f0_mean: float = 200.0  # Hz (typical speech F0)
    f0_std: float = 100.0   # Hz (typical range)

    # Output features
    include_delta: bool = True  # Include first derivative
    include_delta_delta: bool = False  # Include second derivative


# =============================================================================
# YIN Pitch Estimation
# =============================================================================

def difference_function(x: np.ndarray, tau_max: int) -> np.ndarray:
    """
    Compute the YIN difference function.

    d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2

    This is computed efficiently using autocorrelation.
    """
    w = len(x) - tau_max

    # Autocorrelation trick: d(tau) = r(0) + r(tau) - 2*r(tau)
    # where r(tau) = sum(x[j] * x[j+tau])
    np.sum(x[:w] ** 2)

    # Compute d(tau) for all tau
    d = np.zeros(tau_max)
    for tau in range(1, tau_max):
        # Direct computation (could be optimized with FFT)
        diff = x[:w] - x[tau:tau + w]
        d[tau] = np.sum(diff ** 2)

    return d


def cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    """
    Compute cumulative mean normalized difference function (CMNDF).

    d'(tau) = d(tau) / ((1/tau) * sum_{j=1}^{tau} d(j))

    This normalizes the difference function for more robust pitch detection.
    """
    d_prime = np.zeros_like(d)
    d_prime[0] = 1.0

    cumsum = 0.0
    for tau in range(1, len(d)):
        cumsum += d[tau]
        if cumsum > 0:
            d_prime[tau] = d[tau] * tau / cumsum
        else:
            d_prime[tau] = 1.0

    return d_prime


def yin_pitch(
    frame: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 500.0,
    threshold: float = 0.1,
) -> float:
    """
    Estimate fundamental frequency of a single frame using YIN algorithm.

    Args:
        frame: Audio frame (1D array)
        sr: Sample rate
        fmin: Minimum F0 to consider (Hz)
        fmax: Maximum F0 to consider (Hz)
        threshold: YIN threshold (lower = stricter)

    Returns:
        Estimated F0 in Hz, or 0.0 if unvoiced
    """
    # Tau range corresponds to period range
    tau_min = max(2, int(sr / fmax))  # Minimum period (max frequency)
    tau_max = min(len(frame) // 2, int(sr / fmin))  # Maximum period (min frequency)

    if tau_max <= tau_min:
        return 0.0

    # Compute CMNDF
    d = difference_function(frame, tau_max)
    d_prime = cumulative_mean_normalized_difference(d)

    # Find first tau where d'(tau) < threshold
    for tau in range(tau_min, tau_max):
        if d_prime[tau] < threshold:
            # Parabolic interpolation for sub-sample accuracy
            if tau > 0 and tau < len(d_prime) - 1:
                # Fit parabola through 3 points
                s0, s1, s2 = d_prime[tau - 1], d_prime[tau], d_prime[tau + 1]
                delta = 0.5 * (s0 - s2) / (s0 - 2 * s1 + s2 + 1e-10)
                tau_refined = tau + delta
            else:
                tau_refined = tau

            # Convert period to frequency
            if tau_refined > 0:
                return sr / tau_refined
            return 0.0

    return 0.0  # Unvoiced


def extract_f0_yin(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 320,
    frame_length: int = 640,
    fmin: float = 50.0,
    fmax: float = 500.0,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Extract F0 contour using YIN algorithm.

    Args:
        audio: Audio waveform (1D, mono)
        sr: Sample rate
        hop_length: Samples between frames (default 320 = 20ms at 16kHz)
        frame_length: Frame size in samples (default 640 = 40ms)
        fmin: Minimum F0 (Hz)
        fmax: Maximum F0 (Hz)
        threshold: YIN threshold

    Returns:
        F0 contour array, shape (n_frames,)
        Unvoiced frames have value 0.0
    """
    # Ensure audio is float
    audio = np.asarray(audio, dtype=np.float64)

    # Calculate number of frames
    n_frames = max(1, (len(audio) - frame_length) // hop_length + 1)

    f0 = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length

        if end > len(audio):
            # Pad last frame if needed
            frame = np.pad(audio[start:], (0, end - len(audio)))
        else:
            frame = audio[start:end]

        f0[i] = yin_pitch(frame, sr, fmin, fmax, threshold)

    # Apply median filter to remove octave errors (optional but recommended)
    return median_filter_f0(f0, kernel_size=3)



def median_filter_f0(f0: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filter to F0 contour to remove octave errors.

    Only filters voiced frames (f0 > 0).
    """
    result = f0.copy()
    half_k = kernel_size // 2

    for i in range(half_k, len(f0) - half_k):
        if f0[i] > 0:
            # Get voiced neighbors
            window = f0[i - half_k:i + half_k + 1]
            voiced = window[window > 0]
            if len(voiced) > 0:
                result[i] = np.median(voiced)

    return result


# =============================================================================
# Energy Extraction
# =============================================================================

def extract_energy(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 320,
    frame_length: int = 640,
) -> np.ndarray:
    """
    Extract RMS energy per frame.

    Args:
        audio: Audio waveform (1D, mono)
        sr: Sample rate
        hop_length: Samples between frames
        frame_length: Frame size in samples

    Returns:
        Energy contour array, shape (n_frames,)
    """
    audio = np.asarray(audio, dtype=np.float64)

    n_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    energy = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length

        if end > len(audio):
            frame = np.pad(audio[start:], (0, end - len(audio)))
        else:
            frame = audio[start:end]

        # RMS energy
        energy[i] = np.sqrt(np.mean(frame ** 2))

    return energy


# =============================================================================
# Delta Features
# =============================================================================

def compute_delta(features: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Compute delta (derivative) features using first-order difference.

    For more robust deltas, use finite difference with regression:
    delta[t] = (sum_{n=1}^{N} n * (c[t+n] - c[t-n])) / (2 * sum_{n=1}^{N} n^2)

    This implementation uses simple gradient for speed.

    Args:
        features: Feature array, shape (T,) or (T, D)
        order: Derivative order (1=delta, 2=delta-delta)

    Returns:
        Delta features, same shape as input
    """
    if order == 0:
        return features

    # Simple gradient
    delta = np.gradient(features, axis=0)

    if order > 1:
        return compute_delta(delta, order - 1)

    return delta.astype(np.float32)


# =============================================================================
# Main Feature Extraction
# =============================================================================

def extract_prosody_features(
    audio: np.ndarray,
    sr: int = 16000,
    config: ProsodyConfig | None = None,
) -> np.ndarray:
    """
    Extract all prosody features aligned with Whisper frames.

    Features:
    - f0: Normalized fundamental frequency
    - energy: Normalized RMS energy
    - f0_delta: F0 rate of change (if config.include_delta)
    - energy_delta: Energy rate of change (if config.include_delta)

    Args:
        audio: Audio waveform (1D, mono, float)
        sr: Sample rate (should be 16000 for Whisper)
        config: Optional configuration

    Returns:
        Feature array of shape (n_frames, n_features)
        Default: (n_frames, 4) for [f0, energy, f0_delta, energy_delta]
    """
    if config is None:
        config = ProsodyConfig(sample_rate=sr)

    # Extract raw features
    f0 = extract_f0_yin(
        audio,
        sr=sr,
        hop_length=config.hop_length,
        frame_length=config.frame_length,
        fmin=config.f0_min,
        fmax=config.f0_max,
        threshold=config.yin_threshold,
    )

    energy = extract_energy(
        audio,
        sr=sr,
        hop_length=config.hop_length,
        frame_length=config.frame_length,
    )

    # Normalize F0
    # Replace unvoiced (0) with mean for normalization stability
    f0_voiced = f0.copy()
    f0_voiced[f0_voiced == 0] = config.f0_mean
    f0_norm = (f0_voiced - config.f0_mean) / config.f0_std

    # Normalize energy (z-score)
    energy_mean = np.mean(energy) if np.mean(energy) > 0 else 1e-6
    energy_std = np.std(energy) if np.std(energy) > 0 else 1e-6
    energy_norm = (energy - energy_mean) / energy_std

    # Build feature stack
    features = [f0_norm, energy_norm]

    if config.include_delta:
        f0_delta = compute_delta(f0_norm)
        energy_delta = compute_delta(energy_norm)
        features.extend([f0_delta, energy_delta])

    if config.include_delta_delta:
        f0_delta_delta = compute_delta(f0_norm, order=2)
        energy_delta_delta = compute_delta(energy_norm, order=2)
        features.extend([f0_delta_delta, energy_delta_delta])

    # Stack: [T, n_features]
    return np.stack(features, axis=-1).astype(np.float32)



def align_prosody_to_encoder(
    prosody: np.ndarray,
    encoder_len: int,
) -> np.ndarray:
    """
    Align prosody features to encoder output length.

    Whisper encoder output has a specific length based on audio duration.
    This function resamples prosody features to match.

    Args:
        prosody: Prosody features, shape (T_prosody, D)
        encoder_len: Target length (Whisper encoder output frames)

    Returns:
        Aligned features, shape (encoder_len, D)
    """
    T_prosody = prosody.shape[0]
    n_features = prosody.shape[1]

    if T_prosody == encoder_len:
        return prosody

    # Linear interpolation
    x_old = np.linspace(0, 1, T_prosody)
    x_new = np.linspace(0, 1, encoder_len)

    aligned = np.zeros((encoder_len, n_features), dtype=np.float32)
    for i in range(n_features):
        aligned[:, i] = np.interp(x_new, x_old, prosody[:, i])

    return aligned


# =============================================================================
# Batch Processing
# =============================================================================

def extract_prosody_batch(
    audio_batch: list[np.ndarray],
    sr: int = 16000,
    config: ProsodyConfig | None = None,
    max_len: int | None = None,
) -> np.ndarray:
    """
    Extract prosody features for a batch of audio.

    Args:
        audio_batch: List of audio arrays
        sr: Sample rate
        config: Optional configuration
        max_len: Maximum frame length (for padding)

    Returns:
        Batched features, shape (batch, max_len, n_features)
    """
    features_list = []

    for audio in audio_batch:
        features = extract_prosody_features(audio, sr, config)
        features_list.append(features)

    # Find max length
    if max_len is None:
        max_len = max(f.shape[0] for f in features_list)

    # Pad and stack
    n_features = features_list[0].shape[1]
    batch_size = len(features_list)

    result = np.zeros((batch_size, max_len, n_features), dtype=np.float32)

    for i, features in enumerate(features_list):
        T = min(features.shape[0], max_len)
        result[i, :T, :] = features[:T]

    return result


# =============================================================================
# Tests
# =============================================================================

def test_yin_pitch():
    """Test YIN pitch detection on synthetic sine wave."""
    print("Testing YIN pitch detection...")

    sr = 16000
    duration = 0.5  # 500ms

    # Generate 200Hz sine wave
    t = np.linspace(0, duration, int(sr * duration))
    freq = 200.0
    audio = np.sin(2 * np.pi * freq * t)

    # Extract F0
    f0 = extract_f0_yin(audio, sr=sr, hop_length=320, frame_length=640)

    # Check that most frames are close to 200Hz
    voiced = f0[f0 > 0]
    if len(voiced) > 0:
        mean_f0 = np.mean(voiced)
        print(f"  200Hz sine: detected mean F0 = {mean_f0:.1f}Hz")
        assert abs(mean_f0 - freq) < 20, f"F0 detection error: {mean_f0} vs {freq}"
    else:
        print("  WARNING: No voiced frames detected")

    print("  YIN pitch tests PASSED")
    return True


def test_energy_extraction():
    """Test energy extraction."""
    print("Testing energy extraction...")

    sr = 16000
    duration = 0.5

    # Silence
    silence = np.zeros(int(sr * duration))
    energy_silence = extract_energy(silence, sr=sr)
    assert np.all(energy_silence < 1e-6), "Silence should have near-zero energy"

    # White noise
    rng = np.random.default_rng()
    noise = rng.standard_normal(int(sr * duration)) * 0.5
    energy_noise = extract_energy(noise, sr=sr)
    assert np.mean(energy_noise) > 0.1, "Noise should have positive energy"

    print(f"  Silence energy: {np.mean(energy_silence):.6f}")
    print(f"  Noise energy: {np.mean(energy_noise):.4f}")
    print("  Energy tests PASSED")
    return True


def test_prosody_features():
    """Test full prosody feature extraction."""
    print("Testing prosody feature extraction...")

    sr = 16000
    duration = 1.0  # 1 second

    # Generate test audio (200Hz tone with amplitude modulation)
    t = np.linspace(0, duration, int(sr * duration))
    freq = 200.0
    amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2Hz modulation
    audio = amp_mod * np.sin(2 * np.pi * freq * t)

    # Extract features
    config = ProsodyConfig(sample_rate=sr, include_delta=True)
    prosody = extract_prosody_features(audio, sr=sr, config=config)

    print(f"  Audio length: {len(audio)} samples ({duration}s)")
    print(f"  Prosody shape: {prosody.shape}")

    # Should have 4 features: f0, energy, f0_delta, energy_delta
    assert prosody.shape[1] == 4, f"Expected 4 features, got {prosody.shape[1]}"

    # Check frame count (20ms hop = 50fps, 1s = 50 frames approximately)
    expected_frames = int(sr * duration / config.hop_length)
    assert abs(prosody.shape[0] - expected_frames) <= 2, \
        f"Frame count mismatch: {prosody.shape[0]} vs {expected_frames}"

    print("  Prosody feature tests PASSED")
    return True


def test_alignment():
    """Test alignment to encoder length."""
    print("Testing alignment to encoder...")

    rng = np.random.default_rng()
    prosody = rng.standard_normal((100, 4)).astype(np.float32)

    # Upsample
    aligned_up = align_prosody_to_encoder(prosody, 150)
    assert aligned_up.shape == (150, 4), f"Upsample failed: {aligned_up.shape}"

    # Downsample
    aligned_down = align_prosody_to_encoder(prosody, 75)
    assert aligned_down.shape == (75, 4), f"Downsample failed: {aligned_down.shape}"

    # Same size
    aligned_same = align_prosody_to_encoder(prosody, 100)
    assert aligned_same.shape == (100, 4), f"Same size failed: {aligned_same.shape}"
    assert np.allclose(aligned_same, prosody), "Same size should be identity"

    print("  Alignment tests PASSED")
    return True


def test_batch_processing():
    """Test batch prosody extraction."""
    print("Testing batch processing...")

    sr = 16000

    # Different length audios
    rng = np.random.default_rng()
    audio_batch = [
        rng.standard_normal(sr * 1),   # 1 second
        rng.standard_normal(sr * 2),   # 2 seconds
        rng.standard_normal(int(sr * 0.5)),  # 0.5 seconds
    ]

    features = extract_prosody_batch(audio_batch, sr=sr)

    print(f"  Batch shape: {features.shape}")
    assert features.shape[0] == 3, "Should have 3 samples"
    assert features.shape[2] == 4, "Should have 4 features"

    print("  Batch processing tests PASSED")
    return True


if __name__ == "__main__":
    test_yin_pitch()
    test_energy_extraction()
    test_prosody_features()
    test_alignment()
    test_batch_processing()
    print("\nAll prosody feature tests PASSED")
