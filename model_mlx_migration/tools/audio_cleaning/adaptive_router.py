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
Adaptive Audio Router - ASR-informed enhancement selection

Per architecture contracts A1, A6, A7:
- Skip enhancement if audio is clean (SNR > 15, T60 < 0.3)
- Skip enhancement if content is SINGING (destroys harmonics)
- Use ASR confidence to select between raw and enhanced
- Track harm rate (enhanced worse than raw)

Enhancement options:
- DeepFilterNet3 (preferred, if installed)
- Spectral subtraction (fallback)
"""

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tools.audio_cleaning.condition_estimator import (
    AudioCondition,
    ConditionEstimator,
    ContentType,
)
from tools.audio_cleaning.wpe_dereverb import WPEDereverberator


class Denoiser(Protocol):
    """Protocol for denoising implementations."""

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise audio signal."""
        ...


@dataclass
class EnhancementResult:
    """Result of adaptive enhancement."""

    audio: np.ndarray
    was_enhanced: bool
    enhancement_type: str  # "none", "denoise", "dereverb", "both"
    condition: AudioCondition
    raw_confidence: float
    enhanced_confidence: float
    routing_time_ms: float
    enhancement_time_ms: float


class SpectralSubtractionDenoiser:
    """
    Simple spectral subtraction denoiser as fallback.

    Not as good as DeepFilterNet but works without dependencies.
    """

    def __init__(self, noise_reduction_db: float = 15.0):
        self.noise_reduction_db = noise_reduction_db

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply spectral subtraction denoising.

        Args:
            audio: Input audio (float32, mono)
            sample_rate: Sample rate

        Returns:
            Denoised audio
        """
        audio = np.asarray(audio, dtype=np.float64)

        if len(audio) == 0:
            return audio.astype(np.float32)

        # STFT parameters
        n_fft = 2048
        hop_length = n_fft // 4
        window = np.hanning(n_fft)

        # Compute STFT
        n_frames = (len(audio) - n_fft) // hop_length + 1
        if n_frames < 1:
            return audio.astype(np.float32)

        spectra = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.complex128)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spectra[i] = np.fft.rfft(frame * window)

        magnitudes = np.abs(spectra)
        phases = np.angle(spectra)

        # Estimate noise from quietest 10% of frames
        frame_energy = np.sum(magnitudes**2, axis=1)
        n_noise_frames = max(1, n_frames // 10)
        noise_frame_indices = np.argsort(frame_energy)[:n_noise_frames]
        noise_estimate = np.mean(magnitudes[noise_frame_indices], axis=0)

        # Apply spectral subtraction
        reduction_factor = 10 ** (self.noise_reduction_db / 20)
        magnitudes_clean = np.maximum(magnitudes - reduction_factor * noise_estimate, 0)

        # Wiener-style smoothing to reduce musical noise
        gain = magnitudes_clean / (magnitudes + 1e-10)
        gain = np.minimum(gain, 1.0)

        # Apply gain
        spectra_clean = magnitudes * gain * np.exp(1j * phases)

        # Inverse STFT
        output = np.zeros(len(audio), dtype=np.float64)
        window_sum = np.zeros(len(audio), dtype=np.float64)

        for i in range(n_frames):
            start = i * hop_length
            frame = np.fft.irfft(spectra_clean[i])
            end = min(start + n_fft, len(output))
            output[start:end] += (frame * window)[: end - start]
            window_sum[start:end] += window[: end - start] ** 2

        # Normalize
        window_sum = np.maximum(window_sum, 1e-8)
        output = output / window_sum

        return output.astype(np.float32)


class DeepFilterNetDenoiser:
    """
    DeepFilterNet3 denoiser wrapper.

    Requires: pip install deepfilternet
    May also require: python scripts/patch_deepfilternet.py
    """

    def __init__(self):
        self.model = None
        self.df_state = None
        self._available = False
        self._init_error = None

        try:
            from df.enhance import enhance, init_df

            self.model, self.df_state, _ = init_df()
            self._enhance_fn = enhance
            self._available = True
        except ImportError as e:
            self._init_error = f"DeepFilterNet not installed: {e}"
        except Exception as e:
            self._init_error = f"DeepFilterNet init failed: {e}"

    @property
    def available(self) -> bool:
        return self._available

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply DeepFilterNet denoising.

        Args:
            audio: Input audio (float32, mono)
            sample_rate: Sample rate (will resample to 48kHz if needed)

        Returns:
            Denoised audio
        """
        if not self._available:
            raise RuntimeError(f"DeepFilterNet not available: {self._init_error}")

        import torch

        # DeepFilterNet expects 48kHz
        if sample_rate != 48000:
            # Simple linear resample for now
            # In production, use scipy or torchaudio resampling
            factor = 48000 / sample_rate
            n_out = int(len(audio) * factor)
            indices = np.linspace(0, len(audio) - 1, n_out)
            audio_48k = np.interp(indices, np.arange(len(audio)), audio)
        else:
            audio_48k = audio

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_48k).float().unsqueeze(0)

        # Enhance
        enhanced = self._enhance_fn(self.model, self.df_state, audio_tensor)

        # Convert back
        enhanced_np = enhanced.squeeze(0).numpy()

        # Resample back if needed
        if sample_rate != 48000:
            factor = sample_rate / 48000
            n_out = int(len(enhanced_np) * factor)
            indices = np.linspace(0, len(enhanced_np) - 1, n_out)
            enhanced_np = np.interp(indices, np.arange(len(enhanced_np)), enhanced_np)

        return enhanced_np.astype(np.float32)


class AdaptiveRouter:
    """
    Adaptive audio enhancement router.

    Routes audio through appropriate enhancement based on:
    1. Audio condition (SNR, reverb, content type)
    2. ASR confidence comparison (enhanced vs raw)

    Per architecture A7: harm rate must be < 5%
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        snr_threshold: float = 15.0,
        reverb_threshold: float = 0.3,
        harm_threshold: float = 0.05,
        prefer_deepfilternet: bool = True,
    ):
        """
        Initialize adaptive router.

        Args:
            sample_rate: Expected sample rate
            snr_threshold: SNR above this skips denoising
            reverb_threshold: T60 below this skips dereverb
            harm_threshold: Maximum acceptable harm rate
            prefer_deepfilternet: Try DeepFilterNet first
        """
        self.sample_rate = sample_rate
        self.snr_threshold = snr_threshold
        self.reverb_threshold = reverb_threshold
        self.harm_threshold = harm_threshold

        # Initialize components
        self.condition_estimator = ConditionEstimator(
            sample_rate=sample_rate,
            snr_clean_threshold=snr_threshold,
            reverb_clean_threshold=reverb_threshold,
        )

        # Try DeepFilterNet first, fall back to spectral subtraction
        self.denoiser: Denoiser
        if prefer_deepfilternet:
            df_denoiser = DeepFilterNetDenoiser()
            if df_denoiser.available:
                self.denoiser = df_denoiser
                self.denoiser_name = "DeepFilterNet"
            else:
                self.denoiser = SpectralSubtractionDenoiser()
                self.denoiser_name = "SpectralSubtraction"
        else:
            self.denoiser = SpectralSubtractionDenoiser()
            self.denoiser_name = "SpectralSubtraction"

        # Initialize WPE dereverberator
        self.dereverberator = WPEDereverberator()
        self.dereverberator_name = "WPE"

        # Harm tracking
        self.total_processed = 0
        self.harm_count = 0

        # Confidence estimator (placeholder - would use CTC in real implementation)
        self._confidence_fn = self._simple_confidence

    def _simple_confidence(self, audio: np.ndarray) -> float:
        """
        Simple confidence estimate based on signal clarity.

        In production, this would use a CTC model to compute
        actual ASR confidence. For now, use proxy metrics.
        """
        if len(audio) == 0:
            return 0.0

        # Use SNR as proxy for ASR confidence
        # Higher SNR generally means clearer speech
        snr = self.condition_estimator.estimate(audio).snr_db
        # Map SNR to 0-1 confidence
        # SNR 0-10: low confidence, 10-20: medium, 20+: high
        confidence = np.clip((snr - 5) / 25, 0, 1)
        return float(confidence)

    def route(
        self,
        audio: np.ndarray,
        force_enhance: bool = False,
    ) -> EnhancementResult:
        """
        Route audio through appropriate enhancement.

        Args:
            audio: Input audio (float32, mono)
            force_enhance: Force enhancement even if not needed

        Returns:
            EnhancementResult with processed audio and metadata
        """
        start_time = time.perf_counter()

        audio = np.asarray(audio, dtype=np.float32)

        # Analyze condition
        condition = self.condition_estimator.estimate(audio)
        routing_time = time.perf_counter() - start_time

        # Decide on enhancement
        should_denoise = condition.needs_denoising or force_enhance
        # Note: dereverb not yet implemented, tracked in Issue #3

        # BYPASS conditions (A4: singing destroys harmonics)
        if condition.content_type == ContentType.SINGING:
            return EnhancementResult(
                audio=audio,
                was_enhanced=False,
                enhancement_type="none",
                condition=condition,
                raw_confidence=1.0,
                enhanced_confidence=1.0,
                routing_time_ms=routing_time * 1000,
                enhancement_time_ms=0.0,
            )

        # CLEAN PATH (A1: skip if clean)
        if condition.is_clean and not force_enhance:
            return EnhancementResult(
                audio=audio,
                was_enhanced=False,
                enhancement_type="none",
                condition=condition,
                raw_confidence=1.0,
                enhanced_confidence=1.0,
                routing_time_ms=routing_time * 1000,
                enhancement_time_ms=0.0,
            )

        # ENHANCEMENT PATH
        enhance_start = time.perf_counter()
        enhanced = audio.copy()
        enhancement_type = "none"

        if should_denoise:
            try:
                enhanced = self.denoiser.denoise(enhanced, self.sample_rate)
                enhancement_type = "denoise"
            except Exception:
                pass  # Fall back to original

        # Apply dereverberation if needed
        if condition.needs_dereverb:
            try:
                enhanced = self.dereverberator.dereverb(enhanced, self.sample_rate)
                enhancement_type = "dereverb" if enhancement_type == "none" else "both"
            except Exception:
                pass  # Fall back to current state

        enhance_time = time.perf_counter() - enhance_start

        # ASR-INFORMED SELECTION (A6)
        raw_confidence = self._confidence_fn(audio)
        enhanced_confidence = self._confidence_fn(enhanced)

        self.total_processed += 1

        # Select based on confidence
        if enhanced_confidence >= raw_confidence:
            # Enhancement helped or was neutral
            final_audio = enhanced
            was_enhanced = True
        else:
            # Enhancement hurt - use raw
            final_audio = audio
            was_enhanced = False
            self.harm_count += 1

        return EnhancementResult(
            audio=final_audio,
            was_enhanced=was_enhanced,
            enhancement_type=enhancement_type if was_enhanced else "none",
            condition=condition,
            raw_confidence=raw_confidence,
            enhanced_confidence=enhanced_confidence,
            routing_time_ms=routing_time * 1000,
            enhancement_time_ms=enhance_time * 1000,
        )

    @property
    def harm_rate(self) -> float:
        """Fraction of times enhancement would have hurt ASR."""
        if self.total_processed == 0:
            return 0.0
        return self.harm_count / self.total_processed

    def check_harm_gate(self) -> bool:
        """Check if harm rate is within acceptable threshold (A7)."""
        return self.harm_rate < self.harm_threshold

    def warmup(self):
        """Warm up the router to avoid cold-start latency."""
        rng = np.random.default_rng()
        dummy = rng.standard_normal(self.sample_rate).astype(np.float32) * 0.1
        self.route(dummy)
        # Reset counters after warmup
        self.total_processed = 0
        self.harm_count = 0


def test_adaptive_router():
    """Test the adaptive router."""
    print("=" * 60)
    print("AdaptiveRouter Test")
    print("=" * 60)

    sample_rate = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))

    router = AdaptiveRouter(sample_rate=sample_rate)
    print(f"Denoiser: {router.denoiser_name}")
    print(f"Dereverberator: {router.dereverberator_name}")

    # Warmup
    print("\nWarming up...")
    router.warmup()

    # Test 1: Clean signal - should skip enhancement
    print("\n1. Clean signal:")
    clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = router.route(clean)
    print(f"   Was enhanced: {result.was_enhanced}")
    print(f"   Enhancement type: {result.enhancement_type}")
    print(f"   Routing time: {result.routing_time_ms:.2f} ms")

    # Test 2: Noisy signal - should denoise
    print("\n2. Noisy signal:")
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(t)).astype(np.float32) * 0.2
    noisy = clean + noise
    result = router.route(noisy)
    print(f"   Was enhanced: {result.was_enhanced}")
    print(f"   Enhancement type: {result.enhancement_type}")
    print(f"   Routing time: {result.routing_time_ms:.2f} ms")
    print(f"   Enhancement time: {result.enhancement_time_ms:.2f} ms")
    print(f"   Raw confidence: {result.raw_confidence:.3f}")
    print(f"   Enhanced confidence: {result.enhanced_confidence:.3f}")

    # Test 3: Reverberant signal - should dereverb
    print("\n3. Reverberant signal:")
    rir_len = int(0.5 * sample_rate)  # 500ms T60
    rir = np.exp(-np.linspace(0, 6, rir_len)) * rng.standard_normal(rir_len)
    rir = rir.astype(np.float32) / np.max(np.abs(rir))
    rir[0] = 1.0  # Direct path
    reverb = np.convolve(clean, rir, mode="same").astype(np.float32)
    result = router.route(reverb)
    print(f"   Was enhanced: {result.was_enhanced}")
    print(f"   Enhancement type: {result.enhancement_type}")
    print(f"   Needs dereverb: {result.condition.needs_dereverb}")
    print(f"   T60 estimate: {result.condition.reverb_t60:.2f}s")
    print(f"   Enhancement time: {result.enhancement_time_ms:.2f} ms")

    # Test 4: Force enhance clean signal
    print("\n4. Force enhance clean signal:")
    result = router.route(clean, force_enhance=True)
    print(f"   Was enhanced: {result.was_enhanced}")
    print(f"   Enhancement time: {result.enhancement_time_ms:.2f} ms")

    # Report harm rate
    print(f"\nHarm rate: {router.harm_rate * 100:.1f}%")
    print(f"Harm gate: {'PASS' if router.check_harm_gate() else 'FAIL'}")


if __name__ == "__main__":
    test_adaptive_router()
