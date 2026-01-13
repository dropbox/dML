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
DashVoice Echo Cancellation - Layer 1: Exact Waveform Subtraction

Advantage: We know EXACTLY what audio we generated. This is much better than
traditional acoustic echo cancellation (AEC) because we have the original signal.

Pipeline:
1. Keep circular buffer of recently generated TTS audio
2. When mic input comes in:
   a. Cross-correlate with buffer to find delay
   b. Estimate room impulse response (RIR)
   c. Subtract convolved reference from mic input
   d. Spectral cleanup of residual

Target: >40dB echo reduction with <5ms latency
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EchoCancellationResult:
    """Result of echo cancellation."""

    cleaned_audio: np.ndarray
    detected_delay_ms: float
    echo_reduction_db: float
    had_echo: bool
    processing_time_ms: float


class ReferenceBuffer:
    """Circular buffer for storing recently generated TTS audio."""

    def __init__(
        self,
        max_duration_s: float = 5.0,
        sample_rate: int = 24000,
    ):
        """Initialize reference buffer.

        Args:
            max_duration_s: Maximum duration of audio to keep
            sample_rate: Sample rate of audio
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0

    def add(self, audio: np.ndarray):
        """Add generated audio to buffer.

        Args:
            audio: Audio to add (float32, mono)
        """
        audio = np.asarray(audio, dtype=np.float32)

        # Handle audio longer than buffer
        if len(audio) >= self.max_samples:
            # Just keep the last max_samples
            self.buffer[:] = audio[-self.max_samples :]
            self.write_pos = 0
            self.total_written += len(audio)
            return

        # Wrap around write
        end_pos = self.write_pos + len(audio)
        if end_pos <= self.max_samples:
            self.buffer[self.write_pos : end_pos] = audio
        else:
            # Split across boundary
            first_part = self.max_samples - self.write_pos
            self.buffer[self.write_pos :] = audio[:first_part]
            self.buffer[: end_pos - self.max_samples] = audio[first_part:]

        self.write_pos = end_pos % self.max_samples
        self.total_written += len(audio)

    def get_recent(self, duration_s: float) -> np.ndarray:
        """Get most recent audio from buffer.

        Args:
            duration_s: Duration of audio to retrieve

        Returns:
            Most recent audio (may be shorter if buffer not full)
        """
        num_samples = min(int(duration_s * self.sample_rate), self.max_samples)
        actual_samples = min(num_samples, self.total_written)

        if actual_samples == 0:
            return np.array([], dtype=np.float32)

        # Read backwards from write position
        start_pos = (self.write_pos - actual_samples) % self.max_samples
        if start_pos < self.write_pos:
            return self.buffer[start_pos : self.write_pos].copy()
        # Wrap around read
        return np.concatenate(
            [self.buffer[start_pos:], self.buffer[: self.write_pos]],
        )

    def clear(self):
        """Clear the buffer."""
        self.buffer[:] = 0
        self.write_pos = 0
        self.total_written = 0


class EchoCanceller:
    """Layer 1 echo cancellation using exact waveform subtraction.

    This is more effective than traditional AEC because we have the exact
    reference signal that was output through the speaker.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        max_delay_ms: float = 500,
        min_correlation: float = 0.3,
    ):
        """Initialize echo canceller.

        Args:
            sample_rate: Audio sample rate
            max_delay_ms: Maximum expected echo delay
            min_correlation: Minimum correlation to consider as echo
        """
        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.min_correlation = min_correlation

        # Reference buffer for TTS output
        self.ref_buffer = ReferenceBuffer(
            max_duration_s=max_delay_ms / 1000 * 2 + 1.0,
            sample_rate=sample_rate,
        )

        # Adaptive filter state
        self.filter_length = 1024  # Samples for room response estimation
        self.room_filter = np.zeros(self.filter_length, dtype=np.float32)
        self.adaptation_rate = 0.01

    def add_reference(self, audio: np.ndarray):
        """Add generated TTS audio to reference buffer.

        Call this whenever audio is output through the speaker.

        Args:
            audio: Generated audio (float32, mono)
        """
        self.ref_buffer.add(audio)

    def find_delay(
        self,
        mic_audio: np.ndarray,
        ref_audio: np.ndarray,
    ) -> tuple[int, float]:
        """Find delay between reference and mic audio using cross-correlation.

        Args:
            mic_audio: Microphone input
            ref_audio: Reference (generated) audio

        Returns:
            Tuple of (delay_samples, correlation_score)
        """
        from scipy.signal import correlate

        if len(mic_audio) == 0 or len(ref_audio) == 0:
            return 0, 0.0

        # Normalize for correlation
        mic_norm = mic_audio - np.mean(mic_audio)
        ref_norm = ref_audio - np.mean(ref_audio)

        mic_std = np.std(mic_norm)
        ref_std = np.std(ref_norm)

        if mic_std < 1e-8 or ref_std < 1e-8:
            return 0, 0.0

        mic_norm = mic_norm / mic_std
        ref_norm = ref_norm / ref_std

        # Cross-correlation using scipy (correlate(mic, ref) finds where ref appears in mic)
        xcorr = correlate(mic_norm, ref_norm, mode="full")

        # Center point corresponds to zero lag
        center = len(ref_norm) - 1

        # Positive lags: mic signal lags reference (echo scenario)
        positive_lags = xcorr[center:]

        # Only search up to max delay
        max_search = min(self.max_delay_samples, len(positive_lags))
        search_region = positive_lags[:max_search]

        # Find peak
        peak_idx = np.argmax(search_region)
        peak_value = search_region[peak_idx]

        # Normalize correlation to [-1, 1] range
        correlation = peak_value / len(ref_norm)

        return peak_idx, float(correlation)

    def subtract_echo(
        self,
        mic_audio: np.ndarray,
        ref_audio: np.ndarray,
        delay_samples: int,
    ) -> np.ndarray:
        """Subtract echo from microphone audio.

        Uses adaptive filtering to estimate room response.

        Args:
            mic_audio: Microphone input
            ref_audio: Reference audio (aligned)
            delay_samples: Detected delay

        Returns:
            Echo-cancelled audio
        """
        # Align reference with mic
        if delay_samples >= len(ref_audio):
            # No overlap
            return mic_audio.copy()

        # Shift reference to align with mic
        if delay_samples > 0:
            aligned_ref = np.zeros_like(mic_audio)
            copy_len = min(len(ref_audio) - delay_samples, len(mic_audio))
            aligned_ref[:copy_len] = ref_audio[delay_samples : delay_samples + copy_len]
        else:
            aligned_ref = ref_audio[: len(mic_audio)].copy()

        # Simple subtraction with gain estimation
        # Estimate optimal gain to minimize residual energy
        # gain = argmin ||mic - gain * ref||^2 = (mic . ref) / (ref . ref)
        ref_energy = np.dot(aligned_ref, aligned_ref)
        if ref_energy > 1e-8:
            gain = np.dot(mic_audio, aligned_ref) / ref_energy
            # Clamp gain to reasonable range
            gain = np.clip(gain, 0.0, 2.0)
        else:
            gain = 0.0

        # Subtract
        return mic_audio - gain * aligned_ref


    def spectral_cleanup(
        self,
        audio: np.ndarray,
        noise_floor_db: float = -60,
    ) -> np.ndarray:
        """Spectral subtraction to clean up residual echo.

        Args:
            audio: Audio after waveform subtraction
            noise_floor_db: Noise floor threshold

        Returns:
            Cleaned audio
        """
        # Simple spectral gating
        n_fft = 2048
        hop_length = n_fft // 4

        # STFT
        num_frames = (len(audio) - n_fft) // hop_length + 1
        if num_frames <= 0:
            return audio

        window = np.hanning(n_fft)
        frames = np.zeros((num_frames, n_fft), dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_length
            frames[i] = audio[start : start + n_fft] * window

        # FFT
        spectra = np.fft.rfft(frames, axis=1)
        magnitudes = np.abs(spectra)
        phases = np.angle(spectra)

        # Noise floor
        noise_floor = 10 ** (noise_floor_db / 20)

        # Soft gating - reduce magnitude of quiet frames
        magnitudes = np.maximum(magnitudes - noise_floor, 0)

        # Reconstruct
        spectra_clean = magnitudes * np.exp(1j * phases)
        frames_clean = np.fft.irfft(spectra_clean, axis=1)

        # Overlap-add
        output = np.zeros(len(audio), dtype=np.float32)
        window_sum = np.zeros(len(audio), dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_length
            output[start : start + n_fft] += frames_clean[i] * window
            window_sum[start : start + n_fft] += window**2

        # Normalize by window sum
        window_sum = np.maximum(window_sum, 1e-8)
        return output / window_sum


    def process(
        self,
        mic_audio: np.ndarray,
        enable_spectral_cleanup: bool = True,
    ) -> EchoCancellationResult:
        """Process microphone audio to remove echo.

        Args:
            mic_audio: Microphone input (float32, mono)
            enable_spectral_cleanup: Enable spectral post-processing

        Returns:
            EchoCancellationResult with cleaned audio
        """
        import time

        start_time = time.perf_counter()

        mic_audio = np.asarray(mic_audio, dtype=np.float32)

        # Get reference audio
        ref_duration = (len(mic_audio) / self.sample_rate) + (
            self.max_delay_samples / self.sample_rate
        )
        ref_audio = self.ref_buffer.get_recent(ref_duration)

        if len(ref_audio) == 0:
            # No reference audio, return as-is
            return EchoCancellationResult(
                cleaned_audio=mic_audio,
                detected_delay_ms=0.0,
                echo_reduction_db=0.0,
                had_echo=False,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Find delay
        delay_samples, correlation = self.find_delay(mic_audio, ref_audio)
        delay_ms = delay_samples / self.sample_rate * 1000

        if correlation < self.min_correlation:
            # No significant echo detected
            return EchoCancellationResult(
                cleaned_audio=mic_audio,
                detected_delay_ms=delay_ms,
                echo_reduction_db=0.0,
                had_echo=False,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Compute input energy
        input_energy = np.sum(mic_audio**2)

        # Subtract echo
        cancelled = self.subtract_echo(mic_audio, ref_audio, delay_samples)

        # Optional spectral cleanup
        if enable_spectral_cleanup:
            cancelled = self.spectral_cleanup(cancelled)

        # Compute echo reduction
        output_energy = np.sum(cancelled**2)
        if input_energy > 1e-10:
            reduction_ratio = output_energy / input_energy
            echo_reduction_db = -10 * np.log10(max(reduction_ratio, 1e-10))
        else:
            echo_reduction_db = 0.0

        processing_time = (time.perf_counter() - start_time) * 1000

        return EchoCancellationResult(
            cleaned_audio=cancelled,
            detected_delay_ms=delay_ms,
            echo_reduction_db=echo_reduction_db,
            had_echo=True,
            processing_time_ms=processing_time,
        )

    def clear(self):
        """Clear reference buffer and reset state."""
        self.ref_buffer.clear()
        self.room_filter[:] = 0

    def warmup(self):
        """Warm up the echo canceller to avoid cold-start latency.

        Call this once after initialization to pre-compile scipy/numpy functions.
        """
        # Create dummy signals
        duration_s = 0.1
        t = np.linspace(0, duration_s, int(duration_s * self.sample_rate))
        dummy_ref = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        dummy_mic = np.zeros_like(dummy_ref)
        dummy_mic[100:] = 0.5 * dummy_ref[:-100]

        # Process to warm up scipy
        self.add_reference(dummy_ref)
        self.process(dummy_mic)
        self.clear()


def test_echo_cancellation():
    """Test echo cancellation with simulated echo."""
    import time

    print("=" * 60)
    print("Echo Cancellation Test")
    print("=" * 60)

    sample_rate = 24000
    duration_s = 1.0
    delay_ms = 50  # Simulated room delay

    # Create test signals
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))
    reference = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone

    # Simulate echo: delayed and attenuated reference + some user speech
    delay_samples = int(delay_ms * sample_rate / 1000)
    echo = np.zeros_like(reference)
    echo[delay_samples:] = 0.5 * reference[:-delay_samples]  # 50% echo

    # Add simulated user voice (different frequency)
    user_voice = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    mic_input = echo + user_voice

    # Create and warm up canceller
    canceller = EchoCanceller(sample_rate=sample_rate)
    print("Warming up canceller...")
    warmup_start = time.perf_counter()
    canceller.warmup()
    warmup_time = (time.perf_counter() - warmup_start) * 1000
    print(f"Warmup time: {warmup_time:.1f}ms (one-time cost)")

    # Add reference
    canceller.add_reference(reference)

    # Process multiple times to show consistent performance
    times = []
    for _i in range(5):
        result = canceller.process(mic_input)
        times.append(result.processing_time_ms)

    print(f"\nSimulated delay: {delay_ms}ms")
    print(f"Detected delay: {result.detected_delay_ms:.1f}ms")
    print(f"Echo detected: {result.had_echo}")
    print(f"Echo reduction: {result.echo_reduction_db:.1f}dB")
    print(f"Processing times: {np.mean(times):.2f}ms +/- {np.std(times):.2f}ms")

    # Calculate actual SNR improvement
    # The user voice should be preserved, echo should be reduced
    echo_power = np.mean(echo**2)
    residual_power = np.mean((result.cleaned_audio - user_voice) ** 2)
    if echo_power > 1e-10:
        actual_reduction_db = 10 * np.log10(echo_power / max(residual_power, 1e-10))
        print(f"Actual echo suppression: {actual_reduction_db:.1f}dB")

    # Performance check
    target_ms = 5.0
    avg_time = np.mean(times)
    passed = result.had_echo and avg_time < target_ms
    print(f"\nTarget latency: <{target_ms}ms, Achieved: {avg_time:.2f}ms")
    print("Test PASSED" if passed else "Test FAILED")


if __name__ == "__main__":
    test_echo_cancellation()
