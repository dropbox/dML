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
STFT/ISTFT implementation for MLX matching torch.stft/torch.istft behavior.

This module provides Short-Time Fourier Transform operations that match
PyTorch's torch.stft and torch.istft functions for audio processing.

Default parameters match StyleTTS2/Kokoro:
- filter_length=800 (n_fft)
- hop_length=200
- win_length=800
"""

import math

import mlx.core as mx
import mlx.nn as nn


def get_hann_window(win_length: int) -> mx.array:
    """Generate Hann window."""
    n = mx.arange(win_length, dtype=mx.float32)
    return 0.5 * (1 - mx.cos(2 * math.pi * n / win_length))


def frame_signal(x: mx.array, frame_length: int, hop_length: int) -> mx.array:
    """
    Frame a signal into overlapping windows.

    Args:
        x: [batch, samples] - Input signal
        frame_length: Window/frame size
        hop_length: Hop between frames

    Returns:
        frames: [batch, num_frames, frame_length]
    """
    batch, samples = x.shape

    # Calculate number of frames
    num_frames = (samples - frame_length) // hop_length + 1

    # Create frames using indexing
    # This is more memory-efficient than explicit loops
    indices = (
        mx.arange(frame_length)[None, :] + mx.arange(num_frames)[:, None] * hop_length
    )
    # indices: [num_frames, frame_length]

    # Gather frames for each batch
    return x[:, indices.flatten()].reshape(batch, num_frames, frame_length)



def overlap_add(frames: mx.array, hop_length: int, window: mx.array) -> mx.array:
    """
    Reconstruct signal from overlapping frames using overlap-add.

    Args:
        frames: [batch, num_frames, frame_length] - Windowed frames
        hop_length: Hop between frames
        window: [frame_length] - Window function for normalization

    Returns:
        signal: [batch, output_length]
    """
    batch, num_frames, frame_length = frames.shape
    output_length = (num_frames - 1) * hop_length + frame_length

    # Initialize output buffer
    output = mx.zeros((batch, output_length))

    # Overlap-add each frame
    # Note: MLX doesn't support in-place operations, so we build frame contributions
    # and sum them efficiently

    # Create contribution matrix for each frame position
    contributions = []
    for i in range(num_frames):
        start = i * hop_length
        # Create padded frame contribution
        pre_pad = mx.zeros((batch, start))
        post_pad_len = output_length - start - frame_length
        if post_pad_len > 0:
            post_pad = mx.zeros((batch, post_pad_len))
            contribution = mx.concatenate([pre_pad, frames[:, i, :], post_pad], axis=1)
        else:
            contribution = mx.concatenate(
                [pre_pad, frames[:, i, : output_length - start]], axis=1,
            )
        contributions.append(contribution)

    # Sum all contributions
    output = sum(contributions, mx.zeros((batch, output_length)))  # type: ignore[assignment]

    # Compute window SQUARED sum for normalization
    # Window is applied in both STFT (analysis) and ISTFT (synthesis), so
    # the normalization factor is window**2, not window.
    # See: https://pytorch.org/docs/stable/generated/torch.istft.html
    window_sq = window * window
    window_sq_sum = mx.zeros((output_length,))
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, output_length)
        actual_len = end - start
        # Build window squared sum similarly
        pre = mx.zeros((start,))
        if output_length - end > 0:
            post = mx.zeros((output_length - end,))
            ws_contrib = mx.concatenate([pre, window_sq[:actual_len], post], axis=0)
        else:
            ws_contrib = mx.concatenate([pre, window_sq[:actual_len]], axis=0)

        if i == 0:
            window_sq_sum = ws_contrib
        else:
            window_sq_sum = window_sq_sum + ws_contrib

    # Normalize by window squared sum (avoid division by zero)
    window_sq_sum = mx.maximum(window_sq_sum, 1e-8)
    return output / window_sq_sum



class TorchSTFT(nn.Module):
    """
    STFT/ISTFT matching PyTorch torch.stft/torch.istft behavior.

    Default parameters match StyleTTS2:
    - filter_length=800 (n_fft)
    - hop_length=200
    - win_length=800
    """

    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        # Pre-compute window
        self._window = get_hann_window(win_length)

    @property
    def window(self) -> mx.array:
        """Get the window, ensuring it's evaluated."""
        return self._window

    def transform(self, x: mx.array) -> tuple:
        """
        Forward STFT: time-domain -> frequency-domain.

        Args:
            x: [batch, samples] - Time-domain signal

        Returns:
            magnitude: [batch, n_fft//2+1, num_frames]
            phase: [batch, n_fft//2+1, num_frames]
        """
        # Ensure 2D input
        if x.ndim == 1:
            x = x[None, :]

        batch, samples = x.shape

        # Pad signal for complete frames (use constant padding, MLX doesn't support reflect)
        pad_amount = self.filter_length // 2
        x_padded = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])

        # Frame the signal
        frames = frame_signal(x_padded, self.win_length, self.hop_length)
        # frames: [batch, num_frames, win_length]

        # Apply window
        frames = frames * self.window

        # Zero-pad to filter_length if different from win_length
        if self.filter_length > self.win_length:
            pad_amount = self.filter_length - self.win_length
            frames = mx.pad(frames, [(0, 0), (0, 0), (0, pad_amount)])

        # Real FFT
        spectrum = mx.fft.rfft(frames, axis=-1)
        # spectrum: [batch, num_frames, n_fft//2+1]

        # Extract magnitude and phase
        magnitude = mx.abs(spectrum)
        phase = mx.arctan2(spectrum.imag, spectrum.real)

        # Transpose to [batch, n_fft//2+1, num_frames] to match PyTorch convention
        magnitude = magnitude.transpose(0, 2, 1)
        phase = phase.transpose(0, 2, 1)

        return magnitude, phase

    def inverse(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """
        Inverse STFT: frequency-domain -> time-domain.

        Args:
            magnitude: [batch, n_fft//2+1, num_frames]
            phase: [batch, n_fft//2+1, num_frames]

        Returns:
            signal: [batch, 1, samples] (extra dim for compatibility)
        """
        # Transpose to [batch, num_frames, n_fft//2+1]
        magnitude = magnitude.transpose(0, 2, 1)
        phase = phase.transpose(0, 2, 1)

        # Construct complex spectrum from magnitude and phase
        real = magnitude * mx.cos(phase)
        imag = magnitude * mx.sin(phase)

        # MLX rfft outputs complex array - need to construct it properly
        # Use complex arithmetic
        spectrum = real + 1j * imag

        # Inverse real FFT
        frames = mx.fft.irfft(spectrum, n=self.filter_length, axis=-1)
        # frames: [batch, num_frames, filter_length]

        # Truncate to win_length if needed
        if self.filter_length > self.win_length:
            frames = frames[..., : self.win_length]

        # Apply window (for proper overlap-add reconstruction)
        frames = frames * self.window

        # Overlap-add synthesis
        signal = overlap_add(frames, self.hop_length, self.window)

        # Remove padding that was added during forward transform
        pad_amount = self.filter_length // 2
        if signal.shape[1] > 2 * pad_amount:
            signal = signal[:, pad_amount:-pad_amount]

        # Add channel dimension for compatibility: [batch, 1, samples]
        return signal[:, None, :]


class SmallSTFT(nn.Module):
    """
    Small STFT for Generator output (post-convolution ISTFT).

    Uses smaller n_fft (20) and hop_size (5) for final audio reconstruction.
    This matches ISTFTNet's learned spectrogram approach.
    """

    def __init__(
        self,
        n_fft: int = 20,
        hop_size: int = 5,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size

        # Pre-compute window
        self._window = get_hann_window(n_fft)

    @property
    def window(self) -> mx.array:
        return self._window

    def inverse(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """
        Inverse STFT for generator output.

        Args:
            magnitude: [batch, n_fft//2+1, num_frames] - Linear magnitude
            phase: [batch, n_fft//2+1, num_frames] - Phase in radians

        Returns:
            signal: [batch, samples]
        """
        # Transpose to [batch, num_frames, n_fft//2+1]
        magnitude = magnitude.transpose(0, 2, 1)
        phase = phase.transpose(0, 2, 1)

        batch, num_frames, n_bins = magnitude.shape

        # Construct complex spectrum
        real = magnitude * mx.cos(phase)
        imag = magnitude * mx.sin(phase)
        spectrum = real + 1j * imag

        # Inverse real FFT
        frames = mx.fft.irfft(spectrum, n=self.n_fft, axis=-1)
        # frames: [batch, num_frames, n_fft]

        # Apply window
        frames = frames * self.window

        # Overlap-add
        return overlap_add(frames, self.hop_size, self.window)

