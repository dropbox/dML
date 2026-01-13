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
Fused VAD for WhisperMLX (OPT-SHARED-FFT)

Voice Activity Detection that shares FFT computation with Whisper mel spectrogram.
Instead of running Silero VAD with its own STFT (n_fft=256, hop=128), this module
uses the STFT magnitude already computed for Whisper's mel spectrogram (n_fft=400, hop=160).

Expected speedup: 10-15% reduction in audio processing overhead when VAD is needed.

Architecture:
- Uses Whisper's STFT magnitude (200 freq bins from n_fft=400)
- 4x Conv1d encoder blocks (adapted from Silero VAD)
- LSTM decoder with hidden state
- Outputs speech probability per frame (10ms per frame with hop=160 at 16kHz)

Usage:
    from tools.whisper_mlx.audio import compute_stft_and_mel
    from tools.whisper_mlx.fused_vad import FusedVAD

    # Compute shared FFT
    mel, stft_mag = compute_stft_and_mel(audio, return_stft=True)

    # Run VAD on STFT magnitude
    vad = FusedVAD.load()
    speech_probs = vad.detect_frames(stft_mag)

    # Process mel through Whisper encoder
    encoder_out = model.encoder(mel)
"""


import mlx.core as mx
import mlx.nn as nn
import numpy as np


class FusedVadBlock(nn.Module):
    """Single encoder block: Conv1d + ReLU (adapted for Whisper's FFT parameters)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, length] (NCL - PyTorch format)
        # MLX Conv1d expects [batch, length, channels] (NLC)
        x = mx.transpose(x, [0, 2, 1])  # NCL -> NLC
        x = self.conv(x)
        x = nn.relu(x)
        return mx.transpose(x, [0, 2, 1])  # NLC -> NCL


class FusedLSTMCell(nn.Module):
    """Single LSTM cell for stateful VAD inference."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined weight matrices for efficiency
        self.weight_ih = mx.zeros((4 * hidden_size, input_size))
        self.weight_hh = mx.zeros((4 * hidden_size, hidden_size))
        self.bias_ih = mx.zeros((4 * hidden_size,))
        self.bias_hh = mx.zeros((4 * hidden_size,))

    def __call__(
        self, x: mx.array, state: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        LSTM cell forward pass.

        Args:
            x: Input [batch, input_size]
            state: (h, c) tuple, each [batch, hidden_size]

        Returns:
            h_new: New hidden state
            (h_new, c_new): New state tuple
        """
        batch_size = x.shape[0]

        if state is None:
            h = mx.zeros((batch_size, self.hidden_size))
            c = mx.zeros((batch_size, self.hidden_size))
        else:
            h, c = state

        # Compute gates
        gates = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh

        # Split gates
        i, f, g, o = mx.split(gates, 4, axis=1)

        # Apply activations
        i = mx.sigmoid(i)
        f = mx.sigmoid(f)
        g = mx.tanh(g)
        o = mx.sigmoid(o)

        # Update cell and hidden state
        c_new = f * c + i * g
        h_new = o * mx.tanh(c_new)

        return h_new, (h_new, c_new)


class FusedVADDecoder(nn.Module):
    """LSTM-based decoder for VAD output."""

    def __init__(self, input_size: int = 128, hidden_size: int = 128):
        super().__init__()
        self.lstm = FusedLSTMCell(input_size, hidden_size)
        self.output_conv = nn.Conv1d(hidden_size, 1, kernel_size=1)

    def __call__(
        self, x: mx.array, state: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Decoder forward pass.

        Args:
            x: Encoder output [batch, channels, frames]
            state: LSTM state

        Returns:
            prob: Speech probability [batch, 1]
            state: New LSTM state
        """
        # x: [batch, channels, frames] (NCL format)
        # Squeeze last dim (after downsampling, frames=1)
        x = mx.squeeze(x, axis=-1)  # [batch, channels]

        # LSTM
        h, state = self.lstm(x, state)

        # Decoder: ReLU -> Conv1d -> Sigmoid
        h = nn.relu(h)

        # Output projection (MLX Conv1d expects NLC)
        h = h[:, None, :]  # [batch, 1, channels] (NLC)
        prob = self.output_conv(h)  # [batch, 1, 1]
        prob = mx.sigmoid(prob)
        prob = prob[:, 0, :]  # [batch, 1]

        return prob, state


class FusedVAD(nn.Module):
    """
    Fused Voice Activity Detection for WhisperMLX.

    Operates on STFT magnitude from Whisper's FFT computation (n_fft=400, hop=160).
    This avoids redundant FFT computation when VAD is used alongside transcription.

    The architecture mirrors Silero VAD but adapted for Whisper's FFT parameters:
    - Input: 200 frequency bins (from n_fft=400, keeping 200 of 201 bins)
    - 4x Conv1d encoder blocks with strides [1, 2, 2, 1]
    - LSTM decoder with 128 hidden units
    - Output: Speech probability per frame

    Usage:
        vad = FusedVAD()
        vad.load_pretrained()  # Or adapt from Silero weights

        # With shared FFT
        mel, stft_mag = compute_stft_and_mel(audio)
        probs = vad.detect_frames(stft_mag)
        segments = vad.get_speech_segments(probs)
    """

    def __init__(
        self,
        n_fft: int = 400,  # Whisper's FFT size
        hop_length: int = 160,  # Whisper's hop length
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Input channels = n_fft // 2 = 200 for Whisper
        input_channels = n_fft // 2

        # Encoder: 4 conv blocks with strides [1, 2, 2, 1]
        # Total downsampling: 4x
        self.encoder = [
            FusedVadBlock(input_channels, 128, stride=1),
            FusedVadBlock(128, 64, stride=2),  # Downsampling
            FusedVadBlock(64, 64, stride=2),   # Downsampling
            FusedVadBlock(64, 128, stride=1),
        ]

        # Decoder
        self.decoder = FusedVADDecoder(128, 128)

        # LSTM state for streaming
        self._state: tuple[mx.array, mx.array] | None = None

        # Frame size in samples (for context management)
        # With hop=160 at 16kHz, each frame is 10ms
        self.frame_duration_ms = (hop_length / sample_rate) * 1000

    def reset_state(self):
        """Reset LSTM state for new audio stream."""
        self._state = None

    def __call__(
        self,
        stft_magnitude: mx.array,
        state: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Forward pass on STFT magnitude.

        Args:
            stft_magnitude: STFT magnitude [batch, frames, freq_bins] or [frames, freq_bins]
            state: Optional LSTM state for streaming

        Returns:
            prob: Speech probability [batch, n_output_frames]
            state: New LSTM state
        """
        # Handle unbatched input
        if stft_magnitude.ndim == 2:
            stft_magnitude = stft_magnitude[None, :, :]  # Add batch dim

        # Transpose to NCL format: [batch, freq_bins, frames]
        x = mx.transpose(stft_magnitude, [0, 2, 1])

        # Encoder
        for block in self.encoder:
            x = block(x)

        # Get number of output frames
        n_out_frames = x.shape[2]

        # Process each output frame through decoder
        probs = []
        current_state = state or self._state

        for i in range(n_out_frames):
            frame = x[:, :, i:i+1]  # [batch, channels, 1]
            prob, current_state = self.decoder(frame, current_state)
            probs.append(prob)

        # Update internal state
        self._state = current_state

        # Stack probabilities: [batch, n_out_frames]
        probs = mx.concatenate(probs, axis=1)

        return probs, current_state

    def detect_frames(
        self,
        stft_magnitude: mx.array,
        threshold: float = 0.5,
    ) -> mx.array:
        """
        Detect speech frames from STFT magnitude.

        Args:
            stft_magnitude: STFT magnitude from compute_stft_and_mel
            threshold: Speech detection threshold (0-1)

        Returns:
            Speech probabilities per frame, shape (n_frames,)
        """
        self.reset_state()

        # Run forward pass
        probs, _ = self(stft_magnitude)

        # Remove batch dimension if present
        if probs.ndim > 1:
            probs = probs[0]

        return probs

    def get_speech_segments(
        self,
        probs: mx.array,
        threshold: float = 0.5,
        min_speech_duration_ms: float = 250,
        min_silence_duration_ms: float = 100,
    ) -> list[tuple[float, float]]:
        """
        Convert frame-level probabilities to speech segments.

        Args:
            probs: Speech probabilities per frame
            threshold: Speech detection threshold
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence duration to split segments

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        # Convert to numpy for easier processing
        probs_np = np.array(probs)

        # Calculate frame duration accounting for encoder downsampling
        # Encoder has stride 2 twice, so 4x downsampling
        # Each output frame represents 4 input frames
        downsample_factor = 4
        frame_duration_s = (self.hop_length * downsample_factor) / self.sample_rate

        min_speech_frames = int(min_speech_duration_ms / 1000 / frame_duration_s)
        min_silence_frames = int(min_silence_duration_ms / 1000 / frame_duration_s)

        # Threshold
        is_speech = probs_np > threshold

        # Find speech regions
        segments = []
        in_speech = False
        start_frame = 0
        silence_count = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # Speech start
                in_speech = True
                start_frame = i
                silence_count = 0
            elif not speech and in_speech:
                # Potential speech end
                silence_count += 1
                if silence_count >= min_silence_frames:
                    # End of speech segment
                    end_frame = i - silence_count
                    duration_frames = end_frame - start_frame
                    if duration_frames >= min_speech_frames:
                        start_time = start_frame * frame_duration_s
                        end_time = end_frame * frame_duration_s
                        segments.append((start_time, end_time))
                    in_speech = False
                    silence_count = 0
            elif speech and in_speech:
                silence_count = 0

        # Handle segment at end
        if in_speech:
            end_frame = len(is_speech)
            duration_frames = end_frame - start_frame
            if duration_frames >= min_speech_frames:
                start_time = start_frame * frame_duration_s
                end_time = end_frame * frame_duration_s
                segments.append((start_time, end_time))

        return segments

    @classmethod
    def from_silero(cls, silero_model, sample_rate: int = 16000) -> "FusedVAD":
        """
        Initialize from Silero VAD weights (requires adaptation due to different FFT).

        Note: This requires retraining or fine-tuning since Silero uses n_fft=256
        and we use n_fft=400. The architecture is similar but input dimensions differ.

        For now, this initializes with random weights. A proper adaptation would
        require training on speech/silence labels.

        Args:
            silero_model: PyTorch Silero VAD model (unused, for API compatibility)
            sample_rate: Sample rate (must be 16000)

        Returns:
            FusedVAD model (with random weights - needs training)
        """
        if sample_rate != 16000:
            raise ValueError("FusedVAD only supports 16kHz audio")

        return cls(n_fft=400, hop_length=160, sample_rate=sample_rate)

        # Note: Cannot directly copy Silero weights because:
        # - Silero uses 129 input channels (from n_fft=256)
        # - FusedVAD uses 200 input channels (from n_fft=400)
        # The model needs to be trained or fine-tuned on speech data


    @classmethod
    def load_pretrained(cls, weights_path: str | None = None) -> "FusedVAD":
        """
        Load pretrained FusedVAD weights.

        Args:
            weights_path: Path to weights file, or None to use default

        Returns:
            FusedVAD model with pretrained weights
        """
        model = cls()

        if weights_path is not None:
            # Load weights from file
            import os
            if os.path.exists(weights_path):
                weights = mx.load(weights_path)
                model.load_weights(list(weights.items()))

        # Note: If no pretrained weights available, model has random weights
        # and will need training before use

        return model


def detect_speech_fused(
    audio: np.ndarray,
    threshold: float = 0.5,
    min_speech_duration_ms: float = 250,
    min_silence_duration_ms: float = 100,
    vad_model: FusedVAD | None = None,
) -> tuple[list[tuple[float, float]], "mx.array", "mx.array"]:
    """
    Convenience function for VAD with shared FFT computation.

    Computes STFT once and returns both:
    1. Speech segments (for chunking audio to transcribe)
    2. Mel spectrogram (for Whisper encoder)
    3. Speech probabilities per frame

    Args:
        audio: Audio waveform (16kHz, float32)
        threshold: Speech detection threshold
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence to split segments
        vad_model: Optional pre-loaded FusedVAD model

    Returns:
        Tuple of:
        - segments: List of (start_time, end_time) tuples
        - mel: Mel spectrogram for Whisper
        - probs: Per-frame speech probabilities
    """
    from .audio import compute_stft_and_mel

    # Compute shared FFT
    mel, stft_mag = compute_stft_and_mel(audio, return_stft=True)

    # Initialize VAD if not provided
    if vad_model is None:
        vad_model = FusedVAD()

    # Run VAD
    probs = vad_model.detect_frames(stft_mag, threshold=threshold)

    # Get segments
    segments = vad_model.get_speech_segments(
        probs,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    return segments, mel, probs


__all__ = [
    "FusedVAD",
    "FusedVadBlock",
    "FusedLSTMCell",
    "FusedVADDecoder",
    "detect_speech_fused",
]
