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
Silero VAD MLX Implementation

Voice Activity Detection model ported from PyTorch to MLX.
Original: https://github.com/snakers4/silero-vad (MIT License)

Architecture:
- STFT preprocessing (frame-based spectral features)
- 4× Conv1d encoder blocks with ReLU
- LSTM decoder with hidden state
- Conv1d → Sigmoid output (speech probability)

Input: Audio chunks (512 samples at 16kHz = 32ms)
Output: Speech probability [0, 1]
"""


import mlx.core as mx
import mlx.nn as nn
import numpy as np


class STFT(nn.Module):
    """Conv1d-based Short-Time Fourier Transform matching Silero's implementation."""

    def __init__(self, filter_length: int = 256, hop_length: int = 128):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length

        # Create DFT basis as conv filters (will be replaced by from_torch)
        # Shape: [2 * (filter_length//2 + 1), 1, filter_length] = [258, 1, 256]
        n_freq = filter_length // 2 + 1
        self.forward_basis_buffer = mx.zeros((2 * n_freq, 1, filter_length))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Compute STFT magnitude using conv1d (Silero-style).

        Args:
            x: Audio signal [batch, samples]

        Returns:
            Magnitude spectrogram [batch, n_fft//2 + 1, frames]
        """
        # Silero uses reflection padding on the right side (not zero padding)
        # This is critical for numerical exactness with PyTorch implementation
        pad_right = self.filter_length // 4  # 64 for 16kHz

        # Implement reflection padding: mirror the last pad_right samples
        # For input [a, b, c, d, e] with pad=2: [a, b, c, d, e, d, c]
        # The reflection excludes the edge sample itself
        reflected = x[:, -pad_right-1:-1]  # Get samples to reflect (excluding last)
        reflected = reflected[:, ::-1]  # Reverse them
        x = mx.concatenate([x, reflected], axis=1)

        # Add channel dim: [batch, samples] -> [batch, samples, 1] (NLC for MLX)
        x = x[:, :, None]

        # Conv1d with DFT basis
        # forward_basis_buffer: [out_channels, 1, kernel] -> MLX needs [out, kernel, in]
        # Weight is already in [out, 1, kernel] which matches MLX [out, kernel, in] when in=1
        weight = mx.transpose(self.forward_basis_buffer, [0, 2, 1])  # [258, 256, 1]
        forward_transform = mx.conv1d(x, weight, stride=self.hop_length, padding=0)
        # forward_transform: [batch, frames, 258]

        # Transpose to NCL: [batch, 258, frames]
        forward_transform = mx.transpose(forward_transform, [0, 2, 1])

        # Split into real and imaginary parts
        cutoff = self.filter_length // 2 + 1  # 129
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # Compute magnitude
        return mx.sqrt(real_part ** 2 + imag_part ** 2)



class SileroVadBlock(nn.Module):
    """Single encoder block: Conv1d + ReLU.

    Note: Input/output in NCL format (PyTorch convention).
    MLX Conv1d expects NLC, so we transpose internally.
    """

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


class LSTMCell(nn.Module):
    """Single LSTM cell for stateful inference."""

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


class SileroVADDecoder(nn.Module):
    """LSTM-based decoder for VAD."""

    def __init__(self, input_size: int = 128, hidden_size: int = 128):
        super().__init__()
        self.lstm = LSTMCell(input_size, hidden_size)
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

        # Decoder: Dropout → ReLU → Conv1d → Sigmoid
        # (Dropout is identity during inference)
        h = nn.relu(h)

        # Output projection (MLX Conv1d expects NLC)
        h = h[:, None, :]  # [batch, 1, channels] (NLC)
        prob = self.output_conv(h)  # [batch, 1, 1]
        prob = mx.sigmoid(prob)
        prob = prob[:, 0, :]  # [batch, 1]

        return prob, state


class SileroVAD(nn.Module):
    """
    Silero VAD model for MLX.

    Detects voice activity in audio chunks.
    """

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

        # STFT parameters based on sample rate
        if sample_rate == 16000:
            filter_length = 256
            hop_length = 128
            input_channels = 129  # filter_length // 2 + 1
        else:  # 8000
            filter_length = 128
            hop_length = 64
            input_channels = 65

        self.stft = STFT(filter_length=filter_length, hop_length=hop_length)

        # Encoder: 4 conv blocks with strides [1, 2, 2, 1]
        self.encoder = [
            SileroVadBlock(input_channels, 128, stride=1),
            SileroVadBlock(128, 64, stride=2),  # Downsampling
            SileroVadBlock(64, 64, stride=2),   # Downsampling
            SileroVadBlock(64, 128, stride=1),
        ]

        # Decoder
        self.decoder = SileroVADDecoder(128, 128)

        # Context size for streaming (prepended to each chunk)
        self.context_size = 64 if sample_rate == 16000 else 32

        # LSTM state for streaming
        self._state: tuple[mx.array, mx.array] | None = None
        self._context: mx.array | None = None

    def reset_state(self):
        """Reset LSTM state and context for new audio stream."""
        self._state = None
        self._context = None

    def __call__(
        self, x: mx.array, state: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Forward pass.

        Args:
            x: Audio chunk [batch, samples] (512 samples at 16kHz)
            state: Optional LSTM state for streaming

        Returns:
            prob: Speech probability [batch, 1]
            state: New LSTM state
        """
        batch_size = x.shape[0]

        # Initialize context if needed
        if self._context is None:
            self._context = mx.zeros((batch_size, self.context_size))

        # Prepend context to audio (streaming mode)
        x_with_context = mx.concatenate([self._context, x], axis=1)

        # Update context for next chunk (last context_size samples)
        self._context = x_with_context[:, -self.context_size:]

        # STFT
        x = self.stft(x_with_context)

        # Encoder
        for block in self.encoder:
            x = block(x)

        # Decoder
        prob, state = self.decoder(x, state or self._state)

        # Update internal state
        self._state = state

        return prob, state

    def detect(self, audio: mx.array, threshold: float = 0.5) -> mx.array:
        """
        Detect speech in audio.

        Args:
            audio: Full audio signal [samples] or [batch, samples]
            threshold: Speech detection threshold

        Returns:
            is_speech: Boolean array indicating speech presence
        """
        if audio.ndim == 1:
            audio = audio[None, :]  # Add batch dimension

        # Process in chunks
        chunk_size = 512 if self.sample_rate == 16000 else 256
        n_samples = audio.shape[1]
        n_chunks = n_samples // chunk_size

        self.reset_state()
        probs = []

        for i in range(n_chunks):
            chunk = audio[:, i * chunk_size : (i + 1) * chunk_size]
            prob, _ = self(chunk)
            probs.append(prob)

        if probs:
            probs = mx.concatenate(probs, axis=1)
            return probs > threshold
        return mx.array([[False]])

    @classmethod
    def from_torch(cls, torch_model, sample_rate: int = 16000) -> "SileroVAD":
        """
        Convert PyTorch Silero VAD model to MLX.

        Args:
            torch_model: PyTorch Silero VAD model
            sample_rate: Sample rate (16000 or 8000)

        Returns:
            MLX SileroVAD model
        """
        model = cls(sample_rate=sample_rate)

        # Get the appropriate sub-model
        if sample_rate == 16000:
            pt_model = torch_model._model
        else:
            pt_model = torch_model._model_8k

        # Convert STFT forward_basis_buffer
        stft_buffers = dict(pt_model.stft.named_buffers())
        if "forward_basis_buffer" in stft_buffers:
            basis = stft_buffers["forward_basis_buffer"].detach().numpy()
            model.stft.forward_basis_buffer = mx.array(basis)

        # Convert encoder weights
        state_dict = dict(pt_model.named_parameters())

        for i, block in enumerate(model.encoder):
            weight_key = f"encoder.{i}.reparam_conv.weight"
            bias_key = f"encoder.{i}.reparam_conv.bias"

            if weight_key in state_dict:
                # PyTorch Conv1d: [out, in, kernel] -> MLX: [out, kernel, in]
                weight = state_dict[weight_key].detach().numpy()
                weight = np.transpose(weight, (0, 2, 1))
                block.conv.weight = mx.array(weight)
                block.conv.bias = mx.array(state_dict[bias_key].detach().numpy())

        # Convert LSTM weights
        lstm_keys = ["decoder.rnn.weight_ih", "decoder.rnn.weight_hh", "decoder.rnn.bias_ih", "decoder.rnn.bias_hh"]
        for key in lstm_keys:
            if key in state_dict:
                attr_name = key.split(".")[-1]
                setattr(model.decoder.lstm, attr_name, mx.array(state_dict[key].detach().numpy()))

        # Convert output conv
        if "decoder.decoder.2.weight" in state_dict:
            weight = state_dict["decoder.decoder.2.weight"].detach().numpy()
            weight = np.transpose(weight, (0, 2, 1))
            model.decoder.output_conv.weight = mx.array(weight)
            model.decoder.output_conv.bias = mx.array(state_dict["decoder.decoder.2.bias"].detach().numpy())

        return model


def load_silero_vad(sample_rate: int = 16000) -> SileroVAD:
    """
    Load Silero VAD model converted to MLX.

    Args:
        sample_rate: Audio sample rate (16000 or 8000)

    Returns:
        SileroVAD model
    """
    import torch

    # Load PyTorch model
    torch_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=False,
    )

    # Convert to MLX
    return SileroVAD.from_torch(torch_model, sample_rate)

