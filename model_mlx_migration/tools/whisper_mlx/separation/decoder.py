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
Conv-TasNet Decoder module.

Transforms separated latent representations back to waveforms.
"""

import mlx.core as mx
import mlx.nn as nn


class Decoder(nn.Module):
    """
    1D transposed convolutional decoder for Conv-TasNet.

    Transforms separated latent representations back to waveforms using
    transposed convolution (learned synthesis basis).

    Note: MLX uses channels-last format (B, L, C). Input is expected in
    PyTorch-style (B, C, L) and transposed internally.

    The decoder reconstructs waveforms by applying learned synthesis filters,
    essentially performing the inverse operation of the encoder.

    Args:
        n_filters: Number of filters (must match encoder), typically 512.
        kernel_size: Kernel size in samples (must match encoder), typically 16.
        stride: Stride in samples (must match encoder), typically 8.
    """

    def __init__(
        self,
        n_filters: int = 512,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride

        # Transposed 1D convolution: (B, T', N) -> (B, T, 1) in MLX format
        self.deconv = nn.ConvTranspose1d(
            in_channels=n_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def __call__(
        self,
        encoded: mx.array,
        masks: mx.array,
    ) -> mx.array:
        """
        Decode separated sources to waveforms.

        Args:
            encoded: Encoder output of shape (B, N, T') - channels-first.
            masks: Separation masks of shape (B, C, N, T') where C is num_sources.

        Returns:
            Separated waveforms of shape (B, C, T).
        """
        batch_size = encoded.shape[0]
        num_sources = masks.shape[1]
        n_filters = encoded.shape[1]
        time_frames = encoded.shape[2]

        # Apply masks: (B, 1, N, T') * (B, C, N, T') -> (B, C, N, T')
        # Expand encoded for broadcasting: (B, N, T') -> (B, 1, N, T')
        encoded_expanded = mx.expand_dims(encoded, axis=1)
        masked = encoded_expanded * masks  # (B, C, N, T')

        # Reshape for batch processing: (B * C, N, T')
        masked_flat = mx.reshape(masked, (batch_size * num_sources, n_filters, time_frames))

        # Transpose to MLX channels-last format: (B * C, N, T') -> (B * C, T', N)
        masked_flat = mx.transpose(masked_flat, (0, 2, 1))

        # Apply transposed convolution: (B * C, T', N) -> (B * C, T, 1)
        waveforms_flat = self.deconv(masked_flat)

        # Remove channel dimension: (B * C, T, 1) -> (B * C, T)
        waveforms_flat = mx.squeeze(waveforms_flat, axis=-1)

        # Reshape back: (B * C, T) -> (B, C, T)
        output_length = waveforms_flat.shape[1]
        return mx.reshape(waveforms_flat, (batch_size, num_sources, output_length))


    def decode_single(self, masked: mx.array) -> mx.array:
        """
        Decode a single masked representation (no source dimension).

        Args:
            masked: Masked encoding of shape (B, N, T') - channels-first.

        Returns:
            Reconstructed waveform of shape (B, T).
        """
        # Transpose to MLX channels-last: (B, N, T') -> (B, T', N)
        masked = mx.transpose(masked, (0, 2, 1))

        # Apply transposed convolution: (B, T', N) -> (B, T, 1)
        waveform = self.deconv(masked)

        # Remove channel dimension: (B, T, 1) -> (B, T)
        return mx.squeeze(waveform, axis=-1)

    def get_output_length(self, input_frames: int) -> int:
        """
        Calculate output length given input frame count.

        Args:
            input_frames: Number of encoded frames.

        Returns:
            Number of output samples.
        """
        # Output length for transposed conv1d with padding=0
        return (input_frames - 1) * self.stride + self.kernel_size
