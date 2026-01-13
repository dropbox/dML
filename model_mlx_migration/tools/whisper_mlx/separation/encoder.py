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
Conv-TasNet Encoder module.

Transforms raw waveform into latent representation using 1D convolution.
"""

import mlx.core as mx
import mlx.nn as nn


class Encoder(nn.Module):
    """
    1D convolutional encoder for Conv-TasNet.

    Transforms raw waveform (B, T) into a latent representation (B, T', N)
    where N is the number of filters and T' is the compressed time dimension.

    Note: MLX uses channels-last format (B, L, C), not PyTorch's (B, C, L).
    Output is transposed to (B, N, T') for compatibility with separator.

    The encoder is essentially a learned basis decomposition, replacing
    the traditional STFT with learnable filterbanks.

    Args:
        n_filters: Number of filters (latent dimension), typically 512.
        kernel_size: Kernel size in samples, typically 16 (1ms at 16kHz).
        stride: Stride in samples, typically kernel_size // 2 = 8.
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

        # 1D convolution: (B, T, 1) -> (B, T', N)
        # MLX uses channels-last: input (N, L, C_in), output (N, L', C_out)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Encode waveform to latent representation.

        Args:
            x: Input waveform of shape (B, T).

        Returns:
            Encoded representation of shape (B, N, T').
            Note: Output is transposed for compatibility with PyTorch-style separator.
        """
        # Ensure 3D input: (B, T, 1) - MLX channels-last format
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=-1)  # (B, T) -> (B, T, 1)
        elif x.ndim == 3 and x.shape[1] == 1:
            # Handle (B, 1, T) PyTorch-style input
            x = mx.transpose(x, (0, 2, 1))  # (B, 1, T) -> (B, T, 1)

        # Apply convolution: (B, T, 1) -> (B, T', N)
        encoded = self.conv(x)

        # No activation applied - Asteroid uses encoder_activation=None
        # The learned filterbank can produce positive and negative values
        # Masks are applied element-wise and handle both cases

        # Transpose to (B, N, T') for compatibility with separator
        return mx.transpose(encoded, (0, 2, 1))


    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output length given input length.

        Args:
            input_length: Number of input samples.

        Returns:
            Number of output frames.
        """
        # Output length for conv1d with padding=0
        return (input_length - self.kernel_size) // self.stride + 1
