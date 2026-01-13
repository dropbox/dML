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
Temporal Convolutional Network (TCN) for Conv-TasNet separator.

The TCN uses stacked dilated convolutions with exponentially increasing
dilation factors to capture long-range dependencies efficiently.

Note: This implementation uses channels-first format (B, C, T) externally
for compatibility with encoder/decoder, but converts to channels-last (B, T, C)
internally for MLX operations.
"""


import mlx.core as mx
import mlx.nn as nn


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization.

    Unlike standard LayerNorm which normalizes over the last dimension,
    GlobalLayerNorm normalizes over both time and channel dimensions.

    This is used in Conv-TasNet/Asteroid for processing variable-length sequences.

    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    where mean and var are computed over both time and channel dimensions.
    """

    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = mx.ones((num_features,))  # gamma
        self.bias = mx.zeros((num_features,))   # beta

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply global layer normalization.

        Args:
            x: Input tensor of shape (B, C, T) - channels-first format.

        Returns:
            Normalized tensor of same shape.
        """
        # x is (B, C, T)
        # Compute mean and var over both C and T dimensions
        mean = mx.mean(x, axis=(1, 2), keepdims=True)  # (B, 1, 1)
        var = mx.var(x, axis=(1, 2), keepdims=True)    # (B, 1, 1)

        # Normalize
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Apply learnable parameters
        # weight and bias are (C,), need to broadcast to (B, C, T)
        return x_norm * mx.expand_dims(self.weight, axis=(0, 2)) + mx.expand_dims(self.bias, axis=(0, 2))


def _apply_conv1d(conv: nn.Conv1d, x: mx.array) -> mx.array:
    """
    Apply Conv1d with channels-first input/output.

    MLX Conv1d expects (B, L, C_in), produces (B, L', C_out).
    This wrapper handles (B, C_in, L) -> (B, C_out, L') conversion.

    Args:
        conv: MLX Conv1d layer.
        x: Input tensor of shape (B, C_in, L).

    Returns:
        Output tensor of shape (B, C_out, L').
    """
    # Transpose to MLX format: (B, C, L) -> (B, L, C)
    x = mx.transpose(x, (0, 2, 1))
    # Apply convolution
    out = conv(x)
    # Transpose back: (B, L', C) -> (B, C, L')
    return mx.transpose(out, (0, 2, 1))


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution.

    Depthwise separable convolution factorizes a standard convolution into:
    1. Depthwise convolution: applies a single filter per input channel
    2. Pointwise convolution: 1x1 conv that combines depthwise outputs

    This reduces parameters significantly while maintaining representational power.

    Note: Input/output use channels-first format (B, C, T) for compatibility,
    but internal operations use MLX's channels-last format.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding

        # For dilation > 1, we use a standard convolution with manual dilation
        # For dilation = 1, we use depthwise separable

        # Depthwise convolution: each input channel gets its own filter
        # Note: MLX doesn't support groups in the same way as PyTorch
        # We'll use a regular conv and implement depthwise via weight structure
        self.depthwise_weight = mx.zeros((in_channels, kernel_size, 1))

        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply depthwise separable convolution.

        Args:
            x: Input tensor of shape (B, C_in, T) - channels-first.

        Returns:
            Output tensor of shape (B, C_out, T') - channels-first.
        """
        batch_size, channels, time_len = x.shape

        # Apply causal padding if needed
        if self.padding > 0:
            # Causal padding: pad only on the left (in time dimension)
            # x is (B, C, T), pad T dimension
            x = mx.pad(x, [(0, 0), (0, 0), (self.padding, 0)])
            x.shape[2]

        # Apply dilated depthwise convolution
        out = self._dilated_depthwise(x)

        # Apply pointwise convolution (1x1)
        return _apply_conv1d(self.pointwise, out)


    def _dilated_depthwise(self, x: mx.array) -> mx.array:
        """Apply dilated depthwise convolution."""
        batch_size, channels, time_len = x.shape
        k = self.kernel_size
        d = self.dilation

        # Calculate effective receptive field
        receptive_field = k + (k - 1) * (d - 1)

        # Calculate output length
        out_len = time_len - receptive_field + 1
        if out_len <= 0:
            out_len = 1
            # Pad if needed
            pad_needed = receptive_field - time_len
            x = mx.pad(x, [(0, 0), (0, 0), (pad_needed, 0)])
            time_len = x.shape[2]
            out_len = 1

        # For each output position, gather dilated samples and apply weights
        # Create output by gathering inputs at dilated positions
        # indices for kernel: [0, d, 2d, ..., (k-1)*d]

        # More efficient: use reshape and slice
        # For each channel, apply its own kernel
        outputs = []

        for t in range(out_len):
            # Gather dilated samples: positions t, t+d, t+2d, ..., t+(k-1)*d
            samples = []
            for i in range(k):
                idx = t + i * d
                samples.append(x[:, :, idx])  # (B, C)

            # Stack: (B, C, K)
            stacked = mx.stack(samples, axis=-1)

            # Apply depthwise weights: (B, C, K) * (C, K) -> sum -> (B, C)
            # depthwise_weight is (C, K, 1), squeeze to (C, K)
            weights = mx.squeeze(self.depthwise_weight, axis=-1)  # (C, K)
            conv_out = mx.sum(stacked * weights, axis=-1)  # (B, C)
            outputs.append(conv_out)

        # Stack: list of (B, C) -> (B, C, out_len)
        return mx.stack(outputs, axis=-1)



class TCNBlock(nn.Module):
    """
    Single TCN block with dilated convolution and skip connection.

    Each block consists of:
    1. 1x1 convolution (expand from bn_chan to hid_chan)
    2. PReLU activation
    3. GlobalLayerNorm
    4. Dilated depthwise convolution
    5. PReLU activation
    6. GlobalLayerNorm
    7. 1x1 convolution for residual (hid_chan to bn_chan)
    8. 1x1 convolution for skip (hid_chan to skip_chan)
    9. Residual connection

    Architecture matches Asteroid's TDConvNet block.
    Note: Input/output use channels-first format (B, C, T) for compatibility.
    """

    def __init__(
        self,
        bn_chan: int,
        hid_chan: int,
        skip_chan: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = True,
    ):
        super().__init__()
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal

        # Calculate padding for causal convolution
        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2

        # 1x1 expansion conv: bn_chan -> hid_chan
        self.conv1x1_1 = nn.Conv1d(bn_chan, hid_chan, 1, bias=True)

        # PReLU with single weight (broadcasts across channels)
        self.prelu1 = nn.PReLU(1)

        # GlobalLayerNorm on hid_chan (normalizes over both time and channels)
        self.norm1 = GlobalLayerNorm(hid_chan)

        # Dilated depthwise convolution (groups = hid_chan)
        # Note: MLX doesn't support groups, so we store weights and compute manually
        self.dwconv_weight = mx.zeros((hid_chan, kernel_size, 1))
        self.dwconv_bias = mx.zeros((hid_chan,))

        # Second PReLU
        self.prelu2 = nn.PReLU(1)

        # Second GlobalLayerNorm
        self.norm2 = GlobalLayerNorm(hid_chan)

        # Skip connection output (1x1 conv): hid_chan -> skip_chan
        self.conv1x1_skip = nn.Conv1d(hid_chan, skip_chan, 1, bias=True)

        # Residual output (1x1 conv): hid_chan -> bn_chan
        self.conv1x1_res = nn.Conv1d(hid_chan, bn_chan, 1, bias=True)

    def _apply_dilated_depthwise(self, x: mx.array) -> mx.array:
        """
        Apply dilated depthwise convolution.

        Args:
            x: Input tensor of shape (B, C, T) - channels-first.

        Returns:
            Output tensor of shape (B, C, T') - channels-first.
        """
        batch_size, channels, time_len = x.shape
        k = self.kernel_size
        d = self.dilation

        # Apply causal padding
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.padding, 0)])
            time_len = x.shape[2]

        # Calculate effective receptive field
        receptive_field = k + (k - 1) * (d - 1)

        # Calculate output length
        out_len = time_len - receptive_field + 1
        if out_len <= 0:
            out_len = 1
            pad_needed = receptive_field - time_len
            x = mx.pad(x, [(0, 0), (0, 0), (pad_needed, 0)])
            time_len = x.shape[2]

        # Compute dilated convolution per output position
        # For each channel, apply its own kernel (depthwise)
        outputs = []
        weights = mx.squeeze(self.dwconv_weight, axis=-1)  # (C, K)

        for t in range(out_len):
            # Gather dilated samples: positions t, t+d, t+2d, ..., t+(k-1)*d
            samples = []
            for i in range(k):
                idx = t + i * d
                samples.append(x[:, :, idx])  # (B, C)

            # Stack: (B, C, K)
            stacked = mx.stack(samples, axis=-1)

            # Apply depthwise weights: (B, C, K) * (C, K) -> sum -> (B, C)
            conv_out = mx.sum(stacked * weights, axis=-1)  # (B, C)

            # Add bias
            conv_out = conv_out + self.dwconv_bias

            outputs.append(conv_out)

        # Stack: list of (B, C) -> (B, C, out_len)
        return mx.stack(outputs, axis=-1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Apply TCN block.

        Args:
            x: Input tensor of shape (B, bn_chan, T) - channels-first.

        Returns:
            Tuple of (residual_output, skip_output):
                - residual_output: shape (B, bn_chan, T)
                - skip_output: shape (B, skip_chan, T)
        """
        # Store input for residual
        residual = x
        original_len = x.shape[2]

        # 1x1 expansion conv: bn_chan -> hid_chan
        out = _apply_conv1d(self.conv1x1_1, x)  # (B, hid_chan, T)

        # PReLU needs channels-last (B, T, C) format
        out = mx.transpose(out, (0, 2, 1))  # (B, T, hid_chan)
        out = self.prelu1(out)
        out = mx.transpose(out, (0, 2, 1))  # (B, hid_chan, T)

        # GlobalLayerNorm expects channels-first (B, C, T)
        out = self.norm1(out)  # (B, hid_chan, T)

        # Dilated depthwise convolution
        out = self._apply_dilated_depthwise(out)  # (B, hid_chan, T')

        # PReLU in channels-last format
        out = mx.transpose(out, (0, 2, 1))  # (B, T', hid_chan)
        out = self.prelu2(out)
        out = mx.transpose(out, (0, 2, 1))  # (B, hid_chan, T')

        # GlobalLayerNorm expects channels-first
        out = self.norm2(out)  # (B, hid_chan, T')

        # Ensure output matches input length (for residual and skip)
        if out.shape[2] > original_len:
            out = out[:, :, :original_len]
        elif out.shape[2] < original_len:
            pad_len = original_len - out.shape[2]
            out = mx.pad(out, [(0, 0), (0, 0), (0, pad_len)])

        # Skip connection output (1x1 conv): hid_chan -> skip_chan
        skip = _apply_conv1d(self.conv1x1_skip, out)  # (B, skip_chan, T)

        # Residual output (1x1 conv + input): hid_chan -> bn_chan
        res = _apply_conv1d(self.conv1x1_res, out) + residual  # (B, bn_chan, T)

        return res, skip


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network separator for Conv-TasNet.

    The TCN consists of:
    1. GlobalLayerNorm on encoder output
    2. Bottleneck convolution (N -> bn_chan)
    3. R stacks of X dilated conv blocks
    4. PReLU + mask estimation layer

    The exponentially increasing dilation factors (1, 2, 4, 8, ...) allow
    the network to have a large receptive field while keeping computation
    manageable.

    Note: Input/output use channels-first format (B, C, T) for compatibility
    with encoder/decoder.

    Architecture matches Asteroid's TDConvNet masker.

    Args:
        n_filters: Encoder output dimension (N), typically 512.
        bn_chan: Bottleneck channels (B), typically 128.
        hid_chan: Hidden channels inside TCN blocks (H), typically 512.
        skip_chan: Skip connection channels (Sc), typically 128.
        kernel_size: Conv kernel size, typically 3.
        n_layers: Number of layers per stack (X), typically 8.
        n_stacks: Number of stacks (R), typically 3.
        n_sources: Number of sources to separate (C), typically 2 or 3.
        causal: Whether to use causal convolutions for streaming.
    """

    def __init__(
        self,
        n_filters: int = 512,
        bn_chan: int = 128,
        hid_chan: int = 512,
        skip_chan: int = 128,
        kernel_size: int = 3,
        n_layers: int = 8,
        n_stacks: int = 3,
        n_sources: int = 2,
        causal: bool = True,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.n_sources = n_sources
        self.causal = causal

        # GlobalLayerNorm on encoder output (gamma/beta)
        self.layer_norm = GlobalLayerNorm(n_filters)

        # Bottleneck: N -> bn_chan (1x1 conv)
        self.bottleneck = nn.Conv1d(n_filters, bn_chan, 1, bias=True)

        # TCN blocks with exponentially increasing dilation
        self.blocks: list[TCNBlock] = []
        for _stack in range(n_stacks):
            for layer in range(n_layers):
                dilation = 2**layer
                block = TCNBlock(
                    bn_chan=bn_chan,
                    hid_chan=hid_chan,
                    skip_chan=skip_chan,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    causal=causal,
                )
                self.blocks.append(block)

        # PReLU before mask estimation (single weight broadcasts)
        self.mask_prelu = nn.PReLU(1)

        # Mask estimation: project skip outputs to masks (1x1 conv)
        # skip_chan -> n_sources * n_filters
        self.mask_conv = nn.Conv1d(
            skip_chan,
            n_sources * n_filters,
            kernel_size=1,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Estimate separation masks from encoder output.

        Args:
            x: Encoder output of shape (B, N, T') - channels-first.

        Returns:
            Separation masks of shape (B, C, N, T') where C is n_sources.
        """
        batch_size = x.shape[0]
        time_frames = x.shape[2]

        # GlobalLayerNorm expects channels-first (B, C, T)
        x = self.layer_norm(x)  # (B, N, T')

        # Bottleneck (1x1 conv): N -> bn_chan
        x = _apply_conv1d(self.bottleneck, x)  # (B, bn_chan, T')

        # TCN blocks with skip connections
        skip_sum = mx.zeros((batch_size, self.skip_chan, time_frames))

        for block in self.blocks:
            x, skip = block(x)
            # Ensure skip has same shape for addition
            if skip.shape[2] != time_frames:
                if skip.shape[2] > time_frames:
                    skip = skip[:, :, :time_frames]
                else:
                    pad_len = time_frames - skip.shape[2]
                    skip = mx.pad(skip, [(0, 0), (0, 0), (0, pad_len)])
            skip_sum = skip_sum + skip

        # PReLU on skip sum (single weight broadcasts)
        skip_sum = mx.transpose(skip_sum, (0, 2, 1))  # (B, T', skip_chan)
        skip_sum = self.mask_prelu(skip_sum)
        skip_sum = mx.transpose(skip_sum, (0, 2, 1))  # (B, skip_chan, T')

        # Mask estimation (1x1 conv): skip_chan -> n_sources * n_filters
        masks = _apply_conv1d(self.mask_conv, skip_sum)  # (B, C*N, T')

        # Apply ReLU activation for masks (Asteroid uses 'relu' mask_act)
        masks = mx.maximum(masks, 0)

        # Reshape to (B, C, N, T')
        return mx.reshape(
            masks,
            (batch_size, self.n_sources, self.n_filters, time_frames),
        )


    def get_receptive_field(self) -> int:
        """
        Calculate the receptive field of the TCN in samples.

        Returns:
            Receptive field size.
        """
        # Each stack has layers with dilation 1, 2, 4, ..., 2^(n_layers-1)
        # Receptive field per stack: sum of (k-1) * d for each layer
        rf_per_stack = sum(
            (self.kernel_size - 1) * (2**i) for i in range(self.n_layers)
        )
        # Total receptive field
        return rf_per_stack * self.n_stacks + 1
