"""ECAPA-TDNN Layer Implementations in MLX - Native Format.

This module implements the core building blocks for ECAPA-TDNN using
MLX's NATIVE tensor format (Batch, Time, Channels) throughout.

NO TRANSPOSITIONS - all operations use (B, T, C) format.

Layers:
- BatchNorm1d: 1D Batch Normalization (inference mode)
- Conv1d: 1D Convolution (native format)
- TDNNBlock: TDNN layer with Conv1d + BatchNorm + ReLU
- Res2NetBlock: Multi-scale feature extraction
- SEBlock: Squeeze-and-Excitation attention
- SERes2NetBlock: Combined SE + Res2Net block
- AttentiveStatisticsPooling: Temporal pooling with attention
"""

import mlx.core as mx
import mlx.nn as nn


class BatchNorm1d(nn.Module):
    """1D Batch Normalization for inference - Native (B, T, C) format.

    Uses running statistics from training (no updates during inference).

    Args:
        num_features: Number of channels
        eps: Small constant for numerical stability
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Learned parameters
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))

        # Running statistics (set during weight loading)
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply batch normalization.

        Args:
            x: Input tensor of shape (batch, time, channels) - NATIVE MLX FORMAT

        Returns:
            Normalized tensor of same shape (batch, time, channels)
        """
        # x shape: (B, T, C) - native MLX format
        # Normalize along channel dimension (last axis)
        mean = self.running_mean
        var = self.running_var

        # Reshape for broadcasting: (C,) -> (1, 1, C)
        mean = mean[None, None, :]
        var = var[None, None, :]
        weight = self.weight[None, None, :]
        bias = self.bias[None, None, :]

        x_norm = (x - mean) / mx.sqrt(var + self.eps)
        return weight * x_norm + bias


class Conv1d(nn.Module):
    """1D Convolution layer - Native (B, T, C) format.

    NO TRANSPOSITIONS - operates directly on MLX native format.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding added to input
        dilation: Dilation factor
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight shape for MLX conv1d: (out_channels, kernel_size, in_channels)
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply 1D convolution.

        Args:
            x: Input tensor of shape (batch, time, channels) - NATIVE MLX FORMAT

        Returns:
            Output tensor of shape (batch, new_time, out_channels)
        """
        # MLX conv1d: (B, T, C) -> (B, T_out, C_out) - native format, no transpose!

        # Apply padding if needed
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])

        # MLX conv1d
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,  # We already padded
            dilation=self.dilation,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class TDNNBlock(nn.Module):
    """TDNN Block: Conv1d + BatchNorm + ReLU - Native (B, T, C) format.

    Time Delay Neural Network block used in ECAPA-TDNN.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolution
        dilation: Dilation factor for convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        # Calculate padding to maintain time dimension
        padding = (kernel_size - 1) // 2 * dilation

        self.conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = BatchNorm1d(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, time, channels) - NATIVE MLX FORMAT

        Returns:
            Output tensor (batch, time, out_channels)
        """
        x = self.conv(x)
        x = self.norm(x)
        return nn.relu(x)


class Res2NetBlock(nn.Module):
    """Res2Net Block: Multi-scale feature extraction - Native (B, T, C) format.

    Splits channels into multiple groups and processes hierarchically.

    Args:
        channels: Total number of channels (must be divisible by scale)
        kernel_size: Kernel size for convolutions
        dilation: Dilation factor
        scale: Number of feature groups
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        scale: int = 8,
    ):
        super().__init__()
        self.scale = scale
        self.channels_per_scale = channels // scale

        # One TDNNBlock per scale (except first which is identity)
        self.blocks = [
            TDNNBlock(
                self.channels_per_scale,
                self.channels_per_scale,
                kernel_size,
                dilation,
            )
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with multi-scale processing.

        Args:
            x: Input tensor (batch, time, channels) - NATIVE MLX FORMAT

        Returns:
            Output tensor (batch, time, channels)
        """
        # Split into scale groups along channel dimension (axis=2 for B,T,C)
        # x: (B, T, C) -> split into (B, T, C/scale) chunks
        chunks = mx.split(x, self.scale, axis=2)

        outputs = []
        previous = None

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: identity
                outputs.append(chunk)
                previous = chunk
            else:
                # Add previous output and process
                chunk = chunk + previous
                out = self.blocks[i - 1](chunk)
                outputs.append(out)
                previous = out

        # Concatenate along channel dimension (axis=2 for B,T,C)
        return mx.concatenate(outputs, axis=2)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - Native (B, T, C) format.

    Channel attention mechanism that recalibrates channel responses.

    Args:
        channels: Number of input/output channels
        se_channels: Number of channels in bottleneck (squeeze dimension)
    """

    def __init__(self, channels: int, se_channels: int):
        super().__init__()
        self.conv1 = Conv1d(channels, se_channels, kernel_size=1)
        self.conv2 = Conv1d(se_channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with channel attention.

        Args:
            x: Input tensor (batch, time, channels) - NATIVE MLX FORMAT

        Returns:
            Channel-weighted tensor (batch, time, channels)
        """
        # Global average pooling over time (axis=1 for B,T,C)
        # x: (B, T, C) -> (B, 1, C)
        s = mx.mean(x, axis=1, keepdims=True)

        # Squeeze
        s = self.conv1(s)
        s = nn.relu(s)

        # Excitation
        s = self.conv2(s)
        s = mx.sigmoid(s)

        # Scale
        return x * s


class SERes2NetBlock(nn.Module):
    """SE-Res2Net Block: TDNN + Res2Net + TDNN + SE with residual - Native format.

    The main building block of ECAPA-TDNN.

    Args:
        channels: Number of channels (1024 in VoxLingua107)
        kernel_size: Kernel size for Res2Net convolutions
        dilation: Dilation factor
        scale: Res2Net scale factor
        se_channels: SE bottleneck channels
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()
        # First TDNN: pointwise conv
        self.tdnn1 = TDNNBlock(channels, channels, kernel_size=1)

        # Res2Net block
        self.res2net_block = Res2NetBlock(
            channels,
            kernel_size,
            dilation,
            scale,
        )

        # Second TDNN: pointwise conv
        self.tdnn2 = TDNNBlock(channels, channels, kernel_size=1)

        # SE block
        self.se_block = SEBlock(channels, se_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (batch, time, channels) - NATIVE MLX FORMAT

        Returns:
            Output tensor (batch, time, channels)
        """
        residual = x

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)

        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling - Native (B, T, C) format.

    Computes attention-weighted mean and standard deviation over time.

    Args:
        channels: Number of input channels
        attention_channels: Hidden dimension for attention
    """

    def __init__(self, channels: int, attention_channels: int):
        super().__init__()
        # Input: features concatenated with global context
        # For VoxLingua107: 3072 * 3 = 9216 (features, mean, std)
        self.tdnn = TDNNBlock(channels * 3, attention_channels, kernel_size=1)
        self.conv = Conv1d(attention_channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Compute attention-weighted statistics.

        Args:
            x: Input tensor (batch, time, channels) - NATIVE MLX FORMAT

        Returns:
            Pooled tensor (batch, 1, channels * 2) - mean and std concatenated
        """
        # Compute global statistics for context (axis=1 for time in B,T,C)
        mean = mx.mean(x, axis=1, keepdims=True)
        std = mx.sqrt(mx.var(x, axis=1, keepdims=True) + 1e-5)

        # Broadcast to time dimension and concatenate
        mean_broadcast = mx.broadcast_to(mean, x.shape)
        std_broadcast = mx.broadcast_to(std, x.shape)

        # Concatenate features with context (axis=2 for channels in B,T,C)
        context = mx.concatenate([x, mean_broadcast, std_broadcast], axis=2)

        # Compute attention weights
        attn = self.tdnn(context)
        attn = self.conv(attn)
        attn = mx.softmax(attn, axis=1)  # Softmax over time (axis=1)

        # Weighted mean
        weighted_mean = mx.sum(x * attn, axis=1, keepdims=True)

        # Weighted std
        weighted_var = mx.sum((x - weighted_mean) ** 2 * attn, axis=1, keepdims=True)
        weighted_std = mx.sqrt(weighted_var + 1e-5)

        # Concatenate mean and std (axis=2 for channels)
        return mx.concatenate([weighted_mean, weighted_std], axis=2)
