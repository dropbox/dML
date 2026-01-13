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
Kokoro TTS Building Blocks for MLX

Core modules needed for Kokoro-82M StyleTTS2-based model:
- Weight-normalized Conv1d
- Adaptive Instance Normalization (AdaIN)
- AdaLayerNorm
- AdainResBlk1d
- BiLSTM
- Custom LayerNorm with beta/gamma

NOTE: MLX uses NLC format (batch, length, channels), not NCL like PyTorch.
All modules in this file use NLC format consistently.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


def sample_variance(x: mx.array, axis: int, keepdims: bool = False) -> mx.array:
    """
    Compute sample variance with ddof=1 to match PyTorch's default behavior.

    MLX's mx.var() uses ddof=0 (population variance) by default, but PyTorch's
    torch.var() uses ddof=1 (sample variance) by default (unbiased=True).

    This 25% difference causes signal explosion in AdaIN normalization.

    Args:
        x: Input array
        axis: Axis to compute variance over
        keepdims: Whether to keep the reduced dimension

    Returns:
        Sample variance with ddof=1
    """
    mean = mx.mean(x, axis=axis, keepdims=True)
    diff = x - mean
    n = x.shape[axis]
    return mx.sum(diff * diff, axis=axis, keepdims=keepdims) / (n - 1)  # ddof=1


@dataclass
class KokoroConfig:
    """Configuration for Kokoro-82M model."""

    # Model dimensions
    dim_in: int = 64
    hidden_dim: int = 512
    style_dim: int = 128
    max_conv_dim: int = 512
    n_token: int = 178
    n_mels: int = 80
    n_layer: int = 3
    max_dur: int = 50
    dropout: float = 0.2
    text_encoder_kernel_size: int = 5
    multispeaker: bool = True

    # PLBERT config
    plbert_hidden_size: int = 768
    plbert_num_attention_heads: int = 12
    plbert_intermediate_size: int = 2048
    plbert_max_position_embeddings: int = 512
    plbert_num_hidden_layers: int = 12
    plbert_dropout: float = 0.1

    # ALBERT uses smaller embedding dim
    albert_embedding_dim: int = 128

    # ISTFTNet config
    istft_upsample_rates: tuple[int, ...] = (10, 6)
    istft_upsample_kernel_sizes: tuple[int, ...] = (20, 12)
    istft_gen_istft_n_fft: int = 20
    istft_gen_istft_hop_size: int = 5
    istft_resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    istft_resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    istft_upsample_initial_channel: int = 512

    # Audio output config
    sample_rate: int = 24000  # 24kHz audio output (matches CosyVoice2 API)


class WeightNormConv1d(nn.Module):
    """
    Conv1d with weight normalization.

    PyTorch stores weight_g (magnitude) and weight_v (direction) separately.
    Weight = weight_g * weight_v / ||weight_v||

    Input/Output: NLC format (batch, length, channels)

    NOTE: MLX uses weight shape [out_channels, kernel_size, in_channels/groups]
    PyTorch uses weight shape [out_channels, in_channels/groups, kernel_size]
    We store in PyTorch format for weight loading, then transpose for MLX.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight normalization parameters (stored in PyTorch format for easy weight loading)
        # weight_g: [out_channels, 1, 1] - magnitude
        # weight_v: [out_channels, in_channels/groups, kernel_size] - direction (PyTorch format)
        self.weight_g = mx.ones((out_channels, 1, 1))
        self.weight_v = (
            mx.random.normal((out_channels, in_channels // groups, kernel_size)) * 0.02
        )

        self.bias: mx.array | None
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, in_channels] - MLX NLC format

        Returns:
            y: [batch, length', out_channels]
        """
        # Compute normalized weight: g * v / ||v||
        v_norm = mx.sqrt(mx.sum(self.weight_v**2, axis=(1, 2), keepdims=True) + 1e-12)
        weight_pt = self.weight_g * self.weight_v / v_norm

        # Transpose from PyTorch [out, in, kernel] to MLX [out, kernel, in]
        weight = mx.transpose(weight_pt, (0, 2, 1))

        # Apply convolution (MLX uses NLC format with weight [out, kernel, in])
        y = mx.conv1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class WeightNormConvTranspose1d(nn.Module):
    """
    ConvTranspose1d with weight normalization.

    PyTorch stores weight_g (magnitude) and weight_v (direction) separately.
    Weight = weight_g * weight_v / ||weight_v||

    Input/Output: NLC format (batch, length, channels)

    PyTorch ConvTranspose1d weight format: [in_channels, out_channels, kernel_size]
    Note: This is different from Conv1d which is [out_channels, in_channels, kernel_size]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weight normalization parameters (stored in PyTorch format for weight loading)
        # PyTorch ConvTranspose1d: weight_v is [in_channels, out_channels/groups, kernel_size]
        # For groups > 1 (e.g., depthwise), out_channels/groups = 1
        # weight_g: [in_channels, 1, 1] - magnitude (matches first dim of weight_v)
        out_per_group = out_channels // groups
        self.weight_g = mx.ones((in_channels, 1, 1))
        self.weight_v = (
            mx.random.normal((in_channels, out_per_group, kernel_size)) * 0.02
        )

        self.bias: mx.array | None
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, in_channels] - MLX NLC format

        Returns:
            y: [batch, length', out_channels]
        """
        # Compute normalized weight: g * v / ||v||
        v_norm = mx.sqrt(mx.sum(self.weight_v**2, axis=(1, 2), keepdims=True) + 1e-12)
        weight_pt = self.weight_g * self.weight_v / v_norm

        # Transpose for conv_transpose1d
        # PyTorch ConvTranspose1d weight: [in_channels, out_channels/groups, kernel_size]
        # MLX conv_transpose1d weight: [out_channels, kernel_size, in_channels/groups]
        # For groups=in_channels (depthwise): [in, 1, k] -> [out, k, 1]
        # Since in=out=groups: [in, 1, k] -> [in, k, 1] via transpose (0, 2, 1)
        if self.groups > 1:
            # Depthwise: keep first dim, swap last two
            weight = mx.transpose(weight_pt, (0, 2, 1))
        else:
            # Regular: standard transpose
            weight = mx.transpose(weight_pt, (1, 2, 0))

        # Apply transposed convolution with native output_padding support
        y = mx.conv_transpose1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class PlainConv1d(nn.Module):
    """
    Plain Conv1d WITHOUT weight normalization.

    Used for modules like noise_convs in StyleTTS2 ISTFTNet that use
    regular Conv1d, not weight-normalized Conv1d.

    Input/Output: NLC format (batch, length, channels)

    NOTE: MLX uses weight shape [out_channels, kernel_size, in_channels/groups]
    PyTorch uses weight shape [out_channels, in_channels/groups, kernel_size]
    We store in PyTorch format for weight loading, then transpose for MLX.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight stored in PyTorch format for easy weight loading
        # [out_channels, in_channels/groups, kernel_size]
        self.weight = (
            mx.random.normal((out_channels, in_channels // groups, kernel_size)) * 0.02
        )

        self.bias: mx.array | None
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, in_channels] - MLX NLC format

        Returns:
            y: [batch, length', out_channels]
        """
        # Transpose from PyTorch [out, in, kernel] to MLX [out, kernel, in]
        weight = mx.transpose(self.weight, (0, 2, 1))

        # Apply convolution (MLX uses NLC format with weight [out, kernel, in])
        y = mx.conv1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class CustomLayerNorm(nn.Module):
    """
    LayerNorm with learnable beta/gamma (Kokoro style).

    Normalizes over the channel dimension for NLC format.
    Input: [batch, length, channels]

    B2 Optimization: Uses mx.fast.layer_norm for fused kernel execution.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, channels] - NLC format
        """
        # B2: Use fused layer norm kernel for better performance
        return mx.fast.layer_norm(x, weight=self.gamma, bias=self.beta, eps=self.eps)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.

    Normalizes input features and scales/shifts using style-derived parameters.
    Instance normalization is over the length dimension.

    PyTorch's AdaIN1d uses nn.InstanceNorm1d with affine=True, which has learnable
    weight and bias parameters. The formula is:
        norm_out = weight * ((x - mean) / sqrt(var + eps)) + bias
        output = (1 + gamma) * norm_out + beta

    Input: NLC format [batch, length, channels]

    N2 Optimization: Supports cached style parameters to avoid redundant fc() computation.
    """

    def __init__(self, channels: int, style_dim: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        # Linear layer to project style to scale and shift
        self.fc = nn.Linear(style_dim, channels * 2)
        # InstanceNorm1d affine parameters (weight=gamma, bias=beta from the norm)
        # Initialized to weight=1, bias=0 by default in PyTorch
        self.norm_weight = mx.ones((channels,))
        self.norm_bias = mx.zeros((channels,))

    def __call__(
        self,
        x: mx.array,
        s: mx.array,
        cached_style: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """
        Args:
            x: [batch, length, channels] - NLC format
            s: [batch, style_dim] - style vector
            cached_style: Optional (gamma, beta) tuple from style cache.
                         If provided, skips fc(s) computation.
        """
        # N2: Use cached style params if available
        if cached_style is not None:
            gamma, beta = cached_style
            gamma = gamma[:, None, :]  # [batch, 1, channels]
            beta = beta[:, None, :]
        else:
            # Get adaptive parameters from style
            h = self.fc(s)  # [batch, channels * 2]
            gamma, beta = mx.split(h, 2, axis=-1)  # Each [batch, channels]
            gamma = gamma[:, None, :]  # [batch, 1, channels]
            beta = beta[:, None, :]

        # Instance normalization over length dimension with learnable affine
        # NOTE: PyTorch's InstanceNorm1d uses population variance (ddof=0), NOT sample variance!
        # mx.var() uses ddof=0 by default, matching InstanceNorm1d
        mean = mx.mean(x, axis=1, keepdims=True)  # [batch, 1, channels]
        var = mx.var(x, axis=1, keepdims=True)  # ddof=0 to match InstanceNorm1d
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Apply InstanceNorm affine parameters (weight, bias)
        norm_out = self.norm_weight * x_norm + self.norm_bias

        # Apply adaptive scale and shift
        return (1 + gamma) * norm_out + beta


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization for sequence models.

    Used in duration encoder with style conditioning.
    Input: NLC format [batch, length, hidden_dim]

    N2 Optimization: Supports cached style parameters to avoid redundant fc() computation.
    """

    def __init__(self, hidden_dim: int, style_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.fc = nn.Linear(style_dim, hidden_dim * 2)

    def __call__(
        self,
        x: mx.array,
        s: mx.array,
        cached_style: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """
        Args:
            x: [batch, length, hidden_dim] - NLC format
            s: [batch, style_dim] - style vector
            cached_style: Optional (gamma, beta) tuple from style cache.
                         If provided, skips fc(s) computation.
        """
        # N2: Use cached style params if available
        if cached_style is not None:
            gamma, beta = cached_style
            gamma = gamma[:, None, :]  # [batch, 1, hidden_dim]
            beta = beta[:, None, :]
        else:
            # Get adaptive parameters
            h = self.fc(s)  # [batch, hidden_dim * 2]
            gamma, beta = mx.split(h, 2, axis=-1)  # Each [batch, hidden_dim]
            gamma = gamma[:, None, :]  # [batch, 1, hidden_dim]
            beta = beta[:, None, :]

        # Layer normalization over channel dimension
        # PyTorch's LayerNorm uses population variance (ddof=0)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)  # ddof=0 to match LayerNorm
        x = (x - mean) / mx.sqrt(var + self.eps)

        return (1 + gamma) * x + beta


class AdainResBlk1d(nn.Module):
    """
    Residual block with Adaptive Instance Normalization.

    Core building block used in decoder and predictor.
    Input/Output: NLC format [batch, length, channels]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        kernel_size: int = 3,
        upsample: bool = False,
        downsample: bool = False,
        learned_upsample: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.downsample = downsample
        self.learned_upsample = learned_upsample
        # Type annotations for optional attributes
        self.conv1x1: WeightNormConv1d | None = None
        self.pool: WeightNormConv1d | WeightNormConvTranspose1d | None = None

        padding = kernel_size // 2

        # First conv + AdaIN
        # Note: In Kokoro, norm1 operates on in_channels (before channel change)
        self.conv1 = WeightNormConv1d(
            in_channels, out_channels, kernel_size, padding=padding,
        )
        self.norm1 = AdaIN(in_channels, style_dim)

        # Second conv + AdaIN
        self.conv2 = WeightNormConv1d(
            out_channels, out_channels, kernel_size, padding=padding,
        )
        self.norm2 = AdaIN(out_channels, style_dim)

        # Skip connection
        if in_channels != out_channels:
            self.conv1x1 = WeightNormConv1d(in_channels, out_channels, 1)
        else:
            self.conv1x1 = None

        # Pooling for downsample
        if downsample:
            self.pool = WeightNormConv1d(
                out_channels, out_channels, kernel_size, stride=2, padding=padding,
            )

        # Learned upsampling via ConvTranspose1d (for decoder upsample blocks only)
        # This is initialized if learned_upsample=True, weights loaded from checkpoint
        if learned_upsample:
            # Depthwise ConvTranspose1d: groups=in_channels
            self.pool = WeightNormConvTranspose1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                groups=in_channels,
            )

    def __call__(
        self,
        x: mx.array,
        s: mx.array,
        cached_norm1: tuple[mx.array, mx.array] | None = None,
        cached_norm2: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """
        Args:
            x: [batch, length, in_channels] - NLC format
            s: [batch, style_dim]
            cached_norm1: Optional (gamma, beta) for norm1 from style cache
            cached_norm2: Optional (gamma, beta) for norm2 from style cache

        Per StyleTTS2 upstream AdainResBlk1d:
        - Residual path: norm1 -> actv -> pool -> conv1 -> norm2 -> actv -> conv2
        - Shortcut path: upsample (nearest) -> conv1x1
        - Output: (residual + shortcut) * rsqrt(2)
        """
        # === Shortcut path ===
        # For upsample: shortcut uses nearest interpolation (not learned ConvTranspose)
        x_short = x
        if self.upsample:
            # Simple nearest-neighbor upsampling for shortcut
            x_short = mx.repeat(x, 2, axis=1)

        if self.conv1x1 is not None:
            skip = self.conv1x1(x_short)
        else:
            skip = x_short

        # === Residual path ===
        # norm1 -> actv -> pool -> conv1 -> norm2 -> actv -> conv2
        h = self.norm1(x, s, cached_style=cached_norm1)
        h = nn.leaky_relu(h, 0.2)

        # Pool in residual path (for upsample: learned ConvTranspose)
        if self.upsample and self.learned_upsample:
            assert self.pool is not None
            h = self.pool(h)
        elif self.upsample:
            # Simple repeat for predictor upsample blocks
            h = mx.repeat(h, 2, axis=1)

        h = self.conv1(h)
        h = self.norm2(h, s, cached_style=cached_norm2)
        h = nn.leaky_relu(h, 0.2)
        h = self.conv2(h)

        # Residual with rsqrt(2) scaling per StyleTTS2 upstream
        out = (h + skip) * (2**-0.5)

        # Downsample if needed
        if self.downsample:
            assert self.pool is not None
            out = self.pool(out)

        return out


class ProsodyConditionedAdaIN(nn.Module):
    """
    Adaptive Instance Normalization with Prosody Conditioning (Phase C).

    Extends AdaIN to derive gamma/beta from BOTH style AND prosody embeddings.
    This allows prosody to directly influence the normalization scale/shift,
    bypassing the normalization step that washes out prosody effects.

    The key insight: Standard AdaIN normalizes out prosody variations because
    gamma/beta come only from style. By adding prosody conditioning to gamma/beta,
    prosody directly controls the output scale/shift.

    Input: NLC format [batch, length, channels]
    """

    def __init__(
        self,
        channels: int,
        style_dim: int,
        prosody_dim: int,
        eps: float = 1e-5,
        prosody_scale: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.prosody_scale = prosody_scale

        # Style conditioning (same as original AdaIN)
        self.fc_style = nn.Linear(style_dim, channels * 2)

        # Prosody conditioning (new - Phase C)
        # Projects prosody embedding to gamma/beta adjustments
        self.fc_prosody = nn.Linear(prosody_dim, channels * 2)

        # InstanceNorm1d affine parameters
        self.norm_weight = mx.ones((channels,))
        self.norm_bias = mx.zeros((channels,))

    def __call__(
        self,
        x: mx.array,
        s: mx.array,
        prosody_emb: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            x: [batch, length, channels] - NLC format
            s: [batch, style_dim] - style vector
            prosody_emb: [batch, prosody_dim] or [batch, length, prosody_dim] - prosody embedding
                        If None, behaves like standard AdaIN (style only)
        """
        # Get adaptive parameters from style
        h_style = self.fc_style(s)  # [batch, channels * 2]
        gamma_style, beta_style = mx.split(h_style, 2, axis=-1)

        # Add prosody conditioning if provided
        if prosody_emb is not None:
            # Handle both [batch, dim] and [batch, length, dim] formats
            if prosody_emb.ndim == 2:
                # [batch, prosody_dim] - global prosody
                h_prosody = self.fc_prosody(prosody_emb)  # [batch, channels * 2]
            else:
                # [batch, length, prosody_dim] - per-position prosody
                # Average across length for this normalization
                prosody_avg = mx.mean(prosody_emb, axis=1)  # [batch, prosody_dim]
                h_prosody = self.fc_prosody(prosody_avg)  # [batch, channels * 2]

            gamma_prosody, beta_prosody = mx.split(h_prosody, 2, axis=-1)

            # Combine style and prosody (prosody adds to style)
            gamma = gamma_style + self.prosody_scale * gamma_prosody
            beta = beta_style + self.prosody_scale * beta_prosody
        else:
            gamma = gamma_style
            beta = beta_style

        # Expand for broadcasting
        gamma = gamma[:, None, :]  # [batch, 1, channels]
        beta = beta[:, None, :]

        # Instance normalization (same as original)
        mean = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Apply InstanceNorm affine parameters
        norm_out = self.norm_weight * x_norm + self.norm_bias

        # Apply adaptive scale and shift (now includes prosody!)
        return (1 + gamma) * norm_out + beta


class ProsodyConditionedAdainResBlk1d(nn.Module):
    """
    Residual block with Prosody-Conditioned Adaptive Instance Normalization (Phase C).

    Extends AdainResBlk1d to pass prosody embeddings to AdaIN layers.
    Used in F0 predictor to enable prosody control over F0 output.

    Input/Output: NLC format [batch, length, channels]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        prosody_dim: int = 768,
        kernel_size: int = 3,
        upsample: bool = False,
        downsample: bool = False,
        learned_upsample: bool = False,
        prosody_scale: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.downsample = downsample
        self.learned_upsample = learned_upsample
        self.conv1x1: WeightNormConv1d | None = None
        self.pool: WeightNormConv1d | WeightNormConvTranspose1d | None = None

        padding = kernel_size // 2

        # First conv + Prosody-Conditioned AdaIN
        self.conv1 = WeightNormConv1d(
            in_channels, out_channels, kernel_size, padding=padding,
        )
        self.norm1 = ProsodyConditionedAdaIN(
            in_channels, style_dim, prosody_dim, prosody_scale=prosody_scale,
        )

        # Second conv + Prosody-Conditioned AdaIN
        self.conv2 = WeightNormConv1d(
            out_channels, out_channels, kernel_size, padding=padding,
        )
        self.norm2 = ProsodyConditionedAdaIN(
            out_channels, style_dim, prosody_dim, prosody_scale=prosody_scale,
        )

        # Skip connection
        if in_channels != out_channels:
            self.conv1x1 = WeightNormConv1d(in_channels, out_channels, 1)
        else:
            self.conv1x1 = None

        # Pooling for downsample
        if downsample:
            self.pool = WeightNormConv1d(
                out_channels, out_channels, kernel_size, stride=2, padding=padding,
            )

        # Learned upsampling via ConvTranspose1d
        if learned_upsample:
            self.pool = WeightNormConvTranspose1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                groups=in_channels,
            )

    def __call__(
        self,
        x: mx.array,
        s: mx.array,
        prosody_emb: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            x: [batch, length, in_channels] - NLC format
            s: [batch, style_dim] - style vector
            prosody_emb: [batch, prosody_dim] or [batch, length, prosody_dim] - prosody embedding
                        If None, behaves like standard AdainResBlk1d

        Returns:
            out: [batch, length_out, out_channels]
        """
        # === Shortcut path ===
        x_short = x
        if self.upsample:
            x_short = mx.repeat(x, 2, axis=1)

        if self.conv1x1 is not None:
            skip = self.conv1x1(x_short)
        else:
            skip = x_short

        # === Residual path with prosody conditioning ===
        h = self.norm1(x, s, prosody_emb)  # Prosody-conditioned!
        h = nn.leaky_relu(h, 0.2)

        if self.upsample and self.learned_upsample:
            assert self.pool is not None
            h = self.pool(h)
        elif self.upsample:
            h = mx.repeat(h, 2, axis=1)

        h = self.conv1(h)
        h = self.norm2(h, s, prosody_emb)  # Prosody-conditioned!
        h = nn.leaky_relu(h, 0.2)
        h = self.conv2(h)

        # Residual with rsqrt(2) scaling
        out = (h + skip) * (2**-0.5)

        # Downsample if needed
        if self.downsample:
            assert self.pool is not None
            out = self.pool(out)

        return out


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM.

    MLX doesn't have built-in bidirectional, so we run two LSTMs.
    Input/Output: NLC format [batch, length, features]
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_forward = nn.LSTM(input_size, hidden_size, bias=bias)
        self.lstm_backward = nn.LSTM(input_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, input_size]

        Returns:
            output: [batch, length, hidden_size * 2]
        """
        # Forward pass
        out_forward, _ = self.lstm_forward(x)

        # Backward pass - reverse sequence, run LSTM, reverse output
        x_reversed = x[:, ::-1, :]  # Reverse along length dimension
        out_backward, _ = self.lstm_backward(x_reversed)
        out_backward = out_backward[:, ::-1, :]  # Reverse output back

        # Concatenate
        return mx.concatenate([out_forward, out_backward], axis=-1)


class ResBlock1d(nn.Module):
    """
    1D Residual Block used in ISTFTNet.

    Multiple dilated convolutions with LeakyReLU.
    Input/Output: NLC format [batch, length, channels]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = []
        self.convs2 = []

        for d in dilations:
            # Use "same" padding to preserve length
            # For dilated conv: padding = dilation * (kernel_size - 1) // 2
            padding1 = d * (kernel_size - 1) // 2
            padding2 = (kernel_size - 1) // 2  # dilation=1 for conv2
            self.convs1.append(
                WeightNormConv1d(
                    channels, channels, kernel_size, padding=padding1, dilation=d,
                ),
            )
            self.convs2.append(
                WeightNormConv1d(
                    channels, channels, kernel_size, padding=padding2, dilation=1,
                ),
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, channels] - NLC format
        """
        for c1, c2 in zip(self.convs1, self.convs2, strict=False):
            h = nn.leaky_relu(x, 0.1)
            h = c1(h)
            h = nn.leaky_relu(h, 0.1)
            h = c2(h)
            x = x + h
        return x


class AdaINStyleLinear(nn.Module):
    """
    Simple wrapper for AdaIN style projection with InstanceNorm affine parameters.

    Matches checkpoint structure:
    - adain1.0.fc.weight, adain1.0.fc.bias
    - adain1.0.norm.weight, adain1.0.norm.bias (InstanceNorm1d affine params)
    """

    def __init__(self, style_dim: int, output_dim: int):
        super().__init__()
        # output_dim is channels * 2 (for gamma and beta)
        self.channels = output_dim // 2
        self.fc = nn.Linear(style_dim, output_dim)
        # InstanceNorm1d affine parameters (weight, bias)
        # Default: weight=1, bias=0
        self.norm_weight = mx.ones((self.channels,))
        self.norm_bias = mx.zeros((self.channels,))

    def __call__(self, s: mx.array) -> mx.array:
        """Project style to gamma/beta parameters."""
        return self.fc(s)


class AdaINResBlock1dStyled(nn.Module):
    """
    AdaIN-conditioned ResBlock used in ISTFTNet Generator's noise_res blocks.

    This is different from AdainResBlk1d - it has multiple conv layers with
    multiple AdaIN layers and learnable alpha mixing parameters.

    Architecture:
    - convs1: List of 3 weight-normalized Conv1d layers
    - convs2: List of 3 weight-normalized Conv1d layers
    - adain1: List of 3 AdaIN fc layers (style -> gamma, beta)
    - adain2: List of 3 AdaIN fc layers (style -> gamma, beta)
    - alpha1: List of 3 learnable mixing parameters
    - alpha2: List of 3 learnable mixing parameters

    Input/Output: NLC format [batch, length, channels]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        style_dim: int = 128,
        num_layers: int = 3,
        dilations: tuple = (1, 3, 5),
    ):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers

        # Convolution layers - use numbered attributes for proper weight loading
        # Per ISTFTNet/StyleTTS2: convs1 uses different dilations (1, 3, 5)
        # convs2 ALWAYS uses dilation=1 (this is critical for correct output magnitude)
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 1
            # Padding formula for same-size output: dilation * (kernel_size - 1) // 2
            padding = dilation * (kernel_size - 1) // 2
            setattr(
                self,
                f"convs1_{i}",
                WeightNormConv1d(
                    channels, channels, kernel_size, padding=padding, dilation=dilation,
                ),
            )
            # convs2 always uses dilation=1
            padding2 = (kernel_size - 1) // 2
            setattr(
                self,
                f"convs2_{i}",
                WeightNormConv1d(
                    channels, channels, kernel_size, padding=padding2, dilation=1,
                ),
            )

        # AdaIN layers - each produces gamma and beta for channels
        # fc output is 2*channels (gamma + beta)
        # Use nested structure with fc attribute to match checkpoint: adain1.0.fc.weight
        for i in range(num_layers):
            setattr(self, f"adain1_{i}", AdaINStyleLinear(style_dim, channels * 2))
            setattr(self, f"adain2_{i}", AdaINStyleLinear(style_dim, channels * 2))

        # Learnable alpha mixing parameters - direct array attributes
        # Use numbered attributes to match checkpoint: alpha1.0, alpha1.1, alpha1.2
        # Checkpoint shape is [1, channels, 1] (NCL format)
        for i in range(num_layers):
            setattr(self, f"alpha1_{i}", mx.ones((1, channels, 1)))
            setattr(self, f"alpha2_{i}", mx.ones((1, channels, 1)))

    def __call__(
        self,
        x: mx.array,
        s: mx.array,
        cached_styles: dict | None = None,
    ) -> mx.array:
        """
        Forward pass with style conditioning and Snake1D activation.

        Per StyleTTS2 upstream AdaINResBlock1:
        - Flow: AdaIN -> Snake1D -> Conv1 -> AdaIN -> Snake1D -> Conv2 -> Residual

        Snake1D activation: x + (1/alpha) * sin(alpha*x)^2

        Args:
            x: [batch, length, channels] - input features (NLC)
            s: [batch, style_dim] - style vector
            cached_styles: Optional dict mapping "adain1_0", "adain2_0", etc. to (gamma, beta) tuples.
                          N2 optimization - skips fc() computation when provided.

        Returns:
            [batch, length, channels] - output features
        """
        for i in range(self.num_layers):
            # Get numbered attributes
            adain1 = getattr(self, f"adain1_{i}")
            adain2 = getattr(self, f"adain2_{i}")
            alpha1 = getattr(self, f"alpha1_{i}")  # [channels]
            alpha2 = getattr(self, f"alpha2_{i}")  # [channels]
            conv1 = getattr(self, f"convs1_{i}")
            conv2 = getattr(self, f"convs2_{i}")

            # === First path: AdaIN1 -> Snake1D -> Conv1 ===
            # N2: Check for cached style
            cached1 = cached_styles.get(f"adain1_{i}") if cached_styles else None
            if cached1 is not None:
                gamma, beta = cached1
                gamma = gamma[:, None, :]  # [batch, 1, channels]
                beta = beta[:, None, :]
            else:
                style_out = adain1(s)  # [batch, channels*2]
                gamma = style_out[:, : self.channels][:, None, :]  # [batch, 1, channels]
                beta = style_out[:, self.channels :][:, None, :]

            # Instance norm (ddof=0 for InstanceNorm1d) with affine params
            mean = mx.mean(x, axis=1, keepdims=True)
            var = mx.var(x, axis=1, keepdims=True)  # ddof=0 to match InstanceNorm1d
            xt = (x - mean) / mx.sqrt(var + 1e-5)
            # Apply InstanceNorm1d affine parameters (norm_weight, norm_bias)
            xt = adain1.norm_weight * xt + adain1.norm_bias
            xt = (1 + gamma) * xt + beta  # AdaIN: (1 + gamma) per official

            # Snake1D activation: x + (1/alpha) * sin(alpha*x)^2
            # alpha is [1, channels, 1] (NCL format), need to transpose for NLC
            alpha1_nlc = alpha1.transpose(0, 2, 1)  # [1, 1, channels]
            xt = xt + (1 / alpha1_nlc) * (mx.sin(alpha1_nlc * xt) ** 2)

            xt = conv1(xt)

            # === Second path: AdaIN2 -> Snake1D -> Conv2 ===
            # N2: Check for cached style
            cached2 = cached_styles.get(f"adain2_{i}") if cached_styles else None
            if cached2 is not None:
                gamma, beta = cached2
                gamma = gamma[:, None, :]
                beta = beta[:, None, :]
            else:
                style_out = adain2(s)
                gamma = style_out[:, : self.channels][:, None, :]
                beta = style_out[:, self.channels :][:, None, :]

            mean = mx.mean(xt, axis=1, keepdims=True)
            var = mx.var(xt, axis=1, keepdims=True)  # ddof=0 to match InstanceNorm1d
            xt = (xt - mean) / mx.sqrt(var + 1e-5)
            # Apply InstanceNorm1d affine parameters (norm_weight, norm_bias)
            xt = adain2.norm_weight * xt + adain2.norm_bias
            xt = (1 + gamma) * xt + beta

            # Snake1D activation
            alpha2_nlc = alpha2.transpose(0, 2, 1)  # [1, 1, channels]
            xt = xt + (1 / alpha2_nlc) * (mx.sin(alpha2_nlc * xt) ** 2)

            xt = conv2(xt)

            # Residual add
            x = xt + x

        return x
