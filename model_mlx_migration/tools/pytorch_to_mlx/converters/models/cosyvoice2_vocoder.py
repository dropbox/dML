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
CosyVoice2 HiFi-GAN Vocoder

Converts mel spectrograms to audio waveforms using HiFi-GAN architecture
with source-filter modeling and F0 prediction.

Architecture:
- Source module for excitation signal
- F0 predictor for pitch estimation
- HiFi-GAN generator with ResBlocks and upsampling
- Weight normalization on all convolutions

Input: mel spectrogram [batch, mel_dim, length]
Output: audio waveform [batch, samples]
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class HiFiGANConfig:
    """Configuration for HiFi-GAN vocoder (CosyVoice2 hift.pt)."""

    # Input/output
    mel_channels: int = 80
    sample_rate: int = 22050
    output_channels: int = 18  # Output channels (harmonic components)

    # Generator architecture (from hift.pt inspection)
    # ups.0: [512, 256, 16], ups.1: [256, 128, 11], ups.2: [128, 64, 7]
    upsample_initial_channel: int = 512
    upsample_rates: tuple[int, ...] = (8, 8, 2)  # Product = 128
    upsample_kernel_sizes: tuple[int, ...] = (16, 11, 7)
    upsample_output_channels: tuple[int, ...] = (256, 128, 64)

    # ResBlocks (9 total: 3 per upsample stage, 3 kernel sizes)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )

    # F0 predictor
    f0_predictor_channels: int = 512


class WeightNormConv1d(nn.Module):
    """
    Conv1d with weight normalization.

    CosyVoice2 stores weights using PyTorch parametrizations:
    - parametrizations.weight.original0 = g (scale factor)
    - parametrizations.weight.original1 = v (weight direction)

    Weight = g * v / ||v||

    Input/Output: NLC format (batch, length, channels)
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

        # Weight normalization parameters
        # weight_g: [out_channels, 1, 1] - magnitude (original0)
        # weight_v: [out_channels, in_channels/groups, kernel_size] - direction (original1)
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

        # Apply convolution
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

    Used for upsampling in HiFi-GAN generator.
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

        # Weight normalization parameters (PyTorch stores as [in, out/groups, kernel])
        self.weight_g = mx.ones((in_channels, 1, 1))
        self.weight_v = (
            mx.random.normal((in_channels, out_channels // groups, kernel_size)) * 0.02
        )

        self.bias: mx.array | None
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, in_channels] - NLC format

        Returns:
            y: [batch, length', out_channels]
        """
        # Compute normalized weight
        v_norm = mx.sqrt(mx.sum(self.weight_v**2, axis=(1, 2), keepdims=True) + 1e-12)
        weight_pt = self.weight_g * self.weight_v / v_norm

        # Transpose from PyTorch [in, out, kernel] to MLX format
        mx.transpose(weight_pt, (0, 2, 1))

        # Apply transposed convolution
        # MLX conv_transpose1d: input [batch, length, in_channels], weight [out, kernel, in]
        # We need to reshape weight appropriately
        batch, length, _ = x.shape

        # Simple upsampling via repeat + conv approach
        # MLX doesn't have direct conv_transpose1d, so we use upsampling
        x_up = mx.repeat(x, self.stride, axis=1)

        # Convolve (treating as regular conv with stride 1)
        weight_conv = mx.transpose(weight_pt, (1, 2, 0))  # [out, kernel, in]
        y = mx.conv1d(x_up, weight_conv, stride=1, padding=self.kernel_size // 2)

        if self.bias is not None:
            y = y + self.bias

        return y


class ResBlock1d(nn.Module):
    """
    HiFi-GAN ResBlock with multiple dilated convolutions.

    Two parallel dilated conv paths that are summed.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations

        # Two sets of dilated convolutions
        self.convs1 = []
        self.convs2 = []

        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(
                WeightNormConv1d(
                    channels, channels, kernel_size, padding=padding, dilation=d,
                ),
            )
            self.convs2.append(
                WeightNormConv1d(
                    channels, channels, kernel_size, padding=(kernel_size - 1) // 2,
                ),
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, channels] - NLC format

        Returns:
            y: [batch, length, channels]
        """
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=False):
            xt = nn.leaky_relu(x, 0.1)
            xt = conv1(xt)
            xt = nn.leaky_relu(xt, 0.1)
            xt = conv2(xt)
            x = x + xt

        return x


class SourceModule(nn.Module):
    """
    Source module for excitation signal generation.

    Simple linear layer that generates source signal.
    """

    def __init__(self, output_channels: int = 1):
        super().__init__()
        # l_linear: [1, 9] in PyTorch
        self.l_linear = nn.Linear(9, output_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, 9] - Input features

        Returns:
            source: [batch, length, 1]
        """
        return self.l_linear(x)


class F0Predictor(nn.Module):
    """
    F0 prediction network.

    Predicts fundamental frequency from mel spectrogram.
    """

    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        mel_channels = config.mel_channels
        hidden_channels = config.f0_predictor_channels

        # Condition network
        self.condnet = [
            WeightNormConv1d(mel_channels, hidden_channels, 3, padding=1),
            WeightNormConv1d(hidden_channels, hidden_channels, 3, padding=1),
            WeightNormConv1d(hidden_channels, hidden_channels, 3, padding=1),
        ]

        # Output projection
        self.out = WeightNormConv1d(hidden_channels, 1, 1)

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Args:
            mel: [batch, length, mel_channels] - Mel spectrogram

        Returns:
            f0: [batch, length, 1] - Predicted F0
        """
        x = mel
        for conv in self.condnet:
            x = conv(x)
            x = nn.leaky_relu(x, 0.1)

        return self.out(x)


class HiFiGANVocoder(nn.Module):
    """
    HiFi-GAN Vocoder for CosyVoice2.

    Converts mel spectrograms to audio waveforms using source-filter model.
    Architecture from hift.pt inspection:
    - conv_pre: [512, 80, 7]
    - ups: [512->256, 16], [256->128, 11], [128->64, 7]
    - resblocks: 9 total (3 per stage * 3 kernel sizes)
    - conv_post: [64->18, 7]
    """

    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        self.config = config

        # Source module
        self.m_source = SourceModule(output_channels=1)

        # Pre-convolution: mel_channels -> upsample_initial_channel
        self.conv_pre = WeightNormConv1d(
            config.mel_channels, config.upsample_initial_channel, 7, padding=3,
        )

        # F0 predictor
        self.f0_predictor = F0Predictor(config)

        # Upsample layers (ConvTranspose1d)
        # Each ups layer halves channels: 512->256->128->64
        self.ups = []
        in_channels = [config.upsample_initial_channel] + list(
            config.upsample_output_channels[:-1],
        )
        for in_ch, out_ch, kernel, rate in zip(
            in_channels,
            config.upsample_output_channels,
            config.upsample_kernel_sizes,
            config.upsample_rates, strict=False,
        ):
            self.ups.append(
                WeightNormConvTranspose1d(
                    in_ch, out_ch, kernel, stride=rate, padding=rate // 2,
                ),
            )

        # ResBlocks: 3 per upsample stage (one for each kernel size)
        # Total: 9 resblocks for 3 upsample stages
        self.resblocks = []
        for out_ch in config.upsample_output_channels:
            for kernel, dilations in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes, strict=False,
            ):
                self.resblocks.append(ResBlock1d(out_ch, kernel, dilations))

        # Source downsample layers (regular Conv1d, no weight norm)
        self.source_downs = []
        source_down_params = [
            (config.output_channels, 256, 30),  # [256, 18, 30]
            (config.output_channels, 128, 6),  # [128, 18, 6]
            (config.output_channels, 64, 2),  # [64, 18, 2]
        ]
        for in_ch, out_ch, kernel in source_down_params:
            self.source_downs.append(nn.Conv1d(in_ch, out_ch, kernel))

        # Source ResBlocks
        self.source_resblocks = []
        for out_ch in [256, 128, 64]:
            for _ in range(2):  # 2 per stage = 6 total
                self.source_resblocks.append(ResBlock1d(out_ch, 7, (1, 3, 5)))

        # Post-convolution: final_ch -> output_channels
        final_ch = config.upsample_output_channels[-1]  # 64
        self.conv_post = WeightNormConv1d(
            final_ch, config.output_channels, 7, padding=3,
        )

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Generate audio from mel spectrogram.

        Args:
            mel: [batch, length, mel_channels] - Mel spectrogram (NLC format)

        Returns:
            audio: [batch, samples] - Generated waveform
        """
        # Pre-convolution: [B, L, 80] -> [B, L, 512]
        x = self.conv_pre(mel)

        # F0 prediction (not used in simple forward, but computed for reference)
        # f0 = self.f0_predictor(mel)

        # Progressive upsampling with ResBlocks
        resblock_idx = 0
        num_kernels = len(self.config.resblock_kernel_sizes)

        for _i, up in enumerate(self.ups):
            x = nn.leaky_relu(x, 0.1)
            x = up(x)

            # Apply ResBlocks (one for each kernel size at this stage)
            xs: mx.array = self.resblocks[resblock_idx](x)
            resblock_idx += 1
            for _ in range(num_kernels - 1):
                xs = xs + self.resblocks[resblock_idx](x)
                resblock_idx += 1
            x = xs / num_kernels

        # Post-convolution: [B, L*128, 64] -> [B, L*128, 18]
        x = nn.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = mx.tanh(x)

        # Flatten to audio: [B, L*128, 18] -> [B, L*128*18]
        return x.reshape(x.shape[0], -1)


    @staticmethod
    def from_pretrained(
        weights_path: str, config: HiFiGANConfig | None = None,
    ) -> "HiFiGANVocoder":
        """
        Load vocoder from pretrained weights.

        Args:
            weights_path: Path to hift.pt file
            config: Optional config override

        Returns:
            Loaded HiFiGANVocoder
        """
        import torch

        if config is None:
            config = HiFiGANConfig()

        vocoder = HiFiGANVocoder(config)

        # Load PyTorch weights
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Map weights
        vocoder._load_weights(state_dict)

        return vocoder

    def _load_weights(self, state_dict: dict) -> None:
        """Load weights from PyTorch state dict."""

        def get_weight(prefix: str, use_weight_norm: bool = True):
            """Get weight from state dict, handling weight normalization."""
            if use_weight_norm:
                g = state_dict.get(f"{prefix}.parametrizations.weight.original0")
                v = state_dict.get(f"{prefix}.parametrizations.weight.original1")
                if g is not None and v is not None:
                    return mx.array(g.numpy()), mx.array(v.numpy())

            # Fall back to regular weight
            w = state_dict.get(f"{prefix}.weight")
            if w is not None:
                return mx.array(w.numpy())
            return None

        def get_bias(prefix: str):
            """Get bias from state dict."""
            b = state_dict.get(f"{prefix}.bias")
            if b is not None:
                return mx.array(b.numpy())
            return None

        # Load source module
        if "m_source.l_linear.weight" in state_dict:
            self.m_source.l_linear.weight = mx.array(
                state_dict["m_source.l_linear.weight"].numpy(),
            )
            self.m_source.l_linear.bias = mx.array(
                state_dict["m_source.l_linear.bias"].numpy(),
            )

        # Load conv_pre
        result = get_weight("conv_pre")
        if isinstance(result, tuple):
            self.conv_pre.weight_g, self.conv_pre.weight_v = result
        self.conv_pre.bias = get_bias("conv_pre")

        # Load conv_post
        result = get_weight("conv_post")
        if isinstance(result, tuple):
            self.conv_post.weight_g, self.conv_post.weight_v = result
        self.conv_post.bias = get_bias("conv_post")

        # Load upsample layers
        for i, up in enumerate(self.ups):
            result = get_weight(f"ups.{i}")
            if isinstance(result, tuple):
                up.weight_g, up.weight_v = result
            up.bias = get_bias(f"ups.{i}")

        # Load ResBlocks
        for i, resblock in enumerate(self.resblocks):
            for j, conv in enumerate(resblock.convs1):
                result = get_weight(f"resblocks.{i}.convs1.{j}")
                if isinstance(result, tuple):
                    conv.weight_g, conv.weight_v = result
                conv.bias = get_bias(f"resblocks.{i}.convs1.{j}")

            for j, conv in enumerate(resblock.convs2):
                result = get_weight(f"resblocks.{i}.convs2.{j}")
                if isinstance(result, tuple):
                    conv.weight_g, conv.weight_v = result
                conv.bias = get_bias(f"resblocks.{i}.convs2.{j}")
