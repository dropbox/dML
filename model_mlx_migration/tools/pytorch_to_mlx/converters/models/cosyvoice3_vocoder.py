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
CosyVoice3 CausalHiFT Vocoder - MLX Implementation

Causal HiFi-GAN with neural source-filter for streaming TTS.

Architecture matches PyTorch CosyVoice3 hift.pt:
- f0_predictor: condnet (5 Conv1d) + classifier
- m_source: l_linear for harmonic weights
- source_downs: 3 Conv1d for source downsampling
- source_resblocks: 3 ResBlocks with Snake activations
- ups: 3 ConvTranspose1d
- resblocks: 9 ResBlocks with Snake activations
- conv_pre, conv_post: weight-normalized Conv1d
- iSTFT synthesis
"""

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


@dataclass
class CausalHiFTConfig:
    """Configuration for CausalHiFT vocoder."""

    # Input/output
    in_channels: int = 80           # Mel channels
    sample_rate: int = 24000        # Output sample rate

    # Generator
    base_channels: int = 512
    upsample_rates: list[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 11, 7])

    # ResBlocks - 3 per upsample stage = 9 total
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )

    # Source module
    nb_harmonics: int = 8
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10

    # Source ResBlocks - 3 blocks, per-block kernel sizes
    source_resblock_kernel_sizes: list[int] = field(default_factory=lambda: [7, 7, 11])
    source_resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )

    # Source downsampling
    source_down_channels: list[int] = field(default_factory=lambda: [256, 128, 64])
    source_down_kernels: list[int] = field(default_factory=lambda: [30, 6, 1])

    # iSTFT
    istft_n_fft: int = 16
    istft_hop_len: int = 4

    # F0 predictor - 5 conv layers
    f0_channels: int = 512
    f0_num_convs: int = 5
    f0_kernel_sizes: list[int] = field(default_factory=lambda: [4, 3, 3, 3, 3])

    # Other
    lrelu_slope: float = 0.1
    audio_limit: float = 0.99


class Snake(nn.Module):
    """Snake activation: x + (1/alpha) * sin^2(alpha * x)

    Uses custom Metal kernel when available for ~1.4x speedup.
    """

    # Class-level flag to enable/disable Metal kernel
    use_metal_kernel: bool = True

    def __init__(self, channels: int):
        super().__init__()
        # Initialize alpha to 1.0
        self.alpha = mx.ones((channels,))
        self._metal_kernel = None
        self._kernel_failed = False

    def _get_metal_kernel(self):
        """Lazy-load Metal kernel to avoid import overhead."""
        if self._metal_kernel is None and not self._kernel_failed:
            try:
                from tools.metal_kernels.kernels.snake1d import snake1d_custom
                self._metal_kernel = snake1d_custom
            except ImportError:
                self._kernel_failed = True
        return self._metal_kernel

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, L]
        # alpha: [C]

        # Try Metal kernel if enabled
        if Snake.use_metal_kernel and not self._kernel_failed:
            kernel = self._get_metal_kernel()
            if kernel is not None:
                # Kernel expects [..., C], so transpose [B, C, L] -> [B, L, C]
                x_t = x.transpose(0, 2, 1)
                out_t = kernel(x_t, self.alpha)
                # Transpose back [B, L, C] -> [B, C, L]
                return out_t.transpose(0, 2, 1)

        # Fallback: baseline implementation
        alpha = self.alpha[None, :, None]
        return x + (1.0 / (alpha + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


class Conv1d(nn.Module):
    """Conv1d wrapper handling PyTorch [B, C, L] to MLX [B, L, C] format."""

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
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    @property
    def weight(self):
        return self.conv.weight

    @weight.setter
    def weight(self, value):
        self.conv.weight = value

    @property
    def bias(self):
        return self.conv.bias

    @bias.setter
    def bias(self, value):
        self.conv.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, L] (PyTorch format)
        # MLX Conv1d expects [B, L, C]
        x = x.transpose(0, 2, 1)  # [B, L, C]
        x = self.conv(x)
        # Output back to [B, C, L]
        return x.transpose(0, 2, 1)


class ConvTranspose1d(nn.Module):
    """ConvTranspose1d wrapper handling PyTorch [B, C, L] to MLX [B, L, C] format."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    @property
    def weight(self):
        return self.conv.weight

    @weight.setter
    def weight(self, value):
        self.conv.weight = value

    @property
    def bias(self):
        return self.conv.bias

    @bias.setter
    def bias(self, value):
        self.conv.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, L] (PyTorch format)
        # MLX ConvTranspose1d expects [B, L, C]
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        return x.transpose(0, 2, 1)


class ResBlock(nn.Module):
    """HiFi-GAN residual block with Snake activations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.num_layers = len(dilations)

        # First set of convs and activations
        self.convs1 = []
        self.activations1 = []
        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(
                Conv1d(channels, channels, kernel_size, padding=padding, dilation=d),
            )
            self.activations1.append(Snake(channels))

        # Second set of convs and activations
        self.convs2 = []
        self.activations2 = []
        for _ in dilations:
            padding = (kernel_size - 1) // 2
            self.convs2.append(
                Conv1d(channels, channels, kernel_size, padding=padding, dilation=1),
            )
            self.activations2.append(Snake(channels))

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(self.num_layers):
            xt = self.activations1[i](x)
            xt = self.convs1[i](xt)
            xt = self.activations2[i](xt)
            xt = self.convs2[i](xt)
            x = x + xt
        return x


class F0Predictor(nn.Module):
    """F0 predictor matching CosyVoice3 structure."""

    def __init__(self, config: CausalHiFTConfig):
        super().__init__()

        # Condnet: sequence of weight-normalized Conv1d layers
        # Layers at indices 0, 2, 4, 6, 8 (with activations at 1, 3, 5, 7)
        self.condnet = []

        in_ch = config.in_channels
        for i in range(config.f0_num_convs):
            ks = config.f0_kernel_sizes[i] if i < len(config.f0_kernel_sizes) else 3
            padding = (ks - 1) // 2
            self.condnet.append(Conv1d(in_ch, config.f0_channels, ks, padding=padding))
            in_ch = config.f0_channels

        # Classifier: Linear layer
        self.classifier = nn.Linear(config.f0_channels, 1)

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Predict F0 from mel spectrogram.

        Args:
            mel: [B, mel_dim, L]

        Returns:
            F0 in Hz [B, 1, L]
        """
        x = mel
        for _i, conv in enumerate(self.condnet):
            x = conv(x)
            x = nn.leaky_relu(x, 0.1)

        # classifier expects [B, L, C]
        x = x.transpose(0, 2, 1)
        x = self.classifier(x)  # [B, L, 1]
        x = x.transpose(0, 2, 1)  # [B, 1, L]

        return nn.relu(x)  # F0 should be positive


class SourceModule(nn.Module):
    """Neural source-filter source module."""

    def __init__(self, config: CausalHiFTConfig):
        super().__init__()
        self.sample_rate = config.sample_rate
        self.nb_harmonics = config.nb_harmonics

        # l_linear: maps harmonics to weight
        # Input: [B, L, nb_harmonics + 1] (harmonics + noise)
        self.l_linear = nn.Linear(config.nb_harmonics + 1, 1)

    def __call__(
        self,
        f0: mx.array,           # [B, 1, L]
        upsample_factor: int,
    ) -> mx.array:
        """
        Generate source signal from F0.

        Returns:
            source: [B, nb_harmonics * 2 + 2, L * upsample_factor]
        """
        B, _, L = f0.shape

        # Upsample F0 to audio rate
        f0_up = mx.repeat(f0, upsample_factor, axis=2)  # [B, 1, L*up]
        L_audio = f0_up.shape[2]

        # Generate phase
        phase_inc = f0_up / self.sample_rate
        phase = mx.cumsum(phase_inc, axis=2) * 2 * math.pi  # [B, 1, L_audio]

        # Generate harmonics (sin and cos)
        harmonics = []
        for h in range(self.nb_harmonics):
            harmonic_phase = phase * (h + 1)
            harmonics.append(mx.sin(harmonic_phase))
            harmonics.append(mx.cos(harmonic_phase))

        # Add noise channels
        noise = mx.random.normal((B, 2, L_audio)) * 0.003

        # Stack all: [B, nb_harmonics * 2 + 2, L_audio]
        return mx.concatenate(harmonics + [noise], axis=1)



class CausalHiFTGenerator(nn.Module):
    """
    CausalHiFT Generator matching CosyVoice3 hift.pt structure.

    Weight keys expected:
    - conv_pre, conv_post: weight-normalized
    - ups.{0,1,2}: weight-normalized ConvTranspose1d
    - resblocks.{0-8}: Snake activations + weight-normalized convs
    - source_downs.{0,1,2}: regular Conv1d
    - source_resblocks.{0,1,2}: Snake activations + weight-normalized convs
    - f0_predictor.condnet.{0,2,4,6,8}: weight-normalized Conv1d
    - f0_predictor.classifier: Linear
    - m_source.l_linear: Linear
    """

    def __init__(self, config: CausalHiFTConfig):
        super().__init__()
        self.config = config

        # Calculate total upsample factor
        self.upsample_factor = 1
        for r in config.upsample_rates:
            self.upsample_factor *= r

        # F0 predictor
        self.f0_predictor = F0Predictor(config)

        # Source module
        self.m_source = SourceModule(config)

        # Source downsampling: process source signal at each upsample level
        # Source is at full audio rate (L * total_upsample).
        # At each stage i, filter is at L * product(upsample_rates[:i+1]).
        # source_downs stride = total_upsample / cumulative_upsample
        # source_downs.0: [256, 18, 30] - stride=15 (120/8)
        # source_downs.1: [128, 18, 6]  - stride=3 (120/40)
        # source_downs.2: [64, 18, 1]   - stride=1 (120/120)
        source_in_ch = config.nb_harmonics * 2 + 2  # 18
        self.source_downs = []
        cumulative_up = 1
        for i, (out_ch, ks) in enumerate(zip(
            config.source_down_channels,
            config.source_down_kernels, strict=False,
        )):
            cumulative_up *= config.upsample_rates[i]
            stride = self.upsample_factor // cumulative_up
            padding = (ks - stride) // 2 if stride > 1 else 0
            self.source_downs.append(
                Conv1d(source_in_ch, out_ch, ks, stride=stride, padding=padding),
            )

        # Source ResBlocks - one per upsample stage with per-block kernel sizes
        self.source_resblocks = []
        for _i, (out_ch, ks) in enumerate(zip(
            config.source_down_channels,
            config.source_resblock_kernel_sizes, strict=False,
        )):
            self.source_resblocks.append(
                ResBlock(out_ch, kernel_size=ks, dilations=(1, 3, 5)),
            )

        # Pre-conv (weight normalized)
        self.conv_pre = Conv1d(
            config.in_channels,
            config.base_channels,
            kernel_size=5,
            padding=2,
        )

        # Upsampling layers (weight normalized ConvTranspose1d)
        self.ups = []
        in_ch = config.base_channels
        for _i, (rate, ks) in enumerate(zip(
            config.upsample_rates, config.upsample_kernel_sizes, strict=False,
        )):
            out_ch = in_ch // 2
            padding = (ks - rate) // 2
            self.ups.append(
                ConvTranspose1d(in_ch, out_ch, ks, stride=rate, padding=padding),
            )
            in_ch = out_ch

        # ResBlocks - 3 per upsample stage (9 total)
        self.resblocks = []
        in_ch = config.base_channels
        for _i in range(len(config.upsample_rates)):
            in_ch = in_ch // 2
            for ks, dilations in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes, strict=False,
            ):
                self.resblocks.append(
                    ResBlock(in_ch, ks, tuple(dilations)),
                )

        # Post-conv (weight normalized) - output for iSTFT
        final_ch = config.base_channels // (2 ** len(config.upsample_rates))  # 64
        self.conv_post = Conv1d(
            final_ch,
            config.istft_n_fft + 2,  # 18 for magnitude and phase
            kernel_size=7,
            padding=3,
        )

        self.audio_limit = config.audio_limit
        self._compiled_forward = None
        self._is_compiled = False

    def __call__(
        self,
        mel: mx.array,
        f0: mx.array | None = None,
    ) -> mx.array:
        """
        Generate audio from mel spectrogram.

        Args:
            mel: Mel spectrogram [B, mel_dim, L]
            f0: Optional pre-computed F0 [B, 1, L]

        Returns:
            Audio waveform [B, L * upsample_factor]
        """
        if self._is_compiled and self._compiled_forward is not None:
            return self._compiled_forward(mel, f0)
        return self.forward(mel, f0)

    def _istft(self, x: mx.array) -> mx.array:
        """
        Inverse STFT synthesis.

        Args:
            x: [B, n_fft + 2, L] containing magnitude/phase

        Returns:
            Audio [B, L * hop_len]
        """
        n_fft = self.config.istft_n_fft
        hop_len = self.config.istft_hop_len

        # Split into magnitude and phase
        mag = x[:, :n_fft // 2 + 1, :]
        phase = x[:, n_fft // 2 + 1:, :]

        # Convert to complex
        real = mag * mx.cos(phase)
        imag = mag * mx.sin(phase)

        B, _, L = mag.shape
        L_out = L * hop_len

        # Window
        window = mx.array([0.5 * (1 - math.cos(2 * math.pi * i / n_fft))
                          for i in range(n_fft)])

        # Overlap-add synthesis
        audio = mx.zeros((B, L_out))

        for t in range(L):
            frame_complex = real[:, :, t] + 1j * imag[:, :, t]
            frame = mx.fft.irfft(frame_complex, n=n_fft)
            frame = frame * window

            start = t * hop_len
            end = start + n_fft
            if end <= L_out:
                audio = audio.at[:, start:end].add(frame)

        return audio

    def compile_model(self) -> None:
        """
        Compile the vocoder forward pass for faster inference.

        Uses mx.compile to optimize the computation graph.
        Call this after loading weights and before inference.

        Measured speedup: ~1.38x (239ms vs 330ms for 50 mel frames)
        """
        # Store reference to the original forward method
        original_forward = self.forward

        # Compile the forward function
        self._compiled_forward = mx.compile(original_forward)
        self._is_compiled = True

    def forward(self, mel: mx.array, f0: mx.array | None = None) -> mx.array:
        """Forward pass - use __call__ which handles compiled vs uncompiled."""
        # Predict F0 if not provided
        if f0 is None:
            f0 = self.f0_predictor(mel)

        # Generate source signal
        source = self.m_source(f0, self.upsample_factor)

        # Main filter path
        x = self.conv_pre(mel)

        # Upsampling with ResBlocks and source injection
        resblock_idx = 0
        for i, up in enumerate(self.ups):
            x = nn.leaky_relu(x, 0.1)
            x = up(x)

            # Downsample source and add
            source_ds = self.source_downs[i](source)
            source_ds = self.source_resblocks[i](source_ds)

            # Match lengths
            if source_ds.shape[2] < x.shape[2]:
                pad_len = x.shape[2] - source_ds.shape[2]
                source_ds = mx.pad(source_ds, [(0, 0), (0, 0), (0, pad_len)])
            elif source_ds.shape[2] > x.shape[2]:
                source_ds = source_ds[:, :, :x.shape[2]]

            x = x + source_ds

            # Apply ResBlocks
            xs = None
            for _ in range(len(self.config.resblock_kernel_sizes)):
                if xs is None:
                    xs = self.resblocks[resblock_idx](x)
                else:
                    xs = xs + self.resblocks[resblock_idx](x)
                resblock_idx += 1
            x = xs / len(self.config.resblock_kernel_sizes)

        # Post-conv
        x = nn.leaky_relu(x, 0.1)
        x = self.conv_post(x)

        # iSTFT synthesis
        audio = self._istft(x)

        # Clamp output
        return mx.clip(audio, -self.audio_limit, self.audio_limit)



def create_cosyvoice3_vocoder_config() -> CausalHiFTConfig:
    """Create default CosyVoice3 vocoder config matching hift.pt."""
    return CausalHiFTConfig(
        in_channels=80,
        sample_rate=24000,
        base_channels=512,
        nb_harmonics=8,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_n_fft=16,
        istft_hop_len=4,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_down_channels=[256, 128, 64],
        source_down_kernels=[30, 6, 1],
        f0_channels=512,
        f0_num_convs=5,
        f0_kernel_sizes=[4, 3, 3, 3, 3],
        lrelu_slope=0.1,
        audio_limit=0.99,
    )
