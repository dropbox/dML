#!/usr/bin/env python3
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
PyTorch Generator Reference Implementation

Standalone implementation of Kokoro's Generator that can be compared
layer-by-layer with MLX implementation.

Loads weights directly from kokoro-v1_0.pth without needing kokoro package.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Seed for reproducibility
SEED = 42


def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


class WeightNormConv1d(nn.Module):
    """Conv1d with weight normalization."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.conv(x)


class WeightNormConvTranspose1d(nn.Module):
    """ConvTranspose1d with weight normalization."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.conv(x)


class AdaIN1d(nn.Module):
    """Adaptive Instance Normalization for 1D."""

    def __init__(self, channels, style_dim):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)
        self.channels = channels

    def forward(self, x, s):
        # x: [B, C, L], s: [B, style_dim]
        h = self.fc(s)  # [B, C*2]
        gamma, beta = h.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)

        # Instance norm
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        x_norm = (x - mean) / std

        return gamma * x_norm + beta


class AdaINResBlock1d(nn.Module):
    """AdaIN-conditioned ResBlock matching StyleTTS2."""

    def __init__(self, channels, kernel_size, style_dim, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.adain1 = nn.ModuleList()
        self.adain2 = nn.ModuleList()

        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, dilation=d, padding=padding
                    )
                )
            )
            self.convs2.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=(kernel_size - 1) // 2,
                    )
                )
            )
            self.adain1.append(AdaIN1d(channels, style_dim))
            self.adain2.append(AdaIN1d(channels, style_dim))

    def forward(self, x, s):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = a1(xt, s)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            xt = a2(xt, s)
            x = xt + x
        return x


class SourceModuleHnNSF(nn.Module):
    """Harmonic + Noise Source Module matching StyleTTS2."""

    def __init__(self, sample_rate=24000, harmonic_num=8, sine_amp=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        # Linear to combine harmonics to single channel
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def forward(self, f0, upp):
        # f0: [B, T], upp: upsampling factor
        # Output: har_source [B, T*upp, 1], noise [B, T*upp, 1], uv [B, T*upp, 1]
        with torch.no_grad():
            # Upsample F0
            f0_up = F.interpolate(
                f0.unsqueeze(1), scale_factor=upp, mode="nearest"
            ).squeeze(1)

            batch, samples = f0_up.shape

            # Generate harmonics
            harmonics = []
            for h in range(1, self.harmonic_num + 2):  # 1f, 2f, ..., 9f
                # Phase accumulation
                phase = torch.cumsum(f0_up * h / self.sample_rate, dim=1) * 2 * math.pi
                sine = torch.sin(phase) * self.sine_amp
                harmonics.append(sine)

            # Stack: [B, T*upp, num_harmonics]
            harmonics_stack = torch.stack(harmonics, dim=-1)

            # UV mask
            uv = (f0_up > 0).float().unsqueeze(-1)

            # Apply mask
            harmonics_stack = harmonics_stack * uv

        # Combine harmonics
        har_source = torch.tanh(self.l_linear(harmonics_stack))

        # Noise
        noise = torch.randn_like(uv) * self.sine_amp / 3

        return har_source, noise, uv


class PyTorchGenerator(nn.Module):
    """
    Generator matching StyleTTS2/Kokoro architecture.

    Processes features through progressive upsampling with style conditioning.
    """

    def __init__(self, style_dim=128):
        super().__init__()
        self.style_dim = style_dim

        # Config matching Kokoro
        upsample_rates = [10, 6]
        upsample_kernel_sizes = [20, 12]
        upsample_initial_channel = 512
        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        # Source module
        self.m_source = SourceModuleHnNSF()

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            padding = (kernel - rate) // 2
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        in_ch, out_ch, kernel, stride=rate, padding=padding
                    )
                )
            )

        # Noise convolutions (22 input channels from STFT)
        # Stage 0: stride=6 (product of remaining rates), kernel=12
        # Stage 1: stride=1, kernel=1
        self.noise_convs = nn.ModuleList(
            [
                nn.Conv1d(22, 256, 12, stride=6, padding=3),  # stage 0
                nn.Conv1d(22, 128, 1, stride=1, padding=0),  # stage 1
            ]
        )

        # Noise residual blocks
        self.noise_res = nn.ModuleList(
            [
                AdaINResBlock1d(256, 7, style_dim),
                AdaINResBlock1d(128, 11, style_dim),
            ]
        )

        # ResBlocks for each stage
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes
            ):
                self.resblocks.append(AdaINResBlock1d(ch, kernel, style_dim, dilations))

        # Post convolution (n_fft=20 -> 22 output channels)
        final_ch = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(final_ch, 22, 7, padding=3))

        # STFT parameters
        self.post_n_fft = 20
        self.istft_hop_size = 5

    def _source_stft(self, x):
        """STFT for source signal -> 22 channels (11 mag + 11 phase)."""
        n_fft = 20
        hop = 5

        # Center padding
        pad = n_fft // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")

        # STFT
        window = torch.hann_window(n_fft, device=x.device)
        spec = torch.stft(
            x_padded,
            n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=False,
        )
        # spec: [B, n_fft//2+1, frames]

        mag = torch.abs(spec)  # [B, 11, frames]
        phase = torch.angle(spec)  # [B, 11, frames]

        # Concat: [B, 22, frames]
        har = torch.cat([mag, phase], dim=1)

        return har

    def forward(self, x, s, f0):
        """
        Generate audio.

        Args:
            x: [B, L, 512] - Input features (NLC format)
            s: [B, 128] - Style vector
            f0: [B, L] - F0 curve

        Returns:
            audio: [B, samples]
        """
        # NLC -> NCL
        x = x.transpose(1, 2)

        # Calculate total upsampling
        total_upp = 10 * 6 * self.istft_hop_size  # 300

        # Source signal
        har_source, noise, uv = self.m_source(f0, total_upp)  # [B, samples, 1]

        # STFT on source: [B, 22, frames]
        har_1d = har_source.squeeze(-1)  # [B, samples]
        source = self._source_stft(har_1d)  # [B, 22, frames]

        # Progressive upsampling
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)

            # Source convolution BEFORE upsampling
            x_source = self.noise_convs[i](source)
            x_source = self.noise_res[i](x_source, s)

            # Upsample
            x = self.ups[i](x)

            # Reflection pad on last stage
            if i == self.num_upsamples - 1:
                x = F.pad(x, (1, 0), mode="reflect")

            # Align lengths
            if x_source.shape[2] < x.shape[2]:
                x_source = F.pad(x_source, (0, x.shape[2] - x_source.shape[2]))
            elif x_source.shape[2] > x.shape[2]:
                x_source = x_source[..., : x.shape[2]]

            # Add source
            x = x + x_source

            # ResBlocks
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x, s)
                else:
                    xs = xs + self.resblocks[idx](x, s)
            x = xs / self.num_kernels

        # Final conv
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)  # [B, 22, frames]

        # Mag/phase split
        n_bins = self.post_n_fft // 2 + 1  # 11
        log_mag = x[:, :n_bins, :].clamp(-10, 10)
        mag = torch.exp(log_mag)
        phase = torch.sin(x[:, n_bins:, :])

        # ISTFT
        audio = self._istft(mag, phase)

        return audio

    def _istft(self, mag, phase):
        """ISTFT synthesis."""
        n_fft = self.post_n_fft
        hop = self.istft_hop_size

        # Construct complex spectrum
        spectrum = mag * torch.exp(1j * phase)

        # ISTFT
        window = torch.hann_window(n_fft, device=mag.device)
        audio = torch.istft(
            spectrum,
            n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=False,
        )

        return audio.clamp(-1, 1)


def load_generator_weights(generator: PyTorchGenerator, weights_path: str):
    """Load weights from kokoro-v1_0.pth into generator."""
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    decoder = state["decoder"]

    # Build mapping from kokoro weights to our generator
    gen_state = {}

    # m_source.l_linear
    gen_state["m_source.l_linear.weight"] = decoder[
        "module.generator.m_source.l_linear.weight"
    ]
    gen_state["m_source.l_linear.bias"] = decoder[
        "module.generator.m_source.l_linear.bias"
    ]

    # noise_convs
    for i in range(2):
        gen_state[f"noise_convs.{i}.weight"] = decoder[
            f"module.generator.noise_convs.{i}.weight"
        ]
        gen_state[f"noise_convs.{i}.bias"] = decoder[
            f"module.generator.noise_convs.{i}.bias"
        ]

    # noise_res (AdaINResBlock1d)
    for i in range(2):
        prefix_src = f"module.generator.noise_res.{i}"
        prefix_dst = f"noise_res.{i}"

        for j in range(3):  # 3 dilations
            # convs1
            gen_state[f"{prefix_dst}.convs1.{j}.weight_g"] = decoder[
                f"{prefix_src}.convs1.{j}.weight_g"
            ]
            gen_state[f"{prefix_dst}.convs1.{j}.weight_v"] = decoder[
                f"{prefix_src}.convs1.{j}.weight_v"
            ]
            gen_state[f"{prefix_dst}.convs1.{j}.bias"] = decoder[
                f"{prefix_src}.convs1.{j}.bias"
            ]
            # convs2
            gen_state[f"{prefix_dst}.convs2.{j}.weight_g"] = decoder[
                f"{prefix_src}.convs2.{j}.weight_g"
            ]
            gen_state[f"{prefix_dst}.convs2.{j}.weight_v"] = decoder[
                f"{prefix_src}.convs2.{j}.weight_v"
            ]
            gen_state[f"{prefix_dst}.convs2.{j}.bias"] = decoder[
                f"{prefix_src}.convs2.{j}.bias"
            ]
            # adain1, adain2
            gen_state[f"{prefix_dst}.adain1.{j}.fc.weight"] = decoder[
                f"{prefix_src}.adain1.{j}.fc.weight"
            ]
            gen_state[f"{prefix_dst}.adain1.{j}.fc.bias"] = decoder[
                f"{prefix_src}.adain1.{j}.fc.bias"
            ]
            gen_state[f"{prefix_dst}.adain2.{j}.fc.weight"] = decoder[
                f"{prefix_src}.adain2.{j}.fc.weight"
            ]
            gen_state[f"{prefix_dst}.adain2.{j}.fc.bias"] = decoder[
                f"{prefix_src}.adain2.{j}.fc.bias"
            ]

    # ups (ConvTranspose1d with weight_norm)
    for i in range(2):
        gen_state[f"ups.{i}.weight_g"] = decoder[f"module.generator.ups.{i}.weight_g"]
        gen_state[f"ups.{i}.weight_v"] = decoder[f"module.generator.ups.{i}.weight_v"]
        gen_state[f"ups.{i}.bias"] = decoder[f"module.generator.ups.{i}.bias"]

    # resblocks
    for i in range(6):  # 2 stages * 3 kernels
        prefix_src = f"module.generator.resblocks.{i}"
        prefix_dst = f"resblocks.{i}"

        for j in range(3):
            gen_state[f"{prefix_dst}.convs1.{j}.weight_g"] = decoder[
                f"{prefix_src}.convs1.{j}.weight_g"
            ]
            gen_state[f"{prefix_dst}.convs1.{j}.weight_v"] = decoder[
                f"{prefix_src}.convs1.{j}.weight_v"
            ]
            gen_state[f"{prefix_dst}.convs1.{j}.bias"] = decoder[
                f"{prefix_src}.convs1.{j}.bias"
            ]
            gen_state[f"{prefix_dst}.convs2.{j}.weight_g"] = decoder[
                f"{prefix_src}.convs2.{j}.weight_g"
            ]
            gen_state[f"{prefix_dst}.convs2.{j}.weight_v"] = decoder[
                f"{prefix_src}.convs2.{j}.weight_v"
            ]
            gen_state[f"{prefix_dst}.convs2.{j}.bias"] = decoder[
                f"{prefix_src}.convs2.{j}.bias"
            ]
            gen_state[f"{prefix_dst}.adain1.{j}.fc.weight"] = decoder[
                f"{prefix_src}.adain1.{j}.fc.weight"
            ]
            gen_state[f"{prefix_dst}.adain1.{j}.fc.bias"] = decoder[
                f"{prefix_src}.adain1.{j}.fc.bias"
            ]
            gen_state[f"{prefix_dst}.adain2.{j}.fc.weight"] = decoder[
                f"{prefix_src}.adain2.{j}.fc.weight"
            ]
            gen_state[f"{prefix_dst}.adain2.{j}.fc.bias"] = decoder[
                f"{prefix_src}.adain2.{j}.fc.bias"
            ]

    # conv_post
    gen_state["conv_post.weight_g"] = decoder["module.generator.conv_post.weight_g"]
    gen_state["conv_post.weight_v"] = decoder["module.generator.conv_post.weight_v"]
    gen_state["conv_post.bias"] = decoder["module.generator.conv_post.bias"]

    # Load with strict=False to see what's missing
    missing, unexpected = generator.load_state_dict(gen_state, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return generator


def trace_pytorch_generator(generator, x_np, s_np, f0_np) -> Dict[str, np.ndarray]:
    """Run forward pass and capture intermediate values."""
    traces = {}

    x = torch.tensor(x_np)  # [B, L, 512]
    s = torch.tensor(s_np)  # [B, 128]
    f0 = torch.tensor(f0_np)  # [B, L]

    with torch.no_grad():
        # NLC -> NCL
        x_ncl = x.transpose(1, 2)
        traces["input_x_ncl"] = x_ncl.numpy()

        # Source module
        total_upp = 300
        har_source, noise, uv = generator.m_source(f0, total_upp)
        traces["source_har"] = har_source.numpy()
        traces["source_uv"] = uv.numpy()

        # Source STFT
        har_1d = har_source.squeeze(-1)
        source = generator._source_stft(har_1d)
        traces["source_stft"] = source.numpy()

        print(f"Source STFT shape: {source.shape}")
        print(f"Source STFT range: [{source.min():.4f}, {source.max():.4f}]")

        # Full forward
        x = x_ncl
        for i in range(generator.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            traces[f"after_leaky_relu_{i}"] = x.numpy()

            x_source = generator.noise_convs[i](source)
            traces[f"noise_conv_{i}_out"] = x_source.numpy()

            x_source = generator.noise_res[i](x_source, s)
            traces[f"noise_res_{i}_out"] = x_source.numpy()

            x = generator.ups[i](x)
            traces[f"ups_{i}_out"] = x.numpy()

            if i == generator.num_upsamples - 1:
                x = F.pad(x, (1, 0), mode="reflect")
                traces[f"after_reflect_pad_{i}"] = x.numpy()

            if x_source.shape[2] < x.shape[2]:
                x_source = F.pad(x_source, (0, x.shape[2] - x_source.shape[2]))
            elif x_source.shape[2] > x.shape[2]:
                x_source = x_source[..., : x.shape[2]]

            traces[f"x_source_aligned_{i}"] = x_source.numpy()
            traces[f"x_before_add_{i}"] = x.numpy()

            x = x + x_source
            traces[f"after_source_add_{i}"] = x.numpy()

            xs = None
            for j in range(generator.num_kernels):
                idx = i * generator.num_kernels + j
                if xs is None:
                    xs = generator.resblocks[idx](x, s)
                else:
                    xs = xs + generator.resblocks[idx](x, s)
            x = xs / generator.num_kernels
            traces[f"after_resblocks_{i}"] = x.numpy()

        x = F.leaky_relu(x, 0.1)
        traces["before_conv_post"] = x.numpy()

        x = generator.conv_post(x)
        traces["conv_post_out"] = x.numpy()

        # Mag/phase
        n_bins = 11
        log_mag = x[:, :n_bins, :].clamp(-10, 10)
        mag = torch.exp(log_mag)
        phase = torch.sin(x[:, n_bins:, :])

        traces["log_mag"] = log_mag.numpy()
        traces["mag"] = mag.numpy()
        traces["phase"] = phase.numpy()

        # ISTFT
        audio = generator._istft(mag, phase)
        traces["final_audio"] = audio.numpy()

    return traces


def main():
    print("=" * 70)
    print("PYTORCH GENERATOR REFERENCE")
    print("=" * 70)

    set_seeds()

    # Create generator
    generator = PyTorchGenerator(style_dim=128)

    # Load weights
    weights_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
    print(f"\nLoading weights from: {weights_path}")
    generator = load_generator_weights(generator, str(weights_path))
    generator.eval()

    # Create test inputs
    batch = 1
    length = 32
    x_np = np.random.randn(batch, length, 512).astype(np.float32) * 0.1
    s_np = np.random.randn(batch, 128).astype(np.float32) * 0.1
    f0_np = np.full((batch, length), 200.0, dtype=np.float32)

    print("\nRunning PyTorch trace...")
    traces = trace_pytorch_generator(generator, x_np, s_np, f0_np)

    print("\n" + "=" * 70)
    print("PYTORCH TRACE SUMMARY")
    print("=" * 70)

    for key in sorted(traces.keys()):
        val = traces[key]
        print(f"{key}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")

    # Save traces
    output_path = Path("reports/main/pytorch_generator_traces.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **traces)
    print(f"\nTraces saved to {output_path}")


if __name__ == "__main__":
    main()
