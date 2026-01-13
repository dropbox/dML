#!/usr/bin/env python3
"""
Firefly-GAN Vocoder Model for Fish-Speech-1.5

This module provides the vocoder for decoding FSQ codebook indices to audio.
Based on the official Fish-Speech 1.5 implementation.

Architecture:
- 8 parallel FSQ codebooks (NOT residual)
- Each codebook has [8,5,5,5] levels (1000 codes) with 64-dim output
- 2x upsample (4x total) with ConvNeXt blocks
- HiFi-GAN style decoder with 5 upsampling layers

Decode path:
  indices [B, 8, T] -> FSQ decode [B, T, 512] -> upsample [B, 512, T*4]
                     -> HiFiGAN [B, 1, T*4*512] -> audio

Author: Worker #492-493 (AI)
Date: 2025-12-11

Key fixes from #492-493:
- FSQ levels: [8,5,5,5] (1000 codes, not 1024)
- FSQ decode: uses proper half_level formula from official FSQ
- ResBlock: uses SiLU activation (not LeakyReLU)
- ParallelBlock: averages outputs from multiple kernel sizes
- Kernel sizes: (16, 16, 4, 4, 4) and pre/post conv kernel=13
- Uses FishConvNet/FishTransConvNet (causal) convolutions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total=0):
    """Calculate extra padding for causal convolution."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x, paddings, mode='constant', value=0.0):
    """1D padding wrapper with reflect mode handling for small inputs."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def unpad1d(x, paddings):
    """Remove padding from 1D tensor."""
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def init_weights(m, mean=0.0, std=0.01):
    """Initialize Conv1D weights."""
    classname = m.__class__.__name__
    if classname.find("Conv1D") != -1:
        m.weight.data.normal_(mean, std)


class FishConvNet(nn.Module):
    """Causal convolution for Fish-Speech."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation

    def forward(self, x):
        pad = self.kernel_size - self.stride
        extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, pad)
        x = pad1d(x, (pad, int(extra_padding)), mode='constant', value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name='weight', dim=0):
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name=name, dim=dim)
        return self


class FishTransConvNet(nn.Module):
    """Causal transposed convolution for Fish-Speech."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                       stride=stride, dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name='weight', dim=0):
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name=name, dim=dim)
        return self


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first format."""

    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:  # channels_first
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for quantizer upsample."""

    def __init__(self, dim, mlp_ratio=4.0, kernel_size=7, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = FishConvNet(dim, dim, kernel_size=kernel_size, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x, apply_residual=True):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        if apply_residual:
            x = residual + x
        return x


class ResBlock1(nn.Module):
    """HiFi-GAN residual block with dilated convolutions and SiLU activation."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            FishConvNet(channels, channels, kernel_size, stride=1, dilation=d).weight_norm()
            for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            FishConvNet(channels, channels, kernel_size, stride=1, dilation=d).weight_norm()
            for d in dilation
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)  # Key fix: SiLU not LeakyReLU
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x


class ParallelBlock(nn.Module):
    """Parallel residual blocks with different kernel sizes (averaged output)."""

    def __init__(self, channels, kernel_sizes=(3, 7, 11), dilation_sizes=((1, 3, 5),) * 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock1(channels, k, d) for k, d in zip(kernel_sizes, dilation_sizes)
        ])

    def forward(self, x):
        # Key fix: Average outputs from different kernel sizes
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)


class FSQLayer(nn.Module):
    """Single FSQ layer for one codebook."""

    def __init__(self, dim=64, fsq_dim=4, levels=None):
        super().__init__()
        if levels is None:
            levels = [8, 5, 5, 5]  # 1000 codes
        self.dim = dim
        self.fsq_dim = fsq_dim
        self.levels = levels

        self.project_in = nn.Linear(dim, fsq_dim)
        self.project_out = nn.Linear(fsq_dim, dim)

        # Pre-compute bases for index decoding
        bases = [1]
        for l in levels[:-1]:
            bases.append(bases[-1] * l)
        self.register_buffer('bases', torch.tensor(bases))

    def decode(self, indices):
        """Decode indices to continuous values using official FSQ formula."""
        values = []
        for i, (level, base) in enumerate(zip(self.levels, self.bases.tolist())):
            level_idx = (indices // int(base)) % level
            # Official FSQ formula: scale to [-1, 1] range
            half_level = (level - 1) / 2
            val = (level_idx.float() - half_level) / half_level
            values.append(val)

        quantized = torch.stack(values, dim=-1)  # [B, T, fsq_dim]
        return self.project_out(quantized)  # [B, T, dim]


class ParallelFSQ(nn.Module):
    """8 parallel FSQ codebooks (not residual) for Fish-Speech 1.5."""

    def __init__(self, n_codebooks=8, dim_per_codebook=64):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.dim_per_codebook = dim_per_codebook
        self.rvqs = nn.ModuleList([
            FSQLayer(dim=dim_per_codebook) for _ in range(n_codebooks)
        ])

    def decode(self, indices):
        """
        Decode parallel codebook indices.

        Args:
            indices: [B, n_codebooks, T]

        Returns:
            [B, T, n_codebooks * dim_per_codebook] = [B, T, 512]
        """
        decoded = [self.rvqs[i].decode(indices[:, i, :]) for i in range(self.n_codebooks)]
        return torch.cat(decoded, dim=-1)


class QuantizerUpsample(nn.Module):
    """Upsample quantizer output (2x2 = 4x total)."""

    def __init__(self, dim=512, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                FishTransConvNet(dim, dim, kernel_size=2, stride=2),
                ConvNeXtBlock(dim=dim)
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HiFiGANHead(nn.Module):
    """HiFi-GAN decoder head for Fish-Speech 1.5."""

    def __init__(
        self,
        num_mels=512,
        upsample_initial_channel=512,
        upsample_rates=(8, 8, 2, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        pre_conv_kernel_size=13,
        post_conv_kernel_size=13,
    ):
        super().__init__()

        self.conv_pre = FishConvNet(num_mels, upsample_initial_channel,
                                    pre_conv_kernel_size, stride=1).weight_norm()

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                FishTransConvNet(in_ch, out_ch, k, stride=u).weight_norm()
            )
            self.resblocks.append(
                ParallelBlock(out_ch, resblock_kernel_sizes, resblock_dilation_sizes)
            )

        self.ups.apply(init_weights)

        final_ch = upsample_initial_channel // (2 ** len(upsample_rates))
        self.conv_post = FishConvNet(final_ch, 1, post_conv_kernel_size, stride=1).weight_norm()
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for up, resblock in zip(self.ups, self.resblocks):
            x = F.silu(x, inplace=True)
            x = up(x)
            x = resblock(x)

        x = F.silu(x, inplace=True)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class FireflyGANVocoder(nn.Module):
    """
    Complete Firefly-GAN Vocoder for Fish-Speech 1.5.

    Decodes 8-codebook FSQ indices to 44.1kHz audio.
    """

    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate

        self.quantizer_fsq = ParallelFSQ(n_codebooks=8, dim_per_codebook=64)
        self.quantizer_upsample = QuantizerUpsample(dim=512, num_layers=2)
        self.head = HiFiGANHead()

    def decode(self, indices):
        """
        Decode codebook indices to audio.

        Args:
            indices: [B, 8, T] or [8, T] tensor of codebook indices

        Returns:
            [B, 1, T*2048] audio waveform in [-1, 1]
        """
        if indices.dim() == 2:
            indices = indices.unsqueeze(0)

        # FSQ decode: [B, 8, T] -> [B, T, 512]
        z_q = self.quantizer_fsq.decode(indices)

        # Transpose for conv: [B, T, 512] -> [B, 512, T]
        z_q = z_q.transpose(1, 2)

        # Upsample: [B, 512, T] -> [B, 512, T*4]
        z_q = self.quantizer_upsample(z_q)

        # HiFi-GAN: [B, 512, T*4] -> [B, 1, T*4*512]
        audio = self.head(z_q)

        return audio

    def forward(self, indices):
        return self.decode(indices)


def _map_orig_to_model_key(orig_key):
    """Map checkpoint keys to our model structure."""
    if orig_key.startswith('backbone.'):
        return None
    if 'quantizer.downsample' in orig_key:
        return None

    if orig_key.startswith('quantizer.residual_fsq.'):
        return orig_key.replace('quantizer.residual_fsq.', 'quantizer_fsq.')
    if orig_key.startswith('quantizer.upsample.'):
        return orig_key.replace('quantizer.upsample.', 'quantizer_upsample.layers.')
    if orig_key.startswith('head.'):
        return orig_key

    return None


def load_firefly_gan_vocoder(checkpoint_path, device='cpu'):
    """
    Load Firefly-GAN vocoder from checkpoint.

    Args:
        checkpoint_path: Path to firefly-gan-vq-fsq-8x1024-21hz-generator.pth
        device: Device to load model to

    Returns:
        Loaded FireflyGANVocoder model
    """
    model = FireflyGANVocoder()

    orig_sd = torch.load(checkpoint_path, map_location=device, weights_only=True)

    mapped_sd = {}
    for orig_key, value in orig_sd.items():
        model_key = _map_orig_to_model_key(orig_key)
        if model_key:
            mapped_sd[model_key] = value

    result = model.load_state_dict(mapped_sd, strict=False)

    # Check for unexpected issues
    expected_missing = {'bases'}  # Internal buffers
    actual_missing = {k.split('.')[-1] for k in result.missing_keys}
    unexpected = actual_missing - expected_missing
    if unexpected:
        print(f"Warning: unexpected missing keys: {result.missing_keys}")

    if result.unexpected_keys:
        print(f"Warning: unexpected keys: {result.unexpected_keys}")

    model.eval()
    return model.to(device)


if __name__ == '__main__':
    import sys
    import numpy as np

    print("=" * 60)
    print("Testing FireflyGANVocoder (Fish-Speech 1.5)")
    print("=" * 60)

    checkpoint_path = 'models/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth'

    try:
        model = load_firefly_gan_vocoder(checkpoint_path)
        print(f"\nModel loaded successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Load and decode bootstrap voice
        codes = np.load('models/fish-speech-1.5/bootstrap_voice.npy')
        codes_tensor = torch.from_numpy(codes).long().unsqueeze(0)
        print(f"\nInput codes shape: {codes_tensor.shape}")

        with torch.no_grad():
            audio = model.decode(codes_tensor)

        print(f"Output audio shape: {audio.shape}")
        print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
        print(f"Audio mean: {audio.mean():.4f}, std: {audio.std():.4f}")

        saturated = (audio.abs() > 0.99).float().mean()
        print(f"Saturation: {saturated*100:.1f}%")

        # Save test audio
        import soundfile as sf
        audio_np = audio[0, 0].numpy()
        sf.write('test_firefly_gan_output.wav', audio_np, 44100)
        print(f"\nSaved: test_firefly_gan_output.wav ({len(audio_np)/44100:.2f}s)")

    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
