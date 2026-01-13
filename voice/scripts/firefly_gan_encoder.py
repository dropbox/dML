#!/usr/bin/env python3
"""
Firefly-GAN Audio Encoder for Fish-Speech-1.5 (EXPERIMENTAL)

**WARNING**: This encoder is experimental and may not produce codes compatible
with the Fish-Speech transformer. The official approach requires codec.pth from
the gated fishaudio/openaudio-s1-mini repository.

This script attempts to encode audio using the backbone from
firefly-gan-vq-fsq-8x1024-21hz-generator.pth. However, the resulting codes
may differ from what the transformer expects for voice conditioning.

For reliable voice conditioning, use:
1. Bootstrap approach: Generate codes with the transformer, save with --save-codes
2. Official codec: fish_speech/models/dac/inference.py with codec.pth

Architecture:
    Audio (44100 Hz) → Mel-spectrogram (160 bins) → Backbone →
    Downsample → FSQ encode → Codebooks [8, T]

Author: Worker #491 (AI)
Date: 2025-12-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import argparse


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block for backbone and quantizer."""

    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.pwconv2 = nn.Linear(dim * expansion, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # [B, C, T]
        return residual + x


class FireflyBackbone(nn.Module):
    """
    ConvNeXt backbone encoder from Firefly-GAN.

    Takes mel-spectrogram [B, 160, T] and outputs features [B, 512, T/4].
    """

    def __init__(
        self,
        in_channels: int = 160,  # Mel bins
        channels: tuple = (128, 256, 384, 512),
        num_blocks: tuple = (3, 3, 9, 3),
        kernel_size: int = 7,
    ):
        super().__init__()

        # Downsample layers
        self.downsample_layers = nn.ModuleList()

        # First layer: Conv + LayerNorm
        self.downsample_layers.append(nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size, padding=kernel_size//2),
            nn.LayerNorm(channels[0])
        ))

        # Subsequent layers: LayerNorm + Conv
        for i in range(1, len(channels)):
            self.downsample_layers.append(nn.Sequential(
                nn.LayerNorm(channels[i-1]),
                nn.Conv1d(channels[i-1], channels[i], 1)
            ))

        # ConvNeXt stages
        self.stages = nn.ModuleList()
        for i, (ch, num) in enumerate(zip(channels, num_blocks)):
            stage = nn.ModuleList([
                ConvNeXtBlock(ch, kernel_size) for _ in range(num)
            ])
            self.stages.append(stage)

        # Final norm
        self.norm = nn.LayerNorm(channels[-1])

    def forward(self, x):
        """
        Args:
            x: [B, 160, T] mel-spectrogram
        Returns:
            [B, 512, T] features
        """
        for i, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            # Apply LayerNorm if not first layer
            if i > 0:
                x = x.transpose(1, 2)  # [B, T, C]
                x = downsample[0](x)   # LayerNorm
                x = x.transpose(1, 2)  # [B, C, T]
                x = downsample[1](x)   # Conv
            else:
                x = downsample[0](x)   # Conv
                x = x.transpose(1, 2)  # [B, T, C]
                x = downsample[1](x)   # LayerNorm
                x = x.transpose(1, 2)  # [B, C, T]

            # Apply ConvNeXt blocks
            for block in stage:
                x = block(x)

        # Final norm
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, C, T]

        return x


class QuantizerDownsample(nn.Module):
    """Temporal downsampling for quantizer."""

    def __init__(self, dim: int = 512, num_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv1d(dim, dim, 2, stride=2),  # 2x downsample
                ConvNeXtBlock(dim)
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FSQEncoder(nn.Module):
    """
    Finite Scalar Quantization encoder layer.

    Encodes continuous features to discrete indices.
    """

    def __init__(self, dim: int = 64, fsq_dim: int = 4, levels: list = None):
        super().__init__()
        if levels is None:
            levels = [8, 8, 8, 2]  # Product = 1024

        self.levels = levels
        self.dim = dim
        self.fsq_dim = fsq_dim

        self.project_in = nn.Linear(dim, fsq_dim)
        self.project_out = nn.Linear(fsq_dim, dim)

        # Pre-compute level bases
        bases = [1]
        for l in levels[:-1]:
            bases.append(bases[-1] * l)
        self.register_buffer('bases', torch.tensor(bases))
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode features to indices.

        Args:
            x: [B, dim, T] or [B, T, dim] features
        Returns:
            indices: [B, T] codebook indices (int64)
            quantized: [B, T, dim] quantized features
        """
        # Ensure [B, T, dim] format
        if x.dim() == 3 and x.shape[1] == self.dim:
            x = x.transpose(1, 2)  # [B, dim, T] -> [B, T, dim]

        # Project to FSQ dimensions
        z = self.project_in(x)  # [B, T, fsq_dim]

        # Quantize each dimension
        quantized_dims = []
        indices_dims = []

        for i, level in enumerate(self.levels):
            # Scale to [0, level-1]
            z_i = z[..., i]
            z_scaled = (z_i + 1) / 2 * (level - 1)  # Assuming input in [-1, 1]
            z_rounded = torch.round(z_scaled).clamp(0, level - 1)

            indices_dims.append(z_rounded.long())

            # Back to [-1, 1] for reconstruction
            z_reconstructed = 2 * z_rounded / (level - 1) - 1
            quantized_dims.append(z_reconstructed)

        # Combine indices: sum of (index * base) for each dimension
        indices = torch.zeros_like(indices_dims[0])
        for idx, base in zip(indices_dims, self.bases):
            indices = indices + idx * base

        # Stack quantized values and project back
        quantized_z = torch.stack(quantized_dims, dim=-1)  # [B, T, fsq_dim]
        quantized = self.project_out(quantized_z)  # [B, T, dim]

        return indices, quantized

    def forward(self, x):
        indices, _ = self.encode(x)
        return indices


class ResidualFSQEncoder(nn.Module):
    """
    Residual FSQ encoder with multiple codebooks.

    Uses residual quantization: quantize, subtract, repeat.
    """

    def __init__(self, num_codebooks: int = 8, dim: int = 64, fsq_dim: int = 4):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.dim = dim

        self.rvqs = nn.ModuleList([
            FSQEncoder(dim, fsq_dim) for _ in range(num_codebooks)
        ])

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode features using residual quantization.

        Args:
            x: [B, 512, T] features (512 = 8 codebooks * 64 dims)
        Returns:
            indices: [B, 8, T] codebook indices
            quantized: [B, 512, T] reconstructed features
        """
        B, C, T = x.shape

        # Split into codebook chunks [B, T, 8, 64]
        x_split = x.transpose(1, 2).reshape(B, T, self.num_codebooks, self.dim)

        all_indices = []
        all_quantized = []

        # Non-residual: quantize each chunk independently
        # (For FSQ in firefly-gan, each codebook handles its own 64-dim chunk)
        for i, rvq in enumerate(self.rvqs):
            chunk = x_split[:, :, i, :]  # [B, T, 64]
            indices, quantized = rvq.encode(chunk)  # [B, T], [B, T, 64]
            all_indices.append(indices)
            all_quantized.append(quantized)

        # Stack indices: [B, 8, T]
        indices = torch.stack(all_indices, dim=1)

        # Concatenate quantized: [B, T, 512]
        quantized = torch.cat(all_quantized, dim=-1)
        quantized = quantized.transpose(1, 2)  # [B, 512, T]

        return indices, quantized

    def forward(self, x):
        return self.encode(x)[0]


class FireflyGANEncoder(nn.Module):
    """
    Complete Firefly-GAN encoder: audio → codebooks.

    Pipeline:
        Audio → Mel-spectrogram → Backbone → Downsample → FSQ → Codebooks
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: int = 160,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_codebooks: int = 8,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_codebooks = num_codebooks

        # Mel-spectrogram (will be computed externally or via torchaudio)
        # The backbone expects [B, 160, T_mel]

        # Backbone encoder
        self.backbone = FireflyBackbone(in_channels=n_mels)

        # Quantizer downsample (4x temporal reduction)
        self.quantizer_downsample = QuantizerDownsample(512, num_layers=2)

        # FSQ encoder
        self.quantizer_fsq = ResidualFSQEncoder(num_codebooks, dim=64)

    def compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram from audio.

        Args:
            audio: [B, T] or [T] waveform at sample_rate
        Returns:
            [B, n_mels, T_mel] mel-spectrogram
        """
        import torchaudio.transforms as T

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Create mel transform on same device
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            normalized=True,
        ).to(audio.device)

        mel = mel_transform(audio)  # [B, n_mels, T_mel]

        # Log scale
        mel = torch.log(mel.clamp(min=1e-5))

        return mel

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encode mel-spectrogram to codebooks.

        Args:
            mel: [B, n_mels, T_mel] mel-spectrogram
        Returns:
            [B, num_codebooks, T_codes] codebook indices
        """
        # Backbone: [B, 160, T] → [B, 512, T]
        features = self.backbone(mel)

        # Downsample: [B, 512, T] → [B, 512, T/4]
        features = self.quantizer_downsample(features)

        # FSQ encode: [B, 512, T/4] → [B, 8, T/4]
        indices, _ = self.quantizer_fsq.encode(features)

        return indices

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        End-to-end: audio → codebooks.

        Args:
            audio: [B, T] or [T] waveform
        Returns:
            [B, num_codebooks, T_codes] codebook indices
        """
        mel = self.compute_mel(audio)
        return self.encode(mel)


def load_firefly_encoder(checkpoint_path: str, device: str = 'cpu') -> FireflyGANEncoder:
    """
    Load Firefly-GAN encoder from vocoder checkpoint.

    Args:
        checkpoint_path: Path to firefly-gan-vq-fsq-8x1024-21hz-generator.pth
        device: Device to load model to

    Returns:
        Loaded FireflyGANEncoder model
    """
    model = FireflyGANEncoder()

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Map and load backbone weights
    backbone_sd = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            # Map to our model format
            new_key = key.replace('backbone.', '')
            # Handle conv.weight → weight mapping
            new_key = new_key.replace('.conv.weight', '.weight')
            new_key = new_key.replace('.conv.bias', '.bias')
            new_key = new_key.replace('.dwconv.conv.', '.dwconv.')
            backbone_sd[new_key] = value

    # Load backbone weights (strict=False for our structure)
    try:
        model.backbone.load_state_dict(backbone_sd, strict=False)
        print(f"Loaded backbone weights: {len(backbone_sd)} keys")
    except Exception as e:
        print(f"Warning: backbone loading partial: {e}")

    # Load quantizer downsample weights
    downsample_sd = {}
    for key, value in state_dict.items():
        if key.startswith('quantizer.downsample'):
            new_key = key.replace('quantizer.downsample', 'layers')
            new_key = new_key.replace('.conv.weight', '.weight')
            new_key = new_key.replace('.conv.bias', '.bias')
            new_key = new_key.replace('.dwconv.conv.', '.dwconv.')
            downsample_sd[new_key] = value

    try:
        model.quantizer_downsample.load_state_dict(downsample_sd, strict=False)
        print(f"Loaded downsample weights: {len(downsample_sd)} keys")
    except Exception as e:
        print(f"Warning: downsample loading partial: {e}")

    # Load FSQ weights
    fsq_sd = {}
    for key, value in state_dict.items():
        if key.startswith('quantizer.residual_fsq'):
            new_key = key.replace('quantizer.residual_fsq.', '')
            fsq_sd[new_key] = value

    try:
        model.quantizer_fsq.load_state_dict(fsq_sd, strict=False)
        print(f"Loaded FSQ weights: {len(fsq_sd)} keys")
    except Exception as e:
        print(f"Warning: FSQ loading partial: {e}")

    model.eval()
    return model.to(device)


def encode_audio_file(
    audio_path: str,
    checkpoint_path: str,
    output_path: str = None,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Encode an audio file to codebook indices.

    Args:
        audio_path: Path to input audio file
        checkpoint_path: Path to firefly-gan checkpoint
        output_path: Optional path to save .npy file
        device: Device to use

    Returns:
        numpy array of shape [8, T] with codebook indices
    """
    import soundfile as sf
    from scipy import signal

    # Load audio using soundfile
    audio_data, sr = sf.read(audio_path)
    print(f"Loaded audio: {audio_path}")
    print(f"  Sample rate: {sr}")
    print(f"  Duration: {len(audio_data) / sr:.2f}s")

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample if needed
    if sr != 44100:
        num_samples = int(len(audio_data) * 44100 / sr)
        audio_data = signal.resample(audio_data, num_samples)
        print(f"  Resampled to 44100 Hz")

    # Convert to tensor [1, T]
    waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

    waveform = waveform.to(device)

    # Load encoder
    encoder = load_firefly_encoder(checkpoint_path, device)

    # Encode
    with torch.no_grad():
        codes = encoder(waveform)  # [1, 8, T]

    codes_np = codes[0].cpu().numpy()  # [8, T]
    print(f"  Output codes: {codes_np.shape}")

    # Save if requested
    if output_path:
        np.save(output_path, codes_np.astype(np.int64))
        print(f"  Saved to: {output_path}")

    return codes_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode audio to Fish-Speech codebooks')
    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', help='Output .npy file')
    parser.add_argument('--checkpoint', default='models/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth',
                        help='Path to firefly-gan checkpoint')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = args.input.rsplit('.', 1)[0] + '_codes.npy'

    encode_audio_file(args.input, args.checkpoint, args.output, args.device)
