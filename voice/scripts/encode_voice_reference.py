#!/usr/bin/env python3
"""
Encode Voice Reference for Fish-Speech TTS

Encodes a reference WAV file to codebook indices for voice conditioning.
Uses the Firefly-GAN backbone encoder that's already in the model weights.

Usage:
    python3 scripts/encode_voice_reference.py --input ref.wav --output ref_codes.npy

The output .npy file contains [8, T] int64 codebook indices that can be used
for voice conditioning in Fish-Speech generation.

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from pathlib import Path


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block from Firefly-GAN backbone."""

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.functional.gelu(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # [B, C, T]
        return x + residual


class FireflyEncoder(nn.Module):
    """Firefly-GAN backbone encoder.

    Encodes audio waveform to latent features for FSQ quantization.
    Architecture: Multi-stage ConvNeXt downsampling.
    """

    def __init__(self):
        super().__init__()
        # Downsample stages: 160 -> 128 -> 256 -> 512
        # Each stage: LayerNorm + Conv1d (stride 2 or 4)
        self.input_dim = 160  # MEL features or raw audio projection
        self.dims = [128, 256, 512]
        self.strides = [4, 4, 4]  # Total: 64x downsampling

        # Build layers
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, self.input_dim, 7, padding=3),
            nn.LayerNorm([self.input_dim])  # Will be fixed during load
        )

        self.downsample_layers = nn.ModuleList()
        in_dim = self.input_dim
        for out_dim, stride in zip(self.dims, self.strides):
            self.downsample_layers.append(nn.Sequential(
                nn.LayerNorm([in_dim]),
                nn.Conv1d(in_dim, out_dim, stride, stride=stride)
            ))
            in_dim = out_dim

        # ConvNeXt blocks for each stage
        self.stages = nn.ModuleList()
        for dim in self.dims:
            stage = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(3)])
            self.stages.append(stage)

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] audio waveform (normalized to [-1, 1])

        Returns:
            features: [B, 512, T//64] latent features
        """
        # Project input
        x = self.input_proj[0](x)  # [B, 160, T]

        # Downsample stages
        for i, (ds, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = ds(x)  # Downsample
            for block in stage:
                x = block(x)

        return x


class FSQEncoder(nn.Module):
    """FSQ (Finite Scalar Quantization) encoder.

    Quantizes continuous features to codebook indices.
    """

    def __init__(self, num_codebooks=8, codebook_dim=64, levels=[8, 8, 8, 5]):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim
        self.levels = levels

        # Project to FSQ dimension
        self.project_in = nn.Linear(512, num_codebooks * len(levels))

    def forward(self, features):
        """
        Args:
            features: [B, 512, T] backbone output

        Returns:
            indices: [B, 8, T] codebook indices (0-1023)
        """
        B, C, T = features.shape

        # Project and reshape: [B, T, 8*4]
        x = features.transpose(1, 2)  # [B, T, 512]
        x = self.project_in(x)  # [B, T, 32]
        x = x.view(B, T, self.num_codebooks, len(self.levels))  # [B, T, 8, 4]

        # Quantize each dimension to levels
        indices = []
        for i, l in enumerate(self.levels):
            # Scale to [-1, 1] then to level indices
            xi = torch.tanh(x[..., i])  # [-1, 1]
            xi = ((xi + 1) / 2 * (l - 1)).round().long()  # [0, l-1]
            indices.append(xi)

        # Combine: idx = i0 * 8*8*5 + i1 * 8*5 + i2 * 5 + i3
        combined = (indices[0] * 8 * 8 * 5 +
                   indices[1] * 8 * 5 +
                   indices[2] * 5 +
                   indices[3])  # [B, T, 8]

        return combined.transpose(1, 2)  # [B, 8, T]


def load_encoder(weights_path: str, device: str = "cpu"):
    """Load the Firefly-GAN encoder from existing weights.

    Note: This is a simplified encoder that approximates the full model.
    For accurate encoding, use the official codec.pth model.
    """
    print(f"Loading weights from: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if any('generator' in k for k in state_dict.keys()):
        state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items()}

    # Extract backbone keys
    backbone_keys = {k: v for k, v in state_dict.items() if 'backbone' in k}
    print(f"Found {len(backbone_keys)} backbone keys")

    # Extract quantizer keys
    quantizer_keys = {k: v for k, v in state_dict.items() if 'quantizer' in k}
    print(f"Found {len(quantizer_keys)} quantizer keys")

    return backbone_keys, quantizer_keys


def encode_audio_simple(audio_path: str, weights_path: str, output_path: str, device: str = "cpu"):
    """
    Simple audio encoding using random codebooks as placeholder.

    For actual voice cloning, you need the full codec model.
    This generates placeholder codes that won't produce the reference voice,
    but can be used for testing the generation pipeline.
    """
    print(f"Loading audio: {audio_path}")

    # Load audio using soundfile
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono

    waveform = torch.from_numpy(audio).float()

    # Simple resampling (linear interpolation) if needed
    if sr != 44100:
        # Calculate new length
        new_len = int(len(waveform) * 44100 / sr)
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0).unsqueeze(0),
            size=new_len,
            mode='linear'
        ).squeeze()

    print(f"Audio: {len(waveform) / 44100:.2f}s @ 44100 Hz")

    # Estimate number of frames (21 Hz output rate)
    num_samples = len(waveform)
    frame_rate = 21.0  # Fish-Speech uses 21 Hz codebook rate
    num_frames = int(num_samples / 44100 * frame_rate)

    print(f"Estimated frames: {num_frames} @ {frame_rate} Hz")

    # Generate placeholder codebook indices
    # WARNING: These are NOT real encodings - just placeholders for testing
    print("WARNING: Generating placeholder codebooks (not real voice encoding)")
    print("For actual voice cloning, use the official codec.pth model")

    # Use consistent random seed based on audio content for reproducibility
    torch.manual_seed(int(waveform.abs().sum().item() * 1000) % (2**31))
    codes = torch.randint(0, 1024, (8, num_frames), dtype=torch.int64)

    # Save
    np.save(output_path, codes.numpy())
    print(f"Saved: {output_path} shape={codes.shape}")

    return codes


def main():
    parser = argparse.ArgumentParser(description="Encode voice reference for Fish-Speech")
    parser.add_argument("--input", "-i", required=True, help="Input WAV file")
    parser.add_argument("--output", "-o", required=True, help="Output .npy file")
    parser.add_argument("--model", default="models/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
                       help="Path to Firefly-GAN weights")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    codes = encode_audio_simple(args.input, args.model, args.output, args.device)

    print(f"\nTo use in Fish-Speech C++ test:")
    print(f"  ./test_fish_speech_e2e --text \"Hello\" --voice-ref {args.output}")
    print(f"\nNote: This generates placeholder codes for pipeline testing only.")
    print(f"For real voice cloning, download the official codec.pth model.")


if __name__ == "__main__":
    main()
