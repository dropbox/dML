#!/usr/bin/env python3
"""Run Zipformer reference inference with PyTorch.

This script loads the pretrained Zipformer model and runs inference,
saving intermediate outputs for validation against C++ implementation.

Usage:
    python scripts/run_zipformer_reference.py \
        --checkpoint checkpoints/zipformer/en-streaming/exp/pretrained.pt \
        --audio checkpoints/zipformer/en-streaming/test_wavs/1089-134686-0001.wav \
        --output checkpoints/zipformer/en-streaming/reference_outputs.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Add icefall to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools/third_party/icefall"))


def load_audio(audio_path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Load and resample audio file."""
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def compute_fbank(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    num_mel_bins: int = 80
) -> torch.Tensor:
    """Compute filter bank features."""
    # Use torchaudio's fbank implementation
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_rate,
        dither=0.0,
        energy_floor=1.0,
        frame_length=25.0,
        frame_shift=10.0,
        high_freq=0.0,
        htk_compat=True,
        low_freq=20.0,
        preemphasis_coefficient=0.97,
        raw_energy=True,
        round_to_power_of_two=True,
        snip_edges=False,
        subtract_mean=True,
        use_energy=False,
        use_log_fbank=True,
        use_power=True,
        vtln_high=-500.0,
        vtln_low=100.0,
        vtln_warp=1.0,
        window_type="povey",
    )
    return fbank


def run_reference_inference(
    checkpoint_path: Path,
    audio_path: Path,
    output_path: Path,
) -> None:
    """Run reference inference and save outputs."""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    # Load audio
    print(f"Loading audio: {audio_path}")
    waveform = load_audio(audio_path)
    print(f"  Waveform shape: {waveform.shape}")

    # Compute fbank features
    fbank = compute_fbank(waveform)
    print(f"  Fbank shape: {fbank.shape}")

    # Add batch dimension
    fbank = fbank.unsqueeze(0)  # (1, T, 80)
    fbank_lens = torch.tensor([fbank.shape[1]])

    # We can't easily instantiate the full model without icefall setup,
    # so let's manually run subsampling and first encoder layer

    # --- Subsampling (Conv2dSubsampling) ---
    # conv.0: 3x3 conv, 1->8 channels, no stride
    # conv.4: 3x3 conv, 8->32 channels, stride=2
    # conv.7: 3x3 conv, 32->128 channels, stride=(1,2)
    # convnext: 7x7 depthwise, pointwise1 (128->384), pointwise2 (384->128)
    # out: Linear(2432, 192)

    x = fbank.unsqueeze(1)  # (B, 1, T, F) = (1, 1, T, 80)

    # Conv layer 0
    conv0_w = state_dict["encoder_embed.conv.0.weight"]
    conv0_b = state_dict["encoder_embed.conv.0.bias"]
    x = torch.nn.functional.conv2d(x, conv0_w, conv0_b, padding=(0, 1))
    x = swoosh_r(x)  # Simplified - actual has Balancer+ScaleGrad
    print(f"  After conv.0: {x.shape}")

    # Conv layer 4
    conv4_w = state_dict["encoder_embed.conv.4.weight"]
    conv4_b = state_dict["encoder_embed.conv.4.bias"]
    x = torch.nn.functional.conv2d(x, conv4_w, conv4_b, stride=2, padding=0)
    x = swoosh_r(x)
    print(f"  After conv.4: {x.shape}")

    # Conv layer 7
    conv7_w = state_dict["encoder_embed.conv.7.weight"]
    conv7_b = state_dict["encoder_embed.conv.7.bias"]
    x = torch.nn.functional.conv2d(x, conv7_w, conv7_b, stride=(1, 2), padding=0)
    x = swoosh_r(x)
    print(f"  After conv.7: {x.shape}")

    # ConvNeXt (simplified - skip for now)
    # ... depthwise -> pointwise1 -> swoosh_l -> pointwise2 -> residual

    # ConvNeXt depthwise
    dw_w = state_dict["encoder_embed.convnext.depthwise_conv.weight"]
    dw_b = state_dict["encoder_embed.convnext.depthwise_conv.bias"]
    bypass = x
    x_conv = torch.nn.functional.conv2d(x, dw_w, dw_b, padding=3, groups=128)

    # Pointwise1
    pw1_w = state_dict["encoder_embed.convnext.pointwise_conv1.weight"]
    pw1_b = state_dict["encoder_embed.convnext.pointwise_conv1.bias"]
    x_conv = torch.nn.functional.conv2d(x_conv, pw1_w, pw1_b)
    x_conv = swoosh_l(x_conv)

    # Pointwise2
    pw2_w = state_dict["encoder_embed.convnext.pointwise_conv2.weight"]
    pw2_b = state_dict["encoder_embed.convnext.pointwise_conv2.bias"]
    x_conv = torch.nn.functional.conv2d(x_conv, pw2_w, pw2_b)

    x = bypass + x_conv
    print(f"  After convnext: {x.shape}")

    # Reshape and project
    b, c, t, f = x.shape
    x = x.transpose(1, 2).reshape(b, t, c * f)  # (B, T, C*F)
    print(f"  After reshape: {x.shape}")

    # Output linear
    out_w = state_dict["encoder_embed.out.weight"]
    out_b = state_dict["encoder_embed.out.bias"]
    x = torch.nn.functional.linear(x, out_w, out_b)
    print(f"  After out linear: {x.shape}")

    # BiasNorm (icefall formula):
    #   scales = (mean((x - bias)^2, dim=-1, keepdim=True) ^ -0.5) * exp(log_scale)
    #   ans = x * scales
    log_scale = state_dict["encoder_embed.out_norm.log_scale"]
    norm_bias = state_dict["encoder_embed.out_norm.bias"]
    x_minus_bias = x - norm_bias
    variance = (x_minus_bias ** 2).mean(dim=-1, keepdim=True)
    scales = (variance + 1e-8) ** -0.5 * log_scale.exp()
    x = x * scales
    print(f"  After BiasNorm: {x.shape}")

    # Save outputs
    outputs = {
        "input_fbank": fbank.numpy(),
        "fbank_shape": np.array(fbank.shape),
        "after_subsampling": x.detach().numpy(),
        "subsampling_shape": np.array(x.shape),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **outputs)
    print(f"\nSaved reference outputs to: {output_path}")

    # Print stats
    print("\nOutput statistics:")
    for name, arr in outputs.items():
        if isinstance(arr, np.ndarray) and arr.dtype in (np.float32, np.float64):
            print(f"  {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")


def swoosh_l(x: torch.Tensor) -> torch.Tensor:
    """SwooshL activation: x * sigmoid(x - 1)"""
    return x * torch.sigmoid(x - 1)


def swoosh_r(x: torch.Tensor) -> torch.Tensor:
    """SwooshR activation: x * sigmoid(x + 1) - 0.08 * x"""
    return x * torch.sigmoid(x + 1) - 0.08 * x


def main():
    parser = argparse.ArgumentParser(
        description="Run Zipformer reference inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/zipformer/en-streaming/exp/pretrained.pt"),
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("checkpoints/zipformer/en-streaming/test_wavs/1089-134686-0001.wav"),
        help="Path to audio file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/zipformer/en-streaming/reference_outputs.npz"),
        help="Path to save reference outputs"
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.audio.exists():
        print(f"ERROR: Audio file not found: {args.audio}")
        sys.exit(1)

    run_reference_inference(args.checkpoint, args.audio, args.output)


if __name__ == "__main__":
    main()
