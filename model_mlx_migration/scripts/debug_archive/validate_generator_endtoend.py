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
Validate Kokoro Generator end-to-end with deterministic inputs.

Tests the full generator path (not individual components) to verify
that the output audio is reasonable given known input statistics.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import math

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter

# ========== PyTorch Reference Components ==========


def pytorch_source_module(f0_up, weights):
    """PyTorch reference for source module."""
    sampling_rate = 24000
    harmonic_num = 8
    sine_amp = 0.1
    noise_std = 0.003

    # SineGen
    dim = harmonic_num + 1
    fn = f0_up.unsqueeze(-1) * torch.arange(1, dim + 1).float()  # [B, T, 9]

    rad_values = fn / sampling_rate
    # Cumsum phase
    phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi
    sines = torch.sin(phase) * sine_amp  # [B, T, 9]

    # UV mask
    uv = (f0_up > 10).float().unsqueeze(-1)  # [B, T, 1]

    # Add noise
    noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    noise = noise_amp * torch.randn_like(sines)

    sine_waves = sines * uv + noise

    # Linear combination
    linear_w = weights["l_linear.weight"]  # [1, 9]
    linear_b = weights["l_linear.bias"]  # [1]

    har_source = F.linear(sine_waves, linear_w, linear_b)  # [B, T, 1]
    har_source = torch.tanh(har_source)

    # Add output noise
    out_noise = torch.randn_like(har_source[..., 0]) * sine_amp / 3

    return har_source, out_noise, uv[..., 0]


def pytorch_stft(x, n_fft=20, hop=5):
    """PyTorch STFT matching the MLX implementation."""
    window = torch.hann_window(n_fft, periodic=True)
    spec = torch.stft(
        x,
        n_fft,
        hop,
        n_fft,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    return torch.cat([mag, phase], dim=1).transpose(1, 2)  # [B, F, 22]


def pytorch_istft(mag, phase, n_fft=20, hop=5):
    """PyTorch ISTFT matching the MLX implementation."""
    window = torch.hann_window(n_fft, periodic=True)
    # Transpose: [B, T, F] -> [B, F, T]
    mag_t = mag.transpose(1, 2)
    phase_t = phase.transpose(1, 2)

    spec = mag_t * torch.exp(phase_t * 1j)
    audio = torch.istft(spec, n_fft, hop, n_fft, window=window, center=True)
    return audio


# ========== Main Validation ==========


def main():
    print("=" * 60)
    print("Kokoro Generator End-to-End Validation")
    print("=" * 60)

    # Load MLX model
    print("\nLoading MLX model...")
    converter = KokoroConverter()
    mlx_model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_gen = mlx_model.decoder.generator

    # Load PyTorch checkpoint
    print("Loading PyTorch checkpoint...")
    ckpt = torch.load(
        "/Users/ayates/models/kokoro/kokoro-v1_0.pth",
        map_location="cpu",
        weights_only=True,
    )
    pt_decoder = ckpt["decoder"]

    # Extract source module weights
    source_weights = {
        "l_linear.weight": pt_decoder["module.generator.m_source.l_linear.weight"],
        "l_linear.bias": pt_decoder["module.generator.m_source.l_linear.bias"],
    }

    # Test with deterministic inputs
    print("\n" + "=" * 60)
    print("Test 1: Source Module + STFT Comparison")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # Create test F0 (7 phonemes -> 28 samples after predictor)
    batch = 1
    f0_len = 28
    f0_np = np.abs(np.random.randn(batch, f0_len).astype(np.float32)) * 100 + 200

    # Total upsampling: 10 * 6 * 5 = 300
    total_upp = 60 * 5  # (10*6) * 5
    f0_up_np = np.repeat(f0_np, total_upp, axis=1)

    print(f"F0: shape={f0_np.shape}, mean={f0_np.mean():.2f}Hz")
    print(f"F0 upsampled: shape={f0_up_np.shape}")

    # PyTorch source
    f0_up_pt = torch.from_numpy(f0_up_np)
    pt_har, pt_noise, pt_uv = pytorch_source_module(f0_up_pt, source_weights)
    pt_har_np = pt_har.squeeze(-1).numpy()

    print("\nPyTorch harmonic source:")
    print(f"  Shape: {pt_har_np.shape}")
    print(f"  Mean: {pt_har_np.mean():.6f}, Std: {pt_har_np.std():.6f}")

    # MLX source (need to set deterministic)
    # Note: MLX has random components, so we compare statistics not exact values
    mlx_f0 = mx.array(f0_np)
    mlx_har, mlx_noise, mlx_uv = mlx_gen.m_source(mlx_f0, total_upp)
    mx.eval(mlx_har)
    mlx_har_np = np.array(mlx_har.squeeze(-1))

    print("\nMLX harmonic source:")
    print(f"  Shape: {mlx_har_np.shape}")
    print(f"  Mean: {mlx_har_np.mean():.6f}, Std: {mlx_har_np.std():.6f}")

    # Compare STFT
    print("\n" + "=" * 60)
    print("Test 2: STFT Comparison (deterministic)")
    print("=" * 60)

    # Use same harmonic signal for STFT comparison
    test_signal = pt_har_np[0]  # [samples]

    # PyTorch STFT
    pt_spec = pytorch_stft(torch.from_numpy(test_signal[None]))
    pt_spec_np = pt_spec.numpy()

    # MLX STFT
    mlx_spec = mlx_gen._source_stft(mx.array(test_signal[None]))
    mx.eval(mlx_spec)
    mlx_spec_np = np.array(mlx_spec)

    print(f"PyTorch STFT: shape={pt_spec_np.shape}")
    print(f"MLX STFT: shape={mlx_spec_np.shape}")

    min_frames = min(pt_spec_np.shape[1], mlx_spec_np.shape[1])
    mag_diff = np.abs(
        pt_spec_np[:, :min_frames, :11] - mlx_spec_np[:, :min_frames, :11]
    )

    print(f"\nMagnitude comparison (first {min_frames} frames):")
    print(f"  Max diff: {mag_diff.max():.8f}")
    print(f"  Mean diff: {mag_diff.mean():.8f}")

    # Compare complex spectrum (handles phase wrapping)
    pt_complex = pt_spec_np[:, :min_frames, :11] * np.exp(
        1j * pt_spec_np[:, :min_frames, 11:]
    )
    mlx_complex = mlx_spec_np[:, :min_frames, :11] * np.exp(
        1j * mlx_spec_np[:, :min_frames, 11:]
    )
    complex_diff = np.abs(pt_complex - mlx_complex)

    print("\nComplex spectrum comparison:")
    print(f"  Max diff: {complex_diff.max():.8f}")
    print(f"  Mean diff: {complex_diff.mean():.8f}")

    # Test ISTFT
    print("\n" + "=" * 60)
    print("Test 3: ISTFT Comparison (deterministic)")
    print("=" * 60)

    np.random.seed(123)
    frames = 100
    n_bins = 11

    mag_test = np.random.rand(batch, frames, n_bins).astype(np.float32) * 0.5 + 0.1
    phase_test = (
        np.random.rand(batch, frames, n_bins).astype(np.float32) - 0.5
    ) * np.pi

    # PyTorch ISTFT
    pt_audio = pytorch_istft(torch.from_numpy(mag_test), torch.from_numpy(phase_test))
    pt_audio_np = pt_audio.numpy()

    # MLX ISTFT
    mlx_audio = mlx_gen._istft_synthesis(mx.array(mag_test), mx.array(phase_test))
    mx.eval(mlx_audio)
    mlx_audio_np = np.array(mlx_audio)

    print(
        f"PyTorch ISTFT: shape={pt_audio_np.shape}, RMS={np.sqrt((pt_audio_np**2).mean()):.6f}"
    )
    print(
        f"MLX ISTFT: shape={mlx_audio_np.shape}, RMS={np.sqrt((mlx_audio_np**2).mean()):.6f}"
    )

    min_len = min(pt_audio_np.shape[-1], mlx_audio_np.shape[-1])
    audio_diff = np.abs(pt_audio_np[..., :min_len] - mlx_audio_np[..., :min_len])

    print(f"\nAudio comparison (first {min_len} samples):")
    print(f"  Max diff: {audio_diff.max():.8f}")
    print(f"  Mean diff: {audio_diff.mean():.8f}")

    # Full generator test
    print("\n" + "=" * 60)
    print("Test 4: Full Generator Output Statistics")
    print("=" * 60)

    # Create realistic inputs
    np.random.seed(42)
    x_np = np.random.randn(1, 7, 512).astype(np.float32) * 0.1
    s_np = np.random.randn(1, 128).astype(np.float32) * 0.1
    f0_np = np.abs(np.random.randn(1, 28).astype(np.float32)) * 100 + 200

    # MLX generator
    mlx_x = mx.array(x_np)
    mlx_s = mx.array(s_np)
    mlx_f0 = mx.array(f0_np)

    mlx_out = mlx_gen(mlx_x, mlx_s, mlx_f0)
    mx.eval(mlx_out)
    mlx_out_np = np.array(mlx_out)

    print("MLX Generator output:")
    print(f"  Shape: {mlx_out_np.shape}")
    print(f"  RMS: {np.sqrt((mlx_out_np**2).mean()):.6f}")
    print(f"  Range: [{mlx_out_np.min():.4f}, {mlx_out_np.max():.4f}]")

    # Verify output is non-trivial
    if np.sqrt((mlx_out_np**2).mean()) > 0.01:
        print("\n  [PASS] Generator produces non-trivial audio")
    else:
        print("\n  [FAIL] Generator output is near-silent")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    results = {
        "stft_mag_max_diff": float(mag_diff.max()),
        "stft_complex_max_diff": float(complex_diff.max()),
        "istft_max_diff": float(audio_diff.max()),
        "generator_rms": float(np.sqrt((mlx_out_np**2).mean())),
    }

    print(f"\nSTFT magnitude max diff: {results['stft_mag_max_diff']:.2e}")
    print(f"STFT complex spectrum max diff: {results['stft_complex_max_diff']:.2e}")
    print(f"ISTFT max diff: {results['istft_max_diff']:.2e}")
    print(f"Generator output RMS: {results['generator_rms']:.4f}")

    # Overall pass/fail
    all_pass = (
        results["stft_mag_max_diff"] < 1e-5
        and results["stft_complex_max_diff"] < 1e-5
        and results["istft_max_diff"] < 1e-5
        and results["generator_rms"] > 0.01
    )

    if all_pass:
        print("\n[PASS] All validation checks passed")
        print("  - STFT matches PyTorch at <1e-5 precision")
        print("  - ISTFT matches PyTorch at <1e-5 precision")
        print("  - Generator produces reasonable audio output")
    else:
        print("\n[NEEDS REVIEW] Some checks need attention")

    return results


if __name__ == "__main__":
    main()
