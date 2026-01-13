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
Trace ISTFT to understand audio divergence.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load traces
    traces_path = Path("/tmp/kokoro_ref/generator_internal_traces.npz")
    if not traces_path.exists():
        print("No traces available")
        return 1
    pt_traces = np.load(traces_path)

    # Use PyTorch conv_post output as reference
    conv_post_pt = pt_traces["conv_post_out"]  # [1, 22, 7561] NCL
    final_audio_pt = pt_traces["final_audio"]  # [1, 1, 37800]

    print(
        f"PT conv_post: {conv_post_pt.shape}, range: [{conv_post_pt.min():.4f}, {conv_post_pt.max():.4f}]"
    )
    print(
        f"PT final_audio: {final_audio_pt.shape}, range: [{final_audio_pt.min():.4f}, {final_audio_pt.max():.4f}]"
    )

    # Load MLX model
    print("\n=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator

    # Use PyTorch conv_post as input to test ISTFT only
    # Convert NCL to NLC for MLX
    x = mx.array(conv_post_pt).transpose(0, 2, 1)  # [1, 7561, 22]
    mx.eval(x)
    print(
        f"\nMLX conv_post input: {x.shape}, range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
    )

    # ISTFT parameters
    post_n_fft = generator.post_n_fft  # 20
    istft_n_fft = generator.istft_n_fft  # 16
    istft_hop_size = generator.istft_hop_size

    print(f"post_n_fft: {post_n_fft}")
    print(f"istft_n_fft: {istft_n_fft}")
    print(f"istft_hop_size: {istft_hop_size}")

    n_bins = post_n_fft // 2 + 1  # 11

    # Log-magnitude to magnitude
    log_mag = mx.clip(x[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    mx.eval(mag)
    print(
        f"\nmag: {mag.shape}, range: [{float(mx.min(mag)):.4f}, {float(mx.max(mag)):.4f}]"
    )

    # Phase
    phase_logits = x[..., n_bins:]
    phase = mx.sin(phase_logits)
    mx.eval(phase)
    print(
        f"phase: {phase.shape}, range: [{float(mx.min(phase)):.4f}, {float(mx.max(phase)):.4f}]"
    )

    # Build complex STFT
    real = mag * phase
    imag = mag * mx.sqrt(1 - phase**2)
    mx.eval([real, imag])
    print(
        f"real: {real.shape}, range: [{float(mx.min(real)):.4f}, {float(mx.max(real)):.4f}]"
    )
    print(
        f"imag: {imag.shape}, range: [{float(mx.min(imag)):.4f}, {float(mx.max(imag)):.4f}]"
    )

    # Complex STFT matrix [batch, frames, n_fft//2+1, 2]
    mx.stack([real, imag], axis=-1)

    # Construct complex spectrum from magnitude and phase using same formula as PyTorch
    # PyTorch: magnitude * torch.exp(phase * 1j)
    spectrum = mag * mx.exp(1j * phase)
    mx.eval(spectrum)
    print(f"spectrum: {spectrum.shape}")

    # Use IRFFT like in the Generator's _istft_synthesis
    n_fft = post_n_fft
    hop = istft_hop_size

    time_frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)
    mx.eval(time_frames)
    print(
        f"time_frames: {time_frames.shape}, range: [{float(mx.min(time_frames.real)):.4f}, {float(mx.max(time_frames.real)):.4f}]"
    )

    # Hann window
    import math

    n = mx.arange(n_fft, dtype=mx.float32)
    window = 0.5 * (1 - mx.cos(2 * math.pi * n / n_fft))
    print(f"window: {window.shape}")

    # Apply window
    time_frames = time_frames * window
    mx.eval(time_frames)

    # Overlap-add
    frames = spectrum.shape[1]
    output_length = (frames - 1) * hop + n_fft
    print(f"output_length: {output_length}")

    # Simple overlap-add with window normalization
    audio = mx.zeros((1, output_length))
    window_sum = mx.zeros((output_length,))

    for i in range(frames):
        start = i * hop
        end = start + n_fft
        audio = audio.at[:, start:end].add(time_frames[:, i, :])
        window_sum = window_sum.at[start:end].add(
            window**2
        )  # Square for analysis/synthesis

    mx.eval([audio, window_sum])

    # Normalize by window overlap
    window_sum = mx.maximum(window_sum, 1e-8)  # Avoid division by zero
    audio = audio / window_sum
    mx.eval(audio)

    # Trim to expected length (center=True removes padding)
    # PyTorch center=True adds n_fft//2 padding on each side during STFT
    # ISTFT removes this padding
    pad = n_fft // 2
    audio = audio[:, pad : pad + 37800]
    mx.eval(audio)

    print(
        f"\nMLX audio (before clamp): {audio.shape}, range: [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]"
    )

    # Clamp
    audio = mx.clip(audio, -1.0, 1.0)
    print(
        f"MLX audio (after clamp): {audio.shape}, range: [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]"
    )

    # Compare
    pt_audio = final_audio_pt.flatten()
    mlx_audio = np.array(audio).flatten()

    min_len = min(len(pt_audio), len(mlx_audio))
    corr = np.corrcoef(pt_audio[:min_len], mlx_audio[:min_len])[0, 1]
    print(f"\nCorrelation with PyTorch: {corr:.6f}")

    # Check if PT audio is also from same STFT values
    # Re-trace from conv_post values
    print("\n=== Detailed comparison ===")
    print(f"PT audio stats: mean={pt_audio.mean():.6f}, std={pt_audio.std():.6f}")
    print(
        f"MLX audio stats: mean={mlx_audio[:min_len].mean():.6f}, std={mlx_audio[:min_len].std():.6f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
