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
Analyze conv_post differences in detail.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load traces
    ref = np.load("/tmp/kokoro_ref/tensors.npz")
    gen_traces = np.load("/tmp/kokoro_ref/generator_traces.npz")
    internal = np.load("/tmp/kokoro_ref/generator_internal_traces.npz")

    # PT conv_post output
    pt_conv_post = internal["conv_post_out"]  # [1, 22, 7561]
    print(f"PT conv_post shape: {pt_conv_post.shape}")

    # Load MLX model
    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    F0_mx = mx.array(ref["F0_pred"])
    style_mx = mx.array(ref["style_128"])

    # Get PT generator input
    pt_gen_input = gen_traces["generator_input_ncl"]  # [1, 512, 126]
    gen_input = mx.array(pt_gen_input).transpose(0, 2, 1)  # NLC

    generator = model.decoder.generator

    # Run generator to get conv_post output
    x = gen_input

    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval(source)

    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        x = nn.leaky_relu(x, 0.1)
        x_source = noise_conv(source)
        x_source = noise_res(x_source, style_mx)
        x = up(x)

        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)

        if x_source.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x.shape[1]:
            x_source = x_source[:, : x.shape[1], :]

        x = x + x_source

        xs = None
        for j in range(generator.num_kernels):
            block_idx = i * generator.num_kernels + j
            if block_idx < generator._num_resblocks:
                resblock = getattr(generator, f"resblocks_{block_idx}")
                if xs is None:
                    xs = resblock(x, style_mx)
                else:
                    xs = xs + resblock(x, style_mx)
        if xs is not None:
            x = xs / generator.num_kernels
        mx.eval(x)

    x = nn.leaky_relu(x, 0.1)
    mlx_conv_post = generator.conv_post(x)
    mx.eval(mlx_conv_post)

    # Convert to NCL for comparison
    mlx_conv_post_ncl = np.array(mlx_conv_post.transpose(0, 2, 1))

    print(f"\nMLX conv_post shape: {mlx_conv_post_ncl.shape}")
    print(f"PT conv_post shape: {pt_conv_post.shape}")

    # Overall comparison
    diff = np.abs(mlx_conv_post_ncl - pt_conv_post)
    print("\nOverall diff:")
    print(f"  Max: {diff.max():.4f}")
    print(f"  Mean: {diff.mean():.4f}")
    corr = np.corrcoef(mlx_conv_post_ncl.flatten(), pt_conv_post.flatten())[0, 1]
    print(f"  Correlation: {corr:.6f}")

    # Analyze first 11 channels (log_mag) separately
    mlx_log_mag = mlx_conv_post_ncl[:, :11, :]
    pt_log_mag = pt_conv_post[:, :11, :]

    print("\nLog-magnitude (first 11 channels):")
    print(f"  MLX range: [{mlx_log_mag.min():.4f}, {mlx_log_mag.max():.4f}]")
    print(f"  PT range: [{pt_log_mag.min():.4f}, {pt_log_mag.max():.4f}]")
    log_diff = np.abs(mlx_log_mag - pt_log_mag)
    print(f"  Max diff: {log_diff.max():.4f}")
    print(f"  Mean diff: {log_diff.mean():.4f}")
    log_corr = np.corrcoef(mlx_log_mag.flatten(), pt_log_mag.flatten())[0, 1]
    print(f"  Correlation: {log_corr:.6f}")

    # Clip and convert to magnitude
    mlx_mag = np.exp(np.clip(mlx_log_mag, -10, 10))
    pt_mag = np.exp(np.clip(pt_log_mag, -10, 10))

    print("\nMagnitude (after exp):")
    print(f"  MLX range: [{mlx_mag.min():.4f}, {mlx_mag.max():.4f}]")
    print(f"  PT range: [{pt_mag.min():.4f}, {pt_mag.max():.4f}]")
    mag_diff = np.abs(mlx_mag - pt_mag)
    print(f"  Max diff: {mag_diff.max():.4f}")
    print(f"  Mean diff: {mag_diff.mean():.4f}")
    mag_corr = np.corrcoef(mlx_mag.flatten(), pt_mag.flatten())[0, 1]
    print(f"  Correlation: {mag_corr:.6f}")

    # Phase channels
    mlx_phase = mlx_conv_post_ncl[:, 11:, :]
    pt_phase = pt_conv_post[:, 11:, :]

    print("\nPhase logits (last 11 channels):")
    print(f"  MLX range: [{mlx_phase.min():.4f}, {mlx_phase.max():.4f}]")
    print(f"  PT range: [{pt_phase.min():.4f}, {pt_phase.max():.4f}]")
    phase_diff = np.abs(mlx_phase - pt_phase)
    print(f"  Max diff: {phase_diff.max():.4f}")
    print(f"  Mean diff: {phase_diff.mean():.4f}")
    phase_corr = np.corrcoef(mlx_phase.flatten(), pt_phase.flatten())[0, 1]
    print(f"  Correlation: {phase_corr:.6f}")

    # Now run ISTFT with both to see audio difference
    print("\n=== ISTFT Comparison ===")

    # MLX ISTFT
    mlx_log_mag_mx = mx.clip(mlx_conv_post[:, :, :11], -10, 10)
    mlx_mag_mx = mx.exp(mlx_log_mag_mx)
    mlx_phase_mx = mx.sin(mlx_conv_post[:, :, 11:])
    mlx_audio = generator._istft_synthesis(mlx_mag_mx, mlx_phase_mx)
    mlx_audio = mx.clip(mlx_audio, -1.0, 1.0)
    mx.eval(mlx_audio)

    print("MLX audio from MLX conv_post:")
    print(f"  range: [{float(mx.min(mlx_audio)):.4f}, {float(mx.max(mlx_audio)):.4f}]")
    print(f"  std: {float(mx.std(mlx_audio)):.4f}")

    # ISTFT with PT conv_post using MLX ISTFT
    pt_log_mag_mx = mx.clip(
        mx.array(pt_conv_post).transpose(0, 2, 1)[:, :, :11], -10, 10
    )
    pt_mag_mx = mx.exp(pt_log_mag_mx)
    pt_phase_mx = mx.sin(mx.array(pt_conv_post).transpose(0, 2, 1)[:, :, 11:])
    pt_audio = generator._istft_synthesis(pt_mag_mx, pt_phase_mx)
    pt_audio = mx.clip(pt_audio, -1.0, 1.0)
    mx.eval(pt_audio)

    print("\nMLX audio from PT conv_post:")
    print(f"  range: [{float(mx.min(pt_audio)):.4f}, {float(mx.max(pt_audio)):.4f}]")
    print(f"  std: {float(mx.std(pt_audio)):.4f}")

    # Reference audio
    ref_audio = ref["audio"]
    print("\nReference audio:")
    print(f"  range: [{ref_audio.min():.4f}, {ref_audio.max():.4f}]")
    print(f"  std: {ref_audio.std():.4f}")

    # Compare PT conv_post -> MLX ISTFT with reference
    pt_audio_np = np.array(pt_audio).flatten()
    ref_flat = ref_audio.flatten()
    min_len = min(len(pt_audio_np), len(ref_flat))
    corr_pt = np.corrcoef(pt_audio_np[:min_len], ref_flat[:min_len])[0, 1]
    print(f"\nCorrelation (PT conv_post -> MLX ISTFT vs reference): {corr_pt:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
