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
Compare MLX vs PyTorch generator intermediate tensors.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np


def compare(name, mlx_val, pt_val, threshold=0.001):
    """Compare and print differences."""
    mlx_np = np.array(mlx_val) if hasattr(mlx_val, 'tolist') else mlx_val
    pt_np = pt_val.astype(np.float32) if pt_val.dtype != np.float32 else pt_val

    # Handle shape mismatch by trimming to common size
    if mlx_np.shape != pt_np.shape:
        min_shape = tuple(min(m, p) for m, p in zip(mlx_np.shape, pt_np.shape))
        slices = tuple(slice(0, s) for s in min_shape)
        mlx_np = mlx_np[slices]
        pt_np = pt_np[slices]

    diff = np.abs(mlx_np - pt_np)
    max_diff = diff.max()
    mean_diff = diff.mean()

    status = "PASS" if max_diff < threshold else "FAIL"
    print(f"{name:<30} max={max_diff:.6f} mean={mean_diff:.6f} {status}")
    return max_diff


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")

    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    generator = mlx_model.decoder.generator

    print("=" * 72)
    print("Generator Intermediate Comparison")
    print("=" * 72)

    # Load inputs
    generator_x = mx.array(ref["generator_input_x"].astype(np.float32))
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    style = mx.array(ref["style_128"].astype(np.float32))

    print(f"Generator input x shape: {generator_x.shape}")
    print(f"F0 shape: {f0.shape}")

    # Step 1: F0 upsampling
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size
    print(f"Total upsample factor: {total_upp}")

    # MLX SourceModule
    har_source, noise, uv = generator.m_source(f0, total_upp)
    mx.eval(har_source, noise, uv)

    print("\n--- SourceModule outputs ---")
    compare("har_source", har_source, ref["gen_har_source"])
    compare("uv", uv, ref["gen_uv"])

    # Step 2: STFT of harmonic source
    har_source_2d = har_source.squeeze(-1)  # [B, T]
    har_spec, har_phase = generator.stft.transform(har_source_2d)
    mx.eval(har_spec, har_phase)

    print("\n--- STFT outputs ---")
    compare("har_spec", har_spec, ref["gen_har_spec"])
    compare("har_phase", har_phase, ref["gen_har_phase"])

    # har = concat(spec, phase)
    har = mx.concatenate([har_spec, har_phase], axis=1)
    mx.eval(har)
    compare("har (concat)", har, ref["gen_har"])

    # Step 3: Run through generator ups and resblocks
    print("\n--- Generator forward ---")
    # x is [B, T, C] (NLC), PyTorch uses NCL
    x_ncl = generator_x.transpose(0, 2, 1)  # [B, C, T]
    mx.eval(x_ncl)
    print(f"x (NCL) shape: {x_ncl.shape}")

    # Initial convolution (conv_pre)
    x_out = generator.conv_pre(generator_x)
    mx.eval(x_out)
    print(f"After conv_pre: {x_out.shape}")

    # Get har_full for upsampling
    # har is [B, 2*n_fft//2+2, T_stft]
    har_full = har

    # Run through ups and resblocks
    for i in range(len(generator.config.istft_upsample_rates)):
        ups = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")
        resblocks = [getattr(generator, f"resblocks_{i}_{j}") for j in range(3)]

        # Upsample x
        x_out = ups(x_out)
        mx.eval(x_out)

        # Upsample har
        har_up = mx.repeat(har_full, 2**i * generator.config.istft_upsample_rates[0] // generator.config.istft_upsample_rates[i], axis=-1)
        x_len = x_out.shape[1]  # NLC format
        har_up = har_up[:, :, :x_len]

        # Process harmonic and noise
        har_out = noise_conv(har_up.transpose(0, 2, 1))  # NLC
        har_out = noise_res(har_out, style)
        mx.eval(har_out)

        # Add to x
        x_out = x_out + har_out

        # Resblocks
        for rb in resblocks:
            x_out = rb(x_out)
            mx.eval(x_out)

        print(f"After stage {i}: {x_out.shape}")

    # conv_post
    x_out = generator.conv_post(x_out)
    mx.eval(x_out)
    print(f"After conv_post: {x_out.shape}")

    # ISTFT
    spec = x_out[:, :, :generator.n_fft // 2 + 1]
    phase = x_out[:, :, generator.n_fft // 2 + 1:]
    audio = generator.istft.inverse(spec, phase)
    mx.eval(audio)
    print(f"Audio shape: {audio.shape}")

    # Compare final audio
    print("\n--- Final audio ---")
    pt_audio = ref["audio"]
    mlx_audio = np.array(audio).squeeze()
    min_len = min(len(mlx_audio), len(pt_audio))
    diff = np.abs(mlx_audio[:min_len] - pt_audio[:min_len])
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
