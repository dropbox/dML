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
import mlx.nn as nn
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
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    status = "PASS" if max_diff < threshold else "FAIL"
    print(f"{name:<30} max={max_diff:.6f} mean={mean_diff:.6f} {status}")
    return max_diff


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    generator = mlx_model.decoder.generator

    print("=" * 72)
    print("Generator Detailed Comparison")
    print("=" * 72)

    # Load inputs
    x_nlc = mx.array(ref["generator_input_x"].astype(np.float32))  # [1, 126, 512]
    f0 = mx.array(ref["F0_pred"].astype(np.float32))  # [1, 126]
    style = mx.array(ref["style_128"].astype(np.float32))  # [1, 128]

    print(f"Input x shape: {x_nlc.shape}")
    print(f"F0 shape: {f0.shape}")
    print(f"Style shape: {style.shape}")

    compare("gen_input_x", x_nlc, gen_ref["gen_input_x"])

    # Run MLX generator step by step
    x = x_nlc  # Keep in NLC format for MLX

    # SourceModule
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    har_source, noise, uv = generator.m_source(f0, total_upp)
    mx.eval(har_source, noise, uv)

    print("\n--- SourceModule ---")
    compare("har_source", har_source, gen_ref["gen_har_source"])

    # STFT of harmonic source
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval(source)
    compare("har (STFT output)", source, gen_ref["gen_har"])

    # Generator main loop
    print("\n--- Generator Loop ---")
    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        # leaky_relu before ups
        x = nn.leaky_relu(x, 0.1)
        mx.eval(x)
        compare(f"pre_lrelu_{i}", x, gen_ref[f"gen_pre_lrelu_{i}"])

        # noise_convs(source) - before ups
        x_source = noise_conv(source)
        mx.eval(x_source)
        compare(f"noise_convs_{i}", x_source, gen_ref[f"gen_noise_convs_{i}_out"])

        # noise_res
        x_source = noise_res(x_source, style)
        mx.eval(x_source)
        compare(f"noise_res_{i}", x_source, gen_ref[f"gen_noise_res_{i}_out"])

        # ups
        x = up(x)
        mx.eval(x)
        compare(f"ups_{i}", x, gen_ref[f"gen_ups_{i}_out"])

        # reflection pad at last stage
        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)
            mx.eval(x)

        # add
        x = x + x_source
        mx.eval(x)
        compare(f"after_add_{i}", x, gen_ref[f"gen_after_add_{i}"])

        # resblocks
        xs = None
        for j in range(generator.num_kernels):
            block_idx = i * generator.num_kernels + j
            if block_idx < generator._num_resblocks:
                resblock = getattr(generator, f"resblocks_{block_idx}")
                if xs is None:
                    xs = resblock(x, style)
                else:
                    xs = xs + resblock(x, style)
        if xs is not None:
            x = xs / generator.num_kernels
        mx.eval(x)
        compare(f"resblocks_{i}", x, gen_ref[f"gen_resblocks_{i}_out"])

    # Final leaky_relu
    x = nn.leaky_relu(x)
    mx.eval(x)
    compare("final_lrelu", x, gen_ref["gen_final_lrelu"])

    # conv_post
    x = generator.conv_post(x)
    mx.eval(x)
    compare("conv_post", x, gen_ref["gen_conv_post_out"])

    # spec/phase split
    n_bins = generator.post_n_fft // 2 + 1
    raw_spec = x[..., :n_bins]
    raw_phase = x[..., n_bins:]

    compare("raw_spec", raw_spec, gen_ref["gen_raw_spec"])
    compare("raw_phase", raw_phase, gen_ref["gen_raw_phase"])

    # exp/sin
    spec = mx.exp(raw_spec)
    phase = mx.sin(raw_phase)
    mx.eval(spec, phase)

    compare("spec (after exp)", spec, gen_ref["gen_spec"])
    compare("phase (after sin)", phase, gen_ref["gen_phase"])

    # Run full generator
    print("\n--- Full Generator ---")
    audio = generator(x_nlc, style, f0)
    mx.eval(audio)

    pt_audio = gen_ref["gen_audio"]
    mlx_audio = np.array(audio)
    compare("audio", mlx_audio, pt_audio)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
