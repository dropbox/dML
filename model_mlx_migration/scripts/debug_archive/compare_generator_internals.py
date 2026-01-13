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
Compare MLX Generator internals with PyTorch traces.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.pytorch_to_mlx.converters import KokoroConverter


def correlation(a, b):
    """Compute Pearson correlation coefficient."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    min_len = min(len(a_flat), len(b_flat))
    return np.corrcoef(a_flat[:min_len], b_flat[:min_len])[0, 1]


def main():
    # Load PyTorch traces
    pt_traces = dict(np.load("/tmp/kokoro_ref/generator_internal_traces.npz"))
    gen_traces = dict(np.load("/tmp/kokoro_ref/generator_traces.npz"))
    ref = np.load("/tmp/kokoro_ref/tensors.npz")

    print("=== PyTorch Key Traces ===")
    for k in [
        "m_source_out_0",
        "noise_conv_0_in",
        "noise_conv_0_out",
        "noise_res_0_out",
        "ups_0_out",
        "conv_post_out",
        "final_audio",
    ]:
        if k in pt_traces:
            v = pt_traces[k]
            print(f"  {k}: {v.shape}, range: [{v.min():.4f}, {v.max():.4f}]")

    # Load MLX model
    print("\n=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    gen = model.decoder.generator

    # Get inputs - use generator_input from traces (in NCL format, need to convert to NLC)
    gen_input_ncl = gen_traces["generator_input_ncl"]  # [1, 512, 126] NCL
    gen_input_nlc = np.transpose(gen_input_ncl, (0, 2, 1))  # [1, 126, 512] NLC

    style = ref["style_128"]
    f0 = ref["F0_pred"]

    x_mx = mx.array(gen_input_nlc)
    style_mx = mx.array(style)
    f0_mx = mx.array(f0)

    print(f"\nGenerator input (NLC): {x_mx.shape}")
    print(f"Style: {style_mx.shape}")
    print(f"F0: {f0_mx.shape}")

    print("\n=== Tracing MLX Generator ===")

    # Calculate total upsampling factor
    total_upp = 1
    for r in gen.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= gen.istft_hop_size
    print(f"total_upp = {total_upp}")

    # m_source (SourceModule)
    har_source, noise, uv = gen.m_source(f0_mx, total_upp)
    mx.eval(har_source, noise, uv)
    print(f"\nm_source har output: {har_source.shape}")
    print(
        f"  MLX range: [{float(mx.min(har_source)):.4f}, {float(mx.max(har_source)):.4f}]"
    )
    pt_har = pt_traces["m_source_out_0"]
    print(f"  PT range:  [{pt_har.min():.4f}, {pt_har.max():.4f}]")
    print(f"  PT shape:  {pt_har.shape}")

    # Compare har_source (need to align shapes)
    mlx_har = np.array(har_source)
    # MLX: [B, samples, 1], PT: [B, samples, 1]
    har_corr = correlation(mlx_har, pt_har)
    print(f"  har_source correlation: {har_corr:.6f}")

    # Source STFT
    har_source_1d = har_source.squeeze(-1)
    source = gen._source_stft(har_source_1d)
    mx.eval(source)
    print(f"\n_source_stft output: {source.shape}")
    print(f"  MLX range: [{float(mx.min(source)):.4f}, {float(mx.max(source)):.4f}]")

    pt_stft_input = pt_traces["noise_conv_0_in"]  # [1, 22, 7561] NCL
    print(f"  PT noise_conv input shape: {pt_stft_input.shape}")
    print(f"  PT range: [{pt_stft_input.min():.4f}, {pt_stft_input.max():.4f}]")

    # MLX source is NLC [B, frames, 22], PT is NCL [B, 22, frames]
    mlx_source = np.array(source)  # [1, frames, 22]
    mlx_source_ncl = np.transpose(mlx_source, (0, 2, 1))  # [1, 22, frames]
    print(f"  MLX source (NCL): {mlx_source_ncl.shape}")

    # Align lengths
    min_frames = min(mlx_source_ncl.shape[2], pt_stft_input.shape[2])
    stft_corr = correlation(
        mlx_source_ncl[:, :, :min_frames], pt_stft_input[:, :, :min_frames]
    )
    print(f"  STFT output correlation: {stft_corr:.6f}")

    # noise_conv_0
    noise_conv_0 = gen.noise_convs_0
    x_source_0 = noise_conv_0(source)
    mx.eval(x_source_0)
    print(f"\nnoise_conv_0 output: {x_source_0.shape}")
    print(
        f"  MLX range: [{float(mx.min(x_source_0)):.4f}, {float(mx.max(x_source_0)):.4f}]"
    )

    pt_nc0 = pt_traces["noise_conv_0_out"]  # [1, 256, 1260] NCL
    pt_nc0_nlc = np.transpose(pt_nc0, (0, 2, 1))  # [1, 1260, 256]
    print(f"  PT shape (NLC): {pt_nc0_nlc.shape}")
    print(f"  PT range: [{pt_nc0.min():.4f}, {pt_nc0.max():.4f}]")

    mlx_nc0 = np.array(x_source_0)
    min_len = min(mlx_nc0.shape[1], pt_nc0_nlc.shape[1])
    nc0_corr = correlation(mlx_nc0[:, :min_len, :], pt_nc0_nlc[:, :min_len, :])
    print(f"  noise_conv_0 correlation: {nc0_corr:.6f}")

    # noise_res_0
    noise_res_0 = gen.noise_res_0
    x_source_0_res = noise_res_0(x_source_0, style_mx)
    mx.eval(x_source_0_res)
    print(f"\nnoise_res_0 output: {x_source_0_res.shape}")
    print(
        f"  MLX range: [{float(mx.min(x_source_0_res)):.4f}, {float(mx.max(x_source_0_res)):.4f}]"
    )

    pt_nr0 = pt_traces["noise_res_0_out"]  # NCL
    pt_nr0_nlc = np.transpose(pt_nr0, (0, 2, 1))
    print(f"  PT range: [{pt_nr0.min():.4f}, {pt_nr0.max():.4f}]")

    mlx_nr0 = np.array(x_source_0_res)
    min_len = min(mlx_nr0.shape[1], pt_nr0_nlc.shape[1])
    nr0_corr = correlation(mlx_nr0[:, :min_len, :], pt_nr0_nlc[:, :min_len, :])
    print(f"  noise_res_0 correlation: {nr0_corr:.6f}")

    # ups_0
    x_gen = nn.leaky_relu(x_mx, 0.1)
    ups_0 = gen.ups_0
    x_gen = ups_0(x_gen)
    mx.eval(x_gen)
    print(f"\nups_0 output: {x_gen.shape}")
    print(f"  MLX range: [{float(mx.min(x_gen)):.4f}, {float(mx.max(x_gen)):.4f}]")

    pt_ups0 = pt_traces["ups_0_out"]  # NCL
    pt_ups0_nlc = np.transpose(pt_ups0, (0, 2, 1))
    print(f"  PT range: [{pt_ups0.min():.4f}, {pt_ups0.max():.4f}]")

    mlx_ups0 = np.array(x_gen)
    ups0_corr = correlation(mlx_ups0, pt_ups0_nlc)
    print(f"  ups_0 correlation: {ups0_corr:.6f}")

    # Trace further - after adding source
    if x_source_0_res.shape[1] < x_gen.shape[1]:
        pad_len = x_gen.shape[1] - x_source_0_res.shape[1]
        x_source_padded = mx.pad(x_source_0_res, [(0, 0), (0, pad_len), (0, 0)])
    else:
        x_source_padded = x_source_0_res[:, : x_gen.shape[1], :]

    x_combined = x_gen + x_source_padded
    mx.eval(x_combined)
    print(f"\nAfter ups_0 + x_source_0: {x_combined.shape}")
    print(
        f"  MLX range: [{float(mx.min(x_combined)):.4f}, {float(mx.max(x_combined)):.4f}]"
    )

    # Compare with PT trace
    pt_after_add = pt_traces.get("stage0_after_add", None)
    if pt_after_add is not None:
        print(f"  PT range: [{pt_after_add.min():.4f}, {pt_after_add.max():.4f}]")
        pt_after_add_nlc = np.transpose(pt_after_add, (0, 2, 1))
        add_corr = correlation(np.array(x_combined), pt_after_add_nlc)
        print(f"  After add correlation: {add_corr:.6f}")

    # First resblock
    resblock_0 = gen.resblocks_0
    x_res0 = resblock_0(x_combined, style_mx)
    mx.eval(x_res0)
    print(f"\nresblock_0 output: {x_res0.shape}")
    print(f"  MLX range: [{float(mx.min(x_res0)):.4f}, {float(mx.max(x_res0)):.4f}]")

    pt_res0 = pt_traces.get("resblock_0_out", None)
    if pt_res0 is not None:
        pt_res0_nlc = np.transpose(pt_res0, (0, 2, 1))
        print(f"  PT shape (NLC): {pt_res0_nlc.shape}")
        print(f"  PT range: [{pt_res0.min():.4f}, {pt_res0.max():.4f}]")
        res0_corr = correlation(np.array(x_res0), pt_res0_nlc)
        print(f"  resblock_0 correlation: {res0_corr:.6f}")

    # Run full generator
    print("\n=== Full Generator ===")
    audio = gen(x_mx, style_mx, f0_mx)
    mx.eval(audio)
    mlx_audio = np.array(audio)
    print(
        f"MLX audio: {mlx_audio.shape}, range: [{mlx_audio.min():.4f}, {mlx_audio.max():.4f}]"
    )

    pt_audio = pt_traces["final_audio"]
    print(
        f"PT audio: {pt_audio.shape}, range: [{pt_audio.min():.4f}, {pt_audio.max():.4f}]"
    )

    audio_corr = correlation(mlx_audio.flatten(), pt_audio.flatten())
    print(f"Audio correlation: {audio_corr:.6f}")

    # Check conv_post
    pt_conv_post = pt_traces.get("conv_post_out", None)
    if pt_conv_post is not None:
        print(
            f"\nPT conv_post: {pt_conv_post.shape}, range: [{pt_conv_post.min():.4f}, {pt_conv_post.max():.4f}]"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
