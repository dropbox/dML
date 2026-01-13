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
Compare MLX decoder blocks with PyTorch reference traces.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.pytorch_to_mlx.converters import KokoroConverter


def correlation(a, b):
    """Compute Pearson correlation coefficient."""
    a_flat = a.flatten()
    b_flat = b.flatten()[: len(a_flat)]  # Align lengths
    return np.corrcoef(a_flat, b_flat)[0, 1]


def main():
    # Load reference traces
    ref_path = Path("/tmp/kokoro_ref/generator_traces.npz")
    tensors_path = Path("/tmp/kokoro_ref/tensors.npz")
    if not ref_path.exists():
        print("ERROR: Run export_generator_intermediates.py first")
        return 1

    pt_traces = dict(np.load(ref_path))
    ref = np.load(tensors_path)

    print("=== PyTorch Reference Traces ===")
    for k in sorted(pt_traces.keys()):
        v = pt_traces[k]
        print(f"  {k}: {v.shape}, range: [{v.min():.4f}, {v.max():.4f}]")

    # Load MLX model
    print("\n=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Get inputs
    asr_ncl = ref["asr_ncl"]  # [1, 512, 63] NCL
    F0_pred = ref["F0_pred"]  # [1, 126]
    N_pred = ref["N_pred"]  # [1, 126]
    style_128 = ref["style_128"]  # [1, 128]

    # Convert to MLX (NLC format for MLX)
    asr_nlc_mx = mx.transpose(mx.array(asr_ncl), (0, 2, 1))  # [1, 63, 512]
    F0_mx = mx.array(F0_pred)
    N_mx = mx.array(N_pred)
    style_mx = mx.array(style_128)

    decoder = model.decoder

    print("\n=== Tracing MLX Decoder ===")

    # F0 and N conv
    f0_in = F0_mx[:, :, None]  # [1, 126, 1]
    f0_proc = decoder.f0_conv(f0_in)
    mx.eval(f0_proc)
    print(
        f"f0_conv out: {f0_proc.shape}, range: [{float(mx.min(f0_proc)):.4f}, {float(mx.max(f0_proc)):.4f}]"
    )

    # Compare with PyTorch - note PyTorch is NCL [1, 1, 63]
    pt_f0 = pt_traces["f0_conv_out"]  # [1, 1, 63] NCL
    pt_f0_nlc = np.transpose(pt_f0, (0, 2, 1))  # [1, 63, 1]
    mlx_f0 = np.array(f0_proc)
    print(
        f"PT f0_conv: {pt_f0_nlc.shape}, range: [{pt_f0_nlc.min():.4f}, {pt_f0_nlc.max():.4f}]"
    )
    print(f"f0_conv correlation: {correlation(mlx_f0, pt_f0_nlc):.6f}")

    n_in = N_mx[:, :, None]
    n_proc = decoder.n_conv(n_in)
    mx.eval(n_proc)
    print(
        f"n_conv out: {n_proc.shape}, range: [{float(mx.min(n_proc)):.4f}, {float(mx.max(n_proc)):.4f}]"
    )

    # ASR res
    asr_res = decoder.asr_res(asr_nlc_mx)
    mx.eval(asr_res)
    print(
        f"asr_res out: {asr_res.shape}, range: [{float(mx.min(asr_res)):.4f}, {float(mx.max(asr_res)):.4f}]"
    )

    pt_asr_res = pt_traces["asr_res_out"]  # [1, 64, 63] NCL
    pt_asr_res_nlc = np.transpose(pt_asr_res, (0, 2, 1))
    mlx_asr_res = np.array(asr_res)
    print(f"asr_res correlation: {correlation(mlx_asr_res, pt_asr_res_nlc):.6f}")

    # Length alignment
    asr_len = asr_nlc_mx.shape[1]  # 63
    f0_len = f0_proc.shape[1]  # 63

    if f0_len > asr_len:
        scale = f0_len // asr_len
        asr_down = mx.repeat(asr_nlc_mx, scale, axis=1)[:, :f0_len, :]
        asr_res_down = mx.repeat(asr_res, scale, axis=1)[:, :f0_len, :]
    elif asr_len > f0_len:
        stride = asr_len // f0_len
        asr_down = asr_nlc_mx[:, ::stride, :][:, :f0_len, :]
        asr_res_down = asr_res[:, ::stride, :][:, :f0_len, :]
    else:
        asr_down = asr_nlc_mx
        asr_res_down = asr_res

    # Concatenate
    x = mx.concatenate([asr_down, f0_proc, n_proc], axis=-1)
    mx.eval(x)
    print(
        f"\nConcat input: {x.shape}, range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
    )

    pt_concat = pt_traces["concat_input"]  # [1, 514, 63] NCL
    pt_concat_nlc = np.transpose(pt_concat, (0, 2, 1))
    mlx_concat = np.array(x)
    print(f"Concat correlation: {correlation(mlx_concat, pt_concat_nlc):.6f}")

    # Encode
    x = decoder.encode(x, style_mx)
    mx.eval(x)
    print(
        f"\nEncode out: {x.shape}, range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
    )

    pt_encode = pt_traces["encode_out"]  # [1, 1024, 63] NCL
    pt_encode_nlc = np.transpose(pt_encode, (0, 2, 1))
    mlx_encode = np.array(x)
    print(
        f"PT Encode: {pt_encode_nlc.shape}, range: [{pt_encode_nlc.min():.4f}, {pt_encode_nlc.max():.4f}]"
    )
    print(f"Encode correlation: {correlation(mlx_encode, pt_encode_nlc):.6f}")

    # Decode blocks
    decode_blocks = [
        decoder.decode_0,
        decoder.decode_1,
        decoder.decode_2,
        decoder.decode_3,
    ]
    for i, blk in enumerate(decode_blocks):
        # Concatenate residuals
        x_with_res = mx.concatenate([x, asr_res_down, f0_proc, n_proc], axis=-1)
        x = blk(x_with_res, style_mx)
        mx.eval(x)
        print(
            f"\nDecode {i} out: {x.shape}, range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
        )

        pt_decode = pt_traces[f"decode_{i}_out"]  # NCL
        pt_decode_nlc = np.transpose(pt_decode, (0, 2, 1))
        mlx_decode = np.array(x)
        print(
            f"PT Decode {i}: {pt_decode_nlc.shape}, range: [{pt_decode_nlc.min():.4f}, {pt_decode_nlc.max():.4f}]"
        )
        print(f"Decode {i} correlation: {correlation(mlx_decode, pt_decode_nlc):.6f}")

        # Upsample residuals if needed
        if hasattr(blk, "upsample") and blk.upsample:
            new_len = x.shape[1]
            asr_res_down = mx.repeat(asr_res_down, 2, axis=1)[:, :new_len, :]
            f0_proc = mx.repeat(f0_proc, 2, axis=1)[:, :new_len, :]
            n_proc = mx.repeat(n_proc, 2, axis=1)[:, :new_len, :]

    # Compare generator input
    print("\n=== Generator Input ===")
    mlx_gen_input = np.array(x)
    pt_gen_input = pt_traces["generator_input_ncl"]  # [1, 512, 126] NCL
    pt_gen_input_nlc = np.transpose(pt_gen_input, (0, 2, 1))
    print(
        f"MLX gen input: {mlx_gen_input.shape}, range: [{mlx_gen_input.min():.4f}, {mlx_gen_input.max():.4f}]"
    )
    print(
        f"PT gen input: {pt_gen_input_nlc.shape}, range: [{pt_gen_input_nlc.min():.4f}, {pt_gen_input_nlc.max():.4f}]"
    )
    print(
        f"Generator input correlation: {correlation(mlx_gen_input, pt_gen_input_nlc):.6f}"
    )

    # Now call Generator
    print("\n=== Generator Output ===")
    audio = decoder.generator(x, style_mx, F0_mx)
    mx.eval(audio)
    mlx_audio = np.array(audio)
    print(
        f"MLX audio: {mlx_audio.shape}, range: [{mlx_audio.min():.4f}, {mlx_audio.max():.4f}]"
    )

    pt_audio = pt_traces["generator_audio"]
    print(
        f"PT audio: {pt_audio.shape}, range: [{pt_audio.min():.4f}, {pt_audio.max():.4f}]"
    )
    print(
        f"Audio correlation: {correlation(mlx_audio.flatten(), pt_audio.flatten()):.6f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
