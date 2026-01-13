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
Trace through decoder blocks step by step to find where error diverges.
We use exact PyTorch inputs and compare at each stage.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_seed0")
    ref = np.load(ref_dir / "tensors.npz")

    # Load MLX model
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    decoder = mlx_model.decoder

    # Exact PyTorch inputs
    asr_nlc = mx.array(ref["asr_nlc"].astype(np.float32))
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    n = mx.array(ref["N_pred"].astype(np.float32))
    style_128 = mx.array(ref["style_128"].astype(np.float32))

    print("=" * 72)
    print("Decoder Block Trace")
    print("=" * 72)
    print("Input shapes:")
    print(f"  asr_nlc: {asr_nlc.shape}")
    print(f"  f0: {f0.shape}")
    print(f"  n: {n.shape}")
    print(f"  style_128: {style_128.shape}")

    # Step through decoder manually
    f0_in = f0[:, :, None]  # [B, T, 1]
    n_in = n[:, :, None]    # [B, T, 1]

    # f0_conv and n_conv
    f0_proc = decoder.f0_conv(f0_in)
    n_proc = decoder.n_conv(n_in)
    mx.eval(f0_proc, n_proc)

    print("\nAfter f0_conv/n_conv:")
    print(f"  f0_proc: {f0_proc.shape}, range=[{np.array(f0_proc).min():.4f}, {np.array(f0_proc).max():.4f}]")
    print(f"  n_proc: {n_proc.shape}, range=[{np.array(n_proc).min():.4f}, {np.array(n_proc).max():.4f}]")

    # asr_res projection
    asr_res = decoder.asr_res(asr_nlc)
    mx.eval(asr_res)
    print("\nAfter asr_res:")
    print(f"  asr_res: {asr_res.shape}, range=[{np.array(asr_res).min():.4f}, {np.array(asr_res).max():.4f}]")

    # Initial concatenation
    x = mx.concatenate([asr_nlc, f0_proc, n_proc], axis=-1)
    mx.eval(x)
    print("\nInitial concat (before encode):")
    print(f"  x: {x.shape}, range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")

    # Encode block
    x_encode = decoder.encode(x, style_128)
    mx.eval(x_encode)
    print("\nAfter encode block:")
    print(f"  x: {x_encode.shape}, range=[{np.array(x_encode).min():.4f}, {np.array(x_encode).max():.4f}]")

    # Decode blocks
    x = x_encode
    asr_res_down = asr_res
    decode_blocks = [
        ("decode_0", decoder.decode_0),
        ("decode_1", decoder.decode_1),
        ("decode_2", decoder.decode_2),
        ("decode_3", decoder.decode_3),
    ]

    for name, block in decode_blocks:
        # Concatenate residuals
        x_concat = mx.concatenate([x, asr_res_down, f0_proc, n_proc], axis=-1)
        x = block(x_concat, style_128)
        mx.eval(x)

        x_np = np.array(x)
        print(f"\nAfter {name}:")
        print(f"  x: {x.shape}, range=[{x_np.min():.4f}, {x_np.max():.4f}], mean={x_np.mean():.4f}")

        # After upsampling block, adjust residual lengths
        if block.upsample:
            new_len = x.shape[1]
            asr_res_down = mx.repeat(asr_res_down, 2, axis=1)[:, :new_len, :]
            f0_proc = mx.repeat(f0_proc, 2, axis=1)[:, :new_len, :]
            n_proc = mx.repeat(n_proc, 2, axis=1)[:, :new_len, :]
            print(f"    (upsampled to len={new_len})")

    # Now x is the input to Generator
    print("\nGenerator input x:")
    print(f"  shape: {x.shape}")
    x_np = np.array(x)
    print(f"  range: [{x_np.min():.4f}, {x_np.max():.4f}]")
    print(f"  mean: {x_np.mean():.4f}, std: {x_np.std():.4f}")

    # Check if we have PyTorch decoder-to-generator reference
    if "decoder_gen_x" in ref:
        pt_gen_x = ref["decoder_gen_x"].astype(np.float32)
        print("\nPyTorch decoder_gen_x available:")
        print(f"  shape: {pt_gen_x.shape}")
        print(f"  range: [{pt_gen_x.min():.4f}, {pt_gen_x.max():.4f}]")

        # Compare
        min_len = min(x_np.shape[1], pt_gen_x.shape[1])
        diff = np.abs(x_np[:, :min_len, :] - pt_gen_x[:, :min_len, :])
        print("\nDecoder output comparison:")
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")

        # Find where the largest differences are
        flat_diff = diff.reshape(-1)
        top_idx = np.argsort(flat_diff)[-10:][::-1]
        print("\nTop 10 largest diffs:")
        for idx in top_idx:
            loc = np.unravel_index(idx, diff.shape)
            print(f"    {loc}: diff={flat_diff[idx]:.6f}, mlx={x_np[loc]:.4f}, pt={pt_gen_x[loc]:.4f}")
    else:
        print("\n(No PyTorch decoder_gen_x reference available)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
