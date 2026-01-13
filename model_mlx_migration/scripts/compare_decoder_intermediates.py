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
Compare MLX decoder intermediate tensors against PyTorch reference.
This identifies exactly where the decoder error originates.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np


def compare(name, mlx_val, pt_val):
    """Compare two arrays and return stats."""
    mlx_np = np.array(mlx_val).reshape(-1)
    pt_np = pt_val.reshape(-1)
    min_len = min(len(mlx_np), len(pt_np))
    diff = np.abs(mlx_np[:min_len] - pt_np[:min_len])
    return {
        "name": name,
        "max_diff": diff.max() if len(diff) > 0 else 0,
        "mean_diff": diff.mean() if len(diff) > 0 else 0,
        "mlx_shape": list(mlx_val.shape) if hasattr(mlx_val, 'shape') else [],
        "pt_shape": list(pt_val.shape),
    }


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    if not ref_dir.exists():
        print(f"Reference directory not found: {ref_dir}")
        print("Run: /tmp/kokoro_env/bin/python scripts/export_kokoro_decoder_intermediates.py --out-dir /tmp/kokoro_ref_decoder")
        return 1

    ref = np.load(ref_dir / "tensors.npz")

    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    decoder = mlx_model.decoder

    # Load inputs from reference
    asr_nlc = mx.array(ref["asr_nlc"].astype(np.float32))
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    n = mx.array(ref["N_pred"].astype(np.float32))
    style_128 = mx.array(ref["style_128"].astype(np.float32))

    print("=" * 72)
    print("Decoder Intermediate Tensor Comparison")
    print("=" * 72)

    results = []

    # Step 1: F0/N convolutions
    f0_in = f0[:, :, None]
    n_in = n[:, :, None]

    f0_proc = decoder.f0_conv(f0_in)
    n_proc = decoder.n_conv(n_in)
    mx.eval(f0_proc, n_proc)

    results.append(compare("F0_proc", f0_proc, ref["F0_proc"]))
    results.append(compare("N_proc", n_proc, ref["N_proc"]))

    # Step 2: ASR residual
    asr_res = decoder.asr_res(asr_nlc)
    mx.eval(asr_res)
    results.append(compare("asr_res", asr_res, ref["asr_res"]))

    # Step 3: Initial concatenation
    x = mx.concatenate([asr_nlc, f0_proc, n_proc], axis=-1)
    mx.eval(x)
    results.append(compare("decoder_input_concat", x, ref["decoder_input_concat"]))

    # Step 4: Encode block
    x = decoder.encode(x, style_128)
    mx.eval(x)
    results.append(compare("encode_output", x, ref["encode_output"]))

    # Step 5: Decode blocks
    asr_res_down = asr_res
    decode_blocks = [decoder.decode_0, decoder.decode_1, decoder.decode_2, decoder.decode_3]

    for i, block in enumerate(decode_blocks):
        x_cat = mx.concatenate([x, asr_res_down, f0_proc, n_proc], axis=-1)
        x = block(x_cat, style_128)
        mx.eval(x)

        results.append(compare(f"decode_{i}_output", x, ref[f"decode_{i}_output"]))

        if block.upsample:
            new_len = x.shape[1]
            asr_res_down = mx.repeat(asr_res_down, 2, axis=1)[:, :new_len, :]
            f0_proc = mx.repeat(f0_proc, 2, axis=1)[:, :new_len, :]
            n_proc = mx.repeat(n_proc, 2, axis=1)[:, :new_len, :]

    # Step 6: Generator input
    results.append(compare("generator_input_x", x, ref["generator_input_x"]))

    # Print results
    print("\n" + "-" * 72)
    print(f"{'Tensor':<25} {'Max Diff':>12} {'Mean Diff':>12} {'Shapes'}")
    print("-" * 72)

    for r in results:
        status = "✓" if r["max_diff"] < 0.001 else "✗"
        print(f"{r['name']:<25} {r['max_diff']:>12.6f} {r['mean_diff']:>12.6f} {r['mlx_shape']} vs {r['pt_shape']} {status}")

    # Summary
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)

    max_overall = max(r["max_diff"] for r in results)
    first_bad = next((r for r in results if r["max_diff"] > 0.001), None)

    print(f"Overall max diff: {max_overall:.6f}")
    if first_bad:
        print(f"First layer with error > 0.001: {first_bad['name']} (diff={first_bad['max_diff']:.6f})")
    else:
        print("All layers pass threshold 0.001")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
