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
Debug script to pinpoint decoder divergence between MLX and PyTorch.

Exports intermediate tensors from MLX decoder and compares to PyTorch reference.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def compare(name: str, a: np.ndarray, b: np.ndarray, tolerance: float = 0.001) -> dict:
    """Compare two tensors and return statistics."""
    min_len = min(a.size, b.size)
    a1 = a.reshape(-1)[:min_len].astype(np.float32)
    b1 = b.reshape(-1)[:min_len].astype(np.float32)
    diff = np.abs(a1 - b1)

    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0

    # Find where max diff occurs
    max_idx = int(np.argmax(diff)) if diff.size else 0

    status = "PASS" if max_diff <= tolerance else "FAIL"

    return {
        "name": name,
        "status": status,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "max_idx": max_idx,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "a_at_max": float(a1[max_idx]) if diff.size else 0.0,
        "b_at_max": float(b1[max_idx]) if diff.size else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", type=Path, required=True)
    args = parser.parse_args()

    # Load reference
    ref = np.load(args.ref_dir / "tensors.npz")
    metadata = json.loads((args.ref_dir / "metadata.json").read_text())

    print("=" * 72)
    print("Decoder Parity Debug")
    print("=" * 72)
    print(f"Text: {metadata.get('text')!r}")
    print()

    # Load MLX model
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Get inputs
    asr_nlc = mx.array(ref["asr_nlc"].astype(np.float32))
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    n = mx.array(ref["N_pred"].astype(np.float32))
    style_128 = mx.array(ref["style_128"].astype(np.float32))

    print("Input shapes:")
    print(f"  asr_nlc: {asr_nlc.shape}")
    print(f"  f0: {f0.shape}")
    print(f"  n: {n.shape}")
    print(f"  style_128: {style_128.shape}")
    print()

    # Access generator (decoder.generator)
    generator = mlx_model.decoder.generator

    # Step 1: Run F0/N convolutions (same as decoder.forward initial steps)
    decoder = mlx_model.decoder

    # f0_conv: [batch, T] -> [batch, T//2, 1]
    f0_2d = f0[:, :, None]  # [1, 286, 1]
    f0_out = decoder.f0_conv(f0_2d)  # [1, 143, 1]
    mx.eval(f0_out)

    # n_conv: [batch, T] -> [batch, T//2, 1]
    n_2d = n[:, :, None]
    n_out = decoder.n_conv(n_2d)
    mx.eval(n_out)

    print("After F0/N conv:")
    print(f"  f0_out: {f0_out.shape}")
    print(f"  n_out: {n_out.shape}")

    # Run full decoder to get audio
    audio_mlx = mlx_model.decoder(asr_nlc, f0, n, style_128)
    mx.eval(audio_mlx)
    audio_mlx_np = np.array(audio_mlx).reshape(-1)
    audio_ref = ref["audio"].astype(np.float32).reshape(-1)

    result = compare("final_audio", audio_ref, audio_mlx_np, tolerance=0.1)
    print()
    print("Final audio comparison:")
    print(f"  Status: {result['status']}")
    print(f"  Max diff: {result['max_diff']:.6f}")
    print(f"  Mean diff: {result['mean_diff']:.6f}")
    print(f"  Max diff at idx {result['max_idx']}")
    print(f"  Ref value at max: {result['a_at_max']:.6f}")
    print(f"  MLX value at max: {result['b_at_max']:.6f}")
    print()

    # Now let's trace the ISTFT separately
    # We need to get the conv_post output and compare ISTFT directly

    # Run with debugging - extract intermediate tensors
    # This requires modifying the generator temporarily or adding hooks
    # For now, let's just check if the issue is in ISTFT by testing with simple input

    print("=" * 72)
    print("ISTFT Test with synthetic input")
    print("=" * 72)

    # Create simple test: exp(0) magnitude, zero phase -> should give DC signal
    test_frames = 100
    n_bins = 11

    # Test 1: All zeros -> should give near-zero output
    mag_test = mx.ones((1, test_frames, n_bins)) * 0.01
    phase_test = mx.zeros((1, test_frames, n_bins))

    audio_test = generator._istft_synthesis(mag_test, phase_test)
    mx.eval(audio_test)
    print("ISTFT test (small mag, zero phase):")
    print(
        f"  Input mag: {mag_test.shape}, range [{float(mag_test.min()):.3f}, {float(mag_test.max()):.3f}]"
    )
    print(
        f"  Output: {audio_test.shape}, range [{float(audio_test.min()):.6f}, {float(audio_test.max()):.6f}]"
    )
    print(f"  Output RMS: {float(mx.sqrt(mx.mean(audio_test**2))):.6f}")

    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
