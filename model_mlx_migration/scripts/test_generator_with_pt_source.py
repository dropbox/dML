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

"""Test generator with PyTorch's har_source to isolate the error source."""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    generator = mlx_model.decoder.generator

    # Get inputs
    x_nlc = mx.array(ref["generator_input_x"].astype(np.float32))
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    style = mx.array(ref["style_128"].astype(np.float32))

    # Get PyTorch's har_source (kept for reference, could be used for comparison)
    pt_har_source = gen_ref["gen_har_source"]  # [B, samples, 1]
    _ = mx.array(pt_har_source.astype(np.float32))  # Convert to MLX format

    # Option 1: Test with MLX har_source
    print("=== Test 1: MLX har_source ===")
    audio_mlx = generator(x_nlc, style, f0)
    mx.eval(audio_mlx)

    pt_audio = gen_ref["gen_audio"]
    mlx_audio_np = np.array(audio_mlx)

    diff = np.abs(mlx_audio_np - pt_audio)
    print(f"Max abs error: {diff.max():.6f}")
    print(f"Mean abs error: {diff.mean():.6f}")

    # Option 2: Use debug override with PyTorch's har_source
    print("\n=== Test 2: PyTorch har_source ===")
    # Compute what source would be after STFT
    pt_har = gen_ref["gen_har"]  # [B, frames, 22]
    pt_source = mx.array(pt_har.astype(np.float32))

    # Call generator with debug override
    debug_overrides = {"source": pt_source}
    audio_with_pt_source = generator(x_nlc, style, f0, _debug_overrides=debug_overrides)
    mx.eval(audio_with_pt_source)

    mlx_audio_np2 = np.array(audio_with_pt_source)
    diff2 = np.abs(mlx_audio_np2 - pt_audio)
    print(f"Max abs error: {diff2.max():.6f}")
    print(f"Mean abs error: {diff2.mean():.6f}")

    # Correlation
    corr = np.corrcoef(mlx_audio_np2.flatten(), pt_audio.flatten())[0, 1]
    print(f"Correlation: {corr:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
