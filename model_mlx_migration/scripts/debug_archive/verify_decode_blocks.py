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
Verify decode blocks output matches PyTorch generator_traces.npz.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load references
    ref = np.load("/tmp/kokoro_ref/tensors.npz")
    gen_traces = np.load("/tmp/kokoro_ref/generator_traces.npz")

    print("=== PyTorch decode block outputs ===")
    for i in range(4):
        key = f"decode_{i}_out"
        if key in gen_traces:
            v = gen_traces[key]
            print(f"{key}: {v.shape}, range [{v.min():.4f}, {v.max():.4f}]")

    if "generator_input_ncl" in gen_traces:
        gen_in = gen_traces["generator_input_ncl"]
        print(
            f"generator_input_ncl: {gen_in.shape}, range [{gen_in.min():.4f}, {gen_in.max():.4f}]"
        )

    print("\n=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Run decoder with reference inputs
    asr_nlc = mx.array(ref["asr_ncl"]).transpose(0, 2, 1)
    F0_mx = mx.array(ref["F0_pred"])
    N_mx = mx.array(ref["N_pred"])
    style_mx = mx.array(ref["style_128"])

    # Run full decoder
    audio = model.decoder(asr_nlc, F0_mx, N_mx, style_mx)
    mx.eval(audio)

    print(
        f"\nMLX audio: {audio.shape}, range [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]"
    )

    # Compare with reference audio
    ref_audio = ref["audio"]
    mlx_audio = np.array(audio).flatten()
    ref_flat = ref_audio.flatten()
    min_len = min(len(mlx_audio), len(ref_flat))

    corr = np.corrcoef(mlx_audio[:min_len], ref_flat[:min_len])[0, 1]
    print(f"Correlation with reference: {corr:.6f}")

    # Let me also run just the generator with PT decode output
    print("\n=== Testing Generator with PT decode output ===")
    gen_input = mx.array(gen_traces["generator_input_ncl"]).transpose(0, 2, 1)  # NLC
    mx.eval(gen_input)

    # Call generator directly
    generator = model.decoder.generator
    gen_audio = generator(gen_input, style_mx, F0_mx)
    mx.eval(gen_audio)
    print(
        f"MLX generator audio: {gen_audio.shape}, range [{float(mx.min(gen_audio)):.4f}, {float(mx.max(gen_audio)):.4f}]"
    )

    # Compare
    gen_ref = gen_traces["generator_audio"].flatten()
    gen_mlx = np.array(gen_audio).flatten()
    min_len = min(len(gen_mlx), len(gen_ref))
    corr = np.corrcoef(gen_mlx[:min_len], gen_ref[:min_len])[0, 1]
    print(f"Generator correlation: {corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
