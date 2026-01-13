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
Debug where the Generator output is too large (causing clipping).
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load MLX model
    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Load reference
    ref = np.load("/tmp/kokoro_ref/tensors.npz")
    asr_ncl = ref["asr_ncl"]
    F0_pred = ref["F0_pred"]
    N_pred = ref["N_pred"]
    style_128 = ref["style_128"]

    asr_nlc = mx.array(asr_ncl).transpose(0, 2, 1)
    F0_mx = mx.array(F0_pred)
    N_mx = mx.array(N_pred)
    style_mx = mx.array(style_128)

    decoder = model.decoder
    generator = decoder.generator

    # Run decode blocks first (these process asr + F0 + N)
    print("\n=== Running Decode Blocks ===")

    # The decoder forward is: decode_blocks -> generator
    # Let's trace through the decoder manually

    # Check decoder structure
    print("Decoder attributes:")
    for name in dir(decoder):
        if not name.startswith("_"):
            attr = getattr(decoder, name, None)
            if hasattr(attr, "parameters") or isinstance(attr, mx.array):
                print(f"  {name}: {type(attr).__name__}")

    # Run full decode
    # The Decoder should concatenate asr + F0 + N and pass through blocks
    # Then feed to generator

    # Run the actual decoder forward
    print("\n=== Running Full Decoder Forward ===")
    audio = decoder(asr_nlc, F0_mx, N_mx, style_mx)
    mx.eval(audio)

    print(f"Raw audio shape: {audio.shape}")
    print(f"Raw audio range: [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]")
    print(f"Raw audio std: {float(mx.std(audio)):.4f}")

    # Check how much is clipping
    audio_np = np.array(audio).flatten()
    n_clip_high = np.sum(audio_np >= 0.999)
    n_clip_low = np.sum(audio_np <= -0.999)
    print(f"Clipping: {n_clip_high} high, {n_clip_low} low out of {len(audio_np)}")
    print(f"Clipping rate: {(n_clip_high + n_clip_low) / len(audio_np) * 100:.1f}%")

    # Now let's check the Generator directly with the same input
    print("\n=== Running Generator Directly ===")

    # We need to get the generator input from decode blocks
    # Let's trace through decode_blocks manually

    # Check decode blocks structure
    if hasattr(decoder, "asr_res"):
        print("Has asr_res")
    if hasattr(decoder, "decode_blocks"):
        print(f"decode_blocks type: {type(decoder.decode_blocks)}")
    elif hasattr(decoder, "decode"):
        print("Has decode (legacy name)")

    # For now, let's just trace the Generator with a scaled-down input
    print("\n=== Testing Generator with Scaled Input ===")

    # Use reference generator input if available
    gen_traces_path = Path("/tmp/kokoro_ref/generator_traces.npz")
    if gen_traces_path.exists():
        gen_traces = np.load(gen_traces_path)
        if "generator_input_ncl" in gen_traces:
            pt_gen_input = gen_traces["generator_input_ncl"]
            print(
                f"PT generator input range: [{pt_gen_input.min():.4f}, {pt_gen_input.max():.4f}]"
            )
            print(f"PT generator input std: {pt_gen_input.std():.4f}")

            # Convert to MLX
            gen_input = mx.array(pt_gen_input).transpose(0, 2, 1)  # NLC

            # Run generator
            gen_audio = generator(gen_input, style_mx, F0_mx)
            mx.eval(gen_audio)

            print(
                f"\nGenerator output range: [{float(mx.min(gen_audio)):.4f}, {float(mx.max(gen_audio)):.4f}]"
            )
            print(f"Generator output std: {float(mx.std(gen_audio)):.4f}")

            gen_np = np.array(gen_audio).flatten()
            n_clip = np.sum(np.abs(gen_np) >= 0.999)
            print(f"Clipping rate: {n_clip / len(gen_np) * 100:.1f}%")

    # Compare with PT reference audio
    ref_audio = ref["audio"]
    print(f"\nReference audio range: [{ref_audio.min():.4f}, {ref_audio.max():.4f}]")
    print(f"Reference audio std: {ref_audio.std():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
