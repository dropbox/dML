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
Export Generator intermediate tensors from PyTorch kokoro model.

Run this in the kokoro Python environment (separate from MLX):
  python scripts/export_generator_intermediates.py

Outputs:
  /tmp/kokoro_ref/generator_traces.npz
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np


def main():
    # First load existing reference tensors
    ref_path = Path("/tmp/kokoro_ref/tensors.npz")
    if not ref_path.exists():
        print("ERROR: Run export_kokoro_reference.py first")
        return 1

    ref = np.load(ref_path)
    json.loads((ref_path.parent / "metadata.json").read_text())

    import torch
    import torch.nn.functional as F

    try:
        from kokoro import KModel, KPipeline  # noqa: F401
    except ImportError:
        print("ERROR: kokoro package not found. Run in kokoro environment.")
        return 1

    print("=== Loading Kokoro Model ===")
    model = KModel().to("cpu").eval()

    # Get reference inputs
    asr_ncl = ref["asr_ncl"]  # [1, 512, 63]
    F0_pred = ref["F0_pred"]  # [1, 126]
    N_pred = ref["N_pred"]  # [1, 126]
    style_128 = ref["style_128"]  # [1, 128]

    # Convert to PyTorch tensors
    asr_pt = torch.tensor(asr_ncl)  # Keep NCL format for PyTorch
    f0_pt = torch.tensor(F0_pred)
    n_pt = torch.tensor(N_pred)
    style_pt = torch.tensor(style_128)

    print(f"asr shape: {asr_pt.shape}")
    print(f"F0 shape: {f0_pt.shape}")
    print(f"N shape: {n_pt.shape}")
    print(f"style shape: {style_pt.shape}")

    traces = {}

    # Access the actual decoder and generator
    decoder = model.decoder

    print("\n=== Tracing through Decoder ===")

    with torch.no_grad():
        # The PyTorch decoder expects NCL format
        # Let's trace through the decoder's forward pass

        # Store original F0 for generator
        f0_orig = f0_pt

        # Process F0 and N through convs
        # PyTorch: F0_conv operates on [B, 1, L] format
        f0_in = f0_pt.unsqueeze(1)  # [1, 1, 126]
        n_in = n_pt.unsqueeze(1)

        # Trace F0 conv
        f0_proc = decoder.F0_conv(f0_in)  # [1, 1, 63]
        n_proc = decoder.N_conv(n_in)

        traces["f0_conv_out"] = f0_proc.numpy()
        traces["n_conv_out"] = n_proc.numpy()
        print(
            f"f0_conv out: {f0_proc.shape}, range: [{f0_proc.min():.4f}, {f0_proc.max():.4f}]"
        )
        print(
            f"n_conv out: {n_proc.shape}, range: [{n_proc.min():.4f}, {n_proc.max():.4f}]"
        )

        # ASR residual - decoder.asr_res is Sequential(Conv1d)
        asr_res = decoder.asr_res(asr_pt)  # [1, 64, 63]
        traces["asr_res_out"] = asr_res.numpy()
        print(f"asr_res out: {asr_res.shape}")

        # Concatenate: asr + F0 + N
        # x = torch.cat([asr_pt, f0_proc, n_proc], dim=1)  # [1, 514, 63]
        # Actually decoder does length alignment first
        # Let's just call the full decoder

        print("\n=== Running full Decoder forward ===")
        audio = decoder(asr_pt, f0_pt, n_pt, style_pt)
        traces["decoder_audio"] = audio.numpy()
        print(
            f"Decoder audio: {audio.shape}, range: [{audio.min():.4f}, {audio.max():.4f}]"
        )

        # Now let's trace through the generator more explicitly
        # Access generator
        generator = decoder.generator

        print("\n=== Tracing Generator ===")

        # Generator forward expects (asr, s, F0)
        # where asr is the processed features from decode blocks, not raw asr_pt

        # Let's manually call through decoder to get the intermediate x that goes to generator
        # Based on Kokoro/StyleTTS2 decoder structure:
        # encode -> decode blocks -> generator

        # This requires understanding the exact decoder structure
        # Let's try to find it by inspecting the decoder

        print("\nDecoder structure:")
        for name, module in decoder.named_children():
            print(f"  {name}: {type(module).__name__}")

        # Try to access encode/decode blocks
        if hasattr(decoder, "encode"):
            # Match lengths first
            asr_pt.shape[2]  # 63
            f0_proc.shape[2]  # 63

            # Concatenate
            x = torch.cat([asr_pt, f0_proc, n_proc], dim=1)  # [1, 514, 63]
            traces["concat_input"] = x.numpy()
            print(f"Concat input shape: {x.shape}")

            # Encode
            x_encoded = decoder.encode(x, style_pt)
            traces["encode_out"] = x_encoded.numpy()
            print(
                f"Encode out: {x_encoded.shape}, range: [{x_encoded.min():.4f}, {x_encoded.max():.4f}]"
            )

            # Decode blocks (ModuleList)
            for i, blk in enumerate(decoder.decode):
                # Need to concatenate residuals before each block
                x_with_res = torch.cat([x_encoded, asr_res, f0_proc, n_proc], dim=1)
                x_encoded = blk(x_with_res, style_pt)
                traces[f"decode_{i}_out"] = x_encoded.numpy()
                print(
                    f"Decode {i} out: {x_encoded.shape}, range: [{x_encoded.min():.4f}, {x_encoded.max():.4f}]"
                )

                # If this block upsamples, need to upsample residuals too
                if hasattr(blk, "upsample") and blk.upsample:
                    new_len = x_encoded.shape[2]
                    asr_res = F.interpolate(asr_res, size=new_len, mode="nearest")
                    f0_proc = F.interpolate(f0_proc, size=new_len, mode="nearest")
                    n_proc = F.interpolate(n_proc, size=new_len, mode="nearest")

            # Save the generator input (final decode output)
            # This is what goes into the generator
            traces["generator_input_ncl"] = x_encoded.numpy()
            print(
                f"\nGenerator input (final decode): {x_encoded.shape}, range: [{x_encoded.min():.4f}, {x_encoded.max():.4f}]"
            )

            # Now call the actual generator with the decoded features
            # The generator expects: (asr, s, F0) where asr is the decoded features in NCL format
            print("\n=== Calling Generator ===")
            gen = generator
            gen_audio = gen(x_encoded, style_pt, f0_orig)
            traces["generator_audio"] = gen_audio.numpy()
            print(
                f"Generator audio: {gen_audio.shape}, range: [{gen_audio.min():.4f}, {gen_audio.max():.4f}]"
            )

            # Also save Generator structure for reference
            print("\nGenerator structure:")
            for name, module in gen.named_children():
                print(f"  {name}: {type(module).__name__}")

    # Save traces
    output_path = Path("/tmp/kokoro_ref/generator_traces.npz")
    np.savez_compressed(output_path, **traces)
    print(f"\nSaved traces to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
