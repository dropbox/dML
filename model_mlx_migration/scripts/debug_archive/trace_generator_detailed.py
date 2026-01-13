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
Trace Generator layer by layer to find divergence point.

Uses reference tensors from /tmp/kokoro_ref/tensors.npz as inputs,
then traces through the Generator to find where MLX diverges.
"""

# Add project to path
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load reference tensors
    ref_path = Path("/tmp/kokoro_ref/tensors.npz")
    if not ref_path.exists():
        print(f"ERROR: Reference tensors not found at {ref_path}")
        print("Run export_kokoro_reference.py first")
        return 1

    ref = np.load(ref_path)

    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Get reference inputs for Decoder
    # PyTorch decoder inputs: asr (NCL), F0_pred, N_pred, style_128
    asr_ncl = ref["asr_ncl"]  # [1, hidden, T_align] - NCL format
    F0_pred = ref["F0_pred"]  # [1, T_align]
    N_pred = ref["N_pred"]  # [1, T_align]
    style_128 = ref["style_128"]  # [1, 128]

    print(f"Reference asr_ncl shape: {asr_ncl.shape}")
    print(f"Reference F0_pred shape: {F0_pred.shape}")
    print(f"Reference N_pred shape: {N_pred.shape}")
    print(f"Reference style_128 shape: {style_128.shape}")

    # Convert to MLX
    F0_mx = mx.array(F0_pred)
    N_mx = mx.array(N_pred)
    style_mx = mx.array(style_128)

    # For MLX decoder, we need NLC format for asr
    asr_nlc_mx = mx.transpose(mx.array(asr_ncl), (0, 2, 1))  # [1, T_align, hidden]
    print(f"\nMLX asr_nlc shape: {asr_nlc_mx.shape}")

    # === Run Decoder directly ===
    decoder = model.decoder

    # Run full decoder to compare
    print("\n=== Running full Decoder ===")
    audio = decoder(asr_nlc_mx, F0_mx, N_mx, style_mx)
    mx.eval(audio)
    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]")
    print(f"Audio mean: {float(mx.mean(audio)):.6f}")
    print(f"Audio std: {float(mx.std(audio)):.6f}")

    # Compare with reference
    ref_audio = ref["audio"]
    print("\n=== Comparison with Reference ===")
    print(f"Reference audio shape: {ref_audio.shape}")
    print(f"Reference audio range: [{ref_audio.min():.4f}, {ref_audio.max():.4f}]")

    # Align lengths for correlation
    mlx_audio = np.array(audio)
    min_len = min(mlx_audio.shape[-1], ref_audio.shape[-1])
    mlx_trimmed = mlx_audio.flatten()[:min_len]
    ref_trimmed = ref_audio.flatten()[:min_len]

    corr = np.corrcoef(mlx_trimmed, ref_trimmed)[0, 1]
    print(f"Correlation: {corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
