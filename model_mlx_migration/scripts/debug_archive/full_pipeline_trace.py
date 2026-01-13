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
Trace full MLX pipeline vs PyTorch reference with matched inputs.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load reference tensors (from export_kokoro_reference.py)
    ref_path = Path("/tmp/kokoro_ref/tensors.npz")
    if not ref_path.exists():
        print("Reference not found")
        return 1
    ref = np.load(ref_path)

    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Use exact same inputs as PyTorch reference
    asr_ncl = ref["asr_ncl"]  # [1, 512, 63]
    F0_pred = ref["F0_pred"]  # [1, 126]
    N_pred = ref["N_pred"]  # [1, 126]
    style_128 = ref["style_128"]  # [1, 128]

    # Convert to MLX (NLC format)
    asr_nlc = mx.array(asr_ncl).transpose(0, 2, 1)  # [1, 63, 512]
    F0_mx = mx.array(F0_pred)
    N_mx = mx.array(N_pred)
    style_mx = mx.array(style_128)

    print(f"asr_nlc: {asr_nlc.shape}")
    print(
        f"F0: {F0_mx.shape}, range: [{float(mx.min(F0_mx)):.4f}, {float(mx.max(F0_mx)):.4f}]"
    )
    print(f"N: {N_mx.shape}")
    print(f"style: {style_mx.shape}")

    # Run decoder
    decoder = model.decoder
    generator = decoder.generator

    # Manually run through decoder to get generator input
    print("\n=== Running Decoder ===")

    # First need to understand decoder structure
    print("Decoder attributes:")
    for name in dir(decoder):
        if not name.startswith("_"):
            attr = getattr(decoder, name, None)
            if hasattr(attr, "shape") or hasattr(attr, "__call__"):
                print(f"  {name}: {type(attr).__name__}")

    # The decoder forward should produce generator input
    # Let's just trace the generator directly with correct inputs

    # Calculate upsampling factor
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size
    print(f"\nTotal upsampling: {total_upp}")

    # Test SourceModule with same F0
    print("\n=== Testing SourceModule ===")
    har_source, noise, uv = generator.m_source(F0_mx, total_upp)
    mx.eval([har_source, noise, uv])
    print(
        f"har_source: {har_source.shape}, range: [{float(mx.min(har_source)):.4f}, {float(mx.max(har_source)):.4f}]"
    )
    print(
        f"har_source mean: {float(mx.mean(har_source)):.6f}, std: {float(mx.std(har_source)):.6f}"
    )

    # Now run full decoder and get audio
    print("\n=== Running Full Decoder ===")
    audio = decoder(asr_nlc, F0_mx, N_mx, style_mx)
    mx.eval(audio)
    print(
        f"audio: {audio.shape}, range: [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]"
    )
    print(f"audio mean: {float(mx.mean(audio)):.6f}, std: {float(mx.std(audio)):.6f}")

    # Compare with reference
    ref_audio = ref["audio"]
    print(
        f"\nReference audio: {ref_audio.shape}, range: [{ref_audio.min():.4f}, {ref_audio.max():.4f}]"
    )
    print(f"Reference audio mean: {ref_audio.mean():.6f}, std: {ref_audio.std():.6f}")

    # Correlation
    mlx_audio = np.array(audio).flatten()
    ref_flat = ref_audio.flatten()
    min_len = min(len(mlx_audio), len(ref_flat))
    corr = np.corrcoef(mlx_audio[:min_len], ref_flat[:min_len])[0, 1]
    print(f"\nCorrelation: {corr:.6f}")

    # Check if the audio is clipping
    n_clipped = np.sum((np.abs(mlx_audio) > 0.999).astype(int))
    print(f"Number of clipped samples: {n_clipped} / {len(mlx_audio)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
