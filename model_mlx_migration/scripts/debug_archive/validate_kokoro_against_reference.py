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
Validate MLX Kokoro decoder/generator against a PyTorch reference export.

This script is meant to be run in THIS repo's MLX environment (Python 3.14),
using reference tensors produced by `scripts/export_kokoro_reference.py`
running in a separate Python <3.14 environment with the official `kokoro`
package installed.

What it checks:
  - Decoder-only end-to-end: MLX Decoder(asr, F0, N, style128) vs PyTorch audio
  - Basic waveform error statistics

Usage:
  python scripts/validate_kokoro_against_reference.py --ref-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import numpy as np


def compare(name: str, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    min_len = min(a.size, b.size)
    a1 = a.reshape(-1)[:min_len]
    b1 = b.reshape(-1)[:min_len]
    diff = np.abs(a1 - b1)
    return {
        "name": name,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "compare_len": int(min_len),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean((a1 - b1) ** 2))) if diff.size else 0.0,
        "a_rms": float(np.sqrt(np.mean(a1 * a1))) if diff.size else 0.0,
        "b_rms": float(np.sqrt(np.mean(b1 * b1))) if diff.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MLX Kokoro vs PyTorch reference export"
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        required=True,
        help="Reference directory (metadata.json + tensors.npz)",
    )
    parser.add_argument(
        "--mode", choices=["decoder"], default="decoder", help="Validation mode"
    )
    args = parser.parse_args()

    meta_path = args.ref_dir / "metadata.json"
    tensors_path = args.ref_dir / "tensors.npz"
    if not meta_path.exists() or not tensors_path.exists():
        raise SystemExit(f"Missing reference files in {args.ref_dir}")

    metadata = json.loads(meta_path.read_text())
    ref = np.load(tensors_path)

    # Load MLX model (weights via our converter helper)
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Inputs
    asr_nlc = mx.array(ref["asr_nlc"].astype(np.float32))
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    n = mx.array(ref["N_pred"].astype(np.float32))
    style_128 = mx.array(ref["style_128"].astype(np.float32))

    # Run MLX decoder directly
    audio_mlx = mlx_model.decoder(asr_nlc, f0, n, style_128)
    mx.eval(audio_mlx)
    audio_mlx_np = np.array(audio_mlx).reshape(-1)

    audio_ref = ref["audio"].astype(np.float32).reshape(-1)

    stats = compare("decoder_audio", audio_ref, audio_mlx_np)

    print("=" * 72)
    print("Kokoro MLX vs PyTorch Reference (decoder-only)")
    print("=" * 72)
    print(f"Reference: {args.ref_dir}")
    print(f"Text: {metadata.get('text')!r}")
    print(f"Voice: {metadata.get('voice')}")
    print(
        f"Phonemes (len={len(metadata.get('phonemes', ''))}): {metadata.get('phonemes')!r}"
    )
    print()
    print(f"Max abs error: {stats['max_abs']:.6f}")
    print(f"Mean abs error: {stats['mean_abs']:.6f}")
    print(f"RMSE: {stats['rmse']:.6f}")
    print(f"Ref RMS: {stats['a_rms']:.6f}")
    print(f"MLX RMS: {stats['b_rms']:.6f}")

    # Non-zero exit if wildly off (tuned once reference is stable)
    # For now: fail if max error > 0.1
    return 0 if stats["max_abs"] <= 0.1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
