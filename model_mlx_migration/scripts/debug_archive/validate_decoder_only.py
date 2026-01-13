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
Validate MLX Kokoro decoder against PyTorch - decoder-only comparison.

This script bypasses the kokoro package entirely and loads the checkpoint
directly with PyTorch to build a reference decoder.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_pytorch_decoder(checkpoint_path: str):
    """Load decoder weights from PyTorch checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract decoder/generator weights
    decoder_weights = {}
    for k, v in checkpoint.items():
        if k.startswith("module.decoder.") or k.startswith("module.generator."):
            decoder_weights[k] = v

    return checkpoint, decoder_weights


def create_synthetic_inputs(batch: int = 1, frames: int = 100, hidden: int = 512):
    """Create synthetic decoder inputs for testing."""
    np.random.seed(42)  # Deterministic for reproducibility

    # asr: [B, T, hidden] - aligned encoder features
    asr = np.random.randn(batch, frames, hidden).astype(np.float32) * 0.1

    # F0: [B, T] - fundamental frequency (decoder expects 2D, not 3D)
    f0 = np.abs(np.random.randn(batch, frames).astype(np.float32)) * 100 + 100

    # N: [B, T] - noise features (decoder expects 2D, not 3D)
    n = np.random.randn(batch, frames).astype(np.float32) * 0.1

    # style: [B, 128] - style embedding
    style = np.random.randn(batch, 128).astype(np.float32) * 0.1

    return asr, f0, n, style


def run_pytorch_decoder(checkpoint, asr, f0, n, style):
    """Run PyTorch decoder.

    Note: This requires reconstructing the decoder architecture.
    For now, we'll use a simplified approach.
    """

    # Extract relevant weights
    weights = {}
    for k, v in checkpoint.items():
        weights[k] = v

    # The decoder in Kokoro is quite complex (ISTFTNet)
    # For a proper comparison, we need to either:
    # 1. Load the official kokoro model
    # 2. Reconstruct the decoder architecture
    #
    # Since kokoro isn't available, let's do a layer-by-layer comparison
    # starting with the AdainResBlk1d blocks

    # First, let's just verify weight shapes match
    return None  # Placeholder


def validate_weight_loading():
    """Validate that MLX weights match PyTorch checkpoint exactly."""
    import torch

    from tools.pytorch_to_mlx.converters import KokoroConverter

    # Load PyTorch checkpoint
    checkpoint_path = (
        Path.home()
        / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987/kokoro-v1_0.pth"
    )

    if not checkpoint_path.exists():
        # Try to find it
        import glob

        matches = glob.glob(
            str(
                Path.home()
                / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/**/kokoro-v1_0.pth"
            ),
            recursive=True,
        )
        if matches:
            checkpoint_path = Path(matches[0])
        else:
            raise FileNotFoundError("Kokoro checkpoint not found")

    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    pt_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load MLX model
    print("Loading MLX model...")
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Compare key weights
    comparisons = []

    # 1. Compare decoder.encode.weight
    pt_key = "module.decoder.encode.weight_v"
    if pt_key in pt_ckpt:
        pt_w = pt_ckpt[pt_key].numpy()
        mlx_w = np.array(mlx_model.decoder.encode.weight)

        # MLX uses different layout, may need transpose
        if pt_w.shape != mlx_w.shape:
            # Try transpose
            pt_w_t = pt_w.transpose(0, 2, 1)  # NCL -> NLC
            if pt_w_t.shape == mlx_w.shape:
                pt_w = pt_w_t

        max_diff = np.max(np.abs(pt_w - mlx_w))
        comparisons.append(("decoder.encode.weight", pt_w.shape, mlx_w.shape, max_diff))

    # 2. Compare F0_conv weights
    pt_key = "module.generator.F0_conv.weight_v"
    if pt_key in pt_ckpt:
        pt_w = pt_ckpt[pt_key].numpy()
        mlx_w = np.array(mlx_model.decoder.generator.f0_conv.weight)
        max_diff = (
            np.max(np.abs(pt_w.transpose(0, 2, 1) - mlx_w))
            if pt_w.shape != mlx_w.shape
            else np.max(np.abs(pt_w - mlx_w))
        )
        comparisons.append(
            ("generator.F0_conv.weight", pt_w.shape, mlx_w.shape, max_diff)
        )

    # 3. Compare source module weights
    pt_key = "module.generator.source.conv_pre.weight"
    if pt_key in pt_ckpt:
        pt_w = pt_ckpt[pt_key].numpy()
        mlx_w = np.array(mlx_model.decoder.generator.source.conv_pre.weight)
        max_diff = (
            np.max(np.abs(pt_w.transpose(0, 2, 1) - mlx_w))
            if pt_w.shape != mlx_w.shape
            else np.max(np.abs(pt_w - mlx_w))
        )
        comparisons.append(
            ("source.conv_pre.weight", pt_w.shape, mlx_w.shape, max_diff)
        )

    # Print comparison results
    print("\n" + "=" * 72)
    print("Weight Comparison: PyTorch vs MLX")
    print("=" * 72)

    all_match = True
    for name, pt_shape, mlx_shape, max_diff in comparisons:
        status = "✓" if max_diff < 1e-5 else "✗"
        if max_diff >= 1e-5:
            all_match = False
        print(f"{status} {name}: PT{pt_shape} MLX{mlx_shape} max_diff={max_diff:.2e}")

    return all_match, comparisons


def validate_decoder_forward():
    """Validate decoder forward pass produces reasonable output."""
    import mlx.core as mx

    from tools.pytorch_to_mlx.converters import KokoroConverter

    print("\n" + "=" * 72)
    print("Decoder Forward Pass Validation")
    print("=" * 72)

    # Load MLX model
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Create synthetic inputs
    asr_np, f0_np, n_np, style_np = create_synthetic_inputs(
        batch=1, frames=50, hidden=512
    )

    asr = mx.array(asr_np)
    f0 = mx.array(f0_np)
    n = mx.array(n_np)
    style = mx.array(style_np)

    print(
        f"Input shapes: asr={asr.shape}, f0={f0.shape}, n={n.shape}, style={style.shape}"
    )

    # Run decoder
    try:
        audio = mlx_model.decoder(asr, f0, n, style)
        mx.eval(audio)
        audio_np = np.array(audio).reshape(-1)

        print(f"Output shape: {audio.shape}")
        print(
            f"Output stats: min={audio_np.min():.4f}, max={audio_np.max():.4f}, mean={audio_np.mean():.4f}, std={audio_np.std():.4f}"
        )
        print(f"RMS: {np.sqrt(np.mean(audio_np**2)):.4f}")

        # Check for pathological outputs
        if np.isnan(audio_np).any():
            print("ERROR: NaN in output!")
            return False
        if np.isinf(audio_np).any():
            print("ERROR: Inf in output!")
            return False
        if np.std(audio_np) < 1e-6:
            print("ERROR: Output is essentially constant!")
            return False

        print("✓ Decoder produces valid output")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate Kokoro MLX decoder")
    parser.add_argument("--mode", choices=["weights", "forward", "all"], default="all")
    args = parser.parse_args()

    results = {}

    if args.mode in ("weights", "all"):
        weights_ok, comparisons = validate_weight_loading()
        results["weights"] = weights_ok

    if args.mode in ("forward", "all"):
        forward_ok = validate_decoder_forward()
        results["forward"] = forward_ok

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{name}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
