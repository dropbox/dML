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
Convert aiola/whisper-medusa-v1 weights to MLX format.

This script downloads the Medusa head weights from the aiola HuggingFace model
and converts them to the format expected by our WhisperMLX Medusa implementation.

Source format (aiola):
  medusa_heads.<head_idx>.<layer_idx>.linear.{weight,bias}

Target format (MLX):
  heads.<head_idx>.linear.{weight,bias}

Usage:
    python scripts/convert_aiola_medusa.py
    python scripts/convert_aiola_medusa.py --output medusa_weights.npz
    python scripts/convert_aiola_medusa.py --n-heads 5  # Use only first 5 heads
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx


def download_medusa_weights(cache_dir: str = None) -> dict:
    """Download and load Medusa weights from aiola/whisper-medusa-v1."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    # Download the safetensors file containing Medusa weights
    # Based on the index, all Medusa weights are in model-00002-of-00002.safetensors
    model_file = hf_hub_download(
        "aiola/whisper-medusa-v1",
        "model-00002-of-00002.safetensors",
        cache_dir=cache_dir,
    )

    # Load only Medusa-related keys
    weights = {}
    with safe_open(model_file, framework="pt") as f:
        for key in f.keys():
            if "medusa_heads" in key:
                weights[key] = f.get_tensor(key)

    return weights


def convert_weights(
    aiola_weights: dict,
    n_heads: int = 10,
    target_vocab: int = None,
) -> dict:
    """
    Convert aiola Medusa weights to MLX format.

    Args:
        aiola_weights: Dictionary of weights from aiola model
        n_heads: Number of Medusa heads to use (aiola has 11, we might use fewer)
        target_vocab: Target vocabulary size (None to keep original)

    Returns:
        Dictionary of weights in MLX format for load_medusa_weights()
    """
    import torch

    mlx_weights = {}

    # Group weights by head index
    heads = {}
    for key, value in aiola_weights.items():
        # Key format: medusa_heads.<head_idx>.<layer_idx>.linear.{weight,bias}
        parts = key.split(".")
        if len(parts) >= 4 and parts[0] == "medusa_heads":
            head_idx = int(parts[1])
            layer_idx = int(parts[2])
            param_type = ".".join(parts[3:])  # "linear.weight" or "linear.bias"

            if head_idx not in heads:
                heads[head_idx] = {}

            # Convert torch tensor to numpy then to MLX
            if isinstance(value, torch.Tensor):
                value = value.numpy()

            # Store with our naming convention
            # aiola has layer_idx=0 for single-layer heads
            # We just use "linear.{weight,bias}" directly
            if layer_idx == 0:  # Only first layer for Medusa-Linear
                heads[head_idx][param_type] = value

    # Sort heads by index and convert up to n_heads
    sorted_heads = sorted(heads.items())[:n_heads]

    for new_idx, (orig_idx, params) in enumerate(sorted_heads):
        for param_name, value in params.items():
            mlx_key = f"heads.{new_idx}.{param_name}"

            # Handle vocab size adjustment if needed
            if target_vocab is not None and param_name == "linear.weight":
                current_vocab = value.shape[0]
                if current_vocab != target_vocab:
                    print(f"  Warning: Vocab size mismatch for head {new_idx}")
                    print(f"    Source: {current_vocab}, Target: {target_vocab}")
                    if target_vocab > current_vocab:
                        # Pad with zeros
                        padding = mx.zeros((target_vocab - current_vocab, value.shape[1]))
                        value = mx.concatenate([mx.array(value), padding], axis=0)
                        print(f"    Padded to {target_vocab}")
                    else:
                        # Truncate
                        value = value[:target_vocab]
                        print(f"    Truncated to {target_vocab}")

            if target_vocab is not None and param_name == "linear.bias":
                current_vocab = value.shape[0]
                if current_vocab != target_vocab:
                    if target_vocab > current_vocab:
                        padding = mx.zeros((target_vocab - current_vocab,))
                        value = mx.concatenate([mx.array(value), padding], axis=0)
                    else:
                        value = value[:target_vocab]

            mlx_weights[mlx_key] = mx.array(value) if not isinstance(value, mx.array) else value

    return mlx_weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert aiola/whisper-medusa-v1 weights to MLX format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="medusa_aiola_v1.npz",
        help="Output path for converted weights",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=10,
        help="Number of Medusa heads to use (aiola has 11)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to save converted weights (default: project models/)",
    )

    args = parser.parse_args()

    print("Converting aiola/whisper-medusa-v1 weights to MLX format")
    print(f"  Using {args.n_heads} Medusa heads")

    # Determine output path
    if args.models_dir:
        output_dir = Path(args.models_dir)
    else:
        # Default to project models directory
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "models"

    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / args.output

    # Download weights
    print("\n1. Downloading Medusa weights from HuggingFace...")
    try:
        aiola_weights = download_medusa_weights(args.cache_dir)
        print(f"   Downloaded {len(aiola_weights)} weight tensors")
    except Exception as e:
        print(f"   Error downloading weights: {e}")
        return 1

    # Inspect weight shapes
    print("\n2. Inspecting weight shapes...")
    for key, value in sorted(aiola_weights.items())[:6]:  # First 3 heads
        shape = value.shape if hasattr(value, 'shape') else 'N/A'
        print(f"   {key}: {shape}")
    print("   ...")

    # Get source vocab size
    sample_weight = next(v for k, v in aiola_weights.items() if "weight" in k)
    source_vocab = sample_weight.shape[0]
    hidden_dim = sample_weight.shape[1]
    print(f"\n   Source vocab size: {source_vocab}")
    print(f"   Hidden dimension: {hidden_dim}")

    # Check target vocab (large-v3 uses 51866, large-v2 uses 51865)
    # aiola is trained on large-v2 (51865 vocab)
    # Our WhisperMLX large-v3 uses 51866 vocab
    target_vocab = None  # Keep original for now, can adjust if needed

    # Convert weights
    print(f"\n3. Converting {args.n_heads} heads to MLX format...")
    mlx_weights = convert_weights(aiola_weights, n_heads=args.n_heads, target_vocab=target_vocab)
    print(f"   Converted {len(mlx_weights)} weight tensors")

    # Show converted shapes
    print("\n4. Converted weight shapes:")
    for key in sorted(mlx_weights.keys())[:6]:
        print(f"   {key}: {mlx_weights[key].shape}")
    print("   ...")

    # Save weights
    print(f"\n5. Saving to {output_path}...")
    mx.savez(str(output_path), **mlx_weights)

    # Verify saved file
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   Saved {file_size:.1f} MB")

    # Create metadata file
    metadata = {
        "source": "aiola/whisper-medusa-v1",
        "n_heads": args.n_heads,
        "source_vocab": source_vocab,
        "hidden_dim": hidden_dim,
        "target_vocab": target_vocab,
        "base_model": "whisper-large-v2",
        "note": "For use with whisper-large-v2 or compatible models",
    }
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata to {metadata_path}")

    print("\n6. Usage:")
    print("   model = WhisperMLX.from_pretrained('large-v2')  # Must use v2!")
    print(f"   model.load_medusa_heads('{output_path}', n_heads={args.n_heads})")
    print("   result = model.transcribe_medusa('audio.wav')")

    print("\nConversion complete!")
    return 0


if __name__ == "__main__":
    exit(main())
