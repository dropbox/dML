#!/usr/bin/env python3
"""Convert ECAPA-TDNN SpeechBrain checkpoint to MLX format.

This script converts the VoxLingua107 ECAPA-TDNN checkpoint from SpeechBrain
format to MLX format for fast inference on Apple Silicon.

Usage:
    python scripts/convert_ecapa_to_mlx.py

Input:
    models/sota/ecapa-tdnn/embedding_model.ckpt
    models/sota/ecapa-tdnn/classifier.ckpt
    models/sota/ecapa-tdnn/label_encoder.txt

Output:
    models/sota/ecapa-tdnn-mlx/weights.npz
    models/sota/ecapa-tdnn-mlx/config.json
    models/sota/ecapa-tdnn-mlx/label_encoder.txt
"""

import json
import shutil
from pathlib import Path
from typing import Dict

import torch
import numpy as np


def convert_conv1d_weight(weight: torch.Tensor) -> np.ndarray:
    """Convert PyTorch Conv1d weight to MLX format.

    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    MLX conv1d: (out_channels, kernel_size, in_channels)

    Args:
        weight: PyTorch weight tensor

    Returns:
        NumPy array in MLX format
    """
    # Transpose from (O, I, K) to (O, K, I)
    return weight.permute(0, 2, 1).numpy()


def convert_embedding_model(
    pytorch_path: Path,
) -> Dict[str, np.ndarray]:
    """Convert embedding model weights.

    Args:
        pytorch_path: Path to embedding_model.ckpt

    Returns:
        Dictionary of MLX weights
    """
    ckpt = torch.load(pytorch_path, map_location="cpu", weights_only=False)

    mlx_weights = {}

    for pt_key, tensor in ckpt.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        # Skip num_batches_tracked (not needed for inference)
        if "num_batches_tracked" in pt_key:
            continue

        # Map SpeechBrain keys to our MLX model keys
        mlx_key = map_embedding_key(pt_key)
        if mlx_key is None:
            print(f"  Skipping unmapped key: {pt_key}")
            continue

        # Convert tensor
        if "conv.weight" in pt_key or "conv.conv.weight" in pt_key:
            # Conv1d weights need transposition
            value = convert_conv1d_weight(tensor)
        else:
            value = tensor.numpy()

        mlx_weights[mlx_key] = value
        # print(f"  {pt_key} -> {mlx_key}: {value.shape}")

    return mlx_weights


def map_embedding_key(pt_key: str) -> str:
    """Map SpeechBrain key to MLX model key.

    SpeechBrain structure:
        blocks.0.conv.conv.weight -> blocks_0.conv.weight
        blocks.0.norm.norm.weight -> blocks_0.norm.weight
        blocks.1.tdnn1.conv.conv.weight -> blocks_1.tdnn1.conv.weight
        blocks.1.res2net_block.blocks.0.conv.conv.weight -> blocks_1.res2net_block.blocks.0.conv.weight
        blocks.1.se_block.conv1.conv.weight -> blocks_1.se_block.conv1.weight
        mfa.conv.conv.weight -> mfa.conv.weight
        asp.tdnn.conv.conv.weight -> asp.tdnn.conv.weight
        asp_bn.norm.weight -> asp_bn.weight
        fc.conv.weight -> fc.weight

    Args:
        pt_key: SpeechBrain key

    Returns:
        MLX model key
    """
    key = pt_key

    # Handle top-level blocks.N -> blocks_N (only at the start of the key)
    # But NOT res2net_block.blocks.N which should stay as blocks.N
    for i in range(4):
        if key.startswith(f"blocks.{i}."):
            key = f"blocks_{i}." + key[len(f"blocks.{i}."):]

    # Remove redundant .conv in conv paths (but not in se_block which has conv1/conv2)
    # blocks_0.conv.conv.weight -> blocks_0.conv.weight
    # But se_block.conv1.conv.weight -> se_block.conv1.weight

    # Handle se_block specially - remove .conv from conv1.conv and conv2.conv
    key = key.replace(".se_block.conv1.conv.", ".se_block.conv1.")
    key = key.replace(".se_block.conv2.conv.", ".se_block.conv2.")

    # For other paths, remove redundant .conv
    key = key.replace(".conv.conv.", ".conv.")

    # Remove redundant .norm in norm paths
    # blocks_0.norm.norm.weight -> blocks_0.norm.weight
    # But keep asp_bn.norm which becomes asp_bn directly
    if "asp_bn.norm." in key:
        key = key.replace("asp_bn.norm.", "asp_bn.")
    else:
        key = key.replace(".norm.norm.", ".norm.")

    # Handle fc.conv -> fc (Conv1d used as linear)
    key = key.replace("fc.conv.", "fc.")

    # Prefix with embedding_model for full model
    key = f"embedding_model.{key}"

    return key


def convert_classifier(pytorch_path: Path) -> Dict[str, np.ndarray]:
    """Convert classifier weights.

    Args:
        pytorch_path: Path to classifier.ckpt

    Returns:
        Dictionary of MLX weights
    """
    ckpt = torch.load(pytorch_path, map_location="cpu", weights_only=False)

    mlx_weights = {}

    for pt_key, tensor in ckpt.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        if "num_batches_tracked" in pt_key:
            continue

        mlx_key = map_classifier_key(pt_key)
        if mlx_key is None:
            print(f"  Skipping unmapped key: {pt_key}")
            continue

        mlx_weights[mlx_key] = tensor.numpy()

    return mlx_weights


def map_classifier_key(pt_key: str) -> str:
    """Map classifier key to MLX model key.

    SpeechBrain structure:
        norm.norm.weight -> classifier.norm.weight
        DNN.block_0.linear.w.weight -> classifier.linear.weight
        DNN.block_0.norm.norm.weight -> classifier.dnn_norm.weight
        out.w.weight -> classifier.out.weight

    Args:
        pt_key: SpeechBrain key

    Returns:
        MLX model key
    """
    key = pt_key

    # Map DNN.block_0.linear.w -> linear
    key = key.replace("DNN.block_0.linear.w.", "linear.")

    # Map DNN.block_0.norm.norm -> dnn_norm (remove redundant .norm)
    key = key.replace("DNN.block_0.norm.norm.", "dnn_norm.")

    # Map out.w -> out
    key = key.replace("out.w.", "out.")

    # Handle input norm: norm.norm -> norm
    if key.startswith("norm.norm."):
        key = key.replace("norm.norm.", "norm.")

    # Prefix with classifier
    key = f"classifier.{key}"

    return key


def main():
    """Main conversion function."""
    input_dir = Path("models/sota/ecapa-tdnn")
    output_dir = Path("models/sota/ecapa-tdnn-mlx")

    print(f"Converting ECAPA-TDNN from {input_dir} to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert embedding model
    print("\nConverting embedding model...")
    embedding_weights = convert_embedding_model(input_dir / "embedding_model.ckpt")
    print(f"  Converted {len(embedding_weights)} parameters")

    # Convert classifier
    print("\nConverting classifier...")
    classifier_weights = convert_classifier(input_dir / "classifier.ckpt")
    print(f"  Converted {len(classifier_weights)} parameters")

    # Merge weights
    all_weights = {**embedding_weights, **classifier_weights}
    print(f"\nTotal parameters: {len(all_weights)}")

    # Save weights
    weights_path = output_dir / "weights.npz"
    np.savez(str(weights_path), **all_weights)
    print(f"Saved weights to {weights_path}")

    # Save config
    config = {
        "n_mels": 60,
        "sample_rate": 16000,
        "channels": [1024, 1024, 1024, 1024, 3072],
        "kernel_sizes": [5, 3, 3, 3, 1],
        "dilations": [1, 2, 3, 4, 1],
        "attention_channels": 128,
        "lin_neurons": 256,
        "res2net_scale": 8,
        "se_channels": 128,
        "num_languages": 107,
        "classifier_hidden": 512,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Copy label encoder
    label_src = input_dir / "label_encoder.txt"
    label_dst = output_dir / "label_encoder.txt"
    if label_src.exists():
        shutil.copy(label_src, label_dst)
        print(f"Copied label encoder to {label_dst}")

    # Print weight statistics
    print("\nWeight statistics:")
    total_params = 0
    for key, value in all_weights.items():
        total_params += value.size
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
