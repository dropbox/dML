#!/usr/bin/env python3
"""Convert BEATs to MLX format.

This script converts the Microsoft BEATs checkpoint to MLX format.

Usage:
    python scripts/convert_beats_to_mlx.py

Input:
    models/sota/beats/models/sota/beats/BEATs_iter3_plus_AS2M.pt

Output:
    models/sota/beats-mlx/weights.npz
    models/sota/beats-mlx/config.json
"""

import json
from pathlib import Path
from typing import Dict

import torch
import numpy as np


def convert_conv2d_weight(weight: torch.Tensor) -> np.ndarray:
    """Convert PyTorch Conv2d weight to MLX format.

    PyTorch Conv2d: (out_channels, in_channels, kernel_h, kernel_w)
    MLX Conv2d: (out_channels, kernel_h, kernel_w, in_channels)
    """
    return weight.permute(0, 2, 3, 1).numpy()


def convert_conv1d_weight(weight: torch.Tensor) -> np.ndarray:
    """Convert PyTorch Conv1d weight to MLX format.

    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    MLX Conv1d: (out_channels, kernel_size, in_channels)
    """
    return weight.permute(0, 2, 1).numpy()


def map_weight_key(pt_key: str) -> str:
    """Map BEATs checkpoint key to our MLX model key.

    BEATs structure:
        post_extract_proj.weight/bias
        patch_embedding.weight
        encoder.pos_conv.0.weight_g/weight_v/bias
        encoder.layers.N.self_attn.q/k/v/out_proj.weight/bias
        encoder.layers.N.self_attn.relative_attention_bias.weight
        encoder.layers.N.self_attn.grep_a
        encoder.layers.N.self_attn.grep_linear.weight/bias
        encoder.layers.N.self_attn_layer_norm.weight/bias
        encoder.layers.N.fc1/fc2.weight/bias
        encoder.layers.N.final_layer_norm.weight/bias
        encoder.layer_norm.weight/bias
        layer_norm.weight/bias

    Our MLX structure (same but with some key adjustments):
        Same structure but:
        - patch_embedding.weight -> patch_embedding.proj.weight
        - pos_conv.0.X -> pos_conv.X (flatten)
        - relative_attention_bias.weight -> relative_attention_bias.weight (embedding)
    """
    key = pt_key

    # Patch embedding: conv is inside .proj
    key = key.replace("patch_embedding.weight", "patch_embedding.proj.weight")

    # Pos conv: flatten the .0. part
    key = key.replace("pos_conv.0.", "pos_conv.")

    return key


def convert_beats_weights(checkpoint_path: Path) -> Dict[str, np.ndarray]:
    """Convert BEATs PyTorch weights to MLX format.

    Args:
        checkpoint_path: Path to BEATs checkpoint .pt file

    Returns:
        Dictionary of MLX weights
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("model", checkpoint)
    config = checkpoint.get("cfg", {})

    mlx_weights = {}
    skipped_keys = []

    for pt_key, tensor in weights.items():
        # Map key
        mlx_key = map_weight_key(pt_key)

        # Convert tensor based on type
        if "patch_embedding.proj.weight" in mlx_key:
            # Conv2d weight
            value = convert_conv2d_weight(tensor)
            print(f"  {pt_key} -> {mlx_key}: {list(tensor.shape)} -> {list(value.shape)} (conv2d)")
        elif "pos_conv.weight_g" in mlx_key or "pos_conv.weight_v" in mlx_key:
            # Handle weight normalization
            continue  # Processed separately below
        elif mlx_key.endswith(".weight") and tensor.ndim == 2:
            # Linear weight
            value = tensor.numpy()
            print(f"  {pt_key} -> {mlx_key}: {list(tensor.shape)}")
        else:
            value = tensor.numpy()
            print(f"  {pt_key} -> {mlx_key}: {list(tensor.shape)}")

        mlx_weights[mlx_key] = value

    # Pre-compute positional conv weight from weight_g and weight_v
    # BEATs uses weight_norm with dim=2, meaning normalization across dims (0, 1)
    if "encoder.pos_conv.0.weight_g" in weights:
        weight_g = weights["encoder.pos_conv.0.weight_g"]  # (1, 1, 128)
        weight_v = weights["encoder.pos_conv.0.weight_v"]  # (768, 48, 128)

        # Compute effective weight: weight = weight_g * F.normalize(weight_v, dim=[0, 1])
        # F.normalize normalizes along the specified dims, keeping dim=2 intact
        effective_weight = weight_g * torch.nn.functional.normalize(weight_v, dim=[0, 1])

        # Transpose to MLX format: (out, in/groups, kernel) -> (out, kernel, in/groups)
        effective_weight_mlx = effective_weight.permute(0, 2, 1).numpy()
        mlx_weights["encoder.pos_conv.weight"] = effective_weight_mlx
        print(f"  pos_conv.weight_g + weight_v -> encoder.pos_conv.weight: computed -> {list(effective_weight_mlx.shape)}")

    print(f"\nConverted {len(mlx_weights)} parameters")

    return mlx_weights, config


def main():
    """Main conversion function."""
    input_path = Path("models/sota/beats/models/sota/beats/BEATs_iter3_plus_AS2M.pt")
    output_dir = Path("models/sota/beats-mlx")

    print(f"Converting BEATs from {input_path} to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert weights
    print("\nConverting weights...")
    weights, cfg = convert_beats_weights(input_path)

    # Save weights
    weights_path = output_dir / "weights.npz"
    np.savez(str(weights_path), **weights)
    print(f"\nSaved weights to {weights_path}")

    # Save config
    config = {
        "encoder_layers": cfg.get("encoder_layers", 12),
        "encoder_embed_dim": cfg.get("encoder_embed_dim", 768),
        "encoder_ffn_embed_dim": cfg.get("encoder_ffn_embed_dim", 3072),
        "encoder_attention_heads": cfg.get("encoder_attention_heads", 12),
        "activation_fn": cfg.get("activation_fn", "gelu"),
        "layer_norm_first": cfg.get("layer_norm_first", False),
        "conv_pos": cfg.get("conv_pos", 128),
        "conv_pos_groups": cfg.get("conv_pos_groups", 16),
        "relative_position_embedding": cfg.get("relative_position_embedding", True),
        "num_buckets": cfg.get("num_buckets", 320),
        "max_distance": cfg.get("max_distance", 800),
        "gru_rel_pos": cfg.get("gru_rel_pos", True),
        "deep_norm": cfg.get("deep_norm", True),
        "input_patch_size": cfg.get("input_patch_size", 16),
        "embed_dim": cfg.get("embed_dim", 512),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Print statistics
    print("\nWeight statistics:")
    total_params = sum(v.size for v in weights.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
