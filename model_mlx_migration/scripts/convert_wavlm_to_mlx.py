#!/usr/bin/env python3
"""Convert WavLM-large to MLX format.

This script converts the Microsoft wavlm-large checkpoint to MLX format.

Usage:
    python scripts/convert_wavlm_to_mlx.py

Input:
    models/sota/wavlm-large/pytorch_model.bin
    models/sota/wavlm-large/config.json

Output:
    models/sota/wavlm-large-mlx/weights.npz
    models/sota/wavlm-large-mlx/config.json
"""

import json
from pathlib import Path
from typing import Dict

import torch
import numpy as np


def convert_conv1d_weight(weight: torch.Tensor) -> np.ndarray:
    """Convert PyTorch Conv1d weight to MLX format.

    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    MLX Conv1d: (out_channels, kernel_size, in_channels)
    """
    return weight.permute(0, 2, 1).numpy()


def map_weight_key(pt_key: str) -> str:
    """Map WavLM key to our MLX model key.

    WavLM structure (no wav2vec2. prefix):
        feature_extractor.conv_layers.N.conv.weight
        feature_extractor.conv_layers.N.layer_norm.weight
        feature_projection.layer_norm.weight
        feature_projection.projection.weight
        encoder.pos_conv_embed.conv.weight_g
        encoder.pos_conv_embed.conv.weight_v
        encoder.pos_conv_embed.conv.bias
        encoder.layer_norm.weight
        encoder.layers.N.attention.q_proj.weight
        encoder.layers.N.attention.k_proj.weight
        encoder.layers.N.attention.v_proj.weight
        encoder.layers.N.attention.out_proj.weight
        encoder.layers.N.attention.rel_attn_embed.weight
        encoder.layers.N.attention.gru_rel_pos_linear.weight
        encoder.layers.N.attention.gru_rel_pos_const
        encoder.layers.N.feed_forward.intermediate_dense.weight
        encoder.layers.N.feed_forward.output_dense.weight
        encoder.layers.N.layer_norm.weight
        encoder.layers.N.final_layer_norm.weight

    Our MLX structure:
        feature_extractor.conv_layers.N.conv.weight
        feature_extractor.conv_layers.N.layer_norm.weight
        feature_projection.layer_norm.weight
        feature_projection.projection.weight
        encoder.pos_conv_embed.weight (pre-computed from weight_g, weight_v)
        encoder.pos_conv_embed.bias
        encoder.layer_norm.weight
        encoder.layers.N.attention.q_proj.weight
        encoder.layers.N.attention.k_proj.weight
        encoder.layers.N.attention.v_proj.weight
        encoder.layers.N.attention.out_proj.weight
        encoder.layers.N.attention.rel_attn_embed.weight
        encoder.layers.N.attention.gru_rel_pos_linear.weight
        encoder.layers.N.attention.gru_rel_pos_const
        encoder.layers.N.feed_forward.intermediate_dense.weight
        encoder.layers.N.feed_forward.output_dense.weight
        encoder.layers.N.layer_norm.weight
        encoder.layers.N.final_layer_norm.weight
    """
    key = pt_key

    # Positional conv embedding: pos_conv_embed.conv.X -> pos_conv_embed.X (except weight_g/weight_v)
    if "pos_conv_embed.conv.bias" in key:
        key = key.replace("pos_conv_embed.conv.bias", "pos_conv_embed.bias")

    return key


def is_conv1d_weight(pt_key: str) -> bool:
    """Check if a key corresponds to a Conv1d weight."""
    if "feature_extractor.conv_layers" in pt_key and ".conv.weight" in pt_key:
        return True
    return False


def convert_wavlm_weights(pytorch_path: Path) -> Dict[str, np.ndarray]:
    """Convert WavLM PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to pytorch_model.bin

    Returns:
        Dictionary of MLX weights
    """
    weights = torch.load(pytorch_path, map_location="cpu", weights_only=True)

    mlx_weights = {}
    skipped_keys = []

    for pt_key, tensor in weights.items():
        # Skip quantizer, projection heads, and masked_spec_embed (not needed for inference)
        if any(skip in pt_key for skip in ["quantizer", "project_hid", "project_q", "masked_spec_embed"]):
            skipped_keys.append(pt_key)
            continue

        # Skip weight_g and weight_v - we'll combine them below
        if "pos_conv_embed.conv.weight_g" in pt_key or "pos_conv_embed.conv.weight_v" in pt_key:
            continue

        # Map key
        mlx_key = map_weight_key(pt_key)

        # Convert tensor
        if is_conv1d_weight(pt_key):
            value = convert_conv1d_weight(tensor)
        else:
            value = tensor.numpy()

        mlx_weights[mlx_key] = value
        print(f"  {pt_key} -> {mlx_key}: {list(tensor.shape)} -> {list(value.shape)}")

    # Pre-compute positional conv weight from weight_g and weight_v
    if "encoder.pos_conv_embed.conv.weight_g" in weights:
        weight_g = weights["encoder.pos_conv_embed.conv.weight_g"]
        weight_v = weights["encoder.pos_conv_embed.conv.weight_v"]

        # weight_g: (1, 1, 128) - one scalar per kernel position
        # weight_v: (1024, 64, 128) - (out_channels, in_channels/groups, kernel_size)
        # weight_norm: normalize over dim=2 (kernel dimension)
        # For WavLM, dim=2 in weight_norm means kernel dimension

        # Compute L2 norm over the correct dimensions
        # The norm should be computed per filter (over in_channels and kernel dimensions)
        norm = weight_v.pow(2).sum(dim=[1, 2], keepdim=True).sqrt()
        effective_weight = weight_g * weight_v / norm

        # Transpose to MLX format: (out, in/groups, kernel) -> (out, kernel, in/groups)
        effective_weight_mlx = effective_weight.permute(0, 2, 1).numpy()
        mlx_weights["encoder.pos_conv_embed.weight"] = effective_weight_mlx
        print(f"  pos_conv_embed.weight_g + weight_v -> encoder.pos_conv_embed.weight: computed -> {list(effective_weight_mlx.shape)}")

    print(f"\nSkipped {len(skipped_keys)} keys (quantizer/projection)")

    return mlx_weights


def main():
    """Main conversion function."""
    input_dir = Path("models/sota/wavlm-large")
    output_dir = Path("models/sota/wavlm-large-mlx")

    print(f"Converting WavLM-large from {input_dir} to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert weights
    print("\nConverting weights...")
    weights = convert_wavlm_weights(input_dir / "pytorch_model.bin")
    print(f"Converted {len(weights)} parameters")

    # Save weights
    weights_path = output_dir / "weights.npz"
    np.savez(str(weights_path), **weights)
    print(f"\nSaved weights to {weights_path}")

    # Load and save config
    config_src = input_dir / "config.json"
    if config_src.exists():
        with open(config_src) as f:
            config = json.load(f)

        simplified_config = {
            "hidden_size": config.get("hidden_size", 1024),
            "num_hidden_layers": config.get("num_hidden_layers", 24),
            "num_attention_heads": config.get("num_attention_heads", 16),
            "intermediate_size": config.get("intermediate_size", 4096),
            "hidden_act": config.get("hidden_act", "gelu"),
            "layer_norm_eps": config.get("layer_norm_eps", 1e-5),
            "do_stable_layer_norm": config.get("do_stable_layer_norm", True),
            "conv_dim": config.get("conv_dim", [512, 512, 512, 512, 512, 512, 512]),
            "conv_kernel": config.get("conv_kernel", [10, 3, 3, 3, 3, 2, 2]),
            "conv_stride": config.get("conv_stride", [5, 2, 2, 2, 2, 2, 2]),
            "conv_bias": config.get("conv_bias", False),  # WavLM has no conv bias
            "num_conv_pos_embeddings": config.get("num_conv_pos_embeddings", 128),
            "num_conv_pos_embedding_groups": config.get("num_conv_pos_embedding_groups", 16),
            "num_buckets": config.get("num_buckets", 320),
            "max_bucket_distance": config.get("max_bucket_distance", 800),
        }

        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(simplified_config, f, indent=2)
        print(f"Saved config to {config_path}")

    # Print statistics
    print("\nWeight statistics:")
    total_params = sum(v.size for v in weights.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
