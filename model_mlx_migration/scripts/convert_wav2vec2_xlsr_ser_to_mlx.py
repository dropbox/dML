#!/usr/bin/env python3
"""Convert Wav2Vec2-XLSR-SER to MLX format.

This script converts the HuggingFace Wav2Vec2ForSequenceClassification
checkpoint to MLX format for speech emotion recognition.

Usage:
    python scripts/convert_wav2vec2_xlsr_ser_to_mlx.py

Input:
    models/sota/wav2vec2-xlsr-ser/model.safetensors

Output:
    models/sota/wav2vec2-xlsr-ser-mlx/weights.npz
    models/sota/wav2vec2-xlsr-ser-mlx/config.json
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from safetensors import safe_open


def convert_conv1d_weight(weight: torch.Tensor) -> np.ndarray:
    """Convert PyTorch Conv1d weight to MLX format.

    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    MLX Conv1d: (out_channels, kernel_size, in_channels)
    """
    return weight.permute(0, 2, 1).numpy()


def map_weight_key(hf_key: str) -> str:
    """Map HuggingFace key to our MLX model key.

    HF Structure:
        wav2vec2.feature_extractor.conv_layers.N.conv.weight/bias
        wav2vec2.feature_extractor.conv_layers.N.layer_norm.weight/bias
        wav2vec2.feature_projection.layer_norm.weight/bias
        wav2vec2.feature_projection.projection.weight/bias
        wav2vec2.encoder.pos_conv_embed.conv.weight_g/weight_v/bias
        wav2vec2.encoder.layer_norm.weight/bias
        wav2vec2.encoder.layers.N.attention.q/k/v/out_proj.weight/bias
        wav2vec2.encoder.layers.N.layer_norm.weight/bias
        wav2vec2.encoder.layers.N.feed_forward.intermediate_dense.weight/bias
        wav2vec2.encoder.layers.N.feed_forward.output_dense.weight/bias
        wav2vec2.encoder.layers.N.final_layer_norm.weight/bias
        classifier.dense.weight/bias
        classifier.output.weight/bias

    MLX Structure:
        wav2vec2.feature_extractor.conv_layers.N.conv.weight/bias
        wav2vec2.feature_extractor.conv_layers.N.layer_norm.weight/bias
        wav2vec2.feature_projection.layer_norm.weight/bias
        wav2vec2.feature_projection.projection.weight/bias
        wav2vec2.encoder.pos_conv_embed.weight/bias (pre-computed)
        wav2vec2.encoder.layer_norm.weight/bias
        wav2vec2.encoder.layers.N.attention.q/k/v/out_proj.weight/bias
        wav2vec2.encoder.layers.N.layer_norm.weight/bias
        wav2vec2.encoder.layers.N.feed_forward.intermediate_dense.weight/bias
        wav2vec2.encoder.layers.N.feed_forward.output_dense.weight/bias
        wav2vec2.encoder.layers.N.final_layer_norm.weight/bias
        projector.weight/bias
        classifier.weight/bias
    """
    key = hf_key

    # Skip weight normalization components (handled separately)
    if "pos_conv_embed.conv.weight_g" in key or "pos_conv_embed.conv.weight_v" in key:
        return None

    # Map pos_conv_embed.conv.bias -> pos_conv_embed.bias
    key = key.replace("pos_conv_embed.conv.bias", "pos_conv_embed.bias")

    # Map classifier.dense -> projector
    key = key.replace("classifier.dense.", "projector.")

    # Map classifier.output -> classifier
    key = key.replace("classifier.output.", "classifier.")

    return key


def convert_weights(input_path: Path) -> Dict[str, np.ndarray]:
    """Convert HuggingFace weights to MLX format."""
    mlx_weights = {}
    weight_g = None
    weight_v = None

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        for hf_key in keys:
            tensor = f.get_tensor(hf_key)

            # Skip masked_spec_embed (not used for inference)
            if "masked_spec_embed" in hf_key:
                print(f"  Skipping {hf_key}")
                continue

            # Store weight_g and weight_v for later
            if "pos_conv_embed.conv.weight_g" in hf_key:
                weight_g = tensor
                continue
            if "pos_conv_embed.conv.weight_v" in hf_key:
                weight_v = tensor
                continue

            # Map key
            mlx_key = map_weight_key(hf_key)
            if mlx_key is None:
                continue

            # Convert tensor
            if "conv.weight" in hf_key and "feature_extractor" in hf_key:
                # Conv1d weight
                value = convert_conv1d_weight(tensor)
                print(f"  {hf_key} -> {mlx_key}: {list(tensor.shape)} -> {list(value.shape)} (conv1d)")
            elif mlx_key.endswith(".weight") and tensor.ndim == 2:
                # Linear weight (no transpose needed for MLX)
                value = tensor.numpy()
                print(f"  {hf_key} -> {mlx_key}: {list(tensor.shape)}")
            else:
                value = tensor.numpy()
                print(f"  {hf_key} -> {mlx_key}: {list(tensor.shape)}")

            mlx_weights[mlx_key] = value

    # Pre-compute positional conv weight
    if weight_g is not None and weight_v is not None:
        # weight_g: (1024, 1, 1), weight_v: (1024, 64, 128)
        # Weight normalization: g * v / ||v||_2 where norm is per-filter (over in_channels and kernel_size)
        norm = weight_v.pow(2).sum(dim=[1, 2], keepdim=True).sqrt()
        effective_weight = weight_g * weight_v / norm
        # Convert to MLX format: (1024, 64, 128) -> (1024, 128, 64)
        effective_weight_mlx = effective_weight.permute(0, 2, 1).numpy()
        mlx_weights["wav2vec2.encoder.pos_conv_embed.weight"] = effective_weight_mlx
        print(f"  pos_conv_embed.weight_g + weight_v -> wav2vec2.encoder.pos_conv_embed.weight: {list(effective_weight_mlx.shape)}")

    print(f"\nConverted {len(mlx_weights)} parameters")
    return mlx_weights


def main():
    input_dir = Path("models/sota/wav2vec2-xlsr-ser")
    output_dir = Path("models/sota/wav2vec2-xlsr-ser-mlx")
    input_path = input_dir / "model.safetensors"

    print(f"Converting Wav2Vec2-XLSR-SER from {input_path} to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert weights
    print("\nConverting weights...")
    weights = convert_weights(input_path)

    # Save weights
    weights_path = output_dir / "weights.npz"
    np.savez(str(weights_path), **weights)
    print(f"\nSaved weights to {weights_path}")

    # Copy and adapt config
    with open(input_dir / "config.json") as f:
        hf_config = json.load(f)

    mlx_config = {
        "hidden_size": hf_config["hidden_size"],
        "num_hidden_layers": hf_config["num_hidden_layers"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "intermediate_size": hf_config["intermediate_size"],
        "conv_dim": hf_config["conv_dim"],
        "conv_kernel": hf_config["conv_kernel"],
        "conv_stride": hf_config["conv_stride"],
        "conv_bias": hf_config["conv_bias"],
        "feat_extract_norm": hf_config["feat_extract_norm"],
        "do_stable_layer_norm": hf_config["do_stable_layer_norm"],
        "num_conv_pos_embeddings": hf_config["num_conv_pos_embeddings"],
        "num_conv_pos_embedding_groups": hf_config["num_conv_pos_embedding_groups"],
        "layer_norm_eps": hf_config["layer_norm_eps"],
        "id2label": hf_config["id2label"],
        "label2id": hf_config["label2id"],
        "num_labels": len(hf_config["id2label"]),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Print statistics
    print("\nWeight statistics:")
    total_params = sum(v.size for v in weights.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
