#!/usr/bin/env python3
"""Convert AST (Audio Spectrogram Transformer) to MLX format.

This script converts the HuggingFace AST AudioSet checkpoint to MLX format.

Usage:
    python scripts/convert_ast_to_mlx.py

Input:
    models/sota/ast/pytorch_model.bin
    models/sota/ast/config.json

Output:
    models/sota/ast-mlx/weights.npz
    models/sota/ast-mlx/config.json
    models/sota/ast-mlx/labels.json
"""

import json
from pathlib import Path
from typing import Dict

import torch
import numpy as np


def convert_conv2d_weight(weight: torch.Tensor) -> np.ndarray:
    """Convert PyTorch Conv2d weight to MLX format.

    PyTorch Conv2d: (out_channels, in_channels, H, W)
    MLX Conv2d: (out_channels, H, W, in_channels)

    Args:
        weight: PyTorch weight tensor

    Returns:
        NumPy array in MLX format
    """
    # Transpose from (O, I, H, W) to (O, H, W, I)
    return weight.permute(0, 2, 3, 1).numpy()


def map_weight_key(pt_key: str) -> str:
    """Map HuggingFace AST key to our MLX model key.

    HuggingFace structure:
        audio_spectrogram_transformer.embeddings.cls_token
        audio_spectrogram_transformer.embeddings.distillation_token
        audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight
        audio_spectrogram_transformer.embeddings.position_embeddings
        audio_spectrogram_transformer.encoder.layer.N.attention.attention.query.weight
        audio_spectrogram_transformer.encoder.layer.N.attention.output.dense.weight
        audio_spectrogram_transformer.encoder.layer.N.intermediate.dense.weight
        audio_spectrogram_transformer.encoder.layer.N.output.dense.weight
        audio_spectrogram_transformer.encoder.layer.N.layernorm_before.weight
        audio_spectrogram_transformer.encoder.layer.N.layernorm_after.weight
        audio_spectrogram_transformer.layernorm.weight
        classifier.layernorm.weight
        classifier.dense.weight

    Our MLX structure:
        audio_spectrogram_transformer.embeddings.cls_token
        audio_spectrogram_transformer.embeddings.distillation_token
        audio_spectrogram_transformer.embeddings.patch_embeddings.weight
        audio_spectrogram_transformer.embeddings.position_embeddings
        audio_spectrogram_transformer.encoder.layers.N.attention.query.weight
        audio_spectrogram_transformer.encoder.layers.N.attention.output_dense.weight
        audio_spectrogram_transformer.encoder.layers.N.mlp.dense1.weight
        audio_spectrogram_transformer.encoder.layers.N.mlp.dense2.weight
        audio_spectrogram_transformer.encoder.layers.N.layernorm_before.weight
        audio_spectrogram_transformer.encoder.layers.N.layernorm_after.weight
        audio_spectrogram_transformer.layernorm.weight
        classifier.layernorm.weight
        classifier.dense.weight
    """
    key = pt_key

    # Patch embeddings: projection.weight -> weight
    key = key.replace(".patch_embeddings.projection.", ".patch_embeddings.")

    # Encoder layers: layer.N -> layers.N
    key = key.replace(".encoder.layer.", ".encoder.layers.")

    # Attention: attention.attention.X -> attention.X
    key = key.replace(".attention.attention.", ".attention.")

    # Attention output: attention.output.dense -> attention.output_dense
    key = key.replace(".attention.output.dense.", ".attention.output_dense.")

    # MLP: intermediate.dense -> mlp.dense1, output.dense -> mlp.dense2
    # Be careful: only transform within encoder layers
    if ".encoder.layers." in key:
        key = key.replace(".intermediate.dense.", ".mlp.dense1.")
        key = key.replace(".output.dense.", ".mlp.dense2.")

    return key


def convert_ast_weights(pytorch_path: Path) -> Dict[str, np.ndarray]:
    """Convert AST PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to pytorch_model.bin

    Returns:
        Dictionary of MLX weights
    """
    weights = torch.load(pytorch_path, map_location="cpu", weights_only=True)

    mlx_weights = {}

    for pt_key, tensor in weights.items():
        # Map key
        mlx_key = map_weight_key(pt_key)

        # Convert tensor
        if "patch_embeddings.weight" in mlx_key:
            # Conv2d weight needs special handling
            value = convert_conv2d_weight(tensor)
        else:
            value = tensor.numpy()

        mlx_weights[mlx_key] = value
        print(f"  {pt_key} -> {mlx_key}: {list(tensor.shape)} -> {list(value.shape)}")

    return mlx_weights


def main():
    """Main conversion function."""
    input_dir = Path("models/sota/ast")
    output_dir = Path("models/sota/ast-mlx")

    print(f"Converting AST from {input_dir} to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert weights
    print("\nConverting weights...")
    weights = convert_ast_weights(input_dir / "pytorch_model.bin")
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

        # Save simplified config
        simplified_config = {
            "hidden_size": config.get("hidden_size", 768),
            "num_hidden_layers": config.get("num_hidden_layers", 12),
            "num_attention_heads": config.get("num_attention_heads", 12),
            "intermediate_size": config.get("intermediate_size", 3072),
            "hidden_act": config.get("hidden_act", "gelu"),
            "hidden_dropout_prob": config.get("hidden_dropout_prob", 0.0),
            "attention_probs_dropout_prob": config.get("attention_probs_dropout_prob", 0.0),
            "patch_size": config.get("patch_size", 16),
            "num_mel_bins": config.get("num_mel_bins", 128),
            "max_length": config.get("max_length", 1024),
            "time_stride": config.get("time_stride", 10),
            "frequency_stride": config.get("frequency_stride", 10),
            "num_labels": len(config.get("id2label", {})) or 527,
            "qkv_bias": config.get("qkv_bias", True),
            "layer_norm_eps": config.get("layer_norm_eps", 1e-12),
        }

        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(simplified_config, f, indent=2)
        print(f"Saved config to {config_path}")

        # Save labels
        if "id2label" in config:
            labels_path = output_dir / "labels.json"
            with open(labels_path, "w") as f:
                json.dump(config["id2label"], f, indent=2)
            print(f"Saved labels to {labels_path}")

    # Print statistics
    print("\nWeight statistics:")
    total_params = sum(v.size for v in weights.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
