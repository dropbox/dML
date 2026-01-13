#!/usr/bin/env python3
"""Convert Emotion2vec to MLX format.

This script converts the FunASR emotion2vec_base checkpoint to MLX format.

Usage:
    python scripts/convert_emotion2vec_to_mlx.py

Input:
    models/sota/emotion2vec/emotion2vec_base.pt

Output:
    models/sota/emotion2vec-mlx/weights.npz
    models/sota/emotion2vec-mlx/config.json
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

    Args:
        weight: PyTorch weight tensor

    Returns:
        NumPy array in MLX format
    """
    # Transpose from (O, I, K) to (O, K, I)
    return weight.permute(0, 2, 1).numpy()


def map_weight_key(pt_key: str) -> str:
    """Map FunASR emotion2vec key to our MLX model key.

    FunASR structure:
        modality_encoders.AUDIO.extra_tokens
        modality_encoders.AUDIO.alibi_scale
        modality_encoders.AUDIO.local_encoder.conv_layers.N.0.weight    (Conv1d weight)
        modality_encoders.AUDIO.local_encoder.conv_layers.N.2.1.weight  (LayerNorm weight)
        modality_encoders.AUDIO.local_encoder.conv_layers.N.2.1.bias    (LayerNorm bias)
        modality_encoders.AUDIO.project_features.1.weight               (LayerNorm weight)
        modality_encoders.AUDIO.project_features.1.bias                 (LayerNorm bias)
        modality_encoders.AUDIO.project_features.2.weight               (Linear weight)
        modality_encoders.AUDIO.project_features.2.bias                 (Linear bias)
        modality_encoders.AUDIO.relative_positional_encoder.N.0.weight  (Conv1d weight)
        modality_encoders.AUDIO.relative_positional_encoder.N.0.bias    (Conv1d bias)
        modality_encoders.AUDIO.context_encoder.blocks.N.attn.qkv.weight
        modality_encoders.AUDIO.context_encoder.blocks.N.attn.proj.weight
        modality_encoders.AUDIO.context_encoder.blocks.N.mlp.fc1.weight
        modality_encoders.AUDIO.context_encoder.blocks.N.mlp.fc2.weight
        modality_encoders.AUDIO.context_encoder.blocks.N.norm1.weight
        modality_encoders.AUDIO.context_encoder.blocks.N.norm2.weight
        modality_encoders.AUDIO.context_encoder.norm.weight
        blocks.N.attn.qkv.weight
        blocks.N.attn.proj.weight
        blocks.N.mlp.fc1.weight
        blocks.N.mlp.fc2.weight
        blocks.N.norm1.weight
        blocks.N.norm2.weight

    Our MLX structure:
        extra_tokens
        encoder.alibi_scale
        local_encoder.conv_layers.N.conv.weight
        local_encoder.conv_layers.N.layer_norm.weight
        local_encoder.conv_layers.N.layer_norm.bias
        project_features.layer_norm.weight
        project_features.layer_norm.bias
        project_features.projection.weight
        project_features.projection.bias
        relative_positional_encoder.conv_layers.N.weight
        relative_positional_encoder.conv_layers.N.bias
        encoder.context_encoder.N.qkv.weight
        encoder.context_encoder.N.proj.weight
        encoder.context_encoder.N.fc1.weight
        encoder.context_encoder.N.fc2.weight
        encoder.context_encoder.N.norm1.weight
        encoder.context_encoder.N.norm2.weight
        encoder.context_norm.weight
        encoder.blocks.N.qkv.weight
        encoder.blocks.N.proj.weight
        encoder.blocks.N.fc1.weight
        encoder.blocks.N.fc2.weight
        encoder.blocks.N.norm1.weight
        encoder.blocks.N.norm2.weight
    """
    key = pt_key

    # Extra tokens
    if key == "modality_encoders.AUDIO.extra_tokens":
        return "extra_tokens"

    # ALiBi scale
    if key == "modality_encoders.AUDIO.alibi_scale":
        return "encoder.alibi_scale"

    # Local encoder conv layers
    # modality_encoders.AUDIO.local_encoder.conv_layers.N.0.weight -> local_encoder.conv_layers.N.conv.weight
    # modality_encoders.AUDIO.local_encoder.conv_layers.N.2.1.weight -> local_encoder.conv_layers.N.layer_norm.weight
    if "local_encoder.conv_layers" in key:
        key = key.replace("modality_encoders.AUDIO.", "")
        # Conv weight: .N.0.weight -> .N.conv.weight
        key = key.replace(".0.weight", ".conv.weight")
        # LayerNorm: .N.2.1.weight -> .N.layer_norm.weight
        key = key.replace(".2.1.weight", ".layer_norm.weight")
        key = key.replace(".2.1.bias", ".layer_norm.bias")
        return key

    # Project features
    # modality_encoders.AUDIO.project_features.1.weight -> project_features.layer_norm.weight
    # modality_encoders.AUDIO.project_features.2.weight -> project_features.projection.weight
    if "project_features" in key:
        key = key.replace("modality_encoders.AUDIO.", "")
        key = key.replace(".1.weight", ".layer_norm.weight")
        key = key.replace(".1.bias", ".layer_norm.bias")
        key = key.replace(".2.weight", ".projection.weight")
        key = key.replace(".2.bias", ".projection.bias")
        return key

    # Relative positional encoder
    # modality_encoders.AUDIO.relative_positional_encoder.N.0.weight -> relative_positional_encoder.conv_layers.{N-1}.weight
    # Note: PyTorch indices start at 1 (layers 1,2,3,4,5), MLX indices start at 0
    if "relative_positional_encoder" in key:
        key = key.replace("modality_encoders.AUDIO.", "")
        # Extract layer index and adjust (PyTorch uses 1-indexed)
        import re
        match = re.match(r"relative_positional_encoder\.(\d+)\.0\.(weight|bias)", key)
        if match:
            layer_idx = int(match.group(1)) - 1  # Convert from 1-indexed to 0-indexed
            param_type = match.group(2)
            return f"relative_positional_encoder.conv_layers.{layer_idx}.{param_type}"
        return key

    # Context encoder blocks
    # modality_encoders.AUDIO.context_encoder.blocks.N.attn.qkv.weight -> encoder.context_encoder.N.qkv.weight
    if "context_encoder.blocks" in key:
        key = key.replace("modality_encoders.AUDIO.context_encoder.blocks.", "encoder.context_encoder.")
        key = key.replace(".attn.", ".")
        key = key.replace(".mlp.", ".")
        return key

    # Context encoder norm
    if key == "modality_encoders.AUDIO.context_encoder.norm.weight":
        return "encoder.context_norm.weight"
    if key == "modality_encoders.AUDIO.context_encoder.norm.bias":
        return "encoder.context_norm.bias"

    # Main encoder blocks
    # blocks.N.attn.qkv.weight -> encoder.blocks.N.qkv.weight
    if key.startswith("blocks."):
        key = "encoder." + key
        key = key.replace(".attn.", ".")
        key = key.replace(".mlp.", ".")
        return key

    return key


def is_conv1d_weight(pt_key: str) -> bool:
    """Check if a key corresponds to a Conv1d weight."""
    # Local encoder conv weights
    if "local_encoder.conv_layers" in pt_key and ".0.weight" in pt_key:
        return True
    # Relative positional encoder conv weights
    if "relative_positional_encoder" in pt_key and ".0.weight" in pt_key:
        return True
    return False


def convert_emotion2vec_weights(pytorch_path: Path) -> Dict[str, np.ndarray]:
    """Convert Emotion2vec PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to emotion2vec_base.pt

    Returns:
        Dictionary of MLX weights
    """
    checkpoint = torch.load(pytorch_path, map_location="cpu", weights_only=False)

    # Get model weights from checkpoint
    weights = checkpoint.get("model", checkpoint)

    mlx_weights = {}
    skipped_keys = []

    for pt_key, tensor in weights.items():
        # Skip non-model keys and decoder keys (not needed for inference)
        if any(skip in pt_key for skip in ["decoder", "ema_", "target_", "modality_decoders"]):
            skipped_keys.append(pt_key)
            continue

        # Map key
        mlx_key = map_weight_key(pt_key)

        # Convert tensor
        if is_conv1d_weight(pt_key):
            # Conv1d weight needs special handling
            value = convert_conv1d_weight(tensor)
        else:
            value = tensor.numpy()

        mlx_weights[mlx_key] = value
        print(f"  {pt_key} -> {mlx_key}: {list(tensor.shape)} -> {list(value.shape)}")

    print(f"\nSkipped {len(skipped_keys)} keys (decoder/EMA/target)")

    return mlx_weights


def main():
    """Main conversion function."""
    input_dir = Path("models/sota/emotion2vec")
    output_dir = Path("models/sota/emotion2vec-mlx")

    print(f"Converting Emotion2vec from {input_dir} to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert weights
    print("\nConverting weights...")
    weights = convert_emotion2vec_weights(input_dir / "emotion2vec_base.pt")
    print(f"Converted {len(weights)} parameters")

    # Save weights
    weights_path = output_dir / "weights.npz"
    np.savez(str(weights_path), **weights)
    print(f"\nSaved weights to {weights_path}")

    # Save config
    config = {
        "embed_dim": 768,
        "depth": 8,
        "prenet_depth": 4,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "num_extra_tokens": 10,
        "layer_norm_eps": 1e-5,
        "layer_norm_first": False,
        "local_encoder_spec": [
            [512, 10, 5],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 2, 2],
            [512, 2, 2],
        ],
        "conv_pos_width": 95,
        "conv_pos_depth": 5,
        "conv_pos_groups": 16,
        "use_alibi_encoder": True,
        "learned_alibi_scale": True,
        "learned_alibi_scale_per_head": True,
        "num_alibi_heads": 12,
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
