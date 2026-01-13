#!/usr/bin/env python3
"""Export Zipformer weights to safetensors format for C++ inference.

This script loads a Zipformer checkpoint and exports weights in a format
compatible with the C++ MLX inference code.

Usage:
    python scripts/export_zipformer_weights.py \
        --checkpoint checkpoints/zipformer/en-streaming/exp/pretrained.pt \
        --output checkpoints/zipformer/en-streaming/weights.safetensors
"""

import argparse
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)


def flatten_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Flatten nested state dict keys with dots."""
    flat = {}
    for key, value in state_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_state_dict(value, full_key))
        elif isinstance(value, torch.Tensor):
            flat[full_key] = value
    return flat


def convert_key_names(state_dict: dict) -> dict:
    """Convert PyTorch key names to C++ compatible names.

    The C++ code expects keys like:
    - encoder_embed.conv1_weight
    - encoders.0.layers.0.self_attn.in_proj.weight
    """
    converted = {}

    for key, value in state_dict.items():
        # Skip non-tensor values
        if not isinstance(value, torch.Tensor):
            continue

        # Convert to float32 for compatibility
        if value.dtype in (torch.float16, torch.bfloat16):
            value = value.float()

        # Key remapping for C++ compatibility
        new_key = key

        # Handle common patterns
        # model.encoder -> encoder
        if new_key.startswith("model."):
            new_key = new_key[6:]

        # encoder.encoder_embed -> encoder_embed (for top-level encoder)
        if new_key.startswith("encoder."):
            new_key = new_key[8:]

        converted[new_key] = value

    return converted


def analyze_checkpoint(checkpoint_path: Path) -> None:
    """Analyze checkpoint structure and print key statistics."""
    print(f"\nAnalyzing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        print(f"Top-level keys: {list(checkpoint.keys())}")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        print(f"Checkpoint type: {type(checkpoint)}")
        return

    # Count parameters
    total_params = 0
    layer_counts = {}

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            total_params += value.numel()

            # Count by layer type
            parts = key.split(".")
            if len(parts) >= 2:
                layer_type = parts[0]
                layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1

    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Number of tensors: {len(state_dict)}")
    print("\nLayer type counts:")
    for layer_type, count in sorted(layer_counts.items()):
        print(f"  {layer_type}: {count}")

    # Print sample keys
    print("\nSample keys (first 20):")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        value = state_dict[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")


def export_weights(
    checkpoint_path: Path,
    output_path: Path,
    analyze_only: bool = False
) -> None:
    """Export weights from checkpoint to safetensors."""

    if analyze_only:
        analyze_checkpoint(checkpoint_path)
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    # Flatten and convert
    flat_dict = flatten_state_dict(state_dict)
    converted = convert_key_names(flat_dict)

    # Ensure all tensors are contiguous
    for key in converted:
        converted[key] = converted[key].contiguous()

    # Save
    print(f"Saving {len(converted)} tensors to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))

    # Print size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Output size: {size_mb:.2f} MB")

    # Print key summary
    print("\nExported keys (sample):")
    for i, key in enumerate(list(converted.keys())[:10]):
        tensor = converted[key]
        print(f"  {key}: {list(tensor.shape)} {tensor.dtype}")
    if len(converted) > 10:
        print(f"  ... and {len(converted) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Export Zipformer weights to safetensors"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PyTorch checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for safetensors file"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze checkpoint structure, don't export"
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if args.analyze:
        analyze_checkpoint(args.checkpoint)
    else:
        if args.output is None:
            args.output = args.checkpoint.with_suffix(".safetensors")
        export_weights(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
