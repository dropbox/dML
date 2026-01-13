#!/usr/bin/env python3
"""
Export Fish-Speech DualARTransformer weights for C++ libtorch loading.

Uses safetensors format which is efficient and well-supported.

Usage:
    python3 scripts/export_fish_transformer_weights.py

Output:
    models/fish-speech-1.5/transformer_weights.safetensors - Weights file
    models/fish-speech-1.5/transformer_config_cpp.json - Config for C++ loading

Author: Worker #483 (AI)
Date: 2025-12-11
"""

import os
import sys
import json
import torch
from typing import Dict

def main():
    checkpoint_path = "models/fish-speech-1.5"
    output_safetensors = f"{checkpoint_path}/transformer_weights.safetensors"
    config_path = f"{checkpoint_path}/transformer_config_cpp.json"

    # Load config
    with open(f"{checkpoint_path}/config.json") as f:
        config = json.load(f)

    print("Model config:")
    print(f"  dim: {config['dim']}")
    print(f"  vocab_size: {config['vocab_size']}")
    print(f"  n_layer: {config['n_layer']}")
    print(f"  n_fast_layer: {config['n_fast_layer']}")

    # Load weights
    print(f"\nLoading weights from {checkpoint_path}/model.pth...")
    state_dict = torch.load(f"{checkpoint_path}/model.pth", map_location="cpu", weights_only=True)
    print(f"Loaded {len(state_dict)} weight tensors")

    # Try safetensors
    try:
        from safetensors.torch import save_file
        print(f"\nSaving to {output_safetensors} (safetensors format)...")

        # Convert bfloat16 to float32 for better compatibility
        converted: Dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if tensor.dtype == torch.bfloat16:
                converted[name] = tensor.to(torch.float32)
            else:
                converted[name] = tensor.to(torch.float32) if tensor.is_floating_point() else tensor

        save_file(converted, output_safetensors)
        total_size = os.path.getsize(output_safetensors)
        print(f"Saved {len(converted)} weights ({total_size / 1e6:.1f} MB)")
        weight_format = "safetensors"

    except ImportError:
        print("\nError: safetensors not installed. Install with:")
        print("  pip install safetensors")
        return 1

    # Get special token IDs from tokenizer
    try:
        from fish_speech.tokenizer import FishTokenizer
        tokenizer = FishTokenizer.from_pretrained(checkpoint_path)
        im_end_id = tokenizer.get_token_id("<|im_end|>")
        semantic_begin_id = tokenizer.semantic_begin_id
        semantic_end_id = tokenizer.semantic_end_id
        print(f"\nSpecial tokens:")
        print(f"  im_end_id: {im_end_id}")
        print(f"  semantic_begin_id: {semantic_begin_id}")
        print(f"  semantic_end_id: {semantic_end_id}")
    except Exception as e:
        print(f"\nWarning: Could not load tokenizer: {e}")
        print("Using default special token IDs")
        im_end_id = 151645  # Common im_end for tiktoken-based tokenizers
        semantic_begin_id = 100864
        semantic_end_id = 100865

    # Create config for C++
    cpp_config = {
        **config,
        "im_end_id": im_end_id,
        "semantic_begin_id": semantic_begin_id,
        "semantic_end_id": semantic_end_id,
        "weight_format": weight_format,
        "weight_file": os.path.basename(output_safetensors),
        "num_weights": len(state_dict),
    }

    with open(config_path, "w") as f:
        json.dump(cpp_config, f, indent=2)
    print(f"\nSaved {config_path}")

    # Print weight info
    print("\nWeight tensor summary:")
    for name in sorted(state_dict.keys())[:15]:
        tensor = state_dict[name]
        print(f"  {name}: {list(tensor.shape)} {tensor.dtype}")
    print(f"  ... ({len(state_dict) - 15} more)")

    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print(f"Weights: {output_safetensors}")
    print(f"Config:  {config_path}")
    print("\nTo load in C++, use safetensors library or convert to torch format.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
