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
Export CosyVoice3 weights to safetensors format for C++ loading.

Downloads the model from HuggingFace if needed and exports:
- flow.safetensors: DiT Flow model weights
- hift.safetensors: CausalHiFT vocoder weights
"""

import sys
import torch
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
EXPORT_DIR = Path.home() / ".cache" / "cosyvoice3_export"


def download_checkpoint():
    """Download CosyVoice3 checkpoint from HuggingFace."""
    print(f"Downloading {MODEL_ID}...")
    cache_dir = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["flow.pt", "hift.pt", "*.yaml", "*.json"],
    )
    print(f"Downloaded to: {cache_dir}")
    return cache_dir


def export_flow_weights(checkpoint_path: Path, export_dir: Path):
    """Export flow model weights to safetensors."""
    flow_path = checkpoint_path / "flow.pt"
    if not flow_path.exists():
        print(f"Error: {flow_path} not found")
        return False

    print(f"Loading flow weights from {flow_path}...")
    state_dict = torch.load(flow_path, map_location="cpu", weights_only=True)

    # Print some info
    print(f"  Keys: {len(state_dict)}")
    total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
    print(f"  Total params: {total_params:,}")

    # Convert to float32 for safetensors
    export_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            export_dict[key] = value.float()

    # Save
    export_path = export_dir / "flow.safetensors"
    print(f"Saving to {export_path}...")
    save_file(export_dict, export_path)
    print(f"  Saved {len(export_dict)} tensors")
    return True


def export_hift_weights(checkpoint_path: Path, export_dir: Path):
    """Export hift vocoder weights to safetensors."""
    hift_path = checkpoint_path / "hift.pt"
    if not hift_path.exists():
        print(f"Error: {hift_path} not found")
        return False

    print(f"Loading hift weights from {hift_path}...")
    state_dict = torch.load(hift_path, map_location="cpu", weights_only=True)

    # Print some info
    print(f"  Keys: {len(state_dict)}")
    total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
    print(f"  Total params: {total_params:,}")

    # Convert to float32 for safetensors
    export_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            export_dict[key] = value.float()

    # Save
    export_path = export_dir / "hift.safetensors"
    print(f"Saving to {export_path}...")
    save_file(export_dict, export_path)
    print(f"  Saved {len(export_dict)} tensors")
    return True


def print_weight_keys(export_dir: Path):
    """Print weight keys to help with mapping."""
    from safetensors import safe_open

    for name in ["flow.safetensors", "hift.safetensors"]:
        path = export_dir / name
        if path.exists():
            print(f"\n=== {name} ===")
            with safe_open(path, framework="pt") as f:
                keys = list(f.keys())
                for i, key in enumerate(keys[:50]):
                    tensor = f.get_tensor(key)
                    print(f"  {key}: {list(tensor.shape)}")
                if len(keys) > 50:
                    print(f"  ... and {len(keys) - 50} more keys")


def main():
    # Create export directory
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Download if needed
    checkpoint_path = Path(download_checkpoint())

    # Export weights
    print("\n=== Exporting weights ===")
    flow_ok = export_flow_weights(checkpoint_path, EXPORT_DIR)
    hift_ok = export_hift_weights(checkpoint_path, EXPORT_DIR)

    if flow_ok and hift_ok:
        print("\n=== Export complete ===")
        print(f"Weights saved to: {EXPORT_DIR}")
        print_weight_keys(EXPORT_DIR)
        return 0
    else:
        print("\n=== Export failed ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
