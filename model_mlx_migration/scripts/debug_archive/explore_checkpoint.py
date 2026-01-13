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

"""Explore PyTorch checkpoint structure."""

from pathlib import Path

import torch

model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)


def explore_dict(d, prefix="", depth=0, max_depth=3):
    """Recursively explore dict structure."""
    if depth > max_depth:
        return
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            print(f"{'  ' * depth}{key}: dict with {len(value)} keys")
            explore_dict(value, full_key, depth + 1, max_depth)
        elif isinstance(value, torch.Tensor):
            print(f"{'  ' * depth}{key}: Tensor {value.shape}")
        else:
            print(f"{'  ' * depth}{key}: {type(value).__name__}")


print("Checkpoint structure:")
print(f"Type: {type(checkpoint)}")
print()

if isinstance(checkpoint, dict):
    explore_dict(checkpoint, max_depth=2)

    # Look for model weights
    if "net" in checkpoint:
        print("\n\nExploring 'net' key (likely model weights):")
        explore_dict(checkpoint["net"], prefix="net", max_depth=3)

    if "model" in checkpoint:
        print("\n\nExploring 'model' key:")
        explore_dict(checkpoint["model"], prefix="model", max_depth=3)

# Search for source-related keys
print("\n\n=== Searching for source/l_linear ===")


def find_keys(d, prefix=""):
    results = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            results.extend(find_keys(value, full_key))
        elif isinstance(value, torch.Tensor):
            if any(x in key.lower() for x in ["source", "l_linear", "sine"]):
                results.append((full_key, value.shape))
    return results


if isinstance(checkpoint, dict):
    source_keys = find_keys(checkpoint)
    for k, shape in source_keys:
        print(f"{k}: {shape}")

# Find all tensor keys
print("\n\n=== All tensor paths ===")


def get_all_tensors(d, prefix=""):
    results = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            results.extend(get_all_tensors(value, full_key))
        elif isinstance(value, torch.Tensor):
            results.append((full_key, value.shape))
    return results


if isinstance(checkpoint, dict):
    all_tensors = get_all_tensors(checkpoint)
    print(f"Total tensors: {len(all_tensors)}")

    # Group by top-level component
    from collections import defaultdict

    by_component = defaultdict(list)
    for path, shape in all_tensors:
        parts = path.split(".")
        component = parts[0] if len(parts) > 0 else "root"
        by_component[component].append((path, shape))

    for component, items in sorted(by_component.items()):
        print(f"\n{component}: {len(items)} tensors")
        for path, shape in items[:5]:
            print(f"  {path}: {shape}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")
