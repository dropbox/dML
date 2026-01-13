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

"""Check PyTorch state dict keys for AdaIN."""

import torch
from huggingface_hub import hf_hub_download


def main():
    # Download model checkpoint
    model_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    print("Looking for norm keys in decoder...")
    for key in sorted(state.keys()):
        if "decode" in key and "norm" in key:
            print(f"  {key}: {state[key].shape}")

    print("\nLooking for norm keys in encoder...")
    for key in sorted(state.keys()):
        if "encode" in key and "norm" in key:
            print(f"  {key}: {state[key].shape}")

    print("\nLooking for all norm1 keys...")
    for key in sorted(state.keys()):
        if "norm1" in key:
            print(f"  {key}: {state[key].shape}")

    print("\nLooking for norm.weight or norm.bias keys...")
    for key in sorted(state.keys()):
        if "norm.weight" in key or "norm.bias" in key:
            print(f"  {key}: {state[key].shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
