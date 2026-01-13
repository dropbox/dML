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
Inspect PyTorch checkpoint generator structure.
"""

import torch
from huggingface_hub import hf_hub_download


def main():
    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    pt_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    print("=== ALL Generator keys ===")
    gen_keys = sorted([k for k in pt_state.keys() if "generator" in k])
    for k in gen_keys:
        print(f"  {k}: {pt_state[k].shape}")

    print(f"\n=== Total generator keys: {len(gen_keys)} ===")


if __name__ == "__main__":
    main()
