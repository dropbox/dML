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

"""Find SourceModule weights in PyTorch checkpoint."""

from pathlib import Path

import torch

model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

print("Looking for source/sine/harmonic related weights:")
for key in sorted(state_dict.keys()):
    k_lower = key.lower()
    if any(x in k_lower for x in ["source", "sine", "l_linear", "m_source", "har"]):
        weight = state_dict[key]
        print(f"{key}: shape={weight.shape}")

print("\nAll decoder.generator keys:")
for key in sorted(state_dict.keys()):
    if key.startswith("decoder.generator"):
        weight = state_dict[key]
        print(f"{key}: shape={weight.shape}")
