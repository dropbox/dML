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

"""Check if convolution dilations match PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import torch

print("=== PyTorch resblocks conv weights - check for dilation patterns ===")
model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
dec_state = checkpoint.get("decoder", {})

# Check resblocks.0 convs - kernel size should indicate dilation pattern
for key in sorted(dec_state.keys()):
    if "resblocks.0.convs1" in key and "weight_v" in key:
        val = dec_state[key]
        print(f"{key}: shape={val.shape}")
    if "resblocks.1.convs1" in key and "weight_v" in key:
        val = dec_state[key]
        print(f"{key}: shape={val.shape}")

print("\n=== Understanding dilation patterns ===")
print("ISTFTNet uses resblock with dilations (1, 3, 5) for each kernel")
print("The dilations should be applied as dilation parameter in Conv1d")
print("This affects receptive field but weight shapes stay the same")

print("\n=== Let me check the actual PyTorch code's dilation usage ===")
# Dilations are (1, 3, 5) per StyleTTS2/ISTFTNet
# For kernel=3 with dilation=1: effective kernel = 3
# For kernel=3 with dilation=3: effective kernel = 7 (every 3rd)
# For kernel=3 with dilation=5: effective kernel = 11 (every 5th)

print("Typical dilation pattern: (1, 3, 5)")
print("For kernel_size=3:")
print("  dilation=1: padding=1 (standard)")
print("  dilation=3: padding=3 (to maintain same size)")
print("  dilation=5: padding=5")

# Check if MLX module has dilation parameter
import inspect

from converters.models.kokoro_modules import WeightNormConv1d

print("\n=== WeightNormConv1d signature ===")
print(inspect.signature(WeightNormConv1d.__init__))
