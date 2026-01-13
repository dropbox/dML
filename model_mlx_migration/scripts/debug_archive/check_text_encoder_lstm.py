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

"""Check text_encoder LSTM weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import torch

print("=== PyTorch text_encoder LSTM weights ===")
model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
pred_state = checkpoint.get("predictor", {})

# Check lstms.0 weights
for key in sorted(pred_state.keys()):
    if "text_encoder.lstms.0" in key:
        val = pred_state[key]
        print(
            f"{key}: shape={val.shape}, range=[{val.min().item():.4f}, {val.max().item():.4f}]"
        )

print("\n=== Loading MLX model ===")
from converters.kokoro_converter import KokoroConverter

converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

# Check MLX lstms_0 weights
lstm = model.predictor.text_encoder.lstms_0
print("\n=== MLX text_encoder.lstms_0 weights ===")
print(
    f"weight_ih_l0: shape={lstm.weight_ih_l0.shape}, range=[{float(lstm.weight_ih_l0.min()):.4f}, {float(lstm.weight_ih_l0.max()):.4f}]"
)
print(
    f"weight_hh_l0: shape={lstm.weight_hh_l0.shape}, range=[{float(lstm.weight_hh_l0.min()):.4f}, {float(lstm.weight_hh_l0.max()):.4f}]"
)

# Compare specific values
pt_ih = pred_state["module.text_encoder.lstms.0.weight_ih_l0"]
pt_hh = pred_state["module.text_encoder.lstms.0.weight_hh_l0"]

print("\n=== Comparison ===")
print(f"PyTorch weight_ih_l0[:3,:3]: {pt_ih[:3, :3].numpy()}")
print(f"MLX weight_ih_l0[:3,:3]: {lstm.weight_ih_l0[:3, :3].tolist()}")

# Test LSTM on simple input
print("\n=== Test LSTM on sequence ===")
# Different input for each position
test_input = mx.random.normal((1, 7, 640))
mx.eval(test_input)

print(f"Input: shape={test_input.shape}")
print(f"Input[0,0,:5]: {test_input[0, 0, :5].tolist()}")
print(f"Input[0,1,:5]: {test_input[0, 1, :5].tolist()}")
print(f"Input[0,2,:5]: {test_input[0, 2, :5].tolist()}")

output = lstm(test_input)
mx.eval(output)

print(f"\nOutput: shape={output.shape}")
print(f"Output[0,0,:5]: {output[0, 0, :5].tolist()}")
print(f"Output[0,1,:5]: {output[0, 1, :5].tolist()}")
print(f"Output[0,2,:5]: {output[0, 2, :5].tolist()}")

# Are outputs different for different positions?
diff_01 = float(mx.abs(output[0, 0, :] - output[0, 1, :]).max())
diff_12 = float(mx.abs(output[0, 1, :] - output[0, 2, :]).max())
print(f"\nMax diff between position 0 and 1: {diff_01:.6f}")
print(f"Max diff between position 1 and 2: {diff_12:.6f}")

# Test the actual text_encoder
print("\n=== Test full text_encoder ===")
from converters.kokoro_converter import KokoroConverter

converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

# Create distinct input for each position
distinct_input = mx.stack([mx.random.normal((512,)) for _ in range(7)], axis=0)[
    None, :, :
]
mx.eval(distinct_input)

style = mx.random.normal((1, 128))
mx.eval(style)

print(f"Distinct input shape: {distinct_input.shape}")
print(
    f"Input means per position: {[f'{float(mx.mean(distinct_input[0, i, :])):.4f}' for i in range(7)]}"
)

output = model.predictor.text_encoder(distinct_input, style)
mx.eval(output)

print(f"Output shape: {output.shape}")
print(
    f"Output means per position: {[f'{float(mx.mean(output[0, i, :])):.4f}' for i in range(7)]}"
)

# Check if outputs are identical
all_same = True
for i in range(1, 7):
    diff = float(mx.abs(output[0, 0, :] - output[0, i, :]).max())
    if diff > 0.0001:
        all_same = False
    print(f"  Max diff pos 0 vs pos {i}: {diff:.6f}")

print(f"\nAll outputs identical? {all_same}")
