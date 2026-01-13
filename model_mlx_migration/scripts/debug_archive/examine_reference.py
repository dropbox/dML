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
Examine the reference tensors to understand the gen_har format.
"""

import numpy as np

ref_path = "/tmp/kokoro_ref_deterministic_source/tensors.npz"
ref = np.load(ref_path)

print("Available keys:", list(ref.files))
print()

for key in ref.files:
    arr = ref[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    print()

# Look at gen_har specifically
if "gen_har" in ref:
    gen_har = ref["gen_har"]
    print("=" * 72)
    print("gen_har analysis:")
    print("=" * 72)
    print(f"Shape: {gen_har.shape}")  # Expected: [batch, 22, frames]

    # Split into magnitude (first 11) and phase (last 11)
    mag = gen_har[:, :11, :]  # [batch, 11, frames]
    phase = gen_har[:, 11:, :]  # [batch, 11, frames]

    print(f"\nMagnitude: shape={mag.shape}")
    print(f"  min={mag.min():.4f}, max={mag.max():.4f}")
    print(f"  Always positive? {(mag >= 0).all()}")

    print(f"\nPhase: shape={phase.shape}")
    print(f"  min={phase.min():.4f}, max={phase.max():.4f}")
    print(f"  In [-π, π]? {(phase >= -np.pi).all() and (phase <= np.pi).all()}")

    # Check for boundary values (±π)
    boundary_mask = np.abs(np.abs(phase) - np.pi) < 0.01
    print(f"  Bins at ±π boundary: {boundary_mask.sum()} / {phase.size} ({100*boundary_mask.sum()/phase.size:.1f}%)")

# Look at har_source
if "gen_har_source" in ref:
    har_source = ref["gen_har_source"]
    print("\n" + "=" * 72)
    print("gen_har_source analysis:")
    print("=" * 72)
    print(f"Shape: {har_source.shape}")
    print(f"min={har_source.min():.4f}, max={har_source.max():.4f}")

    # This should be the raw waveform before STFT
    # Check if it's a sine-like wave
    flat = har_source.flatten()
    print(f"Sample values: {flat[:10]}")
