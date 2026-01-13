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
Analyze the resblock traces to understand the internal structure.
"""

import numpy as np


def main():
    internal = np.load("/tmp/kokoro_ref/generator_internal_traces.npz")

    print("=== Resblock Trace Analysis ===\n")

    for i in range(6):
        rb_in = internal[f"resblock_{i}_in"]
        rb_out = internal[f"resblock_{i}_out"]

        # Compute statistics
        in_std = rb_in.std()
        out_std = rb_out.std()
        amp = out_std / in_std

        # Check if it's residual connection: out = f(in) + in
        # If so, residual = out - in should be well-structured
        residual = rb_out - rb_in
        residual_std = residual.std()

        # Correlation
        corr = np.corrcoef(rb_out.flatten(), rb_in.flatten())[0, 1]

        print(f"Resblock {i}:")
        print(f"  Input: shape {rb_in.shape}, std={in_std:.4f}")
        print(f"  Output: std={out_std:.4f}, amplification={amp:.4f}")
        print(f"  Residual (out-in): std={residual_std:.4f}")
        print(f"  Output-Input correlation: {corr:.4f}")
        print()

    # Check if inputs to consecutive resblocks are the same
    print("=== Checking if resblock inputs are shared ===")
    for i in range(3):  # Stage 0
        if i > 0:
            corr = np.corrcoef(
                internal[f"resblock_{i}_in"].flatten(),
                internal[f"resblock_{i - 1}_in"].flatten(),
            )[0, 1]
            print(f"resblock_{i}_in vs resblock_{i - 1}_in correlation: {corr:.6f}")

    print("\n=== Generator Flow Analysis ===")

    # ups_0_out + noise_res_0_out should feed into resblocks 0,1,2
    ups_0 = internal["ups_0_out"]
    noise_0 = internal["noise_res_0_out"]
    combined_0 = ups_0 + noise_0

    # Is combined_0 the same as resblock_0_in?
    rb0_in = internal["resblock_0_in"]
    corr = np.corrcoef(combined_0.flatten(), rb0_in.flatten())[0, 1]
    print(f"ups_0 + noise_0 vs resblock_0_in correlation: {corr:.6f}")
    print(
        f"  Combined shape: {combined_0.shape}, range [{combined_0.min():.4f}, {combined_0.max():.4f}]"
    )
    print(
        f"  resblock_0_in shape: {rb0_in.shape}, range [{rb0_in.min():.4f}, {rb0_in.max():.4f}]"
    )

    # Check stage 1
    ups_1 = internal["ups_1_out"]
    noise_1 = internal["noise_res_1_out"]

    print(f"\nups_1_out shape: {ups_1.shape}")
    print(f"noise_res_1_out shape: {noise_1.shape}")
    print(f"resblock_3_in shape: {internal['resblock_3_in'].shape}")

    # Combined should go to resblock 3,4,5
    # But need to add reflection padding first
    print(f"\nups_1 length: {ups_1.shape[2]}")
    print(f"noise_1 length: {noise_1.shape[2]}")
    print(f"resblock_3_in length: {internal['resblock_3_in'].shape[2]}")

    # Reflection pad adds 1 element: F.pad(x, (1, 0), mode='reflect')
    # So ups_1 would become shape [..., 7561]
    print(f"\nups_1 + 1 (reflect pad) = {ups_1.shape[2] + 1}")


if __name__ == "__main__":
    main()
