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

"""Check variance calculation in PyTorch InstanceNorm1d."""

import numpy as np
import torch
import torch.nn as nn


def main():
    # Create test input
    np.random.seed(42)
    x = torch.randn(1, 4, 10)  # NCL format
    print(f"Input shape: {x.shape}")

    # PyTorch InstanceNorm1d
    norm = nn.InstanceNorm1d(4, affine=True, eps=1e-5)
    norm.eval()
    out_pt = norm(x)

    # Manual calculation with sample variance (ddof=1)
    mean = x.mean(dim=2, keepdim=True)
    var_sample = x.var(dim=2, keepdim=True, unbiased=True)  # ddof=1
    out_manual_sample = (x - mean) / torch.sqrt(var_sample + 1e-5)
    out_manual_sample = norm.weight[:, None] * out_manual_sample + norm.bias[:, None]

    # Manual calculation with population variance (ddof=0)
    var_pop = x.var(dim=2, keepdim=True, unbiased=False)  # ddof=0
    out_manual_pop = (x - mean) / torch.sqrt(var_pop + 1e-5)
    out_manual_pop = norm.weight[:, None] * out_manual_pop + norm.bias[:, None]

    print(f"\nPyTorch InstanceNorm1d output range: [{out_pt.min():.6f}, {out_pt.max():.6f}]")
    print(f"Manual (sample var ddof=1) range: [{out_manual_sample.min():.6f}, {out_manual_sample.max():.6f}]")
    print(f"Manual (pop var ddof=0) range: [{out_manual_pop.min():.6f}, {out_manual_pop.max():.6f}]")

    # Compare
    diff_sample = (out_pt - out_manual_sample).abs().max()
    diff_pop = (out_pt - out_manual_pop).abs().max()

    print(f"\nDiff vs sample var (ddof=1): {diff_sample:.6f}")
    print(f"Diff vs pop var (ddof=0): {diff_pop:.6f}")

    if diff_pop < diff_sample:
        print("\n>>> PyTorch InstanceNorm1d uses POPULATION variance (ddof=0)!")
    else:
        print("\n>>> PyTorch InstanceNorm1d uses SAMPLE variance (ddof=1)!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
