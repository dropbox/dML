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
Check dilation configuration for all conv layers in resblocks.
"""

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    print("=== Loading MLX model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator

    print("\n=== Resblock 0 Configuration ===")
    resblock = generator.resblocks_0

    # Print the expected dilations
    print("Expected dilations: convs1=[1,3,5], convs2=[1,1,1]")

    for i in range(3):
        conv1 = getattr(resblock, f"convs1_{i}")
        conv2 = getattr(resblock, f"convs2_{i}")
        print(f"\nIteration {i}:")
        print(
            f"  convs1_{i}: kernel={conv1.kernel_size}, dilation={conv1.dilation}, padding={conv1.padding}"
        )
        print(
            f"  convs2_{i}: kernel={conv2.kernel_size}, dilation={conv2.dilation}, padding={conv2.padding}"
        )

        # Expected padding for dilated conv: (kernel - 1) * dilation / 2
        expected_dilations = [1, 3, 5]
        expected_d1 = expected_dilations[i]
        expected_pad1 = (conv1.kernel_size - 1) * expected_d1 // 2
        expected_pad2 = (conv2.kernel_size - 1) * 1 // 2  # convs2 always dilation=1

        print(f"  Expected convs1_{i}: dilation={expected_d1}, padding={expected_pad1}")
        print(f"  Expected convs2_{i}: dilation=1, padding={expected_pad2}")

        if conv1.dilation != expected_d1:
            print(
                f"  *** MISMATCH: convs1_{i}.dilation={conv1.dilation}, expected={expected_d1}"
            )
        if conv1.padding != expected_pad1:
            print(
                f"  *** MISMATCH: convs1_{i}.padding={conv1.padding}, expected={expected_pad1}"
            )

    # Also check the num_layers attribute
    print(f"\nresblock.num_layers: {resblock.num_layers}")
    print(f"resblock.channels: {resblock.channels}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
