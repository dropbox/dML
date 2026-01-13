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

"""Test reflect padding matches PyTorch."""

import mlx.core as mx
import numpy as np
import torch


def pytorch_reflect_pad(x, pad_left, pad_right):
    """PyTorch reflect pad for reference."""
    return torch.nn.functional.pad(
        x.unsqueeze(0), (pad_left, pad_right), mode="reflect"
    ).squeeze(0)


def mlx_reflect_pad(x, pad_left, pad_right):
    """MLX reflect pad (manual implementation)."""
    if pad_left > 0:
        left_pad = x[:, 1 : pad_left + 1][:, ::-1]
    else:
        left_pad = mx.zeros((x.shape[0], 0))

    if pad_right > 0:
        right_pad = x[:, -(pad_right + 1) : -1][:, ::-1]
    else:
        right_pad = mx.zeros((x.shape[0], 0))

    return mx.concatenate([left_pad, x, right_pad], axis=1)


def main():
    np.random.seed(42)

    # Test with various input sizes (pad must be < length for reflect)
    for length, pad in [(20, 10), (100, 10), (1000, 10), (100, 50)]:
        x_np = np.random.randn(1, length).astype(np.float32)

        # PyTorch
        pt_x = torch.from_numpy(x_np)
        pt_padded = pytorch_reflect_pad(pt_x, pad, pad)

        # MLX
        mlx_x = mx.array(x_np)
        mlx_padded = mlx_reflect_pad(mlx_x, pad, pad)
        mx.eval(mlx_padded)

        # Compare
        pt_np = pt_padded.numpy()
        mlx_np = np.array(mlx_padded)

        diff = np.abs(pt_np - mlx_np).max()
        print(f"Length={length}, pad={pad}: max_diff={diff:.8f}")

    print("\n=== Edge case: short signal ===")
    x_np = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    pad = 2

    pt_x = torch.from_numpy(x_np)
    pt_padded = pytorch_reflect_pad(pt_x, pad, pad)

    mlx_x = mx.array(x_np)
    mlx_padded = mlx_reflect_pad(mlx_x, pad, pad)
    mx.eval(mlx_padded)

    print(f"Input: {x_np.flatten().tolist()}")
    print(f"PyTorch reflect pad({pad},{pad}): {pt_padded.numpy().flatten().tolist()}")
    print(f"MLX reflect pad({pad},{pad}): {np.array(mlx_padded).flatten().tolist()}")
    print(f"Match: {np.allclose(pt_padded.numpy(), np.array(mlx_padded))}")


if __name__ == "__main__":
    main()
