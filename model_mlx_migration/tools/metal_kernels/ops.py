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
Custom Fused Metal Kernels for MLX

This module provides optimized Metal kernels using mx.fast.metal_kernel API.
Each kernel fuses multiple operations to reduce memory bandwidth.

Results Summary (M4 Max):
- Snake1D: 1.40x speedup
- Instance Norm + Style: 1.11-1.35x speedup
- AdaIN + Conv1d (full pipeline): 1.11-1.22x speedup
- INT8 Linear: 4x memory savings (speed limited by MLX's optimized matmul)
"""

import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from tools.metal_kernels.kernels.instance_norm_style import (
    adain_conv1d,
    benchmark_adain_conv,
    fused_instance_norm_style,
    fused_instance_norm_style_baseline,
)
from tools.metal_kernels.kernels.int8_linear import (
    benchmark_int8_linear,
    int8_linear,
    int8_linear_baseline,
    quantize_linear_layer,
    quantize_weights_int8,
)

# Import from individual kernel modules
from tools.metal_kernels.kernels.snake1d import (
    benchmark_snake1d,
    snake1d_baseline,
    snake1d_custom,
)

# Convenient aliases
snake1d = snake1d_custom
snake1d_fused = snake1d_custom
fused_instance_norm = fused_instance_norm_style
fused_instance_norm_baseline = fused_instance_norm_style_baseline
fused_adain_conv1d = adain_conv1d
int8_linear_bias = int8_linear  # bias is optional param


# Baseline for fused adain_conv1d (unfused version)
def fused_adain_conv1d_baseline(x, weight, gamma, beta, kernel_size=3, padding=0, stride=1):
    """Baseline (unfused) AdaIN + Conv1d - for comparison only."""
    import mlx.core as mx

    # Instance normalization
    mean = mx.mean(x, axis=1, keepdims=True)
    var = mx.var(x, axis=1, keepdims=True)
    x_norm = (x - mean) / mx.sqrt(var + 1e-5)

    # Style transform (gamma/beta are per-batch, per-channel)
    x_styled = gamma[:, None, :] * x_norm + beta[:, None, :]

    # Conv1d
    return mx.conv1d(x_styled, weight, stride=stride, padding=padding)


# INT4 baseline (unfused dequantize + matmul)
def int4_linear_baseline(x, weight_q, scales, zeros, group_size=128):
    """Baseline INT4 linear (unfused dequant + matmul)."""
    import mlx.core as mx

    N, K_half = weight_q.shape
    K = K_half * 2

    # Unpack INT4 to INT8
    low = (weight_q & 0xF).astype(mx.int8) - 8
    high = (weight_q >> 4).astype(mx.int8) - 8

    weight_int8 = mx.stack([low, high], axis=-1).reshape(N, K)

    # Dequantize
    num_groups = K // group_size
    weight_int8 = weight_int8.reshape(N, num_groups, group_size)
    scales_expanded = scales[:, :, None]
    zeros_expanded = zeros[:, :, None]

    weight = (weight_int8.astype(mx.float32) - zeros_expanded) * scales_expanded
    weight = weight.reshape(N, K).astype(x.dtype)

    return x @ weight.T


# =============================================================================
# Benchmark Utilities
# =============================================================================


def benchmark_kernel(
    kernel_fn: Callable,
    baseline_fn: Callable,
    inputs: list[mx.array],
    warmup: int = 10,
    iterations: int = 100,
    name: str = "kernel",
) -> dict[str, Any]:
    """
    Benchmark a custom kernel against baseline.

    Args:
        kernel_fn: Custom kernel function
        baseline_fn: Baseline function
        inputs: List of input arrays
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        name: Name for reporting

    Returns:
        Dictionary with benchmark results
    """
    # Warmup
    for _ in range(warmup):
        out_k = kernel_fn(*inputs)
        out_b = baseline_fn(*inputs)
        mx.eval(out_k, out_b)

    # Benchmark custom kernel
    start = time.perf_counter()
    for _ in range(iterations):
        out = kernel_fn(*inputs)
        mx.eval(out)
    custom_time = (time.perf_counter() - start) / iterations

    # Benchmark baseline
    start = time.perf_counter()
    for _ in range(iterations):
        out = baseline_fn(*inputs)
        mx.eval(out)
    baseline_time = (time.perf_counter() - start) / iterations

    return {
        "name": name,
        "custom_ms": custom_time * 1000,
        "baseline_ms": baseline_time * 1000,
        "speedup": baseline_time / custom_time,
        "custom_faster": custom_time < baseline_time,
    }


def verify_numerical_accuracy(
    kernel_fn: Callable,
    baseline_fn: Callable,
    inputs: list[mx.array],
    rtol: float = 1e-5,
    atol: float = 1e-5,
    name: str = "kernel",
) -> dict[str, Any]:
    """
    Verify numerical accuracy of custom kernel vs baseline.

    Args:
        kernel_fn: Custom kernel function
        baseline_fn: Baseline function
        inputs: List of input arrays
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for reporting

    Returns:
        Dictionary with accuracy results
    """
    out_kernel = kernel_fn(*inputs)
    out_baseline = baseline_fn(*inputs)
    mx.eval(out_kernel, out_baseline)

    # Compute differences
    diff = mx.abs(out_kernel - out_baseline)
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))

    # Check if within tolerance
    is_close = mx.allclose(out_kernel, out_baseline, rtol=rtol, atol=atol)

    return {
        "name": name,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rtol": rtol,
        "atol": atol,
        "is_close": bool(is_close),
        "shape": out_kernel.shape,
        "dtype": str(out_kernel.dtype),
    }


# INT4 placeholder (not yet implemented with speedup)
def int4_linear(
    x: mx.array,
    weight_q: mx.array,
    scales: mx.array,
    zeros: mx.array,
    group_size: int = 128,
) -> mx.array:
    """INT4 quantized linear (baseline only - no speedup achieved yet)."""
    # Unpack INT4 to INT8
    N, K_half = weight_q.shape
    K = K_half * 2

    low = (weight_q & 0xF).astype(mx.int8) - 8
    high = (weight_q >> 4).astype(mx.int8) - 8

    weight_int8 = mx.stack([low, high], axis=-1).reshape(N, K)

    num_groups = K // group_size
    weight_int8 = weight_int8.reshape(N, num_groups, group_size)
    scales_expanded = scales[:, :, None]
    zeros_expanded = zeros[:, :, None]

    weight = (weight_int8.astype(mx.float32) - zeros_expanded) * scales_expanded
    weight = weight.reshape(N, K).astype(x.dtype)

    return x @ weight.T


__all__ = [
    # Snake1D
    "snake1d",
    "snake1d_custom",
    "snake1d_baseline",
    "snake1d_fused",
    "benchmark_snake1d",

    # Instance Norm + Style
    "fused_instance_norm",
    "fused_instance_norm_style",
    "fused_instance_norm_style_baseline",
    "fused_instance_norm_baseline",

    # AdaIN + Conv
    "fused_adain_conv1d",
    "fused_adain_conv1d_baseline",
    "adain_conv1d",
    "benchmark_adain_conv",

    # INT8 Linear
    "int8_linear",
    "int8_linear_baseline",
    "int8_linear_bias",
    "quantize_weights_int8",
    "quantize_linear_layer",
    "benchmark_int8_linear",

    # INT4 Linear
    "int4_linear",
    "int4_linear_baseline",

    # Utilities
    "benchmark_kernel",
    "verify_numerical_accuracy",
]
