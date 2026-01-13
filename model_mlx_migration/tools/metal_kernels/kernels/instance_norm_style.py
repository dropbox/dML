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
Fused Instance Normalization + Style Transform Metal Kernel

This kernel fuses:
    1. Instance normalization: (x - mean) / sqrt(var + eps)
    2. Style transform: gamma * norm + beta

Combined with MLX's optimized conv1d, this saves 2-3 memory round-trips
compared to doing each operation separately.

Usage in Kokoro AdaIN blocks:
    styled = fused_instance_norm_style(x, gamma, beta)  # Custom kernel
    out = mx.conv1d(styled, weight)                      # MLX optimized
"""


import mlx.core as mx

# =============================================================================
# Two-Pass Kernel (most reliable)
# =============================================================================
# Pass 1: Compute mean/var per (batch, channel)
# Pass 2: Normalize + style transform

# Pass 2: Normalize and style transform
NORM_STYLE_KERNEL_SOURCE = """
    uint idx = thread_position_in_grid.x;

    // Get dimensions
    uint total_size = (uint)params[0];
    uint batch_size = (uint)params[1];
    uint length = (uint)params[2];
    uint channels = (uint)params[3];

    if (idx >= total_size) return;

    // Compute indices
    uint b = idx / (length * channels);
    uint rem = idx % (length * channels);
    uint l = rem / channels;
    uint c = rem % channels;

    // Load input
    T x_val = x[idx];

    // Load instance norm stats (pre-computed): [B, C]
    uint stat_idx = b * channels + c;
    T m = mean[stat_idx];
    T v = var[stat_idx];

    // Normalize
    T norm_val = (x_val - m) * metal::rsqrt(v + T(1e-5));

    // Style transform: gamma, beta are [B, C]
    T g = gamma[stat_idx];
    T beta_val = beta[stat_idx];

    out[idx] = g * norm_val + beta_val;
"""

_norm_style_kernel: object | None = None


def get_norm_style_kernel():
    """Get or create the fused norm+style kernel."""
    global _norm_style_kernel
    if _norm_style_kernel is None:
        _norm_style_kernel = mx.fast.metal_kernel(
            name="fused_norm_style",
            input_names=["x", "gamma", "beta", "mean", "var", "params"],
            output_names=["out"],
            source=NORM_STYLE_KERNEL_SOURCE,
            header="",
            ensure_row_contiguous=True,
        )
    return _norm_style_kernel


def fused_instance_norm_style(
    x: mx.array,
    gamma: mx.array,
    beta: mx.array,
    eps: float = 1e-5,
) -> mx.array:
    """
    Fused Instance Normalization + Style Transform.

    Computes: gamma * ((x - mean) / sqrt(var + eps)) + beta
    where mean and var are computed over the length dimension.

    Args:
        x: Input tensor [B, L, C]
        gamma: Style scale [B, C]
        beta: Style shift [B, C]
        eps: Numerical stability epsilon

    Returns:
        Styled output tensor [B, L, C]
    """
    B, L, C = x.shape

    # Compute instance norm statistics using MLX (efficient reduction)
    mean = mx.mean(x, axis=1)  # [B, C]
    var = mx.var(x, axis=1)    # [B, C]

    # Parameters
    total_size = x.size
    params = mx.array([total_size, B, L, C], dtype=mx.int32)

    # Grid configuration
    threadgroup_size = 256
    grid_size = ((total_size + threadgroup_size - 1) // threadgroup_size) * threadgroup_size

    # Determine dtype
    if x.dtype == mx.float16:
        template_type = mx.float16
    elif x.dtype == mx.bfloat16:
        template_type = mx.bfloat16
    else:
        template_type = mx.float32

    kernel = get_norm_style_kernel()

    try:
        outputs = kernel(
            inputs=[x.flatten(), gamma, beta, mean, var, params],
            template=[("T", template_type)],
            grid=(grid_size, 1, 1),
            threadgroup=(threadgroup_size, 1, 1),
            output_shapes=[(total_size,)],
            output_dtypes=[x.dtype],
        )
        return outputs[0].reshape(x.shape)
    except Exception as e:
        print(f"Norm+Style kernel failed: {e}")
        return fused_instance_norm_style_baseline(x, gamma, beta, eps)


def fused_instance_norm_style_baseline(
    x: mx.array,
    gamma: mx.array,
    beta: mx.array,
    eps: float = 1e-5,
) -> mx.array:
    """Baseline implementation using MLX ops."""
    # Instance normalization
    mean = mx.mean(x, axis=1, keepdims=True)
    var = mx.var(x, axis=1, keepdims=True)
    normalized = (x - mean) / mx.sqrt(var + eps)

    # Style transform
    gamma = mx.expand_dims(gamma, axis=1)
    beta = mx.expand_dims(beta, axis=1)
    return gamma * normalized + beta


def adain_conv1d(
    x: mx.array,
    weight: mx.array,
    gamma: mx.array,
    beta: mx.array,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    eps: float = 1e-5,
    use_custom: bool = True,
) -> mx.array:
    """
    AdaIN + Conv1d using fused norm+style kernel + MLX conv.

    This is the recommended approach for Kokoro decoder blocks.
    Uses custom kernel for norm+style, then MLX's optimized conv1d.

    Args:
        x: Input tensor [B, L, C_in]
        weight: Conv weight [C_out, K, C_in]
        gamma: Style scale [B, C_in]
        beta: Style shift [B, C_in]
        kernel_size: Conv kernel size
        stride: Conv stride
        padding: Conv padding
        dilation: Conv dilation
        eps: Instance norm epsilon
        use_custom: Whether to use custom kernel (True) or baseline (False)

    Returns:
        Output tensor [B, L_out, C_out]
    """
    if use_custom:
        styled = fused_instance_norm_style(x, gamma, beta, eps)
    else:
        styled = fused_instance_norm_style_baseline(x, gamma, beta, eps)

    return mx.conv1d(styled, weight, stride=stride, padding=padding, dilation=dilation)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_norm_style(
    batch_size: int = 1,
    length: int = 256,
    channels: int = 256,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Benchmark norm+style kernel."""
    import time

    x = mx.random.normal((batch_size, length, channels))
    gamma = mx.random.uniform(0.5, 1.5, shape=(batch_size, channels))
    beta = mx.random.uniform(-0.5, 0.5, shape=(batch_size, channels))
    mx.eval(x, gamma, beta)

    # Warmup
    for _ in range(warmup):
        _ = fused_instance_norm_style(x, gamma, beta)
        _ = fused_instance_norm_style_baseline(x, gamma, beta)
        mx.eval(_)

    # Benchmark custom
    start = time.perf_counter()
    for _ in range(iterations):
        out = fused_instance_norm_style(x, gamma, beta)
        mx.eval(out)
    custom_time = (time.perf_counter() - start) / iterations

    # Benchmark baseline
    start = time.perf_counter()
    for _ in range(iterations):
        out = fused_instance_norm_style_baseline(x, gamma, beta)
        mx.eval(out)
    baseline_time = (time.perf_counter() - start) / iterations

    # Verify
    out_custom = fused_instance_norm_style(x, gamma, beta)
    out_baseline = fused_instance_norm_style_baseline(x, gamma, beta)
    mx.eval(out_custom, out_baseline)

    diff = mx.abs(out_custom - out_baseline)
    max_diff = float(mx.max(diff))

    return {
        "shape": x.shape,
        "custom_ms": custom_time * 1000,
        "baseline_ms": baseline_time * 1000,
        "speedup": baseline_time / custom_time,
        "max_diff": max_diff,
        "is_accurate": max_diff < 1e-4,
    }


def benchmark_adain_conv(
    batch_size: int = 1,
    length: int = 256,
    c_in: int = 256,
    c_out: int = 256,
    kernel_size: int = 3,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Benchmark full AdaIN+Conv pipeline."""
    import time

    x = mx.random.normal((batch_size, length, c_in))
    weight = mx.random.normal((c_out, kernel_size, c_in)) * 0.01
    gamma = mx.random.uniform(0.5, 1.5, shape=(batch_size, c_in))
    beta = mx.random.uniform(-0.5, 0.5, shape=(batch_size, c_in))
    mx.eval(x, weight, gamma, beta)

    # Warmup
    for _ in range(warmup):
        _ = adain_conv1d(x, weight, gamma, beta, kernel_size=kernel_size, use_custom=True)
        _ = adain_conv1d(x, weight, gamma, beta, kernel_size=kernel_size, use_custom=False)
        mx.eval(_)

    # Benchmark with custom kernel
    start = time.perf_counter()
    for _ in range(iterations):
        out = adain_conv1d(x, weight, gamma, beta, kernel_size=kernel_size, use_custom=True)
        mx.eval(out)
    custom_time = (time.perf_counter() - start) / iterations

    # Benchmark baseline
    start = time.perf_counter()
    for _ in range(iterations):
        out = adain_conv1d(x, weight, gamma, beta, kernel_size=kernel_size, use_custom=False)
        mx.eval(out)
    baseline_time = (time.perf_counter() - start) / iterations

    # Verify
    out_custom = adain_conv1d(x, weight, gamma, beta, kernel_size=kernel_size, use_custom=True)
    out_baseline = adain_conv1d(x, weight, gamma, beta, kernel_size=kernel_size, use_custom=False)
    mx.eval(out_custom, out_baseline)

    diff = mx.abs(out_custom - out_baseline)
    max_diff = float(mx.max(diff))

    return {
        "input_shape": x.shape,
        "output_shape": out_custom.shape,
        "custom_ms": custom_time * 1000,
        "baseline_ms": baseline_time * 1000,
        "speedup": baseline_time / custom_time,
        "max_diff": max_diff,
        "is_accurate": max_diff < 1e-3,
    }


if __name__ == "__main__":
    print("Testing Fused Instance Norm + Style Transform Kernel")
    print("=" * 60)

    # Basic test
    x = mx.random.normal((2, 32, 16))
    gamma = mx.random.uniform(0.5, 1.5, shape=(2, 16))
    beta = mx.random.uniform(-0.5, 0.5, shape=(2, 16))
    mx.eval(x, gamma, beta)

    out_custom = fused_instance_norm_style(x, gamma, beta)
    out_baseline = fused_instance_norm_style_baseline(x, gamma, beta)
    mx.eval(out_custom, out_baseline)

    print(f"Input shape: {x.shape}")
    print(f"Output shape (custom): {out_custom.shape}")
    print(f"Output shape (baseline): {out_baseline.shape}")

    diff = mx.abs(out_custom - out_baseline)
    print(f"Max diff: {float(mx.max(diff)):.2e}")
    print(f"Mean diff: {float(mx.mean(diff)):.2e}")

    # Benchmark norm+style only
    print("\nBenchmark: Instance Norm + Style Transform Only")
    for c in [128, 256, 512]:
        result = benchmark_norm_style(
            batch_size=1,
            length=256,
            channels=c,
            warmup=10,
            iterations=100,
        )
        print(f"  C={c}: Custom={result['custom_ms']:.3f}ms, "
              f"Baseline={result['baseline_ms']:.3f}ms, "
              f"Speedup={result['speedup']:.2f}x")

    # Benchmark full AdaIN+Conv
    print("\nBenchmark: Full AdaIN + Conv1d Pipeline")
    for c in [128, 256, 512]:
        result = benchmark_adain_conv(
            batch_size=1,
            length=256,
            c_in=c,
            c_out=c,
            kernel_size=3,
            warmup=10,
            iterations=50,
        )
        print(f"  C={c}: Custom={result['custom_ms']:.3f}ms, "
              f"Baseline={result['baseline_ms']:.3f}ms, "
              f"Speedup={result['speedup']:.2f}x, "
              f"Accurate={result['is_accurate']}")
