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
Fused AdaIN + Conv1d Custom Metal Kernel

This is the core operation in Kokoro's decoder (55% of compute).

AdaIN (Adaptive Instance Normalization):
    1. Instance norm: (x - mean) / sqrt(var + eps)
    2. Style transform: gamma * norm + beta
    3. Conv1d: convolution over styled output

Fusing these operations saves 4 memory round-trips per layer.
"""


import mlx.core as mx

# =============================================================================
# Kernel 1: Fused Style Transform + Conv1d
# =============================================================================
# Assumes instance norm stats (mean, var) are pre-computed
# Fuses: normalize + style_transform + conv1d

STYLE_CONV_KERNEL_SOURCE = """
    // Output indices
    uint b = thread_position_in_grid.z;      // batch
    uint l_out = thread_position_in_grid.y;  // output length position
    uint c_out = thread_position_in_grid.x;  // output channel

    // Get dimensions from param arrays
    uint batch_size = (uint)params[0];
    uint length = (uint)params[1];
    uint c_in = (uint)params[2];
    uint c_out_dim = (uint)params[3];
    uint k_size = (uint)params[4];
    uint padding = (uint)params[5];

    // Bounds check
    if (b >= batch_size || l_out >= length - k_size + 1 + 2*padding || c_out >= c_out_dim) return;

    // Get instance norm stats for this batch (pre-computed)
    // mean and var are [B, C_in]
    // We'll access them per-channel during the conv accumulation

    T acc = T(0);

    // Convolution loop
    for (uint k = 0; k < k_size; k++) {
        int l_in = (int)l_out + (int)k - (int)padding;

        // Skip if out of bounds (zero padding)
        if (l_in < 0 || l_in >= (int)length) continue;

        for (uint c = 0; c < c_in; c++) {
            // Load input value
            uint x_idx = b * length * c_in + (uint)l_in * c_in + c;
            T x_val = x[x_idx];

            // Load instance norm stats
            uint stat_idx = b * c_in + c;
            T m = mean[stat_idx];
            T v = var[stat_idx];

            // Normalize
            T norm_val = (x_val - m) * metal::rsqrt(v + T(eps));

            // Style transform (gamma, beta are [B, C_in])
            T g = gamma[stat_idx];
            T beta_val = beta[stat_idx];
            T styled = g * norm_val + beta_val;

            // Convolution weight: [C_out, K, C_in]
            uint w_idx = c_out * k_size * c_in + k * c_in + c;
            acc += styled * weight[w_idx];
        }
    }

    // Write output: [B, L_out, C_out]
    uint out_length = length - k_size + 1 + 2*padding;
    uint out_idx = b * out_length * c_out_dim + l_out * c_out_dim + c_out;
    out[out_idx] = acc;
"""

STYLE_CONV_HEADER = """
constant float eps = 1e-5f;
"""

_style_conv_kernel: object | None = None


def get_style_conv_kernel():
    """Get or create the fused style+conv kernel."""
    global _style_conv_kernel
    if _style_conv_kernel is None:
        _style_conv_kernel = mx.fast.metal_kernel(
            name="fused_style_conv",
            input_names=["x", "weight", "gamma", "beta", "mean", "var", "params"],
            output_names=["out"],
            source=STYLE_CONV_KERNEL_SOURCE,
            header=STYLE_CONV_HEADER,
            ensure_row_contiguous=True,
        )
    return _style_conv_kernel


def fused_adain_conv1d_custom(
    x: mx.array,
    weight: mx.array,
    gamma: mx.array,
    beta: mx.array,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    eps: float = 1e-5,
) -> mx.array:
    """
    Fused AdaIN + Conv1d using custom Metal kernel.

    Args:
        x: Input tensor [B, L, C_in]
        weight: Conv weight [C_out, K, C_in]
        gamma: Style-conditioned scale [B, C_in]
        beta: Style-conditioned shift [B, C_in]
        kernel_size: Convolution kernel size
        stride: Convolution stride (only 1 supported currently)
        padding: Convolution padding
        dilation: Convolution dilation (only 1 supported currently)
        eps: Instance norm epsilon

    Returns:
        Output tensor [B, L_out, C_out]
    """
    if stride != 1 or dilation != 1:
        # Fall back to baseline for unsupported params
        return fused_adain_conv1d_baseline(
            x, weight, gamma, beta, kernel_size, stride, padding, dilation, eps,
        )

    B, L, C_in = x.shape
    C_out, K, _ = weight.shape

    # Pre-compute instance norm statistics
    mean = mx.mean(x, axis=1)  # [B, C_in]
    var = mx.var(x, axis=1)    # [B, C_in]

    # Output dimensions
    L_out = L - K + 1 + 2 * padding

    # Parameters array
    params = mx.array([B, L, C_in, C_out, K, padding], dtype=mx.int32)

    # Grid configuration
    # Each thread computes one output element
    threadgroup_size = (8, 8, 1)  # 64 threads per threadgroup
    grid = (
        ((C_out + threadgroup_size[0] - 1) // threadgroup_size[0]) * threadgroup_size[0],
        ((L_out + threadgroup_size[1] - 1) // threadgroup_size[1]) * threadgroup_size[1],
        B,
    )

    kernel = get_style_conv_kernel()

    # Determine dtype
    if x.dtype == mx.float16:
        template_type = mx.float16
    elif x.dtype == mx.bfloat16:
        template_type = mx.bfloat16
    else:
        template_type = mx.float32

    try:
        outputs = kernel(
            inputs=[x, weight, gamma, beta, mean, var, params],
            template=[("T", template_type)],
            grid=grid,
            threadgroup=threadgroup_size,
            output_shapes=[(B, L_out, C_out)],
            output_dtypes=[x.dtype],
        )
        return outputs[0]
    except Exception as e:
        print(f"AdaIN+Conv kernel failed: {e}")
        return fused_adain_conv1d_baseline(
            x, weight, gamma, beta, kernel_size, stride, padding, dilation, eps,
        )


def fused_adain_conv1d_baseline(
    x: mx.array,
    weight: mx.array,
    gamma: mx.array,
    beta: mx.array,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    eps: float = 1e-5,
) -> mx.array:
    """Baseline AdaIN + Conv1d using MLX ops."""
    # Instance normalization
    mean = mx.mean(x, axis=1, keepdims=True)
    var = mx.var(x, axis=1, keepdims=True)
    normalized = (x - mean) / mx.sqrt(var + eps)

    # Style transform (gamma and beta are [B, C], need to broadcast to [B, 1, C])
    gamma = mx.expand_dims(gamma, axis=1)
    beta = mx.expand_dims(beta, axis=1)
    styled = gamma * normalized + beta

    # Conv1d
    return mx.conv1d(styled, weight, stride=stride, padding=padding, dilation=dilation)


# =============================================================================
# Kernel 2: Instance Norm Statistics (helper)
# =============================================================================
# Computes mean and variance for instance normalization

def compute_instance_stats(x: mx.array) -> tuple[mx.array, mx.array]:
    """
    Compute instance normalization statistics.

    Args:
        x: Input tensor [B, L, C]

    Returns:
        mean: [B, C]
        var: [B, C]
    """
    mean = mx.mean(x, axis=1)  # [B, C]
    var = mx.var(x, axis=1)    # [B, C]
    return mean, var


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_adain_conv(
    batch_size: int = 1,
    length: int = 256,
    c_in: int = 256,
    c_out: int = 256,
    kernel_size: int = 3,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Benchmark AdaIN+Conv kernel vs baseline."""
    import time

    # Create test data
    x = mx.random.normal((batch_size, length, c_in))
    weight = mx.random.normal((c_out, kernel_size, c_in)) * 0.01
    gamma = mx.random.uniform(0.5, 1.5, shape=(batch_size, c_in))
    beta = mx.random.uniform(-0.5, 0.5, shape=(batch_size, c_in))
    mx.eval(x, weight, gamma, beta)

    # Warmup
    for _ in range(warmup):
        _ = fused_adain_conv1d_custom(x, weight, gamma, beta, kernel_size=kernel_size)
        _ = fused_adain_conv1d_baseline(x, weight, gamma, beta, kernel_size=kernel_size)
        mx.eval(_)

    # Benchmark custom
    start = time.perf_counter()
    for _ in range(iterations):
        out = fused_adain_conv1d_custom(x, weight, gamma, beta, kernel_size=kernel_size)
        mx.eval(out)
    custom_time = (time.perf_counter() - start) / iterations

    # Benchmark baseline
    start = time.perf_counter()
    for _ in range(iterations):
        out = fused_adain_conv1d_baseline(x, weight, gamma, beta, kernel_size=kernel_size)
        mx.eval(out)
    baseline_time = (time.perf_counter() - start) / iterations

    # Verify accuracy
    out_custom = fused_adain_conv1d_custom(x, weight, gamma, beta, kernel_size=kernel_size)
    out_baseline = fused_adain_conv1d_baseline(x, weight, gamma, beta, kernel_size=kernel_size)
    mx.eval(out_custom, out_baseline)

    diff = mx.abs(out_custom - out_baseline)
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))

    return {
        "input_shape": x.shape,
        "output_shape": out_custom.shape,
        "weight_shape": weight.shape,
        "custom_ms": custom_time * 1000,
        "baseline_ms": baseline_time * 1000,
        "speedup": baseline_time / custom_time,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "is_accurate": max_diff < 1e-3,
    }


if __name__ == "__main__":
    print("Testing Fused AdaIN + Conv1d Custom Metal Kernel")
    print("=" * 60)

    # Basic test
    B, L, C_in, C_out, K = 2, 32, 16, 24, 3

    x = mx.random.normal((B, L, C_in))
    weight = mx.random.normal((C_out, K, C_in)) * 0.01
    gamma = mx.random.uniform(0.5, 1.5, shape=(B, C_in))
    beta = mx.random.uniform(-0.5, 0.5, shape=(B, C_in))
    mx.eval(x, weight, gamma, beta)

    out_custom = fused_adain_conv1d_custom(x, weight, gamma, beta, kernel_size=K)
    out_baseline = fused_adain_conv1d_baseline(x, weight, gamma, beta, kernel_size=K)
    mx.eval(out_custom, out_baseline)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape (custom): {out_custom.shape}")
    print(f"Output shape (baseline): {out_baseline.shape}")

    diff = mx.abs(out_custom - out_baseline)
    print(f"Max diff: {float(mx.max(diff)):.2e}")
    print(f"Mean diff: {float(mx.mean(diff)):.2e}")

    # Benchmark with realistic Kokoro decoder sizes
    print("\nBenchmark (Kokoro decoder-like sizes):")
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
        print(f"\n  C={c}:")
        print(f"    Input: {result['input_shape']}, Output: {result['output_shape']}")
        print(f"    Custom: {result['custom_ms']:.3f} ms")
        print(f"    Baseline: {result['baseline_ms']:.3f} ms")
        print(f"    Speedup: {result['speedup']:.2f}x")
        print(f"    Accurate: {result['is_accurate']} (max_diff={result['max_diff']:.2e})")
