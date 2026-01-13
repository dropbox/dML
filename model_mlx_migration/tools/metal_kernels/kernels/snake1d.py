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
Snake1D Activation Custom Metal Kernel

Formula: x + (1/alpha) * sin(alpha*x)^2

This activation is used in Kokoro's ISTFTNet generator.
Fusing with preceding operations saves memory round-trips.
"""


import mlx.core as mx

# Metal kernel source code
# Note: Size and channels are passed as input arrays, not constants
SNAKE1D_KERNEL_SOURCE = """
    uint idx = thread_position_in_grid.x;

    // Get size info from input arrays
    uint total_size = (uint)size_param[0];
    uint num_channels = (uint)channels_param[0];

    // Bounds check
    if (idx >= total_size) return;

    // Get channel index for alpha lookup
    uint c = idx % num_channels;

    // Load values
    T x_val = x[idx];
    T a = alpha[c];

    // Snake1D: x + (1/alpha) * sin(alpha*x)^2
    T sin_val = metal::sin(a * x_val);
    out[idx] = x_val + (T(1.0) / a) * sin_val * sin_val;
"""

# No header needed - all params passed as inputs
SNAKE1D_KERNEL_HEADER = ""

# Cached kernel object
_snake1d_kernel: object | None = None


def get_snake1d_kernel():
    """Get or create the Snake1D Metal kernel."""
    global _snake1d_kernel
    if _snake1d_kernel is None:
        _snake1d_kernel = mx.fast.metal_kernel(
            name="snake1d_fused",
            input_names=["x", "alpha", "size_param", "channels_param"],
            output_names=["out"],
            source=SNAKE1D_KERNEL_SOURCE,
            header=SNAKE1D_KERNEL_HEADER,
            ensure_row_contiguous=True,
        )
    return _snake1d_kernel


def snake1d_custom(x: mx.array, alpha: mx.array) -> mx.array:
    """
    Custom Metal kernel for Snake1D activation.

    Args:
        x: Input tensor of shape [..., C]
        alpha: Per-channel alpha parameter of shape [C]

    Returns:
        Output tensor of same shape as x

    Note:
        Falls back to baseline if kernel execution fails.
    """
    kernel = get_snake1d_kernel()

    # Get dimensions
    size = x.size
    channels = alpha.size

    # Create size parameter arrays
    size_param = mx.array([size], dtype=mx.int32)
    channels_param = mx.array([channels], dtype=mx.int32)

    # Grid/threadgroup configuration
    # Use 256 threads per threadgroup (standard for elementwise)
    threadgroup_size = min(256, size)
    num_threadgroups = (size + threadgroup_size - 1) // threadgroup_size
    grid_size = num_threadgroups * threadgroup_size

    # Determine dtype template
    if x.dtype == mx.float16:
        template_type = mx.float16
    elif x.dtype == mx.bfloat16:
        template_type = mx.bfloat16
    else:
        template_type = mx.float32

    # Flatten input for kernel
    x_flat = x.flatten()

    try:
        outputs = kernel(
            inputs=[x_flat, alpha, size_param, channels_param],
            template=[("T", template_type)],
            grid=(grid_size, 1, 1),
            threadgroup=(threadgroup_size, 1, 1),
            output_shapes=[(size,)],
            output_dtypes=[x.dtype],
        )
        # Reshape output back to original shape
        return outputs[0].reshape(x.shape)
    except Exception as e:
        # Fallback to baseline
        print(f"Snake1D kernel failed, falling back to baseline: {e}")
        return snake1d_baseline(x, alpha)


def snake1d_baseline(x: mx.array, alpha: mx.array) -> mx.array:
    """Baseline Snake1D using MLX ops."""
    return x + (1.0 / alpha) * mx.power(mx.sin(alpha * x), 2)


def benchmark_snake1d(
    batch_size: int = 16,
    length: int = 1024,
    channels: int = 512,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """
    Benchmark Snake1D custom kernel vs baseline.

    Args:
        batch_size: Batch dimension
        length: Sequence length
        channels: Number of channels
        warmup: Warmup iterations
        iterations: Benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    import time

    # Create test data
    x = mx.random.normal((batch_size, length, channels))
    alpha = mx.random.uniform(0.1, 2.0, shape=(channels,))
    mx.eval(x, alpha)

    # Warmup
    for _ in range(warmup):
        _ = snake1d_custom(x, alpha)
        _ = snake1d_baseline(x, alpha)
        mx.eval(_)

    # Benchmark custom kernel
    start = time.perf_counter()
    for _ in range(iterations):
        out = snake1d_custom(x, alpha)
        mx.eval(out)
    custom_time = (time.perf_counter() - start) / iterations

    # Benchmark baseline
    start = time.perf_counter()
    for _ in range(iterations):
        out = snake1d_baseline(x, alpha)
        mx.eval(out)
    baseline_time = (time.perf_counter() - start) / iterations

    # Verify numerical accuracy
    out_custom = snake1d_custom(x, alpha)
    out_baseline = snake1d_baseline(x, alpha)
    mx.eval(out_custom, out_baseline)

    diff = mx.abs(out_custom - out_baseline)
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))

    return {
        "shape": x.shape,
        "custom_ms": custom_time * 1000,
        "baseline_ms": baseline_time * 1000,
        "speedup": baseline_time / custom_time,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "is_accurate": max_diff < 1e-4,
    }


if __name__ == "__main__":
    print("Testing Snake1D Custom Metal Kernel")
    print("=" * 50)

    # Test basic functionality
    x = mx.random.normal((2, 16, 8))
    alpha = mx.random.uniform(0.1, 2.0, shape=(8,))
    mx.eval(x, alpha)

    out_custom = snake1d_custom(x, alpha)
    out_baseline = snake1d_baseline(x, alpha)
    mx.eval(out_custom, out_baseline)

    print(f"Input shape: {x.shape}")
    print(f"Output shape (custom): {out_custom.shape}")
    print(f"Output shape (baseline): {out_baseline.shape}")

    diff = mx.abs(out_custom - out_baseline)
    print(f"Max diff: {float(mx.max(diff)):.2e}")
    print(f"Mean diff: {float(mx.mean(diff)):.2e}")

    # Benchmark
    print("\nBenchmark (realistic Kokoro Generator size):")
    result = benchmark_snake1d(
        batch_size=1,
        length=1024,  # ~4 seconds of audio at 24kHz / 60
        channels=512,
        warmup=10,
        iterations=100,
    )
    print(f"  Shape: {result['shape']}")
    print(f"  Custom: {result['custom_ms']:.3f} ms")
    print(f"  Baseline: {result['baseline_ms']:.3f} ms")
    print(f"  Speedup: {result['speedup']:.2f}x")
    print(f"  Accurate: {result['is_accurate']}")
