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
INT8 Quantized Linear Layer Custom Metal Kernel

Fused dequantization + matrix multiplication for INT8 weights.
Saves memory bandwidth by reading INT8 instead of FP16/FP32.

Memory bandwidth savings:
- FP32 weights: 4 bytes/element
- FP16 weights: 2 bytes/element
- INT8 weights: 1 byte/element + scales/zeros overhead

For a 1024x1024 layer:
- FP32: 4MB read
- FP16: 2MB read
- INT8: ~1MB read (weights) + ~8KB (scales)
"""


import mlx.core as mx

# =============================================================================
# INT8 Fused Dequantize + Linear Kernel
# =============================================================================
# Weight storage: [N, K] as int8
# Scales: [N] per-output-channel (symmetric quantization)
# Output: x @ W.T

INT8_LINEAR_KERNEL_SOURCE = """
    // Thread handles one output element
    uint b = thread_position_in_grid.y;  // batch index
    uint n = thread_position_in_grid.x;  // output feature index

    // Get dimensions
    uint batch_size = (uint)params[0];
    uint in_features = (uint)params[1];
    uint out_features = (uint)params[2];

    if (b >= batch_size || n >= out_features) return;

    // Get scale for this output channel
    float scale = (float)scales[n];

    // Accumulate in float32 for precision
    float acc = 0.0f;

    // Dequantize and multiply-accumulate
    for (uint k = 0; k < in_features; k++) {
        // Load INT8 weight and dequantize
        int8_t w_q = weight_q[n * in_features + k];
        float w = float(w_q) * scale;

        // Load input
        float x_val = (float)x[b * in_features + k];

        acc += x_val * w;
    }

    // Write output
    out[b * out_features + n] = T(acc);
"""

_int8_linear_kernel: object | None = None


def get_int8_linear_kernel():
    """Get or create the INT8 linear kernel."""
    global _int8_linear_kernel
    if _int8_linear_kernel is None:
        _int8_linear_kernel = mx.fast.metal_kernel(
            name="int8_linear",
            input_names=["x", "weight_q", "scales", "params"],
            output_names=["out"],
            source=INT8_LINEAR_KERNEL_SOURCE,
            header="",
            ensure_row_contiguous=True,
        )
    return _int8_linear_kernel


def int8_linear(
    x: mx.array,
    weight_q: mx.array,
    scales: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """
    INT8 quantized linear layer with fused dequantization.

    Uses symmetric quantization: weight_fp = weight_q * scale

    Args:
        x: Input tensor [B, K] or [..., K]
        weight_q: Quantized weights [N, K] as int8
        scales: Per-channel scales [N]
        bias: Optional bias [N]

    Returns:
        Output tensor [B, N] or [..., N]
    """
    # Handle multi-dimensional input
    original_shape = x.shape
    if len(x.shape) > 2:
        x = x.reshape(-1, x.shape[-1])

    B, K = x.shape
    N = weight_q.shape[0]

    # Parameters
    params = mx.array([B, K, N], dtype=mx.int32)

    # Grid configuration
    threadgroup_size = (16, 16, 1)  # 256 threads per group
    grid = (
        ((N + threadgroup_size[0] - 1) // threadgroup_size[0]) * threadgroup_size[0],
        ((B + threadgroup_size[1] - 1) // threadgroup_size[1]) * threadgroup_size[1],
        1,
    )

    kernel = get_int8_linear_kernel()

    # Determine output dtype
    out_dtype = x.dtype
    if out_dtype == mx.float16:
        template_type = mx.float16
    elif out_dtype == mx.bfloat16:
        template_type = mx.bfloat16
    else:
        template_type = mx.float32
        out_dtype = mx.float32

    try:
        outputs = kernel(
            inputs=[x, weight_q, scales, params],
            template=[("T", template_type)],
            grid=grid,
            threadgroup=threadgroup_size,
            output_shapes=[(B, N)],
            output_dtypes=[out_dtype],
        )
        out = outputs[0]
    except Exception as e:
        print(f"INT8 linear kernel failed: {e}")
        out = int8_linear_baseline(x, weight_q, scales)

    # Add bias
    if bias is not None:
        out = out + bias

    # Restore shape
    if len(original_shape) > 2:
        out = out.reshape(*original_shape[:-1], N)

    return out


def int8_linear_baseline(
    x: mx.array,
    weight_q: mx.array,
    scales: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """Baseline INT8 linear using MLX ops."""
    # Dequantize: weight_fp = weight_q * scale
    weight = weight_q.astype(mx.float32) * scales[:, None]
    weight = weight.astype(x.dtype)

    # Matmul: x @ weight.T
    out = x @ weight.T

    if bias is not None:
        out = out + bias

    return out


# =============================================================================
# Quantization Utilities
# =============================================================================

def quantize_weights_int8(
    weight: mx.array,
    symmetric: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Quantize FP weights to INT8.

    Args:
        weight: FP weights [N, K]
        symmetric: Use symmetric quantization (recommended)

    Returns:
        weight_q: INT8 weights [N, K]
        scales: Per-channel scales [N]
    """
    if symmetric:
        # Symmetric: scale = max(abs(w)) / 127
        max_abs = mx.max(mx.abs(weight), axis=1)  # [N]
        scales = max_abs / 127.0
        # Avoid division by zero
        scales = mx.where(scales == 0, mx.ones_like(scales), scales)
        # Quantize
        weight_q = mx.round(weight / scales[:, None])
        weight_q = mx.clip(weight_q, -128, 127).astype(mx.int8)
    else:
        raise NotImplementedError("Asymmetric quantization not implemented")

    return weight_q, scales


def quantize_linear_layer(
    weight: mx.array,
    bias: mx.array | None = None,
) -> dict:
    """
    Quantize a linear layer's weights to INT8.

    Args:
        weight: Weight tensor [N, K]
        bias: Optional bias tensor [N]

    Returns:
        Dictionary with quantized weights and scales
    """
    weight_q, scales = quantize_weights_int8(weight)
    return {
        "weight_q": weight_q,
        "scales": scales,
        "bias": bias,
    }


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_int8_linear(
    batch_size: int = 32,
    in_features: int = 1024,
    out_features: int = 1024,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Benchmark INT8 linear vs FP16 baseline."""
    import time

    # Create FP weights and quantize
    weight_fp = mx.random.normal((out_features, in_features)) * 0.01
    weight_q, scales = quantize_weights_int8(weight_fp)
    x = mx.random.normal((batch_size, in_features))
    mx.eval(x, weight_fp, weight_q, scales)

    # Warmup
    for _ in range(warmup):
        _ = int8_linear(x, weight_q, scales)
        _ = x @ weight_fp.T
        mx.eval(_)

    # Benchmark INT8 custom
    start = time.perf_counter()
    for _ in range(iterations):
        out = int8_linear(x, weight_q, scales)
        mx.eval(out)
    int8_custom_time = (time.perf_counter() - start) / iterations

    # Benchmark INT8 baseline (dequantize then matmul)
    start = time.perf_counter()
    for _ in range(iterations):
        out = int8_linear_baseline(x, weight_q, scales)
        mx.eval(out)
    int8_baseline_time = (time.perf_counter() - start) / iterations

    # Benchmark FP32 matmul (reference)
    start = time.perf_counter()
    for _ in range(iterations):
        out = x @ weight_fp.T
        mx.eval(out)
    fp_time = (time.perf_counter() - start) / iterations

    # Verify accuracy
    out_int8 = int8_linear(x, weight_q, scales)
    out_fp = x @ weight_fp.T
    mx.eval(out_int8, out_fp)

    diff = mx.abs(out_int8 - out_fp)
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))

    # Memory comparison
    fp_memory = weight_fp.size * 4  # FP32 bytes
    int8_memory = weight_q.size * 1 + scales.size * 4  # INT8 + FP32 scales

    return {
        "shape": f"({batch_size}, {in_features}) x ({out_features}, {in_features})",
        "int8_custom_ms": int8_custom_time * 1000,
        "int8_baseline_ms": int8_baseline_time * 1000,
        "fp32_ms": fp_time * 1000,
        "speedup_vs_fp": fp_time / int8_custom_time,
        "speedup_vs_baseline": int8_baseline_time / int8_custom_time,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "fp_memory_mb": fp_memory / 1e6,
        "int8_memory_mb": int8_memory / 1e6,
        "memory_savings": fp_memory / int8_memory,
    }


if __name__ == "__main__":
    print("Testing INT8 Quantized Linear Kernel")
    print("=" * 60)

    # Basic test
    B, K, N = 4, 64, 32

    weight_fp = mx.random.normal((N, K)) * 0.1
    weight_q, scales = quantize_weights_int8(weight_fp)
    x = mx.random.normal((B, K))
    mx.eval(x, weight_fp, weight_q, scales)

    out_int8 = int8_linear(x, weight_q, scales)
    out_fp = x @ weight_fp.T
    mx.eval(out_int8, out_fp)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight_fp.shape}")
    print(f"Output shape: {out_int8.shape}")

    diff = mx.abs(out_int8 - out_fp)
    print(f"Max diff vs FP32: {float(mx.max(diff)):.2e}")
    print(f"Mean diff vs FP32: {float(mx.mean(diff)):.2e}")

    # Benchmark various sizes
    print("\nBenchmark Results:")
    print("-" * 80)

    for K in [256, 512, 1024, 2048]:
        result = benchmark_int8_linear(
            batch_size=32,
            in_features=K,
            out_features=K,
            warmup=10,
            iterations=50,
        )
        print(f"\nSize: {result['shape']}")
        print(f"  INT8 Custom:   {result['int8_custom_ms']:.3f} ms")
        print(f"  INT8 Baseline: {result['int8_baseline_ms']:.3f} ms")
        print(f"  FP32 Matmul:   {result['fp32_ms']:.3f} ms")
        print(f"  Speedup vs FP32: {result['speedup_vs_fp']:.2f}x")
        print(f"  Speedup vs Baseline: {result['speedup_vs_baseline']:.2f}x")
        print(f"  Memory: {result['fp_memory_mb']:.1f}MB (FP32) -> {result['int8_memory_mb']:.1f}MB (INT8) = {result['memory_savings']:.1f}x savings")
        print(f"  Accuracy: max_diff={result['max_diff']:.2e}")
