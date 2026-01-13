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
Numerical validation for PyTorch -> MLX Zipformer conversion.

Compares outputs between PyTorch reference and MLX implementation.
"""

import argparse

import mlx.core as mx
import numpy as np
import torch


def compare_arrays(
    torch_arr: torch.Tensor,
    mlx_arr: mx.array,
    name: str = "output",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict[str, float]:
    """
    Compare PyTorch and MLX arrays and report statistics.

    Args:
        torch_arr: PyTorch tensor
        mlx_arr: MLX array
        name: Name for reporting
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with comparison statistics.
    """
    torch_np = torch_arr.detach().cpu().numpy()
    mlx_np = np.array(mlx_arr)

    # Check shapes match
    if torch_np.shape != mlx_np.shape:
        return {
            "name": name,
            "shape_match": False,
            "torch_shape": torch_np.shape,
            "mlx_shape": mlx_np.shape,
        }

    # Compute differences
    abs_diff = np.abs(torch_np - mlx_np)
    max_abs_error = float(np.max(abs_diff))
    mean_abs_error = float(np.mean(abs_diff))

    # Relative error (avoiding division by zero)
    denom = np.maximum(np.abs(torch_np), np.abs(mlx_np))
    denom = np.where(denom < 1e-10, 1.0, denom)
    rel_diff = abs_diff / denom
    max_rel_error = float(np.max(rel_diff))
    mean_rel_error = float(np.mean(rel_diff))

    # Check if within tolerance
    is_close = np.allclose(torch_np, mlx_np, rtol=rtol, atol=atol)

    return {
        "name": name,
        "shape_match": True,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error,
        "is_close": is_close,
        "rtol": rtol,
        "atol": atol,
    }


def validate_feedforward_module(
    checkpoint_path: str,
    layer_prefix: str = "encoder.encoders.0.layers.0.",
) -> dict[str, float]:
    """
    Validate feedforward module numerical equivalence.

    Args:
        checkpoint_path: Path to PyTorch checkpoint.
        layer_prefix: Prefix for layer weights.

    Returns:
        Comparison statistics.
    """
    from encoder_pretrained import FeedforwardModule as MLXFeedforward

    # Load PyTorch checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = state_dict.get("model", state_dict)

    # Extract feedforward weights
    ff_prefix = layer_prefix + "feed_forward1."
    in_proj_weight = model_dict[ff_prefix + "in_proj.weight"]
    in_proj_bias = model_dict[ff_prefix + "in_proj.bias"]
    out_proj_weight = model_dict[ff_prefix + "out_proj.weight"]
    out_proj_bias = model_dict[ff_prefix + "out_proj.bias"]

    # Create PyTorch feedforward (manual)
    d_model = in_proj_weight.shape[1]
    ff_dim = in_proj_weight.shape[0]

    # PyTorch forward
    x_np = np.random.randn(10, 2, d_model).astype(np.float32)
    x_torch = torch.tensor(x_np)

    # Manual PyTorch forward
    h = torch.nn.functional.linear(x_torch, in_proj_weight, in_proj_bias)
    h = torch.nn.functional.silu(h)
    torch_out = torch.nn.functional.linear(h, out_proj_weight, out_proj_bias)

    # MLX forward
    mlx_ff = MLXFeedforward(d_model, ff_dim)
    mlx_ff.in_proj.weight = mx.array(in_proj_weight.numpy())
    mlx_ff.in_proj.bias = mx.array(in_proj_bias.numpy())
    mlx_ff.out_proj.weight = mx.array(out_proj_weight.numpy())
    mlx_ff.out_proj.bias = mx.array(out_proj_bias.numpy())

    x_mlx = mx.array(x_np)
    mlx_out = mlx_ff(x_mlx)

    return compare_arrays(torch_out, mlx_out, "feedforward", rtol=1e-4, atol=1e-4)


def validate_pooling_module(
    checkpoint_path: str,
    layer_prefix: str = "encoder.encoders.0.layers.0.",
) -> dict[str, float]:
    """
    Validate pooling module numerical equivalence.

    Args:
        checkpoint_path: Path to PyTorch checkpoint.
        layer_prefix: Prefix for layer weights.

    Returns:
        Comparison statistics.
    """
    from encoder_pretrained import PoolingModule as MLXPooling

    # Load PyTorch checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = state_dict.get("model", state_dict)

    # Extract pooling weights
    proj_weight = model_dict[layer_prefix + "pooling.proj.weight"]
    d_model = proj_weight.shape[0]

    # Create test input
    seq_len = 10
    batch_size = 2
    x_np = np.random.randn(seq_len, batch_size, d_model).astype(np.float32)
    x_torch = torch.tensor(x_np)

    # PyTorch forward (manual pooling)
    # Running average: cumsum / cumcount
    cum_x = torch.cumsum(x_torch, dim=0)
    cum_count = torch.arange(1, seq_len + 1, dtype=torch.float32).view(-1, 1, 1)
    pooled = cum_x / cum_count
    torch_out = torch.nn.functional.linear(pooled, proj_weight, None)

    # MLX forward
    mlx_pool = MLXPooling(d_model)
    mlx_pool.proj.weight = mx.array(proj_weight.numpy())

    x_mlx = mx.array(x_np)
    mlx_out = mlx_pool(x_mlx)

    return compare_arrays(torch_out, mlx_out, "pooling", rtol=1e-4, atol=1e-4)


def validate_convolution_module(
    checkpoint_path: str,
    layer_prefix: str = "encoder.encoders.0.layers.0.",
) -> dict[str, float]:
    """
    Validate convolution module numerical equivalence.

    Args:
        checkpoint_path: Path to PyTorch checkpoint.
        layer_prefix: Prefix for layer weights.

    Returns:
        Comparison statistics.
    """
    from encoder_pretrained import ConvolutionModule as MLXConv

    # Load PyTorch checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = state_dict.get("model", state_dict)

    # Extract conv weights
    conv_prefix = layer_prefix + "conv_module1."
    pw1_weight = model_dict[conv_prefix + "pointwise_conv1.weight"]
    pw1_bias = model_dict[conv_prefix + "pointwise_conv1.bias"]
    dw_weight = model_dict[conv_prefix + "depthwise_conv.weight"]
    dw_bias = model_dict[conv_prefix + "depthwise_conv.bias"]
    pw2_weight = model_dict[conv_prefix + "pointwise_conv2.weight"]
    pw2_bias = model_dict[conv_prefix + "pointwise_conv2.bias"]

    d_model = dw_weight.shape[0]
    kernel_size = dw_weight.shape[2]

    # Create test input
    seq_len = 20
    batch_size = 2
    x_np = np.random.randn(seq_len, batch_size, d_model).astype(np.float32)
    x_torch = torch.tensor(x_np)

    # PyTorch forward (manual)
    # Transpose to (batch, d_model, seq) for conv1d
    x_t = x_torch.permute(1, 2, 0)

    # First pointwise conv
    x_t = torch.nn.functional.conv1d(x_t, pw1_weight) + pw1_bias.view(1, -1, 1)

    # GLU
    x_t, gate = x_t.chunk(2, dim=1)
    x_t = x_t * torch.sigmoid(gate)

    # Depthwise conv with padding
    pad_len = (kernel_size - 1) // 2
    x_t = torch.nn.functional.pad(x_t, (pad_len, pad_len))
    x_t = torch.nn.functional.conv1d(x_t, dw_weight, groups=d_model) + dw_bias.view(1, -1, 1)

    # SiLU
    x_t = torch.nn.functional.silu(x_t)

    # Second pointwise conv
    x_t = torch.nn.functional.conv1d(x_t, pw2_weight) + pw2_bias.view(1, -1, 1)

    # Transpose back
    torch_out = x_t.permute(2, 0, 1)

    # MLX forward
    mlx_conv = MLXConv(d_model, kernel_size)

    # Load weights (convert to MLX format)
    mlx_conv.pointwise_conv1_weight = mx.transpose(mx.array(pw1_weight.numpy()), (0, 2, 1))
    mlx_conv.pointwise_conv1_bias = mx.array(pw1_bias.numpy())
    mlx_conv.depthwise_conv_weight = mx.transpose(mx.array(dw_weight.numpy()), (0, 2, 1))
    mlx_conv.depthwise_conv_bias = mx.array(dw_bias.numpy())
    mlx_conv.pointwise_conv2_weight = mx.transpose(mx.array(pw2_weight.numpy()), (0, 2, 1))
    mlx_conv.pointwise_conv2_bias = mx.array(pw2_bias.numpy())

    x_mlx = mx.array(x_np)
    mlx_out = mlx_conv(x_mlx)

    return compare_arrays(torch_out, mlx_out, "convolution", rtol=1e-4, atol=1e-4)


def main():
    parser = argparse.ArgumentParser(
        description="Validate numerical equivalence between PyTorch and MLX",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NUMERICAL VALIDATION: PyTorch vs MLX")
    print("=" * 60)

    # Validate feedforward module
    print("\n[1] Feedforward Module:")
    try:
        result = validate_feedforward_module(args.checkpoint)
        if result.get("shape_match", False):
            print(f"    Max abs error: {result['max_abs_error']:.2e}")
            print(f"    Mean abs error: {result['mean_abs_error']:.2e}")
            print(f"    Max rel error: {result['max_rel_error']:.2e}")
            print(f"    Is close (rtol={result['rtol']}, atol={result['atol']}): {result['is_close']}")
        else:
            print(f"    Shape mismatch: {result.get('torch_shape')} vs {result.get('mlx_shape')}")
    except Exception as e:
        print(f"    Error: {e}")

    # Validate pooling module
    print("\n[2] Pooling Module:")
    try:
        result = validate_pooling_module(args.checkpoint)
        if result.get("shape_match", False):
            print(f"    Max abs error: {result['max_abs_error']:.2e}")
            print(f"    Mean abs error: {result['mean_abs_error']:.2e}")
            print(f"    Max rel error: {result['max_rel_error']:.2e}")
            print(f"    Is close (rtol={result['rtol']}, atol={result['atol']}): {result['is_close']}")
        else:
            print(f"    Shape mismatch: {result.get('torch_shape')} vs {result.get('mlx_shape')}")
    except Exception as e:
        print(f"    Error: {e}")

    # Validate convolution module
    print("\n[3] Convolution Module:")
    try:
        result = validate_convolution_module(args.checkpoint)
        if result.get("shape_match", False):
            print(f"    Max abs error: {result['max_abs_error']:.2e}")
            print(f"    Mean abs error: {result['mean_abs_error']:.2e}")
            print(f"    Max rel error: {result['max_rel_error']:.2e}")
            print(f"    Is close (rtol={result['rtol']}, atol={result['atol']}): {result['is_close']}")
        else:
            print(f"    Shape mismatch: {result.get('torch_shape')} vs {result.get('mlx_shape')}")
    except Exception as e:
        print(f"    Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
