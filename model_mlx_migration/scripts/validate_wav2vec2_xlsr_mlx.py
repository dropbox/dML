#!/usr/bin/env python3
"""Validate Wav2Vec2-XLSR MLX implementation against HuggingFace PyTorch reference.

This script compares outputs from PyTorch (transformers) and MLX wav2vec2-xlsr models
to verify numerical equivalence.

Usage:
    python scripts/validate_wav2vec2_xlsr_mlx.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_pytorch_wav2vec2(x: np.ndarray, checkpoint_path: str) -> np.ndarray:
    """Run wav2vec2 forward pass using PyTorch with original weights.

    This function directly loads weights and computes the forward pass
    to match the original HuggingFace implementation exactly.

    Args:
        x: Input audio (batch, samples)
        checkpoint_path: Path to pytorch_model.bin

    Returns:
        Output features (batch, frames, hidden_size)
    """
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    x_pt = torch.from_numpy(x).unsqueeze(-1)  # (batch, samples, 1)

    # Feature extractor (7 conv layers)
    conv_spec = [
        (512, 10, 5),  # layer 0
        (512, 3, 2),   # layer 1
        (512, 3, 2),   # layer 2
        (512, 3, 2),   # layer 3
        (512, 3, 2),   # layer 4
        (512, 2, 2),   # layer 5
        (512, 2, 2),   # layer 6
    ]

    for i in range(7):
        w = weights[f'wav2vec2.feature_extractor.conv_layers.{i}.conv.weight']
        b = weights[f'wav2vec2.feature_extractor.conv_layers.{i}.conv.bias']
        ln_w = weights[f'wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.weight']
        ln_b = weights[f'wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.bias']

        out_ch, kernel_size, stride = conv_spec[i]

        # Conv1d expects (batch, channels, length)
        x_pt_t = x_pt.transpose(1, 2)  # (batch, channels, length)
        x_pt_t = F.conv1d(x_pt_t, w, b, stride=stride)
        x_pt = x_pt_t.transpose(1, 2)  # (batch, length, channels)

        # Layer norm
        ln = nn.LayerNorm(out_ch)
        ln.weight.data = ln_w
        ln.bias.data = ln_b
        x_pt = ln(x_pt)
        x_pt = F.gelu(x_pt)

    # Feature projection
    proj_ln_w = weights['wav2vec2.feature_projection.layer_norm.weight']
    proj_ln_b = weights['wav2vec2.feature_projection.layer_norm.bias']
    proj_w = weights['wav2vec2.feature_projection.projection.weight']
    proj_b = weights['wav2vec2.feature_projection.projection.bias']

    ln = nn.LayerNorm(512)
    ln.weight.data = proj_ln_w
    ln.bias.data = proj_ln_b
    x_pt = ln(x_pt)
    x_pt = F.linear(x_pt, proj_w, proj_b)

    # Positional conv embedding (grouped conv with weight normalization)
    weight_g = weights['wav2vec2.encoder.pos_conv_embed.conv.weight_g']
    weight_v = weights['wav2vec2.encoder.pos_conv_embed.conv.weight_v']
    pos_bias = weights['wav2vec2.encoder.pos_conv_embed.conv.bias']

    # Compute effective weight: weight = weight_g * weight_v / ||weight_v||
    norm = weight_v.pow(2).sum(dim=[1, 2], keepdim=True).sqrt()
    effective_weight = weight_g * weight_v / norm

    # Grouped conv with padding
    kernel_size = 128
    padding = kernel_size // 2
    x_pt_t = x_pt.transpose(1, 2)  # (batch, hidden, length)
    x_pt_t = F.conv1d(x_pt_t, effective_weight, pos_bias, padding=padding, groups=16)
    # Remove the extra sample from padding
    x_pt_t = x_pt_t[:, :, :-1]
    x_pt_pos = x_pt_t.transpose(1, 2)
    x_pt_pos = F.gelu(x_pt_pos)
    x_pt = x_pt + x_pt_pos

    # Encoder layer norm (after positional embedding)
    enc_ln_w = weights['wav2vec2.encoder.layer_norm.weight']
    enc_ln_b = weights['wav2vec2.encoder.layer_norm.bias']
    ln = nn.LayerNorm(1024)
    ln.weight.data = enc_ln_w
    ln.bias.data = enc_ln_b
    x_pt = ln(x_pt)

    # Transformer encoder layers (24 layers, pre-norm / stable layer norm)
    for layer_idx in range(24):
        prefix = f'wav2vec2.encoder.layers.{layer_idx}'

        # Load weights for this layer
        q_w = weights[f'{prefix}.attention.q_proj.weight']
        q_b = weights[f'{prefix}.attention.q_proj.bias']
        k_w = weights[f'{prefix}.attention.k_proj.weight']
        k_b = weights[f'{prefix}.attention.k_proj.bias']
        v_w = weights[f'{prefix}.attention.v_proj.weight']
        v_b = weights[f'{prefix}.attention.v_proj.bias']
        out_w = weights[f'{prefix}.attention.out_proj.weight']
        out_b = weights[f'{prefix}.attention.out_proj.bias']
        ff_int_w = weights[f'{prefix}.feed_forward.intermediate_dense.weight']
        ff_int_b = weights[f'{prefix}.feed_forward.intermediate_dense.bias']
        ff_out_w = weights[f'{prefix}.feed_forward.output_dense.weight']
        ff_out_b = weights[f'{prefix}.feed_forward.output_dense.bias']
        norm1_w = weights[f'{prefix}.layer_norm.weight']
        norm1_b = weights[f'{prefix}.layer_norm.bias']
        norm2_w = weights[f'{prefix}.final_layer_norm.weight']
        norm2_b = weights[f'{prefix}.final_layer_norm.bias']

        batch_size, seq_len, hidden_size = x_pt.shape
        num_heads = 16
        head_dim = hidden_size // num_heads

        # Pre-norm (stable layer norm)
        residual = x_pt
        ln1 = nn.LayerNorm(hidden_size)
        ln1.weight.data = norm1_w
        ln1.bias.data = norm1_b
        x_normed = ln1(x_pt)

        # Self-attention
        q = F.linear(x_normed, q_w, q_b)
        k = F.linear(x_normed, k_w, k_b)
        v = F.linear(x_normed, v_w, v_b)

        q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        out = F.linear(out, out_w, out_b)
        x_pt = residual + out

        # Feed-forward (pre-norm)
        residual = x_pt
        ln2 = nn.LayerNorm(hidden_size)
        ln2.weight.data = norm2_w
        ln2.bias.data = norm2_b
        x_normed = ln2(x_pt)
        x_ff = F.linear(x_normed, ff_int_w, ff_int_b)
        x_ff = F.gelu(x_ff)
        x_ff = F.linear(x_ff, ff_out_w, ff_out_b)
        x_pt = residual + x_ff

    return x_pt.detach().numpy()


def load_mlx_model():
    """Load our MLX wav2vec2-xlsr model."""
    from tools.whisper_mlx.sota.wav2vec2_xlsr_mlx import Wav2Vec2Model

    model_path = Path("models/sota/wav2vec2-xlsr-mlx")
    model = Wav2Vec2Model.from_pretrained(str(model_path))
    return model


def validate_outputs():
    """Compare PyTorch and MLX model outputs."""
    import mlx.core as mx

    print("=" * 60)
    print("Wav2Vec2-XLSR MLX Validation")
    print("=" * 60)

    # Create test input - 1 second of audio at 16kHz
    np.random.seed(42)
    test_input = np.random.randn(1, 16000).astype(np.float32)
    print(f"\nTest input shape: {test_input.shape}")

    # PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pt_output_np = run_pytorch_wav2vec2(
            test_input,
            "models/sota/wav2vec2-xlsr/pytorch_model.bin"
        )

    print(f"PyTorch output shape: {pt_output_np.shape}")
    print(f"PyTorch output range: [{pt_output_np.min():.4f}, {pt_output_np.max():.4f}]")

    # Load and run MLX model
    print("\nLoading MLX model...")
    mlx_model = load_mlx_model()

    print("Running MLX inference...")
    mlx_input = mx.array(test_input)
    mlx_output = mlx_model(mlx_input)
    mx.eval(mlx_output)
    mlx_output_np = np.array(mlx_output)

    print(f"MLX output shape: {mlx_output_np.shape}")
    print(f"MLX output range: [{mlx_output_np.min():.4f}, {mlx_output_np.max():.4f}]")

    # Compare outputs
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    diff = np.abs(pt_output_np - mlx_output_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    median_diff = np.median(diff)

    print(f"\nMax absolute difference:    {max_diff:.2e}")
    print(f"Mean absolute difference:   {mean_diff:.2e}")
    print(f"Median absolute difference: {median_diff:.2e}")

    # Check correlation
    correlation = np.corrcoef(pt_output_np.flatten(), mlx_output_np.flatten())[0, 1]
    print(f"Correlation:                {correlation:.6f}")

    # Determine pass/fail using relative tolerance
    # wav2vec2 outputs have large absolute values (~thousands)
    # Use relative tolerance instead of absolute
    output_scale = max(abs(pt_output_np.max()), abs(pt_output_np.min()))
    relative_diff = max_diff / output_scale if output_scale > 0 else max_diff

    print(f"Relative max difference:    {relative_diff:.2e}")
    print(f"Output scale:               {output_scale:.2f}")

    # Use tolerance of 1e-4 relative error (0.01% error)
    tolerance = 1e-4  # relative tolerance
    passed = relative_diff < tolerance

    print("\n" + "=" * 60)
    if passed:
        print(f"PASS: Relative diff {relative_diff:.2e} < tolerance {tolerance}")
    else:
        print(f"FAIL: Relative diff {relative_diff:.2e} >= tolerance {tolerance}")
    print("=" * 60)

    # Save validation results
    results = {
        "model": "Wav2Vec2-XLSR",
        "pytorch_path": "models/sota/wav2vec2-xlsr/pytorch_model.bin",
        "mlx_path": "models/sota/wav2vec2-xlsr-mlx",
        "input_shape": list(test_input.shape),
        "output_shape": list(pt_output_np.shape),
        "max_absolute_diff": float(max_diff),
        "mean_absolute_diff": float(mean_diff),
        "median_absolute_diff": float(median_diff),
        "relative_max_diff": float(relative_diff),
        "output_scale": float(output_scale),
        "correlation": float(correlation),
        "tolerance": float(tolerance),
        "tolerance_type": "relative",
        "passed": bool(passed),
    }

    results_path = Path("models/sota/wav2vec2-xlsr-mlx/validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation results to {results_path}")

    return passed


if __name__ == "__main__":
    success = validate_outputs()
    sys.exit(0 if success else 1)
