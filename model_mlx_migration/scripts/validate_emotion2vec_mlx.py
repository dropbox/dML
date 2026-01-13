#!/usr/bin/env python3
"""Validate Emotion2vec MLX implementation against PyTorch reference.

This script compares outputs from PyTorch and MLX emotion2vec models
to verify numerical equivalence.

Since FunASR may not be installed, this uses a minimal PyTorch implementation
that loads the original checkpoint and computes forward pass.

Usage:
    python scripts/validate_emotion2vec_mlx.py
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


def run_pytorch_emotion2vec(x: np.ndarray, checkpoint_path: str) -> np.ndarray:
    """Run emotion2vec forward pass using PyTorch with original weights.

    This function directly loads weights and computes the forward pass
    to match the original FunASR implementation exactly.

    Args:
        x: Input audio (batch, samples)
        checkpoint_path: Path to emotion2vec checkpoint

    Returns:
        Output features (batch, frames + 10, 768)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("model", checkpoint)

    x_pt = torch.from_numpy(x).unsqueeze(-1)  # (batch, samples, 1)

    # Local encoder (7 conv layers)
    spec = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
    for i in range(7):
        w = weights[f'modality_encoders.AUDIO.local_encoder.conv_layers.{i}.0.weight']
        ln_w = weights[f'modality_encoders.AUDIO.local_encoder.conv_layers.{i}.2.1.weight']
        ln_b = weights[f'modality_encoders.AUDIO.local_encoder.conv_layers.{i}.2.1.bias']
        _, kernel_size, stride = spec[i]
        x_pt_t = x_pt.transpose(1, 2)
        x_pt_t = F.conv1d(x_pt_t, w, stride=stride)
        x_pt = x_pt_t.transpose(1, 2)
        ln = nn.LayerNorm(512)
        ln.weight.data = ln_w
        ln.bias.data = ln_b
        x_pt = ln(x_pt)
        x_pt = F.gelu(x_pt)

    # Project features
    proj_ln_w = weights['modality_encoders.AUDIO.project_features.1.weight']
    proj_ln_b = weights['modality_encoders.AUDIO.project_features.1.bias']
    proj_w = weights['modality_encoders.AUDIO.project_features.2.weight']
    proj_b = weights['modality_encoders.AUDIO.project_features.2.bias']
    ln = nn.LayerNorm(512)
    ln.weight.data = proj_ln_w
    ln.bias.data = proj_ln_b
    x_pt = ln(x_pt)
    x_pt = F.linear(x_pt, proj_w, proj_b)

    # Relative positional encoder (5 conv layers)
    for i in range(5):
        w = weights[f'modality_encoders.AUDIO.relative_positional_encoder.{i+1}.0.weight']
        b = weights[f'modality_encoders.AUDIO.relative_positional_encoder.{i+1}.0.bias']
        residual = x_pt
        x_pos_t = x_pt.transpose(1, 2)
        x_pos_t = F.conv1d(x_pos_t, w, b, padding=9, groups=16)
        x_pt = x_pos_t.transpose(1, 2)
        x_pt = F.gelu(x_pt)
        x_pt = x_pt + residual

    # Prepend extra tokens
    extra_tokens = weights['modality_encoders.AUDIO.extra_tokens']
    x_pt = torch.cat([extra_tokens.expand(x_pt.shape[0], -1, -1), x_pt], dim=1)

    # Compute ALiBi bias
    seq_len = x_pt.shape[1]
    positions = torch.arange(seq_len)
    relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
    slopes = torch.tensor([2 ** (-8 * i / 12) for i in range(1, 13)])
    alibi_bias = slopes.view(1, 12, 1, 1) * relative_pos.view(1, 1, seq_len, seq_len).float()
    alibi_scale = weights['modality_encoders.AUDIO.alibi_scale'].squeeze(1)
    alibi_bias = alibi_bias * alibi_scale

    # Helper function for transformer block
    def transformer_block(x, prefix):
        qkv_w = weights[f'{prefix}.attn.qkv.weight']
        qkv_b = weights[f'{prefix}.attn.qkv.bias']
        proj_w = weights[f'{prefix}.attn.proj.weight']
        proj_b = weights[f'{prefix}.attn.proj.bias']
        fc1_w = weights[f'{prefix}.mlp.fc1.weight']
        fc1_b = weights[f'{prefix}.mlp.fc1.bias']
        fc2_w = weights[f'{prefix}.mlp.fc2.weight']
        fc2_b = weights[f'{prefix}.mlp.fc2.bias']
        norm1_w = weights[f'{prefix}.norm1.weight']
        norm1_b = weights[f'{prefix}.norm1.bias']
        norm2_w = weights[f'{prefix}.norm2.weight']
        norm2_b = weights[f'{prefix}.norm2.bias']

        batch_size, seq_len, embed_dim = x.shape
        residual = x
        qkv = F.linear(x, qkv_w, qkv_b)
        qkv = qkv.reshape(batch_size, seq_len, 3, 12, 64)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * (64 ** -0.5)
        attn = attn + alibi_bias
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        out = F.linear(out, proj_w, proj_b)

        x = residual + out
        ln1 = nn.LayerNorm(768)
        ln1.weight.data = norm1_w
        ln1.bias.data = norm1_b
        x = ln1(x)

        residual = x
        x = F.linear(x, fc1_w, fc1_b)
        x = F.gelu(x)
        x = F.linear(x, fc2_w, fc2_b)
        x = residual + x
        ln2 = nn.LayerNorm(768)
        ln2.weight.data = norm2_w
        ln2.bias.data = norm2_b
        x = ln2(x)
        return x

    # Context encoder (4 blocks)
    for block_idx in range(4):
        x_pt = transformer_block(x_pt, f'modality_encoders.AUDIO.context_encoder.blocks.{block_idx}')

    # Context norm
    context_norm_w = weights['modality_encoders.AUDIO.context_encoder.norm.weight']
    context_norm_b = weights['modality_encoders.AUDIO.context_encoder.norm.bias']
    ln = nn.LayerNorm(768)
    ln.weight.data = context_norm_w
    ln.bias.data = context_norm_b
    x_pt = ln(x_pt)

    # Main encoder (8 blocks)
    for block_idx in range(8):
        x_pt = transformer_block(x_pt, f'blocks.{block_idx}')

    return x_pt.detach().numpy()


def load_mlx_model():
    """Load our MLX emotion2vec model."""
    from tools.whisper_mlx.sota.emotion2vec_mlx import Emotion2vecModel

    model_path = Path("models/sota/emotion2vec-mlx")
    model = Emotion2vecModel.from_pretrained(str(model_path))
    return model


def validate_outputs():
    """Compare PyTorch and MLX model outputs."""
    import mlx.core as mx

    print("=" * 60)
    print("Emotion2vec MLX Validation")
    print("=" * 60)

    # Create test input - 1 second of audio at 16kHz
    np.random.seed(42)
    test_input = np.random.randn(1, 16000).astype(np.float32)
    print(f"\nTest input shape: {test_input.shape}")

    # PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pt_output_np = run_pytorch_emotion2vec(test_input, "models/sota/emotion2vec/emotion2vec_base.pt")

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

    # Determine pass/fail
    tolerance = 1e-3
    passed = max_diff < tolerance

    print("\n" + "=" * 60)
    if passed:
        print(f"PASS: Max diff {max_diff:.2e} < tolerance {tolerance}")
    else:
        print(f"FAIL: Max diff {max_diff:.2e} >= tolerance {tolerance}")
    print("=" * 60)

    # Save validation results
    results = {
        "model": "Emotion2vec",
        "pytorch_path": "models/sota/emotion2vec/emotion2vec_base.pt",
        "mlx_path": "models/sota/emotion2vec-mlx",
        "input_shape": list(test_input.shape),
        "output_shape": list(pt_output_np.shape),
        "max_absolute_diff": float(max_diff),
        "mean_absolute_diff": float(mean_diff),
        "median_absolute_diff": float(median_diff),
        "correlation": float(correlation),
        "tolerance": float(tolerance),
        "passed": bool(passed),
    }

    results_path = Path("models/sota/emotion2vec-mlx/validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation results to {results_path}")

    return passed


if __name__ == "__main__":
    success = validate_outputs()
    sys.exit(0 if success else 1)
