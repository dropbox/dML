#!/usr/bin/env python3
"""Validate Wav2Vec2-XLSR-SER MLX conversion against PyTorch.

This script directly loads weights and computes forward pass to avoid
key naming mismatches between different HuggingFace versions.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import mlx.core as mx

from whisper_mlx.sota.wav2vec2_xlsr_mlx import Wav2Vec2ForSequenceClassification


def run_pytorch_forward(test_audio: np.ndarray, weights_path: str) -> np.ndarray:
    """Run wav2vec2-xlsr-ser forward pass using PyTorch with original weights.

    Args:
        test_audio: Input audio (batch, samples)
        weights_path: Path to model.safetensors

    Returns:
        Logits (batch, num_labels)
    """
    # Load weights
    weights = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    x_pt = torch.from_numpy(test_audio).unsqueeze(-1)  # (batch, samples, 1)

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
        x_pt_t = x_pt.transpose(1, 2)
        x_pt_t = F.conv1d(x_pt_t, w, b, stride=stride)
        x_pt = x_pt_t.transpose(1, 2)

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

    # Positional conv embedding
    weight_g = weights['wav2vec2.encoder.pos_conv_embed.conv.weight_g']
    weight_v = weights['wav2vec2.encoder.pos_conv_embed.conv.weight_v']
    pos_bias = weights['wav2vec2.encoder.pos_conv_embed.conv.bias']

    # Weight normalization: g * v / ||v||_2 per filter
    norm = weight_v.pow(2).sum(dim=[1, 2], keepdim=True).sqrt()
    effective_weight = weight_g * weight_v / norm

    kernel_size = 128
    padding = kernel_size // 2
    x_pt_t = x_pt.transpose(1, 2)
    x_pt_t = F.conv1d(x_pt_t, effective_weight, pos_bias, padding=padding, groups=16)
    x_pt_t = x_pt_t[:, :, :-1]
    x_pt_pos = x_pt_t.transpose(1, 2)
    x_pt_pos = F.gelu(x_pt_pos)
    x_pt = x_pt + x_pt_pos

    # Encoder layer norm
    enc_ln_w = weights['wav2vec2.encoder.layer_norm.weight']
    enc_ln_b = weights['wav2vec2.encoder.layer_norm.bias']
    ln = nn.LayerNorm(1024)
    ln.weight.data = enc_ln_w
    ln.bias.data = enc_ln_b
    x_pt = ln(x_pt)

    # Transformer encoder layers (24 layers, pre-norm)
    for layer_idx in range(24):
        prefix = f'wav2vec2.encoder.layers.{layer_idx}'

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

    # Mean pooling
    pooled = x_pt.mean(dim=1)  # (batch, hidden_size)

    # Classification head
    # Note: checkpoint uses classifier.dense/classifier.output naming
    proj_w = weights['classifier.dense.weight']
    proj_b = weights['classifier.dense.bias']
    cls_w = weights['classifier.output.weight']
    cls_b = weights['classifier.output.bias']

    x = F.linear(pooled, proj_w, proj_b)
    x = torch.tanh(x)
    logits = F.linear(x, cls_w, cls_b)

    return logits.detach().numpy()


def main():
    weights_path = "models/sota/wav2vec2-xlsr-ser/model.safetensors"
    mlx_model_path = "models/sota/wav2vec2-xlsr-ser-mlx"

    print("=" * 60)
    print("Wav2Vec2-XLSR-SER MLX Validation")
    print("=" * 60)

    # Create test input
    print("\n1. Creating test input...")
    np.random.seed(42)
    test_audio = np.random.randn(1, 16000).astype(np.float32)
    print(f"   Audio shape: {test_audio.shape}")

    # PyTorch forward (manual)
    print("\n2. Running PyTorch forward (manual)...")
    with torch.no_grad():
        pt_out = run_pytorch_forward(test_audio, weights_path)
    print(f"   PT output shape: {pt_out.shape}")
    print(f"   PT logits: {pt_out[0]}")

    # Load MLX model
    print("\n3. Loading MLX model...")
    mlx_model = Wav2Vec2ForSequenceClassification.from_pretrained(mlx_model_path)
    mx.eval(mlx_model.parameters())
    print(f"   Config: hidden_size={mlx_model.config.hidden_size}, num_labels={mlx_model.num_labels}")

    # MLX forward
    print("\n4. Running MLX forward...")
    test_audio_mlx = mx.array(test_audio)
    mlx_out = mlx_model(test_audio_mlx)
    mx.eval(mlx_out)
    mlx_out_np = np.array(mlx_out)
    print(f"   MLX output shape: {mlx_out_np.shape}")
    print(f"   MLX logits: {mlx_out_np[0]}")

    # Compare
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    abs_diff = np.abs(pt_out - mlx_out_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    print("\nAbsolute Differences:")
    print(f"  Max:  {max_diff:.6e}")
    print(f"  Mean: {mean_diff:.6e}")

    print("\nLogits comparison:")
    id2label = {0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
                4: "happy", 5: "neutral", 6: "sad", 7: "surprised"}
    for i, (pt_val, mlx_val) in enumerate(zip(pt_out[0], mlx_out_np[0])):
        diff = abs(pt_val - mlx_val)
        print(f"  [{i}] {id2label[i]:10s} PT={pt_val:9.5f}, MLX={mlx_val:9.5f}, diff={diff:.6e}")

    pt_pred = np.argmax(pt_out[0])
    mlx_pred = np.argmax(mlx_out_np[0])

    print("\nPredictions:")
    print(f"  PyTorch: {id2label[pt_pred]} ({pt_pred})")
    print(f"  MLX:     {id2label[mlx_pred]} ({mlx_pred})")
    print(f"  Match:   {'Yes' if pt_pred == mlx_pred else 'No'}")

    # Pass/fail
    threshold = 1e-4  # Relaxed threshold for deep network
    if max_diff < threshold:
        print(f"\n PASS: max_diff={max_diff:.6e} < {threshold}")
        return 0
    else:
        print(f"\n FAIL: max_diff={max_diff:.6e} >= {threshold}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
