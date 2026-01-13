#!/usr/bin/env python3
"""Validate BEATs MLX conversion against PyTorch (direct implementation).

Uses a minimal PyTorch BEATs implementation to avoid speechbrain dependency issues.
"""

import sys
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import mlx.core as mx
import torchaudio.compliance.kaldi as ta_kaldi

from whisper_mlx.sota.beats_mlx import BEATsModel


def compute_relative_position_bucket_pt(
    relative_position: torch.Tensor,
    bidirectional: bool = True,
    num_buckets: int = 320,
    max_distance: int = 800,
) -> torch.Tensor:
    """Compute relative position bucket indices (T5-style) in PyTorch."""
    relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)

    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)

    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(
        is_small,
        relative_position.to(torch.long),
        relative_position_if_large,
    )

    return relative_buckets


class PyTorchBEATsAttention(nn.Module):
    """PyTorch BEATs attention for validation."""

    def __init__(self, embed_dim=768, num_heads=12, num_buckets=320, max_distance=800, deep_norm=True, num_layers=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.grep_linear = nn.Linear(self.head_dim, 8)
        self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.scale = self.head_dim ** -0.5
        self.deep_norm_alpha = (2.0 * num_layers) ** 0.25 if deep_norm else 1.0

    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        context_position = torch.arange(query_length)[:, None]
        memory_position = torch.arange(key_length)[None, :]
        relative_position = memory_position - context_position

        bucket = compute_relative_position_bucket_pt(
            relative_position,
            bidirectional=True,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        values = self.relative_attention_bias(bucket)  # (q, k, heads)
        values = values.permute(2, 0, 1).unsqueeze(0)  # (1, heads, q, k)
        return values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        alpha = 32.0

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scale q before reshape
        q = q * self.scale / alpha

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1))

        # Position bias
        position_bias = self.compute_bias(seq_len, seq_len)

        # GREP gating
        query_layer = q * alpha / self.scale
        grep_out = self.grep_linear(query_layer)  # (batch, heads, seq, 8)
        grep_out = grep_out.view(batch_size, self.num_heads, seq_len, 2, 4)
        grep_out = grep_out.sum(dim=-1)  # (batch, heads, seq, 2)
        grep_out = torch.sigmoid(grep_out)

        gate_a = grep_out[:, :, :, 0:1]
        gate_b = grep_out[:, :, :, 1:2]
        gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0

        attn_mask_rel_pos = gate_a_1 * position_bias
        attn = attn + attn_mask_rel_pos

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)

        return out


class PyTorchBEATsEncoderLayer(nn.Module):
    """PyTorch BEATs encoder layer for validation."""

    def __init__(self, embed_dim=768, ffn_dim=3072, num_heads=12, num_buckets=320, max_distance=800, deep_norm=True, num_layers=12):
        super().__init__()
        self.self_attn = PyTorchBEATsAttention(embed_dim, num_heads, num_buckets, max_distance, deep_norm, num_layers)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

        self.deep_norm_alpha = (2.0 * num_layers) ** 0.25 if deep_norm else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn(x)
        x = residual * self.deep_norm_alpha + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = residual * self.deep_norm_alpha + x
        x = self.final_layer_norm(x)

        return x


class PyTorchBEATsPositionalConv(nn.Module):
    """Positional conv with weight normalization (pre-computed weight)."""

    def __init__(self, embed_dim=768, kernel_size=128, groups=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = kernel_size // 2

        # Pre-computed effective weight (from weight_g * normalize(weight_v))
        self.weight = nn.Parameter(torch.zeros(embed_dim, kernel_size, embed_dim // groups))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        group_size = hidden_size // self.groups

        # Pad input
        x_padded = F.pad(x.transpose(1, 2), (self.padding, self.padding))  # (B, C, T+2*pad)

        # Manual grouped convolution
        outputs = []
        for g in range(self.groups):
            x_g = x_padded[:, g * group_size:(g + 1) * group_size, :]  # (B, C/groups, T+2*pad)
            w_g = self.weight[g * group_size:(g + 1) * group_size]  # (C/groups, K, C/groups)
            w_g = w_g.transpose(1, 2)  # (C/groups, C/groups, K) for conv1d
            conv_out = F.conv1d(x_g, w_g)  # (B, C/groups, T+1)
            outputs.append(conv_out)

        out = torch.cat(outputs, dim=1)  # (B, C, T+1)
        out = out[:, :, :-1]  # Trim to match input length
        out = out.transpose(1, 2) + self.bias  # (B, T, C)
        out = F.gelu(out)

        return out


class PyTorchBEATsEncoder(nn.Module):
    """PyTorch BEATs encoder for validation."""

    def __init__(self, embed_dim=768, ffn_dim=3072, num_heads=12, num_layers=12, num_buckets=320, max_distance=800, conv_pos=128, conv_pos_groups=16, deep_norm=True):
        super().__init__()
        self.pos_conv = PyTorchBEATsPositionalConv(embed_dim, conv_pos, conv_pos_groups)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.layers = nn.ModuleList([
            PyTorchBEATsEncoderLayer(embed_dim, ffn_dim, num_heads, num_buckets, max_distance, deep_norm, num_layers)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x


class PyTorchBEATsModel(nn.Module):
    """PyTorch BEATs model for validation."""

    def __init__(self, patch_embed_dim=512, encoder_embed_dim=768, ffn_dim=3072, num_heads=12, num_layers=12, patch_size=16, num_buckets=320, max_distance=800, conv_pos=128, conv_pos_groups=16, deep_norm=True):
        super().__init__()
        self.patch_embedding = nn.Conv2d(1, patch_embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.layer_norm = nn.LayerNorm(patch_embed_dim, eps=1e-5)
        self.post_extract_proj = nn.Linear(patch_embed_dim, encoder_embed_dim)
        self.encoder = PyTorchBEATsEncoder(encoder_embed_dim, ffn_dim, num_heads, num_layers, num_buckets, max_distance, conv_pos, conv_pos_groups, deep_norm)

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        # fbank: (batch, freq, time)
        if fbank.ndim == 3:
            fbank = fbank.unsqueeze(1)  # (batch, 1, freq, time)

        # Patch embedding
        x = self.patch_embedding(fbank)  # (batch, embed_dim, freq//16, time//16)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)

        # Layer norm
        x = self.layer_norm(x)

        # Project to encoder dim
        x = self.post_extract_proj(x)

        # Encoder
        x = self.encoder(x)

        return x


def load_weights_to_pytorch(pt_model: PyTorchBEATsModel, checkpoint_path: str):
    """Load weights from BEATs checkpoint to our PyTorch model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = ckpt["model"]

    # Patch embedding
    pt_model.patch_embedding.weight.data = weights["patch_embedding.weight"]

    # Layer norm
    pt_model.layer_norm.weight.data = weights["layer_norm.weight"]
    pt_model.layer_norm.bias.data = weights["layer_norm.bias"]

    # Post extract projection
    pt_model.post_extract_proj.weight.data = weights["post_extract_proj.weight"]
    pt_model.post_extract_proj.bias.data = weights["post_extract_proj.bias"]

    # Positional conv (compute effective weight from weight_g and weight_v)
    weight_g = weights["encoder.pos_conv.0.weight_g"]  # (1, 1, 768)
    weight_v = weights["encoder.pos_conv.0.weight_v"]  # (768, 128, 48)
    effective_weight = weight_g * F.normalize(weight_v, dim=[0, 1])
    pt_model.encoder.pos_conv.weight.data = effective_weight.transpose(1, 2)  # (768, 48, 128) -> (768, 128, 48)
    pt_model.encoder.pos_conv.bias.data = weights["encoder.pos_conv.0.bias"]

    # Encoder layer norm
    pt_model.encoder.layer_norm.weight.data = weights["encoder.layer_norm.weight"]
    pt_model.encoder.layer_norm.bias.data = weights["encoder.layer_norm.bias"]

    # Encoder layers
    for i, layer in enumerate(pt_model.encoder.layers):
        prefix = f"encoder.layers.{i}"

        # Self attention
        layer.self_attn.q_proj.weight.data = weights[f"{prefix}.self_attn.q_proj.weight"]
        layer.self_attn.q_proj.bias.data = weights[f"{prefix}.self_attn.q_proj.bias"]
        layer.self_attn.k_proj.weight.data = weights[f"{prefix}.self_attn.k_proj.weight"]
        layer.self_attn.k_proj.bias.data = weights[f"{prefix}.self_attn.k_proj.bias"]
        layer.self_attn.v_proj.weight.data = weights[f"{prefix}.self_attn.v_proj.weight"]
        layer.self_attn.v_proj.bias.data = weights[f"{prefix}.self_attn.v_proj.bias"]
        layer.self_attn.out_proj.weight.data = weights[f"{prefix}.self_attn.out_proj.weight"]
        layer.self_attn.out_proj.bias.data = weights[f"{prefix}.self_attn.out_proj.bias"]

        # Relative attention bias
        layer.self_attn.relative_attention_bias.weight.data = weights[f"{prefix}.self_attn.relative_attention_bias.weight"]

        # GREP
        layer.self_attn.grep_linear.weight.data = weights[f"{prefix}.self_attn.grep_linear.weight"]
        layer.self_attn.grep_linear.bias.data = weights[f"{prefix}.self_attn.grep_linear.bias"]
        layer.self_attn.grep_a.data = weights[f"{prefix}.self_attn.grep_a"]

        # Layer norms
        layer.self_attn_layer_norm.weight.data = weights[f"{prefix}.self_attn_layer_norm.weight"]
        layer.self_attn_layer_norm.bias.data = weights[f"{prefix}.self_attn_layer_norm.bias"]
        layer.final_layer_norm.weight.data = weights[f"{prefix}.final_layer_norm.weight"]
        layer.final_layer_norm.bias.data = weights[f"{prefix}.final_layer_norm.bias"]

        # FFN
        layer.fc1.weight.data = weights[f"{prefix}.fc1.weight"]
        layer.fc1.bias.data = weights[f"{prefix}.fc1.bias"]
        layer.fc2.weight.data = weights[f"{prefix}.fc2.weight"]
        layer.fc2.bias.data = weights[f"{prefix}.fc2.bias"]

    return pt_model


def create_fbank_pytorch(waveform: torch.Tensor, fbank_mean: float = 15.41663, fbank_std: float = 6.55582):
    """Create normalized fbank features matching BEATs preprocessing."""
    fbanks = []
    for wav in waveform:
        wav = wav.unsqueeze(0) * 2**15
        fbank = ta_kaldi.fbank(
            wav,
            num_mel_bins=128,
            sample_frequency=16000,
            frame_length=25,
            frame_shift=10,
        )
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    return (fbank - fbank_mean) / (2 * fbank_std)


def main():
    pt_checkpoint = "models/sota/beats/models/sota/beats/BEATs_iter3_plus_AS2M.pt"
    mlx_checkpoint = "models/sota/beats-mlx"

    print("=" * 60)
    print("BEATs MLX Validation (Direct Implementation)")
    print("=" * 60)

    # Create and load PyTorch model
    print("\n1. Loading PyTorch model...")
    pt_model = PyTorchBEATsModel()
    pt_model = load_weights_to_pytorch(pt_model, pt_checkpoint)
    pt_model.eval()
    print("   PyTorch model loaded")

    # Load MLX model
    print("\n2. Loading MLX model...")
    mlx_model = BEATsModel.from_pretrained(mlx_checkpoint)
    mx.eval(mlx_model.parameters())
    print("   MLX model loaded")

    # Create test input
    print("\n3. Creating test input...")
    np.random.seed(42)
    test_audio = np.random.randn(1, 16000).astype(np.float32)
    test_audio_pt = torch.from_numpy(test_audio)

    # Create fbank features
    fbank_pt = create_fbank_pytorch(test_audio_pt)
    fbank_np = fbank_pt.numpy()
    fbank_mlx = mx.array(fbank_np)

    print(f"   Fbank shape: {fbank_pt.shape} (batch, time, freq)")

    # PyTorch forward
    print("\n4. Running PyTorch forward...")
    with torch.no_grad():
        pt_out = pt_model(fbank_pt)
    pt_out_np = pt_out.numpy()
    print(f"   PT output shape: {pt_out_np.shape}")

    # MLX forward
    print("\n5. Running MLX forward...")
    mlx_out = mlx_model(fbank_mlx)
    mx.eval(mlx_out)
    mlx_out_np = np.array(mlx_out)
    print(f"   MLX output shape: {mlx_out_np.shape}")

    # Compare
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    if pt_out_np.shape != mlx_out_np.shape:
        print("\nERROR: Shape mismatch!")
        print(f"  PyTorch: {pt_out_np.shape}")
        print(f"  MLX: {mlx_out_np.shape}")
        return 1

    abs_diff = np.abs(pt_out_np - mlx_out_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    print("\nAbsolute Differences:")
    print(f"  Max:  {max_diff:.6e}")
    print(f"  Mean: {mean_diff:.6e}")

    print("\nSample values (position [0, 0, :5]):")
    print(f"  PyTorch: {pt_out_np[0, 0, :5]}")
    print(f"  MLX:     {mlx_out_np[0, 0, :5]}")

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
