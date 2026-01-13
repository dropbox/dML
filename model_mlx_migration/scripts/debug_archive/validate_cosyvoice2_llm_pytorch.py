#!/usr/bin/env python3
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
CosyVoice2 LLM PyTorch vs MLX Numerical Comparison.

Validates that the MLX implementation matches PyTorch numerically.
Target accuracy: < 1e-5 max error.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from typing import Optional, Tuple, cast

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# PyTorch Reference Implementation
# ============================================================================


class PyTorchQwen2RMSNorm(nn.Module):
    """RMSNorm for Qwen2."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def pytorch_precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE frequencies for PyTorch."""
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def pytorch_apply_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE for PyTorch."""
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    cos = cos[:seq_len][None, None, :, :]
    sin = sin[:seq_len][None, None, :, :]

    half_dim = head_dim // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    cos_half = cos[..., :half_dim]
    sin_half = sin[..., :half_dim]

    q_rotated = torch.cat(
        [
            q1 * cos_half - q2 * sin_half,
            q1 * sin_half + q2 * cos_half,
        ],
        dim=-1,
    )

    k_rotated = torch.cat(
        [
            k1 * cos_half - k2 * sin_half,
            k1 * sin_half + k2 * cos_half,
        ],
        dim=-1,
    )

    return q_rotated, k_rotated


class PyTorchQwen2Attention(nn.Module):
    """Qwen2 GQA attention in PyTorch."""

    def __init__(
        self,
        hidden_size: int = 896,
        num_heads: int = 7,
        num_kv_heads: int = 1,
        head_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.num_heads_per_kv = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = pytorch_apply_rotary_embedding(q, k, cos, sin)

        # GQA: expand KV heads
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads_per_kv, dim=1)
            v = v.repeat_interleave(self.num_heads_per_kv, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        out = self.o_proj(out)

        return cast(torch.Tensor, out)


class PyTorchQwen2MLP(nn.Module):
    """Qwen2 SwiGLU MLP in PyTorch."""

    def __init__(self, hidden_size: int = 896, intermediate_size: int = 4864):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


class PyTorchQwen2DecoderLayer(nn.Module):
    """Qwen2 decoder layer in PyTorch."""

    def __init__(
        self,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
        num_heads: int = 7,
        num_kv_heads: int = 1,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = PyTorchQwen2Attention(
            hidden_size, num_heads, num_kv_heads, head_dim
        )
        self.mlp = PyTorchQwen2MLP(hidden_size, intermediate_size)
        self.input_layernorm = PyTorchQwen2RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = PyTorchQwen2RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class PyTorchQwen2Model(nn.Module):
    """Qwen2 transformer model in PyTorch."""

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
        num_layers: int = 24,
        num_heads: int = 7,
        num_kv_heads: int = 1,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_seq_len: int = 32768,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                PyTorchQwen2DecoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = PyTorchQwen2RMSNorm(hidden_size, rms_norm_eps)

        # Precompute RoPE
        self.register_buffer("rope_cos", None)
        self.register_buffer("rope_sin", None)
        cos, sin = pytorch_precompute_rope_frequencies(
            head_dim, max_seq_len, rope_theta
        )
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        # Causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        mask = mask[None, None, :, :]

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, self.rope_cos, self.rope_sin)

        hidden_states = self.norm(hidden_states)
        return cast(torch.Tensor, hidden_states)


class PyTorchCosyVoice2LLM(nn.Module):
    """CosyVoice2 LLM in PyTorch for validation."""

    def __init__(
        self,
        vocab_size: int = 151936,
        speech_vocab_size: int = 6564,
        llm_embedding_size: int = 2,
        hidden_size: int = 896,
    ):
        super().__init__()
        self.llm_embedding = nn.Embedding(llm_embedding_size, hidden_size)
        self.speech_embedding = nn.Embedding(speech_vocab_size, hidden_size)
        self.llm = PyTorchQwen2Model(vocab_size=vocab_size, hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.llm_decoder = nn.Linear(hidden_size, speech_vocab_size, bias=True)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.llm(input_ids)
        text_logits = self.lm_head(hidden_states)
        speech_logits = self.llm_decoder(hidden_states)
        return text_logits, speech_logits

    def load_weights(self, state_dict: dict):
        """Load weights from CosyVoice2 llm.pt state dict."""
        # LLM embedding
        if "llm_embedding.weight" in state_dict:
            self.llm_embedding.weight.data = state_dict["llm_embedding.weight"]

        # Speech embedding
        if "speech_embedding.weight" in state_dict:
            self.speech_embedding.weight.data = state_dict["speech_embedding.weight"]

        # Text embedding
        if "llm.model.model.embed_tokens.weight" in state_dict:
            self.llm.embed_tokens.weight.data = state_dict[
                "llm.model.model.embed_tokens.weight"
            ]

        # Layers
        for i in range(24):
            prefix = f"llm.model.model.layers.{i}"
            layer = cast(PyTorchQwen2DecoderLayer, self.llm.layers[i])

            # Attention
            if f"{prefix}.self_attn.q_proj.weight" in state_dict:
                layer.self_attn.q_proj.weight.data = state_dict[
                    f"{prefix}.self_attn.q_proj.weight"
                ]
                layer.self_attn.q_proj.bias.data = state_dict[
                    f"{prefix}.self_attn.q_proj.bias"
                ]
                layer.self_attn.k_proj.weight.data = state_dict[
                    f"{prefix}.self_attn.k_proj.weight"
                ]
                layer.self_attn.k_proj.bias.data = state_dict[
                    f"{prefix}.self_attn.k_proj.bias"
                ]
                layer.self_attn.v_proj.weight.data = state_dict[
                    f"{prefix}.self_attn.v_proj.weight"
                ]
                layer.self_attn.v_proj.bias.data = state_dict[
                    f"{prefix}.self_attn.v_proj.bias"
                ]
                layer.self_attn.o_proj.weight.data = state_dict[
                    f"{prefix}.self_attn.o_proj.weight"
                ]

            # MLP
            if f"{prefix}.mlp.gate_proj.weight" in state_dict:
                layer.mlp.gate_proj.weight.data = state_dict[
                    f"{prefix}.mlp.gate_proj.weight"
                ]
                layer.mlp.up_proj.weight.data = state_dict[
                    f"{prefix}.mlp.up_proj.weight"
                ]
                layer.mlp.down_proj.weight.data = state_dict[
                    f"{prefix}.mlp.down_proj.weight"
                ]

            # Norms
            if f"{prefix}.input_layernorm.weight" in state_dict:
                layer.input_layernorm.weight.data = state_dict[
                    f"{prefix}.input_layernorm.weight"
                ]
                layer.post_attention_layernorm.weight.data = state_dict[
                    f"{prefix}.post_attention_layernorm.weight"
                ]

        # Final norm
        if "llm.model.model.norm.weight" in state_dict:
            self.llm.norm.weight.data = state_dict["llm.model.model.norm.weight"]

        # LM head
        if "llm.model.lm_head.weight" in state_dict:
            self.lm_head.weight.data = state_dict["llm.model.lm_head.weight"]

        # Speech decoder
        if "llm_decoder.weight" in state_dict:
            self.llm_decoder.weight.data = state_dict["llm_decoder.weight"]
            self.llm_decoder.bias.data = state_dict["llm_decoder.bias"]


# ============================================================================
# Validation
# ============================================================================


def find_llm_pt():
    """Find llm.pt file."""
    paths = [
        os.path.expanduser("~/.cache/cosyvoice2/cosyvoice2-0.5b/llm.pt"),
        "./models/cosyvoice2/llm.pt",
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def validate_pytorch_vs_mlx():
    """Validate MLX LLM matches PyTorch numerically."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM as MLXCosyVoice2LLM,
    )
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        Qwen2Config,
    )

    print("CosyVoice2 LLM: PyTorch vs MLX Numerical Comparison")
    print("=" * 60)

    # Find weights
    llm_path = find_llm_pt()
    if llm_path is None:
        print("ERROR: llm.pt not found")
        return 1

    print(f"Loading weights from: {llm_path}")
    state_dict = torch.load(llm_path, map_location="cpu", weights_only=False)

    # Create models
    print("\nCreating PyTorch model...")
    pt_model = PyTorchCosyVoice2LLM()
    pt_model.load_weights(state_dict)
    pt_model.eval()

    print("Creating MLX model...")
    config = Qwen2Config()
    mlx_model = MLXCosyVoice2LLM(config)
    mlx_model._load_weights(state_dict)

    # Test inputs - use deterministic values
    print("\n" + "=" * 60)
    print("Testing with small input (batch=1, seq_len=8)")
    print("=" * 60)

    # Use fixed input for reproducibility
    batch, seq_len = 1, 8
    np.random.seed(42)
    input_ids_np = np.random.randint(0, 1000, (batch, seq_len)).astype(np.int64)

    pt_input = torch.tensor(input_ids_np)
    mlx_input = mx.array(input_ids_np.astype(np.int32))

    print(f"Input IDs: {input_ids_np.tolist()}")

    # PyTorch forward pass
    print("\nRunning PyTorch forward pass...")
    with torch.no_grad():
        pt_text_logits, pt_speech_logits = pt_model(pt_input)

    # MLX forward pass
    print("Running MLX forward pass...")
    mlx_text_logits, mlx_speech_logits, _ = mlx_model(mlx_input)
    mx.eval(mlx_text_logits, mlx_speech_logits)

    # Convert to numpy for comparison
    pt_text_np = pt_text_logits.numpy()
    pt_speech_np = pt_speech_logits.numpy()
    mlx_text_np = np.array(mlx_text_logits)
    mlx_speech_np = np.array(mlx_speech_logits)

    # Compare
    print("\n" + "=" * 60)
    print("Numerical Comparison")
    print("=" * 60)

    # Text logits comparison
    text_diff = np.abs(pt_text_np - mlx_text_np)
    text_max_err = np.max(text_diff)
    text_mean_err = np.mean(text_diff)
    text_rel_err = text_max_err / (np.abs(pt_text_np).max() + 1e-8)

    print(f"\nText Logits ({pt_text_np.shape}):")
    print(f"  PyTorch - mean: {pt_text_np.mean():.6f}, std: {pt_text_np.std():.6f}")
    print(f"  MLX     - mean: {mlx_text_np.mean():.6f}, std: {mlx_text_np.std():.6f}")
    print(f"  Max absolute error: {text_max_err:.2e}")
    print(f"  Mean absolute error: {text_mean_err:.2e}")
    print(f"  Relative error: {text_rel_err:.2e}")

    # Speech logits comparison
    speech_diff = np.abs(pt_speech_np - mlx_speech_np)
    speech_max_err = np.max(speech_diff)
    speech_mean_err = np.mean(speech_diff)
    speech_rel_err = speech_max_err / (np.abs(pt_speech_np).max() + 1e-8)

    print(f"\nSpeech Logits ({pt_speech_np.shape}):")
    print(f"  PyTorch - mean: {pt_speech_np.mean():.6f}, std: {pt_speech_np.std():.6f}")
    print(
        f"  MLX     - mean: {mlx_speech_np.mean():.6f}, std: {mlx_speech_np.std():.6f}"
    )
    print(f"  Max absolute error: {speech_max_err:.2e}")
    print(f"  Mean absolute error: {speech_mean_err:.2e}")
    print(f"  Relative error: {speech_rel_err:.2e}")

    # Sample output comparison
    print("\n" + "=" * 60)
    print("Sample Output Comparison (first 5 tokens, position 0)")
    print("=" * 60)
    print("\nText logits [0, 0, :5]:")
    print(f"  PyTorch: {pt_text_np[0, 0, :5]}")
    print(f"  MLX:     {mlx_text_np[0, 0, :5]}")

    print("\nSpeech logits [0, 0, :5]:")
    print(f"  PyTorch: {pt_speech_np[0, 0, :5]}")
    print(f"  MLX:     {mlx_speech_np[0, 0, :5]}")

    # Verdict
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    target = 1e-5
    text_pass = text_max_err < target
    speech_pass = speech_max_err < target

    print(f"\nTarget accuracy: < {target}")
    print(
        f"Text logits:   {'PASS' if text_pass else 'FAIL'} (max error: {text_max_err:.2e})"
    )
    print(
        f"Speech logits: {'PASS' if speech_pass else 'FAIL'} (max error: {speech_max_err:.2e})"
    )

    if text_pass and speech_pass:
        print("\n[SUCCESS] MLX LLM matches PyTorch within tolerance")
        return 0
    else:
        print("\n[FAIL] MLX LLM does not match PyTorch within tolerance")

        # Debug: find where differences are largest
        if not text_pass:
            idx = np.unravel_index(np.argmax(text_diff), text_diff.shape)
            print(f"\nLargest text error at index {idx}:")
            print(f"  PyTorch: {pt_text_np[idx]:.6f}")
            print(f"  MLX:     {mlx_text_np[idx]:.6f}")

        return 1


def main():
    return validate_pytorch_vs_mlx()


if __name__ == "__main__":
    sys.exit(main())
