#!/usr/bin/env python3
"""Generate reference output for Zipformer2EncoderLayer validation.

This script loads checkpoint weights and generates reference outputs
for validating the C++ Zipformer2EncoderLayer implementation.

Usage:
    python scripts/generate_zipformer2_layer_reference.py
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open


def swoosh_l(x: mx.array) -> mx.array:
    """SwooshL activation: x * sigmoid(x - 1)"""
    return x * mx.sigmoid(x - 1)


def swoosh_r(x: mx.array) -> mx.array:
    """SwooshR activation: x * sigmoid(x + 1) - 0.08 * x"""
    return x * mx.sigmoid(x + 1) - 0.08 * x


class BiasNorm:
    """BiasNorm layer matching icefall implementation."""

    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        self.bias = mx.zeros((num_channels,))
        self.log_scale = mx.array(0.0)

    def load_weights(self, weights: dict, prefix: str):
        self.bias = weights[f"{prefix}.bias"]
        self.log_scale = weights[f"{prefix}.log_scale"]

    def __call__(self, x: mx.array) -> mx.array:
        # Formula: scales = (mean((x - bias)^2) + eps)^(-0.5) * exp(log_scale)
        centered = x - self.bias
        variance = mx.mean(centered ** 2, axis=-1, keepdims=True)
        scales = (variance + 1e-8) ** -0.5 * mx.exp(self.log_scale)
        return x * scales


class ScaledLinear:
    """Linear layer with scaled init."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.weight = mx.zeros((out_features, in_features))
        self.bias = mx.zeros((out_features,)) if has_bias else None

    def load_weights(self, weights: dict, prefix: str):
        self.weight = weights[f"{prefix}.weight"]
        if self.has_bias:
            self.bias = weights[f"{prefix}.bias"]

    def __call__(self, x: mx.array) -> mx.array:
        # x @ weight.T + bias
        out = x @ self.weight.T
        if self.has_bias and self.bias is not None:
            out = out + self.bias
        return out


class FeedforwardModule:
    """Feedforward module with SwooshL/SwooshR."""

    def __init__(self, d_model: int, d_ff: int):
        self.norm = BiasNorm(d_model)
        self.in_proj = ScaledLinear(d_model, d_ff)
        self.out_proj = ScaledLinear(d_ff, d_model)

    def load_weights(self, weights: dict, prefix: str):
        self.norm.load_weights(weights, f"{prefix}.norm")
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.out_proj.load_weights(weights, f"{prefix}.out_proj")

    def __call__(self, x: mx.array) -> mx.array:
        # Note: FFN uses norm before, but checkpoint doesn't have per-ff norm
        # Actually checking weights - there's no norm in ff modules
        out = self.in_proj(x)
        out = swoosh_l(out)
        out = self.out_proj(out)
        return out


class FeedforwardModuleNoNorm:
    """Feedforward module without norm (matches checkpoint)."""

    def __init__(self, d_model: int, d_ff: int):
        self.in_proj = ScaledLinear(d_model, d_ff)
        self.out_proj = ScaledLinear(d_ff, d_model)

    def load_weights(self, weights: dict, prefix: str):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.out_proj.load_weights(weights, f"{prefix}.out_proj")

    def __call__(self, x: mx.array) -> mx.array:
        out = self.in_proj(x)
        out = swoosh_l(out)
        out = self.out_proj(out)
        return out


class ConvolutionModule:
    """Convolution module with chunk-causal depthwise conv.

    Implements icefall ChunkCausalDepthwiseConv1d: combines causal conv
    (for streaming) with chunkwise conv (for bidirectional context within chunks).
    """

    def __init__(self, d_model: int, kernel_size: int = 31, causal: bool = True):
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.causal = causal

        self.in_proj = ScaledLinear(d_model, d_model * 2)
        self.out_proj = ScaledLinear(d_model, d_model)

        # Depthwise conv weights
        self.causal_conv_weight = mx.zeros((d_model, 1, 16))  # causal part
        self.causal_conv_bias = mx.zeros((d_model,))
        self.chunkwise_conv_weight = mx.zeros((d_model, 1, kernel_size))
        self.chunkwise_conv_bias = mx.zeros((d_model,))
        self.chunkwise_conv_scale = mx.zeros((2, d_model, kernel_size))

    def load_weights(self, weights: dict, prefix: str):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.out_proj.load_weights(weights, f"{prefix}.out_proj")

        self.causal_conv_weight = weights[f"{prefix}.depthwise_conv.causal_conv.weight"]
        self.causal_conv_bias = weights[f"{prefix}.depthwise_conv.causal_conv.bias"]
        self.chunkwise_conv_weight = weights[f"{prefix}.depthwise_conv.chunkwise_conv.weight"]
        self.chunkwise_conv_bias = weights[f"{prefix}.depthwise_conv.chunkwise_conv.bias"]
        if f"{prefix}.depthwise_conv.chunkwise_conv_scale" in weights:
            self.chunkwise_conv_scale = weights[f"{prefix}.depthwise_conv.chunkwise_conv_scale"]

    def _get_chunk_scale(self, chunk_size: int) -> mx.array:
        """Compute chunk edge scaling (matches icefall _get_chunk_scale)."""
        left_edge = self.chunkwise_conv_scale[0]  # (d_model, kernel_size)
        right_edge = self.chunkwise_conv_scale[1]  # (d_model, kernel_size)

        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, self.kernel_size - chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            pad_arr = mx.zeros((self.d_model, t))
            left_edge = mx.concatenate([left_edge, pad_arr], axis=1)
            right_edge = mx.concatenate([pad_arr, right_edge], axis=1)

        # scale = 1 + (left + right), then transpose to (chunk_size, d_model)
        scale = 1 + left_edge + right_edge
        scale = mx.transpose(scale)  # (chunk_size, d_model)
        scale = mx.expand_dims(scale, 0)  # (1, chunk_size, d_model)
        return scale

    def _depthwise_conv1d(self, x: mx.array, weight: mx.array, bias: mx.array,
                         padding_left: int, padding_right: int) -> mx.array:
        """Manual depthwise conv1d implementation.
        x: (batch, seq, d_model)
        weight: (d_model, 1, kernel_size)
        bias: (d_model,)
        """
        batch_size, seq_len, d_model = x.shape
        kernel_size = weight.shape[2]

        # Pad input
        x_padded = mx.pad(x, [(0, 0), (padding_left, padding_right), (0, 0)])
        padded_len = x_padded.shape[1]
        out_len = padded_len - kernel_size + 1

        # Manual depthwise conv - loop over channels
        out_channels = []
        for c in range(d_model):
            channel_in = x_padded[:, :, c]  # (batch, padded_len)
            w = weight[c, 0, :]  # (kernel_size,)
            channel_out = []
            for t in range(out_len):
                window = channel_in[:, t:t+kernel_size]  # (batch, kernel_size)
                val = mx.sum(window * w, axis=-1) + bias[c]  # (batch,)
                channel_out.append(val)
            channel_out = mx.stack(channel_out, axis=1)  # (batch, out_len)
            out_channels.append(channel_out)
        return mx.stack(out_channels, axis=2)  # (batch, out_len, d_model)

    def __call__(self, x: mx.array, padding_mask: mx.array = None) -> mx.array:
        """
        x: (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, d_model = x.shape

        # Input projection and gating
        proj = self.in_proj(x)  # (seq, batch, 2*d_model)
        x1, x2 = mx.split(proj, 2, axis=-1)
        x1 = x1 * mx.sigmoid(x2)  # GLU gating

        # Transpose for conv: (seq, batch, d_model) -> (batch, seq, d_model)
        x1 = mx.transpose(x1, (1, 0, 2))

        # === Causal component ===
        causal_kernel_size = self.causal_conv_weight.shape[2]  # 16
        left_pad = self.kernel_size // 2  # 15 for kernel_size=31
        x_causal = self._depthwise_conv1d(
            x1, self.causal_conv_weight, self.causal_conv_bias,
            padding_left=left_pad, padding_right=0
        )

        # === Chunkwise component ===
        padding = self.kernel_size // 2  # symmetric padding
        x_chunk = self._depthwise_conv1d(
            x1, self.chunkwise_conv_weight, self.chunkwise_conv_bias,
            padding_left=padding, padding_right=padding
        )

        # Apply chunk edge scaling
        chunk_scale = self._get_chunk_scale(seq_len)  # (1, seq_len, d_model)
        x_chunk = x_chunk * chunk_scale

        # Combine causal + chunkwise
        out = x_causal + x_chunk

        # SwooshR activation
        out = swoosh_r(out)

        # Transpose back: (batch, seq, d_model) -> (seq, batch, d_model)
        out = mx.transpose(out, (1, 0, 2))

        # Output projection
        out = self.out_proj(out)

        return out


class RelPositionMultiheadAttentionWeights:
    """Computes attention weights with relative positional encoding."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int = 4,
        pos_emb_dim: int = 48
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.pos_emb_dim = pos_emb_dim

        # Q, K, P projection: d_model -> 2*num_heads*query_head_dim + num_heads*pos_head_dim
        in_proj_dim = 2 * num_heads * query_head_dim + num_heads * pos_head_dim
        self.in_proj = ScaledLinear(d_model, in_proj_dim)

        # Position projection: pos_emb_dim -> num_heads * pos_head_dim
        self.linear_pos_weight = mx.zeros((num_heads * pos_head_dim, pos_emb_dim))

    def load_weights(self, weights: dict, prefix: str):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.linear_pos_weight = weights[f"{prefix}.linear_pos.weight"]

    def rel_shift(self, x: mx.array, seq_len: int) -> mx.array:
        """Convert relative position scores to absolute positions."""
        # x: (batch_heads, seq_len, 2*seq_len-1)
        # Output: (batch_heads, seq_len, seq_len)

        # Create index matrix: index[q, k] = (seq_len - 1 - q) + k
        rows = mx.arange(seq_len - 1, -1, -1)  # [seq_len-1, ..., 0]
        cols = mx.arange(0, seq_len)  # [0, 1, ..., seq_len-1]

        indexes = mx.expand_dims(rows, 1) + mx.expand_dims(cols, 0)  # (seq_len, seq_len)
        indexes = mx.expand_dims(indexes, 0)  # (1, seq_len, seq_len)

        return mx.take_along_axis(x, indexes.astype(mx.int32), axis=2)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        attn_mask: mx.array = None
    ) -> mx.array:
        """
        x: (seq_len, batch_size, d_model)
        pos_emb: (batch, 2*seq-1, pos_emb_dim)
        """
        seq_len, batch_size, _ = x.shape

        # Project input to Q, K, P
        proj = self.in_proj(x)  # (seq, batch, in_proj_dim)

        query_dim = self.num_heads * self.query_head_dim
        pos_dim = self.num_heads * self.pos_head_dim

        q = proj[:, :, :query_dim]
        k = proj[:, :, query_dim:2*query_dim]
        p = proj[:, :, 2*query_dim:]

        # Reshape for multi-head: (seq, batch, heads*dim) -> (seq, batch, heads, dim)
        q = mx.reshape(q, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        k = mx.reshape(k, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        p = mx.reshape(p, (seq_len, batch_size, self.num_heads, self.pos_head_dim))

        # Transpose to (batch, heads, seq, dim)
        q = mx.transpose(q, (1, 2, 0, 3))
        k = mx.transpose(k, (1, 2, 0, 3))
        p = mx.transpose(p, (1, 2, 0, 3))

        # Project position embeddings: (batch, 2*seq-1, pos_emb_dim) @ (pos_emb_dim, heads*pos_head_dim)
        pos_proj = pos_emb @ self.linear_pos_weight.T
        pos_proj = mx.reshape(pos_proj, (batch_size, -1, self.num_heads, self.pos_head_dim))
        pos_proj = mx.transpose(pos_proj, (0, 2, 1, 3))  # (batch, heads, 2*seq-1, pos_head_dim)

        # Content attention: Q @ K^T
        content_score = q @ mx.transpose(k, (0, 1, 3, 2))  # (batch, heads, seq, seq)

        # Position attention: P @ pos_proj^T
        pos_score = p @ mx.transpose(pos_proj, (0, 1, 3, 2))  # (batch, heads, seq, 2*seq-1)
        pos_score = mx.reshape(pos_score, (batch_size * self.num_heads, seq_len, -1))
        pos_score = self.rel_shift(pos_score, seq_len)
        pos_score = mx.reshape(pos_score, (batch_size, self.num_heads, seq_len, seq_len))

        attn_score = content_score + pos_score

        # Apply mask
        if attn_mask is not None:
            attn_score = attn_score + attn_mask

        # Softmax
        attn_weights = mx.softmax(attn_score, axis=-1)

        # Reshape to (batch * heads, seq, seq)
        attn_weights = mx.reshape(attn_weights, (batch_size * self.num_heads, seq_len, seq_len))

        return attn_weights


class SelfAttention2:
    """Self-attention using pre-computed attention weights."""

    def __init__(self, d_model: int, num_heads: int, value_head_dim: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.value_head_dim = value_head_dim
        self.attention_dim = num_heads * value_head_dim

        self.in_proj = ScaledLinear(d_model, self.attention_dim)
        self.out_proj = ScaledLinear(self.attention_dim, d_model)

    def load_weights(self, weights: dict, prefix: str):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.out_proj.load_weights(weights, f"{prefix}.out_proj")

    def __call__(self, x: mx.array, attn_weights: mx.array) -> mx.array:
        """
        x: (seq_len, batch_size, d_model)
        attn_weights: (batch * heads, seq, seq)
        """
        seq_len, batch_size, _ = x.shape

        # Project values
        v = self.in_proj(x)  # (seq, batch, attention_dim)
        v = mx.reshape(v, (seq_len, batch_size, self.num_heads, self.value_head_dim))
        v = mx.transpose(v, (1, 2, 0, 3))  # (batch, heads, seq, head_dim)
        v = mx.reshape(v, (batch_size * self.num_heads, seq_len, self.value_head_dim))

        # Apply attention
        out = attn_weights @ v  # (batch*heads, seq, head_dim)
        out = mx.reshape(out, (batch_size, self.num_heads, seq_len, self.value_head_dim))
        out = mx.transpose(out, (2, 0, 1, 3))  # (seq, batch, heads, head_dim)
        out = mx.reshape(out, (seq_len, batch_size, self.attention_dim))

        return self.out_proj(out)


class NonlinAttention:
    """Non-linear attention with tanh gating."""

    def __init__(self, d_model: int, hidden_channels: int):
        self.d_model = d_model
        self.hidden_channels = hidden_channels

        self.in_proj = ScaledLinear(d_model, 3 * hidden_channels)
        self.out_proj = ScaledLinear(hidden_channels, d_model)

    def load_weights(self, weights: dict, prefix: str):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.out_proj.load_weights(weights, f"{prefix}.out_proj")

    def __call__(self, x: mx.array, attn_weights: mx.array) -> mx.array:
        """
        x: (seq_len, batch_size, d_model)
        attn_weights: (batch * heads, seq, seq)
        """
        seq_len, batch_size, _ = x.shape

        # Project and split
        proj = self.in_proj(x)  # (seq, batch, 3*hidden)
        s, v, y = mx.split(proj, 3, axis=-1)

        # Tanh gate
        s = mx.tanh(s)
        v = v * s

        # Apply attention using all heads
        total_heads = attn_weights.shape[0]
        num_heads = total_heads // batch_size
        head_dim = self.hidden_channels // num_heads

        # Reshape v for multi-head
        v = mx.reshape(v, (seq_len, batch_size, num_heads, head_dim))
        v = mx.transpose(v, (1, 2, 0, 3))  # (batch, heads, seq, head_dim)
        v = mx.reshape(v, (batch_size * num_heads, seq_len, head_dim))

        # Apply attention
        out = attn_weights @ v  # (batch*heads, seq, head_dim)

        # Reshape back
        out = mx.reshape(out, (batch_size, num_heads, seq_len, head_dim))
        out = mx.transpose(out, (2, 0, 1, 3))  # (seq, batch, heads, head_dim)
        out = mx.reshape(out, (seq_len, batch_size, self.hidden_channels))

        # Output gate and projection
        out = out * y
        return self.out_proj(out)


class BypassModule:
    """Learnable skip connection."""

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.bypass_scale = mx.full((d_model,), 0.5)

    def load_weights(self, weights: dict, prefix: str):
        self.bypass_scale = weights[f"{prefix}.bypass_scale"]

    def __call__(self, src_orig: mx.array, src: mx.array) -> mx.array:
        scale = mx.clip(self.bypass_scale, 0.0, 1.0)
        return src_orig + scale * (src - src_orig)


class Zipformer2EncoderLayer:
    """Full Zipformer2 encoder layer."""

    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        num_heads: int,
        ff1_dim: int,
        ff2_dim: int,
        ff3_dim: int,
        kernel_size: int = 31,
        pos_head_dim: int = 4,
        pos_emb_dim: int = 48,
        value_head_dim: int = 12,
        causal: bool = True
    ):
        self.d_model = d_model
        query_head_dim = attention_dim // num_heads
        hidden_channels = 3 * d_model // 4

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            d_model, num_heads, query_head_dim, pos_head_dim, pos_emb_dim
        )
        self.self_attn1 = SelfAttention2(d_model, num_heads, value_head_dim)
        self.self_attn2 = SelfAttention2(d_model, num_heads, value_head_dim)
        self.nonlin_attention = NonlinAttention(d_model, hidden_channels)

        self.feed_forward1 = FeedforwardModuleNoNorm(d_model, ff1_dim)
        self.feed_forward2 = FeedforwardModuleNoNorm(d_model, ff2_dim)
        self.feed_forward3 = FeedforwardModuleNoNorm(d_model, ff3_dim)

        self.conv_module1 = ConvolutionModule(d_model, kernel_size, causal)
        self.conv_module2 = ConvolutionModule(d_model, kernel_size, causal)

        self.norm = BiasNorm(d_model)
        self.bypass = BypassModule(d_model)
        self.bypass_mid = BypassModule(d_model)
        self.bypass_scale = mx.full((d_model,), 0.5)

    def load_weights(self, weights: dict, prefix: str):
        self.self_attn_weights.load_weights(weights, f"{prefix}.self_attn_weights")
        self.self_attn1.load_weights(weights, f"{prefix}.self_attn1")
        self.self_attn2.load_weights(weights, f"{prefix}.self_attn2")
        self.nonlin_attention.load_weights(weights, f"{prefix}.nonlin_attention")
        self.feed_forward1.load_weights(weights, f"{prefix}.feed_forward1")
        self.feed_forward2.load_weights(weights, f"{prefix}.feed_forward2")
        self.feed_forward3.load_weights(weights, f"{prefix}.feed_forward3")
        self.conv_module1.load_weights(weights, f"{prefix}.conv_module1")
        self.conv_module2.load_weights(weights, f"{prefix}.conv_module2")
        self.norm.load_weights(weights, f"{prefix}.norm")
        self.bypass.load_weights(weights, f"{prefix}.bypass")
        self.bypass_mid.load_weights(weights, f"{prefix}.bypass_mid")
        self.bypass_scale = weights[f"{prefix}.bypass_scale"]

    def __call__(
        self,
        src: mx.array,
        pos_emb: mx.array,
        attn_mask: mx.array = None
    ) -> mx.array:
        """
        src: (seq_len, batch_size, d_model)
        pos_emb: (batch, 2*seq-1, pos_emb_dim)
        """
        src_orig = src
        out = src

        # Compute attention weights (shared)
        attn_weights = self.self_attn_weights(out, pos_emb, attn_mask)

        # First feedforward
        out = out + self.feed_forward1(out)

        # Non-linear attention
        out = out + self.nonlin_attention(out, attn_weights)

        # First self-attention
        out = out + self.self_attn1(out, attn_weights)

        # First convolution
        out = out + self.conv_module1(out)

        # Second feedforward
        out = out + self.feed_forward2(out)

        # Mid-layer bypass
        out = self.bypass_mid(src_orig, out)

        # Second self-attention
        out = out + self.self_attn2(out, attn_weights)

        # Second convolution
        out = out + self.conv_module2(out)

        # Third feedforward
        out = out + self.feed_forward3(out)

        # Normalize
        out = self.norm(out)

        # Final bypass
        out = self.bypass(src_orig, out)

        return out


def main():
    parser = argparse.ArgumentParser(description="Generate Zipformer2 layer reference")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("checkpoints/zipformer/en-streaming/model.safetensors"),
        help="Path to weights file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/zipformer/en-streaming/layer0_reference.npz"),
        help="Path to save reference outputs"
    )
    parser.add_argument(
        "--stage", type=int, default=0,
        help="Encoder stage index"
    )
    parser.add_argument(
        "--layer", type=int, default=0,
        help="Layer index within stage"
    )
    parser.add_argument(
        "--seq-len", type=int, default=32,
        help="Sequence length for test input"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for test input"
    )
    args = parser.parse_args()

    # Load weights
    print(f"Loading weights from {args.weights}")
    weights = {}
    with safe_open(str(args.weights), framework="mlx") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    # Auto-detect config from checkpoint weights
    prefix = f"encoders.{args.stage}.layers.{args.layer}"
    if f"{prefix}.feed_forward1.in_proj.weight" not in weights:
        # Try nested structure for stages > 0
        prefix = f"encoders.{args.stage}.encoder.layers.{args.layer}"
        if f"{prefix}.feed_forward1.in_proj.weight" not in weights:
            raise ValueError(f"Could not find layer weights for stage {args.stage}, layer {args.layer}")

    # Derive dimensions from weight shapes
    ff1_weight = weights[f"{prefix}.feed_forward1.in_proj.weight"]
    ff1_dim = int(ff1_weight.shape[0])
    d_model = int(ff1_weight.shape[1])

    ff2_weight = weights[f"{prefix}.feed_forward2.in_proj.weight"]
    ff2_dim = int(ff2_weight.shape[0])

    ff3_weight = weights[f"{prefix}.feed_forward3.in_proj.weight"]
    ff3_dim = int(ff3_weight.shape[0])

    # Attention config from self_attn
    # in_proj shape is (num_heads * (2 * query_head_dim + pos_head_dim), d_model)
    # But we use self_attn.in_proj which is (num_heads * 2 * query_head_dim, d_model)
    if f"{prefix}.self_attn.in_proj.weight" in weights:
        in_proj_weight = weights[f"{prefix}.self_attn.in_proj.weight"]
        in_proj_dim = int(in_proj_weight.shape[0])
        # in_proj_dim = num_heads * 2 * query_head_dim
        # We need to determine num_heads - check out_proj2
        out_proj2_weight = weights[f"{prefix}.self_attn.out_proj2.weight"]
        # out_proj2 shape is (d_model, num_heads * value_head_dim)
        value_out_dim = int(out_proj2_weight.shape[1])
        # Heuristic: value_head_dim is typically 12, so num_heads = value_out_dim / 12
        value_head_dim = 12
        num_heads = value_out_dim // value_head_dim
        query_head_dim = in_proj_dim // (2 * num_heads)
        attention_dim = query_head_dim * num_heads
    else:
        # Fallback
        num_heads = 4
        attention_dim = 128
        value_head_dim = 12

    # Get kernel size from conv weight (handle both simple and split depthwise conv)
    simple_conv_key = f"{prefix}.conv_module1.depthwise_conv.weight"
    causal_conv_key = f"{prefix}.conv_module1.depthwise_conv.causal_conv.weight"
    chunkwise_conv_key = f"{prefix}.conv_module1.depthwise_conv.chunkwise_conv.weight"

    if simple_conv_key in weights:
        conv_weight = weights[simple_conv_key]
        kernel_size = int(conv_weight.shape[2])
    elif chunkwise_conv_key in weights:
        conv_weight = weights[chunkwise_conv_key]
        kernel_size = int(conv_weight.shape[2])  # Use chunkwise kernel size
    else:
        kernel_size = 31  # Default

    # Position encoding config (typically fixed)
    pos_head_dim = 4
    pos_emb_dim = 48

    print("Creating Zipformer2EncoderLayer with:")
    print(f"  d_model={d_model}, attention_dim={attention_dim}, num_heads={num_heads}")
    print(f"  ff1_dim={ff1_dim}, ff2_dim={ff2_dim}, ff3_dim={ff3_dim}")
    print(f"  kernel_size={kernel_size}")

    # Create layer
    layer = Zipformer2EncoderLayer(
        d_model=d_model,
        attention_dim=attention_dim,
        num_heads=num_heads,
        ff1_dim=ff1_dim,
        ff2_dim=ff2_dim,
        ff3_dim=ff3_dim,
        kernel_size=kernel_size,
        pos_head_dim=pos_head_dim,
        pos_emb_dim=pos_emb_dim,
        value_head_dim=value_head_dim,
        causal=True
    )

    # Load weights using the auto-detected prefix
    print(f"Loading weights with prefix: {prefix}")
    layer.load_weights(weights, prefix)

    # Create deterministic test input
    mx.random.seed(42)
    seq_len = args.seq_len
    batch_size = args.batch_size

    # Input: (seq_len, batch_size, d_model)
    src = mx.random.normal((seq_len, batch_size, d_model)) * 0.1

    # Position embeddings: (batch, 2*seq-1, pos_emb_dim)
    pos_emb = mx.random.normal((batch_size, 2 * seq_len - 1, pos_emb_dim)) * 0.1

    # Run forward pass
    print(f"Running forward pass with input shape: {src.shape}")
    out = layer(src, pos_emb)
    mx.eval(out)

    print(f"Output shape: {out.shape}")
    print(f"Output stats: min={float(mx.min(out)):.6f}, max={float(mx.max(out)):.6f}, mean={float(mx.mean(out)):.6f}")

    # Save reference outputs
    outputs = {
        "input_src": np.array(src),
        "input_pos_emb": np.array(pos_emb),
        "output": np.array(out),
        "config": np.array([
            d_model, attention_dim, num_heads,
            ff1_dim, ff2_dim, ff3_dim,
            kernel_size, pos_head_dim, pos_emb_dim, value_head_dim
        ]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save in both npz and safetensors formats
    np.savez(str(args.output), **outputs)
    print(f"Saved npz reference to: {args.output}")

    # Save as safetensors for C++ loading
    from safetensors.numpy import save_file
    safetensors_path = str(args.output).replace(".npz", ".safetensors")
    save_file(outputs, safetensors_path)
    print(f"Saved safetensors reference to: {safetensors_path}")


if __name__ == "__main__":
    main()
