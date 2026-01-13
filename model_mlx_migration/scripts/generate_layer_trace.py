#!/usr/bin/env python3
"""Generate full layer trace with all intermediate values for C++ comparison."""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def swoosh_l(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x - 1)


def swoosh_r(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x + 1) - 0.08 * x


class BiasNorm:
    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        self.bias = mx.zeros((num_channels,))
        self.log_scale = mx.array(0.0)

    def load_weights(self, weights: dict, prefix: str):
        self.bias = weights[f"{prefix}.bias"]
        self.log_scale = weights[f"{prefix}.log_scale"]

    def __call__(self, x: mx.array) -> mx.array:
        centered = x - self.bias
        variance = mx.mean(centered ** 2, axis=-1, keepdims=True)
        scales = (variance + 1e-8) ** -0.5 * mx.exp(self.log_scale)
        return x * scales


class ScaledLinear:
    def __init__(self, in_features: int, out_features: int, has_bias: bool = True):
        self.weight = mx.zeros((out_features, in_features))
        self.bias = mx.zeros((out_features,)) if has_bias else None
        self.has_bias = has_bias

    def load_weights(self, weights: dict, prefix: str):
        self.weight = weights[f"{prefix}.weight"]
        if self.has_bias:
            self.bias = weights[f"{prefix}.bias"]

    def __call__(self, x: mx.array) -> mx.array:
        out = x @ self.weight.T
        if self.has_bias and self.bias is not None:
            out = out + self.bias
        return out


class FeedforwardModule:
    def __init__(self, d_model: int, d_ff: int):
        self.in_proj = ScaledLinear(d_model, d_ff)
        self.out_proj = ScaledLinear(d_ff, d_model)

    def load_weights(self, weights: dict, prefix: str):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.out_proj.load_weights(weights, f"{prefix}.out_proj")

    def __call__(self, x: mx.array) -> mx.array:
        out = self.in_proj(x)
        out = swoosh_l(out)
        return self.out_proj(out)


class ConvolutionModule:
    def __init__(self, d_model: int, kernel_size: int = 31):
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.in_proj = ScaledLinear(d_model, d_model * 2)
        self.out_proj = ScaledLinear(d_model, d_model)
        self.causal_conv_weight = mx.zeros((d_model, 1, 16))
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

    def _depthwise_conv(self, x, weight, bias, pad_left, pad_right):
        """Manual depthwise conv1d."""
        batch_size, seq_len, d_model = x.shape
        k_size = weight.shape[2]
        x_pad = mx.pad(x, [(0, 0), (pad_left, pad_right), (0, 0)])
        out_len = x_pad.shape[1] - k_size + 1

        out_channels = []
        for c in range(d_model):
            channel_in = x_pad[:, :, c]
            w = weight[c, 0, :]
            channel_out = []
            for t in range(out_len):
                window = channel_in[:, t:t+k_size]
                val = mx.sum(window * w, axis=-1) + bias[c]
                channel_out.append(val)
            out_channels.append(mx.stack(channel_out, axis=1))
        return mx.stack(out_channels, axis=2)

    def __call__(self, x: mx.array) -> mx.array:
        seq_len, batch_size, d_model = x.shape

        proj = self.in_proj(x)
        x1, x2 = mx.split(proj, 2, axis=-1)
        x1 = x1 * mx.sigmoid(x2)
        x1 = mx.transpose(x1, (1, 0, 2))  # (batch, seq, d)

        left_pad = self.kernel_size // 2
        x_causal = self._depthwise_conv(x1, self.causal_conv_weight, self.causal_conv_bias, left_pad, 0)

        padding = self.kernel_size // 2
        x_chunk = self._depthwise_conv(x1, self.chunkwise_conv_weight, self.chunkwise_conv_bias, padding, padding)

        # Chunk scale
        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        if seq_len < self.kernel_size:
            left_edge = left_edge[:, :seq_len]
            right_edge = right_edge[:, self.kernel_size - seq_len:]
        else:
            t = seq_len - self.kernel_size
            pad_arr = mx.zeros((d_model, t))
            left_edge = mx.concatenate([left_edge, pad_arr], axis=1)
            right_edge = mx.concatenate([pad_arr, right_edge], axis=1)
        scale = 1 + left_edge + right_edge
        scale = mx.transpose(scale)
        scale = mx.expand_dims(scale, 0)

        x_chunk = x_chunk * scale
        out = x_causal + x_chunk
        out = swoosh_r(out)
        out = mx.transpose(out, (1, 0, 2))
        return self.out_proj(out)


class RelPositionMultiheadAttentionWeights:
    def __init__(self, d_model, num_heads, query_head_dim, pos_head_dim=4, pos_emb_dim=48):
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.pos_emb_dim = pos_emb_dim
        in_proj_dim = 2 * num_heads * query_head_dim + num_heads * pos_head_dim
        self.in_proj = ScaledLinear(d_model, in_proj_dim)
        self.linear_pos_weight = mx.zeros((num_heads * pos_head_dim, pos_emb_dim))

    def load_weights(self, weights, prefix):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.linear_pos_weight = weights[f"{prefix}.linear_pos.weight"]

    def rel_shift(self, x, seq_len):
        rows = mx.arange(seq_len - 1, -1, -1)
        cols = mx.arange(0, seq_len)
        indexes = mx.expand_dims(rows, 1) + mx.expand_dims(cols, 0)
        indexes = mx.expand_dims(indexes, 0).astype(mx.int32)
        return mx.take_along_axis(x, indexes, axis=2)

    def __call__(self, x, pos_emb):
        seq_len, batch_size, _ = x.shape
        proj = self.in_proj(x)

        query_dim = self.num_heads * self.query_head_dim
        pos_dim = self.num_heads * self.pos_head_dim

        q = proj[:, :, :query_dim]
        k = proj[:, :, query_dim:2*query_dim]
        p = proj[:, :, 2*query_dim:]

        q = mx.reshape(q, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        k = mx.reshape(k, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        p = mx.reshape(p, (seq_len, batch_size, self.num_heads, self.pos_head_dim))

        q = mx.transpose(q, (1, 2, 0, 3))
        k = mx.transpose(k, (1, 2, 0, 3))
        p = mx.transpose(p, (1, 2, 0, 3))

        pos_proj = pos_emb @ self.linear_pos_weight.T
        pos_proj = mx.reshape(pos_proj, (batch_size, -1, self.num_heads, self.pos_head_dim))
        pos_proj = mx.transpose(pos_proj, (0, 2, 1, 3))

        content_score = q @ mx.transpose(k, (0, 1, 3, 2))
        pos_score = p @ mx.transpose(pos_proj, (0, 1, 3, 2))
        pos_score = mx.reshape(pos_score, (batch_size * self.num_heads, seq_len, -1))
        pos_score = self.rel_shift(pos_score, seq_len)
        pos_score = mx.reshape(pos_score, (batch_size, self.num_heads, seq_len, seq_len))

        attn_score = content_score + pos_score
        attn_weights = mx.softmax(attn_score, axis=-1)
        return mx.reshape(attn_weights, (batch_size * self.num_heads, seq_len, seq_len))


class SelfAttention2:
    def __init__(self, d_model, num_heads, value_head_dim):
        self.d_model = d_model
        self.num_heads = num_heads
        self.value_head_dim = value_head_dim
        self.attention_dim = num_heads * value_head_dim
        self.in_proj = ScaledLinear(d_model, self.attention_dim)
        self.out_proj = ScaledLinear(self.attention_dim, d_model)

    def load_weights(self, weights, prefix):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.out_proj.load_weights(weights, f"{prefix}.out_proj")

    def __call__(self, x, attn_weights):
        seq_len, batch_size, _ = x.shape
        v = self.in_proj(x)
        v = mx.reshape(v, (seq_len, batch_size, self.num_heads, self.value_head_dim))
        v = mx.transpose(v, (1, 2, 0, 3))
        v = mx.reshape(v, (batch_size * self.num_heads, seq_len, self.value_head_dim))

        out = attn_weights @ v
        out = mx.reshape(out, (batch_size, self.num_heads, seq_len, self.value_head_dim))
        out = mx.transpose(out, (2, 0, 1, 3))
        out = mx.reshape(out, (seq_len, batch_size, self.attention_dim))
        return self.out_proj(out)


class NonlinAttention:
    def __init__(self, d_model, hidden_channels):
        self.d_model = d_model
        self.hidden_channels = hidden_channels
        self.in_proj = ScaledLinear(d_model, 3 * hidden_channels)
        self.out_proj = ScaledLinear(hidden_channels, d_model)

    def load_weights(self, weights, prefix):
        self.in_proj.load_weights(weights, f"{prefix}.in_proj")
        self.out_proj.load_weights(weights, f"{prefix}.out_proj")

    def __call__(self, x, attn_weights):
        seq_len, batch_size, _ = x.shape
        proj = self.in_proj(x)
        s, v, y = mx.split(proj, 3, axis=-1)
        s = mx.tanh(s)
        v = v * s

        total_heads = attn_weights.shape[0]
        num_heads = total_heads // batch_size
        head_dim = self.hidden_channels // num_heads

        v = mx.reshape(v, (seq_len, batch_size, num_heads, head_dim))
        v = mx.transpose(v, (1, 2, 0, 3))
        v = mx.reshape(v, (batch_size * num_heads, seq_len, head_dim))

        out = attn_weights @ v
        out = mx.reshape(out, (batch_size, num_heads, seq_len, head_dim))
        out = mx.transpose(out, (2, 0, 1, 3))
        out = mx.reshape(out, (seq_len, batch_size, self.hidden_channels))

        out = out * y
        return self.out_proj(out)


class BypassModule:
    def __init__(self, d_model):
        self.d_model = d_model
        self.bypass_scale = mx.full((d_model,), 0.5)

    def load_weights(self, weights, prefix):
        self.bypass_scale = weights[f"{prefix}.bypass_scale"]

    def __call__(self, src_orig, src):
        scale = mx.clip(self.bypass_scale, 0.0, 1.0)
        return src_orig + scale * (src - src_orig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path,
                       default=Path("checkpoints/zipformer/en-streaming/model.safetensors"))
    parser.add_argument("--output", type=Path,
                       default=Path("checkpoints/zipformer/en-streaming/layer0_trace_seq32.safetensors"))
    args = parser.parse_args()

    print("Loading weights...")
    weights = {}
    with safe_open(str(args.weights), framework="mlx") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    prefix = "encoders.0.layers.0"
    d_model = 192
    num_heads = 4
    attention_dim = 128
    query_head_dim = attention_dim // num_heads
    value_head_dim = 12
    hidden_channels = 3 * d_model // 4

    # Create modules
    self_attn_weights = RelPositionMultiheadAttentionWeights(d_model, num_heads, query_head_dim, 4, 48)
    self_attn1 = SelfAttention2(d_model, num_heads, value_head_dim)
    self_attn2 = SelfAttention2(d_model, num_heads, value_head_dim)
    nonlin_attention = NonlinAttention(d_model, hidden_channels)
    ff1 = FeedforwardModule(d_model, 384)
    ff2 = FeedforwardModule(d_model, 512)
    ff3 = FeedforwardModule(d_model, 640)
    conv1 = ConvolutionModule(d_model, 31)
    conv2 = ConvolutionModule(d_model, 31)
    norm = BiasNorm(d_model)
    bypass = BypassModule(d_model)
    bypass_mid = BypassModule(d_model)

    # Load weights
    self_attn_weights.load_weights(weights, f"{prefix}.self_attn_weights")
    self_attn1.load_weights(weights, f"{prefix}.self_attn1")
    self_attn2.load_weights(weights, f"{prefix}.self_attn2")
    nonlin_attention.load_weights(weights, f"{prefix}.nonlin_attention")
    ff1.load_weights(weights, f"{prefix}.feed_forward1")
    ff2.load_weights(weights, f"{prefix}.feed_forward2")
    ff3.load_weights(weights, f"{prefix}.feed_forward3")
    conv1.load_weights(weights, f"{prefix}.conv_module1")
    conv2.load_weights(weights, f"{prefix}.conv_module2")
    norm.load_weights(weights, f"{prefix}.norm")
    bypass.load_weights(weights, f"{prefix}.bypass")
    bypass_mid.load_weights(weights, f"{prefix}.bypass_mid")

    # Create test input
    mx.random.seed(42)
    seq_len = 32
    batch_size = 1
    src = mx.random.normal((seq_len, batch_size, d_model)) * 0.1
    pos_emb = mx.random.normal((batch_size, 2 * seq_len - 1, 48)) * 0.1
    mx.eval(src, pos_emb)

    trace = {"input_src": np.array(src), "input_pos_emb": np.array(pos_emb)}

    def save_step(name, val):
        mx.eval(val)
        trace[name] = np.array(val)
        print(f"{name}: min={float(mx.min(val)):.6f}, max={float(mx.max(val)):.6f}, mean={float(mx.mean(val)):.6f}")

    print(f"\n=== Layer trace (seq_len={seq_len}) ===")
    save_step("input_src", src)

    src_orig = src
    out = src

    # Step 1: Attention weights
    attn_w = self_attn_weights(out, pos_emb)
    save_step("attn_weights", attn_w)

    # Step 2: FF1
    ff1_out = ff1(out)
    save_step("ff1_out", ff1_out)
    out = out + ff1_out
    save_step("after_ff1", out)

    # Step 3: NonlinAttention
    na_out = nonlin_attention(out, attn_w)
    save_step("nonlin_attn_out", na_out)
    out = out + na_out
    save_step("after_nonlin_attn", out)

    # Step 4: SelfAttn1
    sa1_out = self_attn1(out, attn_w)
    save_step("self_attn1_out", sa1_out)
    out = out + sa1_out
    save_step("after_self_attn1", out)

    # Step 5: Conv1
    conv1_out = conv1(out)
    save_step("conv1_out", conv1_out)
    out = out + conv1_out
    save_step("after_conv1", out)

    # Step 6: FF2
    ff2_out = ff2(out)
    save_step("ff2_out", ff2_out)
    out = out + ff2_out
    save_step("after_ff2", out)

    # Step 7: Bypass mid
    out = bypass_mid(src_orig, out)
    save_step("after_bypass_mid", out)

    # Step 8: SelfAttn2
    sa2_out = self_attn2(out, attn_w)
    save_step("self_attn2_out", sa2_out)
    out = out + sa2_out
    save_step("after_self_attn2", out)

    # Step 9: Conv2
    conv2_out = conv2(out)
    save_step("conv2_out", conv2_out)
    out = out + conv2_out
    save_step("after_conv2", out)

    # Step 10: FF3
    ff3_out = ff3(out)
    save_step("ff3_out", ff3_out)
    out = out + ff3_out
    save_step("after_ff3", out)

    # Step 11: Norm
    out = norm(out)
    save_step("after_norm", out)

    # Step 12: Final bypass
    out = bypass(src_orig, out)
    save_step("output", out)

    # Save trace
    save_file(trace, str(args.output))
    print(f"\nSaved trace to {args.output}")


if __name__ == "__main__":
    main()
