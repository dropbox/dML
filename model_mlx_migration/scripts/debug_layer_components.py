#!/usr/bin/env python3
"""Debug script to trace layer components and find divergence point."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path,
                       default=Path("checkpoints/zipformer/en-streaming/model.safetensors"))
    args = parser.parse_args()

    print("Loading weights...")
    weights = {}
    with safe_open(str(args.weights), framework="mlx") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    prefix = "encoders.0.layers.0"
    d_model = 192
    kernel_size = 31

    # Create test input (same seed as reference generation)
    mx.random.seed(42)
    seq_len = 32
    batch_size = 1
    src = mx.random.normal((seq_len, batch_size, d_model)) * 0.1
    mx.eval(src)

    print(f"Input shape: {src.shape}")
    print(f"Input stats: min={float(mx.min(src)):.6f}, max={float(mx.max(src)):.6f}")

    # Test 1: Feedforward module
    print("\n=== Testing FeedforwardModule ===")
    ff1_in_proj = ScaledLinear(d_model, 384)
    ff1_out_proj = ScaledLinear(384, d_model)
    ff1_in_proj.load_weights(weights, f"{prefix}.feed_forward1.in_proj")
    ff1_out_proj.load_weights(weights, f"{prefix}.feed_forward1.out_proj")

    ff1_hidden = ff1_in_proj(src)
    ff1_hidden = swoosh_l(ff1_hidden)
    ff1_out = ff1_out_proj(ff1_hidden)
    mx.eval(ff1_out)
    print(f"FF1 output: min={float(mx.min(ff1_out)):.6f}, max={float(mx.max(ff1_out)):.6f}")

    # Test 2: ConvolutionModule input projection
    print("\n=== Testing ConvolutionModule ===")
    conv_in_proj = ScaledLinear(d_model, d_model * 2)
    conv_in_proj.load_weights(weights, f"{prefix}.conv_module1.in_proj")

    proj = conv_in_proj(src)  # (seq, batch, 2*d_model)
    x1, x2 = mx.split(proj, 2, axis=-1)
    x1_gated = x1 * mx.sigmoid(x2)
    mx.eval(x1_gated)
    print(f"Conv GLU output: min={float(mx.min(x1_gated)):.6f}, max={float(mx.max(x1_gated)):.6f}")

    # Test 3: Causal depthwise conv
    print("\n=== Testing Causal Depthwise Conv ===")
    causal_conv_weight = weights[f"{prefix}.conv_module1.depthwise_conv.causal_conv.weight"]
    causal_conv_bias = weights[f"{prefix}.conv_module1.depthwise_conv.causal_conv.bias"]
    print(f"Causal conv weight shape: {causal_conv_weight.shape}")

    # Transpose for conv: (seq, batch, d_model) -> (batch, seq, d_model)
    x1_t = mx.transpose(x1_gated, (1, 0, 2))

    # Manual depthwise conv
    causal_kernel_size = causal_conv_weight.shape[2]  # 16
    left_pad = kernel_size // 2  # 15
    print(f"Causal kernel size: {causal_kernel_size}, left pad: {left_pad}")

    x1_padded = mx.pad(x1_t, [(0, 0), (left_pad, 0), (0, 0)])
    print(f"Padded input shape: {x1_padded.shape}")

    # Do depthwise conv manually (slow but clear)
    out_channels = []
    for c in range(d_model):
        channel_in = x1_padded[:, :, c]  # (batch, padded_len)
        w = causal_conv_weight[c, 0, :]  # (kernel_size,)
        channel_out = []
        for t in range(seq_len):
            window = channel_in[:, t:t+causal_kernel_size]
            val = mx.sum(window * w, axis=-1) + causal_conv_bias[c]
            channel_out.append(val)
        channel_out = mx.stack(channel_out, axis=1)
        out_channels.append(channel_out)
    x_causal = mx.stack(out_channels, axis=2)
    mx.eval(x_causal)
    print(f"Causal conv output: min={float(mx.min(x_causal)):.6f}, max={float(mx.max(x_causal)):.6f}")
    print(f"Causal conv output shape: {x_causal.shape}")

    # Test 4: Chunkwise depthwise conv
    print("\n=== Testing Chunkwise Depthwise Conv ===")
    chunkwise_conv_weight = weights[f"{prefix}.conv_module1.depthwise_conv.chunkwise_conv.weight"]
    chunkwise_conv_bias = weights[f"{prefix}.conv_module1.depthwise_conv.chunkwise_conv.bias"]
    chunkwise_scale_key = f"{prefix}.conv_module1.depthwise_conv.chunkwise_conv_scale"

    print(f"Chunkwise conv weight shape: {chunkwise_conv_weight.shape}")

    if chunkwise_scale_key in weights:
        chunkwise_conv_scale = weights[chunkwise_scale_key]
        print(f"Chunkwise scale shape: {chunkwise_conv_scale.shape}")
        print(f"Chunkwise scale stats: min={float(mx.min(chunkwise_conv_scale)):.6f}, max={float(mx.max(chunkwise_conv_scale)):.6f}")
    else:
        print("WARNING: chunkwise_conv_scale not found in weights!")
        chunkwise_conv_scale = mx.zeros((2, d_model, kernel_size))

    # Symmetric padding for chunkwise
    padding = kernel_size // 2  # 15
    x1_chunk_padded = mx.pad(x1_t, [(0, 0), (padding, padding), (0, 0)])
    print(f"Chunkwise padded shape: {x1_chunk_padded.shape}")

    out_channels = []
    for c in range(d_model):
        channel_in = x1_chunk_padded[:, :, c]
        w = chunkwise_conv_weight[c, 0, :]
        channel_out = []
        for t in range(seq_len):
            window = channel_in[:, t:t+kernel_size]
            val = mx.sum(window * w, axis=-1) + chunkwise_conv_bias[c]
            channel_out.append(val)
        channel_out = mx.stack(channel_out, axis=1)
        out_channels.append(channel_out)
    x_chunk = mx.stack(out_channels, axis=2)
    mx.eval(x_chunk)
    print(f"Chunkwise conv output (before scale): min={float(mx.min(x_chunk)):.6f}, max={float(mx.max(x_chunk)):.6f}")

    # Compute chunk scale
    left_edge = chunkwise_conv_scale[0]  # (d_model, kernel_size)
    right_edge = chunkwise_conv_scale[1]

    if seq_len < kernel_size:
        left_edge = left_edge[:, :seq_len]
        right_edge = right_edge[:, kernel_size - seq_len:]
    else:
        t = seq_len - kernel_size
        pad_arr = mx.zeros((d_model, t))
        left_edge = mx.concatenate([left_edge, pad_arr], axis=1)
        right_edge = mx.concatenate([pad_arr, right_edge], axis=1)

    scale = 1 + left_edge + right_edge
    scale = mx.transpose(scale)  # (seq_len, d_model)
    scale = mx.expand_dims(scale, 0)  # (1, seq_len, d_model)
    print(f"Chunk scale shape: {scale.shape}")
    print(f"Chunk scale stats: min={float(mx.min(scale)):.6f}, max={float(mx.max(scale)):.6f}")

    x_chunk_scaled = x_chunk * scale
    mx.eval(x_chunk_scaled)
    print(f"Chunkwise conv output (after scale): min={float(mx.min(x_chunk_scaled)):.6f}, max={float(mx.max(x_chunk_scaled)):.6f}")

    # Combine
    conv_combined = x_causal + x_chunk_scaled
    mx.eval(conv_combined)
    print(f"\nCombined causal+chunk: min={float(mx.min(conv_combined)):.6f}, max={float(mx.max(conv_combined)):.6f}")

    # Apply SwooshR
    conv_swoosh = swoosh_r(conv_combined)
    mx.eval(conv_swoosh)
    print(f"After SwooshR: min={float(mx.min(conv_swoosh)):.6f}, max={float(mx.max(conv_swoosh)):.6f}")

    # Output projection
    conv_out_proj = ScaledLinear(d_model, d_model)
    conv_out_proj.load_weights(weights, f"{prefix}.conv_module1.out_proj")

    # Transpose back: (batch, seq, d_model) -> (seq, batch, d_model)
    conv_swoosh_t = mx.transpose(conv_swoosh, (1, 0, 2))
    conv_final = conv_out_proj(conv_swoosh_t)
    mx.eval(conv_final)
    print(f"Conv final output: min={float(mx.min(conv_final)):.6f}, max={float(mx.max(conv_final)):.6f}")

    # Save debug data
    debug_data = {
        "input_src": np.array(src),
        "ff1_out": np.array(ff1_out),
        "conv_glu": np.array(x1_gated),
        "conv_causal": np.array(x_causal),
        "conv_chunk": np.array(x_chunk),
        "conv_chunk_scale": np.array(scale),
        "conv_chunk_scaled": np.array(x_chunk_scaled),
        "conv_combined": np.array(conv_combined),
        "conv_swoosh": np.array(conv_swoosh),
        "conv_final": np.array(conv_final),
    }
    save_file(debug_data, "checkpoints/zipformer/en-streaming/debug_components.safetensors")
    print("\nSaved debug data to debug_components.safetensors")


if __name__ == "__main__":
    main()
