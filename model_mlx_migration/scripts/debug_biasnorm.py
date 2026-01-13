#!/usr/bin/env python3
"""Debug BiasNorm to understand value explosion."""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration/src')

import mlx.core as mx
from safetensors.numpy import load_file

from models.zipformer.zipformer import (
    Zipformer, ZipformerConfig, load_zipformer_weights,
)


def debug_biasnorm(norm, x, name=""):
    """Trace through BiasNorm step by step."""
    print(f"\n=== BiasNorm Debug: {name} ===")
    print(f"Input x: shape={x.shape}, min={float(x.min()):.6f}, max={float(x.max()):.6f}, mean={float(x.mean()):.6f}")

    # Get norm parameters
    bias = norm.bias
    log_scale = norm.log_scale
    mx.eval(bias, log_scale)

    print(f"Bias: shape={bias.shape}, min={float(bias.min()):.6f}, max={float(bias.max()):.6f}, mean={float(bias.mean()):.6f}")
    print(f"log_scale: {float(log_scale):.6f}, exp(log_scale)={float(mx.exp(log_scale)):.6f}")

    # Step through BiasNorm
    channel_dim = norm.channel_dim
    if channel_dim < 0:
        channel_dim = x.ndim + channel_dim

    print(f"channel_dim (resolved): {channel_dim}")

    # Expand bias
    expanded_bias = bias
    for _ in range(channel_dim + 1, x.ndim):
        expanded_bias = mx.expand_dims(expanded_bias, axis=-1)
    mx.eval(expanded_bias)
    print(f"Expanded bias shape: {expanded_bias.shape}")

    # Compute centered
    centered = x - expanded_bias
    mx.eval(centered)
    print(f"Centered: min={float(centered.min()):.6f}, max={float(centered.max()):.6f}, mean={float(centered.mean()):.6f}")

    # Compute variance
    variance = mx.mean(centered * centered, axis=channel_dim, keepdims=True)
    mx.eval(variance)
    print(f"Variance: shape={variance.shape}, min={float(variance.min()):.6f}, max={float(variance.max()):.6f}, mean={float(variance.mean()):.6f}")

    # Check for very small variance values
    small_var_mask = variance < 1e-4
    mx.eval(small_var_mask)
    num_small = float(mx.sum(small_var_mask.astype(mx.float32)))
    print(f"  Variance < 1e-4: {int(num_small)}/{variance.size} elements")

    # Compute rsqrt
    rsqrt_var = mx.rsqrt(variance + 1e-8)
    mx.eval(rsqrt_var)
    print(f"rsqrt(variance+eps): min={float(rsqrt_var.min()):.6f}, max={float(rsqrt_var.max()):.6f}, mean={float(rsqrt_var.mean()):.6f}")

    # Compute scale
    clamped_log_scale = mx.clip(log_scale, norm.log_scale_min, norm.log_scale_max)
    scale = rsqrt_var * mx.exp(clamped_log_scale)
    mx.eval(scale)
    print(f"Scale: min={float(scale.min()):.6f}, max={float(scale.max()):.6f}, mean={float(scale.mean()):.6f}")

    # Compute output
    output = x * scale
    mx.eval(output)
    print(f"Output: min={float(output.min()):.6f}, max={float(output.max()):.6f}, mean={float(output.mean()):.6f}")

    return output


def main():
    # Load model
    weights_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/model.safetensors"
    weights_np = load_file(weights_path)
    weights = {k: mx.array(v) for k, v in weights_np.items()}

    config = ZipformerConfig(
        num_features=80,
        encoder_dims=[192, 256, 384, 512, 384, 256],
        attention_dims=[128, 128, 128, 256, 128, 128],
        num_heads=[4, 4, 4, 4, 4, 4],
        feedforward_dims=[512, 768, 1024, 1536, 1024, 768],
        downsampling_factors=[1, 2, 4, 8, 4, 2],
        num_encoder_layers=[2, 2, 3, 4, 3, 2],
        cnn_module_kernels=[31, 31, 15, 15, 15, 31],
        pos_dim=48,
        pos_head_dim=4,
        value_head_dim=12,
        causal=True,
    )

    model = Zipformer(config)
    load_zipformer_weights(model, weights)

    # Get stage 0 and layer 0
    stage0 = model.encoders[0]
    layer0 = stage0.layers[0]

    # Get input
    embed_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/embed_reference.safetensors")
    embed_out = mx.array(embed_ref["embed_output"])
    src = mx.transpose(embed_out, (1, 0, 2))  # (seq, batch, d_model)

    pos_emb = stage0.pos_encoder(src)
    mx.eval(pos_emb)

    print(f"Input src: {src.shape}, min={float(src.min()):.4f}, max={float(src.max()):.4f}")
    print(f"Pos emb: {pos_emb.shape}")

    # Run through layer components up to norm
    src_orig = src
    out = src

    # 1. Compute attention weights
    attn_weights = layer0.self_attn_weights(out, pos_emb)
    mx.eval(attn_weights)

    # 2. Feed forward 1
    ff1_out = layer0.feed_forward1(out)
    mx.eval(ff1_out)
    out = out + ff1_out

    # 3. NonlinAttention
    na_out = layer0.nonlin_attention(out, attn_weights[0:1])
    mx.eval(na_out)
    out = out + na_out

    # 4. Self attention 1
    sa1_out = layer0.self_attn1(out, attn_weights)
    mx.eval(sa1_out)
    out = out + sa1_out

    # 5. Conv module 1
    conv1_out = layer0.conv_module1(out)
    mx.eval(conv1_out)
    out = out + conv1_out

    # 6. Feed forward 2
    ff2_out = layer0.feed_forward2(out)
    mx.eval(ff2_out)
    out = out + ff2_out

    # 7. Bypass mid
    out = layer0.bypass_mid(src_orig, out)
    mx.eval(out)

    # 8. Self attention 2
    sa2_out = layer0.self_attn2(out, attn_weights)
    mx.eval(sa2_out)
    out = out + sa2_out

    # 9. Conv module 2
    conv2_out = layer0.conv_module2(out)
    mx.eval(conv2_out)
    out = out + conv2_out

    # 10. Feed forward 3
    ff3_out = layer0.feed_forward3(out)
    mx.eval(ff3_out)
    out = out + ff3_out

    print("\n=== Before BiasNorm ===")
    print(f"out: min={float(out.min()):.6f}, max={float(out.max()):.6f}, mean={float(out.mean()):.6f}")

    # Debug the BiasNorm
    norm_out = debug_biasnorm(layer0.norm, out, "layer0.norm")

    # Compare what the loaded weights say
    print("\n=== Direct weight check ===")
    prefix = "encoders.0.layers.0.norm"
    if f"{prefix}.bias" in weights:
        w_bias = weights[f"{prefix}.bias"]
        mx.eval(w_bias)
        print(f"Checkpoint bias: shape={w_bias.shape}, min={float(w_bias.min()):.6f}, max={float(w_bias.max()):.6f}")
    if f"{prefix}.log_scale" in weights:
        w_ls = weights[f"{prefix}.log_scale"]
        mx.eval(w_ls)
        print(f"Checkpoint log_scale: {float(w_ls):.6f}")

    # Check if model weights match checkpoint
    print("\n=== Model vs Checkpoint weights ===")
    model_bias = layer0.norm.bias
    model_ls = layer0.norm.log_scale
    mx.eval(model_bias, model_ls)

    if f"{prefix}.bias" in weights:
        diff = mx.abs(model_bias - weights[f"{prefix}.bias"])
        mx.eval(diff)
        print(f"Bias diff: max={float(diff.max()):.6f}")
    if f"{prefix}.log_scale" in weights:
        diff = mx.abs(model_ls - weights[f"{prefix}.log_scale"])
        mx.eval(diff)
        print(f"log_scale diff: max={float(diff):.6f}")


if __name__ == "__main__":
    main()
