#!/usr/bin/env python3
"""Generate correct layer0 reference using RelPositionalEncoding."""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration/src')

import numpy as np
import mlx.core as mx
from safetensors.numpy import load_file, save_file

from models.zipformer.zipformer import (
    RelPositionalEncoding, Zipformer2EncoderLayer,
)


def main():
    # Load model weights
    weights_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/model.safetensors"
    weights_np = load_file(weights_path)
    weights = {k: mx.array(v) for k, v in weights_np.items()}

    # Stage 0 config
    d_model = 192
    attention_dim = 128
    num_heads = 4
    feedforward_dim = 512
    kernel_size = 31
    pos_dim = 48
    pos_head_dim = 4
    value_head_dim = 12

    # Create layer
    layer = Zipformer2EncoderLayer(
        d_model=d_model,
        attention_dim=attention_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        kernel_size=kernel_size,
        pos_head_dim=pos_head_dim,
        pos_emb_dim=pos_dim,
        value_head_dim=value_head_dim,
        causal=True,
    )

    # Load layer 0 weights (manual loading)
    prefix = "encoders.0.layers.0"

    # Load self_attn_weights
    layer.self_attn_weights.in_proj.weight = weights.get(f"{prefix}.self_attn_weights.in_proj.weight")
    layer.self_attn_weights.in_proj.bias = weights.get(f"{prefix}.self_attn_weights.in_proj.bias")
    layer.self_attn_weights.linear_pos.weight = weights.get(f"{prefix}.self_attn_weights.linear_pos.weight")

    # Load self_attn1, self_attn2
    for attn_name in ["self_attn1", "self_attn2"]:
        attn = getattr(layer, attn_name)
        attn.in_proj.weight = weights.get(f"{prefix}.{attn_name}.in_proj.weight")
        attn.in_proj.bias = weights.get(f"{prefix}.{attn_name}.in_proj.bias")
        attn.out_proj.weight = weights.get(f"{prefix}.{attn_name}.out_proj.weight")
        attn.out_proj.bias = weights.get(f"{prefix}.{attn_name}.out_proj.bias")

    # Load feedforward modules
    for ff_name in ["feed_forward1", "feed_forward2", "feed_forward3"]:
        ff = getattr(layer, ff_name)
        ff.in_proj.weight = weights.get(f"{prefix}.{ff_name}.in_proj.weight")
        ff.in_proj.bias = weights.get(f"{prefix}.{ff_name}.in_proj.bias")
        ff.out_proj.linear.weight = weights.get(f"{prefix}.{ff_name}.out_proj.weight")
        ff.out_proj.linear.bias = weights.get(f"{prefix}.{ff_name}.out_proj.bias")

    # Load nonlin_attention
    layer.nonlin_attention.in_proj.weight = weights.get(f"{prefix}.nonlin_attention.in_proj.weight")
    layer.nonlin_attention.in_proj.bias = weights.get(f"{prefix}.nonlin_attention.in_proj.bias")
    layer.nonlin_attention.out_proj.weight = weights.get(f"{prefix}.nonlin_attention.out_proj.weight")
    layer.nonlin_attention.out_proj.bias = weights.get(f"{prefix}.nonlin_attention.out_proj.bias")

    # Load conv modules (simplified - need proper depthwise loading)
    # This is just structural loading, may need adjustment
    for conv_name in ["conv_module1", "conv_module2"]:
        conv = getattr(layer, conv_name)
        conv.in_proj.weight = weights.get(f"{prefix}.{conv_name}.in_proj.weight")
        conv.in_proj.bias = weights.get(f"{prefix}.{conv_name}.in_proj.bias")
        conv.out_proj.linear.weight = weights.get(f"{prefix}.{conv_name}.out_proj.weight")
        conv.out_proj.linear.bias = weights.get(f"{prefix}.{conv_name}.out_proj.bias")

        # Depthwise conv weights
        if hasattr(conv.depthwise_conv, 'causal_conv'):
            conv.depthwise_conv.causal_conv.weight = weights.get(
                f"{prefix}.{conv_name}.depthwise_conv.causal_conv.weight")
            conv.depthwise_conv.causal_conv.bias = weights.get(
                f"{prefix}.{conv_name}.depthwise_conv.causal_conv.bias")
            conv.depthwise_conv.chunkwise_conv.weight = weights.get(
                f"{prefix}.{conv_name}.depthwise_conv.chunkwise_conv.weight")
            conv.depthwise_conv.chunkwise_conv.bias = weights.get(
                f"{prefix}.{conv_name}.depthwise_conv.chunkwise_conv.bias")
            conv.depthwise_conv.chunkwise_conv_scale = weights.get(
                f"{prefix}.{conv_name}.depthwise_conv.chunkwise_conv_scale")

    # Load bypass modules
    layer.bypass.bypass_scale = weights.get(f"{prefix}.bypass.bypass_scale")
    layer.bypass_mid.bypass_scale = weights.get(f"{prefix}.bypass_mid.bypass_scale")
    layer.bypass_scale = weights.get(f"{prefix}.bypass_scale")

    # Load norm
    layer.norm.bias = weights.get(f"{prefix}.norm.bias")
    layer.norm.log_scale = weights.get(f"{prefix}.norm.log_scale")

    # Use embed reference input
    embed_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/embed_reference.safetensors")
    embed_out = mx.array(embed_ref["embed_output"])  # (1, 327, 192) - (batch, seq, d_model)

    # Transpose to (seq, batch, d_model) for layer input
    src = mx.transpose(embed_out, (1, 0, 2))
    seq_len = src.shape[0]
    batch_size = src.shape[1]

    print(f"Input src shape: {src.shape}")

    # Create pos_emb using RelPositionalEncoding
    pos_enc = RelPositionalEncoding(pos_dim=pos_dim)
    pos_emb = pos_enc(src)
    mx.eval(pos_emb)

    print(f"Pos emb shape: {pos_emb.shape}")
    print(f"Pos emb last col first 5: {[float(pos_emb[0, i, -1]) for i in range(5)]}")

    # Run layer forward
    output = layer(src, pos_emb)
    mx.eval(output)

    print(f"Output shape: {output.shape}")
    print(f"Output stats: min={float(output.min()):.6f} max={float(output.max()):.6f} mean={float(output.mean()):.6f}")

    # Save reference
    output_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/layer0_reference_fixed.safetensors"
    save_file({
        "input_src": np.array(src).astype(np.float32),
        "input_pos_emb": np.array(pos_emb).astype(np.float32),
        "output": np.array(output).astype(np.float32),
    }, output_path)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
