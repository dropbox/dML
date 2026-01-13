#!/usr/bin/env python3
"""Trace through Python layer 0 components to find divergence."""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration/src')

import mlx.core as mx
from safetensors.numpy import load_file

from models.zipformer.zipformer import (
    Zipformer, ZipformerConfig, load_zipformer_weights,
)


def main():
    # Load model
    weights_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/model.safetensors"
    weights_np = load_file(weights_path)
    weights = {k: mx.array(v) for k, v in weights_np.items()}

    config = ZipformerConfig(
        num_features=80,
        encoder_dims=[192, 256, 384, 512, 384, 256],
        attention_dims=[128, 128, 128, 256, 128, 128],
        num_heads=[4, 4, 4, 8, 4, 4],
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
    print(f"Pos emb: {pos_emb.shape}, min={float(pos_emb.min()):.4f}, max={float(pos_emb.max()):.4f}")

    # Step through layer 0 manually
    src_orig = src
    out = src

    # 1. Compute attention weights
    print("\n=== Attention Weights ===")
    attn_weights = layer0.self_attn_weights(out, pos_emb)
    mx.eval(attn_weights)
    print(f"Attention weights: {attn_weights.shape}")
    print(f"  min={float(attn_weights.min()):.6f}, max={float(attn_weights.max()):.6f}")
    print(f"  mean={float(attn_weights.mean()):.6f}")
    # Check if attention weights sum to 1 (should be after softmax)
    attn_sum = mx.sum(attn_weights, axis=-1)
    mx.eval(attn_sum)
    print(f"  sum along key dim (should be ~1.0): min={float(attn_sum.min()):.4f}, max={float(attn_sum.max()):.4f}")

    # 2. Feed forward 1
    print("\n=== Feed Forward 1 ===")
    ff1_out = layer0.feed_forward1(out)
    mx.eval(ff1_out)
    print(f"FF1 output: min={float(ff1_out.min()):.4f}, max={float(ff1_out.max()):.4f}, mean={float(ff1_out.mean()):.4f}")
    out = out + ff1_out

    # 3. NonlinAttention
    print("\n=== NonlinAttention ===")
    # NonlinAttention uses ALL heads of attention weights (not just first)
    na_out = layer0.nonlin_attention(out, attn_weights)
    mx.eval(na_out)
    print(f"NonlinAttn output: min={float(na_out.min()):.4f}, max={float(na_out.max()):.4f}, mean={float(na_out.mean()):.4f}")
    out = out + na_out

    # 4. Self attention 1
    print("\n=== Self Attention 1 ===")
    sa1_out = layer0.self_attn1(out, attn_weights)
    mx.eval(sa1_out)
    print(f"SelfAttn1 output: min={float(sa1_out.min()):.4f}, max={float(sa1_out.max()):.4f}, mean={float(sa1_out.mean()):.4f}")
    out = out + sa1_out

    # 5. Conv module 1
    print("\n=== Conv Module 1 ===")
    conv1_out = layer0.conv_module1(out)
    mx.eval(conv1_out)
    print(f"Conv1 output: min={float(conv1_out.min()):.4f}, max={float(conv1_out.max()):.4f}, mean={float(conv1_out.mean()):.4f}")
    out = out + conv1_out

    # 6. Feed forward 2
    print("\n=== Feed Forward 2 ===")
    ff2_out = layer0.feed_forward2(out)
    mx.eval(ff2_out)
    print(f"FF2 output: min={float(ff2_out.min()):.4f}, max={float(ff2_out.max()):.4f}, mean={float(ff2_out.mean()):.4f}")
    out = out + ff2_out

    # 7. Bypass mid
    print("\n=== Bypass Mid ===")
    out = layer0.bypass_mid(src_orig, out)
    mx.eval(out)
    print(f"After bypass_mid: min={float(out.min()):.4f}, max={float(out.max()):.4f}, mean={float(out.mean()):.4f}")

    # 8. Self attention 2
    print("\n=== Self Attention 2 ===")
    sa2_out = layer0.self_attn2(out, attn_weights)
    mx.eval(sa2_out)
    print(f"SelfAttn2 output: min={float(sa2_out.min()):.4f}, max={float(sa2_out.max()):.4f}, mean={float(sa2_out.mean()):.4f}")
    out = out + sa2_out

    # 9. Conv module 2
    print("\n=== Conv Module 2 ===")
    conv2_out = layer0.conv_module2(out)
    mx.eval(conv2_out)
    print(f"Conv2 output: min={float(conv2_out.min()):.4f}, max={float(conv2_out.max()):.4f}, mean={float(conv2_out.mean()):.4f}")
    out = out + conv2_out

    # 10. Feed forward 3
    print("\n=== Feed Forward 3 ===")
    ff3_out = layer0.feed_forward3(out)
    mx.eval(ff3_out)
    print(f"FF3 output: min={float(ff3_out.min()):.4f}, max={float(ff3_out.max()):.4f}, mean={float(ff3_out.mean()):.4f}")
    out = out + ff3_out

    # 11. Norm and bypass
    print("\n=== Final Norm and Bypass ===")
    out = layer0.norm(out)
    mx.eval(out)
    print(f"After norm: min={float(out.min()):.4f}, max={float(out.max()):.4f}, mean={float(out.mean()):.4f}")

    out = layer0.bypass(src_orig, out)
    mx.eval(out)
    print(f"After final bypass: min={float(out.min()):.4f}, max={float(out.max()):.4f}, mean={float(out.mean()):.4f}")

    print("\n=== Final Output ===")
    print(f"Layer 0 output: min={float(out.min()):.4f}, max={float(out.max()):.4f}, mean={float(out.mean()):.4f}")


if __name__ == "__main__":
    main()
