#!/usr/bin/env python3
"""Trace through layer 0 components to identify where C++ diverges."""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration/src')

import numpy as np
import mlx.core as mx
from safetensors.numpy import load_file, save_file

from models.zipformer.zipformer import Zipformer, ZipformerConfig, load_zipformer_weights


def main():
    # Load model weights
    weights_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/model.safetensors"
    weights_np = load_file(weights_path)
    weights = {k: mx.array(v) for k, v in weights_np.items()}

    # Create model
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

    # Load the ORIGINAL PyTorch embed reference
    embed_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/embed_reference.safetensors")
    embed_out_bt = mx.array(embed_ref["embed_output"])  # (1, 327, 192)
    embed_out = mx.transpose(embed_out_bt, (1, 0, 2))  # (327, 1, 192)
    mx.eval(embed_out)

    print(f"Embed input: {embed_out.shape}, min={float(embed_out.min()):.4f}, max={float(embed_out.max()):.4f}")

    # Get stage 0 encoder
    stage0 = model.encoders[0]
    layer0 = stage0.layers[0]

    # Generate positional encoding
    pos_emb = stage0.pos_encoder(embed_out)
    mx.eval(pos_emb)
    print(f"Pos emb: {pos_emb.shape}, min={float(pos_emb.min()):.4f}, max={float(pos_emb.max()):.4f}")

    # Trace through layer 0 step by step
    src_orig = embed_out
    out = embed_out

    print("\n=== Layer 0 Trace ===")

    # 1. Attention weights
    attn_weights = layer0.self_attn_weights(out, pos_emb)
    mx.eval(attn_weights)
    print(f"1. Attn weights: {attn_weights.shape}, min={float(attn_weights.min()):.6f}, max={float(attn_weights.max()):.6f}")

    # 2. FF1
    ff1_out = layer0.feed_forward1(out)
    mx.eval(ff1_out)
    print(f"2. FF1: min={float(ff1_out.min()):.6f}, max={float(ff1_out.max()):.6f}, mean={float(ff1_out.mean()):.6f}")
    out = out + ff1_out

    # 3. NonlinAttention (uses ALL heads)
    na_out = layer0.nonlin_attention(out, attn_weights)
    mx.eval(na_out)
    print(f"3. NonlinAttn: min={float(na_out.min()):.6f}, max={float(na_out.max()):.6f}")
    out = out + na_out

    # 4. SelfAttn1
    sa1_out = layer0.self_attn1(out, attn_weights)
    mx.eval(sa1_out)
    print(f"4. SelfAttn1: min={float(sa1_out.min()):.6f}, max={float(sa1_out.max()):.6f}")
    out = out + sa1_out

    # 5. Conv1
    conv1_out = layer0.conv_module1(out)
    mx.eval(conv1_out)
    print(f"5. Conv1: min={float(conv1_out.min()):.6f}, max={float(conv1_out.max()):.6f}")
    out = out + conv1_out

    # 6. FF2
    ff2_out = layer0.feed_forward2(out)
    mx.eval(ff2_out)
    print(f"6. FF2: min={float(ff2_out.min()):.6f}, max={float(ff2_out.max()):.6f}")
    out = out + ff2_out

    # 7. Bypass mid
    out = layer0.bypass_mid(src_orig, out)
    mx.eval(out)
    print(f"7. Bypass mid: min={float(out.min()):.6f}, max={float(out.max()):.6f}")

    # 8. SelfAttn2
    sa2_out = layer0.self_attn2(out, attn_weights)
    mx.eval(sa2_out)
    print(f"8. SelfAttn2: min={float(sa2_out.min()):.6f}, max={float(sa2_out.max()):.6f}")
    out = out + sa2_out

    # 9. Conv2
    conv2_out = layer0.conv_module2(out)
    mx.eval(conv2_out)
    print(f"9. Conv2: min={float(conv2_out.min()):.6f}, max={float(conv2_out.max()):.6f}")
    out = out + conv2_out

    # 10. FF3
    ff3_out = layer0.feed_forward3(out)
    mx.eval(ff3_out)
    print(f"10. FF3: min={float(ff3_out.min()):.6f}, max={float(ff3_out.max()):.6f}")
    out = out + ff3_out

    # 11. Norm
    out = layer0.norm(out)
    mx.eval(out)
    print(f"11. Norm: min={float(out.min()):.6f}, max={float(out.max()):.6f}")

    # 12. Final bypass
    out = layer0.bypass(src_orig, out)
    mx.eval(out)
    print(f"12. Bypass: min={float(out.min()):.6f}, max={float(out.max()):.6f}")

    print(f"\nFinal output: min={float(out.min()):.6f}, max={float(out.max()):.6f}, mean={float(out.mean()):.6f}")

    # Save intermediate values for C++ comparison
    save_data = {
        "input_src": np.array(embed_out).astype(np.float32),
        "input_pos_emb": np.array(pos_emb).astype(np.float32),
        "attn_weights": np.array(attn_weights).astype(np.float32),
        "ff1_out": np.array(ff1_out).astype(np.float32),
        "output": np.array(out).astype(np.float32),
    }
    save_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/layer0_trace.safetensors"
    save_file(save_data, save_path)
    print(f"\nSaved trace to: {save_path}")


if __name__ == "__main__":
    main()
