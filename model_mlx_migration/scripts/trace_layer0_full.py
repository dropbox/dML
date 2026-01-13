#!/usr/bin/env python3
"""Trace through layer 0 components to save ALL intermediate values."""

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

    print("\n=== Layer 0 Full Trace ===")
    save_data = {
        "input_src": np.array(embed_out).astype(np.float32),
        "input_pos_emb": np.array(pos_emb).astype(np.float32),
    }

    # 1. Attention weights
    attn_weights = layer0.self_attn_weights(out, pos_emb)
    mx.eval(attn_weights)
    print(f"1. Attn weights: {attn_weights.shape}, min={float(attn_weights.min()):.6f}, max={float(attn_weights.max()):.6f}")
    save_data["attn_weights"] = np.array(attn_weights).astype(np.float32)

    # 2. FF1
    ff1_out = layer0.feed_forward1(out)
    mx.eval(ff1_out)
    print(f"2. FF1: {ff1_out.shape}, min={float(ff1_out.min()):.6f}, max={float(ff1_out.max()):.6f}")
    save_data["ff1_out"] = np.array(ff1_out).astype(np.float32)
    out = out + ff1_out
    save_data["after_ff1"] = np.array(out).astype(np.float32)

    # 3. NonlinAttention
    na_out = layer0.nonlin_attention(out, attn_weights)
    mx.eval(na_out)
    print(f"3. NonlinAttn: {na_out.shape}, min={float(na_out.min()):.6f}, max={float(na_out.max()):.6f}")
    save_data["nonlin_attn_out"] = np.array(na_out).astype(np.float32)
    out = out + na_out
    save_data["after_nonlin_attn"] = np.array(out).astype(np.float32)

    # 4. SelfAttn1
    sa1_out = layer0.self_attn1(out, attn_weights)
    mx.eval(sa1_out)
    print(f"4. SelfAttn1: {sa1_out.shape}, min={float(sa1_out.min()):.6f}, max={float(sa1_out.max()):.6f}")
    save_data["self_attn1_out"] = np.array(sa1_out).astype(np.float32)
    out = out + sa1_out
    save_data["after_self_attn1"] = np.array(out).astype(np.float32)

    # 5. Conv1
    conv1_out = layer0.conv_module1(out)
    mx.eval(conv1_out)
    print(f"5. Conv1: {conv1_out.shape}, min={float(conv1_out.min()):.6f}, max={float(conv1_out.max()):.6f}")
    save_data["conv1_out"] = np.array(conv1_out).astype(np.float32)
    out = out + conv1_out
    save_data["after_conv1"] = np.array(out).astype(np.float32)

    # 6. FF2
    ff2_out = layer0.feed_forward2(out)
    mx.eval(ff2_out)
    print(f"6. FF2: {ff2_out.shape}, min={float(ff2_out.min()):.6f}, max={float(ff2_out.max()):.6f}")
    save_data["ff2_out"] = np.array(ff2_out).astype(np.float32)
    out = out + ff2_out
    save_data["after_ff2"] = np.array(out).astype(np.float32)

    # 7. Bypass mid
    out = layer0.bypass_mid(src_orig, out)
    mx.eval(out)
    print(f"7. Bypass mid: {out.shape}, min={float(out.min()):.6f}, max={float(out.max()):.6f}")
    save_data["after_bypass_mid"] = np.array(out).astype(np.float32)

    # 8. SelfAttn2
    sa2_out = layer0.self_attn2(out, attn_weights)
    mx.eval(sa2_out)
    print(f"8. SelfAttn2: {sa2_out.shape}, min={float(sa2_out.min()):.6f}, max={float(sa2_out.max()):.6f}")
    save_data["self_attn2_out"] = np.array(sa2_out).astype(np.float32)
    out = out + sa2_out
    save_data["after_self_attn2"] = np.array(out).astype(np.float32)

    # 9. Conv2
    conv2_out = layer0.conv_module2(out)
    mx.eval(conv2_out)
    print(f"9. Conv2: {conv2_out.shape}, min={float(conv2_out.min()):.6f}, max={float(conv2_out.max()):.6f}")
    save_data["conv2_out"] = np.array(conv2_out).astype(np.float32)
    out = out + conv2_out
    save_data["after_conv2"] = np.array(out).astype(np.float32)

    # 10. FF3
    ff3_out = layer0.feed_forward3(out)
    mx.eval(ff3_out)
    print(f"10. FF3: {ff3_out.shape}, min={float(ff3_out.min()):.6f}, max={float(ff3_out.max()):.6f}")
    save_data["ff3_out"] = np.array(ff3_out).astype(np.float32)
    out = out + ff3_out
    save_data["after_ff3"] = np.array(out).astype(np.float32)

    # 11. Norm
    out = layer0.norm(out)
    mx.eval(out)
    print(f"11. Norm: {out.shape}, min={float(out.min()):.6f}, max={float(out.max()):.6f}")
    save_data["after_norm"] = np.array(out).astype(np.float32)

    # 12. Final bypass
    out = layer0.bypass(src_orig, out)
    mx.eval(out)
    print(f"12. Bypass: {out.shape}, min={float(out.min()):.6f}, max={float(out.max()):.6f}")
    save_data["output"] = np.array(out).astype(np.float32)

    print(f"\nFinal output: min={float(out.min()):.6f}, max={float(out.max()):.6f}, mean={float(out.mean()):.6f}")

    # Save all intermediate values
    save_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/layer0_full_trace.safetensors"
    save_file(save_data, save_path)
    print(f"\nSaved full trace to: {save_path}")
    print(f"Keys: {list(save_data.keys())}")


if __name__ == "__main__":
    main()
