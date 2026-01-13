#!/usr/bin/env python3
"""Compare C++ and Python layer 0 outputs step by step."""

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

    # Load the ORIGINAL PyTorch embed reference (same as C++ uses)
    embed_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/embed_reference.safetensors")
    embed_out_bt = mx.array(embed_ref["embed_output"])  # (1, 327, 192) - (batch, seq, d)

    # Transpose to (seq, batch, d) for encoder stages
    embed_out = mx.transpose(embed_out_bt, (1, 0, 2))  # (327, 1, 192)
    mx.eval(embed_out)

    print(f"Embed input: {embed_out.shape}")
    print(f"  min={float(embed_out.min()):.6f}, max={float(embed_out.max()):.6f}")

    # Get stage 0 encoder
    stage0 = model.encoders[0]
    layer0 = stage0.layers[0]

    # Generate positional encoding
    pos_emb = stage0.pos_encoder(embed_out)
    mx.eval(pos_emb)
    print(f"\nPos emb: {pos_emb.shape}")
    print(f"  min={float(pos_emb.min()):.6f}, max={float(pos_emb.max()):.6f}")

    # Run layer 0
    layer0_out = layer0(embed_out, pos_emb)
    mx.eval(layer0_out)
    print(f"\nPython Layer 0 output: {layer0_out.shape}")
    print(f"  min={float(layer0_out.min()):.6f}, max={float(layer0_out.max()):.6f}, mean={float(layer0_out.mean()):.6f}")

    # Run layer 1
    layer1_out = stage0.layers[1](layer0_out, pos_emb)
    mx.eval(layer1_out)
    print(f"\nPython Layer 1 output: {layer1_out.shape}")
    print(f"  min={float(layer1_out.min()):.6f}, max={float(layer1_out.max()):.6f}, mean={float(layer1_out.mean()):.6f}")

    # This is stage 0 output
    print("\n=== Stage 0 output (after both layers) ===")
    print(f"  min={float(layer1_out.min()):.6f}, max={float(layer1_out.max()):.6f}")

    # Save references for C++ comparison
    save_data = {
        "embed_input": np.array(embed_out).astype(np.float32),  # (327, 1, 192)
        "pos_emb": np.array(pos_emb).astype(np.float32),  # (1, 653, 48)
        "layer0_output": np.array(layer0_out).astype(np.float32),  # (327, 1, 192)
        "layer1_output": np.array(layer1_out).astype(np.float32),  # (327, 1, 192)
    }

    save_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/layer_debug_reference.safetensors"
    save_file(save_data, save_path)
    print(f"\nSaved debug reference to: {save_path}")


if __name__ == "__main__":
    main()
