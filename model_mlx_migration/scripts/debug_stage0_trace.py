#!/usr/bin/env python3
"""Debug Stage 0 by tracing intermediate outputs."""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration/src')

import numpy as np
import mlx.core as mx
from safetensors.numpy import load_file, save_file

# Import the Python Zipformer model
from models.zipformer.zipformer import Zipformer, ZipformerConfig, load_zipformer_weights


def main():
    # Load model weights
    weights_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/model.safetensors"
    weights_np = load_file(weights_path)
    weights = {k: mx.array(v) for k, v in weights_np.items()}

    # Create model with config matching checkpoint
    config = ZipformerConfig(
        num_features=80,
        encoder_dims=[192, 256, 384, 512, 384, 256],
        attention_dims=[128, 128, 128, 256, 128, 128],
        num_heads=[4, 4, 4, 8, 4, 4],
        feedforward_dims=[512, 768, 1024, 1536, 1024, 768],  # Using ff2 as base
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

    # Load embed reference for input
    embed_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/embed_reference.safetensors")
    input_fbank = mx.array(embed_ref["input_fbank"])  # (1, 661, 80)

    print(f"Input fbank shape: {input_fbank.shape}")

    # Run embed
    embed_out = model.encoder_embed(input_fbank)  # (time', batch, d_model) if it transposes
    mx.eval(embed_out)
    print(f"Embed output shape: {embed_out.shape}")
    print(f"  Min: {float(embed_out.min()):.6f}")
    print(f"  Max: {float(embed_out.max()):.6f}")
    print(f"  Mean: {float(embed_out.mean()):.6f}")

    # Stage 0 expects (seq, batch, d_model)
    x = embed_out
    if x.shape[0] == 1:  # If (batch, seq, d)
        x = mx.transpose(x, (1, 0, 2))
        print(f"Transposed to: {x.shape}")

    # Get stage 0 encoder
    stage0 = model.encoders[0]
    print(f"\nStage 0 type: {type(stage0).__name__}")

    # Compute positional encoding
    pos_emb = stage0.pos_encoder(x)
    mx.eval(pos_emb)
    print("\nPositional encoding:")
    print(f"  Shape: {pos_emb.shape}")
    print(f"  Min: {float(pos_emb.min()):.6f}")
    print(f"  Max: {float(pos_emb.max()):.6f}")
    print(f"  Mean: {float(pos_emb.mean()):.6f}")
    print(f"  Last col first 5: {[float(pos_emb[0, i, -1]) for i in range(5)]}")

    # Run through layers manually
    out = x
    print("\nInput to layers:")
    print(f"  Shape: {out.shape}")
    print(f"  Min: {float(out.min()):.6f}")
    print(f"  Max: {float(out.max()):.6f}")
    print(f"  Mean: {float(out.mean()):.6f}")

    for i, layer in enumerate(stage0.layers):
        out = layer(out, pos_emb)
        mx.eval(out)
        print(f"\nAfter layer {i}:")
        print(f"  Shape: {out.shape}")
        print(f"  Min: {float(out.min()):.6f}")
        print(f"  Max: {float(out.max()):.6f}")
        print(f"  Mean: {float(out.mean()):.6f}")

    # Compare with reference
    stage_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/stage_references.safetensors")
    stage0_ref = mx.array(stage_ref["stage_0_output"])
    mx.eval(stage0_ref)
    print("\nStage 0 reference:")
    print(f"  Shape: {stage0_ref.shape}")
    print(f"  Min: {float(stage0_ref.min()):.6f}")
    print(f"  Max: {float(stage0_ref.max()):.6f}")
    print(f"  Mean: {float(stage0_ref.mean()):.6f}")

    # Compute difference
    diff = mx.abs(out - stage0_ref)
    max_diff = float(diff.max())
    print("\nDifference between Python run and reference:")
    print(f"  Max diff: {max_diff:.6f}")

    # Save trace for C++ comparison
    intermediates = {
        "embed_out": np.array(embed_out),
        "pos_emb": np.array(pos_emb),
        "layer_0_out": np.array(x),  # Will need to run layers
        "final_out": np.array(out),
    }
    output_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/stage0_trace.safetensors"
    save_file({k: v.astype(np.float32) for k, v in intermediates.items()}, output_path)
    print(f"\nSaved trace to: {output_path}")


if __name__ == "__main__":
    main()
