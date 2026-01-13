#!/usr/bin/env python3
"""Trace through Python encoder to generate per-layer references."""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration/src')

import numpy as np
import mlx.core as mx
from safetensors.numpy import load_file, save_file

from models.zipformer.zipformer import (
    Zipformer, ZipformerConfig, load_zipformer_weights,
)


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

    # Use embed reference input
    embed_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/embed_reference.safetensors")
    embed_out = mx.array(embed_ref["embed_output"])  # (1, 327, 192) - (batch, seq, d_model)

    # Transpose to (seq, batch, d_model) for stage processing
    x = mx.transpose(embed_out, (1, 0, 2))
    print(f"Input to Stage 0: {x.shape}")
    print(f"  Stats: min={float(x.min()):.6f} max={float(x.max()):.6f} mean={float(x.mean()):.6f}")

    # Get stage 0 encoder
    stage0 = model.encoders[0]

    # Compute pos_emb
    pos_emb = stage0.pos_encoder(x)
    mx.eval(pos_emb)
    print(f"\nPos emb shape: {pos_emb.shape}")
    print(f"Pos emb last col first 5: {[float(pos_emb[0, i, -1]) for i in range(5)]}")
    print(f"Pos emb stats: min={float(pos_emb.min()):.6f} max={float(pos_emb.max()):.6f}")

    # Save trace data for comparison
    traces = {
        "input": np.array(x).astype(np.float32),
        "pos_emb": np.array(pos_emb).astype(np.float32),
    }

    # Run through layers one by one
    out = x
    for i, layer in enumerate(stage0.layers):
        out = layer(out, pos_emb)
        mx.eval(out)
        print(f"\nAfter layer {i}:")
        print(f"  Shape: {out.shape}")
        print(f"  Stats: min={float(out.min()):.6f} max={float(out.max()):.6f} mean={float(out.mean()):.6f}")
        traces[f"after_layer_{i}"] = np.array(out).astype(np.float32)

    # Compare with stage reference
    stage_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/stage_references.safetensors")
    stage0_ref = mx.array(stage_ref["stage_0_output"])
    mx.eval(stage0_ref)

    print("\nStage 0 reference:")
    print(f"  Stats: min={float(stage0_ref.min()):.6f} max={float(stage0_ref.max()):.6f} mean={float(stage0_ref.mean()):.6f}")

    diff = mx.abs(out - stage0_ref)
    print(f"\nDifference with reference: max_diff={float(diff.max()):.6f}")

    if float(diff.max()) > 1e-5:
        print("  WARNING: Output differs from reference!")

    # Save traces
    output_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/python_stage0_trace.safetensors"
    save_file(traces, output_path)
    print(f"\nSaved traces to: {output_path}")


if __name__ == "__main__":
    main()
