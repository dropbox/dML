#!/usr/bin/env python3
"""Generate stage references using PyTorch embed output as input.

This ensures we're comparing C++ against what it should produce,
using the same embed output that C++ produces.
"""

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

    # Create model with config matching checkpoint
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

    # Use the PyTorch embed output as input to stages
    # embed_output is (batch, seq, d) = (1, 327, 192)
    embed_out_bt = mx.array(embed_ref["embed_output"])  # (1, 327, 192)

    # Transpose to (seq, batch, d) for encoder stages
    embed_out = mx.transpose(embed_out_bt, (1, 0, 2))  # (327, 1, 192)
    mx.eval(embed_out)

    print("Using PyTorch embed output as input:")
    print(f"  Shape: {embed_out.shape}")
    print(f"  min={float(embed_out.min()):.6f}, max={float(embed_out.max()):.6f}")

    # Run through each stage
    x = embed_out
    stage_outputs = {}

    for i, encoder in enumerate(model.encoders):
        # Convert channels
        x = model._convert_num_channels(x, model.encoder_dims[i])
        x = encoder(x)
        mx.eval(x)
        stage_outputs[f"stage_{i}_output"] = np.array(x).astype(np.float32)
        print(f"Stage {i}: {x.shape}, min={float(x.min()):.6f}, max={float(x.max()):.6f}")

    # Combine multi-scale outputs
    outputs = [mx.array(stage_outputs[f"stage_{i}_output"]) for i in range(6)]
    x_full = model._get_full_dim_output(outputs)
    mx.eval(x_full)
    print(f"\nCombined output: {x_full.shape}, min={float(x_full.min()):.6f}, max={float(x_full.max()):.6f}")

    # Final downsampling
    seq_len, batch_size, d_model = x_full.shape
    ds = 2
    d_seq_len = (seq_len + ds - 1) // ds

    pad = d_seq_len * ds - seq_len
    if pad > 0:
        x_full = mx.concatenate([x_full, mx.broadcast_to(x_full[-1:], (pad, batch_size, d_model))], axis=0)

    x_full = mx.reshape(x_full, (d_seq_len, ds, batch_size, d_model))
    weights_ds = mx.softmax(model.downsample_output_bias)[:, None, None]
    x_full = mx.sum(x_full * weights_ds, axis=1)

    # Transpose to (batch, time, d_model)
    x_full = mx.transpose(x_full, (1, 0, 2))
    mx.eval(x_full)

    print(f"\nFull encoder output: {x_full.shape}")
    print(f"  min={float(x_full.min()):.6f}, max={float(x_full.max()):.6f}, mean={float(x_full.mean()):.6f}")

    # Save references
    stage_outputs["encoder_output"] = np.array(x_full).astype(np.float32)

    output_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/stage_references.safetensors"
    save_file(stage_outputs, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
