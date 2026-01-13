#!/usr/bin/env python3
"""Generate encoder reference outputs from Python MLX model."""

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

    # Load input fbank
    embed_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/embed_reference.safetensors")
    input_fbank = mx.array(embed_ref["input_fbank"])  # (1, 661, 80)

    print(f"Input fbank shape: {input_fbank.shape}")

    # Run embed
    embed_out = model.encoder_embed(input_fbank)
    mx.eval(embed_out)
    print(f"Embed output: {embed_out.shape}, min={float(embed_out.min()):.6f}, max={float(embed_out.max()):.6f}")

    # Run through each stage and save intermediate outputs
    x = embed_out
    stage_outputs = {}

    for i, encoder in enumerate(model.encoders):
        # Convert channels
        x = model._convert_num_channels(x, model.encoder_dims[i])
        x = encoder(x)
        mx.eval(x)
        stage_outputs[f"stage_{i}_output"] = np.array(x).astype(np.float32)
        print(f"Stage {i}: {x.shape}, min={float(x.min()):.6f}, max={float(x.max()):.6f}")

    # Run full encoder
    encoder_out, _ = model(input_fbank)
    mx.eval(encoder_out)
    print(f"\nFull encoder output: {encoder_out.shape}")
    print(f"  min={float(encoder_out.min()):.6f}, max={float(encoder_out.max()):.6f}, mean={float(encoder_out.mean()):.6f}")

    # Save references
    output_data = {
        "input_fbank": np.array(input_fbank).astype(np.float32),
        "embed_output": np.array(mx.transpose(embed_out, (1, 0, 2))).astype(np.float32),  # (batch, seq, d)
        "encoder_output": np.array(encoder_out).astype(np.float32),  # (batch, time, d)
    }
    output_data.update(stage_outputs)

    output_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/encoder_reference_mlx.safetensors"
    save_file(output_data, output_path)
    print(f"\nSaved to: {output_path}")

    # Also update the stage references for comparison
    stage_ref_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/stage_references.safetensors"
    save_file(stage_outputs, stage_ref_path)
    print(f"Updated: {stage_ref_path}")


if __name__ == "__main__":
    main()
