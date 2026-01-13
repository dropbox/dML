#!/usr/bin/env python3
"""Test full encoder against reference output."""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration/src')

import mlx.core as mx
from safetensors.numpy import load_file

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

    # Load embed reference for input
    embed_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/embed_reference.safetensors")
    input_fbank = mx.array(embed_ref["input_fbank"])  # (1, 661, 80)

    print(f"Input fbank shape: {input_fbank.shape}")

    # Run embed
    embed_out = model.encoder_embed(input_fbank)
    mx.eval(embed_out)
    print(f"Embed output shape: {embed_out.shape}")
    print(f"  Min: {float(embed_out.min()):.6f}")
    print(f"  Max: {float(embed_out.max()):.6f}")

    # Compare embed output with reference
    ref_embed = mx.array(embed_ref["embed_output"])
    mx.eval(ref_embed)
    # ref_embed is (batch, seq, d) = (1, 327, 192), embed_out is (seq, batch, d) = (327, 1, 192)
    embed_out_compare = mx.transpose(embed_out, (1, 0, 2))
    diff_embed = mx.abs(embed_out_compare - ref_embed)
    mx.eval(diff_embed)
    print(f"  Embed max diff from reference: {float(diff_embed.max()):.6f}")

    # Run full encoder
    encoder_out, _ = model(input_fbank)  # Returns (output, output_lengths)
    mx.eval(encoder_out)
    print(f"\nEncoder output shape: {encoder_out.shape}")
    print(f"  Min: {float(encoder_out.min()):.6f}")
    print(f"  Max: {float(encoder_out.max()):.6f}")
    print(f"  Mean: {float(encoder_out.mean()):.6f}")

    # Load encoder reference
    encoder_ref_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/encoder_reference.safetensors"
    try:
        encoder_ref = load_file(encoder_ref_path)
        ref_out = mx.array(encoder_ref["encoder_output"])
        mx.eval(ref_out)
        print(f"\nReference output shape: {ref_out.shape}")
        print(f"  Min: {float(ref_out.min()):.6f}")
        print(f"  Max: {float(ref_out.max()):.6f}")
        print(f"  Mean: {float(ref_out.mean()):.6f}")

        # Compare - handle shape differences
        if encoder_out.shape != ref_out.shape:
            print(f"\nShape mismatch: encoder={encoder_out.shape}, ref={ref_out.shape}")
            # Try transposing if needed
            if encoder_out.shape[0] == ref_out.shape[1] and encoder_out.shape[1] == ref_out.shape[0]:
                encoder_out_compare = mx.transpose(encoder_out, (1, 0, 2))
                print("Transposed encoder output for comparison")
            else:
                encoder_out_compare = encoder_out
        else:
            encoder_out_compare = encoder_out

        if encoder_out_compare.shape == ref_out.shape:
            diff = mx.abs(encoder_out_compare - ref_out)
            mx.eval(diff)
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            print("\nDifference from reference:")
            print(f"  Max diff: {max_diff:.6f}")
            print(f"  Mean diff: {mean_diff:.6f}")

            if max_diff < 1e-5:
                print("\nNumerical equivalence achieved!")
            elif max_diff < 1e-3:
                print("\nClose match (within 1e-3)")
            else:
                print("\nSignificant difference")
    except FileNotFoundError:
        print(f"\nReference file not found: {encoder_ref_path}")
        print("Run generate_encoder_reference.py first to create the reference")

    # Also check stage references
    print("\n=== Per-stage comparison ===")
    try:
        stage_ref = load_file("/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/stage_references.safetensors")
        for i in range(6):
            key = f"stage_{i}_output"
            if key in stage_ref:
                ref = mx.array(stage_ref[key])
                mx.eval(ref)
                print(f"Stage {i} reference: shape={ref.shape}, min={float(ref.min()):.4f}, max={float(ref.max()):.4f}")
    except FileNotFoundError:
        print("Stage reference file not found")


if __name__ == "__main__":
    main()
