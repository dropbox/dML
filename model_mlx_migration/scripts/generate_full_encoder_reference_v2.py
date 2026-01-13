#!/usr/bin/env python3
"""Generate full encoder reference for C++ validation using Python MLX model.

This script uses the Python MLX Zipformer implementation with the correct
config for the wenetspeech-streaming-small checkpoint.
"""

import sys

import numpy as np
import mlx.core as mx
from safetensors.numpy import load_file, save_file
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    checkpoint_dir = repo_root / "checkpoints" / "zipformer" / "en-streaming"
    weights_path = checkpoint_dir / "model.safetensors"

    # Load checkpoint
    weights_np = load_file(str(weights_path))
    weights = {k: mx.array(v) for k, v in weights_np.items()}
    print(f"Loaded {len(weights)} weight tensors")

    # Config for wenetspeech-streaming-small checkpoint (verified from weights)
    config = {
        'num_encoder_layers': [2, 2, 2, 2, 2, 2],
        'encoder_dims': [192, 256, 256, 256, 256, 256],
        'attention_dims': [128, 128, 128, 256, 128, 128],
        'num_heads': [4, 4, 4, 8, 4, 4],
        'ff1_dims': [384, 576, 576, 576, 576, 576],
        'ff2_dims': [512, 768, 768, 768, 768, 768],
        'ff3_dims': [640, 960, 960, 960, 960, 960],
        'downsampling_factors': [1, 2, 4, 8, 4, 2],
        'cnn_module_kernels': [31, 31, 15, 15, 15, 31],
        'pos_dim': 48,
        'pos_head_dim': 4,
        'value_head_dim': 12,
    }

    output_dim = max(config['encoder_dims'])
    print(f"Config: {config['num_encoder_layers']} layers per stage")
    print(f"Encoder dims: {config['encoder_dims']}")
    print(f"Output dim: {output_dim}")

    # Try to load input fbank from existing reference
    embed_ref_path = checkpoint_dir / "embed_reference.safetensors"
    if embed_ref_path.exists():
        embed_ref = load_file(str(embed_ref_path))
        if "input_fbank" in embed_ref:
            input_fbank = mx.array(embed_ref["input_fbank"])
            print(f"Loaded input fbank: {input_fbank.shape}")
        else:
            # Generate random input
            mx.random.seed(42)
            input_fbank = mx.random.normal((1, 661, 80)) * 0.1
            print(f"Generated random input fbank: {input_fbank.shape}")
    else:
        mx.random.seed(42)
        input_fbank = mx.random.normal((1, 661, 80)) * 0.1
        print(f"Generated random input fbank: {input_fbank.shape}")

    # Import the Python MLX Zipformer model
    try:
        from models.zipformer.zipformer import Zipformer, ZipformerConfig, load_zipformer_weights

        # Create config with correct values
        mlx_config = ZipformerConfig(
            num_features=80,
            num_encoder_layers=tuple(config['num_encoder_layers']),
            encoder_dims=tuple(config['encoder_dims']),
            attention_dims=tuple(config['attention_dims']),
            feedforward_dims=tuple(config['ff1_dims']),  # legacy; see ff{1,2,3}_dims below
            ff1_dims=tuple(config['ff1_dims']),
            ff2_dims=tuple(config['ff2_dims']),
            ff3_dims=tuple(config['ff3_dims']),
            num_heads=tuple(config['num_heads']),
            downsampling_factors=tuple(config['downsampling_factors']),
            cnn_module_kernels=tuple(config['cnn_module_kernels']),
            pos_dim=config['pos_dim'],
            pos_head_dim=config['pos_head_dim'],
            value_head_dim=config['value_head_dim'],
            causal=True,
        )

        print("\nCreating Python MLX Zipformer model...")
        model = Zipformer(mlx_config)
        load_zipformer_weights(model, weights)
        print("Weights loaded successfully")

        # Run inference
        print(f"\nRunning inference on input: {input_fbank.shape}")
        encoder_out, _ = model(input_fbank)
        mx.eval(encoder_out)

        print(f"Python MLX output shape: {encoder_out.shape}")
        print(f"  min={float(encoder_out.min()):.6f}")
        print(f"  max={float(encoder_out.max()):.6f}")
        print(f"  mean={float(encoder_out.mean()):.6f}")

        # Save reference
        output_data = {
            "input_fbank": np.array(input_fbank).astype(np.float32),
            "expected_encoder_out": np.array(encoder_out).astype(np.float32),
        }

        output_path = checkpoint_dir / "full_encoder_reference.safetensors"
        save_file(output_data, str(output_path))
        print(f"\nSaved reference to: {output_path}")
        print(f"  input_fbank: {output_data['input_fbank'].shape}")
        print(f"  expected_encoder_out: {output_data['expected_encoder_out'].shape}")

    except Exception as e:
        print(f"\nError using Python MLX model: {e}")
        raise


if __name__ == "__main__":
    main()
