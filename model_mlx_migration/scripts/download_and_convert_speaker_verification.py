#!/usr/bin/env python3
"""Download and Convert ECAPA-TDNN for Speaker Verification to MLX.

Phase 10.1: Speaker verification model needed for speaker adaptation.

Downloads from: speechbrain/spkrec-ecapa-voxceleb
This is trained on VoxCeleb1+2 for speaker verification (NOT language ID).

Usage:
    python scripts/download_and_convert_speaker_verification.py

Output:
    models/ecapa-spkver/        - PyTorch checkpoint
    models/ecapa-spkver-mlx/    - MLX weights
"""

import json
from pathlib import Path

import numpy as np


def download_speechbrain_model():
    """Download the SpeechBrain speaker verification model using HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    output_dir = Path("models/ecapa-spkver")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading SpeechBrain spkrec-ecapa-voxceleb from HuggingFace Hub...")
    # Download all files from the repo
    snapshot_download(
        repo_id="speechbrain/spkrec-ecapa-voxceleb",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Model downloaded to {output_dir}")
    return output_dir, None


def convert_conv1d_weight(weight) -> np.ndarray:
    """Convert PyTorch Conv1d weight to MLX format.

    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    MLX conv1d: (out_channels, kernel_size, in_channels)
    """
    import torch
    if isinstance(weight, torch.Tensor):
        # Transpose from (O, I, K) to (O, K, I)
        return weight.permute(0, 2, 1).numpy()
    return weight


def map_embedding_key(pt_key: str) -> str:
    """Map SpeechBrain embedding model key to MLX model key."""
    key = pt_key

    # Handle top-level blocks.N -> blocks_N
    for i in range(4):
        if key.startswith(f"blocks.{i}."):
            key = f"blocks_{i}." + key[len(f"blocks.{i}."):]

    # Handle se_block specially
    key = key.replace(".se_block.conv1.conv.", ".se_block.conv1.")
    key = key.replace(".se_block.conv2.conv.", ".se_block.conv2.")

    # Remove redundant .conv
    key = key.replace(".conv.conv.", ".conv.")

    # Handle norm paths
    if "asp_bn.norm." in key:
        key = key.replace("asp_bn.norm.", "asp_bn.")
    else:
        key = key.replace(".norm.norm.", ".norm.")

    # Handle fc.conv -> fc
    key = key.replace("fc.conv.", "fc.")

    # Prefix with embedding_model
    key = f"embedding_model.{key}"

    return key


def convert_embedding_model(pytorch_path: Path) -> dict[str, np.ndarray]:
    """Convert embedding model weights."""
    import torch

    ckpt = torch.load(pytorch_path, map_location="cpu", weights_only=False)

    mlx_weights = {}

    for pt_key, tensor in ckpt.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        # Skip num_batches_tracked
        if "num_batches_tracked" in pt_key:
            continue

        mlx_key = map_embedding_key(pt_key)

        # Convert tensor
        if "conv.weight" in pt_key or "conv.conv.weight" in pt_key:
            value = convert_conv1d_weight(tensor)
        else:
            value = tensor.numpy()

        mlx_weights[mlx_key] = value

    return mlx_weights


def main():
    """Main conversion function."""

    # Download model
    input_dir, _ = download_speechbrain_model()
    output_dir = Path("models/ecapa-spkver-mlx")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting ECAPA-TDNN Speaker Verification from {input_dir} to {output_dir}")

    # Check for embedding model checkpoint
    embedding_path = input_dir / "embedding_model.ckpt"
    if not embedding_path.exists():
        # Try alternate location
        for f in input_dir.glob("*.ckpt"):
            if "embedding" in f.name.lower():
                embedding_path = f
                break

    if not embedding_path.exists():
        print(f"ERROR: Could not find embedding_model.ckpt in {input_dir}")
        print("Available files:", list(input_dir.glob("*")))
        return

    # Convert embedding model
    print(f"\nConverting embedding model from {embedding_path}...")
    mlx_weights = convert_embedding_model(embedding_path)
    print(f"  Converted {len(mlx_weights)} parameters")

    # Save weights
    weights_path = output_dir / "weights.npz"
    np.savez(str(weights_path), **mlx_weights)
    print(f"Saved weights to {weights_path}")

    # Save config - ECAPA-TDNN for speaker verification uses 192-dim embeddings
    # (some checkpoints use 192, some use 256)
    # Let's introspect the model to get the right config
    try:
        # Try to get actual dimensions from model
        fc_weight_key = next((k for k in mlx_weights if "fc.weight" in k), None)
        if fc_weight_key:
            lin_neurons = mlx_weights[fc_weight_key].shape[0]
        else:
            lin_neurons = 192  # Default for VoxCeleb
    except Exception:
        lin_neurons = 192

    config = {
        "n_mels": 80,  # VoxCeleb uses 80 mels
        "sample_rate": 16000,
        "channels": [1024, 1024, 1024, 1024, 3072],
        "kernel_sizes": [5, 3, 3, 3, 1],
        "dilations": [1, 2, 3, 4, 1],
        "attention_channels": 128,
        "lin_neurons": lin_neurons,
        "res2net_scale": 8,
        "se_channels": 128,
        "model_type": "speaker_verification",
        "training_data": "VoxCeleb1+VoxCeleb2",
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Print weight statistics
    print("\nWeight statistics:")
    total_params = sum(v.size for v in mlx_weights.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"  Embedding dimension: {lin_neurons}")

    # Test the model
    print("\nTesting model loading...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import mlx.core as mx

        from tools.whisper_mlx.sota.ecapa_config import ECAPATDNNConfig
        from tools.whisper_mlx.sota.ecapa_tdnn import ECAPATDNN

        # Create config for speaker verification
        spk_config = ECAPATDNNConfig(
            n_mels=config["n_mels"],
            lin_neurons=config["lin_neurons"],
            num_languages=1,  # Not used for verification
        )

        # Load model
        test_model = ECAPATDNN(spk_config)
        weights = mx.load(str(weights_path))

        # Filter to just embedding_model weights
        emb_weights = {k.replace("embedding_model.", ""): v
                       for k, v in weights.items()
                       if k.startswith("embedding_model.")}
        test_model.load_weights(list(emb_weights.items()))

        # Test inference
        test_input = mx.random.normal((1, 100, config["n_mels"]))
        output = test_model(test_input)
        mx.eval(output)

        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dim: {output.shape[-1]} (expected: {lin_neurons})")
        print("  Model loaded and inference successful!")

    except Exception as e:
        print(f"  Warning: Could not test model: {e}")
        import traceback
        traceback.print_exc()

    print("\nConversion complete!")
    print(f"\nModel ready at: {output_dir}")
    print("Use for speaker verification embeddings in Phase 10 speaker adaptation.")


if __name__ == "__main__":
    main()
