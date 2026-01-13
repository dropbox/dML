#!/usr/bin/env python3
"""
Layer-by-layer comparison between PyTorch (icefall) and MLX Zipformer.

This script identifies where numerical divergence begins.

Usage:
    python scripts/compare_pytorch_mlx.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "tools/third_party/icefall/egs/librispeech/ASR/zipformer"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools/third_party/icefall"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import mlx.core as mx

# MLX imports
from src.models.zipformer.asr_model import ASRModelConfig, load_checkpoint
from src.models.zipformer.features import FbankExtractor, FbankConfig, load_audio


def compare_arrays(torch_arr, mlx_arr, name: str, verbose: bool = True):
    """Compare PyTorch and MLX arrays and report statistics."""
    if isinstance(torch_arr, torch.Tensor):
        torch_np = torch_arr.detach().cpu().numpy()
    else:
        torch_np = np.array(torch_arr)

    mlx_np = np.array(mlx_arr)

    if torch_np.shape != mlx_np.shape:
        print(f"{name}: SHAPE MISMATCH torch={torch_np.shape} mlx={mlx_np.shape}")
        return False

    abs_diff = np.abs(torch_np - mlx_np)
    max_abs_error = float(np.max(abs_diff))
    mean_abs_error = float(np.mean(abs_diff))

    # Correlation
    corr = np.corrcoef(torch_np.flatten(), mlx_np.flatten())[0, 1]

    if verbose:
        print(f"{name}:")
        print(f"  shape: {torch_np.shape}")
        print(f"  max_abs_error: {max_abs_error:.6e}")
        print(f"  mean_abs_error: {mean_abs_error:.6e}")
        print(f"  correlation: {corr:.6f}")

        if max_abs_error < 1e-5:
            print("  [PASS]")
        elif max_abs_error < 1e-3:
            print("  [CLOSE - needs investigation]")
        else:
            print("  [FAIL - significant divergence]")

    return max_abs_error < 1e-4


def load_pytorch_model(checkpoint_path: str):
    """Load PyTorch Zipformer model using icefall."""
    print("Loading PyTorch model from icefall...")

    # Import icefall modules
    try:
        from zipformer import Zipformer2
        from scaling import convert_num_channels
    except ImportError:
        print("ERROR: Could not import icefall modules. Check PYTHONPATH.")
        return None

    # Create model with matching config
    model = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=(1, 2, 4, 8, 4, 2),
        encoder_dim=(192, 256, 256, 256, 256, 192),
        num_encoder_layers=(2, 2, 2, 2, 2, 2),
        encoder_unmasked_dim=(192, 192, 192, 192, 192, 192),
        query_head_dim=32,
        pos_head_dim=4,
        value_head_dim=12,
        num_heads=(4, 4, 4, 8, 4, 4),
        feedforward_dim=(384, 512, 512, 512, 512, 384),
        cnn_module_kernel=(31, 31, 15, 15, 15, 31),
        pos_dim=48,
        dropout=0.0,
        causal=True,
        chunk_size=[16],
        left_context_frames=[128],
    )
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # Filter to encoder keys only
    encoder_keys = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder_state = {k.replace("encoder.", ""): v for k, v in encoder_keys.items()}

    # Load weights
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print(f"  Loaded {len(encoder_state) - len(missing)} weights")
    print(f"  Missing: {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")

    return model


def load_mlx_model(checkpoint_path: str):
    """Load MLX ASR model."""
    print("Loading MLX model...")

    config = ASRModelConfig(
        d_model=192,
        num_encoder_layers=(2, 2, 2, 2, 2, 2),
        encoder_dims=(192, 256, 256, 256, 256, 192),
        attention_dims=(128, 128, 128, 256, 128, 128),
        feedforward_dims=(384, 512, 512, 512, 512, 384),
        num_heads=(4, 4, 4, 8, 4, 4),
        downsampling_factors=(1, 2, 4, 8, 4, 2),
        cnn_module_kernels=(31, 31, 15, 15, 15, 31),
        pos_dim=48,
        pos_head_dim=4,
        value_head_dim=12,
        dropout=0.0,
        causal=True,
        vocab_size=500,
        decoder_dim=512,
        blank_id=0,
        context_size=2,
        joiner_dim=512,
    )

    model = load_checkpoint(checkpoint_path, config)
    print("  Loaded MLX model")

    return model


def create_test_input(audio_path: str):
    """Create fbank features from audio file."""
    print(f"Loading audio: {audio_path}")

    # Load audio
    audio = load_audio(audio_path, sample_rate=16000)

    # Extract fbank features
    fbank_config = FbankConfig(
        sample_rate=16000,
        num_mel_bins=80,
    )
    extractor = FbankExtractor(fbank_config)
    features = extractor(mx.array(audio))
    mx.eval(features)

    print(f"  Features shape: {features.shape}")

    # Convert to PyTorch tensor
    features_np = np.array(features)
    features_torch = torch.tensor(features_np)

    return features_torch, features


def test_encoder_embed(pytorch_model, mlx_model, features_torch, features_mlx):
    """Test encoder_embed (Conv2dSubsampling) output."""
    print("\n" + "="*60)
    print("Testing encoder_embed (Conv2dSubsampling)")
    print("="*60)

    # PyTorch forward
    # encoder_embed expects (batch, seq, feat)
    x_torch = features_torch.unsqueeze(0).permute(0, 2, 1)  # (1, 80, T)

    # Run through encoder_embed
    with torch.no_grad():
        # Need to handle the full forward pass since icefall's encoder_embed
        # is integrated into the forward method
        # For now, just test full encoder output
        pass

    # MLX forward
    x_mlx = features_mlx[None, :, :]  # Add batch dim

    # Test full encoder output first
    print("\nComparing full encoder output...")

    # PyTorch full encoder
    with torch.no_grad():
        x_torch_input = features_torch.unsqueeze(0)  # (1, T, 80)
        x_lens = torch.tensor([x_torch_input.shape[1]])

        try:
            pytorch_out, _ = pytorch_model(x_torch_input, x_lens)
            pytorch_out = pytorch_out.squeeze(0)  # Remove batch
            print(f"PyTorch encoder output: {pytorch_out.shape}")
        except Exception as e:
            print(f"PyTorch forward failed: {e}")
            return False

    # MLX full encoder
    try:
        mlx_out = mlx_model.encoder(x_mlx)
        mx.eval(mlx_out)
        print(f"MLX encoder output: {mlx_out.shape}")
    except Exception as e:
        print(f"MLX forward failed: {e}")
        return False

    # Compare
    compare_arrays(pytorch_out, mlx_out[0], "encoder_output")

    return True


def main():
    print("="*60)
    print("PyTorch vs MLX Zipformer Comparison")
    print("="*60)

    # Paths
    checkpoint_path = "checkpoints/zipformer/en-streaming/exp/pretrained.pt"
    audio_path = "checkpoints/zipformer/en-streaming/test_wavs/1089-134686-0001.wav"

    # Verify files exist
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    if not Path(audio_path).exists():
        print(f"ERROR: Audio not found: {audio_path}")
        return 1

    # Load models
    pytorch_model = load_pytorch_model(checkpoint_path)
    if pytorch_model is None:
        print("Failed to load PyTorch model")
        return 1

    mlx_model = load_mlx_model(checkpoint_path)

    # Create test input
    features_torch, features_mlx = create_test_input(audio_path)

    # Test encoder
    test_encoder_embed(pytorch_model, mlx_model, features_torch, features_mlx)

    print("\n" + "="*60)
    print("Comparison complete")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
