#!/usr/bin/env python3
"""Validate BEATs MLX conversion against PyTorch.

Compares outputs from PyTorch (SpeechBrain) and MLX implementations
to verify numerical equivalence.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import mlx.core as mx
import torchaudio.compliance.kaldi as ta_kaldi

from whisper_mlx.sota.beats_mlx import BEATsModel


def create_fbank_pytorch(waveform: torch.Tensor, fbank_mean: float = 15.41663, fbank_std: float = 6.55582):
    """Create normalized fbank features matching BEATs preprocessing."""
    fbanks = []
    for wav in waveform:
        wav = wav.unsqueeze(0) * 2**15
        fbank = ta_kaldi.fbank(
            wav,
            num_mel_bins=128,
            sample_frequency=16000,
            frame_length=25,
            frame_shift=10,
        )
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    return (fbank - fbank_mean) / (2 * fbank_std)


def load_pytorch_beats(checkpoint_path: str):
    """Load PyTorch BEATs model from checkpoint."""
    from speechbrain.lobes.models.beats import BEATs

    model = BEATs(ckp_path=checkpoint_path, freeze=True, output_all_hiddens=False)
    model.eval()
    return model


def validate_layer_by_layer(
    pt_model,
    mlx_model,
    fbank_pt: torch.Tensor,
    fbank_mlx: mx.array,
):
    """Layer-by-layer validation for debugging."""
    print("\n=== Layer-by-Layer Validation ===\n")

    # 1. Patch embedding
    print("1. Patch Embedding")
    with torch.no_grad():
        fbank_4d = fbank_pt.unsqueeze(1)  # (B, 1, F, T)
        pt_patches = pt_model.patch_embedding(fbank_4d)
        pt_patches = pt_patches.reshape(pt_patches.shape[0], pt_patches.shape[1], -1).transpose(1, 2)

    # MLX patch embedding - (B, F, T, 1) input format
    fbank_mlx_4d = fbank_mlx[:, :, :, None]  # Add channel at end
    mlx_patches = mlx_model.patch_embedding(fbank_mlx_4d)
    mx.eval(mlx_patches)

    pt_patches_np = pt_patches.numpy()
    mlx_patches_np = np.array(mlx_patches)

    diff = np.abs(pt_patches_np - mlx_patches_np)
    print(f"   PT shape: {pt_patches_np.shape}, MLX shape: {mlx_patches_np.shape}")
    print(f"   Max diff: {diff.max():.6e}, Mean diff: {diff.mean():.6e}")

    if diff.max() > 0.01:
        print("   Sample values (first patch, first 5 dims):")
        print(f"   PT:  {pt_patches_np[0, 0, :5]}")
        print(f"   MLX: {mlx_patches_np[0, 0, :5]}")

    # 2. Layer norm after patch embedding
    print("\n2. Layer Norm (patch)")
    with torch.no_grad():
        pt_ln = pt_model.layer_norm(pt_patches)

    mlx_ln = mlx_model.layer_norm(mlx_patches)
    mx.eval(mlx_ln)

    pt_ln_np = pt_ln.numpy()
    mlx_ln_np = np.array(mlx_ln)

    diff = np.abs(pt_ln_np - mlx_ln_np)
    print(f"   Max diff: {diff.max():.6e}, Mean diff: {diff.mean():.6e}")

    # 3. Post extract projection (512 -> 768)
    print("\n3. Post Extract Projection (512->768)")
    with torch.no_grad():
        if pt_model.post_extract_proj is not None:
            pt_proj = pt_model.post_extract_proj(pt_ln)
        else:
            pt_proj = pt_ln

    mlx_proj = mlx_model.post_extract_proj(mlx_ln)
    mx.eval(mlx_proj)

    pt_proj_np = pt_proj.numpy()
    mlx_proj_np = np.array(mlx_proj)

    diff = np.abs(pt_proj_np - mlx_proj_np)
    print(f"   Max diff: {diff.max():.6e}, Mean diff: {diff.mean():.6e}")

    # 4. Positional conv embedding
    print("\n4. Positional Conv Embedding")
    with torch.no_grad():
        # SpeechBrain BEATs uses pos_conv (a Sequential)
        pt_pos = pt_model.encoder.pos_conv(pt_proj.transpose(1, 2))
        pt_pos = pt_pos.transpose(1, 2)
        pt_x = pt_proj + pt_pos

    mlx_pos = mlx_model.encoder.pos_conv(mlx_proj)
    mlx_x = mlx_proj + mlx_pos
    mx.eval(mlx_x)

    diff = np.abs(pt_x.numpy() - np.array(mlx_x))
    print(f"   Max diff: {diff.max():.6e}, Mean diff: {diff.mean():.6e}")

    return diff.max()


def main():
    pt_checkpoint = "models/sota/beats/models/sota/beats/BEATs_iter3_plus_AS2M.pt"
    mlx_checkpoint = "models/sota/beats-mlx"

    print("=" * 60)
    print("BEATs MLX Validation")
    print("=" * 60)

    # Load PyTorch model
    print("\n1. Loading PyTorch model...")
    pt_model = load_pytorch_beats(pt_checkpoint)
    print(f"   Config: {pt_model.cfg.__dict__}")

    # Load MLX model
    print("\n2. Loading MLX model...")
    mlx_model = BEATsModel.from_pretrained(mlx_checkpoint)
    mx.eval(mlx_model.parameters())
    print(f"   Config: embed_dim={mlx_model.config.embed_dim}, encoder_embed_dim={mlx_model.config.encoder_embed_dim}")

    # Create test input (1 second of audio at 16kHz)
    print("\n3. Creating test input...")
    np.random.seed(42)
    test_audio = np.random.randn(1, 16000).astype(np.float32)
    test_audio_pt = torch.from_numpy(test_audio)

    # Create fbank features
    fbank_pt = create_fbank_pytorch(test_audio_pt)
    fbank_np = fbank_pt.numpy()
    fbank_mlx = mx.array(fbank_np)

    print(f"   Fbank shape: {fbank_pt.shape} (batch, time, freq)")

    # Run layer-by-layer validation first
    max_intermediate_diff = validate_layer_by_layer(
        pt_model, mlx_model, fbank_pt, fbank_mlx
    )

    # Full model comparison
    print("\n" + "=" * 60)
    print("Full Model Forward Pass")
    print("=" * 60)

    # PyTorch forward (using extract_features)
    print("\n4. Running PyTorch forward...")
    with torch.no_grad():
        # Need to provide wav_lens (relative lengths) to avoid unbound variable bug
        wav_lens = torch.ones(test_audio_pt.shape[0])
        pt_out = pt_model.extract_features(test_audio_pt, wav_lens=wav_lens)
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
    pt_out_np = pt_out.numpy()
    print(f"   PT output shape: {pt_out_np.shape}")

    # MLX forward
    print("\n5. Running MLX forward...")
    # For MLX, we pass fbank directly
    mlx_out = mlx_model(fbank_mlx)
    mx.eval(mlx_out)
    mlx_out_np = np.array(mlx_out)
    print(f"   MLX output shape: {mlx_out_np.shape}")

    # Compare outputs
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    if pt_out_np.shape != mlx_out_np.shape:
        print("\nERROR: Shape mismatch!")
        print(f"  PyTorch: {pt_out_np.shape}")
        print(f"  MLX: {mlx_out_np.shape}")
        return 1

    abs_diff = np.abs(pt_out_np - mlx_out_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    rel_diff = abs_diff / (np.abs(pt_out_np) + 1e-8)
    max_rel_diff = rel_diff.max()

    print("\nAbsolute Differences:")
    print(f"  Max:  {max_diff:.6e}")
    print(f"  Mean: {mean_diff:.6e}")
    print("\nRelative Differences:")
    print(f"  Max:  {max_rel_diff:.6e}")

    # Check a few sample values
    print("\nSample values (position [0, 0, :5]):")
    print(f"  PyTorch: {pt_out_np[0, 0, :5]}")
    print(f"  MLX:     {mlx_out_np[0, 0, :5]}")

    # Pass/fail
    threshold = 1e-5
    if max_diff < threshold:
        print(f"\n✓ PASS: max_diff={max_diff:.6e} < {threshold}")
        return 0
    else:
        print(f"\n✗ FAIL: max_diff={max_diff:.6e} >= {threshold}")

        # Additional debugging
        print("\nDebugging info:")
        print(f"  Intermediate max diff: {max_intermediate_diff:.6e}")

        # Find where largest differences are
        idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        print(f"  Largest diff at index: {idx}")
        print(f"  PT value: {pt_out_np[idx]:.6f}")
        print(f"  MLX value: {mlx_out_np[idx]:.6f}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
