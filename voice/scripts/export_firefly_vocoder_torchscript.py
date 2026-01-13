#!/usr/bin/env python3
"""
Export Firefly-GAN Vocoder to TorchScript for C++ Integration

This script exports the FireflyGANVocoder model to TorchScript format
so it can be loaded by libtorch in the C++ TTS pipeline.

Usage:
    python3 scripts/export_firefly_vocoder_torchscript.py --device cpu
    python3 scripts/export_firefly_vocoder_torchscript.py --device mps

Author: Worker #479 (AI)
Date: 2025-12-11
"""

import argparse
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scripts.firefly_gan_vocoder import load_firefly_gan_vocoder, FireflyGANVocoder


def export_vocoder(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    test_export: bool = True,
) -> bool:
    """
    Export FireflyGANVocoder to TorchScript.

    Args:
        checkpoint_path: Path to firefly-gan weights
        output_path: Output path for TorchScript model
        device: Device for export (cpu, mps, cuda)
        test_export: Whether to test the exported model

    Returns:
        True if export successful, False otherwise
    """
    print(f"Loading vocoder from: {checkpoint_path}")
    print(f"Target device: {device}")

    # Load the model
    model = load_firefly_gan_vocoder(checkpoint_path, device=device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create example input for tracing
    # Shape: [batch, num_codebooks, time_frames]
    batch_size = 1
    num_codebooks = 8
    num_frames = 100  # ~4.76 seconds at 21Hz frame rate

    example_input = torch.randint(
        0, 1024,  # codebook_size
        (batch_size, num_codebooks, num_frames),
        dtype=torch.long,
        device=device
    )

    print(f"Example input shape: {example_input.shape}")

    # Test the model before export
    print("\nRunning forward pass before export...")
    with torch.no_grad():
        output = model.decode(example_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Export using torch.jit.trace
    print("\nExporting to TorchScript (tracing)...")

    try:
        # Wrap the decode method for tracing
        class VocoderWrapper(torch.nn.Module):
            def __init__(self, vocoder):
                super().__init__()
                self.vocoder = vocoder

            def forward(self, indices: torch.Tensor) -> torch.Tensor:
                """
                Decode codebook indices to audio.

                Args:
                    indices: [B, 8, T] long tensor of codebook indices (0-1023)

                Returns:
                    [B, 1, T*2048] float tensor of audio samples in [-1, 1]
                """
                return self.vocoder.decode(indices)

        wrapper = VocoderWrapper(model)
        wrapper.eval()

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example_input)

        # Save the traced model
        traced.save(output_path)
        print(f"Saved TorchScript model to: {output_path}")

        # Get file size
        file_size = os.path.getsize(output_path)
        print(f"Model size: {file_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"ERROR: TorchScript trace failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test the exported model
    if test_export:
        print("\nTesting exported model...")
        try:
            loaded = torch.jit.load(output_path, map_location=device)
            loaded.eval()

            with torch.no_grad():
                output_loaded = loaded(example_input)

            print(f"Loaded model output shape: {output_loaded.shape}")
            print(f"Output range: [{output_loaded.min():.4f}, {output_loaded.max():.4f}]")

            # Compare outputs
            diff = (output - output_loaded).abs().max().item()
            print(f"Max difference vs original: {diff:.6e}")

            if diff < 1e-5:
                print("PASS: Outputs match")
            else:
                print(f"WARNING: Outputs differ by {diff:.6e}")

        except Exception as e:
            print(f"ERROR: Failed to load exported model: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test with different input sizes
    print("\nTesting with different input sizes...")
    test_sizes = [10, 50, 200]

    for n_frames in test_sizes:
        test_input = torch.randint(
            0, 1024,
            (1, 8, n_frames),
            dtype=torch.long,
            device=device
        )
        try:
            with torch.no_grad():
                test_output = loaded(test_input)
            expected_samples = n_frames * 4 * 512  # 4x quantizer, 512x head
            actual_samples = test_output.shape[-1]
            status = "OK" if actual_samples == expected_samples else "MISMATCH"
            print(f"  {n_frames} frames -> {actual_samples} samples ({status})")
        except Exception as e:
            print(f"  {n_frames} frames -> ERROR: {e}")
            return False

    print("\nExport successful!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export Firefly-GAN Vocoder to TorchScript"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        help="Path to vocoder checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for TorchScript model (default: models/fish-speech-1.5/vocoder_{device}.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Device for export",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip testing the exported model",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        output_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(output_dir, f"vocoder_{args.device}.pt")

    # Check device availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("ERROR: MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Run export
    success = export_vocoder(
        args.checkpoint,
        args.output,
        args.device,
        test_export=not args.no_test,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
