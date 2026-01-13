#!/usr/bin/env python3
"""Verify exported CosyVoice2 components work correctly."""

import sys
import os

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'cosyvoice_repo'))

import torch
import numpy as np
from pathlib import Path

EXPORT_DIR = Path(os.path.join(PROJECT_DIR, 'models/cosyvoice/exported'))


def verify_hift():
    """Verify HiFi-GAN export."""
    print("\n" + "=" * 60)
    print("Verifying HiFi-GAN Export")
    print("=" * 60)

    try:
        # Load exported model
        print("Loading exported HiFi-GAN...")
        hift = torch.jit.load(str(EXPORT_DIR / 'hift_traced.pt'))
        print(f"Loaded: {type(hift)}")

        # Test inference
        example_mel = torch.randn(1, 80, 100)
        print(f"Input shape: {example_mel.shape}")

        with torch.no_grad():
            output = hift(example_mel)
        print(f"Output shape: {output.shape}")

        # Verify output is reasonable
        assert output.dim() == 2, f"Expected 2D output, got {output.dim()}D"
        assert output.shape[0] == 1, f"Expected batch=1, got {output.shape[0]}"
        assert output.shape[1] > 0, "Expected non-empty output"

        # Check output values are reasonable (audio range)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        print("SUCCESS: HiFi-GAN export verified!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_flow_encoder():
    """Verify Flow encoder export."""
    print("\n" + "=" * 60)
    print("Verifying Flow Encoder Export")
    print("=" * 60)

    try:
        # Load exported model
        print("Loading exported Flow encoder...")
        encoder = torch.jit.load(str(EXPORT_DIR / 'flow_encoder_traced.pt'))
        print(f"Loaded: {type(encoder)}")

        # Test inference - note: traced with input_size=512
        example_input = torch.randn(1, 100, 512)
        example_len = torch.tensor([100], dtype=torch.int32)
        print(f"Input shape: {example_input.shape}")

        with torch.no_grad():
            output, output_len = encoder(example_input, example_len)
        print(f"Output shape: {output.shape}")
        print(f"Output len: {output_len}")

        # Verify output is reasonable
        assert output.dim() == 3, f"Expected 3D output, got {output.dim()}D"
        assert output.shape[0] == 1, f"Expected batch=1, got {output.shape[0]}"
        assert output.shape[1] > 0, "Expected non-empty sequence"

        # Check output values are reasonable
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        print("SUCCESS: Flow encoder export verified!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("CosyVoice2 Export Verification")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Export directory: {EXPORT_DIR}")

    results = {}
    results['hift'] = verify_hift()
    results['flow_encoder'] = verify_flow_encoder()

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for component, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {component}: {status}")

    if all(results.values()):
        print("\nAll exports verified successfully!")
        return 0
    else:
        print("\nSome exports failed verification.")
        return 1


if __name__ == '__main__':
    exit(main())
