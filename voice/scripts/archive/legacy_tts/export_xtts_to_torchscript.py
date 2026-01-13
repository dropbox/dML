#!/usr/bin/env python3
"""
Export XTTS v2 model to TorchScript format for C++ inference.

This script loads the XTTS v2 PyTorch checkpoint and exports it to TorchScript,
which can be loaded by libtorch in the C++ application.
"""

import torch
import os
import sys
import json
from pathlib import Path

def export_xtts_to_torchscript():
    """Export XTTS v2 model to TorchScript format."""

    # Model paths
    model_dir = Path.home() / "Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"
    output_path = Path("/Users/ayates/voice/stream-tts-cpp/models/xtts_v2.pt")

    print(f"Loading XTTS v2 from: {model_dir}")

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    # Check required files
    config_file = model_dir / "config.json"
    model_file = model_dir / "model.pth"

    if not config_file.exists():
        print(f"ERROR: config.json not found in {model_dir}")
        sys.exit(1)

    if not model_file.exists():
        print(f"ERROR: model.pth not found in {model_dir}")
        sys.exit(1)

    print(f"Found config: {config_file}")
    print(f"Found model: {model_file}")

    try:
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Model config loaded: {config.get('model', 'unknown')}")

        # Try importing TTS library
        try:
            from TTS.tts.models.xtts import Xtts
            print("TTS library found, attempting proper model loading...")

            # Load model using TTS library
            model = Xtts.init_from_config(str(config_file))
            model.load_checkpoint(config, str(model_dir), eval=True, use_deepspeed=False)
            model.eval()

            print("Model loaded successfully via TTS library")

        except ImportError:
            print("TTS library not available, loading checkpoint directly...")

            # Load checkpoint directly
            checkpoint = torch.load(model_file, map_location='cpu')
            print(f"Checkpoint loaded, keys: {list(checkpoint.keys())[:5]}...")

            # This is a fallback - the model architecture needs to be defined
            # For actual inference, we'll need the TTS library
            print("WARNING: Direct checkpoint loading requires model architecture definition")
            print("Recommend installing TTS library: pip install TTS")
            sys.exit(1)

        # Try to export to TorchScript
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\nAttempting TorchScript export (torch.jit.script)...")
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(output_path))
            print(f"SUCCESS! Model exported to: {output_path}")
            print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
            return True

        except Exception as e:
            print(f"Script export failed: {e}")
            print("\nAttempting TorchScript trace with dummy input...")

            try:
                # Create dummy input for tracing
                dummy_text = "Hello world"
                dummy_speaker_wav = torch.randn(1, 1, 22050)  # 1 second at 22050 Hz
                dummy_language = "en"

                # This is simplified - actual XTTS input is more complex
                print("WARNING: Tracing may not capture all model behavior")
                print("XTTS models are complex and may not trace correctly")

                traced_model = torch.jit.trace(
                    model,
                    (dummy_text, dummy_speaker_wav, dummy_language)
                )
                traced_model.save(str(output_path))
                print(f"SUCCESS! Model traced to: {output_path}")
                print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
                return True

            except Exception as e2:
                print(f"Trace export also failed: {e2}")
                print("\nTorchScript export not supported for this model architecture.")
                print("RECOMMENDATION: Use Python subprocess approach instead.")
                return False

    except Exception as e:
        print(f"ERROR during export: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_export(output_path):
    """Verify the exported model can be loaded."""
    try:
        print(f"\nVerifying export at: {output_path}")
        model = torch.jit.load(output_path)
        print("Verification SUCCESS! Model loads correctly.")
        return True
    except Exception as e:
        print(f"Verification FAILED: {e}")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("XTTS v2 TorchScript Export Tool")
    print("=" * 80)
    print()

    success = export_xtts_to_torchscript()

    if success:
        output_path = Path("/Users/ayates/voice/stream-tts-cpp/models/xtts_v2.pt")
        verify_export(output_path)
    else:
        print("\n" + "=" * 80)
        print("ALTERNATIVE APPROACH: Python Subprocess")
        print("=" * 80)
        print("Since TorchScript export is not supported, use Python worker approach:")
        print("1. Keep XTTS in Python worker (like tts_worker_gtts.py)")
        print("2. Call Python from C++ via subprocess (same pattern as current system)")
        print("3. Pipe text to Python, receive audio bytes back")
        print()
        print("This is the proven approach already working in production system.")
        sys.exit(1)
