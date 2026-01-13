#!/usr/bin/env python3
"""Test CosyVoice2 component export feasibility.

This script tests whether the Flow and HiFi-GAN components can be exported
to TorchScript. The LLM component is known to be non-exportable.
"""

import sys
import os

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'cosyvoice_repo'))

import torch
import torchaudio
from pathlib import Path

MODEL_DIR = os.path.join(PROJECT_DIR, 'models/cosyvoice/CosyVoice2-0.5B')
OUTPUT_DIR = Path(os.path.join(PROJECT_DIR, 'models/cosyvoice/exported'))


def test_hift_export():
    """Test HiFi-GAN vocoder export."""
    print("\n" + "=" * 60)
    print("Testing HiFi-GAN Export")
    print("=" * 60)

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        print("Loading CosyVoice2 model...")
        model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, load_vllm=False)
        hift = model.model.hift
        hift.eval()

        print(f"HiFi-GAN type: {type(hift)}")

        # Try to trace the inference method
        print("\nAttempting TorchScript trace...")
        example_mel = torch.randn(1, 80, 100)
        example_source = torch.zeros(1, 1, 0)

        # First test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            output, source = hift.inference(example_mel, example_source)
        print(f"Output shape: {output.shape}")

        # Try scripting instead of tracing (more robust)
        print("\nAttempting torch.jit.script...")
        try:
            scripted_hift = torch.jit.script(hift)
            print("SUCCESS: HiFi-GAN scripted!")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            scripted_hift.save(str(OUTPUT_DIR / 'hift_scripted.pt'))
            print(f"Saved to: {OUTPUT_DIR / 'hift_scripted.pt'}")
            return True
        except Exception as e:
            print(f"Script failed: {e}")

        # Try tracing as fallback
        print("\nAttempting torch.jit.trace...")
        try:
            # Create wrapper for tracing
            class HiFTWrapper(torch.nn.Module):
                def __init__(self, hift):
                    super().__init__()
                    self.hift = hift

                def forward(self, mel):
                    cache_source = torch.zeros(1, 1, 0, device=mel.device)
                    speech, _ = self.hift.inference(mel, cache_source)
                    return speech

            wrapper = HiFTWrapper(hift)
            traced = torch.jit.trace(wrapper, (example_mel,))
            print("SUCCESS: HiFi-GAN traced!")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            traced.save(str(OUTPUT_DIR / 'hift_traced.pt'))
            print(f"Saved to: {OUTPUT_DIR / 'hift_traced.pt'}")
            return True
        except Exception as e:
            print(f"Trace failed: {e}")

        return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flow_encoder_export():
    """Test Flow encoder export."""
    print("\n" + "=" * 60)
    print("Testing Flow Encoder Export")
    print("=" * 60)

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        print("Loading CosyVoice2 model...")
        model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, load_vllm=False)
        flow = model.model.flow
        encoder = flow.encoder
        encoder.eval()

        print(f"Flow encoder type: {type(encoder)}")

        # Try to trace the encoder
        print("\nAttempting TorchScript trace...")
        # The encoder expects embedded tokens, not raw tokens
        example_input = torch.randn(1, 100, flow.input_size)  # [batch, seq_len, input_size]
        example_len = torch.tensor([100], dtype=torch.int32)

        print("Testing forward pass...")
        with torch.no_grad():
            output, output_len = encoder(example_input, example_len)
        print(f"Output shape: {output.shape}")

        print("\nAttempting torch.jit.trace...")
        try:
            traced = torch.jit.trace(encoder, (example_input, example_len))
            print("SUCCESS: Flow encoder traced!")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            traced.save(str(OUTPUT_DIR / 'flow_encoder_traced.pt'))
            print(f"Saved to: {OUTPUT_DIR / 'flow_encoder_traced.pt'}")
            return True
        except Exception as e:
            print(f"Trace failed: {e}")

        return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("CosyVoice2 TorchScript Export Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model: {MODEL_DIR}")

    results = {}

    # Test HiFi-GAN
    results['hift'] = test_hift_export()

    # Test Flow encoder
    results['flow_encoder'] = test_flow_encoder_export()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for component, success in results.items():
        status = "EXPORTABLE" if success else "BLOCKED"
        print(f"  {component}: {status}")

    if all(results.values()):
        print("\nAll tested components are exportable!")
        print("Note: LLM component was NOT tested (known to be non-exportable)")
    else:
        print("\nSome components failed to export.")


if __name__ == '__main__':
    main()
