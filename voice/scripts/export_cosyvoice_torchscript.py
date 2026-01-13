#!/usr/bin/env python3
"""Export CosyVoice2 models to TorchScript for C++ inference.

This script creates wrapper modules with clean tensor-only interfaces
that can be traced/scripted and loaded in C++ via libtorch.

Components:
1. HiFT (vocoder) - converts mel spectrogram to audio
2. Flow decoder - converts embeddings to mel spectrogram
3. LLM - token generation (blocked by HuggingFace transformers)

Usage:
    python3 scripts/export_cosyvoice_torchscript.py [--cpu] [--test]
"""

import sys
import os
import argparse
import time
import torch
import torch.nn as nn

# Patch CUDA before importing CosyVoice
torch.cuda.is_available = lambda: False

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'cosyvoice_repo'))
from cosyvoice.cli.cosyvoice import CosyVoice2

MODEL_DIR = os.path.join(PROJECT_DIR, 'models/cosyvoice/CosyVoice2-0.5B')
EXPORT_DIR = os.path.join(PROJECT_DIR, 'models/cosyvoice/torchscript')


class HiFTWrapper(nn.Module):
    """Wrapper for HiFT vocoder with clean tensor interface.

    The original HiFT.decode takes (mel, source) tensors.
    This wrapper exposes that interface directly.
    """
    def __init__(self, hift):
        super().__init__()
        self.hift = hift

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to audio waveform.

        Args:
            mel: Mel spectrogram tensor (batch, 80, time)

        Returns:
            Audio waveform tensor (batch, samples)
        """
        # Create empty source tensor (no source excitation)
        device = mel.device
        source = torch.zeros(1, 1, 0, device=device)

        # Call decode method which has clean tensor interface
        return self.hift.decode(mel, source)


class HiFTFullWrapper(nn.Module):
    """Full HiFT wrapper that handles f0 prediction internally.

    This matches the forward() behavior of HiFTGenerator.
    """
    def __init__(self, hift):
        super().__init__()
        self.f0_predictor = hift.f0_predictor
        self.f0_upsamp = hift.f0_upsamp
        self.m_source = hift.m_source
        self.decode_fn = hift.decode

    def forward(self, speech_feat: torch.Tensor) -> tuple:
        """Process mel spectrogram through full HiFT pipeline.

        Args:
            speech_feat: Mel features (batch, time, 80) - note: time first!

        Returns:
            Tuple of (audio, f0)
        """
        # Transpose to (batch, 80, time)
        speech_feat = speech_feat.transpose(1, 2)

        # Predict f0
        f0 = self.f0_predictor(speech_feat)

        # Generate source excitation
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)

        # Generate speech
        generated_speech = self.decode_fn(x=speech_feat, s=s)

        return generated_speech, f0


def export_hift(cosyvoice, device, export_dir):
    """Export HiFT vocoder to TorchScript."""
    print('\n=== Exporting HiFT (vocoder) ===')

    hift = cosyvoice.model.hift
    hift.eval()
    hift.to(device)

    # Create wrapper
    wrapper = HiFTWrapper(hift)
    wrapper.eval()

    # Example input for tracing
    example_mel = torch.randn(1, 80, 100, device=device)

    try:
        with torch.no_grad():
            # Trace the wrapper
            traced = torch.jit.trace(wrapper, example_mel, strict=False)

        # Save
        path = os.path.join(export_dir, 'hift_decode.pt')
        traced.save(path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f'  Saved: {path} ({size_mb:.1f} MB)')

        return True
    except Exception as e:
        print(f'  FAILED: {e}')
        return False


def export_hift_full(cosyvoice, device, export_dir):
    """Export full HiFT pipeline (with f0 prediction) to TorchScript."""
    print('\n=== Exporting HiFT Full Pipeline ===')

    hift = cosyvoice.model.hift
    hift.eval()
    hift.to(device)

    # Create full wrapper
    wrapper = HiFTFullWrapper(hift)
    wrapper.eval()
    wrapper.to(device)

    # Example input: (batch, time, 80) - matches forward() expectation
    example_feat = torch.randn(1, 100, 80, device=device)

    try:
        with torch.no_grad():
            # Script instead of trace to handle tuple output
            scripted = torch.jit.script(wrapper)

        # Save
        path = os.path.join(export_dir, 'hift_full.pt')
        scripted.save(path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f'  Saved: {path} ({size_mb:.1f} MB)')

        return True
    except Exception as e:
        print(f'  Script FAILED: {e}')

        # Try tracing instead
        try:
            with torch.no_grad():
                traced = torch.jit.trace(wrapper, example_feat, strict=False)
            path = os.path.join(export_dir, 'hift_full_traced.pt')
            traced.save(path)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'  Saved (traced): {path} ({size_mb:.1f} MB)')
            return True
        except Exception as e2:
            print(f'  Trace also FAILED: {e2}')
            return False


def test_exported_models(export_dir, device):
    """Test loading and running exported models."""
    print('\n=== Testing Exported Models ===')

    for fname in os.listdir(export_dir):
        if not fname.endswith('.pt'):
            continue

        path = os.path.join(export_dir, fname)
        print(f'\nTesting {fname}:')

        try:
            # Load
            model = torch.jit.load(path, map_location='cpu')
            print(f'  Loaded on CPU: OK')

            # Move to target device
            model = model.to(device)
            print(f'  Moved to {device}: OK')

            # Create example input based on model name
            if 'hift_decode' in fname:
                example = torch.randn(1, 80, 100, device=device)
            elif 'hift_full' in fname:
                example = torch.randn(1, 100, 80, device=device)
            else:
                print(f'  Unknown input format, skipping inference test')
                continue

            # Run inference
            with torch.no_grad():
                start = time.time()
                output = model(example)
                elapsed = time.time() - start

            if isinstance(output, tuple):
                shapes = [o.shape for o in output]
                print(f'  Inference: OK - output shapes: {shapes} ({elapsed*1000:.1f}ms)')
            else:
                print(f'  Inference: OK - output shape: {output.shape} ({elapsed*1000:.1f}ms)')

        except Exception as e:
            print(f'  FAILED: {e}')


def main():
    parser = argparse.ArgumentParser(description='Export CosyVoice2 to TorchScript')
    parser.add_argument('--cpu', action='store_true', help='Export for CPU (default: MPS)')
    parser.add_argument('--test', action='store_true', help='Test exported models')
    parser.add_argument('--test-only', action='store_true', help='Only test, skip export')
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu else torch.device('mps')
    print(f'Target device: {device}')
    print(f'PyTorch version: {torch.__version__}')

    os.makedirs(EXPORT_DIR, exist_ok=True)

    if not args.test_only:
        print('\nLoading CosyVoice2...')
        cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)

        # Export HiFT decode (simpler)
        export_hift(cosyvoice, device, EXPORT_DIR)

        # Export HiFT full pipeline
        export_hift_full(cosyvoice, device, EXPORT_DIR)

    if args.test or args.test_only:
        test_exported_models(EXPORT_DIR, device)

    print('\n=== Export Summary ===')
    print(f'Export directory: {EXPORT_DIR}')
    for f in sorted(os.listdir(EXPORT_DIR)):
        path = os.path.join(EXPORT_DIR, f)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'  {f}: {size_mb:.1f} MB')


if __name__ == '__main__':
    main()
