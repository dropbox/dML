#!/usr/bin/env python3
"""Export HiFT vocoder for MPS with correct input format."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cosyvoice_repo'))

import torch
import torch.nn as nn

# Disable CUDA
torch.cuda.is_available = lambda: False


class HiFTTracedWrapper(nn.Module):
    """Wrapper that expects (batch, 80, time) format - matching flow_full output."""
    def __init__(self, hift):
        super().__init__()
        self.hift = hift

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to audio waveform.

        Args:
            mel: (batch, 80, time) - directly from flow_full output

        Returns:
            Audio waveform (batch, samples)
        """
        # HiFT.inference expects speech_feat in (batch, 80, time) format
        # Create device-agnostic cache_source tensor to avoid CPU/MPS mismatch
        # The default torch.zeros(1,1,0) in inference() captures CPU device during trace
        cache_source = mel.new_zeros(1, 1, 0)
        return self.hift.inference(mel, cache_source)[0]


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice2

    project_root = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(project_root, 'models/cosyvoice/CosyVoice2-0.5B')
    output_path = os.path.join(project_root, 'models/cosyvoice/exported/hift_traced.pt')

    print('Loading CosyVoice2...')
    model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

    # Trace on MPS for Metal GPU execution
    # TorchScript tracing captures device-dependent operations at trace time
    # Since we run on MPS, we trace on MPS to ensure device consistency
    trace_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Trace device: {trace_device}')

    hift = model.model.hift
    hift.eval()
    hift.to(trace_device)

    wrapper = HiFTTracedWrapper(hift)
    wrapper.eval()
    wrapper.to(trace_device)

    # Example input in (batch, 80, time) format - same as flow_full output
    example = torch.randn(1, 80, 100, device=trace_device)

    print(f'Tracing HiFT on {trace_device}...')
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)

    # Save
    traced.save(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'Saved: {output_path} ({size_mb:.1f} MB)')

    # Test loading and running on MPS
    runtime_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'\nTesting loaded model on {runtime_device}...')
    loaded = torch.jit.load(output_path, map_location='cpu')
    loaded = loaded.to(runtime_device)

    test_mel = torch.randn(1, 80, 100, device=runtime_device)
    with torch.no_grad():
        output = loaded(test_mel)
    print(f'SUCCESS! Output shape: {output.shape}')


if __name__ == '__main__':
    main()
