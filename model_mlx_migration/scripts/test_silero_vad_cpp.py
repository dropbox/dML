#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Silero VAD C++ implementation against Python.

This test:
1. Generates random audio chunks
2. Runs both Python Silero VAD and exports expected outputs
3. Can be compared with C++ results to verify correctness

The C++ implementation should produce outputs within ~0.02 of Python.
"""

import sys
from pathlib import Path
import numpy as np

# Add converters/models to path for MLX implementation
sys.path.insert(0, str(Path(__file__).parent.parent / "tools/pytorch_to_mlx/converters/models"))

import torch
import mlx.core as mx


def test_silero_vad_python():
    """Test Python Silero VAD and export expected values."""
    from silero_vad_mlx import load_silero_vad

    # Load models
    print("Loading PyTorch Silero VAD...")
    torch_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )

    print("Loading MLX Silero VAD...")
    mlx_model = load_silero_vad(sample_rate=16000)

    # Test with deterministic audio
    np.random.seed(42)

    print("\nTesting single chunk (512 samples)...")
    audio = np.random.randn(512).astype(np.float32) * 0.1

    # PyTorch
    torch_prob = torch_model(torch.tensor(audio), 16000).item()

    # MLX
    mlx_model.reset_state()
    mlx_audio = mx.array(audio[None, :])
    mlx_prob, _ = mlx_model(mlx_audio)
    mx.eval(mlx_prob)
    mlx_val = float(mlx_prob[0, 0])

    print(f"  PyTorch: {torch_prob:.6f}")
    print(f"  MLX:     {mlx_val:.6f}")
    print(f"  Diff:    {abs(torch_prob - mlx_val):.6f}")

    # Export test values for C++ comparison
    output_dir = Path(__file__).parent.parent / "models" / "silero_vad"
    test_file = output_dir / "test_values.txt"

    print(f"\nExporting test values to {test_file}...")

    # Generate multiple test cases
    test_cases = []
    for seed in [42, 123, 456, 789, 1000]:
        np.random.seed(seed)
        audio = np.random.randn(512).astype(np.float32) * 0.1

        # Get PyTorch reference
        torch_model.reset_states()
        torch_prob = torch_model(torch.tensor(audio), 16000).item()

        # Get MLX reference
        mlx_model.reset_state()
        mlx_audio = mx.array(audio[None, :])
        mlx_prob, _ = mlx_model(mlx_audio)
        mx.eval(mlx_prob)
        mlx_val = float(mlx_prob[0, 0])

        test_cases.append({
            "seed": seed,
            "pytorch": torch_prob,
            "mlx": mlx_val,
            "diff": abs(torch_prob - mlx_val)
        })

    with open(test_file, "w") as f:
        f.write("# Silero VAD Test Values\n")
        f.write("# Format: seed, pytorch_prob, mlx_prob, diff\n")
        f.write("#\n")
        for tc in test_cases:
            f.write(f"{tc['seed']}, {tc['pytorch']:.8f}, {tc['mlx']:.8f}, {tc['diff']:.8f}\n")

    print("\nTest Cases:")
    print(f"{'Seed':>6} {'PyTorch':>12} {'MLX':>12} {'Diff':>12}")
    print("-" * 44)
    for tc in test_cases:
        status = "PASS" if tc["diff"] < 0.02 else "FAIL"
        print(f"{tc['seed']:>6} {tc['pytorch']:>12.6f} {tc['mlx']:>12.6f} {tc['diff']:>12.6f} {status}")

    # Export raw audio bytes for C++ to read
    audio_file = output_dir / "test_audio.bin"
    np.random.seed(42)
    audio = np.random.randn(512).astype(np.float32) * 0.1
    audio.tofile(audio_file)
    print(f"\nExported test audio to {audio_file}")

    # Verify all test cases pass
    max_diff = max(tc["diff"] for tc in test_cases)
    avg_diff = sum(tc["diff"] for tc in test_cases) / len(test_cases)

    print("\nSummary:")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Avg diff:  {avg_diff:.6f}")

    if max_diff < 0.02:
        print("\n  PASS: All test cases within tolerance")
        return True
    else:
        print("\n  FAIL: Some test cases exceed tolerance")
        return False


if __name__ == "__main__":
    success = test_silero_vad_python()
    sys.exit(0 if success else 1)
