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
Export Silero VAD weights to binary format for C++ loading.

Exports all weights from the 16kHz Silero VAD model:
- STFT forward_basis_buffer (DFT basis for conv-based STFT)
- Encoder conv weights and biases (4 blocks)
- LSTM weights and biases
- Decoder output conv weights and biases

Output format: Binary file with header describing tensor shapes.
"""

import struct
from pathlib import Path
import numpy as np
import torch


def export_silero_vad_weights(output_dir: Path):
    """Export Silero VAD weights to binary format."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Silero VAD from torch hub
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )

    # Get the 16kHz model
    pt_model = model._model

    # Collect all weights
    weights = {}

    # 1. STFT forward_basis_buffer
    stft_buffers = dict(pt_model.stft.named_buffers())
    if "forward_basis_buffer" in stft_buffers:
        weights["stft_forward_basis_buffer"] = stft_buffers["forward_basis_buffer"].detach().numpy()
        print(f"  stft_forward_basis_buffer: {weights['stft_forward_basis_buffer'].shape}")

    # 2. Encoder conv weights (reparam_conv.weight, reparam_conv.bias)
    state_dict = dict(pt_model.named_parameters())

    for i in range(4):
        weight_key = f"encoder.{i}.reparam_conv.weight"
        bias_key = f"encoder.{i}.reparam_conv.bias"

        if weight_key in state_dict:
            # PyTorch Conv1d: [out, in, kernel]
            weights[f"encoder_{i}_weight"] = state_dict[weight_key].detach().numpy()
            weights[f"encoder_{i}_bias"] = state_dict[bias_key].detach().numpy()
            print(f"  encoder_{i}_weight: {weights[f'encoder_{i}_weight'].shape}")
            print(f"  encoder_{i}_bias: {weights[f'encoder_{i}_bias'].shape}")

    # 3. LSTM weights
    lstm_keys = ["decoder.rnn.weight_ih", "decoder.rnn.weight_hh",
                 "decoder.rnn.bias_ih", "decoder.rnn.bias_hh"]
    for key in lstm_keys:
        if key in state_dict:
            name = key.replace(".", "_")
            weights[name] = state_dict[key].detach().numpy()
            print(f"  {name}: {weights[name].shape}")

    # 4. Decoder output conv
    if "decoder.decoder.2.weight" in state_dict:
        # PyTorch Conv1d: [out, in, kernel]
        weights["decoder_output_weight"] = state_dict["decoder.decoder.2.weight"].detach().numpy()
        weights["decoder_output_bias"] = state_dict["decoder.decoder.2.bias"].detach().numpy()
        print(f"  decoder_output_weight: {weights['decoder_output_weight'].shape}")
        print(f"  decoder_output_bias: {weights['decoder_output_bias'].shape}")

    # Save as NPZ for Python/MLX verification
    npz_path = output_dir / "silero_vad_16k.npz"
    np.savez(npz_path, **weights)
    print(f"\nSaved NPZ: {npz_path}")

    # Save as binary format for C++
    # Format:
    #   4 bytes: magic "SVAD"
    #   4 bytes: version (1)
    #   4 bytes: number of tensors
    #   For each tensor:
    #     4 bytes: name length
    #     N bytes: name
    #     4 bytes: ndim
    #     ndim * 4 bytes: shape
    #     4 bytes: data size in bytes
    #     data_size bytes: float32 data

    bin_path = output_dir / "silero_vad_16k.bin"
    with open(bin_path, "wb") as f:
        # Magic and version
        f.write(b"SVAD")
        f.write(struct.pack("<I", 1))  # version 1
        f.write(struct.pack("<I", len(weights)))  # num tensors

        for name, arr in weights.items():
            # Name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # Shape
            arr = arr.astype(np.float32)
            f.write(struct.pack("<I", arr.ndim))
            for dim in arr.shape:
                f.write(struct.pack("<I", dim))

            # Data
            data = arr.tobytes()
            f.write(struct.pack("<I", len(data)))
            f.write(data)

    print(f"Saved binary: {bin_path}")

    # Print summary
    total_params = sum(w.size for w in weights.values())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size: {total_params * 4 / 1024 / 1024:.2f} MB")

    return weights


def verify_against_mlx_python():
    """Verify exported weights work with MLX Python implementation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools/pytorch_to_mlx/converters/models"))
    from silero_vad_mlx import load_silero_vad
    import mlx.core as mx

    print("\nVerifying against MLX Python implementation...")

    # Load MLX model
    mlx_model = load_silero_vad(sample_rate=16000)

    # Test with random audio
    np.random.seed(42)
    audio = np.random.randn(512).astype(np.float32) * 0.1

    # MLX inference
    mlx_audio = mx.array(audio[None, :])
    mlx_model.reset_state()
    mlx_prob, _ = mlx_model(mlx_audio)
    mx.eval(mlx_prob)
    mlx_val = float(mlx_prob[0, 0])

    # PyTorch inference
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )

    torch_audio = torch.tensor(audio)
    pt_val = model(torch_audio, 16000).item()

    diff = abs(mlx_val - pt_val)
    print(f"  PyTorch: {pt_val:.6f}")
    print(f"  MLX:     {mlx_val:.6f}")
    print(f"  Diff:    {diff:.6f}")

    if diff < 0.02:
        print("  PASS: MLX matches PyTorch within tolerance")
    else:
        print("  WARN: MLX differs from PyTorch more than expected")

    return diff < 0.02


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "models" / "silero_vad"

    print("=" * 60)
    print("Exporting Silero VAD Weights")
    print("=" * 60)

    weights = export_silero_vad_weights(output_dir)

    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    success = verify_against_mlx_python()

    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: Weights exported and verified")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("WARNING: Verification showed larger than expected difference")
        print("=" * 60)
