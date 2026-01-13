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

"""Debug encoder layer by layer to find where divergence starts."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx.core as mx
import mlx.nn as nn

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"


def save_intermediate(name: str, arr: mx.array):
    """Save intermediate array for comparison."""
    mx.eval(arr)
    np_arr = np.array(arr).astype(np.float32)
    np.save(f"/tmp/py_{name}.npy", np_arr)
    print(f"Saved /tmp/py_{name}.npy: shape={np_arr.shape}, min={np_arr.min():.6f}, max={np_arr.max():.6f}")


def main():
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

    print("=" * 70)
    print("ENCODER LAYER-BY-LAYER DEBUG - Python")
    print("=" * 70)

    # Load model
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)
    encoder = model.encoder

    # Load audio and compute mel
    audio = load_audio(TEST_FILE, sample_rate=16000)
    print(f"Loaded audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Pad to 30 seconds (matching C++ behavior after fix)
    padded_audio = np.zeros(480000, dtype=np.float32)
    padded_audio[:min(len(audio), 480000)] = audio[:min(len(audio), 480000)]

    # Compute mel
    mel = log_mel_spectrogram(padded_audio, n_mels=128)
    print(f"Mel shape: {mel.shape}")
    save_intermediate("mel", mel)

    # Add batch dimension
    x = mx.expand_dims(mel, 0)

    # Step 1: conv1
    x = nn.gelu(encoder.conv1(x))
    save_intermediate("after_conv1", x)

    # Step 2: conv2
    x = nn.gelu(encoder.conv2(x))
    save_intermediate("after_conv2", x)

    # Step 3: Add positional embeddings
    seq_len = x.shape[1]
    pos = encoder._positional_embedding[:seq_len]
    save_intermediate("positional_embedding", pos)

    x = x + pos
    save_intermediate("after_pos_emb", x)

    # Step 4: Transformer layers
    for i, block in enumerate(encoder.blocks):
        x, _, _ = block(x)
        mx.eval(x)  # Force evaluation

        # Save more layers around divergence point (layer 20)
        if i == 0 or i == len(encoder.blocks) - 1 or i % 4 == 0 or (17 <= i <= 21):
            save_intermediate(f"after_layer_{i:02d}", x)

    # Step 5: Final layer norm
    x = mx.fast.layer_norm(x, encoder.ln_post.weight, encoder.ln_post.bias, eps=1e-5)
    save_intermediate("encoder_output", x)

    print("\n" + "=" * 70)
    print("COMPARISON INSTRUCTIONS")
    print("=" * 70)
    print("""
To compare with C++ intermediates:

1. Add DEBUG_ENCODER_LAYERS env variable check in whisper_model.cpp:
   - Save intermediates after conv1, conv2, pos_emb, each layer, final output
   - Use same names as Python: /tmp/cpp_after_conv1.bin, etc.

2. Run C++ inference with DEBUG_ENCODER_LAYERS=1

3. Run this script with --compare flag to compare all intermediates
""")

    # Check if C++ intermediates exist
    cpp_files = [
        "after_conv1", "after_conv2", "positional_embedding", "after_pos_emb",
        "after_layer_00", "after_layer_04", "after_layer_08", "after_layer_12",
        "after_layer_16", "after_layer_17", "after_layer_18", "after_layer_19",
        "after_layer_20", "after_layer_21", "after_layer_24", "after_layer_28",
        "after_layer_31", "encoder_output"
    ]

    print("\n" + "=" * 70)
    print("COMPARING WITH C++ (if available)")
    print("=" * 70)

    for name in cpp_files:
        py_file = f"/tmp/py_{name}.npy"
        cpp_file = f"/tmp/cpp_{name}.bin"

        if not os.path.exists(py_file):
            continue

        py_arr = np.load(py_file)

        if os.path.exists(cpp_file):
            cpp_arr = np.fromfile(cpp_file, dtype=np.float32)

            # Try to reshape to match Python
            try:
                cpp_arr = cpp_arr.reshape(py_arr.shape)
            except ValueError:
                print(f"{name}: Shape mismatch! Python {py_arr.shape}, C++ has {cpp_arr.size} elements")
                continue

            diff = np.abs(py_arr - cpp_arr)
            max_diff = diff.max()
            mean_diff = diff.mean()

            status = "✓" if max_diff < 1e-5 else ("!" if max_diff < 0.01 else "✗")
            print(f"{status} {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")

            if max_diff >= 0.01:
                # Find where the biggest difference is
                flat_idx = np.argmax(diff.flatten())
                print(f"    Largest diff at index {flat_idx}: py={py_arr.flatten()[flat_idx]:.6f}, cpp={cpp_arr.flatten()[flat_idx]:.6f}")
        else:
            print(f"? {name}: C++ file not found ({cpp_file})")


if __name__ == "__main__":
    main()
