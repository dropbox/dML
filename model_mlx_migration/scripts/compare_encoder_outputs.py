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

"""Compare encoder outputs between Python and C++ for file 0004."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx.core as mx

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"


def get_python_encoder_output():
    """Get encoder output from Python WhisperMLX."""
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

    # Load model
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    # Load audio directly without VAD
    audio = load_audio(TEST_FILE, sample_rate=16000)
    print(f"Python: Loaded {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Pad to 30 seconds
    padded_audio = np.zeros(480000, dtype=np.float32)
    padded_audio[:min(len(audio), 480000)] = audio[:min(len(audio), 480000)]

    # Compute mel spectrogram
    n_mels = 128  # large-v3
    mel = log_mel_spectrogram(padded_audio, n_mels=n_mels)
    print(f"Python: Mel shape {mel.shape}")

    # Save mel for C++ comparison
    mel_np = np.array(mel, dtype=np.float32)
    np.save("/tmp/python_mel_0004.npy", mel_np)
    print("Python: Saved mel to /tmp/python_mel_0004.npy")

    # Add batch dimension
    mel = mx.expand_dims(mel, axis=0)

    # Encode audio
    audio_features = model.embed_audio(mel)

    # Convert to float32 (as C++ does for parity)
    audio_features = audio_features.astype(mx.float32)
    mx.eval(audio_features)

    print(f"Python: Encoder output shape {audio_features.shape}")

    # Convert to numpy for saving
    enc_np = np.array(audio_features)
    np.save("/tmp/python_encoder_0004.npy", enc_np)
    print("Python: Saved encoder output to /tmp/python_encoder_0004.npy")

    return enc_np, mel_np


def analyze_encoder_output(enc_np, name="Encoder"):
    """Analyze encoder output statistics."""
    print(f"\n{name} output statistics:")
    print(f"  Shape: {enc_np.shape}")
    print(f"  Min: {enc_np.min():.6f}")
    print(f"  Max: {enc_np.max():.6f}")
    print(f"  Mean: {enc_np.mean():.6f}")
    print(f"  Std: {enc_np.std():.6f}")
    print(f"  First 5 values: {enc_np[0, 0, :5]}")
    print(f"  Last 5 values: {enc_np[0, -1, -5:]}")


def main():
    print("=" * 70)
    print("ENCODER OUTPUT COMPARISON - File 0004")
    print("=" * 70)

    # Get Python encoder output
    print("\n--- Python ---")
    py_enc, py_mel = get_python_encoder_output()
    analyze_encoder_output(py_enc, "Python encoder")

    # Check for NaN or Inf
    nan_count = np.isnan(py_enc).sum()
    inf_count = np.isinf(py_enc).sum()
    print(f"\nPython NaN count: {nan_count}, Inf count: {inf_count}")

    print("\n" + "=" * 70)
    print("C++ COMPARISON INSTRUCTIONS")
    print("=" * 70)
    print("""
To compare with C++, add this code to whisper_model.cpp after encoding:

```cpp
// After: auto encoder_output = encode(mel);
// Add:
std::cout << "[DEBUG] Encoder output shape: " << encoder_output.shape()[0]
          << "x" << encoder_output.shape()[1] << "x" << encoder_output.shape()[2] << "\\n";

// Save encoder output to file for Python comparison
auto enc_data = encoder_output.data<float>();
std::ofstream enc_file("/tmp/cpp_encoder_0004.bin", std::ios::binary);
enc_file.write(reinterpret_cast<const char*>(enc_data),
               encoder_output.size() * sizeof(float));
enc_file.close();
std::cout << "[DEBUG] Saved encoder output to /tmp/cpp_encoder_0004.bin\\n";
```

Then run:
```bash
DEBUG_WHISPER=1 ./src/mlx_inference_engine/build/test_mlx_engine \\
  --whisper ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/*/ \\
  --transcribe data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac \\
  --no-vad
```

After that, run this script again with --compare flag to compare:
```bash
python scripts/compare_encoder_outputs.py --compare
```
""")

    # Check if C++ output exists
    cpp_file = "/tmp/cpp_encoder_0004.bin"
    if os.path.exists(cpp_file):
        print("\n" + "=" * 70)
        print("COMPARING WITH C++ OUTPUT")
        print("=" * 70)

        # Load C++ output
        cpp_enc = np.fromfile(cpp_file, dtype=np.float32)
        cpp_enc = cpp_enc.reshape(py_enc.shape)
        analyze_encoder_output(cpp_enc, "C++ encoder")

        # Compare
        diff = np.abs(py_enc - cpp_enc)
        print("\nAbsolute difference:")
        print(f"  Max: {diff.max():.8f}")
        print(f"  Mean: {diff.mean():.8f}")
        print(f"  Std: {diff.std():.8f}")

        # Check where differences are largest
        flat_diff = diff.flatten()
        worst_idx = np.argpartition(flat_diff, -10)[-10:]
        print("\n10 largest differences:")
        for idx in sorted(worst_idx, key=lambda x: flat_diff[x], reverse=True):
            print(f"  Index {idx}: diff={flat_diff[idx]:.8f}")

        # Check if encoder outputs match within tolerance
        if diff.max() < 1e-5:
            print("\n*** SUCCESS: Encoder outputs match within 1e-5 ***")
            print("Issue is in decoder, not encoder.")
        elif diff.max() < 1e-3:
            print(f"\n*** WARNING: Encoder outputs differ by up to {diff.max():.6f} ***")
            print("Small differences may accumulate in decoder.")
        else:
            print(f"\n*** ERROR: Encoder outputs differ significantly (max {diff.max():.4f}) ***")
            print("Need to investigate encoder precision.")
    else:
        print(f"\nC++ encoder output not found at {cpp_file}")
        print("Follow the instructions above to generate it.")


if __name__ == "__main__":
    main()
