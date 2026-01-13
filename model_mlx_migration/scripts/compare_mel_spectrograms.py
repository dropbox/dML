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

"""Compare mel spectrograms between Python and C++."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

def main():
    print("=" * 70)
    print("MEL SPECTROGRAM COMPARISON")
    print("=" * 70)

    # Load Python mel
    py_mel_file = "/tmp/python_mel_0004.npy"
    if not os.path.exists(py_mel_file):
        print(f"Python mel not found at {py_mel_file}")
        print("Run scripts/compare_encoder_outputs.py first")
        return

    py_mel = np.load(py_mel_file)
    print("\nPython mel:")
    print(f"  Shape: {py_mel.shape}")
    print(f"  Min: {py_mel.min():.6f}")
    print(f"  Max: {py_mel.max():.6f}")
    print(f"  Mean: {py_mel.mean():.6f}")
    print(f"  Std: {py_mel.std():.6f}")
    print(f"  First 5 values: {py_mel[0, :5]}")

    # Check for C++ mel (need to add code to save it)
    cpp_mel_file = "/tmp/cpp_mel_0004.bin"
    if os.path.exists(cpp_mel_file):
        # Load as float32, reshape to match
        cpp_mel = np.fromfile(cpp_mel_file, dtype=np.float32)
        cpp_mel = cpp_mel.reshape(py_mel.shape)

        print("\nC++ mel:")
        print(f"  Shape: {cpp_mel.shape}")
        print(f"  Min: {cpp_mel.min():.6f}")
        print(f"  Max: {cpp_mel.max():.6f}")
        print(f"  Mean: {cpp_mel.mean():.6f}")
        print(f"  Std: {cpp_mel.std():.6f}")
        print(f"  First 5 values: {cpp_mel[0, :5]}")

        # Compare
        diff = np.abs(py_mel - cpp_mel)
        print("\nMel spectrogram difference:")
        print(f"  Max: {diff.max():.8f}")
        print(f"  Mean: {diff.mean():.8f}")

        if diff.max() < 1e-5:
            print("\n*** Mel spectrograms match within 1e-5 ***")
            print("Issue is in encoder, not mel computation.")
        elif diff.max() < 1e-3:
            print(f"\n*** Mel spectrograms differ slightly (max {diff.max():.6f}) ***")
        else:
            print("\n*** ERROR: Mel spectrograms differ significantly! ***")
            print("Need to fix mel spectrogram computation.")
    else:
        print(f"\nC++ mel not found at {cpp_mel_file}")
        print("Add this code to C++ whisper_model.cpp to save mel:")
        print("""
// After computing mel, add:
if (std::getenv("DEBUG_ENCODER")) {
    const float* mel_data = mel.data<float>();
    std::ofstream mel_file("/tmp/cpp_mel_0004.bin", std::ios::binary);
    mel_file.write(reinterpret_cast<const char*>(mel_data),
                   mel.size() * sizeof(float));
    mel_file.close();
    std::cerr << "[DEBUG] Saved mel to /tmp/cpp_mel_0004.bin\\n";
}
""")


if __name__ == "__main__":
    main()
