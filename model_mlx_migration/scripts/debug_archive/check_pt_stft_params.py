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
Check PyTorch STFT parameters.
Must run with: /tmp/kokoro_env/bin/python scripts/check_pt_stft_params.py
"""

from kokoro import KModel


def main():
    model = KModel().eval()
    gen = model.decoder.generator
    stft = gen.stft

    print("=" * 72)
    print("PyTorch STFT Parameters")
    print("=" * 72)

    print(f"filter_length: {stft.filter_length}")
    print(f"hop_length: {stft.hop_length}")
    print(f"win_length: {stft.win_length}")
    print(f"window shape: {stft.window.shape}")
    print(f"window type: {type(stft.window)}")
    print(f"window values (first 10): {stft.window[:10].tolist()}")

    # Also check post_n_fft and istft_hop
    print(f"\npost_n_fft: {gen.post_n_fft}")

    # Check n_fft vs filter_length
    print("\nFor inverse ISTFT:")
    print(f"  n_fft (filter_length): {stft.filter_length}")
    print(f"  hop_length: {stft.hop_length}")
    print(f"  win_length: {stft.win_length}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
