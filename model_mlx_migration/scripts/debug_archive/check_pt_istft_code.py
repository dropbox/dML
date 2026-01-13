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
Check PyTorch generator's ISTFT code.
Must run with: /tmp/kokoro_env/bin/python scripts/check_pt_istft_code.py
"""

import inspect

from kokoro import KModel


def main():
    model = KModel().eval()
    gen = model.decoder.generator

    # Print generator class and ISTFT-related code
    print("=" * 72)
    print("PyTorch Generator ISTFT Analysis")
    print("=" * 72)

    print(f"\nGenerator class: {type(gen)}")
    print(f"post_n_fft: {gen.post_n_fft}")

    # Look for ISTFT-related attributes
    print("\n--- ISTFT-related attributes ---")
    for attr in dir(gen):
        if 'istft' in attr.lower() or 'stft' in attr.lower():
            val = getattr(gen, attr, None)
            print(f"  {attr}: {type(val).__name__}")

    # Check for stft attribute
    if hasattr(gen, 'stft'):
        print("\n--- gen.stft ---")
        stft = gen.stft
        print(f"Type: {type(stft)}")
        print(f"Attributes: {[a for a in dir(stft) if not a.startswith('_')][:20]}")

        # Check for inverse method
        if hasattr(stft, 'inverse'):
            print("\n--- stft.inverse source code ---")
            try:
                print(inspect.getsource(stft.inverse))
            except Exception:
                print("Could not get source code")

    # Print forward method to see how ISTFT is called
    print("\n--- Generator forward method (last 100 lines) ---")
    try:
        src = inspect.getsource(gen.forward)
        lines = src.split('\n')
        for line in lines[-100:]:
            print(line)
    except Exception as e:
        print(f"Error: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
