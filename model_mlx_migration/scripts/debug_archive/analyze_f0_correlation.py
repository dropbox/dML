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
Analyze if error correlates with F0 (voiced regions).
"""

from pathlib import Path

import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_seed0")
    ref = np.load(ref_dir / "tensors.npz")

    # F0 prediction at frame level
    f0 = ref["F0_pred"].astype(np.float32).reshape(-1)
    print(f"F0 shape: {f0.shape}")
    print(f"F0 range: {f0.min():.2f} to {f0.max():.2f}")

    # Each F0 frame corresponds to multiple audio samples
    # hop_length = upsample_rates (10*6) * istft_hop_size (5) = 300
    hop_length = 300
    print(f"Hop length: {hop_length} samples")

    # Find voiced frames (F0 > 0)
    voiced_mask = f0 > 0
    print(f"Voiced frames: {voiced_mask.sum()} / {len(f0)}")

    # Map F0 frames to audio sample indices
    voiced_start_frames = np.where(voiced_mask)[0]
    if len(voiced_start_frames) > 0:
        first_voiced = voiced_start_frames[0]
        last_voiced = voiced_start_frames[-1]
        print(f"First voiced frame: {first_voiced} -> sample {first_voiced * hop_length} ({first_voiced * hop_length / 24000:.4f}s)")
        print(f"Last voiced frame: {last_voiced} -> sample {last_voiced * hop_length} ({last_voiced * hop_length / 24000:.4f}s)")

    # UV (unvoiced/voiced) mask
    if "gen_uv" in ref:
        uv = ref["gen_uv"].astype(np.float32).reshape(-1)
        print(f"\nUV shape: {uv.shape}")
        print(f"UV values: unique={np.unique(uv)[:10]}")
        # UV = 1 means unvoiced, UV = 0 means voiced
        voiced_samples = (uv < 0.5).sum()
        print(f"Voiced samples (UV<0.5): {voiced_samples} / {len(uv)}")

    # Check har_source
    if "gen_har_source" in ref:
        har = ref["gen_har_source"].astype(np.float32).reshape(-1)
        print(f"\nHar source shape: {har.shape}")
        print(f"Har source range: {har.min():.4f} to {har.max():.4f}")
        # Find where har_source is non-zero
        har_nonzero = np.abs(har) > 0.001
        if har_nonzero.any():
            first_active = np.where(har_nonzero)[0][0]
            last_active = np.where(har_nonzero)[0][-1]
            print(f"First active har sample: {first_active} ({first_active/24000:.4f}s)")
            print(f"Last active har sample: {last_active} ({last_active/24000:.4f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
