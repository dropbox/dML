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
Find WHERE in the audio the large errors occur.
This can help identify if it's at boundaries, specific frames, etc.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_seed0")
    if not ref_dir.exists():
        print(f"Reference directory not found: {ref_dir}")
        return 1

    ref = np.load(ref_dir / "tensors.npz")

    # Load MLX model
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    # Prepare inputs
    asr_nlc = mx.array(ref["asr_nlc"].astype(np.float32))
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    n = mx.array(ref["N_pred"].astype(np.float32))
    style_128 = mx.array(ref["style_128"].astype(np.float32))

    # Run MLX decoder without overrides
    audio_mlx = mlx_model.decoder(asr_nlc, f0, n, style_128)
    mx.eval(audio_mlx)
    audio_mlx_np = np.array(audio_mlx).reshape(-1)

    # Reference audio
    audio_ref = ref["audio"].astype(np.float32).reshape(-1)

    # Compute error
    min_len = min(len(audio_ref), len(audio_mlx_np))
    diff = np.abs(audio_ref[:min_len] - audio_mlx_np[:min_len])

    print("=" * 72)
    print("Error Location Analysis")
    print("=" * 72)
    print(f"Audio length: {min_len} samples ({min_len / 24000:.2f}s)")
    print(f"Max error: {diff.max():.6f} at sample {diff.argmax()}")
    print(f"Mean error: {diff.mean():.6f}")

    # Find samples with large error (> 0.01)
    large_error_idx = np.where(diff > 0.01)[0]
    print(f"\nSamples with error > 0.01: {len(large_error_idx)}")

    if len(large_error_idx) > 0:
        print(f"  First 20 locations: {large_error_idx[:20]}")
        print(f"  Time range: {large_error_idx[0]/24000:.4f}s to {large_error_idx[-1]/24000:.4f}s")

        # Cluster analysis - are errors grouped?
        gaps = np.diff(large_error_idx)
        cluster_starts = np.where(gaps > 100)[0]  # Gap of 100 samples = new cluster
        n_clusters = len(cluster_starts) + 1 if len(large_error_idx) > 0 else 0
        print(f"  Approximate error clusters: {n_clusters}")

    # Find samples with very large error (> 0.05)
    very_large_idx = np.where(diff > 0.05)[0]
    print(f"\nSamples with error > 0.05: {len(very_large_idx)}")
    if len(very_large_idx) > 0:
        for idx in very_large_idx[:10]:
            print(f"  Sample {idx} ({idx/24000:.4f}s): ref={audio_ref[idx]:.4f}, mlx={audio_mlx_np[idx]:.4f}, diff={diff[idx]:.4f}")

    # Error by segment (divide into 10 segments)
    n_segments = 10
    segment_size = min_len // n_segments
    print(f"\nError by segment ({segment_size} samples each):")
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        seg_diff = diff[start:end]
        print(f"  Segment {i} ({start/24000:.2f}s-{end/24000:.2f}s): max={seg_diff.max():.4f}, mean={seg_diff.mean():.6f}")

    # Compute rolling mean error
    window = 500  # ~20ms at 24kHz
    rolling_mean = np.convolve(diff, np.ones(window)/window, mode='valid')
    peak_rolling_idx = rolling_mean.argmax()
    print(f"\nPeak rolling mean error at sample {peak_rolling_idx} ({peak_rolling_idx/24000:.4f}s): {rolling_mean[peak_rolling_idx]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
