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

"""Check the remaining 2 boundary positions."""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")
    pt_har_source = gen_ref["gen_har_source"]
    pt_har = gen_ref["gen_har"]

    n_bins = pt_har.shape[-1] // 2
    pt_phase = pt_har[..., n_bins:]

    # Load MLX and compute STFT
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    generator = mlx_model.decoder.generator

    har_1d = mx.array(pt_har_source.astype(np.float32)).squeeze(-1)

    # Get complex spectrum
    batch, samples = har_1d.shape
    n_fft = generator.source_stft_n_fft
    hop = generator.source_stft_hop
    pad_amount = n_fft // 2
    x = har_1d
    left_pad = x[:, 1 : pad_amount + 1][:, ::-1]
    right_pad = x[:, -(pad_amount + 1) : -1][:, ::-1]
    x_padded = mx.concatenate([left_pad, x, right_pad], axis=1)
    padded_len = x_padded.shape[1]
    num_frames = (padded_len - n_fft) // hop + 1
    indices = mx.arange(n_fft)[None, :] + mx.arange(num_frames)[:, None] * hop
    frames = x_padded[:, indices.flatten()].reshape(batch, num_frames, n_fft)
    frames = frames * generator._source_window
    spectrum = mx.fft.rfft(frames, axis=-1)
    mx.eval(spectrum)

    real = np.array(spectrum.real)
    imag = np.array(spectrum.imag)

    # Check positions [0,0,3] and [0,0,5]
    for pos in [(0, 0, 3), (0, 0, 5)]:
        b, f, c = pos
        r, im = real[b, f, c], imag[b, f, c]
        pt_p = pt_phase[b, f, c]
        mlx_raw = float(mx.arctan2(spectrum.imag, spectrum.real)[b, f, c])
        print(f"Position {pos}:")
        print(f"  real={r:.9f}, imag={im:.15f}")
        print(f"  MLX arctan2={mlx_raw:.6f}, PT={pt_p:.6f}")
        print(f"  |imag|={abs(im):.15f} < 1e-7? {abs(im) < 1e-7}")
        print()

    # Check what threshold would catch these
    print("Finding appropriate threshold...")
    for thresh in [1e-6, 1e-5, 1e-4]:
        large_diff = 0
        mlx_phase = np.array(mx.arctan2(spectrum.imag, spectrum.real))
        for b in range(real.shape[0]):
            for f in range(real.shape[1]):
                for c in range(real.shape[2]):
                    if abs(imag[b, f, c]) < thresh and real[b, f, c] < 0:
                        mlx_p = mlx_phase[b, f, c]
                        pt_p = pt_phase[b, f, c]
                        diff = abs(mlx_p - pt_p)
                        if diff > 3:
                            large_diff += 1
        print(f"  threshold={thresh}: {large_diff} remaining boundary issues")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
