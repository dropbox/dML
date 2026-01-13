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
Test PyTorch FFT on MPS vs CPU to see if MPS matches MLX.

MLX uses Apple's Accelerate/vDSP FFT implementation.
PyTorch MPS may use the same underlying implementation.
"""

import mlx.core as mx
import numpy as np
import torch


def test_fft_backends():
    """Compare FFT across PyTorch CPU, PyTorch MPS, and MLX."""
    print("=" * 72)
    print("FFT Backend Comparison")
    print("=" * 72)

    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")

    # Test signal
    np.random.seed(42)
    signal = np.random.randn(800).astype(np.float32)

    # PyTorch CPU
    pt_cpu = torch.from_numpy(signal)
    pt_cpu_fft = torch.fft.rfft(pt_cpu)
    pt_cpu_phase = torch.angle(pt_cpu_fft).numpy()

    print("\nPyTorch CPU:")
    print(f"  Phase range: [{pt_cpu_phase.min():.4f}, {pt_cpu_phase.max():.4f}]")

    # PyTorch MPS (if available)
    if mps_available:
        pt_mps = torch.from_numpy(signal).to("mps")
        pt_mps_fft = torch.fft.rfft(pt_mps)
        pt_mps_phase = torch.angle(pt_mps_fft).cpu().numpy()

        print("\nPyTorch MPS:")
        print(f"  Phase range: [{pt_mps_phase.min():.4f}, {pt_mps_phase.max():.4f}]")

        # Compare CPU vs MPS
        phase_diff_cpu_mps = np.abs(pt_cpu_phase - pt_mps_phase)
        print("\nCPU vs MPS:")
        print(f"  max_diff: {phase_diff_cpu_mps.max():.6f}")

        boundary_wraps = phase_diff_cpu_mps > 3.0
        print(f"  boundary wraps: {boundary_wraps.sum()}/{len(phase_diff_cpu_mps)}")
    else:
        pt_mps_phase = None

    # MLX
    mx_signal = mx.array(signal)
    mx_fft = mx.fft.rfft(mx_signal)
    mx.eval(mx_fft)
    mx_phase = np.array(mx.arctan2(mx_fft.imag, mx_fft.real))

    print("\nMLX:")
    print(f"  Phase range: [{mx_phase.min():.4f}, {mx_phase.max():.4f}]")

    # Compare CPU vs MLX
    phase_diff_cpu_mlx = np.abs(pt_cpu_phase - mx_phase)
    print("\nCPU vs MLX:")
    print(f"  max_diff: {phase_diff_cpu_mlx.max():.4f}")
    boundary_wraps = phase_diff_cpu_mlx > 3.0
    print(f"  boundary wraps: {boundary_wraps.sum()}/{len(phase_diff_cpu_mlx)}")

    # Compare MPS vs MLX
    if pt_mps_phase is not None:
        phase_diff_mps_mlx = np.abs(pt_mps_phase - mx_phase)
        print("\nMPS vs MLX:")
        print(f"  max_diff: {phase_diff_mps_mlx.max():.4f}")
        boundary_wraps = phase_diff_mps_mlx > 3.0
        print(f"  boundary wraps: {boundary_wraps.sum()}/{len(phase_diff_mps_mlx)}")

        if phase_diff_mps_mlx.max() < 0.001:
            print("\n*** MPS and MLX produce identical FFT! ***")
            print("This means PyTorch MPS and MLX use the same FFT backend.")


if __name__ == "__main__":
    test_fft_backends()
