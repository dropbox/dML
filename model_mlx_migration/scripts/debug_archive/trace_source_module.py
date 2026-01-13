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
Trace SourceModule step by step comparing MLX vs PyTorch.
"""

import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def pytorch_sine_gen(
    f0,
    upp,
    sample_rate=24000,
    num_harmonics=9,
    sine_amp=0.1,
    voiced_threshold=10,
    l_linear_weight=None,
    l_linear_bias=None,
):
    """
    Reference PyTorch SineGen implementation.
    Based on StyleTTS2/ISTFTNet.
    """
    batch, length = f0.shape
    length * upp

    # Upsample F0 to audio rate
    f0_up = f0.unsqueeze(1)  # [batch, 1, length]
    f0_up = F.interpolate(f0_up, scale_factor=upp, mode="nearest").squeeze(
        1
    )  # [batch, samples]

    # UV mask
    uv = (f0_up > voiced_threshold).float()  # [batch, samples]

    # Generate harmonics
    harmonics = []
    for h in range(1, num_harmonics + 1):
        # rad_values = (f0 * h / sample_rate) mod 1
        rad_values = (f0_up * h / sample_rate) % 1.0  # [batch, samples]

        # Skip random phase for deterministic comparison
        # if h > 1:
        #     rand_ini = torch.rand(batch) - 0.5
        #     rad_values[:, 0] += rand_ini

        # Anti-aliasing: downsample -> cumsum -> upsample
        # Downsample with linear interpolation
        rad_2d = rad_values.unsqueeze(1)  # [batch, 1, samples]
        rad_down = F.interpolate(
            rad_2d, scale_factor=1 / upp, mode="linear", align_corners=True
        )  # [batch, 1, length]
        rad_down = rad_down.squeeze(1)  # [batch, length]

        # Cumulative sum at low rate
        phase_low = torch.cumsum(rad_down, dim=1) * 2 * math.pi  # [batch, length]

        # Scale phase by upp
        phase_scaled = phase_low * upp  # [batch, length]

        # Upsample with linear interpolation
        phase_2d = phase_scaled.unsqueeze(1)  # [batch, 1, length]
        phase = F.interpolate(
            phase_2d, scale_factor=upp, mode="linear", align_corners=True
        )  # [batch, 1, samples]
        phase = phase.squeeze(1)  # [batch, samples]

        # Compute sine
        sine = torch.sin(phase) * sine_amp
        harmonics.append(sine)

    # Stack harmonics: [batch, samples, 9]
    harmonics_stack = torch.stack(harmonics, dim=-1)

    # UV mask for harmonics
    uv_expanded = uv.unsqueeze(-1)  # [batch, samples, 1]

    # Noise: use zeros for deterministic comparison
    noise = torch.zeros_like(harmonics_stack)

    # Apply UV mask and add noise
    sine_waves = harmonics_stack * uv_expanded + noise

    # Combine harmonics using l_linear + tanh
    if l_linear_weight is not None:
        # l_linear is Linear(9, 1)
        har_source = torch.tanh(
            F.linear(sine_waves, l_linear_weight, l_linear_bias)
        )  # [batch, samples, 1]
    else:
        har_source = sine_waves.mean(dim=-1, keepdim=True)  # Fallback

    return har_source, torch.zeros_like(har_source), uv_expanded


def main():
    # Load reference
    ref_path = Path("/tmp/kokoro_ref/tensors.npz")
    if not ref_path.exists():
        print("Reference not found")
        return 1
    ref = np.load(ref_path)

    # Get F0 from reference
    f0_np = ref["F0_pred"]  # [1, 126]
    print(f"F0 shape: {f0_np.shape}")
    print(f"F0 range: [{f0_np.min():.4f}, {f0_np.max():.4f}]")
    print(f"F0 mean: {f0_np.mean():.4f}")

    # Load l_linear weights
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    pt_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = pt_ckpt["decoder"]

    l_weight = dec_state["module.generator.m_source.l_linear.weight"]  # [1, 9]
    l_bias = dec_state["module.generator.m_source.l_linear.bias"]  # [1]
    print(
        f"\nl_linear weight shape: {l_weight.shape}, range: [{l_weight.min():.4f}, {l_weight.max():.4f}]"
    )
    print(f"l_linear bias: {l_bias}")

    # Parameters
    upp = 300  # 5 * 5 * 4 * 3 = 300
    sample_rate = 24000

    # ===== PyTorch reference =====
    print("\n=== PyTorch ===")
    f0_pt = torch.tensor(f0_np)
    with torch.no_grad():
        har_pt, noise_pt, uv_pt = pytorch_sine_gen(
            f0_pt, upp, sample_rate, l_linear_weight=l_weight, l_linear_bias=l_bias
        )
    print(f"har shape: {har_pt.shape}")
    print(f"har range: [{har_pt.min():.4f}, {har_pt.max():.4f}]")
    print(f"har mean: {har_pt.mean():.4f}, std: {har_pt.std():.4f}")

    # ===== MLX =====
    print("\n=== MLX ===")
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    f0_mx = mx.array(f0_np)
    har_mx, noise_mx, uv_mx = model.decoder.generator.m_source(f0_mx, upp)
    mx.eval([har_mx, noise_mx, uv_mx])

    print(f"har shape: {har_mx.shape}")
    print(f"har range: [{float(mx.min(har_mx)):.4f}, {float(mx.max(har_mx)):.4f}]")
    print(f"har mean: {float(mx.mean(har_mx)):.4f}, std: {float(mx.std(har_mx)):.4f}")

    # Compare
    har_pt_np = har_pt.numpy().flatten()
    har_mx_np = np.array(har_mx).flatten()

    min_len = min(len(har_pt_np), len(har_mx_np))
    har_pt_np = har_pt_np[:min_len]
    har_mx_np = har_mx_np[:min_len]

    diff = np.abs(har_pt_np - har_mx_np)
    corr = np.corrcoef(har_pt_np, har_mx_np)[0, 1]

    print("\n=== Comparison ===")
    print(f"Max diff: {diff.max():.6e}")
    print(f"Mean diff: {diff.mean():.6e}")
    print(f"Correlation: {corr:.6f}")

    # Detailed trace: check intermediate values for first harmonic
    print("\n=== Detailed trace for harmonic 1 ===")

    # PyTorch
    f0_up = f0_pt.unsqueeze(1)
    f0_up = F.interpolate(f0_up, scale_factor=upp, mode="nearest").squeeze(1)
    h = 1
    rad_values = (f0_up * h / sample_rate) % 1.0
    rad_2d = rad_values.unsqueeze(1)
    rad_down = F.interpolate(
        rad_2d, scale_factor=1 / upp, mode="linear", align_corners=True
    ).squeeze(1)
    phase_low_pt = torch.cumsum(rad_down, dim=1) * 2 * math.pi
    phase_scaled_pt = phase_low_pt * upp
    phase_2d = phase_scaled_pt.unsqueeze(1)
    phase_pt = F.interpolate(
        phase_2d, scale_factor=upp, mode="linear", align_corners=True
    ).squeeze(1)

    print(
        f"PT rad_down: mean={rad_down.mean():.6f}, range=[{rad_down.min():.6f}, {rad_down.max():.6f}]"
    )
    print(
        f"PT phase_low: mean={phase_low_pt.mean():.6f}, range=[{phase_low_pt.min():.6f}, {phase_low_pt.max():.6f}]"
    )
    print(
        f"PT phase: mean={phase_pt.mean():.6f}, range=[{phase_pt.min():.6f}, {phase_pt.max():.6f}]"
    )

    # MLX
    batch, length = f0_np.shape
    samples = length * upp
    f0_up_mx = mx.repeat(mx.array(f0_np)[:, :, None], upp, axis=1).squeeze(-1)
    rad_values_mx = (f0_up_mx * h / sample_rate) % 1.0

    # Manual linear interpolation downsample
    t_down = mx.arange(length) * (samples - 1) / (length - 1)
    t_floor = mx.floor(t_down).astype(mx.int32)
    t_ceil = mx.minimum(t_floor + 1, samples - 1)
    t_frac = t_down - t_floor.astype(mx.float32)
    rad_floor = rad_values_mx[:, t_floor]
    rad_ceil = rad_values_mx[:, t_ceil]
    rad_down_mx = rad_floor * (1 - t_frac) + rad_ceil * t_frac
    mx.eval(rad_down_mx)

    phase_low_mx = mx.cumsum(rad_down_mx, axis=1) * 2 * math.pi
    phase_scaled_mx = phase_low_mx * upp
    mx.eval(phase_scaled_mx)

    # Manual linear interpolation upsample
    t = mx.arange(samples) / upp
    t_floor = mx.floor(t).astype(mx.int32)
    t_ceil = mx.minimum(t_floor + 1, length - 1)
    t_frac = t - t_floor.astype(mx.float32)
    phase_floor = phase_scaled_mx[:, t_floor]
    phase_ceil = phase_scaled_mx[:, t_ceil]
    phase_mx = phase_floor * (1 - t_frac) + phase_ceil * t_frac
    mx.eval(phase_mx)

    print(
        f"MLX rad_down: mean={float(mx.mean(rad_down_mx)):.6f}, range=[{float(mx.min(rad_down_mx)):.6f}, {float(mx.max(rad_down_mx)):.6f}]"
    )
    print(
        f"MLX phase_low: mean={float(mx.mean(phase_low_mx)):.6f}, range=[{float(mx.min(phase_low_mx)):.6f}, {float(mx.max(phase_low_mx)):.6f}]"
    )
    print(
        f"MLX phase: mean={float(mx.mean(phase_mx)):.6f}, range=[{float(mx.min(phase_mx)):.6f}, {float(mx.max(phase_mx)):.6f}]"
    )

    # Compare rad_down
    rad_down_corr = np.corrcoef(
        rad_down.numpy().flatten(), np.array(rad_down_mx).flatten()
    )[0, 1]
    print(f"\nrad_down correlation: {rad_down_corr:.6f}")

    phase_low_corr = np.corrcoef(
        phase_low_pt.numpy().flatten(), np.array(phase_low_mx).flatten()
    )[0, 1]
    print(f"phase_low correlation: {phase_low_corr:.6f}")

    phase_corr = np.corrcoef(phase_pt.numpy().flatten(), np.array(phase_mx).flatten())[
        0, 1
    ]
    print(f"phase correlation: {phase_corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
