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
Build minimal PyTorch Kokoro inference to get ground truth audio.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load checkpoint
ckpt = torch.load(
    "/Users/ayates/models/kokoro/kokoro-v1_0.pth", map_location="cpu", weights_only=True
)


class TorchSTFT(nn.Module):
    def __init__(self, filter_length=20, hop_length=5, win_length=20):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(input_data.device),
            return_complex=True,
        )
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )
        return inverse_transform.unsqueeze(-2)


class SineGen(nn.Module):
    def __init__(
        self,
        samp_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_values = F.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        sines = torch.sin(phase)
        return sines

    def forward(self, f0):
        fn = torch.multiply(
            f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
        )
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshod,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


def main():
    print("=== PyTorch Reference Inference ===")

    # Create source module
    upsample_rates = [10, 6]
    gen_istft_hop_size = 5
    total_upsample = math.prod(upsample_rates) * gen_istft_hop_size

    m_source = SourceModuleHnNSF(
        sampling_rate=24000,
        upsample_scale=total_upsample,
        harmonic_num=8,
        voiced_threshod=10,
    )

    # Load source module weights
    pt_gen = ckpt["decoder"]
    m_source.l_linear.weight.data = pt_gen["module.generator.m_source.l_linear.weight"]
    m_source.l_linear.bias.data = pt_gen["module.generator.m_source.l_linear.bias"]

    # Create STFT
    stft = TorchSTFT(filter_length=20, hop_length=5, win_length=20)

    # Test F0 input (similar to what predictor outputs)
    torch.manual_seed(42)
    f0 = torch.abs(torch.randn(1, 28)) * 100 + 200  # 28 frames of F0

    # Upsample F0
    f0_up = F.interpolate(
        f0[:, None], scale_factor=total_upsample, mode="nearest"
    ).transpose(1, 2)

    print(f"F0: shape={f0.shape}, mean={f0.mean():.2f}Hz")
    print(f"F0 upsampled: shape={f0_up.shape}")

    # Generate harmonic source
    har_source, noise_sig, uv = m_source(f0_up)
    print(f"\nHarmonic source: shape={har_source.shape}")
    print(f"  mean={har_source.mean():.6f}, std={har_source.std():.6f}")
    print(f"  range=[{har_source.min():.4f}, {har_source.max():.4f}]")

    # Apply STFT
    har_1d = har_source.transpose(1, 2).squeeze(1)
    print(f"\nHarmonic 1D: shape={har_1d.shape}")

    har_spec, har_phase = stft.transform(har_1d)
    print(f"STFT magnitude: shape={har_spec.shape}, mean={har_spec.mean():.6f}")
    print(f"STFT phase: shape={har_phase.shape}, mean={har_phase.mean():.4f}")

    har = torch.cat([har_spec, har_phase], dim=1)
    print(f"Source (har): shape={har.shape}")  # Should be [1, 22, frames]

    # Test ISTFT with synthetic data
    print("\n=== Testing ISTFT ===")
    # Create synthetic log_mag similar to what generator outputs
    frames = 1680
    n_bins = 11
    log_mag = torch.randn(1, n_bins, frames) * 2 - 13  # Similar to generator output
    phase_logits = torch.randn(1, n_bins, frames)

    spec = torch.exp(log_mag)
    phase = torch.sin(phase_logits)

    print(
        f"log_mag: mean={log_mag.mean():.4f}, range=[{log_mag.min():.4f}, {log_mag.max():.4f}]"
    )
    print(f"spec (exp): mean={spec.mean():.6f}")
    print(f"phase (sin): mean={phase.mean():.4f}")

    audio = stft.inverse(spec, phase)
    print(f"\nPyTorch ISTFT output: shape={audio.shape}")
    print(f"  mean={audio.mean():.6f}")
    print(f"  RMS={(audio**2).mean().sqrt():.6f}")
    print(f"  range=[{audio.min():.4f}, {audio.max():.4f}]")

    # Test with higher magnitudes (log_mag around 0)
    print("\n=== Test with realistic magnitudes ===")
    log_mag_high = torch.randn(1, n_bins, frames)  # Mean ~0
    spec_high = torch.exp(log_mag_high)
    audio_high = stft.inverse(spec_high, phase)
    print(f"log_mag_high: mean={log_mag_high.mean():.4f}")
    print(f"spec_high (exp): mean={spec_high.mean():.6f}")
    print(f"Audio RMS: {(audio_high**2).mean().sqrt():.6f}")
    print(f"Audio range: [{audio_high.min():.4f}, {audio_high.max():.4f}]")


if __name__ == "__main__":
    main()
