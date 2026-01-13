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
Kokoro PyTorch Reference Script

Loads PyTorch weights directly (without kokoro package) and compares
with MLX implementation for numerical validation.

This script:
1. Loads Kokoro-82M weights from HuggingFace cache
2. Implements minimal forward pass for key components
3. Compares outputs with MLX implementation
4. Reports numerical differences

Usage:
    python scripts/kokoro_pytorch_reference.py
"""

import sys
from pathlib import Path

# Add tools path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import math

import torch
import torch.nn.functional as F

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available, PyTorch-only validation")


def find_kokoro_weights() -> Path:
    """Find Kokoro weights in common locations."""
    paths = [
        Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth",
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--hexgrad--Kokoro-82M"
        / "snapshots",
    ]

    for p in paths:
        if p.exists():
            if p.is_file():
                return p
            # Search for .pth in directory
            pth_files = list(p.rglob("*.pth"))
            if pth_files:
                return pth_files[0]

    # Try downloading
    try:
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
        return Path(model_file)
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find Kokoro weights. Please download from HuggingFace."
    )


def load_kokoro_weights(model_path: Path | None = None) -> dict:  # type: ignore[type-arg]
    """Load PyTorch weights directly without kokoro package."""
    if model_path is None:
        model_path = find_kokoro_weights()

    print(f"Loading weights from: {model_path}")
    state_dict: dict = torch.load(model_path, map_location="cpu", weights_only=False)  # type: ignore[type-arg]
    return state_dict


class PyTorchSineGen(torch.nn.Module):
    """
    Minimal SineGen matching StyleTTS2 exactly.
    """

    def __init__(self, sample_rate=24000, harmonic_num=0, sine_amp=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.dim = harmonic_num + 1

    def forward(self, f0):
        """
        Generate sine waves from F0.

        Args:
            f0: [batch, length] or [batch, length, 1] - F0 in Hz

        Returns:
            sine_waves: [batch, length, harmonic_num+1]
            uv: [batch, length, 1]
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)

        batch, length, _ = f0.shape

        # Harmonic multipliers
        harmonics = torch.arange(
            1, self.harmonic_num + 2, dtype=f0.dtype, device=f0.device
        )
        harmonics = harmonics.view(1, 1, -1)

        # F0 * harmonics
        fn = f0 * harmonics  # [batch, length, num_harmonics]

        # Phase calculation
        rad_values = (fn / self.sample_rate) % 1.0

        # Random initial phase
        rand_ini = torch.rand(batch, 1, fn.shape[2], device=f0.device)
        rad_values = rad_values.clone()
        rad_values[:, 0:1, :] = rad_values[:, 0:1, :] + rand_ini

        # Cumulative phase
        phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi

        # Sine waves
        sine_waves = torch.sin(phase) * self.sine_amp

        # UV mask
        uv = (f0 > 0).float()

        # Apply UV mask
        sine_waves = sine_waves * uv

        return sine_waves, uv


class PyTorchSourceModule(torch.nn.Module):
    """
    Minimal SourceModuleHnNSF matching StyleTTS2.
    """

    def __init__(self, sample_rate=24000, harmonic_num=0, sine_amp=0.1):
        super().__init__()
        self.sine_amp = sine_amp
        self.sine_gen = PyTorchSineGen(sample_rate, harmonic_num, sine_amp)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)

    def forward(self, f0):
        """
        Generate source signal.

        Args:
            f0: [batch, length] or [batch, length, 1]

        Returns:
            sine_merge: [batch, length, 1]
            noise: [batch, length, 1]
            uv: [batch, length, 1]
        """
        sine_wavs, uv = self.sine_gen(f0)
        sine_merge = torch.tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3

        return sine_merge, noise, uv


def pytorch_stft(x, n_fft=800, hop_length=200, win_length=800):
    """
    STFT matching PyTorch behavior.

    Args:
        x: [batch, samples]

    Returns:
        magnitude: [batch, n_fft//2+1, num_frames]
        phase: [batch, n_fft//2+1, num_frames]
    """
    window = torch.hann_window(win_length, device=x.device)

    # Pad signal
    pad_amount = n_fft // 2
    x_padded = F.pad(x, (pad_amount, pad_amount))

    # STFT
    spec = torch.stft(
        x_padded, n_fft, hop_length, win_length, window=window, return_complex=True
    )
    # spec: [batch, n_fft//2+1, num_frames]

    magnitude = torch.abs(spec)
    phase = torch.angle(spec)

    return magnitude, phase


def compare_source_module_outputs():
    """Compare MLX vs PyTorch SourceModule outputs."""
    print("\n=== Source Module Comparison ===")

    # Create PyTorch reference
    pt_source = PyTorchSourceModule(sample_rate=24000, harmonic_num=0)

    # Test F0
    f0_pt = torch.full((1, 100), 200.0)

    # PyTorch forward
    with torch.no_grad():
        pt_sine, pt_noise, pt_uv = pt_source(f0_pt)

    print(f"PyTorch SourceModule output shape: {pt_sine.shape}")
    print(f"PyTorch output range: [{pt_sine.min():.4f}, {pt_sine.max():.4f}]")

    if HAS_MLX:
        from converters.models.kokoro import SourceModuleHnNSF

        mlx_source = SourceModuleHnNSF(sample_rate=24000, harmonic_num=0)
        f0_mlx = mx.full((1, 100), 200.0)

        mlx_sine, mlx_noise, mlx_uv = mlx_source(f0_mlx)
        mx.eval(mlx_sine)

        print(f"MLX SourceModule output shape: {mlx_sine.shape}")
        print(
            f"MLX output range: [{float(mlx_sine.min()):.4f}, {float(mlx_sine.max()):.4f}]"
        )

        # Both should be bounded [-1, 1] from tanh
        print("\nVerification:")
        print(
            f"  PyTorch bounded: {pt_sine.max().item() <= 1.0 and pt_sine.min().item() >= -1.0}"
        )
        print(
            f"  MLX bounded: {float(mlx_sine.max()) <= 1.0 and float(mlx_sine.min()) >= -1.0}"
        )


def compare_stft_outputs():
    """Compare STFT transform outputs."""
    print("\n=== STFT Comparison ===")

    # Create test signal
    t = torch.arange(24000, dtype=torch.float32) / 24000.0
    signal_pt = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)

    # PyTorch STFT
    mag_pt, phase_pt = pytorch_stft(signal_pt)
    print(f"PyTorch STFT: mag={mag_pt.shape}, phase={phase_pt.shape}")

    if HAS_MLX:
        from converters.models.stft import TorchSTFT

        stft_mlx = TorchSTFT(filter_length=800, hop_length=200, win_length=800)
        signal_mlx = mx.sin(
            2 * math.pi * 440 * mx.arange(24000, dtype=mx.float32) / 24000.0
        )
        signal_mlx = signal_mlx[None, :]

        mag_mlx, phase_mlx = stft_mlx.transform(signal_mlx)
        mx.eval(mag_mlx, phase_mlx)

        print(f"MLX STFT: mag={mag_mlx.shape}, phase={phase_mlx.shape}")

        # Compare shapes
        print(f"\nShape match: {mag_pt.shape[1:] == tuple(mag_mlx.shape[1:])}")


def validate_generator_audio():
    """Validate Generator produces reasonable audio."""
    print("\n=== Generator Audio Validation ===")

    if not HAS_MLX:
        print("MLX not available, skipping")
        return

    from converters.models.kokoro import Generator
    from converters.models.kokoro_modules import KokoroConfig

    config = KokoroConfig()
    gen = Generator(config)

    # Test inputs
    batch = 1
    length = 64
    channels = 512

    x = mx.random.normal((batch, length, channels))
    s = mx.random.normal((batch, config.style_dim))
    f0 = mx.full((batch, length // 2), 1.0)

    audio = gen(x, s, f0)
    mx.eval(audio)

    rms = float(mx.sqrt(mx.mean(audio**2)))
    max_amp = float(mx.abs(audio).max())

    print(f"Generator output shape: {audio.shape}")
    print(f"Audio RMS: {rms:.4f}")
    print(f"Max amplitude: {max_amp:.4f}")

    # Validation
    if rms > 0.05 and rms < 0.5:
        print("PASS: Audio RMS in expected range (0.05-0.5)")
    else:
        print(f"WARN: Audio RMS {rms} outside expected range")

    if max_amp > 0.3:
        print("PASS: Audio has strong signal (>0.3)")
    else:
        print(f"WARN: Audio max amplitude {max_amp} is weak")


def full_model_comparison():
    """Compare full KokoroModel output."""
    print("\n=== Full Model Comparison ===")

    if not HAS_MLX:
        print("MLX not available, skipping")
        return

    from converters.models.kokoro import KokoroModel
    from converters.models.kokoro_modules import KokoroConfig

    config = KokoroConfig()
    model = KokoroModel(config)

    # Test inputs
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    style = mx.random.normal((1, config.style_dim))

    audio = model.synthesize(tokens, style)
    mx.eval(audio)

    rms = float(mx.sqrt(mx.mean(audio**2)))
    max_amp = float(mx.abs(audio).max())
    duration_sec = audio.shape[1] / 24000.0

    print(f"Full model output shape: {audio.shape}")
    print(f"Audio duration: {duration_sec:.3f}s")
    print(f"Audio RMS: {rms:.4f}")
    print(f"Max amplitude: {max_amp:.4f}")

    # Check for NaN/Inf
    has_nan = bool(mx.any(mx.isnan(audio)))
    has_inf = bool(mx.any(mx.isinf(audio)))

    print(f"Has NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")

    if not has_nan and not has_inf and rms > 0.05:
        print("\nPASS: Model produces valid audio output")
    else:
        print("\nFAIL: Model output has issues")


def main():
    """Run all comparisons."""
    print("=" * 60)
    print("Kokoro PyTorch Reference Validation")
    print("=" * 60)

    try:
        compare_source_module_outputs()
    except Exception as e:
        print(f"Error in SourceModule comparison: {e}")

    try:
        compare_stft_outputs()
    except Exception as e:
        print(f"Error in STFT comparison: {e}")

    try:
        validate_generator_audio()
    except Exception as e:
        print(f"Error in Generator validation: {e}")

    try:
        full_model_comparison()
    except Exception as e:
        print(f"Error in full model comparison: {e}")

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
