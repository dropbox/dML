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
Kokoro Debug Script - Export intermediate tensors for C++ comparison.

Runs PyTorch inference on "Hello world" and exports key intermediate values.
"""

import sys
from pathlib import Path

# Add tools path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_PATH = Path.home() / "model_mlx_migration" / "kokoro_cpp_export"


def load_weights():
    """Load safetensors weights."""
    weights = {}
    with safe_open(MODEL_PATH / "weights.safetensors", framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def print_tensor_stats(name: str, t: torch.Tensor):
    """Print tensor statistics for comparison."""
    t_np = t.detach().cpu().numpy().flatten()
    print(f"{name}:")
    print(f"  shape: {list(t.shape)}")
    print(f"  dtype: {t.dtype}")
    print(f"  min: {t_np.min():.6f}")
    print(f"  max: {t_np.max():.6f}")
    print(f"  mean: {t_np.mean():.6f}")
    print(f"  std: {t_np.std():.6f}")
    print(f"  first_5: {t_np[:5]}")
    print()


def conv1d_wn(x, weight, bias, stride=1, padding=0):
    """Conv1d with WeightNorm (pre-folded)."""
    # x: [batch, time, channels] -> [batch, channels, time]
    x_t = x.transpose(1, 2)
    out = F.conv1d(x_t, weight, bias, stride=stride, padding=padding)
    # -> [batch, time, channels]
    return out.transpose(1, 2)


def conv_transpose1d_wn(x, weight, bias, stride=1, padding=0):
    """ConvTranspose1d with WeightNorm (pre-folded)."""
    # x: [batch, time, channels] -> [batch, channels, time]
    x_t = x.transpose(1, 2)
    out = F.conv_transpose1d(x_t, weight, bias, stride=stride, padding=padding)
    # -> [batch, time, channels]
    return out.transpose(1, 2)


def adain(x, style, fc_weight, fc_bias):
    """Adaptive Instance Normalization."""
    # x: [batch, time, channels]
    # style: [batch, style_dim]

    # Get gamma/beta from style
    fc_out = F.linear(style, fc_weight, fc_bias)  # [batch, 2*channels]
    channels = x.shape[-1]
    gamma = fc_out[:, :channels].unsqueeze(1)  # [batch, 1, channels]
    beta = fc_out[:, channels:].unsqueeze(1)  # [batch, 1, channels]

    # Instance norm
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + 1e-5)

    return gamma * x_norm + beta


def source_stft(x, n_fft=20, hop=5):
    """STFT for source signal."""
    # x: [batch, samples]
    window = torch.hann_window(n_fft, device=x.device)

    # Pad signal (center=True, pad_mode=reflect)
    pad_amount = n_fft // 2
    x_padded = F.pad(x, (pad_amount, pad_amount), mode="reflect")

    # STFT
    spec = torch.stft(x_padded, n_fft, hop, n_fft, window, return_complex=True)
    # spec: [batch, n_fft//2+1, frames]

    mag = torch.abs(spec)
    phase = torch.angle(spec)

    # Concatenate: [batch, 22, frames] -> [batch, frames, 22]
    har = torch.cat([mag, phase], dim=1).transpose(1, 2)
    return har


def run_generator(x, style, f0, weights):
    """
    Run Generator forward pass with debug output.

    x: [batch, length, 512] - decoder output
    style: [batch, 128] - style vector
    f0: [batch, f0_frames] - F0 values
    """
    print("=== Generator Debug ===")
    print_tensor_stats("generator_input_x", x)
    print_tensor_stats("generator_input_style", style)
    print_tensor_stats("generator_input_f0", f0)

    # Parameters
    upsample_rates = [10, 6]
    total_upp = 10 * 6 * 5  # 300

    # --- Source module ---
    # Upsample F0
    f0_up = F.interpolate(
        f0.unsqueeze(1), scale_factor=total_upp, mode="nearest"
    ).squeeze(1)
    print_tensor_stats("f0_upsampled", f0_up)

    # Generate harmonics (9 harmonics from l_linear shape [1, 9])
    harmonic_num = 8  # l_linear.weight is [1, 9] -> 9 harmonics
    batch, length = f0_up.shape

    # Harmonic multipliers [1, 2, 3, ..., 9]
    harmonics = torch.arange(1, harmonic_num + 2, dtype=f0_up.dtype).view(1, 1, -1)

    # F0 * harmonics for each harmonic
    fn = f0_up.unsqueeze(-1) * harmonics  # [batch, length, 9]

    # Phase accumulation
    rad = (fn / 24000.0) % 1.0

    # Add random initial phase
    rand_ini = torch.rand(batch, 1, harmonic_num + 1)
    rad[:, 0:1, :] = rad[:, 0:1, :] + rand_ini

    phase = torch.cumsum(rad, dim=1) * 2 * 3.14159265
    sine_waves = torch.sin(phase) * 0.1  # sine_amp = 0.1

    # UV mask (voiced/unvoiced)
    uv = (f0_up > 0).float().unsqueeze(-1)  # [batch, length, 1]
    sine_waves = sine_waves * uv

    # Linear projection (m_source.l_linear)
    l_linear_w = weights["decoder.generator.m_source.l_linear.weight"]  # [1, 9]
    l_linear_b = weights["decoder.generator.m_source.l_linear.bias"]  # [1]
    har_source = torch.tanh(
        F.linear(sine_waves, l_linear_w, l_linear_b)
    )  # [batch, samples, 1]
    print_tensor_stats("har_source", har_source)

    # STFT of source
    source = source_stft(har_source.squeeze(-1))  # [batch, frames, 22]
    print_tensor_stats("source_stft", source)

    # --- Upsampling stages ---
    for i in range(2):
        print(f"\n--- Upsample stage {i} ---")

        # LeakyReLU
        x = F.leaky_relu(x, 0.1)

        # noise_conv
        nc_w = weights[f"decoder.generator.noise_convs_{i}.weight"]
        nc_b = weights[f"decoder.generator.noise_convs_{i}.bias"]

        if i == 0:
            stride = 6
            x_source = F.conv1d(
                source.transpose(1, 2), nc_w, nc_b, stride=stride, padding=3
            ).transpose(1, 2)
        else:
            stride = 1
            x_source = F.conv1d(
                source.transpose(1, 2), nc_w, nc_b, stride=stride, padding=0
            ).transpose(1, 2)

        print_tensor_stats(f"x_source_after_noise_conv_{i}", x_source)

        # noise_res (simplified - just passthrough for debug)
        # In full model this is AdaINResBlock1dStyled

        # Upsample
        up_w = weights[f"decoder.generator.ups_{i}.weight"]
        up_b = weights[f"decoder.generator.ups_{i}.bias"]
        rate = upsample_rates[i]
        kernel = up_w.shape[2]
        padding = (kernel - rate) // 2

        x = conv_transpose1d_wn(x, up_w, up_b, stride=rate, padding=padding)
        print_tensor_stats(f"x_after_ups_{i}", x)

        # Reflection pad at last stage
        if i == 1:
            x = F.pad(x.transpose(1, 2), (1, 0), mode="reflect").transpose(1, 2)

        # Skip x_source add for debug (lengths may not match in simplified version)

        # Skip resblocks for simplicity

    # conv_post
    print("\n--- conv_post ---")
    x = F.leaky_relu(x, 0.01)  # Default slope
    post_w = weights["decoder.generator.conv_post.weight"]  # [22, 128, 7]
    post_b = weights["decoder.generator.conv_post.bias"]
    spec = conv1d_wn(x, post_w, post_b, padding=3)
    print_tensor_stats("conv_post_output", spec)

    # Split mag/phase
    n_bins = 11
    log_mag = spec[..., :n_bins]
    phase_logits = spec[..., n_bins:]

    mag = torch.exp(log_mag)
    phase = torch.sin(phase_logits)

    print_tensor_stats("istft_mag", mag)
    print_tensor_stats("istft_phase", phase)

    # ISTFT
    n_fft = 20
    hop_size = 5

    # Complex spectrum
    spectrum = mag * torch.exp(1j * phase)

    # IRFFT
    time_frames = torch.fft.irfft(spectrum, n=n_fft, dim=-1)
    print_tensor_stats("time_frames", time_frames)

    # Hann window
    window = torch.hann_window(n_fft)
    time_frames = time_frames * window

    # Overlap-add
    frames = time_frames.shape[1]
    output_length = (frames - 1) * hop_size

    audio = torch.zeros(1, output_length + n_fft)
    window_sum = torch.zeros(output_length + n_fft)

    for t in range(frames):
        start = t * hop_size
        audio[0, start : start + n_fft] += time_frames[0, t]
        window_sum[start : start + n_fft] += window**2

    # Normalize
    window_sum = torch.clamp(window_sum, min=1e-8)
    audio = audio / window_sum

    # Trim
    pad = n_fft // 2
    audio = audio[:, pad : pad + output_length]

    print_tensor_stats("final_audio", audio)

    return audio


def main():
    print("=" * 60)
    print("Kokoro Generator Debug - Intermediate Tensors")
    print("=" * 60)

    # Load weights
    print("\nLoading weights...")
    weights = load_weights()
    print(f"Loaded {len(weights)} weight tensors")

    # Load voice style
    with safe_open(MODEL_PATH / "voices" / "af_bella.safetensors", framework="pt") as f:
        style = f.get_tensor("embedding")  # [1, 256]
    print_tensor_stats("voice_style", style)

    # Create test inputs (simulating decoder output)
    # In real model this comes from decoder.decode_3 output
    # For debug, use small random input
    batch = 1
    length = 64  # Frames
    x = torch.randn(batch, length, 512) * 0.1  # Small random values

    # Create F0 (constant 200 Hz for simplicity)
    f0_frames = length // 2
    f0 = torch.full((batch, f0_frames), 200.0)

    print("\n" + "=" * 60)
    print("Running Generator...")
    print("=" * 60)

    with torch.no_grad():
        run_generator(x, style, f0, weights)

    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
