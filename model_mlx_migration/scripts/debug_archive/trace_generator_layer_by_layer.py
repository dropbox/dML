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
Kokoro Generator Layer-by-Layer Comparison: PyTorch vs MLX

This script traces intermediate tensors through both PyTorch and MLX
implementations of the Kokoro Generator to find the exact divergence point.

Steps:
1. Load PyTorch model and extract Generator internals via hooks
2. Feed synthetic but deterministic inputs
3. Load MLX model with identical inputs
4. Compare outputs at each layer to find divergence

Usage:
    python scripts/trace_generator_layer_by_layer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import numpy as np
import torch


def mlx_leaky_relu(x, slope=0.01):
    """Leaky ReLU for MLX."""
    return mx.maximum(slope * x, x)


# Seed for reproducibility
SEED = 42


def set_seeds():
    """Set all seeds for reproducibility."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mx.random.seed(SEED)


def find_kokoro_weights() -> Path:
    """Find Kokoro weights."""
    paths = [
        Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth",
    ]
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("Kokoro weights not found")


def load_pytorch_model():
    """Load the PyTorch Kokoro model."""
    try:
        from kokoro import KModel

        model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
        return model
    except ImportError:
        print("ERROR: kokoro package not installed")
        print("Install with: pip install kokoro (Python < 3.14)")
        return None


def load_mlx_model():
    """Load the MLX Kokoro model."""
    from converters.kokoro_converter import KokoroConverter

    converter = KokoroConverter()
    model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
    return model, config


def create_synthetic_inputs(batch=1, length=32, hidden=512, style_dim=256):
    """Create deterministic synthetic inputs for comparison."""
    set_seeds()

    # Features going into Generator
    x_np = np.random.randn(batch, length, hidden).astype(np.float32) * 0.1
    s_np = np.random.randn(batch, style_dim // 2).astype(np.float32) * 0.1  # 128
    f0_np = np.full((batch, length), 200.0, dtype=np.float32)  # Constant 200 Hz

    return {
        "x": x_np,
        "s": s_np,
        "f0": f0_np,
    }


class PyTorchGeneratorTracer:
    """Trace through PyTorch Generator capturing intermediate outputs."""

    def __init__(self, generator):
        self.generator = generator
        self.traces = {}
        self.hooks = []

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.traces[name] = tuple(
                    o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o
                    for o in output
                )
            else:
                self.traces[name] = output.detach().cpu().numpy()

        return hook

    def trace(self, x, s, f0):
        """Run forward pass and collect traces."""
        self.traces = {}

        with torch.no_grad():
            # Manually trace through the generator
            gen = self.generator

            # Store input
            self.traces["input_x"] = x.cpu().numpy()
            self.traces["input_s"] = s.cpu().numpy()
            self.traces["input_f0"] = f0.cpu().numpy()

            # Calculate total upsampling
            upsample_rates = [10, 6]  # Kokoro config
            istft_hop_size = 5
            total_upp = 1
            for r in upsample_rates:
                total_upp *= r
            total_upp *= istft_hop_size  # 300

            self.traces["total_upp"] = total_upp

            # 1. Source module
            # Access the m_source directly
            har_source, noise, uv = gen.m_source(f0, total_upp)
            self.traces["source_har"] = har_source.cpu().numpy()
            self.traces["source_noise"] = noise.cpu().numpy()
            self.traces["source_uv"] = uv.cpu().numpy()

            # 2. STFT on har_source
            har_source_1d = har_source.squeeze(-1)  # [batch, samples]

            # Extract STFT params from generator if available
            getattr(gen, "post_n_fft", 20)
            getattr(gen, "istft_hop_size", 5)

            # Apply STFT (simplified)
            # The actual gen.source_stft would be more accurate but let's trace the shape
            self.traces["har_source_1d_shape"] = list(har_source_1d.shape)

            # 3. Progressive upsampling
            # Trace ups and resblocks
            # For now, capture the final audio
            audio = gen(x, s, f0)
            self.traces["final_audio"] = audio.cpu().numpy()

        return self.traces


def trace_mlx_generator(model, x_np, s_np, f0_np):
    """Trace through MLX Generator capturing intermediate outputs."""
    traces = {}

    x = mx.array(x_np)
    s = mx.array(s_np)
    f0 = mx.array(f0_np)

    traces["input_x"] = x_np
    traces["input_s"] = s_np
    traces["input_f0"] = f0_np

    gen = model.decoder.generator

    # Calculate total upsampling
    upsample_rates = gen.config.istft_upsample_rates  # (10, 6)
    total_upp = 1
    for r in upsample_rates:
        total_upp *= r
    total_upp *= gen.istft_hop_size  # 5
    traces["total_upp"] = total_upp

    # 1. Source module
    har_source, noise, uv = gen.m_source(f0, total_upp)
    mx.eval(har_source, noise, uv)

    traces["source_har"] = np.array(har_source)
    traces["source_noise"] = np.array(noise)
    traces["source_uv"] = np.array(uv)

    # 2. STFT on har_source
    har_source_1d = har_source.squeeze(-1)
    traces["har_source_1d_shape"] = list(har_source_1d.shape)

    # Get source STFT
    source = gen._source_stft(har_source_1d)
    mx.eval(source)
    traces["source_stft"] = np.array(source)

    # 3. Trace through upsampling stages
    # Start with input features
    x_cur = x

    for i in range(gen.num_upsamples):
        up = getattr(gen, f"ups_{i}")
        noise_conv = getattr(gen, f"noise_convs_{i}")
        noise_res = getattr(gen, f"noise_res_{i}")

        x_cur = mlx_leaky_relu(x_cur, 0.1)
        traces[f"after_leaky_relu_{i}"] = np.array(x_cur)

        # x_source before upsampling
        x_source = noise_conv(source)
        traces[f"noise_conv_{i}_out"] = np.array(x_source)

        x_source = noise_res(x_source, s)
        mx.eval(x_source)
        traces[f"noise_res_{i}_out"] = np.array(x_source)

        # Upsample x
        x_cur = up(x_cur)
        mx.eval(x_cur)
        traces[f"ups_{i}_out"] = np.array(x_cur)

        # Reflection pad on last stage
        if i == gen.num_upsamples - 1:
            x_cur = mx.concatenate([x_cur[:, 1:2, :], x_cur], axis=1)
            traces[f"after_reflect_pad_{i}"] = np.array(x_cur)

        # Length alignment
        if x_source.shape[1] < x_cur.shape[1]:
            pad_len = x_cur.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x_cur.shape[1]:
            x_source = x_source[:, : x_cur.shape[1], :]

        traces[f"x_source_aligned_{i}"] = np.array(x_source)
        traces[f"x_before_add_{i}"] = np.array(x_cur)

        x_cur = x_cur + x_source
        mx.eval(x_cur)
        traces[f"after_source_add_{i}"] = np.array(x_cur)

        # ResBlocks
        xs = None
        for j in range(gen.num_kernels):
            block_idx = i * gen.num_kernels + j
            if block_idx < gen._num_resblocks:
                resblock = getattr(gen, f"resblocks_{block_idx}")
                if xs is None:
                    xs = resblock(x_cur, s)
                else:
                    xs = xs + resblock(x_cur, s)

        if xs is not None:
            x_cur = xs / gen.num_kernels
        mx.eval(x_cur)
        traces[f"after_resblocks_{i}"] = np.array(x_cur)

    # Final conv_post
    x_cur = mlx_leaky_relu(x_cur, 0.1)
    traces["before_conv_post"] = np.array(x_cur)

    x_cur = gen.conv_post(x_cur)
    mx.eval(x_cur)
    traces["conv_post_out"] = np.array(x_cur)

    # Mag/Phase extraction
    n_bins = gen.post_n_fft // 2 + 1  # 11
    log_mag = mx.clip(x_cur[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    phase_logits = x_cur[..., n_bins:]
    phase = mx.sin(phase_logits)

    mx.eval(log_mag, mag, phase_logits, phase)
    traces["log_mag"] = np.array(log_mag)
    traces["mag"] = np.array(mag)
    traces["phase_logits"] = np.array(phase_logits)
    traces["phase"] = np.array(phase)

    # ISTFT synthesis
    audio = gen._istft_synthesis(mag, phase)
    mx.eval(audio)
    traces["final_audio"] = np.array(audio)

    return traces


def compare_traces(pt_traces, mlx_traces):
    """Compare traces and find divergence."""
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 70)

    results = []

    # Compare common keys
    common_keys = set(pt_traces.keys()) & set(mlx_traces.keys())

    for key in sorted(common_keys):
        pt_val = pt_traces[key]
        mlx_val = mlx_traces[key]

        if isinstance(pt_val, (int, float)):
            match = pt_val == mlx_val
            print(f"{key}: PT={pt_val}, MLX={mlx_val}, Match={match}")
            results.append({"key": key, "match": match})
            continue

        if isinstance(pt_val, list):
            match = pt_val == mlx_val
            print(f"{key}: PT={pt_val}, MLX={mlx_val}, Match={match}")
            results.append({"key": key, "match": match})
            continue

        if not isinstance(pt_val, np.ndarray) or not isinstance(mlx_val, np.ndarray):
            continue

        # Compare shapes
        shape_match = pt_val.shape == mlx_val.shape

        if not shape_match:
            print(f"\n{key}:")
            print(f"  Shape mismatch: PT={pt_val.shape}, MLX={mlx_val.shape}")
            results.append({"key": key, "match": False, "reason": "shape_mismatch"})
            continue

        # Numerical comparison
        max_diff = np.abs(pt_val - mlx_val).max()
        mean_diff = np.abs(pt_val - mlx_val).mean()

        # Check correlation
        try:
            corr = np.corrcoef(pt_val.flatten(), mlx_val.flatten())[0, 1]
        except Exception:
            corr = float("nan")

        status = "MATCH" if max_diff < 1e-3 else "DIFF"

        print(f"\n{key}:")
        print(f"  Shape: {pt_val.shape}")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  Correlation: {corr:.6f}")
        print(f"  PT range: [{pt_val.min():.4f}, {pt_val.max():.4f}]")
        print(f"  MLX range: [{mlx_val.min():.4f}, {mlx_val.max():.4f}]")
        print(f"  Status: {status}")

        results.append(
            {
                "key": key,
                "match": max_diff < 1e-3,
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "correlation": float(corr) if not np.isnan(corr) else None,
            }
        )

    # Find first divergence
    print("\n" + "=" * 70)
    print("DIVERGENCE ANALYSIS")
    print("=" * 70)

    diverged = [r for r in results if not r.get("match", True)]
    if diverged:
        print(f"\nFirst divergence at: {diverged[0]['key']}")
        for d in diverged:
            if "max_diff" in d:
                print(f"  {d['key']}: max_diff={d['max_diff']:.6e}")
    else:
        print("\nAll layers match within tolerance!")

    return results


def analyze_source_module_diff():
    """Deep dive into SourceModule differences."""
    print("\n" + "=" * 70)
    print("SOURCE MODULE DEEP ANALYSIS")
    print("=" * 70)

    set_seeds()

    # Create identical F0 input
    f0_np = np.full((1, 32), 200.0, dtype=np.float32)
    torch.tensor(f0_np)
    f0_mlx = mx.array(f0_np)

    upp = 300  # total upsampling

    # MLX SourceModule
    from converters.models.kokoro import SourceModule

    mlx_source = SourceModule(sample_rate=24000)

    har_mlx, noise_mlx, uv_mlx = mlx_source(f0_mlx, upp)
    mx.eval(har_mlx, noise_mlx, uv_mlx)

    print(f"\nF0 input: {f0_np.shape}, value={f0_np[0, 0]} Hz")
    print(f"Upsampling factor: {upp}")
    print("\nMLX SourceModule output:")
    print(f"  har shape: {har_mlx.shape}")
    print(f"  har range: [{float(har_mlx.min()):.4f}, {float(har_mlx.max()):.4f}]")
    print(f"  har mean: {float(mx.mean(har_mlx)):.4f}")
    print(f"  uv mean: {float(mx.mean(uv_mlx)):.4f} (should be 1.0 for voiced)")

    # Check frequency content of har_source
    har_np = np.array(har_mlx.squeeze())

    # FFT analysis
    from scipy.fft import fft, fftfreq

    n_samples = len(har_np)
    sample_rate = 24000

    spectrum = np.abs(fft(har_np))[: n_samples // 2]
    freqs = fftfreq(n_samples, 1 / sample_rate)[: n_samples // 2]

    # Find dominant frequency
    dom_idx = np.argmax(spectrum)
    dom_freq = freqs[dom_idx]

    print("\nHarmonic source frequency analysis:")
    print(f"  Dominant frequency: {dom_freq:.1f} Hz")
    print("  Expected (F0): 200.0 Hz")

    # Check if harmonics are present
    print("\nSpectrum at harmonic frequencies:")
    for h in range(1, 10):
        target_freq = 200 * h
        idx = int(target_freq * n_samples / sample_rate)
        if idx < len(spectrum):
            print(f"  {h}f ({target_freq} Hz): magnitude = {spectrum[idx]:.4f}")


def main():
    print("=" * 70)
    print("KOKORO GENERATOR LAYER-BY-LAYER TRACE COMPARISON")
    print("=" * 70)

    # Analyze source module
    analyze_source_module_diff()

    # Load MLX model
    print("\nLoading MLX model...")
    mlx_model, config = load_mlx_model()
    print("MLX model loaded")

    # Create synthetic inputs
    inputs = create_synthetic_inputs()

    # Trace MLX Generator
    print("\nTracing MLX Generator...")
    mlx_traces = trace_mlx_generator(mlx_model, inputs["x"], inputs["s"], inputs["f0"])

    # Summary of MLX traces
    print("\n" + "=" * 70)
    print("MLX GENERATOR TRACE SUMMARY")
    print("=" * 70)

    for key in sorted(mlx_traces.keys()):
        val = mlx_traces[key]
        if isinstance(val, np.ndarray):
            print(f"{key}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")
        else:
            print(f"{key}: {val}")

    # Try to load PyTorch model for comparison
    print("\n" + "=" * 70)
    print("PYTORCH COMPARISON (if available)")
    print("=" * 70)

    pt_model = load_pytorch_model()
    if pt_model is not None:
        print("PyTorch model loaded, running comparison...")

        x_pt = torch.tensor(inputs["x"])
        s_pt = torch.tensor(inputs["s"])
        f0_pt = torch.tensor(inputs["f0"])

        tracer = PyTorchGeneratorTracer(pt_model.decoder.generator)
        pt_traces = tracer.trace(x_pt, s_pt, f0_pt)

        compare_traces(pt_traces, mlx_traces)
    else:
        print("PyTorch model not available, skipping comparison")
        print("Install kokoro package in Python < 3.14 to enable")

    # Save traces for external analysis
    output_path = Path("reports/main/generator_traces.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert traces to saveable format
    save_dict = {}
    for k, v in mlx_traces.items():
        if isinstance(v, np.ndarray):
            save_dict[f"mlx_{k}"] = v
        elif isinstance(v, (int, float)):
            save_dict[f"mlx_{k}"] = np.array([v])

    np.savez_compressed(output_path, **save_dict)
    print(f"\nTraces saved to {output_path}")

    print("\n" + "=" * 70)
    print("TRACE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
