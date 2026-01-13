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
CosyVoice3 Vocoder Profiler

Profile the CausalHiFT vocoder to find bottlenecks.
Current: 0.3x RTF (slower than real-time)
Target: 50x+ RTF
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


def profile_vocoder():
    print("=" * 70)
    print("CosyVoice3 Vocoder Profiler")
    print("=" * 70)

    # Load vocoder
    print("\n1. Loading vocoder...")
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import (
        CausalHiFTGenerator, create_cosyvoice3_vocoder_config
    )

    vocoder_config = create_cosyvoice3_vocoder_config()
    vocoder = CausalHiFTGenerator(vocoder_config)

    weights = mx.load('models/cosyvoice3_mlx/model.safetensors')
    vocoder_weights = {k[8:]: v for k, v in weights.items() if k.startswith('vocoder.')}
    vocoder.load_weights(list(vocoder_weights.items()))
    mx.eval(vocoder.parameters())

    print(f"   Vocoder loaded: {len(vocoder_weights)} tensors")

    # Test parameters
    sample_rate = 24000
    mel_channels = 80
    hop_size = 200  # CosyVoice uses 200 samples per mel frame

    # ==========================================================================
    # Test 1: Different input sizes
    # ==========================================================================
    print("\n2. Testing different input sizes...")

    test_sizes = [
        ("1s audio", 120),   # 120 mel frames ≈ 1s
        ("2s audio", 240),   # 240 mel frames ≈ 2s
        ("5s audio", 600),   # 600 mel frames ≈ 5s
        ("10s audio", 1200), # 1200 mel frames ≈ 10s
    ]

    for name, mel_frames in test_sizes:
        mel = mx.random.normal((1, mel_channels, mel_frames))
        mx.eval(mel)

        # Warmup
        for _ in range(2):
            audio = vocoder(mel)
            mx.eval(audio)

        times = []
        for _ in range(5):
            start = time.time()
            audio = vocoder(mel)
            mx.eval(audio)
            times.append(time.time() - start)

        avg = np.mean(times[1:])
        audio_duration = (mel_frames * hop_size) / sample_rate
        rtf = audio_duration / avg

        print(f"   {name} ({mel_frames} frames): {avg*1000:.1f}ms, {rtf:.1f}x RTF")

    # ==========================================================================
    # Test 2: Profile individual components
    # ==========================================================================
    print("\n3. Profiling individual components...")

    mel = mx.random.normal((1, mel_channels, 240))  # 2s audio
    mx.eval(mel)

    # Profile conv_pre
    times = []
    for _ in range(10):
        start = time.time()
        x = vocoder.conv_pre(mel)
        mx.eval(x)
        times.append(time.time() - start)
    print(f"   conv_pre: {np.mean(times[3:])*1000:.2f}ms")

    # Profile ups + resblocks
    x = vocoder.conv_pre(mel)
    mx.eval(x)

    for i, (up, resblock_group) in enumerate(zip(vocoder.ups, vocoder.resblocks)):
        # Profile upsampling
        times = []
        for _ in range(10):
            start = time.time()
            x_up = up(x)
            mx.eval(x_up)
            times.append(time.time() - start)
        up_time = np.mean(times[3:])

        # Profile resblocks
        times = []
        for _ in range(10):
            start = time.time()
            x_res = resblock_group(x_up)
            mx.eval(x_res)
            times.append(time.time() - start)
        res_time = np.mean(times[3:])

        print(f"   up[{i}] + resblocks: {up_time*1000:.2f}ms + {res_time*1000:.2f}ms = {(up_time+res_time)*1000:.2f}ms")

        x = resblock_group(up(x))
        mx.eval(x)

    # Profile conv_post
    times = []
    for _ in range(10):
        start = time.time()
        audio = vocoder.conv_post(x)
        mx.eval(audio)
        times.append(time.time() - start)
    print(f"   conv_post: {np.mean(times[3:])*1000:.2f}ms")

    # ==========================================================================
    # Test 3: Profile Snake activation
    # ==========================================================================
    print("\n4. Profiling Snake activation...")

    from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import Snake

    # Create test inputs at different sizes
    snake = Snake(channels=512)
    mx.eval(snake.alpha)

    sizes = [(1, 512, 100), (1, 512, 1000), (1, 512, 10000)]
    for shape in sizes:
        x = mx.random.normal(shape)
        mx.eval(x)

        # Baseline (no Metal kernel)
        Snake.use_metal_kernel = False
        times_baseline = []
        for _ in range(10):
            start = time.time()
            out = snake(x)
            mx.eval(out)
            times_baseline.append(time.time() - start)

        # With Metal kernel (if available)
        Snake.use_metal_kernel = True
        times_metal = []
        for _ in range(10):
            start = time.time()
            out = snake(x)
            mx.eval(out)
            times_metal.append(time.time() - start)

        baseline = np.mean(times_baseline[3:])
        metal = np.mean(times_metal[3:])

        print(f"   Snake {shape}: baseline={baseline*1000:.2f}ms, metal={metal*1000:.2f}ms ({baseline/metal:.1f}x speedup)")

    # ==========================================================================
    # Test 4: Compare to PyTorch vocoder (if possible)
    # ==========================================================================
    print("\n5. Testing PyTorch vocoder performance...")

    try:
        import torch

        # Simple vocoder simulation
        mel_torch = torch.randn(1, 80, 240, device="mps")

        # Create a simple vocoder-like network
        vocoder_torch = torch.nn.Sequential(
            torch.nn.Conv1d(80, 512, 7, padding=3),
            torch.nn.LeakyReLU(0.1),
            torch.nn.ConvTranspose1d(512, 256, 16, stride=8, padding=4),
            torch.nn.LeakyReLU(0.1),
            torch.nn.ConvTranspose1d(256, 128, 11, stride=5, padding=3),
            torch.nn.LeakyReLU(0.1),
            torch.nn.ConvTranspose1d(128, 64, 7, stride=3, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(64, 1, 7, padding=3),
        ).to("mps")

        # Warmup
        for _ in range(5):
            _ = vocoder_torch(mel_torch)
            torch.mps.synchronize()

        times = []
        for _ in range(10):
            start = time.time()
            _ = vocoder_torch(mel_torch)
            torch.mps.synchronize()
            times.append(time.time() - start)

        print(f"   PyTorch simulated vocoder: {np.mean(times[3:])*1000:.2f}ms")

    except Exception as e:
        print(f"   PyTorch test failed: {e}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Run final benchmark
    mel = mx.random.normal((1, mel_channels, 400))  # ~3.3s audio
    mx.eval(mel)

    times = []
    for _ in range(10):
        start = time.time()
        audio = vocoder(mel)
        mx.eval(audio)
        times.append(time.time() - start)

    avg = np.mean(times[3:])
    audio_duration = (400 * hop_size) / sample_rate
    rtf = audio_duration / avg

    print(f"\nFinal benchmark (400 mel frames / {audio_duration:.1f}s audio):")
    print(f"  Time: {avg*1000:.1f}ms")
    print(f"  RTF: {rtf:.1f}x real-time")
    print("  Target: 50x+ RTF")
    print(f"  Gap: {50/rtf:.1f}x improvement needed")


if __name__ == "__main__":
    profile_vocoder()
