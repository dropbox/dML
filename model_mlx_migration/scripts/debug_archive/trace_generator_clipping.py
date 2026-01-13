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

"""Trace through Generator to find where clipping occurs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import math

import mlx.core as mx
import mlx.nn as nn
from converters.kokoro_converter import KokoroConverter

print("=== Loading model ===")
converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

gen = model.decoder.generator

# Create test inputs similar to what decoder produces
batch = 1
length = 32  # input length
channels = 512

# Random input features
x = mx.random.normal((batch, length, channels)) * 0.1  # Small initial values
s = mx.random.normal((batch, config.style_dim)) * 0.5  # Style vector
f0 = mx.full((batch, length // 2), 200.0)  # 200 Hz F0

mx.eval(x, s, f0)

print("\n=== Initial inputs ===")
print(f"x shape: {x.shape}, range: [{float(x.min()):.4f}, {float(x.max()):.4f}]")
print(f"s shape: {s.shape}, range: [{float(s.min()):.4f}, {float(s.max()):.4f}]")
print(f"f0 shape: {f0.shape}, range: [{float(f0.min()):.4f}, {float(f0.max()):.4f}]")

# Manual trace through generator
print("\n=== Tracing through Generator ===")

# Calculate total upsampling factor
total_upp = 1
for r in gen.config.istft_upsample_rates:
    total_upp *= r
total_upp *= gen.istft_hop_size
print(f"Total upsampling: {total_upp}")

# Source signal
har_source, noise, uv = gen.m_source(f0, total_upp)
mx.eval(har_source, noise, uv)
print("\nSource signal (har_source):")
print(f"  shape: {har_source.shape}")
print(f"  range: [{float(har_source.min()):.4f}, {float(har_source.max()):.4f}]")

# STFT on source
har_1d = har_source.squeeze(-1)
source_stft = gen._source_stft(har_1d)
mx.eval(source_stft)
print("\nSource STFT output:")
print(f"  shape: {source_stft.shape}")
print(f"  range: [{float(source_stft.min()):.4f}, {float(source_stft.max()):.4f}]")

# Now trace through upsampling stages
x_trace = x
for i in range(gen.num_upsamples):
    print(f"\n=== Upsample stage {i} ===")

    up = getattr(gen, f"ups_{i}")
    noise_conv = getattr(gen, f"noise_convs_{i}")
    noise_res = getattr(gen, f"noise_res_{i}")

    x_trace = nn.leaky_relu(x_trace, 0.1)
    print(
        f"After leaky_relu: range=[{float(x_trace.min()):.4f}, {float(x_trace.max()):.4f}]"
    )

    # Source signal processing
    x_source = noise_conv(source_stft)
    mx.eval(x_source)
    print(
        f"After noise_conv: shape={x_source.shape}, range=[{float(x_source.min()):.4f}, {float(x_source.max()):.4f}]"
    )

    x_source = noise_res(x_source, s)
    mx.eval(x_source)
    print(
        f"After noise_res: shape={x_source.shape}, range=[{float(x_source.min()):.4f}, {float(x_source.max()):.4f}]"
    )

    # Upsample
    x_trace = up(x_trace)
    mx.eval(x_trace)
    print(
        f"After upsample: shape={x_trace.shape}, range=[{float(x_trace.min()):.4f}, {float(x_trace.max()):.4f}]"
    )

    # Reflection pad at last stage
    if i == gen.num_upsamples - 1:
        x_trace = mx.concatenate([x_trace[:, 1:2, :], x_trace], axis=1)
        print(f"After reflect pad: shape={x_trace.shape}")

    # Length alignment
    if x_source.shape[1] < x_trace.shape[1]:
        pad_len = x_trace.shape[1] - x_source.shape[1]
        x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
    elif x_source.shape[1] > x_trace.shape[1]:
        x_source = x_source[:, : x_trace.shape[1], :]

    # Add source
    x_trace = x_trace + x_source
    mx.eval(x_trace)
    print(
        f"After add source: range=[{float(x_trace.min()):.4f}, {float(x_trace.max()):.4f}]"
    )

    # ResBlocks
    xs = None
    for j in range(gen.num_kernels):
        block_idx = i * gen.num_kernels + j
        if block_idx < gen._num_resblocks:
            resblock = getattr(gen, f"resblocks_{block_idx}")
            if xs is None:
                xs = resblock(x_trace, s)
            else:
                xs = xs + resblock(x_trace, s)
    if xs is not None:
        x_trace = xs / gen.num_kernels
    mx.eval(x_trace)
    print(
        f"After resblocks: range=[{float(x_trace.min()):.4f}, {float(x_trace.max()):.4f}]"
    )

# Final conv
print("\n=== Final stages ===")
x_trace = nn.leaky_relu(x_trace, 0.1)
x_trace = gen.conv_post(x_trace)
mx.eval(x_trace)
print(
    f"After conv_post: shape={x_trace.shape}, range=[{float(x_trace.min()):.4f}, {float(x_trace.max()):.4f}]"
)

# ISTFT preparation
n_bins = gen.post_n_fft // 2 + 1
log_mag = mx.clip(x_trace[..., :n_bins], -10, 10)
mag = mx.exp(log_mag)
mx.eval(mag)
print("\nMagnitude spectrum:")
print(f"  shape: {mag.shape}")
print(f"  range: [{float(mag.min()):.4f}, {float(mag.max()):.4f}]")

phase_logits = x_trace[..., n_bins:]
phase = mx.sin(phase_logits) * math.pi
mx.eval(phase)
print("\nPhase:")
print(f"  shape: {phase.shape}")
print(f"  range: [{float(phase.min()):.4f}, {float(phase.max()):.4f}]")

# ISTFT synthesis
audio = gen._istft_synthesis(mag, phase)
mx.eval(audio)
print("\nISTFT output:")
print(f"  shape: {audio.shape}")
print(f"  range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")
print(f"  RMS: {float(mx.sqrt(mx.mean(audio**2))):.4f}")

# Check clipping before and after
clipped_before = (mx.abs(audio) > 1.0).astype(mx.float32)
print(f"  Samples > 1.0 before clip: {float(mx.sum(clipped_before))}")

audio_clipped = mx.clip(audio, -1.0, 1.0)
mx.eval(audio_clipped)
print("\nAfter clipping:")
print(f"  range: [{float(audio_clipped.min()):.4f}, {float(audio_clipped.max()):.4f}]")
print(f"  RMS: {float(mx.sqrt(mx.mean(audio_clipped**2))):.4f}")
