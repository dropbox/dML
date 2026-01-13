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

"""Trace F0 prediction during actual synthesis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import numpy as np
import torch
from converters.kokoro_converter import KokoroConverter

print("=== Loading model ===")
converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

# Load voice
voice_path = Path.home() / "models" / "kokoro" / "voices" / "af_heart.pt"
voice_pack = torch.load(voice_path, map_location="cpu", weights_only=False)
style = mx.array(voice_pack[5].numpy())

# Test tokens
tokens = mx.array([[0, 47, 44, 51, 51, 54, 0]])

print("\n=== Tracing through synthesis ===")

# Run BERT encoding
input_lengths = mx.array([tokens.shape[1]])
text_mask = mx.zeros((1, tokens.shape[1]), dtype=mx.bool_)

bert_enc = model.bert(
    tokens, attention_mask=mx.ones((1, tokens.shape[1]), dtype=mx.int32)
)
bert_enc = model.bert_encoder(bert_enc)
mx.eval(bert_enc)

print(f"BERT output shape: {bert_enc.shape}")

# Get style components
speaker = style[:, 128:]
style_s = style[:, :128]

# Run predictor to get durations
duration_feats = model.predictor.text_encoder(bert_enc, speaker)
mx.eval(duration_feats)

print(f"Duration feats shape: {duration_feats.shape}")

# LSTM + duration projection
lstm_out, _ = model.predictor.lstm(duration_feats)
mx.eval(lstm_out)

duration_logits = model.predictor.duration_proj(lstm_out)
mx.eval(duration_logits)

duration = mx.sigmoid(duration_logits).sum(axis=-1)
pred_dur = mx.round(mx.clip(duration, 1, 100)).astype(mx.int32).squeeze()
mx.eval(pred_dur)

print(f"Predicted durations: {pred_dur.tolist()}")
total_frames = int(mx.sum(pred_dur))
print(f"Total frames: {total_frames}")

# Expand features
indices_list: list[int] = []
dur_array = np.array(pred_dur)
for i, d in enumerate(dur_array):
    indices_list.extend([i] * int(d))
indices = mx.array(indices_list)

en_expanded = duration_feats[0, indices, :]
en_expanded = en_expanded[None, :, :]
mx.eval(en_expanded)

print(f"Expanded features shape: {en_expanded.shape}")

# Get F0 prediction
enc_out = model.predictor.text_encoder(en_expanded, speaker)
mx.eval(enc_out)

# Apply F0 blocks
x = enc_out
for i in range(3):
    block = getattr(model.predictor, f"F0_{i}")
    x = block(x, speaker)
    mx.eval(x)

f0 = model.predictor.F0_proj(x).squeeze(-1)
mx.eval(f0)

print("\n=== F0 Prediction ===")
print(f"F0 shape: {f0.shape}")
print(f"F0 range: [{float(f0.min()):.4f}, {float(f0.max()):.4f}]")
print(f"F0 mean: {float(mx.mean(f0)):.4f}")
print(f"F0 std: {float(mx.std(f0)):.4f}")

# F0 histogram
f0_flat = f0.flatten()
print(f"\nF0 values (first 50): {[f'{x:.1f}' for x in f0_flat[:50].tolist()]}")

# How many F0 values are in speech range?
f0_in_range = mx.logical_and(f0 > 80, f0 < 400)
pct_in_range = float(mx.mean(f0_in_range.astype(mx.float32))) * 100
print(f"\nF0 values in speech range (80-400 Hz): {pct_in_range:.1f}%")

# Get N prediction
x = enc_out
for i in range(3):
    block = getattr(model.predictor, f"N_{i}")
    x = block(x, speaker)

noise = model.predictor.N_proj(x).squeeze(-1)
mx.eval(noise)

print("\n=== Noise Prediction ===")
print(f"Noise range: [{float(noise.min()):.4f}, {float(noise.max()):.4f}]")
print(f"Noise mean: {float(mx.mean(noise)):.4f}")

# Now trace through decoder
print("\n=== Decoder ===")
t_enc = model.text_encoder(tokens, input_lengths, text_mask)
mx.eval(t_enc)

# Expand text encoder output
asr = t_enc[0, indices, :]
asr = asr[None, :, :]
mx.eval(asr)

print(f"ASR features shape: {asr.shape}")

# Decoder's generator uses F0 * 4 due to upsampling
# Let's trace through decoder.__call__ manually
decoder = model.decoder

# F0 and noise are upsampled 4x in decoder's internal processing
f0_orig = f0

# Trace decoder
f0_in = f0[:, :, None]
n_in = noise[:, :, None]

f0_proc = decoder.f0_conv(f0_in)
n_proc = decoder.n_conv(n_in)
mx.eval(f0_proc, n_proc)

print(
    f"f0_proc shape: {f0_proc.shape}, range: [{float(f0_proc.min()):.4f}, {float(f0_proc.max()):.4f}]"
)
print(
    f"n_proc shape: {n_proc.shape}, range: [{float(n_proc.min()):.4f}, {float(n_proc.max()):.4f}]"
)

# What goes into generator?
print("\n=== Generator F0 input ===")
print(
    f"F0 passed to generator: shape={f0_orig.shape}, range=[{float(f0_orig.min()):.4f}, {float(f0_orig.max()):.4f}]"
)

# Calculate what harmonics this would generate
print(f"\nFor F0={float(f0_orig.mean()):.1f}Hz:")
print(f"  Fundamental period: {24000 / float(f0_orig.mean()):.1f} samples")
print(f"  9th harmonic would be at: {9 * float(f0_orig.mean()):.1f} Hz")
