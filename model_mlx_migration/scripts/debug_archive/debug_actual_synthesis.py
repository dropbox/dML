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

"""Debug actual synthesis F0 values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import torch
from converters.kokoro_converter import KokoroConverter

print("=== Loading model ===")
converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

# Load voice
voice_path = Path.home() / "models" / "kokoro" / "voices" / "af_heart.pt"
voice_pack = torch.load(voice_path, map_location="cpu", weights_only=False)
voice = mx.array(voice_pack[5].numpy())

# Test tokens
tokens = mx.array([[0, 47, 44, 51, 51, 54, 0]])

print("\n=== Running through model manually ===")

# Split voice
style = voice[:, :128]
speaker = voice[:, 128:]

# BERT encoding
bert_out = model.bert(tokens)
bert_enc = model.bert_encoder(bert_out)
mx.eval(bert_enc)
print(f"bert_enc shape: {bert_enc.shape}")

# Duration features
duration_feats = model.predictor.text_encoder(bert_enc, speaker)
mx.eval(duration_feats)
print(f"duration_feats shape: {duration_feats.shape}")

# Duration prediction
batch, length, hidden_dim = duration_feats.shape
style_expanded = mx.broadcast_to(
    speaker[:, None, :], (batch, length, speaker.shape[-1])
)
x_cat = mx.concatenate([duration_feats, style_expanded], axis=-1)
print(f"x_cat shape for LSTM: {x_cat.shape}")

dur_enc = model.predictor.lstm(x_cat)
mx.eval(dur_enc)
print(f"dur_enc shape: {dur_enc.shape}")

duration_logits = model.predictor.duration_proj(dur_enc)
mx.eval(duration_logits)
print(f"duration_logits shape: {duration_logits.shape}")

# Compute alignment
indices, total_frames = model._compute_alignment(duration_logits, 1.0)
mx.eval(indices)
print(f"indices shape: {indices.shape}, total_frames: {total_frames}")

# Expand features
en_expanded = model._expand_features(duration_feats, indices, total_frames)
mx.eval(en_expanded)
print(f"en_expanded shape: {en_expanded.shape}")

# Text encoder for F0/N
enc_out_f0n = model.predictor.text_encoder(en_expanded, speaker)
mx.eval(enc_out_f0n)
print(f"enc_out_f0n shape: {enc_out_f0n.shape}")

# F0 prediction
x = enc_out_f0n
x = model.predictor.F0_0(x, speaker)
x = model.predictor.F0_1(x, speaker)
x = model.predictor.F0_2(x, speaker)
f0 = model.predictor.F0_proj(x).squeeze(-1)
mx.eval(f0)

print("\n=== F0 Analysis ===")
print(f"F0 shape: {f0.shape}")
print(f"F0 range: [{float(f0.min()):.4f}, {float(f0.max()):.4f}]")
print(f"F0 mean: {float(mx.mean(f0)):.4f}")
print(f"F0 std: {float(mx.std(f0)):.4f}")

# Check F0 distribution
f0_flat = f0.flatten()
print(f"\nF0 first 20 values: {[f'{x:.1f}' for x in f0_flat[:20].tolist()]}")
print(f"F0 last 20 values: {[f'{x:.1f}' for x in f0_flat[-20:].tolist()]}")

# What percentage in speech range?
in_speech_range = mx.logical_and(f0 > 80, f0 < 400)
pct = float(mx.mean(in_speech_range.astype(mx.float32))) * 100
print(f"\nF0 in speech range (80-400 Hz): {pct:.1f}%")

# What's the expected fundamental frequency for typical female speech? ~200-400 Hz
# For male speech: ~100-200 Hz
# The voice af_heart should be a female voice

# Noise prediction
x = enc_out_f0n
x = model.predictor.N_0(x, speaker)
x = model.predictor.N_1(x, speaker)
x = model.predictor.N_2(x, speaker)
noise = model.predictor.N_proj(x).squeeze(-1)
mx.eval(noise)

print("\n=== Noise Analysis ===")
print(f"Noise shape: {noise.shape}")
print(f"Noise range: [{float(noise.min()):.4f}, {float(noise.max()):.4f}]")

# ASR features
text_enc = model.text_encoder(tokens)
mx.eval(text_enc)
print(f"\ntext_enc shape: {text_enc.shape}")

asr_expanded = model._expand_features(text_enc, indices, total_frames)
mx.eval(asr_expanded)
print(f"asr_expanded shape: {asr_expanded.shape}")

# Run decoder
print("\n=== Decoder ===")
audio = model.decoder(asr_expanded, f0, noise, style)
mx.eval(audio)

print(f"Audio shape: {audio.shape}")
print(f"Audio range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")
print(f"Audio RMS: {float(mx.sqrt(mx.mean(audio**2))):.4f}")

# What would the harmonics be for the predicted F0?
avg_f0 = float(mx.mean(f0))
print(f"\nFor average F0={avg_f0:.1f} Hz:")
for h in range(1, 10):
    print(f"  Harmonic {h}: {avg_f0 * h:.1f} Hz")
