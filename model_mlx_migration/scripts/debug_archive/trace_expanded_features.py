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

"""Trace expanded features to understand F0 collapse."""

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

voice_path = Path.home() / "models" / "kokoro" / "voices" / "af_heart.pt"
voice_pack = torch.load(voice_path, map_location="cpu", weights_only=False)
voice = mx.array(voice_pack[5].numpy())

tokens = mx.array([[0, 47, 44, 51, 51, 54, 0]])

# Split voice
style = voice[:, :128]
speaker = voice[:, 128:]

# BERT encoding
bert_out = model.bert(tokens)
bert_enc = model.bert_encoder(bert_out)
mx.eval(bert_enc)

# Duration features
duration_feats = model.predictor.text_encoder(bert_enc, speaker)
mx.eval(duration_feats)

print("=== Duration features per token ===")
for i in range(tokens.shape[1]):
    feat = duration_feats[0, i, :]
    mx.eval(feat)
    print(
        f"Token {i}: mean={float(mx.mean(feat)):.4f}, std={float(mx.std(feat)):.4f}, "
        f"range=[{float(feat.min()):.4f}, {float(feat.max()):.4f}]"
    )

# Duration prediction
batch, length, hidden_dim = duration_feats.shape
style_expanded = mx.broadcast_to(
    speaker[:, None, :], (batch, length, speaker.shape[-1])
)
x_cat = mx.concatenate([duration_feats, style_expanded], axis=-1)
dur_enc = model.predictor.lstm(x_cat)
mx.eval(dur_enc)

duration_logits = model.predictor.duration_proj(dur_enc)
mx.eval(duration_logits)

duration = mx.sigmoid(duration_logits).sum(axis=-1)
pred_dur = mx.round(mx.clip(duration, 1, 100)).astype(mx.int32).squeeze()
mx.eval(pred_dur)

print("\n=== Durations ===")
print(f"Duration per token: {pred_dur.tolist()}")
print(f"Total frames: {int(mx.sum(pred_dur))}")

# Compute indices
indices: list[int] = []
dur_array = np.array(pred_dur)
for i, d in enumerate(dur_array):
    indices.extend([i] * int(d))
print(f"\nIndices (first 30): {indices[:30]}")
print(f"Indices (last 30): {indices[-30:]}")

# Expand duration_feats
indices_arr = mx.array(indices)
en_expanded = duration_feats[0, indices_arr, :]
en_expanded = en_expanded[None, :, :]
mx.eval(en_expanded)

print(f"\n=== Expanded features shape: {en_expanded.shape} ===")
print("First 10 frames (should map to token 0):")
for i in range(10):
    feat = en_expanded[0, i, :]
    print(
        f"  Frame {i}: mean={float(mx.mean(feat)):.4f}, std={float(mx.std(feat)):.4f}"
    )

# Check middle frames
mid = en_expanded.shape[1] // 2
print(f"\nMiddle 10 frames (around frame {mid}):")
for i in range(mid - 5, mid + 5):
    feat = en_expanded[0, i, :]
    print(
        f"  Frame {i}: mean={float(mx.mean(feat)):.4f}, std={float(mx.std(feat)):.4f}"
    )

# The key insight: if the middle frames all map to the same token,
# they'll have the same features, leading to the same F0 prediction

# Check which tokens the middle frames map to
print("\nToken mapping for middle frames:")
for i in range(mid - 5, mid + 5):
    if i < len(indices):
        print(f"  Frame {i} -> Token {indices[i]}")
