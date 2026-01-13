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
Trace F0 divergence through layers.

Frame 85 in F0 output has 0.39 Hz difference between C++ and Python.
F0_1 upsamples 2x, so frame 85 corresponds to frame 42 or 43 pre-upsample.

This script saves intermediate tensors at each F0 layer for comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    print("Loading model...")
    converter = KokoroConverter()
    model, config, state_dict = converter.load_from_hf("hexgrad/Kokoro-82M")
    model.set_deterministic(True)
    mx.eval(model.parameters())

    # Same tokens as C++ test
    tokens = [0, 50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46, 0]

    # Load voice
    voice_pack = converter.load_voice_pack("af_bella")
    voice_emb = converter.select_voice_embedding(voice_pack, 12)
    speaker = voice_emb[:, 128:]  # [1, 128]

    input_ids = mx.array([tokens])

    # BERT encoding
    bert_out = model.bert(input_ids, attention_mask=mx.ones((1, len(tokens)), dtype=mx.int32))
    bert_enc = model.bert_encoder(bert_out)
    mx.eval(bert_enc)

    # Duration prediction and expansion
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    mx.eval(duration_feats)
    lstm_out = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(lstm_out)
    mx.eval(duration_logits)

    duration = mx.sigmoid(duration_logits).sum(axis=-1)
    pred_dur = mx.round(mx.clip(duration, 1, 100)).astype(mx.int32).squeeze()
    mx.eval(pred_dur)

    # Expand features
    dur_array = np.array(pred_dur)
    indices_list = []
    for i, d in enumerate(dur_array):
        indices_list.extend([i] * int(d))
    indices = mx.array(indices_list)

    en_expanded = duration_feats[0, indices, :]
    en_expanded_640 = en_expanded[None, :, :]
    mx.eval(en_expanded_640)

    # F0 predictor chain with detailed saves
    print("\n=== Tracing F0 Predictor Chain ===")

    # x_shared = BiLSTM output [1, 63, 512]
    x_shared = model.predictor.shared(en_expanded_640)
    mx.eval(x_shared)
    np.save("/tmp/py_x_shared.npy", np.array(x_shared))
    print(f"x_shared shape: {x_shared.shape}")
    print(f"x_shared [0,42,:5]: {x_shared[0,42,:5].tolist()}")  # Frame 42 (maps to ~84-85 after upsample)

    # F0_0: [1, 63, 512] -> [1, 63, 512]
    x = x_shared
    x = model.predictor.F0_0(x, speaker)
    mx.eval(x)
    np.save("/tmp/py_after_F0_0.npy", np.array(x))
    print(f"\nAfter F0_0 shape: {x.shape}")
    print(f"After F0_0 [0,42,:5]: {x[0,42,:5].tolist()}")

    # F0_1: [1, 63, 512] -> [1, 126, 256] (2x upsample)
    x = model.predictor.F0_1(x, speaker)
    mx.eval(x)
    np.save("/tmp/py_after_F0_1.npy", np.array(x))
    print(f"\nAfter F0_1 shape: {x.shape}")
    print(f"After F0_1 [0,84,:5]: {x[0,84,:5].tolist()}")  # Frame 84 (pre-85)
    print(f"After F0_1 [0,85,:5]: {x[0,85,:5].tolist()}")  # Frame 85

    # F0_2: [1, 126, 256] -> [1, 126, 256]
    x = model.predictor.F0_2(x, speaker)
    mx.eval(x)
    np.save("/tmp/py_after_F0_2.npy", np.array(x))
    print(f"\nAfter F0_2 shape: {x.shape}")
    print(f"After F0_2 [0,84,:5]: {x[0,84,:5].tolist()}")
    print(f"After F0_2 [0,85,:5]: {x[0,85,:5].tolist()}")

    # F0_proj: [1, 126, 256] -> [1, 126]
    f0 = model.predictor.F0_proj(x).squeeze(-1)
    mx.eval(f0)
    np.save("/tmp/py_f0.npy", np.array(f0))
    print(f"\nAfter F0_proj shape: {f0.shape}")
    print(f"F0 [84]: {float(f0[0,84]):.6f}")
    print(f"F0 [85]: {float(f0[0,85]):.6f}")

    # Compare with C++ if files exist
    print("\n=== Comparison with C++ ===")

    cpp_f0_path = Path("/tmp/cpp_f0.npy")
    if cpp_f0_path.exists():
        cpp_f0 = np.load(cpp_f0_path)
        py_f0 = np.array(f0)

        print("F0 at frame 85:")
        print(f"  Python: {float(py_f0[0,85]):.6f}")
        print(f"  C++:    {float(cpp_f0[0,85]):.6f}")
        print(f"  Diff:   {float(abs(py_f0[0,85] - cpp_f0[0,85])):.6f}")
    else:
        print("Run C++ with DEBUG_DECODE_BLOCKS=1 to generate cpp_f0.npy")


if __name__ == "__main__":
    main()
