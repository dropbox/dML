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
Validate MLX Kokoro model against PyTorch reference tensors.

This script compares MLX intermediate outputs against PyTorch reference
exported by export_kokoro_reference.py.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add tools path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

from converters.models.kokoro import KokoroModel


def load_model():
    """Load MLX Kokoro model."""
    from converters.kokoro_converter import KokoroConverter

    converter = KokoroConverter()
    model, config, vocab = converter.load_from_hf("hexgrad/Kokoro-82M")
    mx.eval(model.parameters())
    return model


def compare_arrays(
    name: str, ref: np.ndarray, mlx_arr: mx.array, atol: float = 1e-4
) -> dict:
    """Compare reference and MLX arrays."""
    mlx_np = np.array(mlx_arr)

    # Handle shape differences
    if ref.shape != mlx_np.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "ref_shape": ref.shape,
            "mlx_shape": mlx_np.shape,
        }

    diff = np.abs(ref - mlx_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # Correlation for large arrays
    if ref.size > 10:
        ref_flat = ref.flatten()
        mlx_flat = mlx_np.flatten()
        corr = np.corrcoef(ref_flat, mlx_flat)[0, 1]
    else:
        corr = 1.0 if max_diff < atol else 0.0

    status = "PASS" if max_diff < atol else ("CLOSE" if corr > 0.99 else "FAIL")

    return {
        "name": name,
        "status": status,
        "shape": ref.shape,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "correlation": float(corr),
        "ref_range": [float(ref.min()), float(ref.max())],
        "mlx_range": [float(mlx_np.min()), float(mlx_np.max())],
    }


def validate_pipeline(model: KokoroModel, ref_path: Path) -> list:
    """Run MLX model and compare against reference at each stage."""

    # Load reference
    ref = np.load(ref_path / "tensors.npz")
    with open(ref_path / "metadata.json") as f:
        meta = json.load(f)

    results = []

    # Get reference values
    input_ids = mx.array(ref["input_ids"])
    ref_s = mx.array(ref["ref_s"])
    style = ref_s[:, :128]
    speaker = ref_s[:, 128:]

    print(f"Input text: {meta['text']}")
    print(f"Phonemes: {meta['phonemes']}")
    print(f"Input IDs shape: {input_ids.shape}")
    print()

    # Step 1: BERT forward
    print("=== Step 1: BERT ===")
    attention_mask = mx.ones_like(input_ids)
    bert_out = model.bert(input_ids, attention_mask=attention_mask)
    mx.eval(bert_out)

    results.append(compare_arrays("bert_dur", ref["bert_dur"], bert_out))
    print(f"BERT: {results[-1]}")

    # Step 2: bert_encoder projection
    print("\n=== Step 2: bert_encoder ===")
    bert_enc = model.bert_encoder(bert_out)
    d_en = mx.transpose(bert_enc, (0, 2, 1))  # [B, hidden, T_text] = [B, 512, 14]
    mx.eval(d_en)

    results.append(compare_arrays("d_en", ref["d_en"], d_en))
    print(f"d_en: {results[-1]}")

    # Step 3: Predictor text_encoder
    print("\n=== Step 3: text_encoder (duration features) ===")
    mx.array([input_ids.shape[-1]])
    text_mask = mx.zeros((1, input_ids.shape[-1]), dtype=mx.bool_)

    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    mx.eval(duration_feats)

    results.append(
        compare_arrays("duration_feats", ref["duration_feats"], duration_feats)
    )
    print(f"duration_feats: {results[-1]}")

    # Step 4: Duration LSTM
    print("\n=== Step 4: Duration LSTM ===")
    # PyTorch passes duration_feats directly to LSTM (640-dim), not concatenated with bert_enc
    dur_enc = model.predictor.lstm(duration_feats)  # [B, T, 512]
    mx.eval(dur_enc)

    # Duration projection
    duration_logits = model.predictor.duration_proj(dur_enc)
    mx.eval(duration_logits)

    results.append(
        compare_arrays("duration_logits", ref["duration_logits"], duration_logits)
    )
    print(f"duration_logits: {results[-1]}")

    # Predicted durations
    speed = 1.0
    duration = mx.sigmoid(duration_logits).sum(axis=-1) / speed
    pred_dur = mx.round(duration).astype(mx.int32)
    pred_dur = mx.maximum(pred_dur, 1).squeeze()
    mx.eval(pred_dur)

    results.append(compare_arrays("pred_dur", ref["pred_dur"], pred_dur))
    print(f"pred_dur: {results[-1]}")

    # Step 5: Alignment matrix
    print("\n=== Step 5: Alignment ===")
    T_text = input_ids.shape[1]
    T_align = int(mx.sum(pred_dur).item())

    # Create alignment using repeat_interleave logic
    indices_list: list[int] = []
    pred_dur_np = np.array(pred_dur)
    for i, d in enumerate(pred_dur_np):
        indices_list.extend([i] * int(d))
    indices = mx.array(indices_list)

    pred_aln_trg = mx.zeros((T_text, T_align))
    # One-hot encoding
    for t in range(T_align):
        idx = int(indices[t].item())  # type: ignore[union-attr]
        pred_aln_trg = pred_aln_trg.at[idx, t].add(1.0)
    pred_aln_trg = mx.expand_dims(pred_aln_trg, 0)  # [1, T_text, T_align]
    mx.eval(pred_aln_trg)

    results.append(compare_arrays("pred_aln_trg", ref["pred_aln_trg"], pred_aln_trg))
    print(f"pred_aln_trg: {results[-1]}")

    # Step 6: en = duration_feats @ alignment
    print("\n=== Step 6: en (aligned duration features) ===")
    duration_feats_t = mx.transpose(duration_feats, (0, 2, 1))  # [B, 640, T_text]
    en = duration_feats_t @ pred_aln_trg  # [B, 640, T_align]
    mx.eval(en)

    results.append(compare_arrays("en", ref["en"], en))
    print(f"en: {results[-1]}")

    # Step 7: F0/N prediction
    # PyTorch F0Ntrain: shared LSTM -> F0 blocks -> F0_proj, and same for N
    print("\n=== Step 7: F0/N Prediction ===")

    # en is [B, 640, T_align] NCL format
    # shared LSTM expects NLC, so transpose
    en_nlc = mx.transpose(en, (0, 2, 1))  # [B, T_align, 640]
    shared_out = model.predictor.shared(en_nlc)  # [B, T_align, 512]
    mx.eval(shared_out)

    # F0 prediction: F0_0 -> F0_1 -> F0_2 -> F0_proj
    # AdainResBlk1d expects NLC format [B, T, C]
    x = shared_out  # NLC format
    x = model.predictor.F0_0(x, speaker)
    x = model.predictor.F0_1(x, speaker)
    x = model.predictor.F0_2(x, speaker)
    # F0_proj expects NLC, outputs [B, T, 1]
    F0_pred = model.predictor.F0_proj(x).squeeze(-1)  # [B, T]
    mx.eval(F0_pred)

    # N prediction: N_0 -> N_1 -> N_2 -> N_proj
    x = shared_out  # NLC format
    x = model.predictor.N_0(x, speaker)
    x = model.predictor.N_1(x, speaker)
    x = model.predictor.N_2(x, speaker)
    N_pred = model.predictor.N_proj(x).squeeze(-1)  # [B, T]
    mx.eval(N_pred)

    results.append(compare_arrays("F0_pred", ref["F0_pred"], F0_pred))
    results.append(compare_arrays("N_pred", ref["N_pred"], N_pred))
    print(f"F0_pred: {results[-2]}")
    print(f"N_pred: {results[-1]}")

    # Step 8: Text encoder (for decoder)
    print("\n=== Step 8: Text Encoder t_en ===")
    # TextEncoder takes (input_ids, mask) and returns NLC format
    t_en = model.text_encoder(input_ids, text_mask)
    # Transpose to NCL for comparison with ref which is NCL
    t_en = mx.transpose(t_en, (0, 2, 1))
    mx.eval(t_en)

    results.append(compare_arrays("t_en", ref["t_en"], t_en))
    print(f"t_en: {results[-1]}")

    # Step 9: ASR = t_en @ alignment
    print("\n=== Step 9: ASR ===")
    asr = t_en @ pred_aln_trg  # [B, 512, T_align]
    mx.eval(asr)

    results.append(compare_arrays("asr_ncl", ref["asr_ncl"], asr))
    print(f"asr: {results[-1]}")

    # Step 10: Decoder
    print("\n=== Step 10: Decoder ===")
    # Decoder expects NLC format, so transpose asr from NCL to NLC
    asr_nlc = mx.transpose(asr, (0, 2, 1))  # [B, T_align, 512]
    audio = model.decoder(asr_nlc, F0_pred, N_pred, style)
    audio = audio.squeeze()
    mx.eval(audio)

    results.append(compare_arrays("audio", ref["audio"], audio))
    print(f"audio: {results[-1]}")

    return results


def main():
    ref_path = Path("/tmp/kokoro_ref")

    if not ref_path.exists():
        print(f"ERROR: Reference not found at {ref_path}")
        print("Run scripts/export_kokoro_reference.py first in a PyTorch environment")
        return 1

    print("Loading MLX Kokoro model...")
    model = load_model()
    print("Model loaded.\n")

    print("=" * 60)
    print("VALIDATING MLX vs PYTORCH REFERENCE")
    print("=" * 60)

    results = validate_pipeline(model, ref_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        status = r["status"]
        name = r["name"]
        if status == "SHAPE_MISMATCH":
            print(f"  {name}: {status} (ref={r['ref_shape']}, mlx={r['mlx_shape']})")
        else:
            corr = r.get("correlation", 0)
            max_diff = r.get("max_diff", float("inf"))
            print(f"  {name}: {status} (corr={corr:.4f}, max_diff={max_diff:.2e})")

    # Overall status
    failed = [r for r in results if r["status"] not in ("PASS", "CLOSE")]
    if failed:
        print(f"\nFAILED: {len(failed)}/{len(results)} checks")
        return 1
    else:
        print(f"\nPASSED: All {len(results)} checks")
        return 0


if __name__ == "__main__":
    sys.exit(main())
