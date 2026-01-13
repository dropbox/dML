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
Export a PyTorch reference trace from the official `kokoro` package.

Purpose:
  - Produce a deterministic, checkpoint-faithful reference for Kokoro
  - Export intermediate tensors needed to debug MLX mismatches

Important:
  - The official `kokoro` package (hexgrad/kokoro) requires Python < 3.14.
  - Run this script in a separate Python 3.10–3.13 environment where
    `pip install kokoro` works.

Outputs:
  <out_dir>/metadata.json
  <out_dir>/tensors.npz

Example:
  python scripts/export_kokoro_reference.py \
    --text "Hello, how are you?" \
    --voice af_heart \
    --lang a \
    --out-dir reports/kokoro_reference_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _to_numpy(x):
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return x


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_first_chunk(
    pipeline, text: str, voice: str, speed: float
) -> Tuple[str, str, Any]:
    """
    Returns (graphemes, phonemes, output) for the first chunk.
    """
    for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        return result.graphemes, result.phonemes, result.output
    raise RuntimeError("No chunks produced")


def _phonemes_to_input_ids(model, phonemes: str) -> List[int]:
    # Mirrors kokoro.model.KModel.forward()
    vocab = model.vocab
    ids = [vocab.get(p) for p in phonemes]
    ids = [i for i in ids if i is not None]
    # Add BOS/EOS (0) like kokoro does: [0, *ids, 0]
    return [0, *ids, 0]


def _load_voice_pack(pipeline, voice: str):
    import torch

    pack = pipeline.load_voice(voice)
    if not isinstance(pack, torch.Tensor):
        raise RuntimeError(f"Unexpected voice pack type: {type(pack)}")
    return pack


def _select_ref_s(voice_pack, phonemes: str):
    """
    Mirrors kokoro.pipeline.KPipeline.infer(): pack[len(ps)-1]
    """
    idx = len(phonemes) - 1
    if idx < 0:
        raise RuntimeError("Empty phoneme string")
    return voice_pack[idx]


def export_reference(
    out_dir: Path,
    text: str,
    voice: str,
    lang: str,
    speed: float,
    device: str | None,
    repo_id: str,
) -> None:
    import numpy as np
    import torch

    try:
        from kokoro import KModel, KPipeline
    except Exception as e:
        raise RuntimeError(
            "Failed to import `kokoro`. Install in Python <3.14: "
            "`pip install kokoro` (hexgrad/kokoro)."
        ) from e

    _ensure_dir(out_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build pipeline/model on requested device.
    # Note: kokoro 0.7.x API doesn't use repo_id argument
    model = KModel().to(device).eval()
    pipeline = KPipeline(lang_code=lang, model=model, device=device)

    graphemes, phonemes, output = _pick_first_chunk(pipeline, text, voice, speed)
    voice_pack = _load_voice_pack(pipeline, voice).to(device)
    ref_s = _select_ref_s(voice_pack, phonemes).to(device)
    input_ids_list = _phonemes_to_input_ids(model, phonemes)

    input_ids = torch.tensor([input_ids_list], device=device, dtype=torch.long)

    tensors: Dict[str, Any] = {}

    with torch.no_grad():
        # This is a faithful re-expression of KModel.forward_with_tokens()
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=device,
            dtype=torch.long,
        )

        text_mask = (
            torch.arange(int(input_lengths.max().item()), device=device)
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(
            device
        )  # [B, T]

        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)  # [B, hidden, T]

        # ref_s is typically [1, 256]; KModel splits it
        s = ref_s[:, 128:]
        duration_feats = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        # Duration LSTM + projection
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            duration_feats,
            (
                input_lengths
                if input_lengths.device.type == "cpu"
                else input_lengths.cpu()
            ),
            batch_first=True,
            enforce_sorted=False,
        )
        model.predictor.lstm.flatten_parameters()
        x_lstm, _ = model.predictor.lstm(x_packed)
        x_lstm, _ = torch.nn.utils.rnn.pad_packed_sequence(x_lstm, batch_first=True)
        x_pad = torch.zeros(
            [x_lstm.shape[0], text_mask.shape[-1], x_lstm.shape[-1]], device=device
        )
        x_pad[:, : x_lstm.shape[1], :] = x_lstm
        x_lstm = x_pad

        duration_logits = model.predictor.duration_proj(x_lstm)
        duration = torch.sigmoid(duration_logits).sum(dim=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)  # [B, T_text, T_align]

        en = duration_feats.transpose(-1, -2) @ pred_aln_trg  # [B, hidden, T_align]

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        t_en = model.text_encoder(
            input_ids, input_lengths, text_mask
        )  # [B, hidden, T_text]
        asr = t_en @ pred_aln_trg  # [B, hidden, T_align]

        style128 = ref_s[:, :128]
        audio = model.decoder(asr, F0_pred, N_pred, style128).squeeze()

    # Export tensors. Store both NCL and NLC variants for MLX convenience.
    tensors["input_ids"] = _to_numpy(input_ids)
    tensors["ref_s"] = _to_numpy(ref_s)
    tensors["style_128"] = _to_numpy(ref_s[:, :128])
    tensors["speaker_128"] = _to_numpy(ref_s[:, 128:])

    tensors["text_mask"] = _to_numpy(text_mask)
    tensors["bert_dur"] = _to_numpy(bert_dur)
    tensors["d_en"] = _to_numpy(d_en)
    tensors["duration_feats"] = _to_numpy(duration_feats)
    tensors["duration_logits"] = _to_numpy(duration_logits)
    tensors["pred_dur"] = _to_numpy(pred_dur)
    tensors["pred_aln_trg"] = _to_numpy(pred_aln_trg)
    tensors["en"] = _to_numpy(en)

    tensors["F0_pred"] = _to_numpy(F0_pred)
    tensors["N_pred"] = _to_numpy(N_pred)

    tensors["t_en"] = _to_numpy(t_en)
    tensors["asr_ncl"] = _to_numpy(asr)
    tensors["asr_nlc"] = _to_numpy(asr.transpose(1, 2))  # [B, T_align, hidden]

    tensors["audio"] = _to_numpy(audio)
    if output is not None and hasattr(output, "audio") and output.audio is not None:
        tensors["audio_pipeline"] = _to_numpy(output.audio)

    np.savez_compressed(out_dir / "tensors.npz", **tensors)

    metadata = {
        "repo_id": repo_id,
        "text": text,
        "lang": lang,
        "voice": voice,
        "speed": speed,
        "device": device,
        "graphemes": graphemes,
        "phonemes": phonemes,
        "input_ids_len": len(input_ids_list),
        "note": "Run validate script in MLX env to compare decoder/generator.",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Kokoro PyTorch reference tensors"
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--text", type=str, default="Hello, how are you?", help="Text to synthesize"
    )
    parser.add_argument(
        "--voice", type=str, default="af_heart", help="Voice name (e.g., af_heart)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="a",
        help="Language code (e.g., a for American English)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor")
    parser.add_argument(
        "--device", type=str, default=None, help="cpu/cuda/mps (default auto)"
    )
    parser.add_argument(
        "--repo-id", type=str, default="hexgrad/Kokoro-82M", help="HF repo id"
    )
    args = parser.parse_args()

    export_reference(
        out_dir=args.out_dir,
        text=args.text,
        voice=args.voice,
        lang=args.lang,
        speed=args.speed,
        device=args.device,
        repo_id=args.repo_id,
    )
    print(f"Wrote {args.out_dir}/metadata.json and {args.out_dir}/tensors.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
