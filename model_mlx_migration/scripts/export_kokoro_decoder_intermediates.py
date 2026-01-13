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
Export PyTorch reference tensors WITH decoder intermediate tensors.

This script captures:
- All tensors from export_kokoro_reference.py
- PLUS: decoder encode/decode block outputs
- PLUS: generator input x

Run in Python 3.12 environment with kokoro installed:
  /tmp/kokoro_env/bin/python scripts/export_kokoro_decoder_intermediates.py \
    --text "Hello world" --voice af_bella --seed 0 --out-dir /tmp/kokoro_ref_decoder

Then compare MLX decoder outputs against these tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _set_all_seeds(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return x


def export_reference(
    out_dir: Path,
    text: str,
    voice: str,
    lang: str,
    speed: float,
    device: str | None,
    seed: int = 0,
) -> None:
    from kokoro import KModel, KPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    _set_all_seeds(seed)

    if device is None:
        device = "cpu"

    model = KModel().to(device).eval()
    pipeline = KPipeline(lang_code=lang, model=model, device=device)

    # Get first chunk
    for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        graphemes, phonemes = result.graphemes, result.phonemes
        break

    voice_pack = pipeline.load_voice(voice).to(device)
    ref_s = voice_pack[len(phonemes) - 1].to(device)

    vocab = model.vocab
    ids = [vocab.get(p) for p in phonemes]
    ids = [i for i in ids if i is not None]
    input_ids_list = [0, *ids, 0]
    input_ids = torch.tensor([input_ids_list], device=device, dtype=torch.long)

    tensors: Dict[str, Any] = {}

    with torch.no_grad():
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
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]
        duration_feats = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            duration_feats,
            input_lengths if input_lengths.device.type == "cpu" else input_lengths.cpu(),
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
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = duration_feats.transpose(-1, -2) @ pred_aln_trg

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        style128 = ref_s[:, :128]

        # --- DECODER INTERMEDIATE TENSOR CAPTURE ---
        # Hook into decoder to capture internal state
        decoder = model.decoder
        decoder_intermediates = {}

        # Access internal decoder state by tracing through forward
        # decoder.forward(asr, F0_pred, N_pred, style128)
        # The decoder structure is: f0_conv, n_conv, asr_res, encode, decode[0-3], generator

        # Get decoder input tensors
        asr_T = asr.transpose(-1, -2)  # [B, T, C]
        F0_T = F0_pred
        N_T = N_pred

        # F0/N convolutions
        F0_in = F0_T.unsqueeze(-1).transpose(1, 2)  # [B, 1, T]
        N_in = N_T.unsqueeze(-1).transpose(1, 2)  # [B, 1, T]

        F0_proc = decoder.F0_conv(F0_in)  # [B, 1, T//2]
        N_proc = decoder.N_conv(N_in)  # [B, 1, T//2]

        decoder_intermediates["F0_proc"] = _to_numpy(F0_proc.transpose(1, 2))  # [B, T//2, 1]
        decoder_intermediates["N_proc"] = _to_numpy(N_proc.transpose(1, 2))

        # ASR residual
        asr_res = decoder.asr_res(asr_T.transpose(1, 2))  # [B, 64, T]
        decoder_intermediates["asr_res"] = _to_numpy(asr_res.transpose(1, 2))  # [B, T, 64]

        # Initial concatenation
        # decoder expects NCL format, convert from NLC
        x = torch.cat([asr_T.transpose(1, 2), F0_proc, N_proc], dim=1)  # [B, C+2, T]
        decoder_intermediates["decoder_input_concat"] = _to_numpy(x.transpose(1, 2))

        # Encode block
        x = decoder.encode(x, style128)
        decoder_intermediates["encode_output"] = _to_numpy(x.transpose(1, 2))

        # Decode blocks (0-3)
        asr_res_down = asr_res
        for i, block in enumerate(decoder.decode):
            # Concatenate residuals
            x_cat = torch.cat([x, asr_res_down, F0_proc, N_proc], dim=1)
            x = block(x_cat, style128)
            decoder_intermediates[f"decode_{i}_output"] = _to_numpy(x.transpose(1, 2))

            # After upsampling block, adjust residual lengths
            if hasattr(block, 'upsample') and block.upsample:
                new_len = x.shape[2]
                asr_res_down = torch.nn.functional.interpolate(asr_res_down, size=new_len, mode='nearest')
                F0_proc = torch.nn.functional.interpolate(F0_proc, size=new_len, mode='nearest')
                N_proc = torch.nn.functional.interpolate(N_proc, size=new_len, mode='nearest')

        # Generator input
        decoder_intermediates["generator_input_x"] = _to_numpy(x.transpose(1, 2))

        # --- GENERATOR INTERMEDIATE CAPTURE ---
        gen = decoder.generator

        # F0 upsample
        with torch.no_grad():
            f0_up = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
            har_source, noi_source, uv = gen.m_source(f0_up)
            har_source_squeezed = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = gen.stft.transform(har_source_squeezed)
            har = torch.cat([har_spec, har_phase], dim=1)

        decoder_intermediates["gen_f0_up"] = _to_numpy(f0_up)
        decoder_intermediates["gen_har_source"] = _to_numpy(har_source)
        decoder_intermediates["gen_noi_source"] = _to_numpy(noi_source)
        decoder_intermediates["gen_uv"] = _to_numpy(uv)
        decoder_intermediates["gen_har"] = _to_numpy(har)
        decoder_intermediates["gen_har_spec"] = _to_numpy(har_spec)
        decoder_intermediates["gen_har_phase"] = _to_numpy(har_phase)

        # Run full decoder to get audio
        audio = decoder(asr, F0_pred, N_pred, style128).squeeze()

    # Export all tensors
    tensors["input_ids"] = _to_numpy(input_ids)
    tensors["ref_s"] = _to_numpy(ref_s)
    tensors["style_128"] = _to_numpy(style128)
    tensors["speaker_128"] = _to_numpy(s)
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
    tensors["asr_nlc"] = _to_numpy(asr.transpose(1, 2))
    tensors["audio"] = _to_numpy(audio)

    # Add decoder intermediates
    tensors.update(decoder_intermediates)

    np.savez_compressed(out_dir / "tensors.npz", **tensors)

    metadata = {
        "text": text,
        "lang": lang,
        "voice": voice,
        "speed": speed,
        "device": device,
        "seed": seed,
        "torch_version": torch.__version__,
        "graphemes": graphemes,
        "phonemes": phonemes,
        "input_ids_len": len(input_ids_list),
        "decoder_keys": list(decoder_intermediates.keys()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default="Hello world")
    parser.add_argument("--voice", type=str, default="af_bella")
    parser.add_argument("--lang", type=str, default="a")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    export_reference(
        out_dir=args.out_dir,
        text=args.text,
        voice=args.voice,
        lang=args.lang,
        speed=args.speed,
        device=args.device,
        seed=args.seed,
    )
    print(f"Wrote {args.out_dir}/metadata.json and {args.out_dir}/tensors.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
