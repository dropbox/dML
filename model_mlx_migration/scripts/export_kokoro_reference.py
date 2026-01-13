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
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _set_all_seeds(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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
    seed: int = 0,
    deterministic_source: bool = False,
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

    # Set all seeds BEFORE any model loading or inference
    _set_all_seeds(seed)

    if device is None:
        # Prefer CPU for deterministic export (GPU can have non-determinism)
        device = "cpu"

    # Build pipeline/model on requested device.
    # Note: kokoro 0.7.x API doesn't use repo_id argument
    model = KModel().to(device).eval()
    pipeline = KPipeline(lang_code=lang, model=model, device=device)

    # If deterministic_source is set, patch the SourceModule to use zeros
    # instead of random values (to match MLX deterministic mode)
    if deterministic_source:
        gen = model.decoder.generator
        sine_gen = gen.m_source.l_sin_gen
        source_mod = gen.m_source

        # Patch SineGen._f02sine to use zeros for initial phase instead of torch.rand
        _original_f02sine = sine_gen._f02sine  # noqa: F841 - saved for potential restore

        def deterministic_f02sine(f0_values):
            # Same as original but with zeros instead of torch.rand for initial phase
            rad_values = (f0_values / sine_gen.sampling_rate) % 1
            # CHANGE: Use zeros instead of torch.rand for initial phase
            rand_ini = torch.zeros(
                f0_values.shape[0], f0_values.shape[2], device=f0_values.device
            )
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            if not sine_gen.flag_for_pulse:
                import torch.nn.functional as F

                rad_values = F.interpolate(
                    rad_values.transpose(1, 2),
                    scale_factor=1 / sine_gen.upsample_scale,
                    mode="linear",
                ).transpose(1, 2)
                phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
                phase = F.interpolate(
                    phase.transpose(1, 2) * sine_gen.upsample_scale,
                    scale_factor=sine_gen.upsample_scale,
                    mode="linear",
                ).transpose(1, 2)
                sines = torch.sin(phase)
            else:
                raise RuntimeError("flag_for_pulse mode not supported in deterministic")
            return sines

        sine_gen._f02sine = deterministic_f02sine

        # Patch SineGen.forward to use zeros for noise
        _original_sine_forward = sine_gen.forward  # noqa: F841 - saved for potential restore

        def deterministic_sine_forward(f0):
            fn = torch.multiply(
                f0,
                torch.FloatTensor([[list(range(1, sine_gen.harmonic_num + 2))]]).to(
                    f0.device
                ),
            )
            sine_waves = sine_gen._f02sine(fn) * sine_gen.sine_amp
            uv = sine_gen._f02uv(f0)
            # CHANGE: Use zeros instead of torch.randn_like for noise
            noise = torch.zeros_like(sine_waves)
            sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise

        sine_gen.forward = deterministic_sine_forward

        # Patch SourceModuleHnNSF.forward to use zeros for noise_source
        _original_source_forward = source_mod.forward  # noqa: F841 - saved for potential restore

        def deterministic_source_forward(x):
            with torch.no_grad():
                sine_wavs, uv, _ = sine_gen.forward(x)
            sine_merge = source_mod.l_tanh(source_mod.l_linear(sine_wavs))
            # CHANGE: Use zeros instead of torch.randn_like for noise
            noise = torch.zeros_like(uv)
            return sine_merge, noise, uv

        source_mod.forward = deterministic_source_forward

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

        # Capture Generator intermediate tensors using hooks
        # CRITICAL: We must capture har values from INSIDE the decoder call,
        # not from a separate execution, because SourceModule uses random values.
        gen_intermediates = {}

        gen = model.decoder.generator

        # Hook into m_source to capture har values from the actual decoder run
        original_m_source = gen.m_source.forward
        captured_source_outputs = {}

        def capture_m_source(f0_up):
            har_source, noi_source, uv = original_m_source(f0_up)
            captured_source_outputs["har_source"] = har_source.clone()
            captured_source_outputs["noi_source"] = noi_source.clone()
            captured_source_outputs["uv"] = uv.clone()
            return har_source, noi_source, uv

        gen.m_source.forward = capture_m_source

        # Hook into stft.transform to capture har
        original_transform = gen.stft.transform
        captured_stft_outputs = {}

        def capture_transform(x):
            har_spec, har_phase = original_transform(x)
            captured_stft_outputs["har_spec"] = har_spec.clone()
            captured_stft_outputs["har_phase"] = har_phase.clone()
            return har_spec, har_phase

        gen.stft.transform = capture_transform

        # Hook into stft.inverse to capture FINAL spec/phase going into ISTFT
        # This is DIFFERENT from stft.transform - the generator processes
        # the network output with exp() and sin() before calling inverse
        original_inverse = gen.stft.inverse
        captured_inverse_inputs = {}

        def capture_inverse(magnitude, phase):
            captured_inverse_inputs["istft_magnitude"] = magnitude.clone()
            captured_inverse_inputs["istft_phase"] = phase.clone()
            return original_inverse(magnitude, phase)

        gen.stft.inverse = capture_inverse

        # Hook to capture generator input from decoder
        decoder_gen_input = {}
        original_gen_call = gen.forward

        def capture_gen_input(x, s, F0, *args, **kwargs):
            decoder_gen_input["x"] = _to_numpy(x)
            decoder_gen_input["s"] = _to_numpy(s)
            decoder_gen_input["F0"] = _to_numpy(F0)
            return original_gen_call(x, s, F0, *args, **kwargs)

        gen.forward = capture_gen_input

        # Run decoder - this captures all intermediates from the SAME pass
        audio = model.decoder(asr, F0_pred, N_pred, style128).squeeze()

        # Restore original functions
        gen.m_source.forward = original_m_source
        gen.stft.transform = original_transform
        gen.stft.inverse = original_inverse
        gen.forward = original_gen_call

        # Store captured intermediates (these are now consistent with audio)
        if captured_source_outputs:
            har_source = captured_source_outputs["har_source"]
            noi_source = captured_source_outputs["noi_source"]
            uv = captured_source_outputs["uv"]

            # Compute gen_f0_up for reference
            f0_up = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
            gen_intermediates["gen_f0_up"] = _to_numpy(f0_up)
            gen_intermediates["gen_har_source"] = _to_numpy(har_source)
            gen_intermediates["gen_noi_source"] = _to_numpy(noi_source)
            gen_intermediates["gen_uv"] = _to_numpy(uv)

        if captured_stft_outputs:
            har_spec = captured_stft_outputs["har_spec"]
            har_phase = captured_stft_outputs["har_phase"]
            har = torch.cat([har_spec, har_phase], dim=1)
            gen_intermediates["gen_har"] = _to_numpy(har)
            gen_intermediates["gen_har_spec"] = _to_numpy(har_spec)
            gen_intermediates["gen_har_phase"] = _to_numpy(har_phase)

        # Capture FINAL spec/phase that go into ISTFT (after exp/sin processing)
        if captured_inverse_inputs:
            gen_intermediates["istft_magnitude"] = _to_numpy(
                captured_inverse_inputs["istft_magnitude"]
            )
            gen_intermediates["istft_phase"] = _to_numpy(
                captured_inverse_inputs["istft_phase"]
            )

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

    # Add Generator intermediate tensors
    tensors.update(gen_intermediates)

    # Add decoder-to-generator interface tensors (captured via hook)
    if decoder_gen_input:
        tensors["decoder_gen_x"] = decoder_gen_input.get("x")  # Generator input from decoder
        tensors["decoder_gen_s"] = decoder_gen_input.get("s")  # Style passed to generator
        tensors["decoder_gen_F0"] = decoder_gen_input.get("F0")  # F0 passed to generator

    np.savez_compressed(out_dir / "tensors.npz", **tensors)

    import torch

    metadata = {
        "repo_id": repo_id,
        "text": text,
        "lang": lang,
        "voice": voice,
        "speed": speed,
        "device": device,
        "seed": seed,
        "deterministic_source": deterministic_source,
        "torch_version": torch.__version__,
        "graphemes": graphemes,
        "phonemes": phonemes,
        "input_ids_len": len(input_ids_list),
        "note": "Run validate script in MLX env to compare decoder/generator. Seed ensures deterministic export.",
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
        "--device",
        type=str,
        default=None,
        help="cpu/cuda/mps (default: cpu for determinism)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for deterministic export"
    )
    parser.add_argument(
        "--repo-id", type=str, default="hexgrad/Kokoro-82M", help="HF repo id"
    )
    parser.add_argument(
        "--deterministic-source",
        action="store_true",
        help="Use zeros instead of random values in SourceModule (matches MLX deterministic mode)",
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
        seed=args.seed,
        deterministic_source=args.deterministic_source,
    )
    print(f"Wrote {args.out_dir}/metadata.json and {args.out_dir}/tensors.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
