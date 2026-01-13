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
Test different prosody strength levels to find optimal perceptual impact.
"""
import os
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import mlx.core as mx


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main():
    from tools.pytorch_to_mlx.converters import KokoroConverter
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    output_dir = Path("/Users/ayates/voice/prosody_strength_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test sentence
    text = "How dare you say that to me after everything I have done!"
    prosody_id = 40  # ANGRY

    print("Loading Kokoro model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Load prosody embeddings
    prosody_weights = mx.load("models/prosody_embeddings_orthogonal/final.safetensors")
    embedding_table = prosody_weights["embedding.weight"]

    # Load fc_prosody weights
    fc_weights = mx.load("models/prosody_f0_conditioning_v5/fc_prosody_weights.safetensors")

    # Test different prosody scales
    scales = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]

    for scale in scales:
        print(f"\n{'='*60}")
        print(f"Testing prosody_scale = {scale}")
        print("="*60)

        # Re-enable prosody with new scale
        model.predictor.enable_prosody_conditioning(prosody_dim=768, prosody_scale=scale)
        model.predictor.copy_f0_weights_to_prosody_blocks()

        # Reload fc_prosody weights
        for key, value in fc_weights.items():
            parts = key.split(".")
            if len(parts) < 4:
                continue
            block_name = parts[0]
            if not hasattr(model.predictor, block_name):
                continue
            block = getattr(model.predictor, block_name)
            target = block
            for part in parts[1:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    target = None
                    break
            if target is not None:
                param_name = parts[-1]
                if hasattr(target, param_name):
                    setattr(target, param_name, value)

        # Synthesize
        phonemes, token_ids = phonemize_text(text)
        input_ids = mx.array([token_ids])
        voice = converter.load_voice("af_heart", phoneme_length=len(phonemes))
        mx.eval(voice)

        if voice.shape[-1] == 256:
            style = voice[:, :128]
            speaker = voice[:, 128:]
        else:
            style = voice
            speaker = voice

        prosody_emb = embedding_table[prosody_id:prosody_id+1]

        # Run pipeline
        bert_out = model.bert(input_ids, None)
        bert_enc = model.bert_encoder(bert_out)
        duration_feats = model.predictor.text_encoder(bert_enc, speaker)
        dur_enc = model.predictor.lstm(duration_feats)
        duration_logits = model.predictor.duration_proj(dur_enc)
        indices, total_frames, _ = model._compute_alignment(duration_logits, speed=1.0)
        en_expanded_640 = model._expand_features(duration_feats, indices, total_frames)
        x_shared = model.predictor.shared(en_expanded_640)

        # F0 with prosody
        x = x_shared
        x = model.predictor.F0_0_prosody(x, speaker, prosody_emb)
        x = model.predictor.F0_1_prosody(x, speaker, prosody_emb)
        x = model.predictor.F0_2_prosody(x, speaker, prosody_emb)
        f0 = model.predictor.F0_proj(x).squeeze(-1)

        # Noise
        x = x_shared
        x = model.predictor.N_0(x, speaker)
        x = model.predictor.N_1(x, speaker)
        x = model.predictor.N_2(x, speaker)
        noise = model.predictor.N_proj(x).squeeze(-1)

        text_enc = model.text_encoder(input_ids, None)
        asr_expanded = model._expand_features(text_enc, indices, total_frames)

        audio = model.decoder(asr_expanded, f0, noise, style)
        mx.eval(audio)

        audio_np = np.array(audio).flatten()

        # Compute F0
        try:
            import librosa
            f0_vals = librosa.yin(audio_np, fmin=50, fmax=500, sr=24000)
            f0_valid = f0_vals[(f0_vals > 60) & (f0_vals < 480)]
            f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0
            print(f"  Mean F0: {f0_mean:.1f} Hz")
        except Exception:
            f0_mean = 0

        # Save
        filename = f"angry_scale_{scale:.1f}.wav"
        save_wav(audio_np, str(output_dir / filename))
        print(f"  Saved: {filename}")

    print(f"\n\nFiles saved to: {output_dir}")
    print("\nPlay comparison:")
    print(f"  afplay {output_dir}/angry_scale_0.0.wav  # baseline")
    print(f"  afplay {output_dir}/angry_scale_0.5.wav  # medium")
    print(f"  afplay {output_dir}/angry_scale_1.0.wav  # full strength")


if __name__ == "__main__":
    main()
