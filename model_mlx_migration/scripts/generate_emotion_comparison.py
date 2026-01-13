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
Generate comparison audio: Regular Kokoro vs Emotional Kokoro
For English (af_heart) and Japanese (jf_alpha) voices.
"""

import os
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

# Test sentences
SENTENCES = {
    "en": {
        "neutral": "The weather today is partly cloudy with a chance of rain.",
        "angry": "How dare you say that to me after everything I have done!",
        "sad": "I just received terrible news about the test results.",
        "excited": "We won the championship! This is the best day ever!",
    },
    "ja": {
        "neutral": "今日の天気は晴れです。",
        "angry": "なんてことをしてくれたんだ！許せない！",
        "sad": "悲しいお知らせがあります。",
        "excited": "やったー！優勝だ！最高の日だ！",
    }
}

PROSODY_IDS = {"neutral": 0, "angry": 40, "sad": 41, "excited": 42}


def load_model_with_prosody():
    """Load Kokoro model with v5 prosody conditioning."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    print("Loading Kokoro model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Load prosody embeddings
    prosody_emb_path = "models/prosody_embeddings_orthogonal/final.safetensors"
    fc_weights_path = "models/prosody_f0_conditioning_v5/fc_prosody_weights.safetensors"

    prosody_weights = mx.load(prosody_emb_path)
    embedding_table = prosody_weights["embedding.weight"]

    # Enable Phase C prosody conditioning
    model.predictor.enable_prosody_conditioning(prosody_dim=768, prosody_scale=0.1)
    model.predictor.copy_f0_weights_to_prosody_blocks()

    # Load fc_prosody weights
    fc_weights = mx.load(fc_weights_path)
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

    return model, converter, embedding_table


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000):
    """Save audio to WAV file."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return path


def text_to_tokens(text: str, lang: str = "en"):
    """Convert text to Kokoro tokens."""
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    phonemes, token_ids = phonemize_text(text, language=lang)
    return mx.array([token_ids]), len(phonemes)


def synthesize_baseline(model, converter, text: str, voice_name: str, lang: str):
    """Synthesize without prosody (baseline)."""
    input_ids, phoneme_length = text_to_tokens(text, lang)
    voice = converter.load_voice(voice_name, phoneme_length=phoneme_length)
    mx.eval(voice)
    audio = model.synthesize(input_ids, voice)
    mx.eval(audio)
    return np.array(audio).flatten()


def synthesize_with_prosody(model, converter, embedding_table, text: str, voice_name: str, lang: str, prosody_type: int):
    """Synthesize with Phase C prosody conditioning."""
    input_ids, phoneme_length = text_to_tokens(text, lang)
    voice = converter.load_voice(voice_name, phoneme_length=phoneme_length)
    mx.eval(voice)

    if voice.shape[-1] == 256:
        style = voice[:, :128]
        speaker = voice[:, 128:]
    else:
        style = voice
        speaker = voice

    prosody_emb = embedding_table[prosody_type:prosody_type+1]

    bert_out = model.bert(input_ids, None)
    bert_enc = model.bert_encoder(bert_out)
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    dur_enc = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(dur_enc)
    indices, total_frames, _ = model._compute_alignment(duration_logits, speed=1.0)
    en_expanded_640 = model._expand_features(duration_feats, indices, total_frames)
    x_shared = model.predictor.shared(en_expanded_640)

    # Phase C: F0 with prosody
    x = x_shared
    x = model.predictor.F0_0_prosody(x, speaker, prosody_emb)
    x = model.predictor.F0_1_prosody(x, speaker, prosody_emb)
    x = model.predictor.F0_2_prosody(x, speaker, prosody_emb)
    f0 = model.predictor.F0_proj(x).squeeze(-1)

    # Noise (standard path)
    x = x_shared
    x = model.predictor.N_0(x, speaker)
    x = model.predictor.N_1(x, speaker)
    x = model.predictor.N_2(x, speaker)
    noise = model.predictor.N_proj(x).squeeze(-1)

    text_enc = model.text_encoder(input_ids, None)
    asr_expanded = model._expand_features(text_enc, indices, total_frames)

    audio = model.decoder(asr_expanded, f0, noise, style)
    mx.eval(audio)

    return np.array(audio).flatten()


def main():
    output_dir = Path("/Users/ayates/voice/prosody_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, converter, embedding_table = load_model_with_prosody()

    voices = {
        "en": "af_heart",
        "ja": "jf_alpha",
    }

    for lang, voice in voices.items():
        print(f"\n{'='*60}")
        print(f"Generating {lang.upper()} samples with {voice}")
        print("="*60)

        for emotion, text in SENTENCES[lang].items():
            print(f"\n{emotion.upper()}: {text[:40]}...")

            # Baseline (no prosody)
            baseline_path = output_dir / f"{lang}_{emotion}_baseline.wav"
            audio_baseline = synthesize_baseline(model, converter, text, voice, lang)
            save_wav(audio_baseline, str(baseline_path))
            print(f"  Saved: {baseline_path.name}")

            # With prosody
            prosody_id = PROSODY_IDS.get(emotion, 0)
            prosody_path = output_dir / f"{lang}_{emotion}_v5prosody.wav"
            audio_prosody = synthesize_with_prosody(
                model, converter, embedding_table, text, voice, lang, prosody_id
            )
            save_wav(audio_prosody, str(prosody_path))
            print(f"  Saved: {prosody_path.name}")

    print(f"\n\nAll files saved to: {output_dir}")
    print("\nTo play comparison:")
    print(f"  afplay {output_dir}/en_angry_baseline.wav")
    print(f"  afplay {output_dir}/en_angry_v5prosody.wav")


if __name__ == "__main__":
    main()
