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
v5 Audio Evaluation Script

Generates comprehensive audio samples for perceptual evaluation of Phase C v5 prosody.

Usage:
    python scripts/v5_audio_evaluation.py

Output: tests/prosody/v5_audio_evaluation/
"""

import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

# Prosody type IDs
PROSODY_TYPES = {
    "NEUTRAL": 0,
    "ANGRY": 40,
    "SAD": 41,
    "EXCITED": 42,
    "CALM": 45,
    "SURPRISED": 50,
}

# Test sentences designed to reveal emotional prosody
TEST_SENTENCES = [
    ("How dare you say that to me after everything I have done for you!", "angry_context"),
    ("I just received terrible news about the test results.", "sad_context"),
    ("We won the championship! This is the best day ever!", "excited_context"),
    ("Take a deep breath and relax. Everything will be fine.", "calm_context"),
    ("What? You're kidding me! That's absolutely incredible!", "surprised_context"),
    ("The weather today is partly cloudy with a chance of rain.", "neutral_context"),
]

EMOTIONS = [
    ("baseline", None),  # No prosody
    ("neutral", PROSODY_TYPES["NEUTRAL"]),
    ("angry", PROSODY_TYPES["ANGRY"]),
    ("sad", PROSODY_TYPES["SAD"]),
    ("excited", PROSODY_TYPES["EXCITED"]),
    ("calm", PROSODY_TYPES["CALM"]),
]


def load_model_with_v5_prosody():
    """Load Kokoro model with v5 prosody conditioning."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    print("Loading Kokoro model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Paths
    prosody_emb_path = "models/prosody_embeddings_orthogonal/final.safetensors"
    fc_weights_path = "models/prosody_f0_conditioning_v5/fc_prosody_weights.safetensors"

    if not os.path.exists(prosody_emb_path):
        raise FileNotFoundError(f"Prosody embeddings not found: {prosody_emb_path}")
    if not os.path.exists(fc_weights_path):
        raise FileNotFoundError(f"FC prosody weights not found: {fc_weights_path}")

    # Load prosody embeddings
    print(f"Loading prosody embeddings from {prosody_emb_path}...")
    prosody_weights = mx.load(prosody_emb_path)
    embedding_table = prosody_weights["embedding.weight"]
    print(f"Embedding table shape: {embedding_table.shape}")

    # Enable Phase C prosody conditioning
    print("Enabling Phase C prosody conditioning...")
    model.predictor.enable_prosody_conditioning(prosody_dim=768, prosody_scale=0.1)
    model.predictor.copy_f0_weights_to_prosody_blocks()

    # Load fc_prosody weights
    print(f"Loading fc_prosody weights from {fc_weights_path}...")
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


def text_to_tokens(text: str) -> Tuple[mx.array, int]:
    """Convert text to Kokoro tokens."""
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    phonemes, token_ids = phonemize_text(text)
    return mx.array([token_ids]), len(phonemes)


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


def compute_f0(audio: np.ndarray, sample_rate: int = 24000) -> Tuple[float, float]:
    """Compute mean and std F0 using librosa YIN."""
    try:
        import librosa
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sample_rate)
        f0_valid = f0[(f0 > 60) & (f0 < 480)]
        if len(f0_valid) > 0:
            return float(np.mean(f0_valid)), float(np.std(f0_valid))
    except Exception as e:
        print(f"F0 computation failed: {e}")
    return 0.0, 0.0


def synthesize_baseline(model, converter, text: str) -> np.ndarray:
    """Synthesize without prosody (baseline)."""
    input_ids, phoneme_length = text_to_tokens(text)
    voice = converter.load_voice("af_heart", phoneme_length=phoneme_length)
    mx.eval(voice)

    audio = model.synthesize(input_ids, voice)
    mx.eval(audio)

    return np.array(audio).flatten()


def synthesize_with_prosody(
    model,
    converter,
    embedding_table: mx.array,
    text: str,
    prosody_type: int
) -> np.ndarray:
    """Synthesize with Phase C prosody conditioning."""
    input_ids, phoneme_length = text_to_tokens(text)
    voice = converter.load_voice("af_heart", phoneme_length=phoneme_length)
    mx.eval(voice)

    batch_size, seq_length = input_ids.shape

    if voice.shape[-1] == 256:
        style = voice[:, :128]
        speaker = voice[:, 128:]
    else:
        style = voice
        speaker = voice

    # Get prosody embedding
    prosody_emb = embedding_table[prosody_type:prosody_type+1]  # [1, 768]

    # Run pipeline
    bert_out = model.bert(input_ids, None)
    bert_enc = model.bert_encoder(bert_out)

    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    dur_enc = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(dur_enc)

    indices, total_frames, _ = model._compute_alignment(duration_logits, speed=1.0)
    en_expanded_640 = model._expand_features(duration_feats, indices, total_frames)
    x_shared = model.predictor.shared(en_expanded_640)

    # Phase C: F0 with prosody conditioning
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

    # ASR features
    text_enc = model.text_encoder(input_ids, None)
    asr_expanded = model._expand_features(text_enc, indices, total_frames)

    # Decode
    audio = model.decoder(asr_expanded, f0, noise, style)
    mx.eval(audio)

    return np.array(audio).flatten()


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        return f"[Transcription error: {e}]"


def main():
    output_dir = Path("tests/prosody/v5_audio_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, converter, embedding_table = load_model_with_v5_prosody()

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "v5",
        "prosody_embeddings": "models/prosody_embeddings_orthogonal/final.safetensors",
        "fc_weights": "models/prosody_f0_conditioning_v5/fc_prosody_weights.safetensors",
        "sentences": [],
        "f0_summary": {},
    }

    print("\n" + "=" * 60)
    print("V5 AUDIO EVALUATION")
    print("=" * 60)

    all_f0_diffs = {emo: [] for emo, _ in EMOTIONS if emo != "baseline"}

    for sent_idx, (sentence, context) in enumerate(TEST_SENTENCES):
        print(f"\n[{sent_idx+1}/{len(TEST_SENTENCES)}] {context}")
        print(f"  Text: {sentence[:60]}...")

        sent_result = {
            "context": context,
            "text": sentence,
            "audio_files": {},
            "f0_values": {},
            "f0_diffs": {},
            "transcriptions": {},
        }

        baseline_f0 = None

        for emo_name, emo_id in EMOTIONS:
            try:
                if emo_id is None:
                    # Baseline (no prosody)
                    audio = synthesize_baseline(model, converter, sentence)
                else:
                    # With prosody
                    audio = synthesize_with_prosody(model, converter, embedding_table, sentence, emo_id)

                filename = f"{context}_{emo_name}.wav"
                filepath = str(output_dir / filename)
                save_wav(audio, filepath)

                # Compute F0
                f0_mean, f0_std = compute_f0(audio)

                # Store baseline for comparison
                if emo_name == "baseline":
                    baseline_f0 = f0_mean

                # Compute diff
                f0_diff_pct = 0.0
                if baseline_f0 and baseline_f0 > 0 and emo_name != "baseline":
                    f0_diff_pct = ((f0_mean - baseline_f0) / baseline_f0) * 100
                    all_f0_diffs[emo_name].append(f0_diff_pct)

                sent_result["audio_files"][emo_name] = filename
                sent_result["f0_values"][emo_name] = {"mean": f0_mean, "std": f0_std}
                sent_result["f0_diffs"][emo_name] = f0_diff_pct

                diff_str = f"{f0_diff_pct:+.1f}%" if emo_name != "baseline" else ""
                print(f"  ✓ {emo_name:10}: F0={f0_mean:.1f}Hz {diff_str}")

            except Exception as e:
                print(f"  ✗ {emo_name}: {e}")
                sent_result["audio_files"][emo_name] = None

        results["sentences"].append(sent_result)

    # Summary statistics
    print("\n" + "=" * 60)
    print("F0 CHANGE SUMMARY (vs baseline)")
    print("=" * 60)

    expected = {
        "neutral": 0,
        "angry": +7,
        "sad": -4,
        "excited": +15,
        "calm": 0,
    }

    print(f"{'Emotion':<12} {'Avg F0 Change':>15} {'Expected':>12} {'Status':>10}")
    print("-" * 49)

    for emo_name in ["neutral", "angry", "sad", "excited", "calm"]:
        if emo_name in all_f0_diffs and all_f0_diffs[emo_name]:
            avg = np.mean(all_f0_diffs[emo_name])
            exp = expected.get(emo_name, 0)

            # Evaluate match
            if exp == 0:
                status = "PASS" if abs(avg) < 5 else "FAIL"
            else:
                direction_ok = (avg * exp > 0)  # Same sign
                status = "PASS" if direction_ok else "FAIL"

            results["f0_summary"][emo_name] = {
                "avg_f0_diff_pct": avg,
                "expected_pct": exp,
                "status": status,
            }

            print(f"{emo_name:<12} {avg:>+14.1f}% {exp:>+11d}% {status:>10}")

    # Run Whisper transcription on subset
    print("\n" + "=" * 60)
    print("WHISPER TRANSCRIPTION QUALITY CHECK")
    print("=" * 60)

    try:
        import whisper
        whisper_model = whisper.load_model("base")

        # Test a subset of files
        test_files = [
            ("neutral_context_baseline.wav", "The weather today is partly cloudy with a chance of rain."),
            ("angry_context_angry.wav", "How dare you say that to me after everything I have done for you!"),
            ("sad_context_sad.wav", "I just received terrible news about the test results."),
            ("excited_context_excited.wav", "We won the championship! This is the best day ever!"),
        ]

        transcription_results = []
        for filename, expected_text in test_files:
            filepath = output_dir / filename
            if filepath.exists():
                result = whisper_model.transcribe(str(filepath))
                transcribed = result["text"].strip()

                # Simple match check (case-insensitive, ignore punctuation)
                exp_clean = expected_text.lower().replace("!", "").replace("?", "").replace(".", "").strip()
                trans_clean = transcribed.lower().replace("!", "").replace("?", "").replace(".", "").strip()
                match = exp_clean == trans_clean

                print(f"\n{filename}:")
                print(f"  Expected:    {expected_text}")
                print(f"  Transcribed: {transcribed}")
                print(f"  Match: {'EXACT' if match else 'DIFFERENT'}")

                transcription_results.append({
                    "file": filename,
                    "expected": expected_text,
                    "transcribed": transcribed,
                    "match": match,
                })

        results["transcription_check"] = transcription_results

        # Count matches
        matches = sum(1 for r in transcription_results if r["match"])
        total = len(transcription_results)
        print(f"\nTranscription Accuracy: {matches}/{total} ({matches/total*100:.0f}%)")

    except ImportError:
        print("Whisper not available for transcription check")
        results["transcription_check"] = "whisper_not_available"

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Audio files: {output_dir}/")
    print(f"Results: {results_path}")

    # Print overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)

    all_pass = all(
        results["f0_summary"].get(e, {}).get("status") == "PASS"
        for e in ["neutral", "angry", "sad", "excited", "calm"]
    )

    if all_pass:
        print("✓ All emotion F0 targets PASS")
        print("✓ v5 ready for perceptual evaluation")
        print("\nRECOMMENDATION: Listen to audio files to confirm perceptual quality")
    else:
        print("✗ Some emotion F0 targets FAIL")
        print("\nRECOMMENDATION: Investigate failing emotions before deployment")

    return results


if __name__ == "__main__":
    main()
