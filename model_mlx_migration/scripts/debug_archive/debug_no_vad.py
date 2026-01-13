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
Debug: Compare Python with and without VAD to understand the difference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX

AUDIO_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"

def main():
    print("Loading Python WhisperMLX...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    print(f"\nFile: {AUDIO_FILE}")

    # Get the audio data directly
    from tools.whisper_mlx.audio import load_audio
    audio = load_audio(AUDIO_FILE)
    print(f"Audio length: {len(audio)/16000:.2f}s ({len(audio)} samples)")

    # Transcribe with VAD (default)
    print("\n=== WITH VAD (default) ===")
    result_vad = model.transcribe(AUDIO_FILE, language="en", temperature=0.0)
    print(f"Text: {result_vad['text']}")
    if 'vad_speech_ratio' in result_vad:
        print(f"VAD speech ratio: {result_vad['vad_speech_ratio']:.1%}")

    # Transcribe WITHOUT VAD - use internal decode directly
    print("\n=== WITHOUT VAD (raw audio) ===")
    # Use _decode_with_metrics directly on raw mel features
    from tools.whisper_mlx.audio import log_mel_spectrogram
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

    # Compute mel spectrogram manually
    mel = log_mel_spectrogram(audio, model.config.n_mels, model.config.n_fft, model.config.hop_length)
    # Pad to 30 seconds
    n_frames = 3000  # 30s * 100 frames/s
    if mel.shape[0] < n_frames:
        mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
    mel = mel[:n_frames]

    # Encode
    audio_features = model.encoder(mel[None])
    mx.eval(audio_features)

    # Decode with tokenizer
    tokenizer = get_whisper_tokenizer(multilingual=True)

    # Use the decode_with_metrics directly
    tokens, segments, avg_logprob, no_speech_prob = model._decode_with_metrics(
        audio_features,
        tokenizer,
        temperature=0.0,
        audio_duration=len(audio) / 16000,
    )

    text = tokenizer.decode(tokens)
    print(f"Text: {text}")
    print(f"Tokens ({len(tokens)}): {tokens[:20]}...")

    # Compare texts
    print("\n=== COMPARISON ===")
    text_vad = result_vad['text'].strip()
    text_raw = text.strip()

    print(f"VAD text: {text_vad[:100]}...")
    print(f"Raw text: {text_raw[:100]}...")

    if text_vad == text_raw:
        print("SAME")
    else:
        # Find first difference
        words_vad = text_vad.split()
        words_raw = text_raw.split()
        for i in range(min(len(words_vad), len(words_raw))):
            if words_vad[i] != words_raw[i]:
                print(f"First diff at word {i}: VAD='{words_vad[i]}' vs Raw='{words_raw[i]}'")
                break

if __name__ == "__main__":
    main()
