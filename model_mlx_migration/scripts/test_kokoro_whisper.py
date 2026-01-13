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
Test Kokoro MLX audio quality with Whisper transcription.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def transcribe_with_whisper(audio_path: str) -> str:
    """Transcribe audio file with mlx-whisper."""
    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-tiny",
        )
        text: str = result.get("text", "")
        return text.strip()
    except Exception as e:
        return f"Error: {e}"


def generate_and_test():
    """Generate audio samples and test with Whisper."""
    import wave

    from tools.pytorch_to_mlx.converters import KokoroConverter

    print("=" * 60)
    print("Kokoro MLX + Whisper Quality Test")
    print("=" * 60)

    # Load model
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    # Enable deterministic mode for reproducible results
    model.set_deterministic(True)
    # Load voice pack for proper phoneme-length-based selection
    voice_pack = converter.load_voice_pack("af_heart")
    mx.eval(voice_pack)

    output_dir = Path("/tmp/kokoro_test")
    output_dir.mkdir(exist_ok=True)

    # Import phonemizer for proper token conversion
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    # Test cases with actual text that gets phonemized
    test_cases = [
        {"name": "hello", "text": "Hello", "expected": "hello"},
        {"name": "thank_you", "text": "Thank you", "expected": "thank you"},
        {"name": "hello_world", "text": "Hello world", "expected": "hello world"},
    ]

    results = []
    for tc in test_cases:
        name = tc["name"]
        text = tc["text"]

        # Convert text to phonemes and tokens
        phonemes, tokens = phonemize_text(text)

        print(f"\n=== Test: {name} ===")
        print(f"Text: {text}")
        print(f"Phonemes: {phonemes}")
        print(f"Tokens: {tokens}")

        # Generate audio with properly selected voice embedding
        tokens_mx = mx.array([tokens])
        voice = converter.select_voice_embedding(voice_pack, len(tokens))
        audio = model.synthesize(tokens_mx, voice)
        mx.eval(audio)
        audio_np = np.array(audio).flatten()

        # Audio stats
        rms = np.sqrt(np.mean(audio_np**2))
        max_amp = np.max(np.abs(audio_np))
        duration = len(audio_np) / 24000

        print(f"Duration: {duration:.3f}s")
        print(f"RMS: {rms:.4f}")
        print(f"Max Amp: {max_amp:.4f}")

        # Save to WAV
        audio_path = output_dir / f"{name}.wav"
        audio_int16 = (audio_np * 32767).astype(np.int16)
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_int16.tobytes())

        # Transcribe
        transcript = transcribe_with_whisper(str(audio_path))
        print(f"Whisper: '{transcript}'")

        results.append(
            {
                "name": name,
                "text": text,
                "expected": tc["expected"],
                "phonemes": phonemes,
                "tokens": tokens,
                "rms": rms,
                "max_amp": max_amp,
                "duration": duration,
                "transcript": transcript,
            }
        )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    unique_transcripts = set(r["transcript"] for r in results)
    print(f"Unique transcriptions: {len(unique_transcripts)}")

    for r in results:
        expected = r["expected"]
        transcript = r["transcript"].lower().strip()
        match = "✓" if expected in transcript or transcript in expected else "✗"
        print(
            f"  {r['name']}: expected='{expected}' got='{r['transcript']}' {match} (RMS={r['rms']:.4f})"
        )

    # Check for hallucination pattern
    all_same = len(unique_transcripts) == 1
    if all_same:
        print("\nWARNING: All outputs have same transcription - likely hallucination!")
    else:
        print("\nGood: Different inputs produce different transcriptions")

    return results


if __name__ == "__main__":
    generate_and_test()
