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
Test Kokoro with proper text-to-phoneme conversion.

The previous tests used arbitrary token IDs. This test uses
the actual phonemizer to convert text to proper phoneme tokens.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def get_phoneme_tokenizer():
    """Get the Kokoro phoneme tokenizer."""
    # Kokoro uses a specific vocabulary
    # Based on analysis: tokens 0-177 (178 total)
    # Token 16 = space/padding based on usage patterns

    # Try to use the official tokenizer if available
    try:
        from kokoro import generate  # noqa: F401

        return "official"
    except ImportError:
        pass

    # Fallback: use espeak-ng phonemizer
    try:
        from phonemizer import phonemize  # noqa: F401
        from phonemizer.backend import EspeakBackend  # noqa: F401

        return "espeak"
    except ImportError:
        pass

    return None


def text_to_tokens_espeak(text: str) -> list:
    """Convert text to tokens using espeak phonemizer."""
    from phonemizer import phonemize

    # Phonemize
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )

    print(f"Phonemes: {phonemes}")

    # Kokoro vocabulary mapping (based on IPA phonemes)
    # This is a simplified mapping - actual vocab needs proper analysis
    # Token 16 appears to be word boundary

    # Simple ASCII-based mapping for now
    tokens = [16]  # Start token
    for char in phonemes:
        # Map to token range (16-150 seems to be phoneme range)
        if char == " ":
            tokens.append(16)  # Word boundary
        elif char.isalpha():
            # Map a-z to tokens 40-65
            tokens.append(40 + (ord(char.lower()) - ord("a")) % 26)
        elif char in "ˈˌːˑ":
            # Stress markers
            tokens.append(30 + "ˈˌːˑ".index(char))
        else:
            # Other IPA symbols
            tokens.append(70 + ord(char) % 50)

    tokens.append(16)  # End token

    return tokens


def text_to_tokens_kokoro_style(text: str) -> list:
    """
    Convert text to tokens following Kokoro's tokenization style.

    Based on hexgrad/Kokoro tokenization:
    - Uses IPA phonemes from espeak
    - Has specific token mappings for each phoneme
    """
    # Load the actual vocab if available
    vocab_path = Path.home() / "models" / "kokoro" / "vocab.json"

    if vocab_path.exists():
        import json

        with open(vocab_path) as f:
            vocab = json.load(f)
        print(f"Loaded vocab with {len(vocab)} tokens")
    else:
        # Use default mapping
        vocab = None

    # Simple espeak-based conversion
    try:
        from phonemizer import phonemize

        phonemes = phonemize(
            text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )
        print(f"Phonemes: '{phonemes}'")
    except Exception as e:
        print(f"Phonemizer error: {e}")
        phonemes = text

    # Basic token mapping
    # Based on Kokoro's vocab, common patterns are:
    # Space/boundary: 16
    # Vowels start around 40
    # Consonants around 50-80

    tokens = [0]  # Start of sequence
    for char in phonemes:
        if char == " ":
            tokens.append(16)
        elif char == "ˈ":  # Primary stress
            tokens.append(1)
        elif char == "ˌ":  # Secondary stress
            tokens.append(2)
        elif ord(char) < 128:
            # ASCII range
            tokens.append(16 + ord(char) % 100)
        else:
            # Extended IPA
            tokens.append(100 + hash(char) % 78)

    tokens.append(0)  # End of sequence

    return tokens


def write_wav(filename, audio, sample_rate=24000):
    """Write audio to WAV file without scipy."""
    import wave

    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def test_with_different_tokens():
    """Test audio generation with different token strategies."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    print("=" * 60)
    print("Testing Kokoro with Different Token Strategies")
    print("=" * 60)

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    # Load voice pack for reuse with different phoneme lengths
    voice_pack = converter.load_voice_pack("af_heart")
    mx.eval(voice_pack)

    test_texts = [
        "Hello",
        "Hello world",
        "Thank you",
    ]

    output_dir = Path(__file__).parent.parent / "reports" / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    for text in test_texts:
        print(f"\n=== Testing: '{text}' ===")

        # Method 1: Arbitrary tokens (what previous tests did)
        print("\n1. Arbitrary tokens:")
        arb_tokens = [16] + list(range(43, 43 + len(text))) + [16]
        print(f"   Tokens: {arb_tokens}")

        tokens_mx = mx.array([arb_tokens])
        voice = converter.select_voice_embedding(voice_pack, len(arb_tokens))
        audio = model.synthesize(tokens_mx, voice)
        mx.eval(audio)
        audio_np = np.array(audio).flatten()
        print(
            f"   Audio: duration={len(audio_np) / 24000:.3f}s, RMS={np.sqrt(np.mean(audio_np**2)):.4f}"
        )

        # Save
        filename = f"test_arb_{text.replace(' ', '_')}.wav"
        write_wav(output_dir / filename, audio_np)
        print(f"   Saved: {filename}")

        # Method 2: espeak-based if available
        try:
            print("\n2. Espeak-based tokens:")
            esp_tokens = text_to_tokens_espeak(text)
            print(
                f"   Tokens: {esp_tokens[:20]}..."
                if len(esp_tokens) > 20
                else f"   Tokens: {esp_tokens}"
            )

            tokens_mx = mx.array([esp_tokens])
            voice = converter.select_voice_embedding(voice_pack, len(esp_tokens))
            audio = model.synthesize(tokens_mx, voice)
            mx.eval(audio)
            audio_np = np.array(audio).flatten()
            print(
                f"   Audio: duration={len(audio_np) / 24000:.3f}s, RMS={np.sqrt(np.mean(audio_np**2)):.4f}"
            )

            # Save
            filename = f"test_esp_{text.replace(' ', '_')}.wav"
            write_wav(output_dir / filename, audio_np)
            print(f"   Saved: {filename}")
        except Exception as e:
            print(f"   Error: {e}")

        # Method 3: Simple ASCII-based
        print("\n3. Simple ASCII-based tokens:")
        ascii_tokens = [0]
        for char in text.lower():
            if char == " ":
                ascii_tokens.append(16)
            elif char.isalpha():
                ascii_tokens.append(40 + ord(char) - ord("a"))
            else:
                ascii_tokens.append(30)
        ascii_tokens.append(0)
        print(f"   Tokens: {ascii_tokens}")

        tokens_mx = mx.array([ascii_tokens])
        voice = converter.select_voice_embedding(voice_pack, len(ascii_tokens))
        audio = model.synthesize(tokens_mx, voice)
        mx.eval(audio)
        audio_np = np.array(audio).flatten()
        print(
            f"   Audio: duration={len(audio_np) / 24000:.3f}s, RMS={np.sqrt(np.mean(audio_np**2)):.4f}"
        )

        # Save
        filename = f"test_ascii_{text.replace(' ', '_')}.wav"
        write_wav(output_dir / filename, audio_np)
        print(f"   Saved: {filename}")


def test_with_official_kokoro():
    """Test using official kokoro package if available."""
    print("\n" + "=" * 60)
    print("Testing with Official Kokoro Package")
    print("=" * 60)

    try:
        # This would need the official kokoro package in a Python 3.10-3.13 env
        # For now, just document how to do it
        print("Official kokoro package requires Python 3.10-3.13")
        print("Install: pip install kokoro")
        print("Usage: from kokoro import generate, Voices")
        print("       audio = generate(text, voice=Voices.AF_BELLA)")
        return
    except ImportError:
        print("Official kokoro package not installed")


if __name__ == "__main__":
    test_with_different_tokens()
    test_with_official_kokoro()
