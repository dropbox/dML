#!/usr/bin/env python3
"""
Kokoro TTS Wrapper

Fast, lightweight TTS with Japanese support (82M parameters).

Usage:
    python scripts/kokoro_tts.py "Text to speak" -o output.wav -l ja
    python scripts/kokoro_tts.py "Hello world" -o output.wav -l en

Supported languages: en, ja, es, fr, hi, it, pt, zh

Performance:
    - Model load: ~1-15s (first time downloads weights)
    - Synthesis: ~100-500ms
    - Quality: Good

Voices:
    - Japanese: jf_alpha, jf_gongitsune, jf_nezumi, jm_kumo
    - English: af_heart, af_bella, am_adam, am_michael, etc.

Note: Uses providers.kokoro_provider.KokoroProvider internally.
"""

import argparse
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path for provider import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Re-export constants for backward compatibility
from providers import play_audio
from providers.kokoro_provider import (
    AVAILABLE_VOICES,
    DEFAULT_VOICES,
    LANG_CODE_MAP,
    KokoroProvider,
)

# Global provider instance (lazy initialization)
_provider = None


def get_pipeline(lang_code: str):
    """
    Load Kokoro pipeline (cached per language).

    Deprecated: Use KokoroProvider directly instead.
    """
    global _provider
    if _provider is None:
        _provider = KokoroProvider(verbose=True)
    return _provider._get_pipeline(lang_code)


def synthesize(text: str, output_path: str, language: str = "ja", voice: str = None) -> bool:
    """
    Synthesize speech from text.

    Args:
        text: Text to synthesize
        output_path: Path to save WAV file
        language: Language code (en, ja, es, fr, hi, it, pt, zh)
        voice: Voice name (optional, uses default for language)

    Returns:
        True if successful
    """
    global _provider
    if _provider is None:
        _provider = KokoroProvider(verbose=True)

    voice_id = voice if voice else "default"
    return _provider.synthesize_to_file(text, output_path, language, voice_id)


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("-o", "--output", help="Output WAV file path")
    parser.add_argument("-l", "--language", default="ja",
                       help="Language: en, ja, es, fr, hi, it, pt, zh (default: ja)")
    parser.add_argument("-v", "--voice", default=None,
                       help="Voice name (optional)")
    parser.add_argument("--play", action="store_true",
                       help="Play audio after synthesis")
    parser.add_argument("--list-voices", action="store_true",
                       help="List available voices for language")

    args = parser.parse_args()

    global _provider
    if _provider is None:
        _provider = KokoroProvider(verbose=True)

    if args.list_voices:
        print(f"Available voices for {args.language}:")
        for v in _provider.get_voices(args.language):
            print(f"  - {v}")
        return

    if not args.text:
        text = sys.stdin.read().strip()
    else:
        text = args.text

    if not text:
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        print("Error: -o/--output is required", file=sys.stderr)
        sys.exit(1)

    success = synthesize(text, args.output, args.language, args.voice)

    if success and args.play:
        with open(args.output, 'rb') as f:
            play_audio(f.read())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
