#!/usr/bin/env python3
"""
MeloTTS Wrapper for Multi-language TTS

Faster than XTTS v2 (~1s vs ~7s), good quality for Japanese/Korean/Chinese/English.

Usage:
    python scripts/melotts_tts.py "Text to speak" -o output.wav -l JP
    python scripts/melotts_tts.py "Hello world" -o output.wav -l EN

Supported languages: EN, JP, KR, ZH, FR, ES

Performance:
    - Model load: ~10s (first time)
    - Synthesis: ~1s for short sentences
    - Quality: Good (passes 80%+ STT verification)

Known issues:
    - "こんにちは" may be pronounced as "今日は" (text normalization)
    - Use clear, unambiguous Japanese text for best results
"""

import argparse
import sys
import time
import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global model cache
_models = {}

def get_model(language: str):
    """Load MeloTTS model (cached per language)."""
    global _models

    # Normalize language code
    lang_map = {
        'en': 'EN', 'english': 'EN',
        'ja': 'JP', 'jp': 'JP', 'japanese': 'JP',
        'ko': 'KR', 'kr': 'KR', 'korean': 'KR',
        'zh': 'ZH', 'chinese': 'ZH',
        'fr': 'FR', 'french': 'FR',
        'es': 'ES', 'spanish': 'ES',
    }
    lang = lang_map.get(language.lower(), language.upper())

    if lang not in _models:
        from melo.api import TTS
        print(f"Loading MeloTTS model for {lang}...", file=sys.stderr)
        start = time.time()
        _models[lang] = TTS(language=lang, device='cpu')
        print(f"Model loaded in {time.time()-start:.1f}s", file=sys.stderr)

    return _models[lang], lang


def synthesize(text: str, output_path: str, language: str = "JP", speed: float = 1.0) -> bool:
    """
    Synthesize speech from text.

    Args:
        text: Text to synthesize
        output_path: Path to save WAV file
        language: Language code (EN, JP, KR, ZH, FR, ES)
        speed: Speech speed (1.0 = normal)

    Returns:
        True if successful
    """
    try:
        model, lang = get_model(language)

        # Get speaker ID for language
        speaker_ids = model.hps.data.spk2id
        if lang in speaker_ids:
            speaker_id = speaker_ids[lang]
        elif f'{lang}-Default' in speaker_ids:
            speaker_id = speaker_ids[f'{lang}-Default']
        else:
            # Use first available speaker
            speaker_id = list(speaker_ids.values())[0]

        print(f"Synthesizing ({lang}): {text[:50]}...", file=sys.stderr)
        start = time.time()
        model.tts_to_file(text, speaker_id, output_path, speed=speed)
        print(f"Synthesized in {time.time()-start:.2f}s", file=sys.stderr)

        return True

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="MeloTTS Multi-language TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")
    parser.add_argument("-l", "--language", default="JP",
                       help="Language: EN, JP, KR, ZH, FR, ES (default: JP)")
    parser.add_argument("-s", "--speed", type=float, default=1.0,
                       help="Speech speed (default: 1.0)")
    parser.add_argument("--play", action="store_true",
                       help="Play audio after synthesis")

    args = parser.parse_args()

    success = synthesize(args.text, args.output, args.language, args.speed)

    if success and args.play:
        import subprocess
        subprocess.run(["afplay", args.output])

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
