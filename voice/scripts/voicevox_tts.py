#!/usr/bin/env python3
"""
VOICEVOX TTS Wrapper

Japanese-native TTS engine with high-quality neural voices.

Usage:
    python scripts/voicevox_tts.py "こんにちは世界" -o output.wav
    python scripts/voicevox_tts.py "Hello" -o output.wav --style 3

Popular Style IDs:
    2  - 四国めたん ノーマル
    3  - ずんだもん ノーマル (default)
    8  - 春日部つむぎ ノーマル
    9  - 波音リツ ノーマル
    14 - 冥鳴ひまり ノーマル

Performance:
    - Model load: ~2-5s
    - Synthesis: ~200-500ms
    - Sample rate: 24kHz

Credit: VOICEVOX:ずんだもん (or appropriate character)

Note: Uses providers.voicevox_provider.VoicevoxProvider internally.
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
from providers.voicevox_provider import (
    DEFAULT_STYLE_ID,
    MODEL_DIR,
    ONNXRUNTIME_PATH,
    OPEN_JTALK_DICT,
    STYLE_NAMES,
    VOICEVOX_DIR,
    VoicevoxProvider,
)

# Global provider instance (lazy initialization)
_provider = None


def get_synthesizer():
    """
    Load VOICEVOX synthesizer (cached).

    Deprecated: Use VoicevoxProvider directly instead.
    """
    global _provider
    if _provider is None:
        _provider = VoicevoxProvider(verbose=True)
    return _provider._get_synthesizer()


def list_voices():
    """List all available voices and styles."""
    global _provider
    if _provider is None:
        _provider = VoicevoxProvider(verbose=True)
    _provider.list_all_voices()


def synthesize(text: str, output_path: str, style_id: int = DEFAULT_STYLE_ID) -> bool:
    """
    Synthesize speech from text.

    Args:
        text: Japanese text to synthesize
        output_path: Path to save WAV file
        style_id: VOICEVOX style ID (default: 3 = ずんだもん ノーマル)

    Returns:
        True if successful
    """
    global _provider
    if _provider is None:
        _provider = VoicevoxProvider(verbose=True, style_id=style_id)

    return _provider.synthesize_to_file(text, output_path, "ja", str(style_id))


def main():
    parser = argparse.ArgumentParser(description="VOICEVOX TTS - Japanese neural TTS")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("-o", "--output", help="Output WAV file path")
    parser.add_argument("-s", "--style", type=int, default=DEFAULT_STYLE_ID,
                       help=f"Style ID (default: {DEFAULT_STYLE_ID} = ずんだもん)")
    parser.add_argument("--play", action="store_true",
                       help="Play audio after synthesis")
    parser.add_argument("--list-voices", action="store_true",
                       help="List available voices and styles")

    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    if not args.text:
        parser.error("text is required unless --list-voices is specified")

    if not args.output:
        parser.error("-o/--output is required")

    success = synthesize(args.text, args.output, args.style)

    if success and args.play:
        with open(args.output, 'rb') as f:
            play_audio(f.read())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
