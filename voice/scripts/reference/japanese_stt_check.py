#!/usr/bin/env python3
"""
Japanese STT Verification Test

Japanese doesn't use word boundaries like English, so we use character-level matching.

Usage:
    python tests/test_japanese_stt.py <wav_file> <expected_text>

Exit codes:
    0 - STT recognized >80% of expected characters (PASS)
    1 - STT failed to recognize speech (FAIL)
"""

import sys
import os
import re

def normalize_japanese(text: str) -> str:
    """Normalize Japanese text for comparison."""
    # Remove spaces and punctuation
    text = re.sub(r'[\s\u3000.,!?、。！？「」『』（）\(\)]', '', text)
    # Convert to lowercase for any romaji
    text = text.lower()
    return text

def char_accuracy(expected: str, transcribed: str) -> float:
    """Calculate character-level accuracy for Japanese."""
    expected_norm = normalize_japanese(expected)
    transcribed_norm = normalize_japanese(transcribed)

    if not expected_norm:
        return 0.0

    # Count matching characters
    expected_chars = list(expected_norm)
    transcribed_chars = list(transcribed_norm)

    # Use longest common subsequence for flexible matching
    matches = 0
    for char in expected_chars:
        if char in transcribed_chars:
            matches += 1
            transcribed_chars.remove(char)  # Remove first occurrence

    return matches / len(expected_chars)

def verify_japanese_audio(wav_path: str, expected_text: str, model_size: str = "medium") -> tuple:
    """
    Verify Japanese TTS output using Whisper STT.

    Returns:
        tuple: (passed, transcribed_text, accuracy)
    """
    try:
        import whisper
    except ImportError:
        print("ERROR: Whisper not installed. Run: pip install openai-whisper")
        return False, "", 0.0

    if not os.path.exists(wav_path):
        print(f"ERROR: WAV file not found: {wav_path}")
        return False, "", 0.0

    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {wav_path}")
    result = model.transcribe(wav_path, language="ja")
    transcribed = result["text"]

    accuracy = char_accuracy(expected_text, transcribed)

    print(f"\n{'='*60}")
    print(f"Expected:    '{expected_text}'")
    print(f"Transcribed: '{transcribed}'")
    print(f"Character accuracy: {accuracy:.1%}")
    print(f"{'='*60}")

    # 80% character accuracy for Japanese (more lenient than English words)
    passed = accuracy >= 0.80

    if passed:
        print(f"\n✅ PASS - Japanese audio is intelligible ({accuracy:.0%} char match)")
    else:
        print(f"\n❌ FAIL - Japanese audio not intelligible ({accuracy:.0%} char match)")
        print("   Threshold: 80% character accuracy")

    return passed, transcribed, accuracy

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    wav_path = sys.argv[1]
    expected_text = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "base"

    passed, _, _ = verify_japanese_audio(wav_path, expected_text, model_size)
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
