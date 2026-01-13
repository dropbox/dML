#!/usr/bin/env python3
"""
Speech-to-Text Verification for TTS Output

This is the DEFINITIVE test for TTS quality. If Whisper STT cannot understand
the audio, the TTS is broken - regardless of what other metrics say.

CRITICAL: Previous audio quality tests (RMS, spectral centroid, etc.) PASSED
but audio was unintelligible. This STT test is the only reliable measure.

Usage:
    python tests/test_stt_verification.py <wav_file> <expected_text>

Exit codes:
    0 - STT recognized >50% of expected words (PASS)
    1 - STT failed to recognize speech (FAIL)

Example:
    python tests/test_stt_verification.py /tmp/tts.wav "Hello world"
"""

import sys
import os
import re

def normalize_text(text):
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(text.split())  # Normalize whitespace
    return text


def verify_audio(wav_path, expected_text, model_size="base"):
    """
    Verify TTS output using Whisper STT.

    Args:
        wav_path: Path to WAV file to verify
        expected_text: The text that should have been spoken
        model_size: Whisper model size (tiny, base, small, medium, large)

    Returns:
        tuple: (passed, transcribed_text, match_ratio)
    """
    try:
        import whisper
    except ImportError:
        print("ERROR: Whisper not installed. Run: pip install openai-whisper")
        print("This is REQUIRED for TTS verification.")
        return False, "", 0.0

    if not os.path.exists(wav_path):
        print(f"ERROR: WAV file not found: {wav_path}")
        return False, "", 0.0

    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {wav_path}")
    result = model.transcribe(wav_path)
    transcribed = result["text"]

    print(f"\n{'='*60}")
    print(f"Expected:    '{expected_text}'")
    print(f"Transcribed: '{transcribed}'")
    print(f"{'='*60}")

    # Normalize for comparison
    expected_norm = normalize_text(expected_text)
    transcribed_norm = normalize_text(transcribed)

    expected_words = set(expected_norm.split())
    transcribed_words = set(transcribed_norm.split())

    if not expected_words:
        print("ERROR: No expected words provided")
        return False, transcribed, 0.0

    # Calculate word match ratio
    matched_words = expected_words & transcribed_words
    match_ratio = len(matched_words) / len(expected_words)

    print(f"\nExpected words ({len(expected_words)}): {sorted(expected_words)}")
    print(f"Transcribed words ({len(transcribed_words)}): {sorted(transcribed_words)}")
    print(f"Matched words ({len(matched_words)}): {sorted(matched_words)}")
    print(f"Match ratio: {match_ratio:.1%}")

    # PASS threshold: >50% of expected words recognized
    passed = match_ratio > 0.5

    print(f"\n{'='*60}")
    if passed:
        print("✅ PASS - Audio is intelligible speech")
        print(f"   Whisper recognized {match_ratio:.0%} of expected words")
    else:
        print("❌ FAIL - Audio is NOT intelligible")
        print("")
        print("   The TTS output sounds like GARBAGE to the STT model.")
        print("   This means a human listener would also not understand it.")
        print("")
        print("   DO NOT claim TTS is working until this test passes!")
        print("   DO NOT modify this test to make it pass!")
        print("")
        print("   Possible causes:")
        print("   - Decoder producing noise instead of speech")
        print("   - Wrong style vectors (zeros instead of diffusion output)")
        print("   - F0/pitch information corrupted")
        print("   - Audio post-processing destroying the signal")
    print(f"{'='*60}")

    return passed, transcribed, match_ratio


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nQuick test with macOS say:")
        print("  say -o /tmp/macos_test.wav --data-format=LEI16@24000 'Hello world'")
        print("  python tests/test_stt_verification.py /tmp/macos_test.wav 'Hello world'")
        sys.exit(1)

    wav_path = sys.argv[1]
    expected_text = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "base"

    passed, transcribed, ratio = verify_audio(wav_path, expected_text, model_size)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
