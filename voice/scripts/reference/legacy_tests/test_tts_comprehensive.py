#!/usr/bin/env python3
"""
Comprehensive TTS Testing Suite

This script tests TTS output with multiple phrases and provides detailed feedback.
It's the definitive test for TTS quality - if this passes, the TTS works.

Usage:
    # Test a TTS command
    python tests/test_tts_comprehensive.py --command "echo 'text' | ./tts" --phrases short

    # Test existing WAV files
    python tests/test_tts_comprehensive.py --wav /tmp/test.wav --expected "Hello world"

    # Test with all phrase sets
    python tests/test_tts_comprehensive.py --command "./my_tts.sh" --phrases all

Exit codes:
    0 - All tests passed (>50% word match on all phrases)
    1 - One or more tests failed
"""

import sys
import os
import argparse
import subprocess
import tempfile
import re
from typing import List, Tuple, Optional

# Test phrase sets
PHRASES = {
    "short": [
        "Hello world",
        "Testing one two three",
        "The quick brown fox",
    ],
    "medium": [
        "Hello, this is a test of the text to speech system.",
        "The weather is beautiful today, perfect for a walk.",
        "Please remember to save your work before closing.",
    ],
    "long": [
        "In the beginning, there was silence. Then came the voice, clear and strong, speaking words that would change everything.",
        "The artificial intelligence revolution has transformed how we interact with technology, making speech synthesis more natural than ever.",
    ],
    "numbers": [
        "One two three four five",
        "The year is twenty twenty four",
        "Call me at five five five one two three four",
    ],
    "punctuation": [
        "Hello! How are you?",
        "Wait... what was that?",
        "Yes, no, maybe, I don't know.",
    ],
}


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text


def transcribe_audio(wav_path: str, model_size: str = "base") -> str:
    """Transcribe audio using Whisper."""
    try:
        import whisper
    except ImportError:
        print("ERROR: Whisper not installed. Run: pip install openai-whisper")
        sys.exit(1)

    model = whisper.load_model(model_size)
    result = model.transcribe(wav_path)
    return result["text"]


def calculate_match(expected: str, transcribed: str) -> Tuple[float, set, set]:
    """Calculate word match ratio."""
    expected_norm = normalize_text(expected)
    transcribed_norm = normalize_text(transcribed)

    expected_words = set(expected_norm.split())
    transcribed_words = set(transcribed_norm.split())

    if not expected_words:
        return 0.0, set(), set()

    matched = expected_words & transcribed_words
    ratio = len(matched) / len(expected_words)

    return ratio, expected_words, matched


def test_single(wav_path: str, expected: str, verbose: bool = True) -> bool:
    """Test a single WAV file."""
    transcribed = transcribe_audio(wav_path)
    ratio, expected_words, matched = calculate_match(expected, transcribed)

    passed = ratio > 0.5

    if verbose:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status} ({ratio:.0%} match)")
        print(f"  Expected:    {expected}")
        print(f"  Transcribed: {transcribed.strip()}")
        if not passed:
            missing = expected_words - matched
            print(f"  Missing words: {missing}")

    return passed


def generate_and_test(command_template: str, text: str, verbose: bool = True) -> bool:
    """Generate audio with command and test it."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        # Replace {text} placeholder in command
        command = command_template.replace("{text}", text)
        command = command.replace("{output}", wav_path)

        if verbose:
            print(f"\n📢 Testing: \"{text}\"")
            print(f"   Command: {command}")

        # Run TTS command
        result = subprocess.run(command, shell=True, capture_output=True, timeout=60)

        if result.returncode != 0:
            print(f"   ❌ TTS command failed: {result.stderr.decode()}")
            return False

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            print(f"   ❌ No audio generated or file too small")
            return False

        return test_single(wav_path, text, verbose)

    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def run_test_suite(command_template: str, phrase_sets: List[str], verbose: bool = True) -> Tuple[int, int]:
    """Run full test suite."""
    passed = 0
    total = 0

    print("=" * 60)
    print("TTS COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    for phrase_set in phrase_sets:
        if phrase_set not in PHRASES:
            print(f"Unknown phrase set: {phrase_set}")
            continue

        print(f"\n--- Phrase Set: {phrase_set} ---")

        for phrase in PHRASES[phrase_set]:
            total += 1
            if generate_and_test(command_template, phrase, verbose):
                passed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 60)

    if passed == total:
        print("✅ ALL TESTS PASSED - TTS is working!")
    elif passed > total * 0.5:
        print("⚠️  PARTIAL SUCCESS - TTS mostly works but has issues")
    else:
        print("❌ TESTS FAILED - TTS is NOT producing intelligible speech")

    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Comprehensive TTS testing")
    parser.add_argument("--command", help="TTS command template. Use {text} for input, {output} for WAV path")
    parser.add_argument("--wav", help="Test existing WAV file")
    parser.add_argument("--expected", help="Expected text for --wav mode")
    parser.add_argument("--phrases", default="short", help="Phrase sets: short, medium, long, numbers, punctuation, all")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    if args.wav:
        # Test single WAV file
        if not args.expected:
            print("ERROR: --expected required with --wav")
            sys.exit(1)
        passed = test_single(args.wav, args.expected, not args.quiet)
        sys.exit(0 if passed else 1)

    elif args.command:
        # Run test suite with command
        phrase_sets = list(PHRASES.keys()) if args.phrases == "all" else args.phrases.split(",")
        passed, total = run_test_suite(args.command, phrase_sets, not args.quiet)
        sys.exit(0 if passed == total else 1)

    else:
        # Show help and examples
        print(__doc__)
        print("\nEXAMPLES:")
        print()
        print("# Test macOS say (should pass):")
        print('python tests/test_tts_comprehensive.py --command "say -o {output} {text}" --phrases short')
        print()
        print("# Test espeak-ng (should pass):")
        print('python tests/test_tts_comprehensive.py --command "espeak-ng \\"{text}\\" -w {output}" --phrases short')
        print()
        print("# Test C++ TTS (currently fails):")
        print('python tests/test_tts_comprehensive.py --command "./test_cpp_tts.sh \\"{text}\\" {output}" --phrases short')
        print()
        print("# Test existing WAV:")
        print('python tests/test_tts_comprehensive.py --wav /tmp/test.wav --expected "Hello world"')


if __name__ == "__main__":
    main()
