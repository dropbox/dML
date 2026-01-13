#!/usr/bin/env python3
"""
Compare Kokoro Python output vs C++ output for prosody issues.

Tests:
1. Japanese: こんにちは (should NOT have extra syllable)
2. English: "The lazy dog." (should have FALLING intonation)
3. English: "Are you coming?" (should have RISING intonation)
4. Chinese: 你好 (should have correct tones: rising + dipping)
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_kokoro_python():
    """Generate audio using Kokoro Python package."""
    from kokoro import KPipeline
    import soundfile as sf

    test_cases = [
        ("ja", "j", "こんにちは", "japanese_konnichiwa.wav"),
        ("en", "a", "The quick brown fox jumps over the lazy dog.", "english_statement.wav"),
        ("en", "a", "Are you coming to the party tonight?", "english_question.wav"),
        ("zh", "z", "你好", "chinese_nihao.wav"),
        ("zh", "z", "今天天气很好", "chinese_sentence.wav"),
    ]

    output_dir = PROJECT_ROOT / "reports" / "prosody_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang, lang_code, text, filename in test_cases:
        print(f"\n{'='*60}")
        print(f"Language: {lang}, Text: {text}")
        print(f"{'='*60}")

        try:
            # Initialize pipeline for this language
            pipeline = KPipeline(lang_code=lang_code)

            # Generate audio
            output_path = output_dir / f"python_{filename}"

            # Get the generator - use correct voice names per language
            voice_map = {
                'j': 'jf_alpha',    # Japanese female
                'a': 'af_heart',    # American English female
                'z': 'zf_xiaobei',  # Chinese female
            }
            voice = voice_map.get(lang_code, 'af_heart')
            generator = pipeline(text, voice=voice)

            # Collect all audio chunks
            all_audio = []
            for i, (gs, ps, audio) in enumerate(generator):
                print(f"  Chunk {i}: graphemes='{gs}', phonemes='{ps}'")
                if audio is not None:
                    all_audio.append(audio)

            if all_audio:
                import numpy as np
                combined = np.concatenate(all_audio)
                sf.write(str(output_path), combined, 24000)
                print(f"  Saved: {output_path}")
                print(f"  Duration: {len(combined)/24000:.2f}s")
            else:
                print(f"  WARNING: No audio generated!")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_cpp_output():
    """Generate audio using C++ binary for comparison."""
    cpp_binary = PROJECT_ROOT / "stream-tts-cpp" / "build" / "stream-tts-cpp"

    if not cpp_binary.exists():
        print(f"C++ binary not found: {cpp_binary}")
        return

    test_cases = [
        ("ja", "こんにちは", "japanese_konnichiwa.wav"),
        ("en", "The quick brown fox jumps over the lazy dog.", "english_statement.wav"),
        ("en", "Are you coming to the party tonight?", "english_question.wav"),
        ("zh", "你好", "chinese_nihao.wav"),
        ("zh", "今天天气很好", "chinese_sentence.wav"),
    ]

    output_dir = PROJECT_ROOT / "reports" / "prosody_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang, text, filename in test_cases:
        print(f"\n{'='*60}")
        print(f"C++ - Language: {lang}, Text: {text}")
        print(f"{'='*60}")

        output_path = output_dir / f"cpp_{filename}"

        result = subprocess.run(
            [str(cpp_binary), "--speak", text, "--lang", lang, "--save-audio", str(output_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT / "stream-tts-cpp"),
            timeout=60
        )

        # Print G2P debug output
        for line in result.stderr.split('\n'):
            if 'G2P' in line or 'phoneme' in line.lower():
                print(f"  {line}")

        if output_path.exists():
            print(f"  Saved: {output_path}")
        else:
            print(f"  ERROR: Failed to generate audio")
            print(f"  stdout: {result.stdout[:500]}")
            print(f"  stderr: {result.stderr[:500]}")


if __name__ == "__main__":
    print("="*60)
    print("KOKORO PYTHON vs C++ PROSODY COMPARISON")
    print("="*60)

    print("\n\n### PYTHON OUTPUT ###\n")
    test_kokoro_python()

    print("\n\n### C++ OUTPUT ###\n")
    test_cpp_output()

    print("\n\n### RESULTS ###")
    print("Audio files saved to: reports/prosody_comparison/")
    print("Compare python_*.wav vs cpp_*.wav manually or with LLM-as-Judge")
