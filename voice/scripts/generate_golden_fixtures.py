#!/usr/bin/env python3
"""
Generate golden audio fixtures from working TTS backends.

These fixtures serve as reference audio for:
1. Integration testing (verify optimizations don't degrade quality)
2. Regression detection (compare new output against known-good)
3. Cross-validation (ensure C++ exports match Python)

Usage:
    python scripts/generate_golden_fixtures.py --backend kokoro --langs en,ja
    python scripts/generate_golden_fixtures.py --all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Test sentences for each language
TEST_CORPUS = {
    "en": [
        ("hello", "Hello, how are you?"),
        ("greeting", "Good morning, it's nice to meet you."),
        ("numbers", "The temperature is 72 degrees."),
        ("question", "What time does the meeting start?"),
        ("technical", "The server is processing your request."),
    ],
    "ja": [
        ("konnichiwa", "こんにちは"),
        ("greeting", "お元気ですか"),
        ("speak", "私は日本語を話します"),
        ("server", "サーバーが起動しています"),
        ("error", "エラーが発生しました"),
    ],
    "zh": [
        ("hello", "你好"),
        ("greeting", "很高兴认识你"),
        ("numbers", "今天是十二月四号"),
        ("question", "你想吃什么"),
        ("thanks", "非常感谢"),
    ],
}


def generate_kokoro(output_dir: Path, langs: list[str]) -> list[dict]:
    """Generate fixtures using Kokoro TTS."""
    from scripts.kokoro_tts import synthesize

    results = []
    for lang in langs:
        if lang not in TEST_CORPUS:
            print(f"Skipping unknown language: {lang}", file=sys.stderr)
            continue

        for sentence_id, text in TEST_CORPUS[lang]:
            output_path = output_dir / f"{lang}_{sentence_id}.wav"
            print(f"Generating: {output_path}", file=sys.stderr)

            success = synthesize(text, str(output_path), language=lang)
            if success:
                results.append({
                    "filename": output_path.name,
                    "text": text,
                    "language": lang,
                    "voice": "default",
                    "expected_stt": text,
                    "generated_at": datetime.utcnow().isoformat() + "Z"
                })
            else:
                print(f"FAILED: {text}", file=sys.stderr)

    return results


def generate_melotts(output_dir: Path, langs: list[str]) -> list[dict]:
    """Generate fixtures using MeloTTS."""
    from scripts.melotts_tts import synthesize

    results = []
    for lang in langs:
        if lang not in TEST_CORPUS:
            continue

        # MeloTTS uses different language codes
        melotts_lang = {"en": "EN", "ja": "JP", "zh": "ZH"}.get(lang, lang.upper())

        for sentence_id, text in TEST_CORPUS[lang]:
            output_path = output_dir / f"{lang}_{sentence_id}.wav"
            print(f"Generating: {output_path}", file=sys.stderr)

            success = synthesize(text, str(output_path), language=melotts_lang)
            if success:
                results.append({
                    "filename": output_path.name,
                    "text": text,
                    "language": lang,
                    "voice": "default",
                    "expected_stt": text,
                    "generated_at": datetime.utcnow().isoformat() + "Z"
                })

    return results


def generate_openai(output_dir: Path, langs: list[str]) -> list[dict]:
    """Generate fixtures using OpenAI TTS."""
    from providers.openai_tts_client import synthesize_to_wav

    results = []
    for lang in langs:
        if lang not in TEST_CORPUS:
            continue

        for sentence_id, text in TEST_CORPUS[lang]:
            output_path = output_dir / f"{lang}_{sentence_id}.wav"
            print(f"Generating: {output_path}", file=sys.stderr)

            success = synthesize_to_wav(text, str(output_path), voice="nova", model="tts-1-hd")
            if success:
                results.append({
                    "filename": output_path.name,
                    "text": text,
                    "language": lang,
                    "voice": "nova",
                    "expected_stt": text,
                    "generated_at": datetime.utcnow().isoformat() + "Z"
                })

    return results


GENERATORS = {
    "kokoro": generate_kokoro,
    "melotts": generate_melotts,
    "openai": generate_openai,
}


def main():
    parser = argparse.ArgumentParser(description="Generate golden audio fixtures")
    parser.add_argument("--backend", choices=list(GENERATORS.keys()),
                       help="TTS backend to use")
    parser.add_argument("--langs", default="en,ja",
                       help="Comma-separated language codes (default: en,ja)")
    parser.add_argument("--all", action="store_true",
                       help="Generate for all backends")
    parser.add_argument("--output-dir", default="tests/golden_fixtures",
                       help="Output directory")

    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    langs = [l.strip() for l in args.langs.split(",")]

    backends = list(GENERATORS.keys()) if args.all else [args.backend]
    if not args.backend and not args.all:
        parser.error("Specify --backend or --all")

    for backend in backends:
        if backend not in GENERATORS:
            print(f"Unknown backend: {backend}", file=sys.stderr)
            continue

        output_dir = base_dir / backend
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Generating {backend} fixtures ===", file=sys.stderr)
        generator = GENERATORS[backend]

        try:
            results = generator(output_dir, langs)

            # Write manifest
            manifest_path = output_dir / "manifest.json"
            manifest = {"backend": backend, "files": results}
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

            print(f"Generated {len(results)} fixtures for {backend}", file=sys.stderr)
            print(f"Manifest: {manifest_path}", file=sys.stderr)

        except Exception as e:
            print(f"ERROR generating {backend}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
