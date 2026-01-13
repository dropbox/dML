#!/usr/bin/env python3
"""
XTTS v2 TTS Wrapper for C++ Integration

Usage:
    python scripts/xtts_tts.py "Text to speak" -o output.wav -l ja
    python scripts/xtts_tts.py "Hello world" -o output.wav -l en

Supported languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko

Latency: ~5-8s for a sentence (real-time factor ~1.5-2x)
Quality: SOTA voice cloning, MOS ~4.0
"""

import argparse
import sys
import time

# Apply PyTorch patch before importing TTS
import torch
_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load

# Global model instance for caching
_tts_model = None
_reference_wav = None

def get_tts_model():
    """Load XTTS v2 model (cached)."""
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        print("Loading XTTS v2 model...", file=sys.stderr)
        start = time.time()
        _tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        print(f"Model loaded in {time.time()-start:.1f}s", file=sys.stderr)
    return _tts_model

def get_reference_wav(path=None, language="en"):
    """Get or create reference speaker wav for the target language.

    IMPORTANT: Using a language-matched speaker reference dramatically improves quality.
    Japanese output with English speaker ref produces garbage; with Japanese ref it's perfect.
    """
    import os
    import subprocess

    if path and os.path.exists(path):
        return path

    # Language-specific reference files using macOS TTS voices
    lang_refs = {
        "ja": ("/tmp/xtts_ref_japanese.wav", "Kyoko", "これはテストの音声です"),
        "zh-cn": ("/tmp/xtts_ref_chinese.wav", "Tingting", "这是语音测试"),
        "ko": ("/tmp/xtts_ref_korean.wav", "Yuna", "이것은 음성 테스트입니다"),
        "en": ("/tmp/xtts_ref_english.wav", "Samantha", "This is a test of the voice system"),
    }

    ref_path, voice, text = lang_refs.get(language, lang_refs["en"])

    if not os.path.exists(ref_path):
        # Create reference using macOS TTS with appropriate voice
        try:
            subprocess.run(
                ['say', '-o', ref_path, '--data-format=LEI16@24000', '-v', voice, text],
                capture_output=True, check=True
            )
            print(f"Created {language} speaker reference: {ref_path}", file=sys.stderr)
        except subprocess.CalledProcessError:
            # Fallback to espeak-ng if macOS voice unavailable
            print(f"macOS voice {voice} unavailable, using espeak fallback", file=sys.stderr)
            subprocess.run(
                ['espeak-ng', '-w', ref_path, 'This is a test of the voice system'],
                capture_output=True, check=True
            )

    return ref_path

def synthesize(text: str, output_path: str, language: str = "en", speaker_wav: str = None):
    """Synthesize text to speech."""
    tts = get_tts_model()
    ref_wav = get_reference_wav(speaker_wav, language)

    start = time.time()
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        language=language,
        speaker_wav=ref_wav
    )
    duration = time.time() - start
    print(f"Generated {output_path} in {duration:.1f}s", file=sys.stderr)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="XTTS v2 TTS wrapper")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")
    parser.add_argument("-l", "--language", default="en",
                       help="Language code (en, ja, zh-cn, ko, etc.)")
    parser.add_argument("-s", "--speaker", default=None,
                       help="Reference speaker WAV file for voice cloning")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Suppress progress output")

    args = parser.parse_args()

    if args.quiet:
        import logging
        logging.disable(logging.CRITICAL)

    try:
        synthesize(args.text, args.output, args.language, args.speaker)
        print(args.output)  # Print output path for C++ to capture
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
