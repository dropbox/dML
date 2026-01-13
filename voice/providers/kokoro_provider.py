#!/usr/bin/env python3
"""
Kokoro TTS Provider

Fast, lightweight TTS with multi-language support (82M parameters).

Supported languages: en, ja, es, fr, hi, it, pt, zh

Performance:
    - Model load: ~1-15s (first time downloads weights)
    - Synthesis: ~100-500ms
    - Quality: Good
"""

import io
import sys

import numpy as np

from .base import TtsProviderBase

# Language code mapping (user code -> Kokoro internal code)
LANG_CODE_MAP = {
    "en": "a",  # American English
    "ja": "j",  # Japanese
    "jp": "j",  # Alias
    "es": "e",  # Spanish
    "fr": "f",  # French
    "hi": "h",  # Hindi
    "it": "i",  # Italian
    "pt": "p",  # Portuguese
    "zh": "z",  # Mandarin Chinese
}

# Default voice per language
DEFAULT_VOICES = {
    "a": "af_heart",  # American English female
    "j": "jf_alpha",  # Japanese female
    "e": "ef_dora",  # Spanish female
    "f": "ff_siwis",  # French female
    "h": "hf_alpha",  # Hindi female
    "i": "if_alice",  # Italian female
    "p": "pf_dora",  # Portuguese female
    "z": "zf_xiaobei",  # Mandarin Chinese female
}

# All available voices per language
AVAILABLE_VOICES = {
    "a": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
          "am_adam", "am_michael"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jm_kumo"],
    "e": ["ef_dora"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "i": ["if_alice", "if_sara", "im_nicola"],
    "p": ["pf_dora"],
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_yunxi",
          "zm_yunjian", "zm_yunyang"],
}


class KokoroProvider(TtsProviderBase):
    """Kokoro TTS provider using the hexgrad/Kokoro-82M model."""

    name = "kokoro"
    sample_rate = 24000

    def __init__(self, verbose: bool = True, repo_id: str = "hexgrad/Kokoro-82M"):
        """
        Initialize Kokoro provider.

        Args:
            verbose: Print status messages
            repo_id: HuggingFace model repository ID
        """
        super().__init__(verbose)
        self.repo_id = repo_id
        self._pipelines: dict = {}  # Cached pipelines per language

    def _get_pipeline(self, lang_code: str):
        """Get or create pipeline for language (cached)."""
        if lang_code not in self._pipelines:
            from kokoro import KPipeline

            self._log(f"Loading model for language '{lang_code}'...")
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code, repo_id=self.repo_id
            )
            self._initialized = True

        return self._pipelines[lang_code]

    def _synthesize_impl(self, text: str, lang: str, voice_id: str) -> bytes:
        """
        Synthesize speech using Kokoro.

        Returns WAV bytes at 24kHz mono.
        """
        import soundfile as sf

        # Map to Kokoro internal language code
        kokoro_lang = LANG_CODE_MAP.get(lang.lower(), lang.lower())

        # Get default voice if not specified or "default"
        if voice_id == "default" or not voice_id:
            voice_id = DEFAULT_VOICES.get(kokoro_lang, "af_heart")

        # Get pipeline
        pipe = self._get_pipeline(kokoro_lang)

        # Generate audio - pipe returns a generator
        audio_chunks = []
        for result in pipe(text, voice=voice_id):
            audio_chunks.append(result.output.audio.numpy())

        # Concatenate all chunks
        audio = np.concatenate(audio_chunks)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)

        return buffer.read()

    def get_supported_languages(self) -> list[str]:
        """Return supported language codes."""
        return ["en", "ja", "es", "fr", "hi", "it", "pt", "zh"]

    def get_voices(self, lang: str = None) -> list[str]:
        """Return available voices, optionally filtered by language."""
        if lang is None:
            # Return all voices
            all_voices = []
            for voices in AVAILABLE_VOICES.values():
                all_voices.extend(voices)
            return all_voices

        # Map to Kokoro internal code
        kokoro_lang = LANG_CODE_MAP.get(lang.lower(), lang.lower())
        return AVAILABLE_VOICES.get(kokoro_lang, [])


# CLI support
def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Kokoro TTS Provider")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file")
    parser.add_argument(
        "-l", "--language", default="ja", help="Language: en, ja, es, fr, hi, it, pt, zh"
    )
    parser.add_argument("-v", "--voice", default=None, help="Voice name")
    parser.add_argument("--play", action="store_true", help="Play after synthesis")
    parser.add_argument("--list-voices", action="store_true", help="List voices")

    args = parser.parse_args()

    provider = KokoroProvider()

    if args.list_voices:
        print(f"Available voices for {args.language}:")
        for v in provider.get_voices(args.language):
            print(f"  - {v}")
        return

    if not args.text:
        text = sys.stdin.read().strip()
    else:
        text = args.text

    if not text:
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    voice = args.voice if args.voice else "default"
    success = provider.synthesize_to_file(text, args.output, args.language, voice)

    if success and args.play:
        import subprocess

        subprocess.run(["afplay", args.output])

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
