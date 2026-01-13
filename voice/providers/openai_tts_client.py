#!/usr/bin/env python3
"""
OpenAI TTS Provider

High-quality cloud TTS with excellent multi-language support.

Usage:
    python providers/openai_tts_client.py "Hello world" -o output.wav
    python providers/openai_tts_client.py "こんにちは" -o output.wav -v nova

Voices: alloy, echo, fable, onyx, nova, shimmer
Models: tts-1 (fast), tts-1-hd (high quality)

Requirements:
    pip install openai
    export OPENAI_API_KEY=your-key
"""

import argparse
import os
import sys
import time
from pathlib import Path


class OpenAITTSProvider:
    """OpenAI TTS provider with async support."""

    name = "openai"

    def __init__(self, api_key: str = None, model: str = "tts-1"):
        """
        Initialize OpenAI TTS provider.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use - "tts-1" (fast) or "tts-1-hd" (high quality)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter required")

        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def synthesize(self, text: str, lang: str = "en", voice_id: str = "nova") -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            lang: Language code (auto-detected by OpenAI, but used for voice selection)
            voice_id: Voice name - alloy, echo, fable, onyx, nova, shimmer

        Returns:
            Audio bytes in MP3 format
        """
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice_id,
            input=text,
            response_format="mp3"
        )
        return response.content

    async def synthesize_async(self, text: str, lang: str = "en", voice_id: str = "nova") -> bytes:
        """Async version of synthesize."""
        # OpenAI's Python SDK doesn't have native async, so wrap sync
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.synthesize(text, lang, voice_id)
        )

    def get_supported_languages(self) -> list[str]:
        """Return supported languages (OpenAI supports all major languages)."""
        return ["en", "ja", "zh", "ko", "es", "fr", "de", "it", "pt", "ru", "ar", "hi"]

    def get_voices(self, lang: str = None) -> list[str]:
        """Return available voices."""
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


def synthesize_to_wav(text: str, output_path: str, voice: str = "nova", model: str = "tts-1") -> bool:
    """
    Synthesize text to WAV file (convenience function).

    Args:
        text: Text to synthesize
        output_path: Output WAV file path
        voice: Voice name
        model: Model name (tts-1 or tts-1-hd)

    Returns:
        True if successful
    """
    try:
        provider = OpenAITTSProvider(model=model)

        print(f"Synthesizing ({model}/{voice}): {text[:50]}...", file=sys.stderr)
        start = time.time()

        # Get MP3 audio
        mp3_data = provider.synthesize(text, voice_id=voice)

        # Convert MP3 to WAV using pydub or ffmpeg
        output = Path(output_path)

        # Try pydub first
        try:
            from pydub import AudioSegment
            import io

            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            audio.export(output_path, format="wav")

        except ImportError:
            # Fallback: save as MP3 and use ffmpeg
            mp3_path = output.with_suffix(".mp3")
            mp3_path.write_bytes(mp3_data)

            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", str(mp3_path),
                "-ar", "24000", "-ac", "1",
                str(output_path)
            ], check=True, capture_output=True)

            mp3_path.unlink()  # Clean up temp MP3

        latency = time.time() - start
        print(f"Synthesized in {latency:.2f}s", file=sys.stderr)

        return True

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="OpenAI TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")
    parser.add_argument("-v", "--voice", default="nova",
                       choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                       help="Voice (default: nova)")
    parser.add_argument("-m", "--model", default="tts-1",
                       choices=["tts-1", "tts-1-hd"],
                       help="Model: tts-1 (fast) or tts-1-hd (high quality)")
    parser.add_argument("--play", action="store_true", help="Play audio after synthesis")

    args = parser.parse_args()

    success = synthesize_to_wav(args.text, args.output, args.voice, args.model)

    if success and args.play:
        import subprocess
        subprocess.run(["afplay", args.output])

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
