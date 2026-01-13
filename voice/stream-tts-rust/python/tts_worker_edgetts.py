#!/usr/bin/env python3
"""
TTS Worker - Phase 2 (Edge-TTS)
Converts Japanese text to speech using Microsoft Edge TTS API.

This is 3-6x faster than macOS 'say' command (577ms -> 100-200ms).

Interface:
  - Input: Japanese text on stdin (one line per request)
  - Output: Path to generated audio file on stdout
  - Errors: stderr

Usage:
  echo "こんにちは" | python3 tts_worker_edgetts.py
"""
import sys
import time
import os
import tempfile
import asyncio
import edge_tts

class EdgeTTSWorker:
    def __init__(self):
        print("[EdgeTTS] Initializing Microsoft Edge TTS...", file=sys.stderr)

        # Japanese voice options (high quality):
        # - ja-JP-NanamiNeural (female, natural, default)
        # - ja-JP-KeitaNeural (male, clear)
        # - ja-JP-AoiNeural (female, child-like)
        # - ja-JP-DaichiNeural (male, deeper)
        self.voice = "ja-JP-NanamiNeural"

        # Speaking rate: -50% to +100% (0% = normal)
        # Faster speech = lower latency
        self.rate = "+0%"

        # Volume: -50% to +50% (0% = normal)
        self.volume = "+0%"

        # Pitch: -50Hz to +50Hz (default = +0Hz)
        self.pitch = "+0Hz"

        # Temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp(prefix="edgetts_")
        print(f"[EdgeTTS] Using temp directory: {self.temp_dir}", file=sys.stderr)
        print(f"[EdgeTTS] Using voice: {self.voice}", file=sys.stderr)
        print(f"[EdgeTTS] Rate: {self.rate}, Volume: {self.volume}", file=sys.stderr)

        # Test synthesis to warm up
        print("[EdgeTTS] Warming up with test synthesis...", file=sys.stderr)
        try:
            asyncio.run(self._test_synthesis())
            print("[EdgeTTS] Warm-up successful", file=sys.stderr)
        except Exception as e:
            print(f"[EdgeTTS] WARNING: Warm-up failed: {e}", file=sys.stderr)
            print("[EdgeTTS] Will attempt synthesis on first real request", file=sys.stderr)

        print("[EdgeTTS] Ready to synthesize", file=sys.stderr)
        sys.stderr.flush()

    async def _test_synthesis(self):
        """Warm up the TTS engine with a test synthesis"""
        test_file = os.path.join(self.temp_dir, "warmup.mp3")
        communicate = edge_tts.Communicate(
            "テスト",
            self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch
        )
        await communicate.save(test_file)
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

    async def _synthesize_async(self, text: str, output_file: str) -> float:
        """
        Async synthesis using Edge TTS.
        Returns elapsed time in milliseconds.
        """
        start = time.perf_counter()

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch
        )

        await communicate.save(output_file)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms

    def synthesize(self, text: str) -> str:
        """
        Synthesize text to speech using Edge TTS.
        Returns path to generated audio file.
        """
        if not text.strip():
            return ""

        # Generate unique filename
        audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time() * 1000000)}.mp3")

        try:
            # Run async synthesis
            elapsed_ms = asyncio.run(self._synthesize_async(text, audio_file))

            file_size_kb = os.path.getsize(audio_file) / 1024

            # Truncate text for logging (UTF-8 safe)
            display_text = ''.join(list(text)[:30])
            if len(text) > 30:
                display_text += "..."

            print(f"[EdgeTTS] {elapsed_ms:.1f}ms | {file_size_kb:.1f}KB | {display_text}", file=sys.stderr)
            sys.stderr.flush()

            return audio_file

        except Exception as e:
            print(f"[EdgeTTS] ERROR during synthesis: {e}", file=sys.stderr)
            sys.stderr.flush()
            raise

    def run(self):
        """Main loop: read from stdin, synthesize, write file path to stdout"""
        print("[EdgeTTS] Listening for input on stdin...", file=sys.stderr)
        print("[EdgeTTS] NOTE: Audio files will remain until worker exits", file=sys.stderr)
        sys.stderr.flush()

        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue

            try:
                audio_file = self.synthesize(text)
                print(audio_file)
                sys.stdout.flush()
            except Exception as e:
                print(f"[EdgeTTS] ERROR: {e}", file=sys.stderr)
                sys.stderr.flush()
                # Output empty line on error
                print()
                sys.stdout.flush()

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[EdgeTTS] Cleaned up temp directory", file=sys.stderr)

def main():
    worker = EdgeTTSWorker()
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n[EdgeTTS] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[EdgeTTS] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        worker.cleanup()

if __name__ == "__main__":
    main()
