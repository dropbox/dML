#!/usr/bin/env python3
"""
TTS Worker - Phase 2 (Google TTS)
Converts Japanese text to speech using Google Text-to-Speech API.

Expected performance: 300-500ms (2x faster than macOS say, fallback from Edge-TTS)

Interface:
  - Input: Japanese text on stdin (one line per request)
  - Output: Path to generated audio file on stdout
  - Errors: stderr

Usage:
  echo "こんにちは" | python3 tts_worker_gtts.py
"""
import sys
import time
import os
import tempfile
from gtts import gTTS

class GTTSWorker:
    def __init__(self):
        print("[gTTS] Initializing Google Text-to-Speech...", file=sys.stderr)

        # Language code
        self.lang = 'ja'  # Japanese

        # Speaking speed: True = slower (more natural), False = faster
        self.slow = False

        # Temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp(prefix="gtts_")
        print(f"[gTTS] Using temp directory: {self.temp_dir}", file=sys.stderr)
        print(f"[gTTS] Language: {self.lang}, Slow: {self.slow}", file=sys.stderr)

        # Test synthesis to verify connectivity
        print("[gTTS] Testing connectivity...", file=sys.stderr)
        try:
            test_file = os.path.join(self.temp_dir, "warmup.mp3")
            tts = gTTS("テスト", lang=self.lang, slow=self.slow)
            tts.save(test_file)
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
            print("[gTTS] Connectivity test successful", file=sys.stderr)
        except Exception as e:
            print(f"[gTTS] WARNING: Connectivity test failed: {e}", file=sys.stderr)
            print("[gTTS] Will attempt synthesis on first real request", file=sys.stderr)

        print("[gTTS] Ready to synthesize", file=sys.stderr)
        sys.stderr.flush()

    def synthesize(self, text: str) -> str:
        """
        Synthesize text to speech using Google TTS.
        Returns path to generated audio file.
        """
        if not text.strip():
            return ""

        start = time.perf_counter()

        # Generate unique filename
        audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time() * 1000000)}.mp3")

        try:
            # Create TTS object and save to file
            tts = gTTS(text, lang=self.lang, slow=self.slow)
            tts.save(audio_file)

            elapsed_ms = (time.perf_counter() - start) * 1000
            file_size_kb = os.path.getsize(audio_file) / 1024

            # UTF-8 safe truncation for logging
            display_text = ''.join(list(text)[:30])
            if len(text) > 30:
                display_text += "..."

            print(f"[gTTS] {elapsed_ms:.1f}ms | {file_size_kb:.1f}KB | {display_text}", file=sys.stderr)
            sys.stderr.flush()

            return audio_file

        except Exception as e:
            print(f"[gTTS] ERROR during synthesis: {e}", file=sys.stderr)
            sys.stderr.flush()
            raise

    def run(self):
        """Main loop: read from stdin, synthesize, write file path to stdout"""
        print("[gTTS] Listening for input on stdin...", file=sys.stderr)
        print("[gTTS] NOTE: Audio files will remain until worker exits", file=sys.stderr)
        print("[gTTS] NOTE: Requires internet connection", file=sys.stderr)
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
                print(f"[gTTS] ERROR: {e}", file=sys.stderr)
                sys.stderr.flush()
                # Output empty line on error
                print()
                sys.stdout.flush()

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[gTTS] Cleaned up temp directory", file=sys.stderr)

def main():
    worker = GTTSWorker()
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n[gTTS] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[gTTS] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        worker.cleanup()

if __name__ == "__main__":
    main()
