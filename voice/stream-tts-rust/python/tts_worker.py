#!/usr/bin/env python3
"""
TTS Worker - Phase 1
Converts Japanese text to speech using macOS built-in TTS (say command).

Interface:
  - Input: Japanese text on stdin (one line per request)
  - Output: Path to generated audio file on stdout
  - Errors: stderr

Usage:
  echo "こんにちは" | python3 tts_worker.py
"""
import sys
import time
import os
import tempfile
import subprocess
import yaml

class TTSWorker:
    def __init__(self):
        print("[TTS] Initializing macOS TTS...", file=sys.stderr)

        # Load configuration from config.yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                tts_config = config.get('tts', {})
                self.voice = tts_config.get('voice', 'Kyoko')
                self.rate = tts_config.get('rate', 200)
                print(f"[TTS] Loaded config from {config_path}", file=sys.stderr)
        except Exception as e:
            print(f"[TTS] Warning: Could not load config ({e}), using defaults", file=sys.stderr)
            self.voice = "Kyoko"  # Default Japanese female voice
            self.rate = 200  # Normal speed

        # Temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp(prefix="tts_")
        print(f"[TTS] Using temp directory: {self.temp_dir}", file=sys.stderr)
        print(f"[TTS] Using voice: {self.voice}", file=sys.stderr)

        # Test voice availability
        result = subprocess.run(['say', '-v', self.voice, 'テスト'],
                               capture_output=True)
        if result.returncode != 0:
            print(f"[TTS] WARNING: Voice {self.voice} may not be available", file=sys.stderr)

        print("[TTS] Ready to synthesize", file=sys.stderr)
        sys.stderr.flush()

    def synthesize(self, text: str) -> str:
        """
        Synthesize text to speech using macOS say command.
        Returns path to generated audio file.
        """
        if not text.strip():
            return ""

        start = time.perf_counter()

        # Generate unique filename
        audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time() * 1000000)}.aiff")

        # Use macOS say command to generate audio file
        # Simple AIFF format (default)
        cmd = [
            'say',
            '-v', self.voice,
            '-r', str(self.rate),
            '-o', audio_file,
            text
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"say command failed: {result.stderr}")

        elapsed_ms = (time.perf_counter() - start) * 1000
        file_size_kb = os.path.getsize(audio_file) / 1024

        print(f"[TTS] {elapsed_ms:.1f}ms | {file_size_kb:.1f}KB | {text[:50]}...", file=sys.stderr)
        sys.stderr.flush()

        return audio_file

    def run(self):
        """Main loop: read from stdin, synthesize, write file path to stdout"""
        print("[TTS] Listening for input on stdin...", file=sys.stderr)
        print("[TTS] NOTE: Audio files will remain until worker exits", file=sys.stderr)
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
                print(f"[TTS] ERROR: {e}", file=sys.stderr)
                sys.stderr.flush()
                # Output empty line on error
                print()
                sys.stdout.flush()

        # Cleanup happens in finally block when worker exits

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[TTS] Cleaned up temp directory", file=sys.stderr)

def main():
    worker = TTSWorker()
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n[TTS] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[TTS] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        worker.cleanup()

if __name__ == "__main__":
    main()
