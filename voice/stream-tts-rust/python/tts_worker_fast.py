#!/usr/bin/env python3
"""
Fast TTS Worker using macOS say command
Best quality, configurable speed, local execution.

Copyright 2025 Andrew Yates. All rights reserved.
"""
import sys
import subprocess
import time
import os
import tempfile
import yaml

class MacOSTTSWorker:
    def __init__(self):
        print("[TTS] Initializing macOS say TTS...", file=sys.stderr)

        # Load config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self.voice = config['tts']['voice']
                self.rate = config['tts']['rate']
        else:
            self.voice = 'Kyoko'  # Default Japanese female voice
            self.rate = 280       # Faster than normal (175)

        # Create temp directory for audio files
        self.temp_dir = tempfile.mkdtemp(prefix="tts_")

        print(f"[TTS] Voice: {self.voice}, Rate: {self.rate} WPM", file=sys.stderr)
        print(f"[TTS] Temp dir: {self.temp_dir}", file=sys.stderr)

        # Test voice
        result = subprocess.run(
            ['say', '-v', self.voice, '--', 'テスト'],
            capture_output=True
        )

        if result.returncode == 0:
            print("[TTS] Voice test successful", file=sys.stderr)
        else:
            print(f"[TTS] WARNING: Voice test failed, using default voice", file=sys.stderr)
            self.voice = None  # Use system default

        print("[TTS] Ready", file=sys.stderr)
        sys.stderr.flush()

    def synthesize(self, text: str) -> str:
        """
        Synthesize text to audio file using macOS say.
        Returns path to generated audio file.
        """
        if not text.strip():
            return ""

        start = time.perf_counter()

        # Generate unique filename
        timestamp = int(time.time() * 1000000)
        audio_file = os.path.join(self.temp_dir, f"tts_{timestamp}.aiff")

        # Build say command to generate file
        cmd = ['say', '-o', audio_file]
        if self.voice:
            cmd.extend(['-v', self.voice])
        cmd.extend(['-r', str(self.rate)])
        cmd.append(text)

        # Run say command
        result = subprocess.run(cmd, capture_output=True, text=True)

        elapsed = (time.perf_counter() - start) * 1000

        if result.returncode == 0 and os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file) / 1024  # KB
            print(f"[TTS] {elapsed:.1f}ms | {file_size:.1f}KB | {text[:50]}...", file=sys.stderr)
            sys.stderr.flush()
            return audio_file
        else:
            print(f"[TTS] ERROR: {result.stderr}", file=sys.stderr)
            sys.stderr.flush()
            return ""

    def run(self):
        """Main loop: read Japanese text from stdin, synthesize to file"""
        print("[TTS] Listening for input on stdin...", file=sys.stderr)
        sys.stderr.flush()

        for line in sys.stdin:
            text = line.strip()
            if text:
                try:
                    audio_path = self.synthesize(text)
                    # Return path for Rust to play
                    print(audio_path, flush=True)
                except Exception as e:
                    print(f"[TTS] ERROR: {e}", file=sys.stderr)
                    print("", flush=True)  # Empty line on error

    def cleanup(self):
        """Clean up temp directory"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

if __name__ == "__main__":
    worker = MacOSTTSWorker()
    try:
        worker.run()
    finally:
        worker.cleanup()
