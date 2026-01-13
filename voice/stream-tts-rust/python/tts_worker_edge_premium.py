#!/usr/bin/env python3
"""
Premium EdgeTTS Worker - Best free Japanese TTS
Uses Microsoft Edge Neural voices (cloud-based but free)

Copyright 2025 Andrew Yates. All rights reserved.
"""
import sys
import asyncio
import edge_tts
import tempfile
import time
import os

class EdgeTTSWorker:
    def __init__(self):
        print("[EdgeTTS] Initializing Premium Edge TTS...", file=sys.stderr)

        # Use BEST Japanese voice
        self.voice = "ja-JP-NanamiNeural"  # Most natural female voice
        self.rate = "+10%"  # Slightly faster than normal
        self.volume = "+50%"  # LOUDER for testing

        self.temp_dir = tempfile.mkdtemp(prefix="edge_tts_")

        print(f"[EdgeTTS] Voice: {self.voice}", file=sys.stderr)
        print(f"[EdgeTTS] Rate: {self.rate}, Volume: {self.volume}", file=sys.stderr)
        print(f"[EdgeTTS] Temp dir: {self.temp_dir}", file=sys.stderr)
        print(f"[EdgeTTS] Ready", file=sys.stderr)
        sys.stderr.flush()

    async def synthesize_async(self, text: str) -> str:
        """Synthesize text to audio file"""
        if not text.strip():
            return ""

        start = time.perf_counter()

        # Generate unique filename
        timestamp = int(time.time() * 1000000)
        audio_file = os.path.join(self.temp_dir, f"tts_{timestamp}.mp3")

        # Create TTS
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume
        )

        # Save to file
        await communicate.save(audio_file)

        elapsed = (time.perf_counter() - start) * 1000

        if os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file) / 1024
            print(f"[EdgeTTS] {elapsed:.1f}ms | {file_size:.1f}KB | {text[:50]}...", file=sys.stderr)
            sys.stderr.flush()
            return audio_file
        else:
            print(f"[EdgeTTS] ERROR: File not created", file=sys.stderr)
            sys.stderr.flush()
            return ""

    def synthesize(self, text: str) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self.synthesize_async(text))

    def run(self):
        """Main loop"""
        print("[EdgeTTS] Listening for input on stdin...", file=sys.stderr)
        sys.stderr.flush()

        for line in sys.stdin:
            text = line.strip()
            if text:
                try:
                    audio_path = self.synthesize(text)
                    print(audio_path, flush=True)
                except Exception as e:
                    print(f"[EdgeTTS] ERROR: {e}", file=sys.stderr)
                    print("", flush=True)

    def cleanup(self):
        """Clean up temp files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

if __name__ == "__main__":
    worker = EdgeTTSWorker()
    try:
        worker.run()
    finally:
        worker.cleanup()
