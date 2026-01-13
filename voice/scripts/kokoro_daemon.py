#!/usr/bin/env python3
"""
Kokoro TTS Daemon - Persistent server for fast TTS synthesis.

Keeps the Kokoro model loaded in memory and accepts requests via Unix socket.
Reduces latency from ~7s (subprocess) to ~600ms (warm synthesis).

Usage:
    # Start daemon (blocks, logs to stderr)
    python scripts/kokoro_daemon.py --socket /tmp/kokoro_tts.sock

    # Start daemon in background
    python scripts/kokoro_daemon.py --socket /tmp/kokoro_tts.sock &

    # Send request (from another process)
    echo '{"text": "Hello world", "language": "en"}' | nc -U /tmp/kokoro_tts.sock > output.wav

Protocol:
    Request (JSON line):
        {"text": "...", "language": "en", "voice": "optional"}

    Response:
        - On success: raw WAV bytes
        - On error: JSON with "error" key

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import json
import os
import signal
import socket
import struct
import sys
import time
import warnings
from pathlib import Path

# Suppress warnings before imports
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Language code mapping (same as kokoro_tts.py)
LANG_CODE_MAP = {
    'en': 'a',   # American English
    'ja': 'j',   # Japanese
    'jp': 'j',
    'es': 's',   # Spanish
    'fr': 'f',   # French
    'hi': 'h',   # Hindi
    'it': 'i',   # Italian
    'pt': 'p',   # Portuguese
    'zh': 'z',   # Mandarin Chinese
}

DEFAULT_VOICES = {
    'a': 'af_heart',      # American English female
    'j': 'jf_alpha',      # Japanese female
    's': 'sf_rosa',       # Spanish female
    'f': 'ff_siwis',      # French female
    'h': 'hf_alpha',      # Hindi female
    'i': 'if_alice',      # Italian female
    'p': 'pf_dora',       # Portuguese female
    'z': 'zf_xiaobei',    # Mandarin Chinese female
}


class KokoroDaemon:
    """Persistent Kokoro TTS server with MPS acceleration."""

    def __init__(self, socket_path: str, preload_languages: list = None):
        self.socket_path = socket_path
        self.pipelines = {}  # lang_code -> KPipeline
        self.running = False

        # Detect best device (MPS > CUDA > CPU)
        import torch
        self.device = 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"[daemon] Using Metal GPU (MPS) for acceleration", file=sys.stderr)
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print(f"[daemon] Using CUDA GPU for acceleration", file=sys.stderr)
        else:
            print(f"[daemon] Using CPU (no GPU available)", file=sys.stderr)

        # Import Kokoro once
        print(f"[daemon] Importing Kokoro...", file=sys.stderr)
        from kokoro import KPipeline
        self.KPipeline = KPipeline

        # Preload languages
        if preload_languages:
            for lang in preload_languages:
                self._get_pipeline(lang)

    def _get_pipeline(self, lang_code: str):
        """Get or create pipeline for language."""
        kokoro_code = LANG_CODE_MAP.get(lang_code.lower(), lang_code.lower())

        if kokoro_code not in self.pipelines:
            print(f"[daemon] Loading Kokoro model for '{kokoro_code}'...", file=sys.stderr)
            start = time.time()
            pipeline = self.KPipeline(
                lang_code=kokoro_code,
                repo_id='hexgrad/Kokoro-82M'
            )

            # Move model to GPU if available
            if self.device != 'cpu':
                if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'to'):
                    pipeline.model.to(self.device)
                    print(f"[daemon] Moved model to {self.device}", file=sys.stderr)

            self.pipelines[kokoro_code] = pipeline
            print(f"[daemon] Model loaded in {time.time()-start:.1f}s", file=sys.stderr)

            # Pre-warm with a short synthesis
            print(f"[daemon] Pre-warming model...", file=sys.stderr)
            voice = DEFAULT_VOICES.get(kokoro_code, 'af_heart')
            for _ in pipeline("test", voice=voice):
                pass
            print(f"[daemon] Model ready", file=sys.stderr)

        return self.pipelines[kokoro_code], kokoro_code

    def synthesize(self, text: str, language: str = "en", voice: str = None) -> bytes:
        """Synthesize text to WAV bytes."""
        import numpy as np
        import soundfile as sf
        import io

        pipe, lang_code = self._get_pipeline(language)

        # Get default voice if not specified
        if voice is None:
            voice = DEFAULT_VOICES.get(lang_code, 'af_heart')

        # Generate audio
        start = time.time()
        audio_chunks = []
        for result in pipe(text, voice=voice):
            audio_chunks.append(result.output.audio.numpy())

        audio = np.concatenate(audio_chunks)
        latency = time.time() - start

        print(f"[daemon] Synthesized '{text[:30]}...' ({lang_code}/{voice}) "
              f"in {latency*1000:.0f}ms ({len(audio)/24000:.1f}s audio)", file=sys.stderr)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, 24000, format='WAV', subtype='PCM_16')
        return buffer.getvalue()

    def handle_client(self, conn: socket.socket):
        """Handle a client connection."""
        try:
            # Read request (JSON line terminated by newline)
            data = b''
            while b'\n' not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk

            if not data:
                return

            # Parse JSON request
            request = json.loads(data.decode('utf-8').strip())
            text = request.get('text', '')
            language = request.get('language', 'en')
            voice = request.get('voice')

            if not text:
                error_response = json.dumps({"error": "Missing 'text' field"})
                conn.sendall(error_response.encode('utf-8'))
                return

            # Synthesize
            wav_data = self.synthesize(text, language, voice)

            # Send response: 4-byte length prefix + WAV data
            conn.sendall(struct.pack('<I', len(wav_data)))
            conn.sendall(wav_data)

        except json.JSONDecodeError as e:
            error_response = json.dumps({"error": f"Invalid JSON: {e}"})
            conn.sendall(error_response.encode('utf-8'))
        except Exception as e:
            print(f"[daemon] Error handling client: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            try:
                error_response = json.dumps({"error": str(e)})
                conn.sendall(error_response.encode('utf-8'))
            except:
                pass
        finally:
            conn.close()

    def run(self):
        """Run the daemon server."""
        # Clean up existing socket
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        # Create Unix socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.socket_path)
        server.listen(5)

        # Set permissions (readable/writable by owner)
        os.chmod(self.socket_path, 0o700)

        self.running = True
        print(f"[daemon] Listening on {self.socket_path}", file=sys.stderr)

        # Handle shutdown signals
        def shutdown(signum, frame):
            print(f"\n[daemon] Received signal {signum}, shutting down...", file=sys.stderr)
            self.running = False
            server.close()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            while self.running:
                try:
                    server.settimeout(1.0)  # Check running flag periodically
                    conn, addr = server.accept()
                    self.handle_client(conn)
                except socket.timeout:
                    continue
                except OSError:
                    # Socket closed during shutdown
                    break
        finally:
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
            print("[daemon] Shutdown complete", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS Daemon")
    parser.add_argument("--socket", default="/tmp/kokoro_tts.sock",
                       help="Unix socket path (default: /tmp/kokoro_tts.sock)")
    parser.add_argument("--preload", nargs="*", default=["en", "ja"],
                       help="Languages to preload (default: en ja)")
    args = parser.parse_args()

    daemon = KokoroDaemon(args.socket, args.preload)
    daemon.run()


if __name__ == "__main__":
    main()
