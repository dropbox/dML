#!/usr/bin/env python3
"""
XTTS v2 TTS Daemon - Persistent server for fast TTS synthesis.

Keeps the XTTS v2 model loaded in memory and accepts requests via Unix socket.
Reduces latency from ~42s (subprocess with model load) to ~3-5s (warm synthesis).

Usage:
    # Start daemon (blocks, logs to stderr)
    python scripts/xtts_daemon.py --socket /tmp/xtts_tts.sock

    # Start daemon with preloaded language references
    python scripts/xtts_daemon.py --preload en ja zh-cn

    # Send request (from another process)
    echo '{"text": "Hello world", "language": "en"}' | nc -U /tmp/xtts_tts.sock > output.wav

Protocol:
    Request (JSON line):
        {"text": "...", "language": "en", "speaker_wav": "optional/path.wav"}

    Response:
        - On success: 4-byte length prefix + raw WAV bytes
        - On error: JSON with "error" key

Supported languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Suppress warnings before imports
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Apply PyTorch patch before importing TTS
import torch
_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load

# Language-specific reference generation configs
LANG_REFS = {
    "ja": ("Kyoko", "これはテストの音声です"),
    "zh-cn": ("Tingting", "这是语音测试"),
    "ko": ("Yuna", "이것은 음성 테스트입니다"),
    "en": ("Samantha", "This is a test of the voice system"),
    "es": ("Monica", "Esta es una prueba del sistema de voz"),
    "fr": ("Thomas", "Ceci est un test du systeme vocal"),
    "de": ("Anna", "Dies ist ein Test des Sprachsystems"),
    "it": ("Alice", "Questo e un test del sistema vocale"),
    "pt": ("Joana", "Este e um teste do sistema de voz"),
}


class XTTSDaemon:
    """Persistent XTTS v2 TTS server."""

    def __init__(self, socket_path: str, preload_languages: list = None):
        self.socket_path = socket_path
        self.tts = None
        self.reference_wavs = {}
        self.running = False

        # Import and load model
        print(f"[xtts-daemon] Importing TTS library...", file=sys.stderr)
        from TTS.api import TTS

        print(f"[xtts-daemon] Loading XTTS v2 model (this takes ~24s)...", file=sys.stderr)
        start = time.time()
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        print(f"[xtts-daemon] Model loaded in {time.time()-start:.1f}s", file=sys.stderr)

        # Preload language references
        if preload_languages:
            for lang in preload_languages:
                self._get_reference_wav(lang)

    def _get_reference_wav(self, language: str) -> str:
        """Get or create reference speaker wav for the target language."""
        if language in self.reference_wavs:
            return self.reference_wavs[language]

        ref_path = f"/tmp/xtts_ref_{language.replace('-', '_')}.wav"

        if not os.path.exists(ref_path):
            voice, text = LANG_REFS.get(language, LANG_REFS["en"])
            try:
                subprocess.run(
                    ['say', '-o', ref_path, '--data-format=LEI16@24000', '-v', voice, text],
                    capture_output=True, check=True
                )
                print(f"[xtts-daemon] Created {language} speaker reference: {ref_path}", file=sys.stderr)
            except subprocess.CalledProcessError:
                # Fallback to espeak-ng
                print(f"[xtts-daemon] macOS voice {voice} unavailable, using espeak fallback", file=sys.stderr)
                subprocess.run(
                    ['espeak-ng', '-w', ref_path, 'This is a test of the voice system'],
                    capture_output=True, check=True
                )

        self.reference_wavs[language] = ref_path
        return ref_path

    def synthesize(self, text: str, language: str = "en", speaker_wav: str = None) -> bytes:
        """Synthesize text to WAV bytes."""
        import io
        import wave
        import numpy as np

        # Get reference wav
        ref_wav = speaker_wav if speaker_wav and os.path.exists(speaker_wav) else self._get_reference_wav(language)

        # Generate audio
        start = time.time()

        # Use a temp file since XTTS API writes to file
        temp_path = f"/tmp/xtts_output_{os.getpid()}.wav"
        self.tts.tts_to_file(
            text=text,
            file_path=temp_path,
            language=language,
            speaker_wav=ref_wav
        )

        latency = time.time() - start

        # Read audio, normalize if clipping, then return as WAV bytes
        with wave.open(temp_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)
            duration = n_frames / sample_rate

        os.remove(temp_path)

        # Convert to numpy for normalization check
        if sample_width == 2:
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0

        # Normalize if clipping detected (peak > 0.95)
        peak = np.max(np.abs(samples))
        if peak > 0.95:
            target_peak = 0.85  # Leave headroom
            samples = samples * (target_peak / peak)
            print(f"[xtts-daemon] Normalized audio (peak {peak:.2f} -> {target_peak})", file=sys.stderr)

        # Convert back to int16 WAV bytes
        samples_int = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)

        # Write normalized WAV to bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_out:
            wav_out.setnchannels(n_channels)
            wav_out.setsampwidth(2)  # Always output 16-bit
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(samples_int.tobytes())
        wav_data = wav_buffer.getvalue()

        print(f"[xtts-daemon] Synthesized '{text[:30]}...' ({language}) "
              f"in {latency*1000:.0f}ms ({duration:.1f}s audio)", file=sys.stderr)

        return wav_data

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
            speaker_wav = request.get('speaker_wav')

            if not text:
                error_response = json.dumps({"error": "Missing 'text' field"})
                conn.sendall(error_response.encode('utf-8'))
                return

            # Synthesize
            wav_data = self.synthesize(text, language, speaker_wav)

            # Send response: 4-byte length prefix + WAV data
            conn.sendall(struct.pack('<I', len(wav_data)))
            conn.sendall(wav_data)

        except json.JSONDecodeError as e:
            error_response = json.dumps({"error": f"Invalid JSON: {e}"})
            conn.sendall(error_response.encode('utf-8'))
        except Exception as e:
            print(f"[xtts-daemon] Error handling client: {e}", file=sys.stderr)
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
        print(f"[xtts-daemon] Listening on {self.socket_path}", file=sys.stderr)
        print(f"[xtts-daemon] Ready for synthesis (~3-5s per request)", file=sys.stderr)

        # Handle shutdown signals
        def shutdown(signum, frame):
            print(f"\n[xtts-daemon] Received signal {signum}, shutting down...", file=sys.stderr)
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
            print("[xtts-daemon] Shutdown complete", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="XTTS v2 TTS Daemon")
    parser.add_argument("--socket", default="/tmp/xtts_tts.sock",
                       help="Unix socket path (default: /tmp/xtts_tts.sock)")
    parser.add_argument("--preload", nargs="*", default=["en", "ja"],
                       help="Languages to preload references for (default: en ja)")
    args = parser.parse_args()

    daemon = XTTSDaemon(args.socket, args.preload)
    daemon.run()


if __name__ == "__main__":
    main()
