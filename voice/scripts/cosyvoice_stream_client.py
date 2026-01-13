#!/usr/bin/env python3
"""
CosyVoice2 Streaming Client

Reads text from stdin and streams synthesized Sichuanese audio to the speaker.

Usage:
    # Start server first:
    source cosyvoice_251_venv/bin/activate
    python scripts/cosyvoice_server.py --socket /tmp/cosyvoice.sock &

    # Then stream text:
    echo "你好世界" | python scripts/cosyvoice_stream_client.py
    # Or interactive:
    python scripts/cosyvoice_stream_client.py
    > 你好，今天天气好安逸哦
    > [audio plays]

Copyright 2025 Andrew Yates. All rights reserved.
"""

import socket
import struct
import json
import sys
import wave
import tempfile
import os
import subprocess

SOCKET_PATH = "/tmp/cosyvoice.sock"
SAMPLE_RATE = 24000

def synthesize_and_play(text: str, speed: float = 1.3) -> bool:
    """Synthesize text via CosyVoice2 server and play audio."""
    try:
        # Connect to server
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)

        # Send request
        request = {"text": text.strip(), "speed": speed}
        sock.sendall(json.dumps(request).encode() + b'\n')

        # Read length prefix
        length_data = sock.recv(4)
        if len(length_data) < 4:
            print("[ERROR] Server disconnected", file=sys.stderr)
            return False

        length = struct.unpack('<I', length_data)[0]
        if length == 0:
            print("[ERROR] Server returned empty audio", file=sys.stderr)
            return False

        # Read PCM data
        pcm_data = b''
        while len(pcm_data) < length:
            chunk = sock.recv(min(4096, length - len(pcm_data)))
            if not chunk:
                break
            pcm_data += chunk

        sock.close()

        # Write to temp WAV file and play
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
            with wave.open(f, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm_data)

        # Play audio
        subprocess.run(['afplay', wav_path], check=True)
        os.unlink(wav_path)

        duration = len(pcm_data) / 2 / SAMPLE_RATE
        print(f"[OK] Played {duration:.2f}s audio", file=sys.stderr)
        return True

    except FileNotFoundError:
        print(f"[ERROR] Server not running. Start with:", file=sys.stderr)
        print(f"  source cosyvoice_251_venv/bin/activate", file=sys.stderr)
        print(f"  python scripts/cosyvoice_server.py --socket /tmp/cosyvoice.sock", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return False


def main():
    """Main entry point - read lines from stdin and synthesize."""
    if len(sys.argv) > 1:
        # Text provided as argument
        text = ' '.join(sys.argv[1:])
        synthesize_and_play(text)
    elif sys.stdin.isatty():
        # Interactive mode
        print("CosyVoice2 Streaming TTS (Sichuanese)")
        print("Enter text to synthesize (Ctrl+D to exit):")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    synthesize_and_play(text)
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
    else:
        # Piped input
        for line in sys.stdin:
            if line.strip():
                synthesize_and_play(line)


if __name__ == "__main__":
    main()
