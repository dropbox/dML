#!/usr/bin/env python3
"""
Kokoro TTS Client - Send requests to the Kokoro daemon.

Usage:
    python scripts/kokoro_client.py "Hello world" -o output.wav -l en
    python scripts/kokoro_client.py "こんにちは" -o output.wav -l ja

This is a drop-in replacement for kokoro_tts.py but uses the daemon for speed.
Latency: ~600ms vs ~7s with subprocess.

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import json
import os
import socket
import struct
import sys
import time

# Add project root to path for provider import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from providers import play_audio


def synthesize_via_daemon(
    text: str,
    output_path: str,
    language: str = "en",
    voice: str = None,
    socket_path: str = "/tmp/kokoro_tts.sock"
) -> bool:
    """Send synthesis request to daemon."""
    try:
        start = time.time()

        # Connect to daemon
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)

        # Send request
        request = {"text": text, "language": language}
        if voice:
            request["voice"] = voice

        sock.sendall(json.dumps(request).encode('utf-8') + b'\n')

        # Read response length (4 bytes)
        length_data = sock.recv(4)
        if len(length_data) < 4:
            # Check if it's an error response
            error_data = length_data + sock.recv(4096)
            try:
                error = json.loads(error_data.decode('utf-8'))
                print(f"ERROR: {error.get('error', 'Unknown error')}", file=sys.stderr)
            except:
                print(f"ERROR: Unexpected response: {error_data}", file=sys.stderr)
            return False

        wav_length = struct.unpack('<I', length_data)[0]

        # Read WAV data
        wav_data = b''
        while len(wav_data) < wav_length:
            chunk = sock.recv(min(65536, wav_length - len(wav_data)))
            if not chunk:
                break
            wav_data += chunk

        sock.close()

        if len(wav_data) != wav_length:
            print(f"ERROR: Incomplete response ({len(wav_data)}/{wav_length} bytes)", file=sys.stderr)
            return False

        # Write WAV file
        with open(output_path, 'wb') as f:
            f.write(wav_data)

        latency = time.time() - start
        print(f"Synthesized in {latency*1000:.0f}ms ({len(wav_data)} bytes)", file=sys.stderr)

        return True

    except FileNotFoundError:
        print(f"ERROR: Daemon not running (socket not found: {socket_path})", file=sys.stderr)
        print("Start daemon with: python scripts/kokoro_daemon.py", file=sys.stderr)
        return False
    except ConnectionRefusedError:
        print(f"ERROR: Daemon not accepting connections", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS Client (uses daemon)")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")
    parser.add_argument("-l", "--language", default="en",
                       help="Language: en, ja, es, fr, hi, it, pt, zh (default: en)")
    parser.add_argument("-v", "--voice", default=None, help="Voice name (optional)")
    parser.add_argument("--socket", default="/tmp/kokoro_tts.sock",
                       help="Daemon socket path")
    parser.add_argument("--play", action="store_true", help="Play audio after synthesis")

    args = parser.parse_args()

    success = synthesize_via_daemon(
        args.text,
        args.output,
        args.language,
        args.voice,
        args.socket
    )

    if success and args.play:
        with open(args.output, 'rb') as f:
            play_audio(f.read())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
