#!/usr/bin/env python3
"""
XTTS v2 TTS Daemon Client - Sends requests to the XTTS daemon.

Usage:
    # Ensure daemon is running first:
    python scripts/xtts_daemon.py &

    # Synthesize text
    python scripts/xtts_client.py "Hello world" -o output.wav -l en
    python scripts/xtts_client.py "こんにちは世界" -o output.wav -l ja
    python scripts/xtts_client.py "你好世界" -o output.wav -l zh-cn

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import json
import socket
import struct
import sys
import time


def send_request(socket_path: str, text: str, language: str, speaker_wav: str = None) -> bytes:
    """Send synthesis request to daemon and get WAV bytes."""
    request = {
        "text": text,
        "language": language,
    }
    if speaker_wav:
        request["speaker_wav"] = speaker_wav

    # Connect to daemon
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    try:
        # Send request
        request_bytes = (json.dumps(request) + "\n").encode('utf-8')
        sock.sendall(request_bytes)

        # Read response length (4 bytes)
        length_data = sock.recv(4)
        if len(length_data) < 4:
            raise RuntimeError(f"Unexpected response: {length_data}")

        # Check if it's an error response (starts with '{')
        if length_data[0] == ord('{'):
            # It's a JSON error
            rest = sock.recv(4096)
            error_json = length_data + rest
            error = json.loads(error_json.decode('utf-8'))
            raise RuntimeError(f"Daemon error: {error.get('error', 'Unknown error')}")

        length = struct.unpack('<I', length_data)[0]

        # Read WAV data
        wav_data = b''
        while len(wav_data) < length:
            chunk = sock.recv(min(65536, length - len(wav_data)))
            if not chunk:
                break
            wav_data += chunk

        return wav_data
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="XTTS v2 TTS Daemon Client")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")
    parser.add_argument("-l", "--language", default="en",
                       help="Language code (en, ja, zh-cn, etc.)")
    parser.add_argument("-s", "--speaker", default=None,
                       help="Reference speaker WAV file for voice cloning")
    parser.add_argument("--socket", default="/tmp/xtts_tts.sock",
                       help="Daemon socket path")
    args = parser.parse_args()

    start = time.time()
    try:
        wav_data = send_request(args.socket, args.text, args.language, args.speaker)

        # Write to file
        with open(args.output, 'wb') as f:
            f.write(wav_data)

        latency = (time.time() - start) * 1000
        print(f"Synthesized in {latency:.0f}ms ({len(wav_data)} bytes)", file=sys.stderr)
        print(args.output)
        sys.exit(0)
    except ConnectionRefusedError:
        print(f"ERROR: Could not connect to daemon at {args.socket}", file=sys.stderr)
        print("Start the daemon with: python scripts/xtts_daemon.py &", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
