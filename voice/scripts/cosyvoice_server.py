#!/usr/bin/env python3
"""
CosyVoice2 Socket Server for Sichuanese/Dialect TTS

REQUIRES: cosyvoice_251_venv with PyTorch 2.5.1 (PyTorch 2.9.1 produces frog audio)

Protocol:
    Request: newline-terminated JSON
        {"text": "...", "instruction": "用四川话说这句话", "speed": 1.3}
    Response: 4-byte little-endian length prefix followed by raw int16 PCM audio (24kHz)

Usage:
    source cosyvoice_251_venv/bin/activate
    python scripts/cosyvoice_server.py --socket /tmp/cosyvoice.sock

The server keeps CosyVoice2 loaded with the grandma voice prompt to minimize latency.

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

# Add cosyvoice_repo to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'cosyvoice_repo'))
sys.path.insert(0, str(ROOT_DIR / 'cosyvoice_repo' / 'third_party' / 'Matcha-TTS'))

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class CosyVoice2Server:
    """Persistent CosyVoice2 server using Unix domain sockets with grandma voice."""

    def __init__(self, socket_path: Path, model_dir: str = None, prompt_path: str = None):
        self.socket_path = socket_path
        self.model_dir = model_dir or str(ROOT_DIR / 'models' / 'cosyvoice' / 'CosyVoice2-0.5B')
        self.prompt_path = prompt_path or str(ROOT_DIR / 'tests' / 'golden' / 'hello.wav')
        self.cosyvoice = None
        self.prompt_speech_16k = None
        self.sample_rate = 24000
        self.default_speed = 1.0  # Default speed matches golden reference quality
        self.default_instruction = "用四川话说，像一个四川婆婆在讲故事"

        self._load_model()

    def _load_model(self) -> None:
        """Load CosyVoice2 model and voice prompt."""
        import torch
        import librosa

        print(f"[cosyvoice2] PyTorch version: {torch.__version__}", flush=True)
        if not torch.__version__.startswith("2.5"):
            print("[cosyvoice2] WARNING: PyTorch 2.5.x required for good quality!", flush=True)
            print("[cosyvoice2] Use: source cosyvoice_251_venv/bin/activate", flush=True)

        start = time.time()

        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        print(f"[cosyvoice2] Loading model from {self.model_dir}...", flush=True)
        self.cosyvoice = CosyVoice2(self.model_dir, load_jit=False, load_trt=False)
        self.sample_rate = self.cosyvoice.sample_rate

        # Load voice prompt using CosyVoice's load_wav (returns proper 2D tensor)
        print(f"[cosyvoice2] Loading voice prompt from {self.prompt_path}...", flush=True)
        self.prompt_speech_16k = load_wav(self.prompt_path, 16000)

        print(f"[cosyvoice2] Model loaded in {time.time() - start:.1f}s", flush=True)
        print(f"[cosyvoice2] Voice prompt: {self.prompt_speech_16k.shape[1]/16000:.1f}s", flush=True)
        print(f"[cosyvoice2] Default speed: {self.default_speed}x", flush=True)

        # Store postprocess function
        self.librosa = librosa

    def _postprocess(self, speech):
        """Postprocess audio like webui: trim silence, normalize, add padding."""
        import torch
        speech_np = speech.squeeze().cpu().numpy()
        speech_trimmed, _ = self.librosa.effects.trim(speech_np, top_db=60, frame_length=440, hop_length=220)
        speech_t = torch.from_numpy(speech_trimmed).unsqueeze(0)
        max_val = 0.8
        if speech_t.abs().max() > max_val:
            speech_t = speech_t / speech_t.abs().max() * max_val
        # Add 0.2s padding at end
        padding = torch.zeros(1, int(self.sample_rate * 0.2))
        speech_t = torch.cat([speech_t, padding], dim=1)
        return speech_t

    def synthesize(self, text: str, instruction: str = "", speed: float = None) -> bytes:
        """Synthesize text to raw PCM audio bytes (int16, 24kHz, mono)."""
        import torch
        import numpy as np
        from cosyvoice.utils.common import set_all_random_seed

        if self.cosyvoice is None:
            raise RuntimeError("CosyVoice2 model not loaded")

        instruction = instruction or self.default_instruction
        speed = speed or self.default_speed

        try:
            set_all_random_seed(42)
            for result in self.cosyvoice.inference_instruct2(
                text, instruction, self.prompt_speech_16k, stream=False, speed=speed
            ):
                audio = self._postprocess(result['tts_speech'])
                break

            # Convert to int16 PCM
            audio_np = audio.squeeze().cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            return audio_int16.tobytes()

        except Exception as e:
            print(f"[cosyvoice2] Synthesis error: {e}", flush=True)
            raise

    def synthesize_streaming(self, text: str, instruction: str = "", speed: float = None):
        """Synthesize text with streaming output (yields audio chunks)."""
        import torch
        import numpy as np
        from cosyvoice.utils.common import set_all_random_seed

        if self.cosyvoice is None:
            raise RuntimeError("CosyVoice2 model not loaded")

        instruction = instruction or self.default_instruction
        speed = speed or self.default_speed

        set_all_random_seed(42)
        for result in self.cosyvoice.inference_instruct2(
            text, instruction, self.prompt_speech_16k, stream=True, speed=speed
        ):
            audio = result['tts_speech']
            # Convert to int16 PCM without full postprocessing for streaming
            audio_np = audio.squeeze().cpu().numpy()
            # Simple normalization for streaming
            if np.abs(audio_np).max() > 0.8:
                audio_np = audio_np / np.abs(audio_np).max() * 0.8
            audio_int16 = (audio_np * 32767).astype(np.int16)
            yield audio_int16.tobytes()

    def handle_client(self, conn: socket.socket) -> None:
        """Handle a single client connection."""
        try:
            # Read request until newline
            buffer = b""
            while b"\n" not in buffer:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk

            if not buffer:
                return

            try:
                request = json.loads(buffer.decode("utf-8").strip())
            except json.JSONDecodeError as e:
                print(f"[cosyvoice2] JSON decode error: {e}", flush=True)
                conn.sendall(struct.pack("<I", 0))  # Zero length = error
                return

            text = request.get("text", "")
            instruction = request.get("instruction", "")
            speed = request.get("speed")
            streaming = request.get("stream", False)

            if not text:
                conn.sendall(struct.pack("<I", 0))
                return

            print(f"[cosyvoice2] Synthesizing: text={text[:50]}..., speed={speed or self.default_speed}", flush=True)
            start = time.time()

            if streaming:
                # Streaming mode: send chunks with length prefix
                total_bytes = 0
                for chunk in self.synthesize_streaming(text, instruction, speed):
                    conn.sendall(struct.pack("<I", len(chunk)))
                    conn.sendall(chunk)
                    total_bytes += len(chunk)
                # Send zero-length to signal end
                conn.sendall(struct.pack("<I", 0))
                elapsed = time.time() - start
                duration = total_bytes / 2 / self.sample_rate
                print(f"[cosyvoice2] Streamed {total_bytes} bytes ({duration:.2f}s) in {elapsed:.2f}s", flush=True)
            else:
                # Non-streaming mode
                pcm_data = self.synthesize(text, instruction, speed)
                elapsed = time.time() - start
                duration = len(pcm_data) / 2 / self.sample_rate
                rtf = elapsed / duration if duration > 0 else 0
                print(f"[cosyvoice2] Synthesized {len(pcm_data)} bytes ({duration:.2f}s) in {elapsed:.2f}s (RTF={rtf:.2f}x)", flush=True)

                # Send length-prefixed response
                conn.sendall(struct.pack("<I", len(pcm_data)))
                conn.sendall(pcm_data)

        except Exception as e:
            print(f"[cosyvoice2] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            try:
                conn.sendall(struct.pack("<I", 0))
            except Exception:
                pass
        finally:
            conn.close()

    def serve(self) -> None:
        """Start the Unix socket server loop."""
        if self.socket_path.exists():
            self.socket_path.unlink()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(str(self.socket_path))
        server.listen(5)
        os.chmod(str(self.socket_path), 0o700)

        print(f"[cosyvoice2] Listening on {self.socket_path}", flush=True)
        print(f"[cosyvoice2] CosyVoice2 ready (Sichuanese dialect, grandma voice)", flush=True)

        def shutdown(signum, _frame):
            print(f"[cosyvoice2] Received signal {signum}, shutting down", flush=True)
            server.close()
            if self.socket_path.exists():
                self.socket_path.unlink()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            while True:
                conn, _ = server.accept()
                self.handle_client(conn)
        finally:
            server.close()
            if self.socket_path.exists():
                self.socket_path.unlink()
            print("[cosyvoice2] Shutdown complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice2 Unix socket server (Sichuanese dialect)")
    parser.add_argument("--socket", default="/tmp/cosyvoice.sock", help="Unix socket path")
    parser.add_argument("--model-dir", default=None, help="CosyVoice2 model directory")
    parser.add_argument("--prompt", default=None, help="Voice prompt WAV file (default: grandma)")
    args = parser.parse_args()

    server = CosyVoice2Server(Path(args.socket), model_dir=args.model_dir, prompt_path=args.prompt)
    server.serve()


if __name__ == "__main__":
    main()
