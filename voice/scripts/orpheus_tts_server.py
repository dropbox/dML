#!/usr/bin/env python3
"""
Orpheus-TTS-Turkish Streaming Server (Unix socket)

Orpheus-TTS-Turkish is a 3B parameter model purpose-built for Turkish TTS.
It has 220+ hours of Turkish training data and supports emotion tags:
  <laugh>, <sigh>, <cough>, <cry>, <gasp>, <groan>, <yawn>

Provides a lightweight Unix socket server for Orpheus-TTS synthesis. Designed
to be consumed by the C++ client in stream-tts-cpp/src/orpheus_tts_engine.cpp.

Protocol:
    Request: newline-terminated JSON
        {"text": "...", "emotion": "laugh"}
    Response: 4-byte little-endian length prefix followed by WAV bytes

Usage:
    python scripts/orpheus_tts_server.py --socket /tmp/orpheus_tts.sock --device mps

The server keeps the Orpheus-TTS model loaded to minimize latency.

Installation:
    pip install transformers torch torchaudio snac

Model download:
    huggingface-cli download turkishlanguagemodels/Orpheus-TTS-Turkish-3B --local-dir models/orpheus-tts-turkish

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import json
import os
import signal
import socket
import struct
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Valid emotion tags for Orpheus-TTS
VALID_EMOTIONS = {"laugh", "sigh", "cough", "cry", "gasp", "groan", "yawn"}

from orpheus_snac_utils import extract_audio_codes, snac_tensors_from_interleaved


class OrpheusTTSServer:
    """Persistent Orpheus-TTS server using Unix domain sockets."""

    def __init__(self, socket_path: Path, device: str = "mps", model_path: Optional[str] = None):
        self.socket_path = socket_path
        self.device = device
        self.model_path = model_path or "Karayakar/Orpheus-TTS-Turkish-PT-5000"
        self.model = None
        self.tokenizer = None
        self.snac_model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load Orpheus-TTS model into memory."""
        print("[orpheus-tts] Importing dependencies...", flush=True)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            start = time.time()

            print(f"[orpheus-tts] Loading model from {self.model_path}...", flush=True)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Load model with bfloat16 for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

            # Move to device
            if self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
                print(f"[orpheus-tts] Using MPS (Apple Metal) acceleration", flush=True)
            elif self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                print(f"[orpheus-tts] Using CUDA acceleration", flush=True)
            else:
                print(f"[orpheus-tts] Using CPU (no GPU acceleration)", flush=True)

            # Try to load SNAC decoder for audio generation
            try:
                from snac import SNAC
                self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
                if self.device == "mps":
                    self.snac_model = self.snac_model.to("mps")
                elif self.device == "cuda":
                    self.snac_model = self.snac_model.to("cuda")
                print(f"[orpheus-tts] SNAC decoder loaded", flush=True)
            except ImportError:
                print(f"[orpheus-tts] WARNING: SNAC not installed, audio generation disabled", flush=True)
                print(f"[orpheus-tts] Install with: pip install snac", flush=True)
            except Exception as e:
                print(f"[orpheus-tts] WARNING: Failed to load SNAC: {e}", flush=True)

            print(f"[orpheus-tts] Model loaded in {time.time() - start:.1f}s", flush=True)

        except ImportError as e:
            print(f"[orpheus-tts] ERROR: Missing dependencies: {e}", flush=True)
            print("[orpheus-tts] Install with: pip install transformers torch torchaudio snac", flush=True)
            raise RuntimeError(f"Missing dependencies: {e}") from e
        except Exception as e:
            print(f"[orpheus-tts] ERROR: Failed to load model: {e}", flush=True)
            raise

    def _format_prompt(self, text: str, emotion: Optional[str] = None) -> str:
        """Format text with optional emotion tag for Orpheus-TTS."""
        # Orpheus-TTS uses special tags: <laugh>, <sigh>, etc.
        if emotion and emotion.lower() in VALID_EMOTIONS:
            text = f"<{emotion.lower()}> {text}"
        return text

    def synthesize(self, text: str, emotion: Optional[str] = None) -> bytes:
        """Synthesize text to WAV bytes using Orpheus-TTS."""
        import torch
        import torchaudio

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Orpheus-TTS model not loaded")

        if self.snac_model is None:
            raise RuntimeError("SNAC decoder not loaded - cannot generate audio")

        # Format prompt with optional emotion
        prompt = self._format_prompt(text, emotion)

        with tempfile.NamedTemporaryFile(prefix="orpheus_out_", suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)

            # Generate audio tokens
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Extract generated tokens (skip input tokens)
            generated_ids = outputs[0, input_ids.shape[1]:].tolist()

            # Filter to audio codes (official offset 128262, 7 codes/frame)
            audio_codes = extract_audio_codes(generated_ids)

            if not audio_codes:
                raise RuntimeError("No audio codes generated")

            codes = snac_tensors_from_interleaved(audio_codes, self.snac_model.device)

            # Decode to audio
            with torch.no_grad():
                audio = self.snac_model.decode(codes)

            # Audio is (batch, channels, samples) at 24kHz
            audio = audio.squeeze(0).cpu()

            # Save as WAV
            torchaudio.save(str(tmp_path), audio, 24000)

            data = tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return data

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
                conn.sendall(b'{"error":"invalid json"}')
                return

            text = request.get("text", "")
            emotion = request.get("emotion")

            if not text:
                conn.sendall(b'{"error":"missing text"}')
                return

            print(f"[orpheus-tts] Synthesizing: text={text[:50]}..., emotion={emotion}", flush=True)
            start = time.time()

            wav_data = self.synthesize(text, emotion)

            elapsed = time.time() - start
            print(f"[orpheus-tts] Synthesized {len(wav_data)} bytes in {elapsed:.2f}s", flush=True)

            conn.sendall(struct.pack("<I", len(wav_data)))
            conn.sendall(wav_data)
        except Exception as e:
            print(f"[orpheus-tts] ERROR: {e}", flush=True)
            try:
                msg = json.dumps({"error": str(e)}).encode("utf-8")
                conn.sendall(msg)
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

        print(f"[orpheus-tts] Listening on {self.socket_path}", flush=True)
        print(f"[orpheus-tts] Turkish TTS ready (3B parameters, 220+ hours training)", flush=True)

        def shutdown(signum, _frame):
            print(f"[orpheus-tts] Received signal {signum}, shutting down", flush=True)
            server.close()
            if self.socket_path.exists():
                self.socket_path.unlink()

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
            print("[orpheus-tts] Shutdown complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Orpheus-TTS-Turkish Unix socket server (3B Turkish TTS)")
    parser.add_argument("--socket", default="/tmp/orpheus_tts.sock", help="Unix socket path")
    parser.add_argument("--device", default="mps", help="Target device (mps/cuda/cpu)")
    parser.add_argument("--model", default=None, help="Model path or HuggingFace repo")
    args = parser.parse_args()

    server = OrpheusTTSServer(Path(args.socket), device=args.device, model_path=args.model)
    server.serve()


if __name__ == "__main__":
    main()
