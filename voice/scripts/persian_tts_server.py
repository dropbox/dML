#!/usr/bin/env python3
"""
Persian TTS Streaming Server (Unix socket)

Supports dual engines for Persian (Farsi) TTS:

1. MMS-TTS-FAS (Meta) - VITS architecture, high quality
   - HuggingFace: facebook/mms-tts-fas
   - License: CC-BY-NC 4.0 (non-commercial)
   - Recommended for best quality

2. ManaTTS Tacotron2 - 114h Persian corpus
   - HuggingFace: MahtaFetrat/Persian-Tacotron2-on-ManaTTS
   - License: CC0-1.0 (maximally permissive, commercial friendly)
   - Use for commercial deployment

Protocol:
    Request: newline-terminated JSON
        {"text": "...", "engine": "mms"}  # or "mana"
    Response: 4-byte little-endian length prefix followed by WAV bytes

Usage:
    python scripts/persian_tts_server.py --socket /tmp/persian_tts.sock --device mps

Installation:
    pip install transformers torch torchaudio

Model download (automatic on first use, or manual):
    huggingface-cli download facebook/mms-tts-fas --local-dir models/mms-tts-persian

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import io
import json
import os
import signal
import socket
import struct
import time
import warnings
from pathlib import Path
from typing import Optional

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class PersianTTSServer:
    """Persistent Persian TTS server using Unix domain sockets."""

    def __init__(self, socket_path: Path, device: str = "mps", default_engine: str = "mms"):
        self.socket_path = socket_path
        self.device = device
        self.default_engine = default_engine
        self.mms_model = None
        self.mms_tokenizer = None
        self.mana_model = None
        self.mana_vocoder = None

        self._load_models()

    def _load_models(self) -> None:
        """Load Persian TTS models into memory."""
        print("[persian-tts] Loading Persian TTS models...", flush=True)

        import torch

        # Determine device
        if self.device == "mps" and torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
            print("[persian-tts] Using MPS (Apple Metal) acceleration", flush=True)
        elif self.device == "cuda" and torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
            print("[persian-tts] Using CUDA acceleration", flush=True)
        else:
            self.torch_device = torch.device("cpu")
            print("[persian-tts] Using CPU (no GPU acceleration)", flush=True)

        # Load MMS-TTS-FAS (Meta VITS) - primary engine
        try:
            from transformers import VitsModel, AutoTokenizer

            start = time.time()
            print("[persian-tts] Loading MMS-TTS-FAS (Meta VITS)...", flush=True)

            self.mms_model = VitsModel.from_pretrained("facebook/mms-tts-fas")
            self.mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fas")
            self.mms_model.to(self.torch_device)
            self.mms_model.eval()

            print(f"[persian-tts] MMS-TTS-FAS loaded in {time.time() - start:.1f}s", flush=True)
        except Exception as e:
            print(f"[persian-tts] WARNING: Failed to load MMS-TTS-FAS: {e}", flush=True)
            print("[persian-tts] Install with: pip install transformers torch", flush=True)

        # Load ManaTTS Tacotron2 (optional CC0 fallback)
        # Note: ManaTTS uses a different architecture (Tacotron2 + HiFi-GAN)
        # We only load it if explicitly requested or MMS fails
        if self.default_engine == "mana" or self.mms_model is None:
            try:
                print("[persian-tts] Loading ManaTTS Tacotron2 + HiFi-GAN...", flush=True)
                start = time.time()

                # ManaTTS is loaded via torch.hub
                # Note: This is a placeholder - actual ManaTTS loading may differ
                # based on the repository structure
                try:
                    import torch.hub
                    self.mana_model = torch.hub.load(
                        'MahtaFetrat/Persian-Tacotron2-on-ManaTTS',
                        'tacotron2',
                        trust_repo=True
                    )
                    self.mana_vocoder = torch.hub.load(
                        'MahtaFetrat/Persian-Tacotron2-on-ManaTTS',
                        'hifigan',
                        trust_repo=True
                    )
                    self.mana_model.to(self.torch_device)
                    self.mana_vocoder.to(self.torch_device)
                    self.mana_model.eval()
                    self.mana_vocoder.eval()
                    print(f"[persian-tts] ManaTTS loaded in {time.time() - start:.1f}s", flush=True)
                except Exception as mana_err:
                    print(f"[persian-tts] ManaTTS not available via torch.hub: {mana_err}", flush=True)
                    print("[persian-tts] ManaTTS fallback disabled", flush=True)

            except Exception as e:
                print(f"[persian-tts] WARNING: Failed to load ManaTTS: {e}", flush=True)

        # Check we have at least one working model
        if self.mms_model is None and self.mana_model is None:
            raise RuntimeError("No Persian TTS model available - install transformers and download models")

        print("[persian-tts] Persian TTS ready (Farsi synthesis)", flush=True)

    def synthesize(self, text: str, engine: str = "mms") -> tuple[bytes, int]:
        """Synthesize text to WAV bytes using specified engine.

        Returns:
            tuple: (WAV bytes, sample rate)
        """
        import torch
        import torchaudio

        engine = engine.lower()

        # Select engine
        if engine == "mana" and self.mana_model is not None:
            return self._synthesize_mana(text)
        elif self.mms_model is not None:
            return self._synthesize_mms(text)
        elif self.mana_model is not None:
            return self._synthesize_mana(text)
        else:
            raise RuntimeError("No Persian TTS model available")

    def _synthesize_mms(self, text: str) -> tuple[bytes, int]:
        """Synthesize using MMS-TTS-FAS (Meta VITS)."""
        import tempfile
        import wave
        import numpy as np
        import torch

        # Tokenize and generate
        inputs = self.mms_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.mms_model(**inputs)

        waveform = output.waveform[0].cpu()  # Shape: (1, samples) or (samples,)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)  # Remove channel dimension

        # MMS-TTS outputs at 16000Hz per model config
        sample_rate = self.mms_model.config.sampling_rate

        # Convert to numpy and then to 16-bit PCM
        audio_np = waveform.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Write WAV using wave module (torchaudio.save requires torchcodec on Python 3.14)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with wave.open(str(tmp_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            wav_bytes = tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return wav_bytes, sample_rate

    def _synthesize_mana(self, text: str) -> tuple[bytes, int]:
        """Synthesize using ManaTTS Tacotron2 + HiFi-GAN."""
        import tempfile
        import wave
        import numpy as np
        import torch

        with torch.no_grad():
            # Generate mel spectrogram with Tacotron2
            mel_output, mel_length, alignment = self.mana_model.inference(text)

            # Generate audio with HiFi-GAN vocoder
            audio = self.mana_vocoder(mel_output)
            audio = audio.squeeze().cpu()

        # ManaTTS typically outputs at 22050Hz
        sample_rate = 22050

        # Convert to numpy and then to 16-bit PCM
        audio_np = audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Write WAV using wave module (torchaudio.save requires torchcodec on Python 3.14)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with wave.open(str(tmp_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            wav_bytes = tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return wav_bytes, sample_rate

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
            engine = request.get("engine", self.default_engine)

            if not text:
                conn.sendall(b'{"error":"missing text"}')
                return

            print(f"[persian-tts] Synthesizing ({engine}): {text[:50]}...", flush=True)
            start = time.time()

            wav_data, sample_rate = self.synthesize(text, engine)

            elapsed = time.time() - start
            print(f"[persian-tts] Synthesized {len(wav_data)} bytes at {sample_rate}Hz in {elapsed:.2f}s", flush=True)

            conn.sendall(struct.pack("<I", len(wav_data)))
            conn.sendall(wav_data)
        except Exception as e:
            print(f"[persian-tts] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
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

        print(f"[persian-tts] Listening on {self.socket_path}", flush=True)
        engines = []
        if self.mms_model is not None:
            engines.append("MMS-TTS-FAS (Meta VITS)")
        if self.mana_model is not None:
            engines.append("ManaTTS (CC0)")
        print(f"[persian-tts] Available engines: {', '.join(engines)}", flush=True)

        def shutdown(signum, _frame):
            print(f"[persian-tts] Received signal {signum}, shutting down", flush=True)
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
            print("[persian-tts] Shutdown complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persian TTS Unix socket server (MMS-TTS-FAS / ManaTTS)")
    parser.add_argument("--socket", default="/tmp/persian_tts.sock", help="Unix socket path")
    parser.add_argument("--device", default="mps", help="Target device (mps/cuda/cpu)")
    parser.add_argument("--engine", default="mms", choices=["mms", "mana"],
                        help="Default engine: mms (Meta VITS, CC-BY-NC) or mana (ManaTTS, CC0)")
    args = parser.parse_args()

    server = PersianTTSServer(Path(args.socket), device=args.device, default_engine=args.engine)
    server.serve()


if __name__ == "__main__":
    main()
