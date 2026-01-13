#!/usr/bin/env python3
"""
Fish-Speech Streaming Server (Unix socket)

Fish-Speech is #1 on TTS-Arena2 with 0.008 WER - SOTA for Arabic TTS.
Provides a lightweight Unix socket server for Fish-Speech synthesis. Designed to be
consumed by the C++ client in stream-tts-cpp/src/fish_speech_engine.cpp.

Protocol:
    Request: newline-terminated JSON
        {"text": "...", "language": "ar", "speaker_ref": "/path/to/ref.wav"}
    Response: 4-byte little-endian length prefix followed by WAV bytes

Usage:
    python scripts/fish_speech_server.py --socket /tmp/fish_speech.sock --device mps

The server keeps the Fish-Speech model loaded to minimize latency. Reference WAVs for
each language are generated on first use via macOS `say` (fallback: espeak-ng).

Installation:
    pip install fish-speech

Model download:
    huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish-speech

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
from typing import Dict, Optional

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Language-specific reference generation configs
# Fish-Speech excels at Arabic - that's our primary use case
LANG_REFS: Dict[str, tuple[str, str]] = {
    "ar": ("Tessa", "مرحبا، هذا اختبار لنظام تحويل النص إلى كلام العربي"),
    "en": ("Samantha", "Hello, this is a test of the voice system"),
    "zh": ("Tingting", "你好，这是语音测试系统"),
    "ja": ("Kyoko", "こんにちは、これはテストの音声です"),
    "ko": ("Yuna", "안녕하세요, 이것은 음성 테스트입니다"),
    "es": ("Paulina", "Hola, esta es una prueba del sistema de voz"),
    "fr": ("Amelie", "Bonjour, ceci est un test du systeme vocal"),
}


class FishSpeechServer:
    """Persistent Fish-Speech server using Unix domain sockets."""

    def __init__(self, socket_path: Path, device: str = "mps", model_path: Optional[str] = None):
        self.socket_path = socket_path
        self.device = device
        self.model_path = model_path or "fishaudio/fish-speech-1.5"
        self.model = None
        self.reference_wavs: Dict[str, Path] = {}

        self._load_model()

    def _load_model(self) -> None:
        """Load Fish-Speech model into memory."""
        print("[fish-speech] Importing Fish-Speech library...", flush=True)

        try:
            import torch
            from fish_speech.models.text2semantic import TextToSemanticModel
            from fish_speech.models.vqgan import VQGAN

            start = time.time()

            # Load semantic model (text to tokens)
            self.semantic_model = TextToSemanticModel.from_pretrained(self.model_path)

            # Load VQGAN vocoder (tokens to audio)
            self.vqgan = VQGAN.from_pretrained(self.model_path)

            # Move to device
            if self.device == "mps" and torch.backends.mps.is_available():
                self.semantic_model = self.semantic_model.to("mps")
                self.vqgan = self.vqgan.to("mps")
                print(f"[fish-speech] Using MPS (Apple Metal) acceleration", flush=True)
            elif self.device == "cuda" and torch.cuda.is_available():
                self.semantic_model = self.semantic_model.to("cuda")
                self.vqgan = self.vqgan.to("cuda")
                print(f"[fish-speech] Using CUDA acceleration", flush=True)
            else:
                print(f"[fish-speech] Using CPU (no GPU acceleration)", flush=True)

            self.model = True  # Flag to indicate model is loaded
            print(f"[fish-speech] Fish-Speech model loaded in {time.time() - start:.1f}s", flush=True)

        except ImportError as e:
            print(f"[fish-speech] WARNING: fish-speech library not installed: {e}", flush=True)
            print("[fish-speech] Install with: pip install fish-speech", flush=True)
            print("[fish-speech] Falling back to TTS library with fish-speech model...", flush=True)

            # Fallback: Try using the generic TTS library if fish-speech not installed
            try:
                import torch
                # PyTorch 2.6+ requires weights_only=False for older TTS models
                # Monkey-patch torch.load to use weights_only=False for TTS library
                _original_torch_load = torch.load
                def _patched_torch_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return _original_torch_load(*args, **kwargs)
                torch.load = _patched_torch_load

                from TTS.api import TTS
                self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
                try:
                    self.tts.to(self.device)
                except Exception:
                    pass
                self.model = "tts_fallback"
                print(f"[fish-speech] Fallback TTS model loaded", flush=True)
            except Exception as e2:
                print(f"[fish-speech] ERROR: Could not load any TTS model: {e2}", flush=True)
                raise RuntimeError(f"No TTS model available: {e}") from e

    def _get_reference_wav(self, language: str) -> Path:
        """Return a speaker reference WAV for the target language (generates if missing)."""
        language = language.lower()
        if language in self.reference_wavs:
            return self.reference_wavs[language]

        ref_path = Path(f"/tmp/fish_ref_{language}.wav")

        if not ref_path.exists():
            voice, text = LANG_REFS.get(language, LANG_REFS["en"])
            try:
                # Prefer macOS voices for natural references
                cmd = ["say", "-o", str(ref_path), "--data-format=LEI16@24000", "-v", voice, text]
                os.makedirs(ref_path.parent, exist_ok=True)
                result = os.system(" ".join(cmd) + " >/dev/null 2>&1")
                if result != 0:
                    raise RuntimeError("say command failed")
                print(f"[fish-speech] Created reference WAV for {language}: {ref_path}", flush=True)
            except Exception:
                # Fallback to espeak-ng if macOS voice unavailable
                os.system(f'espeak-ng -w "{ref_path}" "Reference audio for Fish-Speech." >/dev/null 2>&1')
                print(f"[fish-speech] espeak-ng fallback reference for {language}: {ref_path}", flush=True)

        self.reference_wavs[language] = ref_path
        return ref_path

    def synthesize(self, text: str, language: str, speaker_ref: Optional[str]) -> bytes:
        """Synthesize text to WAV bytes using Fish-Speech."""
        if self.model is None:
            raise RuntimeError("Fish-Speech model not loaded")

        language = language or "ar"  # Default to Arabic (Fish-Speech's strength)
        ref_wav = Path(speaker_ref) if speaker_ref else self._get_reference_wav(language)

        with tempfile.NamedTemporaryFile(prefix="fish_out_", suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if self.model == "tts_fallback":
                # Use XTTS fallback
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(tmp_path),
                    language=language if not language.startswith("zh") else "zh-cn",
                    speaker_wav=str(ref_wav),
                )
            else:
                # Use native Fish-Speech API
                import torch
                import torchaudio

                # Load reference audio for voice cloning
                ref_audio, ref_sr = torchaudio.load(str(ref_wav))
                if ref_sr != 24000:
                    ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 24000)

                # Generate semantic tokens from text
                with torch.no_grad():
                    semantic_tokens = self.semantic_model.generate(
                        text,
                        prompt_audio=ref_audio.to(self.semantic_model.device),
                        language=language,
                    )

                    # Generate audio from semantic tokens
                    audio = self.vqgan.decode(semantic_tokens)

                # Save to WAV file
                torchaudio.save(str(tmp_path), audio.cpu(), 24000)

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
            language = request.get("language", "ar")  # Default Arabic
            speaker_ref = request.get("speaker_ref")

            if not text:
                conn.sendall(b'{"error":"missing text"}')
                return

            print(f"[fish-speech] Synthesizing: lang={language}, text={text[:50]}...", flush=True)
            start = time.time()

            wav_data = self.synthesize(text, language, speaker_ref)

            elapsed = time.time() - start
            print(f"[fish-speech] Synthesized {len(wav_data)} bytes in {elapsed:.2f}s", flush=True)

            conn.sendall(struct.pack("<I", len(wav_data)))
            conn.sendall(wav_data)
        except Exception as e:
            print(f"[fish-speech] ERROR: {e}", flush=True)
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

        print(f"[fish-speech] Listening on {self.socket_path}", flush=True)
        print(f"[fish-speech] Arabic TTS ready (#1 TTS-Arena2, 0.008 WER)", flush=True)

        def shutdown(signum, _frame):
            print(f"[fish-speech] Received signal {signum}, shutting down", flush=True)
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
            print("[fish-speech] Shutdown complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fish-Speech Unix socket server (SOTA Arabic TTS)")
    parser.add_argument("--socket", default="/tmp/fish_speech.sock", help="Unix socket path")
    parser.add_argument("--device", default="mps", help="Target device (mps/cuda/cpu)")
    parser.add_argument("--model", default=None, help="Model path or HuggingFace repo")
    args = parser.parse_args()

    server = FishSpeechServer(Path(args.socket), device=args.device, model_path=args.model)
    server.serve()


if __name__ == "__main__":
    main()
