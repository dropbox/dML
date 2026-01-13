#!/usr/bin/env python3
"""
Unified TTS Server with Hybrid Routing.
- Routes EN/JA to Kokoro (faster)
- Routes other 15 languages to XTTS v2
- Unix socket interface for C++ daemon integration
- Pre-loads both models for low latency
"""

import torch
import os
import sys
import json
import time
import socket
import struct
import tempfile
import threading
import traceback
from pathlib import Path

# Add project root to path for provider import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from providers import play_audio

# Patch torch.load for PyTorch 2.6 compatibility
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# XTTS v2 supported languages
XTTS_LANGUAGES = {
    'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr',
    'ru', 'nl', 'cs', 'ar', 'zh-cn', 'zh', 'hu', 'ko', 'ja', 'hi'
}

# Kokoro supported languages (faster, preferred for these)
KOKORO_LANGUAGES = {'en', 'ja'}

SOCKET_PATH = '/tmp/tts_server.sock'
REF_AUDIO_PATH = '/tmp/ref_audio.wav'


class HybridTTSServer:
    def __init__(self, enable_cache: bool = True, lazy_xtts: bool = True):
        self.kokoro = None
        self.xtts = None
        self.xtts_loaded = False
        self.lazy_xtts = lazy_xtts  # Load XTTS only when needed
        self.lock = threading.Lock()
        self.enable_cache = enable_cache
        self.audio_cache = {}  # Cache for common phrases
        self._load_models()
        if enable_cache:
            self._pre_cache_phrases()

    def _load_models(self):
        """Pre-load both TTS models."""
        print("[TTS Server] Loading models...")
        start = time.time()

        # Load Kokoro for EN/JA
        self.kokoro_pipelines = {}
        try:
            import kokoro
            print("[TTS Server] Loading Kokoro...")

            # Determine best device (MPS > CUDA > CPU)
            device = 'cpu'
            if torch.backends.mps.is_available():
                device = 'mps'
                print("[TTS Server] Using Metal GPU (MPS) for acceleration")
            elif torch.cuda.is_available():
                device = 'cuda'
                print("[TTS Server] Using CUDA GPU for acceleration")
            else:
                print("[TTS Server] Using CPU (no GPU available)")

            # Pre-load pipelines for each language
            self.kokoro_pipelines['en'] = kokoro.KPipeline(lang_code='a')  # American English
            self.kokoro_pipelines['ja'] = kokoro.KPipeline(lang_code='j')  # Japanese

            # Move models to GPU if available
            if device != 'cpu':
                for lang, pipeline in self.kokoro_pipelines.items():
                    if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'to'):
                        pipeline.model.to(device)
                        print(f"[TTS Server] Moved {lang} pipeline to {device}")

            self.kokoro = kokoro
            print("[TTS Server] Kokoro loaded (EN + JA pipelines).")

            # Pre-warm with silent synthesis
            print("[TTS Server] Pre-warming Kokoro...")
            for lang, pipeline in self.kokoro_pipelines.items():
                for _, _, _ in pipeline("test", voice='af_heart' if lang == 'en' else 'jf_alpha'):
                    pass
            print("[TTS Server] Kokoro warmed up.")
        except ImportError:
            print("[TTS Server] WARNING: Kokoro not installed, will use XTTS for EN/JA")
            self.kokoro = None

        # Load XTTS v2 for multilingual (lazy by default to preserve MPS performance)
        if not self.lazy_xtts:
            self._load_xtts()
        else:
            print("[TTS Server] XTTS v2 will be loaded on first use (lazy mode)")

        # Ensure reference audio exists for XTTS voice cloning
        self._ensure_ref_audio()

        print(f"[TTS Server] Models loaded in {time.time() - start:.1f}s")

    def _load_xtts(self):
        """Load XTTS v2 model (called lazily or at init)."""
        if self.xtts_loaded:
            return
        try:
            from TTS.api import TTS
            print("[TTS Server] Loading XTTS v2...")
            self.xtts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False)
            self.xtts_loaded = True
            print("[TTS Server] XTTS v2 loaded.")
        except Exception as e:
            print(f"[TTS Server] ERROR loading XTTS: {e}", file=sys.stderr)
            traceback.print_exc()

    def _pre_cache_phrases(self):
        """Pre-synthesize common phrases for instant playback."""
        if not self.kokoro:
            return

        import numpy as np

        # Common coding-related phrases
        common_phrases = [
            ("Starting task", "en"),
            ("Task completed", "en"),
            ("Running tests", "en"),
            ("All tests passed", "en"),
            ("Build succeeded", "en"),
            ("Error occurred", "en"),
            ("Processing", "en"),
            ("Done", "en"),
            ("タスクを開始します", "ja"),
            ("完了しました", "ja"),
            ("テストを実行中", "ja"),
        ]

        print("[TTS Server] Pre-caching common phrases...")
        for text, lang in common_phrases:
            try:
                if lang not in self.kokoro_pipelines:
                    continue
                pipeline = self.kokoro_pipelines[lang]
                voice = 'af_heart' if lang == 'en' else 'jf_alpha'
                audio_chunks = []
                for _, _, audio in pipeline(text, voice=voice):
                    if audio is not None:
                        audio_chunks.append(audio)
                if audio_chunks:
                    self.audio_cache[(text.lower(), lang)] = np.concatenate(audio_chunks)
            except Exception as e:
                print(f"[TTS Server] Cache error for '{text}': {e}", file=sys.stderr)
                traceback.print_exc()
        print(f"[TTS Server] Cached {len(self.audio_cache)} phrases")

    def _ensure_ref_audio(self):
        """Create reference audio for XTTS voice cloning."""
        if not os.path.exists(REF_AUDIO_PATH):
            print("[TTS Server] Creating reference audio...")
            if self.kokoro:
                # Use Kokoro to create reference audio
                import subprocess
                subprocess.run([
                    'python', str(Path(__file__).parent / 'kokoro_tts.py'),
                    'Hello world, this is a reference audio for voice cloning.',
                    '-o', REF_AUDIO_PATH, '-l', 'en'
                ], capture_output=True)

    def synthesize(self, text: str, language: str = 'en', output_path: str = None) -> dict:
        """Synthesize speech with automatic model selection."""
        start = time.time()

        # Normalize language code
        lang = language.lower().split('-')[0] if language != 'zh-cn' else 'zh-cn'

        # Select model
        use_kokoro = lang in KOKORO_LANGUAGES and self.kokoro is not None

        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav')

        try:
            with self.lock:
                if use_kokoro:
                    return self._synthesize_kokoro(text, lang, output_path, start)
                else:
                    return self._synthesize_xtts(text, lang, output_path, start)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': 'kokoro' if use_kokoro else 'xtts',
                'latency': time.time() - start
            }

    def synthesize_stream(self, text: str, language: str = 'en'):
        """Generator that yields audio chunks as they're synthesized.

        Yields:
            dict with keys: 'chunk' (numpy array), 'chunk_idx', 'first_chunk_ms', 'sample_rate'
        """
        start = time.time()
        lang = language.lower().split('-')[0] if language != 'zh-cn' else 'zh-cn'

        if lang not in KOKORO_LANGUAGES or self.kokoro is None:
            # Fallback to non-streaming for XTTS
            yield {'error': 'Streaming only supported for Kokoro (en/ja)'}
            return

        if lang not in self.kokoro_pipelines:
            yield {'error': f'Pipeline not loaded for {lang}'}
            return

        pipeline = self.kokoro_pipelines[lang]
        voice = 'af_heart' if lang == 'en' else 'jf_alpha'

        first_chunk_time = None
        chunk_idx = 0

        with self.lock:
            for _, _, audio in pipeline(text, voice=voice):
                if audio is not None:
                    if first_chunk_time is None:
                        first_chunk_time = (time.time() - start) * 1000
                    yield {
                        'chunk': audio,
                        'chunk_idx': chunk_idx,
                        'first_chunk_ms': first_chunk_time,
                        'sample_rate': 24000
                    }
                    chunk_idx += 1

    def _synthesize_kokoro(self, text: str, lang: str, output_path: str, start: float) -> dict:
        """Synthesize with Kokoro (EN/JA) using pre-loaded pipeline."""
        import soundfile as sf
        import numpy as np

        if lang not in self.kokoro_pipelines:
            return {
                'success': False,
                'error': f'Kokoro pipeline not loaded for language: {lang}',
                'model': 'kokoro',
                'latency': time.time() - start
            }

        # Check cache first for instant playback
        cache_key = (text.lower(), lang)
        if self.enable_cache and cache_key in self.audio_cache:
            full_audio = self.audio_cache[cache_key]
            sf.write(output_path, full_audio, 24000)
            latency = time.time() - start
            duration = len(full_audio) / 24000
            return {
                'success': True,
                'output_path': output_path,
                'model': 'kokoro',
                'language': lang,
                'duration': duration,
                'latency': latency,
                'first_chunk_ms': latency * 1000,  # Cache hit = instant
                'num_chunks': 1,
                'rtf': latency / duration if duration > 0 else 0,
                'cache_hit': True
            }

        pipeline = self.kokoro_pipelines[lang]
        voice = 'af_heart' if lang == 'en' else 'jf_alpha'

        # Collect all audio chunks from generator with timing
        audio_chunks = []
        first_chunk_time = None
        chunk_times = []

        for _, _, audio in pipeline(text, voice=voice):
            if audio is not None:
                chunk_time = time.time() - start
                if first_chunk_time is None:
                    first_chunk_time = chunk_time
                chunk_times.append(chunk_time)
                audio_chunks.append(audio)

        if not audio_chunks:
            return {
                'success': False,
                'error': 'No audio generated',
                'model': 'kokoro',
                'latency': time.time() - start
            }

        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks)

        # Save to file (Kokoro outputs 24kHz audio)
        sf.write(output_path, full_audio, 24000)

        latency = time.time() - start
        duration = len(full_audio) / 24000

        return {
            'success': True,
            'output_path': output_path,
            'model': 'kokoro',
            'language': lang,
            'duration': duration,
            'latency': latency,
            'first_chunk_ms': first_chunk_time * 1000 if first_chunk_time else 0,
            'num_chunks': len(audio_chunks),
            'rtf': latency / duration if duration > 0 else 0,
            'cache_hit': False
        }

    def _synthesize_xtts(self, text: str, lang: str, output_path: str, start: float) -> dict:
        """Synthesize with XTTS v2 (multilingual)."""
        # Lazy load XTTS on first use
        if not self.xtts_loaded:
            self._load_xtts()

        if self.xtts is None:
            return {
                'success': False,
                'error': 'XTTS not loaded',
                'model': 'xtts',
                'latency': time.time() - start
            }

        # Map language codes
        xtts_lang = lang if lang != 'zh' else 'zh-cn'

        self.xtts.tts_to_file(
            text=text,
            speaker_wav=REF_AUDIO_PATH,
            language=xtts_lang,
            file_path=output_path
        )

        latency = time.time() - start

        # Get audio info
        import torchaudio
        audio, sr = torchaudio.load(output_path)
        duration = audio.shape[1] / sr

        return {
            'success': True,
            'output_path': output_path,
            'model': 'xtts',
            'language': lang,
            'duration': duration,
            'latency': latency,
            'rtf': latency / duration if duration > 0 else 0
        }

    def handle_request(self, request: dict) -> dict:
        """Handle a TTS request."""
        command = request.get('command', 'synthesize')

        if command == 'synthesize':
            return self.synthesize(
                text=request.get('text', ''),
                language=request.get('language', 'en'),
                output_path=request.get('output_path')
            )
        elif command == 'status':
            return {
                'success': True,
                'kokoro_loaded': self.kokoro is not None,
                'xtts_loaded': self.xtts is not None,
                'supported_languages': list(XTTS_LANGUAGES)
            }
        elif command == 'shutdown':
            return {'success': True, 'message': 'Shutting down'}
        else:
            return {'success': False, 'error': f'Unknown command: {command}'}

    def run_socket_server(self):
        """Run Unix socket server for C++ daemon integration."""
        # Remove old socket
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(5)
        print(f"[TTS Server] Listening on {SOCKET_PATH}")

        try:
            while True:
                conn, addr = server.accept()
                threading.Thread(target=self._handle_connection, args=(conn,)).start()
        except KeyboardInterrupt:
            print("\n[TTS Server] Shutting down...")
        finally:
            server.close()
            if os.path.exists(SOCKET_PATH):
                os.unlink(SOCKET_PATH)

    def _handle_connection(self, conn):
        """Handle a single client connection."""
        try:
            # Read message length (4 bytes, big endian)
            length_bytes = conn.recv(4)
            if not length_bytes:
                return
            msg_length = struct.unpack('>I', length_bytes)[0]

            # Read message
            data = b''
            while len(data) < msg_length:
                chunk = conn.recv(min(4096, msg_length - len(data)))
                if not chunk:
                    break
                data += chunk

            request = json.loads(data.decode('utf-8'))
            response = self.handle_request(request)

            # Send response
            response_bytes = json.dumps(response).encode('utf-8')
            conn.sendall(struct.pack('>I', len(response_bytes)))
            conn.sendall(response_bytes)

        except Exception as e:
            print(f"[TTS Server] Error handling connection: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            conn.close()


def send_request(request: dict) -> dict:
    """Send a request to the TTS server."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)

    try:
        # Send request
        request_bytes = json.dumps(request).encode('utf-8')
        client.sendall(struct.pack('>I', len(request_bytes)))
        client.sendall(request_bytes)

        # Read response length
        length_bytes = client.recv(4)
        msg_length = struct.unpack('>I', length_bytes)[0]

        # Read response
        data = b''
        while len(data) < msg_length:
            chunk = client.recv(min(4096, msg_length - len(data)))
            if not chunk:
                break
            data += chunk

        return json.loads(data.decode('utf-8'))
    finally:
        client.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hybrid TTS Server')
    parser.add_argument('--server', action='store_true', help='Run as socket server')
    parser.add_argument('--speak', type=str, help='Text to synthesize (direct mode)')
    parser.add_argument('--client', type=str, help='Text to synthesize (via server)')
    parser.add_argument('--language', '-l', type=str, default='en', help='Language code')
    parser.add_argument('--output', '-o', type=str, help='Output WAV path')
    parser.add_argument('--play', action='store_true', help='Play audio after synthesis')
    parser.add_argument('--status', action='store_true', help='Get server status')
    args = parser.parse_args()

    if args.server:
        server = HybridTTSServer()
        server.run_socket_server()
    elif args.status:
        try:
            result = send_request({'command': 'status'})
            print(json.dumps(result, indent=2))
        except FileNotFoundError:
            print("Server not running. Start with: python tts_server.py --server")
    elif args.client:
        # Use server via socket
        try:
            result = send_request({
                'command': 'synthesize',
                'text': args.client,
                'language': args.language,
                'output_path': args.output
            })
            print(json.dumps(result, indent=2))
            if args.play and result.get('success') and result.get('output_path'):
                with open(result['output_path'], 'rb') as f:
                    play_audio(f.read())
        except FileNotFoundError:
            print("Server not running. Start with: python tts_server.py --server")
    elif args.speak:
        # Direct synthesis (no server needed)
        tts = HybridTTSServer()
        output_path = args.output or tempfile.mktemp(suffix='.wav')
        result = tts.synthesize(
            text=args.speak,
            language=args.language,
            output_path=output_path
        )
        print(json.dumps(result, indent=2))
        if args.play and result.get('success') and result.get('output_path'):
            with open(result['output_path'], 'rb') as f:
                play_audio(f.read())
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
