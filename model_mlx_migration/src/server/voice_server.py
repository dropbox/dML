# Copyright 2024-2026 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SOTA++ Voice Server - Streaming ASR with Rich Audio Features.

This server provides WebSocket-based streaming ASR with:
- Real-time transcription via Zipformer (streaming mode)
- High-accuracy ROVER voting mode
- Rich audio features (emotion, pitch, phoneme, etc.)
- Multi-speaker handling via FLASepformer
- Language routing for 100+ languages

Architecture:
    Client → WebSocket → VoiceServer → Pipeline → RichToken → Client

The server handles:
1. Audio chunk reception (raw PCM or compressed)
2. Preprocessing (resample, DC removal, AGC, VAD)
3. Multi-speaker detection and separation
4. Language routing
5. ASR inference (streaming or high-accuracy)
6. Rich head inference
7. RichToken streaming back to client
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .rich_token import (
    ASRMode,
    EmotionLabel,
    HallucinationInfo,
    LanguageInfo,
    PhonemeInfo,
    PitchInfo,
    RichToken,
    SpeakerInfo,
    StreamingResponse,
    WordTimestamp,
    create_final_token,
    create_partial_token,
)

# Optional imports
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketServerProtocol = Any

try:
    import mlx.core as mx  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


logger = logging.getLogger(__name__)


class ServerState(str, Enum):
    """Voice server state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


class SessionState(str, Enum):
    """Client session state."""
    CONNECTED = "connected"
    STREAMING = "streaming"
    PROCESSING = "processing"
    DISCONNECTED = "disconnected"


@dataclass
class ServerConfig:
    """Voice server configuration."""

    # Network
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 10

    # Audio
    sample_rate: int = 16000
    chunk_size_ms: int = 320
    max_audio_duration_s: float = 300.0  # Max 5 minutes per request

    # ASR modes
    default_mode: ASRMode = ASRMode.STREAMING
    enable_high_accuracy: bool = True

    # Features
    enable_emotion: bool = True
    enable_pitch: bool = True
    enable_phoneme: bool = True
    enable_paralinguistics: bool = True
    enable_language: bool = True
    enable_singing: bool = True
    enable_timestamps: bool = True
    enable_hallucination: bool = True
    enable_speaker: bool = True

    # Multi-speaker
    enable_multi_speaker: bool = True
    max_speakers: int = 4

    # Performance
    inference_timeout_ms: int = 5000
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10

    # Logging
    log_latency: bool = True


@dataclass
class SessionMetrics:
    """Metrics for a client session."""
    start_time: float = 0.0
    audio_received_ms: float = 0.0
    chunks_received: int = 0
    tokens_sent: int = 0
    inference_time_ms: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class ClientSession:
    """State for a connected client."""
    session_id: str
    websocket: Any  # WebSocketServerProtocol
    state: SessionState = SessionState.CONNECTED
    mode: ASRMode = ASRMode.STREAMING
    utterance_id: str = ""
    chunk_index: int = 0
    sequence: int = 0
    metrics: SessionMetrics = field(default_factory=SessionMetrics)

    # Audio buffer for current utterance
    audio_buffer: list[np.ndarray] = field(default_factory=list)
    audio_start_ms: float = 0.0

    # Language routing
    detected_language: str | None = None
    language_confidence: float = 0.0


class ASRPipeline:
    """
    ASR inference pipeline abstraction.

    This is the interface that the voice server uses to run inference.
    Implementations can wrap different backends (MLX, PyTorch, etc.).
    """

    async def process_chunk(
        self,
        audio: np.ndarray,
        session: ClientSession,
        mode: ASRMode,
        config: ServerConfig,
    ) -> RichToken | None:
        """
        Process an audio chunk and return partial result.

        Args:
            audio: Audio samples at 16kHz
            session: Current client session
            mode: ASR mode (streaming or high-accuracy)
            config: Server configuration

        Returns:
            RichToken with partial results, or None if no output yet
        """
        raise NotImplementedError

    async def finalize_utterance(
        self,
        session: ClientSession,
        mode: ASRMode,
        config: ServerConfig,
    ) -> RichToken:
        """
        Finalize an utterance and return complete result.

        Args:
            session: Current client session
            mode: ASR mode (streaming or high-accuracy)
            config: Server configuration

        Returns:
            RichToken with final results including all rich features
        """
        raise NotImplementedError

    async def detect_language(
        self,
        audio: np.ndarray,
    ) -> tuple[str, float]:
        """
        Detect language from initial audio.

        Returns:
            Tuple of (language_code, confidence)
        """
        return ("en", 0.95)  # Default to English

    async def detect_speakers(
        self,
        audio: np.ndarray,
    ) -> int:
        """
        Detect number of speakers in audio.

        Returns:
            Number of detected speakers (1 = single speaker)
        """
        return 1  # Default to single speaker


class MockASRPipeline(ASRPipeline):
    """
    Mock ASR pipeline for testing.

    Returns dummy responses with realistic structure.
    """

    def __init__(self):
        self._word_counter = 0
        self._mock_words = [
            "hello", "world", "this", "is", "a", "test",
            "of", "the", "voice", "server", "streaming", "system",
        ]

    async def process_chunk(
        self,
        audio: np.ndarray,
        session: ClientSession,
        mode: ASRMode,
        config: ServerConfig,
    ) -> RichToken | None:
        """Return mock partial result."""
        # Simulate ~50ms processing time
        await asyncio.sleep(0.05)

        # Generate a mock word every ~320ms of audio
        duration_ms = len(audio) / config.sample_rate * 1000
        if duration_ms < 200:
            return None

        self._word_counter += 1
        word = self._mock_words[self._word_counter % len(self._mock_words)]

        return create_partial_token(
            text=word,
            start_ms=session.audio_start_ms,
            end_ms=session.audio_start_ms + duration_ms,
            chunk_index=session.chunk_index,
            utterance_id=session.utterance_id,
        )

    async def finalize_utterance(
        self,
        session: ClientSession,
        mode: ASRMode,
        config: ServerConfig,
    ) -> RichToken:
        """Return mock final result with all features."""
        # Simulate processing time
        await asyncio.sleep(0.1)

        # Build mock result
        text = " ".join(self._mock_words[:6])
        duration_ms = session.metrics.audio_received_ms

        word_timestamps = []
        current_ms = session.audio_start_ms
        for word in text.split():
            word_len_ms = 200.0  # Mock 200ms per word
            word_timestamps.append(WordTimestamp(
                word=word,
                start_ms=current_ms,
                end_ms=current_ms + word_len_ms,
                confidence=0.95,
            ))
            current_ms += word_len_ms + 50  # 50ms gap

        token = create_final_token(
            text=text,
            start_ms=session.audio_start_ms,
            end_ms=session.audio_start_ms + duration_ms,
            word_timestamps=word_timestamps,
            utterance_id=session.utterance_id,
            mode=mode,
        )

        # Add mock rich features
        if config.enable_emotion:
            token.emotion = EmotionLabel.NEUTRAL
            token.emotion_confidence = 0.85

        if config.enable_pitch:
            token.pitch = PitchInfo(
                mean_hz=150.0,
                std_hz=25.0,
                min_hz=100.0,
                max_hz=200.0,
                voiced_ratio=0.7,
            )

        if config.enable_phoneme:
            token.phonemes = PhonemeInfo(
                phonemes=["h", "ɛ", "l", "oʊ", "w", "ɝ", "l", "d"],
                confidences=[0.9] * 8,
            )

        if config.enable_language:
            token.language = LanguageInfo(
                language="en",
                confidence=0.98,
            )

        if config.enable_hallucination:
            token.hallucination = HallucinationInfo(
                score=0.05,
                is_hallucinated=False,
                phoneme_mismatch=0.03,
                energy_silence=0.02,
            )

        if config.enable_speaker:
            token.speaker = SpeakerInfo(
                embedding=[0.1] * 256,  # Mock 256-dim embedding
            )

        return token


class VoiceServer:
    """
    SOTA++ Voice Server.

    Handles WebSocket connections from clients and streams
    RichToken responses with ASR and rich audio features.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        pipeline: ASRPipeline | None = None,
    ):
        self.config = config or ServerConfig()
        self.pipeline = pipeline or MockASRPipeline()
        self.state = ServerState.STOPPED

        # Active sessions
        self._sessions: dict[str, ClientSession] = {}
        self._server: Any | None = None

        # Callbacks
        self._on_connect: Callable | None = None
        self._on_disconnect: Callable | None = None

    @property
    def num_connections(self) -> int:
        """Number of active connections."""
        return len(self._sessions)

    def on_connect(self, callback: Callable[[ClientSession], None]):
        """Register callback for client connection."""
        self._on_connect = callback

    def on_disconnect(self, callback: Callable[[ClientSession], None]):
        """Register callback for client disconnection."""
        self._on_disconnect = callback

    async def start(self):
        """Start the voice server."""
        if not HAS_WEBSOCKETS:
            raise RuntimeError("websockets package required. Install with: pip install websockets")

        if self.state != ServerState.STOPPED:
            raise RuntimeError(f"Server in invalid state: {self.state}")

        self.state = ServerState.STARTING
        logger.info(f"Starting voice server on {self.config.host}:{self.config.port}")

        self._server = await websockets.serve(
            self._handle_client,
            self.config.host,
            self.config.port,
            ping_interval=self.config.websocket_ping_interval,
            ping_timeout=self.config.websocket_ping_timeout,
        )

        self.state = ServerState.RUNNING
        logger.info("Voice server started")

    async def stop(self):
        """Stop the voice server."""
        if self.state != ServerState.RUNNING:
            return

        self.state = ServerState.STOPPING
        logger.info("Stopping voice server...")

        # Close all sessions
        for session in list(self._sessions.values()):
            await self._close_session(session)

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        self.state = ServerState.STOPPED
        logger.info("Voice server stopped")

    async def run_forever(self):
        """Run server until interrupted."""
        await self.start()
        try:
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a WebSocket client connection."""
        session = await self._create_session(websocket)

        try:
            if self._on_connect:
                self._on_connect(session)

            await self._process_messages(session)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {session.session_id} disconnected")

        except Exception as e:
            logger.error(f"Error handling client {session.session_id}: {e}")
            await self._send_error(session, str(e))

        finally:
            await self._close_session(session)
            if self._on_disconnect:
                self._on_disconnect(session)

    async def _create_session(self, websocket: WebSocketServerProtocol) -> ClientSession:
        """Create a new client session."""
        session_id = str(uuid.uuid4())[:8]
        session = ClientSession(
            session_id=session_id,
            websocket=websocket,
            mode=self.config.default_mode,
        )
        session.metrics.start_time = time.time()

        self._sessions[session_id] = session
        logger.info(f"Client {session_id} connected")

        # Send connection acknowledgment
        await self._send_metadata(session, {
            "status": "connected",
            "session_id": session_id,
            "config": {
                "sample_rate": self.config.sample_rate,
                "chunk_size_ms": self.config.chunk_size_ms,
                "default_mode": self.config.default_mode.value,
            },
        })

        return session

    async def _close_session(self, session: ClientSession):
        """Close a client session."""
        session.state = SessionState.DISCONNECTED
        self._sessions.pop(session.session_id, None)
        try:
            await session.websocket.close()
        except Exception:
            pass

    async def _process_messages(self, session: ClientSession):
        """Process incoming messages from a client."""
        async for message in session.websocket:
            if isinstance(message, bytes):
                # Binary message = audio data
                await self._handle_audio(session, message)
            else:
                # Text message = JSON control message
                await self._handle_control(session, message)

    async def _handle_audio(self, session: ClientSession, data: bytes):
        """Handle incoming audio data."""
        session.state = SessionState.STREAMING

        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Track metrics
        duration_ms = len(audio) / self.config.sample_rate * 1000
        session.metrics.audio_received_ms += duration_ms
        session.metrics.chunks_received += 1

        # Start new utterance if needed
        if not session.utterance_id:
            session.utterance_id = str(uuid.uuid4())[:8]
            session.audio_start_ms = time.time() * 1000
            session.chunk_index = 0

        # Add to buffer
        session.audio_buffer.append(audio)
        session.chunk_index += 1

        # Process chunk
        start_time = time.time()
        result = await self.pipeline.process_chunk(
            audio,
            session,
            session.mode,
            self.config,
        )
        inference_time = (time.time() - start_time) * 1000
        session.metrics.inference_time_ms += inference_time

        # Send partial result
        if result:
            await self._send_token(session, result, is_final=False)

    async def _handle_control(self, session: ClientSession, message: str):
        """Handle control message (JSON)."""
        try:
            data = json.loads(message)
            command = data.get("command")

            if command == "start":
                # Start new utterance
                session.utterance_id = str(uuid.uuid4())[:8]
                session.audio_buffer = []
                session.chunk_index = 0
                session.audio_start_ms = time.time() * 1000
                mode_str = data.get("mode", self.config.default_mode.value)
                session.mode = ASRMode(mode_str)
                await self._send_metadata(session, {
                    "status": "started",
                    "utterance_id": session.utterance_id,
                    "mode": session.mode.value,
                })

            elif command == "end":
                # End utterance and get final result
                session.state = SessionState.PROCESSING
                result = await self.pipeline.finalize_utterance(
                    session,
                    session.mode,
                    self.config,
                )
                await self._send_token(session, result, is_final=True)

                # Reset for next utterance
                session.utterance_id = ""
                session.audio_buffer = []
                session.chunk_index = 0
                session.state = SessionState.CONNECTED

            elif command == "cancel":
                # Cancel current utterance
                session.utterance_id = ""
                session.audio_buffer = []
                session.chunk_index = 0
                session.state = SessionState.CONNECTED
                await self._send_metadata(session, {"status": "cancelled"})

            elif command == "config":
                # Update session config
                if "mode" in data:
                    session.mode = ASRMode(data["mode"])
                await self._send_metadata(session, {
                    "status": "configured",
                    "mode": session.mode.value,
                })

            elif command == "ping":
                # Health check
                await self._send_metadata(session, {
                    "status": "pong",
                    "timestamp": time.time(),
                })

            else:
                await self._send_error(session, f"Unknown command: {command}")

        except json.JSONDecodeError:
            await self._send_error(session, "Invalid JSON")
        except Exception as e:
            await self._send_error(session, str(e))

    async def _send_token(
        self,
        session: ClientSession,
        token: RichToken,
        is_final: bool,
    ):
        """Send a RichToken to the client."""
        session.sequence += 1
        session.metrics.tokens_sent += 1

        response = StreamingResponse(
            type="final" if is_final else "partial",
            token=token,
            sequence=session.sequence,
        )

        if self.config.log_latency and is_final:
            total_time = time.time() - session.metrics.start_time
            response.metadata = {
                "total_time_ms": total_time * 1000,
                "inference_time_ms": session.metrics.inference_time_ms,
                "chunks_received": session.metrics.chunks_received,
            }

        await session.websocket.send(response.to_json())

    async def _send_metadata(self, session: ClientSession, metadata: dict):
        """Send metadata to the client."""
        session.sequence += 1
        response = StreamingResponse(
            type="metadata",
            metadata=metadata,
            sequence=session.sequence,
        )
        await session.websocket.send(response.to_json())

    async def _send_error(self, session: ClientSession, error: str):
        """Send error to the client."""
        session.sequence += 1
        response = StreamingResponse(
            type="error",
            error=error,
            sequence=session.sequence,
        )
        await session.websocket.send(response.to_json())


async def main():
    """Run the voice server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = ServerConfig(
        host="localhost",
        port=8765,
    )

    server = VoiceServer(config)
    await server.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
