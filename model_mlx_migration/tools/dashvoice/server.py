#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
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
DashVoice API Server

FastAPI-based server for the DashVoice voice processing pipeline.
Provides endpoints for:
- Audio processing (VAD, echo cancellation, speaker diarization)
- Speech-to-text transcription (Whisper STT)
- Text-to-speech synthesis (Kokoro/CosyVoice)
- Speaker registration and identification
- WebSocket streaming for real-time audio processing

Usage:
    # Start server
    python -m tools.dashvoice.server

    # Or with uvicorn directly
    uvicorn tools.dashvoice.server:app --host 0.0.0.0 --port 8000

    # Test endpoints
    curl -X POST http://localhost:8000/process -F "audio=@audio.wav"

    # WebSocket streaming (JavaScript example):
    ws = new WebSocket("ws://localhost:8000/ws/stream");
    ws.onopen = () => {
        ws.send(JSON.stringify({type: "config", sample_rate: 16000}));
    };
    ws.onmessage = (event) => console.log(JSON.parse(event.data));
    // Send audio chunks as binary data
"""

import asyncio
import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tools.dashvoice.pipeline import DashVoicePipeline
from tools.dashvoice.voice_database import VoiceDatabase

# Import streaming transcription (optional - gracefully handle if not available)
try:
    from tools.whisper_mlx import (  # noqa: F401
        StreamingConfig,
        StreamingResult,
        StreamingWhisper,
        WhisperMLX,
    )
    HAS_STREAMING_WHISPER = True
except ImportError:
    HAS_STREAMING_WHISPER = False
from tools.dashvoice.metrics import (
    instrument_app,
    record_model_loaded,
    record_model_warmup,
    record_pipeline_output,
    record_prewarm_complete,
    record_stt_output,
    record_tts_output,
    record_websocket_audio,
    record_websocket_message,
    semaphore_acquired,
    semaphore_released,
    track_pipeline_processing,
    track_semaphore_wait,
    track_stt_transcription,
    track_tts_generation,
    websocket_session_closed,
    websocket_session_opened,
)

# =============================================================================
# Concurrency Configuration
# =============================================================================

# Maximum concurrent TTS requests (default: 2 for MLX memory efficiency)
MAX_TTS_CONCURRENT = int(os.getenv("DASHVOICE_MAX_TTS_CONCURRENT", "2"))

# Maximum concurrent pipeline requests (default: 4)
MAX_PIPELINE_CONCURRENT = int(os.getenv("DASHVOICE_MAX_PIPELINE_CONCURRENT", "4"))

# Semaphores for concurrency control (allows multiple concurrent requests)
_tts_semaphore = asyncio.Semaphore(MAX_TTS_CONCURRENT)
_pipeline_semaphore = asyncio.Semaphore(MAX_PIPELINE_CONCURRENT)


@asynccontextmanager
async def tracked_semaphore(semaphore: asyncio.Semaphore, semaphore_type: str):
    """Context manager for semaphore acquisition with metrics tracking.

    Tracks:
    - Time spent waiting for the semaphore
    - Number of active requests holding the semaphore
    """
    with track_semaphore_wait(semaphore_type):
        await semaphore.acquire()

    semaphore_acquired(semaphore_type)
    try:
        yield
    finally:
        semaphore.release()
        semaphore_released(semaphore_type)


# =============================================================================
# Streaming WhisperMLX Model Management
# =============================================================================

_streaming_whisper_model = None


def get_streaming_whisper_model():
    """Get or create the global streaming WhisperMLX model instance."""
    global _streaming_whisper_model
    if _streaming_whisper_model is None:
        if not HAS_STREAMING_WHISPER:
            print("Warning: WhisperMLX streaming not available. Using fallback.")
            _streaming_whisper_model = "disabled"
            return _streaming_whisper_model

        try:
            print("Loading WhisperMLX for streaming transcription...")
            _streaming_whisper_model = WhisperMLX.from_pretrained("large-v3-turbo")
            print("  WhisperMLX loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load WhisperMLX: {e}")
            _streaming_whisper_model = "disabled"

    return _streaming_whisper_model


# =============================================================================
# Wake Word Model Management
# =============================================================================

_wakeword_detector = None


def get_wakeword_detector():
    """Get or create the global wake word detector instance."""
    global _wakeword_detector
    if _wakeword_detector is None:
        try:
            from scripts.wakeword_training.inference import WakeWordDetector
            model_path = "models/wakeword/hey_agent/hey_agent_mlx.safetensors"
            if os.path.exists(model_path):
                _wakeword_detector = WakeWordDetector.from_mlx(model_path, use_vad=True)
                print(f"Loaded wake word detector from {model_path}")
            else:
                print(f"Warning: Wake word model not found at {model_path}")
                _wakeword_detector = "disabled"
        except ImportError as e:
            print(f"Warning: Could not import wake word module: {e}")
            _wakeword_detector = "disabled"
        except Exception as e:
            print(f"Warning: Could not load wake word model: {e}")
            _wakeword_detector = "disabled"
    return _wakeword_detector


# =============================================================================
# TTS Model Management
# =============================================================================

_tts_model = None


def get_tts_model():
    """Get or create the global TTS model instance."""
    global _tts_model
    if _tts_model is None:
        try:
            from mlx_audio.tts.utils import load_model
            _tts_model = load_model("prince-canuma/Kokoro-82M")
            print("Loaded Kokoro TTS model")
        except ImportError:
            print("Warning: mlx_audio not installed. TTS disabled.")
            _tts_model = "disabled"
        except Exception as e:
            print(f"Warning: Could not load TTS model: {e}")
            _tts_model = "disabled"
    return _tts_model


# =============================================================================
# Emotional TTS Model Management (Prosody Trifecta)
# =============================================================================

_emotional_tts = None

# Available emotions with prosody type IDs (matching prosody_types.h)
EMOTION_PROSODY_IDS = {
    "neutral": 0,
    "angry": 40,
    "sad": 41,
    "excited": 42,
    "calm": 45,
    "frustrated": 48,
    "nervous": 49,
    "surprised": 50,
}

# Emotional TTS voice packs (subset of Kokoro voices that sound good with prosody)
EMOTIONAL_VOICE_PACKS = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
]


def get_emotional_tts():
    """Get or create the global emotional TTS model instance.

    This loads our custom KokoroModel with prosody trifecta support:
    - F0 Contour v2.4 (pitch modification)
    - Duration v3 (speaking rate)
    - Energy v3 (volume/loudness)
    """
    global _emotional_tts
    if _emotional_tts is None:
        try:
            from pathlib import Path

            import mlx.core as mx

            from tools.pytorch_to_mlx.converters import KokoroConverter

            print("Loading emotional TTS (KokoroModel with prosody)...")

            # Load base model
            converter = KokoroConverter()
            model, config, _ = converter.load_from_hf()
            model.set_deterministic(True)

            # Load prosody contour v2.4
            contour_weights = Path("models/prosody_contour_v2.4/best_model.npz")
            emb_dir = "models/prosody_embeddings_orthogonal"
            embedding_path = Path(f"{emb_dir}/final.safetensors")

            if contour_weights.exists() and embedding_path.exists():
                model.enable_prosody_contour_v2()
                model.load_prosody_contour_v2_weights(contour_weights, embedding_path)
                print(f"  Loaded F0 contour v2.4 from {contour_weights}")

                # Load prosody duration/energy v3
                de_path = "models/prosody_duration_energy_v3/best_model.npz"
                duration_energy_weights = Path(de_path)
                if duration_energy_weights.exists():
                    model.enable_prosody_duration_energy()
                    model.load_prosody_duration_energy_weights(
                        duration_energy_weights, embedding_path,
                    )
                    print(f"  Loaded duration/energy v3 from {duration_energy_weights}")
                else:
                    print(
                        f"  Warning: Duration/energy weights not found: "
                        f"{duration_energy_weights}",
                    )

                # Pre-load voice packs
                voice_packs = {}
                for voice_name in EMOTIONAL_VOICE_PACKS:
                    try:
                        voice_pack = converter.load_voice_pack(voice_name)
                        mx.eval(voice_pack)
                        voice_packs[voice_name] = voice_pack
                    except Exception as e:
                        print(f"  Warning: Could not load voice pack {voice_name}: {e}")

                _emotional_tts = {
                    "model": model,
                    "converter": converter,
                    "voice_packs": voice_packs,
                    "sample_rate": 24000,
                }
                vp_names = list(voice_packs.keys())
                print(f"  Loaded {len(voice_packs)} voice packs: {vp_names}")
                print("Emotional TTS ready (prosody trifecta enabled)")
            else:
                print("Warning: Prosody weights not found. Emotional TTS disabled.")
                print(f"  Expected: {contour_weights}, {embedding_path}")
                _emotional_tts = "disabled"

        except ImportError as e:
            print(f"Warning: Could not import emotional TTS modules: {e}")
            _emotional_tts = "disabled"
        except Exception as e:
            print(f"Warning: Could not load emotional TTS model: {e}")
            import traceback
            traceback.print_exc()
            _emotional_tts = "disabled"

    return _emotional_tts


async def prewarm_models():
    """Pre-warm all models to eliminate cold start latency.

    This function is called during server startup to:
    1. Load and warm the TTS model (Kokoro) - major cold start (~1100ms)
    2. Load and warm the pipeline components (VAD, STT, etc.)

    After pre-warming, first user request will be fast.

    Environment variables:
    - DASHVOICE_PREWARM_ALL_VOICES: Set to "1" to warm all 10 Kokoro voices
      (takes ~30s longer but eliminates voice switching latency)
    """
    import numpy as np

    print("Pre-warming models...")
    start_time = time.time()

    # Check if we should warm all voices
    warm_all_voices = os.getenv("DASHVOICE_PREWARM_ALL_VOICES", "0") == "1"

    # Pre-warm TTS (Kokoro) - this is the main cold start bottleneck
    print("  Pre-warming TTS (Kokoro)...")
    tts_start = time.time()
    tts_model = get_tts_model()
    tts_loaded = tts_model != "disabled"
    if tts_loaded:
        try:
            # Voices to warm up
            voices_to_warm = KOKORO_VOICES if warm_all_voices else ["af_bella"]
            print(f"    Warming {len(voices_to_warm)} voice(s)...")

            for i, voice in enumerate(voices_to_warm):
                voice_start = time.time()
                # Generate dummy audio to trigger MLX graph compilation
                for _result in tts_model.generate(
                    text="Hello.",
                    voice=voice,
                    speed=1.0,
                    verbose=False,
                ):
                    pass  # Just iterate to trigger generation
                voice_time = time.time() - voice_start
                if warm_all_voices:
                    idx = f"[{i+1}/{len(voices_to_warm)}]"
                    print(f"      {idx} {voice}: {voice_time:.2f}s")

            print(f"    TTS warm-up complete ({len(voices_to_warm)} voices)")
        except Exception as e:
            print(f"    TTS warm-up failed: {e}")
            tts_loaded = False
    record_model_loaded("tts", tts_loaded)
    record_model_warmup("tts", time.time() - tts_start)

    # Pre-warm pipeline components
    print("  Pre-warming pipeline components...")
    pipeline_start = time.time()
    pipeline = get_pipeline()

    # Create dummy audio (1 second of silence with some noise)
    rng = np.random.default_rng()
    dummy_audio = rng.standard_normal(16000).astype(np.float32) * 0.01

    # Run through pipeline to warm up VAD, etc.
    vad_loaded = False
    try:
        _ = pipeline.process(dummy_audio, sample_rate=16000)
        print("    Pipeline warm-up complete")
        vad_loaded = True
    except Exception as e:
        print(f"    Pipeline warm-up failed: {e}")
    record_model_loaded("vad", vad_loaded)
    record_model_warmup("vad", time.time() - pipeline_start)

    # Pre-warm Whisper STT if available
    print("  Pre-warming STT (Whisper)...")
    stt_start = time.time()
    stt_loaded = False
    try:
        stt = pipeline._get_stt()
        stt.transcribe(dummy_audio, sample_rate=16000)
        print("    STT warm-up complete")
        stt_loaded = True
    except Exception as e:
        print(f"    STT warm-up failed: {e}")
    record_model_loaded("stt", stt_loaded)
    record_model_warmup("stt", time.time() - stt_start)

    # Pre-warm Emotional TTS (prosody trifecta)
    print("  Pre-warming Emotional TTS (prosody trifecta)...")
    emotional_start = time.time()
    emotional_loaded = False
    try:
        emotional_tts = get_emotional_tts()
        if emotional_tts != "disabled":
            # Generate a short sample to warm up the model
            import mlx.core as mx

            from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                phonemize_text,
            )

            model = emotional_tts["model"]
            converter = emotional_tts["converter"]
            voice_packs = emotional_tts["voice_packs"]

            if voice_packs:
                voice_name = list(voice_packs.keys())[0]
                voice_pack = voice_packs[voice_name]

                # Phonemize test text
                phonemes, tokens = phonemize_text("Test.")
                tokens_mx = mx.array([tokens])

                # Get voice embedding
                voice = converter.select_voice_embedding(voice_pack, len(tokens))

                # Create neutral prosody mask
                prosody_mask = mx.array([[0] * len(tokens)], dtype=mx.int32)

                # Generate audio (triggers MLX graph compilation)
                audio = model(tokens_mx, voice, prosody_mask=prosody_mask)
                mx.eval(audio)

                emotional_loaded = True
                print(f"    Emotional TTS warm-up complete ({len(voice_packs)} voices)")
        else:
            print("    Emotional TTS disabled (missing prosody weights)")
    except Exception as e:
        print(f"    Emotional TTS warm-up failed: {e}")
    record_model_loaded("emotional_tts", emotional_loaded)
    record_model_warmup("emotional_tts", time.time() - emotional_start)

    elapsed = time.time() - start_time
    print(f"Model pre-warming complete in {elapsed:.1f}s")
    record_prewarm_complete()


class ProcessRequest(BaseModel):
    """Request for audio processing."""
    enable_stt: bool = False
    enable_diarization: bool = True
    enable_echo_cancel: bool = True
    enable_voice_fingerprint: bool = True
    enable_noise_reduction: bool = False


class ProcessResponse(BaseModel):
    """Response from audio processing."""
    processing_time_ms: float
    num_segments: int
    num_speakers: int
    segments: list[dict]


class TranscribeResponse(BaseModel):
    """Response from STT transcription."""
    text: str
    language: str | None
    processing_time_ms: float


class SpeakerRegisterRequest(BaseModel):
    """Request to register a speaker."""
    name: str


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app.

    Replaces deprecated @app.on_event("startup") pattern.
    See: https://fastapi.tiangolo.com/advanced/events/
    """
    # Startup: instrument app and pre-warm models
    instrument_app(app)
    await prewarm_models()

    yield  # Server runs here

    # Shutdown: cleanup if needed (none currently)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DashVoice API",
    description="Multi-speaker voice processing pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy-loaded)
_pipeline: DashVoicePipeline | None = None


def get_pipeline() -> DashVoicePipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DashVoicePipeline(
            enable_echo_cancel=True,
            enable_voice_fingerprint=True,
            enable_diarization=True,
            enable_stt=False,  # Enable per-request
            enable_vad=True,
        )
    return _pipeline


def load_audio_from_upload(file: UploadFile) -> tuple[np.ndarray, int]:
    """Load audio from uploaded file."""
    contents = file.file.read()
    audio, sample_rate = sf.read(io.BytesIO(contents))

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32), sample_rate


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "DashVoice API",
        "version": "1.6.0",
        "endpoints": {
            "rest": [
                "POST /process - Process audio through pipeline",
                "POST /transcribe - Speech-to-text transcription",
                "POST /synthesize - Text-to-speech synthesis (Kokoro TTS)",
                "POST /synthesize_emotional - Emotional TTS with prosody",
                "POST /register_speaker - Register a known speaker",
                "POST /separate - Separate overlapping speakers",
                "POST /denoise - Noise reduction using DeepFilterNet3",
                "GET /voices - List available TTS voices and emotions",
                "GET /models - List available models",
                "GET /concurrency - Show concurrency configuration",
                "GET /health - Health check",
            ],
            "wake_word": [
                "POST /wakeword/detect - Detect wake word in audio",
                "GET /wakeword/info - Get wake word model info",
                "WS /ws/wakeword - Streaming wake word detection",
            ],
            "voice_cloning": [
                "POST /clone/extract - Extract speaker embedding from audio",
                "POST /clone/synthesize - Synthesize with cloned voice (CosyVoice2)",
                "GET /clone/voices - List registered cloned voices",
                "DELETE /clone/voices/{name} - Delete a custom cloned voice",
            ],
            "websocket": [
                "WS /ws/stream - Real-time audio streaming with full pipeline",
                "WS /ws/transcribe - Real-time transcription only",
                "WS /ws/synthesize - Streaming TTS synthesis (low TTFA)",
                "WS /ws/synthesize_emotional - Streaming emotional TTS",
                "WS /ws/wakeword - Streaming wake word detection",
                "GET /ws/sessions - List active streaming sessions",
            ],
        },
        "pre_warming": "enabled",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    audio: UploadFile = File(..., description="Audio file to process"),
    enable_stt: bool = Form(default=False, description="Enable speech-to-text"),
    enable_diarization: bool = Form(default=True, description="Enable diarization"),
    enable_echo_cancel: bool = Form(default=True, description="Enable echo cancel"),
    enable_voice_fingerprint: bool = Form(default=True, description="Fingerprint"),
    enable_noise_reduction: bool = Form(
        default=False, description="Enable DeepFilterNet3 noise reduction",
    ),
):
    """
    Process audio through the DashVoice pipeline.

    Returns detected speech segments with speaker identification,
    echo cancellation status, and optional transcription.
    """
    try:
        # Load audio
        audio_data, sample_rate = load_audio_from_upload(audio)

        # Get pipeline and configure
        pipeline = get_pipeline()
        pipeline.enable_echo_cancel = enable_echo_cancel
        pipeline.enable_voice_fingerprint = enable_voice_fingerprint
        pipeline.enable_diarization = enable_diarization
        pipeline.enable_stt = enable_stt
        pipeline.enable_noise_reduction = enable_noise_reduction

        # Build features string for metrics
        features = []
        if enable_echo_cancel:
            features.append("echo")
        if enable_voice_fingerprint:
            features.append("voice_fp")
        if enable_diarization:
            features.append("diarization")
        if enable_stt:
            features.append("stt")
        if enable_noise_reduction:
            features.append("denoise")
        features_str = "+".join(features) if features else "basic"

        # Process audio with metrics tracking (semaphore allows concurrent requests)
        async with tracked_semaphore(_pipeline_semaphore, "pipeline"):
            with track_pipeline_processing(features_str):
                result = pipeline.process(audio_data, sample_rate)

        # Record pipeline metrics
        dashvoice_voice = None
        for seg in result.segments:
            if seg.is_dashvoice and seg.dashvoice_voice:
                dashvoice_voice = seg.dashvoice_voice
                break
        record_pipeline_output(result.num_speakers_detected, dashvoice_voice)

        # Convert segments to dicts
        segments = []
        for seg in result.segments:
            seg_dict = {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "speaker": seg.speaker,
                "is_dashvoice": seg.is_dashvoice,
                "dashvoice_voice": seg.dashvoice_voice,
                "confidence": seg.confidence,
            }
            if enable_stt:
                seg_dict["transcription"] = seg.transcription
                seg_dict["language"] = seg.language
            segments.append(seg_dict)

        return ProcessResponse(
            processing_time_ms=result.processing_time_ms,
            num_segments=len(result.segments),
            num_speakers=result.num_speakers_detected,
            segments=segments,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
):
    """
    Transcribe audio to text using Whisper STT.
    """
    try:
        # Load audio
        audio_data, sample_rate = load_audio_from_upload(audio)

        # Get pipeline
        pipeline = get_pipeline()
        stt = pipeline._get_stt()

        # Transcribe with metrics tracking
        audio_duration_s = len(audio_data) / sample_rate
        with track_stt_transcription():
            start = time.perf_counter()
            text, language = stt.transcribe(audio_data, sample_rate)
            processing_time = (time.perf_counter() - start) * 1000

        # Record STT metrics
        record_stt_output(audio_duration_s, text or "")

        return TranscribeResponse(
            text=text,
            language=language,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/register_speaker")
async def register_speaker(
    audio: UploadFile = File(..., description="Reference audio for speaker"),
    name: str = Form(..., description="Speaker name/ID"),
):
    """
    Register a known speaker for identification.

    Upload a reference audio sample and provide a name.
    The speaker will be identified in future processing.
    """
    try:
        # Load audio
        audio_data, sample_rate = load_audio_from_upload(audio)

        # Register speaker
        pipeline = get_pipeline()
        pipeline.register_speaker(name, audio_data, sample_rate)

        msg = f"Speaker '{name}' registered"
        return {"status": "success", "speaker": name, "message": msg}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/separate")
async def separate_audio(
    audio: UploadFile = File(..., description="Audio with overlapping speech"),
    num_speakers: int = Form(default=2, description="Expected number of speakers"),
):
    """
    Separate overlapping speech into individual speaker tracks.

    Returns multiple audio files, one per detected speaker.
    """
    try:
        # Load audio
        audio_data, sample_rate = load_audio_from_upload(audio)

        # Separate
        pipeline = get_pipeline()
        sources = pipeline.separate_audio(audio_data, sample_rate, num_speakers)

        # Create response with audio data
        result = {
            "num_sources": len(sources),
            "sources": [],
        }

        for i, source in enumerate(sources):
            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, source, sample_rate, format='WAV')
            buffer.seek(0)

            # Add as base64 or just info
            result["sources"].append({
                "index": i,
                "samples": len(source),
                "duration_s": len(source) / sample_rate,
                "max_amplitude": float(np.abs(source).max()),
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/denoise")
async def denoise_audio(
    audio: UploadFile = File(..., description="Audio to denoise"),
):
    """
    Reduce noise in audio using DeepFilterNet3.

    Returns denoised audio as WAV data (base64 encoded).
    """
    try:
        # Load audio
        audio_data, sample_rate = load_audio_from_upload(audio)

        # Denoise
        pipeline = get_pipeline()
        denoised = pipeline.reduce_noise(audio_data, sample_rate)

        # Create response with audio data
        buffer = io.BytesIO()
        sf.write(buffer, denoised, sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.getvalue()

        return {
            "original_samples": len(audio_data),
            "denoised_samples": len(denoised),
            "sample_rate": sample_rate,
            "duration_s": len(denoised) / sample_rate,
            "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# TTS Endpoints
# =============================================================================

# Available Kokoro voices
KOKORO_VOICES = [
    "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]


class SynthesizeRequest(BaseModel):
    """Request for TTS synthesis."""
    text: str
    voice: str = "af_bella"
    speed: float = 1.0


@app.get("/voices")
async def list_voices():
    """List available TTS voices and emotional TTS info."""
    tts_model = get_tts_model()
    enabled = tts_model != "disabled"

    # Check emotional TTS
    emotional_tts = get_emotional_tts()
    emotional_enabled = emotional_tts != "disabled"

    return {
        "tts_enabled": enabled,
        "model": "prince-canuma/Kokoro-82M" if enabled else None,
        "voices": KOKORO_VOICES if enabled else [],
        "default_voice": "af_bella",
        "voice_categories": {
            "american_female": ["af_bella", "af_nicole", "af_sarah", "af_sky"],
            "american_male": ["am_adam", "am_michael"],
            "british_female": ["bf_emma", "bf_isabella"],
            "british_male": ["bm_george", "bm_lewis"],
        },
        "emotional_tts": {
            "enabled": emotional_enabled,
            "voices": (
                list(emotional_tts["voice_packs"].keys()) if emotional_enabled else []
            ),
            "default_voice": "af_heart",
            "emotions": list(EMOTION_PROSODY_IDS.keys()) if emotional_enabled else [],
            "default_emotion": "neutral",
            "features": (
                ["F0 contour v2.4", "Duration v3", "Energy v3"]
                if emotional_enabled else []
            ),
            "endpoint": "/synthesize_emotional",
        },
    }


@app.post("/synthesize")
async def synthesize_audio(
    text: str = Form(..., description="Text to synthesize"),
    voice: str = Form(default="af_bella", description="Voice to use"),
    speed: float = Form(default=1.0, description="Speech speed (0.5-2.0)"),
):
    """
    Synthesize speech from text using Kokoro TTS.

    Returns audio as WAV data (base64 encoded).

    Example usage:
        curl -X POST http://localhost:8000/synthesize \\
            -F "text=Hello, world!" \\
            -F "voice=af_bella" \\
            -F "speed=1.0"
    """
    try:
        tts_model = get_tts_model()
        if tts_model == "disabled":
            raise HTTPException(
                status_code=503,
                detail="TTS is not available. Install mlx_audio package.",
            )

        # Validate voice
        if voice not in KOKORO_VOICES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice '{voice}'. Available: {KOKORO_VOICES}",
            )

        # Validate speed
        if not 0.5 <= speed <= 2.0:
            raise HTTPException(
                status_code=400,
                detail="Speed must be between 0.5 and 2.0",
            )

        # Generate audio with metrics tracking (semaphore allows concurrent requests)
        start_time = time.perf_counter()

        async with tracked_semaphore(_tts_semaphore, "tts"):
            with track_tts_generation(voice=voice):
                result = None
                for r in tts_model.generate(
                    text=text,
                    voice=voice,
                    speed=speed,
                    verbose=False,
                ):
                    result = r

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="TTS generation failed - no output",
            )

        generation_time_s = time.perf_counter() - start_time
        generation_time_ms = generation_time_s * 1000

        # Convert to numpy array
        audio = np.array(result.audio)
        sample_rate = result.sample_rate
        duration_s = len(audio) / sample_rate

        # Record TTS metrics
        record_tts_output(voice, len(audio), sample_rate, generation_time_s)

        # Create WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.getvalue()

        return {
            "text": text,
            "voice": voice,
            "speed": speed,
            "sample_rate": sample_rate,
            "samples": len(audio),
            "duration_s": duration_s,
            "generation_time_ms": generation_time_ms,
            "rtf": (generation_time_ms / 1000) / duration_s,
            "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/synthesize_emotional")
async def synthesize_emotional_audio(
    text: str = Form(..., description="Text to synthesize"),
    voice: str = Form(default="af_heart", description="Voice to use"),
    emotion: str = Form(
        default="neutral",
        description="Emotion (neutral/angry/sad/excited/calm/frustrated/nervous)",
    ),
    speed: float = Form(default=1.0, description="Speech speed (0.5-2.0)"),
):
    """
    Synthesize speech with emotional prosody using the Prosody Trifecta.

    This endpoint uses our custom KokoroModel with:
    - F0 Contour v2.4: Modifies pitch patterns based on emotion
    - Duration v3: Adjusts speaking rate (angry=faster, sad=slower)
    - Energy v3: Adjusts volume/loudness (angry=louder, sad=quieter)

    Available emotions:
    - neutral: Baseline prosody
    - angry: Faster, louder, higher pitch
    - sad: Slower, quieter, lower pitch
    - excited: Faster, louder, higher pitch
    - calm: Similar to neutral (limited training data)
    - frustrated: Slightly slower, louder
    - nervous: Faster, louder
    - surprised: Faster, louder, higher pitch

    Example usage:
        curl -X POST http://localhost:8000/synthesize_emotional \\
            -F "text=Hello, how are you doing today?" \\
            -F "voice=af_heart" \\
            -F "emotion=angry" \\
            -F "speed=1.0"
    """
    try:
        emotional_tts = get_emotional_tts()
        if emotional_tts == "disabled":
            raise HTTPException(
                status_code=503,
                detail="Emotional TTS not available. Missing prosody weights in models/",
            )

        model = emotional_tts["model"]
        converter = emotional_tts["converter"]
        voice_packs = emotional_tts["voice_packs"]
        sample_rate = emotional_tts["sample_rate"]

        # Validate voice
        if voice not in voice_packs:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice '{voice}'. Available: {list(voice_packs.keys())}",
            )

        # Validate emotion
        emotion_lower = emotion.lower()
        if emotion_lower not in EMOTION_PROSODY_IDS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid emotion '{emotion}'. "
                f"Available: {list(EMOTION_PROSODY_IDS.keys())}",
            )

        # Validate speed
        if not 0.5 <= speed <= 2.0:
            raise HTTPException(
                status_code=400,
                detail="Speed must be between 0.5 and 2.0",
            )

        # Import required modules
        import mlx.core as mx

        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text,
        )

        # Phonemize text
        phonemes, tokens = phonemize_text(text)

        # Get voice pack and embedding
        voice_pack = voice_packs[voice]
        voice_embedding = converter.select_voice_embedding(voice_pack, len(tokens))

        # Create prosody mask with emotion ID
        prosody_id = EMOTION_PROSODY_IDS[emotion_lower]
        prosody_mask = mx.array([[prosody_id] * len(tokens)], dtype=mx.int32)

        # Generate audio with metrics tracking
        start_time = time.perf_counter()

        async with tracked_semaphore(_tts_semaphore, "tts"):
            with track_tts_generation(voice=voice):
                tokens_mx = mx.array([tokens])
                audio = model(tokens_mx, voice_embedding, prosody_mask=prosody_mask)
                mx.eval(audio)

        generation_time_s = time.perf_counter() - start_time
        generation_time_ms = generation_time_s * 1000

        # Convert to numpy
        audio_np = np.array(audio).flatten()

        # Apply speed adjustment (simple resampling)
        if speed != 1.0:
            from scipy import signal
            new_length = int(len(audio_np) / speed)
            audio_np = signal.resample(audio_np, new_length).astype(np.float32)

        duration_s = len(audio_np) / sample_rate

        # Record TTS metrics
        voice_emotion = f"{voice}_{emotion_lower}"
        record_tts_output(voice_emotion, len(audio_np), sample_rate, generation_time_s)

        # Create WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.getvalue()

        return {
            "text": text,
            "voice": voice,
            "emotion": emotion_lower,
            "prosody_id": prosody_id,
            "speed": speed,
            "sample_rate": sample_rate,
            "samples": len(audio_np),
            "duration_s": duration_s,
            "generation_time_ms": generation_time_ms,
            "rtf": (generation_time_ms / 1000) / duration_s if duration_s > 0 else 0,
            "phonemes": phonemes,
            "features": ["F0 contour v2.4", "Duration v3", "Energy v3"],
            "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/concurrency")
async def concurrency_info():
    """Show concurrency configuration and current status."""
    return {
        "max_concurrent": {
            "tts": MAX_TTS_CONCURRENT,
            "pipeline": MAX_PIPELINE_CONCURRENT,
        },
        "env_vars": {
            "DASHVOICE_MAX_TTS_CONCURRENT": os.getenv(
                "DASHVOICE_MAX_TTS_CONCURRENT", "2 (default)",
            ),
            "DASHVOICE_MAX_PIPELINE_CONCURRENT": os.getenv(
                "DASHVOICE_MAX_PIPELINE_CONCURRENT", "4 (default)",
            ),
            "DASHVOICE_PREWARM_ALL_VOICES": os.getenv(
                "DASHVOICE_PREWARM_ALL_VOICES", "0 (default)",
            ),
        },
        "description": {
            "tts": "Maximum concurrent TTS synthesis requests",
            "pipeline": "Maximum concurrent audio processing requests",
        },
        "note": "Increase limits for higher throughput, decrease for memory efficiency",
    }


@app.get("/models")
async def list_models():
    """List available models and their status."""
    pipeline = get_pipeline()
    tts_model = get_tts_model()
    emotional_tts = get_emotional_tts()

    return {
        "pipeline": {
            "echo_cancel": pipeline.enable_echo_cancel,
            "voice_fingerprint": pipeline.enable_voice_fingerprint,
            "diarization": pipeline.enable_diarization,
            "stt": pipeline.enable_stt,
            "vad": pipeline.enable_vad,
            "noise_reduction": pipeline.enable_noise_reduction,
            "tts": tts_model != "disabled",
            "emotional_tts": emotional_tts != "disabled",
        },
        "models": {
            "vad": "silero-vad",
            "stt": "mlx-community/whisper-large-v3-turbo",
            "tts": (
                "prince-canuma/Kokoro-82M" if tts_model != "disabled" else "disabled"
            ),
            "emotional_tts": (
                "KokoroModel + Prosody Trifecta (v2.4/v3)"
                if emotional_tts != "disabled" else "disabled"
            ),
            "diarization": "resemblyzer",
            "source_separation": "sepformer (fallback to spectral)",
        },
        "pre_warming": "enabled",
    }


# =============================================================================
# WebSocket Streaming Support
# =============================================================================


class StreamingSession:
    """Manages a WebSocket streaming session for real-time audio processing."""

    def __init__(
        self,
        websocket: WebSocket,
        sample_rate: int = 16000,
        enable_stt: bool = True,
        enable_diarization: bool = True,
        enable_echo_cancel: bool = True,
        enable_noise_reduction: bool = False,
        chunk_duration_ms: int = 500,
    ):
        """Initialize streaming session.

        Args:
            websocket: WebSocket connection
            sample_rate: Audio sample rate
            enable_stt: Enable speech-to-text
            enable_diarization: Enable speaker diarization
            enable_echo_cancel: Enable echo cancellation
            enable_noise_reduction: Enable DeepFilterNet3 noise reduction
            chunk_duration_ms: Processing chunk duration in ms
        """
        self.websocket = websocket
        self.sample_rate = sample_rate
        self.enable_stt = enable_stt
        self.enable_diarization = enable_diarization
        self.enable_echo_cancel = enable_echo_cancel
        self.enable_noise_reduction = enable_noise_reduction
        self.chunk_duration_ms = chunk_duration_ms

        # Audio buffer for accumulating samples
        self.audio_buffer = np.array([], dtype=np.float32)
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)

        # Session state
        self.session_id = f"session_{time.time_ns()}"
        self.total_audio_duration_s = 0.0
        self.segments_processed = 0
        self.is_active = True

    async def process_audio_data(self, data: bytes) -> None:
        """Process incoming audio data.

        Args:
            data: Raw audio bytes (int16 or float32)
        """
        # Determine audio format from data length
        # Assuming int16 by default (2 bytes per sample)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])

        # Process complete chunks
        while len(self.audio_buffer) >= self.chunk_samples:
            chunk = self.audio_buffer[: self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples :]

            # Process chunk
            await self._process_chunk(chunk)

    async def _process_chunk(self, chunk: np.ndarray) -> None:
        """Process a single audio chunk.

        Args:
            chunk: Audio chunk (float32)
        """
        pipeline = get_pipeline()
        pipeline.enable_stt = self.enable_stt
        pipeline.enable_diarization = self.enable_diarization
        pipeline.enable_echo_cancel = self.enable_echo_cancel
        pipeline.enable_noise_reduction = self.enable_noise_reduction

        start_time = time.perf_counter()

        # Use streaming process
        segment = pipeline.process_streaming(chunk, self.sample_rate)

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        chunk_duration_ms = len(chunk) / self.sample_rate * 1000
        self.total_audio_duration_s += chunk_duration_ms / 1000
        self.segments_processed += 1

        # Send result
        if segment is not None:
            result = {
                "type": "segment",
                "session_id": self.session_id,
                "segment": {
                    "start_time": (
                        segment.start_time + self.total_audio_duration_s
                        - chunk_duration_ms / 1000
                    ),
                    "end_time": (
                        segment.end_time + self.total_audio_duration_s
                        - chunk_duration_ms / 1000
                    ),
                    "speaker": segment.speaker,
                    "is_dashvoice": segment.is_dashvoice,
                    "dashvoice_voice": segment.dashvoice_voice,
                    "confidence": segment.confidence,
                    "transcription": segment.transcription,
                    "language": segment.language,
                },
                "processing_time_ms": processing_time_ms,
                "total_duration_s": self.total_audio_duration_s,
            }
        else:
            result = {
                "type": "silence",
                "session_id": self.session_id,
                "processing_time_ms": processing_time_ms,
                "total_duration_s": self.total_audio_duration_s,
            }

        await self.websocket.send_json(result)

    async def flush(self) -> None:
        """Process any remaining audio in the buffer."""
        if len(self.audio_buffer) > 0:
            # Process remaining audio even if less than chunk size
            await self._process_chunk(self.audio_buffer)
            self.audio_buffer = np.array([], dtype=np.float32)

    async def close(self) -> None:
        """Close the session and send summary."""
        self.is_active = False
        await self.flush()

        summary = {
            "type": "session_end",
            "session_id": self.session_id,
            "total_duration_s": self.total_audio_duration_s,
            "segments_processed": self.segments_processed,
        }
        await self.websocket.send_json(summary)


# Store active streaming sessions
_streaming_sessions: dict[str, StreamingSession] = {}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.

    Protocol:
    1. Client connects and sends config message:
       {"type": "config", "sample_rate": 16000, "enable_stt": true, ...}

    2. Client sends audio data as binary messages (int16 PCM)

    3. Server sends back processed segments as JSON:
       {"type": "segment", "transcription": "...", "speaker": "...", ...}

    4. Client sends end message to close:
       {"type": "end"}
    """
    await websocket.accept()
    websocket_session_opened("stream")

    session: StreamingSession | None = None

    try:
        while True:
            # Receive message (can be text or binary)
            message = await websocket.receive()

            if "text" in message:
                # JSON control message
                data = json.loads(message["text"])
                msg_type = data.get("type", "")
                record_websocket_message("stream", msg_type)

                if msg_type == "config":
                    # Initialize session with config
                    session = StreamingSession(
                        websocket=websocket,
                        sample_rate=data.get("sample_rate", 16000),
                        enable_stt=data.get("enable_stt", True),
                        enable_diarization=data.get("enable_diarization", True),
                        enable_echo_cancel=data.get("enable_echo_cancel", True),
                        enable_noise_reduction=data.get("enable_noise_reduction"),
                        chunk_duration_ms=data.get("chunk_duration_ms", 500),
                    )
                    _streaming_sessions[session.session_id] = session

                    await websocket.send_json(
                        {
                            "type": "config_ack",
                            "session_id": session.session_id,
                            "status": "ready",
                        },
                    )

                elif msg_type == "end":
                    # End session
                    if session:
                        await session.close()
                        _streaming_sessions.pop(session.session_id, None)
                    break

                elif msg_type == "ping":
                    # Keep-alive ping
                    pong_msg = {"type": "pong", "timestamp": time.time()}
                    await websocket.send_json(pong_msg)

            elif "bytes" in message:
                # Binary audio data
                record_websocket_message("stream", "audio")
                record_websocket_audio("stream", len(message["bytes"]))

                if session is None:
                    # Auto-create session with defaults
                    session = StreamingSession(websocket=websocket)
                    _streaming_sessions[session.session_id] = session
                    await websocket.send_json(
                        {
                            "type": "config_ack",
                            "session_id": session.session_id,
                            "status": "ready (default config)",
                        },
                    )

                await session.process_audio_data(message["bytes"])

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        websocket_session_closed("stream")
        if session:
            _streaming_sessions.pop(session.session_id, None)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for streaming transcription with VAD-based endpoint detection.

    Enhanced protocol with WhisperMLX streaming:
    1. Client connects
    2. (Optional) Client sends config with use_vad, language, etc.
    3. Client sends audio chunks as binary (int16 PCM, 16kHz)
    4. Server sends back transcription results with partial/final indicators

    Config options:
    - use_vad: bool (default: true) - Use VAD for natural sentence boundaries
    - language: str (default: null) - Language code or null for auto-detect
    - emit_partials: bool (default: true) - Send partial results during speech
    - min_chunk_duration: float (default: 0.5) - Minimum audio before processing
    - max_chunk_duration: float (default: 10.0) - Force processing at this duration
    - silence_threshold: float (default: 0.5) - Silence duration to trigger endpoint

    Response types:
    - {"type": "transcription", "text": ..., "is_final": true/false, ...}
    - {"type": "complete", "full_text": "..."}
    - {"type": "error", "message": "..."}

    This uses WhisperMLX streaming API for better accuracy with natural sentence
    boundaries through VAD-based endpoint detection.
    """
    await websocket.accept()
    websocket_session_opened("transcribe")

    # Check if WhisperMLX streaming is available
    whisper_model = get_streaming_whisper_model()
    use_streaming = whisper_model != "disabled" and HAS_STREAMING_WHISPER

    if use_streaming:
        # Use enhanced streaming mode with VAD
        await _websocket_transcribe_streaming(websocket, whisper_model)
    else:
        # Fallback to basic chunked mode
        await _websocket_transcribe_fallback(websocket)

    websocket_session_closed("transcribe")


async def _websocket_transcribe_streaming(websocket: WebSocket, whisper_model):
    """Enhanced streaming transcription using WhisperMLX with VAD."""
    # Default config
    config_dict = {
        "use_vad": True,
        "language": None,
        "emit_partials": True,
        "min_chunk_duration": 0.5,
        "max_chunk_duration": 10.0,
        "silence_threshold": 0.5,
    }
    total_transcription = []
    streamer = None
    audio_queue = asyncio.Queue()
    processing_task = None

    async def audio_generator():
        """Async generator that yields audio from queue."""
        while True:
            item = await audio_queue.get()
            if item is None:  # End signal
                break
            yield item

    async def process_stream():
        """Process streaming transcription and send results."""
        nonlocal total_transcription
        async for result in streamer.transcribe_stream(audio_generator()):
            response = {
                "type": "transcription",
                "text": result.text,
                "is_final": result.is_final,
                "is_partial": result.is_partial,
                "language": result.language,
                "segment_start": result.segment_start,
                "segment_end": result.segment_end,
                "processing_time_ms": result.processing_time * 1000,
                "audio_duration_s": result.audio_duration,
                "rtf": result.rtf,
            }
            if result.is_final:
                total_transcription.append(result.text)
            await websocket.send_json(response)

    try:
        # Send ready message with capabilities
        await websocket.send_json({
            "type": "ready",
            "mode": "streaming",
            "features": ["vad", "partial_results", "language_detection"],
            "default_config": config_dict,
        })

        while True:
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")
                record_websocket_message("transcribe", msg_type)

                if msg_type == "config":
                    # Update config
                    config_dict.update({
                        "use_vad": data.get("use_vad", config_dict["use_vad"]),
                        "language": data.get("language", config_dict["language"]),
                        "emit_partials": data.get(
                            "emit_partials", config_dict["emit_partials"],
                        ),
                        "min_chunk_duration": data.get(
                            "min_chunk_duration", config_dict["min_chunk_duration"],
                        ),
                        "max_chunk_duration": data.get(
                            "max_chunk_duration", config_dict["max_chunk_duration"],
                        ),
                        "silence_threshold": data.get(
                            "silence_threshold", config_dict["silence_threshold"],
                        ),
                    })

                    # Create streaming config
                    streaming_config = StreamingConfig(
                        use_vad=config_dict["use_vad"],
                        language=config_dict["language"],
                        emit_partials=config_dict["emit_partials"],
                        min_chunk_duration=config_dict["min_chunk_duration"],
                        max_chunk_duration=config_dict["max_chunk_duration"],
                        silence_threshold_duration=config_dict["silence_threshold"],
                    )

                    # Create streamer
                    streamer = StreamingWhisper(whisper_model, streaming_config)

                    # Start processing task
                    processing_task = asyncio.create_task(process_stream())

                    await websocket.send_json({
                        "type": "config_ack",
                        "config": config_dict,
                    })

                elif msg_type == "end":
                    # Signal end to audio generator
                    await audio_queue.put(None)

                    # Wait for processing to complete
                    if processing_task:
                        await processing_task

                    # Send complete transcription
                    await websocket.send_json({
                        "type": "complete",
                        "full_text": " ".join(total_transcription),
                        "segments": len(total_transcription),
                    })
                    break

                elif msg_type == "ping":
                    pong = {"type": "pong", "timestamp": time.time()}
                    await websocket.send_json(pong)

            elif "bytes" in message:
                record_websocket_message("transcribe", "audio")
                record_websocket_audio("transcribe", len(message["bytes"]))

                # Convert audio
                raw = np.frombuffer(message["bytes"], dtype=np.int16)
                audio = raw.astype(np.float32) / 32768.0

                # Initialize streamer on first audio if not configured
                if streamer is None:
                    streaming_config = StreamingConfig(
                        use_vad=config_dict["use_vad"],
                        language=config_dict["language"],
                        emit_partials=config_dict["emit_partials"],
                        min_chunk_duration=config_dict["min_chunk_duration"],
                        max_chunk_duration=config_dict["max_chunk_duration"],
                        silence_threshold_duration=config_dict["silence_threshold"],
                    )
                    streamer = StreamingWhisper(whisper_model, streaming_config)
                    processing_task = asyncio.create_task(process_stream())

                # Queue audio for processing
                await audio_queue.put(audio)

    except WebSocketDisconnect:
        if processing_task:
            await audio_queue.put(None)  # Signal end
            processing_task.cancel()
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _websocket_transcribe_fallback(websocket: WebSocket):
    """Fallback transcription using basic chunked mode (no streaming)."""
    pipeline = get_pipeline()
    stt = pipeline._get_stt()

    audio_buffer = np.array([], dtype=np.float32)
    chunk_samples = 16000  # 1 second chunks
    total_transcription = []

    try:
        # Send ready message indicating fallback mode
        await websocket.send_json({
            "type": "ready",
            "mode": "fallback",
            "features": ["basic_chunked"],
            "note": "WhisperMLX streaming not available, using basic mode",
        })

        while True:
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")
                record_websocket_message("transcribe", msg_type)

                if msg_type == "end":
                    # Process remaining audio
                    if len(audio_buffer) > 1600:
                        text, language = stt.transcribe(audio_buffer, 16000)
                        if text:
                            total_transcription.append(text)
                            await websocket.send_json({
                                "type": "transcription",
                                "text": text,
                                "language": language,
                                "is_final": True,
                                "is_partial": False,
                            })

                    await websocket.send_json({
                        "type": "complete",
                        "full_text": " ".join(total_transcription),
                        "segments": len(total_transcription),
                    })
                    break

                if msg_type == "config":
                    # Acknowledge but ignore in fallback mode
                    await websocket.send_json({
                        "type": "config_ack",
                        "mode": "fallback",
                        "note": "Config options limited in fallback mode",
                    })

            elif "bytes" in message:
                record_websocket_message("transcribe", "audio")
                record_websocket_audio("transcribe", len(message["bytes"]))

                raw = np.frombuffer(message["bytes"], dtype=np.int16)
                audio = raw.astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, audio])

                while len(audio_buffer) >= chunk_samples:
                    chunk = audio_buffer[:chunk_samples]
                    audio_buffer = audio_buffer[chunk_samples:]

                    text, language = stt.transcribe(chunk, 16000)
                    if text:
                        total_transcription.append(text)
                        await websocket.send_json({
                            "type": "transcription",
                            "text": text,
                            "language": language,
                            "is_final": False,
                            "is_partial": False,
                        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/ws/sessions")
async def list_websocket_sessions():
    """List active WebSocket streaming sessions."""
    sessions = []
    for session_id, session in _streaming_sessions.items():
        sessions.append(
            {
                "session_id": session_id,
                "sample_rate": session.sample_rate,
                "total_duration_s": session.total_audio_duration_s,
                "segments_processed": session.segments_processed,
                "is_active": session.is_active,
            },
        )
    return {"active_sessions": len(sessions), "sessions": sessions}


@app.websocket("/ws/synthesize")
async def websocket_synthesize(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS synthesis.

    Protocol:
    1. Client connects
    2. Client sends synthesis request:
       {"type": "synthesize", "text": "Hello world", "voice": "af_bella", "speed": 1.0}
    3. Server streams back audio chunks as they're generated:
       {"type": "audio_chunk", "index": 0, "audio_base64": "...", "samples": 1234}
       {"type": "audio_chunk", "index": 1, "audio_base64": "...", "samples": 5678}
    4. Server sends completion message:
       {"type": "complete", "total_chunks": N, "total_duration_s": 1.5, ...}

    Benefits of streaming:
    - Lower time-to-first-audio (TTFA)
    - Client can start playback before generation completes
    - Useful for long texts with multiple sentences

    Example (JavaScript):
        ws = new WebSocket("ws://localhost:8000/ws/synthesize");
        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: "synthesize",
                text: "Hello world. How are you today?",
                voice: "af_bella",
                speed: 1.0
            }));
        };
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "audio_chunk") {
                // Play audio chunk immediately
                playAudioChunk(msg.audio_base64);
            }
        };
    """
    await websocket.accept()
    websocket_session_opened("synthesize")

    try:
        while True:
            message = await websocket.receive()

            if "text" not in message:
                continue

            data = json.loads(message["text"])
            msg_type = data.get("type", "")
            record_websocket_message("synthesize", msg_type)

            if msg_type == "synthesize":
                # Get TTS model
                tts_model = get_tts_model()
                if tts_model == "disabled":
                    await websocket.send_json({
                        "type": "error",
                        "message": "TTS is not available",
                    })
                    continue

                text = data.get("text", "")
                voice = data.get("voice", "af_bella")
                speed = data.get("speed", 1.0)
                # Split on sentences by default
                split_pattern = data.get("split_pattern", r"[.!?]+\s*")

                # Validate inputs
                if not text:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Text is required",
                    })
                    continue

                if voice not in KOKORO_VOICES:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid voice. Available: {KOKORO_VOICES}",
                    })
                    continue

                # Send acknowledgment
                await websocket.send_json({
                    "type": "synthesis_started",
                    "text_length": len(text),
                    "voice": voice,
                    "speed": speed,
                })

                # Stream audio chunks with metrics tracking
                start_time = time.perf_counter()
                total_samples = 0
                total_duration = 0.0
                chunk_index = 0
                sample_rate = 24000  # Default Kokoro sample rate

                async with tracked_semaphore(_tts_semaphore, "tts"):
                    with track_tts_generation(voice=voice):
                        for result in tts_model.generate(
                            text=text,
                            voice=voice,
                            speed=speed,
                            split_pattern=split_pattern,
                            verbose=False,
                        ):
                            # Convert to numpy and base64
                            audio = np.array(result.audio)
                            sample_rate = result.sample_rate
                            duration = len(audio) / sample_rate

                            # Create WAV chunk
                            buffer = io.BytesIO()
                            sf.write(buffer, audio, sample_rate, format='WAV')
                            buffer.seek(0)
                            audio_bytes = buffer.getvalue()

                            # Calculate timing
                            chunk_time = time.perf_counter() - start_time
                            if chunk_index == 0:
                                ttfa = chunk_time * 1000  # Time to first audio

                            # Send chunk
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "index": chunk_index,
                                "audio_base64": base64.b64encode(audio_bytes).decode(),
                                "samples": len(audio),
                                "sample_rate": sample_rate,
                                "duration_s": duration,
                                "segment_text": (
                                    getattr(result, 'graphemes', "")
                                ),
                                "elapsed_ms": chunk_time * 1000,
                            })

                            total_samples += len(audio)
                            total_duration += duration
                            chunk_index += 1

                # Record TTS metrics
                generation_time_s = time.perf_counter() - start_time
                record_tts_output(voice, total_samples, sample_rate, generation_time_s)

                # Send completion
                total_time = generation_time_s * 1000
                await websocket.send_json({
                    "type": "complete",
                    "total_chunks": chunk_index,
                    "total_samples": total_samples,
                    "total_duration_s": total_duration,
                    "total_time_ms": total_time,
                    "ttfa_ms": ttfa if chunk_index > 0 else 0,
                    "rtf": total_time / 1000 / total_duration if total_duration else 0,
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        websocket_session_closed("synthesize")


@app.websocket("/ws/synthesize_emotional")
async def websocket_synthesize_emotional(websocket: WebSocket):
    """
    WebSocket endpoint for streaming emotional TTS synthesis with prosody trifecta.

    Protocol:
    1. Client connects
    2. Client sends synthesis request:
       {"type": "synthesize", "text": "Hello", "voice": "af_heart", "emotion": "angry"}
    3. Server streams back audio chunks (one per sentence):
       {"type": "audio_chunk", "index": 0, "audio_base64": "...", "samples": 1234}
    4. Server sends completion message:
       {"type": "complete", "total_chunks": N, "total_duration_s": 1.5, ...}

    Prosody Trifecta features:
    - F0 Contour v2.4: Modifies pitch patterns based on emotion
    - Duration v3: Adjusts speaking rate (angry=faster, sad=slower)
    - Energy v3: Adjusts volume/loudness (angry=louder, sad=quieter)

    Available emotions: neutral, angry, sad, excited, calm, frustrated, nervous

    Example (JavaScript):
        ws = new WebSocket("ws://localhost:8000/ws/synthesize_emotional");
        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: "synthesize",
                text: "Hello world. How are you today?",
                voice: "af_heart",
                emotion: "angry",
                speed: 1.0
            }));
        };
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "audio_chunk") {
                playAudioChunk(msg.audio_base64);
            }
        };
    """
    await websocket.accept()
    websocket_session_opened("synthesize_emotional")

    try:
        while True:
            message = await websocket.receive()

            if "text" not in message:
                continue

            data = json.loads(message["text"])
            msg_type = data.get("type", "")
            record_websocket_message("synthesize_emotional", msg_type)

            if msg_type == "synthesize":
                # Get emotional TTS model
                emotional_tts = get_emotional_tts()
                if emotional_tts == "disabled":
                    await websocket.send_json({
                        "type": "error",
                        "message": "Emotional TTS unavailable. Missing prosody weights.",
                    })
                    continue

                model = emotional_tts["model"]
                converter = emotional_tts["converter"]
                voice_packs = emotional_tts["voice_packs"]
                sample_rate = emotional_tts["sample_rate"]

                text = data.get("text", "")
                voice = data.get("voice", "af_heart")
                emotion = data.get("emotion", "neutral").lower()
                speed = data.get("speed", 1.0)
                split_pattern = data.get("split_pattern", r"[.!?]+\s*")

                # Validate inputs
                if not text:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Text is required",
                    })
                    continue

                if voice not in voice_packs:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid voice: {list(voice_packs.keys())}",
                    })
                    continue

                if emotion not in EMOTION_PROSODY_IDS:
                    emotions = list(EMOTION_PROSODY_IDS.keys())
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid emotion: {emotions}",
                    })
                    continue

                if not 0.5 <= speed <= 2.0:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Speed must be between 0.5 and 2.0",
                    })
                    continue

                # Send acknowledgment
                await websocket.send_json({
                    "type": "synthesis_started",
                    "text_length": len(text),
                    "voice": voice,
                    "emotion": emotion,
                    "speed": speed,
                    "features": ["F0 contour v2.4", "Duration v3", "Energy v3"],
                })

                # Import required modules
                import re

                import mlx.core as mx

                from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                    phonemize_text,
                )

                # Split text into segments for streaming
                segments = re.split(split_pattern, text)
                segments = [s.strip() for s in segments if s.strip()]
                if not segments:
                    segments = [text]  # Fall back to full text if no splits

                # Stream audio chunks with metrics tracking
                start_time = time.perf_counter()
                total_samples = 0
                total_duration = 0.0
                chunk_index = 0
                ttfa = 0.0
                prosody_id = EMOTION_PROSODY_IDS[emotion]

                async with tracked_semaphore(_tts_semaphore, "tts"):
                    with track_tts_generation(voice=f"{voice}_{emotion}"):
                        for segment in segments:
                            if not segment:
                                continue

                            # Phonemize segment
                            try:
                                phonemes, tokens = phonemize_text(segment)
                            except Exception as e:
                                await websocket.send_json({
                                    "type": "warning",
                                    "message": f"Skipping segment: {str(e)}",
                                    "segment": segment[:50],
                                })
                                continue

                            # Get voice pack and embedding
                            voice_pack = voice_packs[voice]
                            voice_embedding = converter.select_voice_embedding(
                                voice_pack, len(tokens),
                            )

                            # Create prosody mask
                            mask = [[prosody_id] * len(tokens)]
                            prosody_mask = mx.array(mask, dtype=mx.int32)

                            # Generate audio
                            tokens_mx = mx.array([tokens])
                            audio = model(
                                tokens_mx, voice_embedding, prosody_mask=prosody_mask,
                            )
                            mx.eval(audio)

                            # Convert to numpy
                            audio_np = np.array(audio).flatten()

                            # Apply speed adjustment
                            if speed != 1.0:
                                from scipy import signal
                                new_length = int(len(audio_np) / speed)
                                resampled = signal.resample(audio_np, new_length)
                                audio_np = resampled.astype(np.float32)

                            duration = len(audio_np) / sample_rate

                            # Create WAV chunk
                            buffer = io.BytesIO()
                            sf.write(buffer, audio_np, sample_rate, format='WAV')
                            buffer.seek(0)
                            audio_bytes = buffer.getvalue()

                            # Calculate timing
                            chunk_time = time.perf_counter() - start_time
                            if chunk_index == 0:
                                ttfa = chunk_time * 1000

                            # Send chunk
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "index": chunk_index,
                                "audio_base64": base64.b64encode(audio_bytes).decode(),
                                "samples": len(audio_np),
                                "sample_rate": sample_rate,
                                "duration_s": duration,
                                "segment_text": segment,
                                "phonemes": phonemes,
                                "emotion": emotion,
                                "elapsed_ms": chunk_time * 1000,
                            })

                            total_samples += len(audio_np)
                            total_duration += duration
                            chunk_index += 1

                # Record TTS metrics
                generation_time_s = time.perf_counter() - start_time
                record_tts_output(
                    f"{voice}_{emotion}", total_samples, sample_rate, generation_time_s,
                )

                # Send completion
                total_time = generation_time_s * 1000
                await websocket.send_json({
                    "type": "complete",
                    "total_chunks": chunk_index,
                    "total_samples": total_samples,
                    "total_duration_s": total_duration,
                    "total_time_ms": total_time,
                    "ttfa_ms": ttfa if chunk_index > 0 else 0,
                    "rtf": total_time / 1000 / total_duration if total_duration else 0,
                    "emotion": emotion,
                    "prosody_id": prosody_id,
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        websocket_session_closed("synthesize_emotional")


# =============================================================================
# Wake Word Detection Endpoints
# =============================================================================


class WakeWordResponse(BaseModel):
    """Response model for wake word detection."""
    detected: bool
    probability: float
    threshold: float
    audio_duration_s: float
    vad_enabled: bool
    model_name: str = "hey_agent"


@app.get("/wakeword/info")
async def wakeword_info():
    """
    Get information about the wake word detection model.

    Returns model status, configuration, and supported wake word.
    """
    detector = get_wakeword_detector()

    if detector == "disabled":
        return {
            "enabled": False,
            "reason": "Wake word model not loaded. Check server logs.",
            "model_path": "models/wakeword/hey_agent/hey_agent_mlx.safetensors",
        }

    return {
        "enabled": True,
        "wake_word": "Hey Agent",
        "model_name": "hey_agent",
        "model_type": "CNN",
        "backend": detector.backend,
        "sample_rate": detector.sample_rate,
        "vad_enabled": detector.use_vad,
        "vad_threshold": detector.vad.threshold if detector.vad else None,
        "n_mels": detector.n_mels,
        "max_frames": detector.max_frames,
        "model_path": "models/wakeword/hey_agent/hey_agent_mlx.safetensors",
    }


@app.post("/wakeword/detect", response_model=WakeWordResponse)
async def detect_wakeword(
    audio: UploadFile = File(..., description="Audio file to check for wake word"),
    threshold: float = Form(default=0.5, description="Detection threshold (0.0-1.0)"),
):
    """
    Detect wake word in audio.

    Uses the custom-trained "Hey Agent" CNN model with optional VAD preprocessing
    to filter noise and silence.

    Example usage:
        curl -X POST http://localhost:8000/wakeword/detect \\
            -F "audio=@audio.wav" \\
            -F "threshold=0.5"

    Returns:
        detected: True if wake word probability > threshold
        probability: Raw detection probability (0.0-1.0)
        threshold: The threshold used for detection
        audio_duration_s: Duration of input audio
        vad_enabled: Whether VAD preprocessing was used
    """
    detector = get_wakeword_detector()

    if detector == "disabled":
        raise HTTPException(
            status_code=503,
            detail="Wake word detection not available. Model not loaded.",
        )

    try:
        # Load audio
        audio_data, sample_rate = load_audio_from_upload(audio)
        audio_duration_s = len(audio_data) / sample_rate

        # Resample if needed
        if sample_rate != detector.sample_rate:
            from scipy import signal
            audio_data = signal.resample(
                audio_data,
                int(len(audio_data) * detector.sample_rate / sample_rate),
            )
            audio_data = audio_data.astype(np.float32)

        # Run detection
        probability = detector.detect(audio_data)
        detected = probability > threshold

        return WakeWordResponse(
            detected=detected,
            probability=probability,
            threshold=threshold,
            audio_duration_s=audio_duration_s,
            vad_enabled=detector.use_vad,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/ws/wakeword")
async def websocket_wakeword(websocket: WebSocket):
    """
    WebSocket endpoint for streaming wake word detection.

    Protocol:
    1. Client sends config: {"type": "config", "threshold": 0.5, "sample_rate": 16000}
    2. Client sends audio chunks as binary data
    3. Server responds with detection results for each chunk:
       {"type": "detection", "detected": false, "probability": 0.12}
    4. When wake word detected, server sends:
       {"type": "detected", "probability": 0.95, "chunk_index": 42}

    Audio format:
    - 16-bit signed integer PCM (int16)
    - Little-endian byte order
    - Mono channel
    - Recommended chunk size: 16000 samples (1 second at 16kHz)

    Example (JavaScript):
        ws = new WebSocket("ws://localhost:8000/ws/wakeword");
        ws.onopen = () => {
            ws.send(JSON.stringify({type: "config", threshold: 0.5}));
        };
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === "detected") {
                console.log("Wake word detected!");
            }
        };
        // Send audio chunks as ArrayBuffer
    """
    detector = get_wakeword_detector()

    if detector == "disabled":
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "message": "Wake word detection not available. Model not loaded.",
        })
        await websocket.close()
        return

    await websocket.accept()
    websocket_session_opened("wakeword")

    # Configuration
    config = {
        "threshold": 0.5,
        "sample_rate": 16000,
    }
    chunk_index = 0

    try:
        await websocket.send_json({
            "type": "ready",
            "message": "Send config or audio chunks",
            "expected_sample_rate": detector.sample_rate,
            "vad_enabled": detector.use_vad,
        })

        while True:
            message = await websocket.receive()

            if "text" in message:
                # JSON message - configuration
                data = json.loads(message["text"])
                msg_type = data.get("type", "")

                if msg_type == "config":
                    config["threshold"] = data.get("threshold", config["threshold"])
                    sr = data.get("sample_rate", config["sample_rate"])
                    config["sample_rate"] = sr
                    await websocket.send_json({
                        "type": "config_ack",
                        "threshold": config["threshold"],
                        "sample_rate": config["sample_rate"],
                    })
                    record_websocket_message("wakeword", "config")

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    record_websocket_message("wakeword", "ping")

            elif "bytes" in message:
                # Binary message - audio data
                audio_bytes = message["bytes"]
                record_websocket_audio("wakeword", len(audio_bytes))

                # Convert bytes to numpy array (16-bit signed integer)
                raw_audio = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = raw_audio.astype(np.float32) / 32768.0

                # Resample if needed
                if config["sample_rate"] != detector.sample_rate:
                    from scipy import signal
                    ratio = detector.sample_rate // config["sample_rate"]
                    new_len = len(audio_data) * ratio
                    audio_data = signal.resample(audio_data, int(new_len))

                # Run detection
                probability = detector.detect(audio_data)
                detected = probability > config["threshold"]

                response = {
                    "type": "detected" if detected else "detection",
                    "detected": detected,
                    "probability": round(probability, 4),
                    "chunk_index": chunk_index,
                }

                await websocket.send_json(response)
                record_websocket_message("wakeword", "detection")
                chunk_index += 1

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        websocket_session_closed("wakeword")


# =============================================================================
# Voice Cloning Endpoints
# =============================================================================

# Global voice database instance (lazy-loaded)
_voice_database: VoiceDatabase | None = None


def get_voice_database() -> VoiceDatabase:
    """Get or create the global voice database instance."""
    global _voice_database
    if _voice_database is None:
        _voice_database = VoiceDatabase()
    return _voice_database


@app.post("/clone/extract")
async def extract_speaker_embedding(
    audio: UploadFile = File(..., description="Reference audio for voice cloning"),
    name: str | None = Form(default=None, description="Name to save voice"),
):
    """
    Extract speaker embedding from reference audio for voice cloning.

    This endpoint extracts a speaker embedding that can be used for:
    1. Voice cloning synthesis (via /clone/synthesize)
    2. Voice identification and matching

    The embedding is extracted using Resemblyzer (256-dimensional).

    If a name is provided, the voice is saved to the database for future use.
    Otherwise, only the embedding is returned.

    Example usage:
        curl -X POST http://localhost:8000/clone/extract \\
            -F "audio=@reference.wav" \\
            -F "name=my_voice"
    """
    try:
        # Load audio
        audio_data, sample_rate = load_audio_from_upload(audio)

        # Check minimum duration (need at least 1 second for good embedding)
        duration_s = len(audio_data) / sample_rate
        if duration_s < 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short ({duration_s:.1f}s). Need at least 1 second.",
            )

        # Get voice database and extract embedding
        db = get_voice_database()
        embedding = db.extract_embedding(audio_data, sample_rate)

        # Save if name provided
        saved = False
        if name:
            db.add_fingerprint(
                name=name,
                audio=audio_data,
                sample_rate=sample_rate,
                source="custom",
                reference_text="",
            )
            saved = True

        return {
            "embedding_dim": len(embedding),
            "embedding": embedding.tolist(),
            "saved": saved,
            "name": name,
            "audio_duration_s": duration_s,
            "note": "Embedding uses Resemblyzer (256-dim). For CosyVoice2 synthesis, "
                    "use /clone/synthesize with the name parameter.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/clone/synthesize")
async def synthesize_cloned_voice(
    text: str = Form(..., description="Text to synthesize"),
    voice_name: str | None = Form(default=None, description="Name of cloned voice"),
    speed: float = Form(default=1.0, description="Speech speed (0.5-2.0)"),
):
    """
    Synthesize speech using a cloned voice.

    This endpoint uses CosyVoice2 for voice cloning synthesis.

    IMPORTANT: On Python 3.14, speaker embedding extraction from CAM++ is not
    available (onnxruntime not supported). The synthesis will use a default
    speaker embedding. For full voice cloning, use Python 3.13 or earlier.

    Options:
    1. voice_name: Use a previously saved voice from /clone/extract
    2. No voice_name: Use default CosyVoice2 speaker

    Example usage:
        # With saved voice
        curl -X POST http://localhost:8000/clone/synthesize \\
            -F "text=Hello world" \\
            -F "voice_name=my_voice"

        # Default voice
        curl -X POST http://localhost:8000/clone/synthesize \\
            -F "text=Hello world"
    """
    try:
        # Validate speed
        if not 0.5 <= speed <= 2.0:
            raise HTTPException(
                status_code=400,
                detail="Speed must be between 0.5 and 2.0",
            )

        # Try to load CosyVoice2
        try:
            from tools.pytorch_to_mlx.converters.models.cosyvoice2 import (
                CosyVoice2Model,
            )
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="CosyVoice2 model not available. Check installation.",
            ) from None

        # Load model
        model_path = CosyVoice2Model.get_default_model_path()
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"CosyVoice2 model not found at {model_path}",
            )

        # Get voice embedding if name provided
        speaker_embedding = None
        voice_source = "default"

        if voice_name:
            db = get_voice_database()
            fingerprint = db.get_fingerprint(voice_name)
            if fingerprint is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice '{voice_name}' not found. Use /clone/voices.",
                )
            # Note: Resemblyzer embedding (256-dim) != CosyVoice2 CAM++ (192-dim)
            # This is a limitation on Python 3.14
            voice_source = f"cloned:{voice_name}"

        # Load CosyVoice2 model
        model = CosyVoice2Model.from_pretrained(str(model_path))

        # Get speaker embedding
        # On Python 3.14, we can't use CAM++ (onnxruntime unavailable)
        # Use random embedding for now
        if speaker_embedding is None:
            speaker_embedding = model.tokenizer.random_speaker_embedding(seed=42)

        # Generate audio
        start_time = time.perf_counter()

        audio = model.synthesize_text(
            text=text,
            speaker_embedding=speaker_embedding,
            max_tokens=1000,
            num_flow_steps=10,
        )

        generation_time_s = time.perf_counter() - start_time
        generation_time_ms = generation_time_s * 1000

        # Convert to numpy (CosyVoice2 outputs 24kHz audio)
        audio_np = np.array(audio).squeeze()
        sample_rate = 24000  # CosyVoice2 sample rate
        duration_s = len(audio_np) / sample_rate

        # Create WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.getvalue()

        return {
            "text": text,
            "voice_source": voice_source,
            "speed": speed,
            "sample_rate": sample_rate,
            "samples": len(audio_np),
            "duration_s": duration_s,
            "generation_time_ms": generation_time_ms,
            "rtf": (generation_time_ms / 1000) / duration_s if duration_s > 0 else 0,
            "audio_base64": base64.b64encode(audio_bytes).decode(),
            "note": (
                "CosyVoice2 synthesis. On Python 3.14, speaker embedding uses "
                "random seed. For proper voice cloning, use Python 3.13."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/clone/voices")
async def list_cloned_voices():
    """
    List all registered cloned voices.

    Returns all voices saved via /clone/extract, including:
    - Custom cloned voices
    - Pre-computed Kokoro voice fingerprints
    - Pre-computed CosyVoice2 voice fingerprints

    Example usage:
        curl http://localhost:8000/clone/voices
    """
    try:
        db = get_voice_database()
        voices = []

        for name in db.list_fingerprints():
            fingerprint = db.get_fingerprint(name)
            if fingerprint:
                voices.append({
                    "name": name,
                    "source": fingerprint.source,
                    "created_at": fingerprint.created_at,
                    "embedding_dim": len(fingerprint.embedding),
                })

        # Group by source
        by_source = {}
        for voice in voices:
            source = voice["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(voice["name"])

        return {
            "total_voices": len(voices),
            "voices": voices,
            "by_source": by_source,
            "note": "Use voice names with /clone/synthesize.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/clone/voices/{voice_name}")
async def delete_cloned_voice(voice_name: str):
    """
    Delete a cloned voice from the database.

    Only custom voices can be deleted. Built-in Kokoro/CosyVoice2 voices are protected.

    Example usage:
        curl -X DELETE http://localhost:8000/clone/voices/my_voice
    """
    try:
        db = get_voice_database()
        fingerprint = db.get_fingerprint(voice_name)

        if fingerprint is None:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_name}' not found.",
            )

        if fingerprint.source != "custom":
            raise HTTPException(
                status_code=403,
                detail=f"Cannot delete built-in voice '{voice_name}'",
            )

        # Delete from database
        del db.fingerprints[voice_name]
        db._save_database()

        return {
            "deleted": voice_name,
            "message": f"Voice '{voice_name}' deleted successfully.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def main():
    """Run the server."""
    import uvicorn

    host = os.getenv("DASHVOICE_HOST", "0.0.0.0")
    port = int(os.getenv("DASHVOICE_PORT", "8000"))

    print(f"Starting DashVoice server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
