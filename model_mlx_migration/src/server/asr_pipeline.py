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
Integrated ASR pipeline for SOTA++ Voice Server (Phase 9.4).

Combines:
- Zipformer encoder/decoder for ASR
- Rich audio heads (emotion, pitch, phoneme, etc.)
- ROVER high-accuracy mode
- Multi-speaker handling
- Language routing

This module bridges the server interface (ASRPipeline) with the
actual model inference components.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mlx.core as mx  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .language_router import LanguageRouter, LanguageRouterConfig
from .multi_speaker import MultiSpeakerConfig, MultiSpeakerPipeline
from .rich_token import (
    ASRMode,
    EmotionLabel,
    HallucinationInfo,
    LanguageInfo,
    PhonemeInfo,
    PitchInfo,
    RichToken,
    SpeakerInfo,
    WordTimestamp,
    create_final_token,
    create_partial_token,
)
from .voice_server import ASRPipeline, ClientSession, ServerConfig

# Emotion label mapping (index -> label)
EMOTION_LABELS = [
    EmotionLabel.NEUTRAL,
    EmotionLabel.HAPPY,
    EmotionLabel.SAD,
    EmotionLabel.ANGRY,
    EmotionLabel.FEAR,
    EmotionLabel.DISGUST,
    EmotionLabel.SURPRISE,
    EmotionLabel.CONTEMPT,
]


@dataclass
class IntegratedPipelineConfig:
    """Configuration for integrated ASR pipeline."""
    # Model paths
    zipformer_checkpoint: str | None = None
    zipformer_bpe_model: str | None = None
    whisper_model: str = "large-v3"

    # Feature extraction
    sample_rate: int = 16000
    num_mel_bins: int = 80

    # Rich heads
    enable_rich_heads: bool = True
    encoder_dim: int = 384  # Zipformer encoder dimension

    # Multi-speaker
    enable_multi_speaker: bool = True
    max_speakers: int = 2

    # ROVER
    enable_rover: bool = True

    # Timeouts (ms)
    streaming_timeout_ms: int = 100
    high_accuracy_timeout_ms: int = 5000


class IntegratedASRPipeline(ASRPipeline):
    """
    Integrated ASR pipeline with rich audio features.

    Wraps Zipformer ASR and rich audio heads for unified inference.
    Supports both streaming and high-accuracy (ROVER) modes.
    """

    def __init__(
        self,
        config: IntegratedPipelineConfig | None = None,
    ):
        self.config = config or IntegratedPipelineConfig()

        # Model components (lazy loaded)
        self._zipformer_pipeline: Any = None
        self._whisper_pipeline: Any = None
        self._rich_heads: Any = None
        self._rover_decoder: Any = None

        # Sub-pipelines
        self._multi_speaker = MultiSpeakerPipeline(
            MultiSpeakerConfig(
                enabled=self.config.enable_multi_speaker,
                max_speakers=self.config.max_speakers,
            ),
        ) if self.config.enable_multi_speaker else None

        self._language_router = LanguageRouter(
            LanguageRouterConfig(),
        )

        # Loaded state
        self._models_loaded = False

    def _load_models(self):
        """Lazy load models on first inference."""
        if self._models_loaded:
            return

        # Try to load Zipformer
        if self.config.zipformer_checkpoint and self.config.zipformer_bpe_model:
            try:
                from ..models.zipformer.inference import (
                    ASRPipeline as ZipformerPipeline,
                )
                self._zipformer_pipeline = ZipformerPipeline.from_pretrained(
                    self.config.zipformer_checkpoint,
                    self.config.zipformer_bpe_model,
                )
            except Exception:
                pass

        # Try to load rich heads
        if self.config.enable_rich_heads:
            try:
                from ..models.heads import RichAudioHeads, RichAudioHeadsConfig
                heads_config = RichAudioHeadsConfig(
                    encoder_dim=self.config.encoder_dim,
                )
                self._rich_heads = RichAudioHeads(heads_config)
            except Exception:
                pass

        # Try to load ROVER decoder
        if self.config.enable_rover:
            try:
                from ..decoding import HighAccuracyDecoder
                self._rover_decoder = HighAccuracyDecoder()
            except Exception:
                pass

        self._models_loaded = True

    async def process_chunk(
        self,
        audio: np.ndarray,
        session: ClientSession,
        mode: ASRMode,
        config: ServerConfig,
    ) -> RichToken | None:
        """
        Process an audio chunk for streaming ASR.

        For streaming mode, returns partial results immediately.
        For high-accuracy mode, accumulates audio for batch processing.
        """
        self._load_models()

        # Track timing
        len(audio) / config.sample_rate * 1000

        if mode == ASRMode.HIGH_ACCURACY:
            # In high-accuracy mode, just acknowledge receipt
            # Final result comes from finalize_utterance
            return None

        # Streaming mode: process immediately
        if self._zipformer_pipeline is not None:
            try:
                # Run streaming inference
                text = self._zipformer_pipeline.transcribe(audio, config.sample_rate)
                if text.strip():
                    return create_partial_token(
                        text=text,
                        start_ms=session.audio_start_ms,
                        end_ms=session.audio_start_ms + session.metrics.audio_received_ms,
                        chunk_index=session.chunk_index,
                        utterance_id=session.utterance_id,
                    )
            except Exception:
                pass

        # Fallback: return placeholder
        return create_partial_token(
            text="...",
            start_ms=session.audio_start_ms,
            end_ms=session.audio_start_ms + session.metrics.audio_received_ms,
            chunk_index=session.chunk_index,
            utterance_id=session.utterance_id,
            confidence=0.5,
        )

    async def finalize_utterance(
        self,
        session: ClientSession,
        mode: ASRMode,
        config: ServerConfig,
    ) -> RichToken:
        """
        Finalize utterance and return complete result with all features.

        For streaming mode: combines accumulated partials.
        For high-accuracy mode: runs ROVER voting on full audio.
        """
        self._load_models()

        # Concatenate audio buffer
        if session.audio_buffer:
            full_audio = np.concatenate(session.audio_buffer)
        else:
            full_audio = np.zeros(1600, dtype=np.float32)  # 100ms silence

        duration_ms = len(full_audio) / config.sample_rate * 1000

        # Run ASR
        text = ""
        word_timestamps: list[WordTimestamp] = []
        confidence = 0.9

        if mode == ASRMode.HIGH_ACCURACY and self._rover_decoder is not None:
            # ROVER high-accuracy mode
            try:
                result = await self._run_rover(full_audio, config)
                text = result.get("text", "")
                confidence = result.get("confidence", 0.95)
                word_timestamps = result.get("timestamps", [])
            except Exception:
                pass

        if not text and self._zipformer_pipeline is not None:
            # Zipformer streaming/fallback
            try:
                text = self._zipformer_pipeline.transcribe(full_audio, config.sample_rate)
                confidence = 0.9
            except Exception:
                pass

        if not text:
            text = "[no transcription]"
            confidence = 0.0

        # Create base token
        token = create_final_token(
            text=text,
            start_ms=session.audio_start_ms,
            end_ms=session.audio_start_ms + duration_ms,
            word_timestamps=word_timestamps,
            utterance_id=session.utterance_id,
            confidence=confidence,
            mode=mode,
        )

        # Add rich features if enabled
        if self._rich_heads is not None and config.enable_emotion:
            await self._add_rich_features(token, full_audio, config)

        # Add language info
        if config.enable_language:
            lang_result = self._language_router.detect_language(
                full_audio,
                session_id=session.session_id,
            )
            token.language = LanguageInfo(
                language=lang_result.language,
                confidence=lang_result.confidence,
                all_probs=lang_result.all_probs,
            )

        return token

    async def _run_rover(
        self,
        audio: np.ndarray,
        config: ServerConfig,
    ) -> dict[str, Any]:
        """Run ROVER high-accuracy decoding."""
        if self._rover_decoder is None:
            return {"text": "", "confidence": 0.0}

        try:
            # Run ROVER voting
            result = self._rover_decoder.decode(audio)
            return {
                "text": result.text,
                "confidence": result.confidence,
                "timestamps": [],
            }
        except Exception:
            return {"text": "", "confidence": 0.0}

    async def _add_rich_features(
        self,
        token: RichToken,
        audio: np.ndarray,
        config: ServerConfig,
    ):
        """Add rich audio features to token."""
        if self._rich_heads is None:
            return

        try:
            # This would run the actual rich heads inference
            # For now, we use placeholder logic

            if config.enable_emotion:
                # Mock emotion detection
                token.emotion = EmotionLabel.NEUTRAL
                token.emotion_confidence = 0.75

            if config.enable_pitch:
                # Mock pitch extraction
                token.pitch = PitchInfo(
                    mean_hz=150.0,
                    std_hz=30.0,
                    min_hz=80.0,
                    max_hz=250.0,
                    voiced_ratio=0.65,
                )

            if config.enable_phoneme:
                # Mock phoneme extraction
                token.phonemes = PhonemeInfo(
                    phonemes=["<placeholder>"],
                    confidences=[0.5],
                )

            if config.enable_hallucination:
                token.hallucination = HallucinationInfo(
                    score=0.1,
                    is_hallucinated=False,
                )

            if config.enable_speaker:
                token.speaker = SpeakerInfo(
                    embedding=[0.0] * 256,
                )

        except Exception:
            pass

    async def detect_language(
        self,
        audio: np.ndarray,
    ) -> tuple[str, float]:
        """Detect language from audio."""
        result = self._language_router.detect_language(audio)
        return (result.language, result.confidence)

    async def detect_speakers(
        self,
        audio: np.ndarray,
    ) -> int:
        """Detect number of speakers."""
        if self._multi_speaker is None:
            return 1

        result = await self._multi_speaker.separate_async(audio)
        return result.num_speakers


class ZipformerASRPipeline(ASRPipeline):
    """
    Zipformer-only ASR pipeline.

    Simpler pipeline that only uses Zipformer, without rich heads.
    Useful for low-latency streaming scenarios.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        bpe_model_path: str | None = None,
    ):
        self._checkpoint_path = checkpoint_path
        self._bpe_model_path = bpe_model_path
        self._pipeline: Any = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return

        if self._checkpoint_path and self._bpe_model_path:
            try:
                from ..models.zipformer.inference import (
                    ASRPipeline as ZipformerPipeline,
                )
                self._pipeline = ZipformerPipeline.from_pretrained(
                    self._checkpoint_path,
                    self._bpe_model_path,
                )
            except Exception:
                pass

        self._loaded = True

    async def process_chunk(
        self,
        audio: np.ndarray,
        session: ClientSession,
        mode: ASRMode,
        config: ServerConfig,
    ) -> RichToken | None:
        self._load()

        if self._pipeline is None:
            return None

        try:
            text = self._pipeline.transcribe(audio, config.sample_rate)
            if text.strip():
                return create_partial_token(
                    text=text,
                    start_ms=session.audio_start_ms,
                    end_ms=session.audio_start_ms + session.metrics.audio_received_ms,
                    chunk_index=session.chunk_index,
                    utterance_id=session.utterance_id,
                )
        except Exception:
            pass

        return None

    async def finalize_utterance(
        self,
        session: ClientSession,
        mode: ASRMode,
        config: ServerConfig,
    ) -> RichToken:
        self._load()

        if session.audio_buffer:
            full_audio = np.concatenate(session.audio_buffer)
        else:
            full_audio = np.zeros(1600, dtype=np.float32)

        duration_ms = len(full_audio) / config.sample_rate * 1000
        text = ""

        if self._pipeline is not None:
            try:
                text = self._pipeline.transcribe(full_audio, config.sample_rate)
            except Exception:
                pass

        return create_final_token(
            text=text or "[no transcription]",
            start_ms=session.audio_start_ms,
            end_ms=session.audio_start_ms + duration_ms,
            utterance_id=session.utterance_id,
            mode=mode,
        )
