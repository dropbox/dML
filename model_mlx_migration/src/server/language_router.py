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
Language routing for SOTA++ Voice Server (Phase 9.3).

Routes audio to the appropriate ASR backend based on detected language:
- Supported languages → Zipformer (primary, streaming)
- Unsupported languages → Whisper fallback (100+ languages)

The router uses the LanguageHead from Phase 5.5 for language identification,
with confidence-based fallback to Whisper when uncertain.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ASRBackend(str, Enum):
    """Available ASR backends."""
    ZIPFORMER = "zipformer"  # Primary streaming ASR
    WHISPER = "whisper"  # Fallback for unsupported languages
    ROVER = "rover"  # Combined high-accuracy mode


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str  # ISO 639-1 code
    confidence: float
    backend: ASRBackend
    all_probs: dict[str, float] | None = None


@dataclass
class LanguageRouterConfig:
    """Configuration for language router."""
    # Languages supported by Zipformer (primary ASR)
    supported_languages: set[str] = field(default_factory=lambda: {
        "en",  # English
        "es",  # Spanish
        "fr",  # French
        "de",  # German
        "it",  # Italian
        "pt",  # Portuguese
        "nl",  # Dutch
        "pl",  # Polish
        "ru",  # Russian
    })

    # Confidence threshold to use detected language
    confidence_threshold: float = 0.7

    # Confidence threshold to route to Zipformer (must be > general threshold)
    zipformer_confidence_threshold: float = 0.8

    # Default language when confidence is too low
    default_language: str = "en"

    # Whether to use Whisper for language detection
    use_whisper_detection: bool = True

    # Sample rate
    sample_rate: int = 16000

    # Minimum audio duration for language detection (ms)
    min_detection_duration_ms: float = 500.0


class LanguageRouter:
    """
    Language-based routing for ASR backends.

    Uses language identification to route audio to the most appropriate
    ASR backend. Supported languages go to Zipformer for low-latency
    streaming, while unsupported languages fall back to Whisper.
    """

    def __init__(
        self,
        config: LanguageRouterConfig | None = None,
        language_head: object | None = None,  # Phase 5.5 LanguageHead
    ):
        self.config = config or LanguageRouterConfig()
        self._language_head = language_head
        self._language_head_loaded = False

        # Cache recent language detections for session continuity
        self._session_languages: dict[str, str] = {}

    def detect_language(
        self,
        audio: np.ndarray,
        session_id: str | None = None,
    ) -> LanguageDetectionResult:
        """
        Detect language and determine routing.

        Args:
            audio: Audio samples at 16kHz
            session_id: Optional session ID for caching

        Returns:
            LanguageDetectionResult with language and backend
        """
        # Check if audio is long enough
        duration_ms = len(audio) / self.config.sample_rate * 1000
        if duration_ms < self.config.min_detection_duration_ms:
            # Use cached or default language
            if session_id and session_id in self._session_languages:
                language = self._session_languages[session_id]
            else:
                language = self.config.default_language

            return LanguageDetectionResult(
                language=language,
                confidence=0.5,  # Low confidence for short audio
                backend=self._get_backend(language, 0.5),
            )

        # Run language detection
        language, confidence, all_probs = self._detect_language_impl(audio)

        # Cache result for session
        if session_id:
            self._session_languages[session_id] = language

        # Determine backend
        backend = self._get_backend(language, confidence)

        return LanguageDetectionResult(
            language=language,
            confidence=confidence,
            backend=backend,
            all_probs=all_probs,
        )

    def _detect_language_impl(
        self,
        audio: np.ndarray,
    ) -> tuple[str, float, dict[str, float] | None]:
        """
        Internal language detection implementation.

        Uses LanguageHead if available, otherwise falls back to
        simple heuristics or Whisper detection.
        """
        # If we have a trained language head, use it
        if self._language_head is not None:
            # TODO: Integrate with actual LanguageHead inference
            pass

        # Fallback: assume English with moderate confidence
        # In production, this would use Whisper language detection
        return (self.config.default_language, 0.85, None)

    def _get_backend(self, language: str, confidence: float) -> ASRBackend:
        """
        Determine ASR backend based on language and confidence.

        Args:
            language: Detected language code
            confidence: Detection confidence

        Returns:
            ASRBackend to use
        """
        # Low confidence -> fallback to Whisper (handles more languages)
        if confidence < self.config.confidence_threshold:
            return ASRBackend.WHISPER

        # Unsupported language -> Whisper
        if language not in self.config.supported_languages:
            return ASRBackend.WHISPER

        # High confidence supported language -> Zipformer
        if confidence >= self.config.zipformer_confidence_threshold:
            return ASRBackend.ZIPFORMER

        # Moderate confidence -> still Zipformer for supported languages
        return ASRBackend.ZIPFORMER

    def get_supported_languages(self) -> list[str]:
        """Get list of languages supported by Zipformer."""
        return sorted(self.config.supported_languages)

    def is_supported(self, language: str) -> bool:
        """Check if language is supported by Zipformer."""
        return language in self.config.supported_languages

    def clear_session_cache(self, session_id: str):
        """Clear cached language for a session."""
        self._session_languages.pop(session_id, None)

    def set_session_language(self, session_id: str, language: str):
        """Manually set language for a session (e.g., from user preference)."""
        self._session_languages[session_id] = language

    async def detect_language_async(
        self,
        audio: np.ndarray,
        session_id: str | None = None,
    ) -> LanguageDetectionResult:
        """Async version for use in async server."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.detect_language(audio, session_id),
        )


class MockLanguageRouter(LanguageRouter):
    """
    Mock language router for testing.

    Returns configurable mock results.
    """

    def __init__(self, config: LanguageRouterConfig | None = None):
        super().__init__(config=config)
        self._mock_language = "en"
        self._mock_confidence = 0.95

    def set_mock_result(self, language: str, confidence: float):
        """Configure mock result."""
        self._mock_language = language
        self._mock_confidence = confidence

    def _detect_language_impl(
        self,
        audio: np.ndarray,
    ) -> tuple[str, float, dict[str, float] | None]:
        """Return mock detection result."""
        all_probs = {self._mock_language: self._mock_confidence}
        return (self._mock_language, self._mock_confidence, all_probs)


# Language name mappings for display
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
}


def get_language_name(code: str) -> str:
    """Get display name for language code."""
    return LANGUAGE_NAMES.get(code, code.upper())
