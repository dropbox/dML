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

"""Tests for language routing (Phase 9.3)."""

import numpy as np
import pytest

# Module-level random generator for reproducibility
_rng = np.random.default_rng(42)

from src.server.language_router import (
    LANGUAGE_NAMES,
    ASRBackend,
    LanguageDetectionResult,
    LanguageRouter,
    LanguageRouterConfig,
    MockLanguageRouter,
    get_language_name,
)


class TestASRBackend:
    """Tests for ASRBackend enum."""

    def test_all_backends(self):
        assert ASRBackend.ZIPFORMER.value == "zipformer"
        assert ASRBackend.WHISPER.value == "whisper"
        assert ASRBackend.ROVER.value == "rover"

    def test_from_string(self):
        assert ASRBackend("zipformer") == ASRBackend.ZIPFORMER
        assert ASRBackend("whisper") == ASRBackend.WHISPER


class TestLanguageDetectionResult:
    """Tests for LanguageDetectionResult dataclass."""

    def test_basic_creation(self):
        result = LanguageDetectionResult(
            language="en",
            confidence=0.95,
            backend=ASRBackend.ZIPFORMER,
        )
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.backend == ASRBackend.ZIPFORMER
        assert result.all_probs is None

    def test_with_all_probs(self):
        result = LanguageDetectionResult(
            language="es",
            confidence=0.8,
            backend=ASRBackend.ZIPFORMER,
            all_probs={"es": 0.8, "en": 0.15, "pt": 0.05},
        )
        assert len(result.all_probs) == 3


class TestLanguageRouterConfig:
    """Tests for LanguageRouterConfig dataclass."""

    def test_default_config(self):
        config = LanguageRouterConfig()
        assert "en" in config.supported_languages
        assert config.confidence_threshold == 0.7
        assert config.default_language == "en"

    def test_custom_config(self):
        config = LanguageRouterConfig(
            supported_languages={"en", "es"},
            confidence_threshold=0.8,
            default_language="es",
        )
        assert len(config.supported_languages) == 2
        assert config.confidence_threshold == 0.8
        assert config.default_language == "es"

    def test_supported_languages(self):
        config = LanguageRouterConfig()
        # Check all 9 core languages are supported
        expected = {"en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru"}
        assert config.supported_languages == expected


class TestLanguageRouter:
    """Tests for LanguageRouter."""

    @pytest.fixture
    def router(self):
        return LanguageRouter()

    def test_detect_language_short_audio(self, router):
        # Very short audio (less than min_detection_duration)
        audio = _rng.standard_normal(1600).astype(np.float32)  # 100ms
        result = router.detect_language(audio)
        # Should fall back to default
        assert result.language == "en"
        assert result.confidence == 0.5  # Low confidence for short audio

    def test_detect_language_with_session(self, router):
        audio = _rng.standard_normal(16000).astype(np.float32)  # 1s
        result = router.detect_language(audio, session_id="test123")
        assert result.language is not None

        # Short audio should use cached language
        short_audio = _rng.standard_normal(1600).astype(np.float32)
        result2 = router.detect_language(short_audio, session_id="test123")
        assert result2.language == result.language

    def test_get_supported_languages(self, router):
        languages = router.get_supported_languages()
        assert "en" in languages
        assert isinstance(languages, list)
        assert languages == sorted(languages)  # Should be sorted

    def test_is_supported(self, router):
        assert router.is_supported("en")
        assert router.is_supported("es")
        assert not router.is_supported("zh")  # Chinese not in default
        assert not router.is_supported("ja")  # Japanese not in default

    def test_clear_session_cache(self, router):
        audio = _rng.standard_normal(16000).astype(np.float32)
        router.detect_language(audio, session_id="test")
        router.clear_session_cache("test")
        assert "test" not in router._session_languages

    def test_set_session_language(self, router):
        router.set_session_language("test", "fr")
        # Short audio should use set language
        audio = _rng.standard_normal(1600).astype(np.float32)
        result = router.detect_language(audio, session_id="test")
        assert result.language == "fr"

    @pytest.mark.asyncio
    async def test_detect_language_async(self, router):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = await router.detect_language_async(audio)
        assert isinstance(result, LanguageDetectionResult)


class TestLanguageRouterBackendSelection:
    """Tests for backend selection logic."""

    def test_high_confidence_supported_language(self):
        router = LanguageRouter()
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = router.detect_language(audio)
        # Default fallback returns en with 0.85 confidence
        assert result.backend == ASRBackend.ZIPFORMER

    def test_low_confidence_uses_whisper(self):
        config = LanguageRouterConfig(confidence_threshold=0.9)
        router = LanguageRouter(config=config)
        # Default impl returns 0.85 confidence, which is below 0.9 threshold
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = router.detect_language(audio)
        assert result.backend == ASRBackend.WHISPER

    def test_unsupported_language_uses_whisper(self):
        router = MockLanguageRouter()
        router.set_mock_result("zh", 0.95)  # Chinese, high confidence
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = router.detect_language(audio)
        assert result.backend == ASRBackend.WHISPER


class TestMockLanguageRouter:
    """Tests for MockLanguageRouter."""

    @pytest.fixture
    def mock_router(self):
        return MockLanguageRouter()

    def test_default_result(self, mock_router):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = mock_router.detect_language(audio)
        assert result.language == "en"
        assert result.confidence == 0.95

    def test_set_mock_result(self, mock_router):
        mock_router.set_mock_result("fr", 0.88)
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = mock_router.detect_language(audio)
        assert result.language == "fr"
        assert result.confidence == 0.88

    def test_mock_all_probs(self, mock_router):
        mock_router.set_mock_result("de", 0.92)
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = mock_router.detect_language(audio)
        assert result.all_probs is not None
        assert "de" in result.all_probs

    def test_mock_unsupported_language(self, mock_router):
        mock_router.set_mock_result("ja", 0.9)  # Japanese
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = mock_router.detect_language(audio)
        assert result.language == "ja"
        assert result.backend == ASRBackend.WHISPER


class TestLanguageNames:
    """Tests for language name utilities."""

    def test_language_names_dict(self):
        assert LANGUAGE_NAMES["en"] == "English"
        assert LANGUAGE_NAMES["es"] == "Spanish"
        assert LANGUAGE_NAMES["zh"] == "Chinese"
        assert LANGUAGE_NAMES["ja"] == "Japanese"

    def test_get_language_name_known(self):
        assert get_language_name("en") == "English"
        assert get_language_name("de") == "German"
        assert get_language_name("ru") == "Russian"

    def test_get_language_name_unknown(self):
        # Unknown codes should return uppercase version
        assert get_language_name("xx") == "XX"
        assert get_language_name("abc") == "ABC"

    def test_all_supported_have_names(self):
        config = LanguageRouterConfig()
        for lang in config.supported_languages:
            name = get_language_name(lang)
            assert name != lang.upper()  # Should have proper name


class TestLanguageRouterIntegration:
    """Integration tests for language router."""

    def test_router_with_config(self):
        config = LanguageRouterConfig(
            supported_languages={"en", "es", "fr"},
            confidence_threshold=0.75,
            zipformer_confidence_threshold=0.85,
        )
        router = LanguageRouter(config=config)
        assert len(router.config.supported_languages) == 3

    def test_session_caching(self):
        router = LanguageRouter()

        # First detection
        audio1 = _rng.standard_normal(16000).astype(np.float32)
        result1 = router.detect_language(audio1, session_id="session1")

        # Short audio uses cache
        audio2 = _rng.standard_normal(1600).astype(np.float32)
        result2 = router.detect_language(audio2, session_id="session1")

        assert result2.language == result1.language

        # Different session gets its own cache
        _result3 = router.detect_language(audio1, session_id="session2")
        # Cache for session2 should now exist
        assert "session2" in router._session_languages

    def test_multiple_languages(self):
        mock = MockLanguageRouter()

        # Test routing for different languages
        test_cases = [
            ("en", 0.95, ASRBackend.ZIPFORMER),
            ("es", 0.90, ASRBackend.ZIPFORMER),
            ("zh", 0.95, ASRBackend.WHISPER),  # Unsupported
            ("en", 0.5, ASRBackend.WHISPER),  # Low confidence
        ]

        audio = _rng.standard_normal(16000).astype(np.float32)

        for lang, conf, expected_backend in test_cases:
            mock.set_mock_result(lang, conf)
            result = mock.detect_language(audio)
            assert result.backend == expected_backend, f"Failed for {lang} @ {conf}"
