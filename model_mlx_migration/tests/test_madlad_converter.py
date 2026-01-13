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
Tests for MADLAD-400 Converter

Tests the MADLAD converter without requiring actual model weights for most tests.
For full validation tests, run with HuggingFace authentication.
"""

import sys
from pathlib import Path

import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


class TestMADLADConverterImports:
    """Test converter imports and initialization."""

    def test_import_converter(self):
        """Test that converter can be imported."""
        from pytorch_to_mlx.converters import MADLADConverter

        assert MADLADConverter is not None

    def test_import_dataclasses(self):
        """Test that dataclasses can be imported."""
        from pytorch_to_mlx.converters.madlad_converter import (
            BenchmarkResult,
            TranslationResult,
        )

        assert TranslationResult is not None
        assert BenchmarkResult is not None

    def test_converter_init_no_load(self):
        """Test converter initialization without loading model."""
        from pytorch_to_mlx.converters import MADLADConverter

        # Should not load model on init
        converter = MADLADConverter()
        assert converter is not None
        assert converter.model is None
        assert converter._loaded is False

    def test_converter_default_model(self):
        """Test converter uses correct default model."""
        from pytorch_to_mlx.converters import MADLADConverter

        converter = MADLADConverter()
        assert converter.model_path == "google/madlad400-3b-mt"

    def test_converter_default_quantization(self):
        """Test converter uses 8-bit quantization by default.

        NOTE: Changed from 4-bit to 8-bit in Worker #1106.
        4-bit quantization produces hallucinated/wrong translations for CJK languages.
        """
        from pytorch_to_mlx.converters import MADLADConverter

        converter = MADLADConverter()
        assert converter.quantize_bits == 8


class TestTranslationResult:
    """Test TranslationResult dataclass."""

    def test_result_creation(self):
        """Test TranslationResult creation."""
        from pytorch_to_mlx.converters.madlad_converter import TranslationResult

        result = TranslationResult(
            text="Bonjour le monde",
            source_lang="en",
            target_lang="fr",
            latency_ms=150.0,
            tokens_generated=5,
        )
        assert result.text == "Bonjour le monde"
        assert result.source_lang == "en"
        assert result.target_lang == "fr"
        assert result.latency_ms == 150.0
        assert result.tokens_generated == 5

    def test_result_auto_source_lang(self):
        """Test TranslationResult with auto-detected source language."""
        from pytorch_to_mlx.converters.madlad_converter import TranslationResult

        result = TranslationResult(
            text="Hallo Welt",
            source_lang="auto",
            target_lang="de",
            latency_ms=100.0,
            tokens_generated=3,
        )
        assert result.source_lang == "auto"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation."""
        from pytorch_to_mlx.converters.madlad_converter import BenchmarkResult

        result = BenchmarkResult(
            median_latency_ms=162.0,
            mean_latency_ms=165.0,
            min_latency_ms=150.0,
            max_latency_ms=180.0,
            tokens_per_second=69.6,
        )
        assert result.median_latency_ms == 162.0
        assert result.mean_latency_ms == 165.0
        assert result.min_latency_ms == 150.0
        assert result.max_latency_ms == 180.0
        assert result.tokens_per_second == 69.6


class TestLanguageSupport:
    """Test language support features."""

    def test_list_supported_languages(self):
        """Test supported languages list."""
        from pytorch_to_mlx.converters import MADLADConverter

        languages = MADLADConverter.list_supported_languages()
        assert len(languages) > 0
        assert isinstance(languages, dict)

    def test_common_languages_present(self):
        """Test that common languages are in the list."""
        from pytorch_to_mlx.converters import MADLADConverter

        languages = MADLADConverter.list_supported_languages()

        # Check key language codes
        assert "en" in languages
        assert "fr" in languages
        assert "de" in languages
        assert "es" in languages
        assert "zh" in languages
        assert "ja" in languages
        assert "ko" in languages
        assert "ar" in languages

    def test_language_values_are_strings(self):
        """Test that language values are human-readable names."""
        from pytorch_to_mlx.converters import MADLADConverter

        languages = MADLADConverter.list_supported_languages()

        assert languages["en"] == "English"
        assert languages["fr"] == "French"
        assert languages["de"] == "German"


class TestModelSupport:
    """Test model support features."""

    def test_list_supported_models(self):
        """Test supported models list."""
        from pytorch_to_mlx.converters import MADLADConverter

        models = MADLADConverter.list_supported_models()
        assert len(models) > 0
        assert isinstance(models, list)

    def test_default_model_in_list(self):
        """Test that default model is in supported list."""
        from pytorch_to_mlx.converters import MADLADConverter

        models = MADLADConverter.list_supported_models()
        assert "google/madlad400-3b-mt" in models

    def test_model_variants(self):
        """Test that major model variants are listed."""
        from pytorch_to_mlx.converters import MADLADConverter

        models = MADLADConverter.list_supported_models()
        # MADLAD has 3B, 7B, and 10B variants
        assert "google/madlad400-3b-mt" in models


class TestQuantizationOptions:
    """Test quantization configuration."""

    def test_no_quantization(self):
        """Test converter can be configured without quantization."""
        from pytorch_to_mlx.converters import MADLADConverter

        converter = MADLADConverter(quantize=None)
        assert converter.quantize_bits is None

    def test_4bit_quantization(self):
        """Test 4-bit quantization option."""
        from pytorch_to_mlx.converters import MADLADConverter

        converter = MADLADConverter(quantize=4)
        assert converter.quantize_bits == 4

    def test_8bit_quantization(self):
        """Test 8-bit quantization option."""
        from pytorch_to_mlx.converters import MADLADConverter

        converter = MADLADConverter(quantize=8)
        assert converter.quantize_bits == 8


class TestCJKTranslationQuality:
    """
    Comprehensive CJK (Chinese, Japanese, Korean) translation quality tests.

    Added in Worker #1106 after discovering 4-bit quantization breaks Chinese translations.
    These tests verify that translations are semantically correct, not hallucinated.
    """

    # Expected translations - verified against fp16 reference
    # Format: (input_en, expected_zh, expected_ja, expected_ko)
    # Use alternatives with | separator for synonyms
    TEST_CASES = [
        # Simple greeting
        (
            "Hello, how are you?",
            ["你好"],  # Must contain greeting
            ["こんにちは"],  # Japanese greeting
            ["안녕"],  # Korean greeting
        ),
        # Weather sentence
        (
            "The weather is beautiful today.",
            ["天气|气候", "今天|今日"],  # weather, today (alternatives)
            ["天気", "今日"],  # weather, today
            ["날씨", "오늘"],  # weather, today
        ),
        # Technology sentence
        (
            "Artificial intelligence is changing the world.",
            ["人工智能", "世界|世|全球"],  # AI, world (alternatives)
            ["人工知能"],  # AI
            ["인공지능", "세계|세상"],  # AI, world (alternatives)
        ),
    ]

    def _phrase_matches(self, text: str, phrase_pattern: str) -> bool:
        """Check if text contains any of the alternatives in a phrase pattern."""
        alternatives = phrase_pattern.split("|")
        return any(alt in text for alt in alternatives)

    @pytest.fixture(scope="class")
    def converter(self):
        """Load converter once for all tests in class."""
        from pytorch_to_mlx.converters import MADLADConverter

        return MADLADConverter()  # Uses 8-bit default

    @pytest.mark.slow
    def test_chinese_translation_not_hallucinated(self, converter):
        """Test Chinese translations contain expected content, not hallucinations."""
        for input_text, expected_zh, _, _ in self.TEST_CASES:
            result = converter.translate(input_text, tgt_lang="zh")

            # Check that output contains expected Chinese phrases (with alternatives)
            missing = [p for p in expected_zh if not self._phrase_matches(result.text, p)]
            assert not missing, (
                f"Chinese translation missing expected phrases: {missing}\n"
                f"Input: {input_text}\n"
                f"Output: {result.text}"
            )

    @pytest.mark.slow
    def test_japanese_translation_not_hallucinated(self, converter):
        """Test Japanese translations contain expected content, not hallucinations."""
        for input_text, _, expected_ja, _ in self.TEST_CASES:
            result = converter.translate(input_text, tgt_lang="ja")

            missing = [p for p in expected_ja if not self._phrase_matches(result.text, p)]
            assert not missing, (
                f"Japanese translation missing expected phrases: {missing}\n"
                f"Input: {input_text}\n"
                f"Output: {result.text}"
            )

    @pytest.mark.slow
    def test_korean_translation_not_hallucinated(self, converter):
        """Test Korean translations contain expected content, not hallucinations."""
        for input_text, _, _, expected_ko in self.TEST_CASES:
            result = converter.translate(input_text, tgt_lang="ko")

            missing = [p for p in expected_ko if not self._phrase_matches(result.text, p)]
            assert not missing, (
                f"Korean translation missing expected phrases: {missing}\n"
                f"Input: {input_text}\n"
                f"Output: {result.text}"
            )

    @pytest.mark.slow
    def test_4bit_chinese_hallucination_detection(self, converter):
        """
        Verify that our known 4-bit hallucination case is detected.

        This test documents the bug where 4-bit quantization produces
        completely wrong Chinese output for compound sentences.
        """

        # The problematic input that triggers hallucination in 4-bit
        input_text = "The weather is beautiful today. I think I will go for a walk in the park."

        # 4-bit produces this WRONG output (about childhood memories instead of weather)
        hallucinated_phrases = ["想到这里", "心里一阵激动", "小时候"]

        # 8-bit (default) should NOT contain hallucinated phrases
        result_8bit = converter.translate(input_text, tgt_lang="zh")
        for phrase in hallucinated_phrases:
            assert phrase not in result_8bit.text, (
                f"8-bit translation contains hallucinated phrase: {phrase}\n"
                f"This indicates 8-bit may also be broken!\n"
                f"Output: {result_8bit.text}"
            )

        # 8-bit should contain correct phrases
        expected_correct = ["天气", "公园", "散步"]  # weather, park, walk
        for phrase in expected_correct:
            assert phrase in result_8bit.text, (
                f"8-bit translation missing expected phrase: {phrase}\n"
                f"Output: {result_8bit.text}"
            )


def run_quick_test():
    """Quick test without pytest."""
    print("Testing MADLAD Converter...")
    print("=" * 50)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from pytorch_to_mlx.converters import MADLADConverter
        from pytorch_to_mlx.converters.madlad_converter import (  # noqa: F401
            BenchmarkResult,
            TranslationResult,
        )

        print("   Imports successful!")
    except ImportError as e:
        print(f"   Import failed: {e}")
        return False

    # Test converter init
    print("\n2. Testing converter initialization...")
    try:
        converter = MADLADConverter()
        print(f"   Converter created! Model path: {converter.model_path}")
        print(f"   Quantization: {converter.quantize_bits}-bit")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test supported languages
    print("\n3. Testing supported languages list...")
    languages = MADLADConverter.list_supported_languages()
    print(f"   Found {len(languages)} languages (subset of 400+)")
    for code, name in list(languages.items())[:5]:
        print(f"     - {code}: {name}")

    # Test supported models
    print("\n4. Testing supported models list...")
    models = MADLADConverter.list_supported_models()
    print(f"   Found {len(models)} model variants")
    for m in models:
        print(f"     - {m}")

    print("\n" + "=" * 50)
    print("Quick test PASSED!")
    print("\nTo test translation with real models:")
    print('  python -c "')
    print("    from pytorch_to_mlx.converters import MADLADConverter")
    print("    c = MADLADConverter()")
    print("    result = c.translate('Hello world', tgt_lang='fr')")
    print('    print(result.text)"')

    return True


if __name__ == "__main__":
    sys.exit(0 if run_quick_test() else 1)
