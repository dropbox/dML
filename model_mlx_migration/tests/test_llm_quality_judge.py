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

"""Tests for LLM Quality Judge Framework."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.evaluation.llm_quality_judge import (
    LLMJudge,
    LocalBackend,
    QualityScore,
)


class TestQualityScore:
    """Test QualityScore dataclass."""

    def test_quality_score_creation(self):
        """Test creating a QualityScore."""
        score = QualityScore(
            score=85.0,
            feedback="Good translation",
            breakdown={"accuracy": 90, "fluency": 80},
            model="test/model",
        )
        assert score.score == 85.0
        assert score.feedback == "Good translation"
        assert score.breakdown["accuracy"] == 90
        assert score.model == "test/model"

    def test_quality_score_zero(self):
        """Test QualityScore with zero values."""
        score = QualityScore(
            score=0.0,
            feedback="Evaluation failed",
            breakdown={},
            model="failed/model",
        )
        assert score.score == 0.0
        assert score.breakdown == {}


class TestLocalBackendAliases:
    """Test LocalBackend model aliases."""

    def test_model_aliases_exist(self):
        """Verify expected model aliases are defined."""
        expected_aliases = [
            "llama-3.2-3b",
            "llama-3.2-1b",
            "mistral-7b",
            "qwen-2.5-7b",
            "qwen-2.5-3b",
        ]
        for alias in expected_aliases:
            assert alias in LocalBackend.MODEL_ALIASES

    def test_alias_values_are_mlx_community(self):
        """Test that aliases point to mlx-community models."""
        for alias, full_path in LocalBackend.MODEL_ALIASES.items():
            assert full_path.startswith("mlx-community/"), f"{alias} should point to mlx-community"

    @pytest.mark.slow
    def test_alias_resolution_integration(self):
        """Test that aliases resolve correctly (requires mlx-lm)."""
        try:
            backend = LocalBackend("llama-3.2-3b")
            assert backend.model == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        except ImportError:
            pytest.skip("mlx-lm not installed")

    @pytest.mark.slow
    def test_full_path_passthrough_integration(self):
        """Test that full model paths are not modified (requires mlx-lm)."""
        try:
            custom_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            backend = LocalBackend(custom_path)
            assert backend.model == custom_path
        except ImportError:
            pytest.skip("mlx-lm not installed")


class TestJSONExtraction:
    """Test JSON extraction from LLM responses.

    Note: We test the extraction logic directly since it's a static algorithm
    that doesn't depend on the model loading.
    """

    def _extract_json(self, response: str) -> str:
        """Extract JSON - reimplemented here for unit testing."""
        start = response.find('{')
        if start == -1:
            return response

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(response[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return response[start:i + 1]

        return response

    def test_extract_simple_json(self):
        """Test extracting simple JSON."""
        response = '{"score": 85, "feedback": "Good"}'
        result = self._extract_json(response)
        assert result == '{"score": 85, "feedback": "Good"}'

    def test_extract_json_with_prefix(self):
        """Test extracting JSON with text before it."""
        response = 'Here is my evaluation: {"score": 85, "feedback": "Good"}'
        result = self._extract_json(response)
        assert result == '{"score": 85, "feedback": "Good"}'

    def test_extract_json_with_suffix(self):
        """Test extracting JSON followed by additional text."""
        response = '{"score": 85} Let me explain why...'
        result = self._extract_json(response)
        assert result == '{"score": 85}'

    def test_extract_nested_json(self):
        """Test extracting nested JSON objects."""
        response = '{"outer": {"inner": 42}, "value": "test"}'
        result = self._extract_json(response)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == 42

    def test_extract_json_with_strings(self):
        """Test JSON extraction handles strings with special chars."""
        response = '{"feedback": "The text has {braces} and \\"quotes\\""}'
        result = self._extract_json(response)
        # Should extract the complete JSON
        assert result.startswith('{"feedback":')

    def test_no_json_returns_original(self):
        """Test that non-JSON responses are returned as-is."""
        response = "This is just plain text"
        result = self._extract_json(response)
        assert result == response


class TestLLMJudge:
    """Test LLMJudge class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend for testing."""
        backend = MagicMock()
        backend.model = "test-model"
        return backend

    def test_parse_json_response_simple(self):
        """Test parsing simple JSON response."""
        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            judge = LLMJudge.__new__(LLMJudge)
            response = '{"score": 90, "feedback": "Excellent"}'
            result = judge._parse_json_response(response)
            assert result["score"] == 90
            assert result["feedback"] == "Excellent"

    def test_parse_json_with_markdown_block(self):
        """Test parsing JSON in markdown code block."""
        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            judge = LLMJudge.__new__(LLMJudge)
            response = """Here is my evaluation:
```json
{"score": 85, "feedback": "Good"}
```
"""
            result = judge._parse_json_response(response)
            assert result["score"] == 85

    def test_parse_json_invalid(self):
        """Test handling of invalid JSON."""
        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            judge = LLMJudge.__new__(LLMJudge)
            response = "This is not JSON at all"
            result = judge._parse_json_response(response)
            assert "error" in result

    def test_evaluate_translation_success(self, mock_backend):
        """Test successful translation evaluation."""
        # Mock successful response
        mock_backend.complete.return_value = json.dumps({
            "semantic_accuracy": 95,
            "fluency": 90,
            "terminology": 85,
            "cultural_appropriateness": 100,
            "overall": 92,
            "feedback": "Accurate translation",
        })

        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            judge = LLMJudge.__new__(LLMJudge)
            judge.backend = mock_backend
            judge.backend_name = "test"
            judge.model_name = "test-model"

            score = judge.evaluate_translation(
                "Hello", "Bonjour", "en", "fr",
            )

            assert score.score == 92
            assert score.breakdown["semantic_accuracy"] == 95
            assert "Accurate" in score.feedback

    def test_evaluate_tts_success(self, mock_backend):
        """Test successful TTS evaluation."""
        mock_backend.complete.return_value = json.dumps({
            "accuracy": 100,
            "intelligibility": 95,
            "overall": 97,
            "feedback": "Clear speech",
        })

        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            judge = LLMJudge.__new__(LLMJudge)
            judge.backend = mock_backend
            judge.backend_name = "test"
            judge.model_name = "test-model"

            score = judge.evaluate_tts(
                "Hello world", "Hello world",
            )

            assert score.score == 97
            assert score.breakdown["accuracy"] == 100

    def test_evaluate_handles_parse_error(self, mock_backend):
        """Test handling of unparseable response."""
        mock_backend.complete.return_value = "Invalid response"

        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            judge = LLMJudge.__new__(LLMJudge)
            judge.backend = mock_backend
            judge.backend_name = "test"
            judge.model_name = "test-model"

            score = judge.evaluate_translation(
                "Hello", "Bonjour", "en", "fr",
            )

            assert score.score == 0
            assert "failed" in score.feedback.lower()


class TestBatchEvaluation:
    """Test batch evaluation functionality."""

    def test_batch_evaluate_translations(self):
        """Test batch evaluation of multiple translations."""
        mock_backend = MagicMock()
        mock_backend.model = "test-model"
        mock_backend.complete.side_effect = [
            json.dumps({
                "semantic_accuracy": 90, "fluency": 90,
                "terminology": 90, "cultural_appropriateness": 90,
                "overall": 90, "feedback": "Good",
            }),
            json.dumps({
                "semantic_accuracy": 80, "fluency": 80,
                "terminology": 80, "cultural_appropriateness": 80,
                "overall": 80, "feedback": "OK",
            }),
        ]

        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            judge = LLMJudge.__new__(LLMJudge)
            judge.backend = mock_backend
            judge.backend_name = "test"
            judge.model_name = "test-model"

            samples = [
                ("Hello", "Bonjour", "en", "fr"),
                ("Goodbye", "Au revoir", "en", "fr"),
            ]

            results = judge.batch_evaluate_translations(samples)

            assert len(results) == 2
            assert results[0].score == 90
            assert results[1].score == 80


@pytest.mark.slow
class TestLocalBackendIntegration:
    """Integration tests that use the actual mlx-lm backend.

    These tests are marked slow and require mlx-lm to be installed
    with models downloaded.
    """

    @pytest.fixture
    def judge(self):
        """Create an LLMJudge with local backend."""
        try:
            return LLMJudge(backend="local", model="llama-3.2-3b")
        except ImportError:
            pytest.skip("mlx-lm not installed")

    def test_real_translation_evaluation(self, judge):
        """Test real translation evaluation with local model."""
        score = judge.evaluate_translation(
            source_text="Hello, how are you?",
            translated_text="Bonjour, comment allez-vous?",
            source_lang="en",
            target_lang="fr",
        )

        # Should return a valid score
        assert isinstance(score.score, (int, float))
        assert 0 <= score.score <= 100
        assert score.feedback  # Should have some feedback
        assert "local" in score.model

    def test_real_tts_evaluation_perfect(self, judge):
        """Test TTS evaluation with perfect transcription."""
        score = judge.evaluate_tts(
            original_text="The quick brown fox jumps over the lazy dog.",
            transcription="The quick brown fox jumps over the lazy dog.",
        )

        # Perfect match should get high score
        assert score.score >= 90
        assert score.breakdown.get("accuracy", 0) >= 90

    def test_real_tts_evaluation_error(self, judge):
        """Test TTS evaluation with transcription error."""
        score = judge.evaluate_tts(
            original_text="Hello world",
            transcription="Hello word",  # Missing 'l'
        )

        # Should detect the error
        assert score.score < 100
        # Feedback should mention the error
        assert score.feedback
