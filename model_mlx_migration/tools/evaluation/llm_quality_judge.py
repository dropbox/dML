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
LLM-as-Judge Quality Evaluation Framework

Supports multiple LLM backends for evaluating:
- Translation quality (semantic accuracy, fluency, naturalness)
- TTS quality (from transcriptions)
- Overall pipeline quality

Usage:
    judge = LLMJudge(backend="openai", model="gpt-5.2")
    score = judge.evaluate_translation(source, target, source_lang, target_lang)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Result of a quality evaluation."""
    score: float  # 0-100
    feedback: str
    breakdown: dict  # Component scores
    model: str  # Model used for evaluation


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Generate a completion from the LLM."""


class OpenAIBackend(LLMBackend):
    """OpenAI GPT models backend."""

    def __init__(self, model: str = "gpt-5.2", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai") from None

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent evaluation
        )
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """Anthropic Claude models backend."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from None

    def complete(self, prompt: str, system: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system if system else None,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class LocalBackend(LLMBackend):
    """Local model backend using mlx-lm for Apple Silicon inference."""

    # Map of short names to full model paths
    MODEL_ALIASES = {
        "mistral-7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "llama-3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "qwen-2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "qwen-2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "gemma-2-9b": "mlx-community/gemma-2-9b-it-4bit",
        "phi-3.5-mini": "mlx-community/Phi-3.5-mini-instruct-4bit",
    }

    def __init__(self, model: str = "llama-3.2-3b"):
        """Initialize local MLX backend.

        Args:
            model: Model identifier - either a short name (e.g., 'llama-3.2-3b')
                   or full HuggingFace path (e.g., 'mlx-community/Llama-3.2-3B-Instruct-4bit')
        """
        # Resolve short name to full path
        self.model = self.MODEL_ALIASES.get(model, model)

        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
            self._load = load
            self._generate = generate
            self._make_sampler = make_sampler
        except ImportError:
            raise ImportError("mlx-lm package not installed. Run: pip install mlx-lm") from None

        # Lazy loading - models are loaded on first use
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Load model and tokenizer if not already loaded."""
        if self._model is None:
            print(f"Loading local model: {self.model}")
            self._model, self._tokenizer = self._load(self.model)
            print("Model loaded successfully")

    def complete(self, prompt: str, system: str = "") -> str:
        """Generate a completion using mlx-lm.

        Args:
            prompt: User prompt
            system: System prompt (prepended to conversation)

        Returns:
            Generated text response
        """
        self._ensure_loaded()

        # Use tokenizer's chat template if available (Llama 3.2 Instruct format)
        if hasattr(self._tokenizer, 'apply_chat_template'):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            full_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            # Fallback for tokenizers without chat template
            if system:
                full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
            else:
                full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"

        # Create sampler with low temperature for consistent evaluation
        sampler = self._make_sampler(temp=0.1)

        # Generate with reasonable defaults for evaluation tasks
        response = self._generate(
            self._model,
            self._tokenizer,
            prompt=full_prompt,
            max_tokens=512,  # Shorter to avoid rambling
            sampler=sampler,
            verbose=False,
        )

        # Extract JSON from response if it continues past the JSON block
        return self._extract_json_response(response)


    def _extract_json_response(self, response: str) -> str:
        """Extract the first complete JSON object from a response."""
        # Find the first { and try to find matching }
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

        return response  # Return full response if no complete JSON found


class LLMJudge:
    """LLM-as-Judge for quality evaluation."""

    TRANSLATION_SYSTEM_PROMPT = """You are an expert multilingual translation quality evaluator.
Evaluate the translation on these criteria:
1. Semantic accuracy (0-100): Does the translation convey the same meaning?
2. Fluency (0-100): Is the translation grammatically correct and natural?
3. Terminology (0-100): Are domain-specific terms translated correctly?
4. Cultural appropriateness (0-100): Is the translation culturally appropriate?

Provide your response as JSON with this exact format:
{
    "semantic_accuracy": <score>,
    "fluency": <score>,
    "terminology": <score>,
    "cultural_appropriateness": <score>,
    "overall": <weighted_average>,
    "feedback": "<brief explanation>"
}"""

    TTS_SYSTEM_PROMPT = """You are an expert TTS quality evaluator.
Given the original text and the transcription of synthesized audio, evaluate:
1. Accuracy (0-100): Does the transcription match the original?
2. Intelligibility (0-100): Based on transcription quality, is speech clear?

Provide your response as JSON:
{
    "accuracy": <score>,
    "intelligibility": <score>,
    "overall": <weighted_average>,
    "feedback": "<brief explanation>"
}"""

    def __init__(
        self,
        backend: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the LLM judge.

        Args:
            backend: "openai", "anthropic", or "local"
            model: Specific model to use (defaults vary by backend)
            api_key: API key (uses environment variable if not provided)
        """
        if backend == "openai":
            self.backend = OpenAIBackend(model or "gpt-5.2", api_key)
        elif backend == "anthropic":
            self.backend = AnthropicBackend(model or "claude-3-5-sonnet-20241022", api_key)
        elif backend == "local":
            self.backend = LocalBackend(model or "llama-3.2-3b")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend_name = backend
        self.model_name = model or self.backend.model

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Handle markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"error": "Failed to parse response", "raw": response}

    def evaluate_translation(
        self,
        source_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
    ) -> QualityScore:
        """Evaluate translation quality.

        Args:
            source_text: Original text
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            QualityScore with breakdown and feedback
        """
        prompt = f"""Evaluate this translation:

Source ({source_lang}): {source_text}
Translation ({target_lang}): {translated_text}

Evaluate the quality and provide your assessment as JSON."""

        response = self.backend.complete(prompt, self.TRANSLATION_SYSTEM_PROMPT)
        result = self._parse_json_response(response)

        if "error" in result:
            return QualityScore(
                score=0,
                feedback=f"Evaluation failed: {result.get('error')}",
                breakdown={},
                model=f"{self.backend_name}/{self.model_name}",
            )

        return QualityScore(
            score=result.get("overall", 0),
            feedback=result.get("feedback", ""),
            breakdown={
                "semantic_accuracy": result.get("semantic_accuracy", 0),
                "fluency": result.get("fluency", 0),
                "terminology": result.get("terminology", 0),
                "cultural_appropriateness": result.get("cultural_appropriateness", 0),
            },
            model=f"{self.backend_name}/{self.model_name}",
        )

    def evaluate_tts(
        self,
        original_text: str,
        transcription: str,
    ) -> QualityScore:
        """Evaluate TTS quality based on transcription.

        Args:
            original_text: Text that was synthesized
            transcription: Whisper transcription of the audio

        Returns:
            QualityScore with breakdown and feedback
        """
        prompt = f"""Evaluate this TTS output:

Original text: {original_text}
Transcription of audio: {transcription}

Evaluate the quality and provide your assessment as JSON."""

        response = self.backend.complete(prompt, self.TTS_SYSTEM_PROMPT)
        result = self._parse_json_response(response)

        if "error" in result:
            return QualityScore(
                score=0,
                feedback=f"Evaluation failed: {result.get('error')}",
                breakdown={},
                model=f"{self.backend_name}/{self.model_name}",
            )

        return QualityScore(
            score=result.get("overall", 0),
            feedback=result.get("feedback", ""),
            breakdown={
                "accuracy": result.get("accuracy", 0),
                "intelligibility": result.get("intelligibility", 0),
            },
            model=f"{self.backend_name}/{self.model_name}",
        )

    def batch_evaluate_translations(
        self,
        samples: list[tuple[str, str, str, str]],  # (source, target, src_lang, tgt_lang)
    ) -> list[QualityScore]:
        """Evaluate multiple translations.

        Args:
            samples: List of (source_text, translated_text, source_lang, target_lang)

        Returns:
            List of QualityScore results
        """
        results = []
        for source, target, src_lang, tgt_lang in samples:
            score = self.evaluate_translation(source, target, src_lang, tgt_lang)
            results.append(score)
        return results


def main():
    """Example usage."""
    print("LLM Quality Judge Framework")
    print("=" * 50)

    # Check available backends
    backends = []

    try:
        import openai  # noqa: F401
        if os.environ.get("OPENAI_API_KEY"):
            backends.append("openai")
    except ImportError:
        pass

    try:
        import anthropic  # noqa: F401
        if os.environ.get("ANTHROPIC_API_KEY"):
            backends.append("anthropic")
    except ImportError:
        pass

    # Check for local mlx-lm backend
    try:
        import mlx_lm  # noqa: F401
        backends.append("local")
    except ImportError:
        pass

    if not backends:
        print("No LLM backends available.")
        print("To enable evaluation:")
        print("  1. pip install openai  # or anthropic, or mlx-lm for local")
        print("  2. export OPENAI_API_KEY=your-key  # or ANTHROPIC_API_KEY")
        print("  3. For local: pip install mlx-lm (no API key needed)")
        return

    print(f"Available backends: {backends}")

    # Example evaluation
    judge = LLMJudge(backend=backends[0])

    score = judge.evaluate_translation(
        source_text="Hello, how are you?",
        translated_text="你好，你好吗？",
        source_lang="en",
        target_lang="zh",
    )

    print(f"\nTranslation Quality Score: {score.score}/100")
    print(f"Feedback: {score.feedback}")
    print(f"Breakdown: {score.breakdown}")


if __name__ == "__main__":
    main()
