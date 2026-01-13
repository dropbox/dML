#!/usr/bin/env python3
"""
Semantic Similarity Scoring for Multilingual TTS Tests

Uses GPT-4o to evaluate semantic similarity between expected and actual
transcriptions. This is more robust than keyword matching for:
- Honorific variations (Korean: 안녕하세요 vs 안녕하시오)
- Synonyms and near-synonyms
- Transcription variations that preserve meaning

Worker #274: Initial implementation
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Load .env file for OPENAI_API_KEY
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def semantic_similarity(expected: str, actual: str, language: str) -> Dict:
    """
    Use LLM to evaluate semantic similarity between expected and actual text.

    Args:
        expected: The expected text (original input)
        actual: The actual transcription
        language: Language name (e.g., "Korean", "Japanese", "English")

    Returns:
        dict with keys:
        - score: float 0.0-1.0 (1.0 = same meaning)
        - reason: str explaining the score
        - same_meaning: bool (True if score >= 0.7)
        - error: str if API call failed
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {
            "score": 0.0,
            "reason": "openai package not installed",
            "same_meaning": False,
            "error": "openai package not installed"
        }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "score": 0.0,
            "reason": "OPENAI_API_KEY not set",
            "same_meaning": False,
            "error": "OPENAI_API_KEY not set"
        }

    # Quick exact match check
    if expected.strip() == actual.strip():
        return {
            "score": 1.0,
            "reason": "Exact match",
            "same_meaning": True
        }

    # Build prompt
    prompt = f"""Compare these two texts in {language} for semantic similarity.

Expected: {expected}
Actual: {actual}

Score from 0.0 to 1.0 where:
- 1.0 = Identical or same meaning (including honorific/formality variations)
- 0.8-0.9 = Very similar meaning, minor differences
- 0.5-0.7 = Related but different meaning
- 0.0-0.4 = Unrelated or opposite meaning

Examples for Korean:
- 안녕하세요 vs 안녕하시오 = 0.95 (both mean "hello", different formality)
- 안녕하세요 vs 고맙습니다 = 0.1 (hello vs thank you - different meanings)
- 감사합니다 vs 고맙습니다 = 0.95 (both mean "thank you")

Examples for Japanese:
- こんにちは vs 今日は = 1.0 (same greeting, different writing)
- おはよう vs おはようございます = 0.95 (both "good morning", different formality)

Respond with JSON only: {{"score": 0.X, "reason": "brief explanation"}}"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.1  # Low temp for consistency
        )

        result = json.loads(response.choices[0].message.content)
        result["same_meaning"] = result.get("score", 0.0) >= 0.7
        return result

    except json.JSONDecodeError as e:
        return {
            "score": 0.0,
            "reason": f"JSON parse error: {e}",
            "same_meaning": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "score": 0.0,
            "reason": f"API error: {e}",
            "same_meaning": False,
            "error": str(e)
        }


def batch_similarity(pairs: list, language: str) -> list:
    """
    Evaluate semantic similarity for multiple text pairs.

    Args:
        pairs: List of (expected, actual) tuples
        language: Language name

    Returns:
        List of result dicts
    """
    results = []
    for expected, actual in pairs:
        result = semantic_similarity(expected, actual, language)
        result["expected"] = expected
        result["actual"] = actual
        results.append(result)
    return results


# CLI for testing
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: semantic_similarity.py <expected> <actual> <language>")
        print("Example: semantic_similarity.py '안녕하세요' '안녕하시오' Korean")
        sys.exit(1)

    expected = sys.argv[1]
    actual = sys.argv[2]
    language = sys.argv[3]

    result = semantic_similarity(expected, actual, language)
    print(json.dumps(result, indent=2, ensure_ascii=False))
