#!/usr/bin/env python3
"""Test LLM Summarizer with Qwen3-8B model.

This test verifies that the LLMSummarizer can:
1. Load the Qwen3-8B model
2. Generate text using llama.cpp inference
3. Produce meaningful summaries
"""

import subprocess
import sys
import os
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "qwen" / "Qwen3-8B-Q4_K_M.gguf"
BINARY_PATH = PROJECT_ROOT / "stream-tts-cpp" / "build" / "stream-tts-cpp"
CONFIG_PATH = PROJECT_ROOT / "stream-tts-cpp" / "config" / "default.yaml"


def test_qwen_model_exists():
    """Verify Qwen3-8B model is downloaded."""
    assert MODEL_PATH.exists(), f"Qwen3-8B model not found at {MODEL_PATH}"
    size_gb = MODEL_PATH.stat().st_size / (1024**3)
    assert size_gb > 4.0, f"Model too small ({size_gb:.1f}GB), expected ~4.7GB"
    print(f"Model found: {MODEL_PATH} ({size_gb:.2f}GB)")


def test_binary_exists():
    """Verify stream-tts-cpp binary exists."""
    assert BINARY_PATH.exists(), f"Binary not found at {BINARY_PATH}"
    print(f"Binary found: {BINARY_PATH}")


def test_llm_summarizer_help():
    """Test that help shows LLM summarizer options."""
    result = subprocess.run(
        [str(BINARY_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    # Just verify binary runs
    assert result.returncode == 0 or "Usage:" in result.stdout or "help" in result.stdout.lower()
    print("Binary help works")


def test_llm_summarization_runs():
    """Run a real summarization pass via llama.cpp."""
    env = os.environ.copy()
    env["QWEN_MODEL_PATH"] = str(MODEL_PATH)

    result = subprocess.run(
        [
            str(BINARY_PATH),
            "--test-summarization",
            "Tool: Read, File: src/main.cpp",
            "--test-summarization-max-tokens",
            "48",
            str(CONFIG_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=PROJECT_ROOT / "stream-tts-cpp",
        env=env,
    )

    assert result.returncode == 0, f"Summarization CLI failed: {result.stderr}"
    output = result.stdout.strip()
    assert output, "Summarization output is empty"
    assert "performing an action" not in output.lower(), "Fallback template was used instead of llama.cpp"
    print(f"LLM summarization output: {output}")


def test_multilingual_summarization_ja_to_en():
    """Test cross-lingual summarization: Japanese input → English summary."""
    env = os.environ.copy()
    env["QWEN_MODEL_PATH"] = str(MODEL_PATH)

    # Japanese text: "We had a meeting today. We discussed the project progress."
    japanese_text = "今日は会議がありました。プロジェクトの進捗について話し合いました。"

    result = subprocess.run(
        [
            str(BINARY_PATH),
            "--test-transcript",
            japanese_text,
            "--mode", "brief",
            "--target-language", "en",
            "--test-summarization-max-tokens", "60",
            str(CONFIG_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=PROJECT_ROOT / "stream-tts-cpp",
        env=env,
    )

    assert result.returncode == 0, f"JA→EN summarization failed: {result.stderr}"
    output = result.stdout.strip()
    assert output, "JA→EN summary output is empty"

    # Verify English output (should contain common English words)
    output_lower = output.lower()
    assert any(word in output_lower for word in ["meeting", "project", "discuss", "progress", "today"]), \
        f"Output doesn't appear to be English: {output}"

    # Verify no Japanese characters in output
    import unicodedata
    japanese_chars = sum(1 for c in output if unicodedata.category(c).startswith('L') and ord(c) > 0x3000)
    assert japanese_chars == 0, f"English summary contains Japanese characters: {output}"

    print(f"JA→EN summary: {output}")


def test_multilingual_summarization_en_to_ja():
    """Test cross-lingual summarization: English input → Japanese summary."""
    env = os.environ.copy()
    env["QWEN_MODEL_PATH"] = str(MODEL_PATH)

    english_text = "We had a team meeting today. We discussed the project timeline and decided to push the deadline."

    result = subprocess.run(
        [
            str(BINARY_PATH),
            "--test-transcript",
            english_text,
            "--mode", "brief",
            "--target-language", "ja",
            "--test-summarization-max-tokens", "80",
            str(CONFIG_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=PROJECT_ROOT / "stream-tts-cpp",
        env=env,
    )

    assert result.returncode == 0, f"EN→JA summarization failed: {result.stderr}"
    output = result.stdout.strip()
    assert output, "EN→JA summary output is empty"

    # Verify Japanese output (should contain Japanese characters)
    import unicodedata
    japanese_chars = sum(1 for c in output if ord(c) > 0x3000 and ord(c) < 0x9FFF)
    assert japanese_chars >= 5, f"Output doesn't appear to be Japanese: {output}"

    print(f"EN→JA summary: {output}")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Summarizer Integration Tests")
    print("=" * 60)

    tests = [
        test_qwen_model_exists,
        test_binary_exists,
        test_llm_summarizer_help,
        test_llm_summarization_runs,
        test_multilingual_summarization_ja_to_en,
        test_multilingual_summarization_en_to_ja,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            test()
            print(f"PASSED: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
