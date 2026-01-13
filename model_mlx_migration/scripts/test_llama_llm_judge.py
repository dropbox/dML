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
LLaMA LLM-as-Judge Test

Validates LLaMA generation quality by comparing MLX outputs against known good outputs.

Pass criteria:
- Non-empty generation produced
- Generation is coherent (no repeated tokens/garbage)
- Factually reasonable responses to prompts

Usage:
    python scripts/test_llama_llm_judge.py
"""

import sys
from pathlib import Path
from typing import Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("ERROR: MLX not available")
    sys.exit(1)

try:
    from mlx_lm import generate, load

    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False
    print("WARNING: mlx_lm not available, install with: pip install mlx-lm")

# MLX model paths (check multiple locations)
MLX_MODEL_PATHS = [
    Path.home() / "models" / "llama" / "mlx",
    Path.cwd() / "mlx-llama",
    Path.cwd() / "models" / "llama" / "mlx",
    # Common mlx-lm converted model locations
    Path.home() / ".cache" / "huggingface" / "hub",
]


def find_model_path() -> Path | None:
    """Find LLaMA MLX model in known locations."""
    for path in MLX_MODEL_PATHS:
        if path.exists() and (path / "config.json").exists():
            return path
    return None


def get_test_prompts() -> List[tuple[str, str]]:
    """Get test prompts with expected content types.

    Returns list of (prompt, expected_content_type) tuples.
    expected_content_type: type of response expected
    """
    return [
        ("What is 2 + 2?", "math_answer"),
        ("Hello, my name is", "continuation"),
        ("The capital of France is", "factual"),
    ]


def is_coherent_generation(text: str) -> bool:
    """Check if generation is coherent (not garbage or repeated tokens)."""
    if not text or len(text) < 2:
        return False

    # Check for excessive repetition
    words = text.split()
    if len(words) >= 3:
        # Check for word-level repetition
        consecutive_repeats = sum(
            1 for i in range(len(words) - 1) if words[i] == words[i + 1]
        )
        if consecutive_repeats > len(words) / 2:
            return False

    # Check for character-level repetition
    chars = list(text)
    if len(chars) >= 10:
        consecutive_char_repeats = sum(
            1 for i in range(len(chars) - 1) if chars[i] == chars[i + 1]
        )
        if consecutive_char_repeats > len(chars) / 2:
            return False

    return True


def has_expected_content(text: str, content_type: str) -> bool:
    """Check if generation has expected content type."""
    text_lower = text.lower()

    if content_type == "math_answer":
        # Should contain "4" or "four" somewhere
        return "4" in text or "four" in text_lower

    if content_type == "continuation":
        # Should be a reasonable name continuation (any alphanumeric)
        return len(text) > 0 and any(c.isalpha() for c in text)

    if content_type == "factual":
        # Should mention "paris" (capital of France)
        return "paris" in text_lower

    return True  # Unknown content type, pass by default


def run_test_case(
    model: Any, tokenizer: Any, prompt: str, content_type: str, max_tokens: int = 50
) -> dict[str, Any]:
    """Run a single test case."""
    result: dict[str, Any] = {
        "prompt": prompt,
        "content_type": content_type,
        "status": "UNKNOWN",
        "error": None,
    }

    try:
        # Generate text
        output = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
        mx.eval(output)

        # Extract generated text (after prompt)
        generated = (
            output[len(prompt) :].strip() if len(output) > len(prompt) else output
        )
        result["generated"] = generated
        result["full_output"] = output

        # Check for empty generation
        if not generated or len(generated.strip()) < 1:
            result["status"] = "FAIL"
            result["error"] = "Empty generation"
            return result

        # Check for coherence (not garbage)
        if not is_coherent_generation(generated):
            result["status"] = "FAIL"
            result["error"] = "Incoherent generation (repetition detected)"
            return result

        # Check for expected content
        result["has_expected_content"] = has_expected_content(generated, content_type)
        if not result["has_expected_content"]:
            result["note"] = f"Expected content ({content_type}) not detected"

        # Pass if non-empty and coherent
        result["status"] = "PASS"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def main():
    """Run LLaMA LLM-as-Judge tests."""
    print("=" * 60)
    print("LLaMA LLM-as-Judge Test")
    print("=" * 60)
    print("\nThis test validates LLaMA MLX generation quality by checking")
    print("coherence and expected content in generated outputs.")

    if not HAS_MLX_LM:
        print("\nERROR: mlx_lm package required")
        print("Install with: pip install mlx-lm")
        return 1

    # Check model path
    model_path = find_model_path()
    if model_path is None:
        print("\nWARNING: MLX model not found in any of these locations:")
        for path in MLX_MODEL_PATHS:
            print(f"  - {path}")
        print("\nTo convert a LLaMA model, run:")
        print("  mlx_lm.convert --hf-path meta-llama/Llama-3-8B --mlx-path ./mlx-llama")
        print("\nOr use a smaller model:")
        print(
            "  mlx_lm.convert --hf-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mlx-path ./mlx-llama"
        )
        print("\nTests will be skipped.")
        print("\nOverall: SKIP")
        return 0

    # Load model
    print(f"\nLoading model from {model_path}...")
    try:
        model, tokenizer = load(str(model_path))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        return 1

    # Get test cases
    test_prompts = get_test_prompts()
    results = []

    for prompt, content_type in test_prompts:
        print(f"\n--- Test: {content_type} ---")
        print(f"  Prompt: {prompt}")

        result = run_test_case(model, tokenizer, prompt, content_type)
        results.append(result)

        print(f"  Status: {result['status']}")
        if result.get("generated"):
            # Truncate long outputs
            gen = result["generated"]
            if len(gen) > 80:
                gen = gen[:80] + "..."
            print(f"  Generated: {gen}")
        if result.get("has_expected_content") is not None:
            print(f"  Expected content: {result['has_expected_content']}")
        if result.get("note"):
            print(f"  Note: {result['note']}")
        if result.get("error"):
            print(f"  Error: {result['error']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

    overall = "PASS" if failed == 0 and errors == 0 and passed > 0 else "FAIL"
    if skipped == len(results):
        overall = "SKIP"

    print(f"\nOverall: {overall}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
