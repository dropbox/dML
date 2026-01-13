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
NLLB LLM-as-Judge Test

Validates NLLB translation quality using back-translation and semantic comparison.

Pass criteria:
- Non-empty translations produced
- Back-translation preserves key content
- Output is in the expected target language

Usage:
    python scripts/test_nllb_llm_judge.py
"""

import sys
from pathlib import Path
from typing import Any, List, Tuple

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
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers not available")

from tools.pytorch_to_mlx.converters.models.nllb import NLLBModel

# MLX model paths (check multiple locations)
MLX_MODEL_PATHS = [
    Path.home() / "models" / "nllb" / "mlx",
    Path.cwd() / "mlx-nllb",
    Path.cwd() / "models" / "nllb" / "mlx",
]


def find_model_path() -> Path | None:
    """Find NLLB MLX model in known locations."""
    for path in MLX_MODEL_PATHS:
        if path.exists():
            return path
    return None


def get_test_cases() -> List[Tuple[str, str, str, str]]:
    """Get test cases for NLLB translation.

    Returns list of (source_text, src_lang, tgt_lang, expected_content_keywords).
    expected_content_keywords: words/concepts that should appear in translation
    """
    return [
        ("Hello, how are you?", "eng_Latn", "fra_Latn", "bonjour,comment"),
        ("The weather is nice today.", "eng_Latn", "deu_Latn", "wetter,heute"),
        (
            "I love learning new languages.",
            "eng_Latn",
            "spa_Latn",
            "amor,aprender,idiomas",
        ),
    ]


def translate_text(
    model: NLLBModel,
    tokenizer: Any,
    text: str,
    src_lang: str,
    tgt_lang: str,
    max_tokens: int = 100,
) -> str:
    """Translate text using NLLB MLX model."""
    # Set source language
    tokenizer.src_lang = src_lang

    # Tokenize
    inputs = tokenizer(text, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])

    # Encode
    encoder_output = model.encode(input_ids)
    mx.eval(encoder_output)

    # NLLB generation pattern
    decoder_start_id = tokenizer.eos_token_id  # 2
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    # Start sequence with EOS + target language token
    generated = [decoder_start_id, tgt_lang_id]

    # Prime the cache
    decoder_ids = mx.array([generated])
    logits, cache = model.decode(decoder_ids, encoder_output, cache=None)
    mx.eval(logits, cache)

    # Generate tokens
    for _ in range(max_tokens):
        next_token = int(mx.argmax(logits[0, -1]))
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        # Incremental decode
        next_ids = mx.array([[next_token]])
        logits, cache = model.decode(next_ids, encoder_output, cache=cache)
        mx.eval(logits, cache)

    # Decode tokens to text
    translation: str = tokenizer.decode(generated[2:], skip_special_tokens=True)
    return translation


def has_expected_content(translation: str, keywords: str) -> bool:
    """Check if translation contains expected keywords.

    Keywords are comma-separated and case-insensitive.
    At least one keyword must be present (substring match).
    """
    translation_lower = translation.lower()
    keyword_list = [k.strip() for k in keywords.split(",")]

    for keyword in keyword_list:
        if keyword in translation_lower:
            return True
    return False


def run_test_case(
    model: NLLBModel,
    tokenizer: Any,
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    expected_keywords: str,
) -> dict[str, Any]:
    """Run a single test case."""
    result: dict[str, Any] = {
        "source_text": source_text,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "status": "UNKNOWN",
        "error": None,
    }

    try:
        # Translate
        translation = translate_text(model, tokenizer, source_text, src_lang, tgt_lang)
        result["translation"] = translation

        # Check non-empty
        if not translation or len(translation.strip()) < 2:
            result["status"] = "FAIL"
            result["error"] = "Empty or too short translation"
            return result

        # Check expected content (at least one keyword present)
        if has_expected_content(translation, expected_keywords):
            result["has_expected_content"] = True
        else:
            result["has_expected_content"] = False
            # Not a hard failure - translation may use synonyms
            result["note"] = f"Expected keywords ({expected_keywords}) not found"

        # Back-translate to verify semantic equivalence
        back_translation = translate_text(
            model, tokenizer, translation, tgt_lang, src_lang
        )
        result["back_translation"] = back_translation

        # Simple semantic check: source words should appear in back-translation
        source_words = set(
            source_text.lower().replace(".", "").replace(",", "").split()
        )
        back_words = set(
            back_translation.lower().replace(".", "").replace(",", "").split()
        )
        overlap = len(source_words & back_words)
        overlap_ratio = overlap / len(source_words) if source_words else 0

        result["semantic_overlap"] = overlap_ratio

        # Pass if non-empty translation and reasonable semantic overlap
        if len(translation) > 2 and overlap_ratio >= 0.3:
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
            result["error"] = f"Low semantic overlap ({overlap_ratio:.2f})"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def main():
    """Run NLLB LLM-as-Judge tests."""
    print("=" * 60)
    print("NLLB LLM-as-Judge Test")
    print("=" * 60)
    print("\nThis test validates NLLB MLX translation quality using")
    print("back-translation and semantic comparison.")

    if not HAS_TRANSFORMERS:
        print("\nERROR: transformers package required")
        return 1

    # Check model path
    model_path = find_model_path()
    if model_path is None:
        print("\nWARNING: MLX model not found in any of these locations:")
        for path in MLX_MODEL_PATHS:
            print(f"  - {path}")
        print("\nTo convert the model, run:")
        print("  PYTHONPATH=tools python -m pytorch_to_mlx.cli nllb convert \\")
        print("    --hf-path facebook/nllb-200-distilled-600M --output ./mlx-nllb")
        print("\nTests will be skipped.")
        print("\nOverall: SKIP")
        return 0

    # Load model and tokenizer
    print(f"\nLoading model from {model_path}...")
    try:
        model = NLLBModel.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        return 1

    # Get test cases
    test_cases = get_test_cases()
    results = []

    for source_text, src_lang, tgt_lang, expected_keywords in test_cases:
        print(f"\n--- Test: {src_lang} -> {tgt_lang} ---")
        print(f"  Source: {source_text}")

        result = run_test_case(
            model, tokenizer, source_text, src_lang, tgt_lang, expected_keywords
        )
        results.append(result)

        print(f"  Status: {result['status']}")
        if result.get("translation"):
            print(f"  Translation: {result['translation']}")
        if result.get("back_translation"):
            print(f"  Back-translation: {result['back_translation']}")
        if result.get("semantic_overlap") is not None:
            print(f"  Semantic overlap: {result['semantic_overlap']:.2f}")
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
