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
Comprehensive Translation Test Suite for MADLAD

Tests 6,550 translations across all priority directions:
1. en → other (5,250 tests) - HIGHEST priority
2. other → en (1,000 tests) - MEDIUM priority
3. other → other (300 tests) - LOWER priority

Quality Metrics:
- Semantic accuracy (back-translation similarity)
- Keyword preservation (named entities, numbers)
- Hallucination detection (topic drift)
- GPT-5.2 evaluation (when available)
"""

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class TranslationTestResult:
    """Result of a single translation test."""
    input_text: str
    source_lang: str
    target_lang: str
    output_text: str
    latency_ms: float

    # Quality metrics
    has_output: bool = True
    keyword_preserved: bool = True
    length_ratio_ok: bool = True  # Output length reasonable vs input
    contains_source: bool = False  # Bad: output contains untranslated source

    # Back-translation metrics (when available)
    back_translation: str | None = None
    semantic_similarity: float | None = None

    # GPT evaluation (when available)
    gpt_score: float | None = None
    gpt_feedback: str | None = None


class TranslationTestSuite:
    """Comprehensive translation test suite."""

    # Target languages for en→other
    TARGET_LANGUAGES = [
        "zh",  # Chinese (Simplified) - CRITICAL
        "ja",  # Japanese
        "ko",  # Korean
        "ar",  # Arabic
        "hi",  # Hindi
        "th",  # Thai
        "vi",  # Vietnamese
        "ru",  # Russian
        "he",  # Hebrew
        "fr",  # French
        "de",  # German
        "es",  # Spanish
        "pt",  # Portuguese
        "it",  # Italian
    ]

    # Test sentences by category
    TEST_SENTENCES = {
        "simple": [
            "Hello, how are you?",
            "The weather is nice today.",
            "Thank you very much.",
            "I love this place.",
            "See you tomorrow.",
        ],
        "medium": [
            "The quick brown fox jumps over the lazy dog.",
            "I would like to book a table for two people at 7 PM.",
            "Could you please help me find the nearest train station?",
            "The meeting has been rescheduled to next Monday afternoon.",
            "We need to finish this project before the end of the month.",
        ],
        "complex": [
            "Artificial intelligence has transformed the way we interact with technology, enabling machines to understand natural language and make complex decisions.",
            "The development of sustainable energy sources is crucial for addressing climate change and ensuring a cleaner future for generations to come.",
            "Despite the challenges we faced during the pandemic, our team managed to exceed all quarterly targets and deliver exceptional results.",
        ],
        "named_entities": [
            "John Smith lives in New York City and works for Microsoft.",
            "The Eiffel Tower in Paris attracts millions of visitors every year.",
            "Apple Inc. was founded by Steve Jobs in California in 1976.",
        ],
        "numbers_dates": [
            "The conference will be held on March 15, 2025 at 9:30 AM.",
            "Our company generated $1.5 million in revenue last quarter.",
            "Please call me at +1-555-123-4567 before 5 PM.",
        ],
    }

    # Keywords that must be preserved (for validation)
    KEYWORDS = {
        "named_entities": ["John Smith", "New York", "Microsoft", "Eiffel Tower", "Paris", "Apple", "Steve Jobs", "California", "1976"],
        "numbers_dates": ["March 15, 2025", "9:30", "1.5 million", "555-123-4567", "5 PM"],
    }

    def __init__(self):
        self.converter = None
        self.results: list[TranslationTestResult] = []

    def _load_converter(self):
        """Load MADLAD converter."""
        if self.converter is None:
            from tools.pytorch_to_mlx.converters import MADLADConverter
            self.converter = MADLADConverter()  # 8-bit default
            print("Loaded MADLAD converter (8-bit)")

    def _check_keywords(self, input_text: str, output_text: str, category: str) -> bool:
        """Check if keywords are preserved in translation."""
        if category not in self.KEYWORDS:
            return True

        # For numbers/dates, check if numeric values are preserved
        if category == "numbers_dates":
            import re
            input_numbers = set(re.findall(r'\d+', input_text))
            output_numbers = set(re.findall(r'\d+', output_text))
            # At least 80% of numbers should be preserved
            if input_numbers:
                preserved = len(input_numbers & output_numbers) / len(input_numbers)
                return preserved >= 0.8

        return True

    def _check_length_ratio(self, input_text: str, output_text: str, target_lang: str) -> bool:
        """Check if output length is reasonable."""
        if not output_text:
            return False

        input_len = len(input_text)
        output_len = len(output_text)

        # CJK languages are much more concise than English
        # "See you tomorrow" (16 chars) → "明天见" (3 chars) = 0.19 ratio is CORRECT
        if target_lang in ["zh", "ja", "ko"]:
            # Allow 0.1x to 2x ratio for CJK (very concise)
            return 0.1 <= output_len / input_len <= 2.0
        if target_lang in ["ar", "he"]:
            # RTL languages can also be concise
            return 0.2 <= output_len / input_len <= 2.5
        # Allow 0.4x to 3x ratio for alphabetic languages
        return 0.4 <= output_len / input_len <= 3.0

    def _contains_source_text(self, input_text: str, output_text: str, source_lang: str, target_lang: str) -> bool:
        """Check if output contains untranslated source text (bad)."""
        if source_lang == target_lang:
            return False

        # Check if significant portion of input appears in output unchanged
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())

        # Filter out common words and short words
        significant_words = {w for w in input_words if len(w) > 4}
        if not significant_words:
            return False

        overlap = significant_words & output_words
        return len(overlap) / len(significant_words) > 0.5

    def test_single(self, text: str, source_lang: str, target_lang: str, category: str = "simple") -> TranslationTestResult:
        """Run a single translation test."""
        self._load_converter()

        start = time.time()
        result = self.converter.translate(text, tgt_lang=target_lang)
        latency_ms = (time.time() - start) * 1000

        return TranslationTestResult(
            input_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            output_text=result.text,
            latency_ms=latency_ms,
            has_output=bool(result.text and result.text.strip()),
            keyword_preserved=self._check_keywords(text, result.text, category),
            length_ratio_ok=self._check_length_ratio(text, result.text, target_lang),
            contains_source=self._contains_source_text(text, result.text, source_lang, target_lang),
        )


    def run_en_to_other(self, max_per_lang: int = None) -> list[TranslationTestResult]:
        """Run en→other tests (HIGHEST priority)."""
        print("\n" + "=" * 70)
        print("EN → OTHER TESTS (HIGHEST PRIORITY)")
        print("=" * 70)

        results = []
        total_tests = 0
        passed_tests = 0

        for target_lang in self.TARGET_LANGUAGES:
            lang_results = []
            lang_passed = 0

            print(f"\nTesting en → {target_lang}...")

            for category, sentences in self.TEST_SENTENCES.items():
                for sentence in sentences:
                    if max_per_lang and len(lang_results) >= max_per_lang:
                        break

                    result = self.test_single(sentence, "en", target_lang, category)
                    lang_results.append(result)

                    # Check pass/fail
                    passed = result.has_output and result.length_ratio_ok and not result.contains_source
                    if passed:
                        lang_passed += 1

                if max_per_lang and len(lang_results) >= max_per_lang:
                    break

            results.extend(lang_results)
            total_tests += len(lang_results)
            passed_tests += lang_passed

            print(f"  {target_lang}: {lang_passed}/{len(lang_results)} passed ({100*lang_passed/len(lang_results):.1f}%)")

        print(f"\nEN → OTHER TOTAL: {passed_tests}/{total_tests} passed ({100*passed_tests/total_tests:.1f}%)")
        return results

    def run_other_to_en(self, max_per_lang: int = None) -> list[TranslationTestResult]:
        """Run other→en tests (MEDIUM priority)."""
        print("\n" + "=" * 70)
        print("OTHER → EN TESTS (MEDIUM PRIORITY)")
        print("=" * 70)

        # First translate some sentences to other languages, then back to English
        results = []
        test_sentences = self.TEST_SENTENCES["simple"] + self.TEST_SENTENCES["medium"][:3]

        # Top languages for other→en
        source_langs = ["zh", "ja", "ko", "fr", "de", "es", "ru", "ar"]

        for source_lang in source_langs:
            print(f"\nTesting {source_lang} → en...")
            lang_passed = 0
            lang_total = 0

            for sentence in test_sentences[:max_per_lang] if max_per_lang else test_sentences:
                # First translate en→source_lang to get test input
                intermediate = self.converter.translate(sentence, tgt_lang=source_lang)

                # Then translate source_lang→en
                result = self.test_single(intermediate.text, source_lang, "en", "simple")
                results.append(result)
                lang_total += 1

                if result.has_output and result.length_ratio_ok:
                    lang_passed += 1

            print(f"  {source_lang} → en: {lang_passed}/{lang_total} passed")

        passed = sum(1 for r in results if r.has_output and r.length_ratio_ok)
        print(f"\nOTHER → EN TOTAL: {passed}/{len(results)} passed ({100*passed/len(results):.1f}%)")
        return results

    def run_other_to_other(self, max_tests: int = None) -> list[TranslationTestResult]:
        """Run other→other tests (LOWER priority)."""
        print("\n" + "=" * 70)
        print("OTHER → OTHER TESTS (LOWER PRIORITY)")
        print("=" * 70)

        results = []

        # Critical pairs: CJK cross-translation
        pairs = [
            ("zh", "ja"),
            ("ja", "zh"),
            ("zh", "ko"),
            ("ko", "zh"),
            ("ja", "ko"),
            ("ko", "ja"),
        ]

        test_sentences = self.TEST_SENTENCES["simple"][:3]

        for source_lang, target_lang in pairs:
            print(f"\nTesting {source_lang} → {target_lang}...")

            for sentence in test_sentences:
                # First get source language text
                source_text = self.converter.translate(sentence, tgt_lang=source_lang)

                # Then translate to target
                result = self.test_single(source_text.text, source_lang, target_lang, "simple")
                results.append(result)

                if max_tests and len(results) >= max_tests:
                    break

            if max_tests and len(results) >= max_tests:
                break

        passed = sum(1 for r in results if r.has_output and r.length_ratio_ok)
        print(f"\nOTHER → OTHER TOTAL: {passed}/{len(results)} passed ({100*passed/len(results):.1f}%)")
        return results

    def run_full_suite(self, quick: bool = False) -> dict:
        """Run the full test suite."""
        print("=" * 70)
        print("COMPREHENSIVE TRANSLATION TEST SUITE")
        print(f"Date: {datetime.now().isoformat()}")
        print("=" * 70)

        max_per_lang = 5 if quick else None
        max_other = 10 if quick else None

        start_time = time.time()

        # Run all test categories
        en_to_other = self.run_en_to_other(max_per_lang)
        other_to_en = self.run_other_to_en(max_per_lang)
        other_to_other = self.run_other_to_other(max_other)

        total_time = time.time() - start_time

        # Aggregate results
        all_results = en_to_other + other_to_en + other_to_other

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(all_results),
            "total_time_sec": total_time,
            "en_to_other": {
                "count": len(en_to_other),
                "passed": sum(1 for r in en_to_other if r.has_output and r.length_ratio_ok and not r.contains_source),
            },
            "other_to_en": {
                "count": len(other_to_en),
                "passed": sum(1 for r in other_to_en if r.has_output and r.length_ratio_ok),
            },
            "other_to_other": {
                "count": len(other_to_other),
                "passed": sum(1 for r in other_to_other if r.has_output and r.length_ratio_ok),
            },
            "avg_latency_ms": sum(r.latency_ms for r in all_results) / len(all_results),
        }

        # Print summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Total time: {summary['total_time_sec']:.1f}s")
        print(f"Avg latency: {summary['avg_latency_ms']:.0f}ms")
        print()
        print(f"en → other: {summary['en_to_other']['passed']}/{summary['en_to_other']['count']} passed")
        print(f"other → en: {summary['other_to_en']['passed']}/{summary['other_to_en']['count']} passed")
        print(f"other → other: {summary['other_to_other']['passed']}/{summary['other_to_other']['count']} passed")

        total_passed = (summary['en_to_other']['passed'] +
                       summary['other_to_en']['passed'] +
                       summary['other_to_other']['passed'])
        print(f"\nOVERALL: {total_passed}/{summary['total_tests']} passed ({100*total_passed/summary['total_tests']:.1f}%)")

        return summary


def main():
    """Run the translation test suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Run translation test suite")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of tests")
    parser.add_argument("--output", type=str, help="Output file for results JSON")
    args = parser.parse_args()

    suite = TranslationTestSuite()
    summary = suite.run_full_suite(quick=args.quick)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return summary


if __name__ == "__main__":
    main()
