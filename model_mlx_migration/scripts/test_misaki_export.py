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
Test the exported misaki G2P lexicons.

This script verifies that the exported lexicons can be used for word lookup
and compares results with the live misaki library.
"""

import json
import sys
from pathlib import Path
from typing import Any


def load_lexicon(path: Path) -> dict[str, Any]:
    """Load a JSON lexicon file."""
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def test_english_lexicon(export_dir: Path):
    """Test English lexicon lookup."""
    print("=" * 60)
    print("Testing English Lexicon")
    print("=" * 60)

    # Load exported lexicons
    us_golds = load_lexicon(export_dir / "en" / "us_golds.json")
    us_silvers = load_lexicon(export_dir / "en" / "us_silvers.json")
    gb_golds = load_lexicon(export_dir / "en" / "gb_golds.json")

    # Test words
    test_words = [
        "hello",
        "world",
        "computer",
        "artificial",
        "intelligence",
        "the",
        "a",
        "is",
        "are",
        "have",
        "python",
        "programming",
        "machine",
        "learning",
    ]

    print("\nUS English lookup:")
    for word in test_words:
        gold = us_golds.get(word.lower())
        silver = us_silvers.get(word.capitalize()) if not gold else None
        result = gold or silver or "NOT FOUND"
        source = "gold" if gold else ("silver" if silver else "none")
        print(f"  {word}: {result} ({source})")

    print("\nGB English lookup:")
    for word in ["colour", "favourite", "centre"]:
        result = gb_golds.get(word.lower(), "NOT FOUND")
        print(f"  {word}: {result}")

    # Compare with live misaki
    print("\nComparing with live misaki...")
    try:
        from misaki.en import Lexicon

        lex = Lexicon(british=False)
        mismatches = 0
        sample_size = min(100, len(test_words))

        for word in test_words[:sample_size]:
            exported = us_golds.get(word.lower())
            live = lex.golds.get(word.lower())
            if exported != live:
                print(f"  MISMATCH: {word} - exported={exported}, live={live}")
                mismatches += 1

        if mismatches == 0:
            print(f"  All {sample_size} samples match!")
        else:
            print(f"  {mismatches}/{sample_size} mismatches found")

    except ImportError:
        print("  (misaki not available for comparison)")


def test_japanese_lexicon(export_dir: Path):
    """Test Japanese G2P data."""
    print("\n" + "=" * 60)
    print("Testing Japanese G2P")
    print("=" * 60)

    hepburn = load_lexicon(export_dir / "ja" / "hepburn.json")
    words = load_lexicon(export_dir / "ja" / "words.json")

    print(f"\nHepburn table: {len(hepburn)} entries")
    print("Sample mappings:")
    test_kana = ["あ", "い", "う", "え", "お", "か", "き", "く", "こ", "ん"]
    for kana in test_kana:
        ipa = hepburn.get(kana, "NOT FOUND")
        print(f"  {kana} -> {ipa}")

    print(f"\nJapanese words: {len(words)} entries")
    print("Sample words:")
    # words may be a list stored in JSON
    words_list = list(words.keys()) if isinstance(words, dict) else list(words)[:10]
    for word in words_list[:10]:
        print(f"  {word}")


def test_chinese_lexicon(export_dir: Path):
    """Test Chinese pinyin-to-IPA data."""
    print("\n" + "=" * 60)
    print("Testing Chinese Pinyin-to-IPA")
    print("=" * 60)

    pinyin_to_ipa = load_lexicon(export_dir / "zh" / "pinyin_to_ipa.json")
    test_samples = load_lexicon(export_dir / "zh" / "test_samples.json")

    print(f"\nPinyin-to-IPA mappings: {len(pinyin_to_ipa)} entries")
    print("\nTest samples:")
    for pinyin, ipa_list in test_samples.items():
        if isinstance(ipa_list, list):
            print(f"  {pinyin} -> {', '.join(ipa_list)}")
        else:
            print(f"  {pinyin} -> {ipa_list}")


def test_vocab(export_dir: Path):
    """Test Kokoro vocabulary."""
    print("\n" + "=" * 60)
    print("Testing Kokoro Vocabulary")
    print("=" * 60)

    vocab = load_lexicon(export_dir / "vocab.json")
    vocab_inverse = load_lexicon(export_dir / "vocab_inverse.json")

    print(f"\nVocabulary size: {len(vocab)} tokens")
    print(f"ID range: 1 to {max(vocab.values())}")

    # Check for special tokens
    print("\nSpecial tokens:")
    special = [";", ":", ",", ".", "!", "?", " ", "ˈ", "ˌ"]
    for char in special:
        token_id = vocab.get(char)
        if token_id is not None:
            print(f"  '{char}' -> {token_id}")

    # Verify round-trip
    print("\nRound-trip verification:")
    mismatches = 0
    for phoneme, token_id in vocab.items():
        recovered = vocab_inverse.get(str(token_id))
        if recovered != phoneme:
            print(f"  MISMATCH: {phoneme} -> {token_id} -> {recovered}")
            mismatches += 1

    if mismatches == 0:
        print(f"  All {len(vocab)} tokens round-trip correctly!")


def main():
    export_dir = Path("misaki_export")
    if not export_dir.exists():
        print(f"ERROR: Export directory not found: {export_dir}")
        print("Run export_misaki_g2p.py first")
        sys.exit(1)

    test_english_lexicon(export_dir)
    test_japanese_lexicon(export_dir)
    test_chinese_lexicon(export_dir)
    test_vocab(export_dir)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
