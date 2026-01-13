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
Export ALL misaki G2P lexicons for C++ runtime.

This script exports the complete G2P data from misaki for use in the C++ Kokoro pipeline.
It exports lexicons, phoneme mappings, and vocabulary for all supported languages.

Usage:
    python scripts/export_misaki_g2p.py --output-dir misaki_export

Output structure:
    misaki_export/
    ├── en/
    │   ├── us_golds.json       # 176k American English pronunciations
    │   ├── us_silvers.json     # 186k derived pronunciations
    │   ├── gb_golds.json       # 170k British English pronunciations
    │   ├── gb_silvers.json     # 220k derived pronunciations
    │   ├── vocab.json          # US phoneme vocabulary (45 symbols)
    │   ├── symbols.json        # Symbol pronunciations (%, &, +, @)
    │   ├── currencies.json     # Currency words ($, EUR, GBP)
    │   └── add_symbols.json    # Additional symbols (., /)
    ├── ja/
    │   ├── hepburn.json        # 189 kana-to-IPA mappings
    │   ├── words.json          # 147k Japanese word list
    │   └── katakana_ext.json   # 16 Katakana extensions
    ├── zh/
    │   └── pinyin_to_ipa.json  # Pinyin-to-IPA conversion rules
    └── vocab.json              # Unified Kokoro phoneme vocabulary (178 tokens)
"""

import argparse
import json
import sys
from pathlib import Path


def export_english(output_dir: Path):
    """Export all English G2P data from misaki."""
    from misaki import en
    from misaki.en import Lexicon

    en_dir = output_dir / "en"
    en_dir.mkdir(parents=True, exist_ok=True)

    # American English lexicon
    print("Exporting American English lexicon...")
    lex_us = Lexicon(british=False)

    with open(en_dir / "us_golds.json", "w", encoding="utf-8") as f:
        json.dump(
            lex_us.golds, f, ensure_ascii=False, indent=None, separators=(",", ":")
        )
    print(f"  us_golds.json: {len(lex_us.golds)} entries")

    with open(en_dir / "us_silvers.json", "w", encoding="utf-8") as f:
        json.dump(
            lex_us.silvers, f, ensure_ascii=False, indent=None, separators=(",", ":")
        )
    print(f"  us_silvers.json: {len(lex_us.silvers)} entries")

    # British English lexicon
    print("Exporting British English lexicon...")
    lex_gb = Lexicon(british=True)

    with open(en_dir / "gb_golds.json", "w", encoding="utf-8") as f:
        json.dump(
            lex_gb.golds, f, ensure_ascii=False, indent=None, separators=(",", ":")
        )
    print(f"  gb_golds.json: {len(lex_gb.golds)} entries")

    with open(en_dir / "gb_silvers.json", "w", encoding="utf-8") as f:
        json.dump(
            lex_gb.silvers, f, ensure_ascii=False, indent=None, separators=(",", ":")
        )
    print(f"  gb_silvers.json: {len(lex_gb.silvers)} entries")

    # Vocabulary (phoneme set)
    with open(en_dir / "us_vocab.json", "w", encoding="utf-8") as f:
        json.dump(sorted(en.US_VOCAB), f, ensure_ascii=False, indent=2)
    print(f"  us_vocab.json: {len(en.US_VOCAB)} phonemes")

    with open(en_dir / "gb_vocab.json", "w", encoding="utf-8") as f:
        json.dump(sorted(en.GB_VOCAB), f, ensure_ascii=False, indent=2)
    print(f"  gb_vocab.json: {len(en.GB_VOCAB)} phonemes")

    # Module-level mappings
    with open(en_dir / "symbols.json", "w", encoding="utf-8") as f:
        json.dump(en.SYMBOLS, f, ensure_ascii=False, indent=2)
    print(f"  symbols.json: {len(en.SYMBOLS)} entries")

    # Convert currency tuples to lists for JSON
    currencies = {k: list(v) for k, v in en.CURRENCIES.items()}
    with open(en_dir / "currencies.json", "w", encoding="utf-8") as f:
        json.dump(currencies, f, ensure_ascii=False, indent=2)
    print(f"  currencies.json: {len(en.CURRENCIES)} entries")

    with open(en_dir / "add_symbols.json", "w", encoding="utf-8") as f:
        json.dump(en.ADD_SYMBOLS, f, ensure_ascii=False, indent=2)
    print(f"  add_symbols.json: {len(en.ADD_SYMBOLS)} entries")

    with open(en_dir / "punct_tag_phonemes.json", "w", encoding="utf-8") as f:
        json.dump(en.PUNCT_TAG_PHONEMES, f, ensure_ascii=False, indent=2)
    print(f"  punct_tag_phonemes.json: {len(en.PUNCT_TAG_PHONEMES)} entries")

    # Export consonants, vowels, diphthongs for reference
    phoneme_classes = {
        "consonants": sorted(en.CONSONANTS),
        "vowels": sorted(en.VOWELS),
        "diphthongs": sorted(en.DIPHTHONGS),
        "stresses": en.STRESSES,
        "primary_stress": en.PRIMARY_STRESS,
        "secondary_stress": en.SECONDARY_STRESS,
    }
    with open(en_dir / "phoneme_classes.json", "w", encoding="utf-8") as f:
        json.dump(phoneme_classes, f, ensure_ascii=False, indent=2)
    print("  phoneme_classes.json: exported")

    return {
        "us_golds": len(lex_us.golds),
        "us_silvers": len(lex_us.silvers),
        "gb_golds": len(lex_gb.golds),
        "gb_silvers": len(lex_gb.silvers),
    }


def export_japanese(output_dir: Path):
    """Export all Japanese G2P data from misaki."""
    try:
        from misaki import ja
    except ImportError as e:
        print(f"Skipping Japanese export (missing dependencies): {e}")
        return None

    ja_dir = output_dir / "ja"
    ja_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting Japanese G2P data...")

    # Hepburn romanization table (kana -> IPA)
    with open(ja_dir / "hepburn.json", "w", encoding="utf-8") as f:
        json.dump(ja.HEPBURN, f, ensure_ascii=False, indent=2)
    print(f"  hepburn.json: {len(ja.HEPBURN)} entries")

    # Japanese word list
    with open(ja_dir / "words.json", "w", encoding="utf-8") as f:
        json.dump(
            sorted(ja.JA_WORDS),
            f,
            ensure_ascii=False,
            indent=None,
            separators=(",", ":"),
        )
    print(f"  words.json: {len(ja.JA_WORDS)} entries")

    # Katakana phonetic extensions
    with open(ja_dir / "katakana_ext.json", "w", encoding="utf-8") as f:
        json.dump(ja.Katakana_Phonetic_Extensions, f, ensure_ascii=False, indent=2)
    print(f"  katakana_ext.json: {len(ja.Katakana_Phonetic_Extensions)} entries")

    # Special character sets
    special = {
        "odori": sorted(ja.ODORI),
        "sutegana": sorted(ja.SUTEGANA),
    }
    with open(ja_dir / "special_chars.json", "w", encoding="utf-8") as f:
        json.dump(special, f, ensure_ascii=False, indent=2)
    print("  special_chars.json: exported")

    return {
        "hepburn": len(ja.HEPBURN),
        "words": len(ja.JA_WORDS),
    }


def export_chinese(output_dir: Path):
    """Export Chinese G2P data from misaki."""
    try:
        from misaki import zh
    except ImportError as e:
        print(f"Skipping Chinese export (missing dependencies): {e}")
        return None

    zh_dir = output_dir / "zh"
    zh_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting Chinese G2P data...")

    # Generate pinyin-to-IPA mapping by testing all possible pinyin syllables
    # Chinese uses pypinyin at runtime, but we can export the IPA conversion rules

    # Standard pinyin initials and finals
    initials = [
        "",
        "b",
        "p",
        "m",
        "f",
        "d",
        "t",
        "n",
        "l",
        "g",
        "k",
        "h",
        "j",
        "q",
        "x",
        "zh",
        "ch",
        "sh",
        "r",
        "z",
        "c",
        "s",
        "y",
        "w",
    ]
    finals = [
        "a",
        "o",
        "e",
        "i",
        "u",
        "v",
        "ai",
        "ei",
        "ao",
        "ou",
        "an",
        "en",
        "ang",
        "eng",
        "ong",
        "ia",
        "ie",
        "iao",
        "iu",
        "ian",
        "in",
        "iang",
        "ing",
        "iong",
        "ua",
        "uo",
        "uai",
        "ui",
        "uan",
        "un",
        "uang",
        "ueng",
        "ve",
        "van",
        "vn",
    ]
    tones = ["1", "2", "3", "4", "5"]  # 5 = neutral tone

    # Build pinyin-to-IPA mapping
    # pinyin_to_ipa returns OrderedSet of tuples like [('n', 'i˧˩˧')]
    # We want to flatten each tuple and store as a string
    pinyin_to_ipa = {}
    for initial in initials:
        for final in finals:
            pinyin = initial + final
            if not pinyin:
                continue
            for tone in tones:
                pinyin_tone = pinyin + tone
                try:
                    result = zh.pinyin_to_ipa(pinyin_tone)
                    if result:
                        # Convert OrderedSet of tuples to list of strings
                        ipa_variants = []
                        for variant in result:
                            if isinstance(variant, tuple):
                                ipa_variants.append("".join(variant))
                            else:
                                ipa_variants.append(str(variant))
                        if ipa_variants:
                            pinyin_to_ipa[pinyin_tone] = ipa_variants
                except Exception:
                    pass

    with open(zh_dir / "pinyin_to_ipa.json", "w", encoding="utf-8") as f:
        json.dump(pinyin_to_ipa, f, ensure_ascii=False, indent=2)
    print(f"  pinyin_to_ipa.json: {len(pinyin_to_ipa)} entries")

    # Also test some common syllables directly
    common_syllables = [
        "ni3",
        "hao3",
        "wo3",
        "shi4",
        "de5",
        "bu4",
        "zhe4",
        "ge4",
        "zhong1",
        "guo2",
        "ren2",
        "ming2",
        "tian1",
        "xue2",
        "sheng1",
    ]

    test_results: dict[str, list[str]] = {}
    for syllable in common_syllables:
        try:
            result = zh.pinyin_to_ipa(syllable)
            if result:
                test_ipa_variants: list[str] = []
                for variant in result:
                    if isinstance(variant, tuple):
                        test_ipa_variants.append("".join(variant))
                    else:
                        test_ipa_variants.append(str(variant))
                test_results[syllable] = test_ipa_variants
            else:
                test_results[syllable] = []
        except Exception as e:
            test_results[syllable] = [f"error: {e}"]

    with open(zh_dir / "test_samples.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    print(f"  test_samples.json: {len(test_results)} test cases")

    return {
        "pinyin_to_ipa": len(pinyin_to_ipa),
    }


def export_kokoro_vocab(output_dir: Path):
    """Export the Kokoro phoneme vocabulary (token ID mapping)."""
    # This is the authoritative vocabulary from Kokoro
    # Based on the phonemes.json from kokoro_cpp_export

    # Load from the exported vocab if available
    vocab_paths = [
        Path("kokoro_cpp_export/vocab/phonemes.json"),
        Path("kokoro_export/vocab/phonemes.json"),
    ]

    vocab = None
    for vp in vocab_paths:
        if vp.exists():
            with open(vp, encoding="utf-8") as f:
                vocab = json.load(f)
            print(f"Loaded vocab from {vp}")
            break

    if vocab is None:
        # Fallback: use the phoneme_for_kokoro.py vocab
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from phonemize_for_kokoro import VOCAB

            vocab = VOCAB
            print("Loaded vocab from phonemize_for_kokoro.py")
        except ImportError:
            print("WARNING: Could not load Kokoro vocabulary")
            return None

    # Save unified vocabulary
    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"vocab.json: {len(vocab)} tokens")

    # Also create an inverted mapping (id -> phoneme)
    id_to_phoneme = {v: k for k, v in vocab.items()}
    with open(output_dir / "vocab_inverse.json", "w", encoding="utf-8") as f:
        json.dump(id_to_phoneme, f, ensure_ascii=False, indent=2)
    print(f"vocab_inverse.json: {len(id_to_phoneme)} mappings")

    return {"vocab_size": len(vocab)}


def main():
    parser = argparse.ArgumentParser(description="Export misaki G2P lexicons")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("misaki_export"),
        help="Output directory for exported data",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPORTING MISAKI G2P LEXICONS")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    stats = {}

    # Export English
    print("-" * 40)
    en_stats = export_english(output_dir)
    stats["english"] = en_stats
    print()

    # Export Japanese
    print("-" * 40)
    ja_stats = export_japanese(output_dir)
    if ja_stats:
        stats["japanese"] = ja_stats
    print()

    # Export Chinese
    print("-" * 40)
    zh_stats = export_chinese(output_dir)
    if zh_stats:
        stats["chinese"] = zh_stats
    print()

    # Export Kokoro vocabulary
    print("-" * 40)
    vocab_stats = export_kokoro_vocab(output_dir)
    if vocab_stats:
        stats["vocab"] = vocab_stats
    print()

    # Write summary
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)

    with open(output_dir / "export_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nExport statistics saved to {output_dir}/export_stats.json")
    print("\nSummary:")
    for lang, lang_stats in stats.items():
        print(f"  {lang}:")
        for key, value in lang_stats.items():
            print(f"    {key}: {value:,}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*.json"))
    print(f"\nTotal export size: {total_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
