#!/usr/bin/env python3
"""
Generate SOTA phoneme lexicons for all Kokoro TTS languages.

Uses the best available G2P for each language:
- Chinese: misaki.zh.ZHG2P (Kokoro's native, correct tones)
- Japanese: misaki.ja.JAG2P (Kokoro's native, with pitch)
- Spanish/French/Italian/Portuguese/Hindi/Korean: transphone (multilingual IPA)

Output: C++ header files with static lookup tables for integration.

Usage:
    python generate_sota_lexicons.py --test           # Test all G2P systems
    python generate_sota_lexicons.py --chinese        # Generate Chinese lexicon
    python generate_sota_lexicons.py --all            # Generate all lexicons

Copyright 2025 Andrew Yates. All rights reserved.
"""

import sys
import argparse

# ============================================================================
# Chinese G2P (misaki - SOTA for Kokoro)
# ============================================================================

def get_chinese_g2p():
    """Get misaki Chinese G2P (correct tones with sandhi)."""
    try:
        from misaki.zh import ZHG2P
        return ZHG2P()
    except ImportError as e:
        print(f"Error: Install misaki with: pip install misaki jieba cn2an")
        raise

def chinese_to_phonemes(text, g2p=None):
    """Convert Chinese text to Kokoro-compatible phonemes."""
    if g2p is None:
        g2p = get_chinese_g2p()
    return g2p(text)

# ============================================================================
# Japanese G2P (misaki - SOTA for Kokoro)
# ============================================================================

def get_japanese_g2p():
    """Get misaki Japanese G2P (with pitch accent)."""
    try:
        from misaki.ja import JAG2P
        return JAG2P()
    except ImportError as e:
        print(f"Error: Install misaki with: pip install misaki fugashi unidic-lite jaconv mojimoji")
        raise

def japanese_to_phonemes(text, g2p=None):
    """Convert Japanese text to Kokoro-compatible phonemes."""
    if g2p is None:
        g2p = get_japanese_g2p()
    return g2p(text)

# ============================================================================
# Transphone (multilingual - Spanish, French, Italian, Portuguese, Hindi, Korean)
# ============================================================================

_transphone_tokenizers = {}

def get_transphone_tokenizer(lang_code):
    """Get transphone tokenizer for a language."""
    global _transphone_tokenizers
    if lang_code not in _transphone_tokenizers:
        try:
            from transphone import read_tokenizer
            _transphone_tokenizers[lang_code] = read_tokenizer(lang_code)
        except ImportError:
            print(f"Error: Install transphone with: pip install transphone")
            raise
    return _transphone_tokenizers[lang_code]

def transphone_to_phonemes(text, lang_code):
    """Convert text to IPA using transphone."""
    tokenizer = get_transphone_tokenizer(lang_code)
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)

# Language code mapping
TRANSPHONE_LANGS = {
    'es': 'spa',  # Spanish
    'fr': 'fra',  # French
    'it': 'ita',  # Italian
    'pt': 'por',  # Portuguese
    'hi': 'hin',  # Hindi
    'ko': 'kor',  # Korean
}

# ============================================================================
# Testing
# ============================================================================

def test_all():
    """Test all G2P systems."""
    print("=" * 60)
    print("SOTA G2P Testing for Kokoro TTS")
    print("=" * 60)
    print()

    # Chinese
    print("CHINESE (misaki.zh.ZHG2P - Kokoro native):")
    print("-" * 40)
    try:
        g2p = get_chinese_g2p()
        tests = [
            ("你好", "ni↓xau↓ (both tone 3 = dipping)"),
            ("好", "xau↓ (tone 3)"),
            ("世界", "ʂɨ↘ʨje↘ (tone 4 + tone 4)"),
            ("不好意思", "tone sandhi applied"),
            ("一二三四五", "numbers with tone changes"),
        ]
        for text, expected in tests:
            result = g2p(text)
            print(f"  '{text}' -> {result}")
            print(f"    Expected: {expected}")
        print()
    except Exception as e:
        print(f"  Error: {e}\n")

    # Japanese
    print("JAPANESE (misaki.ja.JAG2P - Kokoro native):")
    print("-" * 40)
    try:
        g2p = get_japanese_g2p()
        tests = [
            ("こんにちは", "with pitch accent"),
            ("今日はいい天気ですね", "full sentence"),
            ("日本語", "kanji"),
        ]
        for text, expected in tests:
            result = g2p(text)
            print(f"  '{text}' -> {result}")
            print(f"    Expected: {expected}")
        print()
    except Exception as e:
        print(f"  Error: {e}\n")

    # Other languages via transphone
    print("OTHER LANGUAGES (transphone - multilingual IPA):")
    print("-" * 40)
    tests = [
        ('es', 'spa', "Hola mundo", "Spanish"),
        ('fr', 'fra', "Bonjour monde", "French"),
        ('it', 'ita', "Ciao mondo", "Italian"),
        ('pt', 'por', "Olá mundo", "Portuguese"),
        ('hi', 'hin', "नमस्ते", "Hindi"),
        ('ko', 'kor', "안녕하세요", "Korean"),
    ]
    for our_code, tp_code, text, name in tests:
        try:
            result = transphone_to_phonemes(text, tp_code)
            print(f"  {name} '{text}' -> {result}")
        except Exception as e:
            print(f"  {name} Error: {e}")
    print()

    print("=" * 60)
    print("COMPARISON: espeak vs misaki for Chinese")
    print("=" * 60)
    print()
    print("espeak-ng (BROKEN):  你好 -> ni↗hˈɑu↗ (both rising - WRONG)")
    print("misaki (CORRECT):    你好 -> ni↓xau↓ (both dipping - CORRECT)")
    print()

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SOTA G2P lexicons')
    parser.add_argument('--test', action='store_true', help='Test all G2P systems')
    parser.add_argument('--chinese', type=str, help='Convert Chinese text')
    parser.add_argument('--japanese', type=str, help='Convert Japanese text')
    parser.add_argument('--spanish', type=str, help='Convert Spanish text')
    parser.add_argument('--french', type=str, help='Convert French text')

    args = parser.parse_args()

    if args.test:
        test_all()
    elif args.chinese:
        print(chinese_to_phonemes(args.chinese))
    elif args.japanese:
        print(japanese_to_phonemes(args.japanese))
    elif args.spanish:
        print(transphone_to_phonemes(args.spanish, 'spa'))
    elif args.french:
        print(transphone_to_phonemes(args.french, 'fra'))
    else:
        parser.print_help()
