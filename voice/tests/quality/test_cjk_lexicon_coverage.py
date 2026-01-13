"""
CJK (Chinese-Japanese-Korean) Lexicon Coverage Tests

Tests coverage of CJK lexicons embedded in C++ headers:
1. Verifies minimum word count thresholds
2. Tests essential vocabulary is present
3. Validates phoneme format (basic sanity checks)

The CJK lexicons are in C++ header files and use the misaki phoneme format
which matches Kokoro's training data.

Usage:
    pytest tests/quality/test_cjk_lexicon_coverage.py -v
"""

import re
import pytest
from pathlib import Path

pytestmark = [pytest.mark.quality, pytest.mark.cjk]

# Paths to C++ header lexicons
PROJECT_ROOT = Path(__file__).parent.parent.parent
JAPANESE_LEXICON = PROJECT_ROOT / "stream-tts-cpp" / "include" / "japanese_misaki_lexicon.hpp"
CHINESE_LEXICON = PROJECT_ROOT / "stream-tts-cpp" / "include" / "chinese_misaki_lexicon.hpp"


# =============================================================================
# Lexicon Parsing
# =============================================================================

def parse_cpp_lexicon(path: Path) -> dict:
    """
    Parse a C++ header lexicon file and extract entries.

    Format: {"text", "phonemes"},
    Returns: {text: phonemes}
    """
    if not path.exists():
        return {}

    content = path.read_text(encoding='utf-8')

    # Match entries like {"ありがとう", "aɾʲiɡatoː"},
    pattern = r'\{"([^"]+)",\s*"([^"]+)"\}'
    matches = re.findall(pattern, content)

    return {text: phonemes for text, phonemes in matches}


# =============================================================================
# Test Data - Essential CJK Vocabulary
# =============================================================================

# Essential Japanese words that should be in the lexicon
ESSENTIAL_JAPANESE = [
    "こんにちは",      # Hello
    "ありがとう",      # Thank you
    "おはよう",        # Good morning
    "さようなら",      # Goodbye
    "すみません",      # Excuse me / Sorry
    "はい",            # Yes
    "いいえ",          # No
    "できる",          # Can do
    "あなた",          # You
    "ファイル",        # File (programming)
    "コミット",        # Commit (programming)
    "エラー",          # Error (programming)
]

# Essential Chinese words that should be in the lexicon
ESSENTIAL_CHINESE = [
    "你好",            # Hello
    "谢谢",            # Thank you
    "再见",            # Goodbye
    "早上好",          # Good morning
    "晚上好",          # Good evening
    "对不起",          # Sorry
    "没问题",          # No problem
    "我知道了",        # I understand
    "代码",            # Code
    "文件",            # File
    "程序",            # Program
    "服务器",          # Server
]


# =============================================================================
# Test Classes
# =============================================================================

class TestJapaneseLexicon:
    """Test Japanese lexicon coverage and format."""

    @pytest.fixture(scope="class")
    def ja_lexicon(self):
        """Load Japanese lexicon."""
        if not JAPANESE_LEXICON.exists():
            pytest.skip(f"Japanese lexicon not found: {JAPANESE_LEXICON}")
        return parse_cpp_lexicon(JAPANESE_LEXICON)

    def test_lexicon_minimum_size(self, ja_lexicon):
        """Verify Japanese lexicon has minimum number of entries."""
        assert len(ja_lexicon) >= 400, \
            f"Japanese lexicon too small: {len(ja_lexicon)} entries (expected 400+)"

    @pytest.mark.parametrize("word", ESSENTIAL_JAPANESE)
    def test_essential_word_present(self, ja_lexicon, word):
        """Verify essential Japanese words are in lexicon."""
        assert word in ja_lexicon, f"Essential word '{word}' not in Japanese lexicon"

    def test_phonemes_contain_ipa(self, ja_lexicon):
        """Verify phonemes use IPA-like characters."""
        ipa_chars = set("aeiouɾʲɡɲʨβɸʥʣʦɕʰɴɯ")

        missing_ipa = []
        for word, phonemes in list(ja_lexicon.items())[:100]:  # Check first 100
            if not any(c in ipa_chars for c in phonemes):
                missing_ipa.append((word, phonemes))

        assert len(missing_ipa) == 0, \
            f"Found {len(missing_ipa)} entries without IPA chars: {missing_ipa[:5]}"

    def test_no_empty_phonemes(self, ja_lexicon):
        """Verify no entries have empty phonemes."""
        empty = [(w, p) for w, p in ja_lexicon.items() if not p.strip()]
        assert len(empty) == 0, f"Found {len(empty)} entries with empty phonemes: {empty[:5]}"


class TestChineseLexicon:
    """Test Chinese lexicon coverage and format."""

    @pytest.fixture(scope="class")
    def zh_lexicon(self):
        """Load Chinese lexicon."""
        if not CHINESE_LEXICON.exists():
            pytest.skip(f"Chinese lexicon not found: {CHINESE_LEXICON}")
        return parse_cpp_lexicon(CHINESE_LEXICON)

    def test_lexicon_minimum_size(self, zh_lexicon):
        """Verify Chinese lexicon has minimum number of entries."""
        assert len(zh_lexicon) >= 1000, \
            f"Chinese lexicon too small: {len(zh_lexicon)} entries (expected 1000+)"

    @pytest.mark.parametrize("word", ESSENTIAL_CHINESE)
    def test_essential_word_present(self, zh_lexicon, word):
        """Verify essential Chinese words are in lexicon."""
        assert word in zh_lexicon, f"Essential word '{word}' not in Chinese lexicon"

    def test_tone_markers_present(self, zh_lexicon):
        """Verify Chinese phonemes include tone markers."""
        tone_markers = set("↗↓↘→")  # Rising, dipping, falling, level

        entries_with_tones = 0
        for phonemes in zh_lexicon.values():
            if any(t in phonemes for t in tone_markers):
                entries_with_tones += 1

        # At least 90% should have tone markers
        ratio = entries_with_tones / len(zh_lexicon) if zh_lexicon else 0
        assert ratio >= 0.9, \
            f"Only {ratio:.1%} of entries have tone markers (expected 90%+)"

    def test_no_empty_phonemes(self, zh_lexicon):
        """Verify no entries have empty phonemes."""
        empty = [(w, p) for w, p in zh_lexicon.items() if not p.strip()]
        assert len(empty) == 0, f"Found {len(empty)} entries with empty phonemes: {empty[:5]}"

    def test_phonemes_use_ipa(self, zh_lexicon):
        """Verify phonemes use IPA-like characters."""
        ipa_chars = set("aeiouɕʨʥʦɤɥŋʂʊɛɨɯwo")

        missing_ipa = []
        for word, phonemes in list(zh_lexicon.items())[:100]:  # Check first 100
            if not any(c in ipa_chars for c in phonemes):
                missing_ipa.append((word, phonemes))

        assert len(missing_ipa) == 0, \
            f"Found {len(missing_ipa)} entries without IPA chars: {missing_ipa[:5]}"


class TestCJKLexiconRegression:
    """
    Regression tests for CJK lexicons.

    These freeze expected phonemes for key words to detect drift.
    """

    # Format: (word, expected_phoneme, lexicon_type)
    CJK_REGRESSION_FIXTURES = [
        # Japanese
        ("こんにちは", "koɲɲiʨiβa", "ja"),
        ("ありがとう", "aɾʲiɡatoː", "ja"),
        ("さようなら", "sajoːnaɾa", "ja"),
        ("ファイル", "ɸaiɾɯ", "ja"),
        ("エラー", "eɾaː", "ja"),

        # Chinese
        ("你好", "ni↓xau↓", "zh"),
        ("早上好", "ʦau↓ʂa↘ŋxau↓", "zh"),
        ("晚上好", "wa↓nʂa↘ŋ xau↓", "zh"),
        ("对不起", "twei↘pu↘ʨʰi↓", "zh"),
        ("代码", "tai↘ma↓", "zh"),
    ]

    @pytest.fixture(scope="class")
    def lexicons(self):
        """Load both lexicons."""
        return {
            "ja": parse_cpp_lexicon(JAPANESE_LEXICON),
            "zh": parse_cpp_lexicon(CHINESE_LEXICON),
        }

    @pytest.mark.parametrize("word,expected,lang", CJK_REGRESSION_FIXTURES)
    def test_phoneme_exact_match(self, lexicons, word, expected, lang):
        """Verify CJK phonemes match frozen expectations."""
        lexicon = lexicons.get(lang, {})

        if not lexicon:
            pytest.skip(f"Lexicon for '{lang}' not loaded")

        actual = lexicon.get(word)
        assert actual is not None, f"'{word}' not found in {lang} lexicon"
        assert actual == expected, (
            f"CJK PHONEME DRIFT for '{word}' ({lang}):\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  Update CJK_REGRESSION_FIXTURES if change is intentional"
        )


class TestCJKLexiconStats:
    """Print statistics about CJK lexicons."""

    def test_print_lexicon_stats(self):
        """Print coverage statistics for CJK lexicons."""
        ja_lex = parse_cpp_lexicon(JAPANESE_LEXICON)
        zh_lex = parse_cpp_lexicon(CHINESE_LEXICON)

        print(f"\n=== CJK Lexicon Coverage Statistics ===")
        print(f"\nJapanese Lexicon ({JAPANESE_LEXICON.name}):")
        print(f"  Total entries: {len(ja_lex)}")
        if ja_lex:
            avg_len = sum(len(w) for w in ja_lex.keys()) / len(ja_lex)
            print(f"  Avg word length: {avg_len:.1f} chars")

        print(f"\nChinese Lexicon ({CHINESE_LEXICON.name}):")
        print(f"  Total entries: {len(zh_lex)}")
        if zh_lex:
            avg_len = sum(len(w) for w in zh_lex.keys()) / len(zh_lex)
            print(f"  Avg word length: {avg_len:.1f} chars")

            # Count tone marker usage
            tone_markers = "↗↓↘→"
            with_tones = sum(1 for p in zh_lex.values() if any(t in p for t in tone_markers))
            print(f"  Entries with tones: {with_tones} ({100*with_tones/len(zh_lex):.1f}%)")

        # Assertions for stats test
        assert len(ja_lex) >= 400, "Japanese lexicon minimum not met"
        assert len(zh_lex) >= 1000, "Chinese lexicon minimum not met"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
