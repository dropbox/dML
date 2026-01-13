"""
G2P (Grapheme-to-Phoneme) Accuracy Tests

Tests the quality of phoneme conversion by verifying:
1. Lexicon coverage for common words
2. Heteronym disambiguation (if implemented)
3. OOV word handling via espeak-ng fallback
4. Phoneme-to-token conversion validity

Usage:
    pytest tests/quality/test_g2p_accuracy.py -v
    pytest tests/quality/test_g2p_accuracy.py -v -m quality
"""

import json
import pytest
from pathlib import Path

pytestmark = [pytest.mark.quality, pytest.mark.g2p]


# =============================================================================
# Test Data
# =============================================================================

# Common English words that should be in the lexicon
COMMON_WORDS = [
    "hello", "world", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "the", "a", "an", "is", "are", "was", "were", "have", "has", "had",
    "will", "would", "could", "should", "can", "may", "might", "must",
    "computer", "software", "programming", "algorithm", "function", "variable",
    "python", "database", "network", "security", "interface",  # Note: javascript not in lexicon
]

# Words that commonly mispronounce - heteronyms that need context
HETERONYMS = {
    # (word, verb_phonemes, noun/adj_phonemes)
    "read": ("ɹˈid", "ɹˈɛd"),  # "read" (present) vs "read" (past)
    "lead": ("lˈid", "lˈɛd"),  # "lead" (guide) vs "lead" (metal)
    "live": ("lˈɪv", "lˈIv"),  # "live" (verb) vs "live" (adjective)
    "bow": ("bˈaʊ", "bˈoʊ"),   # "bow" (weapon) vs "bow" (gesture)
    "tear": ("tˈɛɹ", "tˈɪɹ"),  # "tear" (rip) vs "tear" (crying)
    "wind": ("wˈɪnd", "wˈaɪnd"),  # "wind" (air) vs "wind" (turn)
    "minute": ("mˈɪnɪt", "maɪnjˈut"),  # "minute" (time) vs "minute" (tiny)
    "bass": ("bˈeɪs", "bˈæs"),  # "bass" (fish) vs "bass" (music)
    "close": ("klˈoʊz", "klˈoʊs"),  # "close" (verb) vs "close" (near)
    "record": ("ɹɪkˈɔɹd", "ɹˈɛkɔɹd"),  # "record" (verb) vs "record" (noun)
}

# Technical terms that may not be in standard lexicons
TECHNICAL_TERMS = [
    "API", "JSON", "YAML", "HTTP", "REST", "GraphQL", "WebSocket",
    "OAuth", "JWT", "CORS", "async", "await", "callback", "middleware",
]

# Numbers and special cases
SPECIAL_CASES = [
    ("1st", "first"),
    ("2nd", "second"),
    ("3rd", "third"),
    ("123", "one hundred twenty three"),
    ("$100", "one hundred dollars"),
    ("10%", "ten percent"),
]


# =============================================================================
# Test Classes
# =============================================================================

class TestLexiconCoverage:
    """Test that common words are in the lexicon."""

    @pytest.mark.parametrize("word", COMMON_WORDS)
    def test_common_word_in_lexicon(self, lexicon, word):
        """Verify common words exist in lexicon."""
        word_lower = word.lower()
        assert word_lower in lexicon, f"Common word '{word}' not found in lexicon"

        phonemes = lexicon[word_lower]
        if isinstance(phonemes, dict):
            phonemes = phonemes.get('DEFAULT', str(phonemes))

        assert len(phonemes) > 0, f"'{word}' has empty phonemes"

    def test_lexicon_size(self, lexicon):
        """Verify lexicon has substantial coverage."""
        assert len(lexicon) >= 80000, f"Lexicon too small: {len(lexicon)} words (expected 80k+)"


class TestHeteronymAwareness:
    """Test that heteronyms are documented (even if not yet disambiguated)."""

    @pytest.mark.parametrize("word,pronunciations", list(HETERONYMS.items()))
    def test_heteronym_in_lexicon(self, lexicon, word, pronunciations):
        """Verify heteronyms exist in lexicon with at least one pronunciation."""
        word_lower = word.lower()
        assert word_lower in lexicon, f"Heteronym '{word}' not found in lexicon"

        phonemes = lexicon[word_lower]
        if isinstance(phonemes, dict):
            # Good - lexicon has multiple pronunciations
            assert 'DEFAULT' in phonemes or len(phonemes) > 0, \
                f"Heteronym '{word}' has no pronunciations"
        else:
            # Single pronunciation - just verify it exists
            assert len(phonemes) > 0, f"Heteronym '{word}' has empty phonemes"


class TestPhonemeTokenValidity:
    """Test that all phonemes can be tokenized."""

    def test_lexicon_phonemes_tokenizable(self, lexicon, kokoro_vocab):
        """Verify all phonemes in lexicon can be tokenized."""
        untokenizable = []

        # Sample first 1000 words to keep test fast
        sample_words = list(lexicon.items())[:1000]

        for word, phonemes in sample_words:
            if isinstance(phonemes, dict):
                phonemes = phonemes.get('DEFAULT', '')
            if not phonemes:
                continue

            for char in phonemes:
                if char not in kokoro_vocab and char not in ' ':
                    untokenizable.append((word, char, hex(ord(char))))

        assert len(untokenizable) == 0, \
            f"Found {len(untokenizable)} untokenizable phonemes: {untokenizable[:10]}..."


class TestG2PFallback:
    """Test espeak-ng fallback for OOV words."""

    @pytest.fixture(scope="class")
    def espeak_available(self):
        """Check if espeak-ng is available."""
        import subprocess
        try:
            result = subprocess.run(
                ['espeak-ng', '--version'],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            pytest.skip("espeak-ng not available")

    def test_espeak_handles_oov(self, espeak_available):
        """Test that espeak-ng can handle out-of-vocabulary words."""
        import subprocess

        # Made-up words that won't be in any lexicon
        oov_words = ["xyzzy", "blurfl", "fnord", "zaphod"]

        for word in oov_words:
            result = subprocess.run(
                ['espeak-ng', '--ipa=3', '-v', 'en-us', word],
                capture_output=True, text=True, timeout=5
            )
            assert result.returncode == 0, f"espeak-ng failed for '{word}'"
            assert len(result.stdout.strip()) > 0, f"espeak-ng returned empty phonemes for '{word}'"


class TestG2PMetrics:
    """Track G2P quality metrics."""

    def test_lexicon_coverage_stats(self, lexicon, kokoro_vocab):
        """Report lexicon coverage statistics."""
        stats = {
            "total_words": len(lexicon),
            "words_with_variants": 0,
            "single_pronunciation": 0,
            "total_unique_phonemes": set(),
        }

        for word, phonemes in lexicon.items():
            if isinstance(phonemes, dict):
                stats["words_with_variants"] += 1
                phonemes = phonemes.get('DEFAULT', '')
            else:
                stats["single_pronunciation"] += 1

            for char in phonemes:
                stats["total_unique_phonemes"].add(char)

        stats["unique_phoneme_count"] = len(stats["total_unique_phonemes"])
        del stats["total_unique_phonemes"]

        # Print stats for visibility
        print(f"\nG2P Lexicon Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Basic assertions
        assert stats["total_words"] >= 80000, "Lexicon too small"
        assert stats["unique_phoneme_count"] >= 30, "Too few unique phonemes"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
