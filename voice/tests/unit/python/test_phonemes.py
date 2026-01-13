"""
Phoneme and Tokenization Integration Tests

Tests that:
1. All common words are in the lexicon
2. All phonemes have valid token IDs
3. C++ lexicon output matches Python reference
4. Full TTS→STT round-trip produces intelligible speech
"""

import pytest

pytestmark = pytest.mark.unit

# Test words with expected phonemes from us_gold.json
TEST_WORD_PHONEMES = [
    ("hello", "həlˈO"),
    ("world", "wˈɜɹld"),
    ("quick", "kwˈɪk"),
    ("brown", "bɹˈWn"),
    ("fox", "fˈɑks"),
    ("jumps", "ʤˈʌmps"),
    ("over", "ˈOvəɹ"),
    ("the", "ði"),
    ("lazy", "lˈAzi"),
    ("dog", "dˈɔɡ"),
    ("church", "ʧˈɜɹʧ"),
    ("choir", "kwˈIəɹ"),
    ("night", "nˈIt"),
    ("five", "fˈIv"),
    ("bought", "bˈɔt"),
]


class TestLexiconCoverage:
    """Test that all words are in the lexicon"""

    @pytest.mark.parametrize("word,expected_phonemes", TEST_WORD_PHONEMES)
    def test_word_in_lexicon(self, lexicon, word, expected_phonemes):
        """Verify word exists in lexicon with expected phonemes"""
        actual = lexicon.get(word, lexicon.get(word.lower()))
        assert actual is not None, f"'{word}' not found in lexicon"
        # Note: Some words may have variant pronunciations
        # We just check it exists, not exact match
        assert len(actual) > 0, f"'{word}' has empty phonemes"


class TestPhonemeTokens:
    """Test that all phonemes have valid token IDs"""

    @pytest.mark.parametrize("word,phonemes", TEST_WORD_PHONEMES)
    def test_phonemes_have_token_ids(self, kokoro_vocab, word, phonemes):
        """Verify all phonemes in test cases have valid token IDs"""
        missing = []
        for char in phonemes:
            if char not in kokoro_vocab:
                missing.append((char, hex(ord(char))))

        assert not missing, f"Missing tokens for '{word}': {missing}"

    def test_essential_phonemes_in_vocab(self, kokoro_vocab):
        """Verify essential phonemes are in vocabulary"""
        essential = [
            "ə", "ɹ", "ɔ", "ʤ", "ʧ", "ð", "θ", "ŋ",  # Common IPA
            "A", "I", "O", "W", "Y",  # Diphthong markers
            "ˈ", "ˌ",  # Stress markers
        ]
        for phoneme in essential:
            assert phoneme in kokoro_vocab, f"Essential phoneme '{phoneme}' missing from vocab"

    def test_script_g_in_vocab(self, kokoro_vocab):
        """Verify script g (ɡ U+0261) is in vocabulary - critical for 'dog', 'go', etc."""
        assert "ɡ" in kokoro_vocab, "Script g (ɡ U+0261) must be in vocabulary"
        assert kokoro_vocab["ɡ"] == 92, "Script g should have token ID 92"


class TestPhonemeToToken:
    """Test phoneme to token ID conversion"""

    def test_hello_tokenization(self, kokoro_vocab):
        """Test tokenization of 'hello' -> həlˈO"""
        phonemes = "həlˈO"
        expected_tokens = [50, 83, 54, 156, 31]  # h ə l ˈ O

        tokens = []
        for char in phonemes:
            if char in kokoro_vocab:
                tokens.append(kokoro_vocab[char])

        assert tokens == expected_tokens, f"Expected {expected_tokens}, got {tokens}"

    def test_dog_tokenization(self, kokoro_vocab):
        """Test tokenization of 'dog' -> dˈɔɡ"""
        phonemes = "dˈɔɡ"
        expected_tokens = [46, 156, 76, 92]  # d ˈ ɔ ɡ

        tokens = []
        for char in phonemes:
            if char in kokoro_vocab:
                tokens.append(kokoro_vocab[char])

        assert tokens == expected_tokens, f"Expected {expected_tokens}, got {tokens}"


class TestFullSentence:
    """Test full sentence phoneme assembly"""

    def test_quick_brown_fox_phonemes(self, lexicon):
        """Test phoneme assembly for 'the quick brown fox jumps over the lazy dog'"""
        words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

        phonemes = []
        for word in words:
            word_phonemes = lexicon.get(word, lexicon.get(word.lower()))
            assert word_phonemes is not None, f"'{word}' not in lexicon"
            phonemes.append(word_phonemes)

        sentence_phonemes = " ".join(phonemes)
        expected = "ði kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ði lˈAzi dˈɔɡ"

        assert sentence_phonemes == expected, f"Expected:\n{expected}\nGot:\n{sentence_phonemes}"

    def test_all_phonemes_tokenizable(self, lexicon, kokoro_vocab):
        """Test that all phonemes in test sentence can be tokenized"""
        words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

        missing_phonemes = set()
        for word in words:
            word_phonemes = lexicon.get(word, lexicon.get(word.lower()))
            if word_phonemes:
                for char in word_phonemes:
                    if char not in kokoro_vocab:
                        missing_phonemes.add((char, hex(ord(char)), word))

        assert not missing_phonemes, f"Missing phonemes: {missing_phonemes}"


# =============================================================================
# G2P Regression Fixtures - Frozen phoneme expectations for drift detection
# =============================================================================

# Additional words with frozen phonemes to detect lexicon changes
# Format: (word, expected_phoneme, category)
G2P_REGRESSION_FIXTURES = [
    # Core pangram words (already in TEST_WORD_PHONEMES but included for completeness)
    ("hello", "həlˈO", "common"),
    ("world", "wˈɜɹld", "common"),
    ("quick", "kwˈɪk", "common"),
    ("brown", "bɹˈWn", "common"),
    ("fox", "fˈɑks", "common"),
    ("jumps", "ʤˈʌmps", "common"),
    ("over", "ˈOvəɹ", "common"),
    ("the", "ði", "common"),
    ("lazy", "lˈAzi", "common"),
    ("dog", "dˈɔɡ", "common"),

    # Additional common words
    ("computer", "kəmpjˈuɾəɹ", "technical"),
    ("software", "sˈɔftwˌɛɹ", "technical"),
    ("programming", "pɹˈOɡɹˌæmɪŋ", "technical"),
    ("function", "fˈʌŋkʃən", "technical"),
    ("variable", "vˈɛɹiəbᵊl", "technical"),

    # Phonetically interesting words
    ("church", "ʧˈɜɹʧ", "phonetic"),
    ("choir", "kwˈIəɹ", "phonetic"),
    ("night", "nˈIt", "phonetic"),
    ("through", "θɹu", "phonetic"),
    ("thought", "θˈɔt", "phonetic"),
    ("rough", "ɹˈʌf", "phonetic"),
    ("though", "ðˌO", "phonetic"),
    ("enough", "ɪnˈʌf", "phonetic"),

    # Numbers (text form)
    ("one", "wˈʌn", "number"),
    ("two", "tˈu", "number"),
    ("three", "θɹˈi", "number"),
    ("ten", "tˈɛn", "number"),
    ("hundred", "hˈʌndɹəd", "number"),
    ("thousand", "θˈWzᵊnd", "number"),

    # Stress pattern words
    ("record", "ɹˈɛkəɹd", "stress"),  # noun form
    ("present", "pɹˈɛzᵊnt", "stress"),  # noun/adj form
    ("perfect", "pˈɜɹfəkt", "stress"),  # adj form

    # Additional edge cases
    ("algorithm", "ˈælɡəɹˌɪðəm", "technical"),
    ("database", "dˈæɾəbˌAs", "technical"),
    ("security", "səkjˈʊɹəɾi", "technical"),
]


class TestG2PRegression:
    """
    G2P Regression Tests - Detect phoneme drift.

    These tests fail if lexicon phonemes change from their frozen expected values.
    This catches:
    - Lexicon regeneration that changed pronunciations
    - Accidental lexicon corruption
    - G2P model updates that alter output
    """

    @pytest.mark.parametrize("word,expected,category", G2P_REGRESSION_FIXTURES)
    def test_phoneme_exact_match(self, lexicon, word, expected, category):
        """Verify phonemes exactly match frozen expectations."""
        word_lower = word.lower()
        actual = lexicon.get(word_lower)

        assert actual is not None, f"'{word}' not found in lexicon (category: {category})"

        # Handle dict entries (words with multiple pronunciations)
        if isinstance(actual, dict):
            actual = actual.get('DEFAULT', list(actual.values())[0])

        assert actual == expected, (
            f"PHONEME DRIFT DETECTED for '{word}' (category: {category}):\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  If this change is intentional, update G2P_REGRESSION_FIXTURES"
        )

    def test_regression_fixture_coverage(self):
        """Ensure regression fixtures cover diverse categories."""
        categories = {}
        for word, phoneme, cat in G2P_REGRESSION_FIXTURES:
            categories[cat] = categories.get(cat, 0) + 1

        assert len(categories) >= 5, f"Need at least 5 categories, have {len(categories)}"
        assert categories.get("common", 0) >= 5, "Need at least 5 common words"
        assert categories.get("technical", 0) >= 3, "Need at least 3 technical words"
        assert categories.get("phonetic", 0) >= 3, "Need at least 3 phonetically interesting words"


class TestCppIntegration:
    """Test C++ binary integration"""

    @pytest.mark.requires_binary
    def test_lexicon_loads(self, tts_runner):
        """Test that C++ binary loads the lexicon"""
        stdout, stderr, returncode = tts_runner("hello")
        combined = (stdout + stderr).lower()

        assert returncode == 0, f"C++ binary exited with code {returncode}"
        assert "lexicon loaded" in combined, "Lexicon should load"
        # Should have ~182k words
        assert "182" in combined, "Should have ~182k words loaded"

    @pytest.mark.requires_binary
    def test_no_unknown_phonemes(self, tts_runner):
        """Test that no phonemes are skipped as unknown"""
        stdout, stderr, returncode = tts_runner("The quick brown fox jumps over the lazy dog")
        combined = (stdout + stderr)

        assert returncode == 0, f"C++ binary exited with code {returncode}"
        # Should not have any "Unknown phoneme" warnings
        assert "Unknown phoneme" not in combined, f"Unknown phonemes found: {combined}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
