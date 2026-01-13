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
Phoneme to Word Mapper for Singing Lyrics Detection.

Maps phoneme sequences (IPA or ARPAbet) back to English words using CMU dict.
Key use case: transcribing singing where Whisper may struggle but phonemes are clear.

Strategy:
1. Always output phonemes - captures exactly what was sung
2. Attempt word mapping - if phonemes match dictionary
3. Confidence score - how likely it's a real word vs vocalization

Output examples:
    /h ɛ l oʊ/ → "hello" (conf: 0.95)
    /j ɛ ɛ ɛ/ → "[yeahhh]" (conf: 0.3)  -- brackets = non-word
    /l ɑ l ɑ l ɑ/ → "[la la la]" (conf: 0.1)

Usage:
    from tools.whisper_mlx.phoneme_to_word import PhonemeToWordMapper

    mapper = PhonemeToWordMapper()
    result = mapper.map_phonemes("HH AH L OW")  # ARPAbet
    # Result: {"word": "hello", "confidence": 0.95, "phonemes": "HH AH L OW"}

    result = mapper.map_phonemes_ipa("həloʊ")  # IPA
    # Result: {"word": "hello", "confidence": 0.95, "phonemes": "həloʊ"}

References:
    - CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    - RICHDECODER_V3_ULTRAPLAN.md - Singing Lyrics Pipeline
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# ARPAbet to IPA Mapping
# =============================================================================

# ARPAbet (CMU dict format) to IPA conversion
ARPABET_TO_IPA = {
    # Vowels
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
    'AY': 'aɪ', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ',
    'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u',
    # Consonants
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f',
    'G': 'g', 'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l',
    'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'ɹ',
    'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}

# IPA to ARPAbet (for reverse mapping)
IPA_TO_ARPABET = {v: k for k, v in ARPABET_TO_IPA.items()}

# Extended IPA mappings for variants
IPA_VARIANTS = {
    'ə': 'AH',  # schwa
    'ɚ': 'ER',  # r-colored schwa
    'ɔː': 'AO',  # long o
    'ɑː': 'AA',  # long a
    'iː': 'IY',  # long i
    'uː': 'UW',  # long u
    'ɛː': 'EH',  # long e (rare)
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WordMatch:
    """Result of phoneme-to-word mapping."""
    word: str
    confidence: float
    phonemes_arpabet: str  # Original ARPAbet
    phonemes_ipa: str      # IPA representation
    is_exact_match: bool   # True if exact dictionary match
    alternatives: list[str]  # Other possible words


@dataclass
class MappingStats:
    """Statistics for mapper performance."""
    total_queries: int = 0
    exact_matches: int = 0
    fuzzy_matches: int = 0
    no_matches: int = 0

    @property
    def exact_rate(self) -> float:
        return self.exact_matches / self.total_queries if self.total_queries > 0 else 0.0


# =============================================================================
# Phoneme to Word Mapper
# =============================================================================

class PhonemeToWordMapper:
    """
    Map phoneme sequences to English words using CMU dict.

    Supports both ARPAbet (CMU dict native) and IPA phoneme inputs.
    Uses fuzzy matching for approximate phoneme sequences.
    """

    def __init__(
        self,
        dict_path: str | None = None,
        min_word_length: int = 1,
        max_alternatives: int = 3,
    ):
        """
        Initialize mapper.

        Args:
            dict_path: Path to CMU dict file (downloads if not provided)
            min_word_length: Minimum word length to consider
            max_alternatives: Maximum alternative matches to return
        """
        self.min_word_length = min_word_length
        self.max_alternatives = max_alternatives
        self.stats = MappingStats()

        # Phoneme -> [words] mapping (reversed CMU dict)
        self.phoneme_to_words: dict[str, list[str]] = defaultdict(list)

        # Word -> [phoneme variants] (original CMU dict with variants)
        self.word_to_phonemes: dict[str, list[str]] = {}

        # Load dictionary
        self._load_dict(dict_path)

    def _load_dict(self, dict_path: str | None):
        """Load CMU pronouncing dictionary."""
        if dict_path and Path(dict_path).exists():
            self._load_from_file(dict_path)
        else:
            # Try common locations
            common_paths = [
                Path.home() / ".local/share/cmudict/cmudict.dict",
                Path("/usr/share/dict/cmudict.dict"),
                Path(__file__).parent / "data" / "cmudict.dict",
            ]
            for path in common_paths:
                if path.exists():
                    self._load_from_file(str(path))
                    return

            # Use built-in minimal dictionary
            print("CMU dict not found, using built-in minimal dictionary")
            self._load_builtin_dict()

    def _load_from_file(self, path: str):
        """Load CMU dict from file."""
        count = 0
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';;;'):
                    continue

                # Format: WORD  PHONEME1 PHONEME2 ...
                # Or: WORD(1)  PHONEME1 PHONEME2 ...  (variant)
                parts = line.split()
                if len(parts) < 2:
                    continue

                word = parts[0].upper()
                # Remove variant number e.g., WORD(1) -> WORD
                word = re.sub(r'\(\d+\)$', '', word)

                phonemes = ' '.join(parts[1:])
                # Remove stress markers (0, 1, 2)
                phonemes_clean = re.sub(r'[012]', '', phonemes)

                # Store both directions
                if word not in self.word_to_phonemes:
                    self.word_to_phonemes[word] = []
                self.word_to_phonemes[word].append(phonemes_clean)

                self.phoneme_to_words[phonemes_clean].append(word)
                count += 1

        print(f"Loaded {count} entries, {len(self.word_to_phonemes)} unique words")

    def _load_builtin_dict(self):
        """Load minimal built-in dictionary for common words."""
        # Top 200 English words with phonemes
        builtin = {
            "THE": "DH AH",
            "BE": "B IY",
            "TO": "T UW",
            "OF": "AH V",
            "AND": "AE N D",
            "A": "AH",
            "IN": "IH N",
            "THAT": "DH AE T",
            "HAVE": "HH AE V",
            "I": "AY",
            "IT": "IH T",
            "FOR": "F AO R",
            "NOT": "N AA T",
            "ON": "AA N",
            "WITH": "W IH DH",
            "HE": "HH IY",
            "AS": "AE Z",
            "YOU": "Y UW",
            "DO": "D UW",
            "AT": "AE T",
            "THIS": "DH IH S",
            "BUT": "B AH T",
            "HIS": "HH IH Z",
            "BY": "B AY",
            "FROM": "F R AH M",
            "THEY": "DH EY",
            "WE": "W IY",
            "SAY": "S EY",
            "HER": "HH ER",
            "SHE": "SH IY",
            "OR": "AO R",
            "WILL": "W IH L",
            "MY": "M AY",
            "ONE": "W AH N",
            "ALL": "AO L",
            "WOULD": "W UH D",
            "THERE": "DH EH R",
            "WHAT": "W AH T",
            "SO": "S OW",
            "UP": "AH P",
            "OUT": "AW T",
            "IF": "IH F",
            "ABOUT": "AH B AW T",
            "WHO": "HH UW",
            "GET": "G EH T",
            "WHICH": "W IH CH",
            "GO": "G OW",
            "ME": "M IY",
            "WHEN": "W EH N",
            "MAKE": "M EY K",
            "CAN": "K AE N",
            "LIKE": "L AY K",
            "TIME": "T AY M",
            "NO": "N OW",
            "JUST": "JH AH S T",
            "HIM": "HH IH M",
            "KNOW": "N OW",
            "TAKE": "T EY K",
            "PEOPLE": "P IY P AH L",
            "INTO": "IH N T UW",
            "YEAR": "Y IH R",
            "YOUR": "Y AO R",
            "GOOD": "G UH D",
            "SOME": "S AH M",
            "COULD": "K UH D",
            "THEM": "DH EH M",
            "SEE": "S IY",
            "OTHER": "AH DH ER",
            "THAN": "DH AE N",
            "THEN": "DH EH N",
            "NOW": "N AW",
            "LOOK": "L UH K",
            "ONLY": "OW N L IY",
            "COME": "K AH M",
            "ITS": "IH T S",
            "OVER": "OW V ER",
            "THINK": "TH IH NG K",
            "ALSO": "AO L S OW",
            "BACK": "B AE K",
            "AFTER": "AE F T ER",
            "USE": "Y UW Z",
            "TWO": "T UW",
            "HOW": "HH AW",
            "WORK": "W ER K",
            "FIRST": "F ER S T",
            "WELL": "W EH L",
            "WAY": "W EY",
            "EVEN": "IY V AH N",
            "NEW": "N UW",
            "WANT": "W AA N T",
            "BECAUSE": "B IH K AO Z",
            "ANY": "EH N IY",
            "THESE": "DH IY Z",
            "GIVE": "G IH V",
            "DAY": "D EY",
            "MOST": "M OW S T",
            "US": "AH S",
            # Common singing words
            "LOVE": "L AH V",
            "BABY": "B EY B IY",
            "HEART": "HH AA R T",
            "OH": "OW",
            "OHHH": "OW",
            "LA": "L AA",
            "NA": "N AA",
            "DA": "D AA",
            "YEAH": "Y EH",
            "HEY": "HH EY",
            "GONNA": "G AA N AH",
            "WANNA": "W AA N AH",
            "GOTTA": "G AA T AH",
            "HELLO": "HH AH L OW",
            "WORLD": "W ER L D",
            "NIGHT": "N AY T",
            "DANCE": "D AE N S",
            "MUSIC": "M Y UW Z IH K",
            "SING": "S IH NG",
            "SONG": "S AO NG",
            "DREAM": "D R IY M",
            "FOREVER": "F ER EH V ER",
            "NEVER": "N EH V ER",
            "ALWAYS": "AO L W EY Z",
            "TONIGHT": "T AH N AY T",
            "FEEL": "F IY L",
            "BEAUTIFUL": "B Y UW T AH F AH L",
        }

        for word, phonemes in builtin.items():
            self.word_to_phonemes[word] = [phonemes]
            self.phoneme_to_words[phonemes].append(word)

        print(f"Loaded {len(builtin)} built-in words")

    def arpabet_to_ipa(self, arpabet: str) -> str:
        """Convert ARPAbet phoneme string to IPA."""
        phones = arpabet.split()
        ipa_phones = []
        for p in phones:
            # Remove stress markers
            p_clean = re.sub(r'[012]', '', p)
            ipa = ARPABET_TO_IPA.get(p_clean, p_clean.lower())
            ipa_phones.append(ipa)
        return ''.join(ipa_phones)

    def ipa_to_arpabet(self, ipa: str) -> str:
        """Convert IPA string to ARPAbet (approximate)."""
        # This is lossy - IPA has more detail than ARPAbet
        arpabet = []
        i = 0
        while i < len(ipa):
            # Try 2-char sequences first (digraphs)
            if i + 1 < len(ipa):
                digraph = ipa[i:i+2]
                if digraph in IPA_TO_ARPABET:
                    arpabet.append(IPA_TO_ARPABET[digraph])
                    i += 2
                    continue
                if digraph in IPA_VARIANTS:
                    arpabet.append(IPA_VARIANTS[digraph])
                    i += 2
                    continue

            # Single char
            char = ipa[i]
            if char in IPA_TO_ARPABET:
                arpabet.append(IPA_TO_ARPABET[char])
            elif char in IPA_VARIANTS:
                arpabet.append(IPA_VARIANTS[char])
            elif char.upper() in ARPABET_TO_IPA:
                arpabet.append(char.upper())
            # Skip stress markers, spaces, etc.
            i += 1

        return ' '.join(arpabet)

    def map_phonemes(self, phonemes_arpabet: str) -> WordMatch:
        """
        Map ARPAbet phoneme sequence to word.

        Args:
            phonemes_arpabet: Space-separated ARPAbet phonemes (e.g., "HH AH L OW")

        Returns:
            WordMatch with word, confidence, and alternatives
        """
        self.stats.total_queries += 1

        # Normalize
        phonemes_clean = re.sub(r'[012]', '', phonemes_arpabet.upper()).strip()

        # Exact match
        if phonemes_clean in self.phoneme_to_words:
            words = self.phoneme_to_words[phonemes_clean]
            self.stats.exact_matches += 1
            return WordMatch(
                word=words[0].lower(),
                confidence=0.95,
                phonemes_arpabet=phonemes_clean,
                phonemes_ipa=self.arpabet_to_ipa(phonemes_clean),
                is_exact_match=True,
                alternatives=[w.lower() for w in words[1:self.max_alternatives+1]],
            )

        # Fuzzy match - try common variations
        alternatives = self._fuzzy_match(phonemes_clean)
        if alternatives:
            self.stats.fuzzy_matches += 1
            return WordMatch(
                word=alternatives[0].lower(),
                confidence=0.6,
                phonemes_arpabet=phonemes_clean,
                phonemes_ipa=self.arpabet_to_ipa(phonemes_clean),
                is_exact_match=False,
                alternatives=[w.lower() for w in alternatives[1:self.max_alternatives+1]],
            )

        # No match - return phonemes as "non-word"
        self.stats.no_matches += 1
        ipa = self.arpabet_to_ipa(phonemes_clean)
        return WordMatch(
            word=f"[{ipa}]",  # Brackets indicate non-word
            confidence=0.1,
            phonemes_arpabet=phonemes_clean,
            phonemes_ipa=ipa,
            is_exact_match=False,
            alternatives=[],
        )

    def _fuzzy_match(self, phonemes: str) -> list[str]:
        """Find approximate matches for phoneme sequence."""
        matches = []
        phones = phonemes.split()

        # Try with reduced phonemes (common reductions in speech)
        reductions = [
            # Schwa reduction
            (' AH ', ' '),
            # Common final drops
            (' T$', ''),
            (' D$', ''),
            # Common mergers
            ('AO', 'AA'),
            ('IH', 'EH'),
        ]

        for pattern, replacement in reductions:
            reduced = re.sub(pattern, replacement, phonemes)
            if reduced != phonemes and reduced in self.phoneme_to_words:
                matches.extend(self.phoneme_to_words[reduced])

        # Try prefix matches (partial words)
        for stored_phonemes, words in self.phoneme_to_words.items():
            if stored_phonemes.startswith(phonemes) or phonemes.startswith(stored_phonemes):
                if abs(len(stored_phonemes.split()) - len(phones)) <= 2:
                    matches.extend(words)

        return list(dict.fromkeys(matches))[:self.max_alternatives * 2]

    def map_phonemes_ipa(self, phonemes_ipa: str) -> WordMatch:
        """
        Map IPA phoneme sequence to word.

        Args:
            phonemes_ipa: IPA phoneme string (e.g., "həloʊ")

        Returns:
            WordMatch with word, confidence, and alternatives
        """
        arpabet = self.ipa_to_arpabet(phonemes_ipa)
        return self.map_phonemes(arpabet)

    def get_word_phonemes(self, word: str) -> list[str] | None:
        """Look up phonemes for a word."""
        return self.word_to_phonemes.get(word.upper())

    def get_stats(self) -> MappingStats:
        """Get mapping statistics."""
        return self.stats


# =============================================================================
# Singing Lyrics Processor
# =============================================================================

class SingingLyricsProcessor:
    """
    Process phoneme sequences from singing into lyrics.

    Handles:
    - Word detection from phoneme sequences
    - Vocalization detection (non-word sounds)
    - Confidence scoring
    - Grouping into phrases
    """

    def __init__(self, mapper: PhonemeToWordMapper | None = None):
        self.mapper = mapper or PhonemeToWordMapper()

        # Patterns for common vocalizations
        self.vocalization_patterns = [
            r'^(L AA\s*)+$',           # "la la la"
            r'^(N AA\s*)+$',           # "na na na"
            r'^(D AA\s*)+$',           # "da da da"
            r'^(OW\s*)+$',             # "oh oh oh"
            r'^(AH\s*)+$',             # extended vowel
            r'^(Y EH\s*)+$',           # "yeah yeah"
            r'^HH (AA|AH|EH|IH)\s*$',  # "hah", "heh", etc.
        ]

    def is_vocalization(self, phonemes: str) -> bool:
        """Check if phoneme sequence is likely a vocalization."""
        for pattern in self.vocalization_patterns:
            if re.match(pattern, phonemes):
                return True
        return False

    def process_sequence(
        self,
        phoneme_sequence: list[str],
        timestamps: list[float] | None = None,
    ) -> list[dict]:
        """
        Process a sequence of phonemes into words/vocalizations.

        Args:
            phoneme_sequence: List of ARPAbet phoneme strings
            timestamps: Optional timestamps for each phoneme

        Returns:
            List of dicts with word/vocalization info
        """
        results = []

        for i, phonemes in enumerate(phoneme_sequence):
            timestamp = timestamps[i] if timestamps else None

            if self.is_vocalization(phonemes):
                ipa = self.mapper.arpabet_to_ipa(phonemes)
                results.append({
                    "type": "vocalization",
                    "text": f"[{ipa}]",
                    "phonemes": phonemes,
                    "confidence": 0.1,
                    "timestamp": timestamp,
                })
            else:
                match = self.mapper.map_phonemes(phonemes)
                results.append({
                    "type": "word" if match.is_exact_match else "partial",
                    "text": match.word,
                    "phonemes": match.phonemes_arpabet,
                    "phonemes_ipa": match.phonemes_ipa,
                    "confidence": match.confidence,
                    "alternatives": match.alternatives,
                    "timestamp": timestamp,
                })

        return results


# =============================================================================
# CLI and Testing
# =============================================================================

def test_mapper():
    """Test phoneme to word mapping."""
    print("Testing PhonemeToWordMapper...")

    mapper = PhonemeToWordMapper()

    # Test exact matches
    test_cases = [
        ("HH AH L OW", "hello"),
        ("W ER L D", "world"),
        ("L AH V", "love"),
        ("M Y UW Z IH K", "music"),
    ]

    print("\nExact match tests:")
    for phonemes, expected in test_cases:
        result = mapper.map_phonemes(phonemes)
        status = "PASS" if result.word == expected else "FAIL"
        print(f"  {status}: '{phonemes}' -> '{result.word}' (conf={result.confidence:.2f})")

    # Test IPA conversion
    print("\nIPA conversion tests:")
    ipa_tests = [
        ("HH AH L OW", "hʌloʊ"),
        ("W ER L D", "wɝld"),
    ]
    for arpabet, expected_ipa in ipa_tests:
        ipa = mapper.arpabet_to_ipa(arpabet)
        status = "PASS" if ipa == expected_ipa else "FAIL"
        print(f"  {status}: '{arpabet}' -> '{ipa}' (expected '{expected_ipa}')")

    # Test unknown phonemes
    print("\nUnknown phoneme test:")
    result = mapper.map_phonemes("ZZ XX YY")
    print(f"  Unknown: '{result.word}' (conf={result.confidence:.2f})")

    # Statistics
    print(f"\nMapper stats: exact_rate={mapper.stats.exact_rate:.1%}")

    print("\nAll tests complete.")


def main():
    """CLI for phoneme-to-word mapping."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python phoneme_to_word.py <phonemes>")
        print("       python phoneme_to_word.py --test")
        print()
        print("Examples:")
        print("  python phoneme_to_word.py 'HH AH L OW'")
        print("  python phoneme_to_word.py --ipa 'həloʊ'")
        return

    if sys.argv[1] == "--test":
        test_mapper()
        return

    mapper = PhonemeToWordMapper()

    if sys.argv[1] == "--ipa" and len(sys.argv) > 2:
        phonemes = sys.argv[2]
        result = mapper.map_phonemes_ipa(phonemes)
    else:
        phonemes = ' '.join(sys.argv[1:])
        result = mapper.map_phonemes(phonemes)

    print(f"Word: {result.word}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Phonemes (ARPAbet): {result.phonemes_arpabet}")
    print(f"Phonemes (IPA): {result.phonemes_ipa}")
    print(f"Exact match: {result.is_exact_match}")
    if result.alternatives:
        print(f"Alternatives: {', '.join(result.alternatives)}")


if __name__ == "__main__":
    main()
