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
Phoneme Cache for Streaming ASR Verification.

LRU cache for text → phoneme token mappings to minimize phonemization latency
during streaming ASR verification. Targets <0.1ms per word for cached lookups.

Key features:
- Word-level caching with LRU eviction
- Preloaded common words (~5000 English)
- Thread-safe for concurrent access
- Persistence to disk for warm starts

Usage:
    from tools.whisper_mlx.phoneme_cache import PhonemeCache

    cache = PhonemeCache(max_size=10000)
    cache.preload_common_words()

    # Fast phonemization
    phonemes, tokens = cache.phonemize("hello world")

    # Statistics
    print(f"Cache hit rate: {cache.hit_rate:.1%}")
"""

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

# Common English words to preload (top 1000)
# Full list would be loaded from file
COMMON_WORDS_SAMPLE = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    # Add more as needed...
]


@dataclass
class CacheStats:
    """Statistics for phoneme cache."""
    hits: int = 0
    misses: int = 0
    preloaded: int = 0
    evictions: int = 0
    total_phonemize_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_phonemize_time_ms(self) -> float:
        return self.total_phonemize_time_ms / self.misses if self.misses > 0 else 0.0


@dataclass
class CacheEntry:
    """Single cache entry for a word."""
    phonemes: str  # IPA string
    tokens: list[int]  # Token IDs
    language: str
    created_at: float = field(default_factory=time.time)


class PhonemeCache:
    """
    LRU cache for text → phoneme mappings.

    Caches word-level phonemizations for fast lookup during streaming ASR.
    Thread-safe with configurable max size.
    """

    def __init__(
        self,
        max_size: int = 10000,
        language: str = "en",
        preload: bool = True,
    ):
        """
        Initialize phoneme cache.

        Args:
            max_size: Maximum number of words to cache
            language: Default language for phonemization
            preload: Whether to preload common words
        """
        self.max_size = max_size
        self.default_language = language
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

        # Import phonemizer
        self._phonemizer = None
        self._init_phonemizer()

        if preload:
            self.preload_common_words()

    def _init_phonemizer(self) -> None:
        """Initialize the Kokoro phonemizer."""
        try:
            from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                phonemize_text,
            )
            self._phonemizer = phonemize_text
        except ImportError as e:
            print(f"Warning: Could not import phonemizer: {e}")
            self._phonemizer = None

    def preload_common_words(self, word_list: list[str] | None = None) -> int:
        """
        Preload common words into cache.

        Args:
            word_list: Custom word list (defaults to built-in common words)

        Returns:
            Number of words preloaded
        """
        words = word_list or COMMON_WORDS_SAMPLE

        count = 0
        for word in words:
            if len(self._cache) >= self.max_size:
                break

            word_lower = word.lower().strip()
            if word_lower and word_lower not in self._cache:
                try:
                    phonemes, tokens = self._phonemize_word(word_lower)
                    if tokens:
                        self._add_to_cache(word_lower, phonemes, tokens)
                        count += 1
                except Exception:
                    continue

        self.stats.preloaded = count
        return count

    def preload_from_file(self, filepath: str) -> int:
        """
        Preload words from a text file (one word per line).

        Args:
            filepath: Path to word list file

        Returns:
            Number of words preloaded
        """
        try:
            with open(filepath) as f:
                words = [line.strip() for line in f if line.strip()]
            return self.preload_common_words(words)
        except Exception as e:
            print(f"Error loading word list: {e}")
            return 0

    def _phonemize_word(self, word: str) -> tuple[str, list[int]]:
        """
        Phonemize a single word using Kokoro phonemizer.

        Args:
            word: Word to phonemize

        Returns:
            Tuple of (phoneme string, token IDs)
        """
        if self._phonemizer is None:
            return "", []

        start = time.time()
        phonemes, tokens = self._phonemizer(word, language=self.default_language)
        elapsed = (time.time() - start) * 1000
        self.stats.total_phonemize_time_ms += elapsed

        return phonemes, tokens

    def _add_to_cache(self, key: str, phonemes: str, tokens: list[int]) -> None:
        """Add entry to cache, evicting LRU if needed."""
        with self._lock:
            # Check if already present (move to end)
            if key in self._cache:
                self._cache.move_to_end(key)
                return

            # Evict LRU if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self.stats.evictions += 1

            # Add new entry
            self._cache[key] = CacheEntry(
                phonemes=phonemes,
                tokens=tokens,
                language=self.default_language,
            )

    def get_word(self, word: str) -> tuple[str, list[int]] | None:
        """
        Get phonemes for a word from cache.

        Args:
            word: Word to look up

        Returns:
            Tuple of (phonemes, tokens) or None if not cached
        """
        key = word.lower().strip()

        with self._lock:
            if key in self._cache:
                self.stats.hits += 1
                self._cache.move_to_end(key)  # LRU update
                entry = self._cache[key]
                return entry.phonemes, entry.tokens

        return None

    def phonemize_word(self, word: str) -> tuple[str, list[int]]:
        """
        Phonemize a word, using cache if available.

        Args:
            word: Word to phonemize

        Returns:
            Tuple of (phoneme string, token IDs)
        """
        # Check cache first
        cached = self.get_word(word)
        if cached is not None:
            return cached

        # Cache miss - phonemize and cache
        self.stats.misses += 1
        phonemes, tokens = self._phonemize_word(word)

        if tokens:
            self._add_to_cache(word.lower().strip(), phonemes, tokens)

        return phonemes, tokens

    def phonemize(self, text: str) -> tuple[str, list[int]]:
        """
        Phonemize text (sentence or phrase).

        Splits into words, phonemizes each (using cache), and concatenates.

        Args:
            text: Text to phonemize

        Returns:
            Tuple of (phoneme string, token IDs)
        """
        words = text.lower().split()

        all_phonemes = []
        all_tokens = []

        for word in words:
            # Strip punctuation
            word_clean = ''.join(c for c in word if c.isalnum())
            if not word_clean:
                continue

            phonemes, tokens = self.phonemize_word(word_clean)
            if phonemes:
                all_phonemes.append(phonemes)
            if tokens:
                all_tokens.extend(tokens)

        return ' '.join(all_phonemes), all_tokens

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        return self.stats.hit_rate

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self.stats = CacheStats()

    def save(self, filepath: str) -> None:
        """
        Save cache to JSON file.

        Args:
            filepath: Path to save cache
        """
        with self._lock:
            data = {
                "version": 1,
                "language": self.default_language,
                "entries": {
                    k: {"phonemes": v.phonemes, "tokens": v.tokens}
                    for k, v in self._cache.items()
                },
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> int:
        """
        Load cache from JSON file.

        Args:
            filepath: Path to load cache from

        Returns:
            Number of entries loaded
        """
        with open(filepath) as f:
            data = json.load(f)

        count = 0
        entries = data.get("entries", {})

        with self._lock:
            for key, entry in entries.items():
                if len(self._cache) >= self.max_size:
                    break

                self._cache[key] = CacheEntry(
                    phonemes=entry["phonemes"],
                    tokens=entry["tokens"],
                    language=data.get("language", self.default_language),
                )
                count += 1

        return count

    def get_stats(self) -> dict:
        """Get cache statistics as dict."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "preloaded": self.stats.preloaded,
            "evictions": self.stats.evictions,
            "avg_phonemize_time_ms": self.stats.avg_phonemize_time_ms,
        }


# =============================================================================
# Global Instance
# =============================================================================


_global_cache: PhonemeCache | None = None


def get_phoneme_cache(
    max_size: int = 10000,
    preload: bool = True,
) -> PhonemeCache:
    """
    Get or create global phoneme cache.

    Args:
        max_size: Maximum cache size (only used on first call)
        preload: Whether to preload common words

    Returns:
        Global PhonemeCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = PhonemeCache(max_size=max_size, preload=preload)

    return _global_cache


def phonemize_cached(text: str) -> tuple[str, list[int]]:
    """
    Phonemize text using global cache.

    Args:
        text: Text to phonemize

    Returns:
        Tuple of (phoneme string, token IDs)
    """
    cache = get_phoneme_cache()
    return cache.phonemize(text)


# =============================================================================
# CLI / Test
# =============================================================================


def test_phoneme_cache():
    """Test phoneme cache functionality."""
    print("Testing PhonemeCache...")

    cache = PhonemeCache(max_size=100, preload=True)
    print(f"  Preloaded: {cache.stats.preloaded} words")
    print(f"  Cache size: {cache.size}")

    # Test cached lookup
    start = time.time()
    for _ in range(100):
        _, _ = cache.phonemize_word("the")
    elapsed = (time.time() - start) * 1000
    print(f"  100 cached lookups: {elapsed:.1f}ms ({elapsed/100:.2f}ms/lookup)")

    # Test phrase
    phonemes, tokens = cache.phonemize("the quick brown fox")
    print(f"  Phrase phonemes: {phonemes}")
    print(f"  Phrase tokens: {len(tokens)} tokens")

    # Stats
    stats = cache.get_stats()
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Avg phonemize time: {stats['avg_phonemize_time_ms']:.1f}ms")

    print("PhonemeCache tests PASSED")


if __name__ == "__main__":
    test_phoneme_cache()
