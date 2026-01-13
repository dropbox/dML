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

"""Kokoro phonemizer module.

Converts text to IPA phonemes and tokenizes them for the Kokoro TTS model.
Loads vocabulary from exported artifacts or HuggingFace config.

This module uses misaki G2P (via .venv_phonemizer) when available,
falling back to espeak-ng if misaki is not installed. Misaki is the
official Kokoro G2P and produces better phoneme sequences.

Includes a G2P cache (D8 optimization) to avoid re-phonemizing identical text.
"""

import json
import subprocess
import sys
import warnings
from functools import lru_cache
from pathlib import Path

_VOCAB_CACHE: dict[str, int] | None = None
_MISAKI_AVAILABLE: Path | bool | None = None

# Voice prefix to language mapping for Kokoro voices
_VOICE_LANGUAGE_MAP = {
    "af_": "en",  # American Female
    "am_": "en",  # American Male
    "bf_": "en",  # British Female
    "bm_": "en",  # British Male
    "jf_": "ja",  # Japanese Female
    "jm_": "ja",  # Japanese Male
    "zf_": "zh",  # Chinese Female
    "zm_": "zh",  # Chinese Male
    "hf_": "hi",  # Hindi Female
    "hm_": "hi",  # Hindi Male
    "ef_": "es",  # Spanish Female
    "em_": "es",  # Spanish Male
    "ff_": "fr",  # French Female
    "fm_": "fr",  # French Male
    "if_": "it",  # Italian Female
    "im_": "it",  # Italian Male
    "pf_": "pt",  # Portuguese Female
    "pm_": "pt",  # Portuguese Male
}


def detect_language_from_voice(voice_name: str) -> str:
    """
    Detect language from Kokoro voice name prefix.

    Args:
        voice_name: Voice name (e.g., "af_heart", "jf_alpha")

    Returns:
        Language code (en, ja, zh, etc.) or "en" as default

    Example:
        >>> detect_language_from_voice("af_heart")
        'en'
        >>> detect_language_from_voice("jf_alpha")
        'ja'
    """
    if voice_name and len(voice_name) >= 3:
        prefix = voice_name[:3].lower()
        return _VOICE_LANGUAGE_MAP.get(prefix, "en")
    return "en"


def detect_language_from_text(text: str) -> str:
    """
    Detect language from text content using character analysis.

    Uses Unicode character ranges to identify Japanese, Chinese, Korean, etc.
    Falls back to English for Latin scripts.

    Args:
        text: Input text to analyze

    Returns:
        Language code (en, ja, zh, ko)

    Example:
        >>> detect_language_from_text("Hello world")
        'en'
        >>> detect_language_from_text("こんにちは")
        'ja'
        >>> detect_language_from_text("你好")
        'zh'
    """
    if not text:
        return "en"

    # Count characters in different Unicode ranges
    hiragana = 0  # Japanese hiragana (3040-309F)
    katakana = 0  # Japanese katakana (30A0-30FF)
    cjk = 0       # CJK unified ideographs (4E00-9FFF)
    hangul = 0    # Korean hangul (AC00-D7AF)
    devanagari = 0  # Hindi devanagari (0900-097F)
    latin = 0

    for char in text:
        code = ord(char)
        if 0x3040 <= code <= 0x309F:
            hiragana += 1
        elif 0x30A0 <= code <= 0x30FF:
            katakana += 1
        elif 0x4E00 <= code <= 0x9FFF:
            cjk += 1
        elif 0xAC00 <= code <= 0xD7AF:
            hangul += 1
        elif 0x0900 <= code <= 0x097F:
            devanagari += 1
        elif 0x0041 <= code <= 0x007A:  # Basic Latin letters
            latin += 1

    # Japanese: hiragana/katakana or CJK with some kana
    if hiragana + katakana > 0:
        return "ja"

    # Korean
    if hangul > 0:
        return "ko"

    # Chinese: CJK without kana
    if cjk > 0:
        return "zh"

    # Hindi
    if devanagari > 0:
        return "hi"

    # Default to English for Latin script
    return "en"


def detect_language(text: str, voice_name: str | None = None) -> str:
    """
    Auto-detect language from voice name and/or text content.

    Priority:
    1. Voice name prefix (most reliable for Kokoro)
    2. Text content analysis (fallback)

    Args:
        text: Input text
        voice_name: Optional voice name (e.g., "af_heart")

    Returns:
        Language code (en, ja, zh, ko, hi, es, fr, it, pt)

    Example:
        >>> detect_language("Hello", voice_name="jf_alpha")
        'ja'  # Voice takes priority
        >>> detect_language("こんにちは")
        'ja'  # Text analysis
    """
    # Voice name takes priority (most reliable)
    if voice_name:
        lang = detect_language_from_voice(voice_name)
        if lang != "en":  # If we got a specific language from voice
            return lang

    # Fall back to text analysis
    return detect_language_from_text(text)


# G2P Cache for phonemization results (D8 optimization)
# Maps (text, language) -> (phonemes, token_ids)
# Using lru_cache for thread-safe, bounded caching
_G2P_CACHE_SIZE = 1024  # Cache up to 1024 unique text/language combinations
_g2p_cache_hits = 0
_g2p_cache_misses = 0

PAD_TOKEN = 0
N_TOKENS = 178


def load_vocab(vocab_path: Path | None = None) -> dict[str, int]:
    """Load Kokoro phoneme vocabulary from file.

    Search order:
    1. Explicit vocab_path if provided
    2. misaki_export/vocab.json (repo artifact)
    3. kokoro_cpp_export/g2p/vocab.json (C++ artifact)
    4. ~/models/kokoro/config.json (HuggingFace download)

    Returns:
        Dict mapping phoneme characters to token IDs
    """
    global _VOCAB_CACHE
    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE

    # Find repo root (directory containing misaki_export/)
    repo_root = Path(__file__).parent
    while repo_root.parent != repo_root:
        if (repo_root / "misaki_export").exists():
            break
        repo_root = repo_root.parent

    search_paths = []
    if vocab_path:
        search_paths.append(vocab_path)
    search_paths.extend(
        [
            repo_root / "misaki_export" / "vocab.json",
            repo_root / "kokoro_cpp_export" / "g2p" / "vocab.json",
            Path.home() / "models" / "kokoro" / "config.json",
        ],
    )

    for path in search_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Handle both vocab.json (direct dict) and config.json (nested)
            vocab = data.get("vocab", data) if isinstance(data, dict) else data
            if isinstance(vocab, dict) and len(vocab) > 50:  # Sanity check
                _VOCAB_CACHE = vocab
                return vocab

    raise FileNotFoundError(
        f"Kokoro vocab not found. Searched: {[str(p) for p in search_paths]}",
    )


def _find_misaki_python() -> Path | None:
    """Find the Python interpreter with misaki installed.

    Search order:
    1. .venv_phonemizer/bin/python in repo root
    2. Direct import (misaki in current environment)

    Returns:
        Path to Python with misaki, or None if not found
    """
    global _MISAKI_AVAILABLE

    if _MISAKI_AVAILABLE is not None:
        return _MISAKI_AVAILABLE if isinstance(_MISAKI_AVAILABLE, Path) else None

    # Find repo root
    repo_root = Path(__file__).parent
    while repo_root.parent != repo_root:
        if (repo_root / "misaki_export").exists() or (
            repo_root / ".venv_phonemizer"
        ).exists():
            break
        repo_root = repo_root.parent

    # Check .venv_phonemizer
    venv_python = repo_root / ".venv_phonemizer" / "bin" / "python"
    if venv_python.exists():
        try:
            result = subprocess.run(
                [str(venv_python), "-c", "from misaki import en; print('ok')"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                _MISAKI_AVAILABLE = venv_python
                return venv_python
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Check current environment
    try:
        from misaki import en  # noqa: F401

        _MISAKI_AVAILABLE = Path(sys.executable)
        return Path(sys.executable)
    except ImportError:
        pass

    _MISAKI_AVAILABLE = False  # type: ignore[assignment]
    return None


def _phonemize_with_misaki(
    text: str, misaki_python: Path, language: str = "en",
) -> tuple[str, list[int]] | None:
    """Phonemize text using misaki G2P via subprocess.

    Args:
        text: Input text to phonemize
        misaki_python: Path to Python with misaki installed
        language: Language code (en, ja, zh, ko)

    Returns:
        Tuple of (phonemes, token_ids) or None if failed
    """
    # Find the phonemize script
    repo_root = Path(__file__).parent
    while repo_root.parent != repo_root:
        if (repo_root / "scripts" / "phonemize_for_kokoro.py").exists():
            break
        repo_root = repo_root.parent

    script_path = repo_root / "scripts" / "phonemize_for_kokoro.py"
    if not script_path.exists():
        return None

    try:
        result = subprocess.run(
            [str(misaki_python), str(script_path), "--lang", language, text],
            capture_output=True,
            text=True,
            timeout=30,  # Increased timeout for slower languages
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        return data["phonemes"], data["token_ids"]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError):
        return None


def _phonemize_with_espeak(text: str, language: str = "en-us") -> tuple[str, list[int]]:
    """Phonemize text using espeak-ng (fallback).

    Args:
        text: Input text to convert
        language: Language code for espeak-ng

    Returns:
        Tuple of (phoneme_string, token_ids)
    """
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    vocab = load_vocab()

    # Use separators to get word boundaries
    separator = Separator(phone="", word=" ", syllable="")

    # Phonemize the text using espeak backend
    phonemes = phonemize(
        text,
        language=language,
        backend="espeak",
        separator=separator,
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )

    # Clean up phonemes - remove extra spaces
    phonemes = " ".join(phonemes.split())

    # Tokenize: convert each character to token ID
    # CRITICAL: Emit PAD (0) for unknown chars to preserve length alignment
    # Voice indexing uses phoneme string length, so token count must match
    token_ids = [PAD_TOKEN]  # Start with BOS token
    unknown_chars = set()
    for char in phonemes:
        if char in vocab:
            token_ids.append(vocab[char])
        else:
            # Emit PAD token to preserve length alignment with phoneme string
            token_ids.append(PAD_TOKEN)
            unknown_chars.add(char)
    token_ids.append(PAD_TOKEN)  # End with EOS token

    if unknown_chars:
        warnings.warn(
            f"Unknown phoneme characters (mapped to PAD): {sorted(unknown_chars)!r}. "
            "This may affect voice quality. Consider updating vocabulary.",
            stacklevel=2,
        )

    return phonemes, token_ids


@lru_cache(maxsize=_G2P_CACHE_SIZE)
def _phonemize_text_cached(
    text: str, language: str, prefer_misaki: bool,
) -> tuple[str, tuple[int, ...]]:
    """Internal cached phonemization function.

    Returns tuple of (phonemes, tuple(token_ids)) where token_ids is a tuple
    for hashability in the cache.
    """
    global _g2p_cache_misses
    _g2p_cache_misses += 1

    # Normalize language code for misaki (en-us -> en, ja-jp -> ja, etc.)
    misaki_lang = language.split("-")[0].lower()

    # Try misaki first if preferred
    if prefer_misaki:
        misaki_python = _find_misaki_python()
        if misaki_python:
            result = _phonemize_with_misaki(text, misaki_python, misaki_lang)
            if result is not None:
                return (result[0], tuple(result[1]))

    # Fall back to espeak (use full language code like en-us)
    espeak_lang = language if "-" in language else f"{language}-us" if language == "en" else language
    phonemes, token_ids = _phonemize_with_espeak(text, espeak_lang)
    return (phonemes, tuple(token_ids))


def phonemize_text(
    text: str, language: str = "en", prefer_misaki: bool = True,
) -> tuple[str, list[int]]:
    """Convert text to IPA phonemes and token IDs.

    Uses misaki G2P when available (preferred), falls back to espeak-ng.
    Misaki produces phonemes that match the official Kokoro G2P.

    Results are cached (D8 optimization) to avoid re-phonemizing identical text.
    Cache size: 1024 entries (LRU eviction).

    Args:
        text: Input text to convert
        language: Language code (en, ja, zh, ko for misaki; en-us etc for espeak fallback)
        prefer_misaki: Try misaki first if available (default: True)

    Returns:
        Tuple of (phoneme_string, token_ids)
        - phoneme_string: IPA phoneme representation
        - token_ids: List of token IDs (with PAD tokens at start/end)
    """
    global _g2p_cache_hits

    # Check if result will come from cache (lru_cache handles this internally)
    cache_info_before = _phonemize_text_cached.cache_info()

    phonemes, token_ids_tuple = _phonemize_text_cached(text, language, prefer_misaki)

    # Track cache hits for monitoring
    cache_info_after = _phonemize_text_cached.cache_info()
    if cache_info_after.hits > cache_info_before.hits:
        _g2p_cache_hits += 1

    return (phonemes, list(token_ids_tuple))


def tokenize_phonemes(
    phonemes: str, vocab: dict[str, int] | None = None,
) -> list[int]:
    """Convert IPA phoneme string to token IDs.

    CRITICAL: Unknown characters are mapped to PAD (0) to preserve length alignment.
    Voice indexing uses phoneme string length, so token count must match.

    Args:
        phonemes: IPA phoneme string
        vocab: Optional vocabulary dict (loaded if not provided)

    Returns:
        List of token IDs (with BOS/EOS tokens at start/end)
        Length = len(phonemes) + 2 (BOS + phonemes + EOS)
    """
    if vocab is None:
        vocab = load_vocab()

    token_ids = [PAD_TOKEN]  # BOS
    unknown_chars = set()
    for char in phonemes:
        if char in vocab:
            token_ids.append(vocab[char])
        else:
            # Emit PAD token to preserve length alignment with phoneme string
            token_ids.append(PAD_TOKEN)
            unknown_chars.add(char)
    token_ids.append(PAD_TOKEN)  # EOS

    if unknown_chars:
        warnings.warn(
            f"Unknown phoneme characters (mapped to PAD): {sorted(unknown_chars)!r}. "
            "This may affect voice quality. Consider updating vocabulary.",
            stacklevel=2,
        )

    return token_ids


def get_vocab_size() -> int:
    """Get the vocabulary size (178 tokens)."""
    return N_TOKENS


# D8 Cache Management Functions

def get_g2p_cache_stats() -> dict[str, int]:
    """Get G2P cache statistics.

    Returns:
        Dict with keys: hits, misses, size, maxsize, hit_rate
    """
    cache_info = _phonemize_text_cached.cache_info()
    total = cache_info.hits + cache_info.misses
    hit_rate = cache_info.hits / total if total > 0 else 0.0
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "size": cache_info.currsize,
        "maxsize": cache_info.maxsize,
        "hit_rate": hit_rate,
    }


def clear_g2p_cache() -> None:
    """Clear the G2P cache.

    Useful for freeing memory or testing cache behavior.
    """
    global _g2p_cache_hits, _g2p_cache_misses
    _phonemize_text_cached.cache_clear()
    _g2p_cache_hits = 0
    _g2p_cache_misses = 0


def set_g2p_cache_size(maxsize: int) -> None:
    """Set a new maximum size for the G2P cache.

    Note: This clears the existing cache.

    Args:
        maxsize: New maximum cache size (number of entries)
    """
    global _G2P_CACHE_SIZE, _phonemize_text_cached

    _G2P_CACHE_SIZE = maxsize
    # Recreate the cached function with new size
    _phonemize_text_cached = lru_cache(maxsize=maxsize)(_phonemize_text_cached.__wrapped__)
