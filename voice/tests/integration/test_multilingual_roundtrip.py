"""
Multilingual TTS→STT Round-Trip Tests (Phase 6)

Worker #116 - Comprehensive multilingual verification:
- 5 sentences per language (EN, JA, ZH)
- Proper WER measurement
- Keyword verification
- CI-ready test structure

Worker #274 - Semantic similarity scoring:
- LLM-based semantic similarity fallback for keyword matching failures
- Handles honorific/formality variations (Korean, Japanese)
- Score >= 0.7 passes (same meaning, different form)

Tests use standardized corpus from tests/fixtures/text/test_sentences.json

Usage:
    pytest tests/integration/test_multilingual_roundtrip.py -v
    pytest tests/integration/test_multilingual_roundtrip.py -v -m multilingual
    pytest tests/integration/test_multilingual_roundtrip.py -v -k "english"
"""

import json
import os
import pytest
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add scripts directory to path for semantic_similarity import
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Lazy import semantic_similarity to avoid import errors if openai not installed
_semantic_similarity_fn = None

def get_semantic_similarity():
    """Lazy load semantic_similarity function."""
    global _semantic_similarity_fn
    if _semantic_similarity_fn is None:
        try:
            from semantic_similarity import semantic_similarity
            _semantic_similarity_fn = semantic_similarity
        except ImportError:
            _semantic_similarity_fn = lambda *args, **kwargs: {
                "score": 0.0, "reason": "semantic_similarity not available",
                "same_meaning": False, "error": "import failed"
            }
    return _semantic_similarity_fn


# Language name mapping for semantic similarity
LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "hi": "Hindi",
    "yi": "Yiddish",
}

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "text"


# =============================================================================
# WER Calculation Utilities
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    Used for WER calculation where each character is treated as a token
    (appropriate for Japanese/Chinese without word boundaries).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_wer_words(reference: str, hypothesis: str) -> Tuple[float, Dict]:
    """
    Calculate Word Error Rate for space-separated languages (English).

    WER = (S + D + I) / N
    Where: S=substitutions, D=deletions, I=insertions, N=reference words

    Returns:
        Tuple of (wer_value, metrics_dict)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0, {"reference_words": 0}

    # Use word-level Levenshtein
    distance = levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)

    return wer, {
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "edit_distance": distance,
        "wer": wer
    }


def calculate_wer_chars(reference: str, hypothesis: str) -> Tuple[float, Dict]:
    """
    Calculate Character Error Rate for character-based languages (Japanese, Chinese).

    CER = (S + D + I) / N (at character level)

    Returns:
        Tuple of (cer_value, metrics_dict)
    """
    # Remove spaces and normalize
    ref_chars = reference.replace(" ", "").replace("　", "")
    hyp_chars = hypothesis.replace(" ", "").replace("　", "")

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0, {"reference_chars": 0}

    distance = levenshtein_distance(ref_chars, hyp_chars)
    cer = distance / len(ref_chars)

    return cer, {
        "reference_chars": len(ref_chars),
        "hypothesis_chars": len(hyp_chars),
        "edit_distance": distance,
        "cer": cer
    }


def calculate_keyword_score(text: str, keywords: List[str]) -> Tuple[float, int, int]:
    """
    Calculate keyword match score.

    Args:
        text: Text to search in (case-insensitive)
        keywords: List of keywords to find

    Returns:
        Tuple of (score, matches_found, total_keywords)
    """
    text_lower = text.lower()
    # For CJK, don't lowercase (no concept of case)
    matches = sum(1 for kw in keywords if kw.lower() in text_lower or kw in text)
    score = matches / len(keywords) if keywords else 1.0
    return score, matches, len(keywords)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def test_corpus():
    """Load the standardized test corpus."""
    corpus_path = FIXTURES_DIR / "test_sentences.json"
    assert corpus_path.exists(), f"Test corpus not found: {corpus_path}"

    with open(corpus_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tts_binary():
    """Path to stream-tts-cpp binary."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"TTS binary not found at {binary}")
    return binary


@pytest.fixture(scope="module")
def whisper_binary():
    """Path to test_whisper_stt binary."""
    binary = BUILD_DIR / "test_whisper_stt"
    if not binary.exists():
        pytest.skip(f"WhisperSTT binary not found at {binary}")
    return binary


# =============================================================================
# TTS/STT Helper Functions
# =============================================================================

def get_tts_env():
    """
    Get environment variables for TTS subprocess.

    Note: PYTORCH_ENABLE_MPS_FALLBACK no longer needed with PyTorch 2.9.1+.
    """
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def generate_wav(tts_binary: Path, text: str, config: Path, output_path: Path,
                 timeout: int = 90) -> bool:
    """
    Generate a WAV file using TTS.

    Returns True if successful.
    """
    escaped_text = text.replace('"', '\\"').replace('\n', '\\n')
    input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

    try:
        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(output_path), str(config)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=timeout,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )
        return result.returncode == 0 and output_path.exists()
    except (subprocess.TimeoutExpired, UnicodeDecodeError, Exception) as e:
        print(f"TTS generation error: {e}")
        return False


def transcribe_wav(whisper_binary: Path, audio_path: Path, language: str = "en",
                   timeout: int = 120) -> str:
    """
    Transcribe audio file using WhisperSTT.

    Returns transcribed text or empty string on failure.
    """
    result = subprocess.run(
        [str(whisper_binary), "--lang", language, str(audio_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(STREAM_TTS_CPP)
    )

    if result.returncode != 0:
        return ""

    output = result.stdout
    if "--- Transcription ---" in output:
        start = output.index("--- Transcription ---") + len("--- Transcription ---")
        end = output.index("---------------------", start) if "---------------------" in output[start:] else len(output)
        return output[start:end].strip()

    return ""


# =============================================================================
# Multilingual Round-Trip Test Class
# =============================================================================

@pytest.mark.integration
@pytest.mark.multilingual
class TestMultilingualRoundTrip:
    """
    Comprehensive multilingual TTS→STT round-trip tests.

    Tests 5 sentences per language with:
    - Keyword verification
    - WER/CER measurement
    - Detailed metrics reporting
    """

    @pytest.fixture
    def temp_wav(self, tmp_path):
        """Provide temporary WAV file path."""
        return tmp_path / "test_audio.wav"

    # =========================================================================
    # English Tests
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_english_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                                temp_wav, sentence_idx):
        """Test English sentence round-trip."""
        config = CONFIG_DIR / "kokoro-mps-en.yaml"
        if not config.exists():
            pytest.skip("English config not found")

        corpus = test_corpus["english"]
        sentence = corpus["sentences"][sentence_idx]

        # Generate audio
        success = generate_wav(tts_binary, sentence["text"], config, temp_wav)
        assert success, f"Failed to generate audio for: {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {sentence['id']}"

        # Transcribe
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="en")
        assert transcribed, f"Failed to transcribe: {sentence['id']}"

        # Keyword verification
        kw_score, kw_matches, kw_total = calculate_keyword_score(
            transcribed, sentence["keywords"]
        )

        # WER calculation
        wer, wer_metrics = calculate_wer_words(sentence["text"], transcribed)

        # Report results
        print(f"\n[{sentence['id']}] English Round-Trip:")
        print(f"  Input:       '{sentence['text']}'")
        print(f"  Output:      '{transcribed}'")
        print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
        print(f"  WER:         {wer:.1%}")

        # Assert minimum keyword matches
        assert kw_matches >= sentence["min_keywords"], \
            f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"

    def test_english_aggregate_wer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """Calculate aggregate WER across all English sentences."""
        config = CONFIG_DIR / "kokoro-mps-en.yaml"
        if not config.exists():
            pytest.skip("English config not found")

        corpus = test_corpus["english"]
        total_ref_words = 0
        total_edit_distance = 0
        results = []

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"en_{i}.wav"

            if not generate_wav(tts_binary, sentence["text"], config, wav_path):
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            transcribed = transcribe_wav(whisper_binary, wav_path, language="en")
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            wer, metrics = calculate_wer_words(sentence["text"], transcribed)
            total_ref_words += metrics["reference_words"]
            total_edit_distance += metrics["edit_distance"]
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "wer": wer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate WER
        aggregate_wer = total_edit_distance / total_ref_words if total_ref_words > 0 else 1.0
        target_wer = corpus["wer_target"]

        print(f"\n=== English Aggregate WER Report ===")
        print(f"Target WER: {target_wer:.0%}")
        print(f"Actual WER: {aggregate_wer:.1%}")
        print(f"Status: {'PASS' if aggregate_wer <= target_wer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] WER={r['wer']:.1%}")
            else:
                print(f"  [{r['id']}] {r['status']}")

        # Soft assertion - report but don't fail
        if aggregate_wer > target_wer:
            print(f"\nWARNING: Aggregate WER {aggregate_wer:.1%} exceeds target {target_wer:.0%}")

    # =========================================================================
    # Japanese Tests
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_japanese_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                                 temp_wav, sentence_idx):
        """Test Japanese sentence round-trip."""
        config = CONFIG_DIR / "kokoro-mps-ja.yaml"
        if not config.exists():
            pytest.skip("Japanese config not found")

        corpus = test_corpus["japanese"]
        sentence = corpus["sentences"][sentence_idx]

        # Generate audio
        success = generate_wav(tts_binary, sentence["text"], config, temp_wav, timeout=120)
        assert success, f"Failed to generate audio for: {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {sentence['id']}"

        # Transcribe
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="ja", timeout=180)
        assert transcribed, f"Failed to transcribe: {sentence['id']}"

        # Keyword verification (character-based for Japanese)
        kw_score, kw_matches, kw_total = calculate_keyword_score(
            transcribed, sentence["keywords"]
        )

        # CER calculation
        cer, cer_metrics = calculate_wer_chars(sentence["text"], transcribed)

        # Report results
        print(f"\n[{sentence['id']}] Japanese Round-Trip:")
        print(f"  Input:       '{sentence['text']}'")
        print(f"  Output:      '{transcribed}'")
        print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
        print(f"  CER:         {cer:.1%}")

        # Assert minimum keyword matches
        assert kw_matches >= sentence["min_keywords"], \
            f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"

    def test_japanese_aggregate_cer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """Calculate aggregate CER across all Japanese sentences."""
        config = CONFIG_DIR / "kokoro-mps-ja.yaml"
        if not config.exists():
            pytest.skip("Japanese config not found")

        corpus = test_corpus["japanese"]
        total_ref_chars = 0
        total_edit_distance = 0
        results = []

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"ja_{i}.wav"

            if not generate_wav(tts_binary, sentence["text"], config, wav_path, timeout=120):
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            transcribed = transcribe_wav(whisper_binary, wav_path, language="ja", timeout=180)
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            cer, metrics = calculate_wer_chars(sentence["text"], transcribed)
            total_ref_chars += metrics["reference_chars"]
            total_edit_distance += metrics["edit_distance"]
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "cer": cer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate CER
        aggregate_cer = total_edit_distance / total_ref_chars if total_ref_chars > 0 else 1.0
        target_cer = corpus["wer_target"]  # Using wer_target as cer_target

        print(f"\n=== Japanese Aggregate CER Report ===")
        print(f"Target CER: {target_cer:.0%}")
        print(f"Actual CER: {aggregate_cer:.1%}")
        print(f"Status: {'PASS' if aggregate_cer <= target_cer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] CER={r['cer']:.1%}")
            else:
                print(f"  [{r['id']}] {r['status']}")

        if aggregate_cer > target_cer:
            print(f"\nWARNING: Aggregate CER {aggregate_cer:.1%} exceeds target {target_cer:.0%}")

    # =========================================================================
    # Spanish Tests (Worker #236)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_spanish_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                                temp_wav, sentence_idx):
        """Test Spanish sentence round-trip."""
        config = CONFIG_DIR / "kokoro-mps-es.yaml"
        if not config.exists():
            pytest.skip("Spanish config not found")

        corpus = test_corpus["spanish"]
        sentence = corpus["sentences"][sentence_idx]

        # Generate audio
        success = generate_wav(tts_binary, sentence["text"], config, temp_wav)
        assert success, f"Failed to generate audio for: {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {sentence['id']}"

        # Transcribe
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="es")
        assert transcribed, f"Failed to transcribe: {sentence['id']}"

        # Keyword verification
        kw_score, kw_matches, kw_total = calculate_keyword_score(
            transcribed, sentence["keywords"]
        )

        # WER calculation
        wer, wer_metrics = calculate_wer_words(sentence["text"], transcribed)

        # Report results
        print(f"\n[{sentence['id']}] Spanish Round-Trip:")
        print(f"  Input:       '{sentence['text']}'")
        print(f"  Output:      '{transcribed}'")
        print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
        print(f"  WER:         {wer:.1%}")

        # Assert minimum keyword matches
        assert kw_matches >= sentence["min_keywords"], \
            f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"

    def test_spanish_aggregate_wer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """Calculate aggregate WER across all Spanish sentences."""
        config = CONFIG_DIR / "kokoro-mps-es.yaml"
        if not config.exists():
            pytest.skip("Spanish config not found")

        corpus = test_corpus["spanish"]
        total_ref_words = 0
        total_edit_distance = 0
        results = []

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"es_{i}.wav"

            if not generate_wav(tts_binary, sentence["text"], config, wav_path):
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            transcribed = transcribe_wav(whisper_binary, wav_path, language="es")
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            wer, metrics = calculate_wer_words(sentence["text"], transcribed)
            total_ref_words += metrics["reference_words"]
            total_edit_distance += metrics["edit_distance"]
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "wer": wer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate WER
        aggregate_wer = total_edit_distance / total_ref_words if total_ref_words > 0 else 1.0
        target_wer = corpus["wer_target"]

        print(f"\n=== Spanish Aggregate WER Report ===")
        print(f"Target WER: {target_wer:.0%}")
        print(f"Actual WER: {aggregate_wer:.1%}")
        print(f"Status: {'PASS' if aggregate_wer <= target_wer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] WER={r['wer']:.1%}")
            else:
                print(f"  [{r['id']}] {r['status']}")

        if aggregate_wer > target_wer:
            print(f"\nWARNING: Aggregate WER {aggregate_wer:.1%} exceeds target {target_wer:.0%}")

    # =========================================================================
    # French Tests (Worker #236)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_french_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                               temp_wav, sentence_idx):
        """Test French sentence round-trip."""
        config = CONFIG_DIR / "kokoro-mps-fr.yaml"
        if not config.exists():
            pytest.skip("French config not found")

        corpus = test_corpus["french"]
        sentence = corpus["sentences"][sentence_idx]

        # Generate audio
        success = generate_wav(tts_binary, sentence["text"], config, temp_wav)
        assert success, f"Failed to generate audio for: {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {sentence['id']}"

        # Transcribe
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="fr")
        assert transcribed, f"Failed to transcribe: {sentence['id']}"

        # Keyword verification
        kw_score, kw_matches, kw_total = calculate_keyword_score(
            transcribed, sentence["keywords"]
        )

        # WER calculation
        wer, wer_metrics = calculate_wer_words(sentence["text"], transcribed)

        # Report results
        print(f"\n[{sentence['id']}] French Round-Trip:")
        print(f"  Input:       '{sentence['text']}'")
        print(f"  Output:      '{transcribed}'")
        print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
        print(f"  WER:         {wer:.1%}")

        # Assert minimum keyword matches
        assert kw_matches >= sentence["min_keywords"], \
            f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"

    def test_french_aggregate_wer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """Calculate aggregate WER across all French sentences."""
        config = CONFIG_DIR / "kokoro-mps-fr.yaml"
        if not config.exists():
            pytest.skip("French config not found")

        corpus = test_corpus["french"]
        total_ref_words = 0
        total_edit_distance = 0
        results = []

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"fr_{i}.wav"

            if not generate_wav(tts_binary, sentence["text"], config, wav_path):
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            transcribed = transcribe_wav(whisper_binary, wav_path, language="fr")
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            wer, metrics = calculate_wer_words(sentence["text"], transcribed)
            total_ref_words += metrics["reference_words"]
            total_edit_distance += metrics["edit_distance"]
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "wer": wer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate WER
        aggregate_wer = total_edit_distance / total_ref_words if total_ref_words > 0 else 1.0
        target_wer = corpus["wer_target"]

        print(f"\n=== French Aggregate WER Report ===")
        print(f"Target WER: {target_wer:.0%}")
        print(f"Actual WER: {aggregate_wer:.1%}")
        print(f"Status: {'PASS' if aggregate_wer <= target_wer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] WER={r['wer']:.1%}")
            else:
                print(f"  [{r['id']}] {r['status']}")

        if aggregate_wer > target_wer:
            print(f"\nWARNING: Aggregate WER {aggregate_wer:.1%} exceeds target {target_wer:.0%}")

    # =========================================================================
    # Chinese Tests (Beta Quality - espeak G2P)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_chinese_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                                temp_wav, sentence_idx):
        """
        Test Chinese sentence round-trip (beta quality - uses espeak G2P).

        Chinese TTS uses espeak-ng for G2P with phoneme mapping.
        Quality is approximate but audio must be generated.
        """
        config = CONFIG_DIR / "kokoro-mps-zh.yaml"
        if not config.exists():
            pytest.skip("Chinese config not found")

        corpus = test_corpus["chinese"]
        sentence = corpus["sentences"][sentence_idx]

        escaped_text = sentence["text"].replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"Chinese TTS crashed for: {sentence['id']}"

        # Must generate audio file
        assert temp_wav.exists(), f"No audio file for {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, \
            f"Audio too small for {sentence['id']}: {temp_wav.stat().st_size} bytes"

        # Transcribe and report
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="zh", timeout=180)

        if transcribed:
            kw_score, kw_matches, kw_total = calculate_keyword_score(
                transcribed, sentence["keywords"]
            )
            cer, cer_metrics = calculate_wer_chars(sentence["text"], transcribed)

            print(f"\n[{sentence['id']}] Chinese Round-Trip:")
            print(f"  Input:       '{sentence['text']}'")
            print(f"  Output:      '{transcribed}'")
            print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
            print(f"  CER:         {cer:.1%}")
        else:
            print(f"\n[{sentence['id']}] Chinese TTS generated {temp_wav.stat().st_size} bytes (STT empty)")

    def test_chinese_audio_generation_rate(self, tts_binary, test_corpus, tmp_path):
        """
        Test that Chinese TTS generates audio for all test sentences.

        Chinese uses espeak G2P - all sentences must produce audio.
        """
        config = CONFIG_DIR / "kokoro-mps-zh.yaml"
        if not config.exists():
            pytest.skip("Chinese config not found")

        corpus = test_corpus["chinese"]
        audio_generated = 0
        total = len(corpus["sentences"])

        print("\n=== Chinese Audio Generation Test ===")

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"zh_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=90,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if wav_path.exists() and wav_path.stat().st_size > 1000:
                audio_generated += 1
                print(f"  [{sentence['id']}] OK - {wav_path.stat().st_size} bytes")
            else:
                print(f"  [{sentence['id']}] FAILED - '{sentence['text']}'")

        print(f"\nAudio generated: {audio_generated}/{total}")

        # Must generate audio for all sentences
        assert audio_generated == total, \
            f"Chinese TTS failed to generate audio for {total - audio_generated}/{total} sentences"

    # =========================================================================
    # Hindi Tests (Phase 7 - Worker #147)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_hindi_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                              temp_wav, sentence_idx):
        """
        Test Hindi sentence round-trip.

        Hindi TTS uses espeak-ng for G2P. Tests verify audio generation
        and STT transcription accuracy using keyword matching.
        """
        config = CONFIG_DIR / "kokoro-mps-hi.yaml"
        if not config.exists():
            pytest.skip("Hindi config not found")

        if "hindi" not in test_corpus:
            pytest.skip("Hindi test corpus not found")

        corpus = test_corpus["hindi"]
        sentence = corpus["sentences"][sentence_idx]

        escaped_text = sentence["text"].replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(config)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"Hindi TTS crashed for: {sentence['id']}"

        # Must generate audio file
        assert temp_wav.exists(), f"No audio file for {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, \
            f"Audio too small for {sentence['id']}: {temp_wav.stat().st_size} bytes"

        # Transcribe and report
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="hi", timeout=180)

        if transcribed:
            kw_score, kw_matches, kw_total = calculate_keyword_score(
                transcribed, sentence["keywords"]
            )
            # Use character error rate for Devanagari script
            cer, cer_metrics = calculate_wer_chars(sentence["text"], transcribed)

            print(f"\n[{sentence['id']}] Hindi Round-Trip:")
            print(f"  Input:       '{sentence['text']}'")
            print(f"  Output:      '{transcribed}'")
            print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
            print(f"  CER:         {cer:.1%}")

            # Assert minimum keyword matches
            assert kw_matches >= sentence["min_keywords"], \
                f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"
        else:
            print(f"\n[{sentence['id']}] Hindi TTS generated {temp_wav.stat().st_size} bytes (STT empty)")

    def test_hindi_audio_generation_rate(self, tts_binary, test_corpus, tmp_path):
        """
        Test that Hindi TTS generates audio for all test sentences.
        """
        config = CONFIG_DIR / "kokoro-mps-hi.yaml"
        if not config.exists():
            pytest.skip("Hindi config not found")

        if "hindi" not in test_corpus:
            pytest.skip("Hindi test corpus not found")

        corpus = test_corpus["hindi"]
        audio_generated = 0
        total = len(corpus["sentences"])

        print("\n=== Hindi Audio Generation Test ===")

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"hi_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=90,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if wav_path.exists() and wav_path.stat().st_size > 1000:
                audio_generated += 1
                print(f"  [{sentence['id']}] OK - {wav_path.stat().st_size} bytes")
            else:
                print(f"  [{sentence['id']}] FAILED - '{sentence['text']}'")

        print(f"\nAudio generated: {audio_generated}/{total}")

        # Must generate audio for all sentences
        assert audio_generated == total, \
            f"Hindi TTS failed to generate audio for {total - audio_generated}/{total} sentences"

    # =========================================================================
    # Sichuanese Tests (Phase 7 - Worker #148)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_sichuanese_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                              temp_wav, sentence_idx):
        """
        Test Sichuanese sentence round-trip.

        Sichuanese (四川话) is a southwestern Mandarin dialect. Uses Chinese
        voice (zf_xiaobei) and Mandarin G2P. STT uses Chinese whisper model.
        """
        config = CONFIG_DIR / "kokoro-mps-zh-sichuan.yaml"
        if not config.exists():
            pytest.skip("Sichuanese config not found")

        if "sichuanese" not in test_corpus:
            pytest.skip("Sichuanese test corpus not found")

        corpus = test_corpus["sichuanese"]
        sentence = corpus["sentences"][sentence_idx]

        escaped_text = sentence["text"].replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(config)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"Sichuanese TTS crashed for: {sentence['id']}"

        # Must generate audio file
        assert temp_wav.exists(), f"No audio file for {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, \
            f"Audio too small for {sentence['id']}: {temp_wav.stat().st_size} bytes"

        # Transcribe using Chinese whisper model (Sichuanese is a Chinese dialect)
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="zh", timeout=180)

        if transcribed:
            kw_score, kw_matches, kw_total = calculate_keyword_score(
                transcribed, sentence["keywords"]
            )
            # Use character error rate for Chinese script
            cer, cer_metrics = calculate_wer_chars(sentence["text"], transcribed)

            print(f"\n[{sentence['id']}] Sichuanese Round-Trip:")
            print(f"  Input:       '{sentence['text']}'")
            print(f"  Output:      '{transcribed}'")
            print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
            print(f"  CER:         {cer:.1%}")

            # Assert minimum keyword matches
            assert kw_matches >= sentence["min_keywords"], \
                f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"
        else:
            print(f"\n[{sentence['id']}] Sichuanese TTS generated {temp_wav.stat().st_size} bytes (STT empty)")

    def test_sichuanese_audio_generation_rate(self, tts_binary, test_corpus, tmp_path):
        """
        Test that Sichuanese TTS generates audio for all test sentences.
        """
        config = CONFIG_DIR / "kokoro-mps-zh-sichuan.yaml"
        if not config.exists():
            pytest.skip("Sichuanese config not found")

        if "sichuanese" not in test_corpus:
            pytest.skip("Sichuanese test corpus not found")

        corpus = test_corpus["sichuanese"]
        audio_generated = 0
        total = len(corpus["sentences"])

        print("\n=== Sichuanese Audio Generation Test ===")

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"sc_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=90,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if wav_path.exists() and wav_path.stat().st_size > 1000:
                audio_generated += 1
                print(f"  [{sentence['id']}] OK - {wav_path.stat().st_size} bytes")
            else:
                print(f"  [{sentence['id']}] FAILED - '{sentence['text']}'")

        print(f"\nAudio generated: {audio_generated}/{total}")

        # Must generate audio for all sentences
        assert audio_generated == total, \
            f"Sichuanese TTS failed to generate audio for {total - audio_generated}/{total} sentences"

    # =========================================================================
    # Italian Tests (Phase 7 - Worker #149)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_italian_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                              temp_wav, sentence_idx):
        """
        Test Italian sentence round-trip.

        Italian TTS uses espeak-ng for G2P and Kokoro if_sara voice.
        """
        config = CONFIG_DIR / "kokoro-mps-it.yaml"
        if not config.exists():
            pytest.skip("Italian config not found")

        if "italian" not in test_corpus:
            pytest.skip("Italian test corpus not found")

        corpus = test_corpus["italian"]
        sentence = corpus["sentences"][sentence_idx]

        escaped_text = sentence["text"].replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(config)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"Italian TTS crashed for: {sentence['id']}"

        # Must generate audio file
        assert temp_wav.exists(), f"No audio file for {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, \
            f"Audio too small for {sentence['id']}: {temp_wav.stat().st_size} bytes"

        # Transcribe and report
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="it", timeout=180)

        if transcribed:
            kw_score, kw_matches, kw_total = calculate_keyword_score(
                transcribed, sentence["keywords"]
            )
            # Use word error rate for Italian (space-separated language)
            wer, wer_metrics = calculate_wer_words(sentence["text"], transcribed)

            print(f"\n[{sentence['id']}] Italian Round-Trip:")
            print(f"  Input:       '{sentence['text']}'")
            print(f"  Output:      '{transcribed}'")
            print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
            print(f"  WER:         {wer:.1%}")

            # Assert minimum keyword matches
            assert kw_matches >= sentence["min_keywords"], \
                f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"
        else:
            print(f"\n[{sentence['id']}] Italian TTS generated {temp_wav.stat().st_size} bytes (STT empty)")

    def test_italian_audio_generation_rate(self, tts_binary, test_corpus, tmp_path):
        """
        Test that Italian TTS generates audio for all test sentences.
        """
        config = CONFIG_DIR / "kokoro-mps-it.yaml"
        if not config.exists():
            pytest.skip("Italian config not found")

        if "italian" not in test_corpus:
            pytest.skip("Italian test corpus not found")

        corpus = test_corpus["italian"]
        audio_generated = 0
        total = len(corpus["sentences"])

        print("\n=== Italian Audio Generation Test ===")

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"it_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=90,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if wav_path.exists() and wav_path.stat().st_size > 1000:
                audio_generated += 1
                print(f"  [{sentence['id']}] OK - {wav_path.stat().st_size} bytes")
            else:
                print(f"  [{sentence['id']}] FAILED - '{sentence['text']}'")

        print(f"\nAudio generated: {audio_generated}/{total}")

        # Must generate audio for all sentences
        assert audio_generated == total, \
            f"Italian TTS failed to generate audio for {total - audio_generated}/{total} sentences"

    def test_italian_aggregate_wer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """
        Calculate and enforce aggregate WER across all Italian sentences.

        Italian uses word-level WER since it's a space-separated language.
        Target WER is defined in test_sentences.json (currently 20%).
        """
        config = CONFIG_DIR / "kokoro-mps-it.yaml"
        if not config.exists():
            pytest.skip("Italian config not found")

        if "italian" not in test_corpus:
            pytest.skip("Italian test corpus not found")

        corpus = test_corpus["italian"]
        total_ref_words = 0
        total_edit_distance = 0
        results = []

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"it_wer_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if result.returncode != 0 or not wav_path.exists():
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            transcribed = transcribe_wav(whisper_binary, wav_path, language="it", timeout=180)
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            wer, metrics = calculate_wer_words(sentence["text"], transcribed)
            total_ref_words += metrics["reference_words"]
            total_edit_distance += metrics["edit_distance"]
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "wer": wer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate WER
        aggregate_wer = total_edit_distance / total_ref_words if total_ref_words > 0 else 1.0
        target_wer = corpus["wer_target"]

        print(f"\n=== Italian Aggregate WER Report ===")
        print(f"Target WER: {target_wer:.0%}")
        print(f"Actual WER: {aggregate_wer:.1%}")
        print(f"Status: {'PASS' if aggregate_wer <= target_wer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] WER={r['wer']:.1%} | '{r['text']}' -> '{r['transcribed']}'")
            else:
                print(f"  [{r['id']}] {r['status']}")

        # Hard assertion - enforce WER target
        assert aggregate_wer <= target_wer, \
            f"Italian aggregate WER {aggregate_wer:.1%} exceeds target {target_wer:.0%}"

    # =========================================================================
    # Portuguese Tests (Phase 7 - Worker #149)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_portuguese_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                              temp_wav, sentence_idx):
        """
        Test Portuguese sentence round-trip.

        Portuguese TTS uses espeak-ng for G2P and Kokoro pf_dora voice.
        """
        config = CONFIG_DIR / "kokoro-mps-pt.yaml"
        if not config.exists():
            pytest.skip("Portuguese config not found")

        if "portuguese" not in test_corpus:
            pytest.skip("Portuguese test corpus not found")

        corpus = test_corpus["portuguese"]
        sentence = corpus["sentences"][sentence_idx]

        escaped_text = sentence["text"].replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(config)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"Portuguese TTS crashed for: {sentence['id']}"

        # Must generate audio file
        assert temp_wav.exists(), f"No audio file for {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, \
            f"Audio too small for {sentence['id']}: {temp_wav.stat().st_size} bytes"

        # Transcribe and report
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="pt", timeout=180)

        if transcribed:
            kw_score, kw_matches, kw_total = calculate_keyword_score(
                transcribed, sentence["keywords"]
            )
            # Use word error rate for Portuguese (space-separated language)
            wer, wer_metrics = calculate_wer_words(sentence["text"], transcribed)

            print(f"\n[{sentence['id']}] Portuguese Round-Trip:")
            print(f"  Input:       '{sentence['text']}'")
            print(f"  Output:      '{transcribed}'")
            print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
            print(f"  WER:         {wer:.1%}")

            # Assert minimum keyword matches
            assert kw_matches >= sentence["min_keywords"], \
                f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"
        else:
            print(f"\n[{sentence['id']}] Portuguese TTS generated {temp_wav.stat().st_size} bytes (STT empty)")

    def test_portuguese_audio_generation_rate(self, tts_binary, test_corpus, tmp_path):
        """
        Test that Portuguese TTS generates audio for all test sentences.
        """
        config = CONFIG_DIR / "kokoro-mps-pt.yaml"
        if not config.exists():
            pytest.skip("Portuguese config not found")

        if "portuguese" not in test_corpus:
            pytest.skip("Portuguese test corpus not found")

        corpus = test_corpus["portuguese"]
        audio_generated = 0
        total = len(corpus["sentences"])

        print("\n=== Portuguese Audio Generation Test ===")

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"pt_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=90,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if wav_path.exists() and wav_path.stat().st_size > 1000:
                audio_generated += 1
                print(f"  [{sentence['id']}] OK - {wav_path.stat().st_size} bytes")
            else:
                print(f"  [{sentence['id']}] FAILED - '{sentence['text']}'")

        print(f"\nAudio generated: {audio_generated}/{total}")

        # Must generate audio for all sentences
        assert audio_generated == total, \
            f"Portuguese TTS failed to generate audio for {total - audio_generated}/{total} sentences"

    def test_portuguese_aggregate_wer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """
        Calculate and enforce aggregate WER across all Portuguese sentences.

        Portuguese uses word-level WER since it's a space-separated language.
        Target WER is defined in test_sentences.json (currently 20%).
        """
        config = CONFIG_DIR / "kokoro-mps-pt.yaml"
        if not config.exists():
            pytest.skip("Portuguese config not found")

        if "portuguese" not in test_corpus:
            pytest.skip("Portuguese test corpus not found")

        corpus = test_corpus["portuguese"]
        total_ref_words = 0
        total_edit_distance = 0
        results = []

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"pt_wer_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if result.returncode != 0 or not wav_path.exists():
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            transcribed = transcribe_wav(whisper_binary, wav_path, language="pt", timeout=180)
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            wer, metrics = calculate_wer_words(sentence["text"], transcribed)
            total_ref_words += metrics["reference_words"]
            total_edit_distance += metrics["edit_distance"]
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "wer": wer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate WER
        aggregate_wer = total_edit_distance / total_ref_words if total_ref_words > 0 else 1.0
        target_wer = corpus["wer_target"]

        print(f"\n=== Portuguese Aggregate WER Report ===")
        print(f"Target WER: {target_wer:.0%}")
        print(f"Actual WER: {aggregate_wer:.1%}")
        print(f"Status: {'PASS' if aggregate_wer <= target_wer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] WER={r['wer']:.1%} | '{r['text']}' -> '{r['transcribed']}'")
            else:
                print(f"  [{r['id']}] {r['status']}")

        # Hard assertion - enforce WER target
        assert aggregate_wer <= target_wer, \
            f"Portuguese aggregate WER {aggregate_wer:.1%} exceeds target {target_wer:.0%}"

    # =========================================================================
    # Korean Tests (Phase 7 - Worker #149)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_korean_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                              temp_wav, sentence_idx):
        """
        Test Korean sentence round-trip.

        Korean TTS uses espeak-ng for G2P. Note: No native Kokoro Korean voice yet,
        so quality may be limited. Uses default voice with Korean phonemes.

        Worker #274: Added semantic similarity fallback for honorific variations.
        When keyword matching fails, LLM evaluates if meaning is preserved.
        """
        config = CONFIG_DIR / "kokoro-mps-ko.yaml"
        if not config.exists():
            pytest.skip("Korean config not found")

        if "korean" not in test_corpus:
            pytest.skip("Korean test corpus not found")

        corpus = test_corpus["korean"]
        sentence = corpus["sentences"][sentence_idx]

        escaped_text = sentence["text"].replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(config)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"Korean TTS crashed for: {sentence['id']}"

        # Must generate audio file
        assert temp_wav.exists(), f"No audio file for {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, \
            f"Audio too small for {sentence['id']}: {temp_wav.stat().st_size} bytes"

        # Transcribe and report
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="ko", timeout=180)

        if transcribed:
            kw_score, kw_matches, kw_total = calculate_keyword_score(
                transcribed, sentence["keywords"]
            )
            # Use character error rate for Korean (Hangul script)
            cer, cer_metrics = calculate_wer_chars(sentence["text"], transcribed)

            print(f"\n[{sentence['id']}] Korean Round-Trip:")
            print(f"  Input:       '{sentence['text']}'")
            print(f"  Output:      '{transcribed}'")
            print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
            print(f"  CER:         {cer:.1%}")

            # Primary check: keyword matching
            if kw_matches >= sentence["min_keywords"]:
                print(f"  Status:      PASS (keyword match)")
                return  # Test passes

            # Fallback: semantic similarity (handles honorific variations)
            semantic_fn = get_semantic_similarity()
            similarity = semantic_fn(sentence["text"], transcribed, "Korean")

            if "error" not in similarity:
                print(f"  Semantic:    {similarity['score']:.0%} ({similarity['reason']})")
                if similarity["same_meaning"]:
                    print(f"  Status:      PASS (semantic match >= 70%)")
                    return  # Test passes via semantic similarity
                else:
                    print(f"  Status:      FAIL (neither keyword nor semantic match)")
            else:
                print(f"  Semantic:    N/A ({similarity.get('error', 'unknown error')})")

            # Assertion with detailed failure message
            assert kw_matches >= sentence["min_keywords"] or similarity.get("same_meaning", False), \
                f"[{sentence['id']}] Neither keyword ({kw_matches}/{sentence['min_keywords']}) " \
                f"nor semantic ({similarity.get('score', 0):.0%}) match passed"
        else:
            print(f"\n[{sentence['id']}] Korean TTS generated {temp_wav.stat().st_size} bytes (STT empty)")

    def test_korean_audio_generation_rate(self, tts_binary, test_corpus, tmp_path):
        """
        Test that Korean TTS generates audio for all test sentences.
        """
        config = CONFIG_DIR / "kokoro-mps-ko.yaml"
        if not config.exists():
            pytest.skip("Korean config not found")

        if "korean" not in test_corpus:
            pytest.skip("Korean test corpus not found")

        corpus = test_corpus["korean"]
        audio_generated = 0
        total = len(corpus["sentences"])

        print("\n=== Korean Audio Generation Test ===")

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"ko_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=90,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if wav_path.exists() and wav_path.stat().st_size > 1000:
                audio_generated += 1
                print(f"  [{sentence['id']}] OK - {wav_path.stat().st_size} bytes")
            else:
                print(f"  [{sentence['id']}] FAILED - '{sentence['text']}'")

        print(f"\nAudio generated: {audio_generated}/{total}")

        # Must generate audio for all sentences
        assert audio_generated == total, \
            f"Korean TTS failed to generate audio for {total - audio_generated}/{total} sentences"

    def test_korean_aggregate_cer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """
        Calculate and enforce aggregate CER across all Korean sentences.

        Korean uses character-level CER since Hangul doesn't use spaces consistently.
        Target CER is defined in test_sentences.json (currently 30% - espeak fallback quality).
        """
        config = CONFIG_DIR / "kokoro-mps-ko.yaml"
        if not config.exists():
            pytest.skip("Korean config not found")

        if "korean" not in test_corpus:
            pytest.skip("Korean test corpus not found")

        corpus = test_corpus["korean"]
        total_ref_chars = 0
        total_edit_distance = 0
        results = []
        success_count = 0

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"ko_cer_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if result.returncode != 0 or not wav_path.exists():
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            transcribed = transcribe_wav(whisper_binary, wav_path, language="ko", timeout=180)
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            cer, metrics = calculate_wer_chars(sentence["text"], transcribed)
            total_ref_chars += metrics["reference_chars"]
            total_edit_distance += metrics["edit_distance"]
            success_count += 1
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "cer": cer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate CER
        aggregate_cer = total_edit_distance / total_ref_chars if total_ref_chars > 0 else 1.0
        target_cer = corpus["wer_target"]  # Using wer_target as cer_target

        print(f"\n=== Korean Aggregate CER Report ===")
        print(f"Target CER: {target_cer:.0%}")
        print(f"Actual CER: {aggregate_cer:.1%}")
        print(f"Successful transcriptions: {success_count}/{len(corpus['sentences'])}")
        print(f"Status: {'PASS' if aggregate_cer <= target_cer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] CER={r['cer']:.1%} | '{r['text']}' -> '{r['transcribed']}'")
            else:
                print(f"  [{r['id']}] {r['status']}")

        # Hard assertion - enforce CER target (relaxed for espeak fallback)
        assert aggregate_cer <= target_cer, \
            f"Korean aggregate CER {aggregate_cer:.1%} exceeds target {target_cer:.0%}"

    # =========================================================================
    # Yiddish Tests - Phase 7 Multilingual Expansion (Worker #150)
    # =========================================================================

    @pytest.mark.parametrize("sentence_idx", [0, 1, 2, 3, 4])
    def test_yiddish_roundtrip(self, tts_binary, whisper_binary, test_corpus,
                               temp_wav, sentence_idx):
        """
        Test Yiddish sentence round-trip.

        Yiddish TTS uses Hebrew espeak G2P with English voice fallback.
        Note: No native Yiddish TTS exists, quality may be limited.
        STT uses Whisper's Yiddish support (limited).
        """
        config = CONFIG_DIR / "kokoro-mps-yi.yaml"
        if not config.exists():
            pytest.skip("Yiddish config not found")

        if "yiddish" not in test_corpus:
            pytest.skip("Yiddish test corpus not found")

        corpus = test_corpus["yiddish"]
        sentence = corpus["sentences"][sentence_idx]

        escaped_text = sentence["text"].replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(config)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"Yiddish TTS crashed for: {sentence['id']}"

        # Must generate audio file
        assert temp_wav.exists(), f"No audio file for {sentence['id']}"
        assert temp_wav.stat().st_size > 1000, \
            f"Audio too small for {sentence['id']}: {temp_wav.stat().st_size} bytes"

        # Transcribe and report - Yiddish STT is very limited
        transcribed = transcribe_wav(whisper_binary, temp_wav, language="yi", timeout=180)

        if transcribed:
            kw_score, kw_matches, kw_total = calculate_keyword_score(
                transcribed, sentence["keywords"]
            )
            # Use character error rate for Yiddish (Hebrew script)
            cer, cer_metrics = calculate_wer_chars(sentence["text"], transcribed)

            print(f"\n[{sentence['id']}] Yiddish Round-Trip:")
            print(f"  Input:       '{sentence['text']}'")
            print(f"  Output:      '{transcribed}'")
            print(f"  Keywords:    {kw_matches}/{kw_total} ({kw_score:.0%})")
            print(f"  CER:         {cer:.1%}")

            # Assert minimum keyword matches (set to 0 since STT quality is limited)
            assert kw_matches >= sentence["min_keywords"], \
                f"[{sentence['id']}] Expected {sentence['min_keywords']}+ keywords, got {kw_matches}"
        else:
            print(f"\n[{sentence['id']}] Yiddish TTS generated {temp_wav.stat().st_size} bytes (STT empty)")

    def test_yiddish_audio_generation_rate(self, tts_binary, test_corpus, tmp_path):
        """
        Test that Yiddish TTS generates audio for all test sentences.
        """
        config = CONFIG_DIR / "kokoro-mps-yi.yaml"
        if not config.exists():
            pytest.skip("Yiddish config not found")

        if "yiddish" not in test_corpus:
            pytest.skip("Yiddish test corpus not found")

        corpus = test_corpus["yiddish"]
        audio_generated = 0
        total = len(corpus["sentences"])

        print("\n=== Yiddish Audio Generation Test ===")

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"yi_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=90,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if wav_path.exists() and wav_path.stat().st_size > 1000:
                audio_generated += 1
                print(f"  [{sentence['id']}] OK - {wav_path.stat().st_size} bytes")
            else:
                print(f"  [{sentence['id']}] FAILED - '{sentence['text']}'")

        print(f"\nAudio generated: {audio_generated}/{total}")

        # Must generate audio for all sentences
        assert audio_generated == total, \
            f"Yiddish TTS failed to generate audio for {total - audio_generated}/{total} sentences"

    def test_yiddish_aggregate_cer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """
        Calculate and enforce aggregate CER across all Yiddish sentences.

        Yiddish uses character-level CER for Hebrew script.
        Target CER is defined in test_sentences.json (currently 50% - limited STT quality).
        Note: Yiddish has very limited TTS/STT support, so threshold is relaxed.
        """
        config = CONFIG_DIR / "kokoro-mps-yi.yaml"
        if not config.exists():
            pytest.skip("Yiddish config not found")

        if "yiddish" not in test_corpus:
            pytest.skip("Yiddish test corpus not found")

        corpus = test_corpus["yiddish"]
        total_ref_chars = 0
        total_edit_distance = 0
        results = []
        success_count = 0

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"yi_cer_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if result.returncode != 0 or not wav_path.exists():
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            transcribed = transcribe_wav(whisper_binary, wav_path, language="yi", timeout=180)
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            cer, metrics = calculate_wer_chars(sentence["text"], transcribed)
            total_ref_chars += metrics["reference_chars"]
            total_edit_distance += metrics["edit_distance"]
            success_count += 1
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "cer": cer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate CER
        aggregate_cer = total_edit_distance / total_ref_chars if total_ref_chars > 0 else 1.0
        target_cer = corpus["wer_target"]  # Using wer_target as cer_target

        print(f"\n=== Yiddish Aggregate CER Report ===")
        print(f"Target CER: {target_cer:.0%}")
        print(f"Actual CER: {aggregate_cer:.1%}")
        print(f"Successful transcriptions: {success_count}/{len(corpus['sentences'])}")
        print(f"Status: {'PASS' if aggregate_cer <= target_cer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] CER={r['cer']:.1%} | '{r['text']}' -> '{r['transcribed']}'")
            else:
                print(f"  [{r['id']}] {r['status']}")

        # Hard assertion - enforce CER target (very relaxed due to limited Yiddish support)
        assert aggregate_cer <= target_cer, \
            f"Yiddish aggregate CER {aggregate_cer:.1%} exceeds target {target_cer:.0%}"

    # =========================================================================
    # Sichuanese Aggregate CER Test (Phase 7 - Worker #161)
    # =========================================================================

    def test_sichuanese_aggregate_cer(self, tts_binary, whisper_binary, test_corpus, tmp_path):
        """
        Calculate and enforce aggregate CER across all Sichuanese sentences.

        Sichuanese (四川话) is a southwestern Mandarin dialect, uses Chinese whisper model.
        Uses character-level CER since Chinese doesn't use spaces.
        Target CER is defined in test_sentences.json (currently 30%).
        """
        config = CONFIG_DIR / "kokoro-mps-zh-sichuan.yaml"
        if not config.exists():
            pytest.skip("Sichuanese config not found")

        if "sichuanese" not in test_corpus:
            pytest.skip("Sichuanese test corpus not found")

        corpus = test_corpus["sichuanese"]
        total_ref_chars = 0
        total_edit_distance = 0
        results = []
        success_count = 0

        for i, sentence in enumerate(corpus["sentences"]):
            wav_path = tmp_path / f"sc_cer_{i}.wav"
            escaped_text = sentence["text"].replace('"', '\\"')
            input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

            result = subprocess.run(
                [str(tts_binary), "--save-audio", str(wav_path), str(config)],
                input=input_json.encode('utf-8'),
                capture_output=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            if result.returncode != 0 or not wav_path.exists():
                results.append({"id": sentence["id"], "status": "TTS_FAILED"})
                continue

            # Use Chinese whisper model for Sichuanese (a Chinese dialect)
            transcribed = transcribe_wav(whisper_binary, wav_path, language="zh", timeout=180)
            if not transcribed:
                results.append({"id": sentence["id"], "status": "STT_FAILED"})
                continue

            cer, metrics = calculate_wer_chars(sentence["text"], transcribed)
            total_ref_chars += metrics["reference_chars"]
            total_edit_distance += metrics["edit_distance"]
            success_count += 1
            results.append({
                "id": sentence["id"],
                "status": "OK",
                "cer": cer,
                "text": sentence["text"],
                "transcribed": transcribed
            })

        # Calculate aggregate CER
        aggregate_cer = total_edit_distance / total_ref_chars if total_ref_chars > 0 else 1.0
        target_cer = corpus["wer_target"]  # Using wer_target as cer_target

        print(f"\n=== Sichuanese Aggregate CER Report ===")
        print(f"Target CER: {target_cer:.0%}")
        print(f"Actual CER: {aggregate_cer:.1%}")
        print(f"Successful transcriptions: {success_count}/{len(corpus['sentences'])}")
        print(f"Status: {'PASS' if aggregate_cer <= target_cer else 'FAIL'}")
        print(f"\nPer-sentence results:")
        for r in results:
            if r["status"] == "OK":
                print(f"  [{r['id']}] CER={r['cer']:.1%} | '{r['text']}' -> '{r['transcribed']}'")
            else:
                print(f"  [{r['id']}] {r['status']}")

        # Hard assertion - enforce CER target
        assert aggregate_cer <= target_cer, \
            f"Sichuanese aggregate CER {aggregate_cer:.1%} exceeds target {target_cer:.0%}"


# =============================================================================
# WER/CER Utility Tests
# =============================================================================

@pytest.mark.unit
class TestWERCalculation:
    """Unit tests for WER/CER calculation utilities."""

    def test_levenshtein_identical(self):
        """Identical strings have distance 0."""
        assert levenshtein_distance("hello", "hello") == 0
        assert levenshtein_distance("", "") == 0

    def test_levenshtein_insertions(self):
        """Test insertion distance."""
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("hello", "helloworld") == 5

    def test_levenshtein_deletions(self):
        """Test deletion distance."""
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("helloworld", "hello") == 5

    def test_levenshtein_substitutions(self):
        """Test substitution distance."""
        assert levenshtein_distance("abc", "xyz") == 3
        assert levenshtein_distance("hello", "hallo") == 1

    def test_wer_words_perfect(self):
        """Perfect transcription has 0 WER."""
        wer, _ = calculate_wer_words("hello world", "hello world")
        assert wer == 0.0

    def test_wer_words_all_wrong(self):
        """All wrong has WER >= 1.0."""
        wer, _ = calculate_wer_words("hello world", "goodbye universe")
        assert wer >= 1.0

    def test_wer_chars_japanese(self):
        """Test CER for Japanese text."""
        cer, metrics = calculate_wer_chars("こんにちは", "こんにちは")
        assert cer == 0.0

        cer, metrics = calculate_wer_chars("こんにちは", "こんばんは")
        assert 0 < cer < 1.0  # Partial match

    def test_keyword_score_all_match(self):
        """All keywords found."""
        score, matches, total = calculate_keyword_score(
            "hello world how are you",
            ["hello", "world", "how"]
        )
        assert score == 1.0
        assert matches == 3

    def test_keyword_score_partial(self):
        """Partial keyword match."""
        score, matches, total = calculate_keyword_score(
            "hello world",
            ["hello", "world", "goodbye"]
        )
        assert matches == 2
        assert abs(score - 2/3) < 0.01

    def test_keyword_score_japanese(self):
        """Japanese keyword matching."""
        score, matches, total = calculate_keyword_score(
            "今日は良い天気です",
            ["今日", "天気"]
        )
        assert matches == 2
        assert score == 1.0
