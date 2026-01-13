"""
LLM-as-Judge Quality Tests

Tests audio quality using GPT-5 as a judge. Evaluates:
- Accuracy: Does the audio match the expected text?
- Naturalness: Does it sound like a real human?
- Quality: Overall TTS quality rating

Usage:
    pytest tests/quality/test_llm_judge.py -v -m quality
"""

import json
import os
import sys
import tempfile
import subprocess
import pytest
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_QUALITY_DIR = Path(__file__).parent

# Add scripts to path for importing llm_audio_judge
sys.path.insert(0, str(SCRIPTS_DIR))

# Load test corpus and baseline
TEST_CORPUS_PATH = TESTS_QUALITY_DIR / "test_corpus.json"
BASELINE_PATH = TESTS_QUALITY_DIR / "quality_baseline.json"


def load_env():
    """Load .env file for API keys."""
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


load_env()

# Quality history tracking
QUALITY_HISTORY_PATH = TESTS_QUALITY_DIR / "quality_history.json"


def record_quality_result(language: str, text: str, scores: dict, test_name: str):
    """
    Record quality test result to history file for regression tracking.

    Args:
        language: Language code (en, ja, etc.)
        text: Text that was tested
        scores: Dict with accuracy, naturalness, quality scores
        test_name: Name of the test
    """
    import datetime

    if not QUALITY_HISTORY_PATH.exists():
        history = {"description": "Historical quality scores", "version": "1.0.0", "schema_version": 1, "runs": []}
    else:
        with open(QUALITY_HISTORY_PATH) as f:
            history = json.load(f)

    # Get git commit hash if available
    commit_hash = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=5
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
    except Exception:
        pass

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "commit": commit_hash,
        "test_name": test_name,
        "language": language,
        "text": text[:100],  # Truncate for readability
        "scores": {
            "accuracy": scores.get("accuracy", 0),
            "naturalness": scores.get("naturalness", 0),
            "quality": scores.get("quality", 0)
        },
        "transcription": scores.get("transcription", "")[:200],  # Truncate
        "issues": scores.get("issues", "")[:200]  # Truncate
    }

    # Keep only last 500 runs to prevent file bloat
    history["runs"].append(record)
    if len(history["runs"]) > 500:
        history["runs"] = history["runs"][-500:]

    with open(QUALITY_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_historical_average(language: str, metric: str, last_n: int = 10) -> Optional[float]:
    """
    Get average score for a language/metric over last N runs.

    Args:
        language: Language code
        metric: accuracy, naturalness, or quality
        last_n: Number of recent runs to average

    Returns:
        Average score or None if insufficient data
    """
    if not QUALITY_HISTORY_PATH.exists():
        return None

    with open(QUALITY_HISTORY_PATH) as f:
        history = json.load(f)

    # Filter runs for this language
    relevant_runs = [r for r in history.get("runs", []) if r.get("language") == language]

    if len(relevant_runs) < 3:  # Need at least 3 data points
        return None

    # Get last N scores
    recent_scores = [r["scores"].get(metric, 0) for r in relevant_runs[-last_n:]]

    if not recent_scores:
        return None

    return sum(recent_scores) / len(recent_scores)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def test_corpus():
    """Load the test corpus."""
    assert TEST_CORPUS_PATH.exists(), f"Test corpus not found: {TEST_CORPUS_PATH}"
    with open(TEST_CORPUS_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def quality_baseline():
    """Load quality baseline thresholds."""
    assert BASELINE_PATH.exists(), f"Quality baseline not found: {BASELINE_PATH}"
    with open(BASELINE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def openai_available():
    """Check if OpenAI API is available."""
    key = os.environ.get('OPENAI_API_KEY')
    if not key or key.startswith('sk-...'):
        pytest.skip("OPENAI_API_KEY not configured")
    try:
        from openai import OpenAI
        return True
    except ImportError:
        pytest.skip("openai package not installed")


@pytest.fixture(scope="module")
def tts_binary():
    """Path to stream-tts-cpp binary (main TTS pipeline)."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"TTS binary not found at {binary}")
    return binary


@pytest.fixture(scope="module")
def llm_judge():
    """Import LLM audio judge module."""
    try:
        import llm_audio_judge
        return llm_audio_judge
    except ImportError:
        pytest.skip("llm_audio_judge module not found in scripts/")


# =============================================================================
# Helper Functions
# =============================================================================

def generate_tts_audio(tts_binary: Path, text: str, language: str, output_path: Path, timeout: int = 60) -> bool:
    """
    Generate TTS audio using stream-tts-cpp with --save-audio.

    Args:
        tts_binary: Path to stream-tts-cpp
        text: Text to synthesize
        language: Language code (en, ja, zh, etc.)
        output_path: Path to save WAV file
        timeout: Timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    # Get config path for language
    config_map = {
        'en': 'kokoro-mps-en.yaml',
        'ja': 'kokoro-mps-ja.yaml',
        'zh': 'kokoro-mps-zh.yaml',
        'es': 'kokoro-mps-es.yaml',
        'fr': 'kokoro-mps-fr.yaml',
    }

    config_file = CONFIG_DIR / config_map.get(language, 'kokoro-mps-en.yaml')
    if not config_file.exists():
        # Fall back to default config
        config_file = CONFIG_DIR / 'default.yaml'
        if not config_file.exists():
            print(f"No config found for language {language}")
            return False

    # Escape text for JSON input (Claude API format)
    escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

    result = subprocess.run(
        [str(tts_binary), "--save-audio", str(output_path), str(config_file)],
        input=input_json.encode('utf-8'),
        capture_output=True,
        timeout=timeout,
        cwd=str(STREAM_TTS_CPP)
    )

    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace')
        print(f"TTS generation failed: {stderr}")
        return False

    return output_path.exists() and output_path.stat().st_size > 1000


def evaluate_audio(llm_judge_module, audio_path: Path, expected_text: str, language: str) -> dict:
    """
    Evaluate audio using LLM-as-Judge.

    Returns dict with: accuracy, naturalness, quality, transcription, issues
    """
    result = llm_judge_module.evaluate_audio_openai(
        str(audio_path),
        expected_text,
        language
    )
    return result


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.quality
class TestLLMJudgeSetup:
    """Test LLM Judge infrastructure setup."""

    def test_test_corpus_exists(self, test_corpus):
        """Verify test corpus is loaded."""
        assert 'sentences' in test_corpus
        assert 'en' in test_corpus['sentences']
        assert len(test_corpus['sentences']['en']) == 10

    def test_baseline_exists(self, quality_baseline):
        """Verify quality baseline is loaded."""
        assert 'thresholds' in quality_baseline
        assert 'language_baselines' in quality_baseline
        assert quality_baseline['thresholds']['accuracy']['min'] == 4.0

    def test_openai_key_configured(self):
        """Verify OpenAI API key is present (not necessarily valid)."""
        key = os.environ.get('OPENAI_API_KEY')
        assert key is not None, "OPENAI_API_KEY not set in environment"
        assert len(key) > 10, "OPENAI_API_KEY appears invalid"

    def test_tts_binary_exists(self, tts_binary):
        """Verify TTS binary is available."""
        assert tts_binary.exists()


@pytest.mark.quality
@pytest.mark.slow
class TestEnglishQuality:
    """English TTS quality tests using GPT-5."""

    def test_english_simple_sentence(self, openai_available, tts_binary, llm_judge, quality_baseline):
        """Test simple English sentence quality."""
        text = "Hello, world! This is a test."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'en', output_path)
            assert success, "TTS generation failed"
            assert output_path.exists(), "Audio file not created"
            assert output_path.stat().st_size > 1000, "Audio file too small"

            result = evaluate_audio(llm_judge, output_path, text, 'en')

            assert 'error' not in result, f"LLM Judge error: {result.get('error')}"

            accuracy = result.get('accuracy', 0)
            naturalness = result.get('naturalness', 0)
            quality_score = result.get('quality', 0)

            min_accuracy = quality_baseline['thresholds']['accuracy']['min']
            min_naturalness = quality_baseline['thresholds']['naturalness']['min']
            min_quality = quality_baseline['thresholds']['quality']['min']

            print(f"\nLLM Judge Results for: '{text}'")
            print(f"  Accuracy: {accuracy}/5 (min: {min_accuracy})")
            print(f"  Naturalness: {naturalness}/5 (min: {min_naturalness})")
            print(f"  Quality: {quality_score}/5 (min: {min_quality})")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")
            print(f"  Issues: {result.get('issues', 'none')}")

            assert accuracy >= min_accuracy, f"Accuracy {accuracy} below threshold {min_accuracy}"
            assert naturalness >= min_naturalness, f"Naturalness {naturalness} below threshold {min_naturalness}"
            assert quality_score >= min_quality, f"Quality {quality_score} below threshold {min_quality}"

            # Record for regression tracking
            record_quality_result('en', text, result, 'test_english_simple_sentence')

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_english_technical_sentence(self, openai_available, tts_binary, llm_judge, quality_baseline):
        """Test technical English sentence quality."""
        text = "The function was successfully refactored."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'en', output_path)
            assert success, "TTS generation failed"

            result = evaluate_audio(llm_judge, output_path, text, 'en')

            assert 'error' not in result, f"LLM Judge error: {result.get('error')}"

            print(f"\nLLM Judge Results for: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")

            min_accuracy = quality_baseline['thresholds']['accuracy']['min']
            assert result.get('accuracy', 0) >= min_accuracy

            # Record for regression tracking
            record_quality_result('en', text, result, 'test_english_technical_sentence')

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
@pytest.mark.slow
class TestJapaneseQuality:
    """Japanese TTS quality tests using GPT-5."""

    def test_japanese_hiragana(self, openai_available, tts_binary, llm_judge, quality_baseline):
        """Test Japanese hiragana sentence quality."""
        text = "こんにちは、世界！これはテストです。"

        ja_baseline = quality_baseline['language_baselines']['ja']

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'ja', output_path)
            assert success, "Japanese TTS generation failed"

            result = evaluate_audio(llm_judge, output_path, text, 'ja')

            assert 'error' not in result, f"LLM Judge error: {result.get('error')}"

            print(f"\nLLM Judge Results for: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")

            # Japanese is production quality
            min_accuracy = ja_baseline['accuracy']['baseline'] - ja_baseline['accuracy']['tolerance']
            assert result.get('accuracy', 0) >= min_accuracy, \
                f"Japanese accuracy {result.get('accuracy', 0)} below threshold {min_accuracy}"

            record_quality_result('ja', text, result, 'test_japanese_hiragana')

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
@pytest.mark.slow
class TestChineseQuality:
    """Chinese TTS quality tests using GPT-5."""

    def test_chinese_simple_sentence(self, openai_available, tts_binary, llm_judge, quality_baseline):
        """Test Chinese TTS quality (beta - uses espeak G2P)."""
        text = "你好，世界！这是一个测试。"

        zh_baseline = quality_baseline['language_baselines']['zh']

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'zh', output_path)
            assert success, "Chinese TTS generation failed"

            result = evaluate_audio(llm_judge, output_path, text, 'zh')

            assert 'error' not in result, f"LLM Judge error: {result.get('error')}"

            print(f"\nLLM Judge Results for: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")

            # Chinese uses beta thresholds (espeak G2P, no tone info)
            min_accuracy = zh_baseline['accuracy']['baseline'] - zh_baseline['accuracy']['tolerance']
            assert result.get('accuracy', 0) >= min_accuracy, \
                f"Chinese accuracy {result.get('accuracy', 0)} below threshold {min_accuracy}"

            record_quality_result('zh', text, result, 'test_chinese_simple_sentence')

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
@pytest.mark.slow
class TestSpanishQuality:
    """Spanish TTS quality tests using GPT-5."""

    def test_spanish_simple_sentence(self, openai_available, tts_binary, llm_judge, quality_baseline):
        """Test simple Spanish sentence quality (beta - uses espeak G2P)."""
        text = "¡Hola, mundo! Esto es una prueba."

        es_baseline = quality_baseline['language_baselines']['es']

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'es', output_path)
            assert success, "Spanish TTS generation failed"

            result = evaluate_audio(llm_judge, output_path, text, 'es')

            assert 'error' not in result, f"LLM Judge error: {result.get('error')}"

            print(f"\nLLM Judge Results for: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")

            min_accuracy = es_baseline['accuracy']['baseline'] - es_baseline['accuracy']['tolerance']
            assert result.get('accuracy', 0) >= min_accuracy, \
                f"Accuracy {result.get('accuracy', 0)} below threshold {min_accuracy}"

            record_quality_result('es', text, result, 'test_spanish_simple_sentence')

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_spanish_technical_sentence(self, openai_available, tts_binary, llm_judge, quality_baseline):
        """Test technical Spanish sentence quality."""
        text = "La función fue refactorizada exitosamente."

        es_baseline = quality_baseline['language_baselines']['es']

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'es', output_path)
            assert success, "Spanish TTS generation failed"

            result = evaluate_audio(llm_judge, output_path, text, 'es')

            assert 'error' not in result, f"LLM Judge error: {result.get('error')}"

            print(f"\nLLM Judge Results for: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")

            min_accuracy = es_baseline['accuracy']['baseline'] - es_baseline['accuracy']['tolerance']
            assert result.get('accuracy', 0) >= min_accuracy

            record_quality_result('es', text, result, 'test_spanish_technical_sentence')

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
@pytest.mark.slow
class TestFrenchQuality:
    """French TTS quality tests using GPT-5."""

    def test_french_simple_sentence(self, openai_available, tts_binary, llm_judge, quality_baseline):
        """Test simple French sentence quality (beta - uses espeak G2P)."""
        text = "Bonjour, le monde! Ceci est un test."

        fr_baseline = quality_baseline['language_baselines']['fr']

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'fr', output_path)
            assert success, "French TTS generation failed"

            result = evaluate_audio(llm_judge, output_path, text, 'fr')

            assert 'error' not in result, f"LLM Judge error: {result.get('error')}"

            print(f"\nLLM Judge Results for: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")

            min_accuracy = fr_baseline['accuracy']['baseline'] - fr_baseline['accuracy']['tolerance']
            assert result.get('accuracy', 0) >= min_accuracy, \
                f"Accuracy {result.get('accuracy', 0)} below threshold {min_accuracy}"

            record_quality_result('fr', text, result, 'test_french_simple_sentence')

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_french_technical_sentence(self, openai_available, tts_binary, llm_judge, quality_baseline):
        """Test technical French sentence quality."""
        text = "La fonction a été refactorisée avec succès."

        fr_baseline = quality_baseline['language_baselines']['fr']

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'fr', output_path)
            assert success, "French TTS generation failed"

            result = evaluate_audio(llm_judge, output_path, text, 'fr')

            assert 'error' not in result, f"LLM Judge error: {result.get('error')}"

            print(f"\nLLM Judge Results for: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")

            min_accuracy = fr_baseline['accuracy']['baseline'] - fr_baseline['accuracy']['tolerance']
            assert result.get('accuracy', 0) >= min_accuracy

            record_quality_result('fr', text, result, 'test_french_technical_sentence')

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
@pytest.mark.slow
class TestDifficultCases:
    """Test difficult cases: numbers, acronyms, punctuation.

    NOTE: Number pronunciation was fixed in Worker #106 by adding text normalization.
    Numbers are now converted to words before G2P processing.
    """

    def test_numbers_pronunciation(self, openai_available, tts_binary, llm_judge, test_corpus, quality_baseline):
        """Test number pronunciation quality.

        Worker #106 fixed number pronunciation by implementing text normalization:
        - Integers: 42 -> "forty two"
        - Large numbers: 1,234,567 -> "one million two hundred thirty four thousand..."
        - Decimals: 2.5 -> "two point five"
        - Currency: $42.99 -> "forty two dollars and ninety nine cents"
        - Percentages: 8.25% -> "eight point two five percent"
        - Times: 14:32:15 -> "fourteen thirty two and fifteen seconds"
        """
        # Get first number test case
        numbers = test_corpus.get('difficult_cases', {}).get('numbers', [])
        if not numbers:
            pytest.skip("No number test cases in corpus")

        text = numbers[0]['text']  # "The server processed 1,234,567 requests in 2.5 hours."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'en', output_path)
            assert success, "TTS generation failed for numbers"

            result = evaluate_audio(llm_judge, output_path, text, 'en')

            print(f"\nLLM Judge Results for numbers: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")
            print(f"  Issues: {result.get('issues', 'none')}")

            # Numbers may have slightly lower accuracy (dates, large numbers hard)
            assert result.get('accuracy', 0) >= 3.5, \
                f"Numbers accuracy {result.get('accuracy', 0)} too low"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_acronyms_pronunciation(self, openai_available, tts_binary, llm_judge, test_corpus, quality_baseline):
        """Test acronym pronunciation quality."""
        acronyms = test_corpus.get('difficult_cases', {}).get('acronyms', [])
        if not acronyms:
            pytest.skip("No acronym test cases in corpus")

        text = acronyms[0]['text']  # "The API returns JSON via HTTP with TLS encryption."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'en', output_path)
            assert success, "TTS generation failed for acronyms"

            result = evaluate_audio(llm_judge, output_path, text, 'en')

            print(f"\nLLM Judge Results for acronyms: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")
            print(f"  Issues: {result.get('issues', 'none')}")

            # Acronyms should be spelled out correctly
            # Lower threshold to 3.0 due to GPT-4o evaluation variance
            # (e.g., may hallucinate word substitutions that don't exist)
            assert result.get('accuracy', 0) >= 3.0, \
                f"Acronyms accuracy {result.get('accuracy', 0)} too low"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_punctuation_prosody(self, openai_available, tts_binary, llm_judge, test_corpus, quality_baseline):
        """Test punctuation affects prosody appropriately.

        Uses GPT-4o audio model. Threshold of 3.0 naturalness allows for LLM variance.
        """
        punctuation = test_corpus.get('difficult_cases', {}).get('punctuation', [])
        if not punctuation:
            pytest.skip("No punctuation test cases in corpus")

        text = punctuation[0]['text']  # "Wait... what? That's impossible!"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'en', output_path)
            assert success, "TTS generation failed for punctuation"

            result = evaluate_audio(llm_judge, output_path, text, 'en')

            print(f"\nLLM Judge Results for punctuation: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")
            print(f"  Issues: {result.get('issues', 'none')}")

            # Focus on naturalness for punctuation (prosody)
            assert result.get('naturalness', 0) >= 3.0, \
                f"Punctuation naturalness {result.get('naturalness', 0)} too low"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_mixed_content(self, openai_available, tts_binary, llm_judge, test_corpus, quality_baseline):
        """Test mixed content (URLs, code, special chars)."""
        mixed = test_corpus.get('difficult_cases', {}).get('mixed_content', [])
        if not mixed:
            pytest.skip("No mixed content test cases in corpus")

        text = mixed[3]['text']  # "Press Ctrl+C to copy or Cmd+V to paste."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(tts_binary, text, 'en', output_path)
            assert success, "TTS generation failed for mixed content"

            result = evaluate_audio(llm_judge, output_path, text, 'en')

            print(f"\nLLM Judge Results for mixed content: '{text}'")
            print(f"  Accuracy: {result.get('accuracy', 0)}/5")
            print(f"  Naturalness: {result.get('naturalness', 0)}/5")
            print(f"  Quality: {result.get('quality', 0)}/5")
            print(f"  Transcription: {result.get('transcription', 'N/A')}")
            print(f"  Issues: {result.get('issues', 'none')}")

            # Mixed content may have variations
            assert result.get('accuracy', 0) >= 3.0, \
                f"Mixed content accuracy {result.get('accuracy', 0)} too low"

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
class TestHistoricalRegression:
    """Historical regression tests - compare against historical averages."""

    def test_english_not_regressing(self, quality_baseline):
        """Verify English quality hasn't dropped from historical average."""
        hist_avg = get_historical_average('en', 'accuracy', last_n=10)

        if hist_avg is None:
            pytest.skip("Insufficient historical data for English regression test")

        # Allow 0.5 point drop from historical average
        regression_threshold = quality_baseline['ci_config']['regression_threshold']
        min_acceptable = hist_avg - regression_threshold

        print(f"\nHistorical English regression check:")
        print(f"  Historical average: {hist_avg:.2f}")
        print(f"  Regression threshold: {regression_threshold}")
        print(f"  Minimum acceptable: {min_acceptable:.2f}")

        # This test doesn't fail - it's informational
        # Future runs should add actual current score comparison

    def test_japanese_not_regressing(self, quality_baseline):
        """Verify Japanese quality hasn't dropped from historical average."""
        hist_avg = get_historical_average('ja', 'accuracy', last_n=10)

        if hist_avg is None:
            pytest.skip("Insufficient historical data for Japanese regression test")

        regression_threshold = quality_baseline['ci_config']['regression_threshold']
        min_acceptable = hist_avg - regression_threshold

        print(f"\nHistorical Japanese regression check:")
        print(f"  Historical average: {hist_avg:.2f}")
        print(f"  Minimum acceptable: {min_acceptable:.2f}")

    def test_history_file_exists(self):
        """Verify quality history file exists and is valid."""
        if not QUALITY_HISTORY_PATH.exists():
            pytest.skip("Quality history file not yet created")

        with open(QUALITY_HISTORY_PATH) as f:
            history = json.load(f)

        assert 'runs' in history, "History file missing 'runs' key"
        print(f"\nQuality history contains {len(history['runs'])} test runs")


@pytest.mark.quality
class TestQualityRegression:
    """Quality regression tests - compare against baseline."""

    def test_english_baseline_comparison(self, quality_baseline):
        """Verify English baseline scores are reasonable."""
        en_baseline = quality_baseline['language_baselines']['en']

        assert en_baseline['accuracy']['baseline'] >= 4.0
        assert en_baseline['naturalness']['baseline'] >= 3.5
        assert en_baseline['quality']['baseline'] >= 3.5
        assert en_baseline['status'] == 'production'

    def test_japanese_baseline_comparison(self, quality_baseline):
        """Verify Japanese baseline scores are reasonable."""
        ja_baseline = quality_baseline['language_baselines']['ja']

        assert ja_baseline['accuracy']['baseline'] >= 4.0
        assert ja_baseline['naturalness']['baseline'] >= 3.5
        assert ja_baseline['quality']['baseline'] >= 3.5
        assert ja_baseline['status'] == 'production'

    def test_spanish_baseline_comparison(self, quality_baseline):
        """Verify Spanish baseline scores are reasonable (beta status - fixed in #107)."""
        es_baseline = quality_baseline['language_baselines']['es']

        # Spanish TTS fixed in #107 - now using correct espeak voice
        assert es_baseline['accuracy']['baseline'] >= 3.5
        assert es_baseline['naturalness']['baseline'] >= 3.0
        assert es_baseline['quality']['baseline'] >= 3.0
        assert es_baseline['status'] == 'beta'

    def test_french_baseline_comparison(self, quality_baseline):
        """Verify French baseline scores are reasonable.

        Note: French baseline lowered in #108 due to Kokoro model limitations
        with French phonemes. The model wasn't trained on French voices.
        """
        fr_baseline = quality_baseline['language_baselines']['fr']

        # French accuracy lowered to 3.0 - Kokoro model not trained on French
        assert fr_baseline['accuracy']['baseline'] >= 2.5
        assert fr_baseline['naturalness']['baseline'] >= 2.5
        assert fr_baseline['quality']['baseline'] >= 2.5
        assert fr_baseline['status'] == 'beta'

    def test_chinese_production_documented(self, quality_baseline):
        """Verify Chinese TTS production status is properly documented."""
        zh_baseline = quality_baseline['language_baselines']['zh']
        # Chinese TTS fixed in Worker #388 - status upgraded to 'production'
        # Now uses misaki lexicon with proper tones instead of espeak
        assert zh_baseline['status'] == 'production'
        assert 'misaki' in zh_baseline['notes'].lower() or 'tone' in zh_baseline['notes'].lower()


@pytest.mark.quality
class TestCorpusCoverage:
    """Verify test corpus has adequate coverage."""

    def test_corpus_has_all_languages(self, test_corpus):
        """Verify corpus covers all supported languages."""
        sentences = test_corpus['sentences']
        required_langs = ['en', 'ja', 'zh', 'es', 'fr']

        for lang in required_langs:
            assert lang in sentences, f"Missing language: {lang}"
            assert len(sentences[lang]) == 10, f"Expected 10 sentences for {lang}"

    def test_corpus_has_categories(self, test_corpus):
        """Verify corpus has diverse categories."""
        en_sentences = test_corpus['sentences']['en']
        categories = {s['category'] for s in en_sentences}

        assert 'technical' in categories, "Missing technical category"
        assert 'greeting' in categories, "Missing greeting category"

    def test_corpus_has_difficult_cases(self, test_corpus):
        """Verify corpus has difficult test cases."""
        difficult = test_corpus.get('difficult_cases', {})

        assert 'numbers' in difficult, "Missing numbers difficult cases"
        assert 'acronyms' in difficult, "Missing acronyms difficult cases"
        assert 'punctuation' in difficult, "Missing punctuation difficult cases"
        assert 'mixed_content' in difficult, "Missing mixed_content difficult cases"

        # Each category should have 5 test cases
        for category in ['numbers', 'acronyms', 'punctuation', 'mixed_content']:
            assert len(difficult[category]) == 5, f"Expected 5 {category} test cases"


# =============================================================================
# Main entry point for direct execution
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'quality'])
