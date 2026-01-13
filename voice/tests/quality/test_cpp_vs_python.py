"""
C++ vs Python TTS Head-to-Head Comparison Tests

Uses LLM-as-Judge to compare C++ Kokoro output against Python Kokoro output.
Tests FAIL if C++ loses more than 20% of comparisons for any language.

This catches quality regressions where C++ implementation diverges from Python.

Usage:
    pytest tests/quality/test_cpp_vs_python.py -v -m comparison
"""

import json
import os
import sys
import tempfile
import subprocess
import pytest
from pathlib import Path
from typing import Dict, List, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_QUALITY_DIR = Path(__file__).parent

# Add scripts to path
sys.path.insert(0, str(SCRIPTS_DIR))

# Comparison test sentences - curated for detecting quality differences
COMPARISON_TEXTS = {
    'en': [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Are you coming to the party tonight?",
    ],
    'ja': [
        "こんにちは",
        "今日はいい天気ですね。",
        "お元気ですか?",
    ],
    'zh': [
        "你好",
        "今天天气很好",
        "谢谢你的帮助",
        "我正在学习中文",
        "请问这个多少钱",
    ],
}

# Maximum acceptable loss rate per language
MAX_LOSS_RATE = 0.20  # 20%


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
    """Path to stream-tts-cpp binary."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"TTS binary not found at {binary}")
    return binary


@pytest.fixture(scope="module")
def python_kokoro_available():
    """Check if Python Kokoro is available."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / '.venv_kokoro/lib/python3.11/site-packages'))
        import kokoro
        return True
    except ImportError:
        pytest.skip("Python kokoro package not available")


@pytest.fixture(scope="module")
def llm_judge():
    """Import LLM audio judge module."""
    try:
        import llm_audio_judge
        return llm_audio_judge
    except ImportError:
        pytest.skip("llm_audio_judge module not found in scripts/")


def generate_cpp_audio(tts_binary: Path, text: str, language: str, output_path: Path, high_quality: bool = False) -> bool:
    """Generate audio using C++ Kokoro.

    Uses --speak mode with --lang flag to directly synthesize without translation.
    This ensures we're comparing pure TTS quality, not translation quality.

    Worker #404: Changed default to high_quality=False for FAST GPU (MPS) inference!
    MPS achieves ~0.72 correlation with Python at 8x the speed (40-45ms vs 350ms).
    Use high_quality=True only for parity debugging.
    """
    # Use --speak mode with --lang flag for direct TTS (no translation)
    # Worker #404: Default to MPS (fast GPU) - use --hq only when debugging quality issues
    cmd = [str(tts_binary), "--speak", text, "--lang", language, "--save-audio", str(output_path)]
    if high_quality:
        cmd.insert(1, "--hq")  # Add --hq after binary name for CPU inference

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=180,  # Increased timeout for CPU inference
        cwd=str(STREAM_TTS_CPP)
    )

    return output_path.exists() and output_path.stat().st_size > 1000


def generate_python_audio(text: str, language: str, output_path: Path) -> bool:
    """Generate audio using Python Kokoro."""
    script = f'''
import sys
sys.path.insert(0, "{PROJECT_ROOT}/.venv_kokoro/lib/python3.11/site-packages")
import soundfile as sf
from kokoro import KPipeline

lang_map = {{"en": "a", "ja": "j", "zh": "z"}}
voice_map = {{"en": "af_heart", "ja": "jf_alpha", "zh": "zm_yunjian"}}

lang_code = lang_map.get("{language}", "a")
voice = voice_map.get("{language}", "af_heart")

pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
generator = pipeline(
    "{text}",
    voice=voice,
    speed=1.0
)

# Collect all audio
all_audio = []
for gs, ps, audio in generator:
    all_audio.append(audio)

import numpy as np
full_audio = np.concatenate(all_audio) if all_audio else np.array([])
sf.write("{output_path}", full_audio, 24000)
'''

    result = subprocess.run(
        [str(PROJECT_ROOT / '.venv_kokoro/bin/python'), '-c', script],
        capture_output=True,
        timeout=120,
        cwd=str(PROJECT_ROOT)
    )

    return output_path.exists() and output_path.stat().st_size > 1000


def compare_audio_pair(llm_judge_module, audio1_path: Path, audio2_path: Path,
                       text: str, language: str) -> Dict:
    """
    Compare two audio files using LLM-as-Judge.

    Returns:
        Dict with winner ('audio1', 'audio2', or 'tie'), scores for both, and reasoning
    """
    result = llm_judge_module.compare_audio_pair_openai(
        str(audio1_path),
        str(audio2_path),
        text,
        language
    )
    return result


@pytest.mark.comparison
@pytest.mark.slow
class TestCppVsPythonEnglish:
    """English C++ vs Python comparison tests."""

    def test_english_head_to_head(self, openai_available, tts_binary, python_kokoro_available, llm_judge):
        """C++ vs Python English - C++ must not lose >20%."""
        texts = COMPARISON_TEXTS['en']
        results = []

        for text in texts:
            with tempfile.NamedTemporaryFile(suffix="_cpp.wav", delete=False) as f1:
                cpp_path = Path(f1.name)
            with tempfile.NamedTemporaryFile(suffix="_py.wav", delete=False) as f2:
                py_path = Path(f2.name)

            try:
                cpp_ok = generate_cpp_audio(tts_binary, text, 'en', cpp_path)
                py_ok = generate_python_audio(text, 'en', py_path)

                if not cpp_ok or not py_ok:
                    print(f"  Skipping '{text}' - generation failed (C++: {cpp_ok}, Py: {py_ok})")
                    continue

                result = compare_audio_pair(llm_judge, cpp_path, py_path, text, 'en')
                result['text'] = text
                results.append(result)

                print(f"\n  '{text}':")
                print(f"    C++: acc={result.get('audio1', {}).get('accuracy', 'N/A')}, "
                      f"nat={result.get('audio1', {}).get('naturalness', 'N/A')}")
                print(f"    Python: acc={result.get('audio2', {}).get('accuracy', 'N/A')}, "
                      f"nat={result.get('audio2', {}).get('naturalness', 'N/A')}")
                print(f"    Winner: {result.get('winner', 'N/A')}")

            finally:
                if cpp_path.exists():
                    cpp_path.unlink()
                if py_path.exists():
                    py_path.unlink()

        # Calculate loss rate
        cpp_losses = sum(1 for r in results if r.get('winner') == 'audio2')
        total = len(results)

        if total == 0:
            pytest.skip("No comparisons completed")

        loss_rate = cpp_losses / total

        print(f"\n  ENGLISH RESULTS: C++ lost {cpp_losses}/{total} = {loss_rate:.1%}")

        assert loss_rate <= MAX_LOSS_RATE, \
            f"English C++ loss rate {loss_rate:.1%} exceeds maximum {MAX_LOSS_RATE:.1%}"


@pytest.mark.comparison
@pytest.mark.slow
class TestCppVsPythonJapanese:
    """Japanese C++ vs Python comparison tests."""

    def test_japanese_head_to_head(self, openai_available, tts_binary, python_kokoro_available, llm_judge):
        """C++ vs Python Japanese - C++ must not lose >20%."""
        texts = COMPARISON_TEXTS['ja']
        results = []

        for text in texts:
            with tempfile.NamedTemporaryFile(suffix="_cpp.wav", delete=False) as f1:
                cpp_path = Path(f1.name)
            with tempfile.NamedTemporaryFile(suffix="_py.wav", delete=False) as f2:
                py_path = Path(f2.name)

            try:
                cpp_ok = generate_cpp_audio(tts_binary, text, 'ja', cpp_path)
                py_ok = generate_python_audio(text, 'ja', py_path)

                if not cpp_ok or not py_ok:
                    print(f"  Skipping '{text}' - generation failed")
                    continue

                result = compare_audio_pair(llm_judge, cpp_path, py_path, text, 'ja')
                result['text'] = text
                results.append(result)

                print(f"\n  '{text}':")
                print(f"    Winner: {result.get('winner', 'N/A')}")

            finally:
                if cpp_path.exists():
                    cpp_path.unlink()
                if py_path.exists():
                    py_path.unlink()

        cpp_losses = sum(1 for r in results if r.get('winner') == 'audio2')
        total = len(results)

        if total == 0:
            pytest.skip("No comparisons completed")

        loss_rate = cpp_losses / total

        print(f"\n  JAPANESE RESULTS: C++ lost {cpp_losses}/{total} = {loss_rate:.1%}")

        assert loss_rate <= MAX_LOSS_RATE, \
            f"Japanese C++ loss rate {loss_rate:.1%} exceeds maximum {MAX_LOSS_RATE:.1%}"


@pytest.mark.comparison
@pytest.mark.slow
class TestCppVsPythonChinese:
    """Chinese C++ vs Python comparison tests - CRITICAL for tone quality."""

    def test_chinese_head_to_head(self, openai_available, tts_binary, python_kokoro_available, llm_judge):
        """C++ vs Python Chinese - C++ must not lose >20%.

        This is the critical test for Chinese tone quality.
        Worker #388 fixed Chinese prosody with misaki lexicon + jieba word boundaries.
        """
        texts = COMPARISON_TEXTS['zh']
        results = []

        for text in texts:
            with tempfile.NamedTemporaryFile(suffix="_cpp.wav", delete=False) as f1:
                cpp_path = Path(f1.name)
            with tempfile.NamedTemporaryFile(suffix="_py.wav", delete=False) as f2:
                py_path = Path(f2.name)

            try:
                cpp_ok = generate_cpp_audio(tts_binary, text, 'zh', cpp_path)
                py_ok = generate_python_audio(text, 'zh', py_path)

                if not cpp_ok or not py_ok:
                    print(f"  Skipping '{text}' - generation failed (C++: {cpp_ok}, Py: {py_ok})")
                    continue

                result = compare_audio_pair(llm_judge, cpp_path, py_path, text, 'zh')
                result['text'] = text
                results.append(result)

                print(f"\n  '{text}':")
                print(f"    C++ scores: accuracy={result.get('audio1', {}).get('accuracy', 'N/A')}, "
                      f"naturalness={result.get('audio1', {}).get('naturalness', 'N/A')}, "
                      f"quality={result.get('audio1', {}).get('quality', 'N/A')}")
                print(f"    Python scores: accuracy={result.get('audio2', {}).get('accuracy', 'N/A')}, "
                      f"naturalness={result.get('audio2', {}).get('naturalness', 'N/A')}, "
                      f"quality={result.get('audio2', {}).get('quality', 'N/A')}")
                print(f"    Winner: {result.get('winner', 'N/A')}")
                print(f"    Reasoning: {result.get('reasoning', 'N/A')[:100]}...")

            finally:
                if cpp_path.exists():
                    cpp_path.unlink()
                if py_path.exists():
                    py_path.unlink()

        # Calculate loss rate
        cpp_losses = sum(1 for r in results if r.get('winner') == 'audio2')
        cpp_wins = sum(1 for r in results if r.get('winner') == 'audio1')
        ties = sum(1 for r in results if r.get('winner') == 'tie')
        total = len(results)

        if total == 0:
            pytest.skip("No comparisons completed")

        loss_rate = cpp_losses / total

        print(f"\n  CHINESE RESULTS:")
        print(f"    C++ wins: {cpp_wins}")
        print(f"    Python wins: {cpp_losses}")
        print(f"    Ties: {ties}")
        print(f"    Total: {total}")
        print(f"    C++ loss rate: {loss_rate:.1%} (max allowed: {MAX_LOSS_RATE:.1%})")

        # Save results for analysis
        results_path = TESTS_QUALITY_DIR / "cpp_vs_python_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'language': 'zh',
                'cpp_wins': cpp_wins,
                'python_wins': cpp_losses,
                'ties': ties,
                'total': total,
                'loss_rate': loss_rate,
                'details': results
            }, f, indent=2, ensure_ascii=False)

        assert loss_rate <= MAX_LOSS_RATE, \
            f"Chinese C++ loss rate {loss_rate:.1%} exceeds maximum {MAX_LOSS_RATE:.1%}. " \
            f"C++ is losing to Python too often - check phoneme mapping!"


@pytest.mark.comparison
class TestCppVsPythonSummary:
    """Summary tests for C++ vs Python comparison."""

    def test_comparison_config_valid(self):
        """Verify comparison test configuration is valid."""
        assert MAX_LOSS_RATE == 0.20, "Max loss rate should be 20%"

        for lang, texts in COMPARISON_TEXTS.items():
            assert len(texts) >= 3, f"Need at least 3 test texts for {lang}"

    def test_all_languages_have_comparison_texts(self):
        """Verify all production languages have comparison texts."""
        required = ['en', 'ja', 'zh']
        for lang in required:
            assert lang in COMPARISON_TEXTS, f"Missing comparison texts for {lang}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'comparison'])
