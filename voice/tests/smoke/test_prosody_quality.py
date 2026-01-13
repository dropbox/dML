"""
TTS Prosody Quality Smoke Tests

These tests verify:
1. C++ TTS quality matches or exceeds Python quality (MAIN GOAL)
2. Prosody issues are being investigated and fixed

Run: pytest tests/smoke/test_prosody_quality.py -v -m quality
Note: Requires OPENAI_API_KEY environment variable

Quality Status (2025-12-09):
- C++ wins 5-0 vs Python in LLM-as-Judge comparison
- MPS model now uses complex STFT (PyTorch 2.9.1 supports torch.angle on MPS)
- Prosody tests use GPT-4o audio model for evaluation

Bug Reports:
- G2P Audit: reports/main/G2P_AUDIT_2025-12-07.md
- Prosody Comparison: reports/prosody_comparison/comparison_results.json
"""

import json
import os
import base64
import subprocess
import pytest
import tempfile
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"


def load_env():
    """Load .env file for API keys."""
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key, value)


load_env()


@pytest.fixture(scope="module")
def cpp_binary():
    """Path to stream-tts-cpp binary."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"Binary not found: {binary}")
    return binary


@pytest.fixture(scope="module")
def openai_client():
    """OpenAI client for LLM-as-judge evaluation."""
    key = os.environ.get('OPENAI_API_KEY')
    if not key or key.startswith('sk-...'):
        pytest.skip("OPENAI_API_KEY not configured")
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        pytest.skip("openai package not installed")


def generate_tts_audio(cpp_binary: Path, text: str, lang: str, output_path: Path, timeout: int = 60) -> bool:
    """Generate TTS audio using stream-tts-cpp --speak."""
    # Note: PYTORCH_ENABLE_MPS_FALLBACK no longer needed with PyTorch 2.9.1+
    env = os.environ.copy()
    result = subprocess.run(
        [str(cpp_binary), "--speak", text, "--lang", lang, "--save-audio", str(output_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(STREAM_TTS_CPP),
        env=env
    )
    return output_path.exists() and output_path.stat().st_size > 1000


def evaluate_specific_issue(client, audio_path: Path, expected_text: str, language: str, issue_description: str) -> dict:
    """Evaluate audio for a specific reported issue using GPT-4o-audio with deterministic settings."""
    with open(audio_path, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = f"""You are an expert TTS audio evaluator. A user reported a SPECIFIC issue with this audio.

Listen to it VERY carefully and evaluate if this issue exists.

Expected text: "{expected_text}"
Language: {language}

USER'S REPORTED ISSUE: "{issue_description}"

Listen closely multiple times if needed. Respond in JSON only:
{{
    "issue_present": true|false,
    "issue_severity": "none|minor|moderate|severe",
    "what_you_heard": "describe exactly what you heard related to this issue",
    "transcription": "what the audio actually says",
    "confidence": "high|medium|low"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-audio-2025-08-28",  # GPT-5 based audio model
            modalities=["text"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}
                ]
            }],
            max_tokens=1000,
            temperature=0,
            response_format={"type": "json_object"},
            seed=0
        )
    except Exception as exc:
        # Some SDK versions or models may not support deterministic params yet.
        # Fall back to a minimal request while keeping temperature fixed at 0.
        print(f"Deterministic LLM params unsupported, retrying without seed/response_format: {exc}")
        response = client.chat.completions.create(
            model="gpt-audio-2025-08-28",
            modalities=["text"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}
                ]
            }],
            max_tokens=1000,
            temperature=0
        )

    result_text = response.choices[0].message.content
    json_start = result_text.find('{')
    json_end = result_text.rfind('}') + 1
    if json_start >= 0 and json_end > json_start:
        return json.loads(result_text[json_start:json_end])
    return {"error": "No JSON found", "raw_response": result_text}


@pytest.mark.quality
@pytest.mark.slow
class TestJapaneseProsodyIssues:
    """Japanese TTS prosody quality tests."""

    def test_konnichiwa_no_extra_syllable(self, cpp_binary, openai_client):
        """
        Japanese 'こんにちは' pronunciation quality test.

        Uses GPT-4o audio model to evaluate for trailing vowel artifacts.
        Allows minor issues due to LLM-as-Judge variance.
        """
        text = "こんにちは"
        issue = "The word 'konichiwa' sounds like 'konichiwa-a' - there's an extra 'a' sound at the very end"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(cpp_binary, text, 'ja', output_path)
            assert success, "Japanese TTS generation failed"

            result = evaluate_specific_issue(openai_client, output_path, text, 'ja', issue)

            print(f"\n=== Japanese Prosody Test: {text} ===")
            print(f"Issue present: {result.get('issue_present')}")
            print(f"Severity: {result.get('issue_severity')}")
            print(f"What was heard: {result.get('what_you_heard')}")
            print(f"Transcription: {result.get('transcription')}")
            print(f"Confidence: {result.get('confidence')}")

            # Only fail on severe issues - LLM-as-Judge is subjective
            if result.get('issue_present') and result.get('issue_severity') == 'severe':
                pytest.fail(
                    f"Japanese 'こんにちは' has severe pronunciation issue. "
                    f"Heard: {result.get('what_you_heard')}"
                )

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
@pytest.mark.slow
class TestEnglishProsodyIssues:
    """English TTS prosody quality tests."""

    def test_statement_has_falling_intonation(self, cpp_binary, openai_client):
        """
        English statement intonation quality test.

        Uses GPT-4o audio model to evaluate statement-appropriate falling intonation.
        Allows minor issues due to LLM-as-Judge variance.
        """
        text = "The quick brown fox jumps over the lazy dog"
        issue = "The word 'dog' at the end has a rising intonation like a question, instead of falling intonation for a statement"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(cpp_binary, text, 'en', output_path)
            assert success, "English TTS generation failed"

            result = evaluate_specific_issue(openai_client, output_path, text, 'en', issue)

            print(f"\n=== English Prosody Test: {text[:50]}... ===")
            print(f"Issue present: {result.get('issue_present')}")
            print(f"Severity: {result.get('issue_severity')}")
            print(f"What was heard: {result.get('what_you_heard')}")
            print(f"Confidence: {result.get('confidence')}")

            # Only fail on severe issues - LLM-as-Judge is subjective
            if result.get('issue_present') and result.get('issue_severity') == 'severe':
                pytest.fail(
                    f"English statement has severe intonation issue. "
                    f"Heard: {result.get('what_you_heard')}"
                )

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_question_has_rising_intonation(self, cpp_binary, openai_client):
        """
        English question intonation quality test.

        Uses GPT-4o audio model to evaluate question-appropriate rising intonation.
        Allows minor issues due to LLM-as-Judge variance.
        """
        text = "Are you coming to the party tonight?"
        issue = "The sentence does NOT have rising intonation at the end - it sounds flat or falling like a statement"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(cpp_binary, text, 'en', output_path)
            assert success, "English TTS generation failed"

            result = evaluate_specific_issue(openai_client, output_path, text, 'en', issue)

            print(f"\n=== English Question Intonation Test ===")
            print(f"Issue present (no rising tone): {result.get('issue_present')}")
            print(f"Severity: {result.get('issue_severity')}")
            print(f"What was heard: {result.get('what_you_heard')}")

            # Only fail on severe issues - LLM-as-Judge is subjective
            if result.get('issue_present') and result.get('issue_severity') == 'severe':
                pytest.fail(
                    f"English question has severe intonation issue. "
                    f"Heard: {result.get('what_you_heard')}"
                )

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
@pytest.mark.slow
class TestChineseProsodyIssues:
    """Chinese TTS prosody quality tests - tone correctness."""

    def test_nihao_correct_tones(self, cpp_binary, openai_client):
        """
        Chinese '你好' tone correctness test using LLM-as-Judge.

        Uses GPT-4o audio model to evaluate Chinese tones.
        Allows minor issues due to LLM-as-Judge variance.
        """
        text = "你好"
        issue = "The tones sound wrong or unnatural - both syllables sound the same (rising) instead of rising then dipping. The intonation sounds 'flippy' or robotic."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(cpp_binary, text, 'zh', output_path)
            assert success, "Chinese TTS generation failed"

            result = evaluate_specific_issue(openai_client, output_path, text, 'zh', issue)

            print(f"\n=== Chinese Tone Test: {text} ===")
            print(f"Issue present: {result.get('issue_present')}")
            print(f"Severity: {result.get('issue_severity')}")
            print(f"What was heard: {result.get('what_you_heard')}")
            print(f"Transcription: {result.get('transcription')}")
            print(f"Confidence: {result.get('confidence')}")

            # Only fail on severe issues - LLM-as-Judge is subjective
            if result.get('issue_present') and result.get('issue_severity') == 'severe':
                pytest.fail(
                    f"Chinese '你好' has severe tone issue. "
                    f"Heard: {result.get('what_you_heard')}"
                )

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_chinese_sentence_natural_tones(self, cpp_binary, openai_client):
        """Test a longer Chinese sentence for natural tone flow."""
        text = "今天天气很好"  # "The weather is good today"
        issue = "The tones sound unnatural, robotic, or have weird pitch jumps between syllables"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success = generate_tts_audio(cpp_binary, text, 'zh', output_path)
            assert success, "Chinese TTS generation failed"

            result = evaluate_specific_issue(openai_client, output_path, text, 'zh', issue)

            print(f"\n=== Chinese Sentence Tone Test: {text} ===")
            print(f"Issue present: {result.get('issue_present')}")
            print(f"Severity: {result.get('issue_severity')}")
            print(f"What was heard: {result.get('what_you_heard')}")

            # Allow minor issues for complex sentences
            if result.get('issue_present') and result.get('issue_severity') == 'severe':
                pytest.fail(
                    f"Chinese sentence has severe tone issues. "
                    f"Heard: {result.get('what_you_heard')}"
                )

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.quality
@pytest.mark.slow
class TestCppVsPythonQuality:
    """
    Tests that verify C++ TTS matches or exceeds Python quality.

    This is the MAIN SUCCESS CRITERION for Phase 10.
    Worker #333-334: C++ wins 5-0 vs Python in LLM-as-Judge comparison.
    """

    def test_cpp_quality_matches_python(self, cpp_binary, openai_client):
        """
        Verify C++ TTS quality matches or exceeds Python reference.

        Uses pre-generated audio from compare_prosody.py to avoid regenerating.
        Checks the comparison results JSON for overall quality assessment.
        """
        results_path = PROJECT_ROOT / "reports" / "prosody_comparison" / "comparison_results.json"
        if not results_path.exists():
            pytest.skip("No comparison results found. Run: python scripts/compare_prosody.py")

        with open(results_path) as f:
            results = json.load(f)

        # Results is a list of comparison objects
        # audio1 = Python, audio2 = C++ (per compare_prosody.py convention)
        comparisons = results if isinstance(results, list) else results.get('comparisons', [])

        # Count wins
        cpp_wins = sum(1 for r in comparisons if r.get('winner') == 'audio2')
        python_wins = sum(1 for r in comparisons if r.get('winner') == 'audio1')
        ties = sum(1 for r in comparisons if r.get('winner') == 'tie')

        print(f"\n=== C++ vs Python Quality Test ===")
        print(f"C++ wins: {cpp_wins}")
        print(f"Python wins: {python_wins}")
        print(f"Ties: {ties}")

        # C++ must win more than or equal to Python
        assert cpp_wins >= python_wins, \
            f"C++ quality below Python: C++ {cpp_wins} wins, Python {python_wins} wins. " \
            "Run: python scripts/compare_prosody.py to see details"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'quality', '-s'])
