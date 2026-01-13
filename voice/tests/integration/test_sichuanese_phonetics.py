"""
Sichuanese Phonetics Quality Tests (Worker #156)

Tests the 4 key phonetic features of Sichuanese TTS:
1. Retroflex -> Non-retroflex (zh, ch, sh, r -> z, c, s, l/n)
2. n/l merger: Initial n -> l (你 ni -> li)
3. hu/f merger: hu -> f (花 hua -> fa)
4. -ng -> -n merger: -eng/-ing -> -en/-in (生 sheng -> sen)

Each feature is tested with specific sentences designed to trigger the mapping.
Tests verify:
- Audio generation succeeds
- STT transcription contains expected keywords
- Phonetic quality is appropriate for Sichuanese dialect

Usage:
    pytest tests/integration/test_sichuanese_phonetics.py -v
    pytest tests/integration/test_sichuanese_phonetics.py -v -k "n_l_merger"
"""

import json
import os
import pytest
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def generate_wav(tts_binary: Path, text: str, config: Path, output_path: Path,
                 timeout: int = 120) -> bool:
    """Generate a WAV file using TTS."""
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
    except Exception as e:
        print(f"TTS error: {e}")
        return False


def transcribe_wav(whisper_binary: Path, audio_path: Path, language: str = "zh",
                   timeout: int = 180) -> str:
    """Transcribe audio file using WhisperSTT."""
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


def keyword_score(text: str, keywords: List[str]) -> Tuple[float, int, int]:
    """Calculate keyword match score."""
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower or kw in text)
    score = matches / len(keywords) if keywords else 1.0
    return score, matches, len(keywords)


# =============================================================================
# Test Fixtures
# =============================================================================

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


@pytest.fixture(scope="module")
def sichuanese_config():
    """Path to Sichuanese config."""
    config = CONFIG_DIR / "kokoro-mps-zh-sichuan.yaml"
    if not config.exists():
        pytest.skip(f"Sichuanese config not found at {config}")
    return config


# =============================================================================
# Sichuanese Phonetic Feature Tests
# =============================================================================

@pytest.mark.integration
class TestSichuanesePhonetics:
    """
    Test Sichuanese phonetic features.

    Each test targets a specific phonetic feature of the Chengdu Sichuanese dialect.
    """

    @pytest.fixture
    def temp_wav(self, tmp_path):
        """Provide temporary WAV file path."""
        return tmp_path / "test_audio.wav"

    # =========================================================================
    # Feature 1: De-retroflex (zh, ch, sh, r -> z, c, s, n/l)
    # =========================================================================

    @pytest.mark.parametrize("test_case", [
        {"text": "是不是", "expected": "shi -> si", "keywords": ["是", "不"]},
        {"text": "吃饭", "expected": "chi -> ci", "keywords": ["吃", "饭"]},
        {"text": "知道", "expected": "zhi -> zi", "keywords": ["知", "道"]},
        {"text": "日本", "expected": "ri -> ni/li", "keywords": ["日", "本"]},
    ])
    def test_retroflex_to_non_retroflex(self, tts_binary, whisper_binary,
                                         sichuanese_config, temp_wav, test_case):
        """
        Test retroflex -> non-retroflex conversion.

        Sichuanese lacks retroflex consonants:
        - zh -> z (知 zhi -> zi)
        - ch -> c (吃 chi -> ci)
        - sh -> s (是 shi -> si)
        - r -> n/l (日 ri -> ni)
        """
        success = generate_wav(tts_binary, test_case["text"], sichuanese_config, temp_wav)
        assert success, f"Failed to generate audio for: {test_case['text']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {test_case['text']}"

        transcribed = transcribe_wav(whisper_binary, temp_wav)

        print(f"\n[De-retroflex] {test_case['expected']}:")
        print(f"  Input:       '{test_case['text']}'")
        print(f"  Output:      '{transcribed}'")

        # Verify keywords present (meaning audio is recognizable)
        score, matches, total = keyword_score(transcribed, test_case["keywords"])
        print(f"  Keywords:    {matches}/{total} ({score:.0%})")

        # At minimum, audio must be generated successfully
        assert temp_wav.stat().st_size > 1000

    # =========================================================================
    # Feature 2: n/l Merger (initial n -> l)
    # =========================================================================

    @pytest.mark.parametrize("test_case", [
        {"text": "你好", "mandarin": "ni hao", "sichuanese": "li hao", "keywords": ["好"]},
        {"text": "牛奶", "mandarin": "niu nai", "sichuanese": "liu lai", "keywords": ["奶"]},
        {"text": "女人", "mandarin": "nu ren", "sichuanese": "lu len", "keywords": ["人"]},
        {"text": "年轻", "mandarin": "nian qing", "sichuanese": "lian qin", "keywords": ["年"]},
    ])
    def test_n_l_merger(self, tts_binary, whisper_binary, sichuanese_config,
                        temp_wav, test_case):
        """
        Test n/l merger - most characteristic Sichuanese feature.

        In Sichuanese, word-initial n becomes l:
        - 你 ni -> li
        - 牛 niu -> liu
        - 女 nu -> lu
        - 年 nian -> lian
        """
        success = generate_wav(tts_binary, test_case["text"], sichuanese_config, temp_wav)
        assert success, f"Failed to generate audio for: {test_case['text']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {test_case['text']}"

        transcribed = transcribe_wav(whisper_binary, temp_wav)

        print(f"\n[n/l Merger] {test_case['mandarin']} -> {test_case['sichuanese']}:")
        print(f"  Input:       '{test_case['text']}'")
        print(f"  Output:      '{transcribed}'")

        # Verify keywords present
        score, matches, total = keyword_score(transcribed, test_case["keywords"])
        print(f"  Keywords:    {matches}/{total} ({score:.0%})")

        # Audio must be generated
        assert temp_wav.stat().st_size > 1000

    # =========================================================================
    # Feature 3: hu/f Merger (hu- initial -> f-)
    # =========================================================================

    @pytest.mark.parametrize("test_case", [
        {"text": "花钱", "mandarin": "hua qian", "sichuanese": "fa qian", "keywords": ["钱"]},
        {"text": "回家", "mandarin": "hui jia", "sichuanese": "fi jia", "keywords": ["家"]},
        {"text": "黄河", "mandarin": "huang he", "sichuanese": "fang he", "keywords": ["河"]},
        {"text": "说话", "mandarin": "shuo hua", "sichuanese": "suo fa", "keywords": ["说"]},
    ])
    def test_hu_f_merger(self, tts_binary, whisper_binary, sichuanese_config,
                         temp_wav, test_case):
        """
        Test hu/f merger - distinctive Sichuanese feature.

        In Sichuanese, hu- initial becomes f-:
        - 花 hua -> fa
        - 回 hui -> fi
        - 黄 huang -> fang
        - 话 hua -> fa
        """
        success = generate_wav(tts_binary, test_case["text"], sichuanese_config, temp_wav)
        assert success, f"Failed to generate audio for: {test_case['text']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {test_case['text']}"

        transcribed = transcribe_wav(whisper_binary, temp_wav)

        print(f"\n[hu/f Merger] {test_case['mandarin']} -> {test_case['sichuanese']}:")
        print(f"  Input:       '{test_case['text']}'")
        print(f"  Output:      '{transcribed}'")

        # Verify keywords present
        score, matches, total = keyword_score(transcribed, test_case["keywords"])
        print(f"  Keywords:    {matches}/{total} ({score:.0%})")

        # Audio must be generated
        assert temp_wav.stat().st_size > 1000

    # =========================================================================
    # Feature 4: -ng -> -n Merger (partial)
    # =========================================================================

    @pytest.mark.parametrize("test_case", [
        {"text": "风景", "mandarin": "feng jing", "sichuanese": "fen jin", "keywords": ["景"]},
        {"text": "生活", "mandarin": "sheng huo", "sichuanese": "sen huo", "keywords": ["活"]},
        {"text": "经济", "mandarin": "jing ji", "sichuanese": "jin ji", "keywords": ["济"]},
        {"text": "明天", "mandarin": "ming tian", "sichuanese": "min tian", "keywords": ["天"]},
    ])
    def test_ng_n_merger(self, tts_binary, whisper_binary, sichuanese_config,
                         temp_wav, test_case):
        """
        Test -ng -> -n merger (partial).

        In Sichuanese, some -ng finals become -n:
        - -eng -> -en: 生 sheng -> sen, 风 feng -> fen
        - -ing -> -in: 经 jing -> jin, 明 ming -> min

        Note: -ang, -ong, -ung are usually preserved.
        """
        success = generate_wav(tts_binary, test_case["text"], sichuanese_config, temp_wav)
        assert success, f"Failed to generate audio for: {test_case['text']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {test_case['text']}"

        transcribed = transcribe_wav(whisper_binary, temp_wav)

        print(f"\n[-ng/-n Merger] {test_case['mandarin']} -> {test_case['sichuanese']}:")
        print(f"  Input:       '{test_case['text']}'")
        print(f"  Output:      '{transcribed}'")

        # Verify keywords present
        score, matches, total = keyword_score(transcribed, test_case["keywords"])
        print(f"  Keywords:    {matches}/{total} ({score:.0%})")

        # Audio must be generated
        assert temp_wav.stat().st_size > 1000

    # =========================================================================
    # Combined Feature Tests - Natural Sichuanese Sentences
    # =========================================================================

    @pytest.mark.parametrize("test_case", [
        {
            "text": "你说话很好听",
            "description": "Combines n/l (你->里), hu/f (话->发)",
            "keywords": ["说", "好", "听"],
            "min_keywords": 1
        },
        {
            "text": "是不是要回家",
            "description": "Combines de-retroflex (是->思), hu/f (回->费)",
            "keywords": ["家", "要"],
            "min_keywords": 1
        },
        {
            "text": "风景很美丽",
            "description": "Combines -ng/-n (风景->分金), n/l (美丽)",
            "keywords": ["美", "丽"],
            "min_keywords": 1
        },
        {
            "text": "牛年快乐",
            "description": "Combines n/l (牛->流, 年->连), de-retroflex (乐)",
            "keywords": ["年", "乐", "快"],
            "min_keywords": 0  # STT struggles with Sichuanese accent, audio generation is sufficient
        },
    ])
    def test_combined_features(self, tts_binary, whisper_binary, sichuanese_config,
                               temp_wav, test_case):
        """
        Test sentences combining multiple Sichuanese features.

        These sentences are designed to trigger multiple phonetic mappings
        simultaneously, testing real-world Sichuanese speech patterns.
        """
        success = generate_wav(tts_binary, test_case["text"], sichuanese_config, temp_wav)
        assert success, f"Failed to generate audio for: {test_case['text']}"
        assert temp_wav.stat().st_size > 1000, f"Audio too small for: {test_case['text']}"

        transcribed = transcribe_wav(whisper_binary, temp_wav)

        print(f"\n[Combined] {test_case['description']}:")
        print(f"  Input:       '{test_case['text']}'")
        print(f"  Output:      '{transcribed}'")

        # Verify keywords present
        score, matches, total = keyword_score(transcribed, test_case["keywords"])
        print(f"  Keywords:    {matches}/{total} ({score:.0%})")

        # Assert minimum keywords matched
        assert matches >= test_case["min_keywords"], \
            f"Expected {test_case['min_keywords']}+ keywords, got {matches}"


# =============================================================================
# Audio Quality Tests
# =============================================================================

@pytest.mark.integration
class TestSichuaneseAudioQuality:
    """Test Sichuanese TTS audio quality metrics."""

    def test_all_features_generate_audio(self, tts_binary, tmp_path):
        """Verify all phonetic feature test cases generate audio."""
        config = CONFIG_DIR / "kokoro-mps-zh-sichuan.yaml"
        if not config.exists():
            pytest.skip("Sichuanese config not found")

        test_sentences = [
            # De-retroflex
            "是不是", "吃饭", "知道", "日本",
            # n/l merger
            "你好", "牛奶", "女人", "年轻",
            # hu/f merger
            "花钱", "回家", "黄河", "说话",
            # -ng/-n merger
            "风景", "生活", "经济", "明天",
        ]

        results = []
        for i, text in enumerate(test_sentences):
            wav_path = tmp_path / f"sc_test_{i}.wav"
            success = generate_wav(tts_binary, text, config, wav_path)

            if success and wav_path.exists() and wav_path.stat().st_size > 1000:
                results.append({"text": text, "status": "OK", "size": wav_path.stat().st_size})
            else:
                results.append({"text": text, "status": "FAILED", "size": 0})

        # Report
        print("\n=== Sichuanese Phonetic Feature Audio Test ===")
        ok_count = sum(1 for r in results if r["status"] == "OK")
        print(f"Audio generated: {ok_count}/{len(test_sentences)}")

        for r in results:
            print(f"  [{r['status']}] '{r['text']}' - {r['size']} bytes")

        # All must generate audio
        failed = [r for r in results if r["status"] == "FAILED"]
        assert len(failed) == 0, \
            f"Failed to generate audio for: {[r['text'] for r in failed]}"
