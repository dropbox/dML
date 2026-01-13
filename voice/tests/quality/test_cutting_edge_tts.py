"""
Cutting-Edge TTS Quality Tests (Arabic, Turkish, Persian)

Phase 1 validation tests for:
- Fish-Speech: Arabic TTS (#1 TTS-Arena2)
- Orpheus-TTS: Turkish TTS (3B purpose-built)
- MMS-TTS-FAS / ManaTTS: Persian TTS

These tests validate audio quality via LLM-as-Judge before Phase 2 (pure C++ conversion).
Per MANAGER directive: Pass 7/10 quality before proceeding to TorchScript export.

Usage:
    # Run all cutting-edge TTS tests
    pytest tests/quality/test_cutting_edge_tts.py -v

    # Run specific language tests
    pytest tests/quality/test_cutting_edge_tts.py -v -k "arabic"
    pytest tests/quality/test_cutting_edge_tts.py -v -k "turkish"
    pytest tests/quality/test_cutting_edge_tts.py -v -k "persian"

    # Run LLM-Judge tests only (requires OPENAI_API_KEY)
    pytest tests/quality/test_cutting_edge_tts.py -v -m llm_judge

Copyright 2025 Andrew Yates. All rights reserved.
"""

import base64
import io
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"

# Socket paths for cutting-edge TTS servers
FISH_SPEECH_SOCKET = Path("/tmp/fish_speech.sock")
ORPHEUS_TTS_SOCKET = Path("/tmp/orpheus_tts.sock")
PERSIAN_TTS_SOCKET = Path("/tmp/persian_tts.sock")

# LLM-Judge quality threshold per MANAGER Phase 1 directive
MIN_LLM_JUDGE_SCORE = 7  # 7/10 minimum to pass validation


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


# =============================================================================
# Socket Client Utilities
# =============================================================================

def send_socket_request(socket_path: Path, request: dict, timeout: float = 120.0) -> bytes:
    """Send a JSON request to Unix socket and receive WAV response.

    Protocol:
        Request: newline-terminated JSON
        Response: 4-byte little-endian length prefix + WAV bytes

    Args:
        socket_path: Path to Unix socket
        request: JSON-serializable request dict
        timeout: Socket timeout in seconds

    Returns:
        WAV bytes or raises exception
    """
    if not socket_path.exists():
        raise FileNotFoundError(f"Socket not found: {socket_path}")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(str(socket_path))

        # Send JSON request with newline
        request_bytes = (json.dumps(request) + "\n").encode("utf-8")
        sock.sendall(request_bytes)

        # Receive 4-byte length prefix
        length_data = b""
        while len(length_data) < 4:
            chunk = sock.recv(4 - len(length_data))
            if not chunk:
                break
            length_data += chunk

        if len(length_data) < 4:
            # Check if it's an error response
            remaining = sock.recv(4096)
            error_data = length_data + remaining
            try:
                error = json.loads(error_data.decode("utf-8"))
                raise RuntimeError(f"Server error: {error.get('error', 'unknown')}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Unexpected response: {error_data!r}")

        wav_length = struct.unpack("<I", length_data)[0]

        # Receive WAV data
        wav_data = b""
        while len(wav_data) < wav_length:
            chunk = sock.recv(min(65536, wav_length - len(wav_data)))
            if not chunk:
                break
            wav_data += chunk

        if len(wav_data) != wav_length:
            raise RuntimeError(f"Incomplete WAV data: {len(wav_data)} / {wav_length}")

        return wav_data

    finally:
        sock.close()


def check_socket_server(socket_path: Path) -> bool:
    """Check if socket server is running."""
    if not socket_path.exists():
        return False

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(str(socket_path))
        sock.close()
        return True
    except Exception:
        return False


# =============================================================================
# Audio Analysis Utilities
# =============================================================================

def read_wav_bytes(wav_data: bytes) -> Tuple[np.ndarray, int]:
    """Read WAV bytes and return (samples as float array, sample_rate)."""
    wav_io = io.BytesIO(wav_data)
    with wave.open(wav_io, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        # Convert to numpy array
        if sample_width == 2:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        # Convert stereo to mono if needed
        if channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

        return samples, sample_rate


def get_audio_metrics(samples: np.ndarray, sample_rate: int) -> dict:
    """Calculate audio quality metrics."""
    if len(samples) == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "duration_sec": 0.0,
            "silence_ratio": 1.0,
            "is_silent": True,
        }

    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))
    duration_sec = len(samples) / sample_rate

    # Silence detection (frames below threshold)
    silence_threshold = 0.01
    frame_size = 1024
    silent_frames = 0
    total_frames = len(samples) // frame_size

    for i in range(total_frames):
        frame = samples[i * frame_size:(i + 1) * frame_size]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms < silence_threshold:
            silent_frames += 1

    silence_ratio = silent_frames / max(total_frames, 1)

    return {
        "rms": rms,
        "peak": peak,
        "duration_sec": duration_sec,
        "silence_ratio": float(silence_ratio),
        "is_silent": rms < 0.01,
    }


# =============================================================================
# LLM-Judge Evaluation
# =============================================================================

def evaluate_audio_llm_judge(
    wav_data: bytes,
    expected_text: str,
    language: str,
    language_name: str
) -> dict:
    """Evaluate audio quality using GPT-audio LLM-as-Judge.

    Args:
        wav_data: WAV file bytes
        expected_text: Expected text content
        language: Language code (ar, tr, fa)
        language_name: Full language name for prompt

    Returns:
        Dict with: pronunciation (1-10), intonation (1-10), quality (1-10), overall (1-10)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")

    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("openai package not installed")

    audio_base64 = base64.b64encode(wav_data).decode("utf-8")

    # Language-specific prompts per roadmap
    prompts = {
        "ar": f"""Listen to this Arabic speech synthesis.
Rate the following aspects 1-10:
1. Arabic pronunciation accuracy
2. Natural intonation and rhythm
3. Overall quality and clarity

A score of 8+ means: "Native Arabic speakers would find this natural."
A score of 6-7 means: "Clearly Arabic but with minor issues."
Below 6 means: "Significant problems, not production ready."

Expected text: "{expected_text}"

Output JSON: {{"pronunciation": X, "intonation": X, "quality": X, "overall": X, "issues": "brief description or none"}}""",

        "tr": f"""Listen to this Turkish speech synthesis.
Rate the following aspects 1-10:
1. Turkish pronunciation accuracy (vowel harmony, consonant clarity)
2. Natural intonation and emotional expression
3. Overall quality and clarity

A score of 8+ means: "Native Turkish speakers would be impressed."
A score of 6-7 means: "Clearly Turkish but with minor issues."
Below 6 means: "Significant problems, not production ready."

Expected text: "{expected_text}"

Output JSON: {{"pronunciation": X, "intonation": X, "quality": X, "overall": X, "issues": "brief description or none"}}""",

        "fa": f"""Listen to this Persian (Farsi) speech synthesis.
Rate the following aspects 1-10:
1. Persian pronunciation accuracy (ezafe construction, consonant clusters)
2. Natural intonation and flow
3. Overall quality and clarity

A score of 8+ means: "Native Persian speakers would find this natural."
A score of 6-7 means: "Clearly Persian but with minor issues."
Below 6 means: "Significant problems, not production ready."

Expected text: "{expected_text}"

Output JSON: {{"pronunciation": X, "intonation": X, "quality": X, "overall": X, "issues": "brief description or none"}}""",
    }

    prompt = prompts.get(language, prompts["ar"])

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[
            {
                "role": "system",
                "content": f"You are an expert {language_name} TTS audio evaluator. Output ONLY valid JSON."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}
                ]
            }
        ],
        max_tokens=300
    )

    result_text = response.choices[0].message.content.strip()

    # Parse JSON from response
    try:
        # Handle markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(result_text[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return {
        "pronunciation": 0,
        "intonation": 0,
        "quality": 0,
        "overall": 0,
        "issues": f"Failed to parse response: {result_text[:200]}"
    }


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def fish_speech_available():
    """Check if Fish-Speech server is running."""
    if not check_socket_server(FISH_SPEECH_SOCKET):
        pytest.skip(
            f"Fish-Speech server not running. Start with:\n"
            f"  python {SCRIPTS_DIR}/fish_speech_server.py --socket {FISH_SPEECH_SOCKET}"
        )
    return True


@pytest.fixture(scope="module")
def orpheus_tts_available():
    """Check if Orpheus-TTS server is running."""
    if not check_socket_server(ORPHEUS_TTS_SOCKET):
        pytest.skip(
            f"Orpheus-TTS server not running. Start with:\n"
            f"  python {SCRIPTS_DIR}/orpheus_tts_server.py --socket {ORPHEUS_TTS_SOCKET}"
        )
    return True


@pytest.fixture(scope="module")
def persian_tts_available():
    """Check if Persian TTS server is running."""
    if not check_socket_server(PERSIAN_TTS_SOCKET):
        pytest.skip(
            f"Persian TTS server not running. Start with:\n"
            f"  python {SCRIPTS_DIR}/persian_tts_server.py --socket {PERSIAN_TTS_SOCKET}"
        )
    return True


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


# =============================================================================
# Arabic Tests (Fish-Speech)
# =============================================================================

class TestFishSpeechArabic:
    """Fish-Speech Arabic TTS tests (#1 TTS-Arena2)."""

    ARABIC_TEST_CASES = [
        ("hello", "مرحبا، كيف حالك؟", "Hello, how are you?"),
        ("greeting", "السلام عليكم ورحمة الله وبركاته", "Peace be upon you"),
        ("weather", "الطقس جميل اليوم والشمس مشرقة", "The weather is beautiful today"),
        ("technical", "تم تحديث النظام بنجاح", "System updated successfully"),
    ]

    def test_fish_speech_server_responds(self, fish_speech_available):
        """Verify Fish-Speech server accepts connections and responds."""
        request = {"text": "مرحبا", "language": "ar"}
        wav_data = send_socket_request(FISH_SPEECH_SOCKET, request)

        assert len(wav_data) > 1000, f"WAV data too small: {len(wav_data)} bytes"

        # Verify it's valid WAV
        samples, sample_rate = read_wav_bytes(wav_data)
        assert len(samples) > 0, "No audio samples in WAV"
        assert sample_rate in [22050, 24000, 44100, 48000], f"Unexpected sample rate: {sample_rate}"

    @pytest.mark.parametrize("name,arabic_text,english_meaning", ARABIC_TEST_CASES)
    def test_arabic_synthesis_basic(self, fish_speech_available, name, arabic_text, english_meaning):
        """Test basic Arabic synthesis produces non-silent audio."""
        request = {"text": arabic_text, "language": "ar"}

        start = time.time()
        wav_data = send_socket_request(FISH_SPEECH_SOCKET, request)
        latency = time.time() - start

        samples, sample_rate = read_wav_bytes(wav_data)
        metrics = get_audio_metrics(samples, sample_rate)

        print(f"\n[Arabic:{name}] {arabic_text[:30]}...")
        print(f"  Latency: {latency:.2f}s, Duration: {metrics['duration_sec']:.2f}s")
        print(f"  RMS: {metrics['rms']:.4f}, Peak: {metrics['peak']:.4f}")

        # Quality assertions
        assert not metrics["is_silent"], f"Audio is silent! RMS={metrics['rms']:.4f}"
        assert metrics["duration_sec"] >= 0.3, f"Audio too short: {metrics['duration_sec']:.2f}s"
        assert metrics["silence_ratio"] < 0.9, f"Too much silence: {metrics['silence_ratio']:.2f}"

    @pytest.mark.llm_judge
    def test_arabic_llm_judge_quality(self, fish_speech_available, openai_available):
        """LLM-Judge evaluation for Arabic synthesis quality."""
        arabic_text = "مرحبا، كيف حالك اليوم؟ أتمنى لك يوما سعيدا"

        request = {"text": arabic_text, "language": "ar"}
        wav_data = send_socket_request(FISH_SPEECH_SOCKET, request)

        result = evaluate_audio_llm_judge(wav_data, arabic_text, "ar", "Arabic")

        print(f"\n[Arabic LLM-Judge]")
        print(f"  Text: {arabic_text}")
        print(f"  Pronunciation: {result.get('pronunciation', 0)}/10")
        print(f"  Intonation: {result.get('intonation', 0)}/10")
        print(f"  Quality: {result.get('quality', 0)}/10")
        print(f"  Overall: {result.get('overall', 0)}/10")
        print(f"  Issues: {result.get('issues', 'N/A')}")

        overall = result.get("overall", 0)
        assert overall >= MIN_LLM_JUDGE_SCORE, (
            f"Arabic quality {overall}/10 below Phase 1 threshold {MIN_LLM_JUDGE_SCORE}/10. "
            f"Issues: {result.get('issues', 'unknown')}"
        )

    def test_arabic_rtf_benchmark(self, fish_speech_available):
        """Benchmark Arabic TTS RTF (Real-Time Factor)."""
        # Longer text for meaningful RTF measurement
        arabic_text = "هذا نص طويل لاختبار أداء نظام تحويل النص إلى كلام باللغة العربية. نريد التأكد من أن النظام يعمل بشكل سريع وفعال."

        request = {"text": arabic_text, "language": "ar"}

        start = time.time()
        wav_data = send_socket_request(FISH_SPEECH_SOCKET, request)
        latency = time.time() - start

        samples, sample_rate = read_wav_bytes(wav_data)
        duration = len(samples) / sample_rate

        rtf = latency / duration if duration > 0 else float('inf')

        print(f"\n[Arabic RTF Benchmark]")
        print(f"  Text length: {len(arabic_text)} chars")
        print(f"  Synthesis time: {latency:.2f}s")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  RTF: {rtf:.3f}")

        # Fish-Speech target: RTF < 1.0 (real-time)
        assert rtf < 1.0, f"Arabic RTF too high: {rtf:.3f} (target < 1.0)"


# =============================================================================
# Turkish Tests (Orpheus-TTS)
# =============================================================================

class TestOrpheusTTSTurkish:
    """Orpheus-TTS Turkish TTS tests (3B purpose-built)."""

    TURKISH_TEST_CASES = [
        ("hello", "Merhaba, nasılsınız?", "Hello, how are you?"),
        ("greeting", "Günaydın, bugün hava çok güzel", "Good morning, weather is nice"),
        ("technical", "Sistem başarıyla güncellendi", "System updated successfully"),
        ("emotion", "Bugün çok mutluyum!", "I am very happy today!"),
    ]

    EMOTION_TAGS = ["laugh", "sigh", "gasp"]

    def test_orpheus_server_responds(self, orpheus_tts_available):
        """Verify Orpheus-TTS server accepts connections and responds."""
        request = {"text": "Merhaba"}
        wav_data = send_socket_request(ORPHEUS_TTS_SOCKET, request)

        assert len(wav_data) > 1000, f"WAV data too small: {len(wav_data)} bytes"

        samples, sample_rate = read_wav_bytes(wav_data)
        assert len(samples) > 0, "No audio samples in WAV"

    @pytest.mark.parametrize("name,turkish_text,english_meaning", TURKISH_TEST_CASES)
    def test_turkish_synthesis_basic(self, orpheus_tts_available, name, turkish_text, english_meaning):
        """Test basic Turkish synthesis produces non-silent audio."""
        request = {"text": turkish_text}

        start = time.time()
        wav_data = send_socket_request(ORPHEUS_TTS_SOCKET, request)
        latency = time.time() - start

        samples, sample_rate = read_wav_bytes(wav_data)
        metrics = get_audio_metrics(samples, sample_rate)

        print(f"\n[Turkish:{name}] {turkish_text[:30]}...")
        print(f"  Latency: {latency:.2f}s, Duration: {metrics['duration_sec']:.2f}s")
        print(f"  RMS: {metrics['rms']:.4f}, Peak: {metrics['peak']:.4f}")

        assert not metrics["is_silent"], f"Audio is silent! RMS={metrics['rms']:.4f}"
        assert metrics["duration_sec"] >= 0.3, f"Audio too short: {metrics['duration_sec']:.2f}s"

    @pytest.mark.parametrize("emotion", EMOTION_TAGS)
    def test_turkish_emotion_tags(self, orpheus_tts_available, emotion):
        """Test Turkish synthesis with emotion tags (<laugh>, <sigh>, etc.)."""
        text = "Bu çok ilginç bir durum"
        request = {"text": text, "emotion": emotion}

        wav_data = send_socket_request(ORPHEUS_TTS_SOCKET, request)
        samples, sample_rate = read_wav_bytes(wav_data)
        metrics = get_audio_metrics(samples, sample_rate)

        print(f"\n[Turkish emotion:{emotion}]")
        print(f"  RMS: {metrics['rms']:.4f}, Duration: {metrics['duration_sec']:.2f}s")

        assert not metrics["is_silent"], f"Emotion audio is silent! RMS={metrics['rms']:.4f}"

    @pytest.mark.llm_judge
    def test_turkish_llm_judge_quality(self, orpheus_tts_available, openai_available):
        """LLM-Judge evaluation for Turkish synthesis quality."""
        turkish_text = "Merhaba, bugün hava çok güzel. Umarım iyi bir gün geçirirsiniz."

        request = {"text": turkish_text}
        wav_data = send_socket_request(ORPHEUS_TTS_SOCKET, request)

        result = evaluate_audio_llm_judge(wav_data, turkish_text, "tr", "Turkish")

        print(f"\n[Turkish LLM-Judge]")
        print(f"  Text: {turkish_text}")
        print(f"  Pronunciation: {result.get('pronunciation', 0)}/10")
        print(f"  Intonation: {result.get('intonation', 0)}/10")
        print(f"  Quality: {result.get('quality', 0)}/10")
        print(f"  Overall: {result.get('overall', 0)}/10")
        print(f"  Issues: {result.get('issues', 'N/A')}")

        overall = result.get("overall", 0)
        assert overall >= MIN_LLM_JUDGE_SCORE, (
            f"Turkish quality {overall}/10 below Phase 1 threshold {MIN_LLM_JUDGE_SCORE}/10. "
            f"Issues: {result.get('issues', 'unknown')}"
        )

    def test_turkish_rtf_benchmark(self, orpheus_tts_available):
        """Benchmark Turkish TTS RTF (Real-Time Factor)."""
        turkish_text = "Bu uzun bir metin, Türkçe metin-konuşma sisteminin performansını test etmek için kullanılıyor. Sistemin hızlı ve verimli çalıştığından emin olmak istiyoruz."

        request = {"text": turkish_text}

        start = time.time()
        wav_data = send_socket_request(ORPHEUS_TTS_SOCKET, request)
        latency = time.time() - start

        samples, sample_rate = read_wav_bytes(wav_data)
        duration = len(samples) / sample_rate

        rtf = latency / duration if duration > 0 else float('inf')

        print(f"\n[Turkish RTF Benchmark]")
        print(f"  Text length: {len(turkish_text)} chars")
        print(f"  Synthesis time: {latency:.2f}s")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  RTF: {rtf:.3f}")

        # Orpheus-TTS target: RTF < 1.0 (real-time)
        assert rtf < 1.0, f"Turkish RTF too high: {rtf:.3f} (target < 1.0)"


# =============================================================================
# Persian Tests (MMS-TTS-FAS / ManaTTS)
# =============================================================================

class TestPersianTTS:
    """Persian (Farsi) TTS tests (MMS-TTS-FAS / ManaTTS)."""

    PERSIAN_TEST_CASES = [
        ("hello", "سلام، حال شما چطور است؟", "Hello, how are you?"),
        ("greeting", "صبح بخیر، امروز هوا خیلی خوب است", "Good morning, weather is nice"),
        ("technical", "سیستم با موفقیت به‌روزرسانی شد", "System updated successfully"),
        ("poetry", "در این جهان هر کس به یاد خود است", "In this world everyone remembers themselves"),
    ]

    ENGINES = ["mms", "mana"]

    def test_persian_server_responds(self, persian_tts_available):
        """Verify Persian TTS server accepts connections and responds."""
        request = {"text": "سلام", "engine": "mms"}
        wav_data = send_socket_request(PERSIAN_TTS_SOCKET, request)

        assert len(wav_data) > 1000, f"WAV data too small: {len(wav_data)} bytes"

        samples, sample_rate = read_wav_bytes(wav_data)
        assert len(samples) > 0, "No audio samples in WAV"

    @pytest.mark.parametrize("name,persian_text,english_meaning", PERSIAN_TEST_CASES)
    def test_persian_synthesis_basic(self, persian_tts_available, name, persian_text, english_meaning):
        """Test basic Persian synthesis produces non-silent audio (MMS-TTS default)."""
        request = {"text": persian_text, "engine": "mms"}

        start = time.time()
        wav_data = send_socket_request(PERSIAN_TTS_SOCKET, request)
        latency = time.time() - start

        samples, sample_rate = read_wav_bytes(wav_data)
        metrics = get_audio_metrics(samples, sample_rate)

        print(f"\n[Persian:{name}] {persian_text[:30]}...")
        print(f"  Latency: {latency:.2f}s, Duration: {metrics['duration_sec']:.2f}s")
        print(f"  RMS: {metrics['rms']:.4f}, Peak: {metrics['peak']:.4f}")
        print(f"  Sample rate: {sample_rate}Hz (MMS-TTS outputs 22050Hz)")

        assert not metrics["is_silent"], f"Audio is silent! RMS={metrics['rms']:.4f}"
        assert metrics["duration_sec"] >= 0.3, f"Audio too short: {metrics['duration_sec']:.2f}s"

    @pytest.mark.llm_judge
    def test_persian_llm_judge_quality_mms(self, persian_tts_available, openai_available):
        """LLM-Judge evaluation for Persian MMS-TTS synthesis quality."""
        persian_text = "سلام، امروز هوا بسیار خوب است. امیدوارم روز خوبی داشته باشید."

        request = {"text": persian_text, "engine": "mms"}
        wav_data = send_socket_request(PERSIAN_TTS_SOCKET, request)

        result = evaluate_audio_llm_judge(wav_data, persian_text, "fa", "Persian")

        print(f"\n[Persian MMS-TTS LLM-Judge]")
        print(f"  Text: {persian_text}")
        print(f"  Pronunciation: {result.get('pronunciation', 0)}/10")
        print(f"  Intonation: {result.get('intonation', 0)}/10")
        print(f"  Quality: {result.get('quality', 0)}/10")
        print(f"  Overall: {result.get('overall', 0)}/10")
        print(f"  Issues: {result.get('issues', 'N/A')}")

        overall = result.get("overall", 0)
        assert overall >= MIN_LLM_JUDGE_SCORE, (
            f"Persian MMS-TTS quality {overall}/10 below Phase 1 threshold {MIN_LLM_JUDGE_SCORE}/10. "
            f"Issues: {result.get('issues', 'unknown')}"
        )

    def test_persian_rtf_benchmark(self, persian_tts_available):
        """Benchmark Persian TTS RTF (Real-Time Factor)."""
        persian_text = "این یک متن طولانی برای آزمایش عملکرد سیستم تبدیل متن به گفتار فارسی است. می‌خواهیم مطمئن شویم که سیستم سریع و کارآمد کار می‌کند."

        request = {"text": persian_text, "engine": "mms"}

        start = time.time()
        wav_data = send_socket_request(PERSIAN_TTS_SOCKET, request)
        latency = time.time() - start

        samples, sample_rate = read_wav_bytes(wav_data)
        duration = len(samples) / sample_rate

        rtf = latency / duration if duration > 0 else float('inf')

        print(f"\n[Persian RTF Benchmark]")
        print(f"  Text length: {len(persian_text)} chars")
        print(f"  Synthesis time: {latency:.2f}s")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  RTF: {rtf:.3f}")

        # MMS-TTS target: RTF < 1.0 (real-time)
        assert rtf < 1.0, f"Persian RTF too high: {rtf:.3f} (target < 1.0)"


# =============================================================================
# Cross-Language Integration Tests
# =============================================================================

class TestCuttingEdgeIntegration:
    """Integration tests across all cutting-edge TTS engines."""

    @pytest.mark.llm_judge
    def test_all_languages_pass_quality_threshold(
        self,
        fish_speech_available,
        orpheus_tts_available,
        persian_tts_available,
        openai_available
    ):
        """Validate all three languages pass LLM-Judge quality threshold."""
        results = {}

        # Arabic
        arabic_text = "مرحبا، كيف حالك؟"
        wav_ar = send_socket_request(FISH_SPEECH_SOCKET, {"text": arabic_text, "language": "ar"})
        results["Arabic"] = evaluate_audio_llm_judge(wav_ar, arabic_text, "ar", "Arabic")

        # Turkish
        turkish_text = "Merhaba, nasılsınız?"
        wav_tr = send_socket_request(ORPHEUS_TTS_SOCKET, {"text": turkish_text})
        results["Turkish"] = evaluate_audio_llm_judge(wav_tr, turkish_text, "tr", "Turkish")

        # Persian
        persian_text = "سلام، حال شما چطور است؟"
        wav_fa = send_socket_request(PERSIAN_TTS_SOCKET, {"text": persian_text, "engine": "mms"})
        results["Persian"] = evaluate_audio_llm_judge(wav_fa, persian_text, "fa", "Persian")

        print("\n" + "=" * 60)
        print("CUTTING-EDGE TTS QUALITY VALIDATION")
        print("=" * 60)

        all_pass = True
        for lang, result in results.items():
            overall = result.get("overall", 0)
            status = "PASS" if overall >= MIN_LLM_JUDGE_SCORE else "FAIL"
            if overall < MIN_LLM_JUDGE_SCORE:
                all_pass = False
            print(f"{lang}: {overall}/10 [{status}] - Issues: {result.get('issues', 'none')}")

        print("=" * 60)
        print(f"Threshold: {MIN_LLM_JUDGE_SCORE}/10 (Phase 1 validation)")
        print("=" * 60)

        assert all_pass, "One or more languages failed quality validation"


# =============================================================================
# Audio Sample Generation (for manual review)
# =============================================================================

class TestGenerateAudioSamples:
    """Generate audio samples for manual review and archiving."""

    OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "cutting_edge_samples"

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Ensure output directory exists."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_generate_arabic_samples(self, fish_speech_available):
        """Generate Arabic audio samples for review."""
        samples = [
            ("greeting", "مرحبا، كيف حالك اليوم؟"),
            ("technical", "تم إصلاح الخطأ في الشفرة البرمجية"),
            ("emotional", "أنا سعيد جدا بهذه النتيجة الرائعة"),
        ]

        for name, text in samples:
            request = {"text": text, "language": "ar"}
            wav_data = send_socket_request(FISH_SPEECH_SOCKET, request)

            output_path = self.OUTPUT_DIR / f"arabic_{name}.wav"
            output_path.write_bytes(wav_data)
            print(f"Saved: {output_path}")

    def test_generate_turkish_samples(self, orpheus_tts_available):
        """Generate Turkish audio samples for review."""
        samples = [
            ("greeting", "Merhaba, bugün nasılsınız?"),
            ("technical", "Kod hatası başarıyla düzeltildi"),
            ("emotional_laugh", "Bu çok komik!", "laugh"),
        ]

        for item in samples:
            name = item[0]
            text = item[1]
            emotion = item[2] if len(item) > 2 else None

            request = {"text": text}
            if emotion:
                request["emotion"] = emotion

            wav_data = send_socket_request(ORPHEUS_TTS_SOCKET, request)

            output_path = self.OUTPUT_DIR / f"turkish_{name}.wav"
            output_path.write_bytes(wav_data)
            print(f"Saved: {output_path}")

    def test_generate_persian_samples(self, persian_tts_available):
        """Generate Persian audio samples for review."""
        samples = [
            ("greeting", "سلام، امروز چطورید؟"),
            ("technical", "خطای برنامه‌نویسی با موفقیت اصلاح شد"),
            ("poetry", "دل به دست آور که حج اکبر است"),
        ]

        for name, text in samples:
            request = {"text": text, "engine": "mms"}
            wav_data = send_socket_request(PERSIAN_TTS_SOCKET, request)

            output_path = self.OUTPUT_DIR / f"persian_{name}.wav"
            output_path.write_bytes(wav_data)
            print(f"Saved: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
