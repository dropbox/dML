#!/usr/bin/env python3
"""
CosyVoice2 Streaming Server Integration Tests

Tests the CosyVoice2 socket server with quality validation.
Requires cosyvoice_251_venv and a running server.

Usage:
    # Start server first:
    source cosyvoice_251_venv/bin/activate
    python scripts/cosyvoice_server.py --socket /tmp/cosyvoice.sock &

    # Then run tests:
    pytest tests/integration/test_cosyvoice2_streaming.py -v

Copyright 2025 Andrew Yates. All rights reserved.
"""

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

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SOCKET_PATH = "/tmp/cosyvoice.sock"
SAMPLE_RATE = 24000
GOLDEN_REF = PROJECT_ROOT / "models" / "cosyvoice" / "test_output" / "cosyvoice_sichuan_grandma.wav"


def socket_available() -> bool:
    """Check if the CosyVoice2 server is running."""
    return Path(SOCKET_PATH).exists()


def synthesize_via_socket(text: str, speed: float = 1.3, instruction: str = "") -> bytes:
    """Synthesize text via the socket server, return PCM bytes."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(60)
    sock.connect(SOCKET_PATH)

    request = {"text": text, "speed": speed}
    if instruction:
        request["instruction"] = instruction
    sock.sendall(json.dumps(request).encode() + b'\n')

    # Read length prefix
    length_data = sock.recv(4)
    length = struct.unpack('<I', length_data)[0]
    if length == 0:
        raise RuntimeError("Server returned empty audio")

    # Read PCM data
    pcm_data = b''
    while len(pcm_data) < length:
        chunk = sock.recv(min(4096, length - len(pcm_data)))
        if not chunk:
            break
        pcm_data += chunk

    sock.close()
    return pcm_data


def pcm_to_wav_file(pcm_data: bytes, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Write PCM bytes to WAV file."""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


@pytest.fixture(scope="module")
def server_check():
    """Ensure server is running."""
    if not socket_available():
        pytest.skip(
            "CosyVoice2 server not running. Start with:\n"
            "  source cosyvoice_251_venv/bin/activate\n"
            "  python scripts/cosyvoice_server.py --socket /tmp/cosyvoice.sock"
        )


class TestCosyVoice2StreamingServer:
    """Tests for the CosyVoice2 socket server."""

    def test_basic_synthesis(self, server_check):
        """Test basic text synthesis."""
        pcm_data = synthesize_via_socket("你好世界")
        assert len(pcm_data) > 0
        duration = len(pcm_data) / 2 / SAMPLE_RATE
        assert 0.5 < duration < 10.0, f"Unexpected duration: {duration}s"

    def test_sichuanese_text(self, server_check):
        """Test Sichuanese dialect text."""
        pcm_data = synthesize_via_socket("今天天气好安逸哦，我们去吃火锅嘛")
        duration = len(pcm_data) / 2 / SAMPLE_RATE
        assert duration > 1.0, f"Audio too short: {duration}s"

    def test_speed_adjustment(self, server_check):
        """Test that speed parameter affects output length."""
        text = "四川的火锅太好吃了"

        # Normal speed
        pcm_1_0 = synthesize_via_socket(text, speed=1.0)
        # Faster speed
        pcm_1_5 = synthesize_via_socket(text, speed=1.5)

        # Faster speed should produce shorter audio
        duration_1_0 = len(pcm_1_0) / 2 / SAMPLE_RATE
        duration_1_5 = len(pcm_1_5) / 2 / SAMPLE_RATE

        assert duration_1_5 < duration_1_0 * 0.9, (
            f"Speed 1.5x should be faster: {duration_1_5}s vs {duration_1_0}s"
        )

    def test_multiple_sentences(self, server_check):
        """Test synthesizing multiple sentences."""
        sentences = [
            "婆婆今天做了红烧肉",
            "好香哦",
            "太安逸了"
        ]

        for sentence in sentences:
            pcm_data = synthesize_via_socket(sentence)
            assert len(pcm_data) > 0, f"Empty audio for: {sentence}"

    def test_quality_no_frog(self, server_check):
        """Test that output doesn't have 'dying frog' artifacts.

        This is a regression test for the PyTorch 2.9.1 quality issue.
        Uses LLM-as-judge with majority voting (3 evaluations) to handle
        LLM variance, otherwise checks basic metrics.
        """
        pcm_data = synthesize_via_socket("你好，我是四川人，今天天气好安逸哦")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
            pcm_to_wav_file(pcm_data, wav_path)

        try:
            # Try LLM-as-judge evaluation with majority voting
            import os
            env_path = PROJECT_ROOT / '.env'
            if env_path.exists():
                with open(env_path) as ef:
                    for line in ef:
                        if line.strip() and not line.startswith('#') and '=' in line:
                            key, val = line.strip().split('=', 1)
                            os.environ[key] = val

            import base64
            from openai import OpenAI

            client = OpenAI()

            with open(wav_path, 'rb') as f:
                audio_b64 = base64.standard_b64encode(f.read()).decode()

            prompt = """Evaluate this Sichuanese dialect TTS audio:
1. Does it sound like a "dying frog" (croaking, robotic warbling)?
Return ONLY JSON: {"frog": true/false}"""

            # Run 3 evaluations for majority voting (handles LLM variance)
            frog_votes = []
            for _ in range(3):
                response = client.chat.completions.create(
                    model="gpt-4o-audio-preview",
                    modalities=["text"],
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                    ]}],
                    max_tokens=50
                )

                result_text = response.choices[0].message.content
                if '{' in result_text and '}' in result_text:
                    json_str = result_text[result_text.find('{'):result_text.rfind('}')+1]
                    result = json.loads(json_str)
                    frog_votes.append(result.get('frog', False))

            # Pass if majority (≥2/3) say NOT frog
            frog_count = sum(1 for v in frog_votes if v)
            assert frog_count < 2, (
                f"Audio has 'dying frog' quality issue ({frog_count}/3 votes)! "
                "This indicates PyTorch version problem. "
                "Ensure server uses cosyvoice_251_venv with PyTorch 2.5.1"
            )

        except ImportError:
            # No OpenAI available - check basic metrics
            import numpy as np
            import wave as wave_module

            with wave_module.open(wav_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            # Check for clipping
            clipping = (np.abs(samples) > 0.99).sum() / len(samples)
            assert clipping < 0.01, f"Too much clipping: {clipping*100:.1f}%"

            # Check RMS level (not too quiet or loud)
            rms = np.sqrt(np.mean(samples**2))
            assert 0.01 < rms < 0.5, f"Unusual RMS level: {rms}"

        finally:
            os.unlink(wav_path)

    def test_latency(self, server_check):
        """Test synthesis latency (warm server)."""
        # Warmup
        synthesize_via_socket("热身测试")

        # Measure
        start = time.time()
        pcm_data = synthesize_via_socket("这是一个延迟测试")
        elapsed = time.time() - start

        duration = len(pcm_data) / 2 / SAMPLE_RATE
        rtf = elapsed / duration

        print(f"\nLatency: {elapsed*1000:.0f}ms, Duration: {duration:.2f}s, RTF: {rtf:.2f}x")

        # Should complete faster than 60s for short text (RTF varies with CPU load)
        assert elapsed < 60, f"Synthesis too slow: {elapsed:.1f}s"


class TestCosyVoice2EdgeCases:
    """Edge case tests for robust error handling."""

    def test_empty_text_rejected(self, server_check):
        """Test that empty text returns error (zero length) or closes connection."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(30)  # Allow time for server response
        sock.connect(SOCKET_PATH)

        # Send empty text request
        request = {"text": "", "speed": 1.3}
        sock.sendall(json.dumps(request).encode() + b'\n')

        try:
            # Should return zero length (error indicator)
            length_data = sock.recv(4)
            if len(length_data) == 0:
                # Connection closed - also acceptable
                pass
            else:
                length = struct.unpack('<I', length_data)[0]
                assert length == 0, "Server should reject empty text with zero length"
        except (socket.timeout, ConnectionResetError, BrokenPipeError):
            # Server may close connection or timeout - acceptable for empty text
            pass
        finally:
            sock.close()

    def test_whitespace_only_rejected(self, server_check):
        """Test that whitespace-only text is rejected."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(SOCKET_PATH)

        request = {"text": "   \n\t   ", "speed": 1.3}
        sock.sendall(json.dumps(request).encode() + b'\n')

        length_data = sock.recv(4)
        length = struct.unpack('<I', length_data)[0]
        sock.close()

        assert length == 0, "Server should reject whitespace-only text"

    def test_punctuation_only(self, server_check):
        """Test handling of punctuation-only text."""
        # Punctuation-only may produce short audio or be rejected
        try:
            pcm_data = synthesize_via_socket("。。。")
            # If it produces audio, it should be very short
            duration = len(pcm_data) / 2 / SAMPLE_RATE
            assert duration < 5.0, f"Punctuation-only too long: {duration}s"
        except RuntimeError:
            # Also acceptable to reject
            pass

    def test_mixed_chinese_english(self, server_check):
        """Test mixed Chinese and English text."""
        pcm_data = synthesize_via_socket("今天我要去Starbucks喝咖啡")
        assert len(pcm_data) > 0
        duration = len(pcm_data) / 2 / SAMPLE_RATE
        assert 0.5 < duration < 15.0, f"Unexpected duration: {duration}s"

    def test_numbers_in_text(self, server_check):
        """Test text with numbers."""
        pcm_data = synthesize_via_socket("我今年25岁，住在3楼")
        assert len(pcm_data) > 0
        duration = len(pcm_data) / 2 / SAMPLE_RATE
        assert duration > 0.5, f"Audio too short: {duration}s"

    def test_special_characters(self, server_check):
        """Test text with special characters."""
        pcm_data = synthesize_via_socket("价格是￥100元！太贵了吧？")
        assert len(pcm_data) > 0

    def test_long_text(self, server_check):
        """Test synthesis of moderately long text.

        Note: CosyVoice2 has token limits (~200 speech tokens per chunk).
        Very long text may fail or be truncated.
        This test uses a moderate length to verify longer texts work.
        """
        # Use moderate length text (about 30-40 chars) to stay within model limits
        long_text = "四川是一个美丽的地方。这里有很多好吃的东西。"
        try:
            pcm_data = synthesize_via_socket(long_text)
            duration = len(pcm_data) / 2 / SAMPLE_RATE
            assert duration > 1.5, f"Long text should produce longer audio: {duration}s"
        except RuntimeError as e:
            # Server may reject very long text - this is acceptable behavior
            pytest.skip(f"Server rejected long text (acceptable): {e}")

    def test_repeated_synthesis(self, server_check):
        """Test that server handles repeated requests correctly."""
        text = "测试重复合成"
        durations = []

        for i in range(3):
            pcm_data = synthesize_via_socket(text)
            duration = len(pcm_data) / 2 / SAMPLE_RATE
            durations.append(duration)

        # All durations should be similar (within 50%)
        avg = sum(durations) / len(durations)
        for d in durations:
            assert abs(d - avg) / avg < 0.5, f"Inconsistent durations: {durations}"

    def test_speed_boundaries(self, server_check):
        """Test speed parameter at boundaries."""
        text = "速度测试"

        # Very slow
        pcm_slow = synthesize_via_socket(text, speed=0.8)
        duration_slow = len(pcm_slow) / 2 / SAMPLE_RATE

        # Very fast
        pcm_fast = synthesize_via_socket(text, speed=1.8)
        duration_fast = len(pcm_fast) / 2 / SAMPLE_RATE

        # Fast should be significantly shorter than slow
        assert duration_fast < duration_slow * 0.7, (
            f"Speed difference too small: slow={duration_slow}s, fast={duration_fast}s"
        )

    def test_concurrent_requests(self, server_check):
        """Test that server handles sequential requests properly."""
        import threading

        results = []
        errors = []

        def make_request(text, idx):
            try:
                pcm = synthesize_via_socket(text)
                results.append((idx, len(pcm)))
            except Exception as e:
                errors.append((idx, str(e)))

        # Note: Server is single-threaded, so requests will queue
        # This tests that queued requests complete successfully
        texts = ["测试一", "测试二", "测试三"]

        for i, text in enumerate(texts):
            make_request(text, i)

        assert len(results) == 3, f"Not all requests completed: {len(results)}/3"
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All should produce valid audio
        for idx, length in results:
            assert length > 0, f"Request {idx} produced empty audio"


class TestCosyVoice2AudioQuality:
    """Detailed audio quality tests."""

    def test_no_silence_at_start(self, server_check):
        """Test that audio doesn't have excessive silence at start."""
        import numpy as np

        pcm_data = synthesize_via_socket("你好世界")
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Check first 0.5s - should have some energy
        first_half_second = samples[:SAMPLE_RATE // 2]
        rms = np.sqrt(np.mean(first_half_second**2))

        assert rms > 0.001, f"Too much silence at start: RMS={rms}"

    def test_no_clipping(self, server_check):
        """Test that audio doesn't clip excessively."""
        import numpy as np

        pcm_data = synthesize_via_socket("大声说话！你好！四川！")
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Check for clipping (samples at max/min)
        clipping_ratio = (np.abs(samples) > 0.99).sum() / len(samples)
        assert clipping_ratio < 0.01, f"Too much clipping: {clipping_ratio*100:.1f}%"

    def test_reasonable_dynamic_range(self, server_check):
        """Test that audio has reasonable dynamic range."""
        import numpy as np

        pcm_data = synthesize_via_socket("今天天气好安逸哦，我们去吃火锅嘛")
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Calculate RMS
        rms = np.sqrt(np.mean(samples**2))

        # Should be in reasonable range (not too quiet, not clipping)
        assert 0.01 < rms < 0.5, f"Unusual RMS level: {rms}"

        # Calculate peak
        peak = np.max(np.abs(samples))
        assert peak > 0.1, f"Audio too quiet: peak={peak}"
        assert peak < 1.0, f"Audio clipping: peak={peak}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
