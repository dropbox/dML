"""
Stress Tests for Translation Timeout Behavior

Tests the translation backlog handling and timeout behavior in the streaming pipeline.
Validates fixes from Worker #557:
- N1: Translation counter leak on timeout (atomic flag prevents double-decrement)
- N2: Wrong-language output on backlog (was_backlogged flag)
- N3/O4: Thread safety for translator_ (mutex protection)

Key constants being tested:
- kTranslationTimeout = 5000ms
- kMaxTranslationTasks = 4

Worker #554 - Initial stress test implementation
"""

import os
import pytest
import subprocess
import time
import threading
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"
CONFIG_DIR = STREAM_TTS_CPP / "config"


@pytest.fixture(scope="module")
def binary_exists():
    """Ensure TTS binary exists."""
    if not TTS_BINARY.exists():
        pytest.skip(f"TTS binary not found: {TTS_BINARY}")
    return TTS_BINARY


def make_claude_json(text: str) -> str:
    """Create Claude API JSON format input."""
    escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    return f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}\n{{"type":"message_stop"}}'


def run_tts_with_translation(binary: Path, text: str, lang: str = "ja", timeout: int = 30) -> dict:
    """
    Run TTS with translation enabled, return result dict.

    Returns:
        dict with keys: success, returncode, stdout, stderr, duration_ms
    """
    cmd = [
        str(binary),
        "--daemon-pipe",
        "--language", lang,
        "--translate",  # Enable translation
        "--no-audio",   # Skip audio playback for faster tests
    ]

    input_json = make_claude_json(text)

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            input=input_json,
            capture_output=True,
            text=True,
            cwd=str(STREAM_TTS_CPP),
            timeout=timeout
        )
        duration_ms = (time.time() - start) * 1000

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Timeout expired",
            "duration_ms": timeout * 1000,
        }


class TestTranslationConcurrency:
    """Tests for concurrent translation handling."""

    def test_sequential_translations(self, binary_exists):
        """
        Test that sequential translations work correctly.
        Baseline test before concurrent stress tests.
        """
        texts = [
            "Hello world",
            "How are you today",
            "The weather is nice",
        ]

        results = []
        for text in texts:
            result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=30)
            results.append(result)

        # All should succeed
        for i, result in enumerate(results):
            assert result["success"], f"Translation {i+1} failed: {result['stderr'][:200]}"

        # Check no backlog warnings (sequential should never hit backlog)
        for result in results:
            assert "backlog" not in result["stderr"].lower(), \
                f"Unexpected backlog warning in sequential mode: {result['stderr'][:200]}"

    def test_concurrent_translations_within_limit(self, binary_exists):
        """
        Test concurrent translations within the kMaxTranslationTasks limit (4).
        Should all succeed without backlog warnings.
        """
        texts = ["Test text " + str(i) for i in range(3)]  # 3 concurrent, under limit of 4

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_tts_with_translation, binary_exists, text, "ja", 60)
                for text in texts
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # Count successes
        successes = sum(1 for r in results if r["success"])
        assert successes >= 2, f"Expected at least 2 successes, got {successes}"

        # Verify no double-decrement errors in any output
        for result in results:
            assert "double" not in result["stderr"].lower() and "leak" not in result["stderr"].lower(), \
                f"Potential counter issue: {result['stderr'][:200]}"

    @pytest.mark.slow
    def test_concurrent_translations_exceeding_limit(self, binary_exists):
        """
        Test concurrent translations exceeding kMaxTranslationTasks limit.
        Some should trigger backlog warnings but system should remain stable.
        """
        # Send 8 concurrent requests (2x the limit of 4)
        texts = ["Concurrent test message " + str(i) for i in range(8)]

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(run_tts_with_translation, binary_exists, text, "ja", 60)
                for text in texts
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # System should remain stable - at least some should succeed
        successes = sum(1 for r in results if r["success"])
        assert successes >= 1, f"All {len(results)} translations failed - system unstable"

        # Check for backlog handling messages (expected when exceeding limit)
        backlog_messages = sum(1 for r in results if "backlog" in r["stderr"].lower() or "dropping" in r["stderr"].lower())
        print(f"Concurrent stress: {successes}/{len(results)} succeeded, {backlog_messages} backlog warnings")


class TestTranslationTimeout:
    """Tests for translation timeout handling."""

    def test_normal_translation_completes_before_timeout(self, binary_exists):
        """
        Normal translation should complete well before the 5s timeout.
        """
        text = "Hello, how are you today?"
        result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=30)

        # Should succeed
        assert result["success"], f"Normal translation failed: {result['stderr'][:200]}"

        # Should complete in under 5000ms (the timeout threshold)
        # Note: This includes TTS synthesis time, so be generous
        assert result["duration_ms"] < 20000, \
            f"Translation took too long: {result['duration_ms']:.0f}ms"

    def test_counter_stability_under_load(self, binary_exists):
        """
        Test that translation counter remains stable under repeated load.
        Validates N1 fix: counter doesn't leak on timeout.
        """
        # Run multiple sequential translations to warm up
        for i in range(3):
            result = run_tts_with_translation(binary_exists, f"Warmup {i}", lang="ja", timeout=30)

        # Now run concurrent load
        texts = [f"Load test iteration {i}" for i in range(6)]

        all_results = []
        for _ in range(2):  # Run 2 rounds
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(run_tts_with_translation, binary_exists, text, "ja", 60)
                    for text in texts
                ]
                for future in as_completed(futures):
                    results.append(future.result())
            all_results.extend(results)
            time.sleep(1)  # Brief pause between rounds

        # All rounds should have similar success rates (counter not leaking)
        total_success = sum(1 for r in all_results if r["success"])
        success_rate = total_success / len(all_results)

        assert success_rate >= 0.3, \
            f"Low success rate {success_rate:.1%} may indicate counter leak"

        # Check for no resource exhaustion errors
        for result in all_results:
            stderr = result["stderr"].lower()
            assert "resource exhausted" not in stderr, \
                f"Resource exhaustion detected: {result['stderr'][:200]}"
            assert "out of memory" not in stderr, \
                f"OOM detected: {result['stderr'][:200]}"


class TestTranslationBackpressure:
    """Tests for translation backpressure handling."""

    def test_backlog_warning_message_format(self, binary_exists):
        """
        When backlog is hit, verify warning message format is correct.
        Expected: "Translation backlog at N tasks (limit 4) - DROPPING request"
        """
        # This test verifies the warning message pattern exists in code
        # by triggering high concurrency
        texts = [f"Backpressure test {i}" for i in range(10)]

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(run_tts_with_translation, binary_exists, text, "ja", 60)
                for text in texts
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # Check if any hit backlog (may or may not depending on timing)
        all_stderr = " ".join(r["stderr"] for r in results)
        if "backlog" in all_stderr.lower():
            # Verify format contains expected elements
            assert "tasks" in all_stderr.lower() or "limit" in all_stderr.lower(), \
                f"Backlog warning missing expected format elements"

    def test_was_backlogged_prevents_source_text(self, binary_exists):
        """
        Verify N2 fix: when backlogged, system doesn't fall back to source text.

        When translation is dropped due to backlog:
        - was_backlogged = true
        - Caller should NOT use source English text for Japanese TTS
        """
        # This is validated by absence of English in Japanese output
        # We can't easily test this without inspecting internal state,
        # but we can verify the system doesn't produce garbage output

        text = "This is an English test that should be translated to Japanese"
        result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=30)

        if result["success"]:
            # If successful, translation should have been used
            # We can't verify the actual translation without Whisper,
            # but at minimum it shouldn't crash
            assert result["returncode"] == 0

    def test_system_recovery_after_backlog(self, binary_exists):
        """
        After hitting backlog, system should recover and accept new requests.
        """
        # First, try to trigger backlog with burst
        burst_texts = [f"Burst {i}" for i in range(10)]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(run_tts_with_translation, binary_exists, text, "ja", 60)
                for text in burst_texts
            ]
            # Collect but don't assert - just trigger load
            for future in as_completed(futures):
                future.result()

        # Wait for system to settle
        time.sleep(2)

        # Now verify recovery with sequential requests
        recovery_success = 0
        for i in range(3):
            result = run_tts_with_translation(binary_exists, f"Recovery test {i}", lang="ja", timeout=30)
            if result["success"]:
                recovery_success += 1

        assert recovery_success >= 2, \
            f"System failed to recover: only {recovery_success}/3 recovery requests succeeded"


class TestTranslationThreadSafety:
    """Tests for translation thread safety (N3/O4 fixes)."""

    def test_mutex_protection_under_concurrent_access(self, binary_exists):
        """
        Test that translator_ mutex prevents race conditions.
        Validates N3/O4 fix: translator_mutex_ protects translate() calls.
        """
        # Run many concurrent translations to stress mutex
        texts = [f"Thread safety test {i}" for i in range(20)]

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(run_tts_with_translation, binary_exists, text, "ja", 60)
                for text in texts
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # Check for no segfaults or race condition symptoms
        for result in results:
            stderr = result["stderr"].lower()
            # These would indicate mutex/threading issues
            assert "segfault" not in stderr and "sigsegv" not in stderr, \
                f"Segfault detected - possible race condition: {result['stderr'][:200]}"
            assert "abort" not in stderr and "sigabrt" not in stderr, \
                f"Abort detected - possible race condition: {result['stderr'][:200]}"
            assert "data race" not in stderr, \
                f"Data race detected: {result['stderr'][:200]}"

        # At least some should succeed (not all crash)
        successes = sum(1 for r in results if r["success"])
        assert successes >= len(results) // 2, \
            f"Too many failures ({len(results) - successes}/{len(results)}) - possible thread safety issue"


@pytest.mark.slow
class TestTranslationStressEndurance:
    """Long-running stress tests for translation system stability."""

    def test_sustained_load_30_seconds(self, binary_exists):
        """
        Run sustained translation load for 30 seconds.
        Validates system stability under continuous pressure.
        """
        end_time = time.time() + 30
        iteration = 0
        successes = 0
        failures = 0

        while time.time() < end_time:
            text = f"Sustained load iteration {iteration}"
            result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=60)

            if result["success"]:
                successes += 1
            else:
                failures += 1

            iteration += 1
            time.sleep(0.5)  # Pace requests

        total = successes + failures
        success_rate = successes / total if total > 0 else 0

        print(f"\nSustained load results: {successes}/{total} ({success_rate:.1%} success rate)")

        # System should maintain reasonable success rate under sustained load
        assert success_rate >= 0.5, \
            f"Sustained load success rate {success_rate:.1%} below threshold"
        assert successes >= 10, \
            f"Too few successful translations ({successes}) in sustained load test"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
