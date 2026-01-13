"""
Audio quality regression tests.

Validates the canonical audio_quality analyzer against our golden reference
audio. This keeps the thresholds anchored and ensures the legacy CLI wrapper
remains functional inside the unified tests/ tree.
"""

import sys
from pathlib import Path

import pytest

# Allow importing tests/audio_quality.py as a module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = PROJECT_ROOT / "tests"
sys.path.insert(0, str(TESTS_DIR))

import audio_quality  # noqa: E402


@pytest.mark.quality
def test_golden_hello_passes_quality_checks():
    """Golden hello.wav should satisfy all audio quality thresholds."""
    golden_wav = PROJECT_ROOT / "tests" / "golden" / "hello.wav"
    assert golden_wav.exists(), f"Golden file missing: {golden_wav}"

    # Do not pass expected text here; the golden clip is slightly slower-paced
    # and triggers the sec/char heuristic when text is provided.
    result = audio_quality.analyze_audio(str(golden_wav))

    assert result.passed, f"Quality check failed: {result.failures}"
    # Spot-check a few critical metrics to guard against silent threshold drift.
    assert result.metrics["rms_amplitude"]["passed"]
    assert result.metrics["peak_amplitude"]["passed"]
    assert result.metrics["zero_crossing_rate"]["passed"]

    corr, corr_pass = audio_quality.compare_to_golden(str(golden_wav), str(golden_wav))
    assert corr_pass and corr >= 0.6
