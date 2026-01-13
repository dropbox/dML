#!/usr/bin/env python3
"""
Model instance pooling measurement.

Runs the C++ test_model_pooling utility to load multiple Kokoro TorchScript
instances in a single process, then performs concurrent synthesis to ensure
multiple models can reside on MPS simultaneously.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

STREAM_TTS_CPP = Path(__file__).parent.parent.parent / "stream-tts-cpp"
BINARY_PATH = STREAM_TTS_CPP / "build" / "test_model_pooling"


def parse_pooling_output(output: str) -> dict:
    """Extract JSON summary from test_model_pooling output."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("MODEL_POOLING_RESULT"):
            json_part = line.split("MODEL_POOLING_RESULT", 1)[1].strip()
            return json.loads(json_part)
    raise ValueError("MODEL_POOLING_RESULT line not found")


def test_model_pooling_memory_and_concurrency():
    """Verify per-instance memory cost and concurrent synthesis succeeds."""
    if not BINARY_PATH.exists():
        pytest.skip(f"Binary not found: {BINARY_PATH}")

    env = os.environ.copy()
    # Note: PYTORCH_ENABLE_MPS_FALLBACK no longer needed with PyTorch 2.9.1+
    env.setdefault("SPDLOG_LEVEL", "warn")

    cmd = [
        str(BINARY_PATH),
        "--instances",
        "2",
        "--json",
        "--sequential",
        "--text",
        "Parallel synthesis pooling benchmark.",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(STREAM_TTS_CPP),
        env=env,
    )

    assert result.returncode == 0, f"test_model_pooling failed: {result.stderr[-400:]}"

    try:
        metrics = parse_pooling_output(result.stdout)
    except ValueError as exc:
        pytest.fail(f"{exc}\nstdout tail:\n{result.stdout[-400:]}\nstderr tail:\n{result.stderr[-400:]}")

    assert metrics["instances"] == 2, metrics
    assert metrics["per_instance_mb"] > 100, metrics
    assert metrics["per_instance_mb"] < 3000, metrics
    assert metrics.get("inference_mode") == "sequential", metrics
    assert metrics.get("concurrent_success", False) is True, metrics

    synth_ms = metrics.get("synthesis_ms", [])
    assert len(synth_ms) == 2, metrics
    assert all(ms > 0 for ms in synth_ms), metrics
    assert max(synth_ms) < 5000, metrics


def test_model_pooling_concurrent_mode():
    """Verify concurrent inference works with MPS mutex (Worker #242 fix).

    This test verifies that the global MPS inference mutex properly serializes
    concurrent forward() calls, preventing the Metal command buffer assertion:
    "failed assertion _status < MTLCommandBufferStatusCommitted"
    """
    if not BINARY_PATH.exists():
        pytest.skip(f"Binary not found: {BINARY_PATH}")

    env = os.environ.copy()
    # Note: PYTORCH_ENABLE_MPS_FALLBACK no longer needed with PyTorch 2.9.1+
    env.setdefault("SPDLOG_LEVEL", "warn")

    # Run WITHOUT --sequential to test true concurrent threading
    cmd = [
        str(BINARY_PATH),
        "--instances",
        "2",
        "--json",
        "--text",
        "Concurrent MPS synthesis test.",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(STREAM_TTS_CPP),
        env=env,
    )

    assert result.returncode == 0, f"Concurrent test_model_pooling failed: {result.stderr[-400:]}"

    try:
        metrics = parse_pooling_output(result.stdout)
    except ValueError as exc:
        pytest.fail(f"{exc}\nstdout tail:\n{result.stdout[-400:]}\nstderr tail:\n{result.stderr[-400:]}")

    assert metrics["instances"] == 2, metrics
    assert metrics.get("inference_mode") == "concurrent", metrics
    assert metrics.get("concurrent_success", False) is True, f"Concurrent synthesis failed: {metrics}"

    synth_ms = metrics.get("synthesis_ms", [])
    assert len(synth_ms) == 2, metrics
    assert all(ms > 0 for ms in synth_ms), f"All synthesis should succeed: {metrics}"
    assert max(synth_ms) < 5000, f"Synthesis took too long: {metrics}"
