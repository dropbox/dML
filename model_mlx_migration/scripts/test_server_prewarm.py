#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test server pre-warming and TTS endpoint.

This script tests that:
1. Pre-warming eliminates cold start latency
2. TTS endpoint works correctly
3. First request is fast (no cold start)
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_prewarm():
    """Test pre-warming functionality directly."""
    print("=" * 60)
    print("Testing Pre-warm Functionality")
    print("=" * 60)

    import asyncio

    from tools.dashvoice.server import get_tts_model, prewarm_models

    # Measure pre-warm time
    print("\nRunning pre-warm...")
    start = time.perf_counter()
    asyncio.run(prewarm_models())
    prewarm_time = time.perf_counter() - start
    print(f"Pre-warm completed in {prewarm_time:.1f}s")

    # Test TTS is warm (should be fast now)
    print("\nTesting TTS after pre-warm...")
    tts_model = get_tts_model()
    if tts_model == "disabled":
        print("TTS is disabled (mlx_audio not installed)")
        return

    # First call after pre-warm
    start = time.perf_counter()
    for r in tts_model.generate(
        text="Hello, world!",
        voice="af_bella",
        speed=1.0,
        verbose=False
    ):
        result = r
    first_call_time = (time.perf_counter() - start) * 1000

    # Second call (should be similar to first if properly warmed)
    start = time.perf_counter()
    for r in tts_model.generate(
        text="This is a test.",
        voice="af_bella",
        speed=1.0,
        verbose=False
    ):
        result = r
    second_call_time = (time.perf_counter() - start) * 1000

    print(f"First TTS call (post-warm): {first_call_time:.1f}ms")
    print(f"Second TTS call: {second_call_time:.1f}ms")

    # Check that first call is fast (no cold start overhead)
    # Cold start is ~1100ms, warm should be <300ms
    if first_call_time < 400:
        print("[PASS] Pre-warming eliminated cold start")
    else:
        print(f"[WARN] First call still slow ({first_call_time:.0f}ms > 400ms)")

    import numpy as np
    audio = np.array(result.audio)
    audio_duration = len(audio) / result.sample_rate
    rtf = (second_call_time / 1000) / audio_duration
    print(f"RTF: {rtf:.3f} ({1/rtf:.0f}x real-time)")


def test_cold_vs_warm():
    """Compare cold start vs warm start latency."""
    print("\n" + "=" * 60)
    print("Cold vs Warm Start Comparison")
    print("=" * 60)

    # Create fresh model instance (cold)
    print("\nLoading fresh TTS model (cold start)...")
    from mlx_audio.tts.utils import load_model

    start = time.perf_counter()
    model = load_model("prince-canuma/Kokoro-82M")
    load_time = time.perf_counter() - start
    print(f"Model load time: {load_time*1000:.0f}ms")

    # First generation (cold)
    print("\nFirst generation (cold)...")
    start = time.perf_counter()
    for r in model.generate(
        text="Hello, world!",
        voice="af_bella",
        speed=1.0,
        verbose=False
    ):
        _result = r  # Result consumed by generator
    cold_time = (time.perf_counter() - start) * 1000
    print(f"Cold start generation: {cold_time:.0f}ms")

    # Second generation (warm)
    print("\nSecond generation (warm)...")
    start = time.perf_counter()
    for r in model.generate(
        text="Hello, world!",
        voice="af_bella",
        speed=1.0,
        verbose=False
    ):
        _result = r  # Result consumed by generator
    warm_time = (time.perf_counter() - start) * 1000
    print(f"Warm generation: {warm_time:.0f}ms")

    speedup = cold_time / warm_time
    print(f"\nSpeedup from pre-warming: {speedup:.1f}x")
    print(f"Time saved: {cold_time - warm_time:.0f}ms")


def test_tts_endpoint_simulation():
    """Simulate TTS endpoint calls to verify performance."""
    print("\n" + "=" * 60)
    print("TTS Endpoint Simulation")
    print("=" * 60)

    import asyncio

    import numpy as np

    from tools.dashvoice.server import get_tts_model, prewarm_models

    # Pre-warm first
    print("\nPre-warming...")
    asyncio.run(prewarm_models())

    tts_model = get_tts_model()
    if tts_model == "disabled":
        print("TTS disabled")
        return

    # Simulate multiple endpoint calls
    test_texts = [
        ("Short", "Hello."),
        ("Medium", "The quick brown fox jumps over the lazy dog."),
        ("Long", "This is a longer piece of text to test how the system handles varying input lengths in a production-like scenario."),
    ]

    print("\nSimulating endpoint calls:")
    print(f"{'Type':<10} {'Text Len':<10} {'Time (ms)':<12} {'Audio (s)':<12} {'RTF':<8}")
    print("-" * 60)

    for name, text in test_texts:
        start = time.perf_counter()
        for r in tts_model.generate(
            text=text,
            voice="af_bella",
            speed=1.0,
            verbose=False
        ):
            result = r
        gen_time = (time.perf_counter() - start) * 1000

        audio = np.array(result.audio)
        duration = len(audio) / result.sample_rate
        rtf = (gen_time / 1000) / duration

        print(f"{name:<10} {len(text):<10} {gen_time:<12.1f} {duration:<12.2f} {rtf:<8.4f}")


if __name__ == "__main__":
    try:
        test_prewarm()
        test_cold_vs_warm()
        test_tts_endpoint_simulation()
        print("\n" + "=" * 60)
        print("All tests completed")
        print("=" * 60)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure mlx_audio is installed: pip install mlx_audio")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
