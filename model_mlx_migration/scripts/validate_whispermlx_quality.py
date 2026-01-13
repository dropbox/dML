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
WhisperMLX Quality Validation Script

Validates that WhisperMLX standard mode produces EXACT output matching mlx-whisper.
This is the quality gate for WhisperMLX - all outputs must match the reference.

Usage:
    python scripts/validate_whispermlx_quality.py

Exit codes:
    0: All validations passed (100% exact match)
    1: One or more validations failed
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_test_audio_files() -> list[Path]:
    """Find test audio files for validation."""
    project_root = Path(__file__).parent.parent

    test_files = []

    # Primary test file
    medium_mlx = project_root / "reports" / "audio" / "medium_mlx.wav"
    if medium_mlx.exists():
        test_files.append(medium_mlx)

    # Gradio test audio - SKIP: non-speech test audio, produces empty transcription
    # gradio_test = project_root / ".venv/lib/python3.12/site-packages/gradio/test_data/test_audio.wav"

    # F5-TTS test audio
    f5_test = project_root / ".venv/lib/python3.12/site-packages/f5_tts_mlx/tests/test_en_1_ref_short.wav"
    if f5_test.exists():
        test_files.append(f5_test)

    # RAVDESS prosody samples (diverse emotions, known content)
    ravdess_dir = project_root / "data/prosody/ravdess/Actor_01"
    if ravdess_dir.exists():
        ravdess_files = list(ravdess_dir.glob("*.wav"))[:10]  # Take first 10
        test_files.extend(ravdess_files)

    # CREMA-D samples
    cremad_dir = project_root / "data/prosody/crema-d/AudioMP3"
    if cremad_dir.exists():
        cremad_files = list(cremad_dir.glob("*.mp3"))[:5]  # Take first 5
        test_files.extend(cremad_files)

    return test_files


def validate_whispermlx(audio_path: Path, model, ref_transcribe) -> dict:
    """
    Validate WhisperMLX output matches mlx-whisper reference.

    Returns:
        dict with keys: passed, audio_path, ref_text, our_text, ref_time, our_time, error
    """
    result = {
        "audio_path": str(audio_path),
        "passed": False,
        "ref_text": None,
        "our_text": None,
        "ref_time": None,
        "our_time": None,
        "error": None,
    }

    try:
        # Reference: mlx-whisper
        t0 = time.perf_counter()
        ref_result = ref_transcribe(
            str(audio_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
        )
        result["ref_time"] = time.perf_counter() - t0
        result["ref_text"] = ref_result["text"].strip()

        # Ours: WhisperMLX standard mode (NOT variable_length!)
        t0 = time.perf_counter()
        our_result = model.transcribe(str(audio_path), variable_length=False)
        result["our_time"] = time.perf_counter() - t0
        result["our_text"] = our_result["text"].strip()

        # Exact match required
        if result["our_text"] == result["ref_text"]:
            result["passed"] = True
        else:
            result["error"] = f"Text mismatch: '{result['our_text']}' != '{result['ref_text']}'"

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    print("=" * 70)
    print("WhisperMLX Quality Validation")
    print("=" * 70)
    print()

    # Find test files
    test_files = find_test_audio_files()
    print(f"Found {len(test_files)} test audio files")

    if len(test_files) < 10:
        print(f"WARNING: Only {len(test_files)} files found, need 10+ for full validation")

    if not test_files:
        print("ERROR: No test audio files found!")
        return 1

    # Load models
    print()
    print("Loading models...")

    # Import mlx-whisper
    try:
        import mlx_whisper
        ref_transcribe = mlx_whisper.transcribe
        print("  mlx-whisper: OK")
    except ImportError as e:
        print(f"  mlx-whisper: FAILED - {e}")
        return 1

    # Import WhisperMLX
    try:
        from tools.whisper_mlx import WhisperMLX
        model = WhisperMLX.from_pretrained("large-v3")
        print("  WhisperMLX: OK")
    except ImportError as e:
        print(f"  WhisperMLX: FAILED - {e}")
        return 1

    # Warmup
    print()
    print("Warming up models...")
    if test_files:
        warmup_file = str(test_files[0])
        _ = ref_transcribe(warmup_file, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _ = model.transcribe(warmup_file, variable_length=False)
        print("  Warmup complete")

    # Run validation
    print()
    print("Running validation...")
    print("-" * 70)

    results = []
    passed = 0
    failed = 0

    import warnings
    for i, audio_path in enumerate(test_files):
        print(f"[{i+1}/{len(test_files)}] {audio_path.name}...", end=" ", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = validate_whispermlx(audio_path, model, ref_transcribe)

        results.append(result)

        if result["passed"]:
            passed += 1
            speedup = result["ref_time"] / result["our_time"] if result["our_time"] else 0
            print(f"PASS ({speedup:.2f}x)")
        else:
            failed += 1
            print("FAIL")
            if result["error"]:
                print(f"       Error: {result['error'][:80]}")
            if result["ref_text"]:
                print(f"       Expected: {result['ref_text'][:50]}...")
            if result["our_text"]:
                print(f"       Got:      {result['our_text'][:50]}...")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:  {len(test_files)}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    print(f"Pass rate:    {passed/len(test_files)*100:.1f}%")

    if passed == len(test_files):
        print()
        print("RESULT: ALL VALIDATIONS PASSED - WhisperMLX standard mode is SAFE")

        # Calculate average speedup
        valid_speedups = [
            r["ref_time"] / r["our_time"]
            for r in results
            if r["passed"] and r["ref_time"] and r["our_time"]
        ]
        if valid_speedups:
            avg_speedup = sum(valid_speedups) / len(valid_speedups)
            print(f"Average speedup: {avg_speedup:.2f}x")

        return 0
    else:
        print()
        print("RESULT: VALIDATION FAILED - WhisperMLX output does not match reference")
        return 1


if __name__ == "__main__":
    sys.exit(main())
