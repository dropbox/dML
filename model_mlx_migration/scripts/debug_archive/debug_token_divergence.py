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
Debug token-by-token divergence between C++ and Python Whisper.

Focuses on file 0004 which shows the repetition loop issue.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX

# The problematic file
AUDIO_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"

WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)


def transcribe_python_with_tokens(model, audio_path):
    """Transcribe and return all token details."""
    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        temperature=0.0,
    )
    return result


def transcribe_cpp_with_tokens(audio_path):
    """Get token-level output from C++."""
    audio_path = os.path.abspath(audio_path)

    test_engine = "./src/mlx_inference_engine/test_engine"
    if not os.path.exists(test_engine):
        return {"error": "test_engine not found"}

    cmd = [
        test_engine,
        "--whisper", os.path.expanduser(WHISPER_MODEL_PATH),
        "--transcribe", audio_path,
        "--verbose"  # Request verbose output with tokens
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )

        # Print all output for debugging
        print("=== C++ STDOUT ===")
        print(result.stdout)
        print("=== C++ STDERR ===")
        print(result.stderr)

        # Parse JSON
        json_start = result.stdout.find("{")
        json_end = result.stdout.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = result.stdout[json_start:json_end]
            return json.loads(json_str)
        return {"error": "Could not parse JSON", "stdout": result.stdout}

    except Exception as e:
        return {"error": str(e)}


def main():
    print("Loading Python WhisperMLX...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    print(f"\nProcessing: {AUDIO_FILE}")

    # Python transcription
    print("\n=== PYTHON TRANSCRIPTION ===")
    py_result = transcribe_python_with_tokens(model, AUDIO_FILE)
    print(f"Text: {py_result['text']}")

    # Get tokenizer to decode tokens
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer
    tokenizer = get_whisper_tokenizer(multilingual=True)

    # If we have segments with tokens, print them
    if 'segments' in py_result:
        for i, seg in enumerate(py_result['segments']):
            if 'tokens' in seg:
                print(f"\nSegment {i} tokens ({len(seg['tokens'])}):")
                token_strs = []
                for t in seg['tokens'][:50]:  # First 50
                    try:
                        decoded = tokenizer.decode([t])
                        token_strs.append(f"{t}:{decoded!r}")
                    except:
                        token_strs.append(f"{t}:?")
                print("  " + " ".join(token_strs))

    # C++ transcription
    print("\n=== C++ TRANSCRIPTION ===")
    cpp_result = transcribe_cpp_with_tokens(AUDIO_FILE)
    if "error" in cpp_result:
        print(f"Error: {cpp_result['error']}")
    else:
        print(f"Text: {cpp_result.get('text', 'N/A')}")
        if 'tokens' in cpp_result:
            print(f"\nTokens ({len(cpp_result['tokens'])}):")
            token_strs = []
            for t in cpp_result['tokens'][:50]:
                try:
                    decoded = tokenizer.decode([t])
                    token_strs.append(f"{t}:{decoded!r}")
                except:
                    token_strs.append(f"{t}:?")
            print("  " + " ".join(token_strs))

    # Compare key statistics
    print("\n=== COMPARISON ===")
    py_text = py_result.get('text', '')
    cpp_text = cpp_result.get('text', '')

    py_words = py_text.split()
    cpp_words = cpp_text.split()

    print(f"Python words: {len(py_words)}")
    print(f"C++ words: {len(cpp_words)}")

    # Find first divergence
    min_len = min(len(py_words), len(cpp_words))
    for i in range(min_len):
        if py_words[i].lower() != cpp_words[i].lower():
            print(f"\nFirst divergence at word {i}:")
            print(f"  Python: {' '.join(py_words[max(0,i-3):i+4])}")
            print(f"  C++:    {' '.join(cpp_words[max(0,i-3):i+4])}")
            break
    else:
        if len(py_words) != len(cpp_words):
            print(f"\nOutputs match until word {min_len}, then lengths differ")


if __name__ == "__main__":
    main()
