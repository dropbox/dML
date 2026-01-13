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
Gate 0: C++ vs Python Whisper Comparison
Compare C++ whisper_model.cpp output with Python WhisperMLX output.

Usage:
    python scripts/gate0_compare.py --audio data/librispeech/dev-clean/1272/128104/1272-128104-0000.flac
    python scripts/gate0_compare.py --all  # Run on all test files
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram, pad_or_trim, N_SAMPLES
from tools.whisper_mlx.tokenizer import get_whisper_tokenizer


# Test files from LibriSpeech dev-clean
TEST_FILES = [
    "data/librispeech/dev-clean/1272/128104/1272-128104-0000.flac",
    "data/librispeech/dev-clean/1272/128104/1272-128104-0001.flac",
    "data/librispeech/dev-clean/1272/128104/1272-128104-0002.flac",
    "data/librispeech/dev-clean/1272/128104/1272-128104-0003.flac",
    "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac",
]

# Reference transcripts from LibriSpeech
REFERENCE_TRANSCRIPTS = {
    "1272-128104-0000.flac": "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL",
    "1272-128104-0001.flac": "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER",
    "1272-128104-0002.flac": "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND",
    "1272-128104-0003.flac": "HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS REALLY GREEK AFTER ALL AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHACA",
    "1272-128104-0004.flac": "LINNELL'S PICTURES ARE A SORT OF UP GUARDS AND AT THEM PAINTING AND MASSON'S EXQUISITE IDYLLS ARE AS NATIONAL AS A JINGO POEM MISTER BIRKET FOSTER'S LANDSCAPES SMILE AT ONE MUCH IN THE SAME WAY THAT MISTER CARKER USED TO FLASH HIS TEETH AND MISTER JOHN COLLIER GIVES HIS SITTER A CHEERFUL SLAP ON THE BACK BEFORE HE SAYS LIKE A SHAMPOOER IN A TURKISH BATH NEXT MAN",
}

# Model paths
WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)
CPP_TEST_ENGINE = "src/mlx_inference_engine/test_engine"


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove leading/trailing whitespace
    text = text.strip()
    # Convert to uppercase for comparison
    text = text.upper()
    # Remove punctuation
    import re
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def word_error_rate(ref: str, hyp: str) -> Tuple[float, int, int, int, int]:
    """
    Calculate Word Error Rate using dynamic programming.
    Returns: (wer, substitutions, insertions, deletions, total_words)
    """
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()

    r_len = len(ref_words)
    h_len = len(hyp_words)

    # DP table
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    # Initialize
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # deletion
                    dp[i][j-1] + 1,    # insertion
                    dp[i-1][j-1] + 1   # substitution
                )

    # Backtrack to count edits
    edits = dp[r_len][h_len]
    wer = edits / r_len if r_len > 0 else 0.0

    return wer, edits, 0, 0, r_len  # Simplified: just total edits


def transcribe_python(model: WhisperMLX, audio_path: str, disable_vad: bool = True) -> Dict:
    """Transcribe using Python WhisperMLX."""
    start = time.time()
    # Gate 0: Use vad_aggressiveness=0 (most conservative) for better C++ match
    # When disable_vad=True, use 0 which keeps most audio
    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        temperature=0.0,  # Greedy decoding for determinism
        vad_aggressiveness=0 if disable_vad else 2,  # Gate 0: most conservative VAD
    )
    elapsed = time.time() - start

    return {
        "text": result["text"],
        "tokens": result.get("tokens", []),
        "language": result.get("language", "en"),
        "elapsed_ms": elapsed * 1000,
    }


def transcribe_python_raw(model: WhisperMLX, audio_path: str) -> Dict:
    """
    RAW greedy decode matching C++ --no-vad mode exactly:
    1. Load audio
    2. Pad to 30s (N_SAMPLES = 480000)
    3. Create mel spectrogram
    4. Encode
    5. Greedy decode WITH SuppressTokens filter (matching C++)

    Used for Gate 0 exact comparison.
    """
    start = time.time()

    # 1. Load audio
    audio = load_audio(audio_path)

    # 2. Calculate actual audio duration BEFORE padding (matching C++ exactly)
    # C++ mlx_inference_engine.cpp:360: audio_duration_sec = resampled_audio.size() / 16000.0f
    actual_audio_duration = len(audio) / 16000.0

    # 3. Pad to exactly 30 seconds
    audio = pad_or_trim(audio, N_SAMPLES)

    # 4. Create mel spectrogram
    mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)

    # Pad/trim mel to standard length (3000 frames)
    target_len = model.config.n_audio_ctx * 2  # 1500 * 2 = 3000
    if mel.shape[0] < target_len:
        mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
    elif mel.shape[0] > target_len:
        mel = mel[:target_len, :]

    # Add batch dimension
    mel = mel[None]

    # 4. Encode
    audio_features = model.encoder(mel)
    mx.eval(audio_features)

    # 5. Get tokenizer - use default num_languages (99 for large-v3-turbo)
    # Don't calculate from vocab size as it's error-prone
    is_multilingual = model.config.n_vocab >= 51865
    tokenizer = get_whisper_tokenizer(
        multilingual=is_multilingual,
        language="en",
        task="transcribe",
    )

    # 6. SuppressTokens filter - must match C++ exactly
    # From whisper_model.cpp lines 1217-1225 - these are non-speech tokens
    # C++ suppresses these tokens to prevent garbage output
    # NOTE: Token 0 ("!") is suppressed to prevent spurious exclamation marks
    SUPPRESS_TOKENS = {
        0, 1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
        90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931,
        1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961,
        4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938,
        12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553,
        16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435,
        28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254
    }

    # 7. Greedy decode WITH filters matching C++ exactly
    # Start with SOT sequence
    initial_tokens = list(tokenizer.sot_sequence)
    tokens = mx.array([initial_tokens])

    kv_cache = None
    max_tokens = model.config.n_text_ctx // 2  # 224 tokens max
    all_tokens = list(initial_tokens)
    sample_begin = len(initial_tokens)  # SOT sequence length

    # Timestamp tracking for monotonicity
    timestamp_begin = tokenizer.timestamp_begin  # 50364
    max_initial_timestamp_index = 50  # 1.0 second max initial timestamp

    # Calculate max audio timestamp from raw audio duration
    # Each timestamp token = 0.02 seconds
    # CRITICAL: Use float32 to match C++ precision exactly!
    # Python float64: 9.12/0.02 = 455.999999... → int() gives 455
    # C++ float32:    9.12f/0.02f = 456.0f → static_cast<int>() gives 456
    # Using numpy float32 ensures identical results
    import numpy as np
    max_audio_timestamp_index = int(np.float32(actual_audio_duration) / np.float32(0.02))
    max_audio_timestamp_index = min(max_audio_timestamp_index, 1500)  # Cap at 30s

    for i in range(max_tokens):
        # Decode step
        logits, kv_cache, _, _ = model.decoder(tokens, audio_features, kv_cache=kv_cache)
        mx.eval(logits, kv_cache)

        # Get logits as list for manipulation
        logits_list = logits[:, -1, :].tolist()[0]
        n_vocab = len(logits_list)

        # ==================================================================
        # Compute timestamp dominance BEFORE applying filters
        # This matches C++ logic for forcing timestamps
        # ==================================================================
        import math

        # Compute log-sum-exp normalizer for original logits
        max_logit_orig = max(logits_list)
        sum_exp_orig = sum(math.exp(l - max_logit_orig) for l in logits_list if l > -1e9)
        log_sum_exp_orig = max_logit_orig + math.log(sum_exp_orig + 1e-10)

        # Log probability of sum of timestamp tokens
        timestamp_sum_exp = sum(
            math.exp(logits_list[t] - max_logit_orig)
            for t in range(timestamp_begin, n_vocab)
            if logits_list[t] > -1e9
        )
        timestamp_logprob = (max_logit_orig + math.log(timestamp_sum_exp) - log_sum_exp_orig
                            if timestamp_sum_exp > 0 else float('-inf'))

        # Max log probability of text tokens
        max_text_logit = max((logits_list[t] for t in range(timestamp_begin) if logits_list[t] > -1e9),
                            default=float('-inf'))
        max_text_logprob = max_text_logit - log_sum_exp_orig if max_text_logit > -1e9 else float('-inf')

        # Check if timestamps dominate
        timestamp_dominates = timestamp_logprob > max_text_logprob

        # ==================================================================
        # Apply SuppressTokens filter (suppress non-speech tokens)
        # ==================================================================
        for suppress_token in SUPPRESS_TOKENS:
            if suppress_token < n_vocab:
                logits_list[suppress_token] = float('-inf')

        # ==================================================================
        # ApplyTimestampRules - matching C++ exactly
        # ==================================================================
        sampled_tokens = all_tokens[sample_begin:]  # Tokens after SOT
        sampled_len = len(sampled_tokens)
        current_len = len(all_tokens)  # Total length including SOT

        # Check last two tokens
        last_was_timestamp = (sampled_len >= 1 and
                              sampled_tokens[-1] >= timestamp_begin)
        # In C++: true when less than 2 tokens OR second-to-last is timestamp
        penultimate_was_timestamp = (sampled_len < 2 or
                                     sampled_tokens[-2] >= timestamp_begin)

        # Timestamp pairing rules
        if last_was_timestamp:
            if penultimate_was_timestamp:
                # Two timestamps in a row - next must be text (suppress all timestamps)
                for t in range(timestamp_begin, n_vocab):
                    logits_list[t] = float('-inf')
            else:
                # Timestamp after text - force timestamp or EOT (suppress text)
                for t in range(tokenizer.eot):
                    logits_list[t] = float('-inf')

        # Timestamps must be strictly monotonically increasing
        timestamp_tokens = [t for t in sampled_tokens if t >= timestamp_begin]
        if timestamp_tokens:
            timestamp_last = timestamp_tokens[-1] + 1  # Strictly increasing
            for t in range(timestamp_begin, min(timestamp_last, n_vocab)):
                logits_list[t] = float('-inf')

        # At sample_begin (first decode step), force timestamp token
        if current_len == sample_begin:
            # Suppress all non-timestamp tokens
            for t in range(timestamp_begin):
                if t != tokenizer.eot:
                    logits_list[t] = float('-inf')
            # Max initial timestamp constraint
            last_allowed = timestamp_begin + max_initial_timestamp_index
            for t in range(last_allowed + 1, n_vocab):
                logits_list[t] = float('-inf')
            # SuppressBlank at sample_begin
            logits_list[220] = float('-inf')  # space token
            logits_list[tokenizer.eot] = float('-inf')  # EOT

        # Suppress timestamps beyond audio duration
        max_allowed_timestamp = timestamp_begin + max_audio_timestamp_index
        for t in range(max_allowed_timestamp + 1, n_vocab):
            logits_list[t] = float('-inf')

        # ==================================================================
        # Apply timestamp dominance - force timestamp when it dominates
        # CRITICAL: Do NOT apply when "two timestamps in a row" (text required)
        # ==================================================================
        two_timestamps_in_row = last_was_timestamp and penultimate_was_timestamp
        if timestamp_dominates and not two_timestamps_in_row:
            for t in range(timestamp_begin):
                logits_list[t] = float('-inf')

        # ==================================================================
        # Greedy token selection
        # ==================================================================
        next_token = max(range(n_vocab), key=lambda x: logits_list[x])

        # Check for EOT
        if next_token == tokenizer.eot:
            break

        all_tokens.append(next_token)
        tokens = mx.array([[next_token]])

    # 7. Decode text (skip SOT sequence)
    sample_begin = len(tokenizer.sot_sequence)
    output_tokens = all_tokens[sample_begin:]

    # Filter out special tokens for text (timestamps and other special tokens >= 50257)
    # Standard vocab is 0-50256, special tokens are 50257+
    # Text tokens are < 50257 OR specific punctuation tokens
    # Actually, filter everything >= timestamp_begin which covers timestamps
    # AND filter notimestamps token (50363) and other special tokens
    text_tokens = []
    for t in output_tokens:
        # Keep regular vocab tokens (0-50256)
        # Skip timestamp tokens (>= 50364) and special tokens like notimestamps (50363)
        if t < 50257:  # Regular vocab only
            text_tokens.append(t)
    text = tokenizer.decode(text_tokens)

    elapsed = time.time() - start

    return {
        "text": text,
        "tokens": output_tokens,
        "language": "en",
        "elapsed_ms": elapsed * 1000,
    }


def transcribe_cpp(audio_path: str) -> Dict:
    """Transcribe using C++ test_engine with --transcribe option."""
    # Use absolute path
    audio_path = os.path.abspath(audio_path)

    # Find test_engine - check common locations (prefer build directory)
    test_engine_paths = [
        os.path.join(Path(__file__).parent.parent, "build", "test_mlx_engine"),
        os.path.join(Path(__file__).parent.parent, "build", "test_engine"),
        os.path.join(Path(__file__).parent.parent, "src", "mlx_inference_engine", "test_engine"),
        os.path.join(Path(__file__).parent.parent, "test_engine"),
        "./build/test_mlx_engine",
        "./test_engine",
        "./src/mlx_inference_engine/test_engine",
    ]
    test_engine = None
    for p in test_engine_paths:
        if os.path.exists(p):
            test_engine = p
            break

    if not test_engine:
        return {
            "text": "[test_engine not found]",
            "tokens": [],
            "language": "en",
            "elapsed_ms": 0,
            "error": "test_engine binary not found",
        }

    cmd = [
        test_engine,
        "--whisper", os.path.expanduser(WHISPER_MODEL_PATH),
        "--no-vad",  # Gate 0: Disable VAD for exact comparison
        "--transcribe", audio_path
    ]

    # Set library path for libfvad
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

        # Parse JSON output (ignore stderr which has ffmpeg noise)
        stdout = result.stdout
        # Find the JSON block
        json_start = stdout.find("{")
        json_end = stdout.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = stdout[json_start:json_end]
            data = json.loads(json_str)
            # Extract tokens from segments if not at top level
            tokens = data.get("tokens", [])
            if not tokens and "segments" in data and data["segments"]:
                tokens = data["segments"][0].get("tokens", [])
            return {
                "text": data.get("text", ""),
                "tokens": tokens,
                "language": data.get("language", "en"),
                "elapsed_ms": data.get("elapsed_ms", 0),
            }
        else:
            return {
                "text": f"[Parse error: {stdout[:200]}]",
                "tokens": [],
                "language": "en",
                "elapsed_ms": 0,
                "error": "Could not parse JSON output",
            }
    except subprocess.TimeoutExpired:
        return {
            "text": "[Timeout]",
            "tokens": [],
            "language": "en",
            "elapsed_ms": 0,
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "text": f"[Error: {str(e)}]",
            "tokens": [],
            "language": "en",
            "elapsed_ms": 0,
            "error": str(e),
        }


def compare_outputs(
    audio_file: str,
    python_result: Dict,
    cpp_result: Dict,
    reference: str = None
) -> Dict:
    """Compare Python and C++ outputs including token-level comparison."""
    py_text = python_result["text"]
    cpp_text = cpp_result["text"]
    py_tokens = python_result.get("tokens", [])
    cpp_tokens = cpp_result.get("tokens", [])

    # Normalize for comparison
    py_norm = normalize_text(py_text)
    cpp_norm = normalize_text(cpp_text)

    exact_match = py_norm == cpp_norm

    # Token-level comparison
    # C++ includes SOT sequence (50258, 50259, 50359) and EOT (50257)
    # Python outputs content only (no SOT/EOT wrapper)
    # Strip SOT/EOT from C++ for fair comparison
    cpp_content = cpp_tokens[3:-1] if len(cpp_tokens) > 4 else cpp_tokens

    # Exact content match (the true Gate 0 criterion)
    exact_content_match = py_tokens == cpp_content
    token_match = exact_content_match  # Use content match as primary

    # Separate text tokens (< 50257) from timestamp tokens (>= 50364)
    py_text_tokens = [t for t in py_tokens if t < 50257]
    cpp_text_tokens = [t for t in cpp_content if t < 50257]
    py_ts_tokens = [t for t in py_tokens if t >= 50364]
    cpp_ts_tokens = [t for t in cpp_content if t >= 50364]

    text_token_match = py_text_tokens == cpp_text_tokens
    ts_token_match = py_ts_tokens == cpp_ts_tokens
    length_match = len(py_tokens) == len(cpp_content)

    result = {
        "file": audio_file,
        "python": {
            "text": py_text,
            "normalized": py_norm,
            "tokens": py_tokens,
            "text_tokens": py_text_tokens,
            "ts_tokens": py_ts_tokens,
            "elapsed_ms": python_result["elapsed_ms"],
        },
        "cpp": {
            "text": cpp_text,
            "normalized": cpp_norm,
            "tokens": cpp_tokens,
            "text_tokens": cpp_text_tokens,
            "ts_tokens": cpp_ts_tokens,
            "elapsed_ms": cpp_result.get("elapsed_ms", 0),
            "error": cpp_result.get("error"),
        },
        "exact_match": exact_match,
        "token_match": token_match,
        "text_token_match": text_token_match,
        "ts_token_match": ts_token_match,
        "length_match": length_match,
    }

    # Calculate WER against reference if available
    if reference:
        ref_norm = normalize_text(reference)
        py_wer, _, _, _, _ = word_error_rate(reference, py_text)
        result["reference"] = ref_norm
        result["python_wer"] = py_wer

        if not cpp_result.get("error"):
            cpp_wer, _, _, _, _ = word_error_rate(reference, cpp_text)
            result["cpp_wer"] = cpp_wer

    return result


def find_diverse_files(base_path: str, count: int = 20, max_duration: float = 10.0) -> List[str]:
    """Find diverse test files from different speakers, filtered by duration.

    Default max_duration is 10.0s to match C++ SINGLE_PASS_MAX_DURATION threshold.
    Files > 10s use C++ multi-segment path but still produce identical tokens for files <= 30s.
    Files > 30s require true multi-segment processing in Python (not yet implemented in Gate 0).

    Use --max-duration 30.0 for comprehensive single-chunk testing.
    """
    import random
    import subprocess
    files = []
    base = Path(base_path)

    # Find all flac files
    all_flacs = list(base.rglob("*.flac"))

    # Filter by duration (Gate 0 requires single-segment processing, max 10s)
    valid_flacs = []
    for f in all_flacs:
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(f)],
                capture_output=True, text=True, timeout=5
            )
            duration = float(result.stdout.strip())
            if duration <= max_duration:
                valid_flacs.append(str(f))
        except:
            pass  # Skip files we can't probe

    # Group by speaker
    by_speaker = {}
    for f in valid_flacs:
        speaker = Path(f).stem.split("-")[0]
        if speaker not in by_speaker:
            by_speaker[speaker] = []
        by_speaker[speaker].append(f)

    # Sample from different speakers
    speakers = list(by_speaker.keys())
    random.seed(42)
    random.shuffle(speakers)

    files_per_speaker = max(2, (count // len(speakers)) + 1)

    for speaker in speakers:
        if len(files) >= count:
            break
        speaker_files = by_speaker[speaker][:files_per_speaker]
        files.extend(speaker_files)

    return files[:count]


def main():
    parser = argparse.ArgumentParser(description="Gate 0: C++ vs Python Whisper Comparison")
    parser.add_argument("--audio", type=str, help="Single audio file to test")
    parser.add_argument("--all", action="store_true", help="Test default 5 files")
    parser.add_argument("--diverse", type=int, default=0,
                       help="Test N diverse files from different speakers")
    parser.add_argument("--max-duration", type=float, default=10.0,
                       help="Max duration in seconds for --diverse (default 10.0)")
    parser.add_argument("--model", type=str, default="mlx-community/whisper-large-v3-turbo",
                       help="Model name or path")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--raw", action="store_true", default=True,
                       help="Use raw greedy decode (no VAD, no filters) - DEFAULT for Gate 0")
    args = parser.parse_args()

    if not args.audio and not args.all and args.diverse == 0:
        parser.print_help()
        print("\nError: Specify --audio <file>, --all, or --diverse N")
        sys.exit(1)

    # Load Python model
    print("Loading Python WhisperMLX...")
    start = time.time()
    model = WhisperMLX.from_pretrained(args.model, dtype=mx.float16)
    print(f"  Loaded in {time.time() - start:.1f}s")

    # Determine files to test
    if args.diverse > 0:
        test_files = find_diverse_files("data/librispeech/dev-clean", args.diverse, args.max_duration)
        print(f"  Found {len(test_files)} diverse files from different speakers")
    elif args.all:
        test_files = TEST_FILES
    else:
        test_files = [args.audio]

    results = []
    total_wer = 0.0
    total_files = 0

    print(f"\nTesting {len(test_files)} file(s)...")
    print("=" * 80)

    for audio_path in test_files:
        if not os.path.exists(audio_path):
            print(f"WARNING: File not found: {audio_path}")
            continue

        filename = os.path.basename(audio_path)
        print(f"\nFile: {filename}")

        # Get reference transcript
        reference = REFERENCE_TRANSCRIPTS.get(filename)
        if reference:
            print(f"  Reference: {reference[:60]}...")

        # Transcribe with Python
        mode = "raw greedy" if args.raw else "transcribe"
        print(f"  Python transcription ({mode})...", end=" ", flush=True)
        if args.raw:
            py_result = transcribe_python_raw(model, audio_path)
        else:
            py_result = transcribe_python(model, audio_path)
        print(f"DONE ({py_result['elapsed_ms']:.0f}ms)")
        print(f"    Text: {py_result['text'][:60]}...")

        # Transcribe with C++
        print("  C++ transcription...", end=" ", flush=True)
        cpp_result = transcribe_cpp(audio_path)
        if cpp_result.get("error"):
            print(f"ERROR ({cpp_result.get('error')})")
        else:
            print(f"DONE ({cpp_result['elapsed_ms']:.0f}ms)")
            print(f"    Text: {cpp_result['text'][:60]}...")

        # Compare
        comparison = compare_outputs(audio_path, py_result, cpp_result, reference)
        results.append(comparison)

        if reference and "python_wer" in comparison:
            wer_pct = comparison["python_wer"] * 100
            total_wer += comparison["python_wer"]
            total_files += 1
            print(f"    Python WER: {wer_pct:.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if total_files > 0:
        avg_wer = (total_wer / total_files) * 100
        print(f"Average Python WER: {avg_wer:.2f}% ({total_files} files)")

    cpp_success = sum(1 for r in results if not r["cpp"].get("error"))
    if cpp_success > 0:
        cpp_wer_sum = sum(r.get("cpp_wer", 0) for r in results if "cpp_wer" in r)
        cpp_wer_count = sum(1 for r in results if "cpp_wer" in r)
        if cpp_wer_count > 0:
            avg_cpp_wer = (cpp_wer_sum / cpp_wer_count) * 100
            print(f"Average C++ WER: {avg_cpp_wer:.2f}% ({cpp_wer_count} files)")

        # Check for matches - TEXT LEVEL
        text_matches = sum(1 for r in results if r.get("exact_match", False))
        print(f"\nText matches (normalized): {text_matches}/{len(results)}")

        # Check for matches - TOKEN LEVEL
        token_matches = sum(1 for r in results if r.get("token_match", False))
        text_token_matches = sum(1 for r in results if r.get("text_token_match", False))
        ts_token_matches = sum(1 for r in results if r.get("ts_token_match", False))
        length_matches = sum(1 for r in results if r.get("length_match", False))

        print("\n--- TOKEN-LEVEL ANALYSIS ---")
        print(f"Full token match:       {token_matches}/{len(results)} ({100*token_matches/len(results):.0f}%)")
        print(f"Text token match:       {text_token_matches}/{len(results)} ({100*text_token_matches/len(results):.0f}%)")
        print(f"Timestamp token match:  {ts_token_matches}/{len(results)} ({100*ts_token_matches/len(results):.0f}%)")
        print(f"Length match:           {length_matches}/{len(results)} ({100*length_matches/len(results):.0f}%)")

        # Determine Gate 0 status based on token-level match
        if token_matches == len(results):
            print("\n*** GATE 0 PASS: C++ and Python produce IDENTICAL tokens! ***")
            print("    (C++ SOT/EOT wrapper tokens stripped for comparison)")
        elif text_token_matches == len(results):
            print("\n*** GATE 0 PARTIAL: Text tokens match, timestamp tokens differ ***")
            # Show timestamp differences
            for r in results:
                if not r.get("ts_token_match", False):
                    py_ts = r['python'].get('ts_tokens', [])
                    cpp_ts = r['cpp'].get('ts_tokens', [])
                    if py_ts != cpp_ts:
                        print(f"  {os.path.basename(r['file'])}: py_ts={py_ts[-1] if py_ts else 'N/A'} cpp_ts={cpp_ts[-1] if cpp_ts else 'N/A'}")
        else:
            print("\n*** GATE 0: Token-level discrepancies found ***")
            for r in results:
                if not r.get("text_token_match", False):
                    print(f"\nFile: {os.path.basename(r['file'])}")
                    print(f"  Python text: {r['python']['normalized'][:60]}...")
                    print(f"  C++    text: {r['cpp']['normalized'][:60]}...")
                    # Show first token difference
                    py_txt = r['python'].get('text_tokens', [])
                    cpp_txt = r['cpp'].get('text_tokens', [])
                    for i, (pt, ct) in enumerate(zip(py_txt, cpp_txt)):
                        if pt != ct:
                            print(f"  First diff at token {i}: py={pt} cpp={ct}")
                            break
    else:
        print("\nWARNING: No C++ transcriptions succeeded.")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
