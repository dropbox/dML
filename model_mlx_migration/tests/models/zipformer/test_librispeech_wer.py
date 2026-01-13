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
LibriSpeech test-clean WER evaluation for Zipformer MLX.

This test validates that the MLX Zipformer achieves the target WER
on LibriSpeech test-clean (target: <3.5% for streaming, reference: 2.85%).

Run with: pytest tests/models/zipformer/test_librispeech_wer.py -v --tb=short

For a quick validation on a subset:
    pytest tests/models/zipformer/test_librispeech_wer.py -v -k "subset"

For full evaluation (takes longer):
    pytest tests/models/zipformer/test_librispeech_wer.py -v -k "full"
"""

import sys
import time
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate between reference and hypothesis."""
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()

    # Handle empty cases
    if not ref_words:
        return {
            "wer": 100.0 if hyp_words else 0.0,
            "edit_distance": len(hyp_words),
            "ref_words": 0,
            "hyp_words": len(hyp_words),
            "substitutions": 0,
            "insertions": len(hyp_words),
            "deletions": 0,
        }

    # Dynamic programming for edit distance with alignment tracking
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edit_distance = dp[m][n]
    wer = edit_distance / len(ref_words) * 100

    return {
        "wer": wer,
        "edit_distance": edit_distance,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
    }


def load_transcripts(librispeech_dir: Path) -> dict[str, str]:
    """Load all transcripts from LibriSpeech directory."""
    transcripts = {}
    for trans_file in librispeech_dir.rglob("*.txt"):
        # Skip README files
        if trans_file.name.startswith("README"):
            continue
        # LibriSpeech transcript format: {id} {text}
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text
    return transcripts


def get_audio_files(librispeech_dir: Path) -> list[tuple[str, Path]]:
    """Get list of (utterance_id, audio_path) tuples."""
    audio_files = []
    for flac_file in librispeech_dir.rglob("*.flac"):
        utt_id = flac_file.stem
        audio_files.append((utt_id, flac_file))
    return sorted(audio_files)


@pytest.fixture(scope="module")
def librispeech_test_clean():
    """Return path to LibriSpeech test-clean."""
    path = Path("data/LibriSpeech/test-clean")
    if not path.exists():
        pytest.skip(f"LibriSpeech test-clean not found at {path}")
    return path


@pytest.fixture(scope="module")
def checkpoint_path():
    """Return path to Zipformer checkpoint."""
    return "checkpoints/zipformer/en-streaming/exp/pretrained.pt"


@pytest.fixture(scope="module")
def bpe_model_path():
    """Return path to BPE model."""
    return "checkpoints/zipformer/en-streaming/data/lang_bpe_500/bpe.model"


@pytest.fixture(scope="module")
def pipeline(checkpoint_path, bpe_model_path):
    """Load ASR pipeline once for all tests."""
    from src.models.zipformer.inference import ASRPipeline

    if not Path(checkpoint_path).exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    if not Path(bpe_model_path).exists():
        pytest.skip(f"BPE model not found: {bpe_model_path}")

    return ASRPipeline.from_pretrained(
        checkpoint_path=checkpoint_path,
        bpe_model_path=bpe_model_path,
    )


class TestLibriSpeechWER:
    """WER evaluation on LibriSpeech test-clean."""

    # Target WER for streaming Zipformer (streaming adds latency penalty)
    # Reference non-streaming WER is 2.85%, streaming target is <3.5%
    TARGET_WER = 3.5

    def test_subset_wer(self, pipeline, librispeech_test_clean):
        """Test WER on a subset of LibriSpeech test-clean (faster)."""
        transcripts = load_transcripts(librispeech_test_clean)
        audio_files = get_audio_files(librispeech_test_clean)

        # Take a random subset of 50 utterances for quick validation
        import random

        random.seed(42)  # Reproducible
        subset = random.sample(audio_files, min(50, len(audio_files)))

        total_edit = 0
        total_ref_words = 0
        results = []

        print(f"\nEvaluating {len(subset)} utterances...")

        for utt_id, audio_path in subset:
            if utt_id not in transcripts:
                continue

            ref = transcripts[utt_id]
            hyp = pipeline.transcribe(str(audio_path))

            wer_result = compute_wer(ref, hyp)
            total_edit += wer_result["edit_distance"]
            total_ref_words += wer_result["ref_words"]

            results.append({
                "utt_id": utt_id,
                "ref": ref,
                "hyp": hyp,
                "wer": wer_result["wer"],
            })

        wer = total_edit / max(total_ref_words, 1) * 100

        print(f"Subset WER: {wer:.2f}%")
        print(f"Utterances evaluated: {len(results)}")
        print(f"Total reference words: {total_ref_words}")
        print(f"Total edits: {total_edit}")

        # Show some examples
        print("\nSample results (first 5):")
        for r in results[:5]:
            print(f"  {r['utt_id']}: WER={r['wer']:.1f}%")
            if r["ref"] != r["hyp"]:
                print(f"    REF: {r['ref']}")
                print(f"    HYP: {r['hyp']}")

        # The subset WER may be higher due to variance, use a relaxed target
        assert wer < self.TARGET_WER * 1.5, (
            f"Subset WER {wer:.2f}% exceeds relaxed target {self.TARGET_WER * 1.5:.2f}%"
        )

    @pytest.mark.slow
    def test_full_wer(self, pipeline, librispeech_test_clean):
        """Test WER on full LibriSpeech test-clean (~2620 utterances)."""
        transcripts = load_transcripts(librispeech_test_clean)
        audio_files = get_audio_files(librispeech_test_clean)

        total_edit = 0
        total_ref_words = 0
        num_evaluated = 0

        print(f"\nEvaluating {len(audio_files)} utterances...")
        start_time = time.time()

        for i, (utt_id, audio_path) in enumerate(audio_files):
            if utt_id not in transcripts:
                continue

            ref = transcripts[utt_id]
            hyp = pipeline.transcribe(str(audio_path))

            wer_result = compute_wer(ref, hyp)
            total_edit += wer_result["edit_distance"]
            total_ref_words += wer_result["ref_words"]
            num_evaluated += 1

            # Progress every 100 utterances
            if (i + 1) % 100 == 0:
                current_wer = total_edit / max(total_ref_words, 1) * 100
                elapsed = time.time() - start_time
                rate = num_evaluated / elapsed
                print(
                    f"  Progress: {i + 1}/{len(audio_files)} | "
                    f"WER so far: {current_wer:.2f}% | "
                    f"Rate: {rate:.1f} utt/s",
                )

        wer = total_edit / max(total_ref_words, 1) * 100
        elapsed = time.time() - start_time

        print("\n=== Final Results ===")
        print(f"Full WER: {wer:.2f}%")
        print(f"Utterances evaluated: {num_evaluated}")
        print(f"Total reference words: {total_ref_words}")
        print(f"Total edits: {total_edit}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Rate: {num_evaluated / elapsed:.1f} utt/s")
        print(f"Target WER: <{self.TARGET_WER:.2f}%")

        assert wer < self.TARGET_WER, (
            f"Full WER {wer:.2f}% exceeds target {self.TARGET_WER:.2f}%"
        )


class TestBenchmark:
    """Benchmark inference performance."""

    def test_inference_rtf(self, pipeline, librispeech_test_clean):
        """Measure real-time factor on sample utterances."""
        import mlx.core as mx

        audio_files = get_audio_files(librispeech_test_clean)

        # Select a few utterances of varying length
        samples = audio_files[:10]

        total_audio_duration = 0.0
        total_inference_time = 0.0

        # Warm up
        if samples:
            pipeline.transcribe(str(samples[0][1]))
            mx.eval(mx.ones((1,)))

        for _utt_id, audio_path in samples:
            # Get audio duration (16kHz sample rate)
            import soundfile as sf

            audio_data, sr = sf.read(audio_path)
            audio_duration = len(audio_data) / sr
            total_audio_duration += audio_duration

            # Time inference
            start = time.time()
            pipeline.transcribe(str(audio_path))
            mx.eval(mx.ones((1,)))
            end = time.time()

            total_inference_time += end - start

        rtf = total_inference_time / total_audio_duration
        throughput = total_audio_duration / total_inference_time

        print("\n=== RTF Benchmark ===")
        print(f"Total audio: {total_audio_duration:.2f}s")
        print(f"Total inference: {total_inference_time:.2f}s")
        print(f"RTF: {rtf:.4f}x")
        print(f"Throughput: {throughput:.1f}x real-time")

        # Target: better than 10x real-time
        assert throughput > 10, f"Throughput {throughput:.1f}x below target 10x"
