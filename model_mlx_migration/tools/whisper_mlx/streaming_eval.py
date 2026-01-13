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
Streaming Evaluation Framework for WhisperMLX.

Measures streaming-specific metrics required by CONVERSATIONAL_AI_MASTER_PLAN_v3.md:
- Final WER/CER (quality)
- First partial latency (time to first transcribed token)
- Finalization latency (end-of-speech to final transcript)
- Edit rate (how often partial text changes)
- Committed retractions (stable text that later changed)
- Time-to-commit distribution
- RTF and stage timings

Usage:
    python -m tools.whisper_mlx.streaming_eval \
        --test-set librispeech-test-clean \
        --output reports/streaming_eval.json

References:
    - CONVERSATIONAL_AI_MASTER_PLAN_v3.md (Section 6: Evaluation Protocol)
    - whisper_streaming (IWSLT 2025 benchmarks)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# WER computation utilities
def levenshtein_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """
    Compute Levenshtein distance with substitutions, insertions, deletions.

    Returns:
        (distance, substitutions, insertions, deletions)
    """
    m, n = len(ref), len(hyp)

    # dp[i][j] = (cost, subs, ins, dels)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize first row (insertions)
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, j, 0)

    # Initialize first column (deletions)
    for i in range(1, m + 1):
        dp[i][0] = (i, 0, 0, i)

    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                sub_cost = dp[i - 1][j - 1][0] + 1
                sub_ops = (sub_cost, dp[i - 1][j - 1][1] + 1, dp[i - 1][j - 1][2], dp[i - 1][j - 1][3])

                # Insertion
                ins_cost = dp[i][j - 1][0] + 1
                ins_ops = (ins_cost, dp[i][j - 1][1], dp[i][j - 1][2] + 1, dp[i][j - 1][3])

                # Deletion
                del_cost = dp[i - 1][j][0] + 1
                del_ops = (del_cost, dp[i - 1][j][1], dp[i - 1][j][2], dp[i - 1][j][3] + 1)

                # Choose minimum
                dp[i][j] = min([sub_ops, ins_ops, del_ops], key=lambda x: x[0])

    return dp[m][n]


def normalize_text(text: str) -> str:
    """
    Normalize text for WER computation.

    Standard ASR normalization:
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    - Handle common abbreviations (Mr./Mister, etc.)
    """
    import re

    # Lowercase
    text = text.lower()

    # Common abbreviations (expand before removing punctuation)
    abbrev_map = {
        "mr.": "mister",
        "mrs.": "missus",
        "dr.": "doctor",
        "st.": "saint",
        "vs.": "versus",
        "etc.": "etcetera",
        "m.a.": "ma",  # Master of Arts
    }
    for abbrev, expansion in abbrev_map.items():
        text = text.replace(abbrev, expansion)

    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    return " ".join(text.split())



def compute_wer(reference: str, hypothesis: str) -> dict[str, float]:
    """
    Compute Word Error Rate (WER) and related metrics.

    Args:
        reference: Ground truth transcription
        hypothesis: System output transcription

    Returns:
        Dict with WER, substitutions, insertions, deletions
    """
    # Normalize text (lowercase, remove punctuation, normalize abbreviations)
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return {"wer": 0.0, "substitutions": 0, "insertions": 0, "deletions": 0}
        return {"wer": 1.0, "substitutions": 0, "insertions": len(hyp_words), "deletions": 0}

    distance, subs, ins, dels = levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)

    return {
        "wer": wer,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
    }


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER)."""
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    distance, _, _, _ = levenshtein_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)


@dataclass
class PartialResult:
    """Single partial transcription result."""
    timestamp_ms: float  # Wall clock time since audio start
    audio_time_ms: float  # Audio time transcribed so far
    text: str
    is_final: bool
    is_confirmed: bool = False  # LocalAgreement confirmed


@dataclass
class StreamingMetrics:
    """Comprehensive streaming metrics for a single sample."""

    # Sample identification
    sample_id: str = ""
    reference: str = ""
    final_hypothesis: str = ""

    # Quality metrics
    wer: float = 0.0
    cer: float = 0.0

    # Latency metrics (milliseconds)
    first_partial_latency_ms: float = 0.0  # Time to first partial output
    finalization_latency_ms: float = 0.0   # End-of-speech to final output

    # Stability metrics
    edit_count: int = 0          # Number of times partial text changed
    edit_rate: float = 0.0       # Edit count per second of speech
    committed_retractions: int = 0  # Confirmed text that was later changed
    oscillation_count: int = 0   # Number of flip-flop events (text appearing, disappearing, reappearing)
    oscillation_rate: float = 0.0  # Oscillations per second of speech

    # Timing metrics
    time_to_commit_ms: list[float] = field(default_factory=list)  # Per-word commit latencies

    # Compute metrics
    rtf: float = 0.0             # Real-time factor (processing time / audio duration)
    total_processing_ms: float = 0.0
    audio_duration_ms: float = 0.0

    # Partial results trace (for debugging)
    partial_trace: list[PartialResult] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple samples."""

    # Quality
    mean_wer: float = 0.0
    mean_cer: float = 0.0

    # Latency (milliseconds)
    first_partial_latency_median: float = 0.0
    first_partial_latency_p95: float = 0.0
    finalization_latency_median: float = 0.0
    finalization_latency_p95: float = 0.0

    # Stability
    mean_edit_rate: float = 0.0
    committed_retraction_rate: float = 0.0  # Retractions per sample
    mean_oscillation_rate: float = 0.0  # Oscillations per second of speech (target: <0.05/sec)

    # Time to commit distribution
    time_to_commit_median: float = 0.0
    time_to_commit_p95: float = 0.0

    # Compute
    mean_rtf: float = 0.0

    # Sample counts
    num_samples: int = 0
    num_failed: int = 0


class StreamingReplayHarness:
    """
    Replay audio in simulated real-time for streaming evaluation.

    Feeds audio to the streaming transcriber at a controlled rate,
    collecting partial results with precise timing.
    """

    def __init__(
        self,
        streamer,  # StreamingWhisper or DualPathStreamer
        chunk_duration_ms: float = 100.0,  # Chunk size for replay
        speed_factor: float = 1.0,  # 1.0 = real-time, >1 = faster
    ):
        """
        Initialize replay harness.

        Args:
            streamer: Initialized streaming transcriber
            chunk_duration_ms: Audio chunk size in milliseconds
            speed_factor: Replay speed (1.0 = real-time)
        """
        self.streamer = streamer
        self.chunk_duration_ms = chunk_duration_ms
        self.speed_factor = speed_factor
        self.sample_rate = 16000

    async def replay_audio(
        self,
        audio: np.ndarray,
        reference: str = "",
    ) -> StreamingMetrics:
        """
        Replay audio through streaming transcriber and collect metrics.

        Args:
            audio: Audio waveform (float32, 16kHz mono)
            reference: Ground truth transcription

        Returns:
            StreamingMetrics for this sample
        """
        metrics = StreamingMetrics(reference=reference)
        metrics.audio_duration_ms = len(audio) / self.sample_rate * 1000

        # Reset streamer
        self.streamer.reset()

        # Calculate chunk size
        chunk_samples = int(self.chunk_duration_ms * self.sample_rate / 1000)
        num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

        # Timing
        start_time = time.perf_counter()
        audio_start_time = start_time

        # Partial tracking
        partial_results = []
        prev_text = ""
        confirmed_text = ""
        first_partial_time = None

        # Oscillation tracking: track words that disappeared and might reappear
        # Use a buffer to require words to be missing for 2+ consecutive updates
        # This filters out transient changes that would otherwise count as oscillation
        disappeared_words = set()  # Words confirmed as disappeared (missing 2+ updates)
        pending_disappearance = {}  # word -> count of consecutive absences
        current_words = set()

        # Create async audio generator
        async def audio_generator():
            for i in range(num_chunks):
                chunk_start = i * chunk_samples
                chunk_end = min((i + 1) * chunk_samples, len(audio))
                chunk = audio[chunk_start:chunk_end]

                # Simulate real-time delay if not running at max speed
                if self.speed_factor < 100:
                    target_time = audio_start_time + (chunk_end / self.sample_rate) / self.speed_factor
                    now = time.perf_counter()
                    if target_time > now:
                        await asyncio.sleep(target_time - now)

                yield chunk

        # Run streaming transcription
        async for result in self.streamer.transcribe_stream(audio_generator()):
            now = time.perf_counter()
            wall_time_ms = (now - start_time) * 1000
            audio_time_ms = result.segment_end * 1000 if hasattr(result, 'segment_end') else wall_time_ms

            # Record partial result
            partial = PartialResult(
                timestamp_ms=wall_time_ms,
                audio_time_ms=audio_time_ms,
                text=result.text,
                is_final=result.is_final,
                is_confirmed=getattr(result, 'is_confirmed', False),
            )
            partial_results.append(partial)

            # Track first partial
            if first_partial_time is None and result.text.strip():
                first_partial_time = wall_time_ms

            # Track edits
            if result.text != prev_text and not result.is_final:
                metrics.edit_count += 1

                # Oscillation detection: track words appearing/disappearing/reappearing
                # Uses a buffer to avoid counting transient changes as oscillation
                new_words = set(normalize_text(result.text).split()) if result.text.strip() else set()

                # Track words that disappeared (were in previous, not in current)
                newly_disappeared = current_words - new_words
                for word in newly_disappeared:
                    pending_disappearance[word] = pending_disappearance.get(word, 0) + 1
                    # Confirm disappearance after 2 consecutive absences
                    if pending_disappearance[word] >= 2:
                        disappeared_words.add(word)

                # Words that reappeared - reset their pending count
                for word in new_words:
                    if word in pending_disappearance:
                        del pending_disappearance[word]

                # Check for oscillation: words that reappeared after confirmed disappearing
                reappeared = new_words & disappeared_words
                if reappeared:
                    metrics.oscillation_count += len(reappeared)
                    # Remove reappeared words from tracking (they're back now)
                    disappeared_words -= reappeared

                current_words = new_words

            prev_text = result.text

            # Track confirmed text retractions (within-segment only)
            # Note: We only track retractions within a segment, not across segment boundaries.
            # When a final is emitted, the segment ends and confirmed_text resets.
            # Cross-segment changes are not retractions - they're new sentences.
            if hasattr(result, 'confirmed_text') and result.confirmed_text:
                if confirmed_text and not result.confirmed_text.startswith(confirmed_text):
                    metrics.committed_retractions += 1
                confirmed_text = result.confirmed_text

            # Final result - concatenate all segments (streaming may emit multiple finals)
            if result.is_final:
                if metrics.final_hypothesis:
                    metrics.final_hypothesis += " " + result.text
                else:
                    metrics.final_hypothesis = result.text
                metrics.finalization_latency_ms = wall_time_ms - metrics.audio_duration_ms
                # Reset confirmed text tracking for next segment
                # This prevents counting new segment starts as retractions
                confirmed_text = ""

        # Calculate metrics
        total_time = time.perf_counter() - start_time
        metrics.total_processing_ms = total_time * 1000
        metrics.rtf = metrics.total_processing_ms / metrics.audio_duration_ms if metrics.audio_duration_ms > 0 else 0

        # First partial latency
        metrics.first_partial_latency_ms = first_partial_time if first_partial_time else metrics.audio_duration_ms

        # Edit rate (edits per second of speech)
        speech_duration_s = metrics.audio_duration_ms / 1000
        metrics.edit_rate = metrics.edit_count / speech_duration_s if speech_duration_s > 0 else 0

        # Oscillation rate (oscillations per second of speech)
        metrics.oscillation_rate = metrics.oscillation_count / speech_duration_s if speech_duration_s > 0 else 0

        # WER/CER
        if reference:
            wer_metrics = compute_wer(reference, metrics.final_hypothesis)
            metrics.wer = wer_metrics["wer"]
            metrics.cer = compute_cer(reference, metrics.final_hypothesis)

        # Store partial trace
        metrics.partial_trace = partial_results

        return metrics


class StreamingEvaluator:
    """
    Comprehensive streaming evaluation on test sets.

    Implements the evaluation protocol from CONVERSATIONAL_AI_MASTER_PLAN_v3.md.
    """

    def __init__(
        self,
        model,  # WhisperMLX model
        config=None,  # StreamingConfig
        harness_config: dict = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: WhisperMLX model
            config: StreamingConfig (optional)
            harness_config: Configuration for replay harness
        """
        from .streaming import StreamingConfig, StreamingWhisper

        self.model = model
        self.config = config or StreamingConfig()

        # Create streamer
        self.streamer = StreamingWhisper(model, self.config)

        # Harness config
        self.harness_config = harness_config or {
            "chunk_duration_ms": 100.0,
            "speed_factor": 100.0,  # Run as fast as possible
        }

        # Results storage
        self.sample_metrics: list[StreamingMetrics] = []

    async def evaluate_sample(
        self,
        audio: np.ndarray,
        reference: str,
        sample_id: str = "",
    ) -> StreamingMetrics:
        """
        Evaluate a single audio sample.

        Args:
            audio: Audio waveform (float32, 16kHz mono)
            reference: Ground truth transcription
            sample_id: Sample identifier

        Returns:
            StreamingMetrics for this sample
        """
        harness = StreamingReplayHarness(
            self.streamer,
            chunk_duration_ms=self.harness_config.get("chunk_duration_ms", 100.0),
            speed_factor=self.harness_config.get("speed_factor", 100.0),
        )

        metrics = await harness.replay_audio(audio, reference)
        metrics.sample_id = sample_id

        self.sample_metrics.append(metrics)
        return metrics

    def aggregate_metrics(self) -> AggregatedMetrics:
        """
        Aggregate metrics across all evaluated samples.

        Returns:
            AggregatedMetrics with percentile statistics
        """
        if not self.sample_metrics:
            return AggregatedMetrics()

        # Collect arrays for percentile computation
        wers = [m.wer for m in self.sample_metrics]
        cers = [m.cer for m in self.sample_metrics]
        first_partial_latencies = [m.first_partial_latency_ms for m in self.sample_metrics]
        finalization_latencies = [m.finalization_latency_ms for m in self.sample_metrics]
        edit_rates = [m.edit_rate for m in self.sample_metrics]
        oscillation_rates = [m.oscillation_rate for m in self.sample_metrics]
        rtfs = [m.rtf for m in self.sample_metrics]

        # Time to commit (flatten across all samples)
        all_commit_times = []
        for m in self.sample_metrics:
            all_commit_times.extend(m.time_to_commit_ms)

        # Compute aggregates
        agg = AggregatedMetrics(
            num_samples=len(self.sample_metrics),
            num_failed=sum(1 for m in self.sample_metrics if m.wer >= 1.0),

            # Quality
            mean_wer=np.mean(wers),
            mean_cer=np.mean(cers),

            # Latency
            first_partial_latency_median=np.median(first_partial_latencies),
            first_partial_latency_p95=np.percentile(first_partial_latencies, 95),
            finalization_latency_median=np.median(finalization_latencies),
            finalization_latency_p95=np.percentile(finalization_latencies, 95),

            # Stability
            mean_edit_rate=np.mean(edit_rates),
            committed_retraction_rate=sum(m.committed_retractions for m in self.sample_metrics) / len(self.sample_metrics),
            mean_oscillation_rate=np.mean(oscillation_rates),

            # Compute
            mean_rtf=np.mean(rtfs),
        )

        # Time to commit (only if we have data)
        if all_commit_times:
            agg.time_to_commit_median = np.median(all_commit_times)
            agg.time_to_commit_p95 = np.percentile(all_commit_times, 95)

        return agg

    def save_results(self, output_path: str):
        """
        Save evaluation results to JSON.

        Args:
            output_path: Path to output JSON file
        """
        aggregated = self.aggregate_metrics()

        results = {
            "aggregated": asdict(aggregated),
            "config": {
                "sample_rate": self.config.sample_rate,
                "use_vad": self.config.use_vad,
                "use_local_agreement": self.config.use_local_agreement,
                "agreement_n": self.config.agreement_n,
                "min_chunk_duration": self.config.min_chunk_duration,
                "max_chunk_duration": self.config.max_chunk_duration,
                "latency_mode": self.config.latency_mode,
            },
            "samples": [],
        }

        # Add per-sample results (without partial trace to save space)
        for m in self.sample_metrics:
            sample_dict = asdict(m)
            sample_dict.pop("partial_trace", None)  # Remove trace
            results["samples"].append(sample_dict)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")

    def print_summary(self):
        """Print summary of evaluation results."""
        agg = self.aggregate_metrics()

        print("\n" + "=" * 60)
        print("STREAMING EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nSamples: {agg.num_samples} ({agg.num_failed} failed)")

        print("\n--- Quality ---")
        print(f"  Mean WER:        {agg.mean_wer * 100:.2f}%")
        print(f"  Mean CER:        {agg.mean_cer * 100:.2f}%")

        print("\n--- Latency (ms) ---")
        print(f"  First partial:   median={agg.first_partial_latency_median:.0f}  p95={agg.first_partial_latency_p95:.0f}")
        print(f"  Finalization:    median={agg.finalization_latency_median:.0f}  p95={agg.finalization_latency_p95:.0f}")

        print("\n--- Stability ---")
        print(f"  Edit rate:       {agg.mean_edit_rate:.2f} edits/sec")
        print(f"  Retractions:     {agg.committed_retraction_rate:.3f} per sample")
        oscillation_status = "PASS" if agg.mean_oscillation_rate < 0.05 else "FAIL"
        print(f"  Oscillation:     {agg.mean_oscillation_rate:.3f}/sec (target <0.05) [{oscillation_status}]")

        if agg.time_to_commit_median > 0:
            print("\n--- Time to Commit (ms) ---")
            print(f"  Median:          {agg.time_to_commit_median:.0f}")
            print(f"  P95:             {agg.time_to_commit_p95:.0f}")

        print("\n--- Compute ---")
        print(f"  Mean RTF:        {agg.mean_rtf:.3f}")

        print("\n" + "=" * 60)


def load_test_set(name: str, data_dir: str = "data") -> Iterator[tuple[np.ndarray, str, str]]:
    """
    Load a test set for evaluation.

    Args:
        name: Test set name (e.g., "librispeech-test-clean")
        data_dir: Root data directory

    Yields:
        (audio, reference, sample_id) tuples
    """
    from .audio import load_audio

    data_path = Path(data_dir)

    # LibriSpeech test sets
    if name.startswith("librispeech"):
        # Try multiple potential locations
        libri_paths = [
            data_path / "librispeech" / name.replace("librispeech-", ""),
            data_path / "multilingual" / "english" / name.replace("librispeech-", ""),
            data_path / name,
        ]

        for libri_path in libri_paths:
            if not libri_path.exists():
                continue

            # Look for transcripts file
            trans_file = libri_path / "transcripts.txt"
            if trans_file.exists():
                transcripts = {}
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            transcripts[parts[0]] = parts[1]

                # Find audio files
                for audio_file in sorted(libri_path.rglob("*.flac")):
                    sample_id = audio_file.stem
                    if sample_id in transcripts:
                        audio = load_audio(str(audio_file))
                        yield audio, transcripts[sample_id], sample_id
            else:
                # Try individual transcript files
                for audio_file in sorted(libri_path.rglob("*.flac")):
                    txt_file = audio_file.with_suffix(".txt")
                    if txt_file.exists():
                        with open(txt_file) as f:
                            reference = f.read().strip()
                        audio = load_audio(str(audio_file))
                        yield audio, reference, audio_file.stem
            return

    # JSON manifest format
    manifest_path = data_path / f"{name}.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                audio_path = entry.get("audio_filepath") or entry.get("path")
                reference = entry.get("text") or entry.get("transcript", "")
                sample_id = entry.get("id", Path(audio_path).stem)

                if not Path(audio_path).is_absolute():
                    audio_path = str(data_path / audio_path)

                audio = load_audio(audio_path)
                yield audio, reference, sample_id
        return

    raise ValueError(f"Test set not found: {name}")


async def main():
    """Main entry point for streaming evaluation."""
    parser = argparse.ArgumentParser(description="Streaming evaluation for WhisperMLX")

    parser.add_argument("--test-set", type=str, default="librispeech-test-clean",
                        help="Test set name")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--model", type=str, default="mlx-community/whisper-large-v3-mlx",
                        help="Model name or path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    parser.add_argument("--latency-mode", type=str, default="balanced",
                        choices=["fast", "balanced", "quality"],
                        help="Latency mode")
    parser.add_argument("--no-local-agreement", action="store_true",
                        help="Disable LocalAgreement")
    parser.add_argument("--speed-factor", type=float, default=100.0,
                        help="Replay speed (1.0=real-time, higher=faster)")

    args = parser.parse_args()

    # Import here to avoid circular imports
    from .model import WhisperMLX
    from .streaming import StreamingConfig

    print(f"Loading model: {args.model}")
    model = WhisperMLX.from_pretrained(args.model)

    # Configure streaming
    config = StreamingConfig(
        use_local_agreement=not args.no_local_agreement,
        latency_mode=args.latency_mode,
    )

    # Create evaluator
    evaluator = StreamingEvaluator(
        model,
        config,
        harness_config={"speed_factor": args.speed_factor},
    )

    print(f"Loading test set: {args.test_set}")

    # Evaluate samples
    count = 0
    for audio, reference, sample_id in load_test_set(args.test_set, args.data_dir):
        if args.max_samples and count >= args.max_samples:
            break

        print(f"Evaluating {sample_id}...", end=" ", flush=True)
        metrics = await evaluator.evaluate_sample(audio, reference, sample_id)
        print(f"WER={metrics.wer * 100:.1f}% RTF={metrics.rtf:.2f}")

        count += 1

    if count == 0:
        print("ERROR: No samples found. Check test set path.")
        return

    # Print summary
    evaluator.print_summary()

    # Save results
    if args.output:
        evaluator.save_results(args.output)
    else:
        # Default output path
        output_path = f"reports/streaming_eval_{args.test_set}_{args.latency_mode}.json"
        evaluator.save_results(output_path)


if __name__ == "__main__":
    asyncio.run(main())
