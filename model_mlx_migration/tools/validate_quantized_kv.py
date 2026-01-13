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
Validate INT8 KV Cache Quantization for WhisperMLX (OPT-2.3).

This script validates that INT8 quantization of the cross-attention KV cache
produces acceptable accuracy degradation (<0.1% WER impact).

Usage:
    python tools/validate_quantized_kv.py --audio-dir data/prosody/ravdess --limit 100
    python tools/validate_quantized_kv.py --audio-dir data/prosody/ravdess --full

Output:
    - Console summary with pass/fail metrics
    - QUANTIZATION_AUDIT_RESULTS.md with detailed results
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TranscriptionResult:
    """Result of a single transcription."""
    audio_path: str
    text: str
    time_ms: float
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


@dataclass
class ComparisonResult:
    """Comparison of baseline vs quantized transcription."""
    audio_path: str
    baseline_text: str
    quantized_text: str
    exact_match: bool
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    baseline_time_ms: float
    quantized_time_ms: float
    speedup: float  # quantized is faster if > 1.0


@dataclass
class AuditResults:
    """Aggregate results from the audit."""
    total_files: int = 0
    exact_matches: int = 0
    wer_sum: float = 0.0
    cer_sum: float = 0.0
    baseline_time_sum: float = 0.0
    quantized_time_sum: float = 0.0
    comparisons: list[ComparisonResult] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (path, error_msg)

    @property
    def exact_match_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.exact_matches / self.total_files

    @property
    def avg_wer(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.wer_sum / self.total_files

    @property
    def avg_cer(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.cer_sum / self.total_files

    @property
    def avg_speedup(self) -> float:
        if self.baseline_time_sum == 0:
            return 0.0
        return self.baseline_time_sum / self.quantized_time_sum


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N
    Where S=substitutions, D=deletions, I=insertions, N=reference words
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,    # deletion
                    d[i][j-1] + 1,    # insertion
                    d[i-1][j-1] + 1,   # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate between reference and hypothesis.
    """
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,
                    d[i][j-1] + 1,
                    d[i-1][j-1] + 1,
                )

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def transcribe_audio(model, audio_path: str) -> TranscriptionResult:
    """Transcribe a single audio file and return result with timing."""
    t0 = time.perf_counter()
    result = model.transcribe(audio_path, verbose=False)
    time_ms = (time.perf_counter() - t0) * 1000

    return TranscriptionResult(
        audio_path=audio_path,
        text=result.get("text", ""),
        time_ms=time_ms,
        avg_logprob=result.get("avg_logprob", float("nan")),
        no_speech_prob=result.get("no_speech_prob", float("nan")),
        compression_ratio=result.get("compression_ratio", float("nan")),
    )


def compare_transcriptions(
    baseline: TranscriptionResult,
    quantized: TranscriptionResult,
) -> ComparisonResult:
    """Compare baseline and quantized transcription results."""
    exact_match = baseline.text.strip() == quantized.text.strip()
    wer = calculate_wer(baseline.text, quantized.text)
    cer = calculate_cer(baseline.text, quantized.text)
    speedup = baseline.time_ms / quantized.time_ms if quantized.time_ms > 0 else 0.0

    return ComparisonResult(
        audio_path=baseline.audio_path,
        baseline_text=baseline.text,
        quantized_text=quantized.text,
        exact_match=exact_match,
        wer=wer,
        cer=cer,
        baseline_time_ms=baseline.time_ms,
        quantized_time_ms=quantized.time_ms,
        speedup=speedup,
    )


def run_audit(
    audio_dir: str,
    limit: int | None = None,
    model_name: str = "mlx-community/whisper-large-v3-mlx",
    verbose: bool = False,
) -> AuditResults:
    """
    Run the quantization audit on audio files.

    Args:
        audio_dir: Directory containing audio files
        limit: Maximum number of files to process (None for all)
        model_name: Whisper model to use
        verbose: Print progress for each file

    Returns:
        AuditResults with all comparison data
    """
    from tools.whisper_mlx import WhisperMLX

    # Find all audio files
    audio_path = Path(audio_dir)
    audio_files = list(audio_path.rglob("*.wav"))

    if limit is not None:
        audio_files = audio_files[:limit]

    print(f"Found {len(audio_files)} audio files")
    print("Loading baseline model (quantize_kv=False)...")

    # Load baseline model (no KV quantization)
    baseline_model = WhisperMLX.from_pretrained(
        model_name,
        quantize_kv=False,
        warmup=True,
    )

    print("Loading quantized model (quantize_kv=True)...")

    # Load quantized model (with KV quantization)
    quantized_model = WhisperMLX.from_pretrained(
        model_name,
        quantize_kv=True,
        warmup=True,
    )

    results = AuditResults()

    print(f"\nProcessing {len(audio_files)} files...")
    print("-" * 60)

    for i, audio_file in enumerate(audio_files):
        audio_path_str = str(audio_file)

        try:
            # Transcribe with baseline
            baseline_result = transcribe_audio(baseline_model, audio_path_str)

            # Transcribe with quantized
            quantized_result = transcribe_audio(quantized_model, audio_path_str)

            # Compare
            comparison = compare_transcriptions(baseline_result, quantized_result)

            # Update aggregate stats
            results.total_files += 1
            if comparison.exact_match:
                results.exact_matches += 1
            results.wer_sum += comparison.wer
            results.cer_sum += comparison.cer
            results.baseline_time_sum += comparison.baseline_time_ms
            results.quantized_time_sum += comparison.quantized_time_ms
            results.comparisons.append(comparison)

            if verbose:
                status = "MATCH" if comparison.exact_match else f"WER={comparison.wer:.2%}"
                print(f"[{i+1}/{len(audio_files)}] {audio_file.name}: {status}")
            elif (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(audio_files)}] Processed...")

        except Exception as e:
            results.errors.append((audio_path_str, str(e)))
            if verbose:
                print(f"[{i+1}/{len(audio_files)}] {audio_file.name}: ERROR - {e}")

    print("-" * 60)
    print(f"Completed: {results.total_files} files, {len(results.errors)} errors")

    return results


def generate_report(results: AuditResults, output_path: str) -> None:
    """Generate markdown report from audit results."""

    # Find worst cases (highest WER)
    worst_cases = sorted(
        results.comparisons,
        key=lambda x: x.wer,
        reverse=True,
    )[:10]

    report = f"""# INT8 KV Cache Quantization Audit Results (OPT-2.3)

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Files | {results.total_files} |
| Exact Matches | {results.exact_matches} ({results.exact_match_rate:.1%}) |
| Average WER | {results.avg_wer:.4%} |
| Average CER | {results.avg_cer:.4%} |
| Errors | {len(results.errors)} |

## Performance

| Metric | Value |
|--------|-------|
| Baseline Total Time | {results.baseline_time_sum/1000:.2f}s |
| Quantized Total Time | {results.quantized_time_sum/1000:.2f}s |
| Average Speedup | {results.avg_speedup:.2f}x |

## Quality Assessment

**Target: <0.1% WER degradation**

- Measured WER: **{results.avg_wer:.4%}**
- Status: **{'PASS' if results.avg_wer < 0.001 else 'FAIL' if results.avg_wer >= 0.01 else 'ACCEPTABLE'}**

### Interpretation

- WER < 0.1%: Excellent - negligible quality loss
- WER 0.1% - 1%: Acceptable - minor quality loss
- WER > 1%: Concerning - may need investigation

## Worst Cases (Highest WER)

"""

    if worst_cases:
        report += "| File | WER | Baseline | Quantized |\n"
        report += "|------|-----|----------|----------|\n"
        for case in worst_cases[:10]:
            filename = Path(case.audio_path).name
            baseline_preview = case.baseline_text[:30] + "..." if len(case.baseline_text) > 30 else case.baseline_text
            quantized_preview = case.quantized_text[:30] + "..." if len(case.quantized_text) > 30 else case.quantized_text
            report += f"| {filename} | {case.wer:.2%} | {baseline_preview} | {quantized_preview} |\n"
    else:
        report += "No transcription differences found.\n"

    if results.errors:
        report += "\n## Errors\n\n"
        for path, error in results.errors[:10]:
            report += f"- `{Path(path).name}`: {error}\n"
        if len(results.errors) > 10:
            report += f"\n... and {len(results.errors) - 10} more errors\n"

    report += f"""
## Conclusion

INT8 KV cache quantization (OPT-2.3) provides:
- **{(1 - results.quantized_time_sum/results.baseline_time_sum)*100:.1f}% speedup** (or {results.avg_speedup:.2f}x faster)
- **{results.exact_match_rate:.1%} exact match rate**
- **{results.avg_wer:.4%} average WER** degradation

{'**RECOMMENDATION: SAFE TO ENABLE** - Quality impact is minimal.' if results.avg_wer < 0.01 else '**RECOMMENDATION: INVESTIGATE** - Quality impact exceeds threshold.'}
"""

    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate INT8 KV Cache Quantization for WhisperMLX",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data/prosody/ravdess",
        help="Directory containing audio files (default: data/prosody/ravdess)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of files to process (default: 100)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all files (no limit)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-large-v3-mlx",
        help="Whisper model to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/main/QUANTIZATION_AUDIT_RESULTS.md",
        help="Output report path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress for each file",
    )

    args = parser.parse_args()

    limit = None if args.full else args.limit

    print("=" * 60)
    print("INT8 KV Cache Quantization Audit (OPT-2.3)")
    print("=" * 60)
    print(f"Audio directory: {args.audio_dir}")
    print(f"File limit: {'All' if limit is None else limit}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Run audit
    results = run_audit(
        audio_dir=args.audio_dir,
        limit=limit,
        model_name=args.model,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {results.total_files}")
    print(f"Exact matches: {results.exact_matches} ({results.exact_match_rate:.1%})")
    print(f"Average WER: {results.avg_wer:.4%}")
    print(f"Average CER: {results.avg_cer:.4%}")
    print(f"Speedup: {results.avg_speedup:.2f}x")

    # Quality assessment
    print("\n" + "-" * 60)
    if results.avg_wer < 0.001:
        print("QUALITY: EXCELLENT (<0.1% WER)")
    elif results.avg_wer < 0.01:
        print("QUALITY: ACCEPTABLE (0.1%-1% WER)")
    else:
        print("QUALITY: CONCERNING (>1% WER)")
    print("-" * 60)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate report
    generate_report(results, str(output_path))

    # Return exit code based on quality
    if results.avg_wer >= 0.01:
        print("\nWARNING: Quality degradation exceeds 1% WER threshold")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
