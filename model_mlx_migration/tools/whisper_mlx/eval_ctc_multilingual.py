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
Multilingual CTC Model Evaluation.

Evaluates a trained CTC head checkpoint on multiple language test sets.
Reports WER per language and overall.

Usage:
    # Evaluate on all available test sets
    python -m tools.whisper_mlx.eval_ctc_multilingual \
        --checkpoint checkpoints/ctc_multilingual_v1/step_11000.npz

    # Evaluate on specific languages
    python -m tools.whisper_mlx.eval_ctc_multilingual \
        --checkpoint checkpoints/ctc_multilingual_v1/step_11000.npz \
        --languages en ja zh

    # Quick test (10 samples per language)
    python -m tools.whisper_mlx.eval_ctc_multilingual \
        --checkpoint checkpoints/ctc_multilingual_v1/step_11000.npz \
        --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

# =============================================================================
# WER Computation
# =============================================================================

def levenshtein_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Compute Levenshtein distance with substitutions, insertions, deletions."""
    m, n = len(ref), len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]

    for j in range(1, n + 1):
        dp[0][j] = (j, 0, j, 0)
    for i in range(1, m + 1):
        dp[i][0] = (i, 0, 0, i)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_cost = dp[i - 1][j - 1][0] + 1
                sub_ops = (sub_cost, dp[i - 1][j - 1][1] + 1, dp[i - 1][j - 1][2], dp[i - 1][j - 1][3])
                ins_cost = dp[i][j - 1][0] + 1
                ins_ops = (ins_cost, dp[i][j - 1][1], dp[i][j - 1][2] + 1, dp[i][j - 1][3])
                del_cost = dp[i - 1][j][0] + 1
                del_ops = (del_cost, dp[i - 1][j][1], dp[i - 1][j][2], dp[i - 1][j][3] + 1)
                dp[i][j] = min(sub_ops, ins_ops, del_ops, key=lambda x: x[0])

    return dp[m][n]


def compute_wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """Compute Word Error Rate."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0, 0, len(hyp_words), 0, 0

    _, subs, ins, dels = levenshtein_distance(ref_words, hyp_words)
    wer = (subs + ins + dels) / len(ref_words)
    return wer, subs, ins, dels, len(ref_words)


# =============================================================================
# Test Data Sources
# =============================================================================

@dataclass
class TestSample:
    """A test sample with audio path and transcript."""
    audio_path: str
    transcript: str
    language: str
    sample_id: str


# Test set configurations: (base_path, format, language, test_split)
TEST_SOURCES = [
    # English - LibriSpeech test-clean
    ("data/LibriSpeech/test-clean", "librispeech", "en"),
    # Japanese - CommonVoice test
    ("data/commonvoice/cv-corpus-24.0-2025-12-05/ja", "commonvoice_test", "ja"),
    # Chinese - AISHELL test
    ("data/openslr/zh/aishell", "aishell_test", "zh"),
    # German - MLS test
    ("data/mls/mls_german_opus", "mls_test", "de"),
    # French - MLS test
    ("data/mls/mls_french_opus", "mls_test", "fr"),
    # Spanish - MLS test
    ("data/mls/mls_spanish_opus", "mls_test", "es"),
    # Russian - test split from manifest
    ("data/openslr/ru/russian_librispeech", "manifest_test", "ru"),
    # Korean - OpenSLR
    ("data/openslr/ko", "librispeech", "ko"),
    # Hindi - MUCS test
    ("data/multilingual/hindi_mucs/test", "hindi_mucs", "hi"),
]


def load_librispeech_test(base_path: Path, max_samples: int = 0) -> list[TestSample]:
    """Load LibriSpeech format test samples."""
    samples = []
    transcript_files = list(base_path.rglob("*.txt"))

    for trans_file in transcript_files:
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, transcript = parts
                audio_dir = trans_file.parent
                audio_path = audio_dir / f"{utt_id}.flac"
                if audio_path.exists():
                    samples.append(TestSample(
                        audio_path=str(audio_path),
                        transcript=transcript,
                        language="en",
                        sample_id=utt_id,
                    ))
                    if max_samples > 0 and len(samples) >= max_samples:
                        return samples

    return samples


def load_commonvoice_test(base_path: Path, language: str, max_samples: int = 0) -> list[TestSample]:
    """Load CommonVoice test split."""
    import csv
    samples = []
    test_tsv = base_path / "test.tsv"

    if not test_tsv.exists():
        return samples

    clips_dir = base_path / "clips"
    with open(test_tsv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            audio_path = clips_dir / row['path']
            if audio_path.exists():
                samples.append(TestSample(
                    audio_path=str(audio_path),
                    transcript=row['sentence'],
                    language=language,
                    sample_id=row['path'],
                ))
                if max_samples > 0 and len(samples) >= max_samples:
                    return samples

    return samples


def load_aishell_test(base_path: Path, max_samples: int = 0) -> list[TestSample]:
    """Load AISHELL test split."""
    samples = []
    transcript_file = base_path / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
    test_dir = base_path / "data_aishell" / "wav" / "test"

    if not transcript_file.exists() or not test_dir.exists():
        return samples

    # Load transcripts
    transcripts = {}
    with open(transcript_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id] = text

    # Find test audio files
    for audio_file in test_dir.rglob("*.wav"):
        utt_id = audio_file.stem
        if utt_id in transcripts:
            samples.append(TestSample(
                audio_path=str(audio_file),
                transcript=transcripts[utt_id],
                language="zh",
                sample_id=utt_id,
            ))
            if max_samples > 0 and len(samples) >= max_samples:
                return samples

    return samples


def load_mls_test(base_path: Path, language: str, max_samples: int = 0) -> list[TestSample]:
    """Load MLS test split."""
    samples = []
    test_dir = base_path / "test"
    transcript_file = test_dir / "transcripts.txt"

    if not transcript_file.exists():
        return samples

    with open(transcript_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                utt_id, transcript = parts[0], parts[1]
                # MLS uses speaker/book/segment structure
                parts = utt_id.split('_')
                if len(parts) >= 3:
                    speaker, book = parts[0], parts[1]
                    audio_path = test_dir / "audio" / speaker / book / f"{utt_id}.opus"
                    if audio_path.exists():
                        samples.append(TestSample(
                            audio_path=str(audio_path),
                            transcript=transcript,
                            language=language,
                            sample_id=utt_id,
                        ))
                        if max_samples > 0 and len(samples) >= max_samples:
                            return samples

    return samples


def load_manifest_test(base_path: Path, language: str, max_samples: int = 0) -> list[TestSample]:
    """Load manifest format test split (JSONL)."""
    samples = []
    test_manifest = base_path / "test" / "manifest.jsonl"

    if not test_manifest.exists():
        # Try alternative location
        test_manifest = base_path / "manifest_test.jsonl"

    if not test_manifest.exists():
        return samples

    with open(test_manifest, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            audio_path = base_path / entry.get('audio_filepath', entry.get('audio', ''))
            if audio_path.exists():
                samples.append(TestSample(
                    audio_path=str(audio_path),
                    transcript=entry.get('text', entry.get('transcript', '')),
                    language=language,
                    sample_id=audio_path.stem,
                ))
                if max_samples > 0 and len(samples) >= max_samples:
                    return samples

    return samples


def load_hindi_mucs_test(base_path: Path, max_samples: int = 0) -> list[TestSample]:
    """Load Hindi MUCS test split."""
    samples = []

    if not base_path.exists():
        return samples

    for audio_file in base_path.rglob("*.wav"):
        transcript_file = audio_file.with_suffix('.txt')
        if transcript_file.exists():
            with open(transcript_file, encoding='utf-8') as f:
                transcript = f.read().strip()
            samples.append(TestSample(
                audio_path=str(audio_file),
                transcript=transcript,
                language="hi",
                sample_id=audio_file.stem,
            ))
            if max_samples > 0 and len(samples) >= max_samples:
                return samples

    return samples


def load_test_samples(languages: list[str] | None, max_samples: int = 0) -> dict[str, list[TestSample]]:
    """Load test samples for specified languages."""
    samples_by_lang: dict[str, list[TestSample]] = {}

    for base_path, format_type, language in TEST_SOURCES:
        if languages and language not in languages:
            continue

        path = Path(base_path)
        if not path.exists():
            print(f"  [MISSING] {base_path}")
            continue

        print(f"  [LOADING] {base_path} ({language})")

        if format_type == "librispeech":
            samples = load_librispeech_test(path, max_samples)
        elif format_type == "commonvoice_test":
            samples = load_commonvoice_test(path, language, max_samples)
        elif format_type == "aishell_test":
            samples = load_aishell_test(path, max_samples)
        elif format_type == "mls_test":
            samples = load_mls_test(path, language, max_samples)
        elif format_type == "manifest_test":
            samples = load_manifest_test(path, language, max_samples)
        elif format_type == "hindi_mucs":
            samples = load_hindi_mucs_test(path, max_samples)
        else:
            samples = []

        if samples:
            if language not in samples_by_lang:
                samples_by_lang[language] = []
            samples_by_lang[language].extend(samples)
            print(f"    Loaded {len(samples)} samples")

    return samples_by_lang


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class LanguageResult:
    """Results for a single language."""
    language: str
    num_samples: int
    total_ref_words: int
    total_substitutions: int
    total_insertions: int
    total_deletions: int
    wer: float
    total_audio_duration_s: float
    total_inference_time_s: float
    rtf: float


def unflatten_params(flat_params: dict) -> dict:
    """Convert flat parameter dict to nested dict."""
    nested = {}
    for key, value in flat_params.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested


def evaluate_language(
    samples: list[TestSample],
    whisper_model,
    ctc_head,
    tokenizer,
    language: str,
    verbose: bool = False,
) -> LanguageResult:
    """Evaluate CTC model on samples from a single language."""
    import soundfile as sf
    from scipy import signal

    from .audio import log_mel_spectrogram

    total_ref_words = 0
    total_subs = 0
    total_ins = 0
    total_dels = 0
    total_audio_duration = 0.0
    total_inference_time = 0.0

    for i, sample in enumerate(samples):
        try:
            # Load audio
            audio, sr = sf.read(sample.audio_path)
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert stereo to mono
            if sr != 16000:
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))
                sr = 16000

            audio_duration = len(audio) / sr
            total_audio_duration += audio_duration

            # Run inference
            t0 = time.time()
            mel = log_mel_spectrogram(audio, n_mels=whisper_model.config.n_mels)
            mel_tensor = mx.expand_dims(mx.array(mel), axis=0)
            encoder_output = whisper_model.encoder(mel_tensor, variable_length=True)
            mx.eval(encoder_output)
            logits = ctc_head(encoder_output)
            mx.eval(logits)
            tokens = ctc_head.decode_greedy(logits)
            inference_time = time.time() - t0
            total_inference_time += inference_time

            # Decode to text
            hypothesis = tokenizer.decode(tokens)

            # Compute WER
            wer, subs, ins, dels, ref_words = compute_wer(sample.transcript, hypothesis)
            total_ref_words += ref_words
            total_subs += subs
            total_ins += ins
            total_dels += dels

            if verbose:
                print(f"  [{i+1}/{len(samples)}] WER: {wer:.2%}")
                print(f"    Ref: {sample.transcript[:60]}...")
                print(f"    Hyp: {hypothesis[:60]}...")

            # Progress
            if not verbose and (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(samples)} samples...")

        except Exception as e:
            print(f"  Error processing {sample.sample_id}: {e}")
            continue

    # Compute overall WER
    if total_ref_words > 0:
        overall_wer = (total_subs + total_ins + total_dels) / total_ref_words
    else:
        overall_wer = 0.0

    rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0

    return LanguageResult(
        language=language,
        num_samples=len(samples),
        total_ref_words=total_ref_words,
        total_substitutions=total_subs,
        total_insertions=total_ins,
        total_deletions=total_dels,
        wer=overall_wer,
        total_audio_duration_s=total_audio_duration,
        total_inference_time_s=total_inference_time,
        rtf=rtf,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CTC model on multilingual test sets",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to CTC head checkpoint (.npz file)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        help="Languages to evaluate (default: all available)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples per language (0 = all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MULTILINGUAL CTC MODEL EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model size: {args.model_size}")
    print(f"Languages: {args.languages or 'all available'}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print("=" * 70)

    # Load Whisper model
    print("\n1. Loading Whisper encoder...")
    from .model import WhisperMLX
    from .tokenizer import get_whisper_tokenizer

    model_name = f"mlx-community/whisper-{args.model_size}-mlx"
    whisper_model = WhisperMLX.from_pretrained(model_name)
    tokenizer = get_whisper_tokenizer()
    print(f"   Loaded {model_name}")

    # Load CTC head
    print("\n2. Loading CTC head checkpoint...")
    from .ctc_head import create_ctc_draft_head

    ctc_head = create_ctc_draft_head(args.model_size)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"   ERROR: Checkpoint not found: {args.checkpoint}")
        return

    flat_params = mx.load(str(checkpoint_path))
    nested_params = unflatten_params(flat_params)
    ctc_head.update(nested_params)
    print(f"   Loaded {len(flat_params)} parameters from {args.checkpoint}")

    # Load test samples
    print("\n3. Loading test samples...")
    samples_by_lang = load_test_samples(args.languages, args.max_samples)

    if not samples_by_lang:
        print("   ERROR: No test samples found!")
        return

    total_samples = sum(len(s) for s in samples_by_lang.values())
    print(f"   Total: {total_samples} samples across {len(samples_by_lang)} languages")

    # Warmup
    print("\n4. Warmup...")
    first_lang = next(iter(samples_by_lang.keys()))
    first_sample = samples_by_lang[first_lang][0]
    import soundfile as sf

    from .audio import log_mel_spectrogram

    audio, sr = sf.read(first_sample.audio_path)
    audio = audio.astype(np.float32)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    mel = log_mel_spectrogram(audio, n_mels=whisper_model.config.n_mels)
    mel_tensor = mx.expand_dims(mx.array(mel), axis=0)
    encoder_output = whisper_model.encoder(mel_tensor, variable_length=True)
    mx.eval(encoder_output)
    logits = ctc_head(encoder_output)
    mx.eval(logits)
    _ = ctc_head.decode_greedy(logits)
    print("   Done")

    # Evaluate each language
    print("\n5. Evaluating...")
    results = []

    for language, samples in samples_by_lang.items():
        print(f"\n   [{language.upper()}] Evaluating {len(samples)} samples...")
        result = evaluate_language(
            samples, whisper_model, ctc_head, tokenizer, language, args.verbose,
        )
        results.append(result)
        print(f"   [{language.upper()}] WER: {result.wer:.2%} ({result.num_samples} samples, RTF: {result.rtf:.3f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Language':<10} {'Samples':>8} {'WER':>10} {'S':>6} {'I':>6} {'D':>6} {'RTF':>8}")
    print("-" * 70)

    total_ref_words = 0
    total_errors = 0

    for r in results:
        errors = r.total_substitutions + r.total_insertions + r.total_deletions
        total_ref_words += r.total_ref_words
        total_errors += errors
        print(f"{r.language:<10} {r.num_samples:>8} {r.wer:>9.2%} {r.total_substitutions:>6} {r.total_insertions:>6} {r.total_deletions:>6} {r.rtf:>8.3f}")

    print("-" * 70)
    overall_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0
    print(f"{'OVERALL':<10} {total_samples:>8} {overall_wer:>9.2%}")
    print("=" * 70)

    # Save results
    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "model_size": args.model_size,
            "overall_wer": overall_wer,
            "total_samples": total_samples,
            "languages": {
                r.language: {
                    "wer": r.wer,
                    "num_samples": r.num_samples,
                    "total_ref_words": r.total_ref_words,
                    "substitutions": r.total_substitutions,
                    "insertions": r.total_insertions,
                    "deletions": r.total_deletions,
                    "rtf": r.rtf,
                }
                for r in results
            },
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
