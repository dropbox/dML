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
Combined evaluation of novel streaming ASR techniques.

Evaluates:
1. Round-Trip Verification - TTS-based confidence scoring
2. Prosody-Conditioned Beam Search - Pitch/emotion informed punctuation
3. Emotion-Aware Punctuation - Trained punctuation head with emotion features

Metrics:
- WER (Word Error Rate)
- Punctuation F1 per class (., ?, !, ,)
- Retraction rate (for streaming)
- RTF (Real-Time Factor)

Usage:
    python scripts/evaluate_novel_techniques.py \
        --test-set librispeech-test-clean \
        --enable-roundtrip \
        --enable-prosody-beam \
        --enable-emotion-punct \
        --output reports/main/NOVEL_TECHNIQUES_RESULTS.md
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvaluationConfig:
    """Configuration for combined evaluation."""
    test_set: str = "librispeech-test-clean"
    max_samples: int = 100

    # Technique toggles
    enable_baseline: bool = True
    enable_roundtrip: bool = False
    enable_prosody_beam: bool = False
    enable_emotion_punct: bool = False

    # Model paths
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    pitch_checkpoint: Optional[str] = "checkpoints/pitch_combined_v4/best.npz"
    emotion_checkpoint: Optional[str] = "checkpoints/emotion_unified_v2/best.npz"
    punct_checkpoint: Optional[str] = None  # Will be set after MELD training

    # Prosody beam search config
    beam_size: int = 3

    # Output
    output_path: Optional[str] = None
    verbose: bool = False


@dataclass
class PunctuationMetrics:
    """Per-class punctuation metrics."""
    period_precision: float = 0.0
    period_recall: float = 0.0
    period_f1: float = 0.0

    question_precision: float = 0.0
    question_recall: float = 0.0
    question_f1: float = 0.0

    exclamation_precision: float = 0.0
    exclamation_recall: float = 0.0
    exclamation_f1: float = 0.0

    comma_precision: float = 0.0
    comma_recall: float = 0.0
    comma_f1: float = 0.0

    macro_f1: float = 0.0


@dataclass
class EvaluationResult:
    """Results from evaluation run."""
    technique: str
    wer: float = 0.0
    punctuation: PunctuationMetrics = field(default_factory=PunctuationMetrics)
    retraction_rate: float = 0.0
    rtf: float = 0.0
    samples_evaluated: int = 0
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0


class NovelTechniquesEvaluator:
    """Evaluator for combined novel ASR techniques."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.prosody_search = None
        self.roundtrip_verifier = None
        self.punct_head = None

        # Metrics tracking
        self.results: Dict[str, EvaluationResult] = {}

    def setup(self):
        """Initialize models and components."""
        print("Setting up evaluation...")

        # Load Whisper model
        print(f"  Loading Whisper model: {self.config.whisper_model}")
        from tools.whisper_mlx import WhisperMLX
        self.model = WhisperMLX.from_pretrained(self.config.whisper_model)

        # Setup tokenizer
        from tools.whisper_mlx.tokenizer import get_whisper_tokenizer
        is_multilingual = self.config.whisper_model and "large" in self.config.whisper_model
        self.tokenizer = get_whisper_tokenizer(
            multilingual=is_multilingual,
            num_languages=100 if is_multilingual else 0,
            language="en",
            task="transcribe",
        )

        # Setup prosody beam search if enabled
        if self.config.enable_prosody_beam:
            print("  Setting up prosody beam search...")
            self._setup_prosody_beam()

        # Setup round-trip verification if enabled
        if self.config.enable_roundtrip:
            print("  Setting up round-trip verification...")
            self._setup_roundtrip()

        # Setup emotion-aware punctuation if enabled
        if self.config.enable_emotion_punct:
            print("  Setting up emotion-aware punctuation...")
            self._setup_emotion_punct()

        print("Setup complete")

    def _setup_prosody_beam(self):
        """Initialize prosody beam search components."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyBeamSearch, ProsodyBoostConfig
        from tools.whisper_mlx.multi_head import MultiHeadConfig, CREPEPitchHead, EmotionHead

        # Create pitch and emotion heads if checkpoints available
        pitch_head = None
        emotion_head = None

        d_model = 1280  # Large model dimension

        if self.config.pitch_checkpoint and os.path.exists(self.config.pitch_checkpoint):
            print(f"    Loading pitch head from {self.config.pitch_checkpoint}")
            try:
                multi_config = MultiHeadConfig(d_model=d_model, use_crepe_pitch=True)
                pitch_head = CREPEPitchHead(multi_config)

                # Load weights, filtering for pitch-related parameters
                weights = mx.load(self.config.pitch_checkpoint)
                pitch_weights = {
                    k.replace("pitch.", ""): v
                    for k, v in weights.items()
                    if k.startswith("pitch.")
                }
                if pitch_weights:
                    pitch_head.load_weights(list(pitch_weights.items()))
                    print(f"    Pitch head loaded ({len(pitch_weights)} params)")
            except Exception as e:
                print(f"    Warning: Could not load pitch head: {e}")
                pitch_head = None

        if self.config.emotion_checkpoint and os.path.exists(self.config.emotion_checkpoint):
            print(f"    Loading emotion head from {self.config.emotion_checkpoint}")
            try:
                multi_config = MultiHeadConfig(d_model=d_model)
                emotion_head = EmotionHead(multi_config)

                weights = mx.load(self.config.emotion_checkpoint)
                emotion_weights = {
                    k.replace("emotion.", ""): v
                    for k, v in weights.items()
                    if k.startswith("emotion.")
                }
                if emotion_weights:
                    emotion_head.load_weights(list(emotion_weights.items()))
                    print(f"    Emotion head loaded ({len(emotion_weights)} params)")
            except Exception as e:
                print(f"    Warning: Could not load emotion head: {e}")
                emotion_head = None

        self.prosody_search = ProsodyBeamSearch(
            pitch_head=pitch_head,
            emotion_head=emotion_head,
            config=ProsodyBoostConfig(),
        )

    def _setup_roundtrip(self):
        """Initialize round-trip verification."""
        try:
            from tools.whisper_mlx.roundtrip_verification import RoundTripVerifier
            self.roundtrip_verifier = RoundTripVerifier()
            print("    Round-trip verifier loaded")
        except Exception as e:
            print(f"    Warning: Round-trip verification unavailable: {e}")
            self.config.enable_roundtrip = False

    def _setup_emotion_punct(self):
        """Initialize emotion-aware punctuation head."""
        if not self.config.punct_checkpoint or not os.path.exists(self.config.punct_checkpoint):
            print("    Warning: Punctuation checkpoint not available, disabling emotion-punct")
            self.config.enable_emotion_punct = False
            return

        try:
            print(f"    Loading punctuation head from {self.config.punct_checkpoint}")

            # Load the punctuation head model
            from tools.whisper_mlx.multi_head import PunctuationHead, MultiHeadConfig

            # Load checkpoint state to get config
            state_path = self.config.punct_checkpoint.replace(".npz", "_state.json")
            use_emotion = False
            use_pitch = False

            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = json.load(f)
                    config_data = state.get("config", {})
                    use_emotion = config_data.get("use_emotion", False)
                    use_pitch = config_data.get("use_pitch", False)
                    print(f"    Config from state: use_emotion={use_emotion}, use_pitch={use_pitch}")

            # Create config matching training setup
            head_config = MultiHeadConfig(
                d_model=1280,  # Whisper large encoder dim
                punctuation_hidden_dim=256,
                num_punctuation_classes=6,  # PERIOD, COMMA, QUESTION, EXCLAMATION, ELLIPSIS, NONE
                punctuation_use_emotion=use_emotion,
                punctuation_use_pitch=use_pitch,
                use_layer_norm=True,
                dropout_rate=0.1,
            )

            # Create head with config
            self.punct_head = PunctuationHead(head_config)

            # Load weights
            weights = mx.load(self.config.punct_checkpoint)
            self.punct_head.load_weights(list(weights.items()))
            mx.eval(self.punct_head.parameters())
            print("    Punctuation head loaded successfully")

        except Exception as e:
            print(f"    Warning: Could not load punctuation head: {e}")
            import traceback
            traceback.print_exc()
            self.config.enable_emotion_punct = False

    def load_test_data(self) -> List[Dict]:
        """Load test dataset."""
        print(f"Loading test set: {self.config.test_set}")

        samples = []

        if "librispeech" in self.config.test_set.lower():
            samples = self._load_librispeech()
        elif "meld" in self.config.test_set.lower():
            samples = self._load_meld()
        else:
            raise ValueError(f"Unknown test set: {self.config.test_set}")

        # Limit samples if needed
        if self.config.max_samples > 0 and len(samples) > self.config.max_samples:
            samples = samples[:self.config.max_samples]

        print(f"Loaded {len(samples)} samples")
        return samples

    def _load_librispeech(self) -> List[Dict]:
        """Load LibriSpeech test-clean samples."""
        samples = []

        # Look for LibriSpeech data
        base_paths = [
            Path("data/benchmarks/librispeech/LibriSpeech/test-clean"),
            Path("data/LibriSpeech/test-clean"),
            Path("~/.cache/huggingface/datasets/librispeech/test-clean").expanduser(),
        ]

        data_path = None
        for bp in base_paths:
            if bp.exists():
                data_path = bp
                break

        if data_path is None:
            print("  Warning: LibriSpeech test-clean not found, using dev-clean")
            dev_paths = [
                Path("data/benchmarks/librispeech/LibriSpeech/dev-clean"),
                Path("data/LibriSpeech/dev-clean"),
            ]
            for dp in dev_paths:
                if dp.exists():
                    data_path = dp
                    break

        if data_path is None:
            raise FileNotFoundError("Could not find LibriSpeech data")

        # Find all FLAC files with transcripts
        for trans_file in data_path.glob("**/*.trans.txt"):
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, transcript = parts
                        # Find corresponding audio file
                        audio_file = trans_file.parent / f"{file_id}.flac"
                        if audio_file.exists():
                            samples.append({
                                "id": file_id,
                                "audio_path": str(audio_file),
                                "transcript": transcript,
                            })

        return samples

    def _load_meld(self) -> List[Dict]:
        """Load MELD test samples with punctuation labels."""
        import csv

        samples = []

        # Look for MELD data
        meld_paths = [
            Path("data/emotion_punctuation/MELD.Raw"),
            Path("data/MELD.Raw"),
        ]

        meld_dir = None
        for mp in meld_paths:
            if mp.exists():
                meld_dir = mp
                break

        if meld_dir is None:
            raise FileNotFoundError("Could not find MELD data")

        # Load test split
        audio_dir = meld_dir / "audio_test"
        csv_path = meld_dir / "test_sent_emo.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"MELD CSV not found: {csv_path}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"MELD audio directory not found: {audio_dir}")

        print(f"  Loading MELD test from {csv_path}")

        # Count available audio files
        audio_files = set(p.stem for p in audio_dir.glob("*.wav"))
        print(f"  Found {len(audio_files)} audio files")

        # Read CSV annotations
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Get dialogue and utterance IDs to construct filename
                dia_id = row["Dialogue_ID"]
                utt_id = row["Utterance_ID"]
                text = row.get("Utterance", "")

                # MELD naming convention: dia{dia_id}_utt{utt_id}.wav
                audio_name = f"dia{dia_id}_utt{utt_id}"
                audio_path = audio_dir / f"{audio_name}.wav"

                if audio_path.exists() and text:
                    # Extract punctuation labels from text
                    punct_labels = self._extract_punctuation_from_text(text)

                    samples.append({
                        "id": audio_name,
                        "audio_path": str(audio_path),
                        "transcript": text,
                        "punct_labels": punct_labels,
                        "emotion": row.get("Emotion", "neutral"),
                    })

        return samples

    def _extract_punctuation_from_text(self, text: str) -> List[int]:
        """
        Extract punctuation labels from text.

        Returns a list of punctuation class indices, one per word.

        Punctuation classes:
            0: PERIOD (.)
            1: COMMA (,)
            2: QUESTION (?)
            3: EXCLAMATION (!)
            4: ELLIPSIS (...)
            5: NONE
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Handle empty text
        if not text:
            return []

        # Split into words while preserving punctuation
        import re
        words = re.findall(r'\S+', text)

        if not words:
            return []

        labels = []
        for word in words:
            # Check ending punctuation
            if word.endswith("..."):
                labels.append(4)  # ELLIPSIS
            elif word.endswith("?"):
                labels.append(2)  # QUESTION
            elif word.endswith("!"):
                labels.append(3)  # EXCLAMATION
            elif word.endswith("."):
                labels.append(0)  # PERIOD
            elif word.endswith(","):
                labels.append(1)  # COMMA
            else:
                labels.append(5)  # NONE

        return labels

    def evaluate_baseline(self, samples: List[Dict]) -> EvaluationResult:
        """Evaluate baseline Whisper transcription."""
        print("\nEvaluating baseline (greedy decoding)...")

        result = EvaluationResult(technique="baseline")

        # Punctuation counts for F1
        punct_true = {"period": 0, "question": 0, "exclamation": 0, "comma": 0}
        punct_pred = {"period": 0, "question": 0, "exclamation": 0, "comma": 0}
        punct_correct = {"period": 0, "question": 0, "exclamation": 0, "comma": 0}

        from tools.whisper_mlx.audio import load_audio

        for i, sample in enumerate(samples):
            if self.config.verbose and i % 10 == 0:
                print(f"  Processing sample {i+1}/{len(samples)}")

            try:
                # Load audio
                audio = load_audio(sample["audio_path"], sample_rate=16000)
                audio_duration = len(audio) / 16000
                result.total_audio_duration += audio_duration

                # Transcribe
                t0 = time.perf_counter()
                output = self.model.transcribe(
                    audio,
                    language="en",
                    task="transcribe",
                )
                processing_time = time.perf_counter() - t0
                result.total_processing_time += processing_time

                # Get transcription
                pred_text = output.get("text", "").strip()
                ref_text = sample["transcript"]

                # Update punctuation counts
                self._update_punct_counts(ref_text, pred_text, punct_true, punct_pred, punct_correct)

                result.samples_evaluated += 1

            except Exception as e:
                if self.config.verbose:
                    print(f"  Error processing {sample['id']}: {e}")

        # Calculate metrics
        result.rtf = result.total_processing_time / max(result.total_audio_duration, 0.001)
        result.punctuation = self._compute_punct_f1(punct_true, punct_pred, punct_correct)

        print(f"  Baseline RTF: {result.rtf:.3f}")
        print(f"  Baseline Punct Macro F1: {result.punctuation.macro_f1:.3f}")

        return result

    def evaluate_prosody_beam(self, samples: List[Dict]) -> EvaluationResult:
        """Evaluate prosody-conditioned beam search."""
        print("\nEvaluating prosody beam search...")

        if self.prosody_search is None:
            print("  Warning: Prosody search not initialized, skipping")
            return EvaluationResult(technique="prosody_beam")

        result = EvaluationResult(technique="prosody_beam")

        punct_true = {"period": 0, "question": 0, "exclamation": 0, "comma": 0}
        punct_pred = {"period": 0, "question": 0, "exclamation": 0, "comma": 0}
        punct_correct = {"period": 0, "question": 0, "exclamation": 0, "comma": 0}

        from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

        for i, sample in enumerate(samples):
            if self.config.verbose and i % 10 == 0:
                print(f"  Processing sample {i+1}/{len(samples)}")

            try:
                # Load audio
                audio = load_audio(sample["audio_path"], sample_rate=16000)
                audio_duration = len(audio) / 16000
                result.total_audio_duration += audio_duration

                # Compute mel and encode
                mel = log_mel_spectrogram(audio, n_mels=self.model.config.n_mels)
                target_len = self.model.config.n_audio_ctx * 2
                if mel.shape[0] < target_len:
                    mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
                mel = mel[None]

                t0 = time.perf_counter()

                # Encode
                audio_features = self.model.embed_audio(mel, variable_length=False)
                mx.eval(audio_features)

                # Decode with prosody
                self.prosody_search._initialize_punctuation_ids(self.tokenizer)
                decode_result = self.prosody_search.decode_with_prosody(
                    model=self.model,
                    audio_features=audio_features,
                    tokenizer=self.tokenizer,
                    beam_size=self.config.beam_size,
                    max_tokens=224,
                )
                mx.eval()

                processing_time = time.perf_counter() - t0
                result.total_processing_time += processing_time

                # Decode text
                text_tokens = [t for t in decode_result.tokens
                              if t < self.tokenizer.eot and t not in self.tokenizer.sot_sequence]
                pred_text = self.tokenizer.decode(text_tokens).strip()
                ref_text = sample["transcript"]

                # Update punctuation counts
                self._update_punct_counts(ref_text, pred_text, punct_true, punct_pred, punct_correct)

                result.samples_evaluated += 1

            except Exception as e:
                if self.config.verbose:
                    print(f"  Error processing {sample['id']}: {e}")

        # Calculate metrics
        result.rtf = result.total_processing_time / max(result.total_audio_duration, 0.001)
        result.punctuation = self._compute_punct_f1(punct_true, punct_pred, punct_correct)

        # Get prosody stats
        stats = self.prosody_search.get_stats()
        print(f"  Prosody Beam RTF: {result.rtf:.3f}")
        print(f"  Prosody Beam Punct Macro F1: {result.punctuation.macro_f1:.3f}")
        print(f"  Prosody boosts applied: {stats.get('boosts_applied', 0)}")

        return result

    def _update_punct_counts(
        self,
        ref_text: str,
        pred_text: str,
        punct_true: Dict[str, int],
        punct_pred: Dict[str, int],
        punct_correct: Dict[str, int],
    ):
        """Update punctuation counts for F1 calculation."""
        # Map punctuation to category
        punct_map = {
            ".": "period",
            "?": "question",
            "!": "exclamation",
            ",": "comma",
        }

        # Count punctuation in reference
        for char, category in punct_map.items():
            ref_count = ref_text.count(char)
            pred_count = pred_text.count(char)
            punct_true[category] += ref_count
            punct_pred[category] += pred_count
            punct_correct[category] += min(ref_count, pred_count)

    def _compute_punct_f1(
        self,
        punct_true: Dict[str, int],
        punct_pred: Dict[str, int],
        punct_correct: Dict[str, int],
    ) -> PunctuationMetrics:
        """Compute per-class punctuation F1 scores."""
        metrics = PunctuationMetrics()

        def safe_f1(true: int, pred: int, correct: int) -> Tuple[float, float, float]:
            precision = correct / pred if pred > 0 else 0.0
            recall = correct / true if true > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return precision, recall, f1

        metrics.period_precision, metrics.period_recall, metrics.period_f1 = safe_f1(
            punct_true["period"], punct_pred["period"], punct_correct["period"]
        )
        metrics.question_precision, metrics.question_recall, metrics.question_f1 = safe_f1(
            punct_true["question"], punct_pred["question"], punct_correct["question"]
        )
        metrics.exclamation_precision, metrics.exclamation_recall, metrics.exclamation_f1 = safe_f1(
            punct_true["exclamation"], punct_pred["exclamation"], punct_correct["exclamation"]
        )
        metrics.comma_precision, metrics.comma_recall, metrics.comma_f1 = safe_f1(
            punct_true["comma"], punct_pred["comma"], punct_correct["comma"]
        )

        # Macro F1 (average of per-class F1)
        f1_scores = [
            metrics.period_f1,
            metrics.question_f1,
            metrics.exclamation_f1,
            metrics.comma_f1,
        ]
        metrics.macro_f1 = sum(f1_scores) / len(f1_scores)

        return metrics

    def run_evaluation(self) -> Dict[str, EvaluationResult]:
        """Run full evaluation."""
        # Setup
        self.setup()

        # Load data
        samples = self.load_test_data()

        # Run evaluations
        if self.config.enable_baseline:
            self.results["baseline"] = self.evaluate_baseline(samples)

        if self.config.enable_prosody_beam:
            self.results["prosody_beam"] = self.evaluate_prosody_beam(samples)

        if self.config.enable_roundtrip:
            # TODO: Implement round-trip evaluation
            print("\nRound-trip evaluation not yet implemented")

        if self.config.enable_emotion_punct and self.punct_head is not None:
            punct_result = self.evaluate_emotion_punct(samples)
            self.results["emotion_punct"] = punct_result

        return self.results

    def evaluate_emotion_punct(self, samples: List[Dict]) -> EvaluationResult:
        """Evaluate emotion-aware punctuation head on samples."""
        print("\nEvaluating emotion-aware punctuation...")

        result = EvaluationResult(technique="emotion_punct")

        from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

        # Class names for reporting
        PUNCT_CLASSES = ["PERIOD", "COMMA", "QUESTION", "EXCLAMATION", "ELLIPSIS", "NONE"]

        # Confusion tracking per class
        num_classes = 6
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes
        support = [0] * num_classes

        for i, sample in enumerate(samples):
            if self.config.verbose and i % 10 == 0:
                print(f"  Processing sample {i+1}/{len(samples)}")

            try:
                # Load audio
                audio = load_audio(sample["audio_path"], sample_rate=16000)
                audio_duration = len(audio) / 16000
                result.total_audio_duration += audio_duration

                # Compute mel spectrogram
                mel = log_mel_spectrogram(audio, n_mels=128)

                # Convert to numpy immediately
                if isinstance(mel, mx.array):
                    mel = np.array(mel)

                mel = mel.astype(np.float32)

                # Pad to 30s (3000 frames) for encoder
                target_frames = 3000
                actual_frames = mel.shape[0]

                if actual_frames < target_frames:
                    # Pad with zeros
                    padding = np.zeros((target_frames - actual_frames, mel.shape[1]), dtype=np.float32)
                    mel = np.concatenate([mel, padding], axis=0)
                elif actual_frames > target_frames:
                    # Truncate (shouldn't happen for MELD short clips)
                    mel = mel[:target_frames]

                mel_mx = mx.array(mel)[None, :]  # Add batch dim

                # Run encoder
                t0 = time.perf_counter()
                encoder_out = self.model.encoder(mel_mx)

                # Run punctuation head
                logits = self.punct_head(encoder_out)  # (1, T, 6)
                predictions = mx.argmax(logits, axis=-1)  # (1, T)
                predictions = np.array(predictions[0])  # (T,)
                mx.eval(predictions)

                result.total_processing_time += time.perf_counter() - t0

                # Get ground truth labels
                ground_truth = sample.get("punct_labels", [])
                if not ground_truth:
                    continue

                # Align predictions with ground truth (simple approach)
                # Distribute predictions evenly across ground truth words
                # Only use the non-padded portion of predictions
                # actual_frames is from the original mel, encoder outputs half that
                encoder_frames = actual_frames // 2
                T = min(len(predictions), encoder_frames)
                predictions = predictions[:T]  # Trim to actual audio length
                num_words = len(ground_truth)

                if num_words > 0:
                    for w_idx, gt_label in enumerate(ground_truth):
                        # Map word index to frame index
                        frame_idx = min(int(w_idx * T / num_words), T - 1)
                        pred_label = int(predictions[frame_idx])

                        support[gt_label] += 1
                        if pred_label == gt_label:
                            tp[pred_label] += 1
                        else:
                            fp[pred_label] += 1
                            fn[gt_label] += 1

                result.samples_evaluated += 1

            except Exception as e:
                if self.config.verbose:
                    print(f"  Error processing sample {i}: {e}")
                continue

        # Compute metrics
        per_class = {}
        for i, name in enumerate(PUNCT_CLASSES):
            precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support[i]}

        # Compute macro F1
        macro_f1 = sum(per_class[n]["f1"] for n in PUNCT_CLASSES) / num_classes

        # Update result punctuation metrics
        result.punctuation.period_precision = per_class["PERIOD"]["precision"]
        result.punctuation.period_recall = per_class["PERIOD"]["recall"]
        result.punctuation.period_f1 = per_class["PERIOD"]["f1"]

        result.punctuation.question_precision = per_class["QUESTION"]["precision"]
        result.punctuation.question_recall = per_class["QUESTION"]["recall"]
        result.punctuation.question_f1 = per_class["QUESTION"]["f1"]

        result.punctuation.exclamation_precision = per_class["EXCLAMATION"]["precision"]
        result.punctuation.exclamation_recall = per_class["EXCLAMATION"]["recall"]
        result.punctuation.exclamation_f1 = per_class["EXCLAMATION"]["f1"]

        result.punctuation.comma_precision = per_class["COMMA"]["precision"]
        result.punctuation.comma_recall = per_class["COMMA"]["recall"]
        result.punctuation.comma_f1 = per_class["COMMA"]["f1"]

        result.punctuation.macro_f1 = macro_f1

        # Compute RTF
        if result.total_audio_duration > 0:
            result.rtf = result.total_processing_time / result.total_audio_duration

        # Print summary
        print(f"  Samples evaluated: {result.samples_evaluated}")
        print(f"  RTF: {result.rtf:.3f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Question F1: {per_class['QUESTION']['f1']:.4f} (recall: {per_class['QUESTION']['recall']:.4f})")
        print(f"  Period F1: {per_class['PERIOD']['f1']:.4f}")

        return result

    def generate_report(self) -> str:
        """Generate markdown report of results."""
        lines = [
            "# Novel ASR Techniques Evaluation Report",
            "",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
            f"**Test Set:** {self.config.test_set}",
            f"**Samples:** {self.config.max_samples}",
            "",
            "## Summary",
            "",
            "| Technique | RTF | Punct Macro F1 | ? F1 | ! F1 | Samples |",
            "|-----------|-----|----------------|------|------|---------|",
        ]

        for name, result in self.results.items():
            lines.append(
                f"| {name} | {result.rtf:.3f} | {result.punctuation.macro_f1:.3f} | "
                f"{result.punctuation.question_f1:.3f} | {result.punctuation.exclamation_f1:.3f} | "
                f"{result.samples_evaluated} |"
            )

        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])

        for name, result in self.results.items():
            lines.extend([
                f"### {name}",
                "",
                f"- **RTF:** {result.rtf:.3f}",
                f"- **Total Audio:** {result.total_audio_duration:.1f}s",
                f"- **Total Processing:** {result.total_processing_time:.1f}s",
                "",
                "**Punctuation Metrics:**",
                "",
                "| Class | Precision | Recall | F1 |",
                "|-------|-----------|--------|-----|",
                f"| Period (.) | {result.punctuation.period_precision:.3f} | "
                f"{result.punctuation.period_recall:.3f} | {result.punctuation.period_f1:.3f} |",
                f"| Question (?) | {result.punctuation.question_precision:.3f} | "
                f"{result.punctuation.question_recall:.3f} | {result.punctuation.question_f1:.3f} |",
                f"| Exclamation (!) | {result.punctuation.exclamation_precision:.3f} | "
                f"{result.punctuation.exclamation_recall:.3f} | {result.punctuation.exclamation_f1:.3f} |",
                f"| Comma (,) | {result.punctuation.comma_precision:.3f} | "
                f"{result.punctuation.comma_recall:.3f} | {result.punctuation.comma_f1:.3f} |",
                f"| **Macro** | - | - | **{result.punctuation.macro_f1:.3f}** |",
                "",
            ])

        return "\n".join(lines)

    def save_report(self, output_path: str):
        """Save report to file."""
        report = self.generate_report()

        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(report)

        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate novel streaming ASR techniques")
    parser.add_argument("--test-set", type=str, default="librispeech-test-clean",
                        help="Test dataset to use")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum samples to evaluate (0 for all)")
    parser.add_argument("--enable-baseline", action="store_true", default=True,
                        help="Enable baseline evaluation")
    parser.add_argument("--enable-roundtrip", action="store_true",
                        help="Enable round-trip verification")
    parser.add_argument("--enable-prosody-beam", action="store_true",
                        help="Enable prosody beam search")
    parser.add_argument("--enable-emotion-punct", action="store_true",
                        help="Enable emotion-aware punctuation")
    parser.add_argument("--beam-size", type=int, default=3,
                        help="Beam size for prosody beam search")
    parser.add_argument("--pitch-checkpoint", type=str,
                        default="checkpoints/pitch_combined_v4/best.npz",
                        help="Path to pitch head checkpoint")
    parser.add_argument("--emotion-checkpoint", type=str,
                        default="checkpoints/emotion_unified_v2/best.npz",
                        help="Path to emotion head checkpoint")
    parser.add_argument("--punct-checkpoint", type=str, default=None,
                        help="Path to punctuation head checkpoint")
    parser.add_argument("--output", type=str,
                        default="reports/main/NOVEL_TECHNIQUES_RESULTS.md",
                        help="Output report path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    config = EvaluationConfig(
        test_set=args.test_set,
        max_samples=args.max_samples,
        enable_baseline=args.enable_baseline,
        enable_roundtrip=args.enable_roundtrip,
        enable_prosody_beam=args.enable_prosody_beam,
        enable_emotion_punct=args.enable_emotion_punct,
        beam_size=args.beam_size,
        pitch_checkpoint=args.pitch_checkpoint,
        emotion_checkpoint=args.emotion_checkpoint,
        punct_checkpoint=args.punct_checkpoint,
        output_path=args.output,
        verbose=args.verbose,
    )

    evaluator = NovelTechniquesEvaluator(config)

    # Run evaluation
    results = evaluator.run_evaluation()

    # Generate and save report
    if args.output:
        evaluator.save_report(args.output)
    else:
        print("\n" + evaluator.generate_report())


if __name__ == "__main__":
    main()
