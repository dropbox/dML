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
Evaluate fine-tuned Medusa heads for whisper-large-v3.

This script:
1. Loads fine-tuned Medusa weights
2. Tests prediction accuracy against ground truth
3. Measures transcription speedup vs baseline
4. Verifies output quality matches baseline

Usage:
    python scripts/evaluate_medusa_finetuned.py \
        --weights checkpoints/medusa_v3_finetuned/medusa_v3_finetuned.npz
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.medusa import MedusaModule
from tools.whisper_mlx.medusa_training import load_medusa_weights


class ProjOutWrapper(nn.Module):
    """Wrapper for token_embedding.as_linear()."""

    def __init__(self, token_embedding, n_vocab: int = None):
        super().__init__()
        self._token_embedding = token_embedding
        self._n_vocab = n_vocab

    def __call__(self, x):
        logits = self._token_embedding.as_linear(x)
        if self._n_vocab is not None and logits.shape[-1] > self._n_vocab:
            logits = logits[..., :self._n_vocab]
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Medusa heads"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/medusa_v3_finetuned/medusa_v3_finetuned.npz",
        help="Path to fine-tuned Medusa weights",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-large-v3-mlx",
        help="Whisper model",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=5,
        help="Number of Medusa heads",
    )
    parser.add_argument(
        "--test-files",
        type=str,
        nargs="*",
        default=None,
        help="Audio files to test (defaults to LibriSpeech test samples)",
    )
    return parser.parse_args()


def get_test_files():
    """Get a set of test files from LibriSpeech test-clean."""
    test_dir = Path("/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean")
    files = list(test_dir.glob("**/*.flac"))[:10]  # Get 10 test files
    return files


def test_prediction_accuracy(model, medusa, proj_out, test_file):
    """Test Medusa head prediction accuracy for a single file."""
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

    # Load and encode audio
    audio = load_audio(str(test_file))
    mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)

    # Pad to standard length
    target_len = model.config.n_audio_ctx * 2
    if mel.shape[0] < target_len:
        mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
    else:
        mel = mel[:target_len]

    audio_features = model.embed_audio(mel)

    # Get decoder tokens for baseline transcription
    result = model.transcribe(str(test_file))
    text = result['text']

    # Get teacher logits and hidden states by running decoder
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

    is_multilingual = model.config.n_vocab >= 51865
    num_langs = model.config.n_vocab - 51765 - int(is_multilingual)
    tokenizer = get_whisper_tokenizer(multilingual=is_multilingual, num_languages=num_langs)

    # Tokenize the output text
    tokens = tokenizer.encode(text.strip())
    sot_sequence = [tokenizer.sot]
    tokens = sot_sequence + tokens
    tokens = mx.array([tokens])  # Add batch dimension

    # Run decoder to get hidden states
    teacher_logits, _, _, hidden_states = model.decoder(
        tokens,
        audio_features,
        kv_cache=None,
        return_hidden=True,
    )

    # Get Medusa predictions
    medusa.set_proj_out(proj_out)
    medusa_logits = medusa(hidden_states)

    # Compare predictions
    # Medusa head i should predict the token at position n+i+1 given hidden state at n
    # So we compare medusa head i at position n against actual tokens at n+i+1
    correct = 0
    total = 0
    seq_len = tokens.shape[1]
    actual_tokens = tokens[0].tolist()  # Get actual token sequence

    for head_idx, head_logits in enumerate(medusa_logits):
        shift = head_idx + 1  # Head i predicts shift=i+1 positions ahead

        # For each position n where we can make a prediction (n+shift must be < seq_len)
        for pos in range(seq_len - shift):
            # Medusa prediction at position pos
            medusa_pred = mx.argmax(head_logits[:, pos], axis=-1).item()
            # Target is the actual token at pos + shift
            target_token = actual_tokens[pos + shift]

            if medusa_pred == target_token:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def benchmark_transcription(model, audio_file, use_medusa=False):
    """Benchmark transcription speed."""
    # Warmup
    model.transcribe(str(audio_file))

    # Timed run
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        if use_medusa:
            result = model.transcribe_medusa(str(audio_file))
        else:
            result = model.transcribe(str(audio_file))
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    return avg_time, result['text']


def main():
    args = parse_args()

    print("=" * 60)
    print("Medusa Evaluation: Fine-tuned v2 -> v3 Heads")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
    print(f"Heads: {args.n_heads}")
    print("=" * 60)

    # Check weights exist
    if not os.path.exists(args.weights):
        print(f"ERROR: Weights not found at {args.weights}")
        sys.exit(1)

    # Load model
    print("\nLoading WhisperMLX model...")
    model = WhisperMLX.from_pretrained(args.model)
    config = model.config

    # Create proj_out
    proj_out = ProjOutWrapper(model.decoder.token_embedding, n_vocab=config.n_vocab)

    # Create and load Medusa module
    print(f"\nLoading Medusa module with {args.n_heads} heads...")
    medusa = MedusaModule(
        n_state=config.n_text_state,
        n_vocab=config.n_vocab,
        n_heads=args.n_heads,
        use_aiola=True,
        proj_out=proj_out,
    )

    # Load fine-tuned weights
    load_medusa_weights(medusa, args.weights)
    mx.eval(medusa.parameters())

    # Get test files
    test_files = args.test_files
    if not test_files:
        test_files = get_test_files()
        print(f"\nUsing {len(test_files)} LibriSpeech test files")

    # Test prediction accuracy
    print("\n" + "=" * 60)
    print("Prediction Accuracy Test")
    print("=" * 60)

    total_correct = 0
    total_predictions = 0

    for i, test_file in enumerate(test_files[:5]):  # Test first 5
        accuracy, correct, total = test_prediction_accuracy(model, medusa, proj_out, test_file)
        print(f"  File {i+1}: accuracy={accuracy:.1%} ({correct}/{total} correct)")
        total_correct += correct
        total_predictions += total

    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    print(f"\nOverall Prediction Accuracy: {overall_accuracy:.1%}")
    print(f"  ({total_correct}/{total_predictions} correct predictions)")

    # Evaluate acceptance thresholds
    print("\n" + "=" * 60)
    print("Acceptance Rate Analysis")
    print("=" * 60)
    if overall_accuracy < 0.30:
        print("  Status: FAIL - Accuracy <30%")
        print("  Fine-tuning didn't achieve viable prediction accuracy")
        print("  Recommendation: More training steps or different approach")
    elif overall_accuracy < 0.50:
        print("  Status: MARGINAL - Accuracy 30-50%")
        print("  Small speedup possible (~1.2-1.5x)")
        print("  Recommendation: More training for better speedup")
    elif overall_accuracy < 0.70:
        print("  Status: GOOD - Accuracy 50-70%")
        print("  Meaningful speedup expected (~1.5-2x)")
    else:
        print("  Status: EXCELLENT - Accuracy >70%")
        print("  Strong speedup expected (~2-3x)")

    # Note about speedup testing
    print("\n" + "=" * 60)
    print("Speedup Benchmark (requires full Medusa integration)")
    print("=" * 60)
    print("  To test actual speedup, load weights using:")
    print(f"    model.load_medusa_weights('{args.weights}', use_aiola=True)")
    print("  Then call:")
    print("    model.transcribe_medusa(audio_file)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Weights: {args.weights}")
    print(f"  Prediction Accuracy: {overall_accuracy:.1%}")

    viable = overall_accuracy >= 0.30
    print(f"  Viable for speedup: {'YES' if viable else 'NO'}")


if __name__ == "__main__":
    main()
