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
Test CTC Speculative Decoding.

This script tests the CTCSpeculativeDecoder class with a trained CTC head.
It compares performance against:
1. Greedy decoding (baseline)
2. CTC speculative decoding (new)

Usage:
    python -m scripts.test_ctc_speculative \
        --ctc-checkpoint checkpoints/ctc_english_full/step_48000.npz \
        --audio-file test_audio.wav
"""

import argparse
import time
from pathlib import Path
from typing import Dict

import mlx.core as mx


def test_greedy_decoding(model, audio_path: str) -> Dict:
    """Test greedy decoding via transcribe()."""
    start = time.perf_counter()
    result = model.transcribe(audio_path, language="en", task="transcribe")
    total_time = time.perf_counter() - start

    return {
        "method": "greedy",
        "text": result["text"],
        "total_ms": total_time * 1000,
    }


def test_ctc_speculative_direct(
    model,
    audio_path: str,
    ctc_checkpoint: str,
) -> Dict:
    """
    Test CTC speculative decoding using direct decoder access.

    This bypasses the transcribe() method to use CTCSpeculativeDecoder directly.
    """
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram, pad_or_trim
    from tools.whisper_mlx.ctc_head import CTCDraftHead
    from tools.whisper_mlx.decoding import DecodingOptions
    from tools.whisper_mlx.speculative import CTCSpeculativeDecoder
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

    # Setup tokenizer
    tokenizer = get_whisper_tokenizer(multilingual=True, language="en", task="transcribe")
    # Note: Decoder output vocab (51866) differs from embedding vocab (51872)
    # We need to use the actual output vocab size for logit filters
    # Get it from the decoder's proj_out layer or infer from a dummy forward pass
    n_vocab = 51866  # Whisper large-v3 decoder output size

    start = time.perf_counter()

    # Load and process audio
    audio = load_audio(audio_path)
    audio_duration = len(audio) / 16000
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio, n_mels=128)
    mel_batch = mx.expand_dims(mel, axis=0)

    # Encode audio
    audio_features = model.encoder(mel_batch)
    mx.eval(audio_features)
    encode_time = time.perf_counter() - start

    # Load CTC head
    ctc_load_start = time.perf_counter()
    ctc_head = CTCDraftHead.load_weights(ctc_checkpoint, d_model=1280)
    ctc_load_time = time.perf_counter() - ctc_load_start

    # Create CTC speculative decoder
    ctc_decoder = CTCSpeculativeDecoder(
        main_model=model,
        ctc_head=ctc_head,
        draft_tokens=20,
    )

    # Build decoding options
    options = DecodingOptions(language="en", task="transcribe")
    sot_sequence = tokenizer.sot_sequence
    sample_begin = len(sot_sequence)

    # Decode
    decode_start = time.perf_counter()
    tokens, segments = ctc_decoder.decode(
        audio_features=audio_features,
        tokenizer=tokenizer,
        options=options,
        sample_begin=sample_begin,
        n_vocab=n_vocab,
        audio_duration=audio_duration,
    )
    decode_time = time.perf_counter() - decode_start

    total_time = time.perf_counter() - start

    text = tokenizer.decode(tokens)

    timing = ctc_decoder.timing_breakdown

    return {
        "method": "ctc_speculative",
        "text": text,
        "tokens": len(tokens),
        "encode_ms": encode_time * 1000,
        "ctc_load_ms": ctc_load_time * 1000,
        "decode_ms": decode_time * 1000,
        "total_ms": total_time * 1000,
        "acceptance_rate": ctc_decoder.acceptance_rate,
        "iterations": ctc_decoder.iterations,
        "tokens_per_iter": ctc_decoder.tokens_per_iteration,
        "ctc_draft_ms": timing["ctc_draft_ms"],
        "verify_ms": timing["verify_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description="Test CTC Speculative Decoding")
    parser.add_argument(
        "--ctc-checkpoint",
        type=str,
        default="checkpoints/ctc_english_full/step_48000.npz",
        help="Path to CTC head checkpoint",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default="data/benchmarks/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac",
        help="Audio file to test",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        help="Whisper model size",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run warmup iteration before timing",
    )
    args = parser.parse_args()

    # Check CTC checkpoint exists
    if not Path(args.ctc_checkpoint).exists():
        print(f"CTC checkpoint not found: {args.ctc_checkpoint}")
        print("Listing available checkpoints...")
        ckpt_dir = Path("checkpoints/ctc_english_full")
        if ckpt_dir.exists():
            checkpoints = sorted(ckpt_dir.glob("*.npz"))
            for ckpt in checkpoints[-5:]:
                print(f"  {ckpt}")
            if checkpoints:
                args.ctc_checkpoint = str(checkpoints[-1])
                print(f"\nUsing latest checkpoint: {args.ctc_checkpoint}")
            else:
                print("No checkpoints found. Training still in progress?")
                return
        else:
            print("Checkpoint directory not found")
            return

    # Check audio file exists
    if not Path(args.audio_file).exists():
        print(f"Audio file not found: {args.audio_file}")
        # Try to find a test file
        test_clean = Path("data/benchmarks/librispeech/LibriSpeech/test-clean")
        if test_clean.exists():
            audio_files = list(test_clean.rglob("*.flac"))[:1]
            if audio_files:
                args.audio_file = str(audio_files[0])
                print(f"Using: {args.audio_file}")
            else:
                print("No audio files found in test-clean")
                return
        else:
            print("LibriSpeech test-clean not found")
            return

    print("=" * 60)
    print("CTC Speculative Decoding Test")
    print("=" * 60)
    print(f"Model: {args.model_size}")
    print(f"Audio: {args.audio_file}")
    print(f"CTC checkpoint: {args.ctc_checkpoint}")
    print()

    # Load model
    print("Loading Whisper model...")
    from tools.whisper_mlx.model import WhisperMLX

    model = WhisperMLX.from_pretrained(
        f"mlx-community/whisper-{args.model_size}-mlx",
        dtype=mx.float16,
    )
    n_vocab = model.decoder.token_embedding.weight.shape[0]
    print(f"Model loaded. Vocab size: {n_vocab}")
    print()

    # Warmup if requested
    if args.warmup:
        print("Running warmup...")
        from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram, pad_or_trim

        audio = load_audio(args.audio_file)
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio, n_mels=128)
        mel_batch = mx.expand_dims(mel, axis=0)
        _ = model.encoder(mel_batch)
        mx.eval(_)
        print("Warmup complete")
        print()

    results = []

    # Test 1: Greedy decoding
    print("Test 1: Greedy decoding (baseline)")
    result = test_greedy_decoding(model, args.audio_file)
    results.append(result)
    print(f"  Text: {result['text'][:80]}...")
    print(f"  Total time: {result['total_ms']:.1f}ms")
    print()

    # Test 2: CTC speculative
    print("Test 2: CTC speculative decoding")
    try:
        result = test_ctc_speculative_direct(model, args.audio_file, args.ctc_checkpoint)
        results.append(result)
        print(f"  Text: {result['text'][:80]}...")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Acceptance rate: {result['acceptance_rate']:.1%}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Tokens/iter: {result['tokens_per_iter']:.2f}")
        print(f"  CTC draft time: {result['ctc_draft_ms']:.1f}ms")
        print(f"  Verify time: {result['verify_ms']:.1f}ms")
        print(f"  Total time: {result['total_ms']:.1f}ms")
    except Exception as e:
        import traceback

        print(f"  Error: {e}")
        traceback.print_exc()
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    baseline_time = results[0]["total_ms"] if results else 0
    for r in results:
        speedup = baseline_time / r["total_ms"] if r["total_ms"] > 0 else 0
        print(f"{r['method']:20s}: {r['total_ms']:7.1f}ms ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
