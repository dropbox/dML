#!/usr/bin/env python3
"""Compare sherpa-onnx output with MLX Zipformer output."""

import numpy as np
import wave
import sherpa_onnx

MODEL_DIR = "checkpoints/zipformer/sherpa-onnx-streaming-zipformer-en-2023-06-26"


def run_sherpa_onnx(wav_path: str) -> str:
    """Run sherpa-onnx streaming recognizer on a wav file."""
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=f"{MODEL_DIR}/encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        decoder=f"{MODEL_DIR}/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        joiner=f"{MODEL_DIR}/joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        tokens=f"{MODEL_DIR}/tokens.txt",
        num_threads=1,
    )

    # Load audio
    with wave.open(wav_path, "rb") as f:
        sample_rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    print(f"Audio: {len(samples)} samples @ {sample_rate} Hz ({len(samples)/sample_rate:.2f}s)")

    # Create stream and decode
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)

    # Signal end of audio
    stream.input_finished()

    # Process all audio
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    # Get final result
    result = recognizer.get_result(stream)
    # In newer sherpa-onnx versions, result is already a string
    if isinstance(result, str):
        return result
    return result.text


def run_mlx_pipeline(wav_path: str) -> str:
    """Run MLX ASR pipeline on a wav file."""
    import sys
    sys.path.insert(0, "src")
    from models.zipformer.inference import ASRPipeline

    pipeline = ASRPipeline(MODEL_DIR)
    return pipeline.transcribe(wav_path)


def main():
    # Test on all test wavs
    test_wavs = [
        f"{MODEL_DIR}/test_wavs/0.wav",
        f"{MODEL_DIR}/test_wavs/1.wav",
    ]

    print("=" * 60)
    print("SHERPA-ONNX RESULTS")
    print("=" * 60)

    for wav_path in test_wavs:
        print(f"\nFile: {wav_path}")
        result = run_sherpa_onnx(wav_path)
        print(f"Result: '{result}'")

    print("\n" + "=" * 60)
    print("MLX RESULTS")
    print("=" * 60)

    for wav_path in test_wavs:
        print(f"\nFile: {wav_path}")
        result = run_mlx_pipeline(wav_path)
        print(f"Result: '{result}'")


if __name__ == "__main__":
    main()
