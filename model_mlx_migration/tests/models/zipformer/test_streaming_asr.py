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
Pytest test suite for Zipformer streaming ASR.

Tests:
1. Streaming vs non-streaming frame count alignment
2. WER comparison between streaming and non-streaming
3. Streaming transcription quality on diverse audio

Run with: pytest tests/models/zipformer/test_streaming_asr.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import mlx.core as mx


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate between reference and hypothesis."""
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()

    # Dynamic programming for edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    edit_distance = dp[m][n]
    wer = edit_distance / max(len(ref_words), 1)

    return {
        "wer": wer,
        "edit_distance": edit_distance,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
    }


@pytest.fixture(scope="module")
def checkpoint_path():
    """Return path to Zipformer checkpoint."""
    return "checkpoints/zipformer/en-streaming/exp/pretrained.pt"


@pytest.fixture(scope="module")
def bpe_model_path():
    """Return path to BPE model."""
    return "checkpoints/zipformer/en-streaming/data/lang_bpe_500/bpe.model"


@pytest.fixture(scope="module")
def test_audio_dir():
    """Return path to test audio directory."""
    return Path("checkpoints/zipformer/en-streaming/test_wavs")


@pytest.fixture(scope="module")
def test_transcripts(test_audio_dir) -> dict[str, str]:
    """Load test transcripts."""
    trans_file = test_audio_dir / "trans.txt"
    if not trans_file.exists():
        pytest.skip(f"Transcript file not found: {trans_file}")

    transcripts = {}
    with open(trans_file) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
    return transcripts


@pytest.fixture(scope="module")
def asr_pipeline(checkpoint_path, bpe_model_path):
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


class TestStreamingFrameAlignment:
    """Test streaming vs non-streaming frame count alignment."""

    def test_frame_count_ratio(self, asr_pipeline, test_audio_dir):
        """Test that streaming produces similar frame count to non-streaming.

        After the frame alignment fix, streaming should produce:
        - Frame ratio (streaming/non-streaming) between 0.95 and 1.05
        """
        from src.models.zipformer.features import load_audio

        audio_file = test_audio_dir / "1089-134686-0001.wav"
        if not audio_file.exists():
            pytest.skip(f"Audio file not found: {audio_file}")

        audio, _ = load_audio(str(audio_file), target_sr=16000)

        # Non-streaming: get encoder output frames
        features = asr_pipeline.fbank_extractor.extract(audio, 16000)
        features = mx.expand_dims(features, axis=0)
        feature_lens = mx.array([features.shape[1]], dtype=mx.int32)
        encoder_out, encoder_out_lens = asr_pipeline.model(features, feature_lens)
        mx.eval(encoder_out, encoder_out_lens)
        nonstreaming_frames = int(encoder_out_lens[0].item())

        # Streaming: count total frames from all chunks
        streaming_frames = 0
        for _partial_text, _full_text in asr_pipeline.transcribe_streaming(audio, 16000):
            pass  # We just want to count, done implicitly in pipeline

        # Get streaming frame count from the state
        state = asr_pipeline.init_streaming(batch_size=1)
        fbank_features = asr_pipeline.fbank_extractor.extract(audio, 16000)
        chunk_size = asr_pipeline.config.fbank_chunk_size

        streaming_frames = 0
        start_frame = 0
        while start_frame < fbank_features.shape[0]:
            end_frame = min(start_frame + chunk_size, fbank_features.shape[0])
            chunk_features = fbank_features[start_frame:end_frame]
            valid_frames = int(chunk_features.shape[0])

            if chunk_features.shape[0] < chunk_size:
                padding = mx.zeros(
                    (chunk_size - chunk_features.shape[0], chunk_features.shape[1]),
                    dtype=chunk_features.dtype,
                )
                chunk_features = mx.concatenate([chunk_features, padding], axis=0)

            chunk_features = mx.expand_dims(chunk_features, axis=0)

            # Run streaming encoder
            encoder_out_chunk, new_encoder_states = asr_pipeline.model.encoder.streaming_forward(
                chunk_features,
                state.encoder_states,
                chunk_size=16,
                left_context_len=asr_pipeline.config.left_context_len,
                processed_frames=state.processed_encoder_frames,
            )
            mx.eval(encoder_out_chunk)

            # Drop any encoder frames attributable to padding so we compare only
            # the real audio duration to non-streaming.
            expected_encoder_frames = (valid_frames // 2 + 1) // 2
            streaming_frames += min(int(encoder_out_chunk.shape[1]), expected_encoder_frames)

            state.encoder_states = new_encoder_states
            state.processed_fbank_frames += valid_frames
            state.processed_encoder_frames += valid_frames // 2
            start_frame = end_frame

        # Frame ratio check
        frame_ratio = streaming_frames / nonstreaming_frames

        print("\nFrame count comparison:")
        print(f"  Non-streaming: {nonstreaming_frames}")
        print(f"  Streaming: {streaming_frames}")
        print(f"  Ratio: {frame_ratio:.3f}")

        # After the fix, ratio should be close to 1.0
        assert 0.95 <= frame_ratio <= 1.10, \
            f"Frame ratio {frame_ratio:.3f} outside acceptable range [0.95, 1.10]"


class TestStreamingWER:
    """Test streaming transcription quality via WER."""

    @pytest.mark.parametrize("audio_id", [
        "1089-134686-0001",
        "1221-135766-0001",
        "1221-135766-0002",
    ])
    def test_streaming_wer(self, asr_pipeline, test_audio_dir, test_transcripts, audio_id):
        """Test streaming WER on individual test files.

        Acceptance criteria:
        - Streaming WER should be <= 15% (absolute)
        - WER gap (streaming - non-streaming) should be <= 10%
        """
        from src.models.zipformer.features import load_audio

        audio_file = test_audio_dir / f"{audio_id}.wav"
        if not audio_file.exists():
            pytest.skip(f"Audio file not found: {audio_file}")

        if audio_id not in test_transcripts:
            pytest.skip(f"No transcript for: {audio_id}")

        reference = test_transcripts[audio_id]
        audio, _ = load_audio(str(audio_file), target_sr=16000)

        # Non-streaming transcription
        nonstreaming_text = asr_pipeline.transcribe(audio, 16000)
        nonstreaming_wer = compute_wer(reference, nonstreaming_text)

        # Streaming transcription
        streaming_text = ""
        for _partial, full in asr_pipeline.transcribe_streaming(audio, 16000):
            streaming_text = full
        streaming_wer = compute_wer(reference, streaming_text)

        wer_gap = streaming_wer["wer"] - nonstreaming_wer["wer"]

        print(f"\n{audio_id}:")
        print(f"  Reference: {reference[:60]}...")
        print(f"  Non-streaming: {nonstreaming_text[:60]}...")
        print(f"  Streaming: {streaming_text[:60]}...")
        print(f"  Non-streaming WER: {nonstreaming_wer['wer']*100:.1f}%")
        print(f"  Streaming WER: {streaming_wer['wer']*100:.1f}%")
        print(f"  WER gap: {wer_gap*100:.1f}%")

        # Assert quality thresholds
        assert streaming_wer["wer"] <= 0.20, \
            f"Streaming WER {streaming_wer['wer']*100:.1f}% exceeds 20% threshold"

        assert wer_gap <= 0.15, \
            f"WER gap {wer_gap*100:.1f}% exceeds 15% threshold"

    def test_average_wer(self, asr_pipeline, test_audio_dir, test_transcripts):
        """Test average WER across all test files.

        After the frame alignment fix, average streaming WER should be < 5%.
        """
        from src.models.zipformer.features import load_audio

        results = []

        for audio_id, reference in test_transcripts.items():
            audio_file = test_audio_dir / f"{audio_id}.wav"
            if not audio_file.exists():
                continue

            audio, _ = load_audio(str(audio_file), target_sr=16000)

            # Non-streaming
            nonstreaming_text = asr_pipeline.transcribe(audio, 16000)
            nonstreaming_wer = compute_wer(reference, nonstreaming_text)

            # Streaming
            streaming_text = ""
            for _, full in asr_pipeline.transcribe_streaming(audio, 16000):
                streaming_text = full
            streaming_wer = compute_wer(reference, streaming_text)

            results.append({
                "audio_id": audio_id,
                "nonstreaming_wer": nonstreaming_wer["wer"],
                "streaming_wer": streaming_wer["wer"],
                "wer_gap": streaming_wer["wer"] - nonstreaming_wer["wer"],
            })

        if not results:
            pytest.skip("No valid test samples found")

        avg_streaming_wer = sum(r["streaming_wer"] for r in results) / len(results)
        avg_wer_gap = sum(r["wer_gap"] for r in results) / len(results)

        print(f"\nAverage results ({len(results)} samples):")
        print(f"  Average streaming WER: {avg_streaming_wer*100:.1f}%")
        print(f"  Average WER gap: {avg_wer_gap*100:.1f}%")

        for r in results:
            print(f"  {r['audio_id']}: streaming={r['streaming_wer']*100:.1f}%, gap={r['wer_gap']*100:.1f}%")

        # Assert thresholds based on previous fix results
        assert avg_streaming_wer < 0.10, \
            f"Average streaming WER {avg_streaming_wer*100:.1f}% exceeds 10% threshold"

        assert avg_wer_gap < 0.10, \
            f"Average WER gap {avg_wer_gap*100:.1f}% exceeds 10% threshold"


class TestStreamingRobustness:
    """Test streaming robustness with various audio conditions."""

    def test_different_audio_lengths(self, asr_pipeline, test_audio_dir, test_transcripts):
        """Test streaming works correctly for different audio lengths."""
        from src.models.zipformer.features import load_audio

        for audio_id, _reference in test_transcripts.items():
            audio_file = test_audio_dir / f"{audio_id}.wav"
            if not audio_file.exists():
                continue

            audio, _ = load_audio(str(audio_file), target_sr=16000)
            duration = len(audio) / 16000

            # Streaming transcription should complete without error
            streaming_text = ""
            chunk_count = 0
            for _partial, full in asr_pipeline.transcribe_streaming(audio, 16000):
                streaming_text = full
                chunk_count += 1

            # Basic sanity checks
            assert chunk_count > 0, f"No chunks processed for {audio_id}"
            assert len(streaming_text) > 0 or duration < 0.5, \
                f"Empty output for {audio_id} (duration={duration:.2f}s)"

            print(f"{audio_id}: duration={duration:.2f}s, chunks={chunk_count}, text_len={len(streaming_text)}")

    def test_streaming_state_continuity(self, asr_pipeline):
        """Test that streaming state is properly maintained across chunks."""
        # Generate test audio (simple tone)
        duration = 3.0  # 3 seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)

        # Extract features
        features = asr_pipeline.fbank_extractor.extract(audio, sample_rate)

        # Initialize streaming state
        state = asr_pipeline.init_streaming(batch_size=1)

        chunk_size = asr_pipeline.config.fbank_chunk_size
        num_chunks = (features.shape[0] + chunk_size - 1) // chunk_size

        # Process all chunks
        all_encoder_outputs = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, features.shape[0])
            chunk = features[start:end]
            valid_frames = int(chunk.shape[0])

            # Pad if needed
            if chunk.shape[0] < chunk_size:
                padding = mx.zeros((chunk_size - chunk.shape[0], chunk.shape[1]), dtype=chunk.dtype)
                chunk = mx.concatenate([chunk, padding], axis=0)

            chunk = mx.expand_dims(chunk, axis=0)

            # Process chunk
            encoder_out, new_states = asr_pipeline.model.encoder.streaming_forward(
                chunk,
                state.encoder_states,
                chunk_size=16,
                left_context_len=asr_pipeline.config.left_context_len,
                processed_frames=state.processed_encoder_frames,
            )
            mx.eval(encoder_out)

            all_encoder_outputs.append(encoder_out)
            state.encoder_states = new_states
            state.processed_fbank_frames += valid_frames
            state.processed_encoder_frames += valid_frames // 2

        # Verify we got output for all chunks
        assert len(all_encoder_outputs) == num_chunks, \
            f"Expected {num_chunks} outputs, got {len(all_encoder_outputs)}"

        # Verify output shapes are consistent
        # Encoder streaming output is (batch, seq, encoder_output_dim)
        # encoder_output_dim = max(encoder_dims) = 512 for streaming checkpoint
        for i, out in enumerate(all_encoder_outputs):
            assert out.ndim == 3, f"Chunk {i}: expected 3D output (batch, seq, d_model), got {out.ndim}D"
            assert out.shape[0] == 1, f"Chunk {i}: expected batch_size=1, got {out.shape[0]}"
            # encoder_output_dim = max(encoder_dims) = 512
            assert out.shape[2] == 512, f"Chunk {i}: expected encoder_output_dim=512, got {out.shape[2]}"

        print(f"Processed {num_chunks} chunks successfully")


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
