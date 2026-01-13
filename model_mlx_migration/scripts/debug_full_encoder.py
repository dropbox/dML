#!/usr/bin/env python3
"""
Debug full encoder to trace multi-stage processing.

Traces through all 6 encoder stages and _get_full_dim_output.
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/ayates/model_mlx_migration')

import mlx.core as mx
from src.models.zipformer.features import load_audio, FbankExtractor
from src.models.zipformer.asr_model import ASRModelConfig, load_checkpoint

CHECKPOINT_PATH = 'checkpoints/zipformer/en-streaming/exp/pretrained.pt'
BPE_MODEL_PATH = 'checkpoints/zipformer/en-streaming/data/lang_bpe_500/bpe.model'
AUDIO_PATH = 'checkpoints/zipformer/en-streaming/test_wavs/1089-134686-0001.wav'


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray):
    """Compare arrays and report."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False

    max_diff = np.max(np.abs(a - b))
    mean_diff = np.mean(np.abs(a - b))

    if max_diff < 1e-3:
        status = "OK"
    elif max_diff < 1e-1:
        status = "WARN"
    else:
        status = "FAIL"

    print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.8f}, shape={a.shape}")
    return max_diff < 1e-1


def debug_full_encoder():
    """Debug through all encoder stages."""

    print("=" * 70)
    print("Debugging Full Encoder (All Stages)")
    print("=" * 70)

    # Load MLX model
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(BPE_MODEL_PATH)

    model_config = ASRModelConfig(
        vocab_size=sp.GetPieceSize(),
        decoder_dim=512,
        context_size=2,
        joiner_dim=512,
        blank_id=0,
    )
    mlx_model, _ = load_checkpoint(CHECKPOINT_PATH, model_config)
    mx.eval(mlx_model.parameters())

    # Load audio
    waveform, sr = load_audio(AUDIO_PATH, target_sr=16000)
    fbank_extractor = FbankExtractor()
    features = fbank_extractor.extract(waveform, sr)
    features_batch = mx.expand_dims(features, axis=0)

    print(f"\nInput features: {features_batch.shape}")

    # Get encoder
    encoder = mlx_model.encoder

    # 1. encoder_embed
    print("\n--- encoder_embed ---")
    x = encoder.encoder_embed(features_batch)
    mx.eval(x)
    print(f"  output shape: {x.shape}")
    print(f"  stats: mean={np.mean(np.array(x)):.6f}, std={np.std(np.array(x)):.6f}")

    # Store outputs from each stage
    outputs = []

    # Process through encoder stages
    for i, enc in enumerate(encoder.encoders):
        print(f"\n--- Encoder Stage {i} ---")

        # Convert channels
        target_dim = encoder.encoder_dims[i]
        x = encoder._convert_num_channels(x, target_dim)
        mx.eval(x)
        print(f"  after _convert_num_channels: shape={x.shape}, target={target_dim}")

        # Run encoder
        x = enc(x)
        mx.eval(x)
        outputs.append(x)

        print(f"  output shape: {x.shape}")
        print(f"  stats: mean={np.mean(np.array(x)):.6f}, std={np.std(np.array(x)):.6f}")
        print(f"  range: [{np.min(np.array(x)):.6f}, {np.max(np.array(x)):.6f}]")

    # _get_full_dim_output
    print("\n--- _get_full_dim_output ---")
    print(f"  encoder_dims: {encoder.encoder_dims}")
    print(f"  output_dim (max): {encoder.output_dim}")

    # Manual trace
    output_pieces = [outputs[-1]]
    cur_dim = encoder.encoder_dims[-1]
    print(f"  Start: piece[0] from stage 5 with dim {cur_dim}")

    for i in range(len(encoder.encoder_dims) - 2, -1, -1):
        d = encoder.encoder_dims[i]
        if d > cur_dim:
            this_output = outputs[i]
            piece = this_output[..., cur_dim:d]
            mx.eval(piece)
            output_pieces.append(piece)
            print(f"  Adding: stage {i} channels [{cur_dim}:{d}] shape {piece.shape}")
            cur_dim = d

    combined = mx.concatenate(output_pieces, axis=-1)
    mx.eval(combined)
    print(f"  Combined shape: {combined.shape}")
    print(f"  Combined stats: mean={np.mean(np.array(combined)):.6f}, std={np.std(np.array(combined)):.6f}")

    # Final downsampling
    print("\n--- Final Downsampling ---")
    seq_len, batch_size, d_model = combined.shape
    ds = 2
    d_seq_len = (seq_len + ds - 1) // ds

    pad = d_seq_len * ds - seq_len
    if pad > 0:
        combined = mx.concatenate([combined, mx.broadcast_to(combined[-1:], (pad, batch_size, d_model))], axis=0)

    combined = mx.reshape(combined, (d_seq_len, ds, batch_size, d_model))
    weights = mx.softmax(encoder.downsample_output_bias)[:, None, None]
    combined = mx.sum(combined * weights, axis=1)
    mx.eval(combined)

    # Transpose to (batch, time, d_model)
    final = mx.transpose(combined, (1, 0, 2))
    mx.eval(final)

    print(f"  Final encoder output: {final.shape}")
    print(f"  stats: mean={np.mean(np.array(final)):.6f}, std={np.std(np.array(final)):.6f}")
    print(f"  range: [{np.min(np.array(final)):.6f}, {np.max(np.array(final)):.6f}]")

    # Now run the actual encoder forward and compare
    print("\n--- Comparing with Actual Forward ---")
    x_lens = mx.array([features_batch.shape[1]], dtype=mx.int32)
    actual_out, actual_lens = encoder(features_batch, x_lens)
    mx.eval(actual_out, actual_lens)

    print(f"  Actual output: {actual_out.shape}")
    print(f"  Actual lens: {actual_lens}")
    print(f"  stats: mean={np.mean(np.array(actual_out)):.6f}, std={np.std(np.array(actual_out)):.6f}")

    compare_arrays("manual_vs_actual", np.array(final), np.array(actual_out))

    # Check decoder output for blank token
    print("\n--- Decoder Test ---")
    blank_tokens = mx.array([[0, 0]], dtype=mx.int32)  # context_size=2
    decoder_out = mlx_model.decoder(blank_tokens)
    mx.eval(decoder_out)
    print(f"  Decoder out shape: {decoder_out.shape}")
    print(f"  Decoder out for blank: mean={np.mean(np.array(decoder_out)):.6f}, std={np.std(np.array(decoder_out)):.6f}")

    # Check joiner output
    print("\n--- Joiner Test ---")
    # Take first encoder frame
    enc_frame = actual_out[:, :1, :]  # (1, 1, 512)
    joiner_out = mlx_model.joiner(enc_frame, decoder_out)
    mx.eval(joiner_out)
    print(f"  Joiner out shape: {joiner_out.shape}")

    # Get probabilities
    probs = mx.softmax(joiner_out, axis=-1)
    mx.eval(probs)
    probs_np = np.array(probs)

    print(f"  Blank (id=0) probability: {probs_np[0, 0, 0, 0]:.6f}")
    print(f"  Top 5 tokens: {np.argsort(probs_np[0, 0, 0, :])[-5:][::-1]}")
    print(f"  Top 5 probs: {np.sort(probs_np[0, 0, 0, :])[-5:][::-1]}")

    # Check a few more frames
    print("\n--- Joiner outputs for frames 0-5 ---")
    for frame_idx in range(min(5, actual_out.shape[1])):
        enc_frame = actual_out[:, frame_idx:frame_idx+1, :]
        joiner_out = mlx_model.joiner(enc_frame, decoder_out)
        mx.eval(joiner_out)
        probs = mx.softmax(joiner_out, axis=-1)
        mx.eval(probs)
        probs_np = np.array(probs)

        blank_prob = probs_np[0, 0, 0, 0]
        top_idx = np.argmax(probs_np[0, 0, 0, :])
        top_prob = probs_np[0, 0, 0, top_idx]
        print(f"  Frame {frame_idx}: blank_prob={blank_prob:.4f}, top_idx={top_idx}, top_prob={top_prob:.4f}")


if __name__ == "__main__":
    debug_full_encoder()
