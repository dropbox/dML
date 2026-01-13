#!/usr/bin/env python3
"""
Validation tests for RichCTCHead.

Verifies that RichCTCHead produces outputs that match the individual
trained heads when loaded with the same weights.
"""

import sys
from pathlib import Path

import numpy as np

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available, skipping tests")
    sys.exit(0)

from tools.whisper_mlx.ctc_head import CTCDraftHead
from tools.whisper_mlx.kokoro_phoneme_head import KokoroPhonemeHead
from tools.whisper_mlx.rich_ctc_head import (
    EMOTION_CLASSES_34,
    WHISPER_LANGUAGE_IDS,
    ProsodyConditionedCTC,
    RichCTCConfig,
    RichCTCHead,
    detect_language_from_tokens,
)


def test_ctc_output_matches():
    """Test that RichCTCHead CTC output matches standalone CTCDraftHead."""
    print("Testing CTC output consistency...")

    # Paths
    ctc_path = "checkpoints/ctc_english_full/step_49000.npz"
    if not Path(ctc_path).exists():
        print(f"  SKIP: CTC checkpoint not found at {ctc_path}")
        return  # Skip if checkpoint not available

    # Load standalone CTC head
    standalone_ctc = CTCDraftHead.load_weights(ctc_path)

    # Load RichCTCHead
    rich_head = RichCTCHead.from_pretrained(ctc_path=ctc_path)

    # Generate test input
    mx.random.seed(42)
    encoder_output = mx.random.normal((1, 100, 1280))

    # Get outputs from both
    standalone_logits = standalone_ctc(encoder_output)
    rich_outputs = rich_head(encoder_output)
    rich_logits = rich_outputs["text_logits"]

    mx.eval(standalone_logits)
    mx.eval(rich_logits)

    # Compare
    max_diff = float(mx.max(mx.abs(standalone_logits - rich_logits)))
    print(f"  Max logit difference: {max_diff:.2e}")

    assert max_diff < 1e-5, f"CTC output consistency: FAIL (max_diff={max_diff:.2e})"
    print("  CTC output consistency: PASS")


def test_phoneme_output_matches():
    """Test that RichCTCHead phoneme output matches standalone KokoroPhonemeHead."""
    print("Testing phoneme output consistency...")

    # Paths
    phoneme_path = "checkpoints/kokoro_phoneme_head_v1/kokoro_phoneme_head_1700_best.npz"
    if not Path(phoneme_path).exists():
        print(f"  SKIP: Phoneme checkpoint not found at {phoneme_path}")
        return  # Skip if checkpoint not available

    # Load standalone phoneme head
    standalone_phoneme = KokoroPhonemeHead.from_pretrained(phoneme_path)

    # Load RichCTCHead
    rich_head = RichCTCHead.from_pretrained(phoneme_path=phoneme_path)

    # Generate test input
    mx.random.seed(42)
    encoder_output = mx.random.normal((1, 100, 1280))

    # Get outputs from both
    standalone_logits = standalone_phoneme(encoder_output)
    rich_outputs = rich_head(encoder_output)
    rich_logits = rich_outputs["phoneme"]

    mx.eval(standalone_logits)
    mx.eval(rich_logits)

    # Compare
    max_diff = float(mx.max(mx.abs(standalone_logits - rich_logits)))
    print(f"  Max logit difference: {max_diff:.2e}")

    assert max_diff < 1e-5, f"Phoneme output consistency: FAIL (max_diff={max_diff:.2e})"
    print("  Phoneme output consistency: PASS")


def test_timing_correctness():
    """Test that timing outputs are correct."""
    print("Testing timing correctness...")

    config = RichCTCConfig(frame_rate_hz=50.0)
    head = RichCTCHead(config)

    # 100 frames at 50Hz = 2 seconds
    encoder_output = mx.random.normal((1, 100, 1280))
    outputs = head(encoder_output)

    start_times = np.array(outputs["start_time_ms"][0])
    end_times = np.array(outputs["end_time_ms"][0])

    # First frame: 0-20ms
    assert abs(start_times[0] - 0.0) < 0.001, f"First frame start: {start_times[0]}"
    assert abs(end_times[0] - 20.0) < 0.001, f"First frame end: {end_times[0]}"

    # Last frame: 1980-2000ms
    assert abs(start_times[99] - 1980.0) < 0.001, f"Last frame start: {start_times[99]}"
    assert abs(end_times[99] - 2000.0) < 0.001, f"Last frame end: {end_times[99]}"

    # Frame duration should be 20ms
    for i in range(99):
        duration = end_times[i] - start_times[i]
        assert abs(duration - 20.0) < 0.001, f"Frame {i} duration: {duration}"

    print("  Timing correctness: PASS")


def test_language_detection_mapping():
    """Test that language token IDs are correctly mapped."""
    print("Testing language detection mapping...")

    # English should map to token 50259
    assert WHISPER_LANGUAGE_IDS["en"] == 50259, f"en -> {WHISPER_LANGUAGE_IDS['en']}"

    # Chinese should be 50260
    assert WHISPER_LANGUAGE_IDS["zh"] == 50260, f"zh -> {WHISPER_LANGUAGE_IDS['zh']}"

    # Test detection from tokens
    # English language token
    detected = detect_language_from_tokens([50259, 123, 456])
    assert detected == "en", f"Detected: {detected}"

    # Chinese language token
    detected = detect_language_from_tokens([50260, 123, 456])
    assert detected == "zh", f"Detected: {detected}"

    # No language token
    detected = detect_language_from_tokens([123, 456, 789])
    assert detected is None, f"Detected: {detected}"

    print("  Language detection mapping: PASS")


def test_output_shapes():
    """Test that all output shapes are correct for various input sizes."""
    print("Testing output shapes for various input sizes...")

    config = RichCTCConfig()
    head = RichCTCHead(config)

    test_cases = [
        (1, 50, 1280),   # Short
        (1, 100, 1280),  # Standard
        (1, 500, 1280),  # Medium
        (2, 100, 1280),  # Batched
    ]

    for batch, frames, d_model in test_cases:
        encoder_output = mx.random.normal((batch, frames, d_model))
        outputs = head(encoder_output)
        mx.eval(outputs)

        expected = {
            "text_logits": (batch, frames, config.text_vocab_size),
            "emotion": (batch, frames, config.num_emotions),
            "pitch_bins": (batch, frames, config.pitch_bins),
            "pitch_hz": (batch, frames, 1),
            "para": (batch, frames, config.num_para_classes),
            "phoneme": (batch, frames, config.phoneme_vocab),
        }

        for key, exp_shape in expected.items():
            actual = outputs[key].shape
            assert actual == exp_shape, f"Shape mismatch for {key}: {actual} != {exp_shape}"

        print(f"  Input ({batch}, {frames}, {d_model}): PASS")

    print("  Output shapes: PASS")


def test_unbatched_input():
    """Test that unbatched input (2D) works correctly."""
    print("Testing unbatched input...")

    config = RichCTCConfig()
    head = RichCTCHead(config)

    # 2D input (no batch dimension)
    encoder_output = mx.random.normal((100, 1280))
    outputs = head(encoder_output)
    mx.eval(outputs)

    # Outputs should be 2D (no batch)
    assert outputs["text_logits"].shape == (100, config.text_vocab_size)
    assert outputs["emotion"].shape == (100, config.num_emotions)
    assert outputs["pitch_hz"].shape == (100, 1)

    print("  Unbatched input: PASS")


def test_decoding_methods():
    """Test CTC and phoneme decoding methods."""
    print("Testing decoding methods...")

    head = RichCTCHead.from_pretrained()

    # Use a fixed seed for reproducibility
    mx.random.seed(123)
    encoder_output = mx.random.normal((1, 100, 1280))
    outputs = head(encoder_output)

    # Test text decoding
    tokens, tokens_with_timing = head.decode_text_greedy(outputs)
    assert isinstance(tokens, list), "Tokens should be a list"
    assert len(tokens_with_timing) == len(tokens), "Timing should match tokens"

    for token_id, start_ms, end_ms in tokens_with_timing:
        assert isinstance(token_id, int), f"Token ID should be int: {type(token_id)}"
        assert start_ms < end_ms, f"Start {start_ms} should be < end {end_ms}"
        assert 0 <= start_ms <= 2000, f"Start out of range: {start_ms}"

    # Test phoneme decoding
    phonemes = head.decode_phonemes_greedy(outputs)
    assert isinstance(phonemes, list), "Phonemes should be a list"
    assert all(isinstance(p, int) for p in phonemes), "Phonemes should be ints"

    print(f"  Decoded {len(tokens)} text tokens, {len(phonemes)} phonemes")
    print("  Decoding methods: PASS")


def test_emotion_methods():
    """Test emotion extraction methods."""
    print("Testing emotion methods...")

    head = RichCTCHead.from_pretrained()

    mx.random.seed(456)
    encoder_output = mx.random.normal((1, 100, 1280))
    outputs = head(encoder_output)

    # Test per-frame emotions
    labels, confidences = head.get_emotion_per_frame(outputs)
    assert len(labels) == 100, f"Should have 100 labels: {len(labels)}"
    assert len(confidences) == 100, f"Should have 100 confidences: {len(confidences)}"
    assert all(0 <= c <= 1 for c in confidences), "Confidences should be [0,1]"

    # Test utterance-level emotion
    emotion, conf = head.get_utterance_emotion(outputs)
    assert emotion in EMOTION_CLASSES_34, f"Unknown emotion: {emotion}"
    assert 0 <= conf <= 1, f"Confidence out of range: {conf}"

    print(f"  Utterance emotion: {emotion} ({conf:.3f})")
    print("  Emotion methods: PASS")


def test_pitch_extraction():
    """Test pitch extraction method."""
    print("Testing pitch extraction...")

    head = RichCTCHead.from_pretrained()

    mx.random.seed(789)
    encoder_output = mx.random.normal((1, 100, 1280))
    outputs = head(encoder_output)

    pitch_hz, voicing = head.get_pitch_hz(outputs)

    assert pitch_hz.shape == (100,), f"Pitch shape: {pitch_hz.shape}"
    assert voicing.shape == (100,), f"Voicing shape: {voicing.shape}"
    assert np.all(pitch_hz >= 0), "Pitch should be non-negative"
    assert np.all(voicing >= 0) and np.all(voicing <= 1), "Voicing should be [0,1]"

    print(f"  Mean pitch: {pitch_hz.mean():.1f} Hz, mean voicing: {voicing.mean():.3f}")
    print("  Pitch extraction: PASS")


def test_paralinguistics_detection():
    """Test paralinguistics event detection."""
    print("Testing paralinguistics detection...")

    head = RichCTCHead.from_pretrained()

    mx.random.seed(101112)
    encoder_output = mx.random.normal((1, 100, 1280))
    outputs = head(encoder_output)

    events = head.get_paralinguistics(outputs, threshold=0.01)  # Low threshold for testing

    assert isinstance(events, list), "Events should be a list"

    for event in events:
        class_id, class_name, start_ms, end_ms, confidence = event
        assert isinstance(class_id, int), f"Class ID should be int: {type(class_id)}"
        assert isinstance(class_name, str), f"Class name should be str: {type(class_name)}"
        assert start_ms < end_ms, f"Start {start_ms} should be < end {end_ms}"
        assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"

    print(f"  Detected {len(events)} paralinguistic events")
    print("  Paralinguistics detection: PASS")


def test_prosody_conditioned_ctc():
    """Test ProsodyConditionedCTC forward pass."""
    print("Testing ProsodyConditionedCTC...")

    # Create prosody-conditioned CTC
    prosody_ctc = ProsodyConditionedCTC(
        d_model=1280,
        emotion_dim=34,
        pitch_dim=1,
        prosody_dim=64,
        vocab_size=51865,
    )

    # Create test inputs
    mx.random.seed(9999)
    encoder_output = mx.random.normal((1, 100, 1280))
    emotion_seq = mx.random.normal((1, 100, 34))
    pitch_seq = mx.random.normal((1, 100, 1))

    # Forward pass
    logits = prosody_ctc(encoder_output, emotion_seq, pitch_seq)
    mx.eval(logits)

    # Check shape
    assert logits.shape == (1, 100, 51865), f"Shape mismatch: {logits.shape}"

    print(f"  Output shape: {logits.shape}")
    print("  ProsodyConditionedCTC: PASS")


def test_prosody_conditioned_ctc_init_from_weights():
    """Test initializing ProsodyConditionedCTC from CTC weights."""
    print("Testing ProsodyConditionedCTC from_ctc_head...")

    ctc_path = "checkpoints/ctc_english_full/step_49000.npz"
    if not Path(ctc_path).exists():
        print(f"  SKIP: CTC checkpoint not found at {ctc_path}")
        return  # Skip if checkpoint not available

    # Initialize from trained CTC weights
    prosody_ctc = ProsodyConditionedCTC.from_ctc_head(
        ctc_path,
        emotion_dim=34,
        pitch_dim=1,
        prosody_dim=64,
    )

    # Test forward pass
    mx.random.seed(8888)
    encoder_output = mx.random.normal((1, 100, 1280))
    emotion_seq = mx.random.normal((1, 100, 34))
    pitch_seq = mx.random.normal((1, 100, 1))

    logits = prosody_ctc(encoder_output, emotion_seq, pitch_seq)
    mx.eval(logits)

    assert logits.shape == (1, 100, 51865), f"Shape mismatch: {logits.shape}"

    print(f"  Output shape: {logits.shape}")
    print("  ProsodyConditionedCTC from_ctc_head: PASS")


def test_rich_ctc_head_with_prosody():
    """Test RichCTCHead with prosody conditioning enabled."""
    print("Testing RichCTCHead with prosody conditioning...")

    # Create config with prosody CTC enabled
    config = RichCTCConfig(use_prosody_ctc=True, prosody_dim=64)
    head = RichCTCHead(config)

    # Test forward pass
    mx.random.seed(7777)
    encoder_output = mx.random.normal((1, 100, 1280))
    outputs = head(encoder_output)
    mx.eval(outputs)

    # Check shapes
    assert outputs["text_logits"].shape == (1, 100, config.text_vocab_size)
    assert outputs["emotion"].shape == (1, 100, config.num_emotions)
    assert outputs["pitch_hz"].shape == (1, 100, 1)

    print(f"  text_logits shape: {outputs['text_logits'].shape}")
    print("  RichCTCHead with prosody: PASS")


def test_prosody_ctc_affects_output():
    """Test that prosody features actually affect CTC output."""
    print("Testing prosody features affect output...")

    prosody_ctc = ProsodyConditionedCTC(
        d_model=1280,
        emotion_dim=34,
        pitch_dim=1,
        prosody_dim=64,
        vocab_size=51865,
    )

    # Same encoder output, different prosody
    mx.random.seed(6666)
    encoder_output = mx.random.normal((1, 100, 1280))

    # Two different emotion/pitch sequences
    emotion_seq_1 = mx.zeros((1, 100, 34))
    emotion_seq_1 = emotion_seq_1.at[:, :, 0].add(1.0)  # "neutral"

    emotion_seq_2 = mx.zeros((1, 100, 34))
    emotion_seq_2 = emotion_seq_2.at[:, :, 2].add(1.0)  # "happy"

    pitch_seq_1 = mx.full((1, 100, 1), 100.0)  # Low pitch
    pitch_seq_2 = mx.full((1, 100, 1), 300.0)  # High pitch

    # Get outputs
    logits_1 = prosody_ctc(encoder_output, emotion_seq_1, pitch_seq_1)
    logits_2 = prosody_ctc(encoder_output, emotion_seq_2, pitch_seq_2)
    mx.eval(logits_1)
    mx.eval(logits_2)

    # Outputs should be different
    diff = float(mx.max(mx.abs(logits_1 - logits_2)))
    assert diff > 0.01, f"Prosody should affect output, but diff={diff}"

    print(f"  Max output difference: {diff:.4f}")
    print("  Prosody affects output: PASS")


def test_save_and_load():
    """Test saving and loading RichCTCHead."""
    print("Testing save and load...")

    import os
    import tempfile

    head = RichCTCHead.from_pretrained()

    # Get outputs before save
    mx.random.seed(12345)
    encoder_output = mx.random.normal((1, 50, 1280))
    outputs_before = head(encoder_output)
    mx.eval(outputs_before)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "rich_ctc_head.npz")
        head.save(save_path)

        assert os.path.exists(save_path), "Save file should exist"

        # Note: Loading unified checkpoint would require a different loading method
        # For now, just verify save completes without error
        file_size = os.path.getsize(save_path)
        print(f"  Saved to {save_path} ({file_size / 1024 / 1024:.1f} MB)")

    print("  Save: PASS")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("RichCTCHead Validation Tests")
    print("=" * 60)

    tests = [
        test_timing_correctness,
        test_language_detection_mapping,
        test_output_shapes,
        test_unbatched_input,
        test_decoding_methods,
        test_emotion_methods,
        test_pitch_extraction,
        test_paralinguistics_detection,
        test_ctc_output_matches,
        test_phoneme_output_matches,
        test_prosody_conditioned_ctc,
        test_prosody_conditioned_ctc_init_from_weights,
        test_rich_ctc_head_with_prosody,
        test_prosody_ctc_affects_output,
        test_save_and_load,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()  # Tests now use assertions instead of return values
            passed += 1
        except AssertionError as e:
            print(f"  ASSERTION FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
