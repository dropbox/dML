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
Test ExtendedSingingHead architecture.

Tests:
1. Forward pass shape validation
2. Predict function output format
3. Streaming predictions
4. Loss function
5. WhisperMultiHead integration
6. Backward compatibility with legacy SingingHead
"""

import sys
sys.path.insert(0, '.')

import mlx.core as mx

from tools.whisper_mlx.multi_head import (
    MultiHeadConfig,
    ExtendedSingingHead,
    SingingHead,
    WhisperMultiHead,
    extended_singing_loss,
    SINGING_STYLES,
)


def test_forward_pass():
    """Test forward pass shape validation."""
    print("Test: Forward pass shape validation")

    config = MultiHeadConfig(d_model=1280, use_extended_singing=True)
    head = ExtendedSingingHead(config)

    # Create mock encoder output
    batch_size = 2
    seq_len = 100
    encoder_output = mx.random.normal((batch_size, seq_len, config.d_model))

    # Forward pass (utterance-level)
    singing_logit, style_logits, intensity = head(encoder_output)

    # Check shapes
    assert singing_logit.shape == (batch_size, 1), f"singing_logit shape: {singing_logit.shape}"
    assert style_logits.shape == (batch_size, config.num_singing_styles), f"style_logits shape: {style_logits.shape}"
    assert intensity.shape == (batch_size, 1), f"intensity shape: {intensity.shape}"

    # Check intensity is in [0, 1]
    assert mx.all(intensity >= 0) and mx.all(intensity <= 1), "intensity not in [0, 1]"

    print(f"  singing_logit: {singing_logit.shape} PASS")
    print(f"  style_logits: {style_logits.shape} PASS")
    print(f"  intensity: {intensity.shape} (range: {float(mx.min(intensity)):.3f}-{float(mx.max(intensity)):.3f}) PASS")

    return True


def test_forward_pass_frame_level():
    """Test frame-level forward pass."""
    print("\nTest: Frame-level forward pass")

    config = MultiHeadConfig(d_model=1280, use_extended_singing=True)
    head = ExtendedSingingHead(config)

    batch_size = 2
    seq_len = 100
    encoder_output = mx.random.normal((batch_size, seq_len, config.d_model))

    # Forward pass with frame logits
    singing_logit, style_logits, intensity = head(encoder_output, return_frame_logits=True)

    # Check shapes
    assert singing_logit.shape == (batch_size, seq_len, 1), f"singing_logit shape: {singing_logit.shape}"
    assert style_logits.shape == (batch_size, seq_len, config.num_singing_styles), f"style_logits shape: {style_logits.shape}"
    assert intensity.shape == (batch_size, seq_len, 1), f"intensity shape: {intensity.shape}"

    print(f"  singing_logit: {singing_logit.shape} PASS")
    print(f"  style_logits: {style_logits.shape} PASS")
    print(f"  intensity: {intensity.shape} PASS")

    return True


def test_predict_function():
    """Test predict function."""
    print("\nTest: Predict function")

    config = MultiHeadConfig(d_model=1280, use_extended_singing=True)
    head = ExtendedSingingHead(config)

    # Single utterance
    encoder_output = mx.random.normal((1, 50, config.d_model))

    is_singing, singing_conf, style, style_conf, intensity = head.predict(encoder_output)

    assert isinstance(is_singing, bool), "is_singing should be bool"
    assert 0 <= singing_conf <= 1, "singing_conf should be in [0, 1]"
    assert style in SINGING_STYLES or style.startswith("style_"), f"Invalid style: {style}"
    assert 0 <= style_conf <= 1, "style_conf should be in [0, 1]"
    assert 0 <= intensity <= 1, "intensity should be in [0, 1]"

    print(f"  is_singing: {is_singing}")
    print(f"  singing_confidence: {singing_conf:.3f}")
    print(f"  style: {style} ({style_conf:.3f})")
    print(f"  intensity: {intensity:.3f}")
    print("  PASS")

    return True


def test_predict_label():
    """Test predict_label function."""
    print("\nTest: Predict label function")

    config = MultiHeadConfig(d_model=1280, use_extended_singing=True)
    head = ExtendedSingingHead(config)

    encoder_output = mx.random.normal((1, 50, config.d_model))

    result = head.predict_label(encoder_output)

    assert "is_singing" in result
    assert "singing_confidence" in result
    assert "style" in result
    assert "style_confidence" in result
    assert "intensity" in result

    print(f"  Result: {result}")
    print("  PASS")

    return True


def test_streaming_predictions():
    """Test streaming frame-level predictions."""
    print("\nTest: Streaming predictions")

    config = MultiHeadConfig(d_model=1280, use_extended_singing=True)
    head = ExtendedSingingHead(config)

    encoder_output = mx.random.normal((1, 20, config.d_model))

    results = head.predict_streaming(encoder_output)

    assert len(results) == 20, f"Should have 20 frames, got {len(results)}"

    # Check first frame
    first = results[0]
    assert "frame_idx" in first
    assert "is_singing" in first
    assert "singing_prob" in first
    assert "style" in first
    assert "intensity" in first

    print(f"  Number of frames: {len(results)}")
    print(f"  First frame: {first}")
    print("  PASS")

    return True


def test_loss_function():
    """Test extended singing loss function."""
    print("\nTest: Loss function")

    batch_size = 4
    num_styles = 10

    # Create mock predictions
    singing_logits = mx.random.normal((batch_size, 1))
    style_logits = mx.random.normal((batch_size, num_styles))
    intensity_pred = mx.random.uniform(shape=(batch_size, 1))

    # Create targets
    # 2 singing, 2 speaking
    target_singing = mx.array([1, 1, 0, 0])
    target_style = mx.array([3, 7, 0, 0])  # Style only matters for singing
    target_intensity = mx.array([0.8, 0.5, 0.0, 0.0])

    total_loss, loss_dict = extended_singing_loss(
        singing_logits,
        style_logits,
        intensity_pred,
        target_singing,
        target_style,
        target_intensity,
    )

    assert total_loss.shape == (), f"total_loss should be scalar, got {total_loss.shape}"
    assert "singing_loss" in loss_dict
    assert "style_loss" in loss_dict
    assert "intensity_loss" in loss_dict

    print(f"  Total loss: {float(total_loss):.4f}")
    print(f"  Singing loss: {float(loss_dict['singing_loss']):.4f}")
    print(f"  Style loss: {float(loss_dict['style_loss']):.4f}")
    print(f"  Intensity loss: {float(loss_dict['intensity_loss']):.4f}")
    print("  PASS")

    return True


def test_whisper_multi_head_integration():
    """Test WhisperMultiHead integration."""
    print("\nTest: WhisperMultiHead integration")

    # Test with extended singing head
    config = MultiHeadConfig(
        d_model=1280,
        use_extended_singing=True,
        use_attention_emotion=False,
        use_crepe_pitch=False,
    )

    multi_head = WhisperMultiHead(config)

    # Check singing head type
    assert isinstance(multi_head.singing_head, ExtendedSingingHead), \
        f"Expected ExtendedSingingHead, got {type(multi_head.singing_head)}"

    # Forward pass
    encoder_output = mx.random.normal((1, 50, config.d_model))
    outputs = multi_head(encoder_output)

    # Check outputs
    assert "singing_logit" in outputs
    assert "style_logits" in outputs, "Missing style_logits in outputs"
    assert "intensity" in outputs, "Missing intensity in outputs"
    assert "emotion_logits" in outputs
    assert "pitch_hz" in outputs

    print(f"  singing_logit: {outputs['singing_logit'].shape}")
    print(f"  style_logits: {outputs['style_logits'].shape}")
    print(f"  intensity: {outputs['intensity'].shape}")
    print("  PASS")

    return True


def test_whisper_multi_head_predict_all():
    """Test WhisperMultiHead predict_all with extended singing."""
    print("\nTest: WhisperMultiHead predict_all")

    config = MultiHeadConfig(
        d_model=1280,
        use_extended_singing=True,
    )

    multi_head = WhisperMultiHead(config)
    encoder_output = mx.random.normal((1, 50, config.d_model))

    results = multi_head.predict_all(encoder_output)

    assert "is_singing" in results
    assert "singing_confidence" in results
    # Style and intensity only present if extended singing head
    assert "style" in results, "Missing style in results"
    assert "intensity" in results, "Missing intensity in results"

    print(f"  is_singing: {results['is_singing']}")
    print(f"  singing_confidence: {results['singing_confidence']:.3f}")
    print(f"  style: {results.get('style')}")
    print(f"  intensity: {results.get('intensity', 'N/A')}")
    print("  PASS")

    return True


def test_backward_compatibility():
    """Test backward compatibility with legacy SingingHead."""
    print("\nTest: Backward compatibility")

    # Legacy config (use_extended_singing=False)
    config = MultiHeadConfig(
        d_model=1280,
        use_extended_singing=False,
    )

    multi_head = WhisperMultiHead(config)

    # Check singing head type
    assert isinstance(multi_head.singing_head, SingingHead), \
        f"Expected SingingHead, got {type(multi_head.singing_head)}"

    # Forward pass
    encoder_output = mx.random.normal((1, 50, config.d_model))
    outputs = multi_head(encoder_output)

    # Check that only singing_logit is present (no style or intensity)
    assert "singing_logit" in outputs
    assert "style_logits" not in outputs, "Legacy mode should not have style_logits"
    assert "intensity" not in outputs, "Legacy mode should not have intensity"

    print(f"  singing_logit: {outputs['singing_logit'].shape}")
    print("  Legacy mode: no style_logits, no intensity PASS")

    return True


def test_singing_styles_taxonomy():
    """Test singing styles taxonomy."""
    print("\nTest: Singing styles taxonomy")

    assert len(SINGING_STYLES) == 10, f"Expected 10 styles, got {len(SINGING_STYLES)}"

    expected_styles = [
        "belt", "breathy", "classical", "folk", "jazz",
        "pop", "rock", "soft", "vibrato", "neutral"
    ]

    for i, style in enumerate(expected_styles):
        assert SINGING_STYLES[i] == style, f"Style {i} mismatch: {SINGING_STYLES[i]} != {style}"

    print(f"  Styles: {SINGING_STYLES}")
    print("  PASS")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ExtendedSingingHead Test Suite")
    print("=" * 60)

    tests = [
        test_forward_pass,
        test_forward_pass_frame_level,
        test_predict_function,
        test_predict_label,
        test_streaming_predictions,
        test_loss_function,
        test_whisper_multi_head_integration,
        test_whisper_multi_head_predict_all,
        test_backward_compatibility,
        test_singing_styles_taxonomy,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nTest {test.__name__} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed+failed} tests passed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
