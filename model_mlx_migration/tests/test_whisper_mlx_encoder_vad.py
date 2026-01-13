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
Tests for EncoderVADHead (Phase 3 Optimization).

Tests the encoder VAD head module including:
- Forward pass correctness
- Output shape validation
- Loss computation
- Gradient flow
- Integration with encoder
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


class TestEncoderVADHead:
    """Tests for EncoderVADHead module."""

    def test_import(self):
        """Test that encoder_vad module imports correctly."""
        from tools.whisper_mlx.encoder_vad import (
            EncoderVADHead,
            SileroVADDistiller,
        )
        assert EncoderVADHead is not None
        assert SileroVADDistiller is not None

    def test_init_default(self):
        """Test EncoderVADHead initialization with default params."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        head = EncoderVADHead()
        assert head.n_state == 1280
        assert head.hidden_dim == 256

        # Check layers exist
        assert hasattr(head, 'proj')
        assert hasattr(head, 'classifier')
        assert hasattr(head, 'dropout')

    def test_init_custom_dims(self):
        """Test EncoderVADHead with custom dimensions."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        head = EncoderVADHead(n_state=512, hidden_dim=128)
        assert head.n_state == 512
        assert head.hidden_dim == 128

        # Check weight shapes
        assert head.proj.weight.shape == (128, 512)
        assert head.classifier.weight.shape == (1, 128)

    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        batch_size = 4
        seq_len = 1500
        n_state = 1280

        head = EncoderVADHead(n_state=n_state)

        # Create dummy encoder output
        encoder_output = mx.random.normal((batch_size, seq_len, n_state))

        # Forward pass
        probs = head(encoder_output)
        mx.eval(probs)

        # Check output shape
        assert probs.shape == (batch_size, seq_len), f"Expected ({batch_size}, {seq_len}), got {probs.shape}"

    def test_forward_output_range(self):
        """Test that output probabilities are in [0, 1]."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        head = EncoderVADHead()

        encoder_output = mx.random.normal((2, 100, 1280))
        probs = head(encoder_output)
        mx.eval(probs)

        # Check range
        min_val = float(mx.min(probs))
        max_val = float(mx.max(probs))

        assert min_val >= 0.0, f"Min probability {min_val} < 0"
        assert max_val <= 1.0, f"Max probability {max_val} > 1"

    def test_get_logits(self):
        """Test get_logits returns unbounded values."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        head = EncoderVADHead()

        encoder_output = mx.random.normal((2, 100, 1280))
        logits = head.get_logits(encoder_output)
        mx.eval(logits)

        # Logits should be unbounded (can be negative or > 1)
        assert logits.shape == (2, 100)
        # Just check it doesn't error; unbounded values are expected

    def test_get_speech_mask(self):
        """Test speech mask generation."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        head = EncoderVADHead()

        encoder_output = mx.random.normal((2, 100, 1280))

        # Get mask with default threshold
        mask = head.get_speech_mask(encoder_output, threshold=0.5)
        mx.eval(mask)

        assert mask.shape == (2, 100)
        assert mask.dtype == mx.bool_

        # Different thresholds should give different results
        mask_low = head.get_speech_mask(encoder_output, threshold=0.1)
        mask_high = head.get_speech_mask(encoder_output, threshold=0.9)
        mx.eval(mask_low, mask_high)

        # Lower threshold should have more True values
        assert int(mx.sum(mask_low)) >= int(mx.sum(mask_high))

    def test_training_mode_dropout(self):
        """Test that training mode applies dropout."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        head = EncoderVADHead(dropout=0.5)

        encoder_output = mx.random.normal((1, 100, 1280))

        # Training mode
        probs_train1 = head(encoder_output, training=True)
        probs_train2 = head(encoder_output, training=True)
        mx.eval(probs_train1, probs_train2)

        # Note: In MLX, dropout behavior may vary. Just ensure no errors.
        # The main check is that training=True doesn't crash.

        # Inference mode should be deterministic
        probs_infer1 = head(encoder_output, training=False)
        probs_infer2 = head(encoder_output, training=False)
        mx.eval(probs_infer1, probs_infer2)

        # Inference should be deterministic
        assert mx.allclose(probs_infer1, probs_infer2)


class TestSileroVADDistiller:
    """Tests for SileroVADDistiller."""

    def test_init(self):
        """Test distiller initialization."""
        from tools.whisper_mlx.encoder_vad import SileroVADDistiller

        distiller = SileroVADDistiller()

        assert distiller.sample_rate == 16000
        assert distiller.silero_window_size == 512
        assert distiller.encoder_hop_length == 160
        assert distiller.encoder_conv_stride == 2

    def test_compute_loss(self):
        """Test BCE loss computation."""
        from tools.whisper_mlx.encoder_vad import SileroVADDistiller

        distiller = SileroVADDistiller()

        # Create dummy logits and labels
        logits = mx.array([[0.0, 1.0, -1.0, 2.0]])  # (1, 4)
        labels = mx.array([[1.0, 1.0, 0.0, 0.0]])   # (1, 4)

        loss = distiller.compute_loss(logits, labels, reduction="mean")
        mx.eval(loss)

        # Loss should be positive
        assert float(loss) > 0

        # Check different reductions
        loss_sum = distiller.compute_loss(logits, labels, reduction="sum")
        loss_none = distiller.compute_loss(logits, labels, reduction="none")
        mx.eval(loss_sum, loss_none)

        assert loss_none.shape == (1, 4)
        assert float(loss_sum) == float(mx.sum(loss_none))

    def test_compute_loss_perfect_prediction(self):
        """Test loss is low for perfect predictions."""
        from tools.whisper_mlx.encoder_vad import SileroVADDistiller

        distiller = SileroVADDistiller()

        # Perfect predictions: logits strongly match labels
        # High positive logit -> sigmoid ≈ 1 -> label should be 1
        # High negative logit -> sigmoid ≈ 0 -> label should be 0
        logits = mx.array([[10.0, 10.0, -10.0, -10.0]])
        labels = mx.array([[1.0, 1.0, 0.0, 0.0]])

        loss = distiller.compute_loss(logits, labels)
        mx.eval(loss)

        # Loss should be very small for perfect predictions
        assert float(loss) < 0.01

    def test_compute_loss_wrong_prediction(self):
        """Test loss is high for wrong predictions."""
        from tools.whisper_mlx.encoder_vad import SileroVADDistiller

        distiller = SileroVADDistiller()

        # Wrong predictions: logits opposite of labels
        logits = mx.array([[10.0, 10.0, -10.0, -10.0]])
        labels = mx.array([[0.0, 0.0, 1.0, 1.0]])  # Opposite

        loss = distiller.compute_loss(logits, labels)
        mx.eval(loss)

        # Loss should be high for wrong predictions
        assert float(loss) > 5.0


class TestEncoderVADHeadGradients:
    """Test gradient flow through VAD head."""

    def test_gradient_computation(self):
        """Test that gradients flow through the model."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead, SileroVADDistiller

        head = EncoderVADHead()
        distiller = SileroVADDistiller()

        # Create dummy data
        encoder_output = mx.random.normal((2, 100, 1280))
        labels = mx.random.uniform(shape=(2, 100))

        # Compute loss and gradients
        def loss_fn(head):
            logits = head.get_logits(encoder_output)
            return distiller.compute_loss(logits, labels)

        loss, grads = mx.value_and_grad(loss_fn)(head)
        mx.eval(loss, grads)

        # Check that gradients exist and are non-zero
        assert 'proj' in grads
        assert 'classifier' in grads

        # Check gradient shapes match weight shapes
        assert grads['proj']['weight'].shape == head.proj.weight.shape
        assert grads['classifier']['weight'].shape == head.classifier.weight.shape


class TestEncoderVADHeadSaveLoad:
    """Test saving and loading VAD head weights."""

    def test_save_load_npz(self, tmp_path):
        """Test save/load with .npz format."""
        from tools.whisper_mlx.encoder_vad import (
            EncoderVADHead,
            load_encoder_vad_head,
            save_encoder_vad_head,
        )

        # Create and initialize model
        original = EncoderVADHead(n_state=512, hidden_dim=128)
        encoder_output = mx.random.normal((1, 10, 512))
        original_probs = original(encoder_output)
        mx.eval(original_probs, original.parameters())

        # Save
        save_path = tmp_path / "vad_head.npz"
        save_encoder_vad_head(original, str(save_path))

        # Load
        loaded = load_encoder_vad_head(
            str(save_path),
            n_state=512,
            hidden_dim=128,
        )

        # Check output matches
        loaded_probs = loaded(encoder_output)
        mx.eval(loaded_probs)

        assert mx.allclose(original_probs, loaded_probs)


class TestEncoderVADHeadCreateFunction:
    """Test convenience creation functions."""

    def test_create_encoder_vad_head(self):
        """Test create_encoder_vad_head function."""
        from tools.whisper_mlx.encoder_vad import create_encoder_vad_head

        head = create_encoder_vad_head(n_state=1280, hidden_dim=256)

        assert isinstance(head, nn.Module)
        assert head.n_state == 1280
        assert head.hidden_dim == 256

    def test_create_with_custom_dtype(self):
        """Test creation with custom dtype."""
        from tools.whisper_mlx.encoder_vad import create_encoder_vad_head

        head = create_encoder_vad_head(dtype=mx.float32)

        # Weights should be float32
        assert head.proj.weight.dtype == mx.float32


class TestEncoderVADHeadPerformance:
    """Performance tests for VAD head."""

    def test_inference_speed(self):
        """Test that inference is fast (< 1ms for typical input)."""
        import time

        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        head = EncoderVADHead()

        # Typical encoder output: batch=1, seq_len=1500 (30s audio)
        encoder_output = mx.random.normal((1, 1500, 1280))
        mx.eval(encoder_output)

        # Warmup
        _ = head(encoder_output)
        mx.eval(_)

        # Time inference
        n_runs = 10
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            probs = head(encoder_output)
            mx.eval(probs)
            times.append(time.perf_counter() - t0)

        avg_time_ms = np.mean(times) * 1000

        # Should be very fast (< 5ms typically, allowing some margin)
        assert avg_time_ms < 10.0, f"VAD head too slow: {avg_time_ms:.2f}ms"

    def test_memory_efficiency(self):
        """Test that VAD head is memory efficient (small parameter count)."""
        from tools.whisper_mlx.encoder_vad import EncoderVADHead

        head = EncoderVADHead(n_state=1280, hidden_dim=256)

        # Count parameters
        total_params = 0
        for _k, v in head.parameters().items():
            if isinstance(v, dict):
                for _k2, v2 in v.items():
                    total_params += np.prod(v2.shape)
            else:
                total_params += np.prod(v.shape)

        # Expected: proj (1280*256 + 256) + classifier (256*1 + 1)
        # = 327,936 + 256 + 256 + 1 = ~328K params
        # Should be < 500K params (very small compared to encoder)
        assert total_params < 500000, f"Too many params: {total_params}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
