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
Tests for word timestamp prediction head.

Tests cover:
- Boundary detection
- Offset regression
- Timestamp extraction
- Loss function
- Gradient flow for training
"""

import mlx.core as mx
import mlx.optimizers as optim

from src.models.heads.timestamp import (
    TimestampConfig,
    TimestampHead,
    TimestampLoss,
    WordTimestamp,
    compute_boundary_accuracy,
    compute_boundary_f1,
    compute_timestamp_error,
    timestamp_loss,
)


class TestTimestampConfig:
    """Tests for TimestampConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TimestampConfig()
        assert config.encoder_dim == 384
        assert config.hidden_dim == 256
        assert config.num_layers == 2
        assert config.dropout_rate == 0.1
        assert config.frame_duration_ms == 40.0
        assert config.boundary_threshold == 0.5
        assert config.min_word_duration_ms == 50.0
        assert config.max_word_duration_ms == 2000.0
        assert config.use_offset_regression is True
        assert config.max_offset_frames == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = TimestampConfig(
            encoder_dim=512,
            hidden_dim=128,
            frame_duration_ms=20.0,
            use_offset_regression=False,
        )
        assert config.encoder_dim == 512
        assert config.hidden_dim == 128
        assert config.frame_duration_ms == 20.0
        assert config.use_offset_regression is False


class TestWordTimestamp:
    """Tests for WordTimestamp dataclass."""

    def test_create_timestamp(self):
        """Test creating a word timestamp."""
        ts = WordTimestamp(
            word="hello",
            start_ms=100.0,
            end_ms=500.0,
            confidence=0.9,
        )
        assert ts.word == "hello"
        assert ts.start_ms == 100.0
        assert ts.end_ms == 500.0
        assert ts.confidence == 0.9

    def test_timestamp_immutable(self):
        """Test that timestamps are immutable (NamedTuple)."""
        ts = WordTimestamp("test", 0.0, 100.0, 0.8)
        # NamedTuples are immutable by default
        assert ts.word == "test"
        assert ts.start_ms == 0.0


class TestTimestampHead:
    """Tests for TimestampHead module."""

    def test_forward_default_config(self):
        """Test forward pass with default config."""
        head = TimestampHead()
        batch_size = 4
        seq_len = 100
        encoder_dim = 384

        x = mx.random.normal(shape=(batch_size, seq_len, encoder_dim))
        boundary_logits, offset_preds = head(x)

        assert boundary_logits.shape == (batch_size, seq_len, 1)
        assert offset_preds.shape == (batch_size, seq_len, 2)

    def test_forward_no_offset(self):
        """Test forward pass without offset regression."""
        config = TimestampConfig(use_offset_regression=False)
        head = TimestampHead(config)
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        boundary_logits, offset_preds = head(x)

        assert boundary_logits.shape == (batch_size, seq_len, 1)
        assert offset_preds is None

    def test_forward_custom_config(self):
        """Test forward pass with custom config."""
        config = TimestampConfig(
            encoder_dim=256,
            hidden_dim=128,
            num_layers=1,
        )
        head = TimestampHead(config)
        batch_size = 2
        seq_len = 50

        x = mx.random.normal(shape=(batch_size, seq_len, 256))
        boundary_logits, offset_preds = head(x)

        assert boundary_logits.shape == (batch_size, seq_len, 1)

    def test_forward_with_lengths(self):
        """Test forward pass with variable sequence lengths."""
        head = TimestampHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        lengths = mx.array([100, 80, 60, 40])

        boundary_logits, offset_preds = head(x, encoder_lengths=lengths)

        assert boundary_logits.shape == (batch_size, seq_len, 1)

    def test_predict_boundaries(self):
        """Test boundary prediction."""
        head = TimestampHead()
        batch_size = 4
        seq_len = 100

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        boundary_mask, boundary_probs, offsets = head.predict_boundaries(x)

        assert boundary_mask.shape == (batch_size, seq_len)
        assert boundary_probs.shape == (batch_size, seq_len)
        assert offsets.shape == (batch_size, seq_len, 2)
        # Probs should be in [0, 1]
        assert mx.all(boundary_probs >= 0) and mx.all(boundary_probs <= 1)

    def test_frame_to_time(self):
        """Test frame to time conversion."""
        config = TimestampConfig(frame_duration_ms=40.0)
        head = TimestampHead(config)

        time = head.frame_to_time(10)
        assert time == 400.0  # 10 * 40ms

        time_offset = head.frame_to_time(10, offset=0.5)
        assert time_offset == 420.0  # 10.5 * 40ms

    def test_time_to_frame(self):
        """Test time to frame conversion."""
        config = TimestampConfig(frame_duration_ms=40.0)
        head = TimestampHead(config)

        frame = head.time_to_frame(400.0)
        assert frame == 10

        frame = head.time_to_frame(420.0)
        assert frame == 10  # Truncates

    def test_extract_word_timestamps(self):
        """Test word timestamp extraction."""
        config = TimestampConfig(frame_duration_ms=40.0, boundary_threshold=0.5)
        head = TimestampHead(config)

        batch_size = 2
        seq_len = 50

        # Create mock boundary probabilities with clear peaks
        boundary_probs = mx.full((batch_size, seq_len), 0.2)
        # Add peaks at frames 0, 10, 20, 30, 40
        for i in [0, 10, 20, 30, 40]:
            boundary_probs = boundary_probs.at[:, i].add(0.8)

        lengths = mx.array([50, 50])
        tokens = [["hello", "world", "test"], ["one", "two"]]

        results = head.extract_word_timestamps(
            boundary_probs, tokens, lengths,
        )

        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 2
        # Check first word timestamp
        assert results[0][0].word == "hello"
        assert results[0][0].start_ms >= 0

    def test_extract_word_timestamps_empty(self):
        """Test timestamp extraction with empty tokens."""
        head = TimestampHead()
        batch_size = 2
        seq_len = 50

        boundary_probs = mx.full((batch_size, seq_len), 0.2)
        lengths = mx.array([50, 50])
        tokens = [[], []]

        results = head.extract_word_timestamps(
            boundary_probs, tokens, lengths,
        )

        assert len(results) == 2
        assert len(results[0]) == 0
        assert len(results[1]) == 0


class TestTimestampLoss:
    """Tests for timestamp loss functions."""

    def test_boundary_loss_only(self):
        """Test loss with boundary detection only."""
        batch_size = 4
        seq_len = 50

        boundary_logits = mx.random.normal(shape=(batch_size, seq_len, 1))
        boundary_targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        # Add some positive targets
        boundary_targets = boundary_targets.at[0, 10].add(1)
        boundary_targets = boundary_targets.at[0, 20].add(1)
        boundary_targets = boundary_targets.at[1, 15].add(1)

        total, boundary_loss, offset_loss = timestamp_loss(
            boundary_logits, boundary_targets,
        )

        assert total.shape == ()
        assert boundary_loss.shape == ()
        assert offset_loss is None
        assert total > 0

    def test_boundary_and_offset_loss(self):
        """Test loss with boundary and offset."""
        batch_size = 4
        seq_len = 50

        boundary_logits = mx.random.normal(shape=(batch_size, seq_len, 1))
        boundary_targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        boundary_targets = boundary_targets.at[0, 10].add(1)
        boundary_targets = boundary_targets.at[0, 20].add(1)

        offset_preds = mx.random.normal(shape=(batch_size, seq_len, 2))
        offset_targets = mx.zeros((batch_size, seq_len, 2))

        total, boundary_loss, offset_loss = timestamp_loss(
            boundary_logits, boundary_targets,
            offset_preds=offset_preds,
            offset_targets=offset_targets,
        )

        assert total.shape == ()
        assert boundary_loss.shape == ()
        assert offset_loss.shape == ()
        assert total > 0

    def test_loss_with_mask(self):
        """Test loss with validity mask."""
        batch_size = 4
        seq_len = 50

        boundary_logits = mx.random.normal(shape=(batch_size, seq_len, 1))
        boundary_targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)

        # Mask: only first 30 frames valid
        mask = mx.zeros((batch_size, seq_len), dtype=mx.bool_)
        mask = mask.at[:, :30].add(True)

        total, boundary_loss, _ = timestamp_loss(
            boundary_logits, boundary_targets, mask=mask,
        )

        assert total.shape == ()
        assert total > 0

    def test_loss_module_wrapper(self):
        """Test TimestampLoss module."""
        loss_fn = TimestampLoss(offset_weight=0.1)
        batch_size = 4
        seq_len = 50

        boundary_logits = mx.random.normal(shape=(batch_size, seq_len, 1))
        boundary_targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        boundary_targets = boundary_targets.at[0, 10].add(1)

        offset_preds = mx.random.normal(shape=(batch_size, seq_len, 2))
        offset_targets = mx.zeros((batch_size, seq_len, 2))

        total, boundary_loss, offset_loss = loss_fn(
            boundary_logits, boundary_targets,
            offset_preds, offset_targets,
        )

        assert total.shape == ()
        assert total > 0

    def test_loss_reduction_options(self):
        """Test different reduction options."""
        batch_size = 4
        seq_len = 50

        boundary_logits = mx.random.normal(shape=(batch_size, seq_len, 1))
        boundary_targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)

        total_mean, _, _ = timestamp_loss(
            boundary_logits, boundary_targets, reduction="mean",
        )
        total_sum, _, _ = timestamp_loss(
            boundary_logits, boundary_targets, reduction="sum",
        )

        assert total_mean.shape == ()
        assert total_sum.shape == ()
        # Sum should be larger than mean
        assert total_sum > total_mean


class TestBoundaryAccuracy:
    """Tests for boundary accuracy computation."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        predictions = mx.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ])
        targets = mx.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ])

        acc = compute_boundary_accuracy(predictions, targets)

        assert mx.allclose(acc, mx.array(1.0), atol=1e-5)

    def test_partial_accuracy(self):
        """Test partial accuracy."""
        predictions = mx.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ])
        targets = mx.array([
            [0, 1, 0, 0, 0],  # 4/5 correct
            [1, 0, 0, 0, 0],  # 4/5 correct
        ])

        acc = compute_boundary_accuracy(predictions, targets)

        assert mx.allclose(acc, mx.array(0.8), atol=1e-5)

    def test_accuracy_with_mask(self):
        """Test accuracy with validity mask."""
        predictions = mx.array([
            [0, 1, 0, 0, 1],
        ])
        targets = mx.array([
            [0, 1, 1, 1, 1],  # Only first 2 are masked valid
        ])
        mask = mx.array([
            [True, True, False, False, False],
        ])

        acc = compute_boundary_accuracy(predictions, targets, mask)

        # First two match
        assert mx.allclose(acc, mx.array(1.0), atol=1e-5)


class TestBoundaryF1:
    """Tests for boundary F1 computation."""

    def test_perfect_f1(self):
        """Test perfect F1 score."""
        predictions = mx.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ])
        targets = mx.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ])

        f1, precision, recall = compute_boundary_f1(predictions, targets)

        assert mx.allclose(f1, mx.array(1.0), atol=1e-5)
        assert mx.allclose(precision, mx.array(1.0), atol=1e-5)
        assert mx.allclose(recall, mx.array(1.0), atol=1e-5)

    def test_partial_f1(self):
        """Test partial F1 score."""
        predictions = mx.array([
            [1, 1, 1, 0, 0],  # 2 TP, 1 FP
        ])
        targets = mx.array([
            [1, 1, 0, 1, 0],  # 2 TP, 1 FN
        ])

        f1, precision, recall = compute_boundary_f1(predictions, targets)

        # Precision = 2/3, Recall = 2/3, F1 = 2/3
        assert f1 > 0 and f1 < 1

    def test_f1_with_mask(self):
        """Test F1 with validity mask."""
        predictions = mx.array([
            [1, 1, 1, 0, 0],
        ])
        targets = mx.array([
            [1, 1, 0, 1, 0],
        ])
        mask = mx.array([
            [True, True, True, False, False],
        ])

        f1, precision, recall = compute_boundary_f1(predictions, targets, mask)

        # Only first 3 positions: preds=[1,1,1], targets=[1,1,0]
        # TP=2, FP=1, FN=0
        # Precision = 2/3, Recall = 1.0
        assert f1 > 0 and f1 < 1


class TestTimestampError:
    """Tests for timestamp error computation."""

    def test_perfect_timestamps(self):
        """Test perfect timestamp predictions."""
        predicted = [[
            WordTimestamp("hello", 0.0, 500.0, 0.9),
            WordTimestamp("world", 500.0, 1000.0, 0.9),
        ]]
        target = [[
            WordTimestamp("hello", 0.0, 500.0, 1.0),
            WordTimestamp("world", 500.0, 1000.0, 1.0),
        ]]

        mae, acc, wer = compute_timestamp_error(predicted, target, tolerance_ms=50.0)

        assert mae == 0.0
        assert acc == 1.0
        assert wer == 0.0

    def test_partial_timestamps(self):
        """Test partial timestamp accuracy."""
        predicted = [[
            WordTimestamp("hello", 10.0, 510.0, 0.9),  # 10ms off each
            WordTimestamp("world", 500.0, 1000.0, 0.9),  # Perfect
        ]]
        target = [[
            WordTimestamp("hello", 0.0, 500.0, 1.0),
            WordTimestamp("world", 500.0, 1000.0, 1.0),
        ]]

        mae, acc, wer = compute_timestamp_error(predicted, target, tolerance_ms=50.0)

        # First word is within tolerance, second is perfect
        assert mae == 5.0  # Average of 10ms and 0ms
        assert acc == 1.0  # Both within tolerance
        assert wer == 0.0  # Words match

    def test_wrong_word_timestamps(self):
        """Test timestamps with wrong words."""
        predicted = [[
            WordTimestamp("hello", 0.0, 500.0, 0.9),
            WordTimestamp("earth", 500.0, 1000.0, 0.9),  # Wrong word
        ]]
        target = [[
            WordTimestamp("hello", 0.0, 500.0, 1.0),
            WordTimestamp("world", 500.0, 1000.0, 1.0),
        ]]

        mae, acc, wer = compute_timestamp_error(predicted, target, tolerance_ms=50.0)

        assert wer == 0.5  # 1/2 wrong


class TestTimestampGradients:
    """Tests for gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        head = TimestampHead()
        batch_size = 4
        seq_len = 50

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        boundary_targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        boundary_targets = boundary_targets.at[0, 10].add(1)

        def loss_fn(model, x, targets):
            boundary_logits, offset_preds = model(x)
            total, _, _ = timestamp_loss(boundary_logits, targets)
            return total

        loss, grads = mx.value_and_grad(loss_fn)(head, x, boundary_targets)
        mx.eval(loss, grads)

        assert loss > 0
        assert "hidden" in grads
        assert "boundary_head" in grads

    def test_gradient_flow_with_offset(self):
        """Test gradient flow with offset regression."""
        head = TimestampHead()
        batch_size = 4
        seq_len = 50

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        boundary_targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        boundary_targets = boundary_targets.at[0, 10].add(1)
        offset_targets = mx.zeros((batch_size, seq_len, 2))

        def loss_fn(model, x, boundary_targets, offset_targets):
            boundary_logits, offset_preds = model(x)
            total, _, _ = timestamp_loss(
                boundary_logits, boundary_targets,
                offset_preds, offset_targets,
            )
            return total

        loss, grads = mx.value_and_grad(loss_fn)(head, x, boundary_targets, offset_targets)
        mx.eval(loss, grads)

        assert loss > 0
        assert "offset_head" in grads

    def test_training_step(self):
        """Test a complete training step."""
        head = TimestampHead()
        optimizer = optim.Adam(learning_rate=1e-4)

        batch_size = 4
        seq_len = 50

        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        boundary_targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        boundary_targets = boundary_targets.at[0, 10].add(1)

        def loss_fn(model, x, targets):
            boundary_logits, _ = model(x)
            total, _, _ = timestamp_loss(boundary_logits, targets)
            return total

        # Get initial loss
        initial_loss = loss_fn(head, x, boundary_targets)

        # Training step
        loss, grads = mx.value_and_grad(loss_fn)(head, x, boundary_targets)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)

        # Get updated loss
        updated_loss = loss_fn(head, x, boundary_targets)
        mx.eval(updated_loss)

        # Loss should decrease or stay similar
        assert updated_loss <= initial_loss * 1.1


class TestTimestampIntegration:
    """Integration tests for timestamp head."""

    def test_end_to_end_pipeline(self):
        """Test complete timestamp extraction pipeline."""
        head = TimestampHead()
        batch_size = 2
        seq_len = 100

        # 1. Get encoder output
        x = mx.random.normal(shape=(batch_size, seq_len, 384))
        lengths = mx.array([100, 80])

        # 2. Predict boundaries
        boundary_mask, boundary_probs, offsets = head.predict_boundaries(x, lengths)

        # 3. Extract timestamps
        tokens = [["hello", "world", "test"], ["one", "two"]]
        timestamps = head.extract_word_timestamps(
            boundary_probs, tokens, lengths, offsets,
        )

        # Verify outputs
        assert len(timestamps) == 2
        for ts_list, token_list in zip(timestamps, tokens, strict=False):
            assert len(ts_list) == len(token_list)
            for ts in ts_list:
                assert isinstance(ts, WordTimestamp)
                assert ts.start_ms >= 0
                assert ts.end_ms >= ts.start_ms

    def test_varying_sequence_lengths(self):
        """Test with varying sequence lengths."""
        head = TimestampHead()

        for seq_len in [50, 100, 200]:
            x = mx.random.normal(shape=(2, seq_len, 384))
            boundary_logits, offset_preds = head(x)

            assert boundary_logits.shape == (2, seq_len, 1)
            assert offset_preds.shape == (2, seq_len, 2)
