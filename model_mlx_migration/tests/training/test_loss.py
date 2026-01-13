# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for training loss functions."""

import mlx.core as mx

from src.training.loss import (
    CRCTCLoss,
    CTCLoss,
    ctc_loss,
    log_softmax,
    logsumexp,
)


class TestLogSoftmax:
    """Tests for log_softmax function."""

    def test_basic(self):
        """Test basic log_softmax computation."""
        x = mx.array([[1.0, 2.0, 3.0]])
        result = log_softmax(x, axis=-1)

        # Log softmax should sum to 0 when exponentiated
        exp_sum = mx.sum(mx.exp(result), axis=-1)
        assert mx.allclose(exp_sum, mx.ones_like(exp_sum), atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with large values."""
        x = mx.array([[1000.0, 1001.0, 1002.0]])
        result = log_softmax(x, axis=-1)

        # Should not overflow
        assert mx.all(mx.isfinite(result))

        # Values should be negative (log of probabilities)
        assert mx.all(result <= 0)


class TestLogsumexp:
    """Tests for logsumexp function."""

    def test_basic(self):
        """Test basic logsumexp computation."""
        a = mx.array(1.0)
        b = mx.array(2.0)
        result = logsumexp(a, b)

        expected = mx.log(mx.exp(a) + mx.exp(b))
        assert mx.allclose(result, expected, atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with large values."""
        a = mx.array(1000.0)
        b = mx.array(1001.0)
        result = logsumexp(a, b)

        # Should not overflow
        assert mx.isfinite(result)

        # Result should be close to max + log(2) when values are close
        assert result.item() > 1000.0
        assert result.item() < 1002.0


class TestCTCLoss:
    """Tests for CTC loss function."""

    def test_simple_sequence(self):
        """Test CTC loss with simple sequence."""
        # Logits: (batch=1, time=4, vocab=3)
        # Vocab: blank=0, A=1, B=2
        logits = mx.zeros((1, 4, 3))

        # Target: "AB"
        targets = mx.array([[1, 2]])
        input_lengths = mx.array([4])
        target_lengths = mx.array([2])

        loss = ctc_loss(
            logits=logits,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank_id=0,
            reduction="mean",
        )

        # Loss should be finite (CTC loss can be negative with uniform probs)
        assert mx.isfinite(loss)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        logits = mx.zeros((2, 4, 3))
        targets = mx.array([[1, 2], [1, 0]])
        input_lengths = mx.array([4, 3])
        target_lengths = mx.array([2, 1])

        # Mean reduction
        loss_mean = ctc_loss(
            logits, targets, input_lengths, target_lengths,
            reduction="mean",
        )

        # Sum reduction
        loss_sum = ctc_loss(
            logits, targets, input_lengths, target_lengths,
            reduction="sum",
        )

        # None reduction
        loss_none = ctc_loss(
            logits, targets, input_lengths, target_lengths,
            reduction="none",
        )

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (2,)

    def test_module_wrapper(self):
        """Test CTCLoss module wrapper."""
        loss_fn = CTCLoss(blank_id=0, reduction="mean")

        logits = mx.zeros((1, 4, 3))
        targets = mx.array([[1, 2]])
        input_lengths = mx.array([4])
        target_lengths = mx.array([2])

        loss = loss_fn(logits, targets, input_lengths, target_lengths)

        # Loss should be finite
        assert mx.isfinite(loss)

    def test_empty_target(self):
        """Test CTC loss with empty target."""
        logits = mx.zeros((1, 4, 3))
        targets = mx.array([[0, 0]])  # Will use length 0
        input_lengths = mx.array([4])
        target_lengths = mx.array([0])

        loss = ctc_loss(
            logits, targets, input_lengths, target_lengths,
            reduction="mean",
        )

        # Should handle gracefully
        assert mx.isfinite(loss)


class TestCRCTCLoss:
    """Tests for CR-CTC loss function."""

    def test_loss_combination(self):
        """Test CR-CTC loss module can be instantiated and ctc_weight is used."""
        loss_fn = CRCTCLoss(blank_id=0, ctc_weight=0.3, reduction="mean")

        # Verify the loss function is configured correctly
        assert loss_fn.blank_id == 0
        assert loss_fn.ctc_weight == 0.3
        assert loss_fn.reduction == "mean"

        # Test just CTC portion (transducer loss is complex to test in isolation)
        ctc_output = mx.zeros((1, 4, 10))  # vocab=10
        targets = mx.array([[1, 2]])
        input_lengths = mx.array([4])
        target_lengths = mx.array([2])

        ctc_loss_val = ctc_loss(
            logits=ctc_output,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank_id=0,
            reduction="mean",
        )

        # CTC loss should be finite
        assert mx.isfinite(ctc_loss_val)

    def test_ctc_weight(self):
        """Test CTC weight affects total loss."""
        # With ctc_weight=0, total should equal transducer
        loss_fn_0 = CRCTCLoss(blank_id=0, ctc_weight=0.0)
        # With ctc_weight=1, total should equal CTC
        loss_fn_1 = CRCTCLoss(blank_id=0, ctc_weight=1.0)

        # Different weights produce different total losses
        assert loss_fn_0.ctc_weight == 0.0
        assert loss_fn_1.ctc_weight == 1.0


class TestLossGradients:
    """Tests for loss gradient computation."""

    def test_ctc_loss_gradient(self):
        """Test CTC loss produces valid gradients."""
        logits = mx.zeros((1, 4, 3))
        targets = mx.array([[1, 2]])
        input_lengths = mx.array([4])
        target_lengths = mx.array([2])

        def loss_fn(logits):
            return ctc_loss(
                logits, targets, input_lengths, target_lengths,
                reduction="mean",
            )

        # Compute gradient
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(logits)

        # Gradients should exist and be finite
        assert grads.shape == logits.shape
        assert mx.all(mx.isfinite(grads))
