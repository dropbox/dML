# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for learning rate schedulers."""


from src.training.config import SchedulerConfig
from src.training.scheduler import (
    NoamScheduler,
    WarmupScheduler,
    get_scheduler,
)


class TestWarmupScheduler:
    """Tests for WarmupScheduler."""

    def test_warmup_phase(self):
        """Test learning rate during warmup."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )

        # At step 0, LR should be small
        lr_0 = scheduler.get_lr(0)
        assert lr_0 < 1e-4

        # At step 50 (midway through warmup), LR should be ~0.5 * base_lr
        lr_50 = scheduler.get_lr(50)
        assert 0.4e-3 < lr_50 < 0.6e-3

        # At warmup end, LR should be close to base_lr
        lr_100 = scheduler.get_lr(100)
        assert 0.9e-3 < lr_100 < 1.1e-3

    def test_cosine_decay(self):
        """Test cosine decay after warmup."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
            scheduler_type="warmup_cosine",
        )

        # After warmup, LR should decay
        lr_100 = scheduler.get_lr(100)
        lr_500 = scheduler.get_lr(500)
        lr_900 = scheduler.get_lr(900)

        # LR should decrease
        assert lr_100 > lr_500 > lr_900

        # Should approach min_lr at end
        assert lr_900 < lr_100 / 2

    def test_linear_decay(self):
        """Test linear decay after warmup."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
            scheduler_type="warmup_linear",
        )

        # Linear decay should be roughly proportional
        lr_100 = scheduler.get_lr(100)
        lr_500 = scheduler.get_lr(500)
        lr_900 = scheduler.get_lr(900)

        # Check roughly linear decay
        assert lr_100 > lr_500 > lr_900

    def test_constant_after_warmup(self):
        """Test constant LR after warmup."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            scheduler_type="warmup_constant",
        )

        # After warmup, LR should stay constant
        lr_100 = scheduler.get_lr(100)
        lr_500 = scheduler.get_lr(500)
        lr_900 = scheduler.get_lr(900)

        assert abs(lr_100 - lr_500) < 1e-10
        assert abs(lr_500 - lr_900) < 1e-10

    def test_cooldown_phase(self):
        """Test cooldown phase at end of training."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
            cooldown_steps=100,
            scheduler_type="warmup_constant",
        )

        # During cooldown, LR should decay to min_lr
        lr_900 = scheduler.get_lr(900)  # Start of cooldown
        lr_950 = scheduler.get_lr(950)  # Mid cooldown
        lr_999 = scheduler.get_lr(999)  # End of cooldown

        assert lr_900 > lr_950 > lr_999
        assert lr_999 > 1e-6  # Not quite at min yet

    def test_step_method(self):
        """Test step() method increments correctly."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=10,
            total_steps=100,
        )

        assert scheduler.current_step == 0

        lr1 = scheduler.step()
        assert scheduler.current_step == 1

        lr2 = scheduler.step()
        assert scheduler.current_step == 2

        # LR should increase during warmup
        assert lr2 > lr1

    def test_state_dict(self):
        """Test state serialization."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )

        # Advance some steps
        for _ in range(50):
            scheduler.step()

        state = scheduler.state_dict()

        assert state["step"] == 50
        assert state["base_lr"] == 1e-3
        assert state["warmup_steps"] == 100

    def test_load_state_dict(self):
        """Test state restoration."""
        scheduler1 = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )

        # Advance and save state
        for _ in range(50):
            scheduler1.step()
        state = scheduler1.state_dict()

        # Create new scheduler and load state
        scheduler2 = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )
        scheduler2.load_state_dict(state)

        assert scheduler2.current_step == 50
        assert scheduler2.current_lr == scheduler1.current_lr


class TestNoamScheduler:
    """Tests for NoamScheduler (Transformer-style)."""

    def test_warmup_increase(self):
        """Test LR increases during warmup."""
        scheduler = NoamScheduler(
            d_model=512,
            warmup_steps=4000,
            scale=1.0,
        )

        # LR should increase during warmup
        lr_100 = scheduler.get_lr(100)
        lr_1000 = scheduler.get_lr(1000)
        lr_4000 = scheduler.get_lr(4000)

        assert lr_100 < lr_1000 < lr_4000

    def test_decay_after_warmup(self):
        """Test LR decays after warmup."""
        scheduler = NoamScheduler(
            d_model=512,
            warmup_steps=4000,
            scale=1.0,
        )

        # LR should decrease after warmup
        lr_4000 = scheduler.get_lr(4000)
        lr_8000 = scheduler.get_lr(8000)
        lr_16000 = scheduler.get_lr(16000)

        assert lr_4000 > lr_8000 > lr_16000

    def test_scale_factor(self):
        """Test scale factor affects LR."""
        scheduler_1 = NoamScheduler(d_model=512, warmup_steps=4000, scale=1.0)
        scheduler_2 = NoamScheduler(d_model=512, warmup_steps=4000, scale=2.0)

        lr_1 = scheduler_1.get_lr(1000)
        lr_2 = scheduler_2.get_lr(1000)

        assert abs(lr_2 - 2 * lr_1) < 1e-10


class TestGetScheduler:
    """Tests for get_scheduler factory function."""

    def test_from_config(self):
        """Test creating scheduler from config."""
        config = SchedulerConfig(
            scheduler_type="warmup_cosine",
            warmup_steps=1000,
            total_steps=10000,
            min_lr=1e-6,
            cooldown_steps=500,
        )

        scheduler = get_scheduler(config, base_lr=1e-3)

        assert isinstance(scheduler, WarmupScheduler)
        assert scheduler.warmup_steps == 1000
        assert scheduler.total_steps == 10000
        assert scheduler.base_lr == 1e-3


class TestSchedulerStability:
    """Tests for scheduler stability features."""

    def test_no_negative_lr(self):
        """Test scheduler never produces negative LR."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
            cooldown_steps=100,
            scheduler_type="warmup_cosine",
        )

        # Check many steps
        for step in range(0, 1100, 10):
            lr = scheduler.get_lr(step)
            assert lr >= 0, f"Negative LR at step {step}: {lr}"
            # Allow small tolerance below min_lr (floating point)
            assert lr >= scheduler.min_lr * 0.99 or step < scheduler.warmup_steps

    def test_smooth_transitions(self):
        """Test LR changes smoothly (no sudden jumps after warmup)."""
        scheduler = WarmupScheduler(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
            scheduler_type="warmup_cosine",
        )

        # Test after warmup period (during warmup, changes are proportionally larger)
        prev_lr = scheduler.get_lr(100)
        for step in range(101, 1000):
            lr = scheduler.get_lr(step)
            # LR should not change by more than 5% per step after warmup
            max_change = max(prev_lr, lr) * 0.05 + 1e-8
            assert abs(lr - prev_lr) < max_change, f"Jump at step {step}"
            prev_lr = lr
