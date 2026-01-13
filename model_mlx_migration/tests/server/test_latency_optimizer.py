# Copyright 2024-2026 Andrew Yates
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

"""Tests for Phase 9.7 Latency Optimizer."""

import asyncio
import time

import pytest

from src.server.latency_optimizer import (
    AsyncThrottler,
    BatchAccumulator,
    LatencyConfig,
    LatencyProfiler,
    PipelineOptimizer,
    RequestTrace,
    StageMetrics,
    run_concurrent,
    run_with_timeout,
)


class TestLatencyConfig:
    """Tests for LatencyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LatencyConfig()
        assert config.streaming_target_ms == 100.0
        assert config.high_accuracy_target_ms == 3000.0
        assert config.enable_profiling is True
        assert config.profile_window_size == 1000
        assert config.log_slow_requests is True
        assert config.slow_request_threshold_ms == 500.0
        assert config.enable_batching is True
        assert config.batch_timeout_ms == 50.0
        assert config.max_batch_size == 8
        assert config.max_concurrent_requests == 10
        assert config.request_timeout_ms == 5000.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LatencyConfig(
            streaming_target_ms=50.0,
            max_batch_size=16,
        )
        assert config.streaming_target_ms == 50.0
        assert config.max_batch_size == 16


class TestStageMetrics:
    """Tests for StageMetrics dataclass."""

    def test_default_values(self):
        """Test default metrics values."""
        metrics = StageMetrics(name="test")
        assert metrics.name == "test"
        assert metrics.total_calls == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.min_time_ms == float('inf')
        assert metrics.max_time_ms == 0.0
        assert metrics.recent_times == []

    def test_avg_time_empty(self):
        """Test average with no data."""
        metrics = StageMetrics(name="test")
        assert metrics.avg_time_ms == 0.0

    def test_avg_time(self):
        """Test average calculation."""
        metrics = StageMetrics(name="test")
        metrics.record(10.0)
        metrics.record(20.0)
        metrics.record(30.0)
        assert metrics.avg_time_ms == 20.0

    def test_percentiles(self):
        """Test percentile calculations."""
        metrics = StageMetrics(name="test")
        for i in range(100):
            metrics.record(float(i))

        assert metrics.p50_time_ms == pytest.approx(49.5, rel=0.1)
        assert metrics.p95_time_ms == pytest.approx(95, rel=0.1)
        assert metrics.p99_time_ms == pytest.approx(99, rel=0.1)

    def test_record(self):
        """Test recording measurements."""
        metrics = StageMetrics(name="test")
        metrics.record(50.0)
        metrics.record(100.0)

        assert metrics.total_calls == 2
        assert metrics.total_time_ms == 150.0
        assert metrics.min_time_ms == 50.0
        assert metrics.max_time_ms == 100.0

    def test_window_size(self):
        """Test window size limiting."""
        metrics = StageMetrics(name="test")

        # Record more than window size
        for i in range(20):
            metrics.record(float(i), window_size=10)

        assert len(metrics.recent_times) == 10
        assert metrics.recent_times[0] == 10.0


class TestRequestTrace:
    """Tests for RequestTrace dataclass."""

    def test_default_values(self):
        """Test default trace values."""
        trace = RequestTrace(request_id="test-123")
        assert trace.request_id == "test-123"
        assert trace.start_time == 0.0
        assert trace.end_time == 0.0
        assert trace.stages == {}
        assert trace.metadata == {}

    def test_total_time(self):
        """Test total time calculation."""
        trace = RequestTrace(request_id="test")
        trace.start_time = 1000.0
        trace.end_time = 1000.5
        assert trace.total_time_ms == pytest.approx(500.0, rel=0.01)

    def test_total_time_not_ended(self):
        """Test total time when not ended."""
        trace = RequestTrace(request_id="test")
        trace.start_time = 1000.0
        assert trace.total_time_ms == 0.0


class TestLatencyProfiler:
    """Tests for LatencyProfiler."""

    def test_initialization(self):
        """Test profiler initialization."""
        profiler = LatencyProfiler()
        assert profiler.config is not None

    def test_trace_context_manager(self):
        """Test trace context manager."""
        profiler = LatencyProfiler()

        with profiler.trace("req-1") as trace:
            assert trace.request_id == "req-1"
            assert trace.start_time > 0

        assert trace.end_time > trace.start_time

    def test_stage_context_manager(self):
        """Test stage timing context manager."""
        profiler = LatencyProfiler()

        with profiler.stage("preprocessing"):
            time.sleep(0.01)

        metrics = profiler.get_stage_metrics("preprocessing")
        assert metrics is not None
        assert metrics.total_calls == 1
        assert metrics.total_time_ms >= 10

    def test_stage_recorded_in_trace(self):
        """Test stage timing recorded in trace."""
        profiler = LatencyProfiler()

        with profiler.trace("req-1") as trace:
            with profiler.stage("step1"):
                time.sleep(0.005)
            with profiler.stage("step2"):
                time.sleep(0.005)

        assert "step1" in trace.stages
        assert "step2" in trace.stages
        assert trace.stages["step1"] >= 5

    def test_time_stage_decorator(self):
        """Test timing decorator."""
        profiler = LatencyProfiler()

        @profiler.time_stage("my_function")
        def my_function():
            time.sleep(0.01)
            return "result"

        result = my_function()
        assert result == "result"

        metrics = profiler.get_stage_metrics("my_function")
        assert metrics.total_calls == 1
        assert metrics.total_time_ms >= 10

    @pytest.mark.asyncio
    async def test_time_stage_async_decorator(self):
        """Test timing decorator for async functions."""
        profiler = LatencyProfiler()

        @profiler.time_stage("async_function")
        async def async_function():
            await asyncio.sleep(0.01)
            return "async result"

        result = await async_function()
        assert result == "async result"

        metrics = profiler.get_stage_metrics("async_function")
        assert metrics.total_calls == 1
        assert metrics.total_time_ms >= 10

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        profiler = LatencyProfiler()

        with profiler.stage("a"):
            pass
        with profiler.stage("b"):
            pass

        metrics = profiler.get_all_metrics()
        assert "a" in metrics
        assert "b" in metrics

    def test_get_summary(self):
        """Test getting summary."""
        profiler = LatencyProfiler()

        for _ in range(10):
            with profiler.stage("test"):
                time.sleep(0.001)

        summary = profiler.get_summary()
        assert "test" in summary
        assert "total_calls" in summary["test"]
        assert summary["test"]["total_calls"] == 10

    def test_reset(self):
        """Test reset clears all data."""
        profiler = LatencyProfiler()

        with profiler.stage("test"):
            pass

        assert len(profiler.get_all_metrics()) > 0

        profiler.reset()
        assert len(profiler.get_all_metrics()) == 0


class TestBatchAccumulator:
    """Tests for BatchAccumulator."""

    @pytest.mark.asyncio
    async def test_basic_batching(self):
        """Test basic batching."""
        processed = []

        def process_batch(items):
            processed.append(len(items))
            return [x * 2 for x in items]

        accumulator = BatchAccumulator(
            max_size=3,
            timeout_ms=100.0,
            process_fn=process_batch,
        )

        # Add items concurrently
        results = await asyncio.gather(
            accumulator.add(1),
            accumulator.add(2),
            accumulator.add(3),
        )

        assert results == [2, 4, 6]
        assert processed == [3]

    @pytest.mark.asyncio
    async def test_timeout_flush(self):
        """Test batch flushes on timeout."""
        processed = []

        def process_batch(items):
            processed.append(len(items))
            return items

        accumulator = BatchAccumulator(
            max_size=10,
            timeout_ms=50.0,
            process_fn=process_batch,
        )

        # Add single item - should flush on timeout
        result = await accumulator.add(1)
        assert result == 1
        assert 1 in processed

    @pytest.mark.asyncio
    async def test_no_processor(self):
        """Test accumulator without processor."""
        accumulator = BatchAccumulator(max_size=2, timeout_ms=10.0)

        results = await asyncio.gather(
            accumulator.add("a"),
            accumulator.add("b"),
        )

        assert results == ["a", "b"]


class TestAsyncThrottler:
    """Tests for AsyncThrottler."""

    def test_initialization(self):
        """Test throttler initialization."""
        throttler = AsyncThrottler(max_concurrent=5)
        assert throttler.pending == 0

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test acquire and release."""
        throttler = AsyncThrottler(max_concurrent=2)

        await throttler.acquire()
        assert throttler.pending == 1

        throttler.release()
        await asyncio.sleep(0.01)  # Allow decrement task
        assert throttler.pending == 0


class TestPipelineOptimizer:
    """Tests for PipelineOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = PipelineOptimizer()
        assert optimizer.profiler is not None
        assert optimizer.throttler is not None

    def test_create_batch_accumulator(self):
        """Test creating batch accumulator."""
        optimizer = PipelineOptimizer()

        acc = optimizer.create_batch_accumulator(
            name="test_batch",
            process_fn=lambda items: items,
        )

        assert acc is not None
        assert optimizer.get_batch_accumulator("test_batch") is acc

    @pytest.mark.asyncio
    async def test_profile_request(self):
        """Test profiling a request."""
        optimizer = PipelineOptimizer()

        results = await optimizer.profile_request(
            request_id="test-1",
            stages=[
                ("step1", lambda: "result1"),
                ("step2", lambda: "result2"),
            ],
        )

        assert results == ["result1", "result2"]

        summary = optimizer.profiler.get_summary()
        assert "step1" in summary
        assert "step2" in summary

    def test_get_optimization_recommendations(self):
        """Test getting recommendations."""
        optimizer = PipelineOptimizer(LatencyConfig(streaming_target_ms=10.0))

        # Record some slow measurements
        for _ in range(10):
            with optimizer.profiler.stage("slow_stage"):
                time.sleep(0.02)  # 20ms > 10ms target

        recommendations = optimizer.get_optimization_recommendations()
        assert len(recommendations) > 0
        assert any("slow_stage" in r for r in recommendations)

    def test_get_summary(self):
        """Test getting complete summary."""
        optimizer = PipelineOptimizer()

        with optimizer.profiler.stage("test"):
            pass

        summary = optimizer.get_summary()
        assert "profiler" in summary
        assert "recommendations" in summary
        assert "config" in summary


class TestUtilityFunctions:
    """Tests for utility functions."""

    @pytest.mark.asyncio
    async def test_run_with_timeout_success(self):
        """Test successful execution within timeout."""
        async def fast_operation():
            await asyncio.sleep(0.01)
            return "success"

        result = await run_with_timeout(fast_operation(), timeout_ms=1000)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_run_with_timeout_timeout(self):
        """Test timeout returns default."""
        async def slow_operation():
            await asyncio.sleep(10)
            return "never"

        result = await run_with_timeout(
            slow_operation(),
            timeout_ms=10,
            default="timed_out",
        )
        assert result == "timed_out"

    @pytest.mark.asyncio
    async def test_run_concurrent(self):
        """Test running tasks concurrently."""
        async def task1():
            await asyncio.sleep(0.01)
            return 1

        async def task2():
            await asyncio.sleep(0.01)
            return 2

        output = await run_concurrent([task1, task2], max_concurrent=2)
        assert output == [1, 2]

    @pytest.mark.asyncio
    async def test_run_concurrent_with_limit(self):
        """Test concurrent execution with limit."""
        active = [0]
        max_active = [0]

        async def task():
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0.05)
            active[0] -= 1
            return 1

        tasks = [task for _ in range(10)]
        results = await run_concurrent(tasks, max_concurrent=3)

        assert len(results) == 10
        assert max_active[0] <= 3
