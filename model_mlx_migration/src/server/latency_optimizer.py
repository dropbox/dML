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

"""
Latency optimization for SOTA++ Voice Server (Phase 9.7).

Provides:
- End-to-end latency profiling
- Pipeline stage timing
- Batch processing optimization
- Async execution helpers
"""

import asyncio
import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage."""
    name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    recent_times: list[float] = field(default_factory=list)

    @property
    def avg_time_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_time_ms / self.total_calls

    @property
    def p50_time_ms(self) -> float:
        if not self.recent_times:
            return 0.0
        return statistics.median(self.recent_times)

    @property
    def p95_time_ms(self) -> float:
        if len(self.recent_times) < 2:
            return self.max_time_ms
        sorted_times = sorted(self.recent_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_time_ms(self) -> float:
        if len(self.recent_times) < 2:
            return self.max_time_ms
        sorted_times = sorted(self.recent_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def record(self, time_ms: float, window_size: int = 1000):
        """Record a timing measurement."""
        self.total_calls += 1
        self.total_time_ms += time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
        self.recent_times.append(time_ms)
        # Keep window bounded
        if len(self.recent_times) > window_size:
            self.recent_times = self.recent_times[-window_size:]


@dataclass
class LatencyConfig:
    """Latency optimizer configuration."""
    # Target latencies
    streaming_target_ms: float = 100.0  # Target for streaming mode
    high_accuracy_target_ms: float = 3000.0  # Target for ROVER mode

    # Profiling
    enable_profiling: bool = True
    profile_window_size: int = 1000
    log_slow_requests: bool = True
    slow_request_threshold_ms: float = 500.0

    # Batching
    enable_batching: bool = True
    batch_timeout_ms: float = 50.0
    max_batch_size: int = 8

    # Async
    max_concurrent_requests: int = 10
    request_timeout_ms: float = 5000.0


@dataclass
class RequestTrace:
    """Trace of a single request through the pipeline."""
    request_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        if self.end_time == 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class LatencyProfiler:
    """
    Profiles pipeline latency at each stage.

    Use with context manager or decorators to instrument code.
    """

    def __init__(self, config: LatencyConfig | None = None):
        self.config = config or LatencyConfig()
        self._stages: dict[str, StageMetrics] = {}
        self._current_trace: RequestTrace | None = None
        self._traces: deque = deque(maxlen=100)  # Keep last 100 traces
        self._lock = threading.Lock()

    @contextmanager
    def trace(self, request_id: str):
        """Context manager for tracing a complete request."""
        trace = RequestTrace(request_id=request_id, start_time=time.time())
        self._current_trace = trace

        try:
            yield trace
        finally:
            trace.end_time = time.time()
            self._current_trace = None

            with self._lock:
                self._traces.append(trace)

            if self.config.log_slow_requests:
                if trace.total_time_ms > self.config.slow_request_threshold_ms:
                    self._log_slow_request(trace)

    @contextmanager
    def stage(self, name: str):
        """Context manager for timing a pipeline stage."""
        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self._record_stage(name, elapsed_ms)

    def time_stage(self, name: str):
        """Decorator for timing a function as a pipeline stage."""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    with self.stage(name):
                        return await func(*args, **kwargs)
                return async_wrapper
            def sync_wrapper(*args, **kwargs):
                with self.stage(name):
                    return func(*args, **kwargs)
            return sync_wrapper
        return decorator

    def _record_stage(self, name: str, elapsed_ms: float):
        """Record a stage timing."""
        with self._lock:
            if name not in self._stages:
                self._stages[name] = StageMetrics(name=name)
            self._stages[name].record(elapsed_ms, self.config.profile_window_size)

        # Also record in current trace if active
        if self._current_trace is not None:
            self._current_trace.stages[name] = elapsed_ms

    def _log_slow_request(self, trace: RequestTrace):
        """Log details of a slow request."""
        stages_str = ", ".join(
            f"{name}={time_ms:.1f}ms"
            for name, time_ms in sorted(trace.stages.items(), key=lambda x: -x[1])
        )
        logger.warning(
            f"Slow request {trace.request_id}: {trace.total_time_ms:.1f}ms "
            f"[{stages_str}]",
        )

    def get_stage_metrics(self, name: str) -> StageMetrics | None:
        """Get metrics for a specific stage."""
        with self._lock:
            return self._stages.get(name)

    def get_all_metrics(self) -> dict[str, StageMetrics]:
        """Get metrics for all stages."""
        with self._lock:
            return dict(self._stages)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all stage metrics."""
        with self._lock:
            summary = {}
            for name, metrics in self._stages.items():
                summary[name] = {
                    "total_calls": metrics.total_calls,
                    "avg_ms": round(metrics.avg_time_ms, 2),
                    "min_ms": round(metrics.min_time_ms, 2),
                    "max_ms": round(metrics.max_time_ms, 2),
                    "p50_ms": round(metrics.p50_time_ms, 2),
                    "p95_ms": round(metrics.p95_time_ms, 2),
                    "p99_ms": round(metrics.p99_time_ms, 2),
                }
            return summary

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._stages.clear()
            self._traces.clear()


class BatchAccumulator:
    """
    Accumulates items for batch processing.

    Batches items by waiting up to a timeout or reaching max batch size.
    """

    def __init__(
        self,
        max_size: int = 8,
        timeout_ms: float = 50.0,
        process_fn: Callable[[list[Any]], list[Any]] | None = None,
    ):
        self._max_size = max_size
        self._timeout_ms = timeout_ms
        self._process_fn = process_fn
        self._items: list[tuple[Any, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def add(self, item: Any) -> Any:
        """
        Add an item to the batch.

        Returns the result for this item after batch processing.
        """
        future = asyncio.get_event_loop().create_future()

        async with self._lock:
            self._items.append((item, future))

            # Start flush timer if first item
            if len(self._items) == 1:
                self._flush_task = asyncio.create_task(self._flush_after_timeout())

            # Flush immediately if batch is full
            if len(self._items) >= self._max_size:
                await self._flush()

        return await future

    async def _flush_after_timeout(self):
        """Flush batch after timeout."""
        await asyncio.sleep(self._timeout_ms / 1000)
        async with self._lock:
            await self._flush()

    async def _flush(self):
        """Process and flush the current batch."""
        if not self._items:
            return

        # Cancel any pending flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        self._flush_task = None

        # Get items and clear
        items = self._items
        self._items = []

        # Process batch
        inputs = [item for item, _ in items]
        futures = [future for _, future in items]

        try:
            if self._process_fn:
                results = self._process_fn(inputs)
            else:
                results = inputs  # No-op if no processor

            # Distribute results
            for future, result in zip(futures, results, strict=False):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            # Set exception on all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)


class AsyncThrottler:
    """
    Throttles concurrent async operations.

    Ensures we don't exceed a maximum number of concurrent operations.
    """

    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending = 0
        self._lock = asyncio.Lock()

    @property
    def pending(self) -> int:
        return self._pending

    async def acquire(self):
        """Acquire a slot for execution."""
        await self._semaphore.acquire()
        async with self._lock:
            self._pending += 1

    def release(self):
        """Release an execution slot."""
        self._semaphore.release()
        asyncio.create_task(self._decrement_pending())

    async def _decrement_pending(self):
        async with self._lock:
            self._pending -= 1

    @contextmanager
    def throttle_sync(self):
        """Sync context manager for throttling."""
        # Note: This doesn't actually throttle sync code
        # It's here for API consistency
        yield

    async def throttle(self):
        """Async context manager for throttling."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()


class PipelineOptimizer:
    """
    Optimizes the ASR pipeline for low latency.

    Provides:
    - Profiling instrumentation
    - Batch processing
    - Concurrency control
    """

    def __init__(self, config: LatencyConfig | None = None):
        self.config = config or LatencyConfig()
        self.profiler = LatencyProfiler(config)
        self.throttler = AsyncThrottler(self.config.max_concurrent_requests)

        # Stage-specific batch accumulators
        self._batch_accumulators: dict[str, BatchAccumulator] = {}

    def create_batch_accumulator(
        self,
        name: str,
        process_fn: Callable[[list[Any]], list[Any]],
        max_size: int | None = None,
        timeout_ms: float | None = None,
    ) -> BatchAccumulator:
        """Create a batch accumulator for a pipeline stage."""
        accumulator = BatchAccumulator(
            max_size=max_size or self.config.max_batch_size,
            timeout_ms=timeout_ms or self.config.batch_timeout_ms,
            process_fn=process_fn,
        )
        self._batch_accumulators[name] = accumulator
        return accumulator

    def get_batch_accumulator(self, name: str) -> BatchAccumulator | None:
        """Get an existing batch accumulator."""
        return self._batch_accumulators.get(name)

    async def profile_request(
        self,
        request_id: str,
        stages: list[tuple[str, Callable[[], Any]]],
    ) -> list[Any]:
        """
        Execute stages with profiling.

        Args:
            request_id: Unique request identifier
            stages: List of (stage_name, callable) tuples

        Returns:
            List of results from each stage
        """
        results = []

        with self.profiler.trace(request_id):
            for stage_name, stage_fn in stages:
                with self.profiler.stage(stage_name):
                    if asyncio.iscoroutinefunction(stage_fn):
                        result = await stage_fn()
                    else:
                        result = stage_fn()
                    results.append(result)

        return results

    def get_optimization_recommendations(self) -> list[str]:
        """Get recommendations based on profiling data."""
        recommendations = []
        metrics = self.profiler.get_all_metrics()

        for name, stage_metrics in metrics.items():
            # Check for high latency stages
            if stage_metrics.p95_time_ms > self.config.streaming_target_ms:
                recommendations.append(
                    f"Stage '{name}' p95 latency ({stage_metrics.p95_time_ms:.1f}ms) "
                    f"exceeds streaming target ({self.config.streaming_target_ms}ms)",
                )

            # Check for high variance
            if stage_metrics.max_time_ms > stage_metrics.avg_time_ms * 10:
                recommendations.append(
                    f"Stage '{name}' has high variance "
                    f"(max={stage_metrics.max_time_ms:.1f}ms, avg={stage_metrics.avg_time_ms:.1f}ms)",
                )

        return recommendations

    def get_summary(self) -> dict[str, Any]:
        """Get complete optimization summary."""
        return {
            "profiler": self.profiler.get_summary(),
            "recommendations": self.get_optimization_recommendations(),
            "config": {
                "streaming_target_ms": self.config.streaming_target_ms,
                "high_accuracy_target_ms": self.config.high_accuracy_target_ms,
                "max_concurrent_requests": self.config.max_concurrent_requests,
            },
        }


# Convenience functions for common profiling patterns

def time_function(name: str, profiler: LatencyProfiler | None = None):
    """Decorator to time a function."""
    def decorator(func):
        nonlocal profiler
        if profiler is None:
            profiler = LatencyProfiler()

        return profiler.time_stage(name)(func)
    return decorator


async def run_with_timeout(
    coro,
    timeout_ms: float,
    default: Any = None,
) -> Any:
    """Run coroutine with timeout, returning default on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_ms / 1000)
    except TimeoutError:
        logger.warning(f"Operation timed out after {timeout_ms}ms")
        return default


async def run_concurrent(
    tasks: list[Callable[[], Any]],
    max_concurrent: int = 10,
) -> list[Any]:
    """Run tasks concurrently with a limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def run_with_semaphore(task):
        async with semaphore:
            if asyncio.iscoroutinefunction(task):
                return await task()
            return task()

    coros = [run_with_semaphore(task) for task in tasks]
    results = await asyncio.gather(*coros, return_exceptions=True)

    return results
