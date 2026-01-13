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
Production hardening for SOTA++ Voice Server (Phase 9.8).

Provides:
- Graceful error handling and degradation
- Health check endpoints
- Metrics export (Prometheus-compatible)
- Structured logging
- Circuit breaker pattern
"""

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ==============================================================================
# Error Handling
# ==============================================================================

class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ServerError:
    """Structured server error."""
    code: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "details": self.details,
            "recoverable": self.recoverable,
        }


# Common error codes
class ErrorCode:
    """Common error codes."""
    # Client errors (4xx)
    INVALID_AUDIO = "invalid_audio"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_FORMAT = "unsupported_format"
    RATE_LIMITED = "rate_limited"
    SESSION_EXPIRED = "session_expired"

    # Server errors (5xx)
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    INTERNAL_ERROR = "internal_error"

    # Degradation errors
    FALLBACK_USED = "fallback_used"
    PARTIAL_RESULT = "partial_result"


class ErrorHandler:
    """
    Centralized error handling with recovery strategies.
    """

    def __init__(self, max_errors: int = 1000):
        self._errors: deque = deque(maxlen=max_errors)
        self._error_counts: dict[str, int] = {}
        self._handlers: dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register_handler(self, code: str, handler: Callable[[ServerError], Any]):
        """Register a handler for specific error code."""
        self._handlers[code] = handler

    def handle(self, error: ServerError) -> Any:
        """Handle an error, logging and invoking registered handler."""
        with self._lock:
            self._errors.append(error)
            self._error_counts[error.code] = self._error_counts.get(error.code, 0) + 1

        # Log at appropriate level
        log_fn = {
            ErrorSeverity.DEBUG: logger.debug,
            ErrorSeverity.INFO: logger.info,
            ErrorSeverity.WARNING: logger.warning,
            ErrorSeverity.ERROR: logger.error,
            ErrorSeverity.CRITICAL: logger.critical,
        }.get(error.severity, logger.error)

        log_fn(f"[{error.code}] {error.message}", extra={"error": error.to_dict()})

        # Invoke handler if registered
        handler = self._handlers.get(error.code)
        if handler:
            try:
                return handler(error)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")

        return None

    def get_recent_errors(self, limit: int = 100) -> list[ServerError]:
        """Get recent errors."""
        with self._lock:
            return list(self._errors)[-limit:]

    def get_error_counts(self) -> dict[str, int]:
        """Get error counts by code."""
        with self._lock:
            return dict(self._error_counts)

    def clear(self):
        """Clear error history."""
        with self._lock:
            self._errors.clear()
            self._error_counts.clear()


# ==============================================================================
# Health Checks
# ==============================================================================

class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    checks: list[HealthCheckResult]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(
                self.timestamp, tz=UTC,
            ).isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }


class HealthChecker:
    """
    System health checker with multiple check components.
    """

    def __init__(self):
        self._checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self._lock = threading.Lock()

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult],
    ):
        """Register a health check."""
        with self._lock:
            self._checks[name] = check_fn

    def check(self) -> SystemHealth:
        """Run all health checks."""
        results = []
        overall_status = HealthStatus.HEALTHY

        with self._lock:
            checks = dict(self._checks)

        for name, check_fn in checks.items():
            start = time.time()
            try:
                result = check_fn()
                result.latency_ms = (time.time() - start) * 1000
                results.append(result)

                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED:
                    if overall_status != HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.DEGRADED

            except Exception as e:
                results.append(HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    latency_ms=(time.time() - start) * 1000,
                ))
                overall_status = HealthStatus.UNHEALTHY

        return SystemHealth(
            status=overall_status,
            checks=results,
        )

    async def check_async(self) -> SystemHealth:
        """Run health checks asynchronously."""
        # For now, run sync checks in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check)


def create_model_health_check(
    model_name: str,
    get_model_fn: Callable[[], Any | None],
) -> Callable[[], HealthCheckResult]:
    """Create a health check for a model."""
    def check() -> HealthCheckResult:
        try:
            model = get_model_fn()
            if model is not None:
                return HealthCheckResult(
                    name=f"model_{model_name}",
                    status=HealthStatus.HEALTHY,
                    message="Model loaded",
                )
            return HealthCheckResult(
                name=f"model_{model_name}",
                status=HealthStatus.DEGRADED,
                message="Model not loaded",
            )
        except Exception as e:
            return HealthCheckResult(
                name=f"model_{model_name}",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    return check


def create_memory_health_check(
    max_memory_mb: int = 4096,
    warning_threshold: float = 0.8,
) -> Callable[[], HealthCheckResult]:
    """Create a health check for memory usage."""
    def check() -> HealthCheckResult:
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            ratio = memory_mb / max_memory_mb

            if ratio > 0.95:
                status = HealthStatus.UNHEALTHY
                message = f"Memory critical: {memory_mb:.0f}MB / {max_memory_mb}MB"
            elif ratio > warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"Memory warning: {memory_mb:.0f}MB / {max_memory_mb}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory OK: {memory_mb:.0f}MB / {max_memory_mb}MB"

            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                details={"used_mb": round(memory_mb, 1), "max_mb": max_memory_mb},
            )
        except ImportError:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.DEGRADED,
                message="psutil not available",
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    return check


# ==============================================================================
# Metrics
# ==============================================================================

class MetricType(str, Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """A single metric."""
    name: str
    type: MetricType
    help: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def to_prometheus(self) -> str:
        """Format as Prometheus text exposition."""
        labels_str = ""
        if self.labels:
            pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(pairs) + "}"
        return f"{self.name}{labels_str} {self.value}"


class MetricsRegistry:
    """
    Prometheus-compatible metrics registry.
    """

    def __init__(self, prefix: str = "voiceserver"):
        self._prefix = prefix
        self._metrics: dict[str, Metric] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _full_name(self, name: str) -> str:
        return f"{self._prefix}_{name}"

    def counter(
        self,
        name: str,
        help: str,
        labels: dict[str, str] | None = None,
    ) -> 'Counter':
        """Create or get a counter metric."""
        full_name = self._full_name(name)
        key = self._metric_key(full_name, labels)

        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Metric(
                    name=full_name,
                    type=MetricType.COUNTER,
                    help=help,
                    labels=labels or {},
                )
            return Counter(self, key)

    def gauge(
        self,
        name: str,
        help: str,
        labels: dict[str, str] | None = None,
    ) -> 'Gauge':
        """Create or get a gauge metric."""
        full_name = self._full_name(name)
        key = self._metric_key(full_name, labels)

        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Metric(
                    name=full_name,
                    type=MetricType.GAUGE,
                    help=help,
                    labels=labels or {},
                )
            return Gauge(self, key)

    def histogram(
        self,
        name: str,
        help: str,
        buckets: list[float] | None = None,
        labels: dict[str, str] | None = None,
    ) -> 'Histogram':
        """Create or get a histogram metric."""
        full_name = self._full_name(name)
        key = self._metric_key(full_name, labels)

        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            # Create base metric for sum
            if key not in self._metrics:
                self._metrics[key] = Metric(
                    name=full_name,
                    type=MetricType.HISTOGRAM,
                    help=help,
                    labels=labels or {},
                )
        return Histogram(
            self, key,
            buckets=buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

    def _metric_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create unique key for metric with labels."""
        if not labels:
            return name
        sorted_labels = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{name}{{{label_str}}}"

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []

        with self._lock:
            # Group by metric name for help text
            seen_names: set[str] = set()

            for key, metric in self._metrics.items():
                if metric.name not in seen_names:
                    lines.append(f"# HELP {metric.name} {metric.help}")
                    lines.append(f"# TYPE {metric.name} {metric.type.value}")
                    seen_names.add(metric.name)
                lines.append(metric.to_prometheus())

        return "\n".join(lines) + "\n"

    def export_json(self) -> dict[str, Any]:
        """Export all metrics as JSON."""
        with self._lock:
            return {
                "metrics": [
                    {
                        "name": m.name,
                        "type": m.type.value,
                        "value": m.value,
                        "labels": m.labels,
                    }
                    for m in self._metrics.values()
                ],
            }


class Counter:
    """Counter metric (always increases)."""

    def __init__(self, registry: MetricsRegistry, key: str):
        self._registry = registry
        self._key = key

    def inc(self, value: float = 1.0):
        """Increment counter."""
        with self._registry._lock:
            metric = self._registry._metrics.get(self._key)
            if metric:
                metric.value += value


class Gauge:
    """Gauge metric (can go up or down)."""

    def __init__(self, registry: MetricsRegistry, key: str):
        self._registry = registry
        self._key = key

    def set(self, value: float):
        """Set gauge value."""
        with self._registry._lock:
            metric = self._registry._metrics.get(self._key)
            if metric:
                metric.value = value

    def inc(self, value: float = 1.0):
        """Increment gauge."""
        with self._registry._lock:
            metric = self._registry._metrics.get(self._key)
            if metric:
                metric.value += value

    def dec(self, value: float = 1.0):
        """Decrement gauge."""
        with self._registry._lock:
            metric = self._registry._metrics.get(self._key)
            if metric:
                metric.value -= value


class Histogram:
    """Histogram metric for latency distributions."""

    def __init__(
        self,
        registry: MetricsRegistry,
        key: str,
        buckets: list[float],
    ):
        self._registry = registry
        self._key = key
        self._buckets = sorted(buckets)

    def observe(self, value: float):
        """Record an observation."""
        with self._registry._lock:
            if self._key in self._registry._histograms:
                self._registry._histograms[self._key].append(value)
            metric = self._registry._metrics.get(self._key)
            if metric:
                metric.value += value  # Sum


# ==============================================================================
# Circuit Breaker
# ==============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_s: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker pattern for graceful degradation.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout expired
                if time.time() - self._last_failure_time >= self.config.timeout_s:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                return self._half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name}: CLOSED -> OPEN")

    async def execute(
        self,
        operation: Callable[[], Any],
        fallback: Callable[[], Any] | None = None,
    ) -> Any:
        """Execute operation with circuit breaker protection."""
        if not self.is_available():
            if fallback:
                return fallback() if not asyncio.iscoroutinefunction(fallback) else await fallback()
            raise RuntimeError(f"Circuit {self.name} is OPEN")

        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()
            self.record_success()
            return result

        except Exception:
            self.record_failure()
            if fallback:
                return fallback() if not asyncio.iscoroutinefunction(fallback) else await fallback()
            raise


# ==============================================================================
# Production Server Wrapper
# ==============================================================================

@dataclass
class ProductionConfig:
    """Production configuration."""
    # Error handling
    max_errors: int = 1000
    error_sample_rate: float = 1.0  # Log all errors

    # Health checks
    health_check_interval_s: float = 30.0

    # Metrics
    metrics_enabled: bool = True
    metrics_prefix: str = "voiceserver"

    # Circuit breakers
    enable_circuit_breakers: bool = True


class ProductionServer:
    """
    Production wrapper for voice server.

    Adds error handling, health checks, metrics, and circuit breakers.
    """

    def __init__(
        self,
        config: ProductionConfig | None = None,
    ):
        self.config = config or ProductionConfig()
        self.error_handler = ErrorHandler(self.config.max_errors)
        self.health_checker = HealthChecker()
        self.metrics = MetricsRegistry(self.config.metrics_prefix)

        # Standard metrics
        self._requests_total = self.metrics.counter(
            "requests_total", "Total requests processed",
        )
        self._request_errors = self.metrics.counter(
            "request_errors_total", "Total request errors",
        )
        self._active_sessions = self.metrics.gauge(
            "active_sessions", "Active client sessions",
        )
        self._request_latency = self.metrics.histogram(
            "request_latency_seconds", "Request latency in seconds",
        )

        # Circuit breakers
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Register default health checks
        self.health_checker.register_check(
            "memory",
            create_memory_health_check(),
        )

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name)
        return self._circuit_breakers[name]

    def record_request(self, latency_s: float, error: bool = False):
        """Record a request for metrics."""
        self._requests_total.inc()
        if error:
            self._request_errors.inc()
        self._request_latency.observe(latency_s)

    def set_active_sessions(self, count: int):
        """Update active session count."""
        self._active_sessions.set(count)

    def handle_error(
        self,
        code: str,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        **details,
    ) -> ServerError:
        """Handle and log an error."""
        error = ServerError(
            code=code,
            message=message,
            severity=severity,
            details=details,
        )
        self.error_handler.handle(error)
        return error

    def get_health(self) -> SystemHealth:
        """Get system health status."""
        return self.health_checker.check()

    def get_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return self.metrics.export_prometheus()

    def get_status(self) -> dict[str, Any]:
        """Get complete server status."""
        health = self.get_health()
        return {
            "health": health.to_dict(),
            "errors": {
                "recent_count": len(self.error_handler.get_recent_errors(10)),
                "by_code": self.error_handler.get_error_counts(),
            },
            "circuit_breakers": {
                name: cb.state.value
                for name, cb in self._circuit_breakers.items()
            },
        }
