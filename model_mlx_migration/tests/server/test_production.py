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

"""Tests for Phase 9.8 Production Hardening."""

import time

import pytest

from src.server.production import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ErrorCode,
    # Error handling
    ErrorHandler,
    ErrorSeverity,
    # Health checks
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    Metric,
    # Metrics
    MetricsRegistry,
    MetricType,
    ProductionConfig,
    # Production server
    ProductionServer,
    ServerError,
    SystemHealth,
    create_memory_health_check,
    create_model_health_check,
)


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_all_severities(self):
        """Test all severity levels."""
        assert ErrorSeverity.DEBUG.value == "debug"
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestServerError:
    """Tests for ServerError dataclass."""

    def test_default_values(self):
        """Test default error values."""
        error = ServerError(code="test", message="Test error")
        assert error.code == "test"
        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.ERROR
        assert error.timestamp > 0
        assert error.details == {}
        assert error.recoverable is True

    def test_to_dict(self):
        """Test conversion to dict."""
        error = ServerError(
            code="test",
            message="Test error",
            severity=ErrorSeverity.WARNING,
            details={"extra": "info"},
        )
        d = error.to_dict()
        assert d["code"] == "test"
        assert d["message"] == "Test error"
        assert d["severity"] == "warning"
        assert d["details"]["extra"] == "info"


class TestErrorCode:
    """Tests for ErrorCode constants."""

    def test_client_errors(self):
        """Test client error codes."""
        assert ErrorCode.INVALID_AUDIO == "invalid_audio"
        assert ErrorCode.INVALID_REQUEST == "invalid_request"
        assert ErrorCode.RATE_LIMITED == "rate_limited"

    def test_server_errors(self):
        """Test server error codes."""
        assert ErrorCode.MODEL_LOAD_FAILED == "model_load_failed"
        assert ErrorCode.INFERENCE_FAILED == "inference_failed"
        assert ErrorCode.TIMEOUT == "timeout"


class TestErrorHandler:
    """Tests for ErrorHandler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = ErrorHandler()
        assert len(handler.get_recent_errors()) == 0

    def test_handle_error(self):
        """Test handling an error."""
        handler = ErrorHandler()
        error = ServerError(code="test", message="Test")
        handler.handle(error)

        errors = handler.get_recent_errors()
        assert len(errors) == 1
        assert errors[0].code == "test"

    def test_error_counts(self):
        """Test error counting."""
        handler = ErrorHandler()

        for _ in range(3):
            handler.handle(ServerError(code="type_a", message="A"))
        for _ in range(2):
            handler.handle(ServerError(code="type_b", message="B"))

        counts = handler.get_error_counts()
        assert counts["type_a"] == 3
        assert counts["type_b"] == 2

    def test_register_handler(self):
        """Test registering custom handler."""
        handler = ErrorHandler()
        handled = []

        handler.register_handler("custom", lambda e: handled.append(e.code))

        handler.handle(ServerError(code="custom", message="Test"))
        assert handled == ["custom"]

    def test_max_errors(self):
        """Test max errors limit."""
        handler = ErrorHandler(max_errors=5)

        for i in range(10):
            handler.handle(ServerError(code=f"err_{i}", message="Test"))

        errors = handler.get_recent_errors()
        assert len(errors) == 5

    def test_clear(self):
        """Test clearing errors."""
        handler = ErrorHandler()
        handler.handle(ServerError(code="test", message="Test"))

        handler.clear()
        assert len(handler.get_recent_errors()) == 0
        assert len(handler.get_error_counts()) == 0


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_all_statuses(self):
        """Test all health statuses."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == ""
        assert result.latency_ms == 0.0
        assert result.details == {}

    def test_to_dict(self):
        """Test conversion to dict."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.DEGRADED,
            message="Slow",
            latency_ms=150.5,
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "degraded"
        assert d["message"] == "Slow"
        assert d["latency_ms"] == 150.5


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            checks=[
                HealthCheckResult(name="a", status=HealthStatus.HEALTHY),
                HealthCheckResult(name="b", status=HealthStatus.HEALTHY),
            ],
        )
        d = health.to_dict()
        assert d["status"] == "healthy"
        assert len(d["checks"]) == 2
        assert "timestamp_iso" in d


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_initialization(self):
        """Test checker initialization."""
        checker = HealthChecker()
        health = checker.check()
        assert health.status == HealthStatus.HEALTHY
        assert len(health.checks) == 0

    def test_register_check(self):
        """Test registering a health check."""
        checker = HealthChecker()

        def my_check():
            return HealthCheckResult(name="my_check", status=HealthStatus.HEALTHY)

        checker.register_check("my_check", my_check)
        health = checker.check()

        assert len(health.checks) == 1
        assert health.checks[0].name == "my_check"

    def test_overall_status_degraded(self):
        """Test overall status becomes degraded."""
        checker = HealthChecker()

        checker.register_check(
            "healthy",
            lambda: HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY),
        )
        checker.register_check(
            "degraded",
            lambda: HealthCheckResult(name="degraded", status=HealthStatus.DEGRADED),
        )

        health = checker.check()
        assert health.status == HealthStatus.DEGRADED

    def test_overall_status_unhealthy(self):
        """Test overall status becomes unhealthy."""
        checker = HealthChecker()

        checker.register_check(
            "healthy",
            lambda: HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY),
        )
        checker.register_check(
            "unhealthy",
            lambda: HealthCheckResult(name="unhealthy", status=HealthStatus.UNHEALTHY),
        )

        health = checker.check()
        assert health.status == HealthStatus.UNHEALTHY

    def test_check_exception_handling(self):
        """Test handling of check exceptions."""
        checker = HealthChecker()

        def failing_check():
            raise RuntimeError("Check failed")

        checker.register_check("failing", failing_check)
        health = checker.check()

        assert health.status == HealthStatus.UNHEALTHY
        assert health.checks[0].status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_async(self):
        """Test async health check."""
        checker = HealthChecker()
        checker.register_check(
            "test",
            lambda: HealthCheckResult(name="test", status=HealthStatus.HEALTHY),
        )

        health = await checker.check_async()
        assert health.status == HealthStatus.HEALTHY


class TestCreateHealthChecks:
    """Tests for health check factory functions."""

    def test_model_health_check_loaded(self):
        """Test model health check when loaded."""
        check = create_model_health_check("test", lambda: {"model": True})
        result = check()
        assert result.status == HealthStatus.HEALTHY

    def test_model_health_check_not_loaded(self):
        """Test model health check when not loaded."""
        check = create_model_health_check("test", lambda: None)
        result = check()
        assert result.status == HealthStatus.DEGRADED

    def test_model_health_check_error(self):
        """Test model health check on error."""
        def failing_get():
            raise RuntimeError("Failed")

        check = create_model_health_check("test", failing_get)
        result = check()
        assert result.status == HealthStatus.UNHEALTHY

    def test_memory_health_check(self):
        """Test memory health check."""
        check = create_memory_health_check(max_memory_mb=16384)
        result = check()
        # Should not be unhealthy with generous limit
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_types(self):
        """Test all metric types."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


class TestMetric:
    """Tests for Metric dataclass."""

    def test_to_prometheus_no_labels(self):
        """Test Prometheus format without labels."""
        metric = Metric(
            name="test_metric",
            type=MetricType.COUNTER,
            help="Test",
            value=42.0,
        )
        assert metric.to_prometheus() == "test_metric 42.0"

    def test_to_prometheus_with_labels(self):
        """Test Prometheus format with labels."""
        metric = Metric(
            name="test_metric",
            type=MetricType.COUNTER,
            help="Test",
            value=42.0,
            labels={"method": "GET", "status": "200"},
        )
        output = metric.to_prometheus()
        assert "test_metric{" in output
        assert 'method="GET"' in output
        assert 'status="200"' in output
        assert "42.0" in output


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = MetricsRegistry(prefix="test")
        assert registry.export_json()["metrics"] == []

    def test_counter(self):
        """Test counter metric."""
        registry = MetricsRegistry(prefix="app")
        counter = registry.counter("requests", "Total requests")

        counter.inc()
        counter.inc(5)

        metrics = registry.export_json()["metrics"]
        assert len(metrics) == 1
        assert metrics[0]["value"] == 6

    def test_gauge(self):
        """Test gauge metric."""
        registry = MetricsRegistry(prefix="app")
        gauge = registry.gauge("connections", "Active connections")

        gauge.set(10)
        gauge.inc(5)
        gauge.dec(3)

        metrics = registry.export_json()["metrics"]
        assert len(metrics) == 1
        assert metrics[0]["value"] == 12

    def test_histogram(self):
        """Test histogram metric."""
        registry = MetricsRegistry(prefix="app")
        histogram = registry.histogram(
            "latency",
            "Request latency",
            buckets=[0.1, 0.5, 1.0],
        )

        histogram.observe(0.2)
        histogram.observe(0.7)

        metrics = registry.export_json()["metrics"]
        assert len(metrics) == 1
        assert metrics[0]["value"] == pytest.approx(0.9)  # Sum

    def test_export_prometheus(self):
        """Test Prometheus export format."""
        registry = MetricsRegistry(prefix="app")
        registry.counter("requests", "Total requests").inc(100)

        output = registry.export_prometheus()
        assert "# HELP app_requests Total requests" in output
        assert "# TYPE app_requests counter" in output
        assert "app_requests 100" in output

    def test_counter_with_labels(self):
        """Test counter with labels."""
        registry = MetricsRegistry(prefix="app")
        counter_200 = registry.counter("http_requests", "HTTP requests", {"status": "200"})
        counter_500 = registry.counter("http_requests", "HTTP requests", {"status": "500"})

        counter_200.inc(100)
        counter_500.inc(5)

        metrics = registry.export_json()["metrics"]
        assert len(metrics) == 2


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_all_states(self):
        """Test all circuit states."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_s == 30.0
        assert config.half_open_max_calls == 3


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker("test")
        assert cb.name == "test"
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available() is True

    def test_record_success(self):
        """Test recording success."""
        cb = CircuitBreaker("test")
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_open_on_failures(self):
        """Test circuit opens on failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.is_available() is False

    def test_half_open_after_timeout(self):
        """Test transition to half-open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_s=0.01)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.02)
        assert cb.is_available() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_close_from_half_open(self):
        """Test closing from half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_s=0.01,
        )
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        time.sleep(0.02)
        cb.is_available()  # Transitions to half-open

        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopen_from_half_open(self):
        """Test reopening from half-open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_s=0.01)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        time.sleep(0.02)
        cb.is_available()  # Transitions to half-open

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution through circuit."""
        cb = CircuitBreaker("test")

        async def operation():
            return "success"

        result = await cb.execute(operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test failed execution through circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        async def failing_operation():
            raise RuntimeError("Failed")

        with pytest.raises(RuntimeError):
            await cb.execute(failing_operation)

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self):
        """Test fallback on failure."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        async def failing_operation():
            raise RuntimeError("Failed")

        def fallback():
            return "fallback"

        result = await cb.execute(failing_operation, fallback)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_execute_open_circuit(self):
        """Test execution on open circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_s=1000)
        cb = CircuitBreaker("test", config)
        cb.record_failure()

        async def operation():
            return "success"

        def fallback():
            return "fallback"

        result = await cb.execute(operation, fallback)
        assert result == "fallback"


class TestProductionConfig:
    """Tests for ProductionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = ProductionConfig()
        assert config.max_errors == 1000
        assert config.error_sample_rate == 1.0
        assert config.health_check_interval_s == 30.0
        assert config.metrics_enabled is True
        assert config.metrics_prefix == "voiceserver"
        assert config.enable_circuit_breakers is True


class TestProductionServer:
    """Tests for ProductionServer."""

    def test_initialization(self):
        """Test server initialization."""
        server = ProductionServer()
        assert server.error_handler is not None
        assert server.health_checker is not None
        assert server.metrics is not None

    def test_record_request(self):
        """Test recording request metrics."""
        server = ProductionServer()
        server.record_request(0.5)
        server.record_request(0.3, error=True)

        metrics = server.metrics.export_json()["metrics"]
        # Should have requests_total, request_errors_total, latency, active_sessions
        assert len(metrics) >= 2

    def test_set_active_sessions(self):
        """Test setting active sessions."""
        server = ProductionServer()
        server.set_active_sessions(10)
        server.set_active_sessions(5)

        # The gauge should reflect last value
        metrics = server.metrics.export_json()["metrics"]
        session_metric = next(
            (m for m in metrics if "sessions" in m["name"]), None,
        )
        assert session_metric is not None
        assert session_metric["value"] == 5

    def test_handle_error(self):
        """Test error handling."""
        server = ProductionServer()
        error = server.handle_error(
            code=ErrorCode.TIMEOUT,
            message="Request timed out",
            severity=ErrorSeverity.WARNING,
            duration_ms=5000,
        )

        assert error.code == ErrorCode.TIMEOUT
        assert error.details["duration_ms"] == 5000

    def test_get_health(self):
        """Test getting health status."""
        server = ProductionServer()
        health = server.get_health()

        assert health.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
        ]
        # Should have memory check
        assert any(c.name == "memory" for c in health.checks)

    def test_get_metrics_prometheus(self):
        """Test getting Prometheus metrics."""
        server = ProductionServer()
        server.record_request(0.1)

        output = server.get_metrics_prometheus()
        assert "voiceserver_requests_total" in output

    def test_get_circuit_breaker(self):
        """Test getting circuit breaker."""
        server = ProductionServer()

        cb1 = server.get_circuit_breaker("inference")
        cb2 = server.get_circuit_breaker("inference")

        assert cb1 is cb2
        assert cb1.name == "inference"

    def test_get_status(self):
        """Test getting full status."""
        server = ProductionServer()
        server.handle_error(code="test", message="Test error")

        status = server.get_status()
        assert "health" in status
        assert "errors" in status
        assert "circuit_breakers" in status
