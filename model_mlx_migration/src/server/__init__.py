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
SOTA++ Voice Server - Phase 9 Integration Module.

This package provides the streaming voice server infrastructure:

Phase 9.1: Streaming server framework
- VoiceServer: WebSocket-based streaming ASR server
- RichToken: Unified output format with all rich features
- ASRPipeline: Abstract interface for ASR backends

Phase 9.2: Multi-speaker pipeline
- MultiSpeakerPipeline: FLASepformer-based source separation
- OverlapDetector: Detects overlapping speakers

Phase 9.3: Language routing
- LanguageRouter: Routes audio to appropriate ASR backend
- ASRBackend enum: zipformer, whisper, rover

Phase 9.4-9.5: Integrated ASR pipeline
- IntegratedASRPipeline: Full pipeline with rich heads, ROVER
- Mode switching (streaming vs high-accuracy)

Phase 9.6: Memory management
- MemoryManager: Model memory tracking and LRU eviction
- ModelRegistry: Singleton for model lifecycle management

Phase 9.7: Latency optimization
- LatencyProfiler: Pipeline stage timing
- PipelineOptimizer: Batch processing and async helpers

Phase 9.8: Production hardening
- ErrorHandler: Structured error handling
- HealthChecker: System health monitoring
- MetricsRegistry: Prometheus-compatible metrics
- CircuitBreaker: Graceful degradation
"""

from .asr_pipeline import (
    # Constants
    EMOTION_LABELS,
    # Pipelines
    IntegratedASRPipeline,
    IntegratedPipelineConfig,
    ZipformerASRPipeline,
)
from .language_router import (
    # Utilities
    LANGUAGE_NAMES,
    # Backend
    ASRBackend,
    LanguageDetectionResult,
    # Router
    LanguageRouter,
    LanguageRouterConfig,
    MockLanguageRouter,
    get_language_name,
)
from .latency_optimizer import (
    AsyncThrottler,
    BatchAccumulator,
    LatencyConfig,
    # Profiler
    LatencyProfiler,
    # Optimizer
    PipelineOptimizer,
    RequestTrace,
    StageMetrics,
    run_concurrent,
    # Utilities
    run_with_timeout,
)
from .memory_manager import (
    MemoryConfig,
    # Core
    MemoryManager,
    MemoryStats,
    # Model management
    ModelInfo,
    ModelRegistry,
    ModelState,
)
from .multi_speaker import (
    MockMultiSpeakerPipeline,
    MultiSpeakerConfig,
    # Pipeline
    MultiSpeakerPipeline,
    MultiSpeakerResult,
    # Detection
    OverlapDetector,
    OverlapStatus,
    SpeakerSegment,
)
from .production import (
    # Circuit breaker
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    Counter,
    ErrorCode,
    # Error handling
    ErrorHandler,
    ErrorSeverity,
    Gauge,
    # Health checks
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    Histogram,
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
from .rich_token import (
    ASRMode,
    EmotionLabel,
    HallucinationInfo,
    LanguageInfo,
    ParalinguisticsInfo,
    PhonemeInfo,
    PitchInfo,
    # Core types
    RichToken,
    SingingInfo,
    SpeakerInfo,
    StreamingResponse,
    # Info dataclasses
    WordTimestamp,
    create_final_token,
    # Factory functions
    create_partial_token,
)
from .voice_server import (
    # Pipeline
    ASRPipeline,
    # Session
    ClientSession,
    MockASRPipeline,
    ServerConfig,
    ServerState,
    SessionMetrics,
    SessionState,
    # Server
    VoiceServer,
)

__all__ = [
    # Rich token
    "RichToken",
    "StreamingResponse",
    "ASRMode",
    "EmotionLabel",
    "WordTimestamp",
    "PitchInfo",
    "PhonemeInfo",
    "ParalinguisticsInfo",
    "LanguageInfo",
    "SingingInfo",
    "SpeakerInfo",
    "HallucinationInfo",
    "create_partial_token",
    "create_final_token",
    # Voice server
    "VoiceServer",
    "ServerConfig",
    "ServerState",
    "ClientSession",
    "SessionState",
    "SessionMetrics",
    "ASRPipeline",
    "MockASRPipeline",
    # Multi-speaker (Phase 9.2)
    "MultiSpeakerPipeline",
    "MockMultiSpeakerPipeline",
    "MultiSpeakerConfig",
    "MultiSpeakerResult",
    "OverlapDetector",
    "OverlapStatus",
    "SpeakerSegment",
    # Language routing (Phase 9.3)
    "LanguageRouter",
    "MockLanguageRouter",
    "LanguageRouterConfig",
    "LanguageDetectionResult",
    "ASRBackend",
    "LANGUAGE_NAMES",
    "get_language_name",
    # ASR pipelines (Phase 9.4)
    "IntegratedASRPipeline",
    "ZipformerASRPipeline",
    "IntegratedPipelineConfig",
    "EMOTION_LABELS",
    # Memory management (Phase 9.6)
    "MemoryManager",
    "MemoryConfig",
    "MemoryStats",
    "ModelInfo",
    "ModelState",
    "ModelRegistry",
    # Latency optimization (Phase 9.7)
    "LatencyProfiler",
    "LatencyConfig",
    "StageMetrics",
    "RequestTrace",
    "PipelineOptimizer",
    "BatchAccumulator",
    "AsyncThrottler",
    "run_with_timeout",
    "run_concurrent",
    # Production hardening (Phase 9.8)
    "ErrorHandler",
    "ErrorSeverity",
    "ServerError",
    "ErrorCode",
    "HealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    "SystemHealth",
    "create_model_health_check",
    "create_memory_health_check",
    "MetricsRegistry",
    "MetricType",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "ProductionServer",
    "ProductionConfig",
]
