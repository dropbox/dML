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
Memory management for SOTA++ Voice Server (Phase 9.6).

Provides:
- Model memory tracking and estimation
- Lazy loading with LRU eviction
- Memory limit enforcement
- Model lifecycle management
"""

import gc
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    """State of a managed model."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a managed model."""
    name: str
    state: ModelState = ModelState.UNLOADED
    memory_bytes: int = 0
    estimated_memory_bytes: int = 0
    last_used: float = 0.0
    load_count: int = 0
    error: str | None = None

    # Model instance (when loaded)
    instance: Any = None

    # Loader function
    loader: Callable[[], Any] | None = None

    # Priority (higher = keep loaded longer)
    priority: int = 0


@dataclass
class MemoryConfig:
    """Memory manager configuration."""
    # Memory limits
    max_memory_mb: int = 4096  # 4GB default
    target_memory_mb: int = 3072  # Start evicting at this level
    min_free_memory_mb: int = 512  # Keep at least this much free

    # Eviction policy
    evict_on_load: bool = True  # Evict models before loading new ones
    lru_eviction: bool = True  # Use LRU eviction policy

    # Monitoring
    track_usage: bool = True
    log_memory_ops: bool = True

    # Timeouts
    load_timeout_s: float = 60.0
    unload_timeout_s: float = 10.0


@dataclass
class MemoryStats:
    """Current memory statistics."""
    total_managed_mb: float = 0.0
    loaded_models: int = 0
    unloaded_models: int = 0
    process_memory_mb: float = 0.0
    system_available_mb: float = 0.0
    metal_memory_mb: float = 0.0


class MemoryManager:
    """
    Manages memory for loaded models.

    Tracks memory usage, provides lazy loading with LRU eviction,
    and enforces memory limits.
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self._models: OrderedDict[str, ModelInfo] = OrderedDict()
        self._lock = threading.RLock()
        self._load_callbacks: list[Callable[[str], None]] = []
        self._unload_callbacks: list[Callable[[str], None]] = []

    def register_model(
        self,
        name: str,
        loader: Callable[[], Any],
        estimated_memory_mb: int = 0,
        priority: int = 0,
    ) -> ModelInfo:
        """
        Register a model for management.

        Args:
            name: Unique model identifier
            loader: Function to load the model
            estimated_memory_mb: Estimated memory usage when loaded
            priority: Higher priority models are kept loaded longer

        Returns:
            ModelInfo for the registered model
        """
        with self._lock:
            if name in self._models:
                return self._models[name]

            info = ModelInfo(
                name=name,
                loader=loader,
                estimated_memory_bytes=estimated_memory_mb * 1024 * 1024,
                priority=priority,
            )
            self._models[name] = info

            if self.config.log_memory_ops:
                logger.info(f"Registered model: {name} (estimated {estimated_memory_mb}MB)")

            return info

    def get_model(self, name: str) -> Any | None:
        """
        Get a model instance, loading if necessary.

        Uses lazy loading with LRU eviction if memory limits are exceeded.

        Args:
            name: Model identifier

        Returns:
            Model instance, or None if not registered/failed to load
        """
        with self._lock:
            info = self._models.get(name)
            if info is None:
                logger.warning(f"Model not registered: {name}")
                return None

            # If already loaded, update access time and return
            if info.state == ModelState.LOADED and info.instance is not None:
                info.last_used = time.time()
                # Move to end for LRU tracking
                self._models.move_to_end(name)
                return info.instance

            # Need to load
            return self._load_model(info)

    def _load_model(self, info: ModelInfo) -> Any | None:
        """Load a model, evicting others if necessary."""
        if info.loader is None:
            info.state = ModelState.ERROR
            info.error = "No loader registered"
            return None

        # Check if we need to evict first
        if self.config.evict_on_load:
            self._ensure_memory_available(info.estimated_memory_bytes)

        info.state = ModelState.LOADING

        try:
            if self.config.log_memory_ops:
                logger.info(f"Loading model: {info.name}")

            start_time = time.time()
            instance = info.loader()
            load_time = time.time() - start_time

            info.instance = instance
            info.state = ModelState.LOADED
            info.load_count += 1
            info.last_used = time.time()
            info.error = None

            # Measure actual memory
            info.memory_bytes = self._estimate_model_memory(instance)

            if self.config.log_memory_ops:
                logger.info(
                    f"Loaded model: {info.name} in {load_time:.2f}s "
                    f"({info.memory_bytes / 1024 / 1024:.1f}MB)",
                )

            # Notify callbacks
            for callback in self._load_callbacks:
                try:
                    callback(info.name)
                except Exception as e:
                    logger.error(f"Load callback error: {e}")

            return instance

        except Exception as e:
            info.state = ModelState.ERROR
            info.error = str(e)
            logger.error(f"Failed to load model {info.name}: {e}")
            return None

    def unload_model(self, name: str) -> bool:
        """
        Unload a model to free memory.

        Args:
            name: Model identifier

        Returns:
            True if unloaded successfully
        """
        with self._lock:
            info = self._models.get(name)
            if info is None:
                return False

            if info.state != ModelState.LOADED:
                return False

            return self._unload_model(info)

    def _unload_model(self, info: ModelInfo) -> bool:
        """Unload a model."""
        info.state = ModelState.UNLOADING

        try:
            if self.config.log_memory_ops:
                logger.info(f"Unloading model: {info.name}")

            # Clear instance
            old_instance = info.instance
            info.instance = None
            info.state = ModelState.UNLOADED

            # Delete and garbage collect
            del old_instance
            gc.collect()

            # MLX-specific cleanup
            if HAS_MLX:
                try:
                    mx.metal.clear_cache()
                except Exception:
                    pass

            # Notify callbacks
            for callback in self._unload_callbacks:
                try:
                    callback(info.name)
                except Exception as e:
                    logger.error(f"Unload callback error: {e}")

            if self.config.log_memory_ops:
                logger.info(f"Unloaded model: {info.name}")

            return True

        except Exception as e:
            info.state = ModelState.ERROR
            info.error = str(e)
            logger.error(f"Failed to unload model {info.name}: {e}")
            return False

    def _ensure_memory_available(self, needed_bytes: int):
        """Ensure enough memory is available, evicting models if necessary."""
        stats = self.get_stats()
        self.config.max_memory_mb * 1024 * 1024
        target_bytes = self.config.target_memory_mb * 1024 * 1024

        current_bytes = int(stats.total_managed_mb * 1024 * 1024)

        # Check if we need to evict
        if current_bytes + needed_bytes <= target_bytes:
            return

        # Evict LRU models until we have enough space
        bytes_to_free = current_bytes + needed_bytes - target_bytes
        bytes_freed = 0

        if self.config.lru_eviction:
            # Get models sorted by last_used (oldest first)
            candidates = [
                info for info in self._models.values()
                if info.state == ModelState.LOADED
            ]
            candidates.sort(key=lambda x: (x.priority, x.last_used))

            for info in candidates:
                if bytes_freed >= bytes_to_free:
                    break

                if self._unload_model(info):
                    bytes_freed += info.memory_bytes

    def _estimate_model_memory(self, instance: Any) -> int:
        """Estimate memory usage of a model instance."""
        # Try to get MLX memory info
        if HAS_MLX:
            try:
                # For MLX models, we can estimate from parameters
                if hasattr(instance, 'parameters'):
                    total_bytes = 0
                    params = instance.parameters()
                    if isinstance(params, dict):
                        for key, value in params.items():
                            if hasattr(value, 'nbytes'):
                                total_bytes += value.nbytes
                            elif hasattr(value, 'size') and hasattr(value, 'itemsize'):
                                total_bytes += value.size * value.itemsize
                    return total_bytes
            except Exception:
                pass

        # Fallback: use sys.getsizeof (not accurate for complex objects)
        import sys
        try:
            return sys.getsizeof(instance)
        except Exception:
            return 0

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        stats = MemoryStats()

        with self._lock:
            loaded_bytes = 0
            loaded_count = 0
            unloaded_count = 0

            for info in self._models.values():
                if info.state == ModelState.LOADED:
                    loaded_bytes += info.memory_bytes
                    loaded_count += 1
                elif info.state == ModelState.UNLOADED:
                    unloaded_count += 1

            stats.total_managed_mb = loaded_bytes / 1024 / 1024
            stats.loaded_models = loaded_count
            stats.unloaded_models = unloaded_count

        # Process memory
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                stats.process_memory_mb = process.memory_info().rss / 1024 / 1024
                mem = psutil.virtual_memory()
                stats.system_available_mb = mem.available / 1024 / 1024
            except Exception:
                pass

        # MLX Metal memory
        if HAS_MLX:
            try:
                cache_info = mx.metal.get_cache_memory()
                stats.metal_memory_mb = cache_info / 1024 / 1024
            except Exception:
                pass

        return stats

    def get_model_info(self, name: str) -> ModelInfo | None:
        """Get info for a specific model."""
        with self._lock:
            return self._models.get(name)

    def list_models(self) -> list[ModelInfo]:
        """List all registered models."""
        with self._lock:
            return list(self._models.values())

    def on_load(self, callback: Callable[[str], None]):
        """Register callback for model load events."""
        self._load_callbacks.append(callback)

    def on_unload(self, callback: Callable[[str], None]):
        """Register callback for model unload events."""
        self._unload_callbacks.append(callback)

    def preload(self, names: list[str]) -> dict[str, bool]:
        """
        Preload multiple models.

        Args:
            names: List of model names to preload

        Returns:
            Dict mapping name -> success
        """
        results = {}
        for name in names:
            instance = self.get_model(name)
            results[name] = instance is not None
        return results

    def unload_all(self) -> int:
        """
        Unload all loaded models.

        Returns:
            Number of models unloaded
        """
        count = 0
        with self._lock:
            for info in list(self._models.values()):
                if info.state == ModelState.LOADED:
                    if self._unload_model(info):
                        count += 1
        return count

    def clear_cache(self):
        """Clear MLX metal cache."""
        if HAS_MLX:
            try:
                mx.metal.clear_cache()
            except Exception:
                pass
        gc.collect()


class ModelRegistry:
    """
    Singleton registry for voice server models.

    Provides a central place to register and access models
    with memory management.
    """

    _instance: Optional['ModelRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls, config: MemoryConfig | None = None) -> 'ModelRegistry':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._memory_manager = MemoryManager(config)
                cls._instance._initialized = True
            return cls._instance

    @property
    def memory_manager(self) -> MemoryManager:
        return self._memory_manager

    def register_zipformer(
        self,
        checkpoint_path: str,
        bpe_model_path: str,
        name: str = "zipformer",
    ) -> ModelInfo:
        """Register Zipformer model for lazy loading."""
        def loader():
            from ..models.zipformer.inference import ASRPipeline
            return ASRPipeline.from_pretrained(checkpoint_path, bpe_model_path)

        return self._memory_manager.register_model(
            name=name,
            loader=loader,
            estimated_memory_mb=300,  # ~300MB for Zipformer
            priority=10,  # High priority - keep loaded
        )

    def register_whisper(
        self,
        model_name: str = "large-v3",
        name: str = "whisper",
    ) -> ModelInfo:
        """Register Whisper model for lazy loading."""
        def loader():
            # This would load actual Whisper model
            # For now, return placeholder
            return {"model": model_name, "type": "whisper"}

        return self._memory_manager.register_model(
            name=name,
            loader=loader,
            estimated_memory_mb=3000,  # ~3GB for large-v3
            priority=5,  # Medium priority
        )

    def register_rich_heads(
        self,
        encoder_dim: int = 384,
        name: str = "rich_heads",
    ) -> ModelInfo:
        """Register rich audio heads for lazy loading."""
        def loader():
            from ..models.heads import RichAudioHeads, RichAudioHeadsConfig
            config = RichAudioHeadsConfig(encoder_dim=encoder_dim)
            return RichAudioHeads(config)

        return self._memory_manager.register_model(
            name=name,
            loader=loader,
            estimated_memory_mb=50,  # ~50MB for heads
            priority=8,
        )

    def register_speaker_model(
        self,
        checkpoint_path: str | None = None,
        name: str = "speaker",
    ) -> ModelInfo:
        """Register speaker embedding model for lazy loading."""
        def loader():
            # This would load actual speaker model
            return {"type": "speaker", "checkpoint": checkpoint_path}

        return self._memory_manager.register_model(
            name=name,
            loader=loader,
            estimated_memory_mb=100,  # ~100MB for DELULU
            priority=6,
        )

    def register_separator(
        self,
        checkpoint_path: str | None = None,
        name: str = "separator",
    ) -> ModelInfo:
        """Register source separator (FLASepformer) for lazy loading."""
        def loader():
            # This would load actual separator model
            return {"type": "separator", "checkpoint": checkpoint_path}

        return self._memory_manager.register_model(
            name=name,
            loader=loader,
            estimated_memory_mb=150,  # ~150MB for FLASepformer
            priority=4,  # Lower priority - only needed for multi-speaker
        )

    def get_model(self, name: str) -> Any | None:
        """Get a model by name."""
        return self._memory_manager.get_model(name)

    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        return self._memory_manager.get_stats()

    def unload_unused(self, max_age_s: float = 300.0) -> int:
        """Unload models not used recently."""
        count = 0
        current_time = time.time()

        for info in self._memory_manager.list_models():
            if info.state == ModelState.LOADED:
                age = current_time - info.last_used
                if age > max_age_s:
                    if self._memory_manager.unload_model(info.name):
                        count += 1

        return count

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._memory_manager.unload_all()
            cls._instance = None
