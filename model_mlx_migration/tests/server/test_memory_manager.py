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

"""Tests for Phase 9.6 Memory Manager."""

import time

from src.server.memory_manager import (
    MemoryConfig,
    MemoryManager,
    MemoryStats,
    ModelInfo,
    ModelRegistry,
    ModelState,
)


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MemoryConfig()
        assert config.max_memory_mb == 4096
        assert config.target_memory_mb == 3072
        assert config.min_free_memory_mb == 512
        assert config.evict_on_load is True
        assert config.lru_eviction is True
        assert config.track_usage is True
        assert config.log_memory_ops is True
        assert config.load_timeout_s == 60.0
        assert config.unload_timeout_s == 10.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MemoryConfig(
            max_memory_mb=8192,
            target_memory_mb=6144,
            evict_on_load=False,
        )
        assert config.max_memory_mb == 8192
        assert config.target_memory_mb == 6144
        assert config.evict_on_load is False


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_default_values(self):
        """Test default model info values."""
        info = ModelInfo(name="test_model")
        assert info.name == "test_model"
        assert info.state == ModelState.UNLOADED
        assert info.memory_bytes == 0
        assert info.estimated_memory_bytes == 0
        assert info.last_used == 0.0
        assert info.load_count == 0
        assert info.error is None
        assert info.instance is None
        assert info.loader is None
        assert info.priority == 0


class TestModelState:
    """Tests for ModelState enum."""

    def test_all_states(self):
        """Test all model states."""
        assert ModelState.UNLOADED.value == "unloaded"
        assert ModelState.LOADING.value == "loading"
        assert ModelState.LOADED.value == "loaded"
        assert ModelState.UNLOADING.value == "unloading"
        assert ModelState.ERROR.value == "error"


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = MemoryStats()
        assert stats.total_managed_mb == 0.0
        assert stats.loaded_models == 0
        assert stats.unloaded_models == 0
        assert stats.process_memory_mb == 0.0
        assert stats.system_available_mb == 0.0
        assert stats.metal_memory_mb == 0.0


class TestMemoryManager:
    """Tests for MemoryManager."""

    def test_initialization(self):
        """Test memory manager initialization."""
        manager = MemoryManager()
        assert manager.config is not None
        assert len(manager.list_models()) == 0

    def test_register_model(self):
        """Test model registration."""
        manager = MemoryManager()

        # Simple loader
        def loader():
            return {"type": "test"}

        info = manager.register_model(
            name="test_model",
            loader=loader,
            estimated_memory_mb=100,
            priority=5,
        )

        assert info.name == "test_model"
        assert info.estimated_memory_bytes == 100 * 1024 * 1024
        assert info.priority == 5
        assert info.state == ModelState.UNLOADED

    def test_register_duplicate(self):
        """Test registering same model twice returns existing."""
        manager = MemoryManager()

        def loader():
            return {"type": "test"}

        info1 = manager.register_model("test", loader, 100)
        info2 = manager.register_model("test", loader, 200)

        assert info1 is info2
        assert len(manager.list_models()) == 1

    def test_get_model_lazy_load(self):
        """Test lazy loading via get_model."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))
        load_count = [0]

        def loader():
            load_count[0] += 1
            return {"loaded": True}

        manager.register_model("test", loader, 50)

        # First get should load
        model = manager.get_model("test")
        assert model == {"loaded": True}
        assert load_count[0] == 1

        # Second get should not reload
        model2 = manager.get_model("test")
        assert model2 == {"loaded": True}
        assert load_count[0] == 1

    def test_get_unregistered_model(self):
        """Test getting unregistered model returns None."""
        manager = MemoryManager()
        model = manager.get_model("nonexistent")
        assert model is None

    def test_unload_model(self):
        """Test model unloading."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))

        def loader():
            return {"data": "test"}

        manager.register_model("test", loader, 50)

        # Load then unload
        manager.get_model("test")
        info = manager.get_model_info("test")
        assert info.state == ModelState.LOADED

        result = manager.unload_model("test")
        assert result is True

        info = manager.get_model_info("test")
        assert info.state == ModelState.UNLOADED
        assert info.instance is None

    def test_unload_unloaded_model(self):
        """Test unloading already unloaded model."""
        manager = MemoryManager()

        def loader():
            return {}

        manager.register_model("test", loader, 50)

        # Try to unload without loading first
        result = manager.unload_model("test")
        assert result is False

    def test_get_stats(self):
        """Test getting memory stats."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))

        def loader():
            return {"data": [0] * 1000}

        manager.register_model("test", loader, 10)

        # Before loading
        stats = manager.get_stats()
        assert stats.loaded_models == 0
        assert stats.unloaded_models == 1

        # After loading
        manager.get_model("test")
        stats = manager.get_stats()
        assert stats.loaded_models == 1
        assert stats.unloaded_models == 0

    def test_load_failure(self):
        """Test handling load failure."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))

        def failing_loader():
            raise RuntimeError("Load failed")

        manager.register_model("failing", failing_loader, 50)

        model = manager.get_model("failing")
        assert model is None

        info = manager.get_model_info("failing")
        assert info.state == ModelState.ERROR
        assert "Load failed" in info.error

    def test_unload_all(self):
        """Test unloading all models."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))

        for i in range(3):
            manager.register_model(f"model_{i}", dict, 10)
            manager.get_model(f"model_{i}")

        stats = manager.get_stats()
        assert stats.loaded_models == 3

        count = manager.unload_all()
        assert count == 3

        stats = manager.get_stats()
        assert stats.loaded_models == 0

    def test_lru_updates_on_access(self):
        """Test LRU tracking updates on access."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))

        manager.register_model("model_a", lambda: {"a": 1}, 10)
        manager.register_model("model_b", lambda: {"b": 2}, 10)

        # Load both
        manager.get_model("model_a")
        time.sleep(0.01)
        manager.get_model("model_b")

        info_a = manager.get_model_info("model_a")
        info_b = manager.get_model_info("model_b")

        # B should have later access time
        assert info_b.last_used > info_a.last_used

        # Access A again
        time.sleep(0.01)
        manager.get_model("model_a")
        info_a = manager.get_model_info("model_a")

        # Now A should have later access time
        assert info_a.last_used > info_b.last_used

    def test_preload_multiple(self):
        """Test preloading multiple models."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))

        manager.register_model("a", lambda: {"a": 1}, 10)
        manager.register_model("b", lambda: {"b": 2}, 10)
        manager.register_model("fail", lambda: 1/0, 10)  # Will fail

        results = manager.preload(["a", "b", "fail"])

        assert results["a"] is True
        assert results["b"] is True
        assert results["fail"] is False

    def test_on_load_callback(self):
        """Test load callback."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))
        loaded = []

        manager.on_load(lambda name: loaded.append(name))
        manager.register_model("test", dict, 10)
        manager.get_model("test")

        assert loaded == ["test"]

    def test_on_unload_callback(self):
        """Test unload callback."""
        manager = MemoryManager(MemoryConfig(log_memory_ops=False))
        unloaded = []

        manager.on_unload(lambda name: unloaded.append(name))
        manager.register_model("test", dict, 10)
        manager.get_model("test")
        manager.unload_model("test")

        assert unloaded == ["test"]

    def test_clear_cache(self):
        """Test cache clearing."""
        manager = MemoryManager()
        # Should not raise
        manager.clear_cache()


class TestModelRegistry:
    """Tests for ModelRegistry singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        ModelRegistry.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        ModelRegistry.reset()

    def test_singleton(self):
        """Test singleton pattern."""
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        assert r1 is r2

    def test_memory_manager_access(self):
        """Test access to underlying memory manager."""
        registry = ModelRegistry()
        assert registry.memory_manager is not None
        assert isinstance(registry.memory_manager, MemoryManager)

    def test_get_stats(self):
        """Test getting stats through registry."""
        registry = ModelRegistry()
        stats = registry.get_stats()
        assert isinstance(stats, MemoryStats)

    def test_register_zipformer(self):
        """Test registering Zipformer model."""
        registry = ModelRegistry()
        info = registry.register_zipformer(
            checkpoint_path="/fake/path",
            bpe_model_path="/fake/bpe",
            name="test_zipformer",
        )
        assert info.name == "test_zipformer"
        assert info.estimated_memory_bytes == 300 * 1024 * 1024
        assert info.priority == 10

    def test_register_whisper(self):
        """Test registering Whisper model."""
        registry = ModelRegistry()
        info = registry.register_whisper(
            model_name="large-v3",
            name="test_whisper",
        )
        assert info.name == "test_whisper"
        assert info.estimated_memory_bytes == 3000 * 1024 * 1024
        assert info.priority == 5

    def test_register_rich_heads(self):
        """Test registering rich audio heads."""
        registry = ModelRegistry()
        info = registry.register_rich_heads(
            encoder_dim=384,
            name="test_heads",
        )
        assert info.name == "test_heads"
        assert info.estimated_memory_bytes == 50 * 1024 * 1024
        assert info.priority == 8

    def test_register_speaker_model(self):
        """Test registering speaker model."""
        registry = ModelRegistry()
        info = registry.register_speaker_model(
            checkpoint_path="/fake/path",
            name="test_speaker",
        )
        assert info.name == "test_speaker"
        assert info.estimated_memory_bytes == 100 * 1024 * 1024
        assert info.priority == 6

    def test_register_separator(self):
        """Test registering separator model."""
        registry = ModelRegistry()
        info = registry.register_separator(
            checkpoint_path="/fake/path",
            name="test_separator",
        )
        assert info.name == "test_separator"
        assert info.estimated_memory_bytes == 150 * 1024 * 1024
        assert info.priority == 4

    def test_get_model(self):
        """Test getting model through registry."""
        registry = ModelRegistry()
        # Register a simple test model
        registry.memory_manager.register_model(
            "simple", lambda: {"test": True}, 10,
        )
        model = registry.get_model("simple")
        assert model == {"test": True}

    def test_unload_unused(self):
        """Test unloading unused models."""
        registry = ModelRegistry()

        # Register and load a model
        registry.memory_manager.register_model(
            "old", lambda: {"old": True}, 10,
        )
        registry.get_model("old")

        # Set last_used to very old
        info = registry.memory_manager.get_model_info("old")
        info.last_used = time.time() - 1000

        count = registry.unload_unused(max_age_s=100)
        assert count == 1

    def test_reset(self):
        """Test registry reset."""
        r1 = ModelRegistry()
        r1.memory_manager.register_model("test", dict, 10)

        ModelRegistry.reset()

        r2 = ModelRegistry()
        assert r1 is not r2
        assert len(r2.memory_manager.list_models()) == 0
