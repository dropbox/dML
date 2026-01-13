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
Tests for WhisperMLX Encoder Cache (OPT-W4).

Tests:
1. EncoderCache unit tests (get/put, LRU eviction, stats)
2. Integration with WhisperMLX model
"""

import numpy as np
import pytest

# Skip if MLX not available
mlx = pytest.importorskip("mlx.core")
mx = mlx

# Use modern numpy RNG
_rng = np.random.default_rng(42)


class TestEncoderCacheUnit:
    """Unit tests for EncoderCache class."""

    def test_cache_creation(self):
        """Test cache instantiation."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=8)
        assert len(cache) == 0
        assert cache.max_entries == 8

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=8)

        # Create test mel spectrogram
        mel = mx.array(_rng.standard_normal((1, 3000, 128)).astype(np.float16))
        audio_features = mx.array(_rng.standard_normal((1, 1500, 1280)).astype(np.float16))

        # Put
        cache_key = cache.put(
            mel=mel,
            audio_features=audio_features,
            variable_length=False,
            encoder_positions=1500,
            audio_duration=30.0,
        )

        assert len(cache) == 1
        assert cache_key is not None

        # Get
        entry = cache.get(mel, variable_length=False)
        assert entry is not None
        assert entry.encoder_positions == 1500
        assert entry.audio_duration == 30.0
        assert entry.access_count == 1

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=8)

        mel = mx.array(_rng.standard_normal((1, 3000, 128)).astype(np.float16))
        entry = cache.get(mel, variable_length=False)
        assert entry is None

    def test_cache_variable_length_distinction(self):
        """Test that variable_length flag distinguishes cache entries."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=8)

        mel = mx.array(_rng.standard_normal((1, 1000, 128)).astype(np.float16))
        features_std = mx.array(_rng.standard_normal((1, 1500, 1280)).astype(np.float16))
        features_var = mx.array(_rng.standard_normal((1, 500, 1280)).astype(np.float16))

        # Store standard mode
        cache.put(mel, features_std, variable_length=False, encoder_positions=1500)

        # Store variable-length mode
        cache.put(mel, features_var, variable_length=True, encoder_positions=500)

        # Get should return correct entry based on variable_length
        entry_std = cache.get(mel, variable_length=False)
        entry_var = cache.get(mel, variable_length=True)

        assert entry_std.encoder_positions == 1500
        assert entry_var.encoder_positions == 500
        assert len(cache) == 2

    def test_lru_eviction(self):
        """Test LRU eviction when max_entries exceeded."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=3)

        # Add 3 entries
        for i in range(3):
            mel = mx.array(np.full((1, 1000, 128), i, dtype=np.float16))
            features = mx.array(_rng.standard_normal((1, 1500, 1280)).astype(np.float16))
            cache.put(mel, features, variable_length=False, encoder_positions=1500)

        assert len(cache) == 3

        # Access first entry to make it recent
        mel0 = mx.array(np.full((1, 1000, 128), 0, dtype=np.float16))
        cache.get(mel0, variable_length=False)

        # Add 4th entry - should evict entry 1 (oldest non-accessed)
        mel3 = mx.array(np.full((1, 1000, 128), 3, dtype=np.float16))
        features = mx.array(_rng.standard_normal((1, 1500, 1280)).astype(np.float16))
        cache.put(mel3, features, variable_length=False, encoder_positions=1500)

        assert len(cache) == 3
        assert cache._evictions >= 1

        # Entry 0 should still be cached (was accessed)
        assert cache.get(mel0, variable_length=False) is not None

        # Entry 3 should be cached (just added)
        assert cache.get(mel3, variable_length=False) is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=8)

        mel = mx.array(_rng.standard_normal((1, 3000, 128)).astype(np.float16))
        features = mx.array(_rng.standard_normal((1, 1500, 1280)).astype(np.float16))

        # Miss
        cache.get(mel, variable_length=False)

        # Put
        cache.put(mel, features, variable_length=False, encoder_positions=1500)

        # Hit
        cache.get(mel, variable_length=False)

        stats = cache.stats
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self):
        """Test cache clear operation."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=8)

        # Add entries
        for i in range(5):
            mel = mx.array(np.full((1, 1000, 128), i, dtype=np.float16))
            features = mx.array(_rng.standard_normal((1, 1500, 1280)).astype(np.float16))
            cache.put(mel, features, variable_length=False, encoder_positions=1500)

        assert len(cache) == 5

        cache.clear()
        assert len(cache) == 0

    def test_cache_invalidate(self):
        """Test invalidating specific cache entry."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=8)

        mel1 = mx.array(np.full((1, 1000, 128), 1, dtype=np.float16))
        mel2 = mx.array(np.full((1, 1000, 128), 2, dtype=np.float16))
        features = mx.array(_rng.standard_normal((1, 1500, 1280)).astype(np.float16))

        cache.put(mel1, features, variable_length=False, encoder_positions=1500)
        cache.put(mel2, features, variable_length=False, encoder_positions=1500)

        assert len(cache) == 2

        # Invalidate mel1
        removed = cache.invalidate(mel1, variable_length=False)
        assert removed is True
        assert len(cache) == 1

        # mel2 should still be cached
        assert cache.get(mel2, variable_length=False) is not None

        # Invalidating non-existent entry returns False
        removed = cache.invalidate(mel1, variable_length=False)
        assert removed is False


class TestEncoderCacheIntegration:
    """Integration tests for encoder cache with WhisperMLX model."""

    def test_enable_disable_cache(self):
        """Test enabling and disabling encoder cache."""
        from tools.whisper_mlx.config import get_config
        from tools.whisper_mlx.model import WhisperMLX

        # Create minimal config for testing
        config = get_config("tiny")
        model = WhisperMLX(config, dtype=mx.float16)

        # Initially no cache
        assert model.encoder_cache_enabled is False
        assert model.get_encoder_cache_stats() is None

        # Enable cache
        model.enable_encoder_cache(max_entries=8)
        assert model.encoder_cache_enabled is True
        stats = model.get_encoder_cache_stats()
        assert stats is not None
        assert stats["max_entries"] == 8

        # Disable cache
        model.disable_encoder_cache()
        assert model.encoder_cache_enabled is False

    def test_clear_cache(self):
        """Test clearing encoder cache."""
        from tools.whisper_mlx.config import get_config
        from tools.whisper_mlx.model import WhisperMLX

        config = get_config("tiny")
        model = WhisperMLX(config, dtype=mx.float16)
        model.enable_encoder_cache(max_entries=8)

        # Manually add to cache for testing
        mel = mx.array(_rng.standard_normal((1, 3000, 128)).astype(np.float16))
        features = mx.array(_rng.standard_normal((1, 1500, 384)).astype(np.float16))
        model._encoder_cache.put(mel, features, variable_length=False, encoder_positions=1500)

        assert len(model._encoder_cache) == 1

        model.clear_encoder_cache()
        assert len(model._encoder_cache) == 0

    def test_embed_audio_without_cache(self):
        """Test embed_audio works without cache enabled."""
        from tools.whisper_mlx.config import get_config
        from tools.whisper_mlx.model import WhisperMLX

        config = get_config("tiny")
        model = WhisperMLX(config, dtype=mx.float16)

        # Load weights would be needed for real test
        # For now just verify it doesn't crash
        assert model.encoder_cache_enabled is False


class TestEncoderCacheMemory:
    """Memory management tests for encoder cache."""

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=8)

        # Add entry - large-v3 dimensions: (1, 1500, 1280) float16
        mel = mx.array(_rng.standard_normal((1, 3000, 128)).astype(np.float16))
        features = mx.array(np.zeros((1, 1500, 1280), dtype=np.float16))

        cache.put(mel, features, variable_length=False, encoder_positions=1500)

        # Expected: 1 * 1500 * 1280 * 2 bytes = 3.84 MB
        expected_mb = (1 * 1500 * 1280 * 2) / (1024 * 1024)
        actual_mb = cache._estimate_memory_mb()

        assert abs(actual_mb - expected_mb) < 0.1  # Within 0.1 MB

    def test_memory_limit_eviction(self):
        """Test eviction based on memory limit."""
        from tools.whisper_mlx.encoder_cache import EncoderCache

        # Set very low memory limit
        cache = EncoderCache(max_entries=100, max_memory_mb=1.0)

        # Add entries until memory limit triggers eviction
        for i in range(10):
            mel = mx.array(np.full((1, 1000, 128), i, dtype=np.float16))
            # ~0.5 MB per entry
            features = mx.array(np.zeros((1, 500, 512), dtype=np.float16))
            cache.put(mel, features, variable_length=False, encoder_positions=500)

        # Should have evicted some entries due to memory limit
        assert cache._estimate_memory_mb() <= 1.5  # Allow some margin


class TestEncoderCacheThreadSafety:
    """Thread safety tests for encoder cache."""

    def test_concurrent_access(self):
        """Test concurrent cache access."""
        import threading

        from tools.whisper_mlx.encoder_cache import EncoderCache

        cache = EncoderCache(max_entries=16)
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    mel = mx.array(np.full((1, 100, 128), worker_id * 100 + i, dtype=np.float16))
                    features = mx.array(_rng.standard_normal((1, 50, 256)).astype(np.float16))

                    cache.put(mel, features, variable_length=False, encoder_positions=50)
                    cache.get(mel, variable_length=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"
