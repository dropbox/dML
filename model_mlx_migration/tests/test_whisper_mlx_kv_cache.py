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
Tests for whisper_mlx kv_cache module.

Tests:
- DynamicKVCache: Standard growing cache
- PreallocatedKVCache: Fixed-size preallocated cache
- KVCacheManager: Unified interface
"""

import mlx.core as mx
import pytest


class TestDynamicKVCache:
    """Tests for DynamicKVCache class."""

    def test_init(self):
        """Test initialization."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)

        assert cache.n_layers == 4
        assert len(cache._cache) == 4
        assert all(c is None for c in cache._cache)

    def test_get_empty(self):
        """Test get returns None for empty cache."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)

        for i in range(4):
            assert cache.get(i) is None

    def test_update_and_get(self):
        """Test update and get operations."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)

        # Create dummy K/V tensors
        self_k = mx.random.normal((1, 5, 64))
        self_v = mx.random.normal((1, 5, 64))
        cross_k = mx.random.normal((1, 20, 64))
        cross_v = mx.random.normal((1, 20, 64))

        # Update layer 0
        cache.update(0, (self_k, self_v), (cross_k, cross_v))

        # Get and verify
        result = cache.get(0)
        assert result is not None
        self_kv, cross_kv = result

        assert mx.allclose(self_kv[0], self_k).item()
        assert mx.allclose(self_kv[1], self_v).item()
        assert mx.allclose(cross_kv[0], cross_k).item()
        assert mx.allclose(cross_kv[1], cross_v).item()

    def test_update_multiple_layers(self):
        """Test updating multiple layers."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)

        for layer_idx in range(4):
            self_k = mx.random.normal((1, 5 + layer_idx, 64))
            self_v = mx.random.normal((1, 5 + layer_idx, 64))
            cross_k = mx.random.normal((1, 20, 64))
            cross_v = mx.random.normal((1, 20, 64))

            cache.update(layer_idx, (self_k, self_v), (cross_k, cross_v))

        # Verify each layer
        for layer_idx in range(4):
            result = cache.get(layer_idx)
            assert result is not None
            self_kv, _ = result
            # Each layer has different self-attention length
            assert self_kv[0].shape[1] == 5 + layer_idx

    def test_reset(self):
        """Test reset clears all caches."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)

        # Populate cache
        for layer_idx in range(4):
            self_kv = (mx.random.normal((1, 5, 64)), mx.random.normal((1, 5, 64)))
            cross_kv = (mx.random.normal((1, 20, 64)), mx.random.normal((1, 20, 64)))
            cache.update(layer_idx, self_kv, cross_kv)

        # Reset
        cache.reset()

        # All should be None
        for i in range(4):
            assert cache.get(i) is None

    def test_position_empty(self):
        """Test position is 0 for empty cache."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)
        assert cache.position == 0

    def test_position_after_update(self):
        """Test position reflects self-attention K length."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)

        # Add 5 tokens worth of cache
        self_k = mx.random.normal((1, 5, 64))
        self_v = mx.random.normal((1, 5, 64))
        cross_k = mx.random.normal((1, 20, 64))
        cross_v = mx.random.normal((1, 20, 64))

        cache.update(0, (self_k, self_v), (cross_k, cross_v))

        assert cache.position == 5

    def test_position_grows_with_updates(self):
        """Test position grows as cache is updated."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=2)

        # Start with 3 tokens
        self_k = mx.random.normal((1, 3, 64))
        self_v = mx.random.normal((1, 3, 64))
        cross_kv = (mx.random.normal((1, 20, 64)), mx.random.normal((1, 20, 64)))

        cache.update(0, (self_k, self_v), cross_kv)
        cache.update(1, (self_k, self_v), cross_kv)

        assert cache.position == 3

        # Add more tokens (simulate growing cache)
        self_k_new = mx.random.normal((1, 6, 64))  # Now 6 tokens
        self_v_new = mx.random.normal((1, 6, 64))

        cache.update(0, (self_k_new, self_v_new), cross_kv)

        assert cache.position == 6


class TestPreallocatedKVCache:
    """Tests for PreallocatedKVCache class."""

    @pytest.fixture
    def cache_config(self):
        """Standard cache configuration."""
        return {
            "max_seq_len": 100,
            "n_layers": 4,
            "n_heads": 8,
            "head_dim": 64,
        }

    def test_init(self, cache_config):
        """Test initialization."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config)

        assert cache.max_seq_len == 100
        assert cache.n_layers == 4
        assert cache.n_heads == 8
        assert cache.head_dim == 64

    def test_self_cache_preallocated(self, cache_config):
        """Test self-attention caches are preallocated."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]

        assert cache._self_k.shape == (4, 100, n_state)
        assert cache._self_v.shape == (4, 100, n_state)

    def test_cross_cache_initially_none(self, cache_config):
        """Test cross-attention cache is initially None."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config)

        assert cache._cross_k is None
        assert cache._cross_v is None

    def test_set_cross_attention(self, cache_config):
        """Test setting cross-attention cache."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]
        encoder_len = 50

        cross_k = mx.random.normal((1, encoder_len, n_state))
        cross_v = mx.random.normal((1, encoder_len, n_state))

        cache.set_cross_attention(0, cross_k, cross_v)

        assert cache._cross_k is not None
        assert cache._cross_v is not None
        assert cache._cross_k.shape == (4, encoder_len, n_state)

    def test_update_self_attention(self, cache_config):
        """Test updating self-attention cache."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]

        # Add token at position 0
        k = mx.random.normal((1, 1, n_state))
        v = mx.random.normal((1, 1, n_state))

        cache.update_self_attention(0, k, v)

        # Verify it's stored
        k_stored, v_stored = cache.get_self_attention(0)
        assert k_stored.shape == (1, 1, n_state)

    def test_advance_position(self, cache_config):
        """Test position advances correctly."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config)

        assert cache.position == 0

        cache.advance()
        assert cache.position == 1

        cache.advance()
        assert cache.position == 2

    def test_get_self_attention_returns_slice(self, cache_config):
        """Test get_self_attention returns correct slice."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]

        # Add multiple tokens
        for i in range(5):
            k = mx.ones((1, 1, n_state)) * (i + 1)
            v = mx.ones((1, 1, n_state)) * (i + 1)
            cache.update_self_attention(0, k, v)
            cache.advance()

        # get_self_attention returns position+1 (6 tokens for position 5)
        k_out, v_out = cache.get_self_attention(0)

        # The implementation returns position+1 entries
        assert k_out.shape == (1, 6, n_state)

    def test_get_cross_attention(self, cache_config):
        """Test get_cross_attention returns correct data."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config, dtype=mx.float32)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]
        encoder_len = 50

        # Set cross attention (match dtype)
        cross_k = mx.random.normal((1, encoder_len, n_state)).astype(mx.float32)
        cross_v = mx.random.normal((1, encoder_len, n_state)).astype(mx.float32)

        cache.set_cross_attention(0, cross_k, cross_v)

        # Get it back
        k_out, v_out = cache.get_cross_attention(0)

        assert k_out.shape == (1, encoder_len, n_state)
        assert mx.allclose(k_out.squeeze(0), cross_k.squeeze(0), atol=1e-5).item()

    def test_reset(self, cache_config):
        """Test reset clears cache state."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]

        # Use the cache
        cache.set_cross_attention(0, mx.random.normal((1, 50, n_state)), mx.random.normal((1, 50, n_state)))
        for _ in range(10):
            cache.update_self_attention(0, mx.random.normal((1, 1, n_state)), mx.random.normal((1, 1, n_state)))
            cache.advance()

        assert cache.position == 10

        # Reset
        cache.reset()

        assert cache.position == 0
        assert cache._cross_k is None
        assert cache._cross_v is None

    def test_dtype_float16(self, cache_config):
        """Test float16 dtype."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config, dtype=mx.float16)

        assert cache._self_k.dtype == mx.float16
        assert cache._self_v.dtype == mx.float16

    def test_dtype_float32(self, cache_config):
        """Test float32 dtype."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(**cache_config, dtype=mx.float32)

        assert cache._self_k.dtype == mx.float32
        assert cache._self_v.dtype == mx.float32


class TestKVCacheManager:
    """Tests for KVCacheManager unified interface."""

    @pytest.fixture
    def manager_config(self):
        """Standard manager configuration."""
        return {
            "n_layers": 4,
            "max_seq_len": 100,
            "n_heads": 8,
            "head_dim": 64,
        }

    def test_init_dynamic(self, manager_config):
        """Test initialization with dynamic cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=False)

        assert manager.preallocate is False

    def test_init_preallocated(self, manager_config):
        """Test initialization with preallocated cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=True)

        assert manager.preallocate is True

    def test_dynamic_get(self, manager_config):
        """Test get operation for dynamic cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=False)

        # Initially empty
        assert manager.get(0) is None

    def test_dynamic_update(self, manager_config):
        """Test update operation for dynamic cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=False)

        self_kv = (mx.random.normal((1, 5, 512)), mx.random.normal((1, 5, 512)))
        cross_kv = (mx.random.normal((1, 20, 512)), mx.random.normal((1, 20, 512)))

        manager.update(0, self_kv, cross_kv)

        result = manager.get(0)
        assert result is not None

    def test_preallocated_get_raises(self, manager_config):
        """Test get raises for preallocated cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=True)

        with pytest.raises(ValueError, match="preallocated cache"):
            manager.get(0)

    def test_preallocated_update_raises(self, manager_config):
        """Test update raises for preallocated cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=True)

        self_kv = (mx.random.normal((1, 5, 512)), mx.random.normal((1, 5, 512)))
        cross_kv = (mx.random.normal((1, 20, 512)), mx.random.normal((1, 20, 512)))

        with pytest.raises(ValueError, match="preallocated cache"):
            manager.update(0, self_kv, cross_kv)

    def test_reset_dynamic(self, manager_config):
        """Test reset for dynamic cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=False)

        self_kv = (mx.random.normal((1, 5, 512)), mx.random.normal((1, 5, 512)))
        cross_kv = (mx.random.normal((1, 20, 512)), mx.random.normal((1, 20, 512)))

        manager.update(0, self_kv, cross_kv)
        manager.reset()

        assert manager.get(0) is None

    def test_reset_preallocated(self, manager_config):
        """Test reset for preallocated cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=True)

        # Simulate some usage
        manager._cache.advance()
        manager._cache.advance()

        assert manager.position == 2

        manager.reset()

        assert manager.position == 0

    def test_position_dynamic(self, manager_config):
        """Test position property for dynamic cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=False)

        assert manager.position == 0

        self_kv = (mx.random.normal((1, 7, 512)), mx.random.normal((1, 7, 512)))
        cross_kv = (mx.random.normal((1, 20, 512)), mx.random.normal((1, 20, 512)))

        manager.update(0, self_kv, cross_kv)

        assert manager.position == 7

    def test_position_preallocated(self, manager_config):
        """Test position property for preallocated cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=True)

        assert manager.position == 0

        manager._cache.advance()
        manager._cache.advance()
        manager._cache.advance()

        assert manager.position == 3


class TestKVCacheEdgeCases:
    """Edge case tests for KV cache classes."""

    def test_dynamic_single_layer(self):
        """Test dynamic cache with single layer."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=1)

        self_kv = (mx.random.normal((1, 5, 64)), mx.random.normal((1, 5, 64)))
        cross_kv = (mx.random.normal((1, 20, 64)), mx.random.normal((1, 20, 64)))

        cache.update(0, self_kv, cross_kv)

        assert cache.get(0) is not None

    def test_preallocated_large_encoder(self):
        """Test preallocated cache with large encoder length."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(
            max_seq_len=448,
            n_layers=4,
            n_heads=8,
            head_dim=64,
        )

        n_state = 8 * 64
        encoder_len = 1500  # 30s audio

        cross_k = mx.random.normal((1, encoder_len, n_state))
        cross_v = mx.random.normal((1, encoder_len, n_state))

        cache.set_cross_attention(0, cross_k, cross_v)

        assert cache._cross_k.shape[1] == encoder_len

    def test_dynamic_different_layer_sizes(self):
        """Test dynamic cache can have different sizes per layer."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)

        # Each layer gets different sized cache
        for layer_idx in range(4):
            seq_len = 5 * (layer_idx + 1)
            self_kv = (mx.random.normal((1, seq_len, 64)), mx.random.normal((1, seq_len, 64)))
            cross_kv = (mx.random.normal((1, 20, 64)), mx.random.normal((1, 20, 64)))
            cache.update(layer_idx, self_kv, cross_kv)

        # Verify each layer's size
        for layer_idx in range(4):
            result = cache.get(layer_idx)
            self_kv, _ = result
            assert self_kv[0].shape[1] == 5 * (layer_idx + 1)

    def test_preallocated_position_boundary(self):
        """Test preallocated cache at position boundary."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(
            max_seq_len=10,
            n_layers=1,
            n_heads=2,
            head_dim=32,
        )

        n_state = 2 * 32

        # Fill up to max
        for _i in range(10):
            k = mx.random.normal((1, 1, n_state))
            v = mx.random.normal((1, 1, n_state))
            cache.update_self_attention(0, k, v)
            cache.advance()

        assert cache.position == 10

        # Get full cache
        k_out, v_out = cache.get_self_attention(0)
        assert k_out.shape == (1, 10, n_state)


class TestKVCacheMemoryEfficiency:
    """Tests for memory efficiency properties."""

    def test_preallocated_no_growth(self):
        """Test preallocated cache doesn't reallocate."""
        from tools.whisper_mlx.kv_cache import PreallocatedKVCache

        cache = PreallocatedKVCache(
            max_seq_len=100,
            n_layers=4,
            n_heads=8,
            head_dim=64,
        )

        # Get initial array ID
        initial_k_id = id(cache._self_k)
        initial_v_id = id(cache._self_v)

        n_state = 8 * 64

        # Add many tokens
        for _i in range(50):
            k = mx.random.normal((1, 1, n_state))
            v = mx.random.normal((1, 1, n_state))
            cache.update_self_attention(0, k, v)
            cache.advance()

        # Arrays should not have been reallocated
        assert id(cache._self_k) == initial_k_id
        assert id(cache._self_v) == initial_v_id

    def test_dynamic_cache_position_tracking(self):
        """Test dynamic cache tracks position from first layer."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache

        cache = DynamicKVCache(n_layers=4)

        # Only update layer 0
        self_kv = (mx.random.normal((1, 10, 64)), mx.random.normal((1, 10, 64)))
        cross_kv = (mx.random.normal((1, 20, 64)), mx.random.normal((1, 20, 64)))
        cache.update(0, self_kv, cross_kv)

        # Position should come from layer 0
        assert cache.position == 10

        # Other layers are still None but position is from layer 0
        assert cache.get(1) is None
        assert cache.get(2) is None
        assert cache.get(3) is None


class TestQuantizeToInt8:
    """Tests for quantize_to_int8 helper function."""

    def test_basic_quantization(self):
        """Test basic quantization and dequantization roundtrip."""
        from tools.whisper_mlx.kv_cache import dequantize_from_int8, quantize_to_int8

        # Create random tensor
        x = mx.random.normal((100, 64))

        # Quantize
        x_int8, scale = quantize_to_int8(x)

        # Verify dtype
        assert x_int8.dtype == mx.int8
        assert scale.dtype in [mx.float16, mx.float32]

        # Dequantize
        x_recovered = dequantize_from_int8(x_int8, scale)

        # Should be close to original (within quantization error)
        max_error = float(mx.max(mx.abs(x - x_recovered)))
        assert max_error < 0.1, f"Max error {max_error} too large"

    def test_quantization_preserves_sign(self):
        """Test that quantization preserves sign correctly."""
        from tools.whisper_mlx.kv_cache import dequantize_from_int8, quantize_to_int8

        # Create tensor with positive and negative values
        x = mx.array([[1.0, -1.0], [0.5, -0.5]])

        x_int8, scale = quantize_to_int8(x)
        x_recovered = dequantize_from_int8(x_int8, scale)

        # Signs should match
        assert mx.all((x_recovered > 0) == (x > 0)).item()

    def test_zero_tensor(self):
        """Test quantization of zero tensor."""
        from tools.whisper_mlx.kv_cache import dequantize_from_int8, quantize_to_int8

        x = mx.zeros((10, 10))
        x_int8, scale = quantize_to_int8(x)

        # Should not crash (division by zero protection)
        x_recovered = dequantize_from_int8(x_int8, scale)
        assert mx.allclose(x_recovered, x, atol=1e-5).item()


class TestQuantizedKVCache:
    """Tests for QuantizedKVCache class (OPT-2.3)."""

    @pytest.fixture
    def cache_config(self):
        """Standard cache configuration."""
        return {
            "max_seq_len": 100,
            "n_layers": 4,
            "n_heads": 8,
            "head_dim": 64,
        }

    def test_init(self, cache_config):
        """Test QuantizedKVCache initialization."""
        from tools.whisper_mlx.kv_cache import QuantizedKVCache

        cache = QuantizedKVCache(**cache_config)

        assert cache.n_layers == 4
        assert cache.n_heads == 8
        assert cache.head_dim == 64
        assert cache.max_seq_len == 100
        assert cache.position == 0

    def test_self_attention_uses_float16(self, cache_config):
        """Test that self-attention cache uses float16 (not INT8)."""
        from tools.whisper_mlx.kv_cache import QuantizedKVCache

        cache = QuantizedKVCache(**cache_config)

        # Self-attention should be float16 for accuracy
        assert cache._self_k.dtype == mx.float16
        assert cache._self_v.dtype == mx.float16

    def test_cross_attention_uses_int8(self, cache_config):
        """Test that cross-attention cache uses INT8."""
        from tools.whisper_mlx.kv_cache import QuantizedKVCache

        cache = QuantizedKVCache(**cache_config)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]
        encoder_len = 50

        # Set cross attention
        cross_k = mx.random.normal((1, encoder_len, n_state))
        cross_v = mx.random.normal((1, encoder_len, n_state))
        cache.set_cross_attention(0, cross_k, cross_v)

        # Cross-attention should be INT8
        assert cache._cross_k_int8.dtype == mx.int8
        assert cache._cross_v_int8.dtype == mx.int8

    def test_cross_attention_accuracy(self, cache_config):
        """Test cross-attention quantization accuracy."""
        from tools.whisper_mlx.kv_cache import QuantizedKVCache

        cache = QuantizedKVCache(**cache_config, dtype=mx.float32)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]
        encoder_len = 50

        # Set cross attention
        cross_k = mx.random.normal((1, encoder_len, n_state)).astype(mx.float32)
        cross_v = mx.random.normal((1, encoder_len, n_state)).astype(mx.float32)
        cache.set_cross_attention(0, cross_k, cross_v)

        # Get it back (with dequantization)
        k_out, v_out = cache.get_cross_attention(0)

        # Should be close to original (within INT8 quantization error)
        max_k_error = float(mx.max(mx.abs(k_out - cross_k)))
        max_v_error = float(mx.max(mx.abs(v_out - cross_v)))

        # INT8 per-tensor quantization error typically < 1% of max value
        assert max_k_error < 0.1, f"K error {max_k_error} too large"
        assert max_v_error < 0.1, f"V error {max_v_error} too large"

    def test_self_attention_operations(self, cache_config):
        """Test self-attention update and get operations."""
        from tools.whisper_mlx.kv_cache import QuantizedKVCache

        cache = QuantizedKVCache(**cache_config, dtype=mx.float32)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]

        # Add tokens
        for i in range(5):
            k = mx.ones((1, 1, n_state)) * (i + 1)
            v = mx.ones((1, 1, n_state)) * (i + 1)
            k = k.astype(mx.float32)
            v = v.astype(mx.float32)
            cache.update_self_attention(0, k, v)
            cache.advance()

        # Get self-attention
        k_out, v_out = cache.get_self_attention(0)

        # Should be exact (no quantization for self-attention)
        assert k_out.shape == (1, 6, n_state)  # position+1 = 6

    def test_reset(self, cache_config):
        """Test reset clears cache state."""
        from tools.whisper_mlx.kv_cache import QuantizedKVCache

        cache = QuantizedKVCache(**cache_config)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]

        # Use the cache
        cache.set_cross_attention(
            0,
            mx.random.normal((1, 50, n_state)),
            mx.random.normal((1, 50, n_state)),
        )
        for _ in range(10):
            cache.update_self_attention(
                0,
                mx.random.normal((1, 1, n_state)),
                mx.random.normal((1, 1, n_state)),
            )
            cache.advance()

        assert cache.position == 10

        # Reset
        cache.reset()

        assert cache.position == 0
        assert cache._cross_k_int8 is None
        assert cache._cross_v_int8 is None

    def test_memory_usage_mb(self, cache_config):
        """Test memory_usage_mb property."""
        from tools.whisper_mlx.kv_cache import QuantizedKVCache

        cache = QuantizedKVCache(**cache_config)
        n_state = cache_config["n_heads"] * cache_config["head_dim"]

        # Before setting cross-attention
        mem_before = cache.memory_usage_mb
        assert mem_before > 0

        # Set cross-attention
        cache.set_cross_attention(
            0,
            mx.random.normal((1, 50, n_state)),
            mx.random.normal((1, 50, n_state)),
        )

        mem_after = cache.memory_usage_mb
        # Memory should increase slightly (INT8 cross-attn is small)
        assert mem_after >= mem_before


class TestKVCacheManagerQuantized:
    """Tests for KVCacheManager with quantized cache."""

    @pytest.fixture
    def manager_config(self):
        """Standard manager configuration."""
        return {
            "n_layers": 4,
            "max_seq_len": 100,
            "n_heads": 8,
            "head_dim": 64,
        }

    def test_init_quantized(self, manager_config):
        """Test initialization with quantized cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager, QuantizedKVCache

        manager = KVCacheManager(**manager_config, preallocate=True, quantize=True)

        assert manager.preallocate is True
        assert manager.quantize is True
        assert isinstance(manager._cache, QuantizedKVCache)

    def test_quantize_without_preallocate(self, manager_config):
        """Test quantize flag without preallocate falls back to dynamic."""
        from tools.whisper_mlx.kv_cache import DynamicKVCache, KVCacheManager

        # quantize=True but preallocate=False should use dynamic
        manager = KVCacheManager(**manager_config, preallocate=False, quantize=True)

        assert isinstance(manager._cache, DynamicKVCache)

    def test_quantized_cache_position(self, manager_config):
        """Test position tracking for quantized cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=True, quantize=True)

        assert manager.position == 0

        manager._cache.advance()
        manager._cache.advance()

        assert manager.position == 2

    def test_quantized_cache_reset(self, manager_config):
        """Test reset for quantized cache."""
        from tools.whisper_mlx.kv_cache import KVCacheManager

        manager = KVCacheManager(**manager_config, preallocate=True, quantize=True)

        # Use the cache
        manager._cache.advance()
        manager._cache.advance()

        manager.reset()

        assert manager.position == 0
