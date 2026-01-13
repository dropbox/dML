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
EncoderCache - Cache for WhisperMLX encoder outputs (OPT-W4).

Provides 2x speedup for repeated queries on the same audio by caching
encoder outputs and avoiding redundant computation.

Use cases:
- Multiple transcription attempts with different parameters
- Language detection followed by transcription (same audio processed twice)
- Real-time streaming with overlapping chunks
- A/B testing different decoding strategies on same audio

Usage:
    model = WhisperMLX.from_pretrained("large-v3")
    model.enable_encoder_cache(max_entries=16)  # ~2GB memory for large-v3

    # First call - encodes audio
    result1 = model.transcribe("audio.wav", language="en")

    # Second call - uses cached encoder output (2x faster)
    result2 = model.transcribe("audio.wav", language="ja")  # Translation
"""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx


@dataclass
class CacheEntry:
    """Single cache entry with encoder output and metadata."""
    audio_features: mx.array  # Encoded audio features
    variable_length: bool  # Whether variable-length encoding was used
    encoder_positions: int  # Number of encoder positions (for variable-length)
    audio_duration: float | None  # Actual audio duration (for variable-length)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class EncoderCache:
    """
    LRU cache for WhisperMLX encoder outputs.

    Features:
    - Audio hash-based lookup (content-addressable)
    - LRU eviction policy
    - Thread-safe operations
    - Memory tracking and limits
    - Statistics for monitoring

    Memory estimation (per entry):
    - large-v3: 1500 * 1280 * 2 bytes (float16) = ~3.8MB per 30s audio
    - Default 16 entries = ~61MB max
    """

    def __init__(
        self,
        max_entries: int = 16,
        max_memory_mb: float | None = None,
    ):
        """
        Initialize encoder cache.

        Args:
            max_entries: Maximum number of cached entries (LRU eviction)
            max_memory_mb: Optional memory limit in MB (evicts when exceeded)
        """
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb

        # OrderedDict for LRU - most recently used at end
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _compute_audio_hash(
        self,
        mel: mx.array,
        variable_length: bool,
    ) -> str:
        """
        Compute hash for mel spectrogram.

        Uses SHA-256 for collision resistance. The hash includes:
        - Mel spectrogram data
        - Variable-length flag (same audio may have different encodings)

        Args:
            mel: Mel spectrogram array
            variable_length: Whether variable-length encoding is used

        Returns:
            Hex digest hash string
        """
        # Convert to numpy for hashing (MLX arrays don't have tobytes directly)
        import numpy as np
        mel_np = np.array(mel)

        # Create hash including variable_length flag
        hasher = hashlib.sha256()
        hasher.update(mel_np.tobytes())
        hasher.update(bytes([int(variable_length)]))

        return hasher.hexdigest()[:16]  # 16 chars is enough for uniqueness

    def get(
        self,
        mel: mx.array,
        variable_length: bool,
    ) -> CacheEntry | None:
        """
        Look up cached encoder output.

        Args:
            mel: Mel spectrogram to look up
            variable_length: Whether variable-length encoding is used

        Returns:
            CacheEntry if found, None otherwise
        """
        cache_key = self._compute_audio_hash(mel, variable_length)

        with self._lock:
            if cache_key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                entry = self._cache[cache_key]
                entry.update_access()
                self._hits += 1
                return entry
            self._misses += 1
            return None

    def put(
        self,
        mel: mx.array,
        audio_features: mx.array,
        variable_length: bool,
        encoder_positions: int,
        audio_duration: float | None = None,
    ) -> str:
        """
        Store encoder output in cache.

        Args:
            mel: Original mel spectrogram (for hashing)
            audio_features: Encoded audio features
            variable_length: Whether variable-length encoding was used
            encoder_positions: Number of encoder positions
            audio_duration: Actual audio duration (for variable-length)

        Returns:
            Cache key (hash)
        """
        cache_key = self._compute_audio_hash(mel, variable_length)

        entry = CacheEntry(
            audio_features=audio_features,
            variable_length=variable_length,
            encoder_positions=encoder_positions,
            audio_duration=audio_duration,
        )

        with self._lock:
            # Remove if already exists (to refresh position)
            if cache_key in self._cache:
                del self._cache[cache_key]

            # Add to end (most recently used)
            self._cache[cache_key] = entry

            # Evict if over capacity
            self._evict_if_needed()

        return cache_key

    def _evict_if_needed(self):
        """Evict oldest entries if over capacity (must hold lock)."""
        # Entry count limit
        while len(self._cache) > self.max_entries:
            # Remove oldest (first item)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1

        # Memory limit (optional)
        if self.max_memory_mb is not None:
            while len(self._cache) > 0:
                current_mb = self._estimate_memory_mb()
                if current_mb <= self.max_memory_mb:
                    break
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1

    def _estimate_memory_mb(self) -> float:
        """Estimate current memory usage in MB."""
        total_bytes = 0
        for entry in self._cache.values():
            # MLX array size estimation
            features = entry.audio_features
            # shape[0] = batch, shape[1] = seq_len, shape[2] = hidden_dim
            if len(features.shape) == 3:
                num_elements = features.shape[0] * features.shape[1] * features.shape[2]
            else:
                num_elements = 1
                for dim in features.shape:
                    num_elements *= dim

            # float16 = 2 bytes per element
            bytes_per_element = 2 if features.dtype == mx.float16 else 4
            total_bytes += num_elements * bytes_per_element

        return total_bytes / (1024 * 1024)

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def invalidate(self, mel: mx.array, variable_length: bool) -> bool:
        """
        Remove specific entry from cache.

        Args:
            mel: Mel spectrogram to invalidate
            variable_length: Whether variable-length encoding

        Returns:
            True if entry was found and removed
        """
        cache_key = self._compute_audio_hash(mel, variable_length)

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
                "evictions": self._evictions,
                "memory_mb": self._estimate_memory_mb(),
                "max_memory_mb": self.max_memory_mb,
            }

    def __len__(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    def __contains__(self, key: tuple[mx.array, bool]) -> bool:
        """Check if mel/variable_length combination is cached."""
        mel, variable_length = key
        cache_key = self._compute_audio_hash(mel, variable_length)
        return cache_key in self._cache


class TrainingEncoderCache:
    """
    Disk-based encoder cache for CTC training.

    Since the Whisper encoder is frozen during CTC training, we can pre-compute
    and cache encoder outputs to eliminate the 67% forward pass bottleneck.

    This cache:
    - Uses audio path as key (stable across restarts)
    - Stores to disk with compression
    - Supports incremental building (cache-on-first-access)
    - Tracks actual frame counts for variable-length sequences

    Memory estimation:
    - Encoder output: 1500 frames x 1280 dim x 2 bytes (fp16) = 3.84MB per sample
    - With compression: ~1.5MB per sample
    - 258K samples: ~400GB disk space

    For smaller storage, consider:
    - Storing only up to actual_frames (not padded 1500)
    - Using lower precision (int8 quantization)
    """

    def __init__(
        self,
        cache_dir: str,
        use_compression: bool = True,
    ):
        """
        Initialize training encoder cache.

        Args:
            cache_dir: Directory to store cached encoder outputs
            use_compression: Use numpy's compressed format (saves ~60% space)
        """
        from pathlib import Path


        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_compression = use_compression

        # Stats
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, audio_path: str) -> str:
        """Generate cache key from audio path using SHA256 hash."""
        # Use full path to avoid collisions
        return hashlib.sha256(audio_path.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> "Path":
        """Get file path for cache key."""
        # Use 2-level directory structure to avoid too many files in one dir
        return self.cache_dir / cache_key[:2] / f"{cache_key}.npz"

    def has(self, audio_path: str) -> bool:
        """Check if encoder output is cached for audio path."""
        cache_key = self._get_cache_key(audio_path)
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()

    def load(self, audio_path: str) -> tuple[mx.array, int] | None:
        """
        Load encoder output from cache.

        Returns:
            Tuple of (encoder_output, actual_frames) or None if not cached
        """
        import numpy as np

        cache_key = self._get_cache_key(audio_path)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            self.misses += 1
            return None

        try:
            data = np.load(cache_path)
            encoder_output = mx.array(data["encoder_output"].astype(np.float32))
            actual_frames = int(data["actual_frames"])

            self.hits += 1
            return encoder_output, actual_frames

        except Exception as e:
            print(f"Warning: Failed to load encoder cache for {audio_path}: {e}")
            self.misses += 1
            return None

    def save(
        self,
        audio_path: str,
        encoder_output: mx.array,
        actual_frames: int,
    ) -> bool:
        """
        Save encoder output to cache.

        Args:
            audio_path: Original audio file path (used as key)
            encoder_output: Encoder output array (T, d_model) - single sample, not batched
            actual_frames: Actual number of valid frames (before padding)

        Returns:
            True if saved successfully
        """
        import numpy as np

        cache_key = self._get_cache_key(audio_path)
        cache_path = self._get_cache_path(cache_key)

        try:
            # Ensure directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to numpy and store as float16 to save space
            encoder_np = np.array(encoder_output).astype(np.float16)

            # Save with or without compression
            if self.use_compression:
                np.savez_compressed(
                    cache_path,
                    encoder_output=encoder_np,
                    actual_frames=np.array(actual_frames),
                )
            else:
                np.savez(
                    cache_path,
                    encoder_output=encoder_np,
                    actual_frames=np.array(actual_frames),
                )

            return True

        except Exception as e:
            print(f"Warning: Failed to save encoder cache for {audio_path}: {e}")
            return False

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        # Count cached files
        cached_files = sum(1 for _ in self.cache_dir.rglob("*.npz"))

        # Estimate disk usage
        disk_usage_mb = 0
        for f in self.cache_dir.rglob("*.npz"):
            disk_usage_mb += f.stat().st_size / (1024 * 1024)

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cached_files": cached_files,
            "disk_usage_mb": disk_usage_mb,
        }

    def clear(self):
        """Clear all cached files."""
        import shutil
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                shutil.rmtree(subdir)
