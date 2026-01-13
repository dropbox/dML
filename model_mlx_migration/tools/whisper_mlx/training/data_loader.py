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
Optimized data loading for Whisper MLX training.

Features:
    - Mel spectrogram caching (pre-compute once, load fast)
    - Length-sorted batching (reduce padding waste)
    - Prefetching (overlap I/O with GPU compute)
    - Memory-efficient variable-length batching
"""

import hashlib
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioSample:
    """Single audio sample for training."""
    audio_path: str
    transcript: str
    language: str = "en"
    duration: float = 0.0
    mel_frames: int = 0  # Pre-computed for length sorting


class MelCache:
    """
    Cache for pre-computed mel spectrograms.

    Provides 2-3x speedup by avoiding repeated mel computation.
    Uses compressed numpy files for efficient storage.

    Directory structure:
        cache_dir/
            ab/
                abc123def456...npz  # hash-based filenames
    """

    def __init__(
        self,
        cache_dir: str,
        n_mels: int = 128,
        use_compression: bool = True,
    ):
        """
        Initialize mel cache.

        Args:
            cache_dir: Directory for cached mel files
            n_mels: Number of mel filterbanks (must match training)
            use_compression: Use compressed numpy format
        """
        self.cache_dir = Path(cache_dir)
        self.n_mels = n_mels
        self.use_compression = use_compression
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.hits = 0
        self.misses = 0

    def _get_cache_path(self, audio_path: str) -> Path:
        """Get cache file path for an audio file."""
        # Use SHA256 hash of path for unique filename
        h = hashlib.sha256(audio_path.encode()).hexdigest()
        subdir = self.cache_dir / h[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{h}.npz"

    def has(self, audio_path: str) -> bool:
        """Check if mel is cached."""
        return self._get_cache_path(audio_path).exists()

    def load(self, audio_path: str) -> np.ndarray | None:
        """Load cached mel spectrogram."""
        cache_path = self._get_cache_path(audio_path)
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                self.hits += 1
                return data["mel"].astype(np.float32)
            except Exception:
                # Cache corrupted, return None to recompute
                pass
        self.misses += 1
        return None

    def save(self, audio_path: str, mel: np.ndarray):
        """Save mel spectrogram to cache."""
        cache_path = self._get_cache_path(audio_path)
        # Save as float16 to halve storage
        if self.use_compression:
            np.savez_compressed(cache_path, mel=mel.astype(np.float16))
        else:
            np.savez(cache_path, mel=mel.astype(np.float16))

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
        }

    def clear_stats(self):
        """Reset statistics."""
        self.hits = 0
        self.misses = 0


class BatchPrefetcher:
    """
    Prefetch batches in background threads.

    Overlaps I/O with GPU computation for better throughput.
    """

    def __init__(
        self,
        prepare_fn: Callable,
        num_workers: int = 2,
        prefetch_count: int = 2,
    ):
        """
        Initialize prefetcher.

        Args:
            prepare_fn: Function to prepare a batch of samples
            num_workers: Number of background worker threads
            prefetch_count: Number of batches to prefetch
        """
        self.prepare_fn = prepare_fn
        self.num_workers = num_workers
        self.prefetch_count = prefetch_count
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def iterate(self, batches: list[list[AudioSample]]) -> Iterator:
        """
        Iterate over batches with prefetching.

        Args:
            batches: List of batches, each batch is a list of samples

        Yields:
            Prepared batch data
        """
        futures = []

        # Submit initial prefetch jobs
        for i, batch in enumerate(batches[:self.prefetch_count]):
            future = self.executor.submit(self.prepare_fn, batch)
            futures.append((i, future))

        next_submit = self.prefetch_count

        # Yield completed batches and submit new ones
        while futures:
            idx, future = futures.pop(0)
            yield future.result()

            # Submit next batch if available
            if next_submit < len(batches):
                future = self.executor.submit(self.prepare_fn, batches[next_submit])
                futures.append((next_submit, future))
                next_submit += 1


class OptimizedDataLoader:
    """
    Optimized data loading with caching, prefetching, and length sorting.

    Combines all optimizations for maximum training throughput:
    1. Mel caching: Avoid repeated audio processing
    2. Length sorting: Reduce padding waste
    3. Prefetching: Overlap I/O with compute
    """

    def __init__(
        self,
        mel_cache: MelCache | None = None,
        batch_size: int = 16,
        max_audio_len: float = 30.0,
        prefetch_workers: int = 2,
        sort_by_length: bool = True,
        bucket_boundaries: list[float] | None = None,
    ):
        """
        Initialize data loader.

        Args:
            mel_cache: Optional mel spectrogram cache
            batch_size: Number of samples per batch
            max_audio_len: Maximum audio length in seconds
            prefetch_workers: Number of prefetch threads
            sort_by_length: Sort samples by length before batching
            bucket_boundaries: Duration boundaries for bucket batching
        """
        self.mel_cache = mel_cache
        self.batch_size = batch_size
        self.max_audio_len = max_audio_len
        self.prefetch_workers = prefetch_workers
        self.sort_by_length = sort_by_length
        self.bucket_boundaries = bucket_boundaries or [5.0, 10.0, 15.0, 20.0, 25.0]

    def create_batches(
        self,
        samples: list[AudioSample],
        shuffle: bool = True,
    ) -> list[list[AudioSample]]:
        """
        Create batches from samples with optional length sorting.

        Args:
            samples: List of audio samples
            shuffle: Shuffle samples before batching

        Returns:
            List of batches
        """
        if shuffle:
            samples = list(samples)
            rng = np.random.default_rng()
            rng.shuffle(samples)

        if self.sort_by_length:
            # Sort by duration, then create batches
            # This reduces padding significantly
            samples = sorted(samples, key=lambda s: s.duration)

        # Create batches
        batches = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            batches.append(batch)

        return batches

    def iterate(
        self,
        samples: list[AudioSample],
        prepare_fn: Callable,
        shuffle: bool = True,
    ) -> Iterator:
        """
        Iterate over samples with all optimizations.

        Args:
            samples: List of audio samples
            prepare_fn: Function to prepare a batch
            shuffle: Shuffle before batching

        Yields:
            Prepared batch data
        """
        batches = self.create_batches(samples, shuffle=shuffle)

        if self.prefetch_workers > 0:
            prefetcher = BatchPrefetcher(
                prepare_fn=prepare_fn,
                num_workers=self.prefetch_workers,
            )
            yield from prefetcher.iterate(batches)
        else:
            for batch in batches:
                yield prepare_fn(batch)


def cache_mels_for_dataset(
    samples: list[AudioSample],
    cache_dir: str,
    compute_mel_fn: Callable,
    n_mels: int = 128,
    num_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MelCache:
    """
    Pre-compute and cache mel spectrograms for a dataset.

    Args:
        samples: List of audio samples
        cache_dir: Directory for cache files
        compute_mel_fn: Function to compute mel from audio path
        n_mels: Number of mel filterbanks
        num_workers: Number of parallel workers
        progress_callback: Optional callback(current, total)

    Returns:
        MelCache instance with all mels cached
    """
    cache = MelCache(cache_dir, n_mels=n_mels)

    # Find samples that need caching
    to_cache = [s for s in samples if not cache.has(s.audio_path)]

    print(f"Caching {len(to_cache)} mel spectrograms ({len(samples) - len(to_cache)} already cached)")

    def cache_one(sample: AudioSample) -> bool:
        try:
            mel = compute_mel_fn(sample.audio_path)
            cache.save(sample.audio_path, mel)
            return True
        except Exception as e:
            print(f"Error caching {sample.audio_path}: {e}")
            return False

    # Cache in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(cache_one, s) for s in to_cache]
        for i, future in enumerate(futures):
            future.result()
            if progress_callback:
                progress_callback(i + 1, len(to_cache))

    return cache
