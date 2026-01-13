#!/usr/bin/env python3
"""
TensorPool - Memory pooling for MPS parallel inference.

P3 optimization from EFFICIENCY_ROADMAP.md.
Reduces allocation overhead by reusing pre-allocated tensors.

Usage:
    from tensor_pool import TensorPool, PooledTensorContext

    # Create pool for input tensors
    input_pool = TensorPool(shape=(32, 256), dtype=torch.float32, device='mps', pool_size=8)

    # Acquire and release manually
    idx, tensor = input_pool.acquire()
    tensor.copy_(input_data)  # Fill with data
    result = model(tensor)
    input_pool.release(idx)

    # Or use context manager
    with input_pool.context() as tensor:
        tensor.copy_(input_data)
        result = model(tensor)
"""

import threading
import torch
from typing import Optional, Tuple, List, Any
from contextlib import contextmanager


class TensorPool:
    """
    Thread-safe pool of pre-allocated tensors for reduced allocation overhead.

    Pre-allocates tensors on construction to avoid repeated GPU memory allocation
    during inference. When pool is exhausted, falls back to dynamic allocation.

    Attributes:
        shape: Shape of tensors in pool
        dtype: Data type of tensors
        device: Device (cpu/mps/cuda)
        pool_size: Number of pre-allocated tensors
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str = 'mps',
        pool_size: int = 8
    ):
        """
        Initialize tensor pool.

        Args:
            shape: Shape of tensors to pool
            dtype: Data type (default: float32)
            device: Device to allocate on (default: mps)
            pool_size: Number of tensors to pre-allocate (default: 8)
        """
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.pool_size = pool_size

        # Pre-allocate tensors
        self._pool: List[torch.Tensor] = [
            torch.empty(shape, dtype=dtype, device=device)
            for _ in range(pool_size)
        ]

        # Track available slots (by index)
        self._available: List[int] = list(range(pool_size))
        self._lock = threading.Lock()

        # Statistics
        self._acquires = 0
        self._releases = 0
        self._fallback_allocs = 0

    def acquire(self) -> Tuple[int, torch.Tensor]:
        """
        Acquire a tensor from the pool.

        Returns:
            Tuple of (index, tensor) where index >= 0 means pooled,
            index == -1 means fallback allocation.

        Thread-safe.
        """
        with self._lock:
            self._acquires += 1
            if self._available:
                idx = self._available.pop()
                return idx, self._pool[idx]

            # Pool exhausted - fallback to dynamic allocation
            self._fallback_allocs += 1
            return -1, torch.empty(self.shape, dtype=self.dtype, device=self.device)

    def release(self, idx: int) -> None:
        """
        Release a tensor back to the pool.

        Args:
            idx: Index returned by acquire(). If -1, tensor is garbage collected.

        Thread-safe.
        """
        if idx >= 0:
            with self._lock:
                self._releases += 1
                if idx not in self._available:
                    self._available.append(idx)

    @contextmanager
    def context(self):
        """
        Context manager for automatic acquire/release.

        Usage:
            with pool.context() as tensor:
                tensor.copy_(data)
                result = model(tensor)
        """
        idx, tensor = self.acquire()
        try:
            yield tensor
        finally:
            self.release(idx)

    @property
    def available_count(self) -> int:
        """Number of tensors currently available in pool."""
        with self._lock:
            return len(self._available)

    @property
    def stats(self) -> dict:
        """Pool statistics."""
        with self._lock:
            return {
                'pool_size': self.pool_size,
                'available': len(self._available),
                'acquires': self._acquires,
                'releases': self._releases,
                'fallback_allocs': self._fallback_allocs,
                'hit_rate': (self._acquires - self._fallback_allocs) / max(1, self._acquires),
            }


class MultiShapeTensorPool:
    """
    Pool that handles multiple tensor shapes.

    Useful when inference may have variable input sizes.
    Creates separate pools per shape.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: str = 'mps',
        pool_size_per_shape: int = 4
    ):
        """
        Initialize multi-shape pool.

        Args:
            dtype: Data type for all pools
            device: Device for all pools
            pool_size_per_shape: Pool size for each shape encountered
        """
        self.dtype = dtype
        self.device = device
        self.pool_size_per_shape = pool_size_per_shape

        self._pools: dict[Tuple[int, ...], TensorPool] = {}
        self._lock = threading.Lock()

    def _get_pool(self, shape: Tuple[int, ...]) -> TensorPool:
        """Get or create pool for shape."""
        with self._lock:
            if shape not in self._pools:
                self._pools[shape] = TensorPool(
                    shape=shape,
                    dtype=self.dtype,
                    device=self.device,
                    pool_size=self.pool_size_per_shape
                )
            return self._pools[shape]

    def acquire(self, shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int, torch.Tensor]:
        """
        Acquire tensor of given shape.

        Returns:
            Tuple of (shape, index, tensor)
        """
        pool = self._get_pool(shape)
        idx, tensor = pool.acquire()
        return shape, idx, tensor

    def release(self, shape: Tuple[int, ...], idx: int) -> None:
        """Release tensor back to appropriate pool."""
        if shape in self._pools:
            self._pools[shape].release(idx)

    @contextmanager
    def context(self, shape: Tuple[int, ...]):
        """Context manager for shape-specific acquire/release."""
        shape_key, idx, tensor = self.acquire(shape)
        try:
            yield tensor
        finally:
            self.release(shape_key, idx)

    @property
    def stats(self) -> dict:
        """Statistics for all pools."""
        with self._lock:
            return {
                'num_shapes': len(self._pools),
                'shapes': list(self._pools.keys()),
                'per_pool': {
                    str(shape): pool.stats
                    for shape, pool in self._pools.items()
                }
            }


class PooledInferenceContext:
    """
    High-level context for pooled inference with input and output tensors.

    Example:
        ctx = PooledInferenceContext(
            input_shape=(32, 256),
            output_shape=(32, 128),
            device='mps'
        )

        def worker():
            with ctx.inference() as (input_tensor, output_tensor):
                input_tensor.copy_(data)
                output_tensor.copy_(model(input_tensor))
                return output_tensor.clone()
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.float32,
        device: str = 'mps',
        pool_size: int = 8
    ):
        """
        Initialize pooled inference context.

        Args:
            input_shape: Shape of input tensors
            output_shape: Shape of output tensors (optional - for pre-allocated outputs)
            dtype: Data type
            device: Device
            pool_size: Pool size for each type
        """
        self.input_pool = TensorPool(input_shape, dtype, device, pool_size)
        self.output_pool = (
            TensorPool(output_shape, dtype, device, pool_size)
            if output_shape else None
        )

    @contextmanager
    def inference(self):
        """
        Context manager providing (input_tensor, output_tensor) or just input_tensor.
        """
        input_idx, input_tensor = self.input_pool.acquire()
        output_idx, output_tensor = (-1, None)

        if self.output_pool:
            output_idx, output_tensor = self.output_pool.acquire()

        try:
            if output_tensor is not None:
                yield input_tensor, output_tensor
            else:
                yield input_tensor
        finally:
            self.input_pool.release(input_idx)
            if self.output_pool:
                self.output_pool.release(output_idx)

    @property
    def stats(self) -> dict:
        """Combined statistics."""
        return {
            'input_pool': self.input_pool.stats,
            'output_pool': self.output_pool.stats if self.output_pool else None
        }


# Convenience function for common use case
def create_inference_pool(
    batch_size: int,
    input_dim: int,
    output_dim: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: str = 'mps',
    pool_size: int = 8
) -> PooledInferenceContext:
    """
    Create a pooled inference context for standard 2D tensors.

    Args:
        batch_size: Batch size
        input_dim: Input feature dimension
        output_dim: Output feature dimension (optional)
        dtype: Data type
        device: Device
        pool_size: Number of tensors per pool

    Returns:
        PooledInferenceContext configured for (batch_size, input_dim) inputs
    """
    return PooledInferenceContext(
        input_shape=(batch_size, input_dim),
        output_shape=(batch_size, output_dim) if output_dim else None,
        dtype=dtype,
        device=device,
        pool_size=pool_size
    )
