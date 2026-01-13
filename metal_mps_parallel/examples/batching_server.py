#!/usr/bin/env python3
"""
Dynamic Batching Inference Server for Apple Silicon MPS

This module provides a BatchingInferenceServer that collects inference requests
from multiple clients and batches them together for efficient GPU processing.

Key features:
- Thread-safe request aggregation
- Configurable batch size and timeout
- Safe synchronization (uses .cpu() instead of torch.mps.synchronize())
- Graceful shutdown handling

Usage:
    from batching_server import BatchingInferenceServer

    model = YourModel().to('mps').eval()
    server = BatchingInferenceServer(model, max_batch=32, timeout_ms=10)
    server.start()

    # From multiple client threads:
    result = server.infer(input_tensor)

    server.stop()

See EFFICIENCY_ROADMAP.md for background on why batching is preferred over threading.
"""

import torch
import torch.nn as nn
import queue
import threading
import time
from typing import Optional, List, Tuple, Any
from concurrent.futures import Future


class BatchingInferenceServer:
    """
    A server that batches inference requests from multiple clients.

    Instead of each client thread running inference separately (which causes
    contention on the GPU command queue), this server collects requests and
    processes them in batches, achieving higher throughput.

    Thread safety: All public methods are thread-safe.
    """

    def __init__(
        self,
        model: nn.Module,
        max_batch: int = 32,
        timeout_ms: float = 10.0,
        device: str = 'mps'
    ):
        """
        Initialize the batching server.

        Args:
            model: PyTorch model to use for inference (should be on device already)
            max_batch: Maximum batch size before processing
            timeout_ms: Maximum time to wait for batch to fill (milliseconds)
            device: Device to run inference on ('mps', 'cuda', 'cpu')
        """
        self.model = model
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000.0  # Convert to seconds
        self.device = device

        # Request queue: (input_tensor, result_future)
        self._request_queue: queue.Queue = queue.Queue()

        # Control
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Stats
        self._batches_processed = 0
        self._samples_processed = 0
        self._lock = threading.Lock()

    def start(self):
        """Start the batch processing worker thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._batch_worker,
            name="BatchingServer-Worker",
            daemon=True
        )
        self._worker_thread.start()

    def stop(self, timeout: float = 5.0):
        """
        Stop the server gracefully.

        Args:
            timeout: Maximum time to wait for worker thread to finish
        """
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Submit a single inference request and wait for result.

        This is the client API. Multiple threads can call this concurrently.
        The server will batch requests together for efficient processing.

        Args:
            input_tensor: Input tensor (will be batched with others)

        Returns:
            Output tensor from model inference

        Raises:
            RuntimeError: If server is not running
        """
        if not self._running:
            raise RuntimeError("Server is not running. Call start() first.")

        future: Future = Future()
        self._request_queue.put((input_tensor, future))
        return future.result()

    def infer_async(self, input_tensor: torch.Tensor) -> Future:
        """
        Submit a single inference request asynchronously.

        Returns immediately with a Future that will contain the result.

        Args:
            input_tensor: Input tensor (will be batched with others)

        Returns:
            Future that will contain the output tensor

        Raises:
            RuntimeError: If server is not running
        """
        if not self._running:
            raise RuntimeError("Server is not running. Call start() first.")

        future: Future = Future()
        self._request_queue.put((input_tensor, future))
        return future

    def _batch_worker(self):
        """
        Internal worker that collects and processes batches.

        This runs in a dedicated thread and:
        1. Collects requests until batch is full or timeout
        2. Batches inputs together
        3. Runs inference
        4. Distributes results to waiting clients
        """
        while self._running:
            batch_inputs: List[torch.Tensor] = []
            futures: List[Future] = []

            # Collect requests up to max_batch or until timeout
            deadline = time.time() + self.timeout

            while len(batch_inputs) < self.max_batch:
                remaining = max(0.0, deadline - time.time())
                try:
                    input_tensor, future = self._request_queue.get(timeout=remaining)
                    batch_inputs.append(input_tensor)
                    futures.append(future)
                except queue.Empty:
                    break

            if not batch_inputs:
                # No requests received, continue waiting
                continue

            # Process the batch
            try:
                # MPS tensors are created asynchronously. Sync to ensure data is ready.
                # Without this, worker thread may see uninitialized memory (zeros).
                if self.device == 'mps' and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()

                # Stack inputs into a batch
                # Handle case where inputs have different leading dimensions
                batched_input = torch.cat(batch_inputs, dim=0)

                # Run inference
                with torch.no_grad():
                    batched_output = self.model(batched_input)

                # Safe synchronization: use .cpu() instead of torch.mps.synchronize()
                # This forces GPU completion without using MPS Events
                _ = batched_output.sum().cpu()

                # Distribute results to clients
                # Each client gets their slice of the output (cloned to avoid view issues)
                idx = 0
                for i, future in enumerate(futures):
                    input_size = batch_inputs[i].shape[0]
                    result = batched_output[idx:idx + input_size].clone()
                    future.set_result(result)
                    idx += input_size

                # MPS clone is async - sync to ensure clones complete before clients access them
                if self.device == 'mps' and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()

                # Update stats
                with self._lock:
                    self._batches_processed += 1
                    self._samples_processed += len(batch_inputs)

            except Exception as e:
                # Propagate error to all waiting clients
                for future in futures:
                    future.set_exception(e)

    @property
    def stats(self) -> dict:
        """Get server statistics."""
        with self._lock:
            return {
                'batches_processed': self._batches_processed,
                'samples_processed': self._samples_processed,
                'avg_batch_size': (
                    self._samples_processed / self._batches_processed
                    if self._batches_processed > 0 else 0.0
                )
            }

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop()


class AdaptiveBatchingServer(BatchingInferenceServer):
    """
    BatchingInferenceServer with adaptive batch size based on queue depth.

    Automatically adjusts batch size to balance latency and throughput:
    - High load (deep queue): Increase batch size for throughput
    - Low load (shallow queue): Decrease batch size for latency
    """

    def __init__(
        self,
        model: nn.Module,
        min_batch: int = 1,
        max_batch: int = 64,
        target_latency_ms: float = 50.0,
        timeout_ms: float = 10.0,
        device: str = 'mps'
    ):
        """
        Initialize the adaptive batching server.

        Args:
            model: PyTorch model for inference
            min_batch: Minimum batch size
            max_batch: Maximum batch size
            target_latency_ms: Target latency for adjustment (ms)
            timeout_ms: Maximum wait time for batch collection
            device: Device for inference
        """
        super().__init__(model, max_batch, timeout_ms, device)
        self.min_batch = min_batch
        self.target_latency = target_latency_ms / 1000.0
        self._current_batch = min_batch
        self._avg_inference_time = 0.0
        self._inference_time_alpha = 0.1  # Exponential moving average

    def _batch_worker(self):
        """Worker with adaptive batch size adjustment."""
        while self._running:
            batch_inputs: List[torch.Tensor] = []
            futures: List[Future] = []

            # Get current adaptive batch size
            current_batch_size = self._get_adaptive_batch_size()

            # Collect requests
            deadline = time.time() + self.timeout
            while len(batch_inputs) < current_batch_size:
                remaining = max(0.0, deadline - time.time())
                try:
                    input_tensor, future = self._request_queue.get(timeout=remaining)
                    batch_inputs.append(input_tensor)
                    futures.append(future)
                except queue.Empty:
                    break

            if not batch_inputs:
                continue

            try:
                # MPS tensors are created asynchronously. Sync to ensure data is ready.
                if self.device == 'mps' and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()

                batched_input = torch.cat(batch_inputs, dim=0)

                # Time inference
                start_time = time.time()
                with torch.no_grad():
                    batched_output = self.model(batched_input)
                _ = batched_output.sum().cpu()
                elapsed = time.time() - start_time

                # Update inference time estimate
                self._avg_inference_time = (
                    self._inference_time_alpha * elapsed +
                    (1 - self._inference_time_alpha) * self._avg_inference_time
                )

                # Distribute results (cloned to avoid view issues)
                idx = 0
                for i, future in enumerate(futures):
                    input_size = batch_inputs[i].shape[0]
                    result = batched_output[idx:idx + input_size].clone()
                    future.set_result(result)
                    idx += input_size

                # MPS clone is async - sync to ensure clones complete before clients access them
                if self.device == 'mps' and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()

                with self._lock:
                    self._batches_processed += 1
                    self._samples_processed += len(batch_inputs)

            except Exception as e:
                for future in futures:
                    future.set_exception(e)

    def _get_adaptive_batch_size(self) -> int:
        """Calculate adaptive batch size based on current conditions."""
        queue_depth = self._request_queue.qsize()

        if queue_depth > self._current_batch * 2:
            # Queue backing up - increase batch for throughput
            self._current_batch = min(self._current_batch * 2, self.max_batch)
        elif queue_depth < self._current_batch // 2:
            if self._avg_inference_time < self.target_latency:
                # Low load and fast inference - decrease batch for latency
                self._current_batch = max(self._current_batch // 2, self.min_batch)

        return self._current_batch


def demo():
    """Demonstrate the BatchingInferenceServer."""
    import torch.nn as nn

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = 'cpu'
    else:
        device = 'mps'

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256)
            )

        def forward(self, x):
            return self.layers(x)

    model = SimpleModel().to(device).eval()

    # Create server
    server = BatchingInferenceServer(model, max_batch=16, timeout_ms=5.0, device=device)

    print("Starting batching server demo...")
    print(f"Device: {device}")
    print(f"Max batch: 16, Timeout: 5ms")

    # Simulate multiple client threads
    num_clients = 8
    requests_per_client = 50
    results = []

    def client_worker(client_id: int):
        """Simulate a client sending requests."""
        for i in range(requests_per_client):
            x = torch.randn(1, 256, device=device)
            result = server.infer(x)
            results.append((client_id, i, result.shape))

    with server:
        start = time.time()

        # Launch client threads
        threads = []
        for cid in range(num_clients):
            t = threading.Thread(target=client_worker, args=(cid,))
            threads.append(t)
            t.start()

        # Wait for all clients
        for t in threads:
            t.join()

        elapsed = time.time() - start

    total_requests = num_clients * requests_per_client
    throughput = total_requests / elapsed

    print(f"\nResults:")
    print(f"  Total requests: {total_requests}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.1f} requests/s")
    print(f"  Server stats: {server.stats}")


if __name__ == "__main__":
    demo()
