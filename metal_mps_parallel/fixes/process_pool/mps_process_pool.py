#!/usr/bin/env python3
"""
MPS Process Pool - Bypass Apple's Driver Serialization

Created by Andrew Yates

This module provides a process-based parallel inference pool that bypasses
Apple's per-process Metal driver serialization.

KEY INSIGHT: The serialization bug is per-process. Each process has its own
Metal context. By using multiple processes instead of threads, we can achieve
true parallelism.

Usage:
    from mps_process_pool import MPSProcessPool

    pool = MPSProcessPool(num_workers=8)
    results = pool.forward_batch(inputs)
    pool.shutdown()
"""

import multiprocessing as mp
import torch
import torch.nn as nn
import time
import pickle
from typing import List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np


def _worker_loop(worker_id: int, input_queue: mp.Queue, output_queue: mp.Queue,
                 model_factory: Optional[bytes] = None):
    """
    Worker process main loop.
    Each process has its own Metal context - this is the key to bypassing serialization!
    """
    import torch
    import torch.nn as nn

    device = torch.device('mps')

    # Create model in this process's Metal context
    if model_factory:
        factory_fn = pickle.loads(model_factory)
        model = factory_fn().to(device).eval()
    else:
        # Default model for testing
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ).to(device).eval()

    # Signal ready
    output_queue.put({'status': 'ready', 'worker_id': worker_id})

    while True:
        msg = input_queue.get()

        if msg.get('cmd') == 'STOP':
            break

        elif msg.get('cmd') == 'FORWARD':
            try:
                # Get input data
                input_data = msg.get('data')
                if isinstance(input_data, np.ndarray):
                    x = torch.from_numpy(input_data).to(device)
                else:
                    x = torch.randn(32, 512, device=device)

                # Forward pass in this process's Metal context
                with torch.no_grad():
                    y = model(x)
                torch.mps.synchronize()

                # Return result
                output_queue.put({
                    'status': 'done',
                    'worker_id': worker_id,
                    'result': y.cpu().numpy() if msg.get('return_result') else None
                })

            except Exception as e:
                output_queue.put({
                    'status': 'error',
                    'worker_id': worker_id,
                    'error': str(e)
                })

        elif msg.get('cmd') == 'PING':
            output_queue.put({'status': 'pong', 'worker_id': worker_id})


class MPSProcessPool:
    """
    Process pool for MPS inference that bypasses Apple's driver serialization.

    Each worker is a separate process with its own Metal context,
    achieving true parallelism.
    """

    def __init__(self, num_workers: int = 8, model_factory: Optional[Callable] = None):
        """
        Initialize the process pool.

        Args:
            num_workers: Number of worker processes (each gets its own Metal context)
            model_factory: Optional callable that returns a model to use
        """
        self.num_workers = num_workers
        self.workers = []
        self.input_queues = []
        self.output_queues = []

        # Serialize model factory if provided
        factory_bytes = pickle.dumps(model_factory) if model_factory else None

        # Start workers
        mp.set_start_method('spawn', force=True)

        for i in range(num_workers):
            iq = mp.Queue()
            oq = mp.Queue()
            p = mp.Process(
                target=_worker_loop,
                args=(i, iq, oq, factory_bytes),
                daemon=True
            )
            p.start()
            self.workers.append(p)
            self.input_queues.append(iq)
            self.output_queues.append(oq)

        # Wait for all workers to be ready
        for i in range(num_workers):
            msg = self.output_queues[i].get(timeout=30)
            assert msg['status'] == 'ready', f"Worker {i} failed to start"

        print(f"MPSProcessPool: {num_workers} workers ready")

    def forward_parallel(self, num_requests: int, return_results: bool = False) -> List[Any]:
        """
        Submit requests to all workers in parallel.

        Args:
            num_requests: Number of forward passes to execute
            return_results: Whether to return the actual results

        Returns:
            List of results if return_results=True, else empty list
        """
        # Submit work round-robin
        for i in range(num_requests):
            worker_idx = i % self.num_workers
            self.input_queues[worker_idx].put({
                'cmd': 'FORWARD',
                'return_result': return_results
            })

        # Collect results
        results = []
        for i in range(num_requests):
            worker_idx = i % self.num_workers
            msg = self.output_queues[worker_idx].get(timeout=30)
            if msg['status'] == 'error':
                raise RuntimeError(f"Worker error: {msg['error']}")
            if return_results and msg.get('result') is not None:
                results.append(msg['result'])

        return results

    def shutdown(self):
        """Shutdown all workers."""
        for iq in self.input_queues:
            iq.put({'cmd': 'STOP'})

        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        print("MPSProcessPool: Shutdown complete")


if __name__ == "__main__":
    # Quick test
    print("Testing MPSProcessPool...")

    pool = MPSProcessPool(num_workers=4)

    # Warmup
    pool.forward_parallel(8)

    # Benchmark
    start = time.perf_counter()
    pool.forward_parallel(100)
    elapsed = time.perf_counter() - start

    print(f"100 forward passes in {elapsed:.2f}s = {100/elapsed:.1f} ops/s")

    pool.shutdown()
