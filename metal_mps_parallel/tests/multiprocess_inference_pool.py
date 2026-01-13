#!/usr/bin/env python3
"""
Example: Multi-process MPS inference pool.

This is the recommended production pattern for >2 concurrent nn.Module inference
on Apple MPS: process-based parallelism (one model instance per process).

Usage:
  python tests/multiprocess_inference_pool.py --workers 4 --tasks 32 --iters 20
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_MODEL = None
_DEVICE = None


def _init_worker(model_kind: str, batch: int, in_features: int, hidden: int, out_features: int) -> None:
    global _MODEL, _DEVICE

    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in worker process")

    _DEVICE = torch.device("mps")

    if model_kind == "linear":
        _MODEL = nn.Linear(in_features, out_features).to(_DEVICE)
    elif model_kind == "mlp":
        _MODEL = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(_DEVICE)
    else:
        raise ValueError(f"Unknown model_kind={model_kind!r}")

    _MODEL.eval()

    # Warm up once to initialize MPS/graph caches in this process.
    with torch.no_grad():
        x = torch.randn(batch, in_features, device=_DEVICE)
        _ = _MODEL(x)
        torch.mps.synchronize()


def _run_task(task_id: int, iterations: int, batch: int, in_features: int) -> tuple[int, float]:
    import torch

    if _MODEL is None or _DEVICE is None:
        raise RuntimeError("Worker model not initialized (initializer did not run?)")

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            x = torch.randn(batch, in_features, device=_DEVICE)
            _ = _MODEL(x)
            torch.mps.synchronize()
    end = time.perf_counter()

    return task_id, end - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-process MPS inference example")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--tasks", type=int, default=16, help="Number of tasks to submit")
    parser.add_argument("--iters", type=int, default=10, help="Iterations per task")
    parser.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512, help="MLP hidden size")
    parser.add_argument("--out-features", type=int, default=256)
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.tasks < 1:
        raise SystemExit("--tasks must be >= 1")
    if args.iters < 1:
        raise SystemExit("--iters must be >= 1")

    ctx = mp.get_context("spawn")

    t0 = time.perf_counter()
    durations: list[float] = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(args.model, args.batch, args.in_features, args.hidden, args.out_features),
    ) as executor:
        futures = [
            executor.submit(_run_task, task_id, args.iters, args.batch, args.in_features)
            for task_id in range(args.tasks)
        ]

        for future in as_completed(futures):
            _, dt = future.result()
            durations.append(dt)

    wall = time.perf_counter() - t0
    total_forwards = args.tasks * args.iters

    print("==========================================")
    print("Multi-Process MPS Inference Example")
    print("==========================================")
    print(f"workers: {args.workers}  tasks: {args.tasks}  iters/task: {args.iters}  total iters: {total_forwards}")
    print(f"model: {args.model}  batch: {args.batch}  in_features: {args.in_features}  out_features: {args.out_features}")
    print(f"wall time: {wall:.3f}s  throughput: {total_forwards / wall:.1f} forwards/s")
    print(f"mean task time: {sum(durations) / max(len(durations), 1):.3f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

