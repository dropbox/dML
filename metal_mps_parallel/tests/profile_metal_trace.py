#!/usr/bin/env python3
"""
Minimal workload for capturing Metal System Trace profiles.

This script is intentionally small and self-contained so it can be launched under
Instruments (xctrace). It runs a steady-state multi-threaded MPS workload and
prints a single JSON summary to stdout.

Example:
  /Applications/Xcode.app/Contents/Developer/usr/bin/xctrace record \
    --template "Metal System Trace" --output reports/traces/mps_parallel.trace \
    --launch -- $(command -v python3) tests/profile_metal_trace.py --op matmul --threads 8 --iters 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _require_mps() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required") from exc

    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend not available (run on Apple Silicon with MPS)")


def _set_single_cpu_threading() -> None:
    import torch

    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _build_op(op: str, size: int, dtype: str):
    import torch
    import torch.nn as nn

    device = torch.device("mps")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]

    if op == "matmul":
        a = torch.randn(size, size, device=device, dtype=torch_dtype)
        b = torch.randn(size, size, device=device, dtype=torch_dtype)

        def run_once():
            return a @ b

        return run_once

    if op == "linear":
        model = nn.Linear(size, size, bias=False).to(device=device, dtype=torch_dtype).eval()
        x = torch.randn(size, size, device=device, dtype=torch_dtype)

        def run_once():
            with torch.no_grad():
                return model(x)

        return run_once

    raise ValueError(f"unknown op: {op}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Parallel MPS workload for Instruments tracing")
    parser.add_argument("--op", choices=["matmul", "linear"], default="matmul")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--sync-every", type=int, default=1, help="0 disables per-iter sync")
    parser.add_argument(
        "--sync-mode",
        choices=["device", "event", "none"],
        default="device",
        help="device=torch.mps.synchronize (all streams), event=per-thread Event.synchronize, none=no sync",
    )
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args(argv)

    _require_mps()
    _set_single_cpu_threading()

    import torch

    torch.manual_seed(0)
    torch.mps.synchronize()

    run_once_fns = [_build_op(args.op, args.size, args.dtype) for _ in range(args.threads)]

    def sync_fn(event) -> None:
        if args.sync_mode == "none":
            return
        if args.sync_mode == "device":
            torch.mps.synchronize()
            return
        event.record()
        event.synchronize()

    for run_once in run_once_fns:
        ev = torch.mps.Event(enable_timing=False) if args.sync_mode == "event" else None
        for i in range(args.warmup_iters):
            _ = run_once()
            if args.sync_every and ((i + 1) % args.sync_every == 0):
                sync_fn(ev)
        if args.sync_mode != "none":
            sync_fn(ev)

    torch.mps.synchronize()

    barrier = threading.Barrier(args.threads)

    def worker(tid: int) -> dict:
        ev = torch.mps.Event(enable_timing=False) if args.sync_mode == "event" else None
        latencies_s: list[float] = []
        barrier.wait()
        for i in range(args.iters):
            t0 = time.perf_counter()
            _ = run_once_fns[tid]()
            if args.sync_every and ((i + 1) % args.sync_every == 0):
                sync_fn(ev)
            t1 = time.perf_counter()
            latencies_s.append(t1 - t0)
        if args.sync_mode != "none":
            sync_fn(ev)
        return {"tid": tid, "latencies_s": latencies_s}

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        results = list(pool.map(worker, range(args.threads)))
    if args.sync_mode == "none":
        torch.mps.synchronize()
    t_end = time.perf_counter()

    all_lat = [x for r in results for x in r["latencies_s"]]
    summary = {
        "op": args.op,
        "dtype": args.dtype,
        "size": args.size,
        "threads": args.threads,
        "iters_per_thread": args.iters,
        "sync_every": args.sync_every,
        "total_ops": args.threads * args.iters,
        "wall_time_s": t_end - t_start,
        "throughput_ops_s": (args.threads * args.iters) / (t_end - t_start),
        "latency_ms_mean": statistics.mean(all_lat) * 1000.0 if all_lat else 0.0,
        "latency_ms_p50": statistics.median(all_lat) * 1000.0 if all_lat else 0.0,
        "latency_ms_p95": (statistics.quantiles(all_lat, n=20)[18] * 1000.0) if len(all_lat) >= 20 else 0.0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
