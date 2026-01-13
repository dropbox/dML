#!/usr/bin/env python3
"""
Repro: TransformerEncoderLayer output corruption under parallel MPS streams.

This reproduces the failure seen in `tests/correctness_benchmark.py --parallel` for:
  nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)

Key observation:
- Running the full layer concurrently across threads can intermittently corrupt outputs.
- Adding a CPU barrier between the self-attention stage and the feed-forward stage
  (no GPU synchronization) eliminates the corruption in this repro.
"""

from __future__ import annotations

import argparse
import copy
import threading
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Config:
    threads: int
    iterations: int
    batch: int
    seq_len: int
    d_model: int
    nhead: int
    barrier_between_stages: bool
    reuse_threads: bool
    atol: float


def _forward_staged(
    layer: nn.TransformerEncoderLayer,
    x: torch.Tensor,
    *,
    barrier_attn: Optional[threading.Barrier],
    barrier_rest: Optional[threading.Barrier],
) -> torch.Tensor:
    with torch.no_grad():
        attn_out, _ = layer.self_attn(x, x, x, need_weights=False)
    if barrier_attn is not None:
        barrier_attn.wait(timeout=60)

    with torch.no_grad():
        x1 = x + layer.dropout1(attn_out)
        x2 = layer.norm1(x1)
    if barrier_rest is not None:
        barrier_rest.wait(timeout=60)

    with torch.no_grad():
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
        x3 = x2 + layer.dropout2(ff)
        out = layer.norm2(x3)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--barrier-between-stages", action="store_true")
    parser.add_argument("--reuse-threads", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = Config(
        threads=args.threads,
        iterations=args.iterations,
        batch=args.batch,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        barrier_between_stages=args.barrier_between_stages,
        reuse_threads=args.reuse_threads,
        atol=args.atol,
    )

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    torch.manual_seed(42)
    layer_cpu = nn.TransformerEncoderLayer(
        cfg.d_model, cfg.nhead, dim_feedforward=cfg.d_model * 4, batch_first=True
    ).eval()
    x_cpu = torch.randn(cfg.batch, cfg.seq_len, cfg.d_model, device="cpu", dtype=torch.float32)

    # One module + input per worker thread (stable device objects, avoids .to() races).
    layers = [copy.deepcopy(layer_cpu).to("mps").eval() for _ in range(cfg.threads)]
    xs = [x_cpu.to("mps") for _ in range(cfg.threads)]

    # Golden from a spawned thread (avoids main-thread vs spawned-thread differences).
    golden_holder: list[Optional[torch.Tensor]] = [None]

    def compute_golden() -> None:
        golden_holder[0] = layers[0](xs[0])

    t = threading.Thread(target=compute_golden)
    t.start()
    t.join()
    torch.mps.synchronize()
    golden = golden_holder[0]
    if golden is None:
        raise RuntimeError("Failed to compute golden output")
    golden_cpu = golden.cpu()

    barrier_attn = threading.Barrier(cfg.threads) if cfg.barrier_between_stages else None
    barrier_rest = threading.Barrier(cfg.threads) if cfg.barrier_between_stages else None

    if cfg.reuse_threads:
        outs: list[Optional[torch.Tensor]] = [None] * cfg.threads
        errors: list[str] = []
        barrier_iter = threading.Barrier(cfg.threads + 1)

        def worker(tid: int) -> None:
            nonlocal outs
            try:
                for _ in range(cfg.iterations):
                    barrier_iter.wait(timeout=60)
                    outs[tid] = _forward_staged(
                        layers[tid],
                        xs[tid],
                        barrier_attn=barrier_attn,
                        barrier_rest=barrier_rest,
                    )
                    barrier_iter.wait(timeout=60)
            except Exception as e:
                try:
                    barrier_iter.abort()
                except Exception:
                    pass
                if barrier_attn is not None:
                    try:
                        barrier_attn.abort()
                    except Exception:
                        pass
                if barrier_rest is not None:
                    try:
                        barrier_rest.abort()
                    except Exception:
                        pass
                errors.append(f"thread {tid}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(cfg.threads)]
        for th in threads:
            th.start()

        try:
            for it in range(cfg.iterations):
                barrier_iter.wait(timeout=60)  # start iteration
                barrier_iter.wait(timeout=60)  # wait for workers

                if errors:
                    raise RuntimeError(";\n".join(errors))

                torch.mps.synchronize()
                for tid, out in enumerate(outs):
                    if out is None:
                        raise RuntimeError(f"Missing output for thread {tid}")
                    out_cpu = out.cpu()
                    diff = (out_cpu.float() - golden_cpu.float()).abs().max().item()
                    if diff > cfg.atol:
                        print(
                            f"FAIL: iter={it} thread={tid} max_abs_diff={diff:.6e} atol={cfg.atol} "
                            f"barrier_between_stages={cfg.barrier_between_stages} reuse_threads={cfg.reuse_threads}"
                        )
                        return 1

                if it % 5 == 0:
                    print(
                        f"PASS: iter={it} barrier_between_stages={cfg.barrier_between_stages} "
                        f"reuse_threads={cfg.reuse_threads}"
                    )
        finally:
            try:
                barrier_iter.abort()
            except Exception:
                pass
            for th in threads:
                th.join()
            if errors:
                raise RuntimeError(";\n".join(errors))
    else:
        for it in range(cfg.iterations):
            barrier_start = threading.Barrier(cfg.threads)
            outs: list[Optional[torch.Tensor]] = [None] * cfg.threads
            errors: list[str] = []

            def worker(tid: int) -> None:
                nonlocal outs
                try:
                    barrier_start.wait(timeout=60)
                    outs[tid] = _forward_staged(
                        layers[tid],
                        xs[tid],
                        barrier_attn=barrier_attn,
                        barrier_rest=barrier_rest,
                    )
                except Exception as e:
                    try:
                        barrier_start.abort()
                    except Exception:
                        pass
                    if barrier_attn is not None:
                        try:
                            barrier_attn.abort()
                        except Exception:
                            pass
                    if barrier_rest is not None:
                        try:
                            barrier_rest.abort()
                        except Exception:
                            pass
                    errors.append(f"thread {tid}: {type(e).__name__}: {e}")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(cfg.threads)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()

            if errors:
                raise RuntimeError(";\n".join(errors))

            torch.mps.synchronize()
            for tid, out in enumerate(outs):
                if out is None:
                    raise RuntimeError(f"Missing output for thread {tid}")
                out_cpu = out.cpu()
                diff = (out_cpu.float() - golden_cpu.float()).abs().max().item()
                if diff > cfg.atol:
                    print(
                        f"FAIL: iter={it} thread={tid} max_abs_diff={diff:.6e} atol={cfg.atol} "
                        f"barrier_between_stages={cfg.barrier_between_stages} reuse_threads={cfg.reuse_threads}"
                    )
                    return 1

            if it % 5 == 0:
                print(
                    f"PASS: iter={it} barrier_between_stages={cfg.barrier_between_stages} "
                    f"reuse_threads={cfg.reuse_threads}"
                )

    print(
        f"ALL PASS: barrier_between_stages={cfg.barrier_between_stages} reuse_threads={cfg.reuse_threads}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
