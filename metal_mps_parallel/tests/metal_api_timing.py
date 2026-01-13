#!/usr/bin/env python3
"""
Measure which bare Metal API calls serialize across threads.

This script builds and runs the pure Metal timing repro in:
  tests/metal_pure_objc_repro/main.mm

It sweeps thread counts and reports timings for:
- MTLCommandQueue.commandBuffer (create)
- MTLCommandBuffer.computeCommandEncoder (create)
- MTLCommandBuffer.commit
- MTLCommandBuffer.waitUntilCompleted

This is used for Hole 1 ("Metal driver bottleneck") evidence gathering.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "tests" / "metal_pure_objc_repro" / "main.mm"
DEFAULT_BIN = REPO_ROOT / "tests" / "build" / "metal_pure_objc_repro"


@dataclass(frozen=True)
class RunConfig:
    threads: int
    iters: int
    elements: int
    no_wait: bool


def _parse_thread_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty thread list")
    return out


def _needs_rebuild(src: Path, binary: Path) -> bool:
    if not binary.exists():
        return True
    try:
        return src.stat().st_mtime_ns > binary.stat().st_mtime_ns
    except FileNotFoundError:
        return True


def _build_repro(src: Path, binary: Path) -> None:
    binary.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "clang++",
        "-std=c++17",
        "-O2",
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-o",
        str(binary),
        str(src),
    ]
    subprocess.run(cmd, check=True)


def _run_repro(binary: Path, cfg: RunConfig) -> dict[str, Any]:
    cmd = [
        str(binary),
        "--threads",
        str(cfg.threads),
        "--iters",
        str(cfg.iters),
        "--elements",
        str(cfg.elements),
    ]
    if cfg.no_wait:
        cmd.append("--no-wait")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _fmt_float(x: float) -> str:
    if x == 0.0:
        return "0"
    if x >= 1000.0:
        return f"{x:,.0f}"
    if x >= 100.0:
        return f"{x:.1f}"
    if x >= 10.0:
        return f"{x:.2f}"
    return f"{x:.3f}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Bare Metal API timing sweep")
    parser.add_argument("--threads", default="1,2,4,8", help="comma-separated thread counts")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--elements", type=int, default=262144)
    parser.add_argument("--no-wait", action="store_true", help="skip waitUntilCompleted in repro")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force recompilation of the Objective-C++ repro",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "main"
        / f"metal_api_timing_{_dt.date.today().isoformat()}.json",
    )
    args = parser.parse_args(argv)

    thread_counts = _parse_thread_list(args.threads)
    if min(thread_counts) <= 0:
        raise SystemExit("threads must be positive")
    if args.iters <= 0 or args.elements <= 0:
        raise SystemExit("iters/elements must be positive")

    if args.rebuild or _needs_rebuild(args.src, args.bin):
        _build_repro(args.src, args.bin)

    rows: list[dict[str, Any]] = []
    for t in thread_counts:
        cfg = RunConfig(threads=t, iters=args.iters, elements=args.elements, no_wait=args.no_wait)
        row = _run_repro(args.bin, cfg)
        rows.append(row)

    baseline = next((r for r in rows if r["threads"] == 1), None)
    baseline_tp = float(baseline["throughput_ops_s"]) if baseline else None

    print("# Metal API Timing Sweep")
    print()
    print(f"- src: {os.path.relpath(args.src, REPO_ROOT)}")
    print(f"- bin: {os.path.relpath(args.bin, REPO_ROOT)}")
    print(f"- iters: {args.iters}")
    print(f"- elements: {args.elements}")
    print(f"- no_wait: {args.no_wait}")
    print()
    print("| threads | throughput (ops/s) | speedup | commit p50 (us) | commit p95 (us) | wait p50 (us) | gpu mean (us) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        tp = float(r["throughput_ops_s"])
        speedup = (tp / baseline_tp) if baseline_tp and baseline_tp > 0 else 0.0
        commit_p50 = float(r["commit"]["p50_us"])
        commit_p95 = float(r["commit"]["p95_us"])
        wait_p50 = float(r["wait"]["p50_us"])
        gpu_mean = float(r["gpu"]["mean_us"])
        print(
            f"| {r['threads']} | {_fmt_float(tp)} | {_fmt_float(speedup)}x | "
            f"{_fmt_float(commit_p50)} | {_fmt_float(commit_p95)} | {_fmt_float(wait_p50)} | {_fmt_float(gpu_mean)} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "src": os.path.relpath(args.src, REPO_ROOT),
        "bin": os.path.relpath(args.bin, REPO_ROOT),
        "iters": args.iters,
        "elements": args.elements,
        "no_wait": args.no_wait,
        "results": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print()
    print(f"Wrote: {os.path.relpath(args.output, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

