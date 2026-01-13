#!/usr/bin/env python3
"""
Verify the AGX *binary* driver patch works without libagx_fix.dylib.

Intended use:
  1) Deploy the patched driver via `agx_patch/deploy_patch.sh` (requires SIP disabled)
  2) Reboot
  3) Run this script WITHOUT `DYLD_INSERT_LIBRARIES` set

Safety:
  - By default, this script refuses to run the stress test unless the system AGX driver
    SHA256 matches the known patched hash (macOS 15.7.3 / M4 Max).
  - Use `--force-run` to override this (e.g., different driver build hash).

This script runs an MPS multi-threaded stress workload in subprocesses to avoid
taking down the parent process if the driver crashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_DRIVER_PATH = "/System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X"

# Known SHA256 hashes for macOS 15.7.3 / Apple M4 Max environment (see agx_patch/DEPLOY_INSTRUCTIONS.md).
KNOWN_DRIVER_SHA256 = {
    "universal_original": "fbd62445e186aeee071e65a4baf6a6da5947ca73a0cd65a9f53a6c269f32d345",
    "universal_patched": "3b6813011e481cea46dd2942b966bdc48712d9adcd1a1b836f6710ecb1c3fb0d",
}


@dataclass(frozen=True)
class TrialResult:
    ok: bool
    crashed: bool
    returncode: int
    payload: dict[str, Any] | None
    stdout: str
    stderr: str


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def print_driver_status(driver_path: Path) -> None:
    if not driver_path.exists():
        print(f"WARNING: Driver not found at {driver_path}", file=sys.stderr)
        return

    print(f"Driver path: {driver_path}")
    sha = compute_sha256(driver_path)
    print(f"Driver SHA256: {sha}")

    if sha == KNOWN_DRIVER_SHA256["universal_patched"]:
        print("Driver status: PATCHED (matches known patched SHA256)")
    elif sha == KNOWN_DRIVER_SHA256["universal_original"]:
        print("Driver status: ORIGINAL (matches known original SHA256)")
    else:
        print("Driver status: UNKNOWN (SHA256 does not match known values)")
        print("NOTE: This can happen after macOS updates or on different hardware/driver builds.")


def get_sip_status() -> str:
    try:
        proc = subprocess.run(["csrutil", "status"], capture_output=True, text=True)
    except FileNotFoundError:
        return "unknown"

    out = (proc.stdout or "") + (proc.stderr or "")
    out = out.lower()
    if "enabled" in out:
        return "enabled"
    if "disabled" in out:
        return "disabled"
    return "unknown"


def run_metal_diagnostics(repo_root: Path) -> int:
    diag = repo_root / "tests" / "metal_diagnostics.sh"
    if not diag.exists():
        print("WARNING: tests/metal_diagnostics.sh not found; skipping Metal visibility check.", file=sys.stderr)
        return 2
    proc = subprocess.run([str(diag), "--check"])
    return int(proc.returncode)


def _stress_child_code() -> str:
    # Keep this snippet self-contained: no imports from repo files.
    return r"""
import json
import os
import sys
import threading
import time

os.environ.setdefault("PYTORCH_NO_MPS_PROFILER", "1")

import torch

if not torch.backends.mps.is_available():
    print(json.dumps({"error": "MPS not available", "mps_built": torch.backends.mps.is_built()}))
    sys.exit(2)

threads = int(os.environ["VERIFY_PATCH_THREADS"])
iterations = int(os.environ["VERIFY_PATCH_ITERS"])
size = int(os.environ["VERIFY_PATCH_SIZE"])

# Touch MPS early so failures happen inside the child process.
torch.zeros(1, device="mps")
torch.mps.synchronize()

completed = 0
errors = []
lock = threading.Lock()

def worker(tid: int) -> None:
    global completed
    try:
        for _ in range(iterations):
            x = torch.randn(size, size, device="mps")
            y = torch.randn(size, size, device="mps")
            z = torch.mm(x, y)
            z = torch.relu(z)
            torch.mps.synchronize()
            with lock:
                completed += 1
    except Exception as e:
        with lock:
            errors.append((tid, repr(e)))

ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
start = time.perf_counter()
for t in ts:
    t.start()
for t in ts:
    t.join()
elapsed = time.perf_counter() - start

expected = threads * iterations
ops_per_sec = (completed / elapsed) if elapsed > 0 else 0.0
payload = {
    "torch_version": torch.__version__,
    "threads": threads,
    "iterations": iterations,
    "size": size,
    "completed": completed,
    "expected": expected,
    "errors": errors,
    "elapsed_s": elapsed,
    "ops_per_sec": ops_per_sec,
}
print(json.dumps(payload))
sys.exit(0 if (not errors and completed == expected) else 1)
""".strip()


def run_stress_trial(threads: int, iterations: int, size: int, timeout_s: int) -> TrialResult:
    env = os.environ.copy()
    env.pop("DYLD_INSERT_LIBRARIES", None)
    env["MPS_FORCE_GRAPH_PATH"] = env.get("MPS_FORCE_GRAPH_PATH", "1")
    env["PYTORCH_NO_MPS_PROFILER"] = env.get("PYTORCH_NO_MPS_PROFILER", "1")
    env["VERIFY_PATCH_THREADS"] = str(threads)
    env["VERIFY_PATCH_ITERS"] = str(iterations)
    env["VERIFY_PATCH_SIZE"] = str(size)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _stress_child_code()],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return TrialResult(
            ok=False,
            crashed=False,
            returncode=124,
            payload=None,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + "\nTIMEOUT",
        )

    crashed = proc.returncode < 0 or proc.returncode in (134, 137, 139)
    payload: dict[str, Any] | None = None
    stdout = proc.stdout.strip()
    if stdout:
        last_line = stdout.splitlines()[-1].strip()
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            payload = None

    ok = proc.returncode == 0
    return TrialResult(
        ok=ok,
        crashed=crashed,
        returncode=int(proc.returncode),
        payload=payload,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify AGX binary patch (no libagx_fix.dylib) via MPS stress test",
    )
    parser.add_argument("--threads", type=int, default=8, help="Worker threads (default: 8)")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per thread (default: 100)")
    parser.add_argument("--size", type=int, default=256, help="Matrix size for torch.mm (default: 256)")
    parser.add_argument("--trials", type=int, default=3, help="Repeat stress test N times (default: 3)")
    parser.add_argument("--timeout-s", type=int, default=300, help="Per-trial timeout (default: 300)")
    parser.add_argument(
        "--min-efficiency",
        type=float,
        default=0.0,
        help="If >0, fail when efficiency is below this percent (default: 0 = report only)",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Run stress test even if system driver SHA256 is not recognized as patched",
    )
    parser.add_argument(
        "--skip-driver-hash",
        action="store_true",
        help="Skip computing /System driver SHA256",
    )
    args = parser.parse_args()

    dyld = os.environ.get("DYLD_INSERT_LIBRARIES", "")
    if "libagx_fix" in dyld:
        print("ERROR: libagx_fix.dylib appears to be loaded via DYLD_INSERT_LIBRARIES.", file=sys.stderr)
        print("This test verifies the binary patch alone. Run WITHOUT DYLD_INSERT_LIBRARIES.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent

    print("=== AGX Binary Patch Verification (no libagx_fix) ===")
    print("")

    print("Metal visibility check:")
    metal_rc = run_metal_diagnostics(repo_root)
    if metal_rc == 1:
        print("ERROR: Metal devices are NOT visible to this process.", file=sys.stderr)
        return 2
    if metal_rc == 2:
        print("WARNING: Metal visibility unknown (diagnostics unavailable).", file=sys.stderr)
    print("")

    print(f"SIP status: {get_sip_status()}")
    print("")

    if not args.skip_driver_hash:
        print("System driver fingerprint:")
        driver_path = Path(SYSTEM_DRIVER_PATH)
        if not driver_path.exists():
            print(f"ERROR: System driver not found at {driver_path}", file=sys.stderr)
            return 1

        driver_sha = compute_sha256(driver_path)
        if driver_sha != KNOWN_DRIVER_SHA256["universal_patched"] and not args.force_run:
            print("ERROR: System driver does not match the known PATCHED SHA256.", file=sys.stderr)
            print("Refusing to run the stress test by default because unpatched drivers can crash.", file=sys.stderr)
            print(f"Driver SHA256: {driver_sha}", file=sys.stderr)
            print(f"Known patched: {KNOWN_DRIVER_SHA256['universal_patched']}", file=sys.stderr)
            print("", file=sys.stderr)
            print(
                "If you have deployed a patched driver with a different build hash, re-run with --force-run.",
                file=sys.stderr,
            )
            print(
                "If you have not deployed the patch, run: sudo ./agx_patch/deploy_patch.sh (requires SIP disabled) and reboot.",
                file=sys.stderr,
            )
            return 1

        print_driver_status(driver_path)
        print("")
    elif not args.force_run:
        print("ERROR: --skip-driver-hash requires --force-run (cannot confirm driver is patched).", file=sys.stderr)
        return 1

    # Baseline throughput (single-thread).
    print("Baseline (1 thread):")
    baseline = run_stress_trial(threads=1, iterations=args.iterations, size=args.size, timeout_s=args.timeout_s)
    if not baseline.ok or not baseline.payload:
        print("FAIL: Baseline run failed.", file=sys.stderr)
        if baseline.stdout:
            print(baseline.stdout)
        if baseline.stderr:
            print(baseline.stderr, file=sys.stderr)
        return 1

    baseline_ops = float(baseline.payload.get("ops_per_sec", 0.0))
    print(f"  ops/s: {baseline_ops:.1f}")
    print("")

    # Multi-threaded trials.
    print(f"Stress ({args.threads} threads) x {args.trials} trials:")
    trial_ops: list[float] = []
    failures: list[TrialResult] = []
    crashes = 0

    for i in range(args.trials):
        trial = run_stress_trial(
            threads=args.threads,
            iterations=args.iterations,
            size=args.size,
            timeout_s=args.timeout_s,
        )
        if trial.ok and trial.payload:
            ops = float(trial.payload.get("ops_per_sec", 0.0))
            trial_ops.append(ops)
            print(f"  Trial {i + 1}: PASS ({ops:.1f} ops/s)")
        else:
            failures.append(trial)
            if trial.crashed:
                crashes += 1
            print(f"  Trial {i + 1}: FAIL (rc={trial.returncode})", file=sys.stderr)

    print("")
    crash_rate = (crashes / args.trials) * 100.0 if args.trials > 0 else 0.0
    print(f"Crash rate: {crashes}/{args.trials} ({crash_rate:.1f}%)")

    if trial_ops:
        mean_ops = statistics.mean(trial_ops)
        stdev_ops = statistics.pstdev(trial_ops) if len(trial_ops) > 1 else 0.0
        speedup = mean_ops / baseline_ops if baseline_ops > 0 else 0.0
        efficiency = (speedup / args.threads) * 100.0 if args.threads > 0 else 0.0
        print(f"Throughput: mean={mean_ops:.1f} ops/s (stdev={stdev_ops:.1f})")
        print(f"Speedup: {speedup:.2f}x vs 1T")
        print(f"Efficiency: {efficiency:.1f}% at {args.threads} threads (target: 50%+)")
        if args.min_efficiency > 0 and efficiency < args.min_efficiency:
            print(
                f"FAIL: Efficiency {efficiency:.1f}% < {args.min_efficiency:.1f}% threshold",
                file=sys.stderr,
            )
            return 1

    if failures:
        print("\nFirst failure details:", file=sys.stderr)
        f = failures[0]
        if f.stdout:
            print(f.stdout, file=sys.stderr)
        if f.stderr:
            print(f.stderr, file=sys.stderr)
        return 1

    print("\nPASS: No crashes detected; binary patch appears effective.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
