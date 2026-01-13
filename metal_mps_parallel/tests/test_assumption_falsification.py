#!/usr/bin/env python3
"""
Assumption Falsification Tests for MPS Parallel Inference

This module tests the assumptions underlying our MPS parallelization workarounds.
Each test deliberately disables one protective mechanism and runs stress tests
to capture evidence of Apple MPS framework bugs.

Purpose:
    Turn "Apple is broken" from an assumption into documented evidence.
    Each test produces a PASS/CRASH/RACE result that proves whether
    the workaround is necessary.

Usage:
    python3 tests/test_assumption_falsification.py

Evidence Output:
    reports/main/assumption_falsification_results.json

Reference:
    WORKER_VERIFICATION_PARAGON_CHECKLIST.md Phase 3 requirement
"""

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Force graph path mode for consistent reproduction
os.environ["MPS_FORCE_GRAPH_PATH"] = "1"

import torch
import torch.nn.functional as F


@dataclass
class AssumptionTestResult:
    """Result of a single assumption falsification test."""
    name: str
    assumption: str
    workaround: str
    test_mode: str  # "protected" or "unprotected"
    iterations: int
    threads: int
    passed: int
    failed: int
    crashes: int
    max_diff: float
    result: str  # "PASS", "FAIL", "CRASH"
    evidence: str
    duration_ms: float


@dataclass
class FalsificationReport:
    """Complete falsification test report."""
    timestamp: str
    pytorch_version: str
    mps_available: bool
    device_name: str
    tests: List[AssumptionTestResult] = field(default_factory=list)
    summary: str = ""


class AssumptionFalsificationSuite:
    """Suite of assumption falsification tests."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.report = FalsificationReport(
            timestamp=datetime.now().isoformat(),
            pytorch_version=torch.__version__,
            mps_available=torch.backends.mps.is_available(),
            device_name=self._get_device_name()
        )

    def _get_device_name(self) -> str:
        """Get MPS device name if available."""
        if torch.backends.mps.is_available():
            try:
                return str(torch.device("mps"))
            except:
                return "mps (name unavailable)"
        return "N/A"

    def run_all_tests(self) -> FalsificationReport:
        """Run all assumption falsification tests."""
        if not torch.backends.mps.is_available():
            print("SKIP: MPS not available")
            self.report.summary = "SKIPPED: MPS not available"
            return self.report

        tests = [
            self.test_contiguous_race_assumption,
            self.test_sdpa_parallel_assumption,
            self.test_batch_serialization_assumption,
        ]

        for test_func in tests:
            print(f"\n{'='*70}")
            print(f"Running: {test_func.__name__}")
            print('='*70)
            try:
                result = test_func()
                self.report.tests.append(result)
                print(f"Result: {result.result}")
                print(f"Evidence: {result.evidence}")
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()

        self._generate_summary()
        return self.report

    def _generate_summary(self):
        """Generate summary of falsification results."""
        assumptions_proven = []
        assumptions_unproven = []

        for test in self.report.tests:
            if "unprotected" in test.test_mode and test.result in ["FAIL", "CRASH"]:
                assumptions_proven.append(test.name)
            elif "unprotected" in test.test_mode and test.result == "PASS":
                assumptions_unproven.append(test.name)

        summary_lines = [
            f"Tested {len(self.report.tests)} assumptions",
            f"Bugs proven: {len(assumptions_proven)}",
            f"Bugs not reproduced: {len(assumptions_unproven)}",
        ]

        if assumptions_proven:
            summary_lines.append(f"\nProven Apple MPS bugs requiring workarounds:")
            for name in assumptions_proven:
                summary_lines.append(f"  - {name}")

        if assumptions_unproven:
            summary_lines.append(f"\nAssumptions not proven (may be intermittent):")
            for name in assumptions_unproven:
                summary_lines.append(f"  - {name}")

        self.report.summary = "\n".join(summary_lines)

    def test_contiguous_race_assumption(self) -> AssumptionTestResult:
        """
        Assumption: Apple MPS has a race condition in .contiguous()
        when called from multiple threads.

        Workaround: Use .reshape() instead of .view() + .contiguous()
        """
        assumption = (
            "Apple MPS has a race condition in memory copy operations "
            "triggered by .contiguous() on tensors with complex stride patterns "
            "when called from multiple threads simultaneously."
        )
        workaround = (
            "Use .reshape() which can handle non-contiguous tensors directly, "
            "avoiding the internal .contiguous() call race condition."
        )

        iterations = 30
        threads = 8

        # Test WITHOUT workaround (use contiguous - triggers race)
        start_time = time.time()
        passed, failed, crashes, max_diff = self._run_contiguous_test(
            use_contiguous=True, iterations=iterations, threads=threads
        )
        duration_ms = (time.time() - start_time) * 1000

        if crashes > 0:
            result = "CRASH"
            evidence = f"Process crashed {crashes} times without workaround"
        elif failed > 0:
            result = "FAIL"
            evidence = f"Race detected: {failed}/{iterations} iterations failed, max_diff={max_diff:.2e}"
        else:
            result = "PASS"
            evidence = f"No race detected in {iterations} iterations (bug may be intermittent)"

        return AssumptionTestResult(
            name="contiguous_race",
            assumption=assumption,
            workaround=workaround,
            test_mode="unprotected (use .contiguous())",
            iterations=iterations,
            threads=threads,
            passed=passed,
            failed=failed,
            crashes=crashes,
            max_diff=max_diff,
            result=result,
            evidence=evidence,
            duration_ms=duration_ms
        )

    def _run_contiguous_test(
        self, use_contiguous: bool, iterations: int, threads: int
    ) -> Tuple[int, int, int, float]:
        """Run contiguous race test."""
        embed_dim = 256
        batch_size = 4
        seq_len = 128

        torch.manual_seed(42)
        weight = torch.randn(3 * embed_dim, embed_dim, device="mps")
        bias = torch.randn(3 * embed_dim, device="mps")
        torch.mps.synchronize()

        num_heads = 4
        head_dim = embed_dim // num_heads

        def projection_op(x):
            proj = F.linear(x, weight, bias)
            proj = (
                proj.unflatten(-1, (3, embed_dim))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
            )

            if use_contiguous:
                proj = proj.contiguous()  # Triggers race

            q, k, v = proj[0], proj[1], proj[2]

            if use_contiguous:
                q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            else:
                q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            return out

        passed = 0
        failed = 0
        crashes = 0
        max_diff = 0.0

        for iteration in range(iterations):
            try:
                inputs = []
                for tid in range(threads):
                    torch.manual_seed(iteration * 1000 + tid)
                    inputs.append(torch.randn(batch_size, seq_len, embed_dim, device="mps"))
                torch.mps.synchronize()

                expected = []
                with torch.no_grad():
                    for inp in inputs:
                        expected.append(projection_op(inp).clone())
                torch.mps.synchronize()

                results = [None] * threads
                errors = [None] * threads

                def worker(tid):
                    try:
                        with torch.no_grad():
                            results[tid] = projection_op(inputs[tid])
                        torch.mps.synchronize()
                    except Exception as e:
                        errors[tid] = str(e)

                worker_threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
                for t in worker_threads:
                    t.start()
                for t in worker_threads:
                    t.join()

                iteration_ok = True
                for tid in range(threads):
                    if errors[tid]:
                        iteration_ok = False
                        continue
                    if results[tid] is None:
                        iteration_ok = False
                        continue
                    diff = (results[tid] - expected[tid]).abs().max().item()
                    max_diff = max(max_diff, diff)
                    if diff > 1e-3:
                        iteration_ok = False

                if iteration_ok:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                crashes += 1

        return passed, failed, crashes, max_diff

    def test_sdpa_parallel_assumption(self) -> AssumptionTestResult:
        """
        Assumption: Apple MPS SDPA has internal shared state that causes
        races when called from multiple threads.

        Workaround: Serialize SDPA calls through batch queue (num_workers=1)
        """
        assumption = (
            "Apple MPS's scaled_dot_product_attention has internal shared state "
            "that causes data races when called from multiple threads without "
            "serialization."
        )
        workaround = (
            "Use MPSBatchQueue with num_workers=1 to serialize GPU access, "
            "avoiding the internal race condition in Apple's SDPA implementation."
        )

        iterations = 20
        threads = 8

        start_time = time.time()
        passed, failed, crashes, max_diff = self._run_sdpa_test(
            iterations=iterations, threads=threads
        )
        duration_ms = (time.time() - start_time) * 1000

        if crashes > 0:
            result = "CRASH"
            evidence = f"Process crashed {crashes} times during parallel SDPA"
        elif failed > 0:
            result = "FAIL"
            evidence = f"Race detected: {failed}/{iterations} iterations failed, max_diff={max_diff:.2e}"
        else:
            result = "PASS"
            evidence = f"No race detected in {iterations} iterations (bug may be intermittent)"

        return AssumptionTestResult(
            name="sdpa_parallel_race",
            assumption=assumption,
            workaround=workaround,
            test_mode="unprotected (parallel SDPA)",
            iterations=iterations,
            threads=threads,
            passed=passed,
            failed=failed,
            crashes=crashes,
            max_diff=max_diff,
            result=result,
            evidence=evidence,
            duration_ms=duration_ms
        )

    def _run_sdpa_test(self, iterations: int, threads: int) -> Tuple[int, int, int, float]:
        """Run SDPA parallel test."""
        batch_size = 4
        seq_len = 64
        num_heads = 8
        head_dim = 64

        passed = 0
        failed = 0
        crashes = 0
        max_diff = 0.0

        for iteration in range(iterations):
            try:
                # Create unique Q, K, V for each thread
                qs, ks, vs = [], [], []
                for tid in range(threads):
                    torch.manual_seed(iteration * 1000 + tid)
                    qs.append(torch.randn(batch_size, num_heads, seq_len, head_dim, device="mps"))
                    ks.append(torch.randn(batch_size, num_heads, seq_len, head_dim, device="mps"))
                    vs.append(torch.randn(batch_size, num_heads, seq_len, head_dim, device="mps"))
                torch.mps.synchronize()

                # Compute expected serially
                expected = []
                with torch.no_grad():
                    for tid in range(threads):
                        out = F.scaled_dot_product_attention(qs[tid], ks[tid], vs[tid])
                        expected.append(out.clone())
                torch.mps.synchronize()

                # Run in parallel
                results = [None] * threads
                errors = [None] * threads

                def worker(tid):
                    try:
                        with torch.no_grad():
                            results[tid] = F.scaled_dot_product_attention(qs[tid], ks[tid], vs[tid])
                        torch.mps.synchronize()
                    except Exception as e:
                        errors[tid] = str(e)

                worker_threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
                for t in worker_threads:
                    t.start()
                for t in worker_threads:
                    t.join()

                iteration_ok = True
                for tid in range(threads):
                    if errors[tid]:
                        iteration_ok = False
                        continue
                    if results[tid] is None:
                        iteration_ok = False
                        continue
                    diff = (results[tid] - expected[tid]).abs().max().item()
                    max_diff = max(max_diff, diff)
                    if diff > 1e-3:
                        iteration_ok = False

                if iteration_ok:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                crashes += 1

        return passed, failed, crashes, max_diff

    def test_batch_serialization_assumption(self) -> AssumptionTestResult:
        """
        Assumption: Without batch serialization, 8 threads doing direct MPS
        inference causes more failures than with serialization.

        This test compares protected (serialized) vs unprotected (parallel) behavior.
        """
        assumption = (
            "Running 8+ threads with direct MPS inference (no serialization) "
            "causes more race conditions than with serialization."
        )
        workaround = (
            "MPSBatchQueue with num_workers=1 serializes GPU access, achieving "
            "better correctness at high thread counts."
        )

        iterations = 10
        threads = 8

        start_time = time.time()

        # Run UNPROTECTED (no serialization)
        unprotected_passed, unprotected_failed, unprotected_crashes, unprotected_diff = \
            self._run_batch_serialization_test(iterations=iterations, threads=threads, serialize=False)

        # Run PROTECTED (with serialization)
        protected_passed, protected_failed, protected_crashes, protected_diff = \
            self._run_batch_serialization_test(iterations=iterations, threads=threads, serialize=True)

        duration_ms = (time.time() - start_time) * 1000

        # Compare protected vs unprotected
        if protected_passed > unprotected_passed:
            result = "FAIL"  # "FAIL" means bug proven (unprotected worse than protected)
            evidence = (
                f"Serialization improves correctness: "
                f"protected={protected_passed}/{iterations}, "
                f"unprotected={unprotected_passed}/{iterations}"
            )
        elif unprotected_crashes > 0:
            result = "CRASH"
            evidence = f"Unprotected crashed {unprotected_crashes} times"
        else:
            result = "PASS"  # Bug not proven this run
            evidence = (
                f"No difference: protected={protected_passed}/{iterations}, "
                f"unprotected={unprotected_passed}/{iterations}"
            )

        return AssumptionTestResult(
            name="batch_serialization_needed",
            assumption=assumption,
            workaround=workaround,
            test_mode=f"comparison (protected vs unprotected)",
            iterations=iterations,
            threads=threads,
            passed=unprotected_passed,  # Report unprotected stats
            failed=unprotected_failed,
            crashes=unprotected_crashes,
            max_diff=max(unprotected_diff, protected_diff),
            result=result,
            evidence=evidence,
            duration_ms=duration_ms
        )

    def _run_batch_serialization_test(
        self, iterations: int, threads: int, serialize: bool = True
    ) -> Tuple[int, int, int, float]:
        """Run test with or without serialization."""
        embed_dim = 256
        batch_size = 4

        torch.manual_seed(42)
        model_weight = torch.randn(embed_dim, embed_dim, device="mps")
        torch.mps.synchronize()

        # Serialization via a global lock (simulates batch queue with 1 worker)
        global_lock = threading.Lock() if serialize else None

        def model_forward(x):
            if global_lock:
                with global_lock:
                    out = F.linear(x, model_weight)
                    torch.mps.synchronize()
                    return out
            else:
                out = F.linear(x, model_weight)
                torch.mps.synchronize()
                return out

        passed = 0
        failed = 0
        crashes = 0
        max_diff = 0.0

        for iteration in range(iterations):
            try:
                inputs = []
                for tid in range(threads):
                    torch.manual_seed(iteration * 1000 + tid)
                    inputs.append(torch.randn(batch_size, embed_dim, device="mps"))
                torch.mps.synchronize()

                # Expected (serial)
                expected = []
                with torch.no_grad():
                    for inp in inputs:
                        expected.append(model_forward(inp).clone())

                # Parallel with serialization
                results = [None] * threads
                errors = [None] * threads

                def worker(tid):
                    try:
                        with torch.no_grad():
                            results[tid] = model_forward(inputs[tid])
                    except Exception as e:
                        errors[tid] = str(e)

                worker_threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
                for t in worker_threads:
                    t.start()
                for t in worker_threads:
                    t.join()

                iteration_ok = True
                for tid in range(threads):
                    if errors[tid]:
                        iteration_ok = False
                        continue
                    if results[tid] is None:
                        iteration_ok = False
                        continue
                    diff = (results[tid] - expected[tid]).abs().max().item()
                    max_diff = max(max_diff, diff)
                    if diff > 1e-4:
                        iteration_ok = False

                if iteration_ok:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                crashes += 1

        return passed, failed, crashes, max_diff

    def save_report(self):
        """Save the report to JSON and markdown."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = self.output_dir / "assumption_falsification_results.json"
        with open(json_path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)
        print(f"\nJSON report: {json_path}")

        # Markdown report
        md_path = self.output_dir / "assumption_falsification_report.md"
        with open(md_path, "w") as f:
            f.write(self._to_markdown())
        print(f"Markdown report: {md_path}")

        return json_path, md_path

    def _to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "timestamp": self.report.timestamp,
            "pytorch_version": self.report.pytorch_version,
            "mps_available": self.report.mps_available,
            "device_name": self.report.device_name,
            "summary": self.report.summary,
            "tests": [
                {
                    "name": t.name,
                    "assumption": t.assumption,
                    "workaround": t.workaround,
                    "test_mode": t.test_mode,
                    "iterations": t.iterations,
                    "threads": t.threads,
                    "passed": t.passed,
                    "failed": t.failed,
                    "crashes": t.crashes,
                    "max_diff": t.max_diff,
                    "result": t.result,
                    "evidence": t.evidence,
                    "duration_ms": t.duration_ms
                }
                for t in self.report.tests
            ]
        }

    def _to_markdown(self) -> str:
        """Convert report to markdown."""
        lines = [
            "# Assumption Falsification Report",
            "",
            f"**Date:** {self.report.timestamp}",
            f"**PyTorch Version:** {self.report.pytorch_version}",
            f"**MPS Available:** {self.report.mps_available}",
            f"**Device:** {self.report.device_name}",
            "",
            "## Summary",
            "",
            self.report.summary,
            "",
            "## Test Results",
            "",
        ]

        for test in self.report.tests:
            lines.extend([
                f"### {test.name}",
                "",
                f"**Assumption:** {test.assumption}",
                "",
                f"**Workaround:** {test.workaround}",
                "",
                f"**Test Configuration:**",
                f"- Mode: {test.test_mode}",
                f"- Iterations: {test.iterations}",
                f"- Threads: {test.threads}",
                f"- Duration: {test.duration_ms:.1f}ms",
                "",
                f"**Results:**",
                f"- Passed: {test.passed}",
                f"- Failed: {test.failed}",
                f"- Crashes: {test.crashes}",
                f"- Max Diff: {test.max_diff:.2e}",
                "",
                f"**Verdict: {test.result}**",
                "",
                f"**Evidence:** {test.evidence}",
                "",
                "---",
                "",
            ])

        return "\n".join(lines)


def main():
    print("=" * 70)
    print("Assumption Falsification Tests for MPS Parallel Inference")
    print("=" * 70)
    print()
    print("Purpose: Turn 'Apple is broken' from assumption into documented evidence")
    print()

    # Output directory
    output_dir = Path(__file__).parent.parent / "reports" / "main"

    # Run tests
    suite = AssumptionFalsificationSuite(output_dir)
    report = suite.run_all_tests()

    # Save results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(report.summary)

    suite.save_report()

    # Exit code based on whether bugs were proven
    bugs_proven = sum(
        1 for t in report.tests
        if "unprotected" in t.test_mode and t.result in ["FAIL", "CRASH"]
    )

    if bugs_proven > 0:
        print(f"\n{bugs_proven} Apple MPS bug(s) proven - workarounds necessary")
        return 0  # Success: we proved the bugs exist
    else:
        print("\nNo bugs proven this run (may be intermittent)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
