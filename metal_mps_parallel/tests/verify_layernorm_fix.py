#!/usr/bin/env python3
"""
Verification Script: LayerNorm Verification (MPS)

This script verifies that LayerNorm remains correct under multi-threaded MPS use.

Historically we saw two distinct issues:

1) Thread-affinity / thread-safety:
   Stock PyTorch MPS LayerNorm produced different results from different threads
   with identical inputs (diff=1.5-2.0, not floating-point noise).

2) Correctness-after-threading (N=1867):
   After multi-threaded execution, subsequent LayerNorm outputs could be corrupted
   (large CPU vs MPS diffs) due to an incorrect graph compilation path.
   Fixed by `patches/fix-layernorm-correctness-N1868.patch`.

USAGE:
    # On STOCK PyTorch (should FAIL - bug exists):
    python tests/verify_layernorm_fix.py

    # On PATCHED PyTorch (should PASS - fix works):
    python tests/verify_layernorm_fix.py

EXPECTED RESULTS:
    Stock PyTorch:   Either thread-consistency failure or CPU mismatch (FAIL)
    Patched PyTorch: Thread-consistency PASS and CPU reference match PASS
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import threading
import sys
import os
import pathlib

# Ensure tests directory is in path (allows running from any directory)
_tests_dir = pathlib.Path(__file__).parent.resolve()
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

# Check for AGX fix before running multi-threaded tests
from agx_fix_check import require_agx_fix_for_threading
require_agx_fix_for_threading()

def test_layernorm_thread_safety():
    """Test if LayerNorm produces identical results from different threads."""

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        return False

    # Create deterministic input on CPU
    torch.manual_seed(42)
    x_cpu = torch.randn(16, 512, 768)
    w_cpu = torch.randn(768)
    b_cpu = torch.randn(768)

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS_FORCE_GRAPH_PATH: {os.environ.get('MPS_FORCE_GRAPH_PATH', 'not set')}")
    print()

    # Run from main thread
    x1 = x_cpu.clone().to('mps')
    w1 = w_cpu.clone().to('mps')
    b1 = b_cpu.clone().to('mps')
    out_main = F.layer_norm(x1, [768], w1, b1)
    torch.mps.synchronize()
    out_main_cpu = out_main.cpu().clone()

    # Run from spawned thread
    result = [None]
    error = [None]

    def thread_fn():
        try:
            x2 = x_cpu.clone().to('mps')
            w2 = w_cpu.clone().to('mps')
            b2 = b_cpu.clone().to('mps')
            out = F.layer_norm(x2, [768], w2, b2)
            torch.mps.synchronize()
            result[0] = out.cpu().clone()
        except Exception as e:
            error[0] = str(e)

    t = threading.Thread(target=thread_fn)
    t.start()
    t.join()
    torch.mps.synchronize()

    if error[0]:
        print(f"ERROR in thread: {error[0]}")
        return False

    if result[0] is None:
        print("ERROR: Thread produced no output")
        return False

    # Compare
    diff = (out_main_cpu - result[0]).abs().max().item()

    print(f"LayerNorm Results:")
    print(f"  Main thread output[:3]:    {out_main_cpu[0,0,:3].tolist()}")
    print(f"  Spawned thread output[:3]: {result[0][0,0,:3].tolist()}")
    print(f"  Max absolute difference:   {diff:.6f}")
    print()

    # Threshold: differences > 0.001 indicate the bug
    # (normal FP noise is ~1e-6 for float32)
    if diff > 0.001:
        print(f"FAIL: LayerNorm has thread-affinity bug (diff={diff:.2f})")
        print("      This is EXPECTED on stock PyTorch.")
        print("      The fix (layer_norm_mps_graph) is needed.")
        return False
    else:
        print(f"PASS: LayerNorm is consistent across threads (diff={diff:.2e})")
        print("      Outputs are identical across threads.")
        return True


def test_layernorm_correctness_after_threading():
    """Test CPU vs MPS correctness after multi-threaded MPS usage."""

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        return False

    device = "mps"

    # Step 1: multi-threaded stress (this historically triggered corruption)
    models = [nn.LayerNorm(256).to(device).eval() for _ in range(2)]

    def worker(tid: int):
        for _ in range(10):
            x = torch.randn(4, 32, 256, device=device)
            _ = models[tid](x)
            torch.mps.synchronize()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Step 2: compare a fresh LayerNorm against CPU reference
    torch.manual_seed(0)
    x_cpu = torch.randn(4, 32, 256)

    ln_cpu = nn.LayerNorm(256).cpu().eval()
    ln_mps = nn.LayerNorm(256).to(device).eval()
    ln_mps.load_state_dict(ln_cpu.state_dict())

    y_cpu = ln_cpu(x_cpu)
    y_mps = ln_mps(x_cpu.to(device))
    torch.mps.synchronize()

    diff = (y_cpu - y_mps.cpu()).abs().max().item()

    y_mps_cpu = y_mps.cpu()
    print("LayerNorm Correctness After Threading:")
    print(f"  Max abs diff (CPU vs MPS): {diff:.6f}")
    print(f"  CPU mean/std: {y_cpu.mean().item():.6f} / {y_cpu.std().item():.6f}")
    print(f"  MPS mean/std: {y_mps_cpu.mean().item():.6f} / {y_mps_cpu.std().item():.6f}")
    print()

    if diff > 0.001:
        print(f"FAIL: LayerNorm is incorrect after multi-threading (diff={diff:.2f})")
        return False

    print(f"PASS: LayerNorm matches CPU after multi-threading (diff={diff:.2e})")
    return True


def test_other_ops_thread_safety():
    """Verify other ops are thread-safe (control group)."""

    torch.manual_seed(42)
    x_cpu = torch.randn(16, 512, 768)

    print("Control group (should all PASS):")

    ops = {
        'Softmax': lambda x: F.softmax(x, dim=-1),
        'GELU': lambda x: F.gelu(x),
        'ReLU': lambda x: F.relu(x),
    }

    all_pass = True
    for name, op in ops.items():
        # Main thread
        x1 = x_cpu.clone().to('mps')
        out_main = op(x1)
        torch.mps.synchronize()
        out_main_cpu = out_main.cpu().clone()

        # Spawned thread
        result = [None]
        def thread_fn():
            x2 = x_cpu.clone().to('mps')
            out = op(x2)
            torch.mps.synchronize()
            result[0] = out.cpu().clone()

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join()
        torch.mps.synchronize()

        diff = (out_main_cpu - result[0]).abs().max().item()
        status = "PASS" if diff == 0.0 else f"FAIL (diff={diff:.2e})"
        print(f"  {name}: {status}")
        if diff > 0.001:
            all_pass = False

    print()
    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("LayerNorm Thread-Safety Fix Verification")
    print("=" * 60)
    print()

    # Control group first
    control_pass = test_other_ops_thread_safety()

    # Main test
    layernorm_pass = test_layernorm_thread_safety()

    # Regression: correctness after multi-threaded execution
    layernorm_correctness_pass = test_layernorm_correctness_after_threading()

    print("=" * 60)
    if control_pass and layernorm_pass and layernorm_correctness_pass:
        print("VERIFICATION PASSED: LayerNorm is stable under MPS threading!")
        print("Thread-consistency and CPU reference checks passed.")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED: LayerNorm bug detected.")
        print("This is expected on STOCK PyTorch.")
        print("Build and install the PATCHED PyTorch to fix this.")
        sys.exit(1)
