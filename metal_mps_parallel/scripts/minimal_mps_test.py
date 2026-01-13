#!/usr/bin/env python3
"""
Minimal MPS test for dtrace tracing.

Worker: N=1474
Purpose: Simple MPS operations to trace with dtrace

Usage:
  python3 scripts/minimal_mps_test.py
"""
import torch
import time

def main():
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    device = torch.device("mps")
    print(f"MPS device: {device}")

    # Create tensors
    a = torch.randn(100, 100, device=device)
    b = torch.randn(100, 100, device=device)

    # Do a few operations
    for i in range(5):
        c = torch.matmul(a, b)
        torch.mps.synchronize()
        print(f"Op {i+1} complete")
        time.sleep(0.1)

    print("Done")

if __name__ == "__main__":
    main()
