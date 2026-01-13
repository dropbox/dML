#!/usr/bin/env python3
"""Run 25 rounds of TransformerEncoderLayer verification."""
import torch
import torch.nn as nn
import threading
import sys

def run_test():
    """Run a single test round."""
    model = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128,
        batch_first=True, dropout=0
    ).to('mps').eval()

    x = torch.randn(1, 4, 64, device='mps')
    completed = [0]

    def worker(tid, iterations=20):
        for i in range(iterations):
            try:
                with torch.no_grad():
                    y = model(x)
                torch.mps.synchronize()
                completed[0] += 1
            except:
                pass

    threads = []
    for i in range(8):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return completed[0] == 160

# Run 25 rounds
passed = 0
for i in range(25):
    if run_test():
        passed += 1
        print(f"Round {i+1}: PASS")
    else:
        print(f"Round {i+1}: FAIL")

print(f"\nResult: {passed}/25 passed ({100*passed/25:.1f}%)")
sys.exit(0 if passed == 25 else 1)
