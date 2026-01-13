#!/usr/bin/env python3
"""Test TransformerEncoderLayer with threading - isolate the crash."""

import torch
import torch.nn as nn
import threading
import time
import sys

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model*4,
                batch_first=True, dropout=0
            )
            for _ in range(layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def test_transformer_threads(num_rounds=10):
    device = torch.device("mps")
    num_threads = 8
    iterations = 20

    for round_num in range(num_rounds):
        models = [TransformerBlock().to(device).eval() for _ in range(num_threads)]
        completed = [0] * num_threads
        errors = []

        def worker(tid):
            try:
                for i in range(iterations):
                    x = torch.randn(4, 32, 256, device=device)
                    with torch.no_grad():
                        y = models[tid](x)
                    torch.mps.synchronize()
                    completed[tid] += 1
            except Exception as e:
                errors.append((tid, str(e)))

        start = time.time()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        total = sum(completed)
        expected = num_threads * iterations

        if errors:
            print(f"Round {round_num + 1}: FAIL - {len(errors)} errors")
            for tid, err in errors[:3]:
                print(f"  Thread {tid}: {err}")
            return False
        elif total != expected:
            print(f"Round {round_num + 1}: INCOMPLETE {total}/{expected}")
            return False
        else:
            print(f"Round {round_num + 1}: PASS ({total} ops in {elapsed:.2f}s)")

        # Cleanup
        del models
        torch.mps.empty_cache()
        import gc
        gc.collect()

    return True

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print()
    success = test_transformer_threads()
    print()
    print("RESULT:", "PASS" if success else "FAIL")
    sys.exit(0 if success else 1)
