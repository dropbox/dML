#!/usr/bin/env python3
"""Stress test for v2.3 dylib - aggressive threading."""

import torch
import torch.nn as nn
import threading
import time
import sys

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def stress_test(num_threads=8, iterations=50, num_rounds=10):
    device = torch.device("mps")

    for round_num in range(num_rounds):
        models = [SmallModel().to(device).eval() for _ in range(num_threads)]
        completed = [0] * num_threads
        errors = []

        def worker(tid):
            try:
                for i in range(iterations):
                    x = torch.randn(8, 64, device=device)
                    with torch.no_grad():
                        y = models[tid](x)
                    torch.mps.synchronize()
                    completed[tid] += 1
            except Exception as e:
                errors.append((tid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        start = time.time()
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
        time.sleep(0.1)

    return True

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    success = stress_test()
    print()
    print("RESULT:", "PASS" if success else "FAIL")
    sys.exit(0 if success else 1)
