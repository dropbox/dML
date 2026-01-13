#!/usr/bin/env python3
"""Basic threading test to validate with ThreadSanitizer."""
import torch
import threading
import time

def worker(thread_id, results, errors):
    try:
        for i in range(5):
            x = torch.randn(32, 32, device='mps')
            y = torch.matmul(x, x)
            torch.mps.synchronize()
        results[thread_id] = True
    except Exception as e:
        errors[thread_id] = str(e)

def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    n_threads = 2
    results = {}
    errors = {}
    
    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=worker, args=(i, results, errors))
        threads.append(t)
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        print(f"ERRORS: {errors}")
        return 1
    
    print(f"All {n_threads} threads completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())
