#!/usr/bin/env python3
"""
Quick soak test for MPS parallel inference stability.
Runs for 60 seconds with 8 threads.
"""

import torch
import threading
import time
import sys

def main():
    print('=== 60-Second Soak Test ===')
    print('Testing stability over extended period')
    print()

    stop_flag = threading.Event()
    success_count = [0]
    error_count = [0]
    lock = threading.Lock()

    def worker(worker_id):
        x = torch.randn(128, 128, device='mps')
        local_success = 0
        local_error = 0
        while not stop_flag.is_set():
            try:
                y = torch.matmul(x, x)
                torch.mps.synchronize()
                local_success += 1
            except Exception as e:
                local_error += 1
                print(f'Worker {worker_id} error: {e}')

        with lock:
            success_count[0] += local_success
            error_count[0] += local_error

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    start = time.time()
    for t in threads:
        t.start()

    # Run for 60 seconds
    duration = 60
    print(f'Running for {duration}s with 8 threads...')
    time.sleep(duration)
    stop_flag.set()

    for t in threads:
        t.join()

    elapsed = time.time() - start
    print()
    print(f'Elapsed: {elapsed:.1f}s')
    print(f'Total ops: {success_count[0]}')
    print(f'Throughput: {success_count[0]/elapsed:.1f} ops/s')
    print(f'Errors: {error_count[0]}')
    print(f'Result: {"PASS" if error_count[0] == 0 else "FAIL"}')

    return 0 if error_count[0] == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
