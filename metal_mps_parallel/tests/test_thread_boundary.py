#!/usr/bin/env python3
"""Find the thread count boundary for nn.Module parallel inference.

Each thread count is tested in a separate subprocess to reduce cross-test
state and improve reproducibility.
"""
import subprocess
import sys
import os


TEST_CODE = '''
import threading
import torch
import torch.nn as nn

n_threads = {n_threads}
n_iters = {n_iters}

# Warmup
x = torch.randn(100, 100, device='mps')
y = torch.randn(100, 100, device='mps')
torch.mm(x, y)
torch.mps.synchronize()

model = nn.Linear(256, 256).to('mps')
model.eval()
with torch.no_grad():
    _ = model(torch.randn(4, 256, device='mps'))
    torch.mps.synchronize()

results = []
errors = []
lock = threading.Lock()

def worker(tid):
    try:
        for i in range(n_iters):
            with torch.no_grad():
                out = model(torch.randn(4, 256, device='mps'))
                torch.mps.synchronize()
            with lock:
                results.append(1)
    except Exception as e:
        with lock:
            errors.append(str(e)[:100])

threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
for t in threads:
    t.start()
for t in threads:
    t.join()

expected = n_threads * n_iters
if len(results) == expected and len(errors) == 0:
    print("PASS")
else:
    print(f"FAIL: {{len(results)}}/{{expected}}, errors: {{len(errors)}}")
    if errors:
        print(f"  {{errors[0]}}")
'''


def run_threads_isolated(n_threads, n_iters=10):
    """Run thread test in isolated subprocess to avoid stream pool exhaustion."""
    code = TEST_CODE.format(n_threads=n_threads, n_iters=n_iters)
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(__file__))}
    )
    output = result.stdout.strip()
    return output.startswith("PASS"), output


def test_threads_isolated():
    """Pytest-compatible test that verifies thread boundary behavior."""
    passed, output = run_threads_isolated(4, 10)
    assert passed, f"Thread boundary test failed: {output}"


if __name__ == '__main__':
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Testing thread counts for nn.Linear parallel inference...")
    print(f"(Each test runs in isolated subprocess)")

    # Test incremental thread counts
    for n in [2, 3, 4, 5, 6, 7, 8]:
        try:
            passed, output = run_threads_isolated(n, 10)
            status = "PASS" if passed else "FAIL"
            print(f'{n} threads: {status}')
            if not passed:
                print(f'  {output}')
                break
        except subprocess.TimeoutExpired:
            print(f'{n} threads: TIMEOUT')
            break
        except Exception as e:
            print(f'{n} threads: CRASH ({type(e).__name__}: {e})')
            break
