#!/usr/bin/env python3
"""
Stress test for AGX Fix Comprehensive - Tests ALL protected methods.
Run with: DYLD_INSERT_LIBRARIES=build/libagx_fix_comprehensive.dylib python3 tests/stress_test_comprehensive.py
"""

import os
import sys
import torch
import torch.nn as nn
import threading
import time
import ctypes
from concurrent.futures import ThreadPoolExecutor

# Verify we have MPS
if not torch.backends.mps.is_available():
    print("ERROR: MPS not available")
    sys.exit(1)

# Try to load stats from the dylib
try:
    dylib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libagx_fix_comprehensive.dylib')
    if os.path.exists(dylib_path):
        agx_fix = ctypes.CDLL(dylib_path)
        agx_fix.agx_fix_get_acquisitions.restype = ctypes.c_uint64
        agx_fix.agx_fix_get_contentions.restype = ctypes.c_uint64
        agx_fix.agx_fix_get_null_impl_skips.restype = ctypes.c_uint64
        agx_fix.agx_fix_get_swizzled_count.restype = ctypes.c_size_t
        HAS_STATS = True
    else:
        HAS_STATS = False
except:
    HAS_STATS = False

print("=" * 60)
print("AGX FIX COMPREHENSIVE STRESS TEST")
print("=" * 60)

if HAS_STATS:
    print(f"Swizzled methods: {agx_fix.agx_fix_get_swizzled_count()}")

# Create a model that exercises many encoder methods
class StressModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)

# Create models and data
device = torch.device('mps')
models = [StressModel().to(device).eval() for _ in range(8)]
test_input = torch.randn(4, 3, 32, 32, device=device)

# Statistics
stats = {
    'total_inferences': 0,
    'errors': 0,
    'lock': threading.Lock()
}

def worker(model_idx, iterations):
    """Worker that hammers a model"""
    model = models[model_idx]
    local_count = 0
    local_errors = 0

    for _ in range(iterations):
        try:
            with torch.no_grad():
                # This exercises: setBuffer, setBytes, setTexture, dispatch*, etc.
                output = model(test_input)
                # Force synchronization to stress encoder lifecycle
                output.cpu()
            local_count += 1
        except Exception as e:
            local_errors += 1
            print(f"ERROR in worker {model_idx}: {e}")

    with stats['lock']:
        stats['total_inferences'] += local_count
        stats['errors'] += local_errors

# Run stress test
NUM_THREADS = 16
ITERATIONS_PER_THREAD = 100

print(f"\nRunning {NUM_THREADS} threads, {ITERATIONS_PER_THREAD} iterations each...")
print("This will stress all protected encoder methods.\n")

start = time.time()

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = []
    for i in range(NUM_THREADS):
        # Use modulo to share models across threads (increases contention)
        model_idx = i % len(models)
        futures.append(executor.submit(worker, model_idx, ITERATIONS_PER_THREAD))

    for f in futures:
        f.result()

elapsed = time.time() - start

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total inferences: {stats['total_inferences']}")
print(f"Errors: {stats['errors']}")
print(f"Time: {elapsed:.2f}s")
print(f"Throughput: {stats['total_inferences']/elapsed:.1f} inferences/sec")

if HAS_STATS:
    print(f"\nAGX Fix Stats:")
    print(f"  Mutex acquisitions: {agx_fix.agx_fix_get_acquisitions()}")
    print(f"  Mutex contentions: {agx_fix.agx_fix_get_contentions()}")
    print(f"  NULL impl skips: {agx_fix.agx_fix_get_null_impl_skips()}")

# Determine result
if stats['errors'] == 0:
    print("\n[PASS] No crashes detected!")
    sys.exit(0)
else:
    print(f"\n[FAIL] {stats['errors']} errors occurred!")
    sys.exit(1)
