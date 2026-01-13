#!/usr/bin/env python3
"""
LayerNorm and Transformer stress tests for MPS parallel inference.

KEY FINDINGS (N=1961):
- TransformerEncoderLayer REQUIRES .eval() mode for multi-threaded inference
- Training mode triggers MTLCommandBuffer completion handler race condition
- Separate modules per thread is recommended for complex operations
- LayerNorm works with shared modules, but separate is safer

The AGX encoder fix (libagx_fix_v2.dylib) protects encoder methods.
The .eval() requirement is a separate issue with PyTorch's MPS backend.
"""
import threading
import time
import sys


def test_layernorm_stress(num_threads=8, iterations=50):
    """
    LayerNorm stress test - uses MPSGraph internally.
    """
    import torch
    import torch.nn as nn

    errors = []
    results = []
    lock = threading.Lock()

    # Create LayerNorm module per thread for safety
    # Using .eval() for consistency (LayerNorm has no dropout but this is good practice)
    hidden_size = 256
    layer_norms = [nn.LayerNorm(hidden_size).to('mps').eval() for _ in range(num_threads)]

    def worker(thread_id: int):
        my_layer = layer_norms[thread_id]  # Use thread's own layer
        try:
            for i in range(iterations):
                # Create input on MPS
                batch_size = 16
                seq_len = 32
                x = torch.randn(batch_size, seq_len, hidden_size, device='mps')

                # Apply LayerNorm (uses MPSGraph internally)
                y = my_layer(x)

                # Sync to ensure completion
                torch.mps.synchronize()

                with lock:
                    results.append((thread_id, i))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e)))

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    total_ops = num_threads * iterations

    print(f"\n=== LayerNorm Stress Test ===")
    print(f"Threads: {num_threads}")
    print(f"Iterations/thread: {iterations}")
    print(f"Total operations: {total_ops}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} ops/sec")
    print(f"Success rate: {100*len(results)/total_ops:.1f}%")

    if errors:
        print("\nFirst 5 errors:")
        for thread_id, error in errors[:5]:
            print(f"  Thread {thread_id}: {error}")
        return False

    return len(results) == total_ops


def test_transformer_stress(num_threads=8, iterations=20):
    """
    Transformer stress test - uses LayerNorm + MultiHeadAttention.
    """
    import torch
    import torch.nn as nn

    errors = []
    results = []
    lock = threading.Lock()

    # Create transformer encoder layer per thread (NOT shared)
    # IMPORTANT: Must use .eval() mode - training mode causes MTLCommandBuffer
    # completion handler race condition with multi-threaded MPS inference.
    d_model = 128
    nhead = 4
    encoder_layers = [
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        ).to('mps').eval()
        for _ in range(num_threads)
    ]

    def worker(thread_id: int):
        my_layer = encoder_layers[thread_id]  # Use thread's own layer
        try:
            for i in range(iterations):
                # Create input on MPS
                batch_size = 8
                seq_len = 16
                x = torch.randn(batch_size, seq_len, d_model, device='mps')

                # Apply TransformerEncoderLayer
                with torch.no_grad():
                    y = my_layer(x)

                # Sync to ensure completion
                torch.mps.synchronize()

                with lock:
                    results.append((thread_id, i))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e)))

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    total_ops = num_threads * iterations

    print(f"\n=== Transformer Stress Test ===")
    print(f"Threads: {num_threads}")
    print(f"Iterations/thread: {iterations}")
    print(f"Total operations: {total_ops}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} ops/sec")
    print(f"Success rate: {100*len(results)/total_ops:.1f}%")

    if errors:
        print("\nFirst 5 errors:")
        for thread_id, error in errors[:5]:
            print(f"  Thread {thread_id}: {error}")
        return False

    return len(results) == total_ops


def warmup():
    """Warmup MPS with single-threaded operations."""
    import torch
    import torch.nn as nn

    print("\n--- Warmup: Single-threaded MPS ---")
    x = torch.randn(8, 16, 128, device='mps')
    layer_norm = nn.LayerNorm(128).to('mps')
    y = layer_norm(x)
    torch.mps.synchronize()
    print("Warmup complete")


if __name__ == '__main__':
    print("=" * 60)
    print("MPS LayerNorm/Transformer Stress Tests")
    print("Testing MPSGraph race conditions")
    print("=" * 60)

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Warmup first
    warmup()

    results = {}

    # Test 1: LayerNorm stress
    print("\n" + "-" * 60)
    print("Test 1: LayerNorm (8 threads x 50 iterations)")
    results['layernorm_8t_x50'] = test_layernorm_stress(8, 50)

    # Test 2: Transformer stress
    print("\n" + "-" * 60)
    print("Test 2: Transformer (8 threads x 20 iterations)")
    results['transformer_8t_x20'] = test_transformer_stress(8, 20)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED (expected - MPSGraph race conditions)")
        sys.exit(1)
