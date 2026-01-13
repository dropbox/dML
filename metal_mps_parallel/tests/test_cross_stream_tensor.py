#!/usr/bin/env python3
"""
Test cross-stream tensor correctness.

THREAD SAFETY GUIDANCE:
- Tensors produced on one stream CAN be consumed on another stream AFTER synchronization
- Without explicit synchronization, cross-stream tensor access is UNDEFINED BEHAVIOR
- This test validates the CORRECT usage pattern with explicit sync
"""
import torch
import threading
import queue

def test_cross_stream_with_sync():
    """Test that tensors can be safely passed between threads WITH synchronization."""
    print("Testing cross-stream tensor with explicit synchronization...")
    
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    tensor_queue = queue.Queue()
    
    def producer():
        try:
            # Produce tensor on this thread's stream
            x = torch.randn(64, 64, device='mps')
            y = torch.matmul(x, x)
            # CRITICAL: Synchronize before passing to another thread
            torch.mps.synchronize()
            tensor_queue.put(y.clone())  # Clone to ensure data is complete
        except Exception as e:
            error_queue.put(f"Producer error: {e}")
    
    def consumer():
        try:
            # Wait for tensor from producer
            y = tensor_queue.get(timeout=5.0)
            # Tensor was synced by producer, safe to use
            z = torch.matmul(y, y)
            torch.mps.synchronize()
            result_queue.put(z.sum().item())
        except Exception as e:
            error_queue.put(f"Consumer error: {e}")
    
    # Run producer first, then consumer
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()
    producer_thread.join()
    
    consumer_thread = threading.Thread(target=consumer)
    consumer_thread.start()
    consumer_thread.join()
    
    if not error_queue.empty():
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())
        assert False, f"Errors: {errors}"

    result = result_queue.get()
    print(f"PASS: Cross-stream tensor result = {result:.4f}")


def test_cross_stream_documentation():
    """Document the expected behavior for cross-stream tensors."""
    print("\nCross-stream tensor safety rules:")
    print("1. ALWAYS call torch.mps.synchronize() before passing tensor to another thread")
    print("2. Use tensor.clone() when passing between threads for safety")
    print("3. NEVER mutate a tensor while another thread may be reading it")
    print("4. Each thread should only operate on tensors it owns or tensors that are fully synchronized")
    # This is a documentation-only test - always passes


if __name__ == "__main__":
    success = True

    try:
        test_cross_stream_with_sync()
    except AssertionError as e:
        print(f"FAILED: {e}")
        success = False

    test_cross_stream_documentation()

    if success:
        print("\nAll cross-stream tensor tests PASSED")
    else:
        print("\nSome tests FAILED")
        exit(1)
