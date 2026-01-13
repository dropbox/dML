// metal_stubs.h - Stub definitions for CBMC verification of MPS code
// These stubs replace Objective-C/Metal types that CBMC cannot handle directly.
//
// Purpose: Allow bounded model checking of the C++ synchronization logic
// without requiring actual Metal framework or Objective-C runtime.

#ifndef MPS_VERIFY_METAL_STUBS_H
#define MPS_VERIFY_METAL_STUBS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Metal Device Stubs
// ============================================================================

typedef struct MTLDevice_stub {
    bool is_valid;
    uint64_t recommended_max_working_set_size;
    bool has_unified_memory;
} MTLDevice_stub;

// ============================================================================
// Metal Buffer Stubs
// ============================================================================

typedef enum MTLResourceOptions_stub {
    MTLResourceStorageModeShared = 0,
    MTLResourceStorageModeManaged = 1,
    MTLResourceStorageModePrivate = 2,
    MTLResourceCPUCacheModeDefaultCache = 0,
    MTLResourceCPUCacheModeWriteCombined = 16,
    MTLResourceHazardTrackingModeDefault = 0,
    MTLResourceHazardTrackingModeUntracked = 256
} MTLResourceOptions_stub;

typedef struct MTLBuffer_stub {
    void* contents;
    size_t length;
    uint64_t gpu_address;
    MTLResourceOptions_stub resource_options;
    bool is_valid;
} MTLBuffer_stub;

// ============================================================================
// Metal Heap Stubs
// ============================================================================

typedef enum MTLHeapType_stub {
    MTLHeapTypeAutomatic = 0,
    MTLHeapTypePlacement = 1,
    MTLHeapTypeSparse = 2
} MTLHeapType_stub;

typedef struct MTLHeapDescriptor_stub {
    size_t size;
    MTLResourceOptions_stub resource_options;
    MTLHeapType_stub type;
} MTLHeapDescriptor_stub;

typedef struct MTLHeap_stub {
    size_t size;
    size_t used_size;
    size_t current_allocated_size;
    MTLResourceOptions_stub resource_options;
    MTLHeapType_stub type;
    bool is_valid;
} MTLHeap_stub;

// ============================================================================
// Metal Command Queue Stubs
// ============================================================================

typedef struct MTLCommandQueue_stub {
    MTLDevice_stub* device;
    int queue_id;
    bool is_valid;
} MTLCommandQueue_stub;

typedef struct MTLCommandBuffer_stub {
    MTLCommandQueue_stub* command_queue;
    int status; // 0=NotEnqueued, 1=Enqueued, 2=Committed, 3=Scheduled, 4=Completed, 5=Error
    bool is_valid;
} MTLCommandBuffer_stub;

typedef struct MTLComputeCommandEncoder_stub {
    MTLCommandBuffer_stub* command_buffer;
    bool is_valid;
} MTLComputeCommandEncoder_stub;

// ============================================================================
// Metal Event Stubs
// ============================================================================

typedef struct MTLEvent_stub {
    uint64_t signaled_value;
    MTLDevice_stub* device;
    bool is_valid;
} MTLEvent_stub;

typedef struct MTLSharedEvent_stub {
    uint64_t signaled_value;
    MTLDevice_stub* device;
    bool is_valid;
} MTLSharedEvent_stub;

// ============================================================================
// Objective-C Type Aliases for CBMC
// ============================================================================

// In actual code, these are Objective-C 'id' pointers
// For CBMC, we use plain C pointers to stub structures
typedef MTLDevice_stub* id_MTLDevice;
typedef MTLBuffer_stub* id_MTLBuffer;
typedef MTLHeap_stub* id_MTLHeap;
typedef MTLCommandQueue_stub* id_MTLCommandQueue;
typedef MTLCommandBuffer_stub* id_MTLCommandBuffer;
typedef MTLComputeCommandEncoder_stub* id_MTLComputeCommandEncoder;
typedef MTLEvent_stub* id_MTLEvent;
typedef MTLSharedEvent_stub* id_MTLSharedEvent;

// Generic 'id' type for unknown Objective-C objects
typedef void* objc_id;

// nil equivalent
#define nil_stub ((void*)0)

// ============================================================================
// Mock Metal API Functions
// ============================================================================

// Device creation
static inline id_MTLDevice MTLCreateSystemDefaultDevice_stub(void) {
    // CBMC will use nondet to explore all possibilities
    static MTLDevice_stub device;
    device.is_valid = true;
    device.recommended_max_working_set_size = 16ULL * 1024 * 1024 * 1024; // 16GB
    device.has_unified_memory = true;
    return &device;
}

// Buffer creation (simplified)
static inline id_MTLBuffer device_newBufferWithLength_stub(
    id_MTLDevice device,
    size_t length,
    MTLResourceOptions_stub options)
{
    if (!device || !device->is_valid || length == 0) {
        return nil_stub;
    }
    // In real verification, we'd use CBMC's malloc modeling
    static MTLBuffer_stub buffers[1024];
    static int buffer_count = 0;
    if (buffer_count >= 1024) return nil_stub;

    MTLBuffer_stub* buf = &buffers[buffer_count++];
    buf->length = length;
    buf->resource_options = options;
    buf->is_valid = true;
    buf->contents = (void*)(uintptr_t)(buffer_count * 0x1000); // Fake address
    buf->gpu_address = (uint64_t)(uintptr_t)buf->contents;
    return buf;
}

// Heap creation (simplified)
static inline id_MTLHeap device_newHeapWithDescriptor_stub(
    id_MTLDevice device,
    MTLHeapDescriptor_stub* descriptor)
{
    if (!device || !device->is_valid || !descriptor) {
        return nil_stub;
    }
    static MTLHeap_stub heaps[64];
    static int heap_count = 0;
    if (heap_count >= 64) return nil_stub;

    MTLHeap_stub* heap = &heaps[heap_count++];
    heap->size = descriptor->size;
    heap->used_size = 0;
    heap->current_allocated_size = 0;
    heap->resource_options = descriptor->resource_options;
    heap->type = descriptor->type;
    heap->is_valid = true;
    return heap;
}

// Buffer from heap (simplified)
static inline id_MTLBuffer heap_newBufferWithLength_stub(
    id_MTLHeap heap,
    size_t length,
    MTLResourceOptions_stub options)
{
    if (!heap || !heap->is_valid || length == 0) {
        return nil_stub;
    }
    if (heap->used_size + length > heap->size) {
        return nil_stub; // OOM
    }

    static MTLBuffer_stub heap_buffers[1024];
    static int heap_buffer_count = 0;
    if (heap_buffer_count >= 1024) return nil_stub;

    MTLBuffer_stub* buf = &heap_buffers[heap_buffer_count++];
    buf->length = length;
    buf->resource_options = options;
    buf->is_valid = true;
    buf->contents = (void*)(uintptr_t)((heap_buffer_count + 2000) * 0x1000);
    buf->gpu_address = (uint64_t)(uintptr_t)buf->contents;

    heap->used_size += length;
    heap->current_allocated_size += length;
    return buf;
}

// Buffer contents accessor
static inline void* buffer_contents_stub(id_MTLBuffer buffer) {
    if (!buffer || !buffer->is_valid) return nil_stub;
    return buffer->contents;
}

// Buffer length accessor
static inline size_t buffer_length_stub(id_MTLBuffer buffer) {
    if (!buffer || !buffer->is_valid) return 0;
    return buffer->length;
}

#ifdef __cplusplus
}
#endif

#endif // MPS_VERIFY_METAL_STUBS_H
