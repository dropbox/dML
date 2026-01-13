/**
 * AGX Driver Race Condition Fix - Version 2.3
 *
 * COMBINES THE BEST OF v2.1 AND v2.2:
 *   - v2.2's retain-from-creation (fixes TLC Bug 4: pre-swizzle race)
 *   - v2.1's mutex protection during method calls (fixes data races)
 *
 * BUG IN v2.2:
 *   v2.2 removed mutex protection from encoder method calls, assuming
 *   retain-from-creation was sufficient. This caused crashes at 8 threads.
 *   The driver has internal races that require serialization.
 *
 * v2.3 ARCHITECTURE:
 *   1. Swizzle command buffer creation methods (computeCommandEncoder, etc.)
 *   2. CFRetain encoder immediately on creation (before ANY method call)
 *   3. ALL encoder method calls are mutex-protected (prevents driver races)
 *   4. CFRelease on endEncoding
 *
 * Created by Andrew Yates
 * Part of the MPS Parallel Inference research project
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mutex>
#import <atomic>
#import <unordered_set>
#import <os/log.h>

// ============================================================================
// Configuration
// ============================================================================

#define AGX_FIX_DISABLE_ENV "AGX_FIX_DISABLE"
#define AGX_FIX_VERBOSE_ENV "AGX_FIX_VERBOSE"

// ============================================================================
// Global State
// ============================================================================

namespace {
    // Single global mutex protects ALL shared state AND serializes driver calls
    // Recursive to handle nested calls
    std::recursive_mutex g_encoder_mutex;

    // Set of encoders we're keeping alive
    // Key: encoder pointer - retained via CFRetain on creation
    std::unordered_set<void*> g_active_encoders;

    // Statistics
    std::atomic<uint64_t> g_mutex_acquisitions{0};
    std::atomic<uint64_t> g_mutex_contentions{0};
    std::atomic<uint64_t> g_encoders_retained{0};
    std::atomic<uint64_t> g_encoders_released{0};
    std::atomic<uint64_t> g_null_impl_skips{0};
    std::atomic<uint64_t> g_method_calls{0};

    // Logging
    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    // Original method implementations - Command Buffer methods
    IMP g_original_computeCommandEncoder = nullptr;
    IMP g_original_computeCommandEncoderWithDescriptor = nullptr;
    IMP g_original_computeCommandEncoderWithDispatchType = nullptr;
    IMP g_original_blitCommandEncoder = nullptr;

    // Original method implementations - Encoder methods
    IMP g_original_endEncoding = nullptr;
    IMP g_original_destroyImpl = nullptr;

    // Blit encoder has separate originals (same selector names as compute encoder!)
    IMP g_original_blit_endEncoding = nullptr;
    IMP g_original_blit_deferredEndEncoding = nullptr;
    IMP g_original_blit_dealloc = nullptr;

    // Encoder method originals for _impl check
    constexpr int MAX_SWIZZLED = 128;
    SEL g_swizzled_sels[MAX_SWIZZLED] = {nullptr};
    IMP g_original_imps[MAX_SWIZZLED] = {nullptr};
    int g_swizzle_count = 0;

    IMP get_original_imp(SEL sel) {
        for (int i = 0; i < g_swizzle_count; i++) {
            if (g_swizzled_sels[i] == sel) return g_original_imps[i];
        }
        return nullptr;
    }

    void store_original_imp(SEL sel, IMP imp) {
        if (g_swizzle_count < MAX_SWIZZLED) {
            g_swizzled_sels[g_swizzle_count] = sel;
            g_original_imps[g_swizzle_count] = imp;
            g_swizzle_count++;
        } else {
            // FIX (Bug #175): Log error when limit is reached (was silently failing)
            if (g_log) {
                os_log_error(g_log, "AGX Fix v2.3: MAX_SWIZZLED overflow! count=%d, limit=%d, sel=%s",
                             g_swizzle_count, MAX_SWIZZLED, sel_getName(sel));
            }
        }
    }

    // AGX class info
    Class g_agx_encoder_class = nullptr;
    Class g_agx_command_buffer_class = nullptr;
    Class g_agx_blit_encoder_class = nullptr;
    Class g_agx_render_encoder_class = nullptr;
    Class g_agx_resource_state_encoder_class = nullptr;
    Class g_agx_accel_struct_encoder_class = nullptr;
    ptrdiff_t g_impl_ivar_offset = -1;

    // Render encoder original IMPs (separate from compute/blit due to same selector names)
    IMP g_original_renderCommandEncoderWithDescriptor = nullptr;
    IMP g_original_render_endEncoding = nullptr;
    IMP g_original_render_deferredEndEncoding = nullptr;
    IMP g_original_render_dealloc = nullptr;

    // Resource state encoder original IMPs
    IMP g_original_resourceStateCommandEncoder = nullptr;
    IMP g_original_resource_state_endEncoding = nullptr;
    IMP g_original_resource_state_dealloc = nullptr;
    // FIX (Bug #23): Dedicated IMP storage for updateFence/waitForFence (same selector as compute)
    IMP g_original_resource_state_updateFence = nullptr;
    IMP g_original_resource_state_waitForFence = nullptr;

    // Acceleration structure encoder original IMPs (raytracing)
    IMP g_original_accelerationStructureCommandEncoder = nullptr;
    IMP g_original_accel_struct_endEncoding = nullptr;
    IMP g_original_accel_struct_dealloc = nullptr;
    // FIX (Bug #23): Dedicated IMP storage for updateFence/waitForFence (same selector as compute)
    IMP g_original_accel_struct_updateFence = nullptr;
    IMP g_original_accel_struct_waitForFence = nullptr;
}

// ============================================================================
// Logging
// ============================================================================

#define AGX_LOG(format, ...) \
    do { if (g_verbose && g_log) os_log(g_log, format, ##__VA_ARGS__); } while(0)

#define AGX_LOG_ERROR(format, ...) \
    do { if (g_log) os_log_error(g_log, format, ##__VA_ARGS__); } while(0)

// ============================================================================
// Mutex Guard with Statistics
// ============================================================================

class AGXMutexGuard {
public:
    AGXMutexGuard() : locked_(false) {
        if (!g_enabled) return;
        if (g_encoder_mutex.try_lock()) {
            locked_ = true;
            g_mutex_acquisitions++;
        } else {
            g_mutex_contentions++;
            g_encoder_mutex.lock();
            locked_ = true;
            g_mutex_acquisitions++;
        }
    }
    ~AGXMutexGuard() {
        if (locked_) g_encoder_mutex.unlock();
    }
    AGXMutexGuard(const AGXMutexGuard&) = delete;
    AGXMutexGuard& operator=(const AGXMutexGuard&) = delete;
private:
    bool locked_;
};

// ============================================================================
// Encoder Lifetime Management (v2.3 = v2.2's creation-time retain)
// ============================================================================

// Retain encoder on creation - called from swizzled command buffer methods
// This is the KEY FIX from v2.2: encoder is retained BEFORE any method call
static void retain_encoder_on_creation(id encoder) {
    if (!encoder) return;

    AGXMutexGuard guard;
    void* ptr = (__bridge void*)encoder;

    // Check if already tracked (shouldn't happen, but be safe)
    if (g_active_encoders.count(ptr) > 0) {
        AGX_LOG("AGX Fix v2.3: Encoder %p already tracked", ptr);
        return;
    }

    // FIX (Bug #20 OOM safety): Insert first, only retain on success
    // If insert throws std::bad_alloc, we don't leak a retain
    try {
        g_active_encoders.insert(ptr);
    } catch (const std::bad_alloc&) {
        AGX_LOG_ERROR("AGX Fix v2.3: OOM inserting encoder %p", ptr);
        return;  // Don't retain if we can't track it
    }

    CFRetain((__bridge CFTypeRef)encoder);
    g_encoders_retained++;

    AGX_LOG("AGX Fix v2.3: Retained encoder %p on creation (total: %zu)",
            ptr, g_active_encoders.size());
}

// Release encoder on endEncoding - called from swizzled endEncoding
static void release_encoder_on_end(id encoder) {
    if (!encoder) return;

    // Note: caller already holds mutex (from AGXMutexGuard in swizzled_endEncoding)
    void* ptr = (__bridge void*)encoder;

    // Check if tracked
    auto it = g_active_encoders.find(ptr);
    if (it == g_active_encoders.end()) {
        AGX_LOG("AGX Fix v2.3: Encoder %p not tracked at endEncoding", ptr);
        return;
    }

    // Untrack and release
    g_active_encoders.erase(it);
    CFRelease((__bridge CFTypeRef)encoder);
    g_encoders_released++;

    AGX_LOG("AGX Fix v2.3: Released encoder %p at endEncoding (total: %zu)",
            ptr, g_active_encoders.size());
}


// ============================================================================
// _impl Validity Check
// ============================================================================

static bool is_impl_valid(id encoder) {
    if (g_impl_ivar_offset < 0) return true;

    char* obj_base = (char*)(__bridge void*)encoder;
    void** impl_ptr = (void**)(obj_base + g_impl_ivar_offset);
    void* impl = *impl_ptr;

    if (impl == nullptr) {
        g_null_impl_skips++;
        AGX_LOG("AGX Fix v2.3: NULL _impl in %p", encoder);
        return false;
    }
    return true;
}

// ============================================================================
// Swizzled Command Buffer Methods (CREATE encoders and retain immediately)
// ============================================================================

// computeCommandEncoder
static id swizzled_computeCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_computeCommandEncoder)(self, _cmd);

    if (encoder) {
        retain_encoder_on_creation(encoder);
    }

    return encoder;
}

// computeCommandEncoderWithDescriptor:
static id swizzled_computeCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_computeCommandEncoderWithDescriptor)(self, _cmd, descriptor);

    if (encoder) {
        retain_encoder_on_creation(encoder);
    }

    return encoder;
}

// computeCommandEncoderWithDispatchType:
static id swizzled_computeCommandEncoderWithDispatchType(id self, SEL _cmd, NSUInteger dispatchType) {
    typedef id (*Func)(id, SEL, NSUInteger);
    id encoder = ((Func)g_original_computeCommandEncoderWithDispatchType)(self, _cmd, dispatchType);

    if (encoder) {
        retain_encoder_on_creation(encoder);
    }

    return encoder;
}

// blitCommandEncoder - PyTorch uses this for fillBuffer and copyFromBuffer
static id swizzled_blitCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_blitCommandEncoder)(self, _cmd);

    if (encoder) {
        retain_encoder_on_creation(encoder);
    }

    return encoder;
}

// ============================================================================
// Swizzled Encoder Methods (v2.3 = v2.1's mutex protection RESTORED)
// ============================================================================

// Generic wrapper macro for encoder methods - INCLUDES MUTEX (unlike v2.2)
#define DEFINE_SWIZZLED_METHOD_VOID_0(name) \
static void swizzled_##name(id self, SEL _cmd) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL); \
        ((Func)original)(self, _cmd); \
    } \
}

#define DEFINE_SWIZZLED_METHOD_VOID_1(name, T1) \
static void swizzled_##name(id self, SEL _cmd, T1 a1) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1); \
        ((Func)original)(self, _cmd, a1); \
    } \
}

#define DEFINE_SWIZZLED_METHOD_VOID_2(name, T1, T2) \
static void swizzled_##name(id self, SEL _cmd, T1 a1, T2 a2) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1, T2); \
        ((Func)original)(self, _cmd, a1, a2); \
    } \
}

#define DEFINE_SWIZZLED_METHOD_VOID_3(name, T1, T2, T3) \
static void swizzled_##name(id self, SEL _cmd, T1 a1, T2 a2, T3 a3) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1, T2, T3); \
        ((Func)original)(self, _cmd, a1, a2, a3); \
    } \
}

#define DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE(name) \
static void swizzled_##name(id self, SEL _cmd, MTLSize a1, MTLSize a2) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, MTLSize, MTLSize); \
        ((Func)original)(self, _cmd, a1, a2); \
    } \
}

// Define swizzled methods
DEFINE_SWIZZLED_METHOD_VOID_1(setComputePipelineState, id)
DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE(dispatchThreads)
DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE(dispatchThreadgroups)

// setBuffer:offset:atIndex:
static void swizzled_setBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, buffer, offset, index);
    }
}

// setBuffers:offsets:withRange:
static void swizzled_setBuffers(id self, SEL _cmd, const id* buffers, const NSUInteger* offsets, NSRange range) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const id*, const NSUInteger*, NSRange);
        ((Func)original)(self, _cmd, buffers, offsets, range);
    }
}

// setBytes:length:atIndex:
static void swizzled_setBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, bytes, length, index);
    }
}

DEFINE_SWIZZLED_METHOD_VOID_2(setTexture, id, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(setTextures, const id*, NSRange)
DEFINE_SWIZZLED_METHOD_VOID_2(setSamplerState, id, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(setSamplerStates, const id*, NSRange)
DEFINE_SWIZZLED_METHOD_VOID_2(setThreadgroupMemoryLength, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(useResource, id, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_3(useResources, const id*, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_1(useHeap, id)
DEFINE_SWIZZLED_METHOD_VOID_2(useHeaps, const id*, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_1(memoryBarrierWithScope, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(memoryBarrierWithResources, const id*, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(executeCommandsInBuffer, id, NSRange)

// setStageInRegion:
static void swizzled_setStageInRegion(id self, SEL _cmd, MTLRegion region) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, MTLRegion);
        ((Func)original)(self, _cmd, region);
    }
}

DEFINE_SWIZZLED_METHOD_VOID_2(setImageblockWidth, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(setBufferOffset, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_1(updateFence, id)
DEFINE_SWIZZLED_METHOD_VOID_1(waitForFence, id)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchWaitFlush)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFlushInvalidate)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFlushOnly)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchInvalidateOnly)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFenceOnly)

// dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:
static void swizzled_dispatchThreadgroupsIndirect(id self, SEL _cmd, id buffer, NSUInteger offset, MTLSize tptg) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, MTLSize);
        ((Func)original)(self, _cmd, buffer, offset, tptg);
    }
}

// ============================================================================
// Swizzled BLIT Encoder Methods (PyTorch uses these for memory operations)
// ============================================================================

// fillBuffer:range:value: - used by PyTorch MPSStream.mm:409
static void swizzled_blit_fillBuffer(id self, SEL _cmd, id buffer, NSRange range, uint8_t value) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSRange, uint8_t);
        ((Func)original)(self, _cmd, buffer, range, value);
    }
}

// copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size: - used by PyTorch MPSStream.mm:448
static void swizzled_blit_copyFromBuffer(id self, SEL _cmd, id srcBuffer, NSUInteger srcOffset,
                                          id dstBuffer, NSUInteger dstOffset, NSUInteger size) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, srcBuffer, srcOffset, dstBuffer, dstOffset, size);
    }
}

// synchronizeResource: - common blit operation
static void swizzled_blit_synchronizeResource(id self, SEL _cmd, id resource) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id);
        ((Func)original)(self, _cmd, resource);
    }
}

// Blit encoder endEncoding - releases our retain
static void swizzled_blit_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    // Use dedicated storage (compute encoder has same selector!)
    if (g_original_blit_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_blit_endEncoding)(self, _cmd);
    }

    release_encoder_on_end(self);
}

// Blit encoder deferredEndEncoding - same as endEncoding
static void swizzled_blit_deferredEndEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    // Use dedicated storage (compute encoder has same selector!)
    if (g_original_blit_deferredEndEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_blit_deferredEndEncoding)(self, _cmd);
    }

    release_encoder_on_end(self);
}

// Blit encoder dealloc - force release if still tracked (abnormal termination)
// NOTE: Blit encoder has NO destroyImpl, so we swizzle dealloc instead
static void swizzled_blit_dealloc(id self, SEL _cmd) {
    // Note: Don't use AGXMutexGuard here as encoder is being deallocated
    // We just need to clean up our tracking
    void* ptr = (__bridge void*)self;

    {
        std::lock_guard<std::recursive_mutex> lock(g_encoder_mutex);
        auto it = g_active_encoders.find(ptr);
        if (it != g_active_encoders.end()) {
            g_active_encoders.erase(it);
            // DON'T CFRelease here - we're in dealloc, object is already being freed
            // Our extra retain just delayed the dealloc until now
            g_encoders_released++;
            AGX_LOG("AGX Fix v2.3: Cleaned up blit encoder %p in dealloc", ptr);
        }
    }

    // Call original dealloc (use dedicated storage - compute encoder has same selector)
    if (g_original_blit_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_blit_dealloc)(self, _cmd);
    }
}

// ============================================================================
// Swizzled RENDER Encoder Methods (Gap 3 - LOW priority, not used by PyTorch)
// Added for completeness to cover all Metal encoder types
// ============================================================================

// renderCommandEncoderWithDescriptor: factory method
static id swizzled_renderCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_renderCommandEncoderWithDescriptor)(self, _cmd, descriptor);

    if (encoder) {
        retain_encoder_on_creation(encoder);
    }

    return encoder;
}

// Render encoder common methods - setRenderPipelineState:
static void swizzled_render_setRenderPipelineState(id self, SEL _cmd, id pipelineState) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id);
        ((Func)original)(self, _cmd, pipelineState);
    }
}

// setVertexBuffer:offset:atIndex:
static void swizzled_render_setVertexBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, buffer, offset, index);
    }
}

// setFragmentBuffer:offset:atIndex:
static void swizzled_render_setFragmentBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, buffer, offset, index);
    }
}

// setVertexBytes:length:atIndex:
static void swizzled_render_setVertexBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, bytes, length, index);
    }
}

// setFragmentBytes:length:atIndex:
static void swizzled_render_setFragmentBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, bytes, length, index);
    }
}

// setVertexTexture:atIndex:
static void swizzled_render_setVertexTexture(id self, SEL _cmd, id texture, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger);
        ((Func)original)(self, _cmd, texture, index);
    }
}

// setFragmentTexture:atIndex:
static void swizzled_render_setFragmentTexture(id self, SEL _cmd, id texture, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger);
        ((Func)original)(self, _cmd, texture, index);
    }
}

// drawPrimitives:vertexStart:vertexCount:
static void swizzled_render_drawPrimitives(id self, SEL _cmd, NSUInteger primitiveType, NSUInteger vertexStart, NSUInteger vertexCount) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, primitiveType, vertexStart, vertexCount);
    }
}

// drawPrimitives:vertexStart:vertexCount:instanceCount:
static void swizzled_render_drawPrimitivesInstanced(id self, SEL _cmd, NSUInteger primitiveType, NSUInteger vertexStart, NSUInteger vertexCount, NSUInteger instanceCount) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, primitiveType, vertexStart, vertexCount, instanceCount);
    }
}

// drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:
static void swizzled_render_drawIndexedPrimitives(id self, SEL _cmd, NSUInteger primitiveType, NSUInteger indexCount, NSUInteger indexType, id indexBuffer, NSUInteger indexBufferOffset) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger, id, NSUInteger);
        ((Func)original)(self, _cmd, primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset);
    }
}

// Render encoder endEncoding - releases our retain
static void swizzled_render_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    // Use dedicated storage (other encoders have same selector!)
    if (g_original_render_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_render_endEncoding)(self, _cmd);
    }

    release_encoder_on_end(self);
}

// Render encoder deferredEndEncoding
static void swizzled_render_deferredEndEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_render_deferredEndEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_render_deferredEndEncoding)(self, _cmd);
    }

    release_encoder_on_end(self);
}

// Render encoder dealloc - cleanup fallback
static void swizzled_render_dealloc(id self, SEL _cmd) {
    void* ptr = (__bridge void*)self;

    {
        std::lock_guard<std::recursive_mutex> lock(g_encoder_mutex);
        auto it = g_active_encoders.find(ptr);
        if (it != g_active_encoders.end()) {
            g_active_encoders.erase(it);
            g_encoders_released++;
            AGX_LOG("AGX Fix v2.3: Cleaned up render encoder %p in dealloc", ptr);
        }
    }

    if (g_original_render_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_render_dealloc)(self, _cmd);
    }
}

// ============================================================================
// Swizzled RESOURCE STATE Encoder Methods (Gap 3 - LOW priority)
// Used for sparse texture management, not used by PyTorch
// ============================================================================

// resourceStateCommandEncoder factory method
static id swizzled_resourceStateCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_resourceStateCommandEncoder)(self, _cmd);

    if (encoder) {
        retain_encoder_on_creation(encoder);
    }

    return encoder;
}

// updateTextureMappings:mode:regions:mipLevels:slices:numRegions:
static void swizzled_resource_state_updateTextureMappings(id self, SEL _cmd, id texture,
    NSUInteger mode, const MTLRegion* regions, const NSUInteger* mipLevels,
    const NSUInteger* slices, NSUInteger numRegions) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, const MTLRegion*, const NSUInteger*,
                             const NSUInteger*, NSUInteger);
        ((Func)original)(self, _cmd, texture, mode, regions, mipLevels, slices, numRegions);
    }
}

// updateTextureMapping:mode:region:mipLevel:slice:
static void swizzled_resource_state_updateTextureMapping(id self, SEL _cmd, id texture,
    NSUInteger mode, MTLRegion region, NSUInteger mipLevel, NSUInteger slice) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, MTLRegion, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, texture, mode, region, mipLevel, slice);
    }
}

// Resource state encoder updateFence:
// FIX (Bug #23): Use dedicated IMP storage - selector is same as compute encoder
static void swizzled_resource_state_updateFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    if (g_original_resource_state_updateFence) {
        typedef void (*Func)(id, SEL, id);
        ((Func)g_original_resource_state_updateFence)(self, _cmd, fence);
    }
}

// Resource state encoder waitForFence:
// FIX (Bug #23): Use dedicated IMP storage - selector is same as compute encoder
static void swizzled_resource_state_waitForFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    if (g_original_resource_state_waitForFence) {
        typedef void (*Func)(id, SEL, id);
        ((Func)g_original_resource_state_waitForFence)(self, _cmd, fence);
    }
}

// Resource state encoder endEncoding - releases our retain
static void swizzled_resource_state_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_resource_state_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_resource_state_endEncoding)(self, _cmd);
    }

    release_encoder_on_end(self);
}

// Resource state encoder dealloc - cleanup fallback
static void swizzled_resource_state_dealloc(id self, SEL _cmd) {
    void* ptr = (__bridge void*)self;

    {
        std::lock_guard<std::recursive_mutex> lock(g_encoder_mutex);
        auto it = g_active_encoders.find(ptr);
        if (it != g_active_encoders.end()) {
            g_active_encoders.erase(it);
            g_encoders_released++;
            AGX_LOG("AGX Fix v2.3: Cleaned up resource state encoder %p in dealloc", ptr);
        }
    }

    if (g_original_resource_state_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_resource_state_dealloc)(self, _cmd);
    }
}

// ============================================================================
// Swizzled ACCELERATION STRUCTURE Encoder Methods (Gap 3 - LOW priority)
// Used for Metal raytracing, not used by PyTorch
// ============================================================================

// accelerationStructureCommandEncoder factory method
static id swizzled_accelerationStructureCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_accelerationStructureCommandEncoder)(self, _cmd);

    if (encoder) {
        retain_encoder_on_creation(encoder);
    }

    return encoder;
}

// buildAccelerationStructure:descriptor:scratchBuffer:scratchBufferOffset:
static void swizzled_accel_struct_build(id self, SEL _cmd, id accelStruct, id descriptor,
                                         id scratchBuffer, NSUInteger scratchBufferOffset) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id, id, NSUInteger);
        ((Func)original)(self, _cmd, accelStruct, descriptor, scratchBuffer, scratchBufferOffset);
    }
}

// refitAccelerationStructure:descriptor:destination:scratchBuffer:scratchBufferOffset:
static void swizzled_accel_struct_refit(id self, SEL _cmd, id sourceAccelStruct, id descriptor,
                                         id destAccelStruct, id scratchBuffer, NSUInteger scratchBufferOffset) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id, id, id, NSUInteger);
        ((Func)original)(self, _cmd, sourceAccelStruct, descriptor, destAccelStruct, scratchBuffer, scratchBufferOffset);
    }
}

// copyAccelerationStructure:toAccelerationStructure:
static void swizzled_accel_struct_copy(id self, SEL _cmd, id sourceAccelStruct, id destAccelStruct) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id);
        ((Func)original)(self, _cmd, sourceAccelStruct, destAccelStruct);
    }
}

// writeCompactedAccelerationStructureSize:toBuffer:offset:
static void swizzled_accel_struct_writeCompactedSize(id self, SEL _cmd, id accelStruct,
                                                      id buffer, NSUInteger offset) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id, NSUInteger);
        ((Func)original)(self, _cmd, accelStruct, buffer, offset);
    }
}

// Acceleration structure encoder updateFence:
// FIX (Bug #23): Use dedicated IMP storage - selector is same as compute encoder
static void swizzled_accel_struct_updateFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    if (g_original_accel_struct_updateFence) {
        typedef void (*Func)(id, SEL, id);
        ((Func)g_original_accel_struct_updateFence)(self, _cmd, fence);
    }
}

// Acceleration structure encoder waitForFence:
// FIX (Bug #23): Use dedicated IMP storage - selector is same as compute encoder
static void swizzled_accel_struct_waitForFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (!is_impl_valid(self)) return;
    if (g_original_accel_struct_waitForFence) {
        typedef void (*Func)(id, SEL, id);
        ((Func)g_original_accel_struct_waitForFence)(self, _cmd, fence);
    }
}

// Acceleration structure encoder endEncoding - releases our retain
static void swizzled_accel_struct_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_accel_struct_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_accel_struct_endEncoding)(self, _cmd);
    }

    release_encoder_on_end(self);
}

// Acceleration structure encoder dealloc - cleanup fallback
static void swizzled_accel_struct_dealloc(id self, SEL _cmd) {
    void* ptr = (__bridge void*)self;

    {
        std::lock_guard<std::recursive_mutex> lock(g_encoder_mutex);
        auto it = g_active_encoders.find(ptr);
        if (it != g_active_encoders.end()) {
            g_active_encoders.erase(it);
            g_encoders_released++;
            AGX_LOG("AGX Fix v2.3: Cleaned up accel struct encoder %p in dealloc", ptr);
        }
    }

    if (g_original_accel_struct_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_accel_struct_dealloc)(self, _cmd);
    }
}

// SPECIAL: endEncoding - releases our extra retain
static void swizzled_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    // Call original FIRST (must always call per Metal spec)
    if (g_original_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_endEncoding)(self, _cmd);
    }

    // Now release our extra retain
    release_encoder_on_end(self);
}

// SPECIAL: deferredEndEncoding - same as endEncoding
static void swizzled_deferredEndEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL);
        ((Func)original)(self, _cmd);
    }

    release_encoder_on_end(self);
}

// SPECIAL: destroyImpl - force release if still tracked
static void swizzled_destroyImpl(id self, SEL _cmd) {
    AGXMutexGuard guard;

    AGX_LOG("AGX Fix v2.3: destroyImpl on %p", self);

    // Force release if still tracked (encoder being destroyed before endEncoding)
    void* ptr = (__bridge void*)self;
    auto it = g_active_encoders.find(ptr);
    if (it != g_active_encoders.end()) {
        g_active_encoders.erase(it);
        CFRelease((__bridge CFTypeRef)self);
        g_encoders_released++;
        AGX_LOG("AGX Fix v2.3: Force released encoder %p at destroyImpl", ptr);
    }

    // Call original (outside our extra retain/release)
    typedef void (*Func)(id, SEL);
    ((Func)g_original_destroyImpl)(self, _cmd);
}

// ============================================================================
// Swizzle Helper
// ============================================================================

static bool swizzle_method(Class cls, SEL selector, IMP newImpl, IMP* outOriginal) {
    Method method = class_getInstanceMethod(cls, selector);
    if (!method) {
        return false;
    }
    *outOriginal = method_getImplementation(method);
    store_original_imp(selector, *outOriginal);
    method_setImplementation(method, newImpl);
    return true;
}

// ============================================================================
// Initialization
// ============================================================================

__attribute__((constructor))
static void agx_fix_v2_3_init() {
    g_log = os_log_create("com.agxfix.v2.3", "main");

    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix v2.3: Disabled via environment");
        return;
    }

    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    os_log(g_log, "AGX Fix v2.3: Initializing (retain-from-creation + mutex protection)");

    // Get Metal device and create test objects
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix v2.3: No Metal device");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (!encoder || !commandBuffer) {
        AGX_LOG_ERROR("AGX Fix v2.3: Failed to create test objects");
        return;
    }

    // Get classes
    g_agx_encoder_class = [encoder class];
    g_agx_command_buffer_class = [commandBuffer class];

    os_log(g_log, "AGX Fix v2.3: Encoder class: %s", class_getName(g_agx_encoder_class));
    os_log(g_log, "AGX Fix v2.3: Command buffer class: %s", class_getName(g_agx_command_buffer_class));

    [encoder endEncoding];

    // Discover blit encoder class (PyTorch uses this for fillBuffer and copyFromBuffer)
    id<MTLCommandBuffer> commandBuffer2 = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer2 blitCommandEncoder];
    if (blitEncoder) {
        g_agx_blit_encoder_class = [blitEncoder class];
        os_log(g_log, "AGX Fix v2.3: Blit encoder class: %s", class_getName(g_agx_blit_encoder_class));
        [blitEncoder endEncoding];
    }

    // Discover render encoder class (Gap 3 - LOW priority, not used by PyTorch)
    // Requires a render pass descriptor with valid texture attachment
    MTLRenderPassDescriptor* renderPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                       width:64 height:64 mipmapped:NO];
    texDesc.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> dummyTexture = [device newTextureWithDescriptor:texDesc];
    if (dummyTexture) {
        renderPassDesc.colorAttachments[0].texture = dummyTexture;
        renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionDontCare;

        id<MTLCommandBuffer> commandBuffer3 = [queue commandBuffer];
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer3 renderCommandEncoderWithDescriptor:renderPassDesc];
        if (renderEncoder) {
            g_agx_render_encoder_class = [renderEncoder class];
            os_log(g_log, "AGX Fix v2.3: Render encoder class: %s", class_getName(g_agx_render_encoder_class));
            [renderEncoder endEncoding];
        }
    }

    // Discover resource state encoder class (Gap 3 - LOW priority, used for sparse textures)
    id<MTLCommandBuffer> commandBuffer4 = [queue commandBuffer];
    SEL resourceStateSelector = @selector(resourceStateCommandEncoder);
    if ([commandBuffer4 respondsToSelector:resourceStateSelector]) {
        id resourceStateEncoder = [commandBuffer4 resourceStateCommandEncoder];
        if (resourceStateEncoder) {
            g_agx_resource_state_encoder_class = [resourceStateEncoder class];
            os_log(g_log, "AGX Fix v2.3: Resource state encoder class: %s", class_getName(g_agx_resource_state_encoder_class));
            [resourceStateEncoder endEncoding];
        }
    }

    // Discover acceleration structure encoder class (Gap 3 - LOW priority, used for raytracing)
    id<MTLCommandBuffer> commandBuffer5 = [queue commandBuffer];
    SEL accelStructSelector = @selector(accelerationStructureCommandEncoder);
    if ([commandBuffer5 respondsToSelector:accelStructSelector]) {
        id accelStructEncoder = [commandBuffer5 accelerationStructureCommandEncoder];
        if (accelStructEncoder) {
            g_agx_accel_struct_encoder_class = [accelStructEncoder class];
            os_log(g_log, "AGX Fix v2.3: Accel struct encoder class: %s", class_getName(g_agx_accel_struct_encoder_class));
            [accelStructEncoder endEncoding];
        }
    }

    // Discover _impl ivar offset
    Ivar implIvar = class_getInstanceVariable(g_agx_encoder_class, "_impl");
    if (implIvar) {
        g_impl_ivar_offset = ivar_getOffset(implIvar);
        os_log(g_log, "AGX Fix v2.3: _impl at offset %td", g_impl_ivar_offset);
    } else {
        Class parent = class_getSuperclass(g_agx_encoder_class);
        while (parent) {
            implIvar = class_getInstanceVariable(parent, "_impl");
            if (implIvar) {
                g_impl_ivar_offset = ivar_getOffset(implIvar);
                os_log(g_log, "AGX Fix v2.3: _impl at offset %td in parent %s",
                       g_impl_ivar_offset, class_getName(parent));
                break;
            }
            parent = class_getSuperclass(parent);
        }
    }

    int swizzled_count = 0;
    IMP dummy;

    // ========================================================================
    // PART 1 (from v2.2): Swizzle COMMAND BUFFER encoder creation methods
    // This ensures encoders are retained BEFORE any method can be called
    // ========================================================================

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoder),
                       (IMP)swizzled_computeCommandEncoder, &g_original_computeCommandEncoder)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.3: Swizzled computeCommandEncoder");
    }

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoderWithDescriptor:),
                       (IMP)swizzled_computeCommandEncoderWithDescriptor, &g_original_computeCommandEncoderWithDescriptor)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.3: Swizzled computeCommandEncoderWithDescriptor:");
    }

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoderWithDispatchType:),
                       (IMP)swizzled_computeCommandEncoderWithDispatchType, &g_original_computeCommandEncoderWithDispatchType)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.3: Swizzled computeCommandEncoderWithDispatchType:");
    }

    // Swizzle blitCommandEncoder (PyTorch uses this for fillBuffer and copyFromBuffer)
    if (swizzle_method(g_agx_command_buffer_class, @selector(blitCommandEncoder),
                       (IMP)swizzled_blitCommandEncoder, &g_original_blitCommandEncoder)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.3: Swizzled blitCommandEncoder");
    }

    // ========================================================================
    // PART 2 (from v2.1): Swizzle encoder methods WITH MUTEX PROTECTION
    // This prevents driver-internal races during method execution
    // ========================================================================

    if (swizzle_method(g_agx_encoder_class, @selector(destroyImpl),
                       (IMP)swizzled_destroyImpl, &g_original_destroyImpl)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.3: Swizzled destroyImpl");
    }

    if (swizzle_method(g_agx_encoder_class, @selector(endEncoding),
                       (IMP)swizzled_endEncoding, &g_original_endEncoding)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.3: Swizzled endEncoding");
    }

    #define SWIZZLE(sel, func) \
        if (swizzle_method(g_agx_encoder_class, sel, (IMP)func, &dummy)) swizzled_count++;

    SWIZZLE(@selector(setComputePipelineState:), swizzled_setComputePipelineState)
    SWIZZLE(@selector(dispatchThreads:threadsPerThreadgroup:), swizzled_dispatchThreads)
    SWIZZLE(@selector(dispatchThreadgroups:threadsPerThreadgroup:), swizzled_dispatchThreadgroups)
    SWIZZLE(@selector(setBuffer:offset:atIndex:), swizzled_setBuffer)
    SWIZZLE(@selector(setBuffers:offsets:withRange:), swizzled_setBuffers)
    SWIZZLE(@selector(setBytes:length:atIndex:), swizzled_setBytes)
    SWIZZLE(@selector(setTexture:atIndex:), swizzled_setTexture)
    SWIZZLE(@selector(setTextures:withRange:), swizzled_setTextures)
    SWIZZLE(@selector(setSamplerState:atIndex:), swizzled_setSamplerState)
    SWIZZLE(@selector(setSamplerStates:withRange:), swizzled_setSamplerStates)
    SWIZZLE(@selector(setThreadgroupMemoryLength:atIndex:), swizzled_setThreadgroupMemoryLength)
    SWIZZLE(@selector(useResource:usage:), swizzled_useResource)
    SWIZZLE(@selector(useResources:count:usage:), swizzled_useResources)
    SWIZZLE(@selector(useHeap:), swizzled_useHeap)
    SWIZZLE(@selector(useHeaps:count:), swizzled_useHeaps)
    SWIZZLE(@selector(memoryBarrierWithScope:), swizzled_memoryBarrierWithScope)
    SWIZZLE(@selector(memoryBarrierWithResources:count:), swizzled_memoryBarrierWithResources)
    SWIZZLE(@selector(executeCommandsInBuffer:withRange:), swizzled_executeCommandsInBuffer)
    SWIZZLE(@selector(setStageInRegion:), swizzled_setStageInRegion)
    SWIZZLE(@selector(setImageblockWidth:height:), swizzled_setImageblockWidth)
    SWIZZLE(@selector(deferredEndEncoding), swizzled_deferredEndEncoding)
    SWIZZLE(@selector(setBufferOffset:atIndex:), swizzled_setBufferOffset)
    SWIZZLE(@selector(updateFence:), swizzled_updateFence)
    SWIZZLE(@selector(waitForFence:), swizzled_waitForFence)
    SWIZZLE(@selector(dispatchWaitFlush), swizzled_dispatchWaitFlush)
    SWIZZLE(@selector(dispatchFlushInvalidate), swizzled_dispatchFlushInvalidate)
    SWIZZLE(@selector(dispatchFlushOnly), swizzled_dispatchFlushOnly)
    SWIZZLE(@selector(dispatchInvalidateOnly), swizzled_dispatchInvalidateOnly)
    SWIZZLE(@selector(dispatchFenceOnly), swizzled_dispatchFenceOnly)
    SWIZZLE(@selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:), swizzled_dispatchThreadgroupsIndirect)

    #undef SWIZZLE

    // ========================================================================
    // PART 3: Swizzle BLIT encoder methods
    // PyTorch uses fillBuffer and copyFromBuffer for memory operations
    // ========================================================================

    if (g_agx_blit_encoder_class) {
        IMP blit_dummy;

        // Blit encoder endEncoding (CRITICAL - releases our retain)
        // NOTE: Use dedicated storage - compute encoder has same selector!
        if (swizzle_method(g_agx_blit_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_blit_endEncoding, &g_original_blit_endEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled blit endEncoding");
        }

        // PyTorch-used blit methods (unique selectors - can use dummy)
        if (swizzle_method(g_agx_blit_encoder_class, @selector(fillBuffer:range:value:),
                           (IMP)swizzled_blit_fillBuffer, &blit_dummy)) {
            swizzled_count++;
        }

        if (swizzle_method(g_agx_blit_encoder_class,
                           @selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:),
                           (IMP)swizzled_blit_copyFromBuffer, &blit_dummy)) {
            swizzled_count++;
        }

        if (swizzle_method(g_agx_blit_encoder_class, @selector(synchronizeResource:),
                           (IMP)swizzled_blit_synchronizeResource, &blit_dummy)) {
            swizzled_count++;
        }

        // Blit encoder deferredEndEncoding - alternative encoding end
        // NOTE: Use dedicated storage - compute encoder has same selector!
        if (swizzle_method(g_agx_blit_encoder_class, @selector(deferredEndEncoding),
                           (IMP)swizzled_blit_deferredEndEncoding, &g_original_blit_deferredEndEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled blit deferredEndEncoding");
        }

        // Blit encoder dealloc - cleanup on abnormal termination
        // NOTE: Use dedicated storage - compute encoder has same selector!
        // NOTE: Blit encoder has NO destroyImpl, so we swizzle dealloc instead
        if (swizzle_method(g_agx_blit_encoder_class, @selector(dealloc),
                           (IMP)swizzled_blit_dealloc, &g_original_blit_dealloc)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled blit dealloc (cleanup fallback)");
        }

        os_log(g_log, "AGX Fix v2.3: Blit encoder methods swizzled");
    }

    // ========================================================================
    // PART 4: Swizzle RENDER encoder methods (Gap 3 - LOW priority)
    // Not used by PyTorch, but adds complete Metal encoder coverage
    // ========================================================================

    if (g_agx_render_encoder_class) {
        IMP render_dummy;

        // Render encoder factory on command buffer
        if (swizzle_method(g_agx_command_buffer_class, @selector(renderCommandEncoderWithDescriptor:),
                           (IMP)swizzled_renderCommandEncoderWithDescriptor, &g_original_renderCommandEncoderWithDescriptor)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled renderCommandEncoderWithDescriptor:");
        }

        // Render encoder endEncoding (CRITICAL - releases our retain)
        if (swizzle_method(g_agx_render_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_render_endEncoding, &g_original_render_endEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled render endEncoding");
        }

        // Common render encoder methods
        #define SWIZZLE_RENDER(sel, func) \
            if (swizzle_method(g_agx_render_encoder_class, sel, (IMP)func, &render_dummy)) swizzled_count++;

        SWIZZLE_RENDER(@selector(setRenderPipelineState:), swizzled_render_setRenderPipelineState)
        SWIZZLE_RENDER(@selector(setVertexBuffer:offset:atIndex:), swizzled_render_setVertexBuffer)
        SWIZZLE_RENDER(@selector(setFragmentBuffer:offset:atIndex:), swizzled_render_setFragmentBuffer)
        SWIZZLE_RENDER(@selector(setVertexBytes:length:atIndex:), swizzled_render_setVertexBytes)
        SWIZZLE_RENDER(@selector(setFragmentBytes:length:atIndex:), swizzled_render_setFragmentBytes)
        SWIZZLE_RENDER(@selector(setVertexTexture:atIndex:), swizzled_render_setVertexTexture)
        SWIZZLE_RENDER(@selector(setFragmentTexture:atIndex:), swizzled_render_setFragmentTexture)
        SWIZZLE_RENDER(@selector(drawPrimitives:vertexStart:vertexCount:), swizzled_render_drawPrimitives)
        SWIZZLE_RENDER(@selector(drawPrimitives:vertexStart:vertexCount:instanceCount:), swizzled_render_drawPrimitivesInstanced)
        SWIZZLE_RENDER(@selector(drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:), swizzled_render_drawIndexedPrimitives)

        #undef SWIZZLE_RENDER

        // Render encoder deferredEndEncoding
        if (swizzle_method(g_agx_render_encoder_class, @selector(deferredEndEncoding),
                           (IMP)swizzled_render_deferredEndEncoding, &g_original_render_deferredEndEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled render deferredEndEncoding");
        }

        // Render encoder dealloc - cleanup fallback
        if (swizzle_method(g_agx_render_encoder_class, @selector(dealloc),
                           (IMP)swizzled_render_dealloc, &g_original_render_dealloc)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled render dealloc (cleanup fallback)");
        }

        os_log(g_log, "AGX Fix v2.3: Render encoder methods swizzled");
    }

    // ========================================================================
    // PART 5: Swizzle RESOURCE STATE encoder methods (Gap 3 - LOW priority)
    // Used for sparse texture management, not used by PyTorch
    // ========================================================================

    if (g_agx_resource_state_encoder_class) {
        IMP resource_state_dummy;

        // Resource state encoder factory on command buffer
        if (swizzle_method(g_agx_command_buffer_class, @selector(resourceStateCommandEncoder),
                           (IMP)swizzled_resourceStateCommandEncoder, &g_original_resourceStateCommandEncoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled resourceStateCommandEncoder");
        }

        // Resource state encoder endEncoding (CRITICAL - releases our retain)
        if (swizzle_method(g_agx_resource_state_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_resource_state_endEncoding, &g_original_resource_state_endEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled resource state endEncoding");
        }

        // Resource state encoder methods
        #define SWIZZLE_RESOURCE_STATE(sel, func) \
            if (swizzle_method(g_agx_resource_state_encoder_class, sel, (IMP)func, &resource_state_dummy)) swizzled_count++;

        SWIZZLE_RESOURCE_STATE(@selector(updateTextureMappings:mode:regions:mipLevels:slices:numRegions:), swizzled_resource_state_updateTextureMappings)
        SWIZZLE_RESOURCE_STATE(@selector(updateTextureMapping:mode:region:mipLevel:slice:), swizzled_resource_state_updateTextureMapping)

        // FIX (Bug #23): Use dedicated IMP storage for updateFence/waitForFence (same selector as compute)
        if (swizzle_method(g_agx_resource_state_encoder_class, @selector(updateFence:),
                           (IMP)swizzled_resource_state_updateFence, &g_original_resource_state_updateFence)) {
            swizzled_count++;
        }
        if (swizzle_method(g_agx_resource_state_encoder_class, @selector(waitForFence:),
                           (IMP)swizzled_resource_state_waitForFence, &g_original_resource_state_waitForFence)) {
            swizzled_count++;
        }

        #undef SWIZZLE_RESOURCE_STATE

        // Resource state encoder dealloc - cleanup fallback
        if (swizzle_method(g_agx_resource_state_encoder_class, @selector(dealloc),
                           (IMP)swizzled_resource_state_dealloc, &g_original_resource_state_dealloc)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled resource state dealloc (cleanup fallback)");
        }

        os_log(g_log, "AGX Fix v2.3: Resource state encoder methods swizzled");
    }

    // ========================================================================
    // PART 6: Swizzle ACCELERATION STRUCTURE encoder methods (Gap 3 - final)
    // Used for Metal raytracing, not used by PyTorch
    // ========================================================================

    if (g_agx_accel_struct_encoder_class) {
        IMP accel_struct_dummy;

        // Acceleration structure encoder factory on command buffer
        if (swizzle_method(g_agx_command_buffer_class, @selector(accelerationStructureCommandEncoder),
                           (IMP)swizzled_accelerationStructureCommandEncoder, &g_original_accelerationStructureCommandEncoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled accelerationStructureCommandEncoder");
        }

        // Acceleration structure encoder endEncoding (CRITICAL - releases our retain)
        if (swizzle_method(g_agx_accel_struct_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_accel_struct_endEncoding, &g_original_accel_struct_endEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled accel struct endEncoding");
        }

        // Acceleration structure encoder methods
        #define SWIZZLE_ACCEL_STRUCT(sel, func) \
            if (swizzle_method(g_agx_accel_struct_encoder_class, sel, (IMP)func, &accel_struct_dummy)) swizzled_count++;

        SWIZZLE_ACCEL_STRUCT(@selector(buildAccelerationStructure:descriptor:scratchBuffer:scratchBufferOffset:), swizzled_accel_struct_build)
        SWIZZLE_ACCEL_STRUCT(@selector(refitAccelerationStructure:descriptor:destination:scratchBuffer:scratchBufferOffset:), swizzled_accel_struct_refit)
        SWIZZLE_ACCEL_STRUCT(@selector(copyAccelerationStructure:toAccelerationStructure:), swizzled_accel_struct_copy)
        SWIZZLE_ACCEL_STRUCT(@selector(writeCompactedAccelerationStructureSize:toBuffer:offset:), swizzled_accel_struct_writeCompactedSize)

        // FIX (Bug #23): Use dedicated IMP storage for updateFence/waitForFence (same selector as compute)
        if (swizzle_method(g_agx_accel_struct_encoder_class, @selector(updateFence:),
                           (IMP)swizzled_accel_struct_updateFence, &g_original_accel_struct_updateFence)) {
            swizzled_count++;
        }
        if (swizzle_method(g_agx_accel_struct_encoder_class, @selector(waitForFence:),
                           (IMP)swizzled_accel_struct_waitForFence, &g_original_accel_struct_waitForFence)) {
            swizzled_count++;
        }

        #undef SWIZZLE_ACCEL_STRUCT

        // Acceleration structure encoder dealloc - cleanup fallback
        if (swizzle_method(g_agx_accel_struct_encoder_class, @selector(dealloc),
                           (IMP)swizzled_accel_struct_dealloc, &g_original_accel_struct_dealloc)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.3: Swizzled accel struct dealloc (cleanup fallback)");
        }

        os_log(g_log, "AGX Fix v2.3: Acceleration structure encoder methods swizzled");
    }

    os_log(g_log, "AGX Fix v2.3: COMPLETE - %d methods protected (retain-from-creation + mutex)",
           swizzled_count);
}

// ============================================================================
// Statistics API
// ============================================================================

extern "C" {
    uint64_t agx_fix_v2_3_get_acquisitions() { return g_mutex_acquisitions.load(); }
    uint64_t agx_fix_v2_3_get_contentions() { return g_mutex_contentions.load(); }
    uint64_t agx_fix_v2_3_get_encoders_retained() { return g_encoders_retained.load(); }
    uint64_t agx_fix_v2_3_get_encoders_released() { return g_encoders_released.load(); }
    uint64_t agx_fix_v2_3_get_null_impl_skips() { return g_null_impl_skips.load(); }
    uint64_t agx_fix_v2_3_get_method_calls() { return g_method_calls.load(); }
    size_t agx_fix_v2_3_get_active_count() {
        std::lock_guard<std::recursive_mutex> lock(g_encoder_mutex);
        return g_active_encoders.size();
    }
    bool agx_fix_v2_3_is_enabled() { return g_enabled; }
}
