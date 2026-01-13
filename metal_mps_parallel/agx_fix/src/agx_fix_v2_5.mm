/**
 * AGX Driver Race Condition Fix - Version 2.5
 *
 * FIXES BUG #046 - MPSCommandBuffer encoders not tracked:
 *   v2.4 only swizzled AGXG16XFamilyCommandBuffer's encoder creation methods.
 *   But PyTorch MPS uses MPSCommandBuffer (from MetalPerformanceShaders framework):
 *     _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue].retain;
 *
 *   When PyTorch calls [mpsCommandBuffer computeCommandEncoder], it bypasses
 *   our swizzle entirely. The returned encoder is never tracked, leading to
 *   use-after-free when another thread ends the encoder.
 *
 * v2.5 CHANGES:
 *   1. Discover MPSCommandBuffer class at runtime
 *   2. Swizzle encoder creation methods on BOTH AGX and MPS command buffer classes
 *   3. All encoders are now tracked regardless of creation path
 *
 * Created by Andrew Yates
 * Part of the MPS Parallel Inference research project
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mutex>
#import <atomic>
#import <unordered_map>
#import <os/log.h>

// ============================================================================
// Configuration
// ============================================================================

#define AGX_FIX_DISABLE_ENV "AGX_FIX_DISABLE"
#define AGX_FIX_VERBOSE_ENV "AGX_FIX_VERBOSE"

// ============================================================================
// Per-Encoder State
// ============================================================================

namespace {
    // Note: All fields are protected by g_encoder_mutex - no per-field atomics needed
    struct EncoderState {
        int32_t active_calls = 0;   // Number of method calls in progress
        bool ended = false;          // endEncoding was called
        int32_t retain_count = 0;    // Our retain count (for debugging)

        EncoderState() = default;
    };
}

// ============================================================================
// Global State
// ============================================================================

namespace {
    // Single global mutex protects ALL shared state AND serializes driver calls
    // Recursive to handle nested calls
    std::recursive_mutex g_encoder_mutex;

    // Per-encoder state tracking
    // Key: encoder pointer
    // Value: EncoderState
    std::unordered_map<void*, EncoderState> g_encoder_states;

    // Statistics
    std::atomic<uint64_t> g_mutex_acquisitions{0};
    std::atomic<uint64_t> g_mutex_contentions{0};
    std::atomic<uint64_t> g_encoders_created{0};
    std::atomic<uint64_t> g_encoders_released{0};
    std::atomic<uint64_t> g_null_impl_skips{0};
    std::atomic<uint64_t> g_method_calls{0};
    std::atomic<uint64_t> g_deferred_releases{0};  // Releases that were deferred due to active calls
    std::atomic<uint64_t> g_agx_encoder_creates{0};  // Encoders created via AGX path
    std::atomic<uint64_t> g_mps_encoder_creates{0};  // Encoders created via MPS path (v2.5)

    // Logging
    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    // Original method implementations - AGX Command Buffer methods
    IMP g_original_computeCommandEncoder = nullptr;
    IMP g_original_computeCommandEncoderWithDescriptor = nullptr;
    IMP g_original_computeCommandEncoderWithDispatchType = nullptr;
    IMP g_original_blitCommandEncoder = nullptr;

    // Original method implementations - MPS Command Buffer methods (v2.5)
    IMP g_original_mps_computeCommandEncoder = nullptr;
    IMP g_original_mps_computeCommandEncoderWithDescriptor = nullptr;
    IMP g_original_mps_computeCommandEncoderWithDispatchType = nullptr;
    IMP g_original_mps_blitCommandEncoder = nullptr;
    IMP g_original_mps_renderCommandEncoderWithDescriptor = nullptr;
    IMP g_original_mps_resourceStateCommandEncoder = nullptr;
    IMP g_original_mps_accelerationStructureCommandEncoder = nullptr;

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
            if (g_log) {
                os_log_error(g_log, "AGX Fix v2.5: MAX_SWIZZLED overflow! count=%d, limit=%d, sel=%s",
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

    // MPS class info (v2.5)
    Class g_mps_command_buffer_class = nullptr;

    // Render encoder original IMPs
    IMP g_original_renderCommandEncoderWithDescriptor = nullptr;
    IMP g_original_render_endEncoding = nullptr;
    IMP g_original_render_deferredEndEncoding = nullptr;
    IMP g_original_render_dealloc = nullptr;

    // Resource state encoder original IMPs
    IMP g_original_resourceStateCommandEncoder = nullptr;
    IMP g_original_resource_state_endEncoding = nullptr;
    IMP g_original_resource_state_dealloc = nullptr;
    IMP g_original_resource_state_updateFence = nullptr;
    IMP g_original_resource_state_waitForFence = nullptr;

    // Acceleration structure encoder original IMPs
    IMP g_original_accelerationStructureCommandEncoder = nullptr;
    IMP g_original_accel_struct_endEncoding = nullptr;
    IMP g_original_accel_struct_dealloc = nullptr;
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
// Encoder Lifetime Management (v2.4 - Active Call Tracking)
// ============================================================================

// Called when encoder is created - sets up tracking and retains
static void encoder_created(id encoder) {
    if (!encoder) return;

    AGXMutexGuard guard;
    void* ptr = (__bridge void*)encoder;

    // Check if already tracked (shouldn't happen, but be safe)
    auto it = g_encoder_states.find(ptr);
    if (it != g_encoder_states.end()) {
        AGX_LOG("AGX Fix v2.5: Encoder %p already tracked", ptr);
        return;
    }

    // Create state and retain encoder
    try {
        g_encoder_states.emplace(ptr, EncoderState{});
    } catch (const std::bad_alloc&) {
        AGX_LOG_ERROR("AGX Fix v2.5: OOM creating state for encoder %p", ptr);
        return;
    }

    // Retain encoder - this keeps it alive until we explicitly release
    CFRetain((__bridge CFTypeRef)encoder);
    g_encoder_states[ptr].retain_count = 1;
    g_encoders_created++;

    AGX_LOG("AGX Fix v2.5: Created encoder %p (total: %zu)", ptr, g_encoder_states.size());
}

// Called at START of each method call - increments active count and retains
// Returns true if the encoder is valid and we should proceed
// This MUST be called while holding the mutex
static bool encoder_method_begin(id encoder) {
    if (!encoder) return false;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);

    if (it == g_encoder_states.end()) {
        // Encoder not tracked - might be created before our swizzle was installed
        // Try to track it now
        AGX_LOG("AGX Fix v2.5: Untracked encoder %p in method_begin, tracking now", ptr);
        try {
            g_encoder_states.emplace(ptr, EncoderState{});
            CFRetain((__bridge CFTypeRef)encoder);
            g_encoder_states[ptr].retain_count = 1;
            it = g_encoder_states.find(ptr);
        } catch (const std::bad_alloc&) {
            AGX_LOG_ERROR("AGX Fix v2.5: OOM tracking encoder %p", ptr);
            return false;
        }
    }

    // Increment active call count
    it->second.active_calls++;

    // Additional retain to ensure encoder survives this method call
    CFRetain((__bridge CFTypeRef)encoder);
    it->second.retain_count++;

    return true;
}

// Called at END of each method call - decrements active count
// Releases encoder if ended AND no more active calls
// This MUST be called while holding the mutex
static void encoder_method_end(id encoder) {
    if (!encoder) return;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);

    if (it == g_encoder_states.end()) {
        AGX_LOG("AGX Fix v2.5: Untracked encoder %p in method_end", ptr);
        return;
    }

    // Decrement active call count
    int32_t remaining = --it->second.active_calls;

    // Release our method-level retain
    CFRelease((__bridge CFTypeRef)encoder);
    it->second.retain_count--;

    // If ended and no more active calls, release our creation-level retain
    if (it->second.ended && remaining == 0) {
        AGX_LOG("AGX Fix v2.5: Final release of encoder %p (deferred)", ptr);
        CFRelease((__bridge CFTypeRef)encoder);
        g_encoder_states.erase(it);
        g_encoders_released++;
        g_deferred_releases++;
    }
}

// Called when endEncoding is invoked - marks encoder as ended
// Releases if no active calls, otherwise defers release
// This MUST be called while holding the mutex
static void encoder_ended(id encoder) {
    if (!encoder) return;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);

    if (it == g_encoder_states.end()) {
        AGX_LOG("AGX Fix v2.5: Untracked encoder %p in encoder_ended", ptr);
        return;
    }

    // Mark as ended
    it->second.ended = true;

    // If no active calls, release now
    if (it->second.active_calls == 0) {
        AGX_LOG("AGX Fix v2.5: Immediate release of encoder %p", ptr);
        CFRelease((__bridge CFTypeRef)encoder);
        it->second.retain_count--;
        g_encoder_states.erase(it);
        g_encoders_released++;
    } else {
        AGX_LOG("AGX Fix v2.5: Deferring release of encoder %p (%d active calls)",
                ptr, it->second.active_calls);
    }
}

// Force release - called from destroyImpl or dealloc
static void encoder_force_release(id encoder) {
    if (!encoder) return;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);

    if (it == g_encoder_states.end()) {
        return;  // Not tracked
    }

    // Release all our retains
    int32_t count = it->second.retain_count;
    for (int32_t i = 0; i < count; i++) {
        CFRelease((__bridge CFTypeRef)encoder);
    }

    AGX_LOG("AGX Fix v2.5: Force released encoder %p (%d retains)", ptr, count);
    g_encoder_states.erase(it);
    g_encoders_released++;
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
        AGX_LOG("AGX Fix v2.5: NULL _impl in %p", encoder);
        return false;
    }
    return true;
}

// ============================================================================
// RAII Helper for Method Calls (v2.4)
// ============================================================================

class EncoderMethodScope {
public:
    EncoderMethodScope(id encoder) : encoder_(encoder), valid_(false) {
        if (!encoder_ || !g_enabled) return;

        // Note: We assume mutex is already held (AGXMutexGuard created before this)
        valid_ = encoder_method_begin(encoder_);
    }

    ~EncoderMethodScope() {
        if (valid_) {
            encoder_method_end(encoder_);
        }
    }

    bool is_valid() const { return valid_; }

    EncoderMethodScope(const EncoderMethodScope&) = delete;
    EncoderMethodScope& operator=(const EncoderMethodScope&) = delete;

private:
    id encoder_;
    bool valid_;
};

// ============================================================================
// Swizzled AGX Command Buffer Methods (CREATE encoders)
// ============================================================================

static id swizzled_computeCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_computeCommandEncoder)(self, _cmd);

    if (encoder) {
        encoder_created(encoder);
        g_agx_encoder_creates++;
    }

    return encoder;
}

static id swizzled_computeCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_computeCommandEncoderWithDescriptor)(self, _cmd, descriptor);

    if (encoder) {
        encoder_created(encoder);
        g_agx_encoder_creates++;
    }

    return encoder;
}

static id swizzled_computeCommandEncoderWithDispatchType(id self, SEL _cmd, NSUInteger dispatchType) {
    typedef id (*Func)(id, SEL, NSUInteger);
    id encoder = ((Func)g_original_computeCommandEncoderWithDispatchType)(self, _cmd, dispatchType);

    if (encoder) {
        encoder_created(encoder);
        g_agx_encoder_creates++;
    }

    return encoder;
}

static id swizzled_blitCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_blitCommandEncoder)(self, _cmd);

    if (encoder) {
        encoder_created(encoder);
        g_agx_encoder_creates++;
    }

    return encoder;
}

// ============================================================================
// Swizzled MPS Command Buffer Methods (v2.5 - CREATE encoders)
// ============================================================================

static id swizzled_mps_computeCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_mps_computeCommandEncoder)(self, _cmd);

    if (encoder) {
        encoder_created(encoder);
        g_mps_encoder_creates++;
        AGX_LOG("AGX Fix v2.5: MPS computeCommandEncoder created %p", encoder);
    }

    return encoder;
}

static id swizzled_mps_computeCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_mps_computeCommandEncoderWithDescriptor)(self, _cmd, descriptor);

    if (encoder) {
        encoder_created(encoder);
        g_mps_encoder_creates++;
        AGX_LOG("AGX Fix v2.5: MPS computeCommandEncoderWithDescriptor created %p", encoder);
    }

    return encoder;
}

static id swizzled_mps_computeCommandEncoderWithDispatchType(id self, SEL _cmd, NSUInteger dispatchType) {
    typedef id (*Func)(id, SEL, NSUInteger);
    id encoder = ((Func)g_original_mps_computeCommandEncoderWithDispatchType)(self, _cmd, dispatchType);

    if (encoder) {
        encoder_created(encoder);
        g_mps_encoder_creates++;
        AGX_LOG("AGX Fix v2.5: MPS computeCommandEncoderWithDispatchType created %p", encoder);
    }

    return encoder;
}

static id swizzled_mps_blitCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_mps_blitCommandEncoder)(self, _cmd);

    if (encoder) {
        encoder_created(encoder);
        g_mps_encoder_creates++;
        AGX_LOG("AGX Fix v2.5: MPS blitCommandEncoder created %p", encoder);
    }

    return encoder;
}

static id swizzled_mps_renderCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_mps_renderCommandEncoderWithDescriptor)(self, _cmd, descriptor);

    if (encoder) {
        encoder_created(encoder);
        g_mps_encoder_creates++;
        AGX_LOG("AGX Fix v2.5: MPS renderCommandEncoderWithDescriptor created %p", encoder);
    }

    return encoder;
}

static id swizzled_mps_resourceStateCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_mps_resourceStateCommandEncoder)(self, _cmd);

    if (encoder) {
        encoder_created(encoder);
        g_mps_encoder_creates++;
        AGX_LOG("AGX Fix v2.5: MPS resourceStateCommandEncoder created %p", encoder);
    }

    return encoder;
}

static id swizzled_mps_accelerationStructureCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_mps_accelerationStructureCommandEncoder)(self, _cmd);

    if (encoder) {
        encoder_created(encoder);
        g_mps_encoder_creates++;
        AGX_LOG("AGX Fix v2.5: MPS accelerationStructureCommandEncoder created %p", encoder);
    }

    return encoder;
}

// ============================================================================
// Swizzled Encoder Methods (v2.4 with active call tracking)
// ============================================================================

// Macro for simple void methods with no arguments
#define DEFINE_SWIZZLED_METHOD_VOID_0_V24(name) \
static void swizzled_##name(id self, SEL _cmd) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    EncoderMethodScope scope(self); \
    if (!scope.is_valid() || !is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL); \
        ((Func)original)(self, _cmd); \
    } \
}

// Macro for void methods with 1 argument
#define DEFINE_SWIZZLED_METHOD_VOID_1_V24(name, T1) \
static void swizzled_##name(id self, SEL _cmd, T1 a1) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    EncoderMethodScope scope(self); \
    if (!scope.is_valid() || !is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1); \
        ((Func)original)(self, _cmd, a1); \
    } \
}

// Macro for void methods with 2 arguments
#define DEFINE_SWIZZLED_METHOD_VOID_2_V24(name, T1, T2) \
static void swizzled_##name(id self, SEL _cmd, T1 a1, T2 a2) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    EncoderMethodScope scope(self); \
    if (!scope.is_valid() || !is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1, T2); \
        ((Func)original)(self, _cmd, a1, a2); \
    } \
}

// Macro for void methods with 3 arguments
#define DEFINE_SWIZZLED_METHOD_VOID_3_V24(name, T1, T2, T3) \
static void swizzled_##name(id self, SEL _cmd, T1 a1, T2 a2, T3 a3) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    EncoderMethodScope scope(self); \
    if (!scope.is_valid() || !is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1, T2, T3); \
        ((Func)original)(self, _cmd, a1, a2, a3); \
    } \
}

// Special macro for MTLSize, MTLSize
#define DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE_V24(name) \
static void swizzled_##name(id self, SEL _cmd, MTLSize a1, MTLSize a2) { \
    AGXMutexGuard guard; \
    g_method_calls++; \
    EncoderMethodScope scope(self); \
    if (!scope.is_valid() || !is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, MTLSize, MTLSize); \
        ((Func)original)(self, _cmd, a1, a2); \
    } \
}

// Define swizzled methods using v2.4 macros
DEFINE_SWIZZLED_METHOD_VOID_1_V24(setComputePipelineState, id)
DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE_V24(dispatchThreads)
DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE_V24(dispatchThreadgroups)

// setBuffer:offset:atIndex:
static void swizzled_setBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
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
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
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
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, bytes, length, index);
    }
}

DEFINE_SWIZZLED_METHOD_VOID_2_V24(setTexture, id, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(setTextures, const id*, NSRange)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(setSamplerState, id, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(setSamplerStates, const id*, NSRange)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(setThreadgroupMemoryLength, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(useResource, id, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_3_V24(useResources, const id*, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_1_V24(useHeap, id)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(useHeaps, const id*, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_1_V24(memoryBarrierWithScope, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(memoryBarrierWithResources, const id*, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(executeCommandsInBuffer, id, NSRange)

// setStageInRegion:
static void swizzled_setStageInRegion(id self, SEL _cmd, MTLRegion region) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, MTLRegion);
        ((Func)original)(self, _cmd, region);
    }
}

DEFINE_SWIZZLED_METHOD_VOID_2_V24(setImageblockWidth, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2_V24(setBufferOffset, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_1_V24(updateFence, id)
DEFINE_SWIZZLED_METHOD_VOID_1_V24(waitForFence, id)
DEFINE_SWIZZLED_METHOD_VOID_0_V24(dispatchWaitFlush)
DEFINE_SWIZZLED_METHOD_VOID_0_V24(dispatchFlushInvalidate)
DEFINE_SWIZZLED_METHOD_VOID_0_V24(dispatchFlushOnly)
DEFINE_SWIZZLED_METHOD_VOID_0_V24(dispatchInvalidateOnly)
DEFINE_SWIZZLED_METHOD_VOID_0_V24(dispatchFenceOnly)

// dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:
static void swizzled_dispatchThreadgroupsIndirect(id self, SEL _cmd, id buffer, NSUInteger offset, MTLSize tptg) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, MTLSize);
        ((Func)original)(self, _cmd, buffer, offset, tptg);
    }
}

// ============================================================================
// Swizzled BLIT Encoder Methods
// ============================================================================

static void swizzled_blit_fillBuffer(id self, SEL _cmd, id buffer, NSRange range, uint8_t value) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSRange, uint8_t);
        ((Func)original)(self, _cmd, buffer, range, value);
    }
}

static void swizzled_blit_copyFromBuffer(id self, SEL _cmd, id srcBuffer, NSUInteger srcOffset,
                                          id dstBuffer, NSUInteger dstOffset, NSUInteger size) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, srcBuffer, srcOffset, dstBuffer, dstOffset, size);
    }
}

static void swizzled_blit_synchronizeResource(id self, SEL _cmd, id resource) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id);
        ((Func)original)(self, _cmd, resource);
    }
}

static void swizzled_blit_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_blit_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_blit_endEncoding)(self, _cmd);
    }

    encoder_ended(self);
}

static void swizzled_blit_deferredEndEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_blit_deferredEndEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_blit_deferredEndEncoding)(self, _cmd);
    }

    encoder_ended(self);
}

static void swizzled_blit_dealloc(id self, SEL _cmd) {
    {
        AGXMutexGuard guard;
        encoder_force_release(self);
    }

    if (g_original_blit_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_blit_dealloc)(self, _cmd);
    }
}

// ============================================================================
// Swizzled RENDER Encoder Methods
// ============================================================================

static id swizzled_renderCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_renderCommandEncoderWithDescriptor)(self, _cmd, descriptor);

    if (encoder) {
        encoder_created(encoder);
        g_agx_encoder_creates++;
    }

    return encoder;
}

static void swizzled_render_setRenderPipelineState(id self, SEL _cmd, id pipelineState) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id);
        ((Func)original)(self, _cmd, pipelineState);
    }
}

static void swizzled_render_setVertexBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, buffer, offset, index);
    }
}

static void swizzled_render_setFragmentBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, buffer, offset, index);
    }
}

static void swizzled_render_setVertexBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, bytes, length, index);
    }
}

static void swizzled_render_setFragmentBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, bytes, length, index);
    }
}

static void swizzled_render_setVertexTexture(id self, SEL _cmd, id texture, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger);
        ((Func)original)(self, _cmd, texture, index);
    }
}

static void swizzled_render_setFragmentTexture(id self, SEL _cmd, id texture, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger);
        ((Func)original)(self, _cmd, texture, index);
    }
}

static void swizzled_render_drawPrimitives(id self, SEL _cmd, NSUInteger primitiveType, NSUInteger vertexStart, NSUInteger vertexCount) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, primitiveType, vertexStart, vertexCount);
    }
}

static void swizzled_render_drawPrimitivesInstanced(id self, SEL _cmd, NSUInteger primitiveType, NSUInteger vertexStart, NSUInteger vertexCount, NSUInteger instanceCount) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, primitiveType, vertexStart, vertexCount, instanceCount);
    }
}

static void swizzled_render_drawIndexedPrimitives(id self, SEL _cmd, NSUInteger primitiveType, NSUInteger indexCount, NSUInteger indexType, id indexBuffer, NSUInteger indexBufferOffset) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger, id, NSUInteger);
        ((Func)original)(self, _cmd, primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset);
    }
}

static void swizzled_render_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_render_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_render_endEncoding)(self, _cmd);
    }

    encoder_ended(self);
}

static void swizzled_render_deferredEndEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_render_deferredEndEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_render_deferredEndEncoding)(self, _cmd);
    }

    encoder_ended(self);
}

static void swizzled_render_dealloc(id self, SEL _cmd) {
    {
        AGXMutexGuard guard;
        encoder_force_release(self);
    }

    if (g_original_render_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_render_dealloc)(self, _cmd);
    }
}

// ============================================================================
// Swizzled RESOURCE STATE Encoder Methods
// ============================================================================

static id swizzled_resourceStateCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_resourceStateCommandEncoder)(self, _cmd);

    if (encoder) {
        encoder_created(encoder);
        g_agx_encoder_creates++;
    }

    return encoder;
}

static void swizzled_resource_state_updateTextureMappings(id self, SEL _cmd, id texture,
    NSUInteger mode, const MTLRegion* regions, const NSUInteger* mipLevels,
    const NSUInteger* slices, NSUInteger numRegions) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, const MTLRegion*, const NSUInteger*,
                             const NSUInteger*, NSUInteger);
        ((Func)original)(self, _cmd, texture, mode, regions, mipLevels, slices, numRegions);
    }
}

static void swizzled_resource_state_updateTextureMapping(id self, SEL _cmd, id texture,
    NSUInteger mode, MTLRegion region, NSUInteger mipLevel, NSUInteger slice) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, MTLRegion, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, texture, mode, region, mipLevel, slice);
    }
}

static void swizzled_resource_state_updateFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    if (g_original_resource_state_updateFence) {
        typedef void (*Func)(id, SEL, id);
        ((Func)g_original_resource_state_updateFence)(self, _cmd, fence);
    }
}

static void swizzled_resource_state_waitForFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    if (g_original_resource_state_waitForFence) {
        typedef void (*Func)(id, SEL, id);
        ((Func)g_original_resource_state_waitForFence)(self, _cmd, fence);
    }
}

static void swizzled_resource_state_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_resource_state_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_resource_state_endEncoding)(self, _cmd);
    }

    encoder_ended(self);
}

static void swizzled_resource_state_dealloc(id self, SEL _cmd) {
    {
        AGXMutexGuard guard;
        encoder_force_release(self);
    }

    if (g_original_resource_state_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_resource_state_dealloc)(self, _cmd);
    }
}

// ============================================================================
// Swizzled ACCELERATION STRUCTURE Encoder Methods
// ============================================================================

static id swizzled_accelerationStructureCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_accelerationStructureCommandEncoder)(self, _cmd);

    if (encoder) {
        encoder_created(encoder);
        g_agx_encoder_creates++;
    }

    return encoder;
}

static void swizzled_accel_struct_build(id self, SEL _cmd, id accelStruct, id descriptor,
                                         id scratchBuffer, NSUInteger scratchBufferOffset) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id, id, NSUInteger);
        ((Func)original)(self, _cmd, accelStruct, descriptor, scratchBuffer, scratchBufferOffset);
    }
}

static void swizzled_accel_struct_refit(id self, SEL _cmd, id sourceAccelStruct, id descriptor,
                                         id destAccelStruct, id scratchBuffer, NSUInteger scratchBufferOffset) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id, id, id, NSUInteger);
        ((Func)original)(self, _cmd, sourceAccelStruct, descriptor, destAccelStruct, scratchBuffer, scratchBufferOffset);
    }
}

static void swizzled_accel_struct_copy(id self, SEL _cmd, id sourceAccelStruct, id destAccelStruct) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id);
        ((Func)original)(self, _cmd, sourceAccelStruct, destAccelStruct);
    }
}

static void swizzled_accel_struct_writeCompactedSize(id self, SEL _cmd, id accelStruct,
                                                      id buffer, NSUInteger offset) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id, NSUInteger);
        ((Func)original)(self, _cmd, accelStruct, buffer, offset);
    }
}

static void swizzled_accel_struct_updateFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    if (g_original_accel_struct_updateFence) {
        typedef void (*Func)(id, SEL, id);
        ((Func)g_original_accel_struct_updateFence)(self, _cmd, fence);
    }
}

static void swizzled_accel_struct_waitForFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    if (g_original_accel_struct_waitForFence) {
        typedef void (*Func)(id, SEL, id);
        ((Func)g_original_accel_struct_waitForFence)(self, _cmd, fence);
    }
}

static void swizzled_accel_struct_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_accel_struct_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_accel_struct_endEncoding)(self, _cmd);
    }

    encoder_ended(self);
}

static void swizzled_accel_struct_dealloc(id self, SEL _cmd) {
    {
        AGXMutexGuard guard;
        encoder_force_release(self);
    }

    if (g_original_accel_struct_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_accel_struct_dealloc)(self, _cmd);
    }
}

// COMPUTE: endEncoding
static void swizzled_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    if (g_original_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_endEncoding)(self, _cmd);
    }

    encoder_ended(self);
}

// COMPUTE: deferredEndEncoding
static void swizzled_deferredEndEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;

    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL);
        ((Func)original)(self, _cmd);
    }

    encoder_ended(self);
}

// COMPUTE: destroyImpl
static void swizzled_destroyImpl(id self, SEL _cmd) {
    {
        AGXMutexGuard guard;
        AGX_LOG("AGX Fix v2.5: destroyImpl on %p", self);
        encoder_force_release(self);
    }

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
static void agx_fix_v2_5_init() {
    g_log = os_log_create("com.agxfix.v2.5", "main");

    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix v2.5: Disabled via environment");
        return;
    }

    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    os_log(g_log, "AGX Fix v2.5: Initializing (with MPSCommandBuffer support)");

    // Get Metal device and create test objects
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix v2.5: No Metal device");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (!encoder || !commandBuffer) {
        AGX_LOG_ERROR("AGX Fix v2.5: Failed to create test objects");
        return;
    }

    // Get AGX classes
    g_agx_encoder_class = [encoder class];
    g_agx_command_buffer_class = [commandBuffer class];

    os_log(g_log, "AGX Fix v2.5: AGX Encoder class: %s", class_getName(g_agx_encoder_class));
    os_log(g_log, "AGX Fix v2.5: AGX Command buffer class: %s", class_getName(g_agx_command_buffer_class));

    [encoder endEncoding];

    // ========================================================================
    // v2.5: Discover MPSCommandBuffer class
    // ========================================================================

    @try {
        // Create an MPSCommandBuffer - this is what PyTorch uses
        MPSCommandBuffer* mpsCommandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        if (mpsCommandBuffer) {
            g_mps_command_buffer_class = [mpsCommandBuffer class];
            os_log(g_log, "AGX Fix v2.5: MPS Command buffer class: %s", class_getName(g_mps_command_buffer_class));

            // Verify it's different from the AGX class
            if (g_mps_command_buffer_class == g_agx_command_buffer_class) {
                os_log(g_log, "AGX Fix v2.5: WARNING - MPS and AGX command buffer classes are the same!");
            } else {
                os_log(g_log, "AGX Fix v2.5: MPS and AGX command buffer classes are DIFFERENT - will swizzle both");
            }
        } else {
            os_log(g_log, "AGX Fix v2.5: WARNING - Could not create MPSCommandBuffer");
        }
    } @catch (NSException* e) {
        os_log(g_log, "AGX Fix v2.5: Exception creating MPSCommandBuffer: %s", [[e reason] UTF8String]);
    }

    // Discover blit encoder class
    id<MTLCommandBuffer> commandBuffer2 = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer2 blitCommandEncoder];
    if (blitEncoder) {
        g_agx_blit_encoder_class = [blitEncoder class];
        os_log(g_log, "AGX Fix v2.5: Blit encoder class: %s", class_getName(g_agx_blit_encoder_class));
        [blitEncoder endEncoding];
    }

    // Discover render encoder class
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
            os_log(g_log, "AGX Fix v2.5: Render encoder class: %s", class_getName(g_agx_render_encoder_class));
            [renderEncoder endEncoding];
        }
    }

    // Discover resource state encoder class
    id<MTLCommandBuffer> commandBuffer4 = [queue commandBuffer];
    SEL resourceStateSelector = @selector(resourceStateCommandEncoder);
    if ([commandBuffer4 respondsToSelector:resourceStateSelector]) {
        id resourceStateEncoder = [commandBuffer4 resourceStateCommandEncoder];
        if (resourceStateEncoder) {
            g_agx_resource_state_encoder_class = [resourceStateEncoder class];
            os_log(g_log, "AGX Fix v2.5: Resource state encoder class: %s", class_getName(g_agx_resource_state_encoder_class));
            [resourceStateEncoder endEncoding];
        }
    }

    // Discover acceleration structure encoder class
    id<MTLCommandBuffer> commandBuffer5 = [queue commandBuffer];
    SEL accelStructSelector = @selector(accelerationStructureCommandEncoder);
    if ([commandBuffer5 respondsToSelector:accelStructSelector]) {
        id accelStructEncoder = [commandBuffer5 accelerationStructureCommandEncoder];
        if (accelStructEncoder) {
            g_agx_accel_struct_encoder_class = [accelStructEncoder class];
            os_log(g_log, "AGX Fix v2.5: Accel struct encoder class: %s", class_getName(g_agx_accel_struct_encoder_class));
            [accelStructEncoder endEncoding];
        }
    }

    // Discover _impl ivar offset
    Ivar implIvar = class_getInstanceVariable(g_agx_encoder_class, "_impl");
    if (implIvar) {
        g_impl_ivar_offset = ivar_getOffset(implIvar);
        os_log(g_log, "AGX Fix v2.5: _impl at offset %td", g_impl_ivar_offset);
    } else {
        Class parent = class_getSuperclass(g_agx_encoder_class);
        while (parent) {
            implIvar = class_getInstanceVariable(parent, "_impl");
            if (implIvar) {
                g_impl_ivar_offset = ivar_getOffset(implIvar);
                os_log(g_log, "AGX Fix v2.5: _impl at offset %td in parent %s",
                       g_impl_ivar_offset, class_getName(parent));
                break;
            }
            parent = class_getSuperclass(parent);
        }
    }

    int swizzled_count = 0;
    IMP dummy;

    // ========================================================================
    // PART 1: Swizzle AGX COMMAND BUFFER encoder creation methods
    // ========================================================================

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoder),
                       (IMP)swizzled_computeCommandEncoder, &g_original_computeCommandEncoder)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.5: Swizzled AGX computeCommandEncoder");
    }

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoderWithDescriptor:),
                       (IMP)swizzled_computeCommandEncoderWithDescriptor, &g_original_computeCommandEncoderWithDescriptor)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.5: Swizzled AGX computeCommandEncoderWithDescriptor:");
    }

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoderWithDispatchType:),
                       (IMP)swizzled_computeCommandEncoderWithDispatchType, &g_original_computeCommandEncoderWithDispatchType)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.5: Swizzled AGX computeCommandEncoderWithDispatchType:");
    }

    if (swizzle_method(g_agx_command_buffer_class, @selector(blitCommandEncoder),
                       (IMP)swizzled_blitCommandEncoder, &g_original_blitCommandEncoder)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.5: Swizzled AGX blitCommandEncoder");
    }

    // ========================================================================
    // PART 1b (v2.5): Swizzle MPS COMMAND BUFFER encoder creation methods
    // ========================================================================

    if (g_mps_command_buffer_class && g_mps_command_buffer_class != g_agx_command_buffer_class) {
        if (swizzle_method(g_mps_command_buffer_class, @selector(computeCommandEncoder),
                           (IMP)swizzled_mps_computeCommandEncoder, &g_original_mps_computeCommandEncoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled MPS computeCommandEncoder");
        }

        if (swizzle_method(g_mps_command_buffer_class, @selector(computeCommandEncoderWithDescriptor:),
                           (IMP)swizzled_mps_computeCommandEncoderWithDescriptor, &g_original_mps_computeCommandEncoderWithDescriptor)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled MPS computeCommandEncoderWithDescriptor:");
        }

        if (swizzle_method(g_mps_command_buffer_class, @selector(computeCommandEncoderWithDispatchType:),
                           (IMP)swizzled_mps_computeCommandEncoderWithDispatchType, &g_original_mps_computeCommandEncoderWithDispatchType)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled MPS computeCommandEncoderWithDispatchType:");
        }

        if (swizzle_method(g_mps_command_buffer_class, @selector(blitCommandEncoder),
                           (IMP)swizzled_mps_blitCommandEncoder, &g_original_mps_blitCommandEncoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled MPS blitCommandEncoder");
        }

        if (swizzle_method(g_mps_command_buffer_class, @selector(renderCommandEncoderWithDescriptor:),
                           (IMP)swizzled_mps_renderCommandEncoderWithDescriptor, &g_original_mps_renderCommandEncoderWithDescriptor)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled MPS renderCommandEncoderWithDescriptor:");
        }

        if (swizzle_method(g_mps_command_buffer_class, @selector(resourceStateCommandEncoder),
                           (IMP)swizzled_mps_resourceStateCommandEncoder, &g_original_mps_resourceStateCommandEncoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled MPS resourceStateCommandEncoder");
        }

        if (swizzle_method(g_mps_command_buffer_class, @selector(accelerationStructureCommandEncoder),
                           (IMP)swizzled_mps_accelerationStructureCommandEncoder, &g_original_mps_accelerationStructureCommandEncoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled MPS accelerationStructureCommandEncoder");
        }

        os_log(g_log, "AGX Fix v2.5: MPS Command buffer encoder creation methods swizzled");
    }

    // ========================================================================
    // PART 2: Swizzle encoder methods with active call tracking
    // ========================================================================

    if (swizzle_method(g_agx_encoder_class, @selector(destroyImpl),
                       (IMP)swizzled_destroyImpl, &g_original_destroyImpl)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.5: Swizzled destroyImpl");
    }

    if (swizzle_method(g_agx_encoder_class, @selector(endEncoding),
                       (IMP)swizzled_endEncoding, &g_original_endEncoding)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.5: Swizzled endEncoding");
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
    // ========================================================================

    if (g_agx_blit_encoder_class) {
        IMP blit_dummy;

        if (swizzle_method(g_agx_blit_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_blit_endEncoding, &g_original_blit_endEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled blit endEncoding");
        }

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

        if (swizzle_method(g_agx_blit_encoder_class, @selector(deferredEndEncoding),
                           (IMP)swizzled_blit_deferredEndEncoding, &g_original_blit_deferredEndEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled blit deferredEndEncoding");
        }

        if (swizzle_method(g_agx_blit_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_blit_dealloc, &g_original_blit_dealloc)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled blit dealloc (cleanup fallback)");
        }

        os_log(g_log, "AGX Fix v2.5: Blit encoder methods swizzled");
    }

    // ========================================================================
    // PART 4: Swizzle RENDER encoder methods
    // ========================================================================

    if (g_agx_render_encoder_class) {
        IMP render_dummy;

        if (swizzle_method(g_agx_command_buffer_class, @selector(renderCommandEncoderWithDescriptor:),
                           (IMP)swizzled_renderCommandEncoderWithDescriptor, &g_original_renderCommandEncoderWithDescriptor)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled renderCommandEncoderWithDescriptor:");
        }

        if (swizzle_method(g_agx_render_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_render_endEncoding, &g_original_render_endEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled render endEncoding");
        }

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

        if (swizzle_method(g_agx_render_encoder_class, @selector(deferredEndEncoding),
                           (IMP)swizzled_render_deferredEndEncoding, &g_original_render_deferredEndEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled render deferredEndEncoding");
        }

        if (swizzle_method(g_agx_render_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_render_dealloc, &g_original_render_dealloc)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled render dealloc (cleanup fallback)");
        }

        os_log(g_log, "AGX Fix v2.5: Render encoder methods swizzled");
    }

    // ========================================================================
    // PART 5: Swizzle RESOURCE STATE encoder methods
    // ========================================================================

    if (g_agx_resource_state_encoder_class) {
        IMP resource_state_dummy;

        if (swizzle_method(g_agx_command_buffer_class, @selector(resourceStateCommandEncoder),
                           (IMP)swizzled_resourceStateCommandEncoder, &g_original_resourceStateCommandEncoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled resourceStateCommandEncoder");
        }

        if (swizzle_method(g_agx_resource_state_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_resource_state_endEncoding, &g_original_resource_state_endEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled resource state endEncoding");
        }

        #define SWIZZLE_RESOURCE_STATE(sel, func) \
            if (swizzle_method(g_agx_resource_state_encoder_class, sel, (IMP)func, &resource_state_dummy)) swizzled_count++;

        SWIZZLE_RESOURCE_STATE(@selector(updateTextureMappings:mode:regions:mipLevels:slices:numRegions:), swizzled_resource_state_updateTextureMappings)
        SWIZZLE_RESOURCE_STATE(@selector(updateTextureMapping:mode:region:mipLevel:slice:), swizzled_resource_state_updateTextureMapping)

        if (swizzle_method(g_agx_resource_state_encoder_class, @selector(updateFence:),
                           (IMP)swizzled_resource_state_updateFence, &g_original_resource_state_updateFence)) {
            swizzled_count++;
        }
        if (swizzle_method(g_agx_resource_state_encoder_class, @selector(waitForFence:),
                           (IMP)swizzled_resource_state_waitForFence, &g_original_resource_state_waitForFence)) {
            swizzled_count++;
        }

        #undef SWIZZLE_RESOURCE_STATE

        if (swizzle_method(g_agx_resource_state_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_resource_state_dealloc, &g_original_resource_state_dealloc)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled resource state dealloc (cleanup fallback)");
        }

        os_log(g_log, "AGX Fix v2.5: Resource state encoder methods swizzled");
    }

    // ========================================================================
    // PART 6: Swizzle ACCELERATION STRUCTURE encoder methods
    // ========================================================================

    if (g_agx_accel_struct_encoder_class) {
        IMP accel_struct_dummy;

        if (swizzle_method(g_agx_command_buffer_class, @selector(accelerationStructureCommandEncoder),
                           (IMP)swizzled_accelerationStructureCommandEncoder, &g_original_accelerationStructureCommandEncoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled accelerationStructureCommandEncoder");
        }

        if (swizzle_method(g_agx_accel_struct_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_accel_struct_endEncoding, &g_original_accel_struct_endEncoding)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled accel struct endEncoding");
        }

        #define SWIZZLE_ACCEL_STRUCT(sel, func) \
            if (swizzle_method(g_agx_accel_struct_encoder_class, sel, (IMP)func, &accel_struct_dummy)) swizzled_count++;

        SWIZZLE_ACCEL_STRUCT(@selector(buildAccelerationStructure:descriptor:scratchBuffer:scratchBufferOffset:), swizzled_accel_struct_build)
        SWIZZLE_ACCEL_STRUCT(@selector(refitAccelerationStructure:descriptor:destination:scratchBuffer:scratchBufferOffset:), swizzled_accel_struct_refit)
        SWIZZLE_ACCEL_STRUCT(@selector(copyAccelerationStructure:toAccelerationStructure:), swizzled_accel_struct_copy)
        SWIZZLE_ACCEL_STRUCT(@selector(writeCompactedAccelerationStructureSize:toBuffer:offset:), swizzled_accel_struct_writeCompactedSize)

        if (swizzle_method(g_agx_accel_struct_encoder_class, @selector(updateFence:),
                           (IMP)swizzled_accel_struct_updateFence, &g_original_accel_struct_updateFence)) {
            swizzled_count++;
        }
        if (swizzle_method(g_agx_accel_struct_encoder_class, @selector(waitForFence:),
                           (IMP)swizzled_accel_struct_waitForFence, &g_original_accel_struct_waitForFence)) {
            swizzled_count++;
        }

        #undef SWIZZLE_ACCEL_STRUCT

        if (swizzle_method(g_agx_accel_struct_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_accel_struct_dealloc, &g_original_accel_struct_dealloc)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.5: Swizzled accel struct dealloc (cleanup fallback)");
        }

        os_log(g_log, "AGX Fix v2.5: Acceleration structure encoder methods swizzled");
    }

    os_log(g_log, "AGX Fix v2.5: COMPLETE - %d methods protected (with MPSCommandBuffer support)",
           swizzled_count);
}

// ============================================================================
// Statistics API
// ============================================================================

extern "C" {
    uint64_t agx_fix_v2_5_get_acquisitions() { return g_mutex_acquisitions.load(); }
    uint64_t agx_fix_v2_5_get_contentions() { return g_mutex_contentions.load(); }
    uint64_t agx_fix_v2_5_get_encoders_created() { return g_encoders_created.load(); }
    uint64_t agx_fix_v2_5_get_encoders_released() { return g_encoders_released.load(); }
    uint64_t agx_fix_v2_5_get_null_impl_skips() { return g_null_impl_skips.load(); }
    uint64_t agx_fix_v2_5_get_method_calls() { return g_method_calls.load(); }
    uint64_t agx_fix_v2_5_get_deferred_releases() { return g_deferred_releases.load(); }
    uint64_t agx_fix_v2_5_get_agx_encoder_creates() { return g_agx_encoder_creates.load(); }
    uint64_t agx_fix_v2_5_get_mps_encoder_creates() { return g_mps_encoder_creates.load(); }
    size_t agx_fix_v2_5_get_active_count() {
        std::lock_guard<std::recursive_mutex> lock(g_encoder_mutex);
        return g_encoder_states.size();
    }
    bool agx_fix_v2_5_is_enabled() { return g_enabled; }
}
