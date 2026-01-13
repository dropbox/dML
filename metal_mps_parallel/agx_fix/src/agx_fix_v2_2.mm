/**
 * AGX Driver Race Condition Fix - Version 2.2
 *
 * ⚠️ DEPRECATED: This version CRASHES at 8 threads! Use v2.3 instead.
 * Bug discovered in N=1968: Removing mutex from encoder methods causes
 * driver-internal races. v2.3 adds mutex protection back.
 *
 * FIXES TLC-PROVEN BUGS IN v2.1:
 *   Bug 4 (Pre-Swizzle Race): Retain from CREATION, not first method call
 *   Bug 1 (TOCTOU Race): Eliminated by holding mutex across check-and-act
 *   Bug 2 (Data Race): All shared state access under mutex
 *   Bug 3 (Lost Updates): All increments under mutex
 *
 * KEY ARCHITECTURAL CHANGE (v2.1 -> v2.2):
 *   v2.1 tried to retain encoders inside swizzled encoder METHODS.
 *   Problem: objc_msgSend crashes BEFORE our swizzled code runs.
 *   v2.2 retains encoders in swizzled command buffer CREATION methods.
 *   Solution: Encoder is retained before ANY method can be called on it.
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
    // Single global mutex protects ALL shared state
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

    // Logging
    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    // Original method implementations - Command Buffer methods
    IMP g_original_computeCommandEncoder = nullptr;
    IMP g_original_computeCommandEncoderWithDescriptor = nullptr;
    IMP g_original_computeCommandEncoderWithDispatchType = nullptr;

    // Original method implementations - Encoder methods
    IMP g_original_endEncoding = nullptr;
    IMP g_original_destroyImpl = nullptr;

    // Encoder method originals for _impl check
    constexpr int MAX_SWIZZLED = 64;
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
        }
    }

    // AGX class info
    Class g_agx_encoder_class = nullptr;
    Class g_agx_command_buffer_class = nullptr;
    ptrdiff_t g_impl_ivar_offset = -1;
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
// Encoder Lifetime Management (v2.2)
// ============================================================================

// Retain encoder on creation - called from swizzled command buffer methods
// This is the KEY FIX: encoder is retained BEFORE any method can be called
static void retain_encoder_on_creation(id encoder) {
    if (!encoder) return;

    AGXMutexGuard guard;
    void* ptr = (__bridge void*)encoder;

    // Check if already tracked (shouldn't happen, but be safe)
    if (g_active_encoders.count(ptr) > 0) {
        AGX_LOG("AGX Fix v2.2: Encoder %p already tracked", ptr);
        return;
    }

    // Retain and track
    CFRetain((__bridge CFTypeRef)encoder);
    g_active_encoders.insert(ptr);
    g_encoders_retained++;

    AGX_LOG("AGX Fix v2.2: Retained encoder %p on creation (total: %zu)",
            ptr, g_active_encoders.size());
}

// Release encoder on endEncoding - called from swizzled endEncoding
static void release_encoder_on_end(id encoder) {
    if (!encoder) return;

    AGXMutexGuard guard;
    void* ptr = (__bridge void*)encoder;

    // Check if tracked
    auto it = g_active_encoders.find(ptr);
    if (it == g_active_encoders.end()) {
        AGX_LOG("AGX Fix v2.2: Encoder %p not tracked at endEncoding", ptr);
        return;
    }

    // Untrack and release
    g_active_encoders.erase(it);
    CFRelease((__bridge CFTypeRef)encoder);
    g_encoders_released++;

    AGX_LOG("AGX Fix v2.2: Released encoder %p at endEncoding (total: %zu)",
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
        AGX_LOG("AGX Fix v2.2: NULL _impl in %p", encoder);
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

// ============================================================================
// Swizzled Encoder Methods
// ============================================================================

// Generic wrapper macro for encoder methods - adds _impl check
#define DEFINE_SWIZZLED_METHOD_VOID_0(name) \
static void swizzled_##name(id self, SEL _cmd) { \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL); \
        ((Func)original)(self, _cmd); \
    } \
}

#define DEFINE_SWIZZLED_METHOD_VOID_1(name, T1) \
static void swizzled_##name(id self, SEL _cmd, T1 a1) { \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1); \
        ((Func)original)(self, _cmd, a1); \
    } \
}

#define DEFINE_SWIZZLED_METHOD_VOID_2(name, T1, T2) \
static void swizzled_##name(id self, SEL _cmd, T1 a1, T2 a2) { \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1, T2); \
        ((Func)original)(self, _cmd, a1, a2); \
    } \
}

#define DEFINE_SWIZZLED_METHOD_VOID_3(name, T1, T2, T3) \
static void swizzled_##name(id self, SEL _cmd, T1 a1, T2 a2, T3 a3) { \
    if (!is_impl_valid(self)) return; \
    IMP original = get_original_imp(_cmd); \
    if (original) { \
        typedef void (*Func)(id, SEL, T1, T2, T3); \
        ((Func)original)(self, _cmd, a1, a2, a3); \
    } \
}

#define DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE(name) \
static void swizzled_##name(id self, SEL _cmd, MTLSize a1, MTLSize a2) { \
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
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, buffer, offset, index);
    }
}

// setBuffers:offsets:withRange:
static void swizzled_setBuffers(id self, SEL _cmd, const id* buffers, const NSUInteger* offsets, NSRange range) {
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const id*, const NSUInteger*, NSRange);
        ((Func)original)(self, _cmd, buffers, offsets, range);
    }
}

// setBytes:length:atIndex:
static void swizzled_setBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
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
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, MTLSize);
        ((Func)original)(self, _cmd, buffer, offset, tptg);
    }
}

// SPECIAL: endEncoding - releases our extra retain
static void swizzled_endEncoding(id self, SEL _cmd) {
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
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL);
        ((Func)original)(self, _cmd);
    }

    release_encoder_on_end(self);
}

// SPECIAL: destroyImpl - force release if still tracked
static void swizzled_destroyImpl(id self, SEL _cmd) {
    AGX_LOG("AGX Fix v2.2: destroyImpl on %p", self);

    // Force release if still tracked (encoder being destroyed before endEncoding)
    {
        AGXMutexGuard guard;
        void* ptr = (__bridge void*)self;
        auto it = g_active_encoders.find(ptr);
        if (it != g_active_encoders.end()) {
            g_active_encoders.erase(it);
            CFRelease((__bridge CFTypeRef)self);
            g_encoders_released++;
            AGX_LOG("AGX Fix v2.2: Force released encoder %p at destroyImpl", ptr);
        }
    }

    // Call original
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
static void agx_fix_v2_2_init() {
    g_log = os_log_create("com.agxfix.v2.2", "main");

    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix v2.2: Disabled via environment");
        return;
    }

    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    os_log(g_log, "AGX Fix v2.2: Initializing (retain-from-creation pattern)");

    // Get Metal device and create test objects
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix v2.2: No Metal device");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (!encoder || !commandBuffer) {
        AGX_LOG_ERROR("AGX Fix v2.2: Failed to create test objects");
        return;
    }

    // Get classes
    g_agx_encoder_class = [encoder class];
    g_agx_command_buffer_class = [commandBuffer class];

    os_log(g_log, "AGX Fix v2.2: Encoder class: %s", class_getName(g_agx_encoder_class));
    os_log(g_log, "AGX Fix v2.2: Command buffer class: %s", class_getName(g_agx_command_buffer_class));

    [encoder endEncoding];

    // Discover _impl ivar offset
    Ivar implIvar = class_getInstanceVariable(g_agx_encoder_class, "_impl");
    if (implIvar) {
        g_impl_ivar_offset = ivar_getOffset(implIvar);
        os_log(g_log, "AGX Fix v2.2: _impl at offset %td", g_impl_ivar_offset);
    } else {
        Class parent = class_getSuperclass(g_agx_encoder_class);
        while (parent) {
            implIvar = class_getInstanceVariable(parent, "_impl");
            if (implIvar) {
                g_impl_ivar_offset = ivar_getOffset(implIvar);
                os_log(g_log, "AGX Fix v2.2: _impl at offset %td in parent %s",
                       g_impl_ivar_offset, class_getName(parent));
                break;
            }
            parent = class_getSuperclass(parent);
        }
    }

    int swizzled_count = 0;
    IMP dummy;

    // ========================================================================
    // KEY FIX: Swizzle COMMAND BUFFER encoder creation methods
    // This ensures encoders are retained BEFORE any method can be called
    // ========================================================================

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoder),
                       (IMP)swizzled_computeCommandEncoder, &g_original_computeCommandEncoder)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.2: Swizzled computeCommandEncoder");
    }

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoderWithDescriptor:),
                       (IMP)swizzled_computeCommandEncoderWithDescriptor, &g_original_computeCommandEncoderWithDescriptor)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.2: Swizzled computeCommandEncoderWithDescriptor:");
    }

    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoderWithDispatchType:),
                       (IMP)swizzled_computeCommandEncoderWithDispatchType, &g_original_computeCommandEncoderWithDispatchType)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.2: Swizzled computeCommandEncoderWithDispatchType:");
    }

    // ========================================================================
    // Swizzle encoder methods for _impl NULL checking
    // ========================================================================

    if (swizzle_method(g_agx_encoder_class, @selector(destroyImpl),
                       (IMP)swizzled_destroyImpl, &g_original_destroyImpl)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.2: Swizzled destroyImpl");
    }

    if (swizzle_method(g_agx_encoder_class, @selector(endEncoding),
                       (IMP)swizzled_endEncoding, &g_original_endEncoding)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.2: Swizzled endEncoding");
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

    os_log(g_log, "AGX Fix v2.2: COMPLETE - %d methods protected (retain-from-creation pattern)",
           swizzled_count);
}

// ============================================================================
// Statistics API
// ============================================================================

extern "C" {
    uint64_t agx_fix_v2_2_get_acquisitions() { return g_mutex_acquisitions.load(); }
    uint64_t agx_fix_v2_2_get_contentions() { return g_mutex_contentions.load(); }
    uint64_t agx_fix_v2_2_get_encoders_retained() { return g_encoders_retained.load(); }
    uint64_t agx_fix_v2_2_get_encoders_released() { return g_encoders_released.load(); }
    uint64_t agx_fix_v2_2_get_null_impl_skips() { return g_null_impl_skips.load(); }
    size_t agx_fix_v2_2_get_active_count() {
        std::lock_guard<std::recursive_mutex> lock(g_encoder_mutex);
        return g_active_encoders.size();
    }
    bool agx_fix_v2_2_is_enabled() { return g_enabled; }
}
