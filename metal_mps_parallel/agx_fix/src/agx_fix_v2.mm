/**
 * AGX Driver Race Condition Fix - Version 2
 *
 * This version fixes TWO classes of crashes:
 * 1. NULL _impl crashes (original fix)
 * 2. Use-after-free / PAC failure crashes (NEW in v2)
 *
 * ROOT CAUSE OF PAC FAILURES:
 *   The encoder object is deallocated while another thread still holds a reference.
 *   When objc_msgSend tries to dispatch a method, it reads a corrupted isa pointer,
 *   causing a Pointer Authentication Code (PAC) failure.
 *
 * FIX STRATEGY:
 *   Keep encoders alive from first use until endEncoding is called. This prevents
 *   premature deallocation even if another thread releases the encoder.
 *
 * CRASH SITES FIXED:
 *   1. objc_msgSend + 32 (PAC failure) - FIXED by keeping encoder alive
 *   2. setComputePipelineState: (NULL _impl) - FIXED by _impl check
 *   3. dispatchThreads: (NULL _impl) - FIXED by _impl check
 *   4. ALL encoder methods (NULL _impl or freed object) - FIXED
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
#import <unordered_map>
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
    // The global mutex that prevents race conditions
    std::recursive_mutex g_agx_encoding_mutex;  // Recursive for nested calls

    // Map of encoders we've artificially retained to prevent deallocation
    // Key: encoder pointer, Value: reference count (number of THREADS using it)
    // CRITICAL: Count represents threads, not method calls!
    std::unordered_map<void*, int> g_retained_encoders;

    // Set of encoders that have been destroyed (tombstones)
    std::unordered_set<void*> g_destroyed_encoders;

    // Set of valid contexts (for _impl NULL checking)
    std::unordered_set<void*> g_valid_contexts;

    // PER-THREAD tracking: which encoders this thread is currently using
    // CRITICAL FIX: This prevents the memory leak where each method call
    // incremented the count, but only endEncoding decremented once.
    // Now: increment only when thread STARTS using encoder (first method call)
    //      decrement only when thread STOPS using encoder (endEncoding)
    thread_local std::unordered_set<void*> t_thread_using_encoders;

    // Statistics
    std::atomic<uint64_t> g_mutex_acquisitions{0};
    std::atomic<uint64_t> g_mutex_contentions{0};
    std::atomic<uint64_t> g_encoder_retains{0};
    std::atomic<uint64_t> g_encoder_releases{0};
    std::atomic<uint64_t> g_use_after_free_prevented{0};
    std::atomic<uint64_t> g_null_impl_skips{0};

    // Logging
    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    // Original method implementations
    IMP g_original_destroyImpl = nullptr;
    IMP g_original_initWithQueue = nullptr;

    // Encoder method originals (stored in parallel arrays to avoid hash overflow)
    constexpr int MAX_SWIZZLED = 64;
    SEL g_swizzled_sels[MAX_SWIZZLED] = {nullptr};
    IMP g_original_imps[MAX_SWIZZLED] = {nullptr};
    int g_swizzle_count = 0;

    // Lookup original IMP for a selector
    IMP get_original_imp(SEL sel) {
        for (int i = 0; i < g_swizzle_count; i++) {
            if (g_swizzled_sels[i] == sel) return g_original_imps[i];
        }
        return nullptr;
    }

    // Store original IMP for a selector
    void store_original_imp(SEL sel, IMP imp) {
        if (g_swizzle_count < MAX_SWIZZLED) {
            g_swizzled_sels[g_swizzle_count] = sel;
            g_original_imps[g_swizzle_count] = imp;
            g_swizzle_count++;
        }
    }

    // AGX class info
    Class g_agx_context_class = nullptr;
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
        if (g_agx_encoding_mutex.try_lock()) {
            locked_ = true;
            g_mutex_acquisitions++;
        } else {
            g_mutex_contentions++;
            g_agx_encoding_mutex.lock();
            locked_ = true;
            g_mutex_acquisitions++;
        }
    }
    ~AGXMutexGuard() {
        if (locked_) g_agx_encoding_mutex.unlock();
    }
    AGXMutexGuard(const AGXMutexGuard&) = delete;
    AGXMutexGuard& operator=(const AGXMutexGuard&) = delete;
private:
    bool locked_;
};

// Forward declaration
static bool is_impl_valid(id context);

// ============================================================================
// Encoder Lifetime Management (NEW in v2)
// ============================================================================

// Ensure encoder stays alive - call this on every method call
// Returns false if the encoder is already destroyed (don't use it)
//
// CRITICAL FIX v2.1: Use per-thread tracking to prevent memory leaks!
// Problem: Each method call (setBuffer, setBytes, etc.) was incrementing count,
//          but only endEncoding decremented. Result: count never reached 0.
// Solution: Only increment when THIS THREAD starts using the encoder.
//           Multiple method calls from same thread don't increment again.
static bool ensure_encoder_alive(id encoder) {
    // Note: caller must hold mutex
    void* ptr = (__bridge void*)encoder;

    // Check if THIS THREAD is already using this encoder
    // If yes, don't increment global count (prevents memory leak)
    if (t_thread_using_encoders.count(ptr) > 0) {
        // Thread already using this encoder - no action needed
        AGX_LOG("AGX Fix v2: Thread already using encoder %p", ptr);
        return true;
    }

    // Check if this encoder is already tracked as valid (handles address reuse)
    // CRITICAL: Check retained_encoders BEFORE checking tombstone to handle
    // the case where a new encoder is created at the same address as an old
    // destroyed encoder. This fixes the false-positive UAF detection bug.
    auto it = g_retained_encoders.find(ptr);
    if (it != g_retained_encoders.end()) {
        // This encoder is tracked - address might have been reused
        // Clear from tombstone set just in case
        g_destroyed_encoders.erase(ptr);

        // This thread is starting to use this encoder
        t_thread_using_encoders.insert(ptr);
        it->second++;
        AGX_LOG("AGX Fix v2: Thread using known encoder %p (threads=%d)", ptr, it->second);
        return true;
    }

    // Encoder not tracked yet - check if it's in tombstone
    if (g_destroyed_encoders.count(ptr) > 0) {
        // Address was previously destroyed. Check if this is:
        // a) A NEW encoder at a reused address (has valid _impl) - safe to use
        // b) A truly destroyed encoder (NULL _impl) - use-after-free!
        if (is_impl_valid(encoder)) {
            // Valid _impl means this is a NEW encoder at a reused address
            g_destroyed_encoders.erase(ptr);
            AGX_LOG("AGX Fix v2: Cleared tombstone for reused address %p (valid _impl)", ptr);
        } else {
            // NULL _impl confirms this is a destroyed encoder
            g_use_after_free_prevented++;
            AGX_LOG("AGX Fix v2: Prevented use-after-free on %p (in tombstone, NULL _impl)", ptr);
            return false;
        }
    }

    // This thread is NOT yet using this encoder - register it
    t_thread_using_encoders.insert(ptr);

    // First thread to access: CFRetain and set count to 1
    CFRetain((__bridge CFTypeRef)encoder);
    g_retained_encoders[ptr] = 1;
    g_valid_contexts.insert(ptr);
    g_encoder_retains++;
    AGX_LOG("AGX Fix v2: First thread retained encoder %p (threads=1)", ptr);

    return true;
}

// Release our extra retain on an encoder (called from endEncoding)
// CRITICAL FIX v2.1: Use per-thread tracking to match ensure_encoder_alive
static void release_encoder_retain(id encoder) {
    // Note: caller must hold mutex
    void* ptr = (__bridge void*)encoder;

    // Check if THIS THREAD was using this encoder
    if (t_thread_using_encoders.count(ptr) == 0) {
        // Thread wasn't using this encoder - nothing to release
        AGX_LOG("AGX Fix v2: Thread not using encoder %p, skip release", ptr);
        return;
    }

    // Remove from this thread's set
    t_thread_using_encoders.erase(ptr);

    // Decrement global count
    auto it = g_retained_encoders.find(ptr);
    if (it != g_retained_encoders.end()) {
        it->second--;
        g_encoder_releases++;
        AGX_LOG("AGX Fix v2: Thread released encoder %p (threads=%d)", ptr, it->second);

        if (it->second <= 0) {
            // Last thread done: actually release the encoder
            g_retained_encoders.erase(it);
            AGX_LOG("AGX Fix v2: Final release on %p (all threads done)", ptr);
            CFRelease((__bridge CFTypeRef)encoder);
        }
        // Otherwise: other threads still using, keep the CFRetain in place
    }
}

// Mark encoder as destroyed (called from destroyImpl)
static void mark_encoder_destroyed(id encoder) {
    // Note: caller must hold mutex
    void* ptr = (__bridge void*)encoder;

    // Remove from valid contexts
    g_valid_contexts.erase(ptr);

    // Remove from this thread's using set (cleanup)
    t_thread_using_encoders.erase(ptr);

    // Remove from global retained set (force cleanup)
    g_retained_encoders.erase(ptr);

    // Add to destroyed set (tombstone)
    g_destroyed_encoders.insert(ptr);

    // Limit tombstone set size
    if (g_destroyed_encoders.size() > 10000) {
        g_destroyed_encoders.clear();
    }

    AGX_LOG("AGX Fix v2: Marked encoder %p as destroyed", ptr);
}

// ============================================================================
// _impl Validity Check
// ============================================================================

static bool is_impl_valid(id context) {
    if (g_impl_ivar_offset < 0) return true;

    char* obj_base = (char*)(__bridge void*)context;
    void** impl_ptr = (void**)(obj_base + g_impl_ivar_offset);
    void* impl = *impl_ptr;

    if (impl == nullptr) {
        g_null_impl_skips++;
        AGX_LOG("AGX Fix v2: NULL _impl in %p", context);
        return false;
    }
    return true;
}

// ============================================================================
// Swizzled Method Implementations
// ============================================================================

// Generic wrapper that handles all encoder methods
// This is the CRITICAL function that prevents use-after-free
#define DEFINE_SWIZZLED_METHOD_VOID_0(name) \
static void swizzled_##name(id self, SEL _cmd) { \
    AGXMutexGuard guard; \
    if (!ensure_encoder_alive(self)) return; \
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
    if (!ensure_encoder_alive(self)) return; \
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
    if (!ensure_encoder_alive(self)) return; \
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
    if (!ensure_encoder_alive(self)) return; \
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
    if (!ensure_encoder_alive(self)) return; \
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

// setBuffer:offset:atIndex: - special handling because T1=id, T2=NSUInteger, T3=NSUInteger
static void swizzled_setBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    if (!ensure_encoder_alive(self)) return;
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
    if (!ensure_encoder_alive(self)) return;
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
    if (!ensure_encoder_alive(self)) return;
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

// setStageInRegion - special handling for MTLRegion
static void swizzled_setStageInRegion(id self, SEL _cmd, MTLRegion region) {
    AGXMutexGuard guard;
    if (!ensure_encoder_alive(self)) return;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, MTLRegion);
        ((Func)original)(self, _cmd, region);
    }
}

DEFINE_SWIZZLED_METHOD_VOID_2(setImageblockWidth, NSUInteger, NSUInteger)

// Additional methods needed for LayerNorm and other complex operations
DEFINE_SWIZZLED_METHOD_VOID_2(setBufferOffset, NSUInteger, NSUInteger)  // setBufferOffset:atIndex:
DEFINE_SWIZZLED_METHOD_VOID_1(updateFence, id)
DEFINE_SWIZZLED_METHOD_VOID_1(waitForFence, id)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchWaitFlush)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFlushInvalidate)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFlushOnly)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchInvalidateOnly)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFenceOnly)

// Indirect dispatch methods
static void swizzled_dispatchThreadgroupsIndirect(id self, SEL _cmd, id buffer, NSUInteger offset, MTLSize threadsPerTG) {
    AGXMutexGuard guard;
    if (!ensure_encoder_alive(self)) return;
    if (!is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, MTLSize);
        ((Func)original)(self, _cmd, buffer, offset, threadsPerTG);
    }
}

// SPECIAL: deferredEndEncoding - same lifecycle handling as endEncoding
static void swizzled_deferredEndEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;

    // Call original FIRST (must always call per Metal spec)
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL);
        ((Func)original)(self, _cmd);
    }

    // Release our extra retain - encoder is done
    release_encoder_retain(self);
}

// SPECIAL: endEncoding - releases our extra retain
static void swizzled_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;

    // Call original FIRST (must always call endEncoding per Metal spec)
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL);
        ((Func)original)(self, _cmd);
    }

    // Now release our extra retain - encoder is done
    release_encoder_retain(self);
}

// SPECIAL: destroyImpl - marks encoder as destroyed
static void swizzled_destroyImpl(id self, SEL _cmd) {
    AGXMutexGuard guard;

    AGX_LOG("AGX Fix v2: destroyImpl on %p", self);

    // Mark as destroyed BEFORE calling original
    mark_encoder_destroyed(self);

    // Release our extra retain if we have one
    release_encoder_retain(self);

    // Call original
    typedef void (*Func)(id, SEL);
    ((Func)g_original_destroyImpl)(self, _cmd);
}

// SPECIAL: initWithQueue - register new encoder
static id swizzled_initWithQueue(id self, SEL _cmd, id queue) {
    typedef id (*Func)(id, SEL, id);
    id result = ((Func)g_original_initWithQueue)(self, _cmd, queue);

    if (result) {
        AGXMutexGuard guard;
        void* ptr = (__bridge void*)result;

        // Remove from destroyed set if address was reused
        g_destroyed_encoders.erase(ptr);

        // Immediately retain to keep alive until endEncoding
        if (g_retained_encoders.count(ptr) == 0) {
            CFRetain((__bridge CFTypeRef)result);
            g_retained_encoders[ptr] = 1;  // Initial count of 1
            g_valid_contexts.insert(ptr);
            g_encoder_retains++;
            AGX_LOG("AGX Fix v2: Retained new encoder %p at init (count=1)", ptr);
        }
    }

    return result;
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
    store_original_imp(selector, *outOriginal);  // Store for lookup
    method_setImplementation(method, newImpl);
    return true;
}

// ============================================================================
// Initialization
// ============================================================================

__attribute__((constructor))
static void agx_fix_v2_init() {
    g_log = os_log_create("com.agxfix.v2", "main");

    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix v2: Disabled via environment");
        return;
    }

    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    os_log(g_log, "AGX Fix v2: Initializing (fixing use-after-free + NULL _impl crashes)");

    // Get Metal device and create test encoder
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix v2: No Metal device");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];

    if (!encoder) {
        AGX_LOG_ERROR("AGX Fix v2: Failed to create encoder");
        return;
    }

    Class encoderClass = [encoder class];
    g_agx_context_class = encoderClass;
    os_log(g_log, "AGX Fix v2: Encoder class is %s", class_getName(encoderClass));

    [encoder endEncoding];

    // Discover _impl ivar offset
    Ivar implIvar = class_getInstanceVariable(encoderClass, "_impl");
    if (implIvar) {
        g_impl_ivar_offset = ivar_getOffset(implIvar);
        os_log(g_log, "AGX Fix v2: _impl at offset %td", g_impl_ivar_offset);
    } else {
        // Try parent classes
        Class parent = class_getSuperclass(encoderClass);
        while (parent) {
            implIvar = class_getInstanceVariable(parent, "_impl");
            if (implIvar) {
                g_impl_ivar_offset = ivar_getOffset(implIvar);
                os_log(g_log, "AGX Fix v2: _impl at offset %td in parent %s",
                       g_impl_ivar_offset, class_getName(parent));
                break;
            }
            parent = class_getSuperclass(parent);
        }
    }

    int swizzled_count = 0;
    IMP dummy;

    // Swizzle destroyImpl (CRITICAL)
    if (swizzle_method(encoderClass, @selector(destroyImpl),
                       (IMP)swizzled_destroyImpl, &g_original_destroyImpl)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2: Swizzled destroyImpl");
    }

    // Swizzle initWithQueue if available
    swizzle_method(encoderClass, @selector(initWithQueue:),
                   (IMP)swizzled_initWithQueue, &g_original_initWithQueue);

    // Swizzle all encoder methods
    #define SWIZZLE(sel, func) \
        if (swizzle_method(encoderClass, sel, (IMP)func, &dummy)) swizzled_count++;

    SWIZZLE(@selector(setComputePipelineState:), swizzled_setComputePipelineState)
    SWIZZLE(@selector(dispatchThreads:threadsPerThreadgroup:), swizzled_dispatchThreads)
    SWIZZLE(@selector(dispatchThreadgroups:threadsPerThreadgroup:), swizzled_dispatchThreadgroups)
    SWIZZLE(@selector(endEncoding), swizzled_endEncoding)
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

    // Additional methods for LayerNorm and complex operations
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

    os_log(g_log, "AGX Fix v2: COMPLETE - %d methods protected (use-after-free + NULL _impl)",
           swizzled_count);
}

// ============================================================================
// Statistics API
// ============================================================================

extern "C" {
    uint64_t agx_fix_v2_get_acquisitions() { return g_mutex_acquisitions.load(); }
    uint64_t agx_fix_v2_get_contentions() { return g_mutex_contentions.load(); }
    uint64_t agx_fix_v2_get_encoder_retains() { return g_encoder_retains.load(); }
    uint64_t agx_fix_v2_get_encoder_releases() { return g_encoder_releases.load(); }
    uint64_t agx_fix_v2_get_uaf_prevented() { return g_use_after_free_prevented.load(); }
    uint64_t agx_fix_v2_get_null_impl_skips() { return g_null_impl_skips.load(); }
    size_t agx_fix_v2_get_retained_count() {
        std::lock_guard<std::recursive_mutex> lock(g_agx_encoding_mutex);
        return g_retained_encoders.size();
    }
    bool agx_fix_v2_is_enabled() { return g_enabled; }
}
