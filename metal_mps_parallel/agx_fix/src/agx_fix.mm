/**
 * AGX Driver Race Condition Fix
 *
 * This library intercepts Apple's AGXMetalG16X driver methods and adds
 * proper synchronization to prevent the race conditions that cause crashes.
 *
 * USAGE:
 *   DYLD_INSERT_LIBRARIES=/path/to/libagx_fix.dylib your_app
 *
 * ROOT CAUSE:
 *   The AGX driver's destroyImpl method clears internal state (self->_impl)
 *   AFTER releasing a lock, creating a race window where another thread can
 *   access the context while it's being destroyed.
 *
 * FIX STRATEGY:
 *   A global mutex protects both encoder operations AND destroyImpl, ensuring
 *   mutual exclusion between context destruction and context usage.
 *
 * CRASH SITES FIXED:
 *   1. -[AGXG16XFamilyComputeContext setComputePipelineState:] (0x5c8 NULL deref)
 *   2. AGX::ComputeContext::prepareForEnqueue (0x98 NULL deref)
 *   3. AGX::SpillInfoGen3::allocateUSCSpillBuffer (0x184 NULL deref)
 *   4. -[AGXG16XFamilyComputeContext destroyImpl] (race window closed)
 *   5. -[AGXG16XFamilyComputeContext setBuffer:offset:atIndex:] (+56 NULL deref)
 *   6. ALL other encoder methods that access _impl (comprehensive protection)
 *
 * Created by Andrew Yates
 * Part of the MPS Parallel Inference research project
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <mutex>
#import <atomic>
#import <unordered_set>
#import <os/log.h>

// ============================================================================
// Configuration
// ============================================================================

// Environment variable to disable the fix (for testing)
#define AGX_FIX_DISABLE_ENV "AGX_FIX_DISABLE"

// Environment variable to enable verbose logging
#define AGX_FIX_VERBOSE_ENV "AGX_FIX_VERBOSE"

// ============================================================================
// Global State
// ============================================================================

namespace {
    // The global mutex that prevents the race condition
    std::mutex g_agx_encoding_mutex;

    // Set of valid (not-yet-destroyed) context pointers
    // Protected by g_agx_encoding_mutex
    std::unordered_set<void*> g_valid_contexts;

    // Set of DESTROYED context pointers (tombstones)
    // Used to detect use-after-destroy
    // Protected by g_agx_encoding_mutex
    std::unordered_set<void*> g_destroyed_contexts;

    // Statistics
    std::atomic<uint64_t> g_mutex_acquisitions{0};
    std::atomic<uint64_t> g_mutex_contentions{0};
    std::atomic<uint64_t> g_invalid_context_skips{0};

    // Logging
    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    // Original method implementations (saved during swizzle)
    IMP g_original_setComputePipelineState = nullptr;
    IMP g_original_dispatchThreads = nullptr;
    IMP g_original_dispatchThreadgroups = nullptr;
    IMP g_original_endEncoding = nullptr;
    IMP g_original_destroyImpl = nullptr;
    IMP g_original_initWithQueue = nullptr;

    // Additional encoder methods that access _impl (added after crash at setBuffer:offset:atIndex:)
    IMP g_original_setBuffer = nullptr;
    IMP g_original_setBuffers = nullptr;
    IMP g_original_setBytes = nullptr;
    IMP g_original_setTexture = nullptr;
    IMP g_original_setTextures = nullptr;
    IMP g_original_setSamplerState = nullptr;
    IMP g_original_setSamplerStates = nullptr;
    IMP g_original_setThreadgroupMemoryLength = nullptr;
    IMP g_original_useResource = nullptr;
    IMP g_original_useResources = nullptr;
    IMP g_original_useHeap = nullptr;
    IMP g_original_useHeaps = nullptr;
    IMP g_original_memoryBarrierWithScope = nullptr;
    IMP g_original_memoryBarrierWithResources = nullptr;
    IMP g_original_executeCommandsInBuffer = nullptr;
    IMP g_original_setStageInRegion = nullptr;
    IMP g_original_setImageblockWidth = nullptr;

    // Track the AGX class for destroyImpl swizzle
    Class g_agx_context_class = nullptr;

    // Offset of _impl ivar (discovered at runtime)
    // -1 means not yet discovered
    ptrdiff_t g_impl_ivar_offset = -1;

    // Statistics for NULL _impl detection
    std::atomic<uint64_t> g_null_impl_skips{0};
}

// ============================================================================
// Logging Helpers
// ============================================================================

#define AGX_LOG(format, ...) \
    do { \
        if (g_verbose && g_log) { \
            os_log(g_log, format, ##__VA_ARGS__); \
        } \
    } while(0)

#define AGX_LOG_ERROR(format, ...) \
    do { \
        if (g_log) { \
            os_log_error(g_log, format, ##__VA_ARGS__); \
        } \
    } while(0)

// ============================================================================
// Mutex Wrapper with Statistics
// ============================================================================

class AGXMutexGuard {
public:
    AGXMutexGuard() : locked_(false) {
        if (!g_enabled) return;

        // Try to lock without blocking first
        if (g_agx_encoding_mutex.try_lock()) {
            locked_ = true;
            g_mutex_acquisitions++;
            AGX_LOG("AGX Fix: Mutex acquired (no contention)");
        } else {
            // Contention - have to wait
            g_mutex_contentions++;
            g_agx_encoding_mutex.lock();
            locked_ = true;
            g_mutex_acquisitions++;
            AGX_LOG("AGX Fix: Mutex acquired (after contention)");
        }
    }

    ~AGXMutexGuard() {
        if (locked_) {
            g_agx_encoding_mutex.unlock();
            AGX_LOG("AGX Fix: Mutex released");
        }
    }

    // Non-copyable
    AGXMutexGuard(const AGXMutexGuard&) = delete;
    AGXMutexGuard& operator=(const AGXMutexGuard&) = delete;

private:
    bool locked_;
};

// ============================================================================
// Context Validity Tracking
// ============================================================================

// Register a context as valid
static void register_valid_context(void* ctx) {
    // Note: caller must hold mutex
    g_valid_contexts.insert(ctx);
    AGX_LOG("AGX Fix: Registered context %p (total: %zu)", ctx, g_valid_contexts.size());
}

// Unregister a context (called from destroyImpl)
static void unregister_context(void* ctx) {
    // Note: caller must hold mutex
    g_valid_contexts.erase(ctx);
    g_destroyed_contexts.insert(ctx);  // Mark as destroyed (tombstone)
    AGX_LOG("AGX Fix: Unregistered context %p (remaining: %zu, destroyed: %zu)",
            ctx, g_valid_contexts.size(), g_destroyed_contexts.size());

    // Limit tombstone set size to prevent unbounded growth
    // (addresses can be reused, so old tombstones become irrelevant)
    if (g_destroyed_contexts.size() > 10000) {
        g_destroyed_contexts.clear();
        AGX_LOG("AGX Fix: Cleared tombstone set (size exceeded 10000)");
    }
}

// Check if the internal _impl pointer is NULL
// This is the ROOT CAUSE check - if _impl is NULL, calling any method will crash
static bool is_impl_valid(id context) {
    if (g_impl_ivar_offset < 0) {
        // Ivar offset not discovered yet, assume valid
        return true;
    }

    // Read the _impl pointer directly from the object
    // _impl is a pointer at offset g_impl_ivar_offset from the object base
    char* obj_base = (char*)(__bridge void*)context;
    void** impl_ptr = (void**)(obj_base + g_impl_ivar_offset);
    void* impl = *impl_ptr;

    if (impl == nullptr) {
        g_null_impl_skips++;
        AGX_LOG("AGX Fix: NULL _impl detected in context %p", context);
        return false;
    }

    return true;
}

// Check if context is valid, registering new contexts lazily
// Returns false if context was DESTROYED (use-after-destroy detected)
static bool ensure_context_valid_or_register(void* ctx) {
    // Note: caller must hold mutex

    // Check if already registered as valid - takes priority over tombstones
    // (addresses can be reused after deallocation)
    if (g_valid_contexts.find(ctx) != g_valid_contexts.end()) {
        return true;  // Known valid context
    }

    // Unknown context - could be a new context at a reused address
    // Remove from tombstone set if present (address reuse)
    auto it = g_destroyed_contexts.find(ctx);
    if (it != g_destroyed_contexts.end()) {
        // Address was reused - this is a NEW context, not use-after-destroy
        g_destroyed_contexts.erase(it);
        AGX_LOG("AGX Fix: Address %p reused - removed from tombstones", ctx);
    }

    // Register the new context lazily
    // This handles contexts created before our swizzles were installed
    g_valid_contexts.insert(ctx);
    AGX_LOG("AGX Fix: Lazy registered context %p (total: %zu)", ctx, g_valid_contexts.size());
    return true;
}

// ============================================================================
// Swizzled Method Implementations
// ============================================================================

// Swizzled: -[MTLComputeCommandEncoder setComputePipelineState:]
// This is where crash site 1 occurs
static void swizzled_setComputePipelineState(id self, SEL _cmd, id pipelineState) {
    AGXMutexGuard guard;

    // Check validity and register new contexts lazily
    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setComputePipelineState on destroyed context %p", self);
        return;
    }

    // Check if internal _impl is valid (ROOT CAUSE of crashes)
    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setComputePipelineState - NULL _impl in %p", self);
        return;
    }

    // Call original implementation
    typedef void (*OriginalFunc)(id, SEL, id);
    ((OriginalFunc)g_original_setComputePipelineState)(self, _cmd, pipelineState);
}

// Swizzled: -[MTLComputeCommandEncoder dispatchThreads:threadsPerThreadgroup:]
// This triggers prepareForEnqueue internally
static void swizzled_dispatchThreads(id self, SEL _cmd, MTLSize threads, MTLSize threadsPerGroup) {
    AGXMutexGuard guard;

    // Check validity and register new contexts lazily
    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING dispatchThreads on destroyed context %p", self);
        return;
    }

    // Check if internal _impl is valid (ROOT CAUSE of crashes)
    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING dispatchThreads - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, MTLSize, MTLSize);
    ((OriginalFunc)g_original_dispatchThreads)(self, _cmd, threads, threadsPerGroup);
}

// Swizzled: -[MTLComputeCommandEncoder dispatchThreadgroups:threadsPerThreadgroup:]
static void swizzled_dispatchThreadgroups(id self, SEL _cmd, MTLSize groups, MTLSize threadsPerGroup) {
    AGXMutexGuard guard;

    // Check validity and register new contexts lazily
    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING dispatchThreadgroups on destroyed context %p", self);
        return;
    }

    // Check if internal _impl is valid (ROOT CAUSE of crashes)
    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING dispatchThreadgroups - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, MTLSize, MTLSize);
    ((OriginalFunc)g_original_dispatchThreadgroups)(self, _cmd, groups, threadsPerGroup);
}

// Swizzled: -[MTLComputeCommandEncoder endEncoding]
// IMPORTANT: endEncoding MUST always call the original method because Metal asserts
// "Command encoder released without endEncoding" on dealloc if endEncoding wasn't called.
// The NULL _impl crashes occur in setComputePipelineState/dispatchThreads, NOT endEncoding.
static void swizzled_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;

    // We still take the mutex to serialize with other encoder operations,
    // but we ALWAYS call the original - Metal requires it.
    AGX_LOG("AGX Fix: endEncoding called for context %p", self);

    typedef void (*OriginalFunc)(id, SEL);
    ((OriginalFunc)g_original_endEncoding)(self, _cmd);
}

// Swizzled: -[AGXG16XFamilyComputeContext destroyImpl]
// This is the ROOT CAUSE of the race condition - destroyImpl NULLs internal
// state while another thread may be using the context via encoder methods.
// By holding the same mutex, we ensure destroyImpl waits for any active
// encoder operation to complete before clearing state.
static void swizzled_destroyImpl(id self, SEL _cmd) {
    AGXMutexGuard guard;

    AGX_LOG("AGX Fix: destroyImpl called for context %p", self);

    // Mark context as invalid BEFORE calling original destroyImpl
    unregister_context((__bridge void*)self);

    typedef void (*OriginalFunc)(id, SEL);
    ((OriginalFunc)g_original_destroyImpl)(self, _cmd);

    AGX_LOG("AGX Fix: destroyImpl completed for context %p", self);
}

// Swizzled: -[AGXG16XFamilyComputeContext initWithQueue:]
// Track newly created contexts
static id swizzled_initWithQueue(id self, SEL _cmd, id queue) {
    typedef id (*OriginalFunc)(id, SEL, id);
    id result = ((OriginalFunc)g_original_initWithQueue)(self, _cmd, queue);

    if (result) {
        AGXMutexGuard guard;
        register_valid_context((__bridge void*)result);
    }

    return result;
}

// ============================================================================
// Additional Swizzled Methods (added after crash at setBuffer:offset:atIndex:)
// ALL encoder methods that access _impl must be protected!
// ============================================================================

// Swizzled: -[MTLComputeCommandEncoder setBuffer:offset:atIndex:]
// CRASH SITE: This is where the crash at setBuffer+56 occurred!
static void swizzled_setBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setBuffer on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setBuffer - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, id, NSUInteger, NSUInteger);
    ((OriginalFunc)g_original_setBuffer)(self, _cmd, buffer, offset, index);
}

// Swizzled: -[MTLComputeCommandEncoder setBuffers:offsets:withRange:]
static void swizzled_setBuffers(id self, SEL _cmd, const id* buffers, const NSUInteger* offsets, NSRange range) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setBuffers on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setBuffers - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, const id*, const NSUInteger*, NSRange);
    ((OriginalFunc)g_original_setBuffers)(self, _cmd, buffers, offsets, range);
}

// Swizzled: -[MTLComputeCommandEncoder setBytes:length:atIndex:]
static void swizzled_setBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setBytes on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setBytes - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, const void*, NSUInteger, NSUInteger);
    ((OriginalFunc)g_original_setBytes)(self, _cmd, bytes, length, index);
}

// Swizzled: -[MTLComputeCommandEncoder setTexture:atIndex:]
static void swizzled_setTexture(id self, SEL _cmd, id texture, NSUInteger index) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setTexture on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setTexture - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, id, NSUInteger);
    ((OriginalFunc)g_original_setTexture)(self, _cmd, texture, index);
}

// Swizzled: -[MTLComputeCommandEncoder setTextures:withRange:]
static void swizzled_setTextures(id self, SEL _cmd, const id* textures, NSRange range) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setTextures on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setTextures - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, const id*, NSRange);
    ((OriginalFunc)g_original_setTextures)(self, _cmd, textures, range);
}

// Swizzled: -[MTLComputeCommandEncoder setSamplerState:atIndex:]
static void swizzled_setSamplerState(id self, SEL _cmd, id sampler, NSUInteger index) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setSamplerState on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setSamplerState - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, id, NSUInteger);
    ((OriginalFunc)g_original_setSamplerState)(self, _cmd, sampler, index);
}

// Swizzled: -[MTLComputeCommandEncoder setSamplerStates:withRange:]
static void swizzled_setSamplerStates(id self, SEL _cmd, const id* samplers, NSRange range) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setSamplerStates on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setSamplerStates - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, const id*, NSRange);
    ((OriginalFunc)g_original_setSamplerStates)(self, _cmd, samplers, range);
}

// Swizzled: -[MTLComputeCommandEncoder setThreadgroupMemoryLength:atIndex:]
static void swizzled_setThreadgroupMemoryLength(id self, SEL _cmd, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setThreadgroupMemoryLength on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setThreadgroupMemoryLength - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, NSUInteger, NSUInteger);
    ((OriginalFunc)g_original_setThreadgroupMemoryLength)(self, _cmd, length, index);
}

// Swizzled: -[MTLComputeCommandEncoder useResource:usage:]
static void swizzled_useResource(id self, SEL _cmd, id resource, NSUInteger usage) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING useResource on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING useResource - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, id, NSUInteger);
    ((OriginalFunc)g_original_useResource)(self, _cmd, resource, usage);
}

// Swizzled: -[MTLComputeCommandEncoder useResources:count:usage:]
static void swizzled_useResources(id self, SEL _cmd, const id* resources, NSUInteger count, NSUInteger usage) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING useResources on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING useResources - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, const id*, NSUInteger, NSUInteger);
    ((OriginalFunc)g_original_useResources)(self, _cmd, resources, count, usage);
}

// Swizzled: -[MTLComputeCommandEncoder useHeap:]
static void swizzled_useHeap(id self, SEL _cmd, id heap) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING useHeap on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING useHeap - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, id);
    ((OriginalFunc)g_original_useHeap)(self, _cmd, heap);
}

// Swizzled: -[MTLComputeCommandEncoder useHeaps:count:]
static void swizzled_useHeaps(id self, SEL _cmd, const id* heaps, NSUInteger count) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING useHeaps on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING useHeaps - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, const id*, NSUInteger);
    ((OriginalFunc)g_original_useHeaps)(self, _cmd, heaps, count);
}

// Swizzled: -[MTLComputeCommandEncoder memoryBarrierWithScope:]
static void swizzled_memoryBarrierWithScope(id self, SEL _cmd, NSUInteger scope) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING memoryBarrierWithScope on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING memoryBarrierWithScope - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, NSUInteger);
    ((OriginalFunc)g_original_memoryBarrierWithScope)(self, _cmd, scope);
}

// Swizzled: -[MTLComputeCommandEncoder memoryBarrierWithResources:count:]
static void swizzled_memoryBarrierWithResources(id self, SEL _cmd, const id* resources, NSUInteger count) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING memoryBarrierWithResources on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING memoryBarrierWithResources - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, const id*, NSUInteger);
    ((OriginalFunc)g_original_memoryBarrierWithResources)(self, _cmd, resources, count);
}

// Swizzled: -[MTLComputeCommandEncoder executeCommandsInBuffer:withRange:]
static void swizzled_executeCommandsInBuffer(id self, SEL _cmd, id buffer, NSRange range) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING executeCommandsInBuffer on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING executeCommandsInBuffer - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, id, NSRange);
    ((OriginalFunc)g_original_executeCommandsInBuffer)(self, _cmd, buffer, range);
}

// Swizzled: -[MTLComputeCommandEncoder setStageInRegion:]
static void swizzled_setStageInRegion(id self, SEL _cmd, MTLRegion region) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setStageInRegion on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setStageInRegion - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, MTLRegion);
    ((OriginalFunc)g_original_setStageInRegion)(self, _cmd, region);
}

// Swizzled: -[MTLComputeCommandEncoder setImageblockWidth:height:]
static void swizzled_setImageblockWidth(id self, SEL _cmd, NSUInteger width, NSUInteger height) {
    AGXMutexGuard guard;

    if (!ensure_context_valid_or_register((__bridge void*)self)) {
        g_invalid_context_skips++;
        AGX_LOG("AGX Fix: SKIPPING setImageblockWidth on destroyed context %p", self);
        return;
    }

    if (!is_impl_valid(self)) {
        AGX_LOG("AGX Fix: SKIPPING setImageblockWidth - NULL _impl in %p", self);
        return;
    }

    typedef void (*OriginalFunc)(id, SEL, NSUInteger, NSUInteger);
    ((OriginalFunc)g_original_setImageblockWidth)(self, _cmd, width, height);
}

// ============================================================================
// Swizzle Helper
// ============================================================================

static bool swizzle_method(Class cls, SEL selector, IMP newImpl, IMP* outOriginal) {
    Method method = class_getInstanceMethod(cls, selector);
    if (!method) {
        AGX_LOG_ERROR("AGX Fix: Failed to find method %s", sel_getName(selector));
        return false;
    }

    *outOriginal = method_getImplementation(method);
    method_setImplementation(method, newImpl);

    AGX_LOG("AGX Fix: Swizzled %s", sel_getName(selector));
    return true;
}

// ============================================================================
// Initialization
// ============================================================================

__attribute__((constructor))
static void agx_fix_init() {
    // Initialize logging
    g_log = os_log_create("com.agxfix", "main");

    // Check environment variables
    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix: Disabled via environment variable");
        return;
    }

    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    os_log(g_log, "AGX Fix: Initializing (fixing Apple AGX driver race conditions)");

    // NOTE: AGX driver is loaded lazily when Metal is first accessed.
    // We cannot find the AGX class at constructor time. Instead, we'll
    // swizzle destroyImpl after we create an encoder (when driver is loaded).

    // Get a Metal device to find the actual encoder class
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix: No Metal device available");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        AGX_LOG_ERROR("AGX Fix: Failed to create command queue");
        return;
    }

    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    if (!buffer) {
        AGX_LOG_ERROR("AGX Fix: Failed to create command buffer");
        return;
    }

    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) {
        AGX_LOG_ERROR("AGX Fix: Failed to create compute encoder");
        return;
    }

    // Get the actual class of the encoder
    // NOTE: On Apple Silicon, the encoder class IS the AGX context class
    // (e.g., AGXG16XFamilyComputeContext). This is where we swizzle destroyImpl.
    Class encoderClass = [encoder class];
    os_log(g_log, "AGX Fix: Encoder class is %s", class_getName(encoderClass));
    g_agx_context_class = encoderClass;

    // End the encoder before swizzling
    [encoder endEncoding];

    // Discover the _impl ivar offset - this is CRITICAL for crash prevention
    // The _impl pointer is where the actual GPU state lives
    Ivar implIvar = class_getInstanceVariable(encoderClass, "_impl");
    if (implIvar) {
        g_impl_ivar_offset = ivar_getOffset(implIvar);
        os_log(g_log, "AGX Fix: Found _impl ivar at offset %td in %s",
               g_impl_ivar_offset, class_getName(encoderClass));
    } else {
        // Try parent classes - _impl might be inherited
        Class parentClass = class_getSuperclass(encoderClass);
        while (parentClass) {
            implIvar = class_getInstanceVariable(parentClass, "_impl");
            if (implIvar) {
                g_impl_ivar_offset = ivar_getOffset(implIvar);
                os_log(g_log, "AGX Fix: Found _impl ivar at offset %td in parent %s",
                       g_impl_ivar_offset, class_getName(parentClass));
                break;
            }
            parentClass = class_getSuperclass(parentClass);
        }
        if (g_impl_ivar_offset < 0) {
            AGX_LOG_ERROR("AGX Fix: CRITICAL - Could not find _impl ivar! NULL checks disabled.");
        }
    }

    // Swizzle destroyImpl on the encoder/context class
    // This is the ROOT CAUSE fix - prevents destroyImpl from clearing
    // internal state while another thread is using the context
    if (swizzle_method(encoderClass,
                      @selector(destroyImpl),
                      (IMP)swizzled_destroyImpl,
                      &g_original_destroyImpl)) {
        os_log(g_log, "AGX Fix: destroyImpl swizzled on %s", class_getName(encoderClass));
    } else {
        AGX_LOG_ERROR("AGX Fix: CRITICAL - Failed to swizzle destroyImpl on %s!", class_getName(encoderClass));
    }

    // Try to swizzle initWithQueue: for early context registration
    // NOT CRITICAL: We use lazy registration in encoder methods as fallback
    if (swizzle_method(encoderClass,
                      @selector(initWithQueue:),
                      (IMP)swizzled_initWithQueue,
                      &g_original_initWithQueue)) {
        os_log(g_log, "AGX Fix: initWithQueue: swizzled on %s", class_getName(encoderClass));
    } else {
        // Expected - the selector name varies by driver version
        // Lazy registration in encoder methods will handle context tracking
        os_log(g_log, "AGX Fix: initWithQueue: not found (using lazy registration)");
    }

    // Swizzle encoder methods - CORE methods (must succeed)
    bool success = true;
    int swizzled_count = 0;

    success &= swizzle_method(encoderClass,
                              @selector(setComputePipelineState:),
                              (IMP)swizzled_setComputePipelineState,
                              &g_original_setComputePipelineState);
    if (g_original_setComputePipelineState) swizzled_count++;

    success &= swizzle_method(encoderClass,
                              @selector(dispatchThreads:threadsPerThreadgroup:),
                              (IMP)swizzled_dispatchThreads,
                              &g_original_dispatchThreads);
    if (g_original_dispatchThreads) swizzled_count++;

    success &= swizzle_method(encoderClass,
                              @selector(dispatchThreadgroups:threadsPerThreadgroup:),
                              (IMP)swizzled_dispatchThreadgroups,
                              &g_original_dispatchThreadgroups);
    if (g_original_dispatchThreadgroups) swizzled_count++;

    success &= swizzle_method(encoderClass,
                              @selector(endEncoding),
                              (IMP)swizzled_endEncoding,
                              &g_original_endEncoding);
    if (g_original_endEncoding) swizzled_count++;

    // ========================================================================
    // ADDITIONAL encoder methods - ALL methods that access _impl must be protected!
    // Added after crash in setBuffer:offset:atIndex: revealed gaps in coverage
    // ========================================================================

    // setBuffer:offset:atIndex: - CRITICAL: This is where the crash occurred!
    if (swizzle_method(encoderClass,
                       @selector(setBuffer:offset:atIndex:),
                       (IMP)swizzled_setBuffer,
                       &g_original_setBuffer)) {
        swizzled_count++;
    }

    // setBuffers:offsets:withRange:
    if (swizzle_method(encoderClass,
                       @selector(setBuffers:offsets:withRange:),
                       (IMP)swizzled_setBuffers,
                       &g_original_setBuffers)) {
        swizzled_count++;
    }

    // setBytes:length:atIndex:
    if (swizzle_method(encoderClass,
                       @selector(setBytes:length:atIndex:),
                       (IMP)swizzled_setBytes,
                       &g_original_setBytes)) {
        swizzled_count++;
    }

    // setTexture:atIndex:
    if (swizzle_method(encoderClass,
                       @selector(setTexture:atIndex:),
                       (IMP)swizzled_setTexture,
                       &g_original_setTexture)) {
        swizzled_count++;
    }

    // setTextures:withRange:
    if (swizzle_method(encoderClass,
                       @selector(setTextures:withRange:),
                       (IMP)swizzled_setTextures,
                       &g_original_setTextures)) {
        swizzled_count++;
    }

    // setSamplerState:atIndex:
    if (swizzle_method(encoderClass,
                       @selector(setSamplerState:atIndex:),
                       (IMP)swizzled_setSamplerState,
                       &g_original_setSamplerState)) {
        swizzled_count++;
    }

    // setSamplerStates:withRange:
    if (swizzle_method(encoderClass,
                       @selector(setSamplerStates:withRange:),
                       (IMP)swizzled_setSamplerStates,
                       &g_original_setSamplerStates)) {
        swizzled_count++;
    }

    // setThreadgroupMemoryLength:atIndex:
    if (swizzle_method(encoderClass,
                       @selector(setThreadgroupMemoryLength:atIndex:),
                       (IMP)swizzled_setThreadgroupMemoryLength,
                       &g_original_setThreadgroupMemoryLength)) {
        swizzled_count++;
    }

    // useResource:usage:
    if (swizzle_method(encoderClass,
                       @selector(useResource:usage:),
                       (IMP)swizzled_useResource,
                       &g_original_useResource)) {
        swizzled_count++;
    }

    // useResources:count:usage:
    if (swizzle_method(encoderClass,
                       @selector(useResources:count:usage:),
                       (IMP)swizzled_useResources,
                       &g_original_useResources)) {
        swizzled_count++;
    }

    // useHeap:
    if (swizzle_method(encoderClass,
                       @selector(useHeap:),
                       (IMP)swizzled_useHeap,
                       &g_original_useHeap)) {
        swizzled_count++;
    }

    // useHeaps:count:
    if (swizzle_method(encoderClass,
                       @selector(useHeaps:count:),
                       (IMP)swizzled_useHeaps,
                       &g_original_useHeaps)) {
        swizzled_count++;
    }

    // memoryBarrierWithScope:
    if (swizzle_method(encoderClass,
                       @selector(memoryBarrierWithScope:),
                       (IMP)swizzled_memoryBarrierWithScope,
                       &g_original_memoryBarrierWithScope)) {
        swizzled_count++;
    }

    // memoryBarrierWithResources:count:
    if (swizzle_method(encoderClass,
                       @selector(memoryBarrierWithResources:count:),
                       (IMP)swizzled_memoryBarrierWithResources,
                       &g_original_memoryBarrierWithResources)) {
        swizzled_count++;
    }

    // executeCommandsInBuffer:withRange:
    if (swizzle_method(encoderClass,
                       @selector(executeCommandsInBuffer:withRange:),
                       (IMP)swizzled_executeCommandsInBuffer,
                       &g_original_executeCommandsInBuffer)) {
        swizzled_count++;
    }

    // setStageInRegion:
    if (swizzle_method(encoderClass,
                       @selector(setStageInRegion:),
                       (IMP)swizzled_setStageInRegion,
                       &g_original_setStageInRegion)) {
        swizzled_count++;
    }

    // setImageblockWidth:height:
    if (swizzle_method(encoderClass,
                       @selector(setImageblockWidth:height:),
                       (IMP)swizzled_setImageblockWidth,
                       &g_original_setImageblockWidth)) {
        swizzled_count++;
    }

    os_log(g_log, "AGX Fix: Swizzled %d encoder methods on %s", swizzled_count, class_getName(encoderClass));

    if (success) {
        os_log(g_log, "AGX Fix: Successfully installed (core encoder methods swizzled)");
    } else {
        AGX_LOG_ERROR("AGX Fix: Partial installation (some core encoder methods failed)");
    }

    // Final status
    if (g_original_destroyImpl && success) {
        os_log(g_log, "AGX Fix: COMPLETE - destroyImpl + %d encoder methods protected", swizzled_count);
    } else if (g_original_destroyImpl) {
        os_log(g_log, "AGX Fix: PARTIAL - destroyImpl protected, some encoder methods failed");
    } else if (success) {
        os_log(g_log, "AGX Fix: PARTIAL - encoder methods protected, destroyImpl NOT protected");
    } else {
        AGX_LOG_ERROR("AGX Fix: FAILED - neither destroyImpl nor encoder methods protected");
    }
}

// ============================================================================
// Statistics API (for testing/debugging)
// ============================================================================

extern "C" {
    uint64_t agx_fix_get_acquisitions() {
        return g_mutex_acquisitions.load();
    }

    uint64_t agx_fix_get_contentions() {
        return g_mutex_contentions.load();
    }

    uint64_t agx_fix_get_invalid_skips() {
        return g_invalid_context_skips.load();
    }

    size_t agx_fix_get_tracked_contexts() {
        std::lock_guard<std::mutex> lock(g_agx_encoding_mutex);
        return g_valid_contexts.size();
    }

    size_t agx_fix_get_destroyed_contexts() {
        std::lock_guard<std::mutex> lock(g_agx_encoding_mutex);
        return g_destroyed_contexts.size();
    }

    uint64_t agx_fix_get_null_impl_skips() {
        return g_null_impl_skips.load();
    }

    ptrdiff_t agx_fix_get_impl_offset() {
        return g_impl_ivar_offset;
    }

    void agx_fix_reset_stats() {
        g_mutex_acquisitions = 0;
        g_mutex_contentions = 0;
        g_invalid_context_skips = 0;
    }

    bool agx_fix_is_enabled() {
        return g_enabled;
    }
}
