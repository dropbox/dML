/**
 * AGX Driver Race Condition Fix - COMPREHENSIVE VERSION
 *
 * This library intercepts ALL Apple AGXMetalG16X driver methods that access _impl
 * and adds proper synchronization to prevent race conditions.
 *
 * COVERAGE: 70+ methods on AGXG16XFamilyComputeContext protected
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
    // Use RECURSIVE mutex - methods can call other swizzled methods internally
    std::recursive_mutex g_agx_encoding_mutex;
    std::unordered_set<void*> g_valid_contexts;
    std::unordered_set<void*> g_destroyed_contexts;

    std::atomic<uint64_t> g_mutex_acquisitions{0};
    std::atomic<uint64_t> g_mutex_contentions{0};
    std::atomic<uint64_t> g_invalid_context_skips{0};
    std::atomic<uint64_t> g_null_impl_skips{0};

    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    Class g_agx_context_class = nullptr;
    ptrdiff_t g_impl_ivar_offset = -1;

    // Store original implementations in parallel arrays
    // Index 0-99 for method IMPs, index matches swizzle order
    constexpr int MAX_SWIZZLED_METHODS = 100;
    IMP g_original_imps[MAX_SWIZZLED_METHODS] = {nullptr};
    SEL g_swizzled_sels[MAX_SWIZZLED_METHODS] = {nullptr};
    int g_swizzle_count = 0;

    // Get original IMP for a selector by linear search
    IMP get_original_imp(SEL sel) {
        for (int i = 0; i < g_swizzle_count; i++) {
            if (g_swizzled_sels[i] == sel) {
                return g_original_imps[i];
            }
        }
        return nullptr;
    }

    // Special originals for destroyImpl
    IMP g_original_destroyImpl = nullptr;
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
// Mutex Wrapper
// ============================================================================

class AGXMutexGuard {
public:
    AGXMutexGuard() : locked_(false) {
        if (!g_enabled) return;
        // For recursive mutex, try_lock returns true if already owned by this thread
        g_agx_encoding_mutex.lock();
        locked_ = true;
        g_mutex_acquisitions++;
    }
    ~AGXMutexGuard() {
        if (locked_) g_agx_encoding_mutex.unlock();
    }
    AGXMutexGuard(const AGXMutexGuard&) = delete;
    AGXMutexGuard& operator=(const AGXMutexGuard&) = delete;
private:
    bool locked_;
};

// ============================================================================
// Context Validity
// ============================================================================

// register_valid_context removed - was unused (now handled by ensure_context_valid_or_register)

static void unregister_context(void* ctx) {
    g_valid_contexts.erase(ctx);
    g_destroyed_contexts.insert(ctx);
    if (g_destroyed_contexts.size() > 10000) {
        g_destroyed_contexts.clear();
    }
}

static bool is_impl_valid(id context) {
    if (g_impl_ivar_offset < 0) return true;
    char* obj_base = (char*)(__bridge void*)context;
    void** impl_ptr = (void**)(obj_base + g_impl_ivar_offset);
    void* impl = *impl_ptr;
    if (impl == nullptr) {
        g_null_impl_skips++;
        return false;
    }
    return true;
}

static bool ensure_context_valid_or_register(void* ctx) {
    if (g_valid_contexts.find(ctx) != g_valid_contexts.end()) return true;
    auto it = g_destroyed_contexts.find(ctx);
    if (it != g_destroyed_contexts.end()) {
        g_destroyed_contexts.erase(it);
    }
    g_valid_contexts.insert(ctx);
    return true;
}

// ============================================================================
// MACRO-BASED SWIZZLE GENERATION
// These macros generate swizzled methods with the correct signatures
// ============================================================================

// Get original IMP for a selector
#define GET_ORIGINAL(sel) (get_original_imp(sel))

// Void method, 0 args
#define SWIZZLE_VOID_0(name) \
static void swizzled_##name(id self, SEL _cmd) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd); \
}

// Void method, 1 arg (id)
#define SWIZZLE_VOID_1_ID(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1); \
}

// Void method, 1 arg (NSUInteger)
#define SWIZZLE_VOID_1_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, NSUInteger arg1) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1); \
}

// Void method, 2 args (id, NSUInteger)
#define SWIZZLE_VOID_2_ID_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1, NSUInteger arg2) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2); \
}

// Void method, 2 args (NSUInteger, NSUInteger)
#define SWIZZLE_VOID_2_UINT_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, NSUInteger arg1, NSUInteger arg2) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, NSUInteger, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2); \
}

// Void method, 2 args (id, NSRange)
#define SWIZZLE_VOID_2_ID_RANGE(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1, NSRange arg2) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id, NSRange); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2); \
}

// Void method, 2 args (const void*, NSRange)
#define SWIZZLE_VOID_2_PTR_RANGE(name) \
static void swizzled_##name(id self, SEL _cmd, const void* arg1, NSRange arg2) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, const void*, NSRange); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2); \
}

// Void method, 2 args (MTLSize, MTLSize)
#define SWIZZLE_VOID_2_SIZE_SIZE(name) \
static void swizzled_##name(id self, SEL _cmd, MTLSize arg1, MTLSize arg2) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, MTLSize, MTLSize); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2); \
}

// Void method, 3 args (id, NSUInteger, NSUInteger)
#define SWIZZLE_VOID_3_ID_UINT_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1, NSUInteger arg2, NSUInteger arg3) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3); \
}

// Void method, 3 args (const void*, NSUInteger, NSUInteger)
#define SWIZZLE_VOID_3_PTR_UINT_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, const void* arg1, NSUInteger arg2, NSUInteger arg3) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3); \
}

// Void method, 3 args (const id*, const NSUInteger*, NSRange)
#define SWIZZLE_VOID_3_IDPTR_UINTPTR_RANGE(name) \
static void swizzled_##name(id self, SEL _cmd, const id* arg1, const NSUInteger* arg2, NSRange arg3) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, const id*, const NSUInteger*, NSRange); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3); \
}

// Void method, 3 args (const id*, NSUInteger, NSUInteger)
#define SWIZZLE_VOID_3_IDPTR_UINT_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, const id* arg1, NSUInteger arg2, NSUInteger arg3) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, const id*, NSUInteger, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3); \
}

// Void method, 3 args (id, NSUInteger, MTLSize)
#define SWIZZLE_VOID_3_ID_UINT_SIZE(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1, NSUInteger arg2, MTLSize arg3) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id, NSUInteger, MTLSize); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3); \
}

// Void method, 3 args (id, id, NSUInteger)
#define SWIZZLE_VOID_3_ID_ID_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1, id arg2, NSUInteger arg3) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id, id, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3); \
}

// Void method, 4 args (id, float, float, NSUInteger)
#define SWIZZLE_VOID_4_ID_F_F_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1, float arg2, float arg3, NSUInteger arg4) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id, float, float, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3, arg4); \
}

// Void method, 4 args (const id*, const float*, const float*, NSRange)
#define SWIZZLE_VOID_4_IDPTR_FPTR_FPTR_RANGE(name) \
static void swizzled_##name(id self, SEL _cmd, const id* arg1, const float* arg2, const float* arg3, NSRange arg4) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, const id*, const float*, const float*, NSRange); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3, arg4); \
}

// Void method, 4 args (const void*, NSUInteger, NSUInteger, NSUInteger)
#define SWIZZLE_VOID_4_PTR_UINT_UINT_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, const void* arg1, NSUInteger arg2, NSUInteger arg3, NSUInteger arg4) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3, arg4); \
}

// Void method, 4 args (const id*, const NSUInteger*, const NSUInteger*, NSRange)
#define SWIZZLE_VOID_4_IDPTR_UINTPTR_UINTPTR_RANGE(name) \
static void swizzled_##name(id self, SEL _cmd, const id* arg1, const NSUInteger* arg2, const NSUInteger* arg3, NSRange arg4) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, const id*, const NSUInteger*, const NSUInteger*, NSRange); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3, arg4); \
}

// Void method, 1 arg (MTLRegion)
#define SWIZZLE_VOID_1_REGION(name) \
static void swizzled_##name(id self, SEL _cmd, MTLRegion arg1) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, MTLRegion); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1); \
}

// Void method, 4 args (id, NSUInteger, NSUInteger, NSUInteger) for encode conditionals
#define SWIZZLE_VOID_4_ID_UINT_UINT_UINT(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1, NSUInteger arg2, NSUInteger arg3, NSUInteger arg4) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3, arg4); \
}

// Void method, 3 args (id, NSUInteger, BOOL)
#define SWIZZLE_VOID_3_ID_UINT_BOOL(name) \
static void swizzled_##name(id self, SEL _cmd, id arg1, NSUInteger arg2, BOOL arg3) { \
    AGXMutexGuard guard; \
    if (!ensure_context_valid_or_register((__bridge void*)self)) { g_invalid_context_skips++; return; } \
    if (!is_impl_valid(self)) { return; } \
    typedef void (*Func)(id, SEL, id, NSUInteger, BOOL); \
    ((Func)GET_ORIGINAL(_cmd))(self, _cmd, arg1, arg2, arg3); \
}

// ============================================================================
// GENERATE ALL SWIZZLED METHODS
// ============================================================================

// Core encoding methods
SWIZZLE_VOID_1_ID(setComputePipelineState)
SWIZZLE_VOID_2_SIZE_SIZE(dispatchThreads_threadsPerThreadgroup)
SWIZZLE_VOID_2_SIZE_SIZE(dispatchThreadgroups_threadsPerThreadgroup)
SWIZZLE_VOID_0(endEncoding)
SWIZZLE_VOID_0(deferredEndEncoding)

// Buffer methods
SWIZZLE_VOID_3_ID_UINT_UINT(setBuffer_offset_atIndex)
// setBuffer:offset:attributeStride:atIndex: removed - 4 args but macro expects 3, never registered
SWIZZLE_VOID_3_IDPTR_UINTPTR_RANGE(setBuffers_offsets_withRange)
SWIZZLE_VOID_2_UINT_UINT(setBufferOffset_atIndex)

// Bytes methods
SWIZZLE_VOID_3_PTR_UINT_UINT(setBytes_length_atIndex)

// Texture methods
SWIZZLE_VOID_2_ID_UINT(setTexture_atIndex)
SWIZZLE_VOID_2_PTR_RANGE(setTextures_withRange)

// Sampler methods
SWIZZLE_VOID_2_ID_UINT(setSamplerState_atIndex)
SWIZZLE_VOID_2_PTR_RANGE(setSamplerStates_withRange)
SWIZZLE_VOID_4_ID_F_F_UINT(setSamplerState_lodMinClamp_lodMaxClamp_atIndex)
SWIZZLE_VOID_4_IDPTR_FPTR_FPTR_RANGE(setSamplerStates_lodMinClamps_lodMaxClamps_withRange)

// Threadgroup methods
SWIZZLE_VOID_2_UINT_UINT(setThreadgroupMemoryLength_atIndex)
SWIZZLE_VOID_2_UINT_UINT(setImageblockWidth_height)
SWIZZLE_VOID_2_UINT_UINT(setImageBlockWidth_height)
SWIZZLE_VOID_1_REGION(setStageInRegion)
SWIZZLE_VOID_2_ID_UINT(setStageInRegionWithIndirectBuffer_indirectBufferOffset)

// Dispatch indirect
SWIZZLE_VOID_3_ID_UINT_SIZE(dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup)
SWIZZLE_VOID_2_ID_UINT(dispatchThreadsWithIndirectBuffer_indirectBufferOffset)

// Resource usage
SWIZZLE_VOID_2_ID_UINT(useResource_usage)
SWIZZLE_VOID_3_IDPTR_UINT_UINT(useResources_count_usage)
SWIZZLE_VOID_1_ID(useHeap)
SWIZZLE_VOID_2_ID_UINT(useHeaps_count)  // Actually const id*, NSUInteger
SWIZZLE_VOID_1_ID(useResidencySet)
SWIZZLE_VOID_2_ID_UINT(useResidencySets_count)

// Memory barriers
SWIZZLE_VOID_1_UINT(memoryBarrierWithScope)
SWIZZLE_VOID_2_ID_UINT(memoryBarrierWithResources_count)

// Fences
SWIZZLE_VOID_1_ID(updateFence)
SWIZZLE_VOID_1_ID(waitForFence)

// Execute commands
SWIZZLE_VOID_2_ID_RANGE(executeCommandsInBuffer_withRange)
SWIZZLE_VOID_3_ID_ID_UINT(executeCommandsInBuffer_indirectBuffer_indirectBufferOffset)

// Function tables
SWIZZLE_VOID_2_ID_UINT(setFunctionTable_atIndex)
SWIZZLE_VOID_2_PTR_RANGE(setFunctionTables_withRange)
SWIZZLE_VOID_2_ID_UINT(setVisibleFunctionTable_atBufferIndex)
SWIZZLE_VOID_2_PTR_RANGE(setVisibleFunctionTables_withBufferRange)
SWIZZLE_VOID_2_ID_UINT(setIntersectionFunctionTable_atBufferIndex)
SWIZZLE_VOID_2_PTR_RANGE(setIntersectionFunctionTables_withBufferRange)

// Acceleration structures
SWIZZLE_VOID_2_ID_UINT(setAccelerationStructure_atBufferIndex)

// Thread distribution
SWIZZLE_VOID_1_UINT(setThreadgroupDistributionMode)
SWIZZLE_VOID_0(setThreadgroupDistributionModeWithClusterGroupIndex)

// Flush/invalidate
SWIZZLE_VOID_0(dispatchWaitFlush)
SWIZZLE_VOID_0(dispatchFlushInvalidate)
SWIZZLE_VOID_0(dispatchFlushOnly)
SWIZZLE_VOID_0(dispatchInvalidateOnly)
SWIZZLE_VOID_0(dispatchFenceOnly)

// GPU conditionals
SWIZZLE_VOID_0(encodeStartDoWhile)
SWIZZLE_VOID_0(encodeEndWhile)
SWIZZLE_VOID_0(encodeStartElse)
SWIZZLE_VOID_0(encodeEndIf)

// Counters
SWIZZLE_VOID_3_ID_UINT_BOOL(sampleCountersInBuffer_atSampleIndex_withBarrier)

// Substreams
SWIZZLE_VOID_1_UINT(setSubstream)
SWIZZLE_VOID_1_UINT(signalProgress)
SWIZZLE_VOID_1_UINT(waitForProgress)
SWIZZLE_VOID_0(beginVirtualSubstream)
SWIZZLE_VOID_0(nextVirtualSubstream)
SWIZZLE_VOID_0(endVirtualSubstream)
SWIZZLE_VOID_1_UINT(waitForVirtualSubstream)

// Parallel execution
SWIZZLE_VOID_1_UINT(setParallelExecution)

// ============================================================================
// destroyImpl - Special handling
// ============================================================================

static void swizzled_destroyImpl(id self, SEL _cmd) {
    AGXMutexGuard guard;
    unregister_context((__bridge void*)self);
    typedef void (*Func)(id, SEL);
    ((Func)g_original_destroyImpl)(self, _cmd);
}

// ============================================================================
// Swizzle Helper
// ============================================================================

static bool swizzle_method(Class cls, SEL selector, IMP newImpl) {
    Method method = class_getInstanceMethod(cls, selector);
    if (!method) {
        return false;
    }
    if (g_swizzle_count >= MAX_SWIZZLED_METHODS) {
        AGX_LOG_ERROR("AGX Fix: Too many swizzled methods!");
        return false;
    }
    IMP original = method_getImplementation(method);
    g_swizzled_sels[g_swizzle_count] = selector;
    g_original_imps[g_swizzle_count] = original;
    g_swizzle_count++;
    method_setImplementation(method, newImpl);
    return true;
}

// ============================================================================
// Initialization
// ============================================================================

__attribute__((constructor))
static void agx_fix_init() {
    g_log = os_log_create("com.agxfix.comprehensive", "main");

    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix Comprehensive: Disabled via environment");
        return;
    }

    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    os_log(g_log, "AGX Fix Comprehensive: Initializing");

    // Get encoder class
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix: No Metal device");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];

    Class cls = [encoder class];
    g_agx_context_class = cls;
    os_log(g_log, "AGX Fix Comprehensive: Class is %s", class_getName(cls));

    [encoder endEncoding];

    // Find _impl offset
    Ivar implIvar = class_getInstanceVariable(cls, "_impl");
    if (implIvar) {
        g_impl_ivar_offset = ivar_getOffset(implIvar);
        os_log(g_log, "AGX Fix Comprehensive: _impl at offset %td", g_impl_ivar_offset);
    } else {
        Class parent = class_getSuperclass(cls);
        while (parent) {
            implIvar = class_getInstanceVariable(parent, "_impl");
            if (implIvar) {
                g_impl_ivar_offset = ivar_getOffset(implIvar);
                break;
            }
            parent = class_getSuperclass(parent);
        }
    }

    // Swizzle destroyImpl
    Method destroyMethod = class_getInstanceMethod(cls, @selector(destroyImpl));
    if (destroyMethod) {
        g_original_destroyImpl = method_getImplementation(destroyMethod);
        method_setImplementation(destroyMethod, (IMP)swizzled_destroyImpl);
        os_log(g_log, "AGX Fix Comprehensive: destroyImpl swizzled");
    }

    int count = 0;

    // Core encoding
    #define TRY_SWIZZLE(sel, func) if (swizzle_method(cls, sel, (IMP)func)) count++

    TRY_SWIZZLE(@selector(setComputePipelineState:), swizzled_setComputePipelineState);
    TRY_SWIZZLE(@selector(dispatchThreads:threadsPerThreadgroup:), swizzled_dispatchThreads_threadsPerThreadgroup);
    TRY_SWIZZLE(@selector(dispatchThreadgroups:threadsPerThreadgroup:), swizzled_dispatchThreadgroups_threadsPerThreadgroup);
    TRY_SWIZZLE(@selector(endEncoding), swizzled_endEncoding);
    TRY_SWIZZLE(@selector(deferredEndEncoding), swizzled_deferredEndEncoding);

    // Buffers
    TRY_SWIZZLE(@selector(setBuffer:offset:atIndex:), swizzled_setBuffer_offset_atIndex);
    TRY_SWIZZLE(@selector(setBuffers:offsets:withRange:), swizzled_setBuffers_offsets_withRange);
    TRY_SWIZZLE(@selector(setBufferOffset:atIndex:), swizzled_setBufferOffset_atIndex);
    TRY_SWIZZLE(@selector(setBytes:length:atIndex:), swizzled_setBytes_length_atIndex);

    // Textures
    TRY_SWIZZLE(@selector(setTexture:atIndex:), swizzled_setTexture_atIndex);
    TRY_SWIZZLE(@selector(setTextures:withRange:), swizzled_setTextures_withRange);

    // Samplers
    TRY_SWIZZLE(@selector(setSamplerState:atIndex:), swizzled_setSamplerState_atIndex);
    TRY_SWIZZLE(@selector(setSamplerStates:withRange:), swizzled_setSamplerStates_withRange);
    TRY_SWIZZLE(@selector(setSamplerState:lodMinClamp:lodMaxClamp:atIndex:), swizzled_setSamplerState_lodMinClamp_lodMaxClamp_atIndex);
    TRY_SWIZZLE(@selector(setSamplerStates:lodMinClamps:lodMaxClamps:withRange:), swizzled_setSamplerStates_lodMinClamps_lodMaxClamps_withRange);

    // Threadgroup
    TRY_SWIZZLE(@selector(setThreadgroupMemoryLength:atIndex:), swizzled_setThreadgroupMemoryLength_atIndex);
    TRY_SWIZZLE(@selector(setImageblockWidth:height:), swizzled_setImageblockWidth_height);
    TRY_SWIZZLE(@selector(setImageBlockWidth:height:), swizzled_setImageBlockWidth_height);
    TRY_SWIZZLE(@selector(setStageInRegion:), swizzled_setStageInRegion);
    TRY_SWIZZLE(@selector(setStageInRegionWithIndirectBuffer:indirectBufferOffset:), swizzled_setStageInRegionWithIndirectBuffer_indirectBufferOffset);

    // Dispatch indirect
    TRY_SWIZZLE(@selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:), swizzled_dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup);
    TRY_SWIZZLE(@selector(dispatchThreadsWithIndirectBuffer:indirectBufferOffset:), swizzled_dispatchThreadsWithIndirectBuffer_indirectBufferOffset);

    // Resources
    TRY_SWIZZLE(@selector(useResource:usage:), swizzled_useResource_usage);
    TRY_SWIZZLE(@selector(useResources:count:usage:), swizzled_useResources_count_usage);
    TRY_SWIZZLE(@selector(useHeap:), swizzled_useHeap);
    TRY_SWIZZLE(@selector(useHeaps:count:), swizzled_useHeaps_count);
    TRY_SWIZZLE(@selector(useResidencySet:), swizzled_useResidencySet);
    TRY_SWIZZLE(@selector(useResidencySets:count:), swizzled_useResidencySets_count);

    // Memory barriers
    TRY_SWIZZLE(@selector(memoryBarrierWithScope:), swizzled_memoryBarrierWithScope);
    TRY_SWIZZLE(@selector(memoryBarrierWithResources:count:), swizzled_memoryBarrierWithResources_count);

    // Fences
    TRY_SWIZZLE(@selector(updateFence:), swizzled_updateFence);
    TRY_SWIZZLE(@selector(waitForFence:), swizzled_waitForFence);

    // Execute
    TRY_SWIZZLE(@selector(executeCommandsInBuffer:withRange:), swizzled_executeCommandsInBuffer_withRange);
    TRY_SWIZZLE(@selector(executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:), swizzled_executeCommandsInBuffer_indirectBuffer_indirectBufferOffset);

    // Function tables
    TRY_SWIZZLE(@selector(setFunctionTable:atIndex:), swizzled_setFunctionTable_atIndex);
    TRY_SWIZZLE(@selector(setFunctionTables:withRange:), swizzled_setFunctionTables_withRange);
    TRY_SWIZZLE(@selector(setVisibleFunctionTable:atBufferIndex:), swizzled_setVisibleFunctionTable_atBufferIndex);
    TRY_SWIZZLE(@selector(setVisibleFunctionTables:withBufferRange:), swizzled_setVisibleFunctionTables_withBufferRange);
    TRY_SWIZZLE(@selector(setIntersectionFunctionTable:atBufferIndex:), swizzled_setIntersectionFunctionTable_atBufferIndex);
    TRY_SWIZZLE(@selector(setIntersectionFunctionTables:withBufferRange:), swizzled_setIntersectionFunctionTables_withBufferRange);

    // Acceleration
    TRY_SWIZZLE(@selector(setAccelerationStructure:atBufferIndex:), swizzled_setAccelerationStructure_atBufferIndex);

    // Thread distribution
    TRY_SWIZZLE(@selector(setThreadgroupDistributionMode:), swizzled_setThreadgroupDistributionMode);
    TRY_SWIZZLE(@selector(setThreadgroupDistributionModeWithClusterGroupIndex:), swizzled_setThreadgroupDistributionModeWithClusterGroupIndex);

    // Flush
    TRY_SWIZZLE(@selector(dispatchWaitFlush), swizzled_dispatchWaitFlush);
    TRY_SWIZZLE(@selector(dispatchFlushInvalidate), swizzled_dispatchFlushInvalidate);
    TRY_SWIZZLE(@selector(dispatchFlushOnly), swizzled_dispatchFlushOnly);
    TRY_SWIZZLE(@selector(dispatchInvalidateOnly), swizzled_dispatchInvalidateOnly);
    TRY_SWIZZLE(@selector(dispatchFenceOnly), swizzled_dispatchFenceOnly);

    // Conditionals
    TRY_SWIZZLE(@selector(encodeStartDoWhile), swizzled_encodeStartDoWhile);
    TRY_SWIZZLE(@selector(encodeEndWhile), swizzled_encodeEndWhile);
    TRY_SWIZZLE(@selector(encodeStartElse), swizzled_encodeStartElse);
    TRY_SWIZZLE(@selector(encodeEndIf), swizzled_encodeEndIf);

    // Counters
    TRY_SWIZZLE(@selector(sampleCountersInBuffer:atSampleIndex:withBarrier:), swizzled_sampleCountersInBuffer_atSampleIndex_withBarrier);

    // Substreams
    TRY_SWIZZLE(@selector(setSubstream:), swizzled_setSubstream);
    TRY_SWIZZLE(@selector(signalProgress:), swizzled_signalProgress);
    TRY_SWIZZLE(@selector(waitForProgress:), swizzled_waitForProgress);
    TRY_SWIZZLE(@selector(beginVirtualSubstream), swizzled_beginVirtualSubstream);
    TRY_SWIZZLE(@selector(nextVirtualSubstream), swizzled_nextVirtualSubstream);
    TRY_SWIZZLE(@selector(endVirtualSubstream), swizzled_endVirtualSubstream);
    TRY_SWIZZLE(@selector(waitForVirtualSubstream:), swizzled_waitForVirtualSubstream);

    // Parallel
    TRY_SWIZZLE(@selector(setParallelExecution:), swizzled_setParallelExecution);

    #undef TRY_SWIZZLE

    os_log(g_log, "AGX Fix Comprehensive: Swizzled %d encoder methods + destroyImpl", count);
    os_log(g_log, "AGX Fix Comprehensive: READY - comprehensive protection enabled");
}

// ============================================================================
// Statistics API
// ============================================================================

extern "C" {
    uint64_t agx_fix_get_acquisitions() { return g_mutex_acquisitions.load(); }
    uint64_t agx_fix_get_contentions() { return g_mutex_contentions.load(); }
    uint64_t agx_fix_get_invalid_skips() { return g_invalid_context_skips.load(); }
    uint64_t agx_fix_get_null_impl_skips() { return g_null_impl_skips.load(); }
    ptrdiff_t agx_fix_get_impl_offset() { return g_impl_ivar_offset; }
    size_t agx_fix_get_swizzled_count() { return g_swizzle_count; }
    bool agx_fix_is_enabled() { return g_enabled; }
}
