/**
 * AGX Driver Race Condition Fix - OPTIMIZED Version
 *
 * This library intercepts Apple's AGXMetalG16X driver methods and adds
 * per-encoder synchronization to allow parallel encoding operations.
 *
 * OPTIMIZATION: Instead of a global mutex that serializes ALL operations,
 * this version uses per-encoder mutexes. Different command encoders can
 * operate in parallel while operations on the same encoder are serialized.
 *
 * USAGE:
 *   DYLD_INSERT_LIBRARIES=/path/to/libagx_fix_optimized.dylib your_app
 *
 * THEORY:
 *   The AGX driver race condition occurs when Thread A's encoder context is
 *   invalidated by Thread B. With per-encoder mutexes:
 *   - Same encoder: Operations are serialized → safe
 *   - Different encoders: Operations run in parallel → potentially unsafe
 *
 *   This is a HYPOTHESIS TEST. If crashes still occur, the race is in shared
 *   state between encoders, and global mutex is required.
 *
 * Created by Andrew Yates
 * Part of the MPS Parallel Inference research project - Task 0.6
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <mutex>
#import <atomic>
#import <memory>
#import <os/log.h>

// ============================================================================
// Configuration
// ============================================================================

#define AGX_FIX_DISABLE_ENV "AGX_FIX_DISABLE"
#define AGX_FIX_VERBOSE_ENV "AGX_FIX_VERBOSE"

// Key for associated object (per-encoder mutex)
static const char kMutexKey = 0;

// ============================================================================
// Per-Encoder Mutex Wrapper
// ============================================================================

// Objective-C wrapper class to hold a C++ mutex
// We can't store a std::mutex directly as an associated object
@interface AGXEncoderMutex : NSObject {
@public
    std::mutex mutex;
}
@end

@implementation AGXEncoderMutex
@end

// ============================================================================
// Global State
// ============================================================================

namespace {
    // Statistics
    std::atomic<uint64_t> g_mutex_acquisitions{0};
    std::atomic<uint64_t> g_mutex_contentions{0};
    std::atomic<uint64_t> g_mutex_creations{0};  // How many per-encoder mutexes created

    // Logging
    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    // Original method implementations
    IMP g_original_setComputePipelineState = nullptr;
    IMP g_original_dispatchThreads = nullptr;
    IMP g_original_dispatchThreadgroups = nullptr;
    IMP g_original_endEncoding = nullptr;
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
// Per-Encoder Mutex Helper
// ============================================================================

// Get or create a mutex associated with the given encoder
static std::mutex* getEncoderMutex(id encoder) {
    AGXEncoderMutex* wrapper = objc_getAssociatedObject(encoder, &kMutexKey);
    if (!wrapper) {
        wrapper = [[AGXEncoderMutex alloc] init];
        objc_setAssociatedObject(encoder, &kMutexKey, wrapper,
                                 OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        g_mutex_creations++;
        AGX_LOG("AGX Fix: Created per-encoder mutex for %p", encoder);
    }
    return &wrapper->mutex;
}

// ============================================================================
// Per-Encoder Mutex Guard
// ============================================================================

class AGXEncoderMutexGuard {
public:
    explicit AGXEncoderMutexGuard(id encoder) : mutex_(nullptr), locked_(false) {
        if (!g_enabled) return;

        mutex_ = getEncoderMutex(encoder);
        if (!mutex_) return;

        // Try to lock without blocking first
        if (mutex_->try_lock()) {
            locked_ = true;
            g_mutex_acquisitions++;
            AGX_LOG("AGX Fix: Encoder mutex acquired (no contention) for %p", encoder);
        } else {
            // Contention - have to wait
            g_mutex_contentions++;
            mutex_->lock();
            locked_ = true;
            g_mutex_acquisitions++;
            AGX_LOG("AGX Fix: Encoder mutex acquired (after contention) for %p", encoder);
        }
    }

    ~AGXEncoderMutexGuard() {
        if (locked_ && mutex_) {
            mutex_->unlock();
            AGX_LOG("AGX Fix: Encoder mutex released");
        }
    }

    AGXEncoderMutexGuard(const AGXEncoderMutexGuard&) = delete;
    AGXEncoderMutexGuard& operator=(const AGXEncoderMutexGuard&) = delete;

private:
    std::mutex* mutex_;
    bool locked_;
};

// ============================================================================
// Swizzled Method Implementations
// ============================================================================

static void swizzled_setComputePipelineState(id self, SEL _cmd, id pipelineState) {
    AGXEncoderMutexGuard guard(self);
    typedef void (*OriginalFunc)(id, SEL, id);
    ((OriginalFunc)g_original_setComputePipelineState)(self, _cmd, pipelineState);
}

static void swizzled_dispatchThreads(id self, SEL _cmd, MTLSize threads, MTLSize threadsPerGroup) {
    AGXEncoderMutexGuard guard(self);
    typedef void (*OriginalFunc)(id, SEL, MTLSize, MTLSize);
    ((OriginalFunc)g_original_dispatchThreads)(self, _cmd, threads, threadsPerGroup);
}

static void swizzled_dispatchThreadgroups(id self, SEL _cmd, MTLSize groups, MTLSize threadsPerGroup) {
    AGXEncoderMutexGuard guard(self);
    typedef void (*OriginalFunc)(id, SEL, MTLSize, MTLSize);
    ((OriginalFunc)g_original_dispatchThreadgroups)(self, _cmd, groups, threadsPerGroup);
}

static void swizzled_endEncoding(id self, SEL _cmd) {
    AGXEncoderMutexGuard guard(self);
    typedef void (*OriginalFunc)(id, SEL);
    ((OriginalFunc)g_original_endEncoding)(self, _cmd);
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
// Find AGX Compute Context Class
// ============================================================================

static Class find_agx_compute_context_class() {
    const char* classNames[] = {
        "AGXG16XFamilyComputeContext",   // M4 (G16X)
        "AGXG15XFamilyComputeContext",   // M3 (G15X)
        "AGXG14XFamilyComputeContext",   // M2 (G14X)
        "AGXG13XFamilyComputeContext",   // M1 (G13X)
        "AGXMTLComputeCommandEncoder",   // Generic
        nullptr
    };

    for (const char** name = classNames; *name; name++) {
        Class cls = objc_getClass(*name);
        if (cls) {
            AGX_LOG("AGX Fix: Found AGX class: %s", *name);
            return cls;
        }
    }

    return nullptr;
}

// ============================================================================
// Initialization
// ============================================================================

__attribute__((constructor))
static void agx_fix_optimized_init() {
    // Initialize logging
    g_log = os_log_create("com.agxfix.optimized", "main");

    // Check environment variables
    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix Optimized: Disabled via environment variable");
        return;
    }

    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    os_log(g_log, "AGX Fix Optimized: Initializing (per-encoder mutex strategy)");

    // Find AGX compute context class (logged for diagnostics)
    Class agxClass = find_agx_compute_context_class();
    if (agxClass) {
        os_log(g_log, "AGX Fix Optimized: Found driver class: %s", class_getName(agxClass));
    }

    // Get a Metal device to find the actual encoder class
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix Optimized: No Metal device available");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        AGX_LOG_ERROR("AGX Fix Optimized: Failed to create command queue");
        return;
    }

    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    if (!buffer) {
        AGX_LOG_ERROR("AGX Fix Optimized: Failed to create command buffer");
        return;
    }

    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) {
        AGX_LOG_ERROR("AGX Fix Optimized: Failed to create compute encoder");
        return;
    }

    Class encoderClass = [encoder class];
    os_log(g_log, "AGX Fix Optimized: Encoder class is %s", class_getName(encoderClass));

    [encoder endEncoding];

    // Swizzle the methods
    bool success = true;

    success &= swizzle_method(encoderClass,
                              @selector(setComputePipelineState:),
                              (IMP)swizzled_setComputePipelineState,
                              &g_original_setComputePipelineState);

    success &= swizzle_method(encoderClass,
                              @selector(dispatchThreads:threadsPerThreadgroup:),
                              (IMP)swizzled_dispatchThreads,
                              &g_original_dispatchThreads);

    success &= swizzle_method(encoderClass,
                              @selector(dispatchThreadgroups:threadsPerThreadgroup:),
                              (IMP)swizzled_dispatchThreadgroups,
                              &g_original_dispatchThreadgroups);

    success &= swizzle_method(encoderClass,
                              @selector(endEncoding),
                              (IMP)swizzled_endEncoding,
                              &g_original_endEncoding);

    if (success) {
        os_log(g_log, "AGX Fix Optimized: Successfully installed (per-encoder mutex)");
    } else {
        AGX_LOG_ERROR("AGX Fix Optimized: Partial installation (some methods failed)");
    }
}

// ============================================================================
// Statistics API
// ============================================================================

extern "C" {
    uint64_t agx_fix_get_acquisitions() {
        return g_mutex_acquisitions.load();
    }

    uint64_t agx_fix_get_contentions() {
        return g_mutex_contentions.load();
    }

    uint64_t agx_fix_get_mutex_creations() {
        return g_mutex_creations.load();
    }

    void agx_fix_reset_stats() {
        g_mutex_acquisitions = 0;
        g_mutex_contentions = 0;
        // Note: mutex_creations is cumulative, not reset
    }

    bool agx_fix_is_enabled() {
        return g_enabled;
    }
}
