/**
 * Metal Interposition Library
 *
 * Created by Andrew Yates
 * Date: 2025-12-20
 *
 * Purpose: Hook Metal API calls to identify where Apple's driver
 * serialization occurs. This helps diagnose the MPS threading bug
 * where efficiency drops from 100% to 3% at 8 threads.
 *
 * Usage:
 *   clang -dynamiclib -o libmetal_interpose.dylib metal_interpose.m \
 *       -framework Foundation -framework Metal -framework MetalPerformanceShaders
 *
 *   DYLD_INSERT_LIBRARIES=./libmetal_interpose.dylib \
 *       METAL_INTERPOSE_LOG=1 \
 *       python3 tests/mps_sync_comparison.py
 *
 * Environment Variables:
 *   METAL_INTERPOSE_LOG=1     - Enable logging (default: disabled)
 *   METAL_INTERPOSE_MUTEX=1   - Add mutex around suspected calls (default: disabled)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <pthread.h>
#import <dlfcn.h>
#import <objc/runtime.h>
#import <stdatomic.h>

// Global state
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static BOOL g_log_enabled = NO;
static BOOL g_mutex_enabled = NO;
static uint64_t g_timebase_num = 0;
static uint64_t g_timebase_denom = 0;

// Statistics
static _Atomic uint64_t g_encoder_count = 0;
static _Atomic uint64_t g_encoder_wait_ns = 0;
static _Atomic uint64_t g_commit_count = 0;
static _Atomic uint64_t g_commit_wait_ns = 0;
static _Atomic uint64_t g_sync_count = 0;
static _Atomic uint64_t g_sync_wait_ns = 0;

// Convert mach time to nanoseconds
static uint64_t mach_to_ns(uint64_t mach_time) {
    return mach_time * g_timebase_num / g_timebase_denom;
}

// Original method implementations (saved during swizzling)
static IMP g_original_computeCommandEncoder = NULL;
static IMP g_original_computeCommandEncoderWithDescriptor = NULL;
static IMP g_original_commit = NULL;
static IMP g_original_waitUntilCompleted = NULL;

// Hooked: computeCommandEncoder
static id hooked_computeCommandEncoder(id self, SEL _cmd) {
    uint64_t start = mach_absolute_time();

    if (g_mutex_enabled) {
        pthread_mutex_lock(&g_mutex);
    }

    uint64_t after_lock = mach_absolute_time();

    // Call original
    id result = ((id (*)(id, SEL))g_original_computeCommandEncoder)(self, _cmd);

    uint64_t end = mach_absolute_time();

    if (g_mutex_enabled) {
        pthread_mutex_unlock(&g_mutex);
    }

    uint64_t lock_wait_ns = mach_to_ns(after_lock - start);
    uint64_t total_ns = mach_to_ns(end - start);

    atomic_fetch_add(&g_encoder_count, 1);
    atomic_fetch_add(&g_encoder_wait_ns, lock_wait_ns);

    if (g_log_enabled && lock_wait_ns > 1000000) {  // > 1ms
        NSLog(@"[MetalInterpose] computeCommandEncoder waited %llu ns (%.2f ms)",
              lock_wait_ns, lock_wait_ns / 1000000.0);
    }

    return result;
}

// Hooked: computeCommandEncoderWithDescriptor:
static id hooked_computeCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    uint64_t start = mach_absolute_time();

    if (g_mutex_enabled) {
        pthread_mutex_lock(&g_mutex);
    }

    uint64_t after_lock = mach_absolute_time();

    // Call original
    id result = ((id (*)(id, SEL, id))g_original_computeCommandEncoderWithDescriptor)(self, _cmd, descriptor);

    uint64_t end = mach_absolute_time();

    if (g_mutex_enabled) {
        pthread_mutex_unlock(&g_mutex);
    }

    uint64_t lock_wait_ns = mach_to_ns(after_lock - start);

    atomic_fetch_add(&g_encoder_count, 1);
    atomic_fetch_add(&g_encoder_wait_ns, lock_wait_ns);

    if (g_log_enabled && lock_wait_ns > 1000000) {
        NSLog(@"[MetalInterpose] computeCommandEncoderWithDescriptor waited %llu ns",
              lock_wait_ns);
    }

    return result;
}

// Hooked: commit
static void hooked_commit(id self, SEL _cmd) {
    uint64_t start = mach_absolute_time();

    if (g_mutex_enabled) {
        pthread_mutex_lock(&g_mutex);
    }

    uint64_t after_lock = mach_absolute_time();

    // Call original
    ((void (*)(id, SEL))g_original_commit)(self, _cmd);

    uint64_t end = mach_absolute_time();

    if (g_mutex_enabled) {
        pthread_mutex_unlock(&g_mutex);
    }

    uint64_t lock_wait_ns = mach_to_ns(after_lock - start);
    uint64_t total_ns = mach_to_ns(end - start);

    atomic_fetch_add(&g_commit_count, 1);
    atomic_fetch_add(&g_commit_wait_ns, total_ns);

    if (g_log_enabled && total_ns > 1000000) {
        NSLog(@"[MetalInterpose] commit took %llu ns (%.2f ms), lock wait: %llu ns",
              total_ns, total_ns / 1000000.0, lock_wait_ns);
    }
}

// Hooked: waitUntilCompleted
static void hooked_waitUntilCompleted(id self, SEL _cmd) {
    uint64_t start = mach_absolute_time();

    // Call original - no mutex here, this is expected to block
    ((void (*)(id, SEL))g_original_waitUntilCompleted)(self, _cmd);

    uint64_t end = mach_absolute_time();
    uint64_t wait_ns = mach_to_ns(end - start);

    atomic_fetch_add(&g_sync_count, 1);
    atomic_fetch_add(&g_sync_wait_ns, wait_ns);

    if (g_log_enabled && wait_ns > 5000000) {  // > 5ms
        NSLog(@"[MetalInterpose] waitUntilCompleted took %llu ns (%.2f ms)",
              wait_ns, wait_ns / 1000000.0);
    }
}

// Print statistics on exit
static void print_stats(void) {
    uint64_t encoder_count = atomic_load(&g_encoder_count);
    uint64_t encoder_wait = atomic_load(&g_encoder_wait_ns);
    uint64_t commit_count = atomic_load(&g_commit_count);
    uint64_t commit_wait = atomic_load(&g_commit_wait_ns);
    uint64_t sync_count = atomic_load(&g_sync_count);
    uint64_t sync_wait = atomic_load(&g_sync_wait_ns);

    NSLog(@"[MetalInterpose] === STATISTICS ===");
    NSLog(@"[MetalInterpose] computeCommandEncoder: %llu calls, avg lock wait: %.2f ms",
          encoder_count, encoder_count > 0 ? (encoder_wait / 1000000.0 / encoder_count) : 0);
    NSLog(@"[MetalInterpose] commit: %llu calls, avg time: %.2f ms",
          commit_count, commit_count > 0 ? (commit_wait / 1000000.0 / commit_count) : 0);
    NSLog(@"[MetalInterpose] waitUntilCompleted: %llu calls, avg time: %.2f ms",
          sync_count, sync_count > 0 ? (sync_wait / 1000000.0 / sync_count) : 0);
    NSLog(@"[MetalInterpose] =================");
}

// Swizzle a method on a class
static BOOL swizzle_method(Class cls, SEL selector, IMP new_imp, IMP *original_imp) {
    Method method = class_getInstanceMethod(cls, selector);
    if (!method) {
        NSLog(@"[MetalInterpose] WARNING: Could not find method %@ on %@",
              NSStringFromSelector(selector), NSStringFromClass(cls));
        return NO;
    }

    *original_imp = method_setImplementation(method, new_imp);
    NSLog(@"[MetalInterpose] Hooked %@.%@",
          NSStringFromClass(cls), NSStringFromSelector(selector));
    return YES;
}

// Library constructor - runs when loaded
__attribute__((constructor))
static void metal_interpose_init(void) {
    // Get timebase for mach_absolute_time conversion
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    g_timebase_num = timebase.numer;
    g_timebase_denom = timebase.denom;

    // Check environment variables
    g_log_enabled = getenv("METAL_INTERPOSE_LOG") != NULL;
    g_mutex_enabled = getenv("METAL_INTERPOSE_MUTEX") != NULL;

    NSLog(@"[MetalInterpose] Initializing... log=%d, mutex=%d",
          g_log_enabled, g_mutex_enabled);

    // Find the concrete implementation class for MTLCommandBuffer
    // This is device-specific (e.g., AGXMTLCommandBuffer, IGPU...)
    // We need to swizzle the concrete class, not the protocol

    // Get a device to find the concrete class
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        NSLog(@"[MetalInterpose] ERROR: No Metal device available");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        NSLog(@"[MetalInterpose] ERROR: Could not create command queue");
        return;
    }

    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    if (!buffer) {
        NSLog(@"[MetalInterpose] ERROR: Could not create command buffer");
        return;
    }

    Class bufferClass = [buffer class];
    NSLog(@"[MetalInterpose] Command buffer class: %@", NSStringFromClass(bufferClass));

    // Swizzle computeCommandEncoder
    swizzle_method(bufferClass,
                   @selector(computeCommandEncoder),
                   (IMP)hooked_computeCommandEncoder,
                   &g_original_computeCommandEncoder);

    // Swizzle computeCommandEncoderWithDescriptor:
    swizzle_method(bufferClass,
                   @selector(computeCommandEncoderWithDescriptor:),
                   (IMP)hooked_computeCommandEncoderWithDescriptor,
                   &g_original_computeCommandEncoderWithDescriptor);

    // Swizzle commit
    swizzle_method(bufferClass,
                   @selector(commit),
                   (IMP)hooked_commit,
                   &g_original_commit);

    // Swizzle waitUntilCompleted
    swizzle_method(bufferClass,
                   @selector(waitUntilCompleted),
                   (IMP)hooked_waitUntilCompleted,
                   &g_original_waitUntilCompleted);

    // Register exit handler
    atexit(print_stats);

    NSLog(@"[MetalInterpose] Initialization complete");
}
