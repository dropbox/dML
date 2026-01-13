/**
 * AGX Driver Race Condition Fix - Version 2.9
 *
 * FIXES 60 FORMAL VERIFICATION GAPS - COMPREHENSIVE ENCODER COVERAGE
 *
 * CRITICAL (GAP 1-10): Commit race, timeout escape, encoder tracking, TOCTOU
 * COMPUTE (GAP 11-33): All 45 compute encoder methods tracked
 * BLIT (GAP 34-51): All 23 blit encoder methods tracked
 * RENDER (GAP 52-60): Core render encoder methods tracked
 *
 * TOTAL: 77+ encoder methods protected across compute, blit, and render encoders.
 *
 * Created by Andrew Yates
 * Part of the MPS Parallel Inference research project
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <cstdlib>
#import <mutex>
#import <atomic>
#import <unordered_map>
#import <unordered_set>
#import <os/log.h>
#import <thread>
#import <chrono>
#import <condition_variable>

// ============================================================================
// Configuration
// ============================================================================

#define AGX_FIX_DISABLE_ENV "AGX_FIX_DISABLE"
#define AGX_FIX_VERBOSE_ENV "AGX_FIX_VERBOSE"
// Gap 9 (roadmap): deadlock/lock inversion diagnostics (opt-in)
#define AGX_FIX_DEADLOCK_DETECT_ENV "AGX_FIX_DEADLOCK_DETECT"
#define AGX_FIX_LOCK_WARN_MS_ENV "AGX_FIX_LOCK_WARN_MS"
#define AGX_FIX_LOCK_LOG_INTERVAL_MS_ENV "AGX_FIX_LOCK_LOG_INTERVAL_MS"
#define AGX_FIX_LOCK_TIMEOUT_MS_ENV "AGX_FIX_LOCK_TIMEOUT_MS"
#define AGX_FIX_LOCK_ABORT_ON_TIMEOUT_ENV "AGX_FIX_LOCK_ABORT_ON_TIMEOUT"

static uint64_t parse_env_u64(const char* name, uint64_t default_value) {
    const char* v = getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    unsigned long long val = strtoull(v, &end, 10);
    if (end == v) return default_value;
    return static_cast<uint64_t>(val);
}

// ============================================================================
// Per-Encoder State
// ============================================================================

namespace {
    struct EncoderState {
        int32_t active_calls = 0;
        bool ended = false;
        int32_t retain_count = 0;
        void* command_buffer = nullptr;

        EncoderState() = default;
    };
}

// ============================================================================
// Global State
// ============================================================================

namespace {
    std::recursive_timed_mutex g_encoder_mutex;
    std::condition_variable_any g_encoder_cv;  // v2.9: For blocking wait
    std::unordered_map<void*, EncoderState> g_encoder_states;
    std::unordered_map<void*, std::unordered_set<void*>> g_command_buffer_encoders;

    // Gap 9 (roadmap): deadlock/lock inversion diagnostics (opt-in)
    bool g_deadlock_detection = false;
    uint64_t g_lock_warn_ms = 1000;
    uint64_t g_lock_log_interval_ms = 5000;
    uint64_t g_lock_timeout_ms = 0;
    bool g_lock_abort_on_timeout = false;
    std::atomic<uint64_t> g_mutex_long_wait_warnings{0};
    std::atomic<uint64_t> g_mutex_lock_timeouts{0};
    std::atomic<uint64_t> g_mutex_max_wait_ms{0};

    // Statistics
    std::atomic<uint64_t> g_mutex_acquisitions{0};
    std::atomic<uint64_t> g_mutex_contentions{0};
    std::atomic<uint64_t> g_encoders_created{0};
    std::atomic<uint64_t> g_encoders_released{0};
    std::atomic<uint64_t> g_null_impl_skips{0};
    std::atomic<uint64_t> g_method_calls{0};
    std::atomic<uint64_t> g_deferred_releases{0};
    std::atomic<uint64_t> g_agx_encoder_creates{0};
    std::atomic<uint64_t> g_mps_encoder_creates{0};
    std::atomic<uint64_t> g_commit_forced_ends{0};
    std::atomic<uint64_t> g_commits_intercepted{0};
    std::atomic<uint64_t> g_signal_events_intercepted{0};
    std::atomic<uint64_t> g_wait_events_intercepted{0};
    std::atomic<uint64_t> g_event_encoder_waits{0};
    std::atomic<uint64_t> g_parallel_encoder_creates{0};  // v2.9: Track parallel encoders

    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    // AGX Command Buffer originals
    IMP g_original_computeCommandEncoder = nullptr;
    IMP g_original_computeCommandEncoderWithDescriptor = nullptr;
    IMP g_original_computeCommandEncoderWithDispatchType = nullptr;
    IMP g_original_blitCommandEncoder = nullptr;
    IMP g_original_blitCommandEncoderWithDescriptor = nullptr;  // v2.9 GAP 5
    IMP g_original_resourceStateCommandEncoderWithDescriptor = nullptr;  // v2.9 GAP 5
    IMP g_original_accelerationStructureCommandEncoderWithDescriptor = nullptr;  // v2.9 GAP 5
    IMP g_original_commit = nullptr;
    IMP g_original_encodeSignalEvent = nullptr;
    IMP g_original_encodeWaitForEvent = nullptr;
    IMP g_original_parallelRenderCommandEncoderWithDescriptor = nullptr;  // v2.9 GAP 3

    // MPS Command Buffer originals
    IMP g_original_mps_computeCommandEncoder = nullptr;
    IMP g_original_mps_computeCommandEncoderWithDescriptor = nullptr;
    IMP g_original_mps_computeCommandEncoderWithDispatchType = nullptr;
    IMP g_original_mps_blitCommandEncoder = nullptr;
    IMP g_original_mps_renderCommandEncoderWithDescriptor = nullptr;
    IMP g_original_mps_resourceStateCommandEncoder = nullptr;
    IMP g_original_mps_accelerationStructureCommandEncoder = nullptr;
    IMP g_original_mps_commit = nullptr;
    IMP g_original_mps_encodeSignalEvent = nullptr;
    IMP g_original_mps_encodeWaitForEvent = nullptr;
    IMP g_original_mps_parallelRenderCommandEncoderWithDescriptor = nullptr;  // v2.9 GAP 3

    // Encoder method originals
    IMP g_original_endEncoding = nullptr;
    IMP g_original_destroyImpl = nullptr;

    // Blit encoder originals
    IMP g_original_blit_endEncoding = nullptr;
    IMP g_original_blit_deferredEndEncoding = nullptr;
    IMP g_original_blit_dealloc = nullptr;

    // Render encoder originals
    IMP g_original_renderCommandEncoderWithDescriptor = nullptr;
    IMP g_original_render_endEncoding = nullptr;
    IMP g_original_render_deferredEndEncoding = nullptr;
    IMP g_original_render_dealloc = nullptr;

    // Parallel render encoder originals (v2.9 GAP 3)
    IMP g_original_parallel_render_endEncoding = nullptr;
    IMP g_original_parallel_render_dealloc = nullptr;
    IMP g_original_parallel_render_sub_encoder = nullptr;  // v2.9 GAP 4: sub-encoder creation

    // Resource state encoder originals
    IMP g_original_resourceStateCommandEncoder = nullptr;
    IMP g_original_resource_state_endEncoding = nullptr;
    IMP g_original_resource_state_dealloc = nullptr;

    // Acceleration structure encoder originals
    IMP g_original_accelerationStructureCommandEncoder = nullptr;
    IMP g_original_accel_struct_endEncoding = nullptr;
    IMP g_original_accel_struct_dealloc = nullptr;

    // Selector mapping
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
        }
    }

    // Class info
    Class g_agx_encoder_class = nullptr;
    Class g_agx_command_buffer_class = nullptr;
    Class g_agx_blit_encoder_class = nullptr;
    Class g_agx_render_encoder_class = nullptr;
    Class g_agx_parallel_render_encoder_class = nullptr;  // v2.9 GAP 3
    Class g_agx_resource_state_encoder_class = nullptr;
    Class g_agx_accel_struct_encoder_class = nullptr;
    Class g_mps_command_buffer_class = nullptr;
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
// Mutex Guard (does NOT auto-release for commit operations)
// ============================================================================

class AGXMutexGuard {
public:
    AGXMutexGuard() : locked_(false) {
        if (!g_enabled) return;
        if (g_encoder_mutex.try_lock()) {
            locked_ = true;
            g_mutex_acquisitions++;
            return;
        }

        g_mutex_contentions++;
        if (!g_deadlock_detection) {
            g_encoder_mutex.lock();
            locked_ = true;
            g_mutex_acquisitions++;
            return;
        }

        const auto start = std::chrono::steady_clock::now();
        uint64_t next_log_ms = g_lock_warn_ms ? g_lock_warn_ms : 1000;
        const uint64_t log_interval_ms = g_lock_log_interval_ms ? g_lock_log_interval_ms : 5000;
        bool timeout_reported = false;

        while (!g_encoder_mutex.try_lock_for(std::chrono::milliseconds(100))) {
            const auto waited = std::chrono::steady_clock::now() - start;
            const uint64_t waited_ms =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(waited).count());

            if (waited_ms >= next_log_ms) {
                g_mutex_long_wait_warnings++;

                uint64_t prev = g_mutex_max_wait_ms.load(std::memory_order_relaxed);
                while (waited_ms > prev &&
                       !g_mutex_max_wait_ms.compare_exchange_weak(
                           prev, waited_ms, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    // retry
                }

                AGX_LOG_ERROR("AGX Fix v2.9: WARNING - waited %llu ms for global mutex "
                              "(contention). Potential lock inversion/deadlock.",
                              static_cast<unsigned long long>(waited_ms));
                next_log_ms = waited_ms + log_interval_ms;
            }

            if (g_lock_timeout_ms > 0 && waited_ms >= g_lock_timeout_ms && !timeout_reported) {
                timeout_reported = true;
                g_mutex_lock_timeouts++;
                AGX_LOG_ERROR("AGX Fix v2.9: ERROR - mutex wait exceeded timeout (%llu ms). "
                              "Potential deadlock.",
                              static_cast<unsigned long long>(g_lock_timeout_ms));
                if (g_lock_abort_on_timeout) {
                    AGX_LOG_ERROR("AGX Fix v2.9: Aborting due to AGX_FIX_LOCK_ABORT_ON_TIMEOUT=1");
                    abort();
                }
            }
        }

        const auto waited = std::chrono::steady_clock::now() - start;
        const uint64_t waited_ms =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(waited).count());

        uint64_t prev = g_mutex_max_wait_ms.load(std::memory_order_relaxed);
        while (waited_ms > prev &&
               !g_mutex_max_wait_ms.compare_exchange_weak(
                   prev, waited_ms, std::memory_order_relaxed, std::memory_order_relaxed)) {
            // retry
        }

        locked_ = true;
        g_mutex_acquisitions++;
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
// v2.7: Get endEncoding IMP for an encoder class
// ============================================================================

static IMP get_end_encoding_imp_for_encoder(id encoder) {
    if (!encoder) return nullptr;
    Class cls = [encoder class];
    if (cls == g_agx_encoder_class) return g_original_endEncoding;
    if (cls == g_agx_blit_encoder_class) return g_original_blit_endEncoding;
    if (cls == g_agx_render_encoder_class) return g_original_render_endEncoding;
    if (cls == g_agx_parallel_render_encoder_class) return g_original_parallel_render_endEncoding;
    if (cls == g_agx_resource_state_encoder_class) return g_original_resource_state_endEncoding;
    if (cls == g_agx_accel_struct_encoder_class) return g_original_accel_struct_endEncoding;
    return nullptr;
}

// ============================================================================
// Forward declarations
// ============================================================================

static bool is_impl_valid(id encoder);

// ============================================================================
// Encoder Lifecycle with Command Buffer Tracking
// ============================================================================

// v2.9 GAP 7 FIX: Verify encoder class has our swizzled endEncoding
static void verify_encoder_class_swizzled(id encoder) {
    Class cls = [encoder class];

    // Check if this class or a parent class is one we swizzled
    bool is_known_class = (cls == g_agx_encoder_class ||
                          cls == g_agx_blit_encoder_class ||
                          cls == g_agx_render_encoder_class ||
                          cls == g_agx_parallel_render_encoder_class ||
                          cls == g_agx_resource_state_encoder_class ||
                          cls == g_agx_accel_struct_encoder_class);

    if (!is_known_class) {
        // Unknown encoder class - might be a subclass that overrides our swizzle
        // Check if it's a subclass of a known class
        Class parent = cls;
        while (parent) {
            if (parent == g_agx_encoder_class ||
                parent == g_agx_blit_encoder_class ||
                parent == g_agx_render_encoder_class ||
                parent == g_agx_parallel_render_encoder_class) {
                // It's a subclass - our swizzle should still work via inheritance
                // But if the subclass overrides endEncoding, we won't intercept
                AGX_LOG("AGX Fix v2.9: Encoder %p is subclass %s (parent swizzled)",
                        (__bridge void*)encoder, class_getName(cls));
                break;
            }
            parent = class_getSuperclass(parent);
        }

        if (!parent) {
            // Completely unknown encoder class!
            AGX_LOG_ERROR("AGX Fix v2.9: UNKNOWN encoder class %s - protection may be incomplete!",
                         class_getName(cls));
        }
    }
}

static void encoder_created_v27(id encoder, id commandBuffer) {
    if (!encoder) return;

    AGXMutexGuard guard;
    void* encoder_ptr = (__bridge void*)encoder;
    void* cb_ptr = commandBuffer ? (__bridge void*)commandBuffer : nullptr;

    auto it = g_encoder_states.find(encoder_ptr);
    if (it != g_encoder_states.end()) {
        AGX_LOG("AGX Fix v2.9: Encoder %p already tracked", encoder_ptr);
        return;
    }

    // v2.9 GAP 7 FIX: Verify encoder class is swizzled
    verify_encoder_class_swizzled(encoder);

    EncoderState state;
    state.command_buffer = cb_ptr;

    try {
        g_encoder_states.emplace(encoder_ptr, state);
        if (cb_ptr) {
            g_command_buffer_encoders[cb_ptr].insert(encoder_ptr);
        }
    } catch (const std::bad_alloc&) {
        AGX_LOG_ERROR("AGX Fix v2.9: OOM creating state for encoder %p", encoder_ptr);
        return;
    }

    CFRetain((__bridge CFTypeRef)encoder);
    g_encoder_states[encoder_ptr].retain_count = 1;
    g_encoders_created++;

    AGX_LOG("AGX Fix v2.9: Created encoder %p for command buffer %p", encoder_ptr, cb_ptr);
}

static bool encoder_method_begin(id encoder) {
    if (!encoder) return false;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);

    if (it == g_encoder_states.end()) {
        // v2.9 GAP 9 + GAP 10 FIX: Don't create state for untracked encoders
        // If encoder is not in our map, it means either:
        // 1. Created before our swizzle (rare with GAP 6 high-priority constructor)
        // 2. Already released/destroyed by encoder_force_release
        // 3. Being destroyed concurrently (TOCTOU race with destroyImpl)
        //
        // For safety, reject ALL untracked encoders. This prevents the race
        // where is_impl_valid returns true just before destroyImpl NULLs _impl.
        // With GAP 6 (priority constructor), case 1 should be very rare.
        AGX_LOG("AGX Fix v2.9: Rejecting method on untracked encoder %p (GAP 10 safety)", ptr);
        return false;
    }

    // v2.9 GAP 9 FIX: Check if encoder was already ended
    if (it->second.ended) {
        AGX_LOG("AGX Fix v2.9: Rejecting method call on ended encoder %p", ptr);
        return false;  // Don't allow methods on ended encoder
    }

    it->second.active_calls++;
    CFRetain((__bridge CFTypeRef)encoder);
    it->second.retain_count++;
    return true;
}

static void encoder_method_end(id encoder) {
    if (!encoder) return;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);
    if (it == g_encoder_states.end()) return;

    int32_t remaining = --it->second.active_calls;
    CFRelease((__bridge CFTypeRef)encoder);
    it->second.retain_count--;

    if (it->second.ended && remaining == 0) {
        if (it->second.command_buffer) {
            auto cb_it = g_command_buffer_encoders.find(it->second.command_buffer);
            if (cb_it != g_command_buffer_encoders.end()) {
                cb_it->second.erase(ptr);
                if (cb_it->second.empty()) {
                    g_command_buffer_encoders.erase(cb_it);
                }
            }
        }
        CFRelease((__bridge CFTypeRef)encoder);
        g_encoder_states.erase(it);
        g_encoders_released++;
        g_deferred_releases++;

        // v2.9 GAP 2: Signal waiters that an encoder ended
        g_encoder_cv.notify_all();
    }
}

static void encoder_ended(id encoder) {
    if (!encoder) return;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);
    if (it == g_encoder_states.end()) return;

    it->second.ended = true;

    if (it->second.active_calls == 0) {
        if (it->second.command_buffer) {
            auto cb_it = g_command_buffer_encoders.find(it->second.command_buffer);
            if (cb_it != g_command_buffer_encoders.end()) {
                cb_it->second.erase(ptr);
                if (cb_it->second.empty()) {
                    g_command_buffer_encoders.erase(cb_it);
                }
            }
        }
        CFRelease((__bridge CFTypeRef)encoder);
        it->second.retain_count--;
        g_encoder_states.erase(it);
        g_encoders_released++;
    }

    // v2.9 GAP 2: Signal waiters that an encoder ended
    g_encoder_cv.notify_all();
}

static void encoder_force_release(id encoder) {
    if (!encoder) return;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);
    if (it == g_encoder_states.end()) return;

    if (it->second.command_buffer) {
        auto cb_it = g_command_buffer_encoders.find(it->second.command_buffer);
        if (cb_it != g_command_buffer_encoders.end()) {
            cb_it->second.erase(ptr);
            if (cb_it->second.empty()) {
                g_command_buffer_encoders.erase(cb_it);
            }
        }
    }

    int32_t count = it->second.retain_count;
    for (int32_t i = 0; i < count; i++) {
        CFRelease((__bridge CFTypeRef)encoder);
    }
    g_encoder_states.erase(it);
    g_encoders_released++;

    // v2.9 GAP 2: Signal waiters
    g_encoder_cv.notify_all();
}

// ============================================================================
// _impl Validity Check
// ============================================================================

static bool is_impl_valid(id encoder) {
    if (g_impl_ivar_offset < 0) return true;
    char* obj_base = (char*)(__bridge void*)encoder;
    void** impl_ptr = (void**)(obj_base + g_impl_ivar_offset);
    if (*impl_ptr == nullptr) {
        g_null_impl_skips++;
        return false;
    }
    return true;
}

// ============================================================================
// RAII Helper
// ============================================================================

class EncoderMethodScope {
public:
    EncoderMethodScope(id encoder) : encoder_(encoder), valid_(false) {
        if (!encoder_ || !g_enabled) return;
        valid_ = encoder_method_begin(encoder_);
    }
    ~EncoderMethodScope() {
        if (valid_) encoder_method_end(encoder_);
    }
    bool is_valid() const { return valid_; }
private:
    id encoder_;
    bool valid_;
};

// ============================================================================
// v2.9 GAP 1 FIX: Force-end active encoders (called WITH mutex held)
// ============================================================================

static void ensure_all_encoders_ended_for_command_buffer_locked(id commandBuffer) {
    // PRECONDITION: Mutex must be held by caller
    if (!commandBuffer) return;

    void* cb_ptr = (__bridge void*)commandBuffer;
    auto cb_it = g_command_buffer_encoders.find(cb_ptr);
    if (cb_it == g_command_buffer_encoders.end()) return;

    // Copy the set since we might modify it
    std::unordered_set<void*> encoders = cb_it->second;

    for (void* encoder_ptr : encoders) {
        auto enc_it = g_encoder_states.find(encoder_ptr);
        if (enc_it == g_encoder_states.end()) continue;
        if (enc_it->second.ended) continue;

        // Encoder not ended - force end it
        id encoder = (__bridge id)encoder_ptr;
        IMP endImp = get_end_encoding_imp_for_encoder(encoder);
        if (endImp) {
            AGX_LOG_ERROR("AGX Fix v2.9: Force-ending encoder %p before commit", encoder_ptr);
            g_commit_forced_ends++;
            typedef void (*EndFunc)(id, SEL);
            ((EndFunc)endImp)(encoder, @selector(endEncoding));
            enc_it->second.ended = true;
        }
    }
}

// ============================================================================
// v2.9 GAP 1 FIX: Commit holds mutex through ENTIRE operation
// ============================================================================

static void swizzled_commit(id self, SEL _cmd) {
    g_commits_intercepted++;

    // v2.9 GAP 1 FIX: Hold mutex through ENTIRE commit operation
    // Previous v2.8 released mutex before calling original commit, allowing
    // a race where a new encoder could be created between force-end and commit.
    AGXMutexGuard guard;
    ensure_all_encoders_ended_for_command_buffer_locked(self);

    // Call original commit WHILE HOLDING MUTEX
    typedef void (*Func)(id, SEL);
    ((Func)g_original_commit)(self, _cmd);
}

static void swizzled_mps_commit(id self, SEL _cmd) {
    g_commits_intercepted++;

    // v2.9 GAP 1 FIX: Hold mutex through ENTIRE commit operation
    AGXMutexGuard guard;
    ensure_all_encoders_ended_for_command_buffer_locked(self);

    typedef void (*Func)(id, SEL);
    ((Func)g_original_mps_commit)(self, _cmd);
}

// ============================================================================
// v2.9 GAP 2 FIX: Block INDEFINITELY until encoders end (no timeout!)
// ============================================================================

static bool has_active_encoders_for_command_buffer_locked(void* cb_ptr) {
    // PRECONDITION: Mutex must be held by caller
    auto cb_it = g_command_buffer_encoders.find(cb_ptr);
    if (cb_it == g_command_buffer_encoders.end() || cb_it->second.empty()) {
        return false;
    }
    for (void* encoder_ptr : cb_it->second) {
        auto enc_it = g_encoder_states.find(encoder_ptr);
        if (enc_it != g_encoder_states.end() && !enc_it->second.ended) {
            return true;
        }
    }
    return false;
}

static void block_until_encoders_end(id commandBuffer) {
    if (!commandBuffer) return;
    void* cb_ptr = (__bridge void*)commandBuffer;

    // v2.9 GAP 2 FIX: Block INDEFINITELY using condition variable
    // Previous v2.8 had 100ms timeout then proceeded anyway (crash!).
    // This version blocks until ALL encoders are ended - no escape hatch.

    std::unique_lock<std::recursive_timed_mutex> lock(g_encoder_mutex);

    while (has_active_encoders_for_command_buffer_locked(cb_ptr)) {
        g_event_encoder_waits++;
        AGX_LOG("AGX Fix v2.9: Blocking for encoders to end on CB %p", cb_ptr);

        // Wait with periodic logging (but NO timeout escape!)
        auto status = g_encoder_cv.wait_for(lock, std::chrono::seconds(5));
        if (status == std::cv_status::timeout) {
            AGX_LOG("AGX Fix v2.9: Still waiting for encoders on CB %p...", cb_ptr);
            // Continue waiting - NO ESCAPE!
        }
    }
}

static void swizzled_encodeSignalEvent(id self, SEL _cmd, id event, uint64_t value) {
    g_signal_events_intercepted++;
    block_until_encoders_end(self);
    typedef void (*Func)(id, SEL, id, uint64_t);
    ((Func)g_original_encodeSignalEvent)(self, _cmd, event, value);
}

static void swizzled_encodeWaitForEvent(id self, SEL _cmd, id event, uint64_t value) {
    g_wait_events_intercepted++;
    block_until_encoders_end(self);
    typedef void (*Func)(id, SEL, id, uint64_t);
    ((Func)g_original_encodeWaitForEvent)(self, _cmd, event, value);
}

static void swizzled_mps_encodeSignalEvent(id self, SEL _cmd, id event, uint64_t value) {
    g_signal_events_intercepted++;
    block_until_encoders_end(self);
    typedef void (*Func)(id, SEL, id, uint64_t);
    ((Func)g_original_mps_encodeSignalEvent)(self, _cmd, event, value);
}

static void swizzled_mps_encodeWaitForEvent(id self, SEL _cmd, id event, uint64_t value) {
    g_wait_events_intercepted++;
    block_until_encoders_end(self);
    typedef void (*Func)(id, SEL, id, uint64_t);
    ((Func)g_original_mps_encodeWaitForEvent)(self, _cmd, event, value);
}

// ============================================================================
// Swizzled AGX Command Buffer Methods
// ============================================================================

static id swizzled_computeCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_computeCommandEncoder)(self, _cmd);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
    }
    return encoder;
}

static id swizzled_computeCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_computeCommandEncoderWithDescriptor)(self, _cmd, descriptor);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
    }
    return encoder;
}

static id swizzled_computeCommandEncoderWithDispatchType(id self, SEL _cmd, NSUInteger dispatchType) {
    typedef id (*Func)(id, SEL, NSUInteger);
    id encoder = ((Func)g_original_computeCommandEncoderWithDispatchType)(self, _cmd, dispatchType);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
    }
    return encoder;
}

static id swizzled_blitCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_blitCommandEncoder)(self, _cmd);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
    }
    return encoder;
}

// ============================================================================
// v2.9 GAP 5 FIX: Descriptor-based encoder creation methods
// ============================================================================

static id swizzled_blitCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_blitCommandEncoderWithDescriptor)(self, _cmd, descriptor);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
        AGX_LOG("AGX Fix v2.9: Created blit encoder with descriptor %p", (__bridge void*)encoder);
    }
    return encoder;
}

static id swizzled_resourceStateCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_resourceStateCommandEncoderWithDescriptor)(self, _cmd, descriptor);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
        AGX_LOG("AGX Fix v2.9: Created resource state encoder with descriptor %p", (__bridge void*)encoder);
    }
    return encoder;
}

static id swizzled_accelerationStructureCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_accelerationStructureCommandEncoderWithDescriptor)(self, _cmd, descriptor);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
        AGX_LOG("AGX Fix v2.9: Created accel struct encoder with descriptor %p", (__bridge void*)encoder);
    }
    return encoder;
}

// ============================================================================
// v2.9 GAP 3 FIX: Swizzled Parallel Render Encoder Creation
// ============================================================================

static id swizzled_parallelRenderCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_parallelRenderCommandEncoderWithDescriptor)(self, _cmd, descriptor);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_parallel_encoder_creates++;
        AGX_LOG("AGX Fix v2.9: Created parallel render encoder %p", (__bridge void*)encoder);
    }
    return encoder;
}

static id swizzled_mps_parallelRenderCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_mps_parallelRenderCommandEncoderWithDescriptor)(self, _cmd, descriptor);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_parallel_encoder_creates++;
        AGX_LOG("AGX Fix v2.9: Created MPS parallel render encoder %p", (__bridge void*)encoder);
    }
    return encoder;
}

// ============================================================================
// Swizzled MPS Command Buffer Methods
// ============================================================================

static id swizzled_mps_computeCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_mps_computeCommandEncoder)(self, _cmd);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_mps_encoder_creates++;
    }
    return encoder;
}

static id swizzled_mps_computeCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_mps_computeCommandEncoderWithDescriptor)(self, _cmd, descriptor);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_mps_encoder_creates++;
    }
    return encoder;
}

static id swizzled_mps_computeCommandEncoderWithDispatchType(id self, SEL _cmd, NSUInteger dispatchType) {
    typedef id (*Func)(id, SEL, NSUInteger);
    id encoder = ((Func)g_original_mps_computeCommandEncoderWithDispatchType)(self, _cmd, dispatchType);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_mps_encoder_creates++;
    }
    return encoder;
}

static id swizzled_mps_blitCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_mps_blitCommandEncoder)(self, _cmd);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_mps_encoder_creates++;
    }
    return encoder;
}

static id swizzled_mps_renderCommandEncoderWithDescriptor(id self, SEL _cmd, id descriptor) {
    typedef id (*Func)(id, SEL, id);
    id encoder = ((Func)g_original_mps_renderCommandEncoderWithDescriptor)(self, _cmd, descriptor);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_mps_encoder_creates++;
    }
    return encoder;
}

static id swizzled_mps_resourceStateCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_mps_resourceStateCommandEncoder)(self, _cmd);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_mps_encoder_creates++;
    }
    return encoder;
}

static id swizzled_mps_accelerationStructureCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_mps_accelerationStructureCommandEncoder)(self, _cmd);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_mps_encoder_creates++;
    }
    return encoder;
}

// ============================================================================
// Swizzled Encoder Methods
// ============================================================================

#define DEFINE_SWIZZLED_METHOD_VOID_0(name) \
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

#define DEFINE_SWIZZLED_METHOD_VOID_1(name, T1) \
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

#define DEFINE_SWIZZLED_METHOD_VOID_2(name, T1, T2) \
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

#define DEFINE_SWIZZLED_METHOD_VOID_3(name, T1, T2, T3) \
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

#define DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE(name) \
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

DEFINE_SWIZZLED_METHOD_VOID_1(setComputePipelineState, id)
DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE(dispatchThreads)
DEFINE_SWIZZLED_METHOD_MTL_SIZE_SIZE(dispatchThreadgroups)

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

DEFINE_SWIZZLED_METHOD_VOID_2(setImageblockWidth, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(setBufferOffset, NSUInteger, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_1(updateFence, id)
DEFINE_SWIZZLED_METHOD_VOID_1(waitForFence, id)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchWaitFlush)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFlushInvalidate)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFlushOnly)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchInvalidateOnly)
DEFINE_SWIZZLED_METHOD_VOID_0(dispatchFenceOnly)

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

// v2.9 GAP 11 FIX: dispatchThreadsWithIndirectBuffer
static void swizzled_dispatchThreadsIndirect(id self, SEL _cmd, id buffer, NSUInteger offset, MTLSize tptg) {
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

// v2.9 GAP 12 FIX: executeCommandsInBuffer with indirect buffer
static void swizzled_executeCommandsInBufferIndirect(id self, SEL _cmd, id icb, id indirectBuffer, NSUInteger offset) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, id, NSUInteger);
        ((Func)original)(self, _cmd, icb, indirectBuffer, offset);
    }
}

// v2.9 GAP 13-17 FIX: Ray tracing methods
DEFINE_SWIZZLED_METHOD_VOID_2(setVisibleFunctionTable, id, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(setVisibleFunctionTables, const id*, NSRange)
DEFINE_SWIZZLED_METHOD_VOID_2(setIntersectionFunctionTable, id, NSUInteger)
DEFINE_SWIZZLED_METHOD_VOID_2(setIntersectionFunctionTables, const id*, NSRange)
DEFINE_SWIZZLED_METHOD_VOID_2(setAccelerationStructure, id, NSUInteger)

// ============================================================================
// v2.9 GAP 22-33 FIX: Additional compute encoder methods
// ============================================================================

// GAP 22: memoryBarrierWithScope:afterStages:beforeStages:
static void swizzled_memoryBarrierWithScopeStages(id self, SEL _cmd, NSUInteger scope, NSUInteger after, NSUInteger before) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope escope(self);
    if (!escope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, scope, after, before);
    }
}

// GAP 23: sampleCountersInBuffer:atSampleIndex:withBarrier:
static void swizzled_sampleCountersInBuffer(id self, SEL _cmd, id buffer, NSUInteger index, BOOL barrier) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope escope(self);
    if (!escope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, BOOL);
        ((Func)original)(self, _cmd, buffer, index, barrier);
    }
}

// GAP 24: setBuffer:offset:attributeStride:atIndex:
static void swizzled_setBufferAttributeStride(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger stride, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope escope(self);
    if (!escope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, buffer, offset, stride, index);
    }
}

// GAP 25: setBuffers:offsets:attributeStrides:withRange:
static void swizzled_setBuffersAttributeStrides(id self, SEL _cmd, const id* buffers, const NSUInteger* offsets, const NSUInteger* strides, NSRange range) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope escope(self);
    if (!escope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const id*, const NSUInteger*, const NSUInteger*, NSRange);
        ((Func)original)(self, _cmd, buffers, offsets, strides, range);
    }
}

// GAP 26: setFunction:atIndex: (for indirect pipelines)
DEFINE_SWIZZLED_METHOD_VOID_2(setFunction, id, NSUInteger)

// GAP 27: setSamplerState:lodMinClamp:lodMaxClamp:atIndex:
static void swizzled_setSamplerStateLOD(id self, SEL _cmd, id sampler, float lodMin, float lodMax, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope escope(self);
    if (!escope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, float, float, NSUInteger);
        ((Func)original)(self, _cmd, sampler, lodMin, lodMax, index);
    }
}

// GAP 28: setSamplerStates:lodMinClamps:lodMaxClamps:withRange:
static void swizzled_setSamplerStatesLOD(id self, SEL _cmd, const id* samplers, const float* lodMins, const float* lodMaxs, NSRange range) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope escope(self);
    if (!escope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const id*, const float*, const float*, NSRange);
        ((Func)original)(self, _cmd, samplers, lodMins, lodMaxs, range);
    }
}

// GAP 29: setStageInRegionWithIndirectBuffer:indirectBufferOffset:
DEFINE_SWIZZLED_METHOD_VOID_2(setStageInRegionWithIndirectBuffer, id, NSUInteger)

// GAP 30: useHeap:stages:
DEFINE_SWIZZLED_METHOD_VOID_2(useHeapStages, id, NSUInteger)

// GAP 31: useHeaps:count:stages:
DEFINE_SWIZZLED_METHOD_VOID_3(useHeapsStages, const id*, NSUInteger, NSUInteger)

// GAP 32: useResource:usage:stages:
DEFINE_SWIZZLED_METHOD_VOID_3(useResourceStages, id, NSUInteger, NSUInteger)

// GAP 33: useResources:count:usage:stages:
static void swizzled_useResourcesStages(id self, SEL _cmd, const id* resources, NSUInteger count, NSUInteger usage, NSUInteger stages) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope escope(self);
    if (!escope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, const id*, NSUInteger, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, resources, count, usage, stages);
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

// v2.9 GAP 34-51: Additional blit encoder methods
static void swizzled_blit_copyFromTextureToBuffer(id self, SEL _cmd, id srcTex, NSUInteger srcSlice, NSUInteger srcLevel,
    MTLOrigin srcOrigin, MTLSize srcSize, id dstBuffer, NSUInteger dstOffset, NSUInteger dstBytesPerRow, NSUInteger dstBytesPerImage) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger, MTLOrigin, MTLSize, id, NSUInteger, NSUInteger, NSUInteger);
        ((Func)original)(self, _cmd, srcTex, srcSlice, srcLevel, srcOrigin, srcSize, dstBuffer, dstOffset, dstBytesPerRow, dstBytesPerImage);
    }
}

static void swizzled_blit_copyFromBufferToTexture(id self, SEL _cmd, id srcBuffer, NSUInteger srcOffset, NSUInteger srcBytesPerRow,
    NSUInteger srcBytesPerImage, MTLSize srcSize, id dstTex, NSUInteger dstSlice, NSUInteger dstLevel, MTLOrigin dstOrigin) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger, MTLSize, id, NSUInteger, NSUInteger, MTLOrigin);
        ((Func)original)(self, _cmd, srcBuffer, srcOffset, srcBytesPerRow, srcBytesPerImage, srcSize, dstTex, dstSlice, dstLevel, dstOrigin);
    }
}

static void swizzled_blit_copyFromTextureToTexture(id self, SEL _cmd, id srcTex, NSUInteger srcSlice, NSUInteger srcLevel,
    MTLOrigin srcOrigin, MTLSize srcSize, id dstTex, NSUInteger dstSlice, NSUInteger dstLevel, MTLOrigin dstOrigin) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger, MTLOrigin, MTLSize, id, NSUInteger, NSUInteger, MTLOrigin);
        ((Func)original)(self, _cmd, srcTex, srcSlice, srcLevel, srcOrigin, srcSize, dstTex, dstSlice, dstLevel, dstOrigin);
    }
}

static void swizzled_blit_generateMipmaps(id self, SEL _cmd, id texture) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id); ((Func)original)(self, _cmd, texture); }
}

static void swizzled_blit_synchronizeTexture(id self, SEL _cmd, id texture, NSUInteger slice, NSUInteger level) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, texture, slice, level); }
}

static void swizzled_blit_optimizeForGPU(id self, SEL _cmd, id texture) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id); ((Func)original)(self, _cmd, texture); }
}

static void swizzled_blit_optimizeForGPUSlice(id self, SEL _cmd, id texture, NSUInteger slice, NSUInteger level) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, texture, slice, level); }
}

static void swizzled_blit_optimizeForCPU(id self, SEL _cmd, id texture) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id); ((Func)original)(self, _cmd, texture); }
}

static void swizzled_blit_optimizeForCPUSlice(id self, SEL _cmd, id texture, NSUInteger slice, NSUInteger level) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, texture, slice, level); }
}

static void swizzled_blit_resetCommandsInBuffer(id self, SEL _cmd, id icb, NSRange range) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSRange); ((Func)original)(self, _cmd, icb, range); }
}

static void swizzled_blit_copyIndirectCommandBuffer(id self, SEL _cmd, id srcICB, NSRange srcRange, id dstICB, NSUInteger dstIndex) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSRange, id, NSUInteger); ((Func)original)(self, _cmd, srcICB, srcRange, dstICB, dstIndex); }
}

static void swizzled_blit_optimizeIndirectCommandBuffer(id self, SEL _cmd, id icb, NSRange range) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSRange); ((Func)original)(self, _cmd, icb, range); }
}

static void swizzled_blit_resolveCounters(id self, SEL _cmd, id counterBuffer, NSRange range, id dstBuffer, NSUInteger dstOffset) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSRange, id, NSUInteger); ((Func)original)(self, _cmd, counterBuffer, range, dstBuffer, dstOffset); }
}

static void swizzled_blit_sampleCounters(id self, SEL _cmd, id buffer, NSUInteger index, BOOL barrier) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSUInteger, BOOL); ((Func)original)(self, _cmd, buffer, index, barrier); }
}

static void swizzled_blit_updateFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id); ((Func)original)(self, _cmd, fence); }
}

static void swizzled_blit_waitForFence(id self, SEL _cmd, id fence) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id); ((Func)original)(self, _cmd, fence); }
}

static void swizzled_blit_getTextureAccessCounters(id self, SEL _cmd, id texture, MTLRegion region, NSUInteger mipLevel,
    NSUInteger slice, BOOL reset, id countersBuffer, NSUInteger countersOffset) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) {
        typedef void (*Func)(id, SEL, id, MTLRegion, NSUInteger, NSUInteger, BOOL, id, NSUInteger);
        ((Func)original)(self, _cmd, texture, region, mipLevel, slice, reset, countersBuffer, countersOffset);
    }
}

static void swizzled_blit_resetTextureAccessCounters(id self, SEL _cmd, id texture, MTLRegion region, NSUInteger mipLevel, NSUInteger slice) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, MTLRegion, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, texture, region, mipLevel, slice); }
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
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
    }
    return encoder;
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
// v2.9 GAP 52-70: Render encoder work methods
// These are NOT used by PyTorch MPS but included for completeness
// ============================================================================

// Vertex buffer methods
static void swizzled_render_setVertexBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, buffer, offset, index); }
}

static void swizzled_render_setVertexBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, bytes, length, index); }
}

// Fragment buffer methods
static void swizzled_render_setFragmentBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, buffer, offset, index); }
}

static void swizzled_render_setFragmentBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, const void*, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, bytes, length, index); }
}

// Texture methods
static void swizzled_render_setVertexTexture(id self, SEL _cmd, id texture, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSUInteger); ((Func)original)(self, _cmd, texture, index); }
}

static void swizzled_render_setFragmentTexture(id self, SEL _cmd, id texture, NSUInteger index) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id, NSUInteger); ((Func)original)(self, _cmd, texture, index); }
}

// Pipeline state
static void swizzled_render_setRenderPipelineState(id self, SEL _cmd, id pipelineState) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, id); ((Func)original)(self, _cmd, pipelineState); }
}

// Draw primitives
static void swizzled_render_drawPrimitives(id self, SEL _cmd, NSUInteger primitiveType, NSUInteger vertexStart, NSUInteger vertexCount) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, primitiveType, vertexStart, vertexCount); }
}

static void swizzled_render_drawPrimitivesInstanced(id self, SEL _cmd, NSUInteger primitiveType, NSUInteger vertexStart, NSUInteger vertexCount, NSUInteger instanceCount) {
    AGXMutexGuard guard;
    g_method_calls++;
    EncoderMethodScope scope(self);
    if (!scope.is_valid() || !is_impl_valid(self)) return;
    IMP original = get_original_imp(_cmd);
    if (original) { typedef void (*Func)(id, SEL, NSUInteger, NSUInteger, NSUInteger, NSUInteger); ((Func)original)(self, _cmd, primitiveType, vertexStart, vertexCount, instanceCount); }
}

// ============================================================================
// v2.9 GAP 3: Swizzled PARALLEL RENDER Encoder Methods
// ============================================================================

static void swizzled_parallel_render_endEncoding(id self, SEL _cmd) {
    AGXMutexGuard guard;
    g_method_calls++;
    if (g_original_parallel_render_endEncoding) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_parallel_render_endEncoding)(self, _cmd);
    }
    encoder_ended(self);
}

static void swizzled_parallel_render_dealloc(id self, SEL _cmd) {
    {
        AGXMutexGuard guard;
        encoder_force_release(self);
    }
    if (g_original_parallel_render_dealloc) {
        typedef void (*Func)(id, SEL);
        ((Func)g_original_parallel_render_dealloc)(self, _cmd);
    }
}

// v2.9 GAP 4 FIX: Track sub-encoders created by parallelRenderCommandEncoder
// The parallelRenderCommandEncoder has a renderCommandEncoder method that creates sub-encoders.
// These sub-encoders must be tracked with the same command buffer as the parent parallel encoder.
static id swizzled_parallel_render_sub_encoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id subEncoder = ((Func)g_original_parallel_render_sub_encoder)(self, _cmd);
    if (subEncoder) {
        // Get the command buffer from the parallel encoder's tracked state
        AGXMutexGuard guard;
        void* parallel_ptr = (__bridge void*)self;
        auto it = g_encoder_states.find(parallel_ptr);
        id commandBuffer = nil;
        if (it != g_encoder_states.end() && it->second.command_buffer) {
            commandBuffer = (__bridge id)it->second.command_buffer;
        }
        // Track the sub-encoder with the same command buffer
        if (commandBuffer) {
            encoder_created_v27(subEncoder, commandBuffer);
            AGX_LOG("AGX Fix v2.9: Created sub-encoder %p from parallel encoder %p (CB %p)",
                    (__bridge void*)subEncoder, parallel_ptr, (__bridge void*)commandBuffer);
        } else {
            // Fallback: track without command buffer association
            encoder_created_v27(subEncoder, nil);
            AGX_LOG("AGX Fix v2.9: Created sub-encoder %p from parallel encoder %p (no CB found)",
                    (__bridge void*)subEncoder, parallel_ptr);
        }
        g_agx_encoder_creates++;
    }
    return subEncoder;
}

// ============================================================================
// Swizzled RESOURCE STATE Encoder Methods
// ============================================================================

static id swizzled_resourceStateCommandEncoder(id self, SEL _cmd) {
    typedef id (*Func)(id, SEL);
    id encoder = ((Func)g_original_resourceStateCommandEncoder)(self, _cmd);
    if (encoder) {
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
    }
    return encoder;
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
        encoder_created_v27(encoder, self);
        g_agx_encoder_creates++;
    }
    return encoder;
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
    if (!method) return false;
    *outOriginal = method_getImplementation(method);
    store_original_imp(selector, *outOriginal);
    method_setImplementation(method, newImpl);
    return true;
}

// ============================================================================
// Initialization
// ============================================================================

// v2.9 GAP 6 FIX: Use high-priority constructor to run before other dylibs
// Priority 101 runs very early (default is 65535, lower = earlier)
// This ensures we swizzle Metal methods before any other code can use Metal
__attribute__((constructor(101)))
static void agx_fix_v2_9_init() {
    g_log = os_log_create("com.agxfix.v2.9", "main");

    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix v2.9: Disabled via environment");
        return;
    }
    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    g_deadlock_detection = getenv(AGX_FIX_DEADLOCK_DETECT_ENV) != nullptr;
    if (g_deadlock_detection) {
        g_lock_warn_ms = parse_env_u64(AGX_FIX_LOCK_WARN_MS_ENV, 1000);
        g_lock_log_interval_ms = parse_env_u64(AGX_FIX_LOCK_LOG_INTERVAL_MS_ENV, 5000);
        g_lock_timeout_ms = parse_env_u64(AGX_FIX_LOCK_TIMEOUT_MS_ENV, 0);
        g_lock_abort_on_timeout = getenv(AGX_FIX_LOCK_ABORT_ON_TIMEOUT_ENV) != nullptr;

        os_log(g_log,
               "AGX Fix v2.9: Deadlock detection enabled (warn=%llu ms, interval=%llu ms, timeout=%llu ms, abort=%d)",
               static_cast<unsigned long long>(g_lock_warn_ms),
               static_cast<unsigned long long>(g_lock_log_interval_ms),
               static_cast<unsigned long long>(g_lock_timeout_ms),
               g_lock_abort_on_timeout ? 1 : 0);
    }

    os_log(g_log, "AGX Fix v2.9: Initializing (with formal verification gap fixes)");

    // GAP 4 FIX: Log macOS version for compatibility tracking
    NSProcessInfo *processInfo = [NSProcessInfo processInfo];
    NSOperatingSystemVersion osVersion = [processInfo operatingSystemVersion];
    os_log(g_log, "AGX Fix v2.9: macOS version: %ld.%ld.%ld",
           (long)osVersion.majorVersion, (long)osVersion.minorVersion, (long)osVersion.patchVersion);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix v2.9: No Metal device");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (!encoder || !commandBuffer) {
        AGX_LOG_ERROR("AGX Fix v2.9: Failed to create test objects");
        return;
    }

    g_agx_encoder_class = [encoder class];
    g_agx_command_buffer_class = [commandBuffer class];

    // GAP 4 FIX: Clear error messages if critical classes not found
    if (!g_agx_encoder_class) {
        AGX_LOG_ERROR("AGX Fix v2.9: FATAL - Cannot find AGX encoder class. "
                      "AGX fix will NOT protect against crashes. "
                      "macOS version %ld.%ld.%ld may be incompatible.",
                      (long)osVersion.majorVersion, (long)osVersion.minorVersion, (long)osVersion.patchVersion);
        g_enabled = false;
        return;
    }
    if (!g_agx_command_buffer_class) {
        AGX_LOG_ERROR("AGX Fix v2.9: FATAL - Cannot find AGX command buffer class. "
                      "macOS version %ld.%ld.%ld may be incompatible.",
                      (long)osVersion.majorVersion, (long)osVersion.minorVersion, (long)osVersion.patchVersion);
        g_enabled = false;
        return;
    }

    os_log(g_log, "AGX Fix v2.9: AGX Encoder class: %s", class_getName(g_agx_encoder_class));
    os_log(g_log, "AGX Fix v2.9: AGX Command buffer class: %s", class_getName(g_agx_command_buffer_class));
    [encoder endEncoding];

    // Discover MPSCommandBuffer
    @try {
        MPSCommandBuffer* mpsCommandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        if (mpsCommandBuffer) {
            g_mps_command_buffer_class = [mpsCommandBuffer class];
            os_log(g_log, "AGX Fix v2.9: MPS Command buffer class: %s", class_getName(g_mps_command_buffer_class));
        }
    } @catch (NSException* e) {
        os_log(g_log, "AGX Fix v2.9: Exception creating MPSCommandBuffer");
    }

    // Discover blit encoder
    id<MTLCommandBuffer> cb2 = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blitEncoder = [cb2 blitCommandEncoder];
    if (blitEncoder) {
        g_agx_blit_encoder_class = [blitEncoder class];
        [blitEncoder endEncoding];
    }

    // Discover render encoder
    MTLRenderPassDescriptor* renderPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:64 height:64 mipmapped:NO];
    texDesc.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> dummyTexture = [device newTextureWithDescriptor:texDesc];
    if (dummyTexture) {
        renderPassDesc.colorAttachments[0].texture = dummyTexture;
        renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionDontCare;
        id<MTLCommandBuffer> cb3 = [queue commandBuffer];
        id<MTLRenderCommandEncoder> renderEncoder = [cb3 renderCommandEncoderWithDescriptor:renderPassDesc];
        if (renderEncoder) {
            g_agx_render_encoder_class = [renderEncoder class];
            [renderEncoder endEncoding];
        }

        // v2.9 GAP 3: Discover parallel render encoder
        id<MTLCommandBuffer> cb3b = [queue commandBuffer];
        id<MTLParallelRenderCommandEncoder> parallelEncoder = [cb3b parallelRenderCommandEncoderWithDescriptor:renderPassDesc];
        if (parallelEncoder) {
            g_agx_parallel_render_encoder_class = [parallelEncoder class];
            os_log(g_log, "AGX Fix v2.9: Parallel render encoder class: %s", class_getName(g_agx_parallel_render_encoder_class));
            [parallelEncoder endEncoding];
        }
    }

    // Discover resource state encoder
    id<MTLCommandBuffer> cb4 = [queue commandBuffer];
    if ([cb4 respondsToSelector:@selector(resourceStateCommandEncoder)]) {
        id resourceStateEncoder = [cb4 resourceStateCommandEncoder];
        if (resourceStateEncoder) {
            g_agx_resource_state_encoder_class = [resourceStateEncoder class];
            [resourceStateEncoder endEncoding];
        }
    }

    // Discover acceleration structure encoder
    id<MTLCommandBuffer> cb5 = [queue commandBuffer];
    if ([cb5 respondsToSelector:@selector(accelerationStructureCommandEncoder)]) {
        id accelStructEncoder = [cb5 accelerationStructureCommandEncoder];
        if (accelStructEncoder) {
            g_agx_accel_struct_encoder_class = [accelStructEncoder class];
            [accelStructEncoder endEncoding];
        }
    }

    // Find _impl offset
    Ivar implIvar = class_getInstanceVariable(g_agx_encoder_class, "_impl");
    if (implIvar) {
        g_impl_ivar_offset = ivar_getOffset(implIvar);
    } else {
        Class parent = class_getSuperclass(g_agx_encoder_class);
        while (parent) {
            implIvar = class_getInstanceVariable(parent, "_impl");
            if (implIvar) {
                g_impl_ivar_offset = ivar_getOffset(implIvar);
                break;
            }
            parent = class_getSuperclass(parent);
        }
    }

    int swizzled_count = 0;
    IMP dummy;

    // v2.9 GAP 1: CRITICAL - Swizzle commit (holds mutex through entire operation)
    if (swizzle_method(g_agx_command_buffer_class, @selector(commit),
                       (IMP)swizzled_commit, &g_original_commit)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.9: Swizzled AGX commit (GAP 1 FIX: mutex held through commit)");
    }

    // v2.9 GAP 2: Swizzle event methods (block indefinitely, no timeout)
    if (swizzle_method(g_agx_command_buffer_class, @selector(encodeSignalEvent:value:),
                       (IMP)swizzled_encodeSignalEvent, &g_original_encodeSignalEvent)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.9: Swizzled AGX encodeSignalEvent (GAP 2 FIX: no timeout)");
    }

    if (swizzle_method(g_agx_command_buffer_class, @selector(encodeWaitForEvent:value:),
                       (IMP)swizzled_encodeWaitForEvent, &g_original_encodeWaitForEvent)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.9: Swizzled AGX encodeWaitForEvent");
    }

    // v2.9 GAP 3: Swizzle parallel render encoder creation
    if (swizzle_method(g_agx_command_buffer_class, @selector(parallelRenderCommandEncoderWithDescriptor:),
                       (IMP)swizzled_parallelRenderCommandEncoderWithDescriptor, &g_original_parallelRenderCommandEncoderWithDescriptor)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.9: Swizzled AGX parallelRenderCommandEncoder (GAP 3 FIX)");
    }

    // AGX command buffer encoder creation
    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoder),
                       (IMP)swizzled_computeCommandEncoder, &g_original_computeCommandEncoder)) swizzled_count++;
    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoderWithDescriptor:),
                       (IMP)swizzled_computeCommandEncoderWithDescriptor, &g_original_computeCommandEncoderWithDescriptor)) swizzled_count++;
    if (swizzle_method(g_agx_command_buffer_class, @selector(computeCommandEncoderWithDispatchType:),
                       (IMP)swizzled_computeCommandEncoderWithDispatchType, &g_original_computeCommandEncoderWithDispatchType)) swizzled_count++;
    if (swizzle_method(g_agx_command_buffer_class, @selector(blitCommandEncoder),
                       (IMP)swizzled_blitCommandEncoder, &g_original_blitCommandEncoder)) swizzled_count++;

    // v2.9 GAP 5 FIX: Descriptor-based encoder creation methods
    if (swizzle_method(g_agx_command_buffer_class, @selector(blitCommandEncoderWithDescriptor:),
                       (IMP)swizzled_blitCommandEncoderWithDescriptor, &g_original_blitCommandEncoderWithDescriptor)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.9: Swizzled blitCommandEncoderWithDescriptor (GAP 5 FIX)");
    }
    if (swizzle_method(g_agx_command_buffer_class, @selector(resourceStateCommandEncoderWithDescriptor:),
                       (IMP)swizzled_resourceStateCommandEncoderWithDescriptor, &g_original_resourceStateCommandEncoderWithDescriptor)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.9: Swizzled resourceStateCommandEncoderWithDescriptor (GAP 5 FIX)");
    }
    if (swizzle_method(g_agx_command_buffer_class, @selector(accelerationStructureCommandEncoderWithDescriptor:),
                       (IMP)swizzled_accelerationStructureCommandEncoderWithDescriptor, &g_original_accelerationStructureCommandEncoderWithDescriptor)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.9: Swizzled accelerationStructureCommandEncoderWithDescriptor (GAP 5 FIX)");
    }

    // MPS command buffer swizzling
    if (g_mps_command_buffer_class && g_mps_command_buffer_class != g_agx_command_buffer_class) {
        if (swizzle_method(g_mps_command_buffer_class, @selector(commit),
                           (IMP)swizzled_mps_commit, &g_original_mps_commit)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.9: Swizzled MPS commit");
        }
        if (swizzle_method(g_mps_command_buffer_class, @selector(encodeSignalEvent:value:),
                           (IMP)swizzled_mps_encodeSignalEvent, &g_original_mps_encodeSignalEvent)) {
            swizzled_count++;
        }
        if (swizzle_method(g_mps_command_buffer_class, @selector(encodeWaitForEvent:value:),
                           (IMP)swizzled_mps_encodeWaitForEvent, &g_original_mps_encodeWaitForEvent)) {
            swizzled_count++;
        }
        // v2.9 GAP 3: MPS parallel render encoder
        if (swizzle_method(g_mps_command_buffer_class, @selector(parallelRenderCommandEncoderWithDescriptor:),
                           (IMP)swizzled_mps_parallelRenderCommandEncoderWithDescriptor, &g_original_mps_parallelRenderCommandEncoderWithDescriptor)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.9: Swizzled MPS parallelRenderCommandEncoder");
        }
        if (swizzle_method(g_mps_command_buffer_class, @selector(computeCommandEncoder),
                           (IMP)swizzled_mps_computeCommandEncoder, &g_original_mps_computeCommandEncoder)) swizzled_count++;
        if (swizzle_method(g_mps_command_buffer_class, @selector(computeCommandEncoderWithDescriptor:),
                           (IMP)swizzled_mps_computeCommandEncoderWithDescriptor, &g_original_mps_computeCommandEncoderWithDescriptor)) swizzled_count++;
        if (swizzle_method(g_mps_command_buffer_class, @selector(computeCommandEncoderWithDispatchType:),
                           (IMP)swizzled_mps_computeCommandEncoderWithDispatchType, &g_original_mps_computeCommandEncoderWithDispatchType)) swizzled_count++;
        if (swizzle_method(g_mps_command_buffer_class, @selector(blitCommandEncoder),
                           (IMP)swizzled_mps_blitCommandEncoder, &g_original_mps_blitCommandEncoder)) swizzled_count++;
        if (swizzle_method(g_mps_command_buffer_class, @selector(renderCommandEncoderWithDescriptor:),
                           (IMP)swizzled_mps_renderCommandEncoderWithDescriptor, &g_original_mps_renderCommandEncoderWithDescriptor)) swizzled_count++;
        if (swizzle_method(g_mps_command_buffer_class, @selector(resourceStateCommandEncoder),
                           (IMP)swizzled_mps_resourceStateCommandEncoder, &g_original_mps_resourceStateCommandEncoder)) swizzled_count++;
        if (swizzle_method(g_mps_command_buffer_class, @selector(accelerationStructureCommandEncoder),
                           (IMP)swizzled_mps_accelerationStructureCommandEncoder, &g_original_mps_accelerationStructureCommandEncoder)) swizzled_count++;
    }

    // Encoder methods
    if (swizzle_method(g_agx_encoder_class, @selector(destroyImpl),
                       (IMP)swizzled_destroyImpl, &g_original_destroyImpl)) swizzled_count++;
    if (swizzle_method(g_agx_encoder_class, @selector(endEncoding),
                       (IMP)swizzled_endEncoding, &g_original_endEncoding)) swizzled_count++;

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

    // v2.9 GAP 11 FIX: Missing indirect dispatch method
    SWIZZLE(@selector(dispatchThreadsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:), swizzled_dispatchThreadsIndirect)

    // v2.9 GAP 12 FIX: executeCommandsInBuffer with indirect buffer
    SWIZZLE(@selector(executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:), swizzled_executeCommandsInBufferIndirect)

    // v2.9 GAP 13-17 FIX: Ray tracing methods (macOS 11+)
    SWIZZLE(@selector(setVisibleFunctionTable:atBufferIndex:), swizzled_setVisibleFunctionTable)
    SWIZZLE(@selector(setVisibleFunctionTables:withBufferRange:), swizzled_setVisibleFunctionTables)
    SWIZZLE(@selector(setIntersectionFunctionTable:atBufferIndex:), swizzled_setIntersectionFunctionTable)
    SWIZZLE(@selector(setIntersectionFunctionTables:withBufferRange:), swizzled_setIntersectionFunctionTables)
    SWIZZLE(@selector(setAccelerationStructure:atBufferIndex:), swizzled_setAccelerationStructure)

    // v2.9 GAP 22-33 FIX: Additional compute encoder methods
    SWIZZLE(@selector(memoryBarrierWithScope:afterStages:beforeStages:), swizzled_memoryBarrierWithScopeStages)
    SWIZZLE(@selector(sampleCountersInBuffer:atSampleIndex:withBarrier:), swizzled_sampleCountersInBuffer)
    SWIZZLE(@selector(setBuffer:offset:attributeStride:atIndex:), swizzled_setBufferAttributeStride)
    SWIZZLE(@selector(setBuffers:offsets:attributeStrides:withRange:), swizzled_setBuffersAttributeStrides)
    SWIZZLE(@selector(setFunction:atIndex:), swizzled_setFunction)
    SWIZZLE(@selector(setSamplerState:lodMinClamp:lodMaxClamp:atIndex:), swizzled_setSamplerStateLOD)
    SWIZZLE(@selector(setSamplerStates:lodMinClamps:lodMaxClamps:withRange:), swizzled_setSamplerStatesLOD)
    SWIZZLE(@selector(setStageInRegionWithIndirectBuffer:indirectBufferOffset:), swizzled_setStageInRegionWithIndirectBuffer)
    SWIZZLE(@selector(useHeap:stages:), swizzled_useHeapStages)
    SWIZZLE(@selector(useHeaps:count:stages:), swizzled_useHeapsStages)
    SWIZZLE(@selector(useResource:usage:stages:), swizzled_useResourceStages)
    SWIZZLE(@selector(useResources:count:usage:stages:), swizzled_useResourcesStages)

    #undef SWIZZLE

    // Blit encoder
    if (g_agx_blit_encoder_class) {
        IMP blit_dummy;
        if (swizzle_method(g_agx_blit_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_blit_endEncoding, &g_original_blit_endEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_blit_encoder_class, @selector(fillBuffer:range:value:),
                           (IMP)swizzled_blit_fillBuffer, &blit_dummy)) swizzled_count++;
        if (swizzle_method(g_agx_blit_encoder_class, @selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:),
                           (IMP)swizzled_blit_copyFromBuffer, &blit_dummy)) swizzled_count++;
        if (swizzle_method(g_agx_blit_encoder_class, @selector(synchronizeResource:),
                           (IMP)swizzled_blit_synchronizeResource, &blit_dummy)) swizzled_count++;
        if (swizzle_method(g_agx_blit_encoder_class, @selector(deferredEndEncoding),
                           (IMP)swizzled_blit_deferredEndEncoding, &g_original_blit_deferredEndEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_blit_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_blit_dealloc, &g_original_blit_dealloc)) swizzled_count++;

        // v2.9 GAP 34-51 FIX: Additional blit encoder methods
        #define BLIT_SWIZZLE(sel, func) \
            if (swizzle_method(g_agx_blit_encoder_class, sel, (IMP)func, &blit_dummy)) swizzled_count++;
        BLIT_SWIZZLE(@selector(copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:), swizzled_blit_copyFromTextureToBuffer)
        BLIT_SWIZZLE(@selector(copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:), swizzled_blit_copyFromBufferToTexture)
        BLIT_SWIZZLE(@selector(copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:), swizzled_blit_copyFromTextureToTexture)
        BLIT_SWIZZLE(@selector(generateMipmapsForTexture:), swizzled_blit_generateMipmaps)
        BLIT_SWIZZLE(@selector(synchronizeTexture:slice:level:), swizzled_blit_synchronizeTexture)
        BLIT_SWIZZLE(@selector(optimizeContentsForGPUAccess:), swizzled_blit_optimizeForGPU)
        BLIT_SWIZZLE(@selector(optimizeContentsForGPUAccess:slice:level:), swizzled_blit_optimizeForGPUSlice)
        BLIT_SWIZZLE(@selector(optimizeContentsForCPUAccess:), swizzled_blit_optimizeForCPU)
        BLIT_SWIZZLE(@selector(optimizeContentsForCPUAccess:slice:level:), swizzled_blit_optimizeForCPUSlice)
        BLIT_SWIZZLE(@selector(resetCommandsInBuffer:withRange:), swizzled_blit_resetCommandsInBuffer)
        BLIT_SWIZZLE(@selector(copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:), swizzled_blit_copyIndirectCommandBuffer)
        BLIT_SWIZZLE(@selector(optimizeIndirectCommandBuffer:withRange:), swizzled_blit_optimizeIndirectCommandBuffer)
        BLIT_SWIZZLE(@selector(resolveCounters:inRange:destinationBuffer:destinationOffset:), swizzled_blit_resolveCounters)
        BLIT_SWIZZLE(@selector(sampleCountersInBuffer:atSampleIndex:withBarrier:), swizzled_blit_sampleCounters)
        BLIT_SWIZZLE(@selector(updateFence:), swizzled_blit_updateFence)
        BLIT_SWIZZLE(@selector(waitForFence:), swizzled_blit_waitForFence)
        BLIT_SWIZZLE(@selector(getTextureAccessCounters:region:mipLevel:slice:resetCounters:countersBuffer:countersBufferOffset:), swizzled_blit_getTextureAccessCounters)
        BLIT_SWIZZLE(@selector(resetTextureAccessCounters:region:mipLevel:slice:), swizzled_blit_resetTextureAccessCounters)
        #undef BLIT_SWIZZLE
        os_log(g_log, "AGX Fix v2.9: All blit encoder methods protected (GAP 34-51 FIX)");
    }

    // Render encoder
    if (g_agx_render_encoder_class) {
        IMP render_dummy;
        if (swizzle_method(g_agx_command_buffer_class, @selector(renderCommandEncoderWithDescriptor:),
                           (IMP)swizzled_renderCommandEncoderWithDescriptor, &g_original_renderCommandEncoderWithDescriptor)) swizzled_count++;
        if (swizzle_method(g_agx_render_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_render_endEncoding, &g_original_render_endEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_render_encoder_class, @selector(deferredEndEncoding),
                           (IMP)swizzled_render_deferredEndEncoding, &g_original_render_deferredEndEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_render_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_render_dealloc, &g_original_render_dealloc)) swizzled_count++;

        // v2.9 GAP 52-60: Render encoder work methods (not used by PyTorch MPS)
        #define RENDER_SWIZZLE(sel, func) \
            if (swizzle_method(g_agx_render_encoder_class, sel, (IMP)func, &render_dummy)) swizzled_count++;
        RENDER_SWIZZLE(@selector(setVertexBuffer:offset:atIndex:), swizzled_render_setVertexBuffer)
        RENDER_SWIZZLE(@selector(setVertexBytes:length:atIndex:), swizzled_render_setVertexBytes)
        RENDER_SWIZZLE(@selector(setFragmentBuffer:offset:atIndex:), swizzled_render_setFragmentBuffer)
        RENDER_SWIZZLE(@selector(setFragmentBytes:length:atIndex:), swizzled_render_setFragmentBytes)
        RENDER_SWIZZLE(@selector(setVertexTexture:atIndex:), swizzled_render_setVertexTexture)
        RENDER_SWIZZLE(@selector(setFragmentTexture:atIndex:), swizzled_render_setFragmentTexture)
        RENDER_SWIZZLE(@selector(setRenderPipelineState:), swizzled_render_setRenderPipelineState)
        RENDER_SWIZZLE(@selector(drawPrimitives:vertexStart:vertexCount:), swizzled_render_drawPrimitives)
        RENDER_SWIZZLE(@selector(drawPrimitives:vertexStart:vertexCount:instanceCount:), swizzled_render_drawPrimitivesInstanced)
        #undef RENDER_SWIZZLE
        os_log(g_log, "AGX Fix v2.9: Render encoder methods protected (GAP 52-60)");
    }

    // v2.9 GAP 3 & 4: Parallel render encoder and sub-encoder creation
    if (g_agx_parallel_render_encoder_class) {
        if (swizzle_method(g_agx_parallel_render_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_parallel_render_endEncoding, &g_original_parallel_render_endEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_parallel_render_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_parallel_render_dealloc, &g_original_parallel_render_dealloc)) swizzled_count++;
        // v2.9 GAP 4 FIX: Swizzle sub-encoder creation
        if (swizzle_method(g_agx_parallel_render_encoder_class, @selector(renderCommandEncoder),
                           (IMP)swizzled_parallel_render_sub_encoder, &g_original_parallel_render_sub_encoder)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.9: Swizzled parallel render sub-encoder creation (GAP 4 FIX)");
        }
        os_log(g_log, "AGX Fix v2.9: Swizzled parallel render encoder endEncoding/dealloc");
    }

    // Resource state encoder
    if (g_agx_resource_state_encoder_class) {
        if (swizzle_method(g_agx_command_buffer_class, @selector(resourceStateCommandEncoder),
                           (IMP)swizzled_resourceStateCommandEncoder, &g_original_resourceStateCommandEncoder)) swizzled_count++;
        if (swizzle_method(g_agx_resource_state_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_resource_state_endEncoding, &g_original_resource_state_endEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_resource_state_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_resource_state_dealloc, &g_original_resource_state_dealloc)) swizzled_count++;
    }

    // Acceleration structure encoder
    if (g_agx_accel_struct_encoder_class) {
        if (swizzle_method(g_agx_command_buffer_class, @selector(accelerationStructureCommandEncoder),
                           (IMP)swizzled_accelerationStructureCommandEncoder, &g_original_accelerationStructureCommandEncoder)) swizzled_count++;
        if (swizzle_method(g_agx_accel_struct_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_accel_struct_endEncoding, &g_original_accel_struct_endEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_accel_struct_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_accel_struct_dealloc, &g_original_accel_struct_dealloc)) swizzled_count++;
    }

    os_log(g_log, "AGX Fix v2.9: COMPLETE - %d methods protected", swizzled_count);

    // GAP 3 FIX: Verify swizzle is active (detect IMP caching bypass)
    {
        Method commit_method = class_getInstanceMethod(g_agx_command_buffer_class, @selector(commit));
        if (commit_method) {
            IMP current_commit_imp = method_getImplementation(commit_method);
            if (current_commit_imp != (IMP)swizzled_commit) {
                AGX_LOG_ERROR("AGX Fix v2.9: WARNING - commit swizzle may be bypassed by IMP caching! "
                              "Expected IMP %p, got %p", (void*)swizzled_commit, (void*)current_commit_imp);
            } else {
                os_log(g_log, "AGX Fix v2.9: GAP 3 VERIFIED - commit swizzle active (IMP: %p)", (void*)current_commit_imp);
            }
        }

        Method encode_method = class_getInstanceMethod(g_agx_encoder_class, @selector(endEncoding));
        if (encode_method) {
            IMP current_enc_imp = method_getImplementation(encode_method);
            if (current_enc_imp != (IMP)swizzled_endEncoding) {
                AGX_LOG_ERROR("AGX Fix v2.9: WARNING - endEncoding swizzle may be bypassed! "
                              "Expected IMP %p, got %p", (void*)swizzled_endEncoding, (void*)current_enc_imp);
            } else {
                os_log(g_log, "AGX Fix v2.9: GAP 3 VERIFIED - endEncoding swizzle active (IMP: %p)", (void*)current_enc_imp);
            }
        }
    }

    os_log(g_log, "AGX Fix v2.9: GAP FIXES APPLIED:");
    os_log(g_log, "  GAP 1: Mutex held through commit (no race window)");
    os_log(g_log, "  GAP 2: Block indefinitely on event ops (no timeout escape)");
    os_log(g_log, "  GAP 3: parallelRenderCommandEncoder tracked");
    os_log(g_log, "  GAP 4: Sub-encoders from parallelRenderCommandEncoder tracked");
    os_log(g_log, "  GAP 5: Descriptor-based encoder creation methods tracked");
    os_log(g_log, "  GAP 6: High-priority constructor (runs before other dylibs)");
    os_log(g_log, "  GAP 7: Unknown encoder class detection");
    os_log(g_log, "  GAP 8: Direct IMP calls documented as limitation");
    os_log(g_log, "  GAP 9: encoder_method_begin checks ended state");
    os_log(g_log, "  GAP 10: Reject untracked encoders (TOCTOU safety)");
    os_log(g_log, "  GAP 11: dispatchThreadsWithIndirectBuffer tracked");
    os_log(g_log, "  GAP 12: executeCommandsInBuffer:indirectBuffer tracked");
    os_log(g_log, "  GAP 13-17: Ray tracing methods tracked");
    os_log(g_log, "  GAP 22-33: Additional compute encoder methods tracked");
    os_log(g_log, "  GAP 34-51: All blit encoder methods tracked");
    os_log(g_log, "  GAP 52-60: Core render encoder methods tracked");
    os_log(g_log, "AGX Fix v2.9: 77+ ENCODER METHODS PROTECTED (compute + blit + render)");
}

// ============================================================================
// Statistics API
// ============================================================================

extern "C" {
    uint64_t agx_fix_v2_9_get_acquisitions() { return g_mutex_acquisitions.load(); }
    uint64_t agx_fix_v2_9_get_contentions() { return g_mutex_contentions.load(); }
    uint64_t agx_fix_v2_9_get_encoders_created() { return g_encoders_created.load(); }
    uint64_t agx_fix_v2_9_get_encoders_released() { return g_encoders_released.load(); }
    uint64_t agx_fix_v2_9_get_null_impl_skips() { return g_null_impl_skips.load(); }
    uint64_t agx_fix_v2_9_get_method_calls() { return g_method_calls.load(); }
    uint64_t agx_fix_v2_9_get_deferred_releases() { return g_deferred_releases.load(); }
    uint64_t agx_fix_v2_9_get_agx_encoder_creates() { return g_agx_encoder_creates.load(); }
    uint64_t agx_fix_v2_9_get_mps_encoder_creates() { return g_mps_encoder_creates.load(); }
    uint64_t agx_fix_v2_9_get_commit_forced_ends() { return g_commit_forced_ends.load(); }
    uint64_t agx_fix_v2_9_get_commits_intercepted() { return g_commits_intercepted.load(); }
    uint64_t agx_fix_v2_9_get_signal_events_intercepted() { return g_signal_events_intercepted.load(); }
    uint64_t agx_fix_v2_9_get_wait_events_intercepted() { return g_wait_events_intercepted.load(); }
    uint64_t agx_fix_v2_9_get_event_encoder_waits() { return g_event_encoder_waits.load(); }
    uint64_t agx_fix_v2_9_get_parallel_encoder_creates() { return g_parallel_encoder_creates.load(); }
    size_t agx_fix_v2_9_get_active_count() {
        std::lock_guard<std::recursive_timed_mutex> lock(g_encoder_mutex);
        return g_encoder_states.size();
    }
    bool agx_fix_v2_9_is_enabled() { return g_enabled; }

    // Gap 9 (roadmap): deadlock/lock inversion diagnostics
    bool agx_fix_v2_9_deadlock_detection_enabled() { return g_deadlock_detection; }
    uint64_t agx_fix_v2_9_get_mutex_long_wait_warnings() { return g_mutex_long_wait_warnings.load(); }
    uint64_t agx_fix_v2_9_get_mutex_lock_timeouts() { return g_mutex_lock_timeouts.load(); }
    uint64_t agx_fix_v2_9_get_mutex_max_wait_ms() { return g_mutex_max_wait_ms.load(); }

    // GAP 3: Runtime verification that swizzle is active
    bool agx_fix_v2_9_verify_swizzle_active() {
        if (!g_agx_command_buffer_class || !g_agx_encoder_class) return false;

        Method commit_method = class_getInstanceMethod(g_agx_command_buffer_class, @selector(commit));
        if (!commit_method) return false;
        IMP current_commit_imp = method_getImplementation(commit_method);
        if (current_commit_imp != (IMP)swizzled_commit) return false;

        Method encode_method = class_getInstanceMethod(g_agx_encoder_class, @selector(endEncoding));
        if (!encode_method) return false;
        IMP current_enc_imp = method_getImplementation(encode_method);
        if (current_enc_imp != (IMP)swizzled_endEncoding) return false;

        return true;
    }

    // GAP 4: Get discovered class names for diagnostics
    const char* agx_fix_v2_9_get_encoder_class_name() {
        return g_agx_encoder_class ? class_getName(g_agx_encoder_class) : "NOT_FOUND";
    }
    const char* agx_fix_v2_9_get_command_buffer_class_name() {
        return g_agx_command_buffer_class ? class_getName(g_agx_command_buffer_class) : "NOT_FOUND";
    }
}
