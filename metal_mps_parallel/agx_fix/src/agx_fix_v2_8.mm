/**
 * AGX Driver Race Condition Fix - Version 2.8
 *
 * FIXES BUG #048 - encodeSignalEvent:value: with uncommitted encoder:
 *   v2.7 swizzled commit to ensure encoders are ended before commit.
 *   However, encodeSignalEvent:value: has the same requirement - Metal
 *   asserts that all encoders must be ended before signal/wait event encoding.
 *
 *   PyTorch's MPSEvent::record() calls encodeSignalEvent:value:, which
 *   caused SIGABRT with "encodeSignalEvent:value: with uncommitted encoder".
 *
 * v2.8 CHANGES from v2.7:
 *   1. Swizzle encodeSignalEvent:value: to ensure encoders ended first
 *   2. Swizzle encodeWaitForEvent:value: for completeness (same validation)
 *   3. All encoder-ending operations now go through ensure_all_encoders_ended
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
#import <unordered_set>
#import <os/log.h>
#import <thread>
#import <chrono>

// ============================================================================
// Configuration
// ============================================================================

#define AGX_FIX_DISABLE_ENV "AGX_FIX_DISABLE"
#define AGX_FIX_VERBOSE_ENV "AGX_FIX_VERBOSE"

// ============================================================================
// Per-Encoder State
// ============================================================================

namespace {
    struct EncoderState {
        int32_t active_calls = 0;
        bool ended = false;
        int32_t retain_count = 0;
        void* command_buffer = nullptr;  // v2.7: Track owning command buffer

        EncoderState() = default;
    };
}

// ============================================================================
// Global State
// ============================================================================

namespace {
    std::recursive_mutex g_encoder_mutex;
    std::unordered_map<void*, EncoderState> g_encoder_states;
    
    // v2.7: Track which encoders belong to which command buffer
    std::unordered_map<void*, std::unordered_set<void*>> g_command_buffer_encoders;

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
    std::atomic<uint64_t> g_commit_forced_ends{0};  // v2.7: Encoders force-ended at commit
    std::atomic<uint64_t> g_commits_intercepted{0}; // v2.7: Total commits intercepted
    std::atomic<uint64_t> g_signal_events_intercepted{0}; // v2.8: Signal event calls intercepted
    std::atomic<uint64_t> g_wait_events_intercepted{0};   // v2.8: Wait event calls intercepted

    os_log_t g_log = nullptr;
    bool g_verbose = false;
    bool g_enabled = true;

    // AGX Command Buffer originals
    IMP g_original_computeCommandEncoder = nullptr;
    IMP g_original_computeCommandEncoderWithDescriptor = nullptr;
    IMP g_original_computeCommandEncoderWithDispatchType = nullptr;
    IMP g_original_blitCommandEncoder = nullptr;
    IMP g_original_commit = nullptr;  // v2.7
    IMP g_original_encodeSignalEvent = nullptr;  // v2.8
    IMP g_original_encodeWaitForEvent = nullptr; // v2.8

    // MPS Command Buffer originals
    IMP g_original_mps_computeCommandEncoder = nullptr;
    IMP g_original_mps_computeCommandEncoderWithDescriptor = nullptr;
    IMP g_original_mps_computeCommandEncoderWithDispatchType = nullptr;
    IMP g_original_mps_blitCommandEncoder = nullptr;
    IMP g_original_mps_renderCommandEncoderWithDescriptor = nullptr;
    IMP g_original_mps_resourceStateCommandEncoder = nullptr;
    IMP g_original_mps_accelerationStructureCommandEncoder = nullptr;
    IMP g_original_mps_commit = nullptr;  // v2.7
    IMP g_original_mps_encodeSignalEvent = nullptr;  // v2.8
    IMP g_original_mps_encodeWaitForEvent = nullptr; // v2.8

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
// Mutex Guard
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
// v2.7: Get endEncoding IMP for an encoder class
// ============================================================================

static IMP get_end_encoding_imp_for_encoder(id encoder) {
    if (!encoder) return nullptr;
    Class cls = [encoder class];
    if (cls == g_agx_encoder_class) return g_original_endEncoding;
    if (cls == g_agx_blit_encoder_class) return g_original_blit_endEncoding;
    if (cls == g_agx_render_encoder_class) return g_original_render_endEncoding;
    if (cls == g_agx_resource_state_encoder_class) return g_original_resource_state_endEncoding;
    if (cls == g_agx_accel_struct_encoder_class) return g_original_accel_struct_endEncoding;
    return nullptr;
}

// ============================================================================
// Encoder Lifecycle with Command Buffer Tracking
// ============================================================================

static void encoder_created_v27(id encoder, id commandBuffer) {
    if (!encoder) return;

    AGXMutexGuard guard;
    void* encoder_ptr = (__bridge void*)encoder;
    void* cb_ptr = commandBuffer ? (__bridge void*)commandBuffer : nullptr;

    auto it = g_encoder_states.find(encoder_ptr);
    if (it != g_encoder_states.end()) {
        AGX_LOG("AGX Fix v2.8: Encoder %p already tracked", encoder_ptr);
        return;
    }

    EncoderState state;
    state.command_buffer = cb_ptr;

    try {
        g_encoder_states.emplace(encoder_ptr, state);
        if (cb_ptr) {
            g_command_buffer_encoders[cb_ptr].insert(encoder_ptr);
        }
    } catch (const std::bad_alloc&) {
        AGX_LOG_ERROR("AGX Fix v2.8: OOM creating state for encoder %p", encoder_ptr);
        return;
    }

    CFRetain((__bridge CFTypeRef)encoder);
    g_encoder_states[encoder_ptr].retain_count = 1;
    g_encoders_created++;

    AGX_LOG("AGX Fix v2.8: Created encoder %p for command buffer %p", encoder_ptr, cb_ptr);
}

static bool encoder_method_begin(id encoder) {
    if (!encoder) return false;

    void* ptr = (__bridge void*)encoder;
    auto it = g_encoder_states.find(ptr);

    if (it == g_encoder_states.end()) {
        try {
            g_encoder_states.emplace(ptr, EncoderState{});
            CFRetain((__bridge CFTypeRef)encoder);
            g_encoder_states[ptr].retain_count = 1;
            it = g_encoder_states.find(ptr);
        } catch (const std::bad_alloc&) {
            return false;
        }
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
// v2.7: Commit Safety - Force-end active encoders before commit
// ============================================================================

static void ensure_all_encoders_ended_for_command_buffer(id commandBuffer) {
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
            AGX_LOG_ERROR("AGX Fix v2.8: Force-ending encoder %p before commit", encoder_ptr);
            g_commit_forced_ends++;
            typedef void (*EndFunc)(id, SEL);
            ((EndFunc)endImp)(encoder, @selector(endEncoding));
            enc_it->second.ended = true;
        }
    }
}

static void swizzled_commit(id self, SEL _cmd) {
    g_commits_intercepted++;
    {
        AGXMutexGuard guard;
        ensure_all_encoders_ended_for_command_buffer(self);
    }
    typedef void (*Func)(id, SEL);
    ((Func)g_original_commit)(self, _cmd);
}

static void swizzled_mps_commit(id self, SEL _cmd) {
    g_commits_intercepted++;
    {
        AGXMutexGuard guard;
        ensure_all_encoders_ended_for_command_buffer(self);
    }
    typedef void (*Func)(id, SEL);
    ((Func)g_original_mps_commit)(self, _cmd);
}

// ============================================================================
// v2.8: Signal/Wait Event Safety - Wait for active encoders before event ops
//
// CRITICAL: We must NOT force-end encoders for event operations because that
// truncates GPU work and causes timeouts. Instead, we wait for encoders to
// finish naturally. This is different from commit, where force-ending is safe
// because commit is the final operation on a command buffer.
// ============================================================================

// Statistics for event waits
namespace {
    std::atomic<uint64_t> g_event_encoder_waits{0};  // Times we had to wait for encoders
}

static bool has_active_encoders_for_command_buffer(void* cb_ptr) {
    // Must be called with g_encoder_mutex held
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

static void wait_for_encoders_to_end(id commandBuffer) {
    if (!commandBuffer) return;
    void* cb_ptr = (__bridge void*)commandBuffer;

    // Short-circuit: check without waiting first
    {
        AGXMutexGuard guard;
        if (!has_active_encoders_for_command_buffer(cb_ptr)) {
            return;
        }
    }

    // There are active encoders, we need to wait
    g_event_encoder_waits++;
    AGX_LOG("AGX Fix v2.8: Waiting for encoders to end before event on CB %p", cb_ptr);

    // Spin-wait with yields (max 1000 iterations = ~100ms at 100us/iter)
    for (int i = 0; i < 1000; i++) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        {
            AGXMutexGuard guard;
            if (!has_active_encoders_for_command_buffer(cb_ptr)) {
                AGX_LOG("AGX Fix v2.8: Encoders ended after %d waits", i + 1);
                return;
            }
        }
    }

    // Timeout - log error but proceed (will likely crash or timeout anyway)
    AGX_LOG_ERROR("AGX Fix v2.8: TIMEOUT waiting for encoders to end on CB %p - proceeding anyway", cb_ptr);
}

static void swizzled_encodeSignalEvent(id self, SEL _cmd, id event, uint64_t value) {
    g_signal_events_intercepted++;
    wait_for_encoders_to_end(self);
    typedef void (*Func)(id, SEL, id, uint64_t);
    ((Func)g_original_encodeSignalEvent)(self, _cmd, event, value);
}

static void swizzled_encodeWaitForEvent(id self, SEL _cmd, id event, uint64_t value) {
    g_wait_events_intercepted++;
    wait_for_encoders_to_end(self);
    typedef void (*Func)(id, SEL, id, uint64_t);
    ((Func)g_original_encodeWaitForEvent)(self, _cmd, event, value);
}

static void swizzled_mps_encodeSignalEvent(id self, SEL _cmd, id event, uint64_t value) {
    g_signal_events_intercepted++;
    wait_for_encoders_to_end(self);
    typedef void (*Func)(id, SEL, id, uint64_t);
    ((Func)g_original_mps_encodeSignalEvent)(self, _cmd, event, value);
}

static void swizzled_mps_encodeWaitForEvent(id self, SEL _cmd, id event, uint64_t value) {
    g_wait_events_intercepted++;
    wait_for_encoders_to_end(self);
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

__attribute__((constructor))
static void agx_fix_v2_8_init() {
    g_log = os_log_create("com.agxfix.v2.8", "main");

    if (getenv(AGX_FIX_DISABLE_ENV)) {
        g_enabled = false;
        os_log(g_log, "AGX Fix v2.8: Disabled via environment");
        return;
    }
    if (getenv(AGX_FIX_VERBOSE_ENV)) {
        g_verbose = true;
    }

    os_log(g_log, "AGX Fix v2.8: Initializing (with event-safety for Bug #048 fix)");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        AGX_LOG_ERROR("AGX Fix v2.8: No Metal device");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (!encoder || !commandBuffer) {
        AGX_LOG_ERROR("AGX Fix v2.8: Failed to create test objects");
        return;
    }

    g_agx_encoder_class = [encoder class];
    g_agx_command_buffer_class = [commandBuffer class];
    os_log(g_log, "AGX Fix v2.8: AGX Encoder class: %s", class_getName(g_agx_encoder_class));
    os_log(g_log, "AGX Fix v2.8: AGX Command buffer class: %s", class_getName(g_agx_command_buffer_class));
    [encoder endEncoding];

    // Discover MPSCommandBuffer
    @try {
        MPSCommandBuffer* mpsCommandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        if (mpsCommandBuffer) {
            g_mps_command_buffer_class = [mpsCommandBuffer class];
            os_log(g_log, "AGX Fix v2.8: MPS Command buffer class: %s", class_getName(g_mps_command_buffer_class));
        }
    } @catch (NSException* e) {
        os_log(g_log, "AGX Fix v2.8: Exception creating MPSCommandBuffer");
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

    // v2.7: CRITICAL - Swizzle commit on AGX command buffer
    if (swizzle_method(g_agx_command_buffer_class, @selector(commit),
                       (IMP)swizzled_commit, &g_original_commit)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.8: Swizzled AGX commit (CRITICAL for validation fix)");
    }

    // v2.8: CRITICAL - Swizzle encodeSignalEvent:value: on AGX command buffer (Bug #048 fix)
    if (swizzle_method(g_agx_command_buffer_class, @selector(encodeSignalEvent:value:),
                       (IMP)swizzled_encodeSignalEvent, &g_original_encodeSignalEvent)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.8: Swizzled AGX encodeSignalEvent:value: (CRITICAL for Bug #048)");
    }

    // v2.8: Swizzle encodeWaitForEvent:value: for completeness
    if (swizzle_method(g_agx_command_buffer_class, @selector(encodeWaitForEvent:value:),
                       (IMP)swizzled_encodeWaitForEvent, &g_original_encodeWaitForEvent)) {
        swizzled_count++;
        os_log(g_log, "AGX Fix v2.8: Swizzled AGX encodeWaitForEvent:value:");
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

    // v2.7: Swizzle commit on MPS command buffer
    if (g_mps_command_buffer_class && g_mps_command_buffer_class != g_agx_command_buffer_class) {
        if (swizzle_method(g_mps_command_buffer_class, @selector(commit),
                           (IMP)swizzled_mps_commit, &g_original_mps_commit)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.8: Swizzled MPS commit");
        }
        // v2.8: Swizzle encodeSignalEvent:value: on MPS command buffer
        if (swizzle_method(g_mps_command_buffer_class, @selector(encodeSignalEvent:value:),
                           (IMP)swizzled_mps_encodeSignalEvent, &g_original_mps_encodeSignalEvent)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.8: Swizzled MPS encodeSignalEvent:value:");
        }
        // v2.8: Swizzle encodeWaitForEvent:value: on MPS command buffer
        if (swizzle_method(g_mps_command_buffer_class, @selector(encodeWaitForEvent:value:),
                           (IMP)swizzled_mps_encodeWaitForEvent, &g_original_mps_encodeWaitForEvent)) {
            swizzled_count++;
            os_log(g_log, "AGX Fix v2.8: Swizzled MPS encodeWaitForEvent:value:");
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
    }

    // Render encoder
    if (g_agx_render_encoder_class) {
        if (swizzle_method(g_agx_command_buffer_class, @selector(renderCommandEncoderWithDescriptor:),
                           (IMP)swizzled_renderCommandEncoderWithDescriptor, &g_original_renderCommandEncoderWithDescriptor)) swizzled_count++;
        if (swizzle_method(g_agx_render_encoder_class, @selector(endEncoding),
                           (IMP)swizzled_render_endEncoding, &g_original_render_endEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_render_encoder_class, @selector(deferredEndEncoding),
                           (IMP)swizzled_render_deferredEndEncoding, &g_original_render_deferredEndEncoding)) swizzled_count++;
        if (swizzle_method(g_agx_render_encoder_class, sel_registerName("dealloc"),
                           (IMP)swizzled_render_dealloc, &g_original_render_dealloc)) swizzled_count++;
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

    os_log(g_log, "AGX Fix v2.8: COMPLETE - %d methods protected (with event-safety for Bug #048)", swizzled_count);
}

// ============================================================================
// Statistics API
// ============================================================================

extern "C" {
    uint64_t agx_fix_v2_8_get_acquisitions() { return g_mutex_acquisitions.load(); }
    uint64_t agx_fix_v2_8_get_contentions() { return g_mutex_contentions.load(); }
    uint64_t agx_fix_v2_8_get_encoders_created() { return g_encoders_created.load(); }
    uint64_t agx_fix_v2_8_get_encoders_released() { return g_encoders_released.load(); }
    uint64_t agx_fix_v2_8_get_null_impl_skips() { return g_null_impl_skips.load(); }
    uint64_t agx_fix_v2_8_get_method_calls() { return g_method_calls.load(); }
    uint64_t agx_fix_v2_8_get_deferred_releases() { return g_deferred_releases.load(); }
    uint64_t agx_fix_v2_8_get_agx_encoder_creates() { return g_agx_encoder_creates.load(); }
    uint64_t agx_fix_v2_8_get_mps_encoder_creates() { return g_mps_encoder_creates.load(); }
    uint64_t agx_fix_v2_8_get_commit_forced_ends() { return g_commit_forced_ends.load(); }
    uint64_t agx_fix_v2_8_get_commits_intercepted() { return g_commits_intercepted.load(); }
    uint64_t agx_fix_v2_8_get_signal_events_intercepted() { return g_signal_events_intercepted.load(); }
    uint64_t agx_fix_v2_8_get_wait_events_intercepted() { return g_wait_events_intercepted.load(); }
    uint64_t agx_fix_v2_8_get_event_encoder_waits() { return g_event_encoder_waits.load(); }
    size_t agx_fix_v2_8_get_active_count() {
        std::lock_guard<std::recursive_mutex> lock(g_encoder_mutex);
        return g_encoder_states.size();
    }
    bool agx_fix_v2_8_is_enabled() { return g_enabled; }
}
