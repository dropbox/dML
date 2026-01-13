// clang_annotations.h - Clang Thread Safety Annotations
//
// These macros enable compile-time thread safety analysis via Clang's
// -Wthread-safety flag. They have zero runtime cost.
//
// Usage:
//   #include "clang_annotations.h"
//
//   class MyClass {
//       std::mutex mutex_ CAPABILITY("mutex");
//       int data_ GUARDED_BY(mutex_);
//
//   public:
//       void write(int x) EXCLUDES(mutex_) {
//           std::lock_guard<std::mutex> lock(mutex_);
//           data_ = x;
//       }
//
//       int read() const REQUIRES(mutex_) {
//           return data_;
//       }
//   };
//
// Compile with:
//   clang++ -Wthread-safety -Wthread-safety-attributes ...
//
// Reference: https://clang.llvm.org/docs/ThreadSafetyAnalysis.html

#ifndef MPS_VERIFY_CLANG_ANNOTATIONS_H
#define MPS_VERIFY_CLANG_ANNOTATIONS_H

// Check for Clang thread safety annotation support
#if defined(__clang__) && defined(__clang_major__) && (__clang_major__ >= 3)
#define MPS_THREAD_ANNOTATION_ATTRIBUTE__(x) __attribute__((x))
#else
#define MPS_THREAD_ANNOTATION_ATTRIBUTE__(x)  // no-op on non-Clang
#endif

// ============================================================================
// Capability Annotations
// ============================================================================

// CAPABILITY("name") - Declares a type as a capability (mutex-like)
// Use on mutex declarations to give them a name for error messages
#define CAPABILITY(x) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(capability(x))

// SCOPED_CAPABILITY - For RAII lock guards
// Automatically handles acquire/release semantics
#define SCOPED_CAPABILITY \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

// ============================================================================
// Data Protection Annotations
// ============================================================================

// GUARDED_BY(mutex) - This data is protected by the named mutex
// Read or write without holding the lock is an error
#define GUARDED_BY(x) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(guarded_by(x))

// PT_GUARDED_BY(mutex) - For pointer types: the *pointed-to* data is protected
// The pointer itself can be read without the lock
#define PT_GUARDED_BY(x) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_by(x))

// ============================================================================
// Function Precondition Annotations
// ============================================================================

// REQUIRES(mutex...) - Function requires these locks be held when called
// Caller must hold the lock(s)
#define REQUIRES(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(requires_capability(__VA_ARGS__))

// REQUIRES_SHARED(mutex...) - Requires shared (read) access
#define REQUIRES_SHARED(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(requires_shared_capability(__VA_ARGS__))

// EXCLUDES(mutex...) - Function must NOT hold these locks when called
// Used to prevent deadlock from recursive lock attempts
#define EXCLUDES(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

// ============================================================================
// Function Effect Annotations
// ============================================================================

// ACQUIRE(mutex...) - Function acquires these locks
// Lock is held after function returns (normal exit)
#define ACQUIRE(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(acquire_capability(__VA_ARGS__))

// ACQUIRE_SHARED(mutex...) - Acquires shared (read) access
#define ACQUIRE_SHARED(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))

// TRY_ACQUIRE(bool, mutex...) - May acquire lock; returns bool success
// If function returns true, lock was acquired
#define TRY_ACQUIRE(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_capability(__VA_ARGS__))

// TRY_ACQUIRE_SHARED(bool, mutex...) - May acquire shared lock
#define TRY_ACQUIRE_SHARED(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_shared_capability(__VA_ARGS__))

// RELEASE(mutex...) - Function releases these locks
#define RELEASE(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(release_capability(__VA_ARGS__))

// RELEASE_SHARED(mutex...) - Releases shared access
#define RELEASE_SHARED(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(release_shared_capability(__VA_ARGS__))

// RELEASE_GENERIC(mutex...) - Releases any type of access
#define RELEASE_GENERIC(...) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(release_generic_capability(__VA_ARGS__))

// ============================================================================
// Special Annotations
// ============================================================================

// NO_THREAD_SAFETY_ANALYSIS - Disable analysis for a function
// Use sparingly, only when analysis is too imprecise
#define NO_THREAD_SAFETY_ANALYSIS \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(no_thread_safety_analysis)

// ASSERT_CAPABILITY(mutex) - Assert that we hold this lock
// For runtime assertions that analysis can't prove
#define ASSERT_CAPABILITY(x) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(assert_capability(x))

// ASSERT_SHARED_CAPABILITY(mutex) - Assert shared access
#define ASSERT_SHARED_CAPABILITY(x) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_capability(x))

// RETURN_CAPABILITY(mutex) - Function returns a reference to a mutex
#define RETURN_CAPABILITY(x) \
    MPS_THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

// ============================================================================
// Recursive Mutex Support
// ============================================================================

// For recursive_mutex, the standard annotations work but re-entrancy
// must be marked with TRY_ACQUIRE that can "succeed" multiple times

// RECURSIVE_CAPABILITY - Marker for recursive mutex types
// Currently no special Clang support, but documents intent
#define RECURSIVE_CAPABILITY CAPABILITY("recursive_mutex")

// ============================================================================
// Portable Macros (for non-Clang compilers)
// ============================================================================

// These expand to nothing on non-Clang but document locking requirements
#ifndef GUARDED_VAR
#define GUARDED_VAR  // Marks a variable as requiring lock; docs-only on non-Clang
#endif

#ifndef THREAD_SAFE
#define THREAD_SAFE  // Marks a function as thread-safe; docs-only
#endif

// ============================================================================
// Example MPS Usage Patterns
// ============================================================================

/*
 * // MPSAllocator example with annotations:
 *
 * class MPSHeapAllocatorImpl {
 * private:
 *     std::recursive_mutex m_mutex CAPABILITY("allocator_mutex");
 *
 *     ska::flat_hash_map<const void*, BufferBlock*> m_allocated_buffers
 *         GUARDED_BY(m_mutex);
 *
 *     std::vector<BufferPool> m_pools GUARDED_BY(m_mutex);
 *
 * public:
 *     // Public API - must not hold lock when called (prevents deadlock)
 *     void* allocate(size_t size) EXCLUDES(m_mutex);
 *     void free(void* ptr) EXCLUDES(m_mutex);
 *
 * private:
 *     // Internal helpers - require lock to be held
 *     bool get_free_buffer_internal(size_t size, BufferBlock** block)
 *         REQUIRES(m_mutex);
 *
 *     void release_buffer(BufferBlock* block) REQUIRES(m_mutex);
 * };
 *
 *
 * // MPSStream example with annotations:
 *
 * class MPSStream {
 * private:
 *     std::recursive_mutex _streamMutex CAPABILITY("stream_mutex");
 *
 *     id<MTLCommandBuffer> _commandBuffer GUARDED_BY(_streamMutex);
 *     id<MTLComputeCommandEncoder> _commandEncoder GUARDED_BY(_streamMutex);
 *
 * public:
 *     // Thread-safe public API
 *     dispatch_queue_t queue() EXCLUDES(_streamMutex);
 *     void synchronize(SyncType type) EXCLUDES(_streamMutex);
 *
 * private:
 *     // Must be called under lock
 *     void endKernelCoalescing() REQUIRES(_streamMutex);
 * };
 */

#endif // MPS_VERIFY_CLANG_ANNOTATIONS_H
