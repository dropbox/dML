// runtime_assumptions.h - Runtime verification of platform assumptions
// Part of MPS Parallel Inference verification infrastructure
//
// Our formal proofs (TLA+, Iris/Coq, CBMC) assume certain platform behaviors.
// This module verifies those assumptions hold at runtime on the current platform.

#pragma once

#include <string>
#include <vector>
#include <chrono>

namespace mps_verification {

// Result of a single assumption check
struct AssumptionResult {
    std::string id;           // e.g., "A.001"
    std::string name;         // e.g., "MTLSharedEvent atomicity"
    bool passed;
    std::string details;      // Failure details if not passed
    double duration_ms;       // How long the check took
};

// Complete platform verification report
struct PlatformReport {
    // Hardware info
    std::string chip_name;           // e.g., "Apple M4 Max"
    int gpu_core_count;
    uint64_t memory_bytes;
    bool has_dynamic_caching;        // M3+ feature
    bool is_ultra_chip;              // Dual-die variant

    // Software info
    std::string macos_version;
    std::string metal_version;

    // Assumption check results
    std::vector<AssumptionResult> results;

    // Summary
    int total_checks;
    int passed_checks;
    int failed_checks;
    bool all_passed() const { return failed_checks == 0; }

    // Generate markdown report
    std::string to_markdown() const;

    // Generate JSON report
    std::string to_json() const;
};

// Platform assumption verification class
class PlatformAssumptions {
public:
    // Run all assumption checks
    static PlatformReport verify_all();

    // Individual assumption checks

    // A.001: MTLSharedEvent.signaledValue updates are atomic
    // Tests concurrent increments from multiple threads
    static AssumptionResult verify_event_atomicity();

    // A.002: Multiple MTLCommandQueues can submit concurrently
    // Tests concurrent command buffer submission
    static AssumptionResult verify_command_queue_thread_safety();

    // A.003: Memory barriers provide sequential consistency
    // Uses Dekker's algorithm to verify memory ordering
    static AssumptionResult verify_memory_ordering();

    // A.007: std::mutex lock/unlock provides acquire/release barriers
    // Verifies a producer's writes become visible after consumer lock acquisition
    static AssumptionResult verify_mutex_memory_barriers();

    // A.008: release/acquire message passing works as expected
    // Verifies non-atomic data published before a release store is visible after acquire load
    static AssumptionResult verify_release_acquire_message_passing();

    // A.004: CPU-GPU memory coherency after synchronization
    // Tests read-after-write across CPU/GPU boundary
    static AssumptionResult verify_unified_memory_coherency();

    // A.005: @autoreleasepool correctly releases objects
    // Tests ARC semantics under various conditions
    static AssumptionResult verify_autorelease_semantics();

    // A.006: Concurrent tensor operations don't corrupt
    // Tests our stream pool isolation
    static AssumptionResult verify_stream_isolation();

private:
    // Helper to collect hardware info
    static void collect_hardware_info(PlatformReport& report);

    // Helper to collect software info
    static void collect_software_info(PlatformReport& report);
};

// Utility functions for platform detection
namespace platform {
    // Get chip generation (1=M1, 2=M2, 3=M3, 4=M4, 0=unknown)
    int get_chip_generation();

    // Check if running on Ultra variant (dual-die)
    bool is_ultra_chip();

    // Check if chip has Dynamic Caching (M3+)
    bool has_dynamic_caching();

    // Get GPU core count
    int get_gpu_core_count();

    // Get chip name string
    std::string get_chip_name();

    // Get macOS version string
    std::string get_macos_version();

    // Get total system memory in bytes
    uint64_t get_memory_bytes();
}

} // namespace mps_verification
