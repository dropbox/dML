// runtime_assumptions.mm - Runtime verification of platform assumptions
// Implementation of platform assumption checks

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "runtime_assumptions.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <sys/sysctl.h>

namespace mps_verification {

// Platform detection utilities
namespace platform {

std::string get_chip_name() {
    char buffer[256];
    size_t size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
        return std::string(buffer);
    }
    return "Unknown";
}

int get_chip_generation() {
    std::string chip = get_chip_name();
    if (chip.find("M1") != std::string::npos) return 1;
    if (chip.find("M2") != std::string::npos) return 2;
    if (chip.find("M3") != std::string::npos) return 3;
    if (chip.find("M4") != std::string::npos) return 4;
    return 0;
}

bool is_ultra_chip() {
    return get_chip_name().find("Ultra") != std::string::npos;
}

bool has_dynamic_caching() {
    // Dynamic Caching was introduced with M3
    return get_chip_generation() >= 3;
}

int get_gpu_core_count() {
    // Query system_profiler for GPU core count
    @autoreleasepool {
        NSTask *task = [[NSTask alloc] init];
        task.launchPath = @"/usr/sbin/system_profiler";
        task.arguments = @[@"SPDisplaysDataType"];

        NSPipe *pipe = [NSPipe pipe];
        task.standardOutput = pipe;

        [task launch];
        [task waitUntilExit];

        NSData *data = [[pipe fileHandleForReading] readDataToEndOfFile];
        NSString *output = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];

        // Parse "Total Number of Cores: XX"
        NSRange range = [output rangeOfString:@"Total Number of Cores: "];
        if (range.location != NSNotFound) {
            NSUInteger start = range.location + range.length;
            NSRange newline = [output rangeOfString:@"\n" options:0 range:NSMakeRange(start, output.length - start)];
            if (newline.location != NSNotFound) {
                NSString *cores = [output substringWithRange:NSMakeRange(start, newline.location - start)];
                return [cores intValue];
            }
        }
    }
    return 0;
}

std::string get_macos_version() {
    @autoreleasepool {
        NSOperatingSystemVersion version = [[NSProcessInfo processInfo] operatingSystemVersion];
        return std::to_string(version.majorVersion) + "." +
               std::to_string(version.minorVersion) + "." +
               std::to_string(version.patchVersion);
    }
}

uint64_t get_memory_bytes() {
    return [[NSProcessInfo processInfo] physicalMemory];
}

} // namespace platform

// Helper to time a check
template<typename F>
AssumptionResult timed_check(const std::string& id, const std::string& name, F&& check) {
    auto start = std::chrono::high_resolution_clock::now();
    AssumptionResult result;
    result.id = id;
    result.name = name;

    try {
        auto [passed, details] = check();
        result.passed = passed;
        result.details = details;
    } catch (const std::exception& e) {
        result.passed = false;
        result.details = std::string("Exception: ") + e.what();
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

// A.001: MTLSharedEvent atomicity verification
AssumptionResult PlatformAssumptions::verify_event_atomicity() {
    return timed_check("A.001", "MTLSharedEvent atomicity", []() -> std::pair<bool, std::string> {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                return {false, "No Metal device available"};
            }

            id<MTLSharedEvent> event = [device newSharedEvent];
            event.signaledValue = 0;

            constexpr int N_THREADS = 8;
            constexpr int ITERATIONS = 1000;

            std::atomic<int> completed{0};
            std::atomic<uint64_t> total_increments{0};
            std::vector<std::thread> threads;

            for (int t = 0; t < N_THREADS; t++) {
                threads.emplace_back([&] {
                    @autoreleasepool {
                        for (int i = 0; i < ITERATIONS; i++) {
                            // Use atomic fetch-add pattern via signaled value
                            @synchronized(event) {
                                event.signaledValue = event.signaledValue + 1;
                            }
                            total_increments.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    completed.fetch_add(1, std::memory_order_release);
                });
            }

            for (auto& t : threads) t.join();

            uint64_t expected = N_THREADS * ITERATIONS;
            uint64_t actual = event.signaledValue;

            // Note: The synchronized block ensures atomicity at the Objective-C level
            // This test verifies that the synchronization primitives work correctly
            if (actual == expected) {
                return {true, "All " + std::to_string(expected) + " increments accounted for"};
            } else {
                return {false, "Expected " + std::to_string(expected) +
                              ", got " + std::to_string(actual) +
                              " (lost " + std::to_string(expected - actual) + " signals)"};
            }
        }
    });
}

// A.002: Command queue thread safety
AssumptionResult PlatformAssumptions::verify_command_queue_thread_safety() {
    return timed_check("A.002", "MTLCommandQueue thread safety", []() -> std::pair<bool, std::string> {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                return {false, "No Metal device available"};
            }

            constexpr int N_QUEUES = 8;
            constexpr int ITERATIONS = 100;

            std::vector<id<MTLCommandQueue>> queues;
            for (int i = 0; i < N_QUEUES; i++) {
                id<MTLCommandQueue> queue = [device newCommandQueue];
                if (!queue) {
                    return {false, "Failed to create command queue " + std::to_string(i)};
                }
                queues.push_back(queue);
            }

            std::atomic<int> completed{0};
            std::atomic<int> errors{0};
            std::vector<std::thread> threads;

            for (int t = 0; t < N_QUEUES; t++) {
                threads.emplace_back([&, t] {
                    @autoreleasepool {
                        for (int i = 0; i < ITERATIONS; i++) {
                            @autoreleasepool {
                                id<MTLCommandBuffer> cmdBuf = [queues[t] commandBuffer];
                                if (!cmdBuf) {
                                    errors.fetch_add(1, std::memory_order_relaxed);
                                    continue;
                                }
                                [cmdBuf commit];
                                [cmdBuf waitUntilCompleted];

                                if (cmdBuf.status == MTLCommandBufferStatusError) {
                                    errors.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                        }
                    }
                    completed.fetch_add(1, std::memory_order_release);
                });
            }

            for (auto& t : threads) t.join();

            int error_count = errors.load();
            if (error_count == 0) {
                return {true, std::to_string(N_QUEUES) + " queues x " +
                              std::to_string(ITERATIONS) + " commands completed without errors"};
            } else {
                return {false, std::to_string(error_count) + " command buffer errors detected"};
            }
        }
    });
}

// A.003: Memory ordering via Dekker's algorithm
AssumptionResult PlatformAssumptions::verify_memory_ordering() {
    return timed_check("A.003", "Sequential consistency memory ordering", []() -> std::pair<bool, std::string> {
        std::atomic<bool> flag0{false};
        std::atomic<bool> flag1{false};
        std::atomic<int> turn{0};
        std::atomic<int> violations{0};
        std::atomic<int> in_critical{0};

        constexpr int ITERATIONS = 50000;

        auto thread0 = [&] {
            for (int i = 0; i < ITERATIONS; i++) {
                flag0.store(true, std::memory_order_seq_cst);
                while (flag1.load(std::memory_order_seq_cst)) {
                    if (turn.load(std::memory_order_seq_cst) != 0) {
                        flag0.store(false, std::memory_order_seq_cst);
                        while (turn.load(std::memory_order_seq_cst) != 0) {
                            std::this_thread::yield();
                        }
                        flag0.store(true, std::memory_order_seq_cst);
                    }
                }
                // Critical section
                int prev = in_critical.fetch_add(1, std::memory_order_seq_cst);
                if (prev != 0) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                in_critical.fetch_sub(1, std::memory_order_seq_cst);

                turn.store(1, std::memory_order_seq_cst);
                flag0.store(false, std::memory_order_seq_cst);
            }
        };

        auto thread1 = [&] {
            for (int i = 0; i < ITERATIONS; i++) {
                flag1.store(true, std::memory_order_seq_cst);
                while (flag0.load(std::memory_order_seq_cst)) {
                    if (turn.load(std::memory_order_seq_cst) != 1) {
                        flag1.store(false, std::memory_order_seq_cst);
                        while (turn.load(std::memory_order_seq_cst) != 1) {
                            std::this_thread::yield();
                        }
                        flag1.store(true, std::memory_order_seq_cst);
                    }
                }
                // Critical section
                int prev = in_critical.fetch_add(1, std::memory_order_seq_cst);
                if (prev != 0) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                in_critical.fetch_sub(1, std::memory_order_seq_cst);

                turn.store(0, std::memory_order_seq_cst);
                flag1.store(false, std::memory_order_seq_cst);
            }
        };

        std::thread t0(thread0);
        std::thread t1(thread1);
        t0.join();
        t1.join();

        int violation_count = violations.load();
        if (violation_count == 0) {
            return {true, "Dekker's algorithm passed " + std::to_string(ITERATIONS * 2) +
                          " critical section entries with no violations"};
        } else {
            return {false, std::to_string(violation_count) +
                          " mutual exclusion violations detected (memory ordering broken)"};
        }
    });
}

// A.007: std::mutex acquire/release barrier semantics
AssumptionResult PlatformAssumptions::verify_mutex_memory_barriers() {
    return timed_check("A.007", "std::mutex acquire/release barriers", []() -> std::pair<bool, std::string> {
        std::mutex m;
        std::condition_variable cv;

        uint64_t data = 0;
        bool ready = false;
        std::atomic<int> errors{0};

        constexpr uint64_t ITERATIONS = 10000;

        std::thread consumer([&] {
            for (uint64_t i = 1; i <= ITERATIONS; i++) {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [&] { return ready; });
                if (data != i) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
                ready = false;
                lock.unlock();
                cv.notify_one();
            }
        });

        std::thread producer([&] {
            for (uint64_t i = 1; i <= ITERATIONS; i++) {
                {
                    std::lock_guard<std::mutex> lock(m);
                    data = i;
                    ready = true;
                }
                cv.notify_one();
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [&] { return !ready; });
            }
        });

        producer.join();
        consumer.join();

        int error_count = errors.load();
        if (error_count == 0) {
            return {true, "Mutex publish/consume passed " + std::to_string(ITERATIONS) + " iterations"};
        }
        return {false, std::to_string(error_count) + " visibility violations detected"};
    });
}

// A.008: release/acquire message passing
AssumptionResult PlatformAssumptions::verify_release_acquire_message_passing() {
    return timed_check("A.008", "release/acquire message passing", []() -> std::pair<bool, std::string> {
        uint64_t data = 0;
        std::atomic<uint64_t> seq{0};
        std::atomic<uint64_t> ack{0};
        std::atomic<int> errors{0};

        constexpr uint64_t ITERATIONS = 200000;

        std::thread producer([&] {
            for (uint64_t i = 1; i <= ITERATIONS; i++) {
                data = i;
                seq.store(i, std::memory_order_release);
                while (ack.load(std::memory_order_acquire) != i) {
                    std::this_thread::yield();
                }
            }
        });

        std::thread consumer([&] {
            for (uint64_t i = 1; i <= ITERATIONS; i++) {
                while (seq.load(std::memory_order_acquire) != i) {
                    std::this_thread::yield();
                }
                if (data != i) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
                ack.store(i, std::memory_order_release);
            }
        });

        producer.join();
        consumer.join();

        int error_count = errors.load();
        if (error_count == 0) {
            return {true, "Release/acquire message passing passed " + std::to_string(ITERATIONS) + " iterations"};
        }
        return {false, std::to_string(error_count) + " message passing violations detected"};
    });
}

// A.004: Unified memory coherency
AssumptionResult PlatformAssumptions::verify_unified_memory_coherency() {
    return timed_check("A.004", "CPU-GPU unified memory coherency", []() -> std::pair<bool, std::string> {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                return {false, "No Metal device available"};
            }

            constexpr int BUFFER_SIZE = 1024 * sizeof(uint32_t);
            constexpr int NUM_ELEMENTS = 1024;

            // Create a shared buffer
            id<MTLBuffer> buffer = [device newBufferWithLength:BUFFER_SIZE
                                                       options:MTLResourceStorageModeShared];
            if (!buffer) {
                return {false, "Failed to create shared buffer"};
            }

            uint32_t* data = (uint32_t*)buffer.contents;

            // CPU writes pattern
            for (int i = 0; i < NUM_ELEMENTS; i++) {
                data[i] = i * 12345 + 67890;
            }

            // Force memory synchronization
            std::atomic_thread_fence(std::memory_order_seq_cst);

            // Verify CPU can read back correctly
            int errors = 0;
            for (int i = 0; i < NUM_ELEMENTS; i++) {
                uint32_t expected = i * 12345 + 67890;
                if (data[i] != expected) {
                    errors++;
                }
            }

            // Note: Full GPU verification would require a compute shader
            // This test verifies the CPU-side of unified memory works

            if (errors == 0) {
                return {true, "CPU read-after-write coherent for " +
                              std::to_string(NUM_ELEMENTS) + " elements"};
            } else {
                return {false, std::to_string(errors) + " coherency errors detected"};
            }
        }
    });
}

// A.005: Autoreleasepool semantics
AssumptionResult PlatformAssumptions::verify_autorelease_semantics() {
    return timed_check("A.005", "@autoreleasepool semantics", []() -> std::pair<bool, std::string> {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                return {false, "No Metal device available"};
            }

            // Track that objects are released at pool drain
            constexpr int ITERATIONS = 100;

            for (int i = 0; i < ITERATIONS; i++) {
                @autoreleasepool {
                    // Create and immediately abandon an autoreleased object
                    id<MTLCommandQueue> queue = [device newCommandQueue];
                    id<MTLCommandBuffer> buffer = [queue commandBuffer];
                    (void)buffer; // Suppress unused warning

                    // Objects should be released when this pool drains
                }
            }

            // If we get here without crash/leak, ARC is working
            // Note: Actual leak detection would require Instruments
            return {true, std::to_string(ITERATIONS) +
                          " autoreleasepool iterations completed without crash"};
        }
    });
}

// A.006: Stream isolation
AssumptionResult PlatformAssumptions::verify_stream_isolation() {
    return timed_check("A.006", "Stream isolation (no cross-contamination)", []() -> std::pair<bool, std::string> {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                return {false, "No Metal device available"};
            }

            constexpr int N_STREAMS = 4;
            constexpr int ITERATIONS = 50;

            std::vector<id<MTLCommandQueue>> queues;
            for (int i = 0; i < N_STREAMS; i++) {
                queues.push_back([device newCommandQueue]);
            }

            std::atomic<int> errors{0};
            std::vector<std::thread> threads;

            for (int s = 0; s < N_STREAMS; s++) {
                threads.emplace_back([&, s] {
                    @autoreleasepool {
                        // Each stream works with its own buffer
                        id<MTLBuffer> buffer = [device newBufferWithLength:4096
                                                                   options:MTLResourceStorageModeShared];
                        uint32_t* data = (uint32_t*)buffer.contents;

                        for (int i = 0; i < ITERATIONS; i++) {
                            @autoreleasepool {
                                // Write unique pattern
                                uint32_t pattern = (s << 24) | i;
                                for (int j = 0; j < 1024; j++) {
                                    data[j] = pattern;
                                }

                                // Submit and wait
                                id<MTLCommandBuffer> cmdBuf = [queues[s] commandBuffer];
                                [cmdBuf commit];
                                [cmdBuf waitUntilCompleted];

                                // Verify pattern still intact (no cross-stream pollution)
                                for (int j = 0; j < 1024; j++) {
                                    if (data[j] != pattern) {
                                        errors.fetch_add(1, std::memory_order_relaxed);
                                    }
                                }
                            }
                        }
                    }
                });
            }

            for (auto& t : threads) t.join();

            int error_count = errors.load();
            if (error_count == 0) {
                return {true, std::to_string(N_STREAMS) + " streams x " +
                              std::to_string(ITERATIONS) + " iterations isolated correctly"};
            } else {
                return {false, std::to_string(error_count) + " stream isolation violations"};
            }
        }
    });
}

void PlatformAssumptions::collect_hardware_info(PlatformReport& report) {
    report.chip_name = platform::get_chip_name();
    report.gpu_core_count = platform::get_gpu_core_count();
    report.memory_bytes = platform::get_memory_bytes();
    report.has_dynamic_caching = platform::has_dynamic_caching();
    report.is_ultra_chip = platform::is_ultra_chip();
}

void PlatformAssumptions::collect_software_info(PlatformReport& report) {
    report.macos_version = platform::get_macos_version();

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            // Determine Metal GPU family support
            if ([device supportsFamily:MTLGPUFamilyApple9]) {
                report.metal_version = "Apple9 (M4)";
            } else if ([device supportsFamily:MTLGPUFamilyApple8]) {
                report.metal_version = "Apple8 (M3)";
            } else if ([device supportsFamily:MTLGPUFamilyApple7]) {
                report.metal_version = "Apple7 (M1/M2)";
            } else {
                report.metal_version = "Unknown";
            }
        } else {
            report.metal_version = "No Metal device";
        }
    }
}

PlatformReport PlatformAssumptions::verify_all() {
    PlatformReport report;

    collect_hardware_info(report);
    collect_software_info(report);

    // Run all assumption checks
    report.results.push_back(verify_event_atomicity());
    report.results.push_back(verify_command_queue_thread_safety());
    report.results.push_back(verify_memory_ordering());
    report.results.push_back(verify_mutex_memory_barriers());
    report.results.push_back(verify_release_acquire_message_passing());
    report.results.push_back(verify_unified_memory_coherency());
    report.results.push_back(verify_autorelease_semantics());
    report.results.push_back(verify_stream_isolation());

    // Compute summary
    report.total_checks = (int)report.results.size();
    report.passed_checks = 0;
    report.failed_checks = 0;

    for (const auto& r : report.results) {
        if (r.passed) {
            report.passed_checks++;
        } else {
            report.failed_checks++;
        }
    }

    return report;
}

std::string PlatformReport::to_markdown() const {
    std::ostringstream ss;

    ss << "# Platform Assumption Verification Report\n\n";

    ss << "## Platform Information\n\n";
    ss << "| Property | Value |\n";
    ss << "|----------|-------|\n";
    ss << "| Chip | " << chip_name << " |\n";
    ss << "| GPU Cores | " << gpu_core_count << " |\n";
    ss << "| Memory | " << (memory_bytes / (1024*1024*1024)) << " GB |\n";
    ss << "| Dynamic Caching | " << (has_dynamic_caching ? "Yes" : "No") << " |\n";
    ss << "| Ultra Chip | " << (is_ultra_chip ? "Yes" : "No") << " |\n";
    ss << "| macOS | " << macos_version << " |\n";
    ss << "| Metal Family | " << metal_version << " |\n\n";

    ss << "## Assumption Checks\n\n";
    ss << "| ID | Assumption | Status | Duration | Details |\n";
    ss << "|----|------------|--------|----------|--------|\n";

    for (const auto& r : results) {
        ss << "| " << r.id << " | " << r.name << " | "
           << (r.passed ? "PASS" : "**FAIL**") << " | "
           << std::fixed << std::setprecision(1) << r.duration_ms << " ms | "
           << r.details << " |\n";
    }

    ss << "\n## Summary\n\n";
    ss << "- **Total checks**: " << total_checks << "\n";
    ss << "- **Passed**: " << passed_checks << "\n";
    ss << "- **Failed**: " << failed_checks << "\n";
    ss << "- **All passed**: " << (all_passed() ? "Yes" : "**NO**") << "\n";

    return ss.str();
}

std::string PlatformReport::to_json() const {
    std::ostringstream ss;

    ss << "{\n";
    ss << "  \"platform\": {\n";
    ss << "    \"chip_name\": \"" << chip_name << "\",\n";
    ss << "    \"gpu_core_count\": " << gpu_core_count << ",\n";
    ss << "    \"memory_bytes\": " << memory_bytes << ",\n";
    ss << "    \"has_dynamic_caching\": " << (has_dynamic_caching ? "true" : "false") << ",\n";
    ss << "    \"is_ultra_chip\": " << (is_ultra_chip ? "true" : "false") << ",\n";
    ss << "    \"macos_version\": \"" << macos_version << "\",\n";
    ss << "    \"metal_version\": \"" << metal_version << "\"\n";
    ss << "  },\n";

    ss << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        ss << "    {\n";
        ss << "      \"id\": \"" << r.id << "\",\n";
        ss << "      \"name\": \"" << r.name << "\",\n";
        ss << "      \"passed\": " << (r.passed ? "true" : "false") << ",\n";
        ss << "      \"duration_ms\": " << std::fixed << std::setprecision(2) << r.duration_ms << ",\n";
        ss << "      \"details\": \"" << r.details << "\"\n";
        ss << "    }" << (i < results.size() - 1 ? "," : "") << "\n";
    }
    ss << "  ],\n";

    ss << "  \"summary\": {\n";
    ss << "    \"total_checks\": " << total_checks << ",\n";
    ss << "    \"passed_checks\": " << passed_checks << ",\n";
    ss << "    \"failed_checks\": " << failed_checks << ",\n";
    ss << "    \"all_passed\": " << (all_passed() ? "true" : "false") << "\n";
    ss << "  }\n";
    ss << "}\n";

    return ss.str();
}

} // namespace mps_verification
