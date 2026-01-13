// run_platform_checks.mm - Command-line tool to verify platform assumptions
// Usage: ./run_platform_checks [--json] [--markdown]

#import <Foundation/Foundation.h>
#include "runtime_assumptions.h"
#include <iostream>
#include <cstring>

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        bool output_json = false;
        bool output_markdown = false;

        // Parse arguments
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--json") == 0) {
                output_json = true;
            } else if (strcmp(argv[i], "--markdown") == 0) {
                output_markdown = true;
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                std::cout << "Usage: " << argv[0] << " [--json] [--markdown]\n";
                std::cout << "\nVerifies platform assumptions for MPS parallel inference.\n";
                std::cout << "\nOptions:\n";
                std::cout << "  --json      Output results in JSON format\n";
                std::cout << "  --markdown  Output results in Markdown format\n";
                std::cout << "  --help, -h  Show this help message\n";
                std::cout << "\nDefault output is human-readable text.\n";
                return 0;
            }
        }

        std::cout << "Running platform assumption checks...\n\n";

        auto report = mps_verification::PlatformAssumptions::verify_all();

        if (output_json) {
            std::cout << report.to_json();
        } else if (output_markdown) {
            std::cout << report.to_markdown();
        } else {
            // Default: human-readable output
            std::cout << "=== Platform Information ===\n";
            std::cout << "Chip: " << report.chip_name << "\n";
            std::cout << "GPU Cores: " << report.gpu_core_count << "\n";
            std::cout << "Memory: " << (report.memory_bytes / (1024*1024*1024)) << " GB\n";
            std::cout << "Dynamic Caching: " << (report.has_dynamic_caching ? "Yes" : "No") << "\n";
            std::cout << "Ultra Chip: " << (report.is_ultra_chip ? "Yes" : "No") << "\n";
            std::cout << "macOS: " << report.macos_version << "\n";
            std::cout << "Metal Family: " << report.metal_version << "\n";
            std::cout << "\n=== Assumption Checks ===\n";

            for (const auto& r : report.results) {
                std::cout << (r.passed ? "[PASS]" : "[FAIL]") << " "
                          << r.id << " " << r.name << "\n";
                std::cout << "       " << r.details << " (" << r.duration_ms << " ms)\n";
            }

            std::cout << "\n=== Summary ===\n";
            std::cout << "Total: " << report.total_checks << ", ";
            std::cout << "Passed: " << report.passed_checks << ", ";
            std::cout << "Failed: " << report.failed_checks << "\n";

            if (report.all_passed()) {
                std::cout << "\nAll platform assumptions verified.\n";
            } else {
                std::cout << "\nWARNING: Some assumptions failed! Proofs may not hold on this platform.\n";
            }
        }

        return report.all_passed() ? 0 : 1;
    }
}
