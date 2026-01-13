#!/bin/bash
# collect_platform_fingerprint.sh - Collect comprehensive platform information
# Used for cross-platform verification and debugging platform-specific issues

set -e

echo "=== MPS Platform Fingerprint ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Hostname: $(hostname)"
echo ""

echo "=== Hardware ==="
echo "Chip:"
sysctl -n machdep.cpu.brand_string

echo ""
echo "Hardware Overview:"
system_profiler SPHardwareDataType | grep -E "Chip|Cores|Memory|Model"

echo ""
echo "=== GPU Information ==="
system_profiler SPDisplaysDataType | grep -E "Chipset|Cores|Metal|Vendor|Type|Bus"

echo ""
echo "=== Software ==="
echo "macOS Version:"
sw_vers

echo ""
echo "Xcode Version:"
xcodebuild -version 2>/dev/null || echo "Xcode not installed"

echo ""
echo "Clang Version:"
clang --version 2>/dev/null | head -1 || echo "clang not found"

echo ""
echo "=== Python Environment ==="
if command -v python3 &> /dev/null; then
    echo "Python: $(python3 --version)"
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch not available"
    python3 -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')" 2>/dev/null || echo "Cannot check MPS"
else
    echo "Python3 not found"
fi

echo ""
echo "=== Metal Capabilities ==="
# Try to run our Metal capability query tool if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../verification/run_platform_checks" ]; then
    "$SCRIPT_DIR/../verification/run_platform_checks" 2>/dev/null || echo "Platform checks not available"
elif [ -f "./verification/run_platform_checks" ]; then
    ./verification/run_platform_checks 2>/dev/null || echo "Platform checks not available"
else
    echo "Platform checks tool not built. Run: cd verification && make"
fi

echo ""
echo "=== System Load ==="
uptime

echo ""
echo "=== Memory Pressure ==="
vm_stat | head -10

echo ""
echo "=== Thermal State ==="
# Query thermal state if pmset is available (with timeout to avoid hanging)
timeout 5 pmset -g therm 2>/dev/null || echo "Thermal state not available"

echo ""
echo "=== Fingerprint Complete ==="
