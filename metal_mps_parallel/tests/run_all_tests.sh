#!/bin/bash
# MPS Stream Pool - Comprehensive Test Runner
# Runs all tests in subprocess for clean MPS state

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
REPO_ROOT="$(pwd)"
FORK_DIR="${REPO_ROOT}/pytorch-mps-fork"
CRASH_DIR="${REPO_ROOT}/crash_logs"
# Allow time for crash_monitor.sh / DiagnosticReports to land in crash_logs/
CRASH_WAIT_SECS="${MPS_TEST_CRASH_WAIT_SECS:-6}"

mkdir -p "$CRASH_DIR"

# Use a repo-local temp directory so this test suite can run under sandboxed
# environments that restrict writes to system temp locations (e.g. /tmp).
TMP_ROOT="$(mktemp -d "${REPO_ROOT}/.mps_tests_tmp.XXXXXX")"
cleanup_tmp_root() {
    rm -rf "${TMP_ROOT}"
}
trap cleanup_tmp_root EXIT

# Activate venv if available
if [ -d "venv_mps_test" ]; then
    source venv_mps_test/bin/activate
fi

# Resolve Python binary (venv provides `python`, but many systems only have `python3`).
PYTHON_BIN="python"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: python not found (tried: python, python3)." >&2
    exit 2
fi

count_crash_logs() {
    find "$CRASH_DIR" -maxdepth 1 -type f \( -name "*.ips" -o -name "*.crash" \) 2>/dev/null | wc -l | tr -d ' '
}

print_latest_crash() {
    "${PYTHON_BIN}" scripts/check_crashes.py --latest 2>/dev/null | head -30 || true
}

already_captured() {
    local crash_file="$1"
    local base
    base="$(basename "$crash_file")"

    # We store captures as "<timestamp>_<original_basename>".
    if find "$CRASH_DIR" -maxdepth 1 -name "*_${base}" -print -quit 2>/dev/null | grep -q .; then
        return 0
    fi
    return 1
}

capture_new_crash_reports_since() {
    local since_epoch="$1"
    local copied=0

    # Create a marker file so we can use `find -newer` to detect new crash logs.
    # `touch -t` expects local time: [[CC]YY]MMDDhhmm[.SS]
    local marker
    marker="$(mktemp "${TMP_ROOT}/mps_crash_marker.XXXXXX")"
    if ! touch -t "$(date -r "$since_epoch" +'%Y%m%d%H%M.%S' 2>/dev/null)" "$marker" 2>/dev/null; then
        touch "$marker"
    fi

    local user_dir="$HOME/Library/Logs/DiagnosticReports"
    local system_dir="/Library/Logs/DiagnosticReports"

    for dir in "$user_dir" "$system_dir"; do
        if [ ! -d "$dir" ]; then
            continue
        fi

        while IFS= read -r crash_file; do
            if [ ! -f "$crash_file" ]; then
                continue
            fi

            if already_captured "$crash_file"; then
                continue
            fi

            # Filter for our target domain to reduce false positives.
            if ! grep -q -E "Python|AGX|Metal|mps|torch" "$crash_file" 2>/dev/null; then
                continue
            fi

            local timestamp
            timestamp="$(date +%Y%m%d_%H%M%S)"
            local base
            base="$(basename "$crash_file")"
            local dest="$CRASH_DIR/${timestamp}_${base}"

            cp "$crash_file" "$dest"
            copied=$((copied + 1))
        done < <(
            find "$dir" -maxdepth 1 -type f \
                \( -name "*.ips" -o -name "*.crash" \) \
                -newer "$marker" 2>/dev/null || true
        )
    done

    rm -f "$marker" 2>/dev/null || true
    echo "$copied"
}

# Enable MPS graph path for thread-safe nn.Module inference with 3+ threads.
# Apple's MPSNDArrayMatrixMultiplication (used by default _mps_linear_nograph)
# has internal thread-safety issues that crash with 3+ threads. The graph path
# (MPSGraph) IS thread-safe and works with 8+ threads.
# See reports/main/verification_N48_2025-12-13.md for details.
export MPS_FORCE_GRAPH_PATH=1

# Multi-threaded MPS workloads can still crash due to a documented AGX driver
# race. When available, inject libagx_fix to make the test suite reliable.
# (Opt out with: MPS_TESTS_DISABLE_AGX_FIX=1)
AGX_FIX_LIB=""
AGX_FIX_CANDIDATES=(
    "${REPO_ROOT}/agx_fix/build/libagx_fix_v2_9.dylib"  # recommended - closes formal verification gaps
    "${REPO_ROOT}/agx_fix/build/libagx_fix_v2_8.dylib"  # superseded - event-safety (Bug #048) + commit-safety
    "${REPO_ROOT}/agx_fix/build/libagx_fix_v2_7.dylib"  # superseded - commit-safety only
    "${REPO_ROOT}/agx_fix/build/libagx_fix_v2_5.dylib"  # recommended - fixes PAC trap crash
    "${REPO_ROOT}/agx_fix/build/libagx_fix_v2_3.dylib"
    "${REPO_ROOT}/agx_fix/build/libagx_fix_v2.dylib"
    "${REPO_ROOT}/agx_fix/build/libagx_fix.dylib"
)
for candidate in "${AGX_FIX_CANDIDATES[@]}"; do
    if [ -f "${candidate}" ]; then
        AGX_FIX_LIB="${candidate}"
        break
    fi
done

if [ "${MPS_TESTS_DISABLE_AGX_FIX:-0}" != "1" ] && [ -n "${AGX_FIX_LIB}" ]; then
    if [[ "${DYLD_INSERT_LIBRARIES:-}" != *"libagx_fix"* ]]; then
        if [ -n "${DYLD_INSERT_LIBRARIES:-}" ]; then
            export DYLD_INSERT_LIBRARIES="${AGX_FIX_LIB}:${DYLD_INSERT_LIBRARIES}"
        else
            export DYLD_INSERT_LIBRARIES="${AGX_FIX_LIB}"
        fi
    fi
    echo "AGX fix loaded: ${AGX_FIX_LIB}"
else
    echo "AGX fix not loaded (build: cd agx_fix && make; opt out: MPS_TESTS_DISABLE_AGX_FIX=1)"
fi

# Allow running tests even when the imported torch build git hash does not match
# the current fork HEAD. This is useful when the fork has moved forward but the
# local in-place build has not been rebuilt/reinstalled yet.
# NOTE: We still require that torch is imported from pytorch-mps-fork to avoid
# accidentally testing a system/baseline PyTorch build.
ALLOW_TORCH_MISMATCH="${MPS_TESTS_ALLOW_TORCH_MISMATCH:-0}"

METAL_VISIBLE="unknown"

print_metal_diagnostics() {
    echo ""
    echo "=== MPS Diagnostics ==="
    echo "uname -m: $(uname -m)"
    echo "sw_vers:"
    sw_vers || true
    echo ""
    echo "system_profiler SPDisplaysDataType:"
    system_profiler SPDisplaysDataType 2>/dev/null | sed -n '1,120p' || true

    if command -v clang >/dev/null 2>&1; then
        local tmp_dir
        if [ -n "${TMP_ROOT:-}" ] && [ -d "${TMP_ROOT}" ]; then
            tmp_dir="$(mktemp -d "${TMP_ROOT}/metal_diag.XXXXXX")"
        else
            tmp_dir="$(mktemp -d)"
        fi
        local src="${tmp_dir}/metal_diag.m"
        local bin="${tmp_dir}/metal_diag"
        cat > "${src}" <<'EOF'
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main() {
  @autoreleasepool {
    id<MTLDevice> defaultDev = MTLCreateSystemDefaultDevice();
    if (defaultDev) {
      NSLog(@"MTLCreateSystemDefaultDevice: %@", [defaultDev name]);
    } else {
      NSLog(@"MTLCreateSystemDefaultDevice: nil");
    }
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    NSLog(@"MTLCopyAllDevices count: %lu", (unsigned long)[devices count]);
    for (id<MTLDevice> dev in devices) {
      BOOL mac2 = NO;
      if ([dev respondsToSelector:@selector(supportsFamily:)]) {
        mac2 = [dev supportsFamily:MTLGPUFamilyMac2];
      }
      NSLog(@"Device: %@ lowPower=%d headless=%d mac2=%d", [dev name], [dev isLowPower], [dev isHeadless], mac2);
    }
    id mpsCD = NSClassFromString(@"MPSGraph");
    SEL sel = @selector(HermiteanToRealFFTWithTensor:axes:descriptor:name:);
    BOOL responds = (mpsCD != nil) ? [mpsCD instancesRespondToSelector:sel] : NO;
    NSLog(@"MPSGraph class: %@", mpsCD);
    NSLog(@"MPSGraph responds to HermiteanToRealFFT...: %d", responds);
  }
  return 0;
}
EOF
        if clang -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph "${src}" -o "${bin}" 2>/dev/null; then
            echo ""
            echo "Metal framework probe (MTLCreateSystemDefaultDevice/MTLCopyAllDevices):"
            local probe_out
            probe_out="$("${bin}" 2>&1 || true)"
            echo "${probe_out}"
            if echo "${probe_out}" | grep -q "MTLCreateSystemDefaultDevice: nil" && echo "${probe_out}" | grep -q "MTLCopyAllDevices count: 0"; then
                METAL_VISIBLE="no"
            else
                METAL_VISIBLE="yes"
            fi
        else
            echo ""
            echo "Metal framework probe: skipped (compile failed)"
        fi
        rm -rf "${tmp_dir}"
    else
        echo ""
        echo "Metal framework probe: skipped (clang not found)"
    fi

    echo "======================="
    echo ""
}

# Preflight: ensure the imported torch build matches the fork HEAD.
# This repo has previously produced false-positive "tests pass" results when the
# environment imported baseline PyTorch (or a stale build) instead of the fork.
if [ -d "pytorch-mps-fork/.git" ]; then
    # torch.__version__ embeds a short git hash; force 7 chars to match.
    FORK_HEAD_SHORT="$(git -C pytorch-mps-fork rev-parse --short=7 HEAD)"
    TORCH_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)')"
    TORCH_FILE="$("${PYTHON_BIN}" -c 'import torch; print(torch.__file__)')"
    TORCH_C_FILE="$("${PYTHON_BIN}" -c 'import torch; print(torch._C.__file__)')"
    echo "Torch: $TORCH_VERSION"
    echo "Torch location: $TORCH_FILE"
    echo "Torch _C: $TORCH_C_FILE"
    echo "Fork HEAD: git$FORK_HEAD_SHORT"
    echo ""

    # Hard safety check: never allow running tests against a torch import that
    # does not come from the local fork, even if mismatch is allowed.
    if [[ "${TORCH_FILE}" != "${FORK_DIR}/torch/"* ]]; then
        echo "ERROR: Imported torch is not from ${FORK_DIR} (got: ${TORCH_FILE})."
        echo "Fix: activate the venv / adjust PYTHONPATH so torch imports from pytorch-mps-fork."
        exit 2
    fi

    if [[ "$TORCH_VERSION" != *"git${FORK_HEAD_SHORT}"* ]]; then
        echo "WARNING: Imported torch ($TORCH_VERSION) does not match pytorch-mps-fork HEAD (git$FORK_HEAD_SHORT)."
        echo "This usually means you updated the fork but did not rebuild/reinstall torch."
        if [ "${ALLOW_TORCH_MISMATCH}" != "1" ]; then
            echo ""
            echo "ERROR: Refusing to run tests against a potentially stale build."
            echo "Fix: rebuild/reinstall torch from pytorch-mps-fork, then re-run this test suite."
            echo "Suggested rebuild command:"
            echo "  (cd pytorch-mps-fork && USE_MPS=1 USE_CUDA=0 BUILD_TEST=0 ${PYTHON_BIN} -m pip install -e . -v --no-build-isolation)"
            echo "Override (unsafe): set MPS_TESTS_ALLOW_TORCH_MISMATCH=1"
            exit 2
        fi
        echo "Continuing because MPS_TESTS_ALLOW_TORCH_MISMATCH=1"
        echo ""
    fi
fi

# Preflight: ensure MPS is available before running MPS-specific tests.
if ! "${PYTHON_BIN}" - <<'PY'
import torch
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY
then
    echo "ERROR: Failed to import torch or query MPS availability." >&2
    exit 2
fi

if ! "${PYTHON_BIN}" - <<'PY'
import sys
import torch
sys.exit(0 if torch.backends.mps.is_available() else 1)
PY
then
    echo "ERROR: MPS is not available (torch.backends.mps.is_available() == False)." >&2
    OS_VERSION="$(sw_vers -productVersion 2>/dev/null || true)"
    OS_MAJOR="${OS_VERSION%%.*}"
    if [[ -n "${OS_VERSION}" && "${OS_MAJOR}" =~ ^[0-9]+$ && "${OS_MAJOR}" -ge 14 ]]; then
        echo "NOTE: sw_vers reports macOS ${OS_VERSION} (>=14.0)." >&2
        echo "      If torch claims 'MPS backend is supported on MacOS 14.0+' anyway, this usually means Metal devices are not visible to this process (sandbox/VM/headless runner)." >&2
    fi
    # Provide a more actionable root cause when possible (common in VMs / CI).
    "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import platform
import torch

print("Platform:", platform.platform())
try:
    torch.empty(1, device="mps")
    print("NOTE: Created an MPS tensor even though is_available() is False.")
except Exception as e:
    print("MPS tensor creation error:", repr(e))
    if "supported on MacOS 14.0+" in str(e):
        print("NOTE: This error can also occur when Metal devices are not visible to this process (e.g. MTLCreateSystemDefaultDevice: nil).")
PY
    print_metal_diagnostics || true
    if [ "${METAL_VISIBLE}" = "no" ]; then
        echo "Detected: Metal devices are NOT visible to this process (MTLCreateSystemDefaultDevice: nil, MTLCopyAllDevices: 0)." >&2
        echo "This is usually a sandbox/VM/headless runner limitation, not a PyTorch/MPS build issue." >&2
        echo 'Fix: run from a normal Terminal session with Metal device access (or use `run_worker.sh`, which runs Codex with `--dangerously-bypass-approvals-and-sandbox`).' >&2
        echo "Standalone check: ./tests/metal_diagnostics.sh" >&2
        if [ "${MPS_TESTS_SKIP_IF_NO_METAL:-1}" = "1" ]; then
            echo "" >&2
            echo "SKIP: Exiting 0 because Metal is not visible to this process." >&2
            echo "Set MPS_TESTS_SKIP_IF_NO_METAL=0 to treat this as a failure." >&2
            exit 0
        fi
    else
        echo "NOTE: Metal probe did not detect the common sandbox signature (MTLCreateSystemDefaultDevice: nil + 0 devices)." >&2
        echo "      If MPS is still unavailable, check OS version, device policy (e.g. MDM), and the PyTorch build configuration." >&2
    fi
    echo "This test suite must run in a macOS environment with Metal device access." >&2
    echo "If this is unexpected, verify you are not running under a restricted sandbox/VM/headless session (no MTLDevice visible)." >&2
    exit 2
fi

echo "=========================================="
echo "MPS Stream Pool - Test Suite"
echo "=========================================="
echo "Date: $(date)"
echo ""
echo "Platform:"
"${PYTHON_BIN}" tests/platform_utils.py || true
echo ""

PASS=0
FAIL=0

# Use mktemp for safe parallel execution (scoped to TMP_ROOT).
TEST_OUTPUT="$(mktemp "${TMP_ROOT}/test_output.XXXXXX")"

run_test() {
    local name=$1
    local cmd=$2
    local max_retries=${3:-1}  # Default 1 attempt, pass 2 for retry on flaky tests
    local attempt=1

    echo "--- $name ---"

    while [ $attempt -le $max_retries ]; do
        # Sync MPS and give GPU time to clean up state between tests
        # This reduces intermittent failures from MPS race conditions
        "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import gc
try:
    import torch
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
except Exception:
    pass
gc.collect()
PY
        sleep 1

        # Avoid `bash -c` here: on macOS, spawning a new `/bin/bash` process can
        # strip `DYLD_*` env vars (SIP/restricted binaries), which breaks
        # DYLD_INSERT_LIBRARIES-based injection (used by the AGX fix).
        local before_time
        before_time="$(date +%s)"
        local crash_before
        crash_before="$(count_crash_logs)"
        local cmd_ok=0

        if eval "$cmd" > "$TEST_OUTPUT" 2>&1; then
            cmd_ok=1
        fi

        # Small delay to ensure crash logs are written.
        sleep "$CRASH_WAIT_SECS"

        # Opportunistically copy new crash reports from DiagnosticReports into crash_logs/
        # (covers the case where crash_monitor.sh isn't running).
        local copied_crash_reports
        copied_crash_reports="$(capture_new_crash_reports_since "$before_time")"
        if [ "${copied_crash_reports:-0}" -gt 0 ]; then
            echo "Captured $copied_crash_reports new crash report(s) from DiagnosticReports."
        fi

        local crash_after
        crash_after="$(count_crash_logs)"
        local new_crashes=$((crash_after - crash_before))

        if [ "$new_crashes" -gt 0 ]; then
            echo "FAIL (NEW CRASHES: $new_crashes)"
            echo ""
            echo "Latest crash details:"
            print_latest_crash
            echo ""
            echo "A test that crashes is NOT passing."
            exit 1
        fi

        if [ "$cmd_ok" -eq 1 ]; then
            if grep -q "PASS\|passed\|ALL.*PASS" "$TEST_OUTPUT"; then
                echo "PASS"
                ((PASS++))
                echo ""
                return 0
            else
                echo "WARN (no PASS indicator)"
                cat "$TEST_OUTPUT"
                ((PASS++))
                echo ""
                return 0
            fi
        else
            if [ $attempt -lt $max_retries ]; then
                echo "RETRY (attempt $attempt failed, retrying...)"
                ((attempt++))
                sleep 2
            else
                echo "FAIL"
                tail -20 "$TEST_OUTPUT"
                ((FAIL++))
                echo ""
                return 1
            fi
        fi
    done
}

# Test 1: Fork safety regression (MPS must fail safely in forked child)
run_test "Fork Safety" "${PYTHON_BIN} tests/test_fork_safety.py" || true

# Test 2: Simple parallel test
run_test "Simple Parallel MPS" "${PYTHON_BIN} tests/test_parallel_mps_simple.py" || true

# Test 3: Extended stress test
run_test "Extended Stress Test" "${PYTHON_BIN} tests/test_stress_extended.py" || true

# Test 4: Thread boundary test
run_test "Thread Boundary" "${PYTHON_BIN} tests/test_thread_boundary.py" || true

# Test 5: Stream assignment
run_test "Stream Assignment" "${PYTHON_BIN} tests/test_stream_assignment.py" || true

# Test 6: Benchmark (verify no crashes)
run_test "Benchmark (nn.Linear)" "${PYTHON_BIN} tests/benchmark_parallel_mps.py --model linear" || true

# Test 7: Real models (nn.Sequential + MLP + Conv1D)
# Uses retry because Apple MPS framework has intermittent race conditions at 2+ threads
run_test "Real Models Parallel" "${PYTHON_BIN} tests/test_real_models_parallel.py" 2 || true

# Test 8: Stream pool wraparound (>31 sequential threads)
run_test "Stream Pool Wraparound" "${PYTHON_BIN} tests/test_oversubscription.py" || true

# Test 9: Thread churn
run_test "Thread Churn" "${PYTHON_BIN} tests/test_thread_churn.py" || true

# Test 10: Cross-stream tensor correctness (Phase 17 review item)
run_test "Cross-Stream Tensor" "${PYTHON_BIN} tests/test_cross_stream_tensor.py" || true

# Test 11: Linear algebra ops parallel (Phase 23 mutex validation)
run_test "Linalg Ops Parallel" "${PYTHON_BIN} tests/test_linalg_ops_parallel.py" || true

# Test 12: Large workload efficiency (validates 50%+ efficiency at 2 threads)
run_test "Large Workload Efficiency" "${PYTHON_BIN} tests/test_efficiency_large_workload.py" || true

# Test 13: Max streams stress test (31 threads - Phase 39.1)
run_test "Max Streams Stress (31t)" "${PYTHON_BIN} tests/test_max_streams_stress.py" || true

# Test 14: OOM recovery under parallel load (Phase 39.2)
run_test "OOM Recovery Parallel" "${PYTHON_BIN} tests/test_oom_recovery_parallel.py" || true

# Test 15: Graph compilation race stress (Phase 39.3)
# Note: Flaky test - occasional segfault from accumulated MPS state across test suite
run_test "Graph Compilation Stress" "${PYTHON_BIN} tests/test_graph_compilation_stress.py" 2 || true

# Test 16: Stream pool boundary - round robin wraparound (Phase 39.4)
run_test "Stream Pool: Round Robin" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_stream_pool_boundaries import test_round_robin_wraparound; test_round_robin_wraparound()\"" || true

# Test 17: Stream pool boundary - stream reuse under churn (Phase 39.4)
run_test "Stream Pool: Reuse Churn" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_stream_pool_boundaries import test_stream_reuse_under_churn; test_stream_reuse_under_churn()\"" || true

# Test 18: Stream pool boundary - sync during active use (Phase 39.4)
run_test "Stream Pool: Sync Active" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_stream_pool_boundaries import test_synchronize_all_during_active_use; test_synchronize_all_during_active_use()\"" || true

# Test 19: Static destruction - MPS state cleanup (Phase 39.5)
run_test "Static: MPS Cleanup" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_static_destruction import test_mps_state_during_cleanup; test_mps_state_during_cleanup()\"" || true

# Test 20: Static destruction - atexit handlers (Phase 39.5)
run_test "Static: Atexit" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_static_destruction import test_atexit_cleanup; test_atexit_cleanup()\"" || true

# Test 21: Static destruction - thread pool cleanup order (Phase 39.5)
# Note: Flaky test - occasional segfault during Python interpreter startup (not test logic)
run_test "Static: Thread Cleanup" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_static_destruction import test_thread_pool_cleanup_order; test_thread_pool_cleanup_order()\"" 2 || true

# Test 22: Fork safety stress - fork with active threads (Phase 39.6)
run_test "Fork: Active Threads" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_fork_safety_stress import test_fork_with_active_threads; test_fork_with_active_threads()\"" || true

# Test 23: Fork safety stress - parent continues after fork (Phase 39.6)
run_test "Fork: Parent Continues" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_fork_safety_stress import test_parent_continues_after_fork; test_parent_continues_after_fork()\"" || true

# Test 24: Fork safety stress - multiprocessing spawn (Phase 39.6)
run_test "Fork: MP Spawn" "${PYTHON_BIN} -c \"import sys; sys.path.insert(0, 'tests'); from test_fork_safety_stress import test_multiprocessing_spawn; test_multiprocessing_spawn()\"" || true

# Test 25: LayerNorm correctness + thread-consistency regression (N=1868)
run_test "LayerNorm Verification" "${PYTHON_BIN} tests/verify_layernorm_fix.py" || true

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
