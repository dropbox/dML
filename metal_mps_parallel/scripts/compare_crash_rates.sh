#!/bin/bash
# Compare crash rates with and without the AGX fix dylib.
#
# WARNING: This script intentionally measures crash rates and may generate
# crash logs under `crash_logs/` (and in macOS DiagnosticReports). Run only
# when you're prepared to handle/clean crash artifacts.
#
# Defaults:
# - Uses MPSGraph path (`MPS_FORCE_GRAPH_PATH=1`) unless overridden.
# - Chooses the newest available `libagx_fix_*.dylib` unless `AGX_FIX_DYLIB` is set.

N=${1:-20}

export MPS_FORCE_GRAPH_PATH="${MPS_FORCE_GRAPH_PATH:-1}"

AGX_FIX_DYLIB_EFFECTIVE="${AGX_FIX_DYLIB:-}"
if [ -z "${AGX_FIX_DYLIB_EFFECTIVE}" ]; then
    AGX_FIX_CANDIDATES=(
        "agx_fix/build/libagx_fix_v2_9.dylib"
        "agx_fix/build/libagx_fix_v2_8.dylib"
        "agx_fix/build/libagx_fix_v2_7.dylib"
        "agx_fix/build/libagx_fix_v2_5.dylib"
        "agx_fix/build/libagx_fix_v2_3.dylib"
        "agx_fix/build/libagx_fix_v2.dylib"
        "agx_fix/build/libagx_fix.dylib"
    )
    for candidate in "${AGX_FIX_CANDIDATES[@]}"; do
        if [ -f "${candidate}" ]; then
            AGX_FIX_DYLIB_EFFECTIVE="${candidate}"
            break
        fi
    done
fi

if [ -z "${AGX_FIX_DYLIB_EFFECTIVE}" ]; then
    echo "ERROR: No AGX fix dylib found. Build first: (cd agx_fix && make)" >&2
    exit 2
fi

echo "=== Testing WITHOUT AGX fix ($N runs) ==="
no_fix_pass=0
no_fix_fail=0
for i in $(seq 1 $N); do
    if MPS_SUPPRESS_AGX_FIX_WARNING=1 python3 scripts/test_transformer_threads.py > /dev/null 2>&1; then
        no_fix_pass=$((no_fix_pass + 1))
        echo -n "."
    else
        no_fix_fail=$((no_fix_fail + 1))
        echo -n "X"
    fi
done
echo ""
echo "Without fix: $no_fix_pass passed, $no_fix_fail crashed ($(echo "scale=1; $no_fix_pass * 100 / $N" | bc)% pass rate)"
echo ""

echo "=== Testing WITH AGX fix ($(basename "${AGX_FIX_DYLIB_EFFECTIVE}")) ($N runs) ==="
with_fix_pass=0
with_fix_fail=0
for i in $(seq 1 $N); do
    if DYLD_INSERT_LIBRARIES="${AGX_FIX_DYLIB_EFFECTIVE}" python3 scripts/test_transformer_threads.py > /dev/null 2>&1; then
        with_fix_pass=$((with_fix_pass + 1))
        echo -n "."
    else
        with_fix_fail=$((with_fix_fail + 1))
        echo -n "X"
    fi
done
echo ""
echo "With AGX fix: $with_fix_pass passed, $with_fix_fail crashed ($(echo "scale=1; $with_fix_pass * 100 / $N" | bc)% pass rate)"
echo ""

echo "=== SUMMARY ==="
echo "WITHOUT fix: $(echo "scale=1; $no_fix_pass * 100 / $N" | bc)% pass rate"
echo "WITH fix:    $(echo "scale=1; $with_fix_pass * 100 / $N" | bc)% pass rate"
