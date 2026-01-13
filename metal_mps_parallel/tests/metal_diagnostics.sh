#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./tests/metal_diagnostics.sh [--check] [--full]

Options:
  --check   Exit 0 if Metal devices are visible, 1 if not visible, 2 if unknown
            (defaults to minimal output; see --full)
  --full    Print full diagnostics (sw_vers, system_profiler) even in --check mode
EOF
}

CHECK_MODE=0
FULL_OUTPUT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --check)
      CHECK_MODE=1
      FULL_OUTPUT=0
      ;;
    --full) FULL_OUTPUT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
REPO_ROOT="$(pwd)"

if [ "${FULL_OUTPUT}" -eq 1 ]; then
  echo ""
  echo "=== Metal Diagnostics ==="
  echo "Date: $(date)"
  echo "uname -m: $(uname -m)"
  echo ""

  echo "sw_vers:"
  sw_vers || true
  echo ""

  echo "system_profiler SPDisplaysDataType:"
  system_profiler SPDisplaysDataType 2>/dev/null | sed -n '1,120p' || true
  echo ""
fi

if ! command -v clang >/dev/null 2>&1; then
  echo "Metal framework probe: skipped (clang not found)"
  if [ "${CHECK_MODE}" -eq 1 ]; then
    exit 2
  fi
  exit 0
fi

tmp_dir="$(mktemp -d "${REPO_ROOT}/.metal_diag_tmp.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT

src="${tmp_dir}/metal_diag.m"
bin="${tmp_dir}/metal_diag"

cat > "${src}" <<'EOF'
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
int main() {
  @autoreleasepool {
    id<MTLDevice> defaultDev = MTLCreateSystemDefaultDevice();
    if (defaultDev) {
      printf("MTLCreateSystemDefaultDevice: %s\n", [[defaultDev name] UTF8String]);
    } else {
      printf("MTLCreateSystemDefaultDevice: nil\n");
    }
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    printf("MTLCopyAllDevices count: %lu\n", (unsigned long)[devices count]);
    for (id<MTLDevice> dev in devices) {
      BOOL mac2 = NO;
      if ([dev respondsToSelector:@selector(supportsFamily:)]) {
        mac2 = [dev supportsFamily:MTLGPUFamilyMac2];
      }
      printf(
          "Device: %s lowPower=%d headless=%d mac2=%d\n",
          [[dev name] UTF8String],
          [dev isLowPower],
          [dev isHeadless],
          mac2);
    }
  }
  return 0;
}
EOF

if [ "${FULL_OUTPUT}" -eq 1 ]; then
  echo "Metal framework probe (MTLCreateSystemDefaultDevice/MTLCopyAllDevices):"
fi
METAL_VISIBLE="unknown"
if clang -fobjc-arc -framework Foundation -framework Metal "${src}" -o "${bin}" 2>/dev/null; then
  probe_out="$("${bin}" 2>&1 || true)"
  echo "${probe_out}"
  if echo "${probe_out}" | grep -q "MTLCreateSystemDefaultDevice: nil" && echo "${probe_out}" | grep -q "MTLCopyAllDevices count: 0"; then
    METAL_VISIBLE="no"
  else
    METAL_VISIBLE="yes"
  fi
else
  echo "Metal framework probe: skipped (compile failed)"
  METAL_VISIBLE="unknown"
fi

if [ "${FULL_OUTPUT}" -eq 1 ]; then
  echo ""

  cat <<'EOF'
NOTE:
- If `MTLCreateSystemDefaultDevice: nil` and `MTLCopyAllDevices count: 0`, Metal devices are not visible to this process.
- This commonly happens under sandboxed/VM/headless runners. Run from a normal Terminal session with Metal device access.
- For this repo's autonomous worker loop, see `run_worker.sh` (Codex runs with `--dangerously-bypass-approvals-and-sandbox` to allow Metal/MPS access).
EOF
fi

if [ "${CHECK_MODE}" -eq 1 ]; then
  case "${METAL_VISIBLE}" in
    yes) exit 0 ;;
    no) exit 1 ;;
    *) exit 2 ;;
  esac
fi
