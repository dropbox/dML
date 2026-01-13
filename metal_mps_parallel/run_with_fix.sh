#!/bin/bash
# Run Python with the AGX fix dylib loaded (prefers newest available).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

AGX_FIX=""
AGX_FIX_CANDIDATES=(
  "$REPO_ROOT/agx_fix/build/libagx_fix_v2_9.dylib"
  "$REPO_ROOT/agx_fix/build/libagx_fix_v2_8.dylib"
  "$REPO_ROOT/agx_fix/build/libagx_fix_v2_7.dylib"
  "$REPO_ROOT/agx_fix/build/libagx_fix_v2_6.dylib"
  "$REPO_ROOT/agx_fix/build/libagx_fix_v2_5.dylib"
)
for candidate in "${AGX_FIX_CANDIDATES[@]}"; do
  if [ -f "$candidate" ]; then
    AGX_FIX="$candidate"
    break
  fi
done

if [ -n "$AGX_FIX" ]; then
  if [ -n "${DYLD_INSERT_LIBRARIES:-}" ]; then
    if [[ "${DYLD_INSERT_LIBRARIES}" != *"libagx_fix"* ]]; then
      export DYLD_INSERT_LIBRARIES="${AGX_FIX}:${DYLD_INSERT_LIBRARIES}"
    fi
  else
    export DYLD_INSERT_LIBRARIES="$AGX_FIX"
  fi
else
  echo "WARNING: AGX fix dylib not found (build: cd agx_fix && make)" >&2
fi

# Force MPS Graph API path which is designed for thread-safety
# The non-graph path has race conditions that can cause crashes
export MPS_FORCE_GRAPH_PATH=1

# Optional: Enable verbose logging for debugging
# export AGX_FIX_VERBOSE=1

exec python3 "$@"
