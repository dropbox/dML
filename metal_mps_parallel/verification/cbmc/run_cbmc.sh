#!/bin/bash
# CBMC Verification Runner for MPS Parallel Models
#
# Usage:
#   ./run_cbmc.sh [model_name] [options]
#
# Models:
#   batch_queue - MPSBatchQueue concurrent verification (default)
#   stream_pool - MPSStreamPool TLS binding verification
#   allocator   - MPSAllocator ABA detection verification
#   event       - MPSEvent callback survival verification
#   all         - Run all models
#
# Options:
#   --unwind N  - Set loop unwinding bound (default: 15)
#   --trace     - Show counterexample trace on failure
#   --verbose   - Verbose CBMC output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-batch_queue}"
shift 2>/dev/null || true

# Defaults
UNWIND=15
TRACE=""
VERBOSE=""

# Parse options
while [[ $# -gt 0 ]]; do
  case $1 in
    --unwind)
      UNWIND="$2"
      shift 2
      ;;
    --trace)
      TRACE="--trace"
      shift
      ;;
    --verbose)
      VERBOSE="--verbosity 9"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

run_model() {
  local MODEL_NAME="$1"
  local HARNESS=""

  echo "=== CBMC Verification: $MODEL_NAME ==="
  echo "Configuration:"
  echo "  Unwind bound: $UNWIND"
  echo ""

  case "$MODEL_NAME" in
    batch_queue)
      HARNESS="batch_queue_c_model.c"
      ;;
    stream_pool)
      HARNESS="stream_pool_c_model.c"
      ;;
    allocator)
      HARNESS="allocator_c_model.c"
      ;;
    event)
      HARNESS="event_c_model.c"
      ;;
    *)
      echo "Unknown model: $MODEL_NAME"
      echo "Available: batch_queue, stream_pool, allocator, event, all"
      return 1
      ;;
  esac

  if [[ ! -f "$HARNESS" ]]; then
    echo "ERROR: Harness file not found: $HARNESS"
    return 1
  fi

  echo "Running: cbmc $HARNESS"
  echo ""

  # Run CBMC
  set +e
  cbmc "$HARNESS" \
    --unwind "$UNWIND" \
    --bounds-check \
    --pointer-check \
    --div-by-zero-check \
    --slice-formula \
    $TRACE \
    $VERBOSE \
    2>&1

  local RESULT=$?
  set -e

  echo ""
  if [[ $RESULT -eq 0 ]]; then
    echo "VERIFICATION SUCCESSFUL: $MODEL_NAME"
  else
    echo "VERIFICATION FAILED: $MODEL_NAME (exit code: $RESULT)"
  fi
  echo ""

  return $RESULT
}

if [[ "$MODEL" == "all" ]]; then
  FAILED=0

  for m in batch_queue stream_pool allocator event; do
    if ! run_model "$m"; then
      FAILED=1
    fi
  done

  echo "=========================================="
  if [[ $FAILED -eq 0 ]]; then
    echo "ALL MODELS VERIFIED SUCCESSFULLY"
  else
    echo "SOME MODELS FAILED VERIFICATION"
  fi
  exit $FAILED
else
  run_model "$MODEL"
fi
