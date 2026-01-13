#!/bin/bash
# run_all_verification.sh - Run complete verification suite
# Usage: ./tools/run_all_verification.sh

set -euo pipefail

JAVA_HOME_DEFAULT="/opt/homebrew/opt/openjdk"
export JAVA_HOME="${JAVA_HOME:-$JAVA_HOME_DEFAULT}"
if [ -x "$JAVA_HOME/bin/java" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== MPS Parallel Inference Verification Suite ==="
echo "Project: $PROJECT_DIR"
echo "Java: $JAVA_HOME"
echo ""

# Check tools
echo "=== Checking Tools ==="
echo -n "Apalache: "
/bin/bash "$SCRIPT_DIR/apalache/bin/apalache-mc" version 2>/dev/null || echo "NOT FOUND"

echo -n "TLC: "
if [ -x "$JAVA_HOME/bin/java" ] && [ -f "$SCRIPT_DIR/tla2tools.jar" ]; then
    # The jar may exit non-zero when invoked without args; we only care that it exists.
    $JAVA_HOME/bin/java -jar "$SCRIPT_DIR/tla2tools.jar" 2>&1 | sed -n '1p' || true
else
    echo "NOT FOUND"
fi

echo -n "Coq: "
coqc --version 2>/dev/null | head -1 || echo "NOT FOUND"

echo -n "CBMC: "
cbmc --version 2>/dev/null | head -1 || echo "NOT FOUND"

echo -n "Lean 4: "
lake --version 2>/dev/null | head -1 || echo "NOT FOUND"

echo ""

# Track Apalache availability for later
APALACHE_AVAILABLE=false
if [ -x "$SCRIPT_DIR/apalache/bin/apalache-mc" ]; then
    APALACHE_AVAILABLE=true
fi

# TLA+ with TLC (bounded)
echo "=== TLA+ Bounded Verification (TLC) ==="
cd "$PROJECT_DIR/specs"
for cfg in *.cfg; do
    spec="${cfg%.cfg}.tla"
    if [ -f "$spec" ]; then
        if [ "$cfg" = "MPSStreamPoolParallel.cfg" ]; then
            echo "  Skipping $spec (expected-fail existence check: $cfg)"
            echo ""
            continue
        fi
        echo "  Checking $spec..."
        $JAVA_HOME/bin/java -jar "$SCRIPT_DIR/tla2tools.jar" \
            -deadlock -config "$cfg" "$spec" 2>&1 | tail -5
        echo ""
    fi
done

echo "=== TLA+ Expected-Fail Witness (TLC) ==="
if [ -f "MPSStreamPoolParallel.cfg" ] && [ -f "MPSStreamPoolParallel.tla" ]; then
    echo "  Checking MPSStreamPoolParallel.tla (expected invariant violation)..."
    if $JAVA_HOME/bin/java -jar "$SCRIPT_DIR/tla2tools.jar" \
        -deadlock -config "MPSStreamPoolParallel.cfg" "MPSStreamPoolParallel.tla" 2>&1 | tail -5; then
        echo "ERROR: expected TLC failure for MPSStreamPoolParallel (NoParallelEver), but exit code was 0"
        exit 1
    fi
    echo ""
fi

# Apalache Symbolic Verification (if available)
if [ "$APALACHE_AVAILABLE" = true ]; then
    echo "=== Apalache Symbolic TLA+ Verification ==="
    cd "$PROJECT_DIR/specs"
    for apalache_cfg in *_Apalache.cfg; do
        if [ -f "$apalache_cfg" ]; then
            # Extract base spec name (e.g., MPSStreamPool_Apalache.cfg -> MPSStreamPool.tla)
            spec_base="${apalache_cfg%_Apalache.cfg}.tla"
            if [ -f "$spec_base" ]; then
                echo "  Checking $spec_base with Apalache (config: $apalache_cfg)..."
                /bin/bash "$SCRIPT_DIR/apalache/bin/apalache-mc" check \
                    --config="$apalache_cfg" "$spec_base" 2>&1 | tail -5
                echo ""
            fi
        fi
    done
else
    echo "=== Apalache Symbolic TLA+ Verification ==="
    echo "  Skipped (Apalache not installed)"
    echo "  To install, run: ./tools/setup_apalache.sh"
    echo ""
fi

# Iris/Coq
echo "=== Iris/Coq Proofs ==="
if [ -d "$PROJECT_DIR/verification/iris" ]; then
    cd "$PROJECT_DIR/verification/iris"
    if [ -f "Makefile" ]; then
        make 2>&1 | tail -10
    else
        echo "No Makefile found, checking individual files..."
        for v in theories/*.v; do
            echo "  Checking $v..."
            coqc -Q theories MPS "$v" 2>&1 | tail -3
        done
    fi
else
    echo "Iris proofs directory not found"
fi

# CBMC
echo "=== CBMC Harnesses ==="
CBMC_DIR="$PROJECT_DIR/mps-verify/verification/cbmc/harnesses"
if [ -d "$CBMC_DIR" ]; then
    cd "$CBMC_DIR"
    for harness in *_harness.c; do
        if [ -f "$harness" ]; then
            echo "  Checking $harness..."
            cbmc --unwind 11 "$harness" 2>&1 | tail -3
        fi
    done
else
    echo "CBMC harnesses directory not found"
fi

# Lean 4 Proofs
echo "=== Lean 4 Proofs ==="
# mps-verify uses lakefile.toml (newer lake format)
if [ -d "$PROJECT_DIR/mps-verify" ] && [ -f "$PROJECT_DIR/mps-verify/lakefile.toml" ]; then
    cd "$PROJECT_DIR/mps-verify"
    if command -v lake >/dev/null 2>&1; then
        echo "  Building mps-verify..."
        lake build 2>&1 | tail -5
    else
        echo "  Skipped (lake not found)"
    fi
else
    echo "  Skipped (mps-verify directory not found)"
fi

echo ""
echo "=== Verification Complete ==="
