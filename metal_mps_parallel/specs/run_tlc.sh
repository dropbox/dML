#!/bin/bash
# Run TLC model checker on MPS specifications
# Usage: ./run_tlc.sh <spec_name>
# Example: ./run_tlc.sh MPSStreamPool

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TLA_TOOLS="$SCRIPT_DIR/tools/tla2tools.jar"

# Configure Java
export JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home -v 21 2>/dev/null || echo /opt/homebrew/opt/openjdk@21)}"
export PATH="$JAVA_HOME/bin:$PATH"

if [ ! -f "$TLA_TOOLS" ]; then
    echo "ERROR: TLA+ tools not found at $TLA_TOOLS"
    echo "Download from: https://github.com/tlaplus/tlaplus/releases"
    exit 1
fi

SPEC_NAME="${1:-MPSStreamPool}"
SPEC_FILE="$SCRIPT_DIR/${SPEC_NAME}.tla"
CFG_FILE="$SCRIPT_DIR/${SPEC_NAME}.cfg"

if [ ! -f "$SPEC_FILE" ]; then
    echo "ERROR: Specification not found: $SPEC_FILE"
    exit 1
fi

if [ ! -f "$CFG_FILE" ]; then
    echo "ERROR: Configuration not found: $CFG_FILE"
    exit 1
fi

echo "=== TLC Model Checker ==="
echo "Specification: $SPEC_NAME"
echo "Java: $(java -version 2>&1 | head -1)"
echo ""

# Run TLC with:
# -workers auto: Use all available cores
# -deadlock: Check for deadlocks
# -cleanup: Clean up after run
cd "$SCRIPT_DIR"
java -XX:+UseParallelGC -Xmx4g -cp "$TLA_TOOLS" tlc2.TLC \
    -workers auto \
    -deadlock \
    -cleanup \
    "$SPEC_NAME"

echo ""
echo "=== TLC Complete ==="
