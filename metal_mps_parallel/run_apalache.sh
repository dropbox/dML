#!/bin/bash
# Run Apalache verification on all TLA+ specs

set -e

export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"

APALACHE="$HOME/tools/apalache/bin/apalache-mc"
SPECS_DIR="$(dirname "$0")/specs"
RESULTS_FILE="$(dirname "$0")/apalache_results.json"

echo "{"
echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
echo "  \"apalache_version\": \"$($APALACHE version 2>/dev/null || echo 'unknown')\","
echo "  \"results\": ["

first=true
for cfg in "$SPECS_DIR"/*_Apalache.cfg; do
    spec_name=$(basename "$cfg" "_Apalache.cfg")
    spec_file="$SPECS_DIR/${spec_name}.tla"

    if [ ! -f "$spec_file" ]; then
        echo "  Skipping $spec_name - no .tla file" >&2
        continue
    fi

    if [ "$first" = true ]; then
        first=false
    else
        echo ","
    fi

    echo "    {"
    echo "      \"spec\": \"$spec_name\","

    # Run Apalache with timeout of 180 seconds
    output=$(cd "$SPECS_DIR" && timeout 180 "$APALACHE" check --config="$(basename "$cfg")" "$(basename "$spec_file")" 2>&1) || true

    if echo "$output" | grep -q "Invariant.*violated"; then
        status="FAIL"
    elif echo "$output" | grep -q "PASS"; then
        status="PASS"
    elif echo "$output" | grep -q "Error\|error"; then
        status="ERROR"
    elif echo "$output" | grep -q "timeout"; then
        status="TIMEOUT"
    else
        # Check for successful completion pattern
        if echo "$output" | grep -q "state invariant.*holds"; then
            status="PASS"
        else
            status="UNKNOWN"
        fi
    fi

    echo "      \"status\": \"$status\""
    echo -n "    }"
done

echo ""
echo "  ]"
echo "}"
