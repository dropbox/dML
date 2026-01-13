#!/bin/bash
# setup_apalache.sh - Download and install Apalache for TLA+ symbolic verification
#
# Apalache is a symbolic model checker for TLA+ that uses Z3/SMT solvers.
# Unlike TLC (bounded enumeration), Apalache can verify inductive invariants
# for unbounded parameters.
#
# Usage: ./tools/setup_apalache.sh
#
# Requirements:
#   - Java 17+ (Homebrew: brew install openjdk)
#   - curl or wget
#
# After installation, run verification with:
#   ./tools/run_all_verification.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APALACHE_VERSION="0.52.1"
APALACHE_URL="https://github.com/apalache-mc/apalache/releases/download/v${APALACHE_VERSION}/apalache.tgz"
APALACHE_DIR="$SCRIPT_DIR/apalache"

echo "=== Apalache Setup Script ==="
echo "Version: $APALACHE_VERSION"
echo "Install location: $APALACHE_DIR"
echo ""

# Check Java
echo "Checking Java..."
if [ -x "/opt/homebrew/opt/openjdk/bin/java" ]; then
    JAVA_CMD="/opt/homebrew/opt/openjdk/bin/java"
elif command -v java &> /dev/null; then
    JAVA_CMD="$(command -v java)"
else
    echo "ERROR: Java not found. Install with: brew install openjdk"
    exit 1
fi

JAVA_VERSION=$($JAVA_CMD -version 2>&1 | head -1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 17 ] 2>/dev/null; then
    echo "WARNING: Java 17+ recommended (found Java $JAVA_VERSION)"
fi
echo "  Using: $JAVA_CMD (version $JAVA_VERSION)"

# Check if already installed
if [ -x "$APALACHE_DIR/bin/apalache-mc" ]; then
    CURRENT_VERSION=$("$APALACHE_DIR/bin/apalache-mc" version 2>/dev/null | head -1 || echo "unknown")
    echo ""
    echo "Apalache already installed: $CURRENT_VERSION"
    echo "To reinstall, remove $APALACHE_DIR and run again."
    exit 0
fi

# Download
echo ""
echo "Downloading Apalache v${APALACHE_VERSION}..."
TMP_FILE="$(mktemp /tmp/apalache-XXXXXX.tgz)"
trap "rm -f $TMP_FILE" EXIT

if command -v curl &> /dev/null; then
    curl -L -o "$TMP_FILE" "$APALACHE_URL"
elif command -v wget &> /dev/null; then
    wget -O "$TMP_FILE" "$APALACHE_URL"
else
    echo "ERROR: Neither curl nor wget found"
    exit 1
fi

# Extract
echo ""
echo "Extracting to $APALACHE_DIR..."
mkdir -p "$APALACHE_DIR"
tar -xzf "$TMP_FILE" -C "$APALACHE_DIR" --strip-components=1

# Verify installation
if [ -x "$APALACHE_DIR/bin/apalache-mc" ]; then
    echo ""
    echo "=== Installation Successful ==="
    # Apalache's wrapper uses 'java' from PATH, so ensure Homebrew OpenJDK is first
    APALACHE_VERSION_CHECK=$(PATH=/opt/homebrew/opt/openjdk/bin:$PATH /bin/bash "$APALACHE_DIR/bin/apalache-mc" version 2>/dev/null || echo "unknown")
    echo "Version: $APALACHE_VERSION_CHECK"
    echo ""
    echo "Apalache is ready. Run verification with:"
    echo "  ./tools/run_all_verification.sh"
    echo ""
    echo "Or run directly:"
    echo "  cd specs"
    echo "  PATH=/opt/homebrew/opt/openjdk/bin:\$PATH $APALACHE_DIR/bin/apalache-mc check --config=MPSStreamPool_Apalache.cfg MPSStreamPool.tla"
    echo ""
    echo "NOTE: Apalache requires Java in PATH. The verification script handles this automatically."
else
    echo "ERROR: Installation failed - apalache-mc not found"
    exit 1
fi
