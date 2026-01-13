#!/bin/bash
# Deploy patched AGX driver
# REQUIRES: SIP disabled (csrutil disable in recovery mode)
#
# This script:
# 1. Verifies SIP is disabled
# 2. Backs up the original driver
# 3. Deploys the patched driver
# 4. Clears kernel cache
#
# After running, you must REBOOT for changes to take effect.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVER_PATH="/System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X"
PATCHED_FILE="$SCRIPT_DIR/AGXMetalG16X_universal_patched"
BACKUP_FILE="$SCRIPT_DIR/AGXMetalG16X_backup_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "AGX Driver Patch Deployment"
echo "========================================"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (sudo)."
    echo "Usage: sudo $0"
    exit 1
fi

# Check SIP status
SIP_STATUS=$(csrutil status 2>&1 || true)
if echo "$SIP_STATUS" | grep -q "enabled"; then
    echo "ERROR: System Integrity Protection (SIP) is enabled."
    echo
    echo "To deploy the patch, you must disable SIP:"
    echo "  1. Shut down your Mac"
    echo "  2. Hold the power button until 'Loading startup options' appears"
    echo "  3. Select Options → Continue"
    echo "  4. Open Terminal from Utilities menu"
    echo "  5. Run: csrutil disable"
    echo "  6. Reboot and run this script again"
    exit 1
fi

echo "SIP Status: DISABLED (OK)"
echo

# Check patched file exists
if [ ! -f "$PATCHED_FILE" ]; then
    echo "ERROR: Patched driver not found at: $PATCHED_FILE"
    echo
    echo "Run create_patch.py first to generate the patched binary:"
    echo "  python3 create_patch.py --input AGXMetalG16X_universal_original --output AGXMetalG16X_universal_patched"
    exit 1
fi

echo "Patched file: $PATCHED_FILE"
echo "Driver path:  $DRIVER_PATH"
echo

# Verify patched file is valid Mach-O
if ! file "$PATCHED_FILE" | grep -q "Mach-O"; then
    echo "ERROR: Patched file does not appear to be a valid Mach-O binary."
    exit 1
fi

# Show checksums
echo "Checksums:"
PATCHED_SHA=$(shasum -a 256 "$PATCHED_FILE" | cut -d' ' -f1)
CURRENT_SHA=$(shasum -a 256 "$DRIVER_PATH" | cut -d' ' -f1)
echo "  Patched: $PATCHED_SHA"
echo "  Current: $CURRENT_SHA"

# Expected checksums (update these if patch changes)
EXPECTED_PATCHED="3b6813011e481cea46dd2942b966bdc48712d9adcd1a1b836f6710ecb1c3fb0d"
EXPECTED_ORIGINAL="fbd62445e186aeee071e65a4baf6a6da5947ca73a0cd65a9f53a6c269f32d345"

if [ "$PATCHED_SHA" != "$EXPECTED_PATCHED" ]; then
    echo
    echo "WARNING: Patched file checksum does not match expected value."
    echo "  Expected: $EXPECTED_PATCHED"
    echo "  Got:      $PATCHED_SHA"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo

# Check if already patched
if [ "$CURRENT_SHA" = "$PATCHED_SHA" ]; then
    echo "Driver appears to already be patched (checksums match)."
    read -p "Re-deploy anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Confirm deployment
echo "This will:"
echo "  1. Backup current driver to: $BACKUP_FILE"
echo "  2. Replace driver with patched version"
echo "  3. Clear kernel cache"
echo
echo "IMPORTANT: A reboot is required after deployment."
echo
read -p "Proceed with deployment? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Backup original
echo "Backing up current driver..."
cp "$DRIVER_PATH" "$BACKUP_FILE"
echo "Backup saved to: $BACKUP_FILE"

# Deploy patch
echo "Deploying patched driver..."
cp "$PATCHED_FILE" "$DRIVER_PATH"
chmod 755 "$DRIVER_PATH"
chown root:wheel "$DRIVER_PATH"

# Clear kernel cache
echo "Clearing kernel cache..."
kextcache -invalidate / 2>/dev/null || true

echo
echo "========================================"
echo "DEPLOYMENT COMPLETE"
echo "========================================"
echo
echo "Backup saved to: $BACKUP_FILE"
echo
echo "IMPORTANT: You must REBOOT for the patch to take effect."
echo
echo "After reboot, test with:"
echo "  python3 tests/verify_patch.py"
echo
echo "To revert if there are problems:"
echo "  sudo ./revert_patch.sh"
echo
