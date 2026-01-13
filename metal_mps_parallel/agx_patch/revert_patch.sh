#!/bin/bash
# Revert AGX driver to original (unpatched) version
# REQUIRES: SIP disabled (csrutil disable in recovery mode)
#
# This script restores the original AGX driver from backup.
# Run this if the patched driver causes issues.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVER_PATH="/System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X"

echo "========================================"
echo "AGX Driver Revert Script"
echo "========================================"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (sudo)."
    echo "Usage: sudo $0 [backup_file]"
    exit 1
fi

# Check SIP status
SIP_STATUS=$(csrutil status 2>&1 || true)
if echo "$SIP_STATUS" | grep -q "enabled"; then
    echo "ERROR: System Integrity Protection (SIP) is enabled."
    echo
    echo "To revert the driver, you must disable SIP:"
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

# Find backup file
if [ -n "$1" ]; then
    BACKUP_FILE="$1"
else
    # Look for most recent backup in script directory
    BACKUP_FILE=$(ls -t "$SCRIPT_DIR"/AGXMetalG16X_backup_* 2>/dev/null | head -1)

    if [ -z "$BACKUP_FILE" ]; then
        # Try the universal original
        if [ -f "$SCRIPT_DIR/AGXMetalG16X_universal_original" ]; then
            BACKUP_FILE="$SCRIPT_DIR/AGXMetalG16X_universal_original"
        fi
    fi
fi

if [ -z "$BACKUP_FILE" ] || [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: No backup file found."
    echo
    echo "Please specify the backup file:"
    echo "  sudo $0 /path/to/AGXMetalG16X_backup_YYYYMMDD_HHMMSS"
    echo
    echo "Or use the original copy:"
    echo "  sudo $0 $SCRIPT_DIR/AGXMetalG16X_universal_original"
    exit 1
fi

echo "Backup file: $BACKUP_FILE"
echo "Driver path: $DRIVER_PATH"
echo

# Verify backup file looks like a valid Mach-O
if ! file "$BACKUP_FILE" | grep -q "Mach-O"; then
    echo "ERROR: Backup file does not appear to be a valid Mach-O binary."
    exit 1
fi

echo "Backup file is valid Mach-O."
echo

# Show checksums
echo "Checksums:"
echo "  Backup:  $(shasum -a 256 "$BACKUP_FILE" | cut -d' ' -f1)"
if [ -f "$DRIVER_PATH" ]; then
    echo "  Current: $(shasum -a 256 "$DRIVER_PATH" | cut -d' ' -f1)"
fi
echo

# Confirm
read -p "Restore original driver? This will overwrite the current driver. [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Restore
echo "Restoring original driver..."
cp "$BACKUP_FILE" "$DRIVER_PATH"
chmod 755 "$DRIVER_PATH"
chown root:wheel "$DRIVER_PATH"

echo "Clearing kernel cache..."
kextcache -invalidate / 2>/dev/null || true

echo
echo "========================================"
echo "REVERT COMPLETE"
echo "========================================"
echo
echo "The original AGX driver has been restored."
echo
echo "IMPORTANT: You must reboot for changes to take effect."
echo
echo "After reboot, you can re-enable SIP if desired:"
echo "  1. Boot to Recovery Mode"
echo "  2. Run: csrutil enable"
echo "  3. Reboot"
echo
