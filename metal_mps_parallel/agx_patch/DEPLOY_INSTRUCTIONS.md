# AGX Driver Patch Deployment Instructions

**Created by Andrew Yates**

This document provides step-by-step instructions to deploy and revert the AGX driver binary patch.

---

## Quick Reference

| Action | Command |
|--------|---------|
| Verify patch | `python3 create_patch.py --verify` |
| Deploy patch | `sudo ./deploy_patch.sh` |
| Revert patch | `sudo ./revert_patch.sh` |
| Test patch | `python3 ../tests/verify_patch.py` |

---

## Pre-Deployment Checklist

- [ ] macOS 15.x on Apple Silicon (M4 Max verified)
- [ ] `AGXMetalG16X_universal_patched` exists and has correct checksum
- [ ] Understand the risks (see Risks section below)
- [ ] Know how to boot to Recovery Mode

### Verify Patch Checksum

```bash
cd agx_patch
shasum -a 256 AGXMetalG16X_universal_patched
# Expected: 3b6813011e481cea46dd2942b966bdc48712d9adcd1a1b836f6710ecb1c3fb0d
```

### Verify Patch Correctness

```bash
python3 create_patch.py --verify
# All 9 locations should show "OK"
```

---

## Step 1: Disable SIP

System Integrity Protection must be disabled to modify system drivers.

1. **Shut down** your Mac completely
2. **Press and hold** the power button until "Loading startup options" appears
3. Click **Options** → **Continue**
4. Select your user account and enter password if prompted
5. From the menu bar, select **Utilities** → **Terminal**
6. Run:
   ```bash
   csrutil disable
   ```
7. Type `reboot` or select **Apple menu** → **Restart**

After reboot, verify SIP is disabled:
```bash
csrutil status
# Should show: System Integrity Protection status: disabled
```

---

## Step 2: Deploy the Patch

```bash
cd ~/metal_mps_parallel/agx_patch
sudo ./deploy_patch.sh
```

The script will:
1. Verify SIP is disabled
2. Check the patched binary exists and is valid
3. Backup the current driver to `AGXMetalG16X_backup_YYYYMMDD_HHMMSS`
4. Copy the patched driver to the system location
5. Set correct permissions (755, root:wheel)
6. Clear the kernel cache

**IMPORTANT: Reboot after deployment.**

---

## Step 3: Verify the Patch Works

After reboot:

```bash
cd ~/metal_mps_parallel

# Test WITHOUT the method swizzling workaround
# This proves the binary patch works on its own
python3 tests/verify_patch.py
```

Expected output:
- 0% crash rate
- 50%+ efficiency at 8 threads (goal)

---

## Reverting the Patch

If something goes wrong, revert to the original driver:

### Option A: From Normal Boot (if system boots)

```bash
cd ~/metal_mps_parallel/agx_patch
sudo ./revert_patch.sh
# Reboot after completion
```

### Option B: From Recovery Mode (if system won't boot)

1. **Shut down** your Mac (force shutdown if necessary: hold power 10 seconds)
2. **Boot to Recovery Mode** (hold power button until options appear)
3. Select **Options** → **Continue**
4. Open **Utilities** → **Terminal**
5. Mount your main volume:
   ```bash
   diskutil list
   # Find your main volume (usually disk3s1 or similar)
   diskutil mount disk3s1
   ```
6. Restore the backup:
   ```bash
   # Find your backup
   ls /Volumes/Macintosh\ HD/Users/*/metal_mps_parallel/agx_patch/AGXMetalG16X_backup_*

   # Copy it back
   cp /Volumes/Macintosh\ HD/Users/ayates/metal_mps_parallel/agx_patch/AGXMetalG16X_backup_* \
      /Volumes/Macintosh\ HD/System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X

   # Clear kernel cache
   kextcache -invalidate /Volumes/Macintosh\ HD/
   ```
7. Reboot

### Option C: Reinstall macOS (last resort)

If you cannot recover:
1. Boot to Recovery Mode
2. Select **Reinstall macOS**
3. This preserves your data but reinstalls system files

---

## Risks

### What Could Go Wrong

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| System won't boot | Low | Recovery Mode restore |
| Graphics glitches | Low | Revert patch |
| Kernel panic | Very Low | Recovery Mode restore |
| Data loss | None | Patch doesn't touch user data |

### Why This Is Relatively Safe

1. **Only one file is modified** - the AGX GPU driver
2. **Backup is automatic** - original driver is saved before patching
3. **Recovery Mode always works** - can restore from there
4. **macOS reinstall preserves data** - worst case is reinstalling macOS

### What the Patch Does NOT Do

- Does NOT modify the kernel
- Does NOT modify other system files
- Does NOT require disabling other security features
- Does NOT persist across macOS updates (update will restore original)

---

## Re-Enabling SIP

After testing, you may want to re-enable SIP:

1. Boot to Recovery Mode
2. Open Terminal
3. Run:
   ```bash
   csrutil enable
   ```
4. Reboot

**Note:** The patch will continue to work with SIP enabled. SIP only protects against *new* modifications. Once deployed, the patched driver runs normally.

---

## Checksums

| File | SHA256 |
|------|--------|
| AGXMetalG16X_universal_original | `fbd62445e186aeee071e65a4baf6a6da5947ca73a0cd65a9f53a6c269f32d345` |
| AGXMetalG16X_universal_patched | `3b6813011e481cea46dd2942b966bdc48712d9adcd1a1b836f6710ecb1c3fb0d` |
| AGXMetalG16X_arm64e | `3f9c4e77f09ed624dbfe288a25f3bb717c067b5fd44e09994938fec496bf6601` |
| AGXMetalG16X_patched | `db8c76d46bb6d6053055b2cb26ffffba6e8d5874af45906122b3b0d819734409` |

---

## Technical Details

### Patch Locations (9 instructions modified)

| Address | Original | Patched | Purpose |
|---------|----------|---------|---------|
| 0x2be05c | b.hi 0x2be07c | b.hi 0x2be080 | Redirect Path 2 |
| 0x2be070 | add x0, x25, x21 | str xzr, [x19, x24] | Path 1: NULL first |
| 0x2be074 | bl unlock | add x0, x25, x21 | Path 1: Move add |
| 0x2be078 | b 0x2be08c | bl unlock | Path 1: Unlock |
| 0x2be07c | add x0, x25, x21 | b 0x2be090 | Path 1: To epilogue |
| 0x2be080 | bl unlock | str xzr, [x19, x24] | Path 2: NULL first |
| 0x2be084 | mov x0, x20 | add x0, x25, x21 | Path 2: Prep lock |
| 0x2be088 | bl free | bl unlock | Path 2: Unlock |
| 0x2be08c | str xzr | b 0x2be090 | Path 2: To epilogue |

### Known Limitation

Path 2 (freelist full) skips the `free()` call due to binary space constraints. This causes a small memory leak when the freelist is full (rare condition). For production use without this leak, use the method swizzling approach (`libagx_fix.dylib`).

---

## Support

If you encounter issues:

1. Check `reports/main/` for detailed technical documentation
2. Run `python3 create_patch.py --verify` to check patch integrity
3. Use `./revert_patch.sh` to restore original driver
