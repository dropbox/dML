# AGX Binary Patch for Race Condition Fix

**Created by Andrew Yates**

This directory contains tools to directly patch the Apple AGX GPU driver to fix
a race condition that causes crashes during multi-threaded Metal compute operations.

## Background

The AGXMetalG16X driver (used on M4 and similar Apple Silicon GPUs) has a race
condition in `-[AGXG16XFamilyComputeContext destroyImpl]` where the `self->_impl`
pointer is NULLed AFTER the lock is released. This creates a window where another
thread can read the pointer while it's being invalidated, causing crashes at
offsets 0x5c8, 0x98, and 0x184.

## The Bug

```
VA 0x2be070: add x0, x25, x21     ; prepare lock address
VA 0x2be074: bl unlock            ; UNLOCK first
VA 0x2be078: b 0x2be08c           ; jump to str xzr
...
VA 0x2be08c: str xzr, [x19, x24]  ; NULL _impl AFTER unlock <- BUG
```

The race window exists between `bl unlock` and `str xzr`.

## The Fix

Move the `str xzr` instruction to execute BEFORE the unlock:

```
VA 0x2be070: str xzr, [x19, x24]  ; NULL _impl FIRST (fixed!)
VA 0x2be074: add x0, x25, x21     ; prepare lock address
VA 0x2be078: bl unlock            ; UNLOCK after NULL
VA 0x2be07c: b 0x2be090           ; skip to epilogue
```

## Files

- `create_patch.py` - Python script that creates the patched binary
- `AGXMetalG16X_arm64e` - Fat Mach-O containing only the arm64e slice (used as the default input)
- `AGXMetalG16X_patched` - Patched output when using the default input
- `AGXMetalG16X_universal_original` - Universal (x86_64 + arm64e) copy of the driver
- `AGXMetalG16X_universal_patched` - Universal patched output (recommended if replacing the system driver)

## Usage

### 1. Verify Patch Locations

```bash
python3 create_patch.py --verify
```

This checks that the expected instructions are present at the patch locations.

### 2. Create Patched Binary

```bash
python3 create_patch.py
```

This creates `AGXMetalG16X_patched` with the race condition fixed.

To patch a universal (x86_64 + arm64e) driver while preserving the x86_64 slice:

```bash
python3 create_patch.py --input AGXMetalG16X_universal_original --output AGXMetalG16X_universal_patched
```

### 3. Apply to System (REQUIRES SIP DISABLED)

**WARNING**: Modifying system drivers is dangerous and requires disabling SIP.

```bash
# 1. Boot into Recovery Mode (hold power button on Apple Silicon)
# 2. Open Terminal and run:
csrutil disable

# 3. Reboot normally

# 4. Create a patched copy of the *universal* driver (recommended)
python3 create_patch.py \
  --input /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X \
  --output AGXMetalG16X_universal_patched

# 5. Backup original driver
sudo cp /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X \
        ~/AGXMetalG16X_backup

# 6. Copy patched binary
sudo cp AGXMetalG16X_universal_patched \
        /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X

# 7. Fix permissions
sudo chmod 755 /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X
sudo chown root:wheel /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X

# 8. Clear kext cache and reboot
sudo kextcache -invalidate /

# 9. Reboot and test
```

## Patch Details

### Path 1 (Freelist Not Full - Common Case)

Original:
```
0x2be070: add x0, x25, x21      ; prep lock addr
0x2be074: bl unlock             ; UNLOCK
0x2be078: b 0x2be08c            ; to shared str xzr
```

Patched:
```
0x2be070: str xzr, [x19, x24]   ; NULL FIRST
0x2be074: add x0, x25, x21      ; prep lock addr
0x2be078: bl unlock             ; UNLOCK
0x2be07c: b 0x2be090            ; to epilogue
```

### Path 2 (Freelist Full - Rare Case)

Original:
```
0x2be07c: add x0, x25, x21      ; prep lock addr
0x2be080: bl unlock             ; UNLOCK
0x2be084: mov x0, x20           ; prep for free
0x2be088: bl free               ; free the impl
0x2be08c: str xzr, [x19, x24]   ; NULL AFTER unlock <- BUG
```

Patched:
```
0x2be080: str xzr, [x19, x24]   ; NULL FIRST
0x2be084: add x0, x25, x21      ; prep lock addr
0x2be088: bl unlock             ; UNLOCK
0x2be08c: b 0x2be090            ; to epilogue (SKIPS FREE!)
```

**Limitation**: Path 2 skips the `free()` call due to space constraints.
This causes a memory leak when the freelist is full (rare). For a complete
fix without memory leak, use the runtime dylib approach.

## ⚠️ Known Limitation: Memory Leak in Path 2

**The binary patch has a memory leak when the freelist is full (rare).**

Due to space constraints in the binary, PATH 2 (freelist full) cannot fit the
`free()` call after the race fix. The patch trades a small memory leak for
crash prevention:

- **PATH 1 (common)**: Fully fixed, no leak
- **PATH 2 (rare)**: Fixed for race, leaks one impl object per destroyImpl call

This leak only occurs when:
1. The internal freelist is full (count > 7)
2. A compute context is being destroyed

In practice, this is extremely rare. The freelist is almost always below
capacity, so PATH 2 is seldom executed.

### Why No Code Cave?

A "code cave" (unused space in the binary) could hold a trampoline for the
`free()` call. However:
- The `__text` section has **zero** padding or NOP sleds
- All functions are tightly packed (function N ends, function N+1 begins)
- No inter-segment padding available

The binary was compiled with maximum optimization and no debug padding.

## Recommended: Runtime Dylib Fix

For a **complete fix** without memory leak or SIP modification, use the
runtime dylib injection approach:

```bash
DYLD_INSERT_LIBRARIES=/path/to/libagx_fix.dylib your_app
```

This intercepts the destroyImpl method at runtime and properly fixes both
paths without any space constraints.

See `agx_fix/src/agx_fix.mm` for the implementation.

## Verification

After patching, verify the fix with:

```bash
otool -tv AGXMetalG16X_patched | grep -A10 0x2be070
```

Expected output should show `str xzr` before `bl unlock`.

## Reverting

To revert to the original driver:

```bash
sudo cp ~/AGXMetalG16X_backup \
        /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X
sudo kextcache -invalidate /
# Reboot
```

## Technical Notes

- Slice offset depends on the input file format:
  - `AGXMetalG16X_arm64e` (fat, arm64e-only): arm64e slice offset 0x4000
  - `AGXMetalG16X_universal_original` (fat, x86_64 + arm64e): arm64e slice offset 0xA64000
- Bug instruction at VA 0x2be08c, file offset = slice_offset + 0x2be08c (0x2c208c or 0xD2208C)
- Instruction encoding for `str xzr, [x19, x24]`: 0xf8386a7f (little-endian: 7f 6a 38 f8)
- The patched binary will need re-signing for macOS to load it

## References

- `reports/main/context_lifecycle_analysis_N1474_2025-12-21.md` - Full RE analysis
- `reports/main/context_common_structure_N1473_2025-12-21.md` - Structure analysis
- `BLOG_POST.md` - Project blog with research timeline
