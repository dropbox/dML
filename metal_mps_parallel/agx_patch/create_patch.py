#!/usr/bin/env python3
"""
AGX Driver Binary Patch for Race Condition Fix
Created by Andrew Yates

This script patches the AGXMetalG16X driver to fix a race condition in
-[AGXG16XFamilyComputeContext destroyImpl] where self->_impl is NULLed
AFTER the lock is released, allowing another thread to access the
now-invalid pointer.

The fix: Move the `str xzr, [x19, x24]` instruction to execute BEFORE
the unlock, while still holding the lock.

USAGE:
    python3 create_patch.py [--verify] [--dry-run] [--input PATH] [--output PATH]

OPTIONS:
    --verify    Only verify the patch locations, don't patch
    --dry-run   Show what would be patched without writing
    --input     Input Mach-O (fat or thin). Defaults to AGXMetalG16X_arm64e.
    --output    Output path for patched binary.

NOTE: Creating a patched copy does NOT require SIP changes. Applying the patch
to the system driver requires disabling SIP and re-signing, and should only be
done with explicit user approval.
"""

import argparse
import struct
import hashlib
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PatchLocation:
    """Describes a single instruction patch."""
    va: int           # Offset within the arm64e slice (__TEXT vmaddr==fileoff for this driver)
    old_bytes: bytes  # Expected original bytes
    new_bytes: bytes  # Replacement bytes
    description: str


class AGXPatcher:
    """Patches AGX driver to fix race condition in destroyImpl"""

    # Fat binary structure
    FAT_MAGIC = 0xcafebabe
    FAT_MAGIC_64 = 0xcafebabf
    MH_MAGIC_64 = 0xfeedfacf

    CPU_TYPE_ARM64 = 0x0100000c
    CPU_SUBTYPE_ARM64E = 2
    CPU_SUBTYPE_MASK = 0x00ffffff

    # ARM64 instruction encodings (little-endian)
    NOP = bytes([0x1f, 0x20, 0x03, 0xd5])

    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.data = bytearray(self.input_path.read_bytes())
        self.slice_offset = self._find_arm64e_slice()

    def _find_arm64e_slice(self) -> int:
        """Find the offset of the arm64e slice in the fat binary"""
        magic = struct.unpack('>I', self.data[0:4])[0]

        if magic == self.FAT_MAGIC:
            nfat = struct.unpack('>I', self.data[4:8])[0]
            for i in range(nfat):
                base = 8 + i * 20
                cputype = struct.unpack('>I', self.data[base:base + 4])[0]
                cpusubtype = struct.unpack('>I', self.data[base + 4:base + 8])[0]
                offset = struct.unpack('>I', self.data[base + 8:base + 12])[0]

                if cputype == self.CPU_TYPE_ARM64 and (cpusubtype & self.CPU_SUBTYPE_MASK) == self.CPU_SUBTYPE_ARM64E:
                    print(f"Found arm64e slice at offset 0x{offset:x}")
                    return offset
            raise ValueError("No arm64e slice found in fat binary")
        elif magic == self.FAT_MAGIC_64:
            nfat = struct.unpack('>I', self.data[4:8])[0]
            for i in range(nfat):
                base = 8 + i * 32
                cputype = struct.unpack('>I', self.data[base:base + 4])[0]
                cpusubtype = struct.unpack('>I', self.data[base + 4:base + 8])[0]
                offset = struct.unpack('>Q', self.data[base + 8:base + 16])[0]

                if cputype == self.CPU_TYPE_ARM64 and (cpusubtype & self.CPU_SUBTYPE_MASK) == self.CPU_SUBTYPE_ARM64E:
                    print(f"Found arm64e slice at offset 0x{offset:x}")
                    return offset
            raise ValueError("No arm64e slice found in fat64 binary")
        elif magic == self.MH_MAGIC_64:
            # Thin 64-bit Mach-O. slice_offset is 0. Validate arch when possible.
            cputype = struct.unpack('<I', self.data[4:8])[0]
            cpusubtype = struct.unpack('<I', self.data[8:12])[0]
            if cputype != self.CPU_TYPE_ARM64 or (cpusubtype & self.CPU_SUBTYPE_MASK) != self.CPU_SUBTYPE_ARM64E:
                raise ValueError(f"Thin Mach-O is not arm64e: cputype=0x{cputype:x} cpusubtype=0x{cpusubtype:x}")
            return 0
        else:
            raise ValueError(f"Unknown binary format: magic=0x{magic:x}")

    def _file_offset(self, va: int) -> int:
        """Convert virtual address to file offset"""
        return self.slice_offset + va

    def read_instruction(self, va: int) -> bytes:
        """Read 4-byte instruction at virtual address"""
        offset = self._file_offset(va)
        return bytes(self.data[offset:offset+4])

    def write_instruction(self, va: int, instr: bytes):
        """Write 4-byte instruction at virtual address"""
        assert len(instr) == 4
        offset = self._file_offset(va)
        self.data[offset:offset+4] = instr

    @staticmethod
    def encode_b(from_va: int, to_va: int) -> bytes:
        """Encode unconditional branch: b <target>"""
        offset = (to_va - from_va) // 4
        if not (-0x2000000 <= offset < 0x2000000):
            raise ValueError(f"Branch offset out of range: {offset}")
        imm26 = offset & 0x3ffffff
        instr = 0x14000000 | imm26
        return struct.pack('<I', instr)

    @staticmethod
    def encode_b_cond(from_va: int, to_va: int, cond: int) -> bytes:
        """Encode conditional branch: b.<cond> <target>"""
        offset = (to_va - from_va) // 4
        if not (-0x40000 <= offset < 0x40000):
            raise ValueError(f"Conditional branch offset out of range: {offset}")
        imm19 = offset & 0x7ffff
        instr = 0x54000000 | (imm19 << 5) | cond
        return struct.pack('<I', instr)

    @staticmethod
    def encode_bl(from_va: int, to_va: int) -> bytes:
        """Encode branch-with-link: bl <target>"""
        offset = (to_va - from_va) // 4
        if not (-0x2000000 <= offset < 0x2000000):
            raise ValueError(f"Branch offset out of range: {offset}")
        imm26 = offset & 0x3ffffff
        instr = 0x94000000 | imm26
        return struct.pack('<I', instr)

    def get_patches(self) -> list[PatchLocation]:
        """
        Generate the patch list for fixing the race condition.

        Original code structure:

        0x2be05c: b.hi 0x2be07c        ; if freelist full, go Path 2
        ...
        Path 1 (freelist not full):
        0x2be06c: str x20, [x9, #0x1a68] ; store to freelist
        0x2be070: add x0, x25, x21       ; prep lock addr
        0x2be074: bl unlock
        0x2be078: b 0x2be08c             ; jump to shared str xzr

        Path 2 (freelist full):
        0x2be07c: add x0, x25, x21
        0x2be080: bl unlock
        0x2be084: mov x0, x20
        0x2be088: bl free
        0x2be08c: str xzr, [x19, x24]    ; BUG: NULL after unlock!

        Fixed code (inline, within space constraints):

        Path 1 gets modified to NULL before unlock:
        0x2be070: str xzr, [x19, x24]    ; NULL first!
        0x2be074: add x0, x25, x21       ; prep lock addr
        0x2be078: bl unlock
        0x2be07c: b 0x2be090             ; skip to epilogue

        Path 2 is also fixed to NULL before unlock, but due to space constraints
        the binary patch skips the `free()` call (memory leak when freelist is full).
        For a complete fix without leak, use the runtime dylib (agx_fix.mm).
        """
        patches = []

        # Known instruction encodings (verified from disassembly)
        STR_XZR_X19_X24 = bytes([0x7f, 0x6a, 0x38, 0xf8])  # str xzr, [x19, x24]
        ADD_X0_X25_X21 = bytes([0x20, 0x03, 0x15, 0x8b])   # add x0, x25, x21

        # Addresses from disassembly
        UNLOCK_STUB = 0x6d49e4
        EPILOGUE = 0x2be090

        # === PATH 1 FIX (inline) ===

        # 0x2be070: Change from "add x0, x25, x21" to "str xzr, [x19, x24]"
        patches.append(PatchLocation(
            va=0x2be070,
            old_bytes=ADD_X0_X25_X21,
            new_bytes=STR_XZR_X19_X24,
            description="Path 1: NULL _impl before unlock"
        ))

        # 0x2be074: Change from "bl unlock" to "add x0, x25, x21"
        original_bl_unlock = self.encode_bl(0x2be074, UNLOCK_STUB)
        patches.append(PatchLocation(
            va=0x2be074,
            old_bytes=original_bl_unlock,
            new_bytes=ADD_X0_X25_X21,
            description="Path 1: Move add to here"
        ))

        # 0x2be078: Change from "b 0x2be08c" to "bl unlock"
        original_b_str = self.encode_b(0x2be078, 0x2be08c)
        new_bl_unlock = self.encode_bl(0x2be078, UNLOCK_STUB)
        patches.append(PatchLocation(
            va=0x2be078,
            old_bytes=original_b_str,
            new_bytes=new_bl_unlock,
            description="Path 1: Unlock here"
        ))

        # 0x2be07c: Change from "add x0, x25, x21" (Path 2 entry) to "b epilogue"
        b_epilogue = self.encode_b(0x2be07c, EPILOGUE)
        patches.append(PatchLocation(
            va=0x2be07c,
            old_bytes=ADD_X0_X25_X21,
            new_bytes=b_epilogue,
            description="Path 1: Jump to epilogue (overwrites Path 2 entry)"
        ))

        # === PATH 2 REDIRECT ===
        # Redirect Path 2 entry from 0x2be07c to 0x2be080
        # 0x2be05c: b.hi 0x2be07c -> b.hi 0x2be080

        COND_HI = 0x8  # higher (unsigned >)
        old_b_hi = self.encode_b_cond(0x2be05c, 0x2be07c, COND_HI)
        new_b_hi = self.encode_b_cond(0x2be05c, 0x2be080, COND_HI)
        patches.append(PatchLocation(
            va=0x2be05c,
            old_bytes=old_b_hi,
            new_bytes=new_b_hi,
            description="Redirect Path 2 to start at 0x2be080"
        ))

        # === PATH 2 FIX (partial - limited by space) ===
        # Path 2 now starts at 0x2be080 (was bl unlock)
        # We can squeeze in str xzr but need to tail-call free

        # 0x2be080: Change "bl unlock" to "str xzr"
        old_p2_unlock = self.encode_bl(0x2be080, UNLOCK_STUB)
        patches.append(PatchLocation(
            va=0x2be080,
            old_bytes=old_p2_unlock,
            new_bytes=STR_XZR_X19_X24,
            description="Path 2: NULL _impl first"
        ))

        # 0x2be084: Change "mov x0, x20" to "add x0, x25, x21"
        MOV_X0_X20 = bytes([0xe0, 0x03, 0x14, 0xaa])
        patches.append(PatchLocation(
            va=0x2be084,
            old_bytes=MOV_X0_X20,
            new_bytes=ADD_X0_X25_X21,
            description="Path 2: Prep lock address"
        ))

        # 0x2be088: Change "bl free" to "bl unlock"
        FREE_STUB = 0x6d46c4
        old_bl_free = self.encode_bl(0x2be088, FREE_STUB)
        new_bl_unlock_p2 = self.encode_bl(0x2be088, UNLOCK_STUB)
        patches.append(PatchLocation(
            va=0x2be088,
            old_bytes=old_bl_free,
            new_bytes=new_bl_unlock_p2,
            description="Path 2: Unlock"
        ))

        # 0x2be08c: Change "str xzr" (now unused) to "mov x0, x20"
        patches.append(PatchLocation(
            va=0x2be08c,
            old_bytes=STR_XZR_X19_X24,
            new_bytes=MOV_X0_X20,
            description="Path 2: Prep for free"
        ))

        # 0x2be090: PROBLEM - this is the epilogue start!
        # We need "bl free" here but would overwrite "ldp x29, x30, [sp, #0x90]"
        #
        # SOLUTION: Use tail-call "b free" and rely on free() returning to our caller
        # BUT: x30 was clobbered by "bl unlock", so this won't work!
        #
        # ACTUAL SOLUTION: We need the epilogue, but we also need to call free.
        # The only way to do this in available space is to NOT call free
        # and leak the memory (acceptable for a rare path) OR use a code cave.
        #
        # For now, we'll implement a memory-leaking fix that at least prevents
        # the crash. The proper fix requires the runtime dylib.

        # Skip the free call - go directly to epilogue
        # This leaks memory when freelist is full (rare) but prevents crash
        b_epilogue_p2 = self.encode_b(0x2be08c, EPILOGUE)
        patches[-1] = PatchLocation(
            va=0x2be08c,
            old_bytes=STR_XZR_X19_X24,
            new_bytes=b_epilogue_p2,
            description="Path 2: Skip to epilogue (leaks memory but prevents crash)"
        )

        return patches

    def verify_patches(self, patches: list[PatchLocation]) -> bool:
        """Verify all patch locations have expected original bytes"""
        all_ok = True
        for p in patches:
            actual = self.read_instruction(p.va)
            if actual != p.old_bytes:
                print(f"MISMATCH at 0x{p.va:06x}: expected {p.old_bytes.hex()}, got {actual.hex()}")
                print(f"  Description: {p.description}")
                all_ok = False
            else:
                print(f"OK at 0x{p.va:06x}: {p.description}")
        return all_ok

    def apply_patches(self, patches: list[PatchLocation], dry_run: bool = False):
        """Apply all patches to the binary"""
        for p in patches:
            if not dry_run:
                self.write_instruction(p.va, p.new_bytes)
            print(f"{'Would patch' if dry_run else 'Patched'} 0x{p.va:06x}: "
                  f"{p.old_bytes.hex()} -> {p.new_bytes.hex()}")
            print(f"  {p.description}")

    def save(self, output_path: str):
        """Save the patched binary"""
        output = Path(output_path)
        output.write_bytes(self.data)
        print(f"Saved patched binary to {output}")

        # Compute hash
        sha256 = hashlib.sha256(self.data).hexdigest()
        print(f"SHA256: {sha256}")


def main():
    parser = argparse.ArgumentParser(description='Patch AGX driver race condition')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify patch locations')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show patches without applying')
    parser.add_argument('--input', default='AGXMetalG16X_arm64e',
                        help='Input binary (default: AGXMetalG16X_arm64e)')
    parser.add_argument('--output', default='AGXMetalG16X_patched',
                        help='Output binary (default: AGXMetalG16X_patched)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Please copy the AGX driver or extract the arm64e slice first:")
        print("  lipo -thin arm64e /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X -output AGXMetalG16X_arm64e")
        return 1

    print(f"Loading {input_path}...")
    patcher = AGXPatcher(str(input_path))
    patches = patcher.get_patches()

    print(f"\nGenerated {len(patches)} patches:")
    print("-" * 60)

    if not patcher.verify_patches(patches):
        print("\nERROR: Patch verification failed!")
        print("The binary may have been updated or already patched.")
        return 1

    print("-" * 60)

    if args.verify:
        print("\nVerification complete. No changes made.")
        return 0

    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Applying patches...")
    patcher.apply_patches(patches, dry_run=args.dry_run)

    if not args.dry_run:
        patcher.save(str(output_path))

        print("\n" + "=" * 60)
        print("IMPORTANT NOTES:")
        print("=" * 60)
        print("""
1. This patch fixes the race condition for Path 1 (common case).
   Path 2 (freelist full) has a memory leak to avoid overflow.

2. For a complete fix without memory leak, use the runtime dylib:
   DYLD_INSERT_LIBRARIES=/path/to/libagx_fix.dylib your_app

3. To apply this patch to the system driver:
   a. Boot into Recovery Mode (hold power button on Apple Silicon)
   b. Run: csrutil disable
   c. Reboot normally
   d. Copy patched binary to /System/Library/Extensions/...
   e. Re-sign or disable signature checking
   f. Reboot and test

4. The patched binary must be re-signed for macOS to load it.
   This requires disabling SIP or using a valid signing identity.

5. SHA256 of the patched binary is printed above for verification.
""")

    return 0


if __name__ == '__main__':
    exit(main())
