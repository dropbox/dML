# MPS Binary Analysis: Technical and Legal Assessment

**Date**: 2025-12-17
**Status**: RESEARCH ONLY - Not recommended for production

## Executive Summary

While technically possible to analyze and potentially patch Apple's MPS binary, this approach has significant **legal risks** and **practical limitations**. This document explores the technical feasibility while recommending against implementation.

## Technical Feasibility

### MPS Binary Location

On modern macOS (11+), MPS is embedded in the dyld shared cache:

```
/System/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e
```

The framework no longer exists as a standalone binary at:
```
/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders
# Returns: No such file or directory
```

### Extracting from Dyld Cache

Tools that can extract from dyld cache:
1. `dyld_shared_cache_extract_dylibs` (Xcode tools)
2. Hopper Disassembler
3. IDA Pro
4. Ghidra (free, NSA open-source)

```bash
# Example extraction (requires SIP disabled)
dyld_shared_cache_util -extract /tmp/extracted /System/Library/dyld/dyld_shared_cache_arm64e
```

### System Integrity Protection (SIP)

macOS SIP prevents:
1. Modifying system frameworks
2. Injecting into system processes
3. Loading modified system libraries

Would require:
- Booting to Recovery Mode
- Running `csrutil disable`
- **This disables critical security protections**

### Potential Patch Points

If we could analyze the binary, we'd look for:

```asm
; Hypothetical global state access
_MPSNDArrayMatrixMultiplication_encode:
    adrp    x0, _g_shared_encoder@PAGE      ; GLOBAL STATE
    ldr     x0, [x0, _g_shared_encoder@PAGEOFF]
    ; ... encoding logic using shared state ...
```

A patch would replace global state access with per-instance state:

```asm
; Hypothetical fix
_MPSNDArrayMatrixMultiplication_encode:
    ldr     x0, [x19, #OFFSET_INSTANCE_ENCODER]  ; PER-INSTANCE
    ; ... encoding logic using instance state ...
```

### Technical Challenges

1. **ARM64e PAC**: Pointer Authentication prevents simple binary patches
2. **Code signing**: Modified binaries won't load without re-signing
3. **Dyld cache**: Can't patch individual frameworks without rebuilding cache
4. **ASLR**: Address Space Layout Randomization complicates patching
5. **Updates**: Every macOS update invalidates patches

## Legal Analysis

### Apple EULA Restrictions

Apple's macOS Software License Agreement (Section 2.B):

> "You may not copy (except as expressly permitted by this License), decompile,
> reverse engineer, disassemble, attempt to derive the source code of, decrypt,
> modify, or create derivative works of the Software..."

### DMCA Considerations

The Digital Millennium Copyright Act (DMCA) Section 1201 prohibits:
- Circumventing technological protection measures
- Developing/distributing circumvention tools

**Exception**: Security research exemption may apply, but is narrow.

### Risk Assessment

| Action | Legal Risk |
|--------|------------|
| Static analysis of symbols | Low |
| Disassembly for understanding | Medium |
| Creating patches | High |
| Distributing patches | Very High |
| Using patches in production | High |

## Alternative Approaches (Legal)

### 1. Steel Integration (Recommended)
Use Apple's MIT-licensed MLX kernels instead of patching MPS.
- **Legal**: Yes (MIT license)
- **Effort**: 20-40 commits
- **Risk**: Low

### 2. Interposition/Hooking
Create a shim library that intercepts MPS calls:

```c
// Legal: DYLD_INSERT_LIBRARIES approach
// Intercept MPSNDArrayMatrixMultiplication encoding
DYLD_INTERPOSE(my_encode, original_encode);

void my_encode(id self, SEL cmd, ...) {
    @synchronized(global_lock) {  // Add serialization
        original_encode(self, cmd, ...);
    }
}
```

- **Legal**: Possibly (doesn't modify Apple code)
- **Practical**: What we already do with mutexes

### 3. Runtime Swizzling
Use Objective-C runtime to add thread-safety:

```objc
// Swizzle encodeToCommandBuffer: to add locking
Method original = class_getInstanceMethod([MPSNDArrayMatrixMultiplication class],
                                          @selector(encodeToCommandBuffer:...));
method_setImplementation(original, (IMP)safe_encode);
```

- **Legal**: Gray area (modifies runtime behavior)
- **Risk**: Medium (could break with updates)

### 4. Apple Bug Report
Submit detailed radar and wait for official fix.
- **Legal**: Yes
- **Timeline**: Unknown (could be years)

## Recommendation

**Do NOT attempt to patch MPS binary.** Instead:

1. **Use Steel Integration** - Port MLX kernels (MIT licensed, legal)
2. **Keep current mitigations** - Mutexes work, just limit scaling
3. **Submit Apple Radar** - Document the issue officially
4. **Monitor MLX** - Apple may improve MPSGraph threading

## Conclusion

While binary patching is technically interesting, it is:
- **Legally risky** under Apple EULA and DMCA
- **Practically fragile** due to SIP, PAC, code signing
- **Unnecessary** given MLX Steel kernels are available

The legal, supported path is to use Apple's own open-source alternative (MLX Steel) rather than attempting to fix their closed-source code.

## References

- Apple macOS EULA: https://www.apple.com/legal/sla/
- DMCA Section 1201: https://www.law.cornell.edu/uscode/text/17/1201
- MLX License: MIT (https://github.com/ml-explore/mlx/blob/main/LICENSE)
