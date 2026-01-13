# URGENT: Fix Global Encoding Mutex Misdiagnosis

**Created by: Manager**
**Date: 2025-12-20**
**Priority: CRITICAL**

## The Problem

We added a **global encoding mutex** (`getGlobalMetalEncodingMutex()`) that serializes ALL Metal encoding operations. This was based on a **potentially incorrect diagnosis** that Apple's AGX driver has a race condition.

### Evidence the Diagnosis May Be Wrong

1. **Pure Metal test** (per-thread queues, per-thread buffers) achieves **5.6x scaling at 8 threads** with NO global mutex
2. **Apple bug report** has been corrected to note: "Raw Metal API achieves 62% efficiency at 8 threads"
3. **PyTorch MPS plateaus at 3,800 ops/s** due to the global mutex WE added

### Current Impact

| Configuration | Throughput | Cause |
|---------------|------------|-------|
| Pure Metal (multi-queue) | 29,298 ops/s | No global mutex |
| PyTorch MPS | 3,819 ops/s | **Global mutex we added** |
| **Lost performance** | **7.7x** | Our misdiagnosis |

## The Plan

### Phase 1: Prove Parallel Encoding is Safe (TLA+)

**File**: `mps-verify/specs/MPSEncodingPath.tla` (already created)

1. Install/configure Java for TLC
2. Run: `java -jar tla2tools.jar -config MPSEncodingPath.cfg MPSEncodingPath.tla`
3. Verify:
   - `ParallelEncodingWitnessed` property holds (multiple threads can encode simultaneously)
   - `NoBufferSharing` invariant holds (each buffer used by one thread)
   - `NoEncoderSharing` invariant holds (each stream encoder used by one thread)

**Expected Result**: Proof shows parallel encoding is safe when each thread has its own stream and buffer.

### Phase 2: Empirical Test - Remove Global Mutex

**Environment variable exists**: `MPS_DISABLE_ENCODING_MUTEX=1`

1. Run test suite with mutex disabled:
   ```bash
   MPS_DISABLE_ENCODING_MUTEX=1 python3 tests/benchmark_comprehensive_final.py
   ```

2. Run ThreadSanitizer to check for races:
   ```bash
   # Build with TSan
   MPS_DISABLE_ENCODING_MUTEX=1 ./tests/build/test_with_tsan
   ```

3. Record results:
   - Does it crash?
   - Are there data races?
   - What's the throughput?

### Phase 3: Root Cause of Original 20% Crash Rate

The comment says removing mutex caused "~20% SIGSEGV crash rate" in N=1068.

Questions to answer:
1. Was this BEFORE or AFTER other thread-safety fixes?
2. Did we fix other bugs that made the mutex unnecessary?
3. Was the crash from sharing buffers (our bug) not AGX driver (Apple bug)?

**Investigation**:
```bash
# Check git history around N=1068
git log --oneline | grep -E "N=106[0-9]|N=107[0-9]"

# Check what was different in that test
git show <commit-hash>
```

### Phase 4: If Safe, Remove the Mutex

If Phase 1 and 2 confirm parallel encoding is safe:

1. **Remove global mutex**: Edit `MPSStream.mm` to remove `MPSEncodingLock` from encoding path
2. **Keep per-stream mutex**: Per-stream protection is still needed
3. **Verify**: Run full test suite
4. **Benchmark**: Measure throughput improvement

### Phase 5: Fix Documentation

1. **Apple bug report**: Either withdraw or update with corrected analysis
2. **BLOG_POST.md**: Update the "AGX driver bug" section
3. **README.md**: Update performance claims
4. **WORKER_DIRECTIVE.md**: Document the fix

### Phase 6: Fix Formal Verification

1. **Add encoding path to existing specs** - not just stream allocation
2. **Add serialization detection invariant**:
   ```tla
   NoGlobalSerialization ==
       \A state : ~(only_one_thread_can_be_encoding)
   ```
3. **Run all specs** to verify no other hidden serialization

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| Threading throughput (8 threads) | 3,752 ops/s | 20,000+ ops/s |
| Threading scaling | 1.14x | 5x+ |
| TLA+ proof | Stream pool only | Full encoding path |
| Apple report | Possibly false | Corrected or withdrawn |

## Commands to Execute

```bash
# Phase 1: Run TLA+ proof
cd mps-verify/specs
export JAVA_HOME=/path/to/jdk
java -jar ../tools/tla2tools.jar -config MPSEncodingPath.cfg MPSEncodingPath.tla

# Phase 2: Empirical test without mutex
MPS_DISABLE_ENCODING_MUTEX=1 python3 tests/benchmark_comprehensive_final.py

# Phase 3: Check git history
git log --oneline -100 | grep -i "mutex\|encoding\|race"

# Phase 4: If safe, edit MPSStream.mm (specific changes TBD based on findings)

# Phase 5: Update documentation (BLOG_POST.md, README.md, apple_feedback/)

# Phase 6: Run all TLA+ specs
for spec in mps-verify/specs/*.tla; do
    java -jar tla2tools.jar "$spec"
done
```

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Mutex IS necessary | Medium | Phase 2 will reveal crashes/races |
| Other hidden serialization | Low | Phase 6 TLA+ analysis |
| Performance regression | Low | Keep old code path as fallback |

## Worker Instructions

1. Start with Phase 2 (empirical test) - fastest way to get signal
2. If no crashes/races, proceed to Phase 4 (remove mutex)
3. If crashes occur, investigate Phase 3 (root cause)
4. Update all documentation regardless of outcome
5. Commit with detailed findings

**This is a CRITICAL fix. We may have crippled performance by 7.7x based on a misdiagnosis.**
