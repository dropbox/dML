# Rigorous Analysis: What Can We Actually Prove?

**Created by Andrew Yates**
**Date: 2025-12-20**

## The Claim Under Scrutiny

**Claim**: "Apple's Metal driver has a serialization bug that causes threading efficiency to drop to 3% at 8 threads."

## What Formal Methods CAN Prove

TLA+ and Lean can prove:
1. **Our code permits parallelism** - No accidental serialization in MPSStreamPool
2. **No deadlocks** - The design is livelock-free
3. **Correctness** - Invariants hold, no data races

## What Formal Methods CANNOT Prove

1. **Performance characteristics** - "Threading is 30x slower than batching"
2. **External system behavior** - "Apple's driver serializes"
3. **Root cause attribution** - "The driver, not our code, is the bottleneck"

## Alternative Hypotheses to Rule Out

| Hypothesis | Test | Result |
|------------|------|--------|
| H1: Python GIL causes serialization | Test with pure C++ | **TODO** |
| H2: PyTorch has internal locks | Test with raw Metal | **TODO** |
| H3: Our MPSStreamPool mutex | Check TLA+ spec | See below |
| H4: Memory bandwidth saturation | Vary tensor sizes | **TODO** |
| H5: Command queue hardware limit | Check Metal docs | **TODO** |

## TLA+ Verification of Our Code

### MPSStreamPoolParallel.tla

This spec verifies that our design PERMITS parallelism:

```tla
\* Property: Parallelism is achievable (existential property)
ParallelismPossible ==
    <>parallel_witnessed  \* Eventually, 2+ threads are in_use simultaneously
```

**If this passes**: Our code design does NOT accidentally serialize.
**If this fails**: We have a bug in our design.

### Running the Verification

```bash
cd specs
java -jar tla2tools.jar -config MPSStreamPoolParallel.cfg MPSStreamPoolParallel.tla
```

## The Honest Assessment

### What We Know (Empirical)

1. Threading efficiency: 12% → 6% → 3% → 1.6% as N doubles
2. Batching efficiency: 84% → 91% → 96% → 95% (near-linear)
3. Process pool efficiency: ~23% (6.8x better than threading)

### What We DON'T Know (Not Proven)

1. **WHERE** exactly the serialization occurs
2. **WHETHER** it's Apple's driver vs something else
3. **WHY** batching bypasses it

### What Would Constitute Proof

To PROVE "Apple's driver serializes", we need:

1. **Minimal reproduction in pure Metal/C++** (no Python, no PyTorch)
2. **Instrumentation showing lock contention** in Apple code
3. **Comparison with non-MPS Metal** (custom kernels)

## Conclusion

**I cannot formally prove the claim "Apple's driver has a serialization bug."**

What I CAN say:
- Empirical measurements show threading efficiency drops to 3% at 8 threads
- Our TLA+ specs verify our code PERMITS parallelism
- The bottleneck is NOT in our code (if TLA+ passes)
- SOMETHING serializes threading but not batching

To prove it's specifically Apple's driver, we need:
1. Pure Metal/C++ minimal reproduction
2. DYLD interposition to instrument Apple's code
3. Comparison with custom Metal kernels

**The rigorous answer: We have strong EVIDENCE, not PROOF.**
