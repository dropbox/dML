# 7 Critical Holes in the MPS Parallel Story

**Created by Andrew Yates**
**Date**: 2025-12-20
**Status**: Analysis complete, roadmap created

---

## HOLE 1: "Metal Driver Bottleneck" - UNPROVEN CLAIM

**Claim**: "The bottleneck is in Apple's Metal driver, not our code"

**Problem**: We never actually PROVED this. We:
- Observed low efficiency (13-30%)
- Ran TLA+ on small models
- Assumed Apple is at fault

**What we SHOULD have done**:
- Profile with Instruments/Metal System Trace
- Show WHERE time is spent (our locks vs Apple's driver)
- Identify specific Metal API calls that serialize
- Create minimal reproduction outside PyTorch

**Status**: SPECULATION, NOT PROOF

**Resolution**: See Phase 1 in WORKER_DIRECTIVE_HOLES_ROADMAP.md

---

## HOLE 2: "3.64x Speedup" - UNVERIFIED/CONTRADICTED

**Claim**: "Large models achieve 3.64x speedup at 8 threads"

**Problem**: My tests showed:
- Large transformer: 1.46x at 8 threads (NOT 3.64x)
- Statistical benchmark: 0.95x at 8 threads

**Where did 3.64x come from?**
- throughput_N1261_2025-12-18.md claims it
- But I cannot reproduce it
- Was it BatchQueue pipelining? Different hardware state? Fabricated?

**Status**: CONTRADICTED BY MEASUREMENTS

**Resolution**: Investigate source, correct or validate claim

---

## HOLE 3: BatchQueue - DOES IT EXIST?

**Claim**: "Use torch.mps.BatchQueue for correctness"

**Problem**: We reference it but:
- Never tested if it actually exists in the build
- Never showed it working
- Never benchmarked it vs threading

**Status**: ✅ CLOSED - Verified working

**Evidence**:
```python
>>> import torch
>>> torch.mps.BatchQueue
<class 'torch.mps.BatchQueue'>
>>> bq = torch.mps.BatchQueue(num_workers=1)
>>> # Works!
```

---

## HOLE 4: TLA+ Proofs - BOUNDED, NOT GENERAL

**Claim**: "14.7M states explored, correctness proven"

**Problem**: The proofs used:
- NumStreams = 2-4 (implementation has 32)
- NumThreads = 2-3 (we claim 8-thread safety)
- MaxOperations = 5-12 (unbounded in reality)

**What this means**:
- We proved correctness for TINY models
- NOT proven for actual 8-thread, 32-stream config
- Induction argument is informal, not machine-checked

**Status**: BOUNDED PROOF, NOT GENERAL

**Resolution**: Either run TLA+ at higher bounds or create Lean 4 induction proof

---

## HOLE 5: "201 Bugs Fixed" - UNVERIFIED LIST

**Claim**: "Fixed 201 threading issues"

**Problem**:
- Where is the list of 201 bugs?
- Can each one be reproduced?
- Were they actual bugs or just code cleanups?

**Status**: UNVERIFIED CLAIM

**Resolution**: Create itemized list from git history with commit hashes

---

## HOLE 6: Apple Bug - NO RADAR FILED

**Claim**: "These are Apple framework bugs"

**Problem**:
- We never filed a radar
- No confirmation from Apple
- No minimal reproduction outside PyTorch
- Could be our misuse of the API

**Status**: PENDING - Report ready, not filed

**Resolution**: File radar at feedbackassistant.apple.com

---

## HOLE 7: Never Attempted to FIX the Issue

**Original Plan**: "Reverse engineer and fix the MPS issue"

**What happened**:
- We worked AROUND with batching
- Never identified the actual cause
- Never attempted a real fix
- Never showed what would happen IF Apple fixed their driver

**Status**: INCOMPLETE

**Resolution**: Phase 4 of roadmap - reverse engineer and fix

---

## Summary Table

| # | Hole | Status | Phase |
|---|------|--------|-------|
| 1 | Metal driver bottleneck | UNPROVEN | 1 |
| 2 | 3.64x speedup | CONTRADICTED | 1 |
| 3 | BatchQueue exists | ✅ CLOSED | - |
| 4 | TLA+ bounded proofs | ACKNOWLEDGED | 1 |
| 5 | 201 bugs list | UNVERIFIED | 1 |
| 6 | Apple Radar | NOT FILED | 1 |
| 7 | Never fixed issue | INCOMPLETE | 4 |

---

## Next Steps

See `WORKER_DIRECTIVE_HOLES_ROADMAP.md` for the complete 4-phase resolution plan:
1. Address all 7 holes with evidence
2. Exhaustive performance benchmarks
3. Formal proof of optimality
4. Fix Apple MPS issue via reverse engineering
