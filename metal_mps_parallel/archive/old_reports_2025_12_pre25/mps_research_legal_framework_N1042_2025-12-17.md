# MPS Research: Legal Framework for Analysis and Publication

**Date**: 2025-12-17
**Status**: RESEARCH IS LEGAL - with responsible disclosure

## Executive Summary

**YES**, we can:
1. Reverse engineer MPS for research/interoperability
2. Publish technical findings
3. Submit detailed bug reports to Apple
4. Present at security conferences

The key legal protections are:
- DMCA security research exemption (17 U.S.C. § 1201(j))
- Interoperability exception (17 U.S.C. § 1201(f))
- First Amendment (publication of research findings)

## Legal Framework

### DMCA Security Research Exemption

**17 U.S.C. § 1201(j)** - Security Testing:

> "it is not a violation... to engage in an act of security testing, if such act does not constitute infringement... and... the information derived from such act is used solely to promote the security of the owner or operator of such computer..."

**Key requirements**:
1. Good faith security research
2. Information used to improve security
3. Responsible disclosure to vendor

### DMCA Interoperability Exception

**17 U.S.C. § 1201(f)** - Reverse Engineering:

> "a person who has lawfully obtained the right to use a copy of a computer program may circumvent a technological measure... for the sole purpose of identifying and analyzing those elements of the program that are necessary to achieve interoperability"

**Our case**: We're analyzing MPS to achieve interoperability with PyTorch for multi-threaded ML inference.

### CFAA Safe Harbor

The Computer Fraud and Abuse Act has a good-faith security research safe harbor (added 2022):

> Good faith security research is not unauthorized access if done to identify vulnerabilities in a manner designed to improve security.

### Apple's Security Bounty Program

Apple explicitly invites security research:
- https://security.apple.com/bounty/
- Covers macOS and frameworks
- Provides legal safe harbor for researchers

## What We CAN Do

### 1. Static Analysis of Symbols ✅

```bash
# Extract and analyze symbols
nm -g /path/to/extracted/MetalPerformanceShaders | grep MPSNDArray
strings /path/to/extracted/MetalPerformanceShaders | grep -i thread
```

### 2. Dynamic Analysis with Instruments ✅

```bash
# Profile MPS behavior legally using Apple's own tools
xcrun xctrace record --template "Time Profiler" --launch -- ./mps_test
```

### 3. Disassembly for Understanding ✅

Using Ghidra, Hopper, or IDA to understand the crash:
- Identify the global state causing races
- Document the internal architecture
- Find the specific unsafe code patterns

### 4. Publish Research Findings ✅

We can publish:
- Technical analysis of the thread-safety bug
- Reproduction code (we already have this)
- Proposed fixes (even if we can't implement them)
- Comparison with MLX's thread-safe approach

### 5. Submit to Apple ✅

Via multiple channels:
- Feedback Assistant (bug report)
- security@apple.com (if security-relevant)
- Apple Security Bounty (if applicable)

### 6. Present at Conferences ✅

Appropriate venues:
- WWDC (Apple's own conference)
- Black Hat / DEF CON (security research)
- USENIX Security
- Academic conferences (OSDI, SOSP, etc.)

## What We Should NOT Do

| Action | Risk | Recommendation |
|--------|------|----------------|
| Distribute binary patches | High | Don't do this |
| Create circumvention tools | High | Don't distribute |
| Bypass SIP in production | Medium | Research only |
| Exploit for malicious purposes | Very High | Never |

## Research Plan

### Phase 1: Extract and Analyze (Legal)

```bash
# 1. Extract MPS from dyld cache (requires SIP disabled on research machine)
dyld_shared_cache_util -extract /tmp/mps_research \
    /System/Library/dyld/dyld_shared_cache_arm64e

# 2. Analyze with Ghidra
ghidra /tmp/mps_research/MetalPerformanceShaders

# 3. Find the problematic code
# Look for: global variables, static state, shared encoders
```

### Phase 2: Document Findings

Create detailed technical report:
1. **Root cause identification**: Which global state causes races
2. **Crash analysis**: Exact sequence leading to `MPSSetResourcesOnCommandEncoder` crash
3. **Code patterns**: Document the unsafe patterns
4. **Proposed fix**: How Apple could fix it (conceptual)

### Phase 3: Responsible Disclosure

1. **Submit to Apple first** (30-90 day disclosure window)
2. **Wait for response/fix**
3. **Publish after disclosure period** or after Apple responds

### Phase 4: Publication

Options:
- Technical blog post
- Academic paper
- Conference presentation
- GitHub repository with findings

## Sample Research Publication Outline

```
Title: Thread-Safety Analysis of Apple MetalPerformanceShaders Framework

Abstract:
We present a detailed analysis of thread-safety issues in Apple's
MetalPerformanceShaders (MPS) framework that prevent concurrent GPU
kernel encoding. Through reverse engineering and dynamic analysis,
we identify internal global state in MPSNDArrayMatrixMultiplication
that causes crashes when multiple threads encode concurrently...

1. Introduction
   - MPS is used by PyTorch, TensorFlow, etc.
   - Multi-threaded ML inference is increasingly important
   - MPS has undocumented thread-safety limitations

2. Background
   - Metal GPU programming model
   - MPS architecture
   - Expected thread-safety guarantees

3. Methodology
   - Dynamic analysis with Instruments
   - Static analysis with Ghidra
   - Crash reproduction and analysis

4. Findings
   - Global state in MPSNDArrayMatrixMultiplication
   - Unsafe encoding in MPSSetResourcesOnCommandEncoder
   - Comparison with thread-safe MLX approach

5. Impact
   - Affects all ML frameworks using MPS
   - Limits parallel inference to ~30% efficiency
   - Forces serialization via mutexes

6. Proposed Mitigations
   - Current: Global mutexes (PyTorch approach)
   - Alternative: Custom Metal kernels (MLX approach)
   - Ideal: Apple fixes internal state management

7. Responsible Disclosure
   - Submitted to Apple on [DATE]
   - Apple response: [RESPONSE]

8. Conclusion
```

## Timeline

| Week | Activity |
|------|----------|
| 1-2 | Extract MPS, set up Ghidra analysis |
| 3-4 | Identify global state, document crash path |
| 5 | Write detailed technical report |
| 6 | Submit to Apple via Feedback Assistant |
| 7-12 | Disclosure window (wait for Apple) |
| 13+ | Publish findings |

## Legal Contacts (If Needed)

For legal questions about security research:
- EFF (Electronic Frontier Foundation): https://www.eff.org/
- ACLU Technology & Civil Liberties: https://www.aclu.org/
- Stanford CIS: https://cyberlaw.stanford.edu/

## Conclusion

**Research and publication are LEGAL** under:
- DMCA security research exemption
- DMCA interoperability exception
- First Amendment protections

The key is **responsible disclosure** - submit to Apple first, give them time to respond, then publish.

This is exactly how security research works. Apple benefits from the research, users benefit from the fix, and researchers benefit from publication credit.
