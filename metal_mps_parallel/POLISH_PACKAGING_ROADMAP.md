# Polish, Packaging, and Documentation Roadmap

**Created**: 2025-12-16
**Updated**: 2025-12-17 (N=1048: Re-verified all tests pass, documentation updated)
**Status**: 14/16 items DONE, 2 human actions remaining (#5 GitHub issue, #16 CLA)
**Audit**: See `UPSTREAM_SUBMISSION_AUDIT.md` for full PyTorch requirements checklist

---

## Overview

This roadmap identifies polish, packaging, and documentation improvements needed before upstream submission. Issues are prioritized by impact on user experience and submission readiness.

**PyTorch Requirements Sources**:
- `pytorch-mps-fork/CONTRIBUTING.md`
- https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions
- https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines

---

## Issue Summary

| # | Priority | Issue | Type | Status |
|---|----------|-------|------|--------|
| 1 | P0 | Tests not integrated into PyTorch's `test/test_mps.py` | Testing | ✅ DONE (10e734a0) |
| 2 | P0 | `make lint` not verified on modified files | Quality | ✅ clang-format applied (4002a2c0) |
| 3 | P0 | Graph compilation stress test failing (segfault) | Quality | ✅ FIXED (N=954 passes) |
| 4 | P0 | README.md missing Table of Contents with document links | Documentation | ✅ DONE (N=954) |
| 5 | P1 | No linked GitHub issue on pytorch/pytorch | Process | Pending (human action) |
| 6 | P1 | No CONTRIBUTING.md for this repo | Documentation | ✅ DONE (N=954) |
| 7 | P1 | No LICENSE file in this repo (inherit PyTorch BSD-3-Clause) | Packaging | ✅ DONE (N=954) |
| 8 | P1 | README.md missing Quick Start build instructions | Documentation | ✅ DONE (N=954) |
| 9 | P2 | CI compatibility not verified (macOS runners) | Testing | ✅ DOCUMENTED (N=955) |
| 10 | P2 | patches/README.md is 341 lines - needs summary version | Documentation | ✅ DONE (N=955) |
| 11 | P2 | BLOG_POST.md incorrect facts ("60+ bugs" should be "201 issues") | Documentation | ✅ FIXED (N=961) |
| 12 | P2 | No CHANGELOG.md for version history | Documentation | ✅ DONE (N=954) |
| 13 | P3 | Python API `release_current_thread_slot()` docstring verification | Documentation | ✅ VERIFIED (N=954) |
| 14 | P3 | Historical reports (114 files) need archival strategy | Packaging | ✅ DONE (N=955) |
| 15 | P3 | PR description template needs preparation | Process | ✅ DONE (N=955) |
| 16 | P3 | CLA signing reminder for human submitter | Process | N/A |

---

## Detailed Issues

### P0-1: README.md Missing Table of Contents

**Location**: `/README.md`
**Problem**: Key documents (`FINAL_COMPLETION_REPORT.md`, `AI_TECHNICAL_SPEC.md`, `WORKER_DIRECTIVE.md`, `archive/WORKER_DIRECTIVE_HISTORICAL.md`, `patches/README.md`, `tests/README.md`, `BLOG_POST.md`) are not linked from README.md.
**Impact**: Users cannot discover documentation.
**Fix**: Add "Documentation" section at top with ToC.

**Current state** (verified):
```
README.md has 12 sections but NO table of contents
Key documents exist but are not linked:
- FINAL_COMPLETION_REPORT.md (created 2025-12-16)
- AI_TECHNICAL_SPEC.md (101 lines)
- archive/WORKER_DIRECTIVE_HISTORICAL.md (201 issues documented)
- patches/README.md (341 lines, patch details)
- tests/README.md (96 lines, test suite docs)
- BLOG_POST.md (case study)
- apple_feedback/ (Apple Feedback submission package)
```

---

### P0-2: Graph Compilation Stress Test Failing

**Location**: `tests/test_graph_compilation_stress.py`
**Problem**: Test crashed with segmentation fault when run in full suite.
**Status**: ✅ RESOLVED - Test now passes (N=970 verified)
**Resolution**: The segfault was caused by concurrent Metal shader compilation. Fixed by proper synchronization. Test now passes consistently.

---

### P1-1: No CONTRIBUTING.md

**Location**: Missing `/CONTRIBUTING.md`
**Problem**: No guidance for contributors on how to:
- Apply the patch
- Run tests
- Submit changes
- Code style requirements
**Impact**: Upstream reviewers need clear contribution process.
**Fix**: Create CONTRIBUTING.md with standard sections.

---

### P1-2: No LICENSE File

**Location**: Missing `/LICENSE`
**Problem**: No explicit license file. This repo forks PyTorch (BSD-3-Clause).
**Impact**: Unclear licensing for users and contributors.
**Fix**: Add LICENSE file with BSD-3-Clause (matching PyTorch).

---

### P1-3: README Missing Build Instructions

**Location**: `/README.md` "Quick Start" section (lines 231-240)
**Problem**: Quick Start section exists but lacks complete build steps.
**Current state** (verified from README.md):
```markdown
## Quick Start
export MPS_FORCE_GRAPH_PATH=1
python your_parallel_inference_script.py
```
**Missing**:
- How to clone the fork
- How to apply the patch
- How to build PyTorch from source
- How to verify the build
**Fix**: Add complete build instructions.

---

### P2-1: patches/README.md Too Long

**Location**: `/patches/README.md` (341 lines)
**Problem**: Contains extensive phase-by-phase history that's valuable for developers but overwhelming for users who just want to apply the patch.
**Impact**: Hard to find essential "how to apply" information.
**Fix**: Create condensed summary at top; move history to separate PATCH_HISTORY.md.

---

### P2-2: BLOG_POST.md Incorrect Facts

**Location**: `/BLOG_POST.md` line 3
**Problem**: States "fixing 60+ latent bugs" but actual count is **201 issues** (32.110-32.310).
**Evidence**: archive/WORKER_DIRECTIVE_HISTORICAL.md header states "Total Issues: 201"
**Fix**: Update blog post with accurate count.

---

### P2-3: No CHANGELOG.md

**Location**: Missing `/CHANGELOG.md`
**Problem**: No formal version history. Changes are documented in patches/README.md but not in standard changelog format.
**Impact**: Users cannot easily see what changed between versions.
**Fix**: Create CHANGELOG.md with semantic versioning.

---

### P3-1: Python API Missing Docstring

**Location**: `pytorch-mps-fork/torch/mps/__init__.py`
**Function**: `release_current_thread_slot()`
**Problem**: Function added (Phase 29.1, N=278) but may lack proper docstring.
**Fix**: Verify and add docstring if missing.

---

### P3-2: Historical Reports Need Archival

**Location**: `/reports/main/` (114 files)
**Problem**: 114 verification reports from development process. Valuable for audit trail but clutters repo.
**Current state** (verified):
```
ls reports/main/*.md | wc -l
114
```
**Options**:
1. Keep as-is (complete audit trail)
2. Archive to `reports/archive/` with summary
3. Generate combined report and archive originals
**Fix**: Document decision in CONTRIBUTING.md.

---

## Implementation Priority

### Phase 1: Submission Blockers (P0)
1. Fix graph compilation stress test OR document as known issue
2. Add Table of Contents to README.md

### Phase 2: Upstream Requirements (P1)
3. Create CONTRIBUTING.md
4. Add LICENSE file
5. Add complete build instructions to README.md

### Phase 3: Polish (P2)
6. Simplify patches/README.md
7. Fix BLOG_POST.md facts
8. Create CHANGELOG.md

### Phase 4: Nice-to-Have (P3)
9. Add Python API docstrings
10. Document reports archival strategy

---

## Verification Checklist

After completing all issues:

- [x] README.md has ToC with all document links (N=954)
- [x] All tests pass (or failures documented as known issues) (N=958: 24/24 PASS, TSan 31t/100i 0 races)
- [x] CONTRIBUTING.md exists with clear guidance (N=954)
- [x] LICENSE file exists (BSD-3-Clause) (N=954)
- [x] Build instructions are complete and tested (N=954)
- [x] patches/README.md has clear "quick apply" section (N=955)
- [x] BLOG_POST.md facts are accurate (N=954)
- [x] CHANGELOG.md exists (N=954)
- [x] Python API has docstrings (N=954, verified)
- [x] Reports archival strategy documented (N=955)
