# Verification Impact Statement Template

**Purpose:** This template formalizes the verification impact statement discipline required for any commit that touches concurrency protocol code.

---

## When to Use This Template

Use this template (or include answers in your commit message) when your commit:
- Modifies atomic operations or memory ordering
- Changes mutex/lock acquisition patterns
- Modifies thread-local storage bindings
- Changes callback/event lifetime management
- Modifies stream pool allocation/release logic
- Changes the batching architecture

---

## Required Questions

### 1. Which properties changed?

**Reference:** Property catalog in `VERIFICATION_TRACEABILITY.md`

| Property ID | Status | Notes |
|-------------|--------|-------|
| SP.001-SP.009 | unchanged / modified / added | |
| LP.001-LP.002 | unchanged / modified / added | |
| ST.001-ST.010 | unchanged / modified / added | |

If "modified" or "added", describe the change:
```
[Property changes description]
```

### 2. Which tools were re-run? What were the results?

| Tool | Target | Result | States/Checks |
|------|--------|--------|---------------|
| TLC (TLA+) | [spec name] | PASS/FAIL | [count] |
| CBMC | [harness name] | PASS/FAIL | [checks] |
| TSA | [file] | [warnings] | - |
| Structural | full suite | [pass/fail/warn] | [31 total] |
| Iris/Coq | [module] | builds/fails | - |

### 3. Did any assumption ledger entries change?

**Assumption categories:**
- Global mutexes (intentional serialization points)
- Apple MPS framework limitations
- Memory ordering assumptions
- Thread count limitations

| Assumption | Changed? | Old Value | New Value |
|------------|----------|-----------|-----------|
| [assumption] | yes/no | | |

### 4. What new artifacts were produced?

List paths to any new verification artifacts:
```
specs/[new-spec].tla
verification/cbmc/harnesses/[new-harness].c
mps-verify/states/[timestamp]/
reports/main/[report].md
```

---

## Example: Verification Impact Statement

```markdown
## Verification Impact Statement

1. **Properties changed:** None. This commit fixes a lock acquisition bug
   but doesn't change the verified protocol properties.

2. **Tools re-run:**
   - TLC: MPSStreamPool - PASS (7,981 states)
   - Structural: 31 checks, 26 pass, 0 failures, 5 warnings

3. **Assumptions changed:** No.

4. **Artifacts produced:**
   - mps-verify/states/25-12-19-12-30-00/verification_report.json
```

---

## Abbreviated Form for Commit Messages

For non-major changes, the abbreviated form is acceptable:

```
# 1303: Fix lock acquisition in commitAndWait

[commit description]

**Verification Impact:**
- Properties: unchanged
- Re-ran: Structural checks (31/26 pass, 0 fail)
- Assumptions: no change
- Artifacts: mps-verify/structural_check_results.json
```

---

## Checklist Integration

This template fulfills the requirement in `WORKER_VERIFICATION_PARAGON_CHECKLIST.md`:

> **Phase 1.4 - Adopt the "verification impact statement" discipline**
> - Outcome: Any commit touching concurrency protocol code includes a short verification impact statement.
> - Evidence: Commit message or `reports/main/*.md` entry answering the 4 questions.
> - Why: Prevents invisible regressions and spec drift (Design §5.6).

---

**Created:** N=1303 (2025-12-19)
