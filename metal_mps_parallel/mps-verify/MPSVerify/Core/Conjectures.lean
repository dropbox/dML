/-
  MPSVerify.Core.Conjectures
  Former conjectures, now proven theorems

  This module originally contained unproven conjectures. As of N=1343,
  the seq_cst race freedom theorem has been proven by extending the
  memory model to distinguish atomic from non-atomic operations.

  The proof leverages C++11 §1.10/21: "The execution of a program contains
  a data race if it contains two conflicting actions in different threads,
  at least one of which is not atomic, and neither happens before the other."

  Since all seq_cst operations are atomic, they cannot participate in data races.
-/

import MPSVerify.Core.MemoryModel

namespace MPSVerify.Core.Conjectures

open MPSVerify.Core.MemoryModel

/-
## seq_cst Race Freedom (PROVEN)

**THEOREM PROVEN (N=1343)**

Programs using only seq_cst operations are race-free. This is because:

1. All seq_cst operations are atomic (read, write, rmw, fence all have memory orders)
2. Data races require at least one non-atomic operation (C++11 §1.10/21)
3. Therefore, a trace of only seq_cst operations has no non-atomic operations
4. Therefore, no data race is possible

The proof is in MPSVerify.Core.MemoryModel as `seq_cst_race_free`.
-/
theorem seq_cst_race_free_conjecture (events : List MemoryEvent) :
    (∀ e ∈ events, e.op.order = .seq_cst) →
    isRaceFree events :=
  seq_cst_race_free events

end MPSVerify.Core.Conjectures
