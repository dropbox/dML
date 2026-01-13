/-
  MPSVerify - MPS Parallel Inference Verification Platform

  A multi-language verification platform for the MPS parallel inference
  implementation. Integrates TLA+, Lean proofs, Iris/Coq, CBMC, and
  static analysis tools.

  Author: MPS Parallel Inference Project
  License: MIT
-/

-- Core definitions
import MPSVerify.Core
-- Domain-specific language for MPS patterns
import MPSVerify.DSL
-- Custom verification tactics
import MPSVerify.Tactics
-- External tool bridges
import MPSVerify.Bridges
-- AGX driver race condition proofs
import MPSVerify.AGX
