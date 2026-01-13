/-
  MPSVerify.Core.Types
  Core type definitions for the MPS Verification Platform

  This module defines the fundamental types used across all verification
  components: thread identifiers, memory locations, values, and timestamps.
-/

namespace MPSVerify.Core

/-- Thread identifier -/
abbrev ThreadId := Nat

/-- Memory location identifier -/
abbrev Location := Nat

/-- Abstract value type for verification -/
inductive Value where
  | unit : Value
  | bool : Bool → Value
  | nat : Nat → Value
  | int : Int → Value
  | ptr : Location → Value
  | null : Value
  deriving Repr, DecidableEq, Inhabited

/-- Timestamp for memory ordering -/
abbrev Timestamp := Nat

/-- Stream identifier (0 = default, 1-31 = worker streams) -/
structure StreamId where
  id : Fin 32
  deriving Repr, DecidableEq, Inhabited

/-- Default stream (stream 0) -/
def StreamId.default : StreamId := ⟨0, by decide⟩

/-- Check if this is the default stream -/
def StreamId.isDefault (s : StreamId) : Bool := s.id = 0

/-- Check if this is a worker stream -/
def StreamId.isWorker (s : StreamId) : Bool := s.id ≠ 0

/-- Buffer block identifier for allocator verification -/
structure BufferId where
  id : Nat
  generation : Nat  -- For ABA detection
  deriving Repr, DecidableEq, Inhabited

/-- Event identifier for synchronization -/
structure EventId where
  id : Nat
  counter : Nat
  deriving Repr, DecidableEq, Inhabited

/-- Verification result status -/
inductive VerificationStatus where
  | verified : VerificationStatus
  | failed : String → VerificationStatus
  | timeout : VerificationStatus
  | unknown : VerificationStatus
  deriving Repr, Inhabited

/-- Property being verified -/
structure Property where
  name : String
  description : String
  priority : Nat  -- 1 = HIGH, 2 = MEDIUM, 3 = LOW
  deriving Repr, Inhabited

/-- File dependency for incremental verification -/
structure FileDependency where
  specFile : String
  sourceFiles : List String
  verifiers : List String
  deriving Repr, Inhabited

/-- Hash for file change detection -/
abbrev FileHash := String

/-- Verification cache entry -/
structure CacheEntry where
  fileHash : FileHash
  status : VerificationStatus
  timestamp : Nat
  deriving Repr, Inhabited

end MPSVerify.Core
