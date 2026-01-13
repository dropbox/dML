/-
  MPSVerify.DSL.Syntax
  Custom Syntax and Notation for MPS Verification DSL

  This module provides domain-specific notation for expressing
  concurrent patterns in a readable way.
-/

import MPSVerify.Core
import MPSVerify.DSL.StreamPool
import MPSVerify.DSL.Allocator

namespace MPSVerify.DSL.Syntax

open MPSVerify.Core
open MPSVerify.Core.Concurrency
open MPSVerify.Core.MemoryModel
open MPSVerify.DSL.StreamPool
open MPSVerify.DSL.Allocator

-- Use qualified type for clarity
abbrev TId := MPSVerify.Core.ThreadId

/-!
## Atomic Operation Notation

Provide concise notation for atomic operations with memory orderings.
-/

/-- Notation for atomic load with memory order -/
macro "atomic_load[" ord:term "]" x:term : term =>
  `(atomicLoad $x $ord)

/-- Notation for atomic store with memory order -/
macro "atomic_store[" ord:term "]" x:term "," v:term : term =>
  `(atomicStore $x $v $ord)

/-- Acquire memory order -/
abbrev acquire := MemoryOrder.acquire

/-- Release memory order -/
abbrev release := MemoryOrder.release

/-- Sequentially consistent memory order -/
abbrev seq_cst := MemoryOrder.seq_cst

/-- Relaxed memory order -/
abbrev relaxed := MemoryOrder.relaxed

/-!
## Stream Pool DSL

High-level operations on stream pool.
-/

/-- Get current thread's stream -/
def getCurrentStream (pool : PoolState) (thread : TId) : Option StreamId :=
  pool.tlsBindings thread

/-- Check if thread has a bound stream -/
def hasBoundStream (pool : PoolState) (thread : TId) : Bool :=
  (pool.tlsBindings thread).isSome

/-- Get stream count -/
def streamCount (pool : PoolState) : Nat :=
  pool.streams.length

/-- Get free stream count -/
def freeStreamCount (pool : PoolState) : Nat :=
  pool.freeList.length

/-!
## Allocator DSL

High-level operations on allocator.
-/

/-- Get buffer by ID -/
def getBuffer (state : AllocatorState) (id : BufferId) : Option BufferMeta :=
  state.mainPool.buffers.find? (·.id = id)

/-- Check if buffer is allocated -/
def isBufferAllocated (state : AllocatorState) (id : BufferId) : Bool :=
  match getBuffer state id with
  | some bufMeta => bufMeta.state = .allocated
  | none => false

/-- Get ABA generation for buffer -/
def getABAGeneration (state : AllocatorState) (id : BufferId) : Nat :=
  state.abaCounters id

/-!
## Verification DSL

DSL for expressing verification conditions.
-/

/-- Verification condition -/
inductive VerificationCondition where
  | noDataRace : List MemoryEvent → VerificationCondition
  | deadlockFree : VerificationCondition
  | abaDetected : BufferId → Nat → Nat → VerificationCondition
  | mutualExclusion : MutexState → VerificationCondition
  | progressGuarantee : VerificationCondition
  deriving Repr

/-- Result of checking a verification condition -/
inductive VCResult where
  | satisfied : VCResult
  | violated : String → VCResult
  | unknown : VCResult
  deriving Repr, Inhabited

/-- Check ABA detection condition -/
def checkABACondition (gen1 gen2 : Nat) : VCResult :=
  if gen1 < gen2 then .satisfied
  else .violated "ABA generation not incremented"

/-- Check mutual exclusion condition -/
def checkMutualExclusion (state : MutexState) : VCResult :=
  if state.recursionCount ≤ 1 || state.owner.isSome then .satisfied
  else .violated "Mutual exclusion violated"

/-!
## Invariant Macros

Macros for declaring invariants.
-/

/-- Declare a safety invariant -/
macro "safety_invariant" name:ident ":" type:term ":=" body:term : command =>
  `(def $name : $type := $body)

/-- Declare a liveness property -/
macro "liveness_property" name:ident ":" type:term ":=" body:term : command =>
  `(def $name : $type := $body)

/-!
## Thread Safety Annotations (DSL equivalent of Clang TSA)
-/

/-- Mark a resource as guarded by a mutex -/
structure GuardedBy (μ : MutexState) (α : Type) where
  value : α
  deriving Repr, Inhabited

/-- Acquire annotation for function precondition -/
structure AcquiresCapability (μ : MutexState) where
  deriving Repr, Inhabited

/-- Release annotation for function postcondition -/
structure ReleasesCapability (μ : MutexState) where
  deriving Repr, Inhabited

/-- Requires capability annotation -/
structure RequiresCapability (μ : MutexState) where
  deriving Repr, Inhabited

end MPSVerify.DSL.Syntax
