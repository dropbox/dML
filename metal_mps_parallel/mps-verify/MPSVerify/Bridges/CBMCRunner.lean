/-
  MPSVerify.Bridges.CBMCRunner
  CBMC (C Bounded Model Checker) Process Runner

  CBMC is typically run as:
    cbmc <source.c> [options] --unwind N

  Common options:
  - --unwind N: Loop unwinding bound
  - --pointer-check: Enable pointer safety checks
  - --bounds-check: Enable array bounds checks
  - --memory-leak-check: Check for memory leaks
  - -I <path>: Include path for headers
-/

import MPSVerify.Bridges.CBMC
import MPSVerify.Core.Types

namespace MPSVerify.Bridges.CBMCRunner

open MPSVerify.Bridges.CBMC
open MPSVerify.Core

/-- Best-effort flush for long-running commands when stdout is piped (e.g. worker logs). -/
def flushStdout : IO Unit := do
  let out ← IO.getStdout
  out.flush

/-- Check if `timeout` is available (GNU coreutils on macOS: `brew install coreutils`). -/
def checkTimeoutAvailable : IO Bool := do
  try
    let result ← IO.Process.output { cmd := "which", args := #["timeout"] }
    return result.exitCode == 0
  catch _ =>
    return false

/-- CBMC execution options -/
structure CBMCOptions where
  unwind : Nat := 15               -- Loop unwinding bound (15 needed for complex harnesses)
  pointerCheck : Bool := true      -- Enable pointer checks
  boundsCheck : Bool := true       -- Enable bounds checks
  memoryLeakCheck : Bool := false  -- Check for memory leaks (slower)
  includePaths : List String := [] -- Include paths for headers
  timeout : Nat := 900             -- Timeout in seconds (some harnesses are slow)
  verbose : Bool := false          -- Extra output
  cwd : Option System.FilePath := none -- Working directory for CBMC execution
  -- macOS system include paths for standard library headers (stdlib.h, stdbool.h, etc.)
  -- These are needed because CBMC uses GCC-style preprocessing
  systemIncludePaths : List String := [
    "/Library/Developer/CommandLineTools/usr/lib/clang/17/include",  -- stdbool.h, stddef.h, etc.
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"  -- stdlib.h, etc.
  ]
  deriving Repr, Inhabited

/-- Default CBMC options -/
def CBMCOptions.default : CBMCOptions := {}

/-- Harness specification to verify -/
structure CBMCHarness where
  sourceFile : System.FilePath    -- Path to .c harness file
  name : String                   -- Human-readable name
  includePaths : List String := [] -- Additional include paths for this harness
  deriving Repr, Inhabited

/-- Check if CBMC is available as a command -/
def checkCBMCAvailable : IO Bool := do
  try
    let result ← IO.Process.output {
      cmd := "which"
      args := #["cbmc"]
    }
    return result.exitCode == 0
  catch _ =>
    return false

/-- Get CBMC version string -/
def getCBMCVersion : IO (Option String) := do
  try
    let result ← IO.Process.output {
      cmd := "cbmc"
      args := #["--version"]
    }
    if result.exitCode == 0 then
      let firstLine := result.stdout.splitOn "\n" |>.head!
      return some firstLine
    else
      return none
  catch _ =>
    return none

/-- Build CBMC command arguments -/
def buildCBMCArgs (harness : CBMCHarness) (opts : CBMCOptions) : Array String :=
  let base := #[harness.sourceFile.toString]

  -- Add system include paths (macOS SDK and clang builtins) for stdlib.h, stdbool.h, etc.
  let withSysInc := opts.systemIncludePaths.foldl (init := base) fun acc path =>
    acc ++ #["-I", path]

  -- Add include paths (from options and harness)
  let allIncludes := opts.includePaths ++ harness.includePaths
  let withIncludes := allIncludes.foldl (init := withSysInc) fun acc path =>
    acc ++ #["-I", path]

  -- Add unwind bound
  let withUnwind := withIncludes ++ #["--unwind", toString opts.unwind]

  -- Add optional checks
  let withPointer := if opts.pointerCheck
    then withUnwind ++ #["--pointer-check"]
    else withUnwind
  let withBounds := if opts.boundsCheck
    then withPointer ++ #["--bounds-check"]
    else withPointer
  let withMemory := if opts.memoryLeakCheck
    then withBounds ++ #["--memory-leak-check"]
    else withBounds

  withMemory

/-- Run CBMC and return parsed result -/
def runCBMC (harness : CBMCHarness) (opts : CBMCOptions := CBMCOptions.default) : IO CBMCResult := do
  -- Check CBMC availability
  let available ← checkCBMCAvailable
  if !available then
    return CBMCResult.failed "CBMC not found. Install CBMC (e.g., brew install cbmc)"

  let args := buildCBMCArgs harness opts
  if opts.verbose then
    IO.println s!"Running: cbmc {String.intercalate " " (args.toList)}"
    flushStdout

  try
    let useTimeout := opts.timeout > 0 && (← checkTimeoutAvailable)
    let (cmd, finalArgs) :=
      if useTimeout then
        ("timeout", #[toString opts.timeout, "cbmc"] ++ args)
      else
        ("cbmc", args)
    let result ← IO.Process.output {
      cmd := cmd
      args := finalArgs
      cwd := opts.cwd
    }
    let combinedOutput := result.stdout ++ result.stderr
    let parsed := parseCBMCOutput combinedOutput
    if useTimeout && result.exitCode == 124 then
      return { parsed with
        success := false
        rawOutput := s!"TIMEOUT after {opts.timeout}s\n\n{combinedOutput}"
      }
    return parsed
  catch e =>
    return CBMCResult.failed s!"CBMC execution failed: {e}"

/-- Run CBMC on a harness and return verification status -/
def verifyHarness (harness : CBMCHarness) (opts : CBMCOptions := CBMCOptions.default) : IO VerificationStatus := do
  let result ← runCBMC harness opts
  if result.success then
    return .verified
  else if result.failedAssertions == 0 then
    -- Failed but no specific failures (maybe timeout or crash)
    return .failed result.rawOutput
  else
    return .failed s!"CBMC found {result.failedAssertions} failures"

/-- Predefined harnesses in this project -/
def knownHarnesses : List CBMCHarness := [
  {
    sourceFile := "verification/cbmc/harnesses/aba_detection_harness.c"
    name := "ABA Detection"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/alloc_free_harness.c"
    name := "Alloc/Free"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/tls_cache_harness.c"
    name := "TLS Cache"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/stream_pool_harness.c"
    name := "Stream Pool"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/event_pool_harness.c"
    name := "Event Pool"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/batch_queue_harness.c"
    name := "Batch Queue"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/graph_cache_harness.c"
    name := "Graph Cache"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/command_buffer_harness.c"
    name := "Command Buffer"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/tls_binding_harness.c"
    name := "TLS Binding"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  },
  {
    sourceFile := "verification/cbmc/harnesses/fork_safety_harness.c"
    name := "Fork Safety"
    includePaths := ["verification/cbmc/models", "verification/cbmc/stubs"]
  }
]

/-- Find a harness by name -/
def findHarnessByName (name : String) : Option CBMCHarness :=
  knownHarnesses.find? (·.name.toLower == name.toLower)

/-- Run all known CBMC harnesses and return aggregated results -/
def runAllHarnesses (opts : CBMCOptions := CBMCOptions.default) : IO (List (String × CBMCResult)) := do
  let mut results : List (String × CBMCResult) := []
  for harness in knownHarnesses do
    IO.println s!"Verifying {harness.name}..."
    flushStdout
    let result ← runCBMC harness opts
    results := results ++ [(harness.name, result)]
    let statusStr :=
      if result.success then "PASS"
      else if result.rawOutput.startsWith "TIMEOUT" then "TIMEOUT"
      else "FAIL"
    IO.println s!"  {harness.name}: {statusStr} ({result.failedAssertions}/{result.totalAssertions} failed)"
  return results

/-- Summary of all verification results -/
def summarizeResults (results : List (String × CBMCResult)) : String :=
  let passed := results.filter (·.2.success) |>.length
  let total := results.length
  let totalAssertions := results.foldl (init := 0) fun acc (_, r) => acc + r.totalAssertions
  let failedAssertions := results.foldl (init := 0) fun acc (_, r) => acc + r.failedAssertions
  let header := s!"CBMC Verification Summary: {passed}/{total} harnesses passed, {failedAssertions}/{totalAssertions} assertions failed\n"

  let details := results.map fun (name, result) =>
    let status :=
      if result.success then "PASS"
      else if result.rawOutput.startsWith "TIMEOUT" then "TIMEOUT"
      else "FAIL"
    s!"  {name}: {status} ({result.failedAssertions}/{result.totalAssertions})"

  header ++ String.intercalate "\n" details

end MPSVerify.Bridges.CBMCRunner
