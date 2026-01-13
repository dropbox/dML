/-
  MPSVerify.Bridges.StaticAnalysis
  Static Analysis Tool Integration

  This module provides integration with:
  1. Clang Thread Safety Analysis (-Wthread-safety)
  2. Facebook Infer (racerd, starvation checkers)

  These tools provide lightweight continuous verification without
  the heavyweight setup of formal verification.
-/

import MPSVerify.Core.Types

namespace MPSVerify.Bridges.StaticAnalysis

open MPSVerify.Core

/-- Check if needle appears in haystack -/
def containsSubstr (haystack : String) (needle : String) : Bool :=
  decide ((haystack.splitOn needle).length > 1)

/-- Clang TSA warning severity levels -/
inductive WarningSeverity where
  | note : WarningSeverity
  | warning : WarningSeverity
  | error : WarningSeverity
  deriving Repr, BEq, Inhabited

/-- A single Clang warning/error -/
structure ClangWarning where
  file : String           -- Source file path
  line : Nat              -- Line number
  column : Nat            -- Column number
  severity : WarningSeverity
  code : String           -- Warning code (e.g., "-Wthread-safety-analysis")
  message : String        -- Warning message
  deriving Repr, Inhabited

/-- Result of Clang analysis -/
structure ClangResult where
  success : Bool
  warnings : List ClangWarning
  errors : List ClangWarning
  rawOutput : String
  deriving Repr, Inhabited

/-- Parse severity from string -/
def parseSeverity (s : String) : WarningSeverity :=
  if containsSubstr s "error" then .error
  else if containsSubstr s "warning" then .warning
  else .note

/-- Parse a Clang warning line
    Format: file:line:col: severity: message [-Wcode] -/
def parseClangLine (line : String) : Option ClangWarning := Id.run do
  -- Split on ": "
  let parts := line.splitOn ": "
  if parts.length < 3 then return none

  -- Parse location (file:line:col)
  let locParts := parts[0]!.splitOn ":"
  if locParts.length < 3 then return none

  let file := locParts[0]!
  let lineNum := locParts[1]!.toNat?.getD 0
  let col := locParts[2]!.toNat?.getD 0

  -- Parse severity
  let severity := parseSeverity parts[1]!

  -- Parse message and code
  let restParts := parts.drop 2
  let message := String.intercalate ": " restParts

  -- Extract warning code if present (e.g., [-Wthread-safety])
  let code := if containsSubstr message "[-W"
    then
      let startIdx := message.splitOn "[-W" |>.drop 1 |>.head?.getD ""
      let endIdx := startIdx.splitOn "]" |>.head?.getD ""
      s!"-W{endIdx}"
    else ""

  return some {
    file := file
    line := lineNum
    column := col
    severity := severity
    code := code
    message := message.trim
  }

/-- Parse Clang output -/
def parseClangOutput (output : String) : ClangResult := Id.run do
  let lines := output.splitOn "\n"
  let mut warnings : List ClangWarning := []
  let mut errors : List ClangWarning := []

  for line in lines do
    if let some warning := parseClangLine line then
      match warning.severity with
      | .error => errors := errors ++ [warning]
      | .warning => warnings := warnings ++ [warning]
      | .note => pure ()  -- Skip notes

  {
    success := errors.isEmpty
    warnings := warnings
    errors := errors
    rawOutput := output
  }

/-- Format Clang result for display -/
def ClangResult.format (result : ClangResult) : String :=
  let status := if result.success then "PASS" else "FAIL"
  let stats := s!"{result.warnings.length} warnings, {result.errors.length} errors"
  let header := s!"Clang TSA Result: {status} ({stats})"

  if result.warnings.isEmpty && result.errors.isEmpty then
    header
  else
    let formatWarning (w : ClangWarning) : String :=
      s!"  {w.file}:{w.line}:{w.column}: {w.message}"
    let warningStrs := result.warnings.map formatWarning
    let errorStrs := result.errors.map formatWarning
    let errSection := if result.errors.isEmpty then ""
      else "\nErrors:\n" ++ String.intercalate "\n" errorStrs
    let warnSection := if result.warnings.isEmpty then ""
      else "\nWarnings:\n" ++ String.intercalate "\n" warningStrs
    header ++ errSection ++ warnSection

/-- Infer issue type -/
inductive InferIssueType where
  | dataRace : InferIssueType        -- RacerD data race
  | deadlock : InferIssueType        -- Starvation deadlock
  | lockingViolation : InferIssueType
  | other : String → InferIssueType
  deriving Repr, Inhabited

/-- A single Infer issue -/
structure InferIssue where
  issueType : InferIssueType
  severity : String       -- "ERROR", "WARNING", etc.
  file : String
  line : Nat
  procedure : String      -- Function name
  description : String
  deriving Repr, Inhabited

/-- Result of Infer analysis -/
structure InferResult where
  success : Bool
  issues : List InferIssue
  rawOutput : String
  deriving Repr, Inhabited

/-- Static analysis options -/
structure AnalysisOptions where
  runClangTSA : Bool := true
  runInfer : Bool := true
  verbose : Bool := false
  timeout : Nat := 600    -- 10 minute timeout
  deriving Repr, Inhabited

/-- Combined result from all static analysis tools -/
structure StaticAnalysisResult where
  clang : Option ClangResult
  infer : Option InferResult
  deriving Repr, Inhabited

/-- Check if Clang is available -/
def checkClangAvailable : IO Bool := do
  try
    let result ← IO.Process.output {
      cmd := "which"
      args := #["clang++"]
    }
    return result.exitCode == 0
  catch _ =>
    return false

/-- Check if Infer is available -/
def checkInferAvailable : IO Bool := do
  try
    let result ← IO.Process.output {
      cmd := "which"
      args := #["infer"]
    }
    return result.exitCode == 0
  catch _ =>
    return false

/-- Run the analysis script and parse output -/
def runAnalysis (scriptPath : System.FilePath) (opts : AnalysisOptions := {}) : IO StaticAnalysisResult := do
  let args := if opts.runClangTSA && !opts.runInfer then #["--clang-only"]
              else if !opts.runClangTSA && opts.runInfer then #["--infer-only"]
              else #["--all"]

  try
    let result ← IO.Process.output {
      cmd := scriptPath.toString
      args := args
    }
    -- For now, return raw output; parsing can be added later
    -- when we have actual output format to work with
    let clangResult := if opts.runClangTSA
      then some (parseClangOutput result.stdout)
      else none

    return {
      clang := clangResult
      infer := none  -- TODO: Parse Infer JSON output
    }
  catch e =>
    return {
      clang := some (ClangResult.failed s!"Analysis failed: {e}")
      infer := none
    }
  where
    ClangResult.failed (msg : String) : ClangResult := {
      success := false
      warnings := []
      errors := []
      rawOutput := msg
    }

/-- Format combined result -/
def StaticAnalysisResult.format (result : StaticAnalysisResult) : String :=
  let clangSection := match result.clang with
    | some r => s!"## Clang Thread Safety Analysis\n{r.format}"
    | none => "## Clang Thread Safety Analysis\nNot run"

  let inferSection := match result.infer with
    | some _ => "## Facebook Infer\nCompleted (see JSON output)"
    | none => "## Facebook Infer\nNot run or not installed"

  s!"# Static Analysis Results\n\n{clangSection}\n\n{inferSection}"

end MPSVerify.Bridges.StaticAnalysis
