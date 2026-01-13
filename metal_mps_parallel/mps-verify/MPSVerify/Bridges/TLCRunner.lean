/-
  MPSVerify.Bridges.TLCRunner
  TLC (TLA+ Model Checker) Process Runner

  This module handles executing TLC as an external process and
  collecting its output. TLC is typically run via:

    java -jar tla2tools.jar <spec>.tla -config <spec>.cfg

  Or if TLC is available as a command:

    tlc <spec>.tla -config <spec>.cfg

  The runner:
  1. Checks if TLC is available (tlc command or tla2tools.jar)
  2. Executes TLC with appropriate arguments
  3. Captures stdout/stderr
  4. Parses output using TLAPlus.parseTLCOutput
  5. Returns structured TLCResult
-/

import MPSVerify.Bridges.TLAPlus
import MPSVerify.Core.Types

namespace MPSVerify.Bridges.TLCRunner

open MPSVerify.Bridges.TLAPlus
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

/-- TLC execution options -/
structure TLCOptions where
  timeoutSecs : Nat := 1800      -- 30 minute default timeout (MPSEvent.tla can be slow)
  workers : Nat := 4             -- Parallel workers (TLC scales well on multi-core)
  checkDeadlock : Bool := true   -- Check for deadlocks
  checkLiveness : Bool := false  -- Check liveness properties (slower)
  verbose : Bool := false        -- Extra output
  cwd : Option System.FilePath := none -- Working directory for TLC execution
  deriving Repr, Inhabited

/-- Default TLC options -/
def TLCOptions.default : TLCOptions := {}

/-- Specification to verify -/
structure TLCSpec where
  specFile : System.FilePath    -- Path to .tla file
  configFile : System.FilePath  -- Path to .cfg file
  name : String                 -- Human-readable name
  deriving Repr, Inhabited

/-- Check if TLC is available as a command -/
def checkTLCCommand : IO Bool := do
  try
    let result ← IO.Process.output {
      cmd := "which"
      args := #["tlc"]
    }
    return result.exitCode == 0
  catch _ =>
    return false

/-- Check if tla2tools.jar exists at common locations -/
def findTLA2Tools : IO (Option System.FilePath) := do
  -- First, try to find relative to executable (vendored jar in project)
  -- Executable is at .lake/build/bin/mpsverify, jar is at tools/tla2tools.jar
  let appPath ← IO.appPath
  let appDir := appPath.parent.getD (System.FilePath.mk ".")
  let projectRoot := appDir / ".." / ".." / ".."  -- .lake/build/bin -> project root
  let vendoredJar := projectRoot / "tools" / "tla2tools.jar"

  -- Try vendored jar first (normalized path)
  if ← vendoredJar.pathExists then
    return some vendoredJar

  -- Standard locations as fallback
  let locations : List System.FilePath := [
    "/usr/local/lib/tla2tools.jar",
    "/opt/tla2tools.jar",
    (System.FilePath.mk (← IO.getEnv "HOME" |>.map (·.getD "~"))) / ".tla" / "tla2tools.jar",
    (System.FilePath.mk ".") / "tools" / "tla2tools.jar",
    (System.FilePath.mk ".") / "tla2tools.jar"
  ]
  for loc in locations do
    if ← loc.pathExists then
      return some loc
  return none

/-- Find vendored or system Java -/
def findJava : IO (Option System.FilePath) := do
  -- First, try to find vendored JDK relative to executable
  let appPath ← IO.appPath
  let appDir := appPath.parent.getD (System.FilePath.mk ".")
  let projectRoot := appDir / ".." / ".." / ".."
  -- Check for macOS JDK bundle structure
  let vendoredJava := projectRoot / "tools" / "jdk-21.0.2.jdk" / "Contents" / "Home" / "bin" / "java"
  if ← vendoredJava.pathExists then
    return some vendoredJava

  -- Also check cwd-relative path
  let cwdVendored := (System.FilePath.mk ".") / "tools" / "jdk-21.0.2.jdk" / "Contents" / "Home" / "bin" / "java"
  if ← cwdVendored.pathExists then
    return some cwdVendored

  -- Check if system java exists
  try
    let result ← IO.Process.output {
      cmd := "which"
      args := #["java"]
    }
    if result.exitCode == 0 && !result.stdout.trim.isEmpty then
      return some (System.FilePath.mk result.stdout.trim)
  catch _ =>
    pure ()

  return none

/-- Result of TLC availability check -/
inductive TLCAvailability where
  | tlcCommand : TLCAvailability           -- tlc command available
  | jarFile : System.FilePath → System.FilePath → TLCAvailability  -- (jarPath, javaPath)
  | jarOnly : System.FilePath → TLCAvailability  -- jar found but no java
  | notFound : TLCAvailability             -- TLC not installed
  deriving Repr

/-- Check TLC availability -/
def checkTLCAvailability : IO TLCAvailability := do
  if ← checkTLCCommand then
    return .tlcCommand
  if let some jarPath ← findTLA2Tools then
    if let some javaPath ← findJava then
      return .jarFile jarPath javaPath
    else
      return .jarOnly jarPath
  return .notFound

/-- Build TLC command arguments -/
def buildTLCArgs (spec : TLCSpec) (opts : TLCOptions) : Array String :=
  let base := #[spec.specFile.toString, "-config", spec.configFile.toString]
  let withWorkers := if opts.workers > 1
    then base ++ #["-workers", toString opts.workers]
    else base
  let withDeadlock := if !opts.checkDeadlock
    then withWorkers ++ #["-deadlock"]
    else withWorkers
  withDeadlock

/-- Run TLC and return parsed result -/
def runTLC (spec : TLCSpec) (opts : TLCOptions := TLCOptions.default) : IO TLCResult := do
  -- Check TLC availability
  let availability ← checkTLCAvailability

  match availability with
  | .notFound =>
    return TLCResult.failed "TLC not found. Install TLA+ tools or set tla2tools.jar path."

  | .jarOnly _ =>
    return TLCResult.failed "TLC jar found but no Java runtime. Install Java or use vendored JDK."

  | .tlcCommand =>
    let args := buildTLCArgs spec opts
    if opts.verbose then
      IO.println s!"Running: tlc {String.intercalate " " (args.toList)}"
      flushStdout

    try
      let useTimeout := opts.timeoutSecs > 0 && (← checkTimeoutAvailable)
      let (cmd, finalArgs) :=
        if useTimeout then
          ("timeout", #[toString opts.timeoutSecs, "tlc"] ++ args)
        else
          ("tlc", args)
      let result ← IO.Process.output {
        cmd := cmd
        args := finalArgs
        cwd := opts.cwd
      }
      let combinedOutput := result.stdout ++ result.stderr
      let parsed := parseTLCOutput combinedOutput
      if useTimeout && result.exitCode == 124 then
        return { parsed with
          success := false
          rawOutput := s!"TIMEOUT after {opts.timeoutSecs}s\n\n{combinedOutput}"
        }
      return parsed
    catch e =>
      return TLCResult.failed s!"TLC execution failed: {e}"

  | .jarFile jarPath javaPath =>
    let args := buildTLCArgs spec opts
    let javaArgs := #["-XX:+UseParallelGC", "-jar", jarPath.toString] ++ args
    if opts.verbose then
      IO.println s!"Running: {javaPath.toString} {String.intercalate " " (javaArgs.toList)}"
      flushStdout

    try
      let useTimeout := opts.timeoutSecs > 0 && (← checkTimeoutAvailable)
      let (cmd, finalArgs) :=
        if useTimeout then
          ("timeout", #[toString opts.timeoutSecs, javaPath.toString] ++ javaArgs)
        else
          (javaPath.toString, javaArgs)
      let result ← IO.Process.output {
        cmd := cmd
        args := finalArgs
        cwd := opts.cwd
      }
      let combinedOutput := result.stdout ++ result.stderr
      let parsed := parseTLCOutput combinedOutput
      if useTimeout && result.exitCode == 124 then
        return { parsed with
          success := false
          rawOutput := s!"TIMEOUT after {opts.timeoutSecs}s\n\n{combinedOutput}"
        }
      return parsed
    catch e =>
      return TLCResult.failed s!"TLC execution failed: {e}"

/-- Run TLC on a spec and return verification status -/
def verifySpec (spec : TLCSpec) (opts : TLCOptions := TLCOptions.default) : IO VerificationStatus := do
  let result ← runTLC spec opts
  if result.success then
    return .verified
  else if result.violations.isEmpty then
    -- Failed but no specific violations (maybe timeout or crash)
    return .failed result.rawOutput
  else
    let violationNames := result.violations.map (·.invariant) |> String.intercalate ", "
    return .failed s!"TLC found violations: {violationNames}"

/-- Predefined specs in this project -/
def knownSpecs : List TLCSpec := [
  -- Core MPS specs (original 3)
  {
    specFile := "specs/MPSStreamPool.tla"
    configFile := "specs/MPSStreamPool.cfg"
    name := "MPSStreamPool"
  },
  {
    specFile := "specs/MPSAllocator.tla"
    configFile := "specs/MPSAllocator.cfg"
    name := "MPSAllocator"
  },
  {
    specFile := "specs/MPSEvent.tla"
    configFile := "specs/MPSEvent.cfg"
    name := "MPSEvent"
  },
  -- Extended specs (added N=1392)
  {
    specFile := "specs/MPSBatchQueue.tla"
    configFile := "specs/MPSBatchQueue.cfg"
    name := "MPSBatchQueue"
  },
  {
    specFile := "specs/MPSCommandBuffer.tla"
    configFile := "specs/MPSCommandBuffer.cfg"
    name := "MPSCommandBuffer"
  },
  {
    specFile := "specs/MPSForkHandler.tla"
    configFile := "specs/MPSForkHandler.cfg"
    name := "MPSForkHandler"
  },
  {
    specFile := "specs/MPSGraphCache.tla"
    configFile := "specs/MPSGraphCache.cfg"
    name := "MPSGraphCache"
  },
  {
    specFile := "specs/MPSKernelCache.tla"
    configFile := "specs/MPSKernelCache.cfg"
    name := "MPSKernelCache"
  },
  {
    specFile := "specs/MPSTLSBinding.tla"
    configFile := "specs/MPSTLSBinding.cfg"
    name := "MPSTLSBinding"
  },
  {
    specFile := "specs/MPSFullSystem.tla"
    configFile := "specs/MPSFullSystem.cfg"
    name := "MPSFullSystem"
  }
]

/-- Find a spec by name -/
def findSpecByName (name : String) : Option TLCSpec :=
  knownSpecs.find? (·.name.toLower == name.toLower)

/-- Run all known TLA+ specs and return aggregated result -/
def runAllSpecs (opts : TLCOptions := TLCOptions.default) : IO (List (String × TLCResult)) := do
  let mut results : List (String × TLCResult) := []
  for spec in knownSpecs do
    IO.println s!"Verifying {spec.name}..."
    flushStdout
    let result ← runTLC spec opts
    results := results ++ [(spec.name, result)]
    let statusStr :=
      if result.success then "PASS"
      else if result.rawOutput.startsWith "TIMEOUT" then "TIMEOUT"
      else "FAIL"
    IO.println s!"  {spec.name}: {statusStr} ({result.statesGenerated} states, {result.timeMs}ms)"
  return results

/-- Summary of all verification results -/
def summarizeResults (results : List (String × TLCResult)) : String :=
  let passed := results.filter (·.2.success) |>.length
  let total := results.length
  let header := s!"TLA+ Verification Summary: {passed}/{total} passed\n"

  let details := results.map fun (name, result) =>
    let status :=
      if result.success then "PASS"
      else if result.rawOutput.startsWith "TIMEOUT" then "TIMEOUT"
      else "FAIL"
    s!"  {name}: {status}"

  header ++ String.intercalate "\n" details

end MPSVerify.Bridges.TLCRunner
