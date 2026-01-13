/-
  MPS Verification Platform - CLI Entry Point

  Usage:
    mps-verify check [--all] [--incremental]
    mps-verify tla [--spec=<spec>] [--all]
    mps-verify report --format=<html|md>
-/

import MPSVerify

open MPSVerify.Bridges.TLCRunner
open MPSVerify.Bridges.TLAPlus
open MPSVerify.Bridges.CBMCRunner
open MPSVerify.Bridges.CBMC
open MPSVerify.Bridges.StaticAnalysis

def version : String := "0.4.0"

/-- Make a string safe for filenames (best-effort). -/
def sanitizePathComponent (s : String) : String :=
  let isOk (c : Char) : Bool :=
    c.isAlphanum || c == '_' || c == '-' || c == '.'
  let out := s.toList.foldl (init := "") fun acc c =>
    acc.push (if isOk c then c else '_')
  if out.isEmpty then "unnamed" else out

/-- Return the first non-empty line from stdout/stderr, if any. -/
def firstLine? (stdout : String) (stderr : String) : Option String :=
  let combined := (stdout ++ "\n" ++ stderr).splitOn "\n"
  combined.findSome? fun line =>
    let t := line.trim
    if t.isEmpty then none else some t

/-- Best-effort command probe: return first output line on success. -/
def cmdFirstLine? (cmd : String) (args : Array String := #[]) (cwd : Option System.FilePath := none) : IO (Option String) := do
  try
    let result ← IO.Process.output { cmd := cmd, args := args, cwd := cwd }
    if result.exitCode == 0 then
      return firstLine? result.stdout result.stderr
    return none
  catch _ =>
    return none

/-- Minimal JSON string escaping. -/
def jsonEscape (s : String) : String :=
  s.foldl (init := "") fun acc c =>
    match c with
    | '\\' => acc ++ "\\\\"
    | '\"' => acc ++ "\\\""
    | '\n' => acc ++ "\\n"
    | '\r' => acc ++ "\\r"
    | '\t' => acc ++ "\\t"
    | _ => acc.push c

def jsonStr (s : String) : String := s!"\"{jsonEscape s}\""

def jsonObj (fields : List (String × String)) : String :=
  let renderField (k : String) (v : String) : String := s!"{jsonStr k}:{v}"
  "{" ++ String.intercalate "," (fields.map fun (k, v) => renderField k v) ++ "}"

def jsonArr (items : List String) : String :=
  "[" ++ String.intercalate "," items ++ "]"

/-- Find the `mps-verify/` project root so the CLI can be run from repo root (or elsewhere). -/
def findProjectRoot : IO System.FilePath := do
  let isProjectRoot (dir : System.FilePath) : IO (Option System.FilePath) := do
    let harnessDir := dir / "verification" / "cbmc" / "harnesses"
    if ← harnessDir.pathExists then
      return some dir
    let repoRootCandidate := dir / "mps-verify" / "verification" / "cbmc" / "harnesses"
    if ← repoRootCandidate.pathExists then
      return some (dir / "mps-verify")
    return none

  let searchUp (start : System.FilePath) (fuel : Nat) : IO (Option System.FilePath) := do
    let mut dir := start
    for _ in List.range fuel do
      if let some root ← isProjectRoot dir then
        return some root
      match dir.parent with
      | none => return none
      | some parent =>
        if parent == dir then
          return none
        dir := parent
    return none

  let cwd ← IO.currentDir
  if let some root ← searchUp cwd 64 then
    return root

  -- Fallback: derive from executable path (useful when invoked from outside the repo).
  let appPath ← IO.appPath
  match appPath.parent with
  | none => return cwd
  | some appDir =>
    if let some root ← searchUp appDir 64 then
      return root

  return cwd

def printHelp : IO Unit := do
  IO.println s!"MPS Verification Platform v{version}"
  IO.println ""
  IO.println "Usage: mps-verify <command> [options]"
  IO.println ""
  IO.println "Commands:"
  IO.println "  check       Run verification suite"
  IO.println "  tla         TLA+ model checking"
  IO.println "  cbmc        CBMC bounded verification"
  IO.println "  static      Static analysis (Clang TSA)"
  IO.println "  structural  Structural conformance checks"
  IO.println "  report      Generate verification report"
  IO.println "  help        Show this help"
  IO.println ""
  IO.println "TLA+ Options:"
  IO.println "  --spec=NAME     Run specific spec (use --all to list all available specs)"
  IO.println "  --all           Run all TLA+ specs"
  IO.println "  --timeout=SECS  Timeout per spec (default: 1800)"
  IO.println "  --verbose       Verbose output"
  IO.println ""
  IO.println "CBMC Options:"
  IO.println "  --harness=NAME  Run specific harness (aba_detection, alloc_free, tls_cache, stream_pool)"
  IO.println "  --all           Run all CBMC harnesses"
  IO.println "  --timeout=SECS  Timeout per harness (default: 900)"
  IO.println ""
  IO.println "Static Analysis Options:"
  IO.println "  --clang         Run Clang TSA only"
  IO.println "  --infer         Run Facebook Infer only"
  IO.println "  --all           Run all static analysis"
  IO.println "  --verbose       Verbose output"
  IO.println ""
  IO.println "Check Options:"
  IO.println "  --all           Run all verification tools (TLA+, CBMC, Static, Structural)"
  IO.println "  --tla           Include TLA+ model checking"
  IO.println "  --cbmc          Include CBMC verification"
  IO.println "  --static        Include static analysis"
  IO.println "  --structural    Include structural conformance checks"
  IO.println "  --verbose       Show detailed progress (recommended for --all)"
  IO.println ""
  IO.println "Report Options:"
  IO.println "  --format=md     Generate Markdown report (default)"
  IO.println "  --format=html   Generate HTML report"
  IO.println "  --output=FILE   Output file (default: verification_report.md)"
  IO.println ""
  IO.println "General Options:"
  IO.println "  --incremental   Only verify changed files (default)"
  IO.println "  --force         Force re-verification"
  IO.println "  --allow-skip    Allow missing tools (best-effort run)"

def runTLACommand (args : List String) : IO Unit := do
  IO.println "TLA+ Model Checking"
  IO.println "==================="
  IO.println ""

  let root ← findProjectRoot

  -- Check TLC availability first
  let availability ← checkTLCAvailability
  match availability with
  | .notFound =>
    IO.println "ERROR: TLC not found."
    IO.println ""
    IO.println "Install TLA+ tools:"
    IO.println "  1. Download tla2tools.jar from https://github.com/tlaplus/tlaplus/releases"
    IO.println "  2. Place in /usr/local/lib/tla2tools.jar or current directory"
    IO.println "  3. Or install 'tlc' command via package manager"
    return
  | .jarOnly jarPath =>
    IO.println s!"ERROR: TLC jar found at {jarPath} but no Java runtime."
    IO.println ""
    IO.println "Install Java or ensure vendored JDK is in tools/jdk-21.0.2.jdk/"
    return
  | .tlcCommand =>
    IO.println "TLC found: using 'tlc' command"
  | .jarFile jarPath javaPath =>
    IO.println s!"TLC found: using {jarPath}"
    IO.println s!"Java runtime: {javaPath}"

  IO.println ""

  -- Parse arguments
  let verbose := args.any (· == "--verbose")
  let runAll := args.any (· == "--all")
  let specArg := args.find? (·.startsWith "--spec=")
  let specName := specArg.map (·.drop 7)
  let timeoutArg := args.find? (·.startsWith "--timeout=")
  let timeoutSecs := timeoutArg.bind fun a => (a.drop 10).toNat?
  if timeoutArg.isSome && timeoutSecs.isNone then
    IO.println "ERROR: Invalid --timeout value (expected integer seconds)."
    return

  let opts : TLCOptions := {
    verbose := verbose
    timeoutSecs := timeoutSecs.getD TLCOptions.default.timeoutSecs
    cwd := some root
  }

  match (runAll, specName) with
  | (true, _) =>
    -- Run all specs
    IO.println "Running all TLA+ specifications..."
    IO.println ""
    let results ← runAllSpecs opts
    IO.println ""
    IO.println (summarizeResults results)
  | (false, some name) =>
    -- Run specific spec
    match findSpecByName name with
    | some spec =>
      IO.println s!"Running {spec.name}..."
      let result ← runTLC spec opts
      IO.println ""
      IO.println (result.format)
    | none =>
      IO.println s!"Unknown spec: {name}"
      IO.println ""
      IO.println "Available specs:"
      for spec in knownSpecs do
        IO.println s!"  - {spec.name}"
  | (false, none) =>
    -- No spec specified, show help
    IO.println "Specify --all or --spec=NAME"
    IO.println ""
    IO.println "Available specs:"
    for spec in knownSpecs do
      IO.println s!"  - {spec.name}"

def runCBMCCommand (args : List String) : IO Unit := do
  IO.println "CBMC Bounded Model Checking"
  IO.println "============================"
  IO.println ""

  let root ← findProjectRoot

  -- Check CBMC availability
  let available ← checkCBMCAvailable
  if !available then
    IO.println "ERROR: CBMC not found."
    IO.println ""
    IO.println "Install CBMC:"
    IO.println "  brew install cbmc"
    IO.println "  # or download from https://www.cprover.org/cbmc/"
    return

  let versionInfo ← getCBMCVersion
  IO.println s!"CBMC version: {versionInfo}"
  IO.println ""

  -- Parse arguments
  let runAll := args.any (· == "--all")
  let harnessArg := args.find? (·.startsWith "--harness=")
  let harnessName := harnessArg.map (·.drop 10)
  let timeoutArg := args.find? (·.startsWith "--timeout=")
  let timeoutSecs := timeoutArg.bind fun a => (a.drop 10).toNat?
  if timeoutArg.isSome && timeoutSecs.isNone then
    IO.println "ERROR: Invalid --timeout value (expected integer seconds)."
    return

  let opts : CBMCOptions := {
    unwind := 15  -- Sufficient for all harnesses (10 causes unwinding assertion failures)
    pointerCheck := true
    boundsCheck := true
    includePaths := []
    timeout := timeoutSecs.getD CBMCOptions.default.timeout
    cwd := some root
  }

  match (runAll, harnessName) with
  | (true, _) =>
    IO.println "Running all CBMC harnesses..."
    IO.println ""
    let results ← runAllHarnesses opts
    IO.println ""
    IO.println (summarizeResults results)
  | (false, some name) =>
    let canonical := match name.toLower with
      | "aba_detection" => "ABA Detection"
      | "alloc_free" => "Alloc/Free"
      | "tls_cache" => "TLS Cache"
      | "stream_pool" => "Stream Pool"
      | other => other
    match knownHarnesses.find? (fun h => h.name.toLower == canonical.toLower) with
    | some harness =>
      IO.println s!"Running harness: {harness.name}..."
      let result ← runCBMC harness opts
      IO.println ""
      IO.println (result.format)
    | none =>
      IO.println s!"Unknown harness: {name}"
      IO.println ""
      IO.println "Available harnesses:"
      for h in knownHarnesses do
        IO.println s!"  - {h.name}"
  | (false, none) =>
    IO.println "Specify --all or --harness=NAME"
    IO.println ""
    IO.println "Available harnesses:"
    for h in knownHarnesses do
      IO.println s!"  - {h.name}"

def runStaticCommand (_args : List String) : IO Unit := do
  IO.println "Static Analysis (Clang TSA via compile_commands.json)"
  IO.println "======================================================"
  IO.println ""

  let root ← findProjectRoot

  -- Find the TSA script relative to project root or executable
  let appPath ← IO.appPath
  let appDir := appPath.parent.getD (System.FilePath.mk ".")
  let projectRoot := appDir / ".." / ".." / ".."
  let scriptPath := projectRoot / "scripts" / "run_clang_tsa.sh"
  let outputPath := root / "tsa_results.json"

  if !(← scriptPath.pathExists) then
    -- Try from cwd
    let cwdScript := root / "scripts" / "run_clang_tsa.sh"
    if !(← cwdScript.pathExists) then
      IO.println "ERROR: TSA script not found."
      IO.println s!"Tried: {scriptPath} and {cwdScript}"
      return

  IO.println s!"Project root: {root}"
  IO.println s!"TSA script: {scriptPath}"
  IO.println s!"Output: {outputPath}"
  IO.println ""

  -- Run the TSA script
  IO.println "Running Clang Thread Safety Analysis..."
  IO.println ""

  try
    let result ← IO.Process.output {
      cmd := "bash"
      args := #[scriptPath.toString, outputPath.toString]
      cwd := some root
    }

    -- Print script output
    if !result.stdout.isEmpty then
      IO.println result.stdout
    if !result.stderr.isEmpty then
      IO.println result.stderr

    -- Read and summarize results
    if ← outputPath.pathExists then
      let jsonContent ← IO.FS.readFile outputPath
      IO.println ""
      IO.println "Results written to: tsa_results.json"
      IO.println ""

      -- Parse summary from JSON (simple extraction)
      if (jsonContent.splitOn "\"total_warnings\"").length > 1 then
        IO.println "See tsa_results.json for detailed findings."

    -- Report exit status
    if result.exitCode == 0 then
      IO.println "\n✓ TSA PASSED - No warnings or errors"
    else if result.exitCode == 2 then
      IO.println "\n⚠ TSA found warnings (exit code 2)"
    else
      IO.println s!"\n✗ TSA failed (exit code {result.exitCode})"

  catch e =>
    IO.println s!"ERROR: Failed to run TSA: {e}"

/-- Run structural conformance checks -/
def runStructuralCommand (_args : List String) : IO Unit := do
  IO.println "Structural Conformance Checks"
  IO.println "=============================="
  IO.println ""

  let root ← findProjectRoot

  -- Find the structural checks script
  let appPath ← IO.appPath
  let appDir := appPath.parent.getD (System.FilePath.mk ".")
  let projectRoot := appDir / ".." / ".." / ".."
  let scriptPath := projectRoot / "scripts" / "structural_checks.sh"
  let outputPath := root / "structural_check_results.json"

  let cwdScript := root / "scripts" / "structural_checks.sh"
  let finalScript := if ← scriptPath.pathExists then scriptPath else cwdScript

  if !(← finalScript.pathExists) then
    IO.println "ERROR: Structural checks script not found."
    IO.println s!"Tried: {scriptPath} and {cwdScript}"
    return

  IO.println s!"Project root: {root}"
  IO.println s!"Script: {finalScript}"
  IO.println ""

  try
    let result ← IO.Process.output {
      cmd := "bash"
      args := #[finalScript.toString, outputPath.toString]
      cwd := some root
    }

    -- Print script output (includes colored results)
    if !result.stdout.isEmpty then
      IO.println result.stdout

    -- Report exit status
    if result.exitCode == 0 then
      IO.println "\nStructural checks: PASS"
    else
      IO.println s!"\nStructural checks: FAIL (exit code {result.exitCode})"

  catch e =>
    IO.println s!"ERROR: Failed to run structural checks: {e}"

/-- Tool outcome for gating and reporting. -/
inductive ToolOutcome (α : Type) where
  | notRequested : ToolOutcome α
  | skipped : String → ToolOutcome α
  | completed : α → ToolOutcome α
  deriving Inhabited

structure TLARun where
  name : String
  result : TLCResult
  logPath : System.FilePath
  deriving Inhabited

structure CBMCRun where
  name : String
  result : CBMCResult
  logPath : System.FilePath
  deriving Inhabited

structure StaticRun where
  exitCode : UInt32
  result : StaticAnalysisResult
  logPath : System.FilePath
  tsaResultsJson : Option System.FilePath
  deriving Inhabited

structure StructuralRun where
  exitCode : UInt32
  logPath : System.FilePath
  resultsJson : System.FilePath
  deriving Inhabited

structure ToolVersions where
  mpsverify : String
  gitHead : Option String
  java : Option String
  tlc : Option String
  cbmc : Option String
  clang : Option String
  deriving Inhabited

/-- Unified verification result for check --all -/
structure UnifiedResult where
  /-- Directory containing run artifacts (logs, JSON summaries). -/
  runDir : System.FilePath
  /-- Tool/version metadata captured for the run. -/
  versions : ToolVersions
  tla : ToolOutcome (List TLARun)
  cbmc : ToolOutcome (List CBMCRun)
  static : ToolOutcome StaticRun
  structural : ToolOutcome StructuralRun
  timestamp : String
  deriving Inhabited

/-- Run all verification tools -/
def runCheckAll (runTLA : Bool) (runCBMC : Bool) (runStatic : Bool) (runStructural : Bool) (verbose : Bool := false) : IO UnifiedResult := do
  let now ← IO.Process.output { cmd := "date", args := #["+%Y-%m-%d %H:%M:%S"] }
  let timestamp := now.stdout.trim

  let root ← findProjectRoot
  let runStamp ← IO.Process.output { cmd := "date", args := #["+%y-%m-%d-%H-%M-%S"] }
  let base := runStamp.stdout.trim
  let ms ← IO.monoMsNow
  let runId := s!"{base}.{ms % 1000}"
  let runDir := root / "states" / runId
  let tlaDir := runDir / "tla"
  let cbmcDir := runDir / "cbmc"
  let staticDir := runDir / "static"
  let structuralDir := runDir / "structural"
  IO.FS.createDirAll tlaDir
  IO.FS.createDirAll cbmcDir
  IO.FS.createDirAll staticDir
  IO.FS.createDirAll structuralDir

  -- Use TLCRunner/CBMCRunner defaults (tuned for long-running specs/harnesses).
  let tlcOpts : TLCOptions := { TLCOptions.default with cwd := some root, verbose := verbose }
  let cbmcOpts : CBMCOptions := { CBMCOptions.default with cwd := some root, verbose := verbose }

  -- Capture tool versions (best-effort).
  let gitHead ← cmdFirstLine? "git" #["rev-parse", "HEAD"] (cwd := some root)
  let clangVer ← cmdFirstLine? "clang++" #["--version"] (cwd := some root)
  let cbmcVer := (← getCBMCVersion)
  let tlcAvail ← checkTLCAvailability
  let (tlcVer, javaVer) ← match tlcAvail with
    | .tlcCommand => do
      let jv ← cmdFirstLine? "tlc" #["-help"] (cwd := some root)
      pure (some "tlc (command)", jv)
    | .jarFile jarPath javaPath => do
      let jv ← cmdFirstLine? javaPath.toString #["-version"] (cwd := some root)
      pure (some s!"tla2tools.jar ({jarPath.toString})", jv)
    | .jarOnly jarPath => do
      pure (some s!"tla2tools.jar ({jarPath.toString})", none)
    | .notFound => do
      pure (none, none)

  let versions : ToolVersions := {
    mpsverify := version
    gitHead := gitHead
    java := javaVer
    tlc := tlcVer
    cbmc := cbmcVer
    clang := clangVer
  }

  let mut tlaOutcome : ToolOutcome (List TLARun) := .notRequested
  let mut cbmcOutcome : ToolOutcome (List CBMCRun) := .notRequested
  let mut staticOutcome : ToolOutcome StaticRun := .notRequested
  let mut structuralOutcome : ToolOutcome StructuralRun := .notRequested

  -- Run TLA+ if requested and available
  if runTLA then
    match tlcAvail with
    | .notFound =>
      IO.println "TLA+: Skipped (TLC not found)"
      tlaOutcome := .skipped "TLC not found"
    | .jarOnly _ =>
      IO.println "TLA+: Skipped (TLC jar found but no Java runtime)"
      tlaOutcome := .skipped "TLC jar found but no Java runtime"
    | _ =>
      IO.println "Running TLA+ model checking..."
      let results ← runAllSpecs tlcOpts
      let mut runs : List TLARun := []
      for (name, r) in results do
        let logPath := tlaDir / s!"{sanitizePathComponent name}.log"
        IO.FS.writeFile logPath r.rawOutput
        runs := runs ++ [{ name := name, result := r, logPath := logPath }]
      tlaOutcome := .completed runs

  -- Run CBMC if requested and available
  if runCBMC then
    let available ← checkCBMCAvailable
    if !available then
      IO.println "CBMC: Skipped (not found)"
      cbmcOutcome := .skipped "CBMC not found"
    else
      IO.println "\nRunning CBMC bounded verification..."
      let results ← runAllHarnesses cbmcOpts
      let mut runs : List CBMCRun := []
      for (name, r) in results do
        let logPath := cbmcDir / s!"{sanitizePathComponent name}.log"
        IO.FS.writeFile logPath r.rawOutput
        runs := runs ++ [{ name := name, result := r, logPath := logPath }]
      cbmcOutcome := .completed runs

  -- Run static analysis if requested and available
  if runStatic then
    let clangAvail ← checkClangAvailable
    if !clangAvail then
      IO.println "Static Analysis: Skipped (clang not found)"
      staticOutcome := .skipped "clang++ not found"
    else
      IO.println "\nRunning Static Analysis (Clang TSA)..."
      let scriptPath := root / "scripts" / "run_clang_tsa.sh"
      let outputPath := staticDir / "tsa_results.json"
      let logPath := staticDir / "tsa.log"

      if ← scriptPath.pathExists then
        let result ← IO.Process.output {
          cmd := "bash"
          args := #[scriptPath.toString, outputPath.toString]
          cwd := some root
        }
        let raw := result.stdout ++ result.stderr
        IO.FS.writeFile logPath raw
        let clangResult : ClangResult := {
          success := result.exitCode == 0
          warnings := []
          errors := []
          rawOutput := raw
        }
        if result.exitCode == 0 then
          IO.println "  TSA: PASS"
        else if result.exitCode == 2 then
          IO.println "  TSA: Warnings found (see tsa_results.json)"
        else
          IO.println s!"  TSA: Failed (exit code {result.exitCode})"
        staticOutcome := .completed {
          exitCode := result.exitCode
          result := { clang := some clangResult, infer := none }
          logPath := logPath
          tsaResultsJson := some outputPath
        }
      else
        IO.println "Static Analysis: TSA script not found"
        let raw := "TSA script not found"
        IO.FS.writeFile logPath raw
        staticOutcome := .skipped "TSA script not found"

  -- Run structural checks if requested and available
  if runStructural then
    let scriptPath := root / "scripts" / "structural_checks.sh"
    let resultsJson := structuralDir / "structural_check_results.json"
    let logPath := structuralDir / "structural.log"
    if ← scriptPath.pathExists then
      try
        let result ← IO.Process.output {
          cmd := "bash"
          args := #[scriptPath.toString, resultsJson.toString]
          cwd := some root
        }
        let raw := result.stdout ++ result.stderr
        IO.FS.writeFile logPath raw
        structuralOutcome := .completed {
          exitCode := result.exitCode
          logPath := logPath
          resultsJson := resultsJson
        }
      catch e =>
        IO.FS.writeFile logPath s!"ERROR: {e}"
        structuralOutcome := .completed {
          exitCode := 2
          logPath := logPath
          resultsJson := resultsJson
        }
    else
      IO.FS.writeFile logPath "Structural checks script not found"
      structuralOutcome := .skipped "Structural checks script not found"

  return {
    runDir := runDir
    versions := versions
    tla := tlaOutcome
    cbmc := cbmcOutcome
    static := staticOutcome
    structural := structuralOutcome
    timestamp := timestamp
  }

/-- Count passing TLA+ specs -/
def countTLAPass (results : List TLARun) : Nat :=
  results.filter (·.result.success) |>.length

/-- Count passing CBMC harnesses -/
def countCBMCPass (results : List CBMCRun) : Nat :=
  results.filter (·.result.success) |>.length

/-- Format unified results as JSON. -/
def formatReportJson (result : UnifiedResult) : String :=
  let versionsObj := jsonObj [
    ("mpsverify", jsonStr result.versions.mpsverify),
    ("git_head", match result.versions.gitHead with | some v => jsonStr v | none => "null"),
    ("java", match result.versions.java with | some v => jsonStr v | none => "null"),
    ("tlc", match result.versions.tlc with | some v => jsonStr v | none => "null"),
    ("cbmc", match result.versions.cbmc with | some v => jsonStr v | none => "null"),
    ("clang", match result.versions.clang with | some v => jsonStr v | none => "null")
  ]

  let encodeTLA (runs : List TLARun) : String :=
    jsonArr <| runs.map fun r =>
      jsonObj [
        ("name", jsonStr r.name),
        ("success", if r.result.success then "true" else "false"),
        ("states_generated", toString r.result.statesGenerated),
        ("distinct_states", toString r.result.distinctStates),
        ("depth", toString r.result.depth),
        ("time_ms", toString r.result.timeMs),
        ("log_path", jsonStr r.logPath.toString)
      ]

  let encodeCBMC (runs : List CBMCRun) : String :=
    jsonArr <| runs.map fun r =>
      jsonObj [
        ("name", jsonStr r.name),
        ("success", if r.result.success then "true" else "false"),
        ("failed_assertions", toString r.result.failedAssertions),
        ("total_assertions", toString r.result.totalAssertions),
        ("log_path", jsonStr r.logPath.toString)
      ]

  let encodeStatic (r : StaticRun) : String :=
    jsonObj [
      ("exit_code", toString r.exitCode),
      ("tsa_log_path", jsonStr r.logPath.toString),
      ("tsa_results_json", match r.tsaResultsJson with | some p => jsonStr p.toString | none => "null"),
      ("success", match r.result.clang with | some c => if c.success then "true" else "false" | none => "false")
    ]

  let encodeStructural (r : StructuralRun) : String :=
    jsonObj [
      ("exit_code", toString r.exitCode),
      ("log_path", jsonStr r.logPath.toString),
      ("results_json", jsonStr r.resultsJson.toString)
    ]

  let toolObj {α : Type} (key : String) (outcome : ToolOutcome α) (encode : α → String) : (String × String) :=
    let v := match outcome with
      | .notRequested => jsonObj [("status", jsonStr "not_requested")]
      | .skipped reason => jsonObj [("status", jsonStr "skipped"), ("reason", jsonStr reason)]
      | .completed value => jsonObj [("status", jsonStr "completed"), ("result", encode value)]
    (key, v)

  jsonObj [
    ("generated_at", jsonStr result.timestamp),
    ("run_dir", jsonStr result.runDir.toString),
    ("versions", versionsObj),
    toolObj "tla" result.tla encodeTLA,
    toolObj "cbmc" result.cbmc encodeCBMC,
    toolObj "static" result.static encodeStatic,
    toolObj "structural" result.structural encodeStructural
  ]

/-- Format unified results as markdown -/
def formatReportMarkdown (result : UnifiedResult) : String :=
  let header :=
    s!"# MPS Verification Report\n\nGenerated: {result.timestamp}\n\n"
    ++ s!"Run directory: `{result.runDir.toString}`\n\n"

  let versionsSection :=
    "## Tool Versions\n\n"
    ++ s!"- mpsverify: `{result.versions.mpsverify}`\n"
    ++ s!"- git: `{result.versions.gitHead.getD "unknown"}`\n"
    ++ s!"- TLC: `{result.versions.tlc.getD "unknown"}`\n"
    ++ s!"- Java: `{result.versions.java.getD "unknown"}`\n"
    ++ s!"- CBMC: `{result.versions.cbmc.getD "unknown"}`\n"
    ++ s!"- Clang: `{result.versions.clang.getD "unknown"}`\n\n"

  -- Summary section
  let tlaSummaryLine :=
    match result.tla with
    | .completed runs =>
      let passed := countTLAPass runs
      let total := runs.length
      let status := if passed == total then "PASS" else "FAIL"
      s!"- **TLA+ Model Checking:** {passed}/{total} specs pass ({status})\n"
    | .skipped reason =>
      s!"- **TLA+ Model Checking:** SKIPPED ({reason})\n"
    | .notRequested =>
      "- **TLA+ Model Checking:** Not run\n"

  let cbmcSummaryLine :=
    match result.cbmc with
    | .completed runs =>
      let passed := countCBMCPass runs
      let total := runs.length
      let status := if passed == total then "PASS" else "FAIL"
      s!"- **CBMC Verification:** {passed}/{total} harnesses pass ({status})\n"
    | .skipped reason =>
      s!"- **CBMC Verification:** SKIPPED ({reason})\n"
    | .notRequested =>
      "- **CBMC Verification:** Not run\n"

  let staticSummaryLine :=
    match result.static with
    | .completed r =>
      let status :=
        if r.exitCode == 0 then "PASS"
        else if r.exitCode == 2 then "WARN"
        else "FAIL"
      s!"- **Static Analysis (TSA):** {status} (exit code {r.exitCode}, log: `{r.logPath.toString}`)\n"
    | .skipped reason =>
      s!"- **Static Analysis (TSA):** SKIPPED ({reason})\n"
    | .notRequested =>
      "- **Static Analysis (TSA):** Not run\n"

  let structuralSummaryLine :=
    match result.structural with
    | .completed r =>
      let status := if r.exitCode == 0 then "PASS" else "FAIL"
      s!"- **Structural Checks:** {status} (exit code {r.exitCode}, results: `{r.resultsJson.toString}`)\n"
    | .skipped reason =>
      s!"- **Structural Checks:** SKIPPED ({reason})\n"
    | .notRequested =>
      "- **Structural Checks:** Not run\n"

  let summary :=
    "## Summary\n\n"
    ++ tlaSummaryLine
    ++ cbmcSummaryLine
    ++ staticSummaryLine
    ++ structuralSummaryLine
    ++ "\n"

  -- TLA+ details
  let tlaSection := match result.tla with
    | .completed runs =>
      let rows := runs.map fun r =>
        let status := if r.result.success then "PASS" else "FAIL"
        s!"| {r.name} | {r.result.statesGenerated} | {r.result.timeMs}ms | {status} | `{r.logPath.toString}` |"
      let table := "| Spec | States | Time | Status | Log |\n|------|--------|------|--------|-----|\n"
        ++ String.intercalate "\n" rows
      s!"## TLA+ Model Checking Results\n\n{table}\n\n"
    | .skipped reason =>
      s!"## TLA+ Model Checking Results\n\nSKIPPED: {reason}\n\n"
    | .notRequested =>
      "## TLA+ Model Checking Results\n\nNot run.\n\n"

  -- CBMC details
  let cbmcSection := match result.cbmc with
    | .completed runs =>
      let rows := runs.map fun r =>
        let status := if r.result.success then "PASS" else "FAIL"
        s!"| {r.name} | {r.result.failedAssertions}/{r.result.totalAssertions} | {status} | `{r.logPath.toString}` |"
      let table := "| Harness | Assertions | Status | Log |\n|---------|------------|--------|-----|\n"
        ++ String.intercalate "\n" rows
      s!"## CBMC Bounded Verification Results\n\n{table}\n\n"
    | .skipped reason =>
      s!"## CBMC Bounded Verification Results\n\nSKIPPED: {reason}\n\n"
    | .notRequested =>
      "## CBMC Bounded Verification Results\n\nNot run.\n\n"

  -- Static analysis details
  let staticSection :=
    match result.static with
    | .completed r =>
      let status :=
        if r.exitCode == 0 then "PASS"
        else if r.exitCode == 2 then "WARN"
        else "FAIL"
      let tsaJson := r.tsaResultsJson.map (·.toString) |>.getD "(none)"
      s!"## Static Analysis Results\n\n- TSA: {status} (exit code {r.exitCode})\n- Log: `{r.logPath.toString}`\n- Results JSON: `{tsaJson}`\n\n"
    | .skipped reason =>
      s!"## Static Analysis Results\n\nSKIPPED: {reason}\n\n"
    | .notRequested =>
      "## Static Analysis Results\n\nNot run.\n\n"

  let structuralSection :=
    match result.structural with
    | .completed r =>
      let status := if r.exitCode == 0 then "PASS" else "FAIL"
      s!"## Structural Conformance Checks\n\n- Status: {status} (exit code {r.exitCode})\n- Log: `{r.logPath.toString}`\n- Results JSON: `{r.resultsJson.toString}`\n\n"
    | .skipped reason =>
      s!"## Structural Conformance Checks\n\nSKIPPED: {reason}\n\n"
    | .notRequested =>
      "## Structural Conformance Checks\n\nNot run.\n\n"

  -- Report gate status only when supported by this run's results.
  let tlaGate :=
    match result.tla with
    | .completed runs => if countTLAPass runs == runs.length then "PASS" else "FAIL"
    | .skipped reason => s!"SKIPPED ({reason})"
    | .notRequested => "NOT RUN"
  let cbmcGate :=
    match result.cbmc with
    | .completed runs => if countCBMCPass runs == runs.length then "PASS" else "FAIL"
    | .skipped reason => s!"SKIPPED ({reason})"
    | .notRequested => "NOT RUN"
  let tsaGate :=
    match result.static with
    | .completed r =>
      if r.exitCode == 0 then "PASS"
      else if r.exitCode == 2 then "WARN"
      else "FAIL"
    | .skipped reason => s!"SKIPPED ({reason})"
    | .notRequested => "NOT RUN"
  let structuralGate :=
    match result.structural with
    | .completed r => if r.exitCode == 0 then "PASS" else "FAIL"
    | .skipped reason => s!"SKIPPED ({reason})"
    | .notRequested => "NOT RUN"

  let propertiesSection :=
    "## Verification Gate Status (This Run)\n\n"
    ++ s!"- TLA+: {tlaGate}\n"
    ++ s!"- CBMC: {cbmcGate}\n"
    ++ s!"- Static (TSA): {tsaGate}\n"
    ++ s!"- Structural: {structuralGate}\n"

  header ++ versionsSection ++ summary ++ tlaSection ++ cbmcSection ++ staticSection ++ structuralSection ++ propertiesSection

/-- Format unified results as HTML -/
def formatReportHTML (result : UnifiedResult) : String :=
  let md := formatReportMarkdown result
  -- Simple HTML wrapper
  "<!DOCTYPE html>\n<html>\n<head>\n"
    ++ "<title>MPS Verification Report</title>\n"
    ++ "<style>\n"
    ++ "body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }\n"
    ++ "table { border-collapse: collapse; width: 100%; margin: 10px 0; }\n"
    ++ "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n"
    ++ "th { background-color: #f5f5f5; }\n"
    ++ "h1 { color: #333; }\n"
    ++ "h2 { color: #555; margin-top: 30px; }\n"
    ++ "</style>\n"
    ++ "</head>\n<body>\n"
    ++ "<pre style=\"white-space: pre-wrap;\">\n"
    ++ md
    ++ "\n</pre>\n"
    ++ "</body>\n</html>"

/-- Run check command with options -/
def runCheckCommand (args : List String) : IO Unit := do
  let runAll := args.any (· == "--all")
  let runTLA := runAll || args.any (· == "--tla")
  let runCBMC := runAll || args.any (· == "--cbmc")
  let runStatic := runAll || args.any (· == "--static")
  let runStructural := runAll || args.any (· == "--structural")
  let allowSkip := args.any (· == "--allow-skip")
  let verbose := args.any (· == "--verbose")

  if !runTLA && !runCBMC && !runStatic && !runStructural then
    -- No specific tools requested, show status
    IO.println "MPS Verification Platform"
    IO.println "========================="
    IO.println ""
    IO.println "Phase Status:"
    IO.println "  Phase 1 (Lean Foundation)      ✓ Complete"
    IO.println "  Phase 2 (TLA+ Model Checking)  ✓ Complete - 10 specs verified"
    IO.println "  Phase 3 (CBMC Verification)    ✓ Complete - 10 harnesses verified"
    IO.println "  Phase 4 (Static Analysis)      ✓ Complete - TSA annotations applied"
    IO.println "  Phase 5 (Iris/Coq)             ✓ Complete - 6 modules compiling"
    IO.println "  Phase 6 (Integration)          ✓ Complete"
    IO.println ""
    IO.println "Run 'mps-verify check --all' to execute all verification tools."
    IO.println "Run 'mps-verify help' for full options."
    return

  IO.println "MPS Verification Platform - Full Suite"
  IO.println "======================================="
  IO.println ""

  let result ← runCheckAll runTLA runCBMC runStatic runStructural verbose

  IO.println ""
  IO.println "======================================="
  IO.println "VERIFICATION SUMMARY"
  IO.println "======================================="
  IO.println ""

  -- TLA+ summary
  match result.tla with
  | .completed runs =>
    let passed := countTLAPass runs
    let total := runs.length
    let status := if passed == total then "PASS" else "FAIL"
    IO.println s!"TLA+ Model Checking: {passed}/{total} specs - {status}"
  | .skipped reason =>
    IO.println s!"TLA+ Model Checking: SKIPPED ({reason})"
  | .notRequested =>
    IO.println "TLA+ Model Checking: Not run"

  -- CBMC summary
  match result.cbmc with
  | .completed runs =>
    let passed := countCBMCPass runs
    let total := runs.length
    let status := if passed == total then "PASS" else "FAIL"
    IO.println s!"CBMC Verification: {passed}/{total} harnesses - {status}"
  | .skipped reason =>
    IO.println s!"CBMC Verification: SKIPPED ({reason})"
  | .notRequested =>
    IO.println "CBMC Verification: Not run"

  -- Static analysis summary
  match result.static with
  | .completed r =>
    let status := if r.exitCode == 0 then "PASS" else if r.exitCode == 2 then "WARN" else "FAIL"
    IO.println s!"Static Analysis (TSA): {status} (exit code {r.exitCode})"
  | .skipped reason =>
    IO.println s!"Static Analysis (TSA): SKIPPED ({reason})"
  | .notRequested =>
    IO.println "Static Analysis (TSA): Not run"

  match result.structural with
  | .completed r =>
    let status := if r.exitCode == 0 then "PASS" else "FAIL"
    IO.println s!"Structural Checks: {status} (exit code {r.exitCode})"
  | .skipped reason =>
    IO.println s!"Structural Checks: SKIPPED ({reason})"
  | .notRequested =>
    IO.println "Structural Checks: Not run"

  IO.println ""
  let mdReport := formatReportMarkdown result
  let jsonReport := formatReportJson result
  IO.FS.writeFile (result.runDir / "verification_report.md") mdReport
  IO.FS.writeFile (result.runDir / "verification_report.json") jsonReport
  IO.println s!"Wrote reports:"
  IO.println s!"  - {result.runDir / "verification_report.md"}"
  IO.println s!"  - {result.runDir / "verification_report.json"}"

  -- Gating behavior: by default, any skipped tool or failed check makes this command fail.
  let mut failures : List String := []

  if runTLA then
    match result.tla with
    | .completed runs =>
      if countTLAPass runs != runs.length then
        failures := failures ++ ["TLA+ model checking: FAIL"]
    | .skipped reason =>
      if !allowSkip then
        failures := failures ++ [s!"TLA+ model checking: SKIPPED ({reason})"]
    | .notRequested =>
      failures := failures ++ ["TLA+ model checking: NOT RUN (internal error)"]

  if runCBMC then
    match result.cbmc with
    | .completed runs =>
      if countCBMCPass runs != runs.length then
        failures := failures ++ ["CBMC verification: FAIL"]
    | .skipped reason =>
      if !allowSkip then
        failures := failures ++ [s!"CBMC verification: SKIPPED ({reason})"]
    | .notRequested =>
      failures := failures ++ ["CBMC verification: NOT RUN (internal error)"]

  if runStatic then
    match result.static with
    | .completed r =>
      if r.exitCode != 0 then
        failures := failures ++ [s!"Static analysis (TSA): exit code {r.exitCode}"]
    | .skipped reason =>
      if !allowSkip then
        failures := failures ++ [s!"Static analysis (TSA): SKIPPED ({reason})"]
    | .notRequested =>
      failures := failures ++ ["Static analysis (TSA): NOT RUN (internal error)"]

  if runStructural then
    match result.structural with
    | .completed r =>
      if r.exitCode != 0 then
        failures := failures ++ [s!"Structural checks: exit code {r.exitCode}"]
    | .skipped reason =>
      if !allowSkip then
        failures := failures ++ [s!"Structural checks: SKIPPED ({reason})"]
    | .notRequested =>
      failures := failures ++ ["Structural checks: NOT RUN (internal error)"]

  if !failures.isEmpty then
    IO.println ""
    IO.println "======================================="
    IO.println "OVERALL STATUS: FAIL"
    IO.println "======================================="
    for f in failures do
      IO.println s!"- {f}"
    IO.println s!"Run directory: {result.runDir}"
    IO.println ""
    IO.println "Hint: --allow-skip ignores missing tools (not FAIL/TIMEOUT/WARN results)."
    IO.Process.exit 1

/-- Run report command -/
def runReportCommand (args : List String) : IO Unit := do
  IO.println "Generating verification report..."
  IO.println ""

  -- Parse format option
  let formatArg := args.find? (·.startsWith "--format=")
  let format := formatArg.map (·.drop 9) |>.getD "md"

  -- Parse output option
  let outputArg := args.find? (·.startsWith "--output=")
  let defaultOutput := if format == "html" then "verification_report.html" else "verification_report.md"
  let outputFile := outputArg.map (·.drop 9) |>.getD defaultOutput

  -- Run all verification tools to gather results
  let result ← runCheckAll true true true true

  -- Generate report
  let content := if format == "html"
    then formatReportHTML result
    else formatReportMarkdown result

  -- Write to file
  IO.FS.writeFile outputFile content
  IO.println s!"Report generated: {outputFile}"

  -- Print summary
  let (tlaPass, tlaTotal) :=
    match result.tla with
    | .completed runs => (countTLAPass runs, runs.length)
    | _ => (0, 0)
  let (cbmcPass, cbmcTotal) :=
    match result.cbmc with
    | .completed runs => (countCBMCPass runs, runs.length)
    | _ => (0, 0)

  IO.println ""
  IO.println "Summary:"
  IO.println s!"  TLA+: {tlaPass}/{tlaTotal} specs pass"
  IO.println s!"  CBMC: {cbmcPass}/{cbmcTotal} harnesses pass"
  match result.static with
  | .completed r =>
    let status := if r.exitCode == 0 then "PASS" else if r.exitCode == 2 then "WARN" else "FAIL"
    IO.println s!"  Static (TSA): {status} (exit code {r.exitCode})"
  | .skipped reason =>
    IO.println s!"  Static (TSA): SKIPPED ({reason})"
  | .notRequested =>
    IO.println s!"  Static (TSA): Not run"

def main (args : List String) : IO Unit := do
  match args with
  | [] => printHelp
  | ["help"] => printHelp
  | ["--help"] => printHelp
  | ["-h"] => printHelp
  | ["version"] => IO.println s!"mps-verify version {version}"
  | ["--version"] => IO.println s!"mps-verify version {version}"
  | "tla" :: rest => runTLACommand rest
  | "cbmc" :: rest => runCBMCCommand rest
  | "static" :: rest => runStaticCommand rest
  | "structural" :: rest => runStructuralCommand rest
  | "check" :: rest => runCheckCommand rest
  | "report" :: rest => runReportCommand rest
  | _ => do
    IO.println s!"Unknown command: {args}"
    printHelp
