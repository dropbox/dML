/-
  MPSVerify.Bridges.TLAPlus
  TLA+ TLC Model Checker Output Parser
-/

namespace MPSVerify.Bridges.TLAPlus

/-- Check if needle appears in haystack at position i -/
def matchAt (haystack : String) (needle : String) (i : Nat) : Bool :=
  let needleChars := needle.toList
  let haystackChars := haystack.toList
  needleChars.zipIdx.all fun (c, j) =>
    match haystackChars[i + j]? with
    | some c' => c == c'
    | none => false

/-- Check if haystack contains needle as substring -/
def containsSubstr (haystack : String) (needle : String) : Bool :=
  if needle.isEmpty then true
  else if haystack.length < needle.length then false
  else
    let maxStart := haystack.length - needle.length
    (List.range (maxStart + 1)).any fun i => matchAt haystack needle i

/-- Parsed TLC violation with counterexample trace -/
structure Violation where
  invariant : String
  message : String
  trace : List String
  deriving Repr, Inhabited

/-- TLC execution result -/
structure TLCResult where
  success : Bool
  statesGenerated : Nat
  distinctStates : Nat
  depth : Nat
  violations : List Violation
  timeMs : Nat
  rawOutput : String
  deriving Repr, Inhabited

/-- Default result for failed/timeout runs -/
def TLCResult.failed (msg : String) : TLCResult := {
  success := false
  statesGenerated := 0
  distinctStates := 0
  depth := 0
  violations := []
  timeMs := 0
  rawOutput := msg
}

/-- Parse state count from TLC output line -/
def parseStateLine (line : String) : Option (Nat × Nat) :=
  if containsSubstr line "states generated" then
    let parts := line.splitOn " "
    match parts[0]?.bind String.toNat? with
    | none => none
    | some generated =>
      let distinctIdx := parts.findIdx? (· == "distinct")
      match distinctIdx with
      | none => none
      | some idx =>
        if idx > 0 then
          match parts[idx - 1]?.bind (fun s => (s.replace "," "").toNat?) with
          | some distinct => some (generated, distinct)
          | none => none
        else
          none
  else
    none

/-- Parse depth from TLC output line -/
def parseDepthLine (line : String) : Option Nat :=
  if containsSubstr line "depth of the complete state graph" then
    let parts := line.splitOn " "
    parts.findSome? fun part =>
      (part.replace "." "").toNat?
  else
    none

/-- Check if output indicates success -/
def isSuccessOutput (output : String) : Bool :=
  containsSubstr output "Model checking completed. No error has been found." ||
  containsSubstr output "No error has been found."

/-- Parse invariant violation from TLC output -/
def parseViolation (output : String) : List Violation := Id.run do
  let mut violations : List Violation := []
  let lines := output.splitOn "\n"
  let mut inViolation := false
  let mut currentInvariant := ""
  let mut currentTrace : List String := []

  for line in lines do
    if containsSubstr line "Error: Invariant" && containsSubstr line "is violated" then
      if inViolation then
        violations := violations ++ [{
          invariant := currentInvariant
          message := s!"Invariant {currentInvariant} violated"
          trace := currentTrace
        }]
      inViolation := true
      let stripped := line.replace "Error: Invariant " "" |>.replace " is violated." ""
      currentInvariant := stripped.trim
      currentTrace := []
    else if inViolation && line.startsWith "/\\ " then
      currentTrace := currentTrace ++ [line]
    else if containsSubstr line "State " && containsSubstr line ":" then
      currentTrace := currentTrace ++ [line]

  if inViolation then
    violations := violations ++ [{
      invariant := currentInvariant
      message := s!"Invariant {currentInvariant} violated"
      trace := currentTrace
    }]

  violations

/-- Parse time from output -/
def parseTimeMs (output : String) : Nat := Id.run do
  let lines := output.splitOn "\n"
  for line in lines do
    if containsSubstr line "Finished in " then
      let parts := line.splitOn "Finished in "
      if let some rest := parts[1]? then
        -- TLC prints either:
        --   Finished in 73s at (...)
        -- or:
        --   Finished in 01min 13s at (...)
        let timeStr := (rest.splitOn " at ").head!.trim
        if containsSubstr timeStr "min" then
          let minParts := timeStr.splitOn "min"
          let minStr := minParts[0]!.trim
          let secPart := minParts[1]?.getD ""
          let secStr := (secPart.splitOn "s").head!.trim
          if let some mins := minStr.toNat? then
            if let some secs := secStr.toNat? then
              return (mins * 60 + secs) * 1000
        else
          let secPart := (timeStr.splitOn "s").head!.trim
          let secStr := (secPart.splitOn ".").head!.trim
          if let some secs := secStr.toNat? then
            return secs * 1000
  return 0

/-- Parse full TLC output into structured result -/
def parseTLCOutput (output : String) : TLCResult := Id.run do
  let success := isSuccessOutput output
  let violations := if success then [] else parseViolation output

  let mut statesGen : Nat := 0
  let mut distinctStates : Nat := 0
  let mut depth : Nat := 0

  for line in output.splitOn "\n" do
    if let some (gen, dist) := parseStateLine line then
      statesGen := gen
      distinctStates := dist
    if let some d := parseDepthLine line then
      depth := d

  let timeMs := parseTimeMs output

  {
    success := success
    statesGenerated := statesGen
    distinctStates := distinctStates
    depth := depth
    violations := violations
    timeMs := timeMs
    rawOutput := output
  }

/-- Format TLC result for display -/
def TLCResult.format (result : TLCResult) : String :=
  let status :=
    if result.success then "PASS"
    else if result.rawOutput.startsWith "TIMEOUT" then "TIMEOUT"
    else "FAIL"
  let stats := s!"{result.statesGenerated} states generated, {result.distinctStates} distinct"
  let timing := s!"{result.timeMs}ms"
  let header := s!"TLC Result: {status} ({stats}, depth={result.depth}, time={timing})"

  if result.violations.isEmpty then
    header
  else
    let violationStrs := result.violations.map fun v =>
      let traceStr := if v.trace.isEmpty then "" else s!"\n  Trace:\n    {String.intercalate "\n    " v.trace}"
      s!"  - {v.invariant}: {v.message}{traceStr}"
    header ++ "\nViolations:\n" ++ String.intercalate "\n" violationStrs

end MPSVerify.Bridges.TLAPlus
