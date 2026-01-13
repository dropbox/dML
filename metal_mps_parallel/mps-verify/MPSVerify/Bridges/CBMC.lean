/-
  MPSVerify.Bridges.CBMC
  CBMC (C Bounded Model Checker) Output Parser

  CBMC output format:
  - Header: "CBMC version X.Y.Z"
  - Each assertion: "[function.type.N] line L description: SUCCESS|FAILURE"
  - Summary: "** X of Y failed (Z iterations)"
  - Final: "VERIFICATION SUCCESSFUL" or "VERIFICATION FAILED"
-/

namespace MPSVerify.Bridges.CBMC

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

/-- Assertion result from CBMC output -/
inductive AssertionResult where
  | success : AssertionResult
  | failure : AssertionResult
  deriving Repr, BEq, Inhabited

/-- Parsed CBMC assertion -/
structure CBMCAssertion where
  function : String        -- Function containing the assertion
  checkType : String       -- Type: assertion, pointer_dereference, array_bounds, etc.
  line : Nat               -- Source line number
  description : String     -- Assertion description
  result : AssertionResult -- SUCCESS or FAILURE
  deriving Repr, Inhabited

/-- CBMC execution result -/
structure CBMCResult where
  success : Bool              -- Overall verification success
  totalAssertions : Nat       -- Total number of assertions checked
  failedAssertions : Nat      -- Number of failed assertions
  assertions : List CBMCAssertion  -- Individual assertion results
  iterations : Nat            -- BMC iterations
  rawOutput : String          -- Full CBMC output
  deriving Repr, Inhabited

/-- Default result for failed/timeout runs -/
def CBMCResult.failed (msg : String) : CBMCResult := {
  success := false
  totalAssertions := 0
  failedAssertions := 0
  assertions := []
  iterations := 0
  rawOutput := msg
}

/-- Parse a single assertion line -/
def parseAssertionLine (line : String) : Option CBMCAssertion := Id.run do
  -- Format: [function.type.N] line L description: SUCCESS|FAILURE
  if !line.startsWith "[" then return none

  -- Find the closing bracket
  let closeBracket := line.toList.findIdx? (· == ']')
  match closeBracket with
  | none => return none
  | some idx =>
    let lineChars := line.toList
    let id := String.ofList (lineChars.drop 1 |>.take (idx - 1))
    let rest := String.ofList (lineChars.drop (idx + 2))

    -- Parse function.type.N from id
    let idParts := id.splitOn "."
    let function := idParts.head?.getD ""
    let checkType := if idParts.length > 1 then idParts[1]!  else ""

    -- Parse "line L description: RESULT"
    let restParts := rest.splitOn " "
    if restParts.length < 3 then return none
    if restParts[0]! != "line" then return none

    let lineNum := restParts[1]!.toNat?.getD 0

    -- Find the result (last word)
    let resultStr := rest.splitOn ": " |>.getLast!
    let result := if resultStr == "SUCCESS" then AssertionResult.success
                  else AssertionResult.failure

    -- Description is everything between "line N" and ": RESULT"
    let colonIdx := rest.toList.findIdx? (· == ':')
    let description := match colonIdx with
      | none => ""
      | some ci =>
        let startIdx := 5 + (restParts[1]!.length) + 1  -- "line N "
        (rest.toList.drop startIdx |>.take (ci - startIdx) |> String.ofList).trim

    return some {
      function := function
      checkType := checkType
      line := lineNum
      description := description.trim
      result := result
    }

/-- Parse the summary line: "** X of Y failed (Z iterations)" -/
def parseSummaryLine (line : String) : Option (Nat × Nat × Nat) := Id.run do
  if !containsSubstr line "of" || !containsSubstr line "failed" then
    return none

  let stripped := line.replace "** " "" |>.replace "*" ""
  let parts := stripped.splitOn " "

  let failedIdx := parts.findIdx? (· == "failed")
  match failedIdx with
  | none => return none
  | some fi =>
    if fi < 3 then return none
    let failed := parts[fi - 3]!.toNat?.getD 0
    let total := parts[fi - 1]!.toNat?.getD 0

    -- Parse iterations from "(Z iterations)"
    let iterPart := parts.find? (fun p => p.startsWith "(")
    let iterations := match iterPart with
      | none => 1
      | some ip => (ip.replace "(" "").toNat?.getD 1

    return some (failed, total, iterations)

/-- Check if output indicates success -/
def isSuccessOutput (output : String) : Bool :=
  containsSubstr output "VERIFICATION SUCCESSFUL"

/-- Parse full CBMC output into structured result -/
def parseCBMCOutput (output : String) : CBMCResult := Id.run do
  let success := isSuccessOutput output
  let lines := output.splitOn "\n"

  let mut assertions : List CBMCAssertion := []
  let mut failed : Nat := 0
  let mut total : Nat := 0
  let mut iterations : Nat := 1

  for line in lines do
    -- Parse assertion lines
    if let some assertion := parseAssertionLine line then
      assertions := assertions ++ [assertion]

    -- Parse summary line
    if let some (f, t, i) := parseSummaryLine line then
      failed := f
      total := t
      iterations := i

  {
    success := success
    totalAssertions := total
    failedAssertions := failed
    assertions := assertions
    iterations := iterations
    rawOutput := output
  }

/-- Get only failed assertions -/
def CBMCResult.failures (result : CBMCResult) : List CBMCAssertion :=
  result.assertions.filter (·.result == AssertionResult.failure)

/-- Format CBMC result for display -/
def CBMCResult.format (result : CBMCResult) : String :=
  let status :=
    if result.success then "PASS"
    else if result.rawOutput.startsWith "TIMEOUT" then "TIMEOUT"
    else "FAIL"
  let stats := s!"{result.failedAssertions}/{result.totalAssertions} assertions failed"
  let header := s!"CBMC Result: {status} ({stats})"

  if result.failures.isEmpty then
    header
  else
    let failureStrs := result.failures.map fun a =>
      s!"  - [{a.function}.{a.checkType}] line {a.line}: {a.description}"
    header ++ "\nFailures:\n" ++ String.intercalate "\n" failureStrs

/-- Summarize assertions by type -/
def CBMCResult.summarizeByType (result : CBMCResult) : List (String × Nat × Nat) := Id.run do
  let mut counts : List (String × Nat × Nat) := []
  let mut seen : List String := []

  for assertion in result.assertions do
    if !seen.contains assertion.checkType then
      seen := seen ++ [assertion.checkType]
      let typeAssertions := result.assertions.filter (·.checkType == assertion.checkType)
      let total := typeAssertions.length
      let failed := typeAssertions.filter (·.result == AssertionResult.failure) |>.length
      counts := counts ++ [(assertion.checkType, failed, total)]

  counts

end MPSVerify.Bridges.CBMC
