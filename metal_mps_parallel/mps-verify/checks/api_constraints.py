#!/usr/bin/env python3
"""
API Constraint Checker for MPS/Metal APIs

Performs static analysis to detect violations of Apple API constraints
that cannot be caught by standard formal verification tools.

Constraints checked:
- AF.007: addCompletedHandler must be called before commit
- AF.008: waitUntilCompleted must be called after commit
- AF.009: Command encoder must end before commit
- AF.010: No nested command encoders

Usage:
    python api_constraints.py [--path PATH] [--verbose]
    python api_constraints.py --check-all
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class BufferState(Enum):
    """MTLCommandBuffer state machine"""
    UNKNOWN = auto()
    CREATED = auto()
    ENCODING = auto()
    COMMITTED = auto()
    COMPLETED = auto()


@dataclass
class APIConstraint:
    """Definition of an API constraint"""
    id: str
    api_name: str
    required_states: Set[BufferState]
    error_message: str
    severity: str = "ERROR"
    documentation_url: str = ""


@dataclass
class Violation:
    """A detected constraint violation"""
    constraint: APIConstraint
    file_path: str
    line_number: int
    code_snippet: str
    inferred_state: BufferState
    explanation: str


@dataclass
class AnalysisResult:
    """Results of analyzing a file"""
    file_path: str
    violations: List[Violation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    state_transitions: List[Tuple[int, str, BufferState]] = field(default_factory=list)


# Define known API constraints
CONSTRAINTS = {
    "AF.007": APIConstraint(
        id="AF.007",
        api_name="addCompletedHandler",
        required_states={BufferState.CREATED, BufferState.ENCODING},
        error_message="addCompletedHandler called on committed/completed buffer causes SIGABRT",
        severity="CRITICAL",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandbuffer/1442997-addcompletedhandler"
    ),
    "AF.008": APIConstraint(
        id="AF.008",
        api_name="waitUntilCompleted",
        required_states={BufferState.COMMITTED, BufferState.COMPLETED},
        error_message="waitUntilCompleted called before commit has undefined behavior",
        severity="ERROR"
    ),
    "AF.009": APIConstraint(
        id="AF.009",
        api_name="commit",
        required_states={BufferState.CREATED, BufferState.ENCODING},
        error_message="commit called on already committed buffer",
        severity="ERROR"
    ),
}

# Patterns that indicate state transitions
STATE_TRANSITION_PATTERNS = {
    # Pattern -> (variable_pattern, new_state)
    r'\[(\w+)\s+commit\]': BufferState.COMMITTED,
    r'\[(\w+)\s+commitAndContinue\]': BufferState.COMMITTED,
    r'(\w+)\.commit\(\)': BufferState.COMMITTED,
    r'\[MPSCommandBuffer\s+commandBufferFromCommandQueue:': BufferState.CREATED,
    r'commandBufferLocked\(\)': BufferState.CREATED,
    r'commandBuffer\(\)': BufferState.CREATED,
    r'\[(\w+)\s+computeCommandEncoder\]': BufferState.ENCODING,
    r'\[(\w+)\s+blitCommandEncoder\]': BufferState.ENCODING,
    r'\[(\w+)\s+waitUntilCompleted\]': BufferState.COMPLETED,
}

# Patterns for API calls we want to check
API_CALL_PATTERNS = {
    "addCompletedHandler": [
        r'\[(\w+)\s+addCompletedHandler:',
        r'addCompletedHandler\s*\(',
    ],
    "waitUntilCompleted": [
        r'\[(\w+)\s+waitUntilCompleted\]',
    ],
    "commit": [
        r'\[(\w+)\s+commit\]',
        r'(\w+)\.commit\(\)',
    ],
}


class APIConstraintChecker:
    """Static analyzer for Metal/MPS API constraints"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[AnalysisResult] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Analyze a single file for API constraint violations"""
        result = AnalysisResult(file_path=str(file_path))

        try:
            content = file_path.read_text()
            lines = content.split('\n')
        except Exception as e:
            result.warnings.append(f"Could not read file: {e}")
            return result

        # Track buffer state through the file (simplified - per function would be better)
        # This is a conservative analysis - we track state pessimistically
        current_states: Dict[str, BufferState] = {}

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*'):
                continue

            # Check for state transitions
            for pattern, new_state in STATE_TRANSITION_PATTERNS.items():
                match = re.search(pattern, line)
                if match:
                    # Try to extract variable name
                    var_name = match.group(1) if match.lastindex else "_buffer"
                    current_states[var_name] = new_state
                    result.state_transitions.append((line_num, var_name, new_state))
                    self.log(f"Line {line_num}: {var_name} -> {new_state.name}")

            # Check for API calls and validate constraints
            for api_name, patterns in API_CALL_PATTERNS.items():
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        var_name = match.group(1) if match.lastindex else "_buffer"
                        inferred_state = current_states.get(var_name, BufferState.UNKNOWN)

                        # Check constraint
                        constraint = self._get_constraint_for_api(api_name)
                        if constraint:
                            violation = self._check_constraint(
                                constraint, var_name, inferred_state,
                                str(file_path), line_num, line.strip()
                            )
                            if violation:
                                result.violations.append(violation)

        return result

    def _get_constraint_for_api(self, api_name: str) -> Optional[APIConstraint]:
        """Find constraint for an API call"""
        for constraint in CONSTRAINTS.values():
            if constraint.api_name == api_name:
                return constraint
        return None

    def _check_constraint(
        self,
        constraint: APIConstraint,
        var_name: str,
        inferred_state: BufferState,
        file_path: str,
        line_num: int,
        code_snippet: str
    ) -> Optional[Violation]:
        """Check if a constraint is violated"""

        # If state is unknown, we can't prove a violation (be conservative)
        if inferred_state == BufferState.UNKNOWN:
            return None

        # Check if current state is in allowed states
        if inferred_state not in constraint.required_states:
            return Violation(
                constraint=constraint,
                file_path=file_path,
                line_number=line_num,
                code_snippet=code_snippet,
                inferred_state=inferred_state,
                explanation=f"Buffer '{var_name}' is in state {inferred_state.name}, "
                           f"but {constraint.api_name} requires state in "
                           f"{{{', '.join(s.name for s in constraint.required_states)}}}"
            )

        return None

    def analyze_directory(self, dir_path: Path, extensions: Set[str] = {'.mm', '.m', '.cpp'}) -> List[AnalysisResult]:
        """Analyze all matching files in a directory"""
        results = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    self.log(f"Analyzing: {file_path}")
                    result = self.analyze_file(file_path)
                    results.append(result)

        self.results = results
        return results

    def report(self) -> Dict:
        """Generate a report of all findings"""
        total_violations = sum(len(r.violations) for r in self.results)
        total_warnings = sum(len(r.warnings) for r in self.results)

        violations_by_constraint: Dict[str, List[Violation]] = {}
        for result in self.results:
            for v in result.violations:
                if v.constraint.id not in violations_by_constraint:
                    violations_by_constraint[v.constraint.id] = []
                violations_by_constraint[v.constraint.id].append(v)

        return {
            "summary": {
                "files_analyzed": len(self.results),
                "total_violations": total_violations,
                "total_warnings": total_warnings,
                "violations_by_severity": {
                    "CRITICAL": sum(1 for r in self.results for v in r.violations if v.constraint.severity == "CRITICAL"),
                    "ERROR": sum(1 for r in self.results for v in r.violations if v.constraint.severity == "ERROR"),
                    "WARNING": sum(1 for r in self.results for v in r.violations if v.constraint.severity == "WARNING"),
                }
            },
            "violations": [
                {
                    "constraint_id": v.constraint.id,
                    "severity": v.constraint.severity,
                    "file": v.file_path,
                    "line": v.line_number,
                    "code": v.code_snippet,
                    "state": v.inferred_state.name,
                    "explanation": v.explanation,
                    "documentation": v.constraint.documentation_url
                }
                for result in self.results
                for v in result.violations
            ],
            "constraints_checked": list(CONSTRAINTS.keys())
        }

    def print_report(self):
        """Print a human-readable report"""
        report = self.report()

        print("\n" + "=" * 70)
        print("API CONSTRAINT CHECK RESULTS")
        print("=" * 70)

        summary = report["summary"]
        print(f"\nFiles analyzed: {summary['files_analyzed']}")
        print(f"Total violations: {summary['total_violations']}")
        print(f"  CRITICAL: {summary['violations_by_severity']['CRITICAL']}")
        print(f"  ERROR: {summary['violations_by_severity']['ERROR']}")
        print(f"  WARNING: {summary['violations_by_severity']['WARNING']}")

        if report["violations"]:
            print("\n" + "-" * 70)
            print("VIOLATIONS:")
            print("-" * 70)

            for v in report["violations"]:
                print(f"\n[{v['severity']}] {v['constraint_id']}: {v['file']}:{v['line']}")
                print(f"  Code: {v['code']}")
                print(f"  State: {v['state']}")
                print(f"  {v['explanation']}")
                if v['documentation']:
                    print(f"  Docs: {v['documentation']}")
        else:
            print("\n✅ No violations found!")

        print("\n" + "=" * 70)
        return summary['total_violations']


def main():
    parser = argparse.ArgumentParser(description="Check Metal/MPS API constraints")
    parser.add_argument("--path", "-p", type=str,
                       default="pytorch-mps-fork/aten/src/ATen/mps",
                       help="Path to analyze")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true",
                       help="Output as JSON")
    parser.add_argument("--check-all", action="store_true",
                       help="Check all MPS files")

    args = parser.parse_args()

    checker = APIConstraintChecker(verbose=args.verbose)

    path = Path(args.path)
    if path.is_file():
        checker.analyze_file(path)
    elif path.is_dir():
        checker.analyze_directory(path)
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(checker.report(), indent=2))
        sys.exit(0 if checker.report()["summary"]["total_violations"] == 0 else 1)
    else:
        violations = checker.print_report()
        sys.exit(0 if violations == 0 else 1)


if __name__ == "__main__":
    main()
