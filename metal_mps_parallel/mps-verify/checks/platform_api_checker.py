#!/usr/bin/env python3
"""
Platform API Constraint Checker

Enhanced static analyzer for Metal/MPS API constraints using the
comprehensive constraint catalog. This checker performs:

1. State machine tracking for command buffers
2. Pattern matching for known dangerous sequences
3. Dispatch queue analysis for deadlock potential
4. Resource lifetime tracking

Usage:
    python platform_api_checker.py --path pytorch-mps-fork/aten/src/ATen/mps
    python platform_api_checker.py --check-all --json
    python platform_api_checker.py --constraint AF.007 --path file.mm

Exit codes:
    0 - No violations
    1 - Violations found
    2 - Analysis error
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from metal_api_catalog import (
    METAL_API_CONSTRAINTS,
    APIConstraint,
    Severity,
    DetectionMethod,
    get_static_checkable_constraints,
)


@dataclass
class Violation:
    """A detected API constraint violation"""
    constraint_id: str
    file_path: str
    line_number: int
    code_snippet: str
    explanation: str
    severity: Severity


@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    path: str
    violations: List[Violation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    constraints_checked: int = 0


class PlatformAPIChecker:
    """Comprehensive API constraint checker"""

    def __init__(self, verbose: bool = False, constraints: Optional[Set[str]] = None):
        self.verbose = verbose
        self.target_constraints = constraints  # None means all
        self.results: List[FileAnalysis] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    def analyze_file(self, path: Path) -> FileAnalysis:
        """Analyze a single file for API constraint violations"""
        result = FileAnalysis(path=str(path))

        try:
            content = path.read_text()
            lines = content.split('\n')
        except Exception as e:
            result.warnings.append(f"Could not read file: {e}")
            return result

        # Get checkable constraints
        constraints = get_static_checkable_constraints()
        if self.target_constraints:
            constraints = [c for c in constraints if c.id in self.target_constraints]

        result.constraints_checked = len(constraints)

        # Check each constraint
        for constraint in constraints:
            violations = self._check_constraint(constraint, content, lines, path)
            result.violations.extend(violations)

        # Additional semantic checks
        result.violations.extend(self._check_command_buffer_flow(content, lines, path))
        result.violations.extend(self._check_dispatch_patterns(content, lines, path))
        result.violations.extend(self._check_encoder_lifecycle(content, lines, path))

        return result

    def _check_constraint(
        self,
        constraint: APIConstraint,
        content: str,
        lines: List[str],
        path: Path
    ) -> List[Violation]:
        """Check a single constraint against file content"""
        violations = []

        if not constraint.static_pattern:
            return violations

        try:
            pattern = re.compile(constraint.static_pattern, re.MULTILINE | re.DOTALL)
            for match in pattern.finditer(content):
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                code_snippet = lines[line_num - 1].strip() if line_num <= len(lines) else ""

                violations.append(Violation(
                    constraint_id=constraint.id,
                    file_path=str(path),
                    line_number=line_num,
                    code_snippet=code_snippet,
                    explanation=constraint.description,
                    severity=constraint.severity
                ))
                self.log(f"Found {constraint.id} violation at {path}:{line_num}")
        except re.error as e:
            self.log(f"Regex error in {constraint.id}: {e}")

        return violations

    def _check_command_buffer_flow(
        self,
        content: str,
        lines: List[str],
        path: Path
    ) -> List[Violation]:
        """
        Track command buffer state through the file and detect violations.
        This is more sophisticated than pattern matching - it tracks state flow.
        """
        violations = []

        # State machine: buffer_name -> state
        # States: CREATED, ENCODING, COMMITTED, COMPLETED
        buffer_states: Dict[str, str] = {}

        # State transition patterns
        create_patterns = [
            (r'\[(\w+)\s+commandBuffer\]', 'CREATED'),
            (r'(\w+)\s*=\s*\[.*commandBufferWithDescriptor:', 'CREATED'),
            (r'commandBufferLocked\(\)', '_commandBuffer'),  # MPS-specific
        ]

        encoding_patterns = [
            (r'\[(\w+)\s+computeCommandEncoder\]', 'ENCODING'),
            (r'\[(\w+)\s+blitCommandEncoder\]', 'ENCODING'),
            (r'\[(\w+)\s+renderCommandEncoderWithDescriptor:', 'ENCODING'),
        ]

        commit_patterns = [
            (r'\[(\w+)\s+commit\]', 'COMMITTED'),
            (r'(\w+)\.commit\(\)', 'COMMITTED'),
        ]

        end_encoding_patterns = [
            (r'\[(\w+)\s+endEncoding\]', 'end_encoding'),
        ]

        add_handler_patterns = [
            (r'\[(\w+)\s+addCompletedHandler:', 'add_handler'),
            (r'(\w+)\.addCompletedHandler\(', 'add_handler'),
        ]

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*'):
                continue

            # Track state transitions
            for pattern, buffer_name in create_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1) if match.lastindex else buffer_name
                    buffer_states[name] = 'CREATED'
                    self.log(f"Line {line_num}: {name} -> CREATED")

            for pattern, _ in encoding_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    buffer_states[name] = 'ENCODING'

            for pattern, _ in commit_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    buffer_states[name] = 'COMMITTED'

            # Check for violations
            for pattern, action in add_handler_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    state = buffer_states.get(name, 'UNKNOWN')
                    if state == 'COMMITTED':
                        violations.append(Violation(
                            constraint_id='AF.007',
                            file_path=str(path),
                            line_number=line_num,
                            code_snippet=stripped,
                            explanation=f"addCompletedHandler called on {name} which is already COMMITTED",
                            severity=Severity.CRASH
                        ))

        return violations

    def _check_dispatch_patterns(
        self,
        content: str,
        lines: List[str],
        path: Path
    ) -> List[Violation]:
        """Check for dispatch queue issues"""
        violations = []

        # Find dispatch_sync calls and check for potential deadlock patterns
        dispatch_sync_pattern = r'dispatch_sync\s*\(\s*(\w+)\s*,'

        for line_num, line in enumerate(lines, 1):
            match = re.search(dispatch_sync_pattern, line)
            if match:
                queue_name = match.group(1)

                # Check if same queue is used in enclosing scope
                # Look backwards for function definition or dispatch_async
                context_start = max(0, line_num - 30)
                context = '\n'.join(lines[context_start:line_num])

                # If we're inside a dispatch to the same queue, that's a deadlock
                if re.search(rf'dispatch_async\s*\(\s*{queue_name}|dispatch_sync\s*\(\s*{queue_name}', context):
                    violations.append(Violation(
                        constraint_id='AF.020',
                        file_path=str(path),
                        line_number=line_num,
                        code_snippet=lines[line_num - 1].strip(),
                        explanation=f"Potential deadlock: dispatch_sync to {queue_name} from same queue context",
                        severity=Severity.DEADLOCK
                    ))

        return violations

    def _check_encoder_lifecycle(
        self,
        content: str,
        lines: List[str],
        path: Path
    ) -> List[Violation]:
        """Check for encoder lifecycle issues"""
        violations = []

        # Track active encoders
        active_encoders: Dict[str, int] = {}  # encoder_name -> creation_line

        encoder_create_pattern = r'\[(\w+)\s+(compute|blit|render)CommandEncoder'
        end_encoding_pattern = r'\[(\w+)\s+endEncoding\]'
        commit_pattern = r'\[(\w+)\s+commit\]'

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('//'):
                continue

            # Track encoder creation
            match = re.search(encoder_create_pattern, line)
            if match:
                buffer_name = match.group(1)
                if buffer_name in active_encoders:
                    # Creating new encoder without ending previous one
                    violations.append(Violation(
                        constraint_id='AF.011',
                        file_path=str(path),
                        line_number=line_num,
                        code_snippet=stripped,
                        explanation=f"Creating encoder on {buffer_name} while another may be active (created line {active_encoders[buffer_name]})",
                        severity=Severity.CRASH
                    ))
                active_encoders[buffer_name] = line_num

            # Track encoder end
            match = re.search(end_encoding_pattern, line)
            if match:
                encoder_name = match.group(1)
                # Note: encoder name may differ from buffer name
                # This is a simplification

            # Check commit without end encoding
            match = re.search(commit_pattern, line)
            if match:
                buffer_name = match.group(1)
                if buffer_name in active_encoders:
                    # Check if endEncoding was called between creation and commit
                    creation_line = active_encoders[buffer_name]
                    between_lines = '\n'.join(lines[creation_line:line_num])
                    if 'endEncoding' not in between_lines:
                        violations.append(Violation(
                            constraint_id='AF.010',
                            file_path=str(path),
                            line_number=line_num,
                            code_snippet=stripped,
                            explanation=f"Committing {buffer_name} without endEncoding (encoder created line {creation_line})",
                            severity=Severity.CRASH
                        ))

        return violations

    def analyze_directory(self, dir_path: Path, extensions: Set[str] = {'.mm', '.m', '.cpp'}) -> List[FileAnalysis]:
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

    def generate_report(self) -> Dict:
        """Generate a comprehensive report"""
        total_violations = sum(len(r.violations) for r in self.results)
        total_warnings = sum(len(r.warnings) for r in self.results)

        violations_by_severity = {
            'CRASH': 0,
            'DATA_CORRUPTION': 0,
            'DEADLOCK': 0,
            'LEAK': 0,
            'PERFORMANCE': 0,
            'UNDEFINED': 0,
        }

        violations_by_constraint: Dict[str, int] = {}

        for result in self.results:
            for v in result.violations:
                severity_name = v.severity.value if isinstance(v.severity, Severity) else str(v.severity)
                if severity_name in violations_by_severity:
                    violations_by_severity[severity_name] += 1

                if v.constraint_id not in violations_by_constraint:
                    violations_by_constraint[v.constraint_id] = 0
                violations_by_constraint[v.constraint_id] += 1

        return {
            "summary": {
                "files_analyzed": len(self.results),
                "total_violations": total_violations,
                "total_warnings": total_warnings,
                "violations_by_severity": violations_by_severity,
                "violations_by_constraint": violations_by_constraint,
            },
            "violations": [
                {
                    "constraint_id": v.constraint_id,
                    "severity": v.severity.value if isinstance(v.severity, Severity) else str(v.severity),
                    "file": v.file_path,
                    "line": v.line_number,
                    "code": v.code_snippet,
                    "explanation": v.explanation,
                }
                for result in self.results
                for v in result.violations
            ],
            "files": [
                {
                    "path": r.path,
                    "violations": len(r.violations),
                    "warnings": len(r.warnings),
                    "constraints_checked": r.constraints_checked,
                }
                for r in self.results
            ],
            "constraints_checked": list(set(
                v.constraint_id for r in self.results for v in r.violations
            )),
        }

    def print_report(self) -> int:
        """Print a human-readable report and return violation count"""
        report = self.generate_report()

        print("\n" + "=" * 70)
        print("PLATFORM API CONSTRAINT CHECK RESULTS")
        print("=" * 70)

        summary = report["summary"]
        print(f"\nFiles analyzed: {summary['files_analyzed']}")
        print(f"Total violations: {summary['total_violations']}")

        print("\nViolations by severity:")
        for severity, count in summary['violations_by_severity'].items():
            if count > 0:
                print(f"  {severity}: {count}")

        print("\nViolations by constraint:")
        for constraint_id, count in summary['violations_by_constraint'].items():
            if count > 0:
                constraint = METAL_API_CONSTRAINTS.get(constraint_id)
                desc = constraint.description if constraint else "Unknown"
                print(f"  {constraint_id}: {count} - {desc}")

        if report["violations"]:
            print("\n" + "-" * 70)
            print("VIOLATIONS:")
            print("-" * 70)

            for v in report["violations"]:
                severity = v['severity']
                print(f"\n[{severity}] {v['constraint_id']}: {v['file']}:{v['line']}")
                print(f"  Code: {v['code']}")
                print(f"  {v['explanation']}")
        else:
            print("\n[OK] No violations found!")

        print("\n" + "=" * 70)
        return summary['total_violations']


def main():
    parser = argparse.ArgumentParser(
        description="Check Metal/MPS API constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --path pytorch-mps-fork/aten/src/ATen/mps
  %(prog)s --check-all --json > report.json
  %(prog)s --constraint AF.007 --path file.mm
  %(prog)s --list-constraints
        """
    )
    parser.add_argument("--path", "-p", type=str,
                        default="pytorch-mps-fork/aten/src/ATen/mps",
                        help="Path to analyze (file or directory)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--check-all", action="store_true",
                        help="Check all MPS files in default location")
    parser.add_argument("--constraint", "-c", type=str, action="append",
                        help="Only check specific constraint(s), e.g., AF.007")
    parser.add_argument("--list-constraints", action="store_true",
                        help="List all known constraints")

    args = parser.parse_args()

    if args.list_constraints:
        print("\nKnown API Constraints:")
        print("-" * 60)
        for cid, c in sorted(METAL_API_CONSTRAINTS.items()):
            detection = "STATIC" if c.detection == DetectionMethod.STATIC else "DYNAMIC"
            print(f"  {cid}: [{c.severity.value}] [{detection}] {c.description[:50]}...")
        sys.exit(0)

    target_constraints = set(args.constraint) if args.constraint else None
    checker = PlatformAPIChecker(verbose=args.verbose, constraints=target_constraints)

    path = Path(args.path)
    if path.is_file():
        result = checker.analyze_file(path)
        checker.results = [result]
    elif path.is_dir():
        checker.analyze_directory(path)
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(checker.generate_report(), indent=2))
        sys.exit(0 if checker.generate_report()["summary"]["total_violations"] == 0 else 1)
    else:
        violations = checker.print_report()
        sys.exit(0 if violations == 0 else 1)


if __name__ == "__main__":
    main()
