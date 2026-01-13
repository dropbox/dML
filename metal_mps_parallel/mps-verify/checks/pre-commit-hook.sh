#!/bin/bash
#
# Git Pre-Commit Hook for MPS Parallel Inference
#
# Runs verification checks before allowing commits:
# 1. API constraint checker (AF.* constraints)
# 2. Structural checks (ST.* patterns)
# 3. TSA compilation check (optional, slow)
#
# Install:
#   ln -sf ../../mps-verify/checks/pre-commit-hook.sh .git/hooks/pre-commit
#
# Skip checks (emergency):
#   git commit --no-verify
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPS_VERIFY_DIR="$PROJECT_ROOT/mps-verify"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MPS Pre-Commit Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FAILED=0

# Check 1: API Constraints
echo ""
echo -e "${YELLOW}[1/3] Checking API constraints (AF.*)...${NC}"

if [ -f "$MPS_VERIFY_DIR/checks/api_constraints.py" ]; then
    # Only check staged .mm/.cpp files in pytorch-mps-fork
    STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(mm|cpp|m)$' | grep -E '^pytorch-mps-fork/' || true)

    if [ -n "$STAGED_FILES" ]; then
        for file in $STAGED_FILES; do
            if [ -f "$PROJECT_ROOT/$file" ]; then
                python3 "$MPS_VERIFY_DIR/checks/api_constraints.py" --path "$PROJECT_ROOT/$file" 2>/dev/null
                if [ $? -ne 0 ]; then
                    echo -e "${RED}  ✗ API constraint violation in $file${NC}"
                    FAILED=1
                fi
            fi
        done
        if [ $FAILED -eq 0 ]; then
            echo -e "${GREEN}  ✓ API constraints OK${NC}"
        fi
    else
        echo "  (no MPS files staged)"
    fi
else
    echo -e "${YELLOW}  ⚠ API constraint checker not found, skipping${NC}"
fi

# Check 2: Structural Checks (quick subset)
echo ""
echo -e "${YELLOW}[2/3] Running structural checks (ST.*)...${NC}"

if [ -f "$MPS_VERIFY_DIR/scripts/structural_checks.sh" ]; then
    # Run quick structural checks
    cd "$PROJECT_ROOT"
    bash "$MPS_VERIFY_DIR/scripts/structural_checks.sh" --quick 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}  ✗ Structural check failed${NC}"
        FAILED=1
    else
        echo -e "${GREEN}  ✓ Structural checks OK${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ Structural checks not found, skipping${NC}"
fi

# Check 3: Dangerous patterns
echo ""
echo -e "${YELLOW}[3/3] Checking for dangerous patterns...${NC}"

DANGEROUS_PATTERNS=(
    "_prevCommandBuffer.*addCompletedHandler"  # AF.007 violation pattern
    "dispatch_async.*_streamMutex"              # Potential deadlock
    "delete.*this.*callback"                    # UAF risk
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
    MATCHES=$(git diff --cached | grep -E "^\+" | grep -E "$pattern" || true)
    if [ -n "$MATCHES" ]; then
        echo -e "${RED}  ✗ Dangerous pattern found: $pattern${NC}"
        echo "$MATCHES"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}  ✓ No dangerous patterns${NC}"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $FAILED -ne 0 ]; then
    echo -e "${RED}Pre-commit checks FAILED${NC}"
    echo ""
    echo "To bypass (emergency only): git commit --no-verify"
    echo "To fix: Review violations above and update code"
    exit 1
else
    echo -e "${GREEN}All pre-commit checks passed ✓${NC}"
    exit 0
fi
