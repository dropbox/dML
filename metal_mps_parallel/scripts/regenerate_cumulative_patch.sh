#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_DIR="${REPO_ROOT}/pytorch-mps-fork"

print_usage() {
  cat <<'EOF'
Usage:
  regenerate_cumulative_patch.sh [--check] [BASE_REF] [OUT_FILE]

Options:
  --check   Do not write; verify OUT_FILE matches `git diff BASE_REF` and that
            any configured patch aliases are byte-identical.
EOF
}

mode="write"
while [ "${#}" -gt 0 ]; do
  case "${1}" in
    --check)
      mode="check"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: Unknown option: ${1}" >&2
      print_usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

BASE_REF="${1:-v2.9.1}"
PATCH_DIR="${REPO_ROOT}/patches"
OUT_FILE="${2:-${PATCH_DIR}/cumulative-v2.9.1-to-mps-stream-pool.patch}"

compute_md5() {
  local file="$1"
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "${file}"
    return 0
  fi
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "${file}" | awk '{print $1}'
    return 0
  fi
  return 1
}

alias_files=(
  "${PATCH_DIR}/cumulative-v2.9.1-to-mps-stream-pool.patch"
  # Keep historical cumulative patches in patches/archive/ (e.g. 022/029) out of the auto-synced alias set.
  "${PATCH_DIR}/archive/015-cumulative-final.patch"
)

if [ ! -d "${FORK_DIR}/.git" ]; then
  echo "ERROR: Missing PyTorch fork repo at: ${FORK_DIR}" >&2
  exit 2
fi

if ! git -C "${FORK_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: Not a git worktree: ${FORK_DIR}" >&2
  exit 2
fi

if ! git -C "${FORK_DIR}" rev-parse "${BASE_REF}^{commit}" >/dev/null 2>&1; then
  echo "ERROR: Base ref not found in fork: ${BASE_REF}" >&2
  exit 2
fi

HEAD_SHA="$(git -C "${FORK_DIR}" rev-parse HEAD)"

TMP_ROOT="$(mktemp -d "${REPO_ROOT}/.patch_tmp.XXXXXX")"
cleanup_tmp_root() {
  rm -rf "${TMP_ROOT}"
}
trap cleanup_tmp_root EXIT

changed_files=()
while IFS= read -r file; do
  [ -n "${file}" ] || continue
  changed_files+=("${file}")
done < <(git -C "${FORK_DIR}" diff --name-only "${BASE_REF}")

if [ "${#changed_files[@]}" -eq 0 ]; then
  echo "ERROR: No changes detected in pytorch-mps-fork versus ${BASE_REF}." >&2
  echo "Base: ${BASE_REF}" >&2
  echo "Head: ${HEAD_SHA}" >&2
  echo "" >&2
  echo "This usually means your fork checkout is still at the baseline ref (or otherwise missing the MPS stream pool changes)." >&2
  echo "Fix: check out the patched fork commit/branch (see patches/README.md for the expected fork HEAD), then re-run:" >&2
  echo "  ./scripts/regenerate_cumulative_patch.sh --check" >&2
  echo "" >&2
  echo "Refusing to continue to avoid accidentally overwriting ${OUT_FILE} with an empty patch." >&2
  exit 3
fi

allowed_prefixes=(
  "aten/src/ATen/mps/"
  "aten/src/ATen/native/mps/"
  # Sparse MPS operations (32.300-32.305 deadlock fixes)
  "aten/src/ATen/native/sparse/mps/"
  "aten/src/ATen/native/native_functions.yaml"
  # MPS hooks interface (for extending MPSHooksInterface with new methods)
  "aten/src/ATen/detail/MPSHooksInterface.h"
  # Python bindings for MPS module (release_current_thread_slot API)
  "torch/csrc/mps/Module.cpp"
  "torch/mps/__init__.py"
  "torch/mps/batch_queue.py"
  # PyTorch test file - parallel inference tests (P0-1 roadmap item)
  "test/test_mps.py"
)

unexpected_files=()
for file in "${changed_files[@]+"${changed_files[@]}"}"; do
  allowed=false
  for prefix in "${allowed_prefixes[@]+"${allowed_prefixes[@]}"}"; do
    if [[ "${file}" == "${prefix}"* ]]; then
      allowed=true
      break
    fi
  done
  if [ "${allowed}" = false ]; then
    unexpected_files+=("${file}")
  fi
done

if [ "${#unexpected_files[@]}" -gt 0 ]; then
  echo "ERROR: Refusing to generate patch; fork diff includes unexpected paths." >&2
  echo "Base: ${BASE_REF}" >&2
  echo "Head: ${HEAD_SHA}" >&2
  echo "" >&2
  printf '%s\n' "${unexpected_files[@]}" | sed 's/^/  - /' >&2
  echo "" >&2
  echo "Fix: remove accidental commits/files in pytorch-mps-fork (e.g. reports/), then retry." >&2
  exit 3
fi

expected_file="${TMP_ROOT}/expected.patch"
git -C "${FORK_DIR}" diff "${BASE_REF}" > "${expected_file}"

if [ "${mode}" = "check" ]; then
  if [ ! -f "${OUT_FILE}" ]; then
    echo "ERROR: Patch file does not exist: ${OUT_FILE}" >&2
    exit 4
  fi

  if ! cmp -s "${expected_file}" "${OUT_FILE}"; then
    echo "ERROR: Patch file is out-of-sync with fork diff." >&2
    echo "Base: ${BASE_REF}" >&2
    echo "Head: ${HEAD_SHA}" >&2
    echo "Patch: ${OUT_FILE}" >&2
    echo "" >&2
    echo "Fix: run ./scripts/regenerate_cumulative_patch.sh to rewrite the patch." >&2
    exit 5
  fi

  if [[ "${OUT_FILE}" == "${PATCH_DIR}/"* ]]; then
    for alias in "${alias_files[@]}"; do
      [ "${alias}" = "${OUT_FILE}" ] && continue
      if [ ! -f "${alias}" ]; then
        echo "ERROR: Missing patch alias: ${alias}" >&2
        exit 6
      fi
      if ! cmp -s "${OUT_FILE}" "${alias}"; then
        echo "ERROR: Patch alias out-of-sync: ${alias}" >&2
        echo "Fix: run ./scripts/regenerate_cumulative_patch.sh to sync aliases." >&2
        exit 6
      fi
    done
  fi
else
  mkdir -p "$(dirname "${OUT_FILE}")"
  cp -f "${expected_file}" "${OUT_FILE}"
fi

if [ "${mode}" != "check" ] && [[ "${OUT_FILE}" == "${PATCH_DIR}/"* ]]; then
  for alias in "${alias_files[@]}"; do
    [ "${alias}" = "${OUT_FILE}" ] && continue
    mkdir -p "$(dirname "${alias}")"
    cp -f "${OUT_FILE}" "${alias}"
  done
fi

if [ "${mode}" = "check" ]; then
  echo "Verified patch: ${OUT_FILE}"
else
  echo "Wrote patch: ${OUT_FILE}"
fi
echo "Base: ${BASE_REF}"
echo "Head: ${HEAD_SHA}"

if [[ "${OUT_FILE}" == "${PATCH_DIR}/"* ]]; then
  md5_value=""
  if md5_value="$(compute_md5 "${PATCH_DIR}/cumulative-v2.9.1-to-mps-stream-pool.patch")"; then
    echo "MD5: ${md5_value}"
  fi
  if [ "${mode}" = "check" ]; then
    echo "Verified patch aliases:"
  else
    echo "Synced patch aliases:"
  fi
  printf '  - %s\n' "${alias_files[@]}"
fi

git -C "${FORK_DIR}" diff --stat "${BASE_REF}"
