#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/data}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN=$(command -v "${PYTHON_BIN}") || {
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  }
fi

export ALFWORLD_DATA=${ALFWORLD_DATA:-${DATA_ROOT}/alfworld}
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare.py" \
  --task alfworld \
  --local-dir "${DATA_ROOT}/alfworld/metadata" \
  "$@"
