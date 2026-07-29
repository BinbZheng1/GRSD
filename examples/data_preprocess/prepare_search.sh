#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/data}
PYTHON_BIN=${PYTHON_BIN:-python}
DOWNLOAD_RETRIEVER_ASSETS=${DOWNLOAD_RETRIEVER_ASSETS:-true}

if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN=$(command -v "${PYTHON_BIN}") || {
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  }
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/preprocess_search_r1_dataset.py" \
  --local_dir "${DATA_ROOT}/search" \
  "$@"

case "${DOWNLOAD_RETRIEVER_ASSETS,,}" in
  true) ;;
  false) exit 0 ;;
  *)
    echo "DOWNLOAD_RETRIEVER_ASSETS must be true or false." >&2
    exit 1
    ;;
esac

SEARCH_ASSET_ROOT=${SEARCH_ASSET_ROOT:-${DATA_ROOT}/searchR1}
mkdir -p "${SEARCH_ASSET_ROOT}"
"${PYTHON_BIN}" "${REPO_DIR}/examples/search/searchr1_download.py" \
  --local_dir "${SEARCH_ASSET_ROOT}"

if [[ ! -f "${SEARCH_ASSET_ROOT}/e5_Flat.index" ]]; then
  cat "${SEARCH_ASSET_ROOT}/part_aa" "${SEARCH_ASSET_ROOT}/part_ab" \
    > "${SEARCH_ASSET_ROOT}/e5_Flat.index"
fi
if [[ ! -f "${SEARCH_ASSET_ROOT}/wiki-18.jsonl" ]]; then
  gzip -dk "${SEARCH_ASSET_ROOT}/wiki-18.jsonl.gz"
fi
