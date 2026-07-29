#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/data}
SEARCH_ASSET_ROOT=${SEARCH_ASSET_ROOT:-${DATA_ROOT}/searchR1}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN=$(command -v "${PYTHON_BIN}") || {
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  }
fi

INDEX_FILE=${INDEX_FILE:-${SEARCH_ASSET_ROOT}/e5_Flat.index}
CORPUS_FILE=${CORPUS_FILE:-${SEARCH_ASSET_ROOT}/wiki-18.jsonl}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-intfloat/e5-base-v2}
TOPK=${TOPK:-3}
PORT=${PORT:-8000}

for path in "${INDEX_FILE}" "${CORPUS_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing Search retrieval asset: ${path}" >&2
    echo "Run examples/data_preprocess/prepare_search.sh first." >&2
    exit 1
  fi
done

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/retrieval_server.py" \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk "${TOPK}" \
  --retriever_name "${RETRIEVER_NAME}" \
  --retriever_model "${RETRIEVER_MODEL}" \
  --faiss_gpu \
  --port "${PORT}"
