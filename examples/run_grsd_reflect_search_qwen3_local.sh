#!/usr/bin/env bash
set -euo pipefail

# Policy-native GRSD-Reflect SearchQA launcher. All shared SearchQA settings
# and the Reflect-specific defaults live in run_grsd_search_qwen3_local.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-$(dirname "${REPO_DIR}")}

export GRSD_VARIANT=reflect
export GRSD_MODULATION_LEVEL=${GRSD_MODULATION_LEVEL:-turn}
exec "${REPO_DIR}/examples/run_grsd_search_qwen3_local.sh" "$@"
