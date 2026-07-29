#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the policy-native GRSD SearchQA launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
export GRSD_VARIANT=grsd
export GRSD_MODULATION_LEVEL=${GRSD_MODULATION_LEVEL:-turn}
exec "${REPO_DIR}/examples/run_grsd_search_qwen3_local.sh" "$@"
