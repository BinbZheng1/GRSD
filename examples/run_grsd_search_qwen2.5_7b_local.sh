#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-$(dirname "${REPO_DIR}")}
export GRSD_VARIANT=${GRSD_VARIANT:-reflect}
export MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/Qwen2.5-7B-Instruct}
export TP=${TP:-4}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-8}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-16}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_search/grsd_${GRSD_VARIANT}_qwen2.5_7b_local}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-${GRSD_VARIANT}-Qwen2.5-7B-Search}

exec "${SCRIPT_DIR}/run_grsd_search_qwen3_local.sh" "$@"
