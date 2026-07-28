#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-$(dirname "${REPO_DIR}")}
export GRSD_VARIANT=${GRSD_VARIANT:-reflect}
export MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/Qwen2.5-7B-Instruct}
export TP=${TP:-2}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-32}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_alfworld/grsd_${GRSD_VARIANT}_qwen2.5_7b_local}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-${GRSD_VARIANT}-Qwen2.5-7B-ALFWorld}

exec "${SCRIPT_DIR}/run_grsd_alfworld_qwen3_local.sh" "$@"
