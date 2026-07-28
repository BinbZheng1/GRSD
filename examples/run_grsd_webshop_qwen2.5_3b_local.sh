#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-$(dirname "${REPO_DIR}")}
export GRSD_VARIANT=${GRSD_VARIANT:-reflect}
export MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/Qwen2.5-3B-Instruct}
export TP=${TP:-2}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-8}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-16}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_webshop/grsd_${GRSD_VARIANT}_qwen2.5_3b_local}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-${GRSD_VARIANT}-Qwen2.5-3B-WebShop}

exec "${SCRIPT_DIR}/run_grsd_webshop_qwen3_local.sh" "$@"
