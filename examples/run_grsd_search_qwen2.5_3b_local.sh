#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-${REPO_DIR}}
MODEL_ROOT=${MODEL_ROOT:-${WORKSPACE}/models}
export GRSD_VARIANT=${GRSD_VARIANT:-grsd}
export MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-3B-Instruct}
export TP=${TP:-1}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-16}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
case "${GRSD_VARIANT}" in
  external|original)
    export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_search/external_reflection_qwen2.5_3b}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-External-Reflection-Ablation-Qwen2.5-3B-Search}
    ;;
  *)
    export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_search/grsd_qwen2.5_3b}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-Qwen2.5-3B-Search}
    ;;
esac

exec "${SCRIPT_DIR}/run_grsd_search_qwen3_local.sh" "$@"
