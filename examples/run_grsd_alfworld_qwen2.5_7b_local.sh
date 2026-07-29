#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-${REPO_DIR}}
MODEL_ROOT=${MODEL_ROOT:-${WORKSPACE}/models}
export GRSD_VARIANT=${GRSD_VARIANT:-grsd}
export MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-7B-Instruct}
export TP=${TP:-2}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-32}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-240}
case "${GRSD_VARIANT}" in
  external|original)
    export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_alfworld/external_reflection_qwen2.5_7b}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-External-Reflection-Ablation-Qwen2.5-7B-ALFWorld}
    ;;
  *)
    export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_alfworld/grsd_qwen2.5_7b}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-Qwen2.5-7B-ALFWorld}
    ;;
esac

exec "${SCRIPT_DIR}/run_grsd_alfworld_qwen3_local.sh" "$@"
