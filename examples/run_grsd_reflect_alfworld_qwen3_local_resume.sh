#!/usr/bin/env bash
set -euo pipefail

# Resume launcher for a GRSD Qwen3-1.7B ALFWorld run.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${HERE}/.." && pwd)}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_alfworld/grsd_qwen3_1.7b}

RESUME_FROM_STEP=${RESUME_FROM_STEP:-240}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-${CHECKPOINT_DIR}/global_step_${RESUME_FROM_STEP}}

if [[ ! -f "${RESUME_FROM_PATH}/data.pt" ]]; then
  echo "Missing dataloader state: ${RESUME_FROM_PATH}/data.pt" >&2
  exit 1
fi

for rank in {0..7}; do
  for state in model optim extra_state; do
    shard="${RESUME_FROM_PATH}/actor/${state}_world_size_8_rank_${rank}.pt"
    if [[ ! -s "${shard}" ]]; then
      echo "Missing or empty checkpoint shard: ${shard}" >&2
      exit 1
    fi
  done
done

# Set WANDB_RUN_ID/WANDB_ENTITY explicitly when resuming a hosted run. Leaving
# WANDB_RUN_ID unset starts a new run and avoids coupling this launcher to a
# personal project.
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  export WANDB_RESUME=${WANDB_RESUME:-must}
fi
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-Qwen3-1.7B-ALFWorld-Resume}

# Continue to the paper's Qwen3-1.7B ALFWorld horizon by default.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-600}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

echo "== GRSD resume =="
echo "  resume_from : ${RESUME_FROM_PATH}"
echo "  wandb run   : ${WANDB_ENTITY:-<unset>}/verl_agent_alfworld/${WANDB_RUN_ID:-<unset>} (resume=${WANDB_RESUME:-<unset>})"
echo "  exp name    : ${EXPERIMENT_NAME}"
echo "  total_steps : ${TOTAL_TRAINING_STEPS}"
echo "  val_before  : ${VAL_BEFORE_TRAIN} (logs validation at step ${RESUME_FROM_STEP})"
echo "  first train : step $((RESUME_FROM_STEP + 1))"

exec bash "${HERE}/run_grsd_reflect_alfworld_qwen3_local.sh" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  "$@"
