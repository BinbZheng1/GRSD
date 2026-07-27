#!/usr/bin/env bash
set -euo pipefail

# Resume launcher for the GRSD-Reflect Qwen3-1.7B ALFWorld run.
#
# global_step_240 was saved after the step-240 update. Resume the original W&B
# run and keep initial validation enabled: it is explicitly logged at step 240,
# so the continued curve starts at 240 even if step 239 is absent. The first
# resumed training update is then logged at step 241.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${HERE}/.." && pwd)}
WORKSPACE=${WORKSPACE:-$(dirname "${REPO_DIR}")}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_alfworld/grsd_reflect_qwen3_1.7b_local_turn_alpha0.01_0713}

RESUME_FROM_STEP=${RESUME_FROM_STEP:-420}
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
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-Reflect-ALFWorld-Resume}

# Continue to step 320 by default (80 more updates); callers can override this.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-800}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

echo "== GRSD-Reflect resume =="
echo "  resume_from : ${RESUME_FROM_PATH}"
echo "  wandb run   : ${WANDB_ENTITY}/verl_agent_alfworld/${WANDB_RUN_ID} (resume=${WANDB_RESUME})"
echo "  exp name    : ${EXPERIMENT_NAME}"
echo "  total_steps : ${TOTAL_TRAINING_STEPS}"
echo "  val_before  : ${VAL_BEFORE_TRAIN} (logs validation at step ${RESUME_FROM_STEP})"
echo "  first train : step $((RESUME_FROM_STEP + 1))"

exec bash "${HERE}/run_grsd_reflect_alfworld_qwen3_local.sh" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  "$@"
