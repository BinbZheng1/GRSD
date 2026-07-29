#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-${REPO_DIR}}
MODEL_ROOT=${MODEL_ROOT:-${WORKSPACE}/models}
DATA_ROOT=${DATA_ROOT:-${WORKSPACE}/data}

# GRSD is the policy-native reflection method. Legacy names remain aliases so
# existing launch commands continue to work.
GRSD_VARIANT=${GRSD_VARIANT:-grsd}
GRSD_MODULATION_LEVEL=${GRSD_MODULATION_LEVEL:-turn}
case "${GRSD_VARIANT}" in
  grsd|reflect)
    export GRSD_MODULATION_LEVEL
    exec "${REPO_DIR}/examples/run_grsd_reflect_alfworld_qwen3_local.sh" "$@"
    ;;
  external|original)
    ;;
  *)
    echo "GRSD_VARIANT must be 'grsd' or 'external' (legacy: 'reflect' or 'original'), got: ${GRSD_VARIANT}" >&2
    exit 1
    ;;
esac

PYTHON_BIN=${PYTHON_BIN:-python}
if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN=$(command -v "${PYTHON_BIN}") || {
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  }
fi

MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/Qwen3-1.7B}
DATA_DIR=${DATA_DIR:-${DATA_ROOT}/alfworld/metadata}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.parquet}
ALFWORLD_DATA=${ALFWORLD_DATA:-${DATA_ROOT}/alfworld}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_alfworld/external_reflection_qwen3_1.7b}

ENGINE=${ENGINE:-vllm}
N_GPUS=${N_GPUS:-8}
TP=${TP:-2}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
RAY_ADDRESS=${RAY_ADDRESS:-local}
ENV_CPUS=${ENV_CPUS:-0.1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-32}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-32}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-50}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}

LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-0.01}
INVALID_ACTION_PENALTY_COEF=${INVALID_ACTION_PENALTY_COEF:-0.1}

# GRSD hyperparameters (bidirectional turn-level advantage modulation).
# lambda=0.5  modulation strength.
# g_hat_max=0.2 matches RLSD clip_eps; with lambda=0.5 it bounds modulation
# to +/-10% (turn-level, after group mean-abs normalization).
# eta=0.0     dead-zone for |q_norm|; turns with |q_norm|<=eta get no modulation.
GRSD_LAMBDA=${GRSD_LAMBDA:-0.5}
GRSD_ETA=${GRSD_ETA:-0.0}
GRSD_G_HAT_MAX=${GRSD_G_HAT_MAX:-0.2}
# -1 keeps lambda fixed; a positive N linearly decays it to zero by step N.
GRSD_DECAY_STEPS=${GRSD_DECAY_STEPS:--1}
GRSD_WARMDOWN_STEPS=${GRSD_WARMDOWN_STEPS:-${GRSD_DECAY_STEPS}}
SKILLS_DIR=${SKILLS_DIR:-skills/alfworld}
SKILL_ALL=${SKILL_ALL:-false}

# External LLM for online skill reflection / contrastive prior.
# temperature=0.0 for deterministic (reproducible) priors.
JUDGE_API_BASE=${JUDGE_API_BASE:-${LLM_API_BASE:-https://api.openai.com/v1}}
JUDGE_API_KEY=${JUDGE_API_KEY:-${LLM_API_KEY:-}}
JUDGE_MODEL=${JUDGE_MODEL:-${LLM_MODEL:-gpt-4o-mini}}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-600}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-600}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-20}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRSD-External-Reflection-Ablation-Qwen3-1.7B-ALFWorld}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-${WORKSPACE}/wandb}
WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE:-${REPO_DIR}/.secrets/wandb_api_key}
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
  WANDB_API_KEY=$(<"${WANDB_API_KEY_FILE}")
fi

export ALFWORLD_DATA
export HF_HOME=${HF_HOME:-${WORKSPACE}/.cache/huggingface}
export WANDB_MODE
export WANDB_DIR
export WANDB_API_KEY
export JUDGE_API_BASE
export JUDGE_API_KEY
export JUDGE_MODEL
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export no_proxy=${no_proxy:-localhost,127.0.0.1,0.0.0.0}
export NO_PROXY=${NO_PROXY:-${no_proxy}}
# Pipe Ray worker stdout/stderr back to the parent process so all output
# appears in the caller's log file (e.g. grsd_retry.log).
export RAY_LOG_TO_STDOUT=1

cd "${REPO_DIR}"
mkdir -p "${WANDB_DIR}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
  echo "WANDB_MODE=online but WANDB_API_KEY is not set and ${HOME}/.netrc does not exist." >&2
  echo "Run: export WANDB_API_KEY=<your_wandb_key>" >&2
  echo "Or disable online logging with: WANDB_MODE=offline TRAINER_LOGGER=\"['console']\" bash $0" >&2
  exit 1
fi

for path in \
  "${PYTHON_BIN}" \
  "${MODEL_PATH}/config.json" \
  "${MODEL_PATH}/tokenizer.json" \
  "${TRAIN_FILE}" \
  "${VAL_FILE}" \
  "${SKILLS_DIR}" \
  "${ALFWORLD_DATA}/json_2.1.1/train" \
  "${ALFWORLD_DATA}/json_2.1.1/valid_seen" \
  "${ALFWORLD_DATA}/json_2.1.1/valid_unseen" \
  "${ALFWORLD_DATA}/logic/alfred.pddl" \
  "${ALFWORLD_DATA}/logic/alfred.twl2"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("CUDA is not visible. Run this script inside a GPU job/container.")
PY

# Sanity-check the external LLM endpoint before launching the long job.
GRSD_PROBE_ATTEMPTS=${GRSD_PROBE_ATTEMPTS:-6}
"${PYTHON_BIN}" - "${GRSD_PROBE_ATTEMPTS}" <<'PY'
import os
import sys
import time
from openai import OpenAI

base = os.environ["JUDGE_API_BASE"]
key = os.environ["JUDGE_API_KEY"]
model = os.environ["JUDGE_MODEL"]
attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 6
print(f"[GRSD] probing LLM endpoint base={base} model={model} attempts={attempts}")

client = OpenAI(api_key=key, base_url=base, timeout=30.0)
last_err = None
for i in range(1, attempts + 1):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            temperature=0.0,
            max_tokens=256,
        )
        content = resp.choices[0].message.content
        print(f"[GRSD] probe {i}/{attempts} OK: finish_reason={resp.choices[0].finish_reason}, content={content!r}")
        print("[GRSD] endpoint reachable, proceeding.")
        break
    except Exception as e:
        last_err = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[GRSD] probe {i}/{attempts} FAILED: {last_err}")
        if i < attempts:
            time.sleep(2.0 * i)
else:
    raise SystemExit(f"[GRSD] external LLM endpoint unreachable after {attempts} attempts: {last_err}")
PY

set -x
"${PYTHON_BIN}" -u -m verl.trainer.main_grsd \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP}" \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef="${INVALID_ACTION_PENALTY_COEF}" \
  "actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model']" \
  algorithm.use_kl_in_reward=False \
  +algorithm.grsd.grsd_lambda="${GRSD_LAMBDA}" \
  +algorithm.grsd.eta="${GRSD_ETA}" \
  +algorithm.grsd.g_hat_max="${GRSD_G_HAT_MAX}" \
  +algorithm.grsd.warmdown_steps="${GRSD_WARMDOWN_STEPS}" \
  +algorithm.grsd.skills_dir="${SKILLS_DIR}" \
  +algorithm.grsd.skill_all="${SKILL_ALL}" \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps="${MAX_ENV_STEPS}" \
  env.rollout.n="${GROUP_SIZE}" \
  env.resources_per_worker.num_cpus="${ENV_CPUS}" \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=verl_agent_alfworld \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.ray_wait_register_center_timeout=600 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  +ray_init.address="${RAY_ADDRESS}" \
  ray_init.num_cpus="${RAY_NUM_CPUS}" \
  "$@"
