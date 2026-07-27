#!/usr/bin/env bash
set -euo pipefail

# Local GRSD + Search-based QA (Search-R1 style) launcher.
# Defaults to policy-native GRSD-Reflect with turn-level modulation. Set
# GRSD_VARIANT=original explicitly for the external-LLM GRSD ablation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-$(dirname "${REPO_DIR}")}
PYTHON_BIN=${PYTHON_BIN:-${WORKSPACE}/venv_search_qa/bin/python}
GRSD_VARIANT=${GRSD_VARIANT:-reflect}

case "${GRSD_VARIANT}" in
  original)
    TRAINER_MODULE=verl.trainer.main_grsd
    DEFAULT_CHECKPOINT_NAME=grsd_qwen3_1.7b_local
    DEFAULT_EXPERIMENT_NAME=GRSD-Qwen3-1.7B-Search
    ;;
  reflect)
    TRAINER_MODULE=verl.trainer.main_grsd_reflect
    DEFAULT_CHECKPOINT_NAME=grsd_reflect_qwen3_1.7b_local
    DEFAULT_EXPERIMENT_NAME=GRSD-Reflect-Qwen3-1.7B-Search
    ;;
  *)
    echo "GRSD_VARIANT must be 'original' or 'reflect', got: ${GRSD_VARIANT}" >&2
    exit 1
    ;;
esac

MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/Qwen3-1.7B}
DATA_DIR=${DATA_DIR:-${WORKSPACE}/data/searchR1_processed_direct}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.parquet}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_search/${DEFAULT_CHECKPOINT_NAME}}

SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:8000/retrieve}
SEARCH_TOPK=${SEARCH_TOPK:-3}

ENGINE=${ENGINE:-vllm}
N_GPUS=${N_GPUS:-8}
TP=${TP:-2}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
RAY_ADDRESS=${RAY_ADDRESS:-local}

# Reflection/prior generations add policy rollouts beyond the task rollout.
# Keep the default smaller than GRPO's 128 prompts for the first full run.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
GROUP_SIZE=${GROUP_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-16}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-32}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-4}
HISTORY_LENGTH=${HISTORY_LENGTH:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}

LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-0.001}
INVALID_ACTION_PENALTY_COEF=${INVALID_ACTION_PENALTY_COEF:-0.01}

# GRSD bidirectional turn-level advantage modulation.
GRSD_LAMBDA=${GRSD_LAMBDA:-0.5}
GRSD_ETA=${GRSD_ETA:-0.0}
# Match RLSD clip_eps=0.2. With lambda=0.5, both turn- and token-level
# modulation scale task advantages within [0.9, 1.1].
GRSD_G_HAT_MAX=${GRSD_G_HAT_MAX:-0.2}
# -1 keeps lambda fixed; a positive N linearly decays it to zero by step N.
GRSD_DECAY_STEPS=${GRSD_DECAY_STEPS:--1}
GRSD_WARMDOWN_STEPS=${GRSD_WARMDOWN_STEPS:-${GRSD_DECAY_STEPS}}
# "turn" shares one modulation coefficient across a turn; "token" computes
# it per token. token_norm is only used by token-level modulation.
GRSD_MODULATION_LEVEL=${GRSD_MODULATION_LEVEL:-turn}
GRSD_TOKEN_NORM=${GRSD_TOKEN_NORM:-none}
SKILLS_DIR=${SKILLS_DIR:-skills/search}
SKILL_ALL=${SKILL_ALL:-false}

# Original GRSD: the external LLM synthesizes reflections and group priors.
REFLECT_TEMPERATURE=${REFLECT_TEMPERATURE:-0.0}
REFLECT_MAX_TOKENS=${REFLECT_MAX_TOKENS:-1024}
REFLECT_TIMEOUT=${REFLECT_TIMEOUT:-60.0}
REFLECT_MAX_RETRIES=${REFLECT_MAX_RETRIES:-2}

# GRSD-Reflect only: the policy synthesizes reflections/priors and the external
# LLM scores them. These options are ignored by the original GRSD variant.
REFLECT_LOSS_COEF=${REFLECT_LOSS_COEF:-0.1}
REFLECT_DO_SAMPLE=${REFLECT_DO_SAMPLE:-true}
REFLECT_MAX_TURNS=${REFLECT_MAX_TURNS:-4}
REFLECT_MAX_CHARS_PER_OBS=${REFLECT_MAX_CHARS_PER_OBS:-1200}
REFLECT_SAMPLE_FREQ=${REFLECT_SAMPLE_FREQ:-5}
REFLECT_SAMPLE_DIR=${REFLECT_SAMPLE_DIR:-${CHECKPOINT_DIR}/reflect_samples}
JUDGE_TEMPERATURE=${JUDGE_TEMPERATURE:-0.0}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-16}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-60.0}
JUDGE_MAX_RETRIES=${JUDGE_MAX_RETRIES:-2}
# ReflectionJudge uses a bounded ThreadPoolExecutor for network-bound scoring.
JUDGE_MAX_CONCURRENCY=${JUDGE_MAX_CONCURRENCY:-16}

# External LLM is used only as the rubric judge, not to synthesize priors.
# JUDGE_* takes precedence; LLM_* aliases are accepted for compatibility.
JUDGE_API_BASE=${JUDGE_API_BASE:-${LLM_API_BASE:-https://api.openai.com/v1}}
JUDGE_API_KEY=${JUDGE_API_KEY:-${LLM_API_KEY:-}}
JUDGE_MODEL=${JUDGE_MODEL:-${LLM_MODEL:-gpt-4o-mini}}
GRSD_PROBE_ATTEMPTS=${GRSD_PROBE_ATTEMPTS:-3}

if [[ "${GRSD_VARIANT}" == "original" ]]; then
  VARIANT_ARGS=(
    +algorithm.grsd.reflect_temperature="${REFLECT_TEMPERATURE}"
    +algorithm.grsd.reflect_max_tokens="${REFLECT_MAX_TOKENS}"
    +algorithm.grsd.reflect_timeout="${REFLECT_TIMEOUT}"
    +algorithm.grsd.reflect_max_retries="${REFLECT_MAX_RETRIES}"
    +algorithm.grsd.reflect_max_turns="${REFLECT_MAX_TURNS}"
    +algorithm.grsd.reflect_max_chars_per_obs="${REFLECT_MAX_CHARS_PER_OBS}"
  )
else
  VARIANT_ARGS=(
    +algorithm.grsd.modulation_level="${GRSD_MODULATION_LEVEL}"
    +algorithm.grsd.token_norm="${GRSD_TOKEN_NORM}"
    +algorithm.grsd.reflect_loss_coef="${REFLECT_LOSS_COEF}"
    +algorithm.grsd.reflect_do_sample="${REFLECT_DO_SAMPLE}"
    +algorithm.grsd.reflect_max_turns="${REFLECT_MAX_TURNS}"
    +algorithm.grsd.reflect_max_chars_per_obs="${REFLECT_MAX_CHARS_PER_OBS}"
    +algorithm.grsd.reflect_sample_freq="${REFLECT_SAMPLE_FREQ}"
    +algorithm.grsd.reflect_sample_dir="${REFLECT_SAMPLE_DIR}"
    +algorithm.grsd.judge_temperature="${JUDGE_TEMPERATURE}"
    +algorithm.grsd.judge_max_tokens="${JUDGE_MAX_TOKENS}"
    +algorithm.grsd.judge_timeout="${JUDGE_TIMEOUT}"
    +algorithm.grsd.judge_max_retries="${JUDGE_MAX_RETRIES}"
    +algorithm.grsd.judge_max_concurrency="${JUDGE_MAX_CONCURRENCY}"
  )
fi

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-240}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-40}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_DIR=${WANDB_DIR:-${WORKSPACE}/wandb}
WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE:-${WORKSPACE}/.secrets/wandb_api_key}
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
  WANDB_API_KEY=$(<"${WANDB_API_KEY_FILE}")
fi

export HF_HOME=${HF_HOME:-${WORKSPACE}/.cache/huggingface}
export WANDB_MODE
export WANDB_DIR
export WANDB_API_KEY
export JUDGE_API_BASE
export JUDGE_API_KEY
export JUDGE_MODEL
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
export RAY_LOG_TO_STDOUT=${RAY_LOG_TO_STDOUT:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export no_proxy=${no_proxy:-localhost,127.0.0.1,0.0.0.0}
export NO_PROXY=${NO_PROXY:-${no_proxy}}

cd "${REPO_DIR}"
mkdir -p "${WANDB_DIR}" "${CHECKPOINT_DIR}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
  echo "WANDB_MODE=online but WANDB_API_KEY is not set and ${HOME}/.netrc does not exist." >&2
  exit 1
fi

if [[ -z "${JUDGE_API_KEY}" ]]; then
  echo "GRSD requires JUDGE_API_KEY (or LLM_API_KEY)." >&2
  echo "Export one of those variables before launching." >&2
  exit 1
fi

for path in \
  "${PYTHON_BIN}" \
  "${MODEL_PATH}/config.json" \
  "${MODEL_PATH}/tokenizer.json" \
  "${TRAIN_FILE}" \
  "${VAL_FILE}" \
  "${SKILLS_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

# Retrieval must be available before either validation or rollout starts.
retrieve_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${SEARCH_URL}" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "topk": 1, "return_scores": false}' --max-time 10 || true)
if [[ "${retrieve_status}" != "200" ]]; then
  echo "Retrieval server at ${SEARCH_URL} is not responding (HTTP ${retrieve_status})." >&2
  exit 1
fi
echo "Retrieval server OK: ${SEARCH_URL}"

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("CUDA is not visible. Run this script inside a GPU job/container.")
PY

# Fail before allocating training workers when the judge is misconfigured.
"${PYTHON_BIN}" - "${GRSD_PROBE_ATTEMPTS}" <<'PY'
import os
import sys
import time
from openai import OpenAI

attempts = int(sys.argv[1])
base = os.environ["JUDGE_API_BASE"]
model = os.environ["JUDGE_MODEL"]
print(f"[GRSD] probing external endpoint base={base} model={model}")
client = OpenAI(api_key=os.environ["JUDGE_API_KEY"], base_url=base, timeout=30.0)
last_error = None
for attempt in range(1, attempts + 1):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with only the digit 3."}],
            temperature=0.0,
            max_tokens=16,
            extra_body={"thinking": {"type": "disabled"}},
        )
        print(f"[GRSD] endpoint probe succeeded: {response.choices[0].message.content!r}")
        break
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"[GRSD] probe {attempt}/{attempts} failed: {last_error}")
        if attempt < attempts:
            time.sleep(2 * attempt)
else:
    raise SystemExit(f"GRSD external endpoint unavailable: {last_error}")
PY

set -x
"${PYTHON_BIN}" -u -m "${TRAINER_MODULE}" \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
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
  actor_rollout_ref.rollout.enforce_eager=False \
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
  "${VARIANT_ARGS[@]}" \
  env.env_name=search \
  env.seed=0 \
  env.max_steps="${MAX_ENV_STEPS}" \
  env.rollout.n="${GROUP_SIZE}" \
  env.history_length="${HISTORY_LENGTH}" \
  env.search.search_url="${SEARCH_URL}" \
  env.search.topk="${SEARCH_TOPK}" \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=verl_agent_search \
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
