#!/usr/bin/env bash
set -euo pipefail

# Local GRSD + WebShop launcher for Qwen3-1.7B. WebShop requires Python 3.10,
# JDK >= 11, product JSON files, and a local Pyserini/Lucene index.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-$(dirname "${REPO_DIR}")}
PYTHON_BIN=${PYTHON_BIN:-${WORKSPACE}/venv_webshop/bin/python}
GRSD_VARIANT=${GRSD_VARIANT:-reflect}

case "${GRSD_VARIANT}" in
  original)
    TRAINER_MODULE=verl.trainer.main_grsd
    DEFAULT_CHECKPOINT_NAME=grsd_qwen3_1.7b_local
    DEFAULT_EXPERIMENT_NAME=GRSD-Qwen3-1.7B-WebShop
    ;;
  reflect)
    TRAINER_MODULE=verl.trainer.main_grsd_reflect
    DEFAULT_CHECKPOINT_NAME=grsd_reflect_qwen3_1.7b_local
    DEFAULT_EXPERIMENT_NAME=GRSD-Reflect-Qwen3-1.7B-WebShop
    ;;
  *)
    echo "GRSD_VARIANT must be 'original' or 'reflect', got: ${GRSD_VARIANT}" >&2
    exit 1
    ;;
esac

MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/Qwen3-1.7B}
DATA_DIR=${DATA_DIR:-${WORKSPACE}/data/verl-agent/text}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.parquet}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/verl_agent_webshop/${DEFAULT_CHECKPOINT_NAME}}

WEBSHOP_SOURCE_DIR=${REPO_DIR}/agent_system/environments/env_package/webshop/webshop
WEBSHOP_ASSET_ROOT=${WEBSHOP_ASSET_ROOT:-${WORKSPACE}/data/webshop}
WEBSHOP_DATA_DIR=${WEBSHOP_DATA_DIR:-${WEBSHOP_ASSET_ROOT}/data}
WEBSHOP_INDEX_DIR=${WEBSHOP_INDEX_DIR:-${WEBSHOP_ASSET_ROOT}/search_engine/indexes}
WEBSHOP_JAVA_HOME=${WEBSHOP_JAVA_HOME:-${WEBSHOP_ASSET_ROOT}/jdk-11}
WEBSHOP_USE_SMALL=${WEBSHOP_USE_SMALL:-True}
WEBSHOP_HUMAN_GOALS=${WEBSHOP_HUMAN_GOALS:-False}

ENGINE=${ENGINE:-vllm}
N_GPUS=${N_GPUS:-8}
TP=${TP:-2}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
RAY_ADDRESS=${RAY_ADDRESS:-local}
ENV_CPUS=${ENV_CPUS:-0.1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-8}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-16}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
REFLECT_MAX_PROMPT_LENGTH=${REFLECT_MAX_PROMPT_LENGTH:-8096}
TEACHER_MAX_PROMPT_LENGTH=${TEACHER_MAX_PROMPT_LENGTH:-4096}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-8608}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-15}
HISTORY_LENGTH=${HISTORY_LENGTH:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}

LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-0.01}
INVALID_ACTION_PENALTY_COEF=${INVALID_ACTION_PENALTY_COEF:-0.1}

GRSD_LAMBDA=${GRSD_LAMBDA:-0.5}
GRSD_ETA=${GRSD_ETA:-0.0}
GRSD_G_HAT_MAX=${GRSD_G_HAT_MAX:-0.2}
GRSD_DECAY_STEPS=${GRSD_DECAY_STEPS:--1}
GRSD_WARMDOWN_STEPS=${GRSD_WARMDOWN_STEPS:-${GRSD_DECAY_STEPS}}
GRSD_MODULATION_LEVEL=${GRSD_MODULATION_LEVEL:-turn}
GRSD_TOKEN_NORM=${GRSD_TOKEN_NORM:-none}
SKILLS_DIR=${SKILLS_DIR:-skills/webshop}
SKILL_ALL=${SKILL_ALL:-false}

# Original GRSD uses the external service to synthesize reflections and priors.
REFLECT_TEMPERATURE=${REFLECT_TEMPERATURE:-0.0}
REFLECT_MAX_TOKENS=${REFLECT_MAX_TOKENS:-1024}
REFLECT_TIMEOUT=${REFLECT_TIMEOUT:-60.0}
REFLECT_MAX_RETRIES=${REFLECT_MAX_RETRIES:-2}

# GRSD-Reflect uses the policy for reflection/prior generation and an optional
# external rubric judge for the auxiliary reflection update.
REFLECT_LOSS_COEF=${REFLECT_LOSS_COEF:-0.1}
REFLECT_DO_SAMPLE=${REFLECT_DO_SAMPLE:-true}
REFLECT_MAX_TURNS=${REFLECT_MAX_TURNS:-15}
REFLECT_MAX_CHARS_PER_OBS=${REFLECT_MAX_CHARS_PER_OBS:-1200}
REFLECT_SAMPLE_FREQ=${REFLECT_SAMPLE_FREQ:-5}
REFLECT_SAMPLE_DIR=${REFLECT_SAMPLE_DIR:-${CHECKPOINT_DIR}/reflect_samples}
JUDGE_ENABLED=${JUDGE_ENABLED:-true}
case "${JUDGE_ENABLED,,}" in
  true|false) ;;
  *)
    echo "JUDGE_ENABLED must be true or false, got: ${JUDGE_ENABLED}" >&2
    exit 1
    ;;
esac
JUDGE_TEMPERATURE=${JUDGE_TEMPERATURE:-0.0}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-16}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-60.0}
JUDGE_MAX_RETRIES=${JUDGE_MAX_RETRIES:-2}
JUDGE_MAX_CONCURRENCY=${JUDGE_MAX_CONCURRENCY:-16}

JUDGE_API_BASE=${JUDGE_API_BASE:-${LLM_API_BASE:-https://api.openai.com/v1}}
JUDGE_API_KEY=${JUDGE_API_KEY:-${LLM_API_KEY:-}}
JUDGE_MODEL=${JUDGE_MODEL:-${LLM_MODEL:-gpt-4o-mini}}
GRSD_PROBE_ATTEMPTS=${GRSD_PROBE_ATTEMPTS:-3}

if [[ "${GRSD_VARIANT}" == "original" ]]; then
  VARIANT_ARGS=(
    +algorithm.grsd.teacher_max_prompt_length="${TEACHER_MAX_PROMPT_LENGTH}"
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
    +algorithm.grsd.reflect_max_prompt_length="${REFLECT_MAX_PROMPT_LENGTH}"
    +algorithm.grsd.reflect_loss_coef="${REFLECT_LOSS_COEF}"
    +algorithm.grsd.reflect_do_sample="${REFLECT_DO_SAMPLE}"
    +algorithm.grsd.reflect_max_turns="${REFLECT_MAX_TURNS}"
    +algorithm.grsd.reflect_max_chars_per_obs="${REFLECT_MAX_CHARS_PER_OBS}"
    +algorithm.grsd.reflect_sample_freq="${REFLECT_SAMPLE_FREQ}"
    +algorithm.grsd.reflect_sample_dir="${REFLECT_SAMPLE_DIR}"
    +algorithm.grsd.judge_enabled="${JUDGE_ENABLED}"
    +algorithm.grsd.judge_temperature="${JUDGE_TEMPERATURE}"
    +algorithm.grsd.judge_max_tokens="${JUDGE_MAX_TOKENS}"
    +algorithm.grsd.judge_timeout="${JUDGE_TIMEOUT}"
    +algorithm.grsd.judge_max_retries="${JUDGE_MAX_RETRIES}"
    +algorithm.grsd.judge_max_concurrency="${JUDGE_MAX_CONCURRENCY}"
  )
fi

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-240}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-20}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-${WORKSPACE}/wandb}
WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE:-${WORKSPACE}/.secrets/wandb_api_key}
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
  WANDB_API_KEY=$(<"${WANDB_API_KEY_FILE}")
fi

if [[ "${SKILLS_DIR}" = /* ]]; then
  SKILLS_PATH=${SKILLS_DIR}
else
  SKILLS_PATH=${REPO_DIR}/${SKILLS_DIR}
fi

export HF_HOME=${HF_HOME:-${WORKSPACE}/.cache/huggingface}
export JAVA_HOME=${WEBSHOP_JAVA_HOME}
export PATH=${JAVA_HOME}/bin:$(dirname "${PYTHON_BIN}"):${PATH}
export _JAVA_OPTIONS=${_JAVA_OPTIONS:-"-XX:+UseSerialGC -Xss512k -Xmx1g"}
export WANDB_MODE
export WANDB_DIR
export WANDB_API_KEY
export JUDGE_API_BASE
export JUDGE_API_KEY
export JUDGE_MODEL
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
export RAY_LOG_TO_STDOUT=${RAY_LOG_TO_STDOUT:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export DATASETS_MAX_WORKERS=${DATASETS_MAX_WORKERS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export no_proxy=${no_proxy:-localhost,127.0.0.1,0.0.0.0}
export NO_PROXY=${NO_PROXY:-${no_proxy}}

ulimit -u 65536 2>/dev/null || true

case "${WEBSHOP_USE_SMALL,,}" in
  true|1)
    PRODUCT_FILE=${WEBSHOP_DATA_DIR}/items_shuffle_1000.json
    ATTRIBUTE_FILE=${WEBSHOP_DATA_DIR}/items_ins_v2_1000.json
    ;;
  false|0)
    PRODUCT_FILE=${WEBSHOP_DATA_DIR}/items_shuffle.json
    ATTRIBUTE_FILE=${WEBSHOP_DATA_DIR}/items_ins_v2.json
    ;;
  *)
    echo "WEBSHOP_USE_SMALL must be True/False or 1/0, got: ${WEBSHOP_USE_SMALL}" >&2
    exit 1
    ;;
esac
HUMAN_GOALS_FILE=${WEBSHOP_DATA_DIR}/items_human_ins.json

cd "${REPO_DIR}"
mkdir -p "${WANDB_DIR}" "${CHECKPOINT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing WebShop Python executable: ${PYTHON_BIN}" >&2
  echo "Create a Python 3.10 environment with the WebShop requirements." >&2
  exit 1
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
  echo "WANDB_MODE=online but WANDB_API_KEY is not set and ${HOME}/.netrc does not exist." >&2
  exit 1
fi

if [[ "${GRSD_VARIANT}" == "original" || "${JUDGE_ENABLED,,}" == "true" ]]; then
  if [[ -z "${JUDGE_API_KEY}" ]]; then
    echo "This GRSD configuration requires JUDGE_API_KEY (or LLM_API_KEY)." >&2
    exit 1
  fi
fi

for path in \
  "${MODEL_PATH}/config.json" \
  "${MODEL_PATH}/tokenizer.json" \
  "${TRAIN_FILE}" \
  "${VAL_FILE}" \
  "${PRODUCT_FILE}" \
  "${ATTRIBUTE_FILE}" \
  "${HUMAN_GOALS_FILE}" \
  "${WEBSHOP_INDEX_DIR}" \
  "${SKILLS_PATH}/skill_mapping.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

if ! compgen -G "${WEBSHOP_INDEX_DIR}/*" >/dev/null; then
  echo "Lucene index is empty: ${WEBSHOP_INDEX_DIR}" >&2
  exit 1
fi

if [[ ! -x "${JAVA_HOME}/bin/java" ]]; then
  echo "JDK not found at ${JAVA_HOME}/bin/java; WebShop requires JDK >= 11." >&2
  exit 1
fi

java_version_output=$("${JAVA_HOME}/bin/java" -version 2>&1 | grep -m1 -E '^(openjdk|java) version' || true)
if [[ "${java_version_output}" =~ \"([0-9]+)(\.([0-9]+))? ]]; then
  java_major=${BASH_REMATCH[1]}
  if [[ "${java_major}" == "1" ]]; then
    java_major=${BASH_REMATCH[3]}
  fi
else
  echo "Cannot parse Java version: ${java_version_output}" >&2
  exit 1
fi
if (( java_major < 11 )); then
  echo "WebShop/Pyserini requires JDK >= 11; found: ${java_version_output}" >&2
  exit 1
fi

link_webshop_asset() {
  local source_path=$1
  local target_path=$2

  if [[ -e "${target_path}" && ! -L "${target_path}" ]]; then
    echo "Refusing to replace the real path: ${target_path}" >&2
    exit 1
  fi
  ln -sfnT "${source_path}" "${target_path}"
}

link_webshop_asset "${WEBSHOP_DATA_DIR}" "${WEBSHOP_SOURCE_DIR}/data"
link_webshop_asset "${WEBSHOP_INDEX_DIR}" "${WEBSHOP_SOURCE_DIR}/search_engine/indexes"

"${PYTHON_BIN}" - "${SKILLS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

skills_dir = Path(sys.argv[1])
with (skills_dir / "skill_mapping.json").open(encoding="utf-8") as file:
    mapping = json.load(file)
missing = [
    filename
    for filename in mapping.get("skill_files", {}).values()
    if not (skills_dir / filename).is_file()
]
if missing:
    raise SystemExit(f"Missing skill files under {skills_dir}: {missing}")
print(f"GRSD skills OK: {skills_dir} ({len(mapping.get('skill_files', {}))} skill files)")
PY

"${PYTHON_BIN}" - "${TRAIN_FILE}" "${VAL_FILE}" "${TRAIN_BATCH_SIZE}" <<'PY'
import sys
import pandas as pd

train_file, val_file = sys.argv[1], sys.argv[2]
train_batch_size = int(sys.argv[3])
train_rows = len(pd.read_parquet(train_file))
val_rows = len(pd.read_parquet(val_file))
print(f"dataset train_rows={train_rows} val_rows={val_rows} train_batch_size={train_batch_size}")
if train_rows < train_batch_size:
    raise SystemExit(
        f"Train parquet has {train_rows} rows, but TRAIN_BATCH_SIZE={train_batch_size}; "
        "drop_last=True would produce an empty dataloader."
    )
if train_rows % train_batch_size:
    print(f"warning: {train_rows} train rows is not divisible by batch size {train_batch_size}")
if val_rows < 1:
    raise SystemExit("Validation parquet is empty.")
PY

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

if sys.version_info > (3, 10, 99):
    raise SystemExit(f"WebShop requires Python <= 3.10; found {sys.version.split()[0]}")
required = [
    "bs4", "cleantext", "flask", "gym", "pyserini", "rank_bm25",
    "ray", "selenium", "spacy", "thefuzz", "torch", "vllm",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name} ({type(exc).__name__}: {exc})")
if missing:
    raise SystemExit("Missing or broken WebShop dependencies:\n  - " + "\n  - ".join(missing))

import spacy
try:
    spacy.load("en_core_web_sm")
except OSError as exc:
    raise SystemExit("Missing spaCy model en_core_web_sm") from exc

import torch
print(
    f"python={sys.version.split()[0]} torch={torch.__version__} "
    f"cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}"
)
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("CUDA is not visible. Run this script inside a GPU job/container.")
PY

if [[ "${GRSD_VARIANT}" == "original" || "${JUDGE_ENABLED,,}" == "true" ]]; then
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
fi

echo "WebShop assets OK (${PRODUCT_FILE}, ${WEBSHOP_INDEX_DIR})"
echo "Java OK: ${java_version_output}"

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
  data.truncation=error \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ROLLOUT_MAX_MODEL_LEN}" \
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
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_MODEL_LEN}" \
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
  env.env_name=Webshop \
  env.seed=0 \
  env.max_steps="${MAX_ENV_STEPS}" \
  env.history_length="${HISTORY_LENGTH}" \
  env.rollout.n="${GROUP_SIZE}" \
  env.resources_per_worker.num_cpus="${ENV_CPUS}" \
  env.webshop.use_small="${WEBSHOP_USE_SMALL}" \
  env.webshop.human_goals="${WEBSHOP_HUMAN_GOALS}" \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=verl_agent_webshop \
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
