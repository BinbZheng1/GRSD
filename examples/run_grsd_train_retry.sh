# Wrapper that launches GRSD ALFWorld training and automatically retries
# ONLY when the rank-0 worker hits a torch.distributed EADDRINUSE port race
# during init (a known hazard on this shared multi-node MPI host) BEFORE any
# training step has run. Any other failure, or a crash after training started,
# is NOT retried (so real bugs surface immediately).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_DIR}"

MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}
LOG=${LOG:-logs/grsd_train_$(date +%Y%m%d_%H%M%S).log}

# Use all 8 GPUs by default. To restrict, set both CUDA_VISIBLE_DEVICES and
# a matching N_GPUS via env (e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 bash ...).
export N_GPUS=${N_GPUS:-8}
export GRSD_VARIANT=${GRSD_VARIANT:-reflect}
export GRSD_MODULATION_LEVEL=${GRSD_MODULATION_LEVEL:-turn}

# Force the intended experiment name regardless of any inherited EXPERIMENT_NAME.
unset EXPERIMENT_NAME
export EXPERIMENT_NAME="${GRSD_EXPERIMENT_NAME:-GRSD-Reflect-Qwen3-1.7B-ALFWorld-Turn}"

echo "[retry] LOG=${LOG}"
echo "[retry] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} N_GPUS=${N_GPUS}"
echo "[retry] GRSD_VARIANT=${GRSD_VARIANT} GRSD_MODULATION_LEVEL=${GRSD_MODULATION_LEVEL}"
echo "[retry] EXPERIMENT_NAME=${EXPERIMENT_NAME} MAX_ATTEMPTS=${MAX_ATTEMPTS}"

mkdir -p logs

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  echo "=========== GRSD TRAIN ATTEMPT ${attempt}/${MAX_ATTEMPTS} ===========" | tee -a "${LOG}"
  # Clean any stragglers from a previous failed attempt before retrying.
  if [[ "${attempt}" -gt 1 ]]; then
    pkill -9 -f "verl.trainer.main_grsd" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    sleep 8
  fi

  bash examples/run_grsd_alfworld_qwen3_local.sh >>"${LOG}" 2>&1
  rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    echo "[retry] training finished cleanly (rc=0)." | tee -a "${LOG}"
    exit 0
  fi

  # Did we get past worker init into actual training? If a step ran, do NOT retry.
  if grep -aqE "Training: *[0-9]+%|step:[0-9]+ - " "${LOG}"; then
    echo "[retry] failure occurred AFTER training started (rc=${rc}); not retrying." | tee -a "${LOG}"
    exit "${rc}"
  fi

  # Only retry the known port-race during init.
  if grep -aq "EADDRINUSE" "${LOG}"; then
    echo "[retry] EADDRINUSE during init on attempt ${attempt} (rc=${rc}); retrying..." | tee -a "${LOG}"
    continue
  fi

  echo "[retry] non-retriable failure on attempt ${attempt} (rc=${rc}); aborting." | tee -a "${LOG}"
  exit "${rc}"
done

echo "[retry] exhausted ${MAX_ATTEMPTS} attempts; giving up." | tee -a "${LOG}"
exit 1
