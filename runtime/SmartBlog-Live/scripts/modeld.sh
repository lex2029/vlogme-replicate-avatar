#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime_env.sh"
LOCKED_CONF="${WORKER_LOCKED_CONF:-$ROOT_DIR/config/worker_locked.conf}"
RUNTIME_CONF="${WORKER_RUNTIME_CONF:-$ROOT_DIR/config/worker_runtime.conf}"
PROFILE_CONF="${WORKER_PROFILE_CONF:-$ROOT_DIR/config/worker_profile.local.conf}"
if [[ ! -f "$PROFILE_CONF" ]]; then
  PROFILE_CONF="$ROOT_DIR/config/worker_profile.render_allinone.conf"
fi
SECRETS_ENV="$ROOT_DIR/config/worker_secrets.conf"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-$ROOT_DIR/logs/torchrun}"
TORCHRUN_RUN_ID="${TORCHRUN_RUN_ID:-smartblog-live-modeld}"
SINGLETON_LOCK_FILE="${WORKER_MODEL_SINGLETON_LOCK_FILE:-/run/lock/smartblog-live-modeld.lock}"
GPU_LOCK_DIR="${WORKER_GPU_LOCK_DIR:-/run/lock/smartblog-live-gpus}"

cd "$ROOT_DIR"

mkdir -p "$TORCHRUN_LOG_DIR" "$(dirname "$SINGLETON_LOCK_FILE")" "$GPU_LOCK_DIR"

exec 200>"$SINGLETON_LOCK_FILE"
if ! flock -n 200; then
  echo "Another model runtime instance is already running (lock: $SINGLETON_LOCK_FILE)."
  exit 1
fi
echo "$$" 1>&200

_acquire_gpu_locks() {
  local visible="${1:-}"
  local gpu lock_file fd
  local -A seen=()
  [[ -n "$visible" ]] || { echo "CUDA_VISIBLE_DEVICES is empty"; exit 1; }
  IFS=',' read -r -a _gpu_items <<<"$visible"
  for gpu in "${_gpu_items[@]}"; do
    gpu="$(echo "$gpu" | xargs)"
    [[ -z "$gpu" ]] && continue
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Unsupported CUDA_VISIBLE_DEVICES entry '$gpu'"; exit 1; }
    [[ -n "${seen[$gpu]:-}" ]] && continue
    seen["$gpu"]=1
    lock_file="$GPU_LOCK_DIR/gpu-${gpu}.lock"
    exec {fd}>"$lock_file"
    if ! flock -n "$fd"; then
      echo "GPU lock is already held for gpu=${gpu} (lock: $lock_file)."
      exit 1
    fi
  done
}

load_worker_env "$LOCKED_CONF" "$RUNTIME_CONF" "$PROFILE_CONF" "$SECRETS_ENV"

required_runtime=(
  CUDA_VISIBLE_DEVICES
  USE_FP8
  TORCHRUN_NPROC
  NUM_GPUS_DIT
  ULYSSES_SIZE
  MASTER_PORT
  WORKER_TASK
  SIZE
  BASE_SEED
  TRAINING_CONFIG
  WORKER_FPS
  WORKER_AUDIO_SAMPLE_RATE
  INFER_FRAMES
  WORKER_SAMPLE_STEPS
  WORKER_LIVEAUDIO_MICRO_CHUNK_SCHEDULE_SAMPLES
  GUIDE_SCALE
  SAMPLE_SOLVER
  CKPT_DIR
  SAVE_DIR
  SERVER_PORT
  SERVER_NAME
)
for key in "${required_runtime[@]}"; do
  require_var "$key"
done

USE_MERGED_CKPT="${USE_MERGED_CKPT:-0}"
MERGED_CKPT_DIR="${MERGED_CKPT_DIR:-}"
MERGED_NOISE_MODEL_DIR="${MERGED_NOISE_MODEL_DIR:-$MERGED_CKPT_DIR}"
USE_MERGED_CKPT_EFFECTIVE=0
if is_true "$USE_MERGED_CKPT" || [[ -n "$MERGED_CKPT_DIR" ]]; then
  USE_MERGED_CKPT_EFFECTIVE=1
  if [[ -z "$MERGED_NOISE_MODEL_DIR" ]]; then
    echo "Missing required parameter: MERGED_NOISE_MODEL_DIR"
    exit 1
  fi
else
  require_var "LORA_PATH_DMD"
fi

_acquire_gpu_locks "$CUDA_VISIBLE_DEVICES"

FP8_FLAG=""
if [[ "${USE_FP8}" == "true" || "${USE_FP8}" == "1" ]]; then
  FP8_FLAG="--fp8"
fi

PYTHON_BIN="${WORKER_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  SHARED_PYTHON_BIN="${WORKER_SHARED_PYTHON_BIN:-}"
  if [[ -n "$SHARED_PYTHON_BIN" && -x "$SHARED_PYTHON_BIN" ]]; then
    PYTHON_BIN="$SHARED_PYTHON_BIN"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

TORCHRUN_CMD=()
if [[ -n "${TORCHRUN_BIN:-}" ]]; then
  TORCHRUN_CMD=("$TORCHRUN_BIN")
elif [[ -x "$PYTHON_BIN" ]]; then
  TORCHRUN_CMD=("$PYTHON_BIN" -m torch.distributed.run)
elif [[ -n "${WORKER_SHARED_TORCHRUN_BIN:-}" && -x "${WORKER_SHARED_TORCHRUN_BIN:-}" ]]; then
  TORCHRUN_CMD=("${WORKER_SHARED_TORCHRUN_BIN}")
else
  TORCHRUN_CMD=("$(command -v torchrun)")
fi

boot_log "Starting SmartBlog Live model runtime profile=${WORKER_PROFILE_NAME:-prod} model_log=${MODEL_LOG_LEVEL:-INFO} timing=${MODEL_TIMING_LOG:-0}"
boot_log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
boot_log "TORCHRUN_RUN_ID=$TORCHRUN_RUN_ID"
boot_log "NUM_GPUS_DIT=${NUM_GPUS_DIT} TORCHRUN_NPROC=${TORCHRUN_NPROC}"
boot_log "TORCHRUN_CMD=${TORCHRUN_CMD[*]}"
main_args=(
  --nproc_per_node="$TORCHRUN_NPROC"
  --master_port="$MASTER_PORT"
  --rdzv-id "$TORCHRUN_RUN_ID"
  --log-dir "$TORCHRUN_LOG_DIR"
  --tee 3
  --module avalife.model.main
  --ulysses_size "$ULYSSES_SIZE"
  --task "$WORKER_TASK"
  --size "$SIZE"
  --base_seed "$BASE_SEED"
  --training_config "$TRAINING_CONFIG"
  --offload_model False
  --convert_model_dtype
  --infer_frames "$INFER_FRAMES"
  --sample_steps "$WORKER_SAMPLE_STEPS"
  --sample_guide_scale "$GUIDE_SCALE"
  --num_gpus_dit "$NUM_GPUS_DIT"
  --sample_solver "$SAMPLE_SOLVER"
  --ckpt_dir "$CKPT_DIR"
  --save_dir "$SAVE_DIR"
  --server_port "$SERVER_PORT"
  --server_name "$SERVER_NAME"
)
if [[ -n "${SAMPLE_SHIFT:-}" ]]; then
  main_args+=(--sample_shift "$SAMPLE_SHIFT")
fi
if is_true "${START_FROM_REF:-0}"; then
  main_args+=(--start_from_ref)
fi
if is_true "${DROP_MOTION_NOISY:-0}"; then
  main_args+=(--drop_motion_noisy)
fi
if is_true "${ENABLE_T5_FSDP:-0}"; then
  main_args+=(--t5_fsdp)
fi
if is_true "${ENABLE_DIT_FSDP:-0}"; then
  main_args+=(--dit_fsdp)
fi
if is_true "${ENABLE_VAE_PARALLEL:-0}"; then
  main_args+=(--enable_vae_parallel)
fi
if is_true "${OFFLOAD_KV_CACHE:-0}"; then
  main_args+=(--offload_kv_cache)
fi
if [[ "$USE_MERGED_CKPT_EFFECTIVE" == "1" ]]; then
  main_args+=(--using_merged_ckpt --merged_noise_model_dir "$MERGED_NOISE_MODEL_DIR")
else
  main_args+=(--load_lora --lora_path_dmd "$LORA_PATH_DMD")
fi
if [[ -n "$FP8_FLAG" ]]; then
  main_args+=("$FP8_FLAG")
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "${TORCHRUN_CMD[@]}" "${main_args[@]}"
