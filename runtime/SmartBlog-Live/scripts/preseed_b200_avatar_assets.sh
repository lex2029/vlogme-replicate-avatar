#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-verify}"
ASSET_ROOT="${WORKER_ASSET_ROOT:-/root/smartblog-assets}"
HF_HOME_DIR="${HF_HOME:-$ASSET_ROOT/hf}"

log() {
  printf '[avatar-preseed] %s\n' "$*"
}

fail() {
  printf '[avatar-preseed] ERROR: %s\n' "$*" >&2
  exit 1
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

required_asset_files=(
  "ckpt/Wan2.2-S2V-14B/config.json"
  "ckpt/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
  "ckpt/Wan2.2-S2V-14B/Wan2.1_VAE.pth"
  "ckpt/Wan2.2-S2V-14B-merged-liveavatar-prefp8-test/config.json"
  "ckpt/LiveAvatar/liveavatar.safetensors"
  "worker_assets/enchenh2d/models/GFPGANv1.4.pth"
  "gfpgan/weights/detection_Resnet50_Final.pth"
  "gfpgan/weights/parsing_parsenet.pth"
  "RIFE-safetensors/flownet.safetensors"
  "RIFE-safetensors/interpolation_model.py"
)

required_asset_min_bytes() {
  case "$1" in
    "ckpt/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth") printf '%s\n' 1000000000 ;;
    "ckpt/Wan2.2-S2V-14B/Wan2.1_VAE.pth") printf '%s\n' 100000000 ;;
    "ckpt/LiveAvatar/liveavatar.safetensors") printf '%s\n' 1000000 ;;
    "worker_assets/enchenh2d/models/GFPGANv1.4.pth") printf '%s\n' 1000000 ;;
    "gfpgan/weights/detection_Resnet50_Final.pth") printf '%s\n' 1000000 ;;
    "gfpgan/weights/parsing_parsenet.pth") printf '%s\n' 1000000 ;;
    "RIFE-safetensors/flownet.safetensors") printf '%s\n' 10000000 ;;
    "RIFE-safetensors/interpolation_model.py") printf '%s\n' 1000 ;;
    *) printf '%s\n' 1 ;;
  esac
}

python_bin() {
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/python"
  else
    command -v python3
  fi
}

verify_asset_files() {
  local missing=()
  local rel
  local min_bytes
  local size
  for rel in "${required_asset_files[@]}"; do
    min_bytes="$(required_asset_min_bytes "$rel")"
    if [[ ! -f "$ASSET_ROOT/$rel" ]]; then
      missing+=("$rel (missing)")
      continue
    fi
    size="$(wc -c < "$ASSET_ROOT/$rel" | tr -d '[:space:]')"
    if (( size < min_bytes )); then
      missing+=("$rel (${size} bytes < ${min_bytes})")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    printf '[avatar-preseed] missing required avatar assets under %s:\n' "$ASSET_ROOT" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    return 1
  fi
  log "required avatar assets present under $ASSET_ROOT"
}

verify_wav2vec_cache() {
  export HF_HOME="$HF_HOME_DIR"
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-}}}}"
  "$(python_bin)" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/wav2vec2-base-960h",
    cache_dir=None,
    local_files_only=True,
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
    ],
)
PY
  log "Wav2Vec cache present under HF_HOME=$HF_HOME"
}

verify_all() {
  verify_asset_files && verify_wav2vec_cache
}

preseed_worker_assets() {
  mkdir -p "$ASSET_ROOT" "$HF_HOME_DIR"
  export WORKER_ASSET_ROOT="$ASSET_ROOT"
  export HF_HOME="$HF_HOME_DIR"
  export HF_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}}"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  log "preseeding LiveAvatar worker assets into $ASSET_ROOT"
  PATH="$ROOT_DIR/.venv/bin:$PATH" bash "$ROOT_DIR/scripts/download_worker_weights.sh"
}

preseed_wav2vec_cache() {
  mkdir -p "$HF_HOME_DIR"
  export HF_HOME="$HF_HOME_DIR"
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-}}}}"
  log "preseeding Wav2Vec cache into HF_HOME=$HF_HOME"
  "$(python_bin)" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/wav2vec2-base-960h",
    cache_dir=None,
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
    ],
)
PY
}

preseed_all() {
  preseed_worker_assets
  preseed_wav2vec_cache
  verify_all
}

case "$MODE" in
  verify)
    verify_all
    ;;
  preseed)
    preseed_all
    ;;
  verify-or-preseed)
    if verify_all; then
      exit 0
    fi
    log "verification failed; running preseed"
    preseed_all
    ;;
  *)
    fail "unsupported mode: $MODE (expected verify, preseed, verify-or-preseed)"
    ;;
esac
