#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${APP_DIR:-$ROOT_DIR}"
ASSET_ROOT="${WORKER_ASSET_ROOT:-$APP_DIR}"
ENV_FILE="$APP_DIR/config/worker_secrets.conf"
BOOTSTRAP_VENV_DIR="${HF_BOOTSTRAP_VENV_DIR:-$APP_DIR/.bootstrap-venv}"
PYTHON_BIN_NAME="${PYTHON_BIN_NAME:-python3.10}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

log() {
  printf '[weights] %s\n' "$*"
}

fail() {
  printf '[weights] ERROR: %s\n' "$*" >&2
  exit 1
}

source_env() {
  [[ -f "$ENV_FILE" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

ensure_hf_cli() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI_BIN="$(command -v huggingface-cli)"
    return 0
  fi
  if [[ ! -x "$BOOTSTRAP_VENV_DIR/bin/huggingface-cli" ]]; then
    local pybin
    pybin="$(command -v "$PYTHON_BIN_NAME" 2>/dev/null || true)"
    if [[ -z "$pybin" ]]; then
      pybin="$(command -v python3 2>/dev/null || true)"
    fi
    [[ -n "$pybin" ]] || fail "python3.10/python3 not found; cannot bootstrap huggingface-cli"
    log "bootstrapping huggingface-cli in $BOOTSTRAP_VENV_DIR"
    "$pybin" -m venv "$BOOTSTRAP_VENV_DIR"
    "$BOOTSTRAP_VENV_DIR/bin/pip" install --upgrade pip "huggingface_hub[cli]==0.36.2" hf_transfer
  fi
  HF_CLI_BIN="$BOOTSTRAP_VENV_DIR/bin/huggingface-cli"
}

ensure_layout() {
  mkdir -p \
    "$ASSET_ROOT/ckpt" \
    "$ASSET_ROOT/worker_assets/enchenh2d/models" \
    "$ASSET_ROOT/gfpgan/weights"
}

download_tree() {
  local label="$1"
  local repo_id="$2"
  local include_glob="$3"
  local remote_root="$4"
  local local_root="$5"
  local required_rel="$6"
  local revision="${7:-}"

  [[ -n "$repo_id" ]] || fail "missing repo id for $label"
  [[ -n "$include_glob" ]] || fail "missing include glob for $label"
  [[ -n "$remote_root" ]] || fail "missing remote root for $label"
  [[ -n "$local_root" ]] || fail "missing local root for $label"
  [[ -n "$required_rel" ]] || fail "missing required path for $label"

  if [[ "$FORCE_DOWNLOAD" != "1" && -f "$ASSET_ROOT/$required_rel" ]]; then
    log "skip $label; already present: $required_rel"
    return 0
  fi

  local cmd=("$HF_CLI_BIN" download "$repo_id" --local-dir "$ASSET_ROOT" --include "$include_glob")
  if [[ -n "$revision" ]]; then
    cmd+=(--revision "$revision")
  fi

  log "downloading $label from $repo_id include=$include_glob"
  "${cmd[@]}"
  if [[ -d "$ASSET_ROOT/$remote_root" && "$remote_root" != "$local_root" ]]; then
    mkdir -p "$(dirname "$ASSET_ROOT/$local_root")"
    rm -rf "$ASSET_ROOT/$local_root"
    mv "$ASSET_ROOT/$remote_root" "$ASSET_ROOT/$local_root"
  fi
  [[ -f "$ASSET_ROOT/$required_rel" ]] || fail "$label download finished but required file missing: $required_rel"
}

download_tree_if_missing_any() {
  local label="$1"
  local repo_id="$2"
  local include_glob="$3"
  local remote_root="$4"
  local local_root="$5"
  local revision="$6"
  shift 6

  local rel
  if [[ "$FORCE_DOWNLOAD" != "1" ]]; then
    local missing=0
    for rel in "$@"; do
      if [[ ! -s "$ASSET_ROOT/$rel" ]]; then
        missing=1
        break
      fi
    done
    if [[ "$missing" == "0" ]]; then
      log "skip $label; required files already present"
      return 0
    fi
  fi

  FORCE_DOWNLOAD=1 download_tree "$label" "$repo_id" "$include_glob" "$remote_root" "$local_root" "$1" "$revision"
  for rel in "$@"; do
    [[ -s "$ASSET_ROOT/$rel" ]] || fail "$label download finished but required file missing: $rel"
  done
}

download_enhancers() {
  local repo_id="$1"
  local include_glob="$2"
  local revision="${3:-}"

  if [[ "$FORCE_DOWNLOAD" != "1" \
    && -f "$ASSET_ROOT/worker_assets/enchenh2d/models/GFPGANv1.4.pth" \
    && -f "$ASSET_ROOT/gfpgan/weights/detection_Resnet50_Final.pth" \
    && -f "$ASSET_ROOT/gfpgan/weights/parsing_parsenet.pth" ]]; then
    log "skip enhancers; already present"
    return 0
  fi

  local cmd=("$HF_CLI_BIN" download "$repo_id" --local-dir "$ASSET_ROOT" --include "$include_glob")
  if [[ -n "$revision" ]]; then
    cmd+=(--revision "$revision")
  fi

  log "downloading enhancers from $repo_id include=$include_glob"
  "${cmd[@]}"

  local src="$ASSET_ROOT/enhancers"
  [[ -d "$src" ]] || fail "enhancer download finished but enhancers/ folder missing"
  install -D -m 0644 "$src/GFPGANv1.4.pth" "$ASSET_ROOT/worker_assets/enchenh2d/models/GFPGANv1.4.pth"
  if [[ -f "$src/RealESRGAN_x2plus.pth" ]]; then
    install -D -m 0644 "$src/RealESRGAN_x2plus.pth" "$ASSET_ROOT/worker_assets/enchenh2d/models/RealESRGAN_x2plus.pth"
  fi
  if [[ -f "$src/codeformer.pth" ]]; then
    install -D -m 0644 "$src/codeformer.pth" "$ASSET_ROOT/worker_assets/enchenh2d/models/codeformer.pth"
  fi
  install -D -m 0644 "$src/detection_Resnet50_Final.pth" "$ASSET_ROOT/gfpgan/weights/detection_Resnet50_Final.pth"
  install -D -m 0644 "$src/parsing_parsenet.pth" "$ASSET_ROOT/gfpgan/weights/parsing_parsenet.pth"
  rm -rf "$src"
}

verify_required_files() {
  local required_files=(
    "ckpt/Wan2.2-S2V-14B/config.json"
    "ckpt/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
    "ckpt/Wan2.2-S2V-14B/Wan2.1_VAE.pth"
    "ckpt/Wan2.2-S2V-14B-merged-liveavatar-prefp8-test/config.json"
    "ckpt/LiveAvatar/liveavatar.safetensors"
    "worker_assets/enchenh2d/models/GFPGANv1.4.pth"
    "gfpgan/weights/detection_Resnet50_Final.pth"
    "gfpgan/weights/parsing_parsenet.pth"
  )
  local rel
  for rel in "${required_files[@]}"; do
    [[ -f "$ASSET_ROOT/$rel" ]] || fail "required weight missing: $rel"
  done
}

main() {
  source_env
  ensure_hf_cli
  ensure_layout

  export HF_TOKEN="${AVALIFE_HF_TOKEN:-${SMARTBLOG_HF_TOKEN:-}}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

  local worker_repo="${HF_WORKER_REPO_ID:-abalex2029/smartblog}"
  local base_repo="${HF_BASE_MODEL_REPO_ID:-$worker_repo}"
  local merged_repo="${HF_MERGED_MODEL_REPO_ID:-$worker_repo}"
  local lora_repo="${HF_LORA_REPO_ID:-$worker_repo}"
  local enhancer_repo="${HF_ENHANCER_MODELS_REPO_ID:-$worker_repo}"
  local face_repo="${HF_FACE_WEIGHTS_REPO_ID:-$worker_repo}"

  download_tree_if_missing_any \
    "base model" \
    "$base_repo" \
    "${HF_BASE_MODEL_INCLUDE:-Wan2.2-S2V-14B/*}" \
    "Wan2.2-S2V-14B" \
    "ckpt/Wan2.2-S2V-14B" \
    "${HF_BASE_MODEL_REVISION:-}" \
    "ckpt/Wan2.2-S2V-14B/config.json" \
    "ckpt/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth" \
    "ckpt/Wan2.2-S2V-14B/Wan2.1_VAE.pth"

  download_tree \
    "merged model" \
    "$merged_repo" \
    "${HF_MERGED_MODEL_INCLUDE:-Wan2.2-S2V-14B-merged-liveavatar-prefp8-test/*}" \
    "Wan2.2-S2V-14B-merged-liveavatar-prefp8-test" \
    "ckpt/Wan2.2-S2V-14B-merged-liveavatar-prefp8-test" \
    "ckpt/Wan2.2-S2V-14B-merged-liveavatar-prefp8-test/config.json" \
    "${HF_MERGED_MODEL_REVISION:-}"

  download_tree \
    "lora" \
    "$lora_repo" \
    "${HF_LORA_INCLUDE:-LiveAvatar/*}" \
    "LiveAvatar" \
    "ckpt/LiveAvatar" \
    "ckpt/LiveAvatar/liveavatar.safetensors" \
    "${HF_LORA_REVISION:-}"

  download_enhancers \
    "$enhancer_repo" \
    "${HF_ENHANCER_MODELS_INCLUDE:-enhancers/*}" \
    "${HF_ENHANCER_MODELS_REVISION:-}"
  if [[ "$face_repo" != "$enhancer_repo" || "${HF_FACE_WEIGHTS_INCLUDE:-enhancers/*}" != "${HF_ENHANCER_MODELS_INCLUDE:-enhancers/*}" ]]; then
    download_enhancers \
      "$face_repo" \
      "${HF_FACE_WEIGHTS_INCLUDE:-enhancers/*}" \
      "${HF_FACE_WEIGHTS_REVISION:-}"
  fi

  verify_required_files
  log "all required worker weights are present under $ASSET_ROOT"
}

main "$@"
