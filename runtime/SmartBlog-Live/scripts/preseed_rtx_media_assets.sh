#!/usr/bin/env bash
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

load_worker_env "$LOCKED_CONF" "$RUNTIME_CONF" "$PROFILE_CONF" "$SECRETS_ENV"

MODE="${1:-preseed}"
ASSET_ROOT="${WORKER_ASSET_ROOT:-/workspace/smartblog-assets}"
HUNYUAN_DIR="${SMARTBLOG_HUNYUAN_TENCENT_LOCAL_DIR:-$ASSET_ROOT/hunyuan/tencent/HunyuanVideo-1.5}"
HUNYUAN_REPO="${SMARTBLOG_HUNYUAN_TENCENT_MODEL_PATH:-${SMARTBLOG_HUNYUAN_MODEL_ID:-tencent/HunyuanVideo-1.5}}"
HUNYUAN_ALLOW="${SMARTBLOG_HUNYUAN_TENCENT_ALLOW_PATTERNS:-transformer/480p_i2v_step_distilled/*,transformer/480p_t2v_distilled/*,vae/*,scheduler/*,text_encoder/*,vision_encoder/*}"
GLYPH_REPO="${SMARTBLOG_HUNYUAN_TENCENT_GLYPH_MODEL_ID:-AI-ModelScope/Glyph-SDXL-v2}"
LLM_REPO="${SMARTBLOG_HUNYUAN_TENCENT_LLM_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
BYT5_REPO="${SMARTBLOG_HUNYUAN_TENCENT_BYT5_MODEL_ID:-google/byt5-small}"
VISION_REPO="${SMARTBLOG_HUNYUAN_TENCENT_VISION_MODEL_ID:-black-forest-labs/FLUX.1-Redux-dev}"
LLM_ALLOW="${SMARTBLOG_HUNYUAN_TENCENT_LLM_ALLOW_PATTERNS:-*.json,*.safetensors,*.jinja,*.txt,tokenizer*,merges.txt,vocab.json,*.model}"
BYT5_ALLOW="${SMARTBLOG_HUNYUAN_TENCENT_BYT5_ALLOW_PATTERNS:-config.json,pytorch_model.bin,special_tokens_map.json,tokenizer_config.json,spiece.model}"
VISION_ALLOW="${SMARTBLOG_HUNYUAN_TENCENT_VISION_ALLOW_PATTERNS:-image_encoder/config.json,image_encoder/model.safetensors,image_encoder/*.json}"
HF_CLI_BIN="${SMARTBLOG_HF_CLI_BIN:-}"
if [[ -z "$HF_CLI_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/huggingface-cli" ]]; then
    HF_CLI_BIN="$ROOT_DIR/.venv/bin/huggingface-cli"
  elif [[ -x /opt/hunyuan/venv/bin/huggingface-cli ]]; then
    HF_CLI_BIN="/opt/hunyuan/venv/bin/huggingface-cli"
  else
    HF_CLI_BIN="huggingface-cli"
  fi
fi
MODELSCOPE_BIN="${SMARTBLOG_MODELSCOPE_BIN:-}"
if [[ -z "$MODELSCOPE_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/modelscope" ]]; then
    MODELSCOPE_BIN="$ROOT_DIR/.venv/bin/modelscope"
  elif [[ -x /opt/hunyuan/venv/bin/modelscope ]]; then
    MODELSCOPE_BIN="/opt/hunyuan/venv/bin/modelscope"
  else
    MODELSCOPE_BIN="modelscope"
  fi
fi

export HF_HOME="${SMARTBLOG_HUNYUAN_HF_HOME:-$ASSET_ROOT/hunyuan/hf}"
export HF_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [rtx-assets] $*"; }

has_glob() {
  local pattern="$1"
  compgen -G "$pattern" >/dev/null
}

need_file() {
  local path="$1"
  [[ -f "$path" ]] || { log "missing file: $path"; return 1; }
}

need_glob() {
  local pattern="$1"
  has_glob "$pattern" || { log "missing glob: $pattern"; return 1; }
}

split_patterns() {
  local raw="$1"
  python3 - "$raw" <<'PY'
import sys
raw = sys.argv[1]
for item in raw.replace(";", ",").split(","):
    item = item.strip()
    if item:
        print(item)
PY
}

hf_download() {
  local repo="$1"
  local dir="$2"
  local patterns="$3"
  shift 3 || true
  mkdir -p "$dir"
  local args=(download "$repo" --local-dir "$dir")
  while IFS= read -r pattern; do
    [[ -n "$pattern" ]] && args+=(--include "$pattern")
  done < <(split_patterns "$patterns")
  log "download/resolve hf repo=$repo local_dir=$dir include=${patterns:-all}"
  "$HF_CLI_BIN" "${args[@]}"
}

modelscope_download_glyph() {
  local dir="$HUNYUAN_DIR/text_encoder/Glyph-SDXL-v2"
  mkdir -p "$dir"
  log "download/resolve modelscope glyph repo=$GLYPH_REPO local_dir=$dir"
  "$MODELSCOPE_BIN" download \
    --model "$GLYPH_REPO" \
    --local_dir "$dir" \
    --include \
    assets/color_idx.json \
    assets/multilingual_10-lang_idx.json \
    checkpoints/byt5_model.pt \
    --max-workers 4
}

preseed_mmaudio() {
  local enabled="${SMARTBLOG_MMAUDIO_SERVICE_ENABLED:-0}"
  if ! is_true "$enabled"; then
    log "MMAudio disabled; skip preseed"
    return 0
  fi
  local root="${SMARTBLOG_MMAUDIO_ROOT:-/opt/MMAudio}"
  local python="${SMARTBLOG_MMAUDIO_PYTHON:-$root/venv/bin/python}"
  local variant="${SMARTBLOG_MMAUDIO_VARIANT:-large_44k_v2}"
  if [[ ! -x "$python" || ! -d "$root" ]]; then
    log "MMAudio runtime missing; skip preseed python=$python root=$root"
    return 0
  fi
  local mmaudio_asset_root="${SMARTBLOG_MMAUDIO_ASSET_ROOT:-$ASSET_ROOT/mmaudio}"
  local weights_dir="${SMARTBLOG_MMAUDIO_WEIGHTS_DIR:-$mmaudio_asset_root/weights}"
  local ext_weights_dir="${SMARTBLOG_MMAUDIO_EXT_WEIGHTS_DIR:-$mmaudio_asset_root/ext_weights}"
  mkdir -p "$weights_dir" "$ext_weights_dir"
  for name in weights ext_weights; do
    local src="$root/$name"
    local dst="$weights_dir"
    [[ "$name" == "ext_weights" ]] && dst="$ext_weights_dir"
    if [[ ! -L "$src" ]]; then
      if [[ -d "$src" ]]; then
        shopt -s nullglob dotglob
        for item in "$src"/*; do
          local base
          base="$(basename "$item")"
          [[ -e "$dst/$base" ]] || mv "$item" "$dst/"
        done
        shopt -u nullglob dotglob
        rm -rf "$src"
      fi
      ln -s "$dst" "$src"
    fi
  done
  log "download/resolve MMAudio variant=$variant"
  (
    cd "$root"
    SMARTBLOG_MMAUDIO_VARIANT="$variant" \
      HF_HOME="${SMARTBLOG_MMAUDIO_HF_HOME:-$mmaudio_asset_root/hf}" \
      PYTHONPATH="$ROOT_DIR:$root:${PYTHONPATH:-}" \
      "$python" - <<'PY'
import os
from mmaudio.eval_utils import all_model_cfg
variant = os.environ.get("SMARTBLOG_MMAUDIO_VARIANT", "large_44k_v2")
cfg = all_model_cfg[variant]
cfg.download_if_needed()
print(f"MMAudio assets ready: {variant}")
PY
  )
}

preseed() {
  mkdir -p "$ASSET_ROOT" "$HF_HOME"
  hf_download "$HUNYUAN_REPO" "$HUNYUAN_DIR" "$HUNYUAN_ALLOW"
  modelscope_download_glyph
  hf_download "$LLM_REPO" "$HUNYUAN_DIR/text_encoder/llm" "$LLM_ALLOW"
  hf_download "$BYT5_REPO" "$HUNYUAN_DIR/text_encoder/byt5-small" "$BYT5_ALLOW"
  hf_download "$VISION_REPO" "$HUNYUAN_DIR/vision_encoder/siglip" "$VISION_ALLOW"
  preseed_mmaudio
}

verify() {
  local ok=1
  need_glob "$HUNYUAN_DIR/transformer/480p_i2v_step_distilled/*" || ok=0
  need_glob "$HUNYUAN_DIR/vae/*" || ok=0
  need_glob "$HUNYUAN_DIR/scheduler/*" || ok=0
  need_file "$HUNYUAN_DIR/text_encoder/Glyph-SDXL-v2/assets/color_idx.json" || ok=0
  need_file "$HUNYUAN_DIR/text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json" || ok=0
  need_file "$HUNYUAN_DIR/text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt" || ok=0
  need_file "$HUNYUAN_DIR/text_encoder/llm/config.json" || ok=0
  need_glob "$HUNYUAN_DIR/text_encoder/llm/*.safetensors" || ok=0
  need_file "$HUNYUAN_DIR/text_encoder/byt5-small/config.json" || ok=0
  if [[ ! -f "$HUNYUAN_DIR/text_encoder/byt5-small/pytorch_model.bin" ]] && ! has_glob "$HUNYUAN_DIR/text_encoder/byt5-small/*.safetensors"; then
    log "missing ByT5 weights: $HUNYUAN_DIR/text_encoder/byt5-small/pytorch_model.bin or *.safetensors"
    ok=0
  fi
  need_file "$HUNYUAN_DIR/vision_encoder/siglip/image_encoder/config.json" || ok=0
  need_file "$HUNYUAN_DIR/vision_encoder/siglip/image_encoder/model.safetensors" || ok=0
  [[ "$ok" == "1" ]] || return 1
  log "required RTX media assets are present in $ASSET_ROOT"
}

case "$MODE" in
  preseed)
    preseed
    verify
    ;;
  verify)
    verify
    ;;
  verify-or-preseed)
    verify || { log "asset verification failed; running preseed"; preseed; verify; }
    ;;
  *)
    echo "Usage: $0 preseed|verify|verify-or-preseed" >&2
    exit 2
    ;;
esac
