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
RUN_STATE_DIR="${WORKER_RUN_STATE_DIR:-$ROOT_DIR/runtime}"
PID_FILE="${SMARTBLOG_MMAUDIO_SERVICE_PID_FILE:-$RUN_STATE_DIR/mmaudio_service.pid}"
LOG_DIR="${SMARTBLOG_MMAUDIO_SERVICE_LOG_DIR:-$ROOT_DIR/logs/mmaudio_service}"
CURRENT_LOG_FILE="${SMARTBLOG_MMAUDIO_SERVICE_CURRENT_LOG_FILE:-$RUN_STATE_DIR/current_mmaudio_service_log.txt}"

load_worker_env "$LOCKED_CONF" "$RUNTIME_CONF" "$PROFILE_CONF" "$SECRETS_ENV"

ENABLED="${SMARTBLOG_MMAUDIO_SERVICE_ENABLED:-0}"
HOST="${SMARTBLOG_MMAUDIO_SERVICE_HOST:-127.0.0.1}"
PORT="${SMARTBLOG_MMAUDIO_SERVICE_PORT:-8799}"
PYTHON="${SMARTBLOG_MMAUDIO_PYTHON:-/opt/MMAudio/venv/bin/python}"
ROOT="${SMARTBLOG_MMAUDIO_ROOT:-/opt/MMAudio}"
VARIANT="${SMARTBLOG_MMAUDIO_VARIANT:-large_44k_v2}"
TIMEOUT_SEC="${SMARTBLOG_MMAUDIO_SERVICE_READY_TIMEOUT_SEC:-600}"
ASSET_ROOT="${WORKER_ASSET_ROOT:-/workspace/smartblog-assets}"
MMAUDIO_ASSET_ROOT="${SMARTBLOG_MMAUDIO_ASSET_ROOT:-$ASSET_ROOT/mmaudio}"
MMAUDIO_WEIGHTS_DIR="${SMARTBLOG_MMAUDIO_WEIGHTS_DIR:-$MMAUDIO_ASSET_ROOT/weights}"
MMAUDIO_EXT_WEIGHTS_DIR="${SMARTBLOG_MMAUDIO_EXT_WEIGHTS_DIR:-$MMAUDIO_ASSET_ROOT/ext_weights}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

prepare_asset_dirs() {
  mkdir -p "$MMAUDIO_WEIGHTS_DIR" "$MMAUDIO_EXT_WEIGHTS_DIR"
  for name in weights ext_weights; do
    local src="$ROOT/$name"
    local dst
    if [[ "$name" == "weights" ]]; then
      dst="$MMAUDIO_WEIGHTS_DIR"
    else
      dst="$MMAUDIO_EXT_WEIGHTS_DIR"
    fi
    if [[ -L "$src" ]]; then
      continue
    fi
    if [[ -d "$src" ]]; then
      shopt -s nullglob dotglob
      for item in "$src"/*; do
        local base
        base="$(basename "$item")"
        if [[ ! -e "$dst/$base" ]]; then
          mv "$item" "$dst/"
        fi
      done
      shopt -u nullglob dotglob
      rm -rf "$src"
    fi
    ln -s "$dst" "$src"
  done
}

md5_ok() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || return 1
  local actual
  actual="$(md5sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]]
}

preload_primary_weight() {
  if ! is_true "${SMARTBLOG_MMAUDIO_PRELOAD_WEIGHTS:-1}"; then
    return 0
  fi
  if [[ "$VARIANT" != "large_44k_v2" ]]; then
    return 0
  fi
  local target="$ROOT/weights/mmaudio_large_44k_v2.pth"
  local md5="01ad4464f049b2d7efdaa4c1a59b8dfe"
  local url="${SMARTBLOG_MMAUDIO_LARGE_44K_V2_URL:-https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k_v2.pth}"
  if md5_ok "$target" "$md5"; then
    log "MMAudio primary weight ready: $target"
    return 0
  fi
  log "preloading MMAudio primary weight with resume: $target"
  for attempt in 1 2 3 4 5; do
    curl -fL --retry 8 --retry-delay 2 --retry-all-errors -C - -o "$target" "$url" && {
      if md5_ok "$target" "$md5"; then
        log "MMAudio primary weight downloaded"
        return 0
      fi
      log "MMAudio primary weight md5 mismatch after attempt=$attempt"
    }
    sleep $(( attempt * 2 ))
  done
  log "MMAudio primary weight preload failed; service download fallback will run"
  return 0
}

pid_alive() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

current_pid() {
  if [[ -f "$PID_FILE" ]]; then
    head -n1 "$PID_FILE" 2>/dev/null | xargs || true
  fi
}

health() {
  "$PYTHON" - "$HOST" "$PORT" <<'PY'
import json, sys, urllib.request
host, port = sys.argv[1], int(sys.argv[2])
try:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data.get("ok") and data.get("ready"):
        print(json.dumps(data, ensure_ascii=False))
        raise SystemExit(0)
except Exception as e:
    print(str(e), file=sys.stderr)
raise SystemExit(1)
PY
}

start() {
  if ! is_true "$ENABLED"; then
    log "MMAudio service disabled"
    return 0
  fi
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    if health >/dev/null 2>&1; then
      log "MMAudio service already ready pid=$pid"
    else
      log "MMAudio service already running but not ready yet pid=$pid"
    fi
    return 0
  fi
  if [[ ! -x "$PYTHON" ]]; then
    log "MMAudio python not executable: $PYTHON"
    return 1
  fi
  if [[ ! -d "$ROOT" ]]; then
    log "MMAudio root not found: $ROOT"
    return 1
  fi
  prepare_asset_dirs
  preload_primary_weight

  mkdir -p "$RUN_STATE_DIR" "$LOG_DIR"
  local log_file="$LOG_DIR/mmaudio_service_$(date '+%Y%m%d_%H%M%S').log"
  printf '%s\n' "$log_file" >"$CURRENT_LOG_FILE"
  log "starting MMAudio service host=$HOST port=$PORT root=$ROOT variant=$VARIANT python=$PYTHON log=$log_file"
  (
    cd "$ROOT"
    export PYTHONPATH="$ROOT_DIR:$ROOT:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${SMARTBLOG_MMAUDIO_CUDA_VISIBLE_DEVICES:-0}"
    export HF_HOME="${SMARTBLOG_MMAUDIO_HF_HOME:-$MMAUDIO_ASSET_ROOT/hf}"
    export HUGGING_FACE_HUB_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
    export HF_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HF_TOKEN:-}}}"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    exec "$PYTHON" -m avalife.mmaudio_service \
      --host "$HOST" \
      --port "$PORT" \
      --variant "$VARIANT"
  ) >"$log_file" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$PID_FILE"
  log "MMAudio service pid=$pid"
}

wait_ready() {
  if ! is_true "$ENABLED"; then
    return 0
  fi
  local deadline=$(( $(date +%s) + TIMEOUT_SEC ))
  while (( $(date +%s) < deadline )); do
    if health >/dev/null 2>&1; then
      log "MMAudio service ready"
      return 0
    fi
    local pid
    pid="$(current_pid)"
    if [[ -n "$pid" ]] && ! pid_alive "$pid"; then
      log "MMAudio service exited before ready pid=$pid"
      tail -n 160 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
      return 1
    fi
    sleep 3
  done
  log "MMAudio service not ready after ${TIMEOUT_SEC}s"
  tail -n 200 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
  return 1
}

stop() {
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    log "stopping MMAudio service pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      pid_alive "$pid" || break
      sleep 1
    done
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

status() {
  local pid
  pid="$(current_pid)"
  log "enabled=$ENABLED pid=${pid:-none} host=$HOST port=$PORT root=$ROOT variant=$VARIANT"
  if pid_alive "$pid"; then
    log "process=alive"
  else
    log "process=absent"
  fi
  health || true
}

case "${1:-status}" in
  start)
    start
    ;;
  start-wait)
    start
    wait_ready
    ;;
  wait)
    wait_ready
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  restart-wait)
    stop
    start
    wait_ready
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 {start|start-wait|wait|stop|restart|restart-wait|status}" >&2
    exit 2
    ;;
esac
