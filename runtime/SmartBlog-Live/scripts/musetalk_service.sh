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
PID_FILE="${SMARTBLOG_MUSETALK_SERVICE_PID_FILE:-$RUN_STATE_DIR/musetalk_service.pid}"
LOG_DIR="${SMARTBLOG_MUSETALK_SERVICE_LOG_DIR:-$ROOT_DIR/logs/musetalk_service}"
CURRENT_LOG_FILE="${SMARTBLOG_MUSETALK_SERVICE_CURRENT_LOG_FILE:-$RUN_STATE_DIR/current_musetalk_service_log.txt}"

load_worker_env "$LOCKED_CONF" "$RUNTIME_CONF" "$PROFILE_CONF" "$SECRETS_ENV"

ENABLED="${SMARTBLOG_MUSETALK_SERVICE_ENABLED:-0}"
HOST="${SMARTBLOG_MUSETALK_SERVICE_HOST:-0.0.0.0}"
PORT="${SMARTBLOG_MUSETALK_SERVICE_PORT:-8800}"
PYTHON="${SMARTBLOG_MUSETALK_SERVICE_PYTHON:-${SMARTBLOG_UPSCALE_PYTHON:-$ROOT_DIR/.venv/bin/python}}"
ROOT="${SMARTBLOG_MUSETALK_ROOT:-/opt/MuseTalk}"
TIMEOUT_SEC="${SMARTBLOG_MUSETALK_SERVICE_READY_TIMEOUT_SEC:-60}"

is_true_local() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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
host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
try:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as r:
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
  if ! is_true_local "$ENABLED"; then
    log "MuseTalk service disabled"
    return 0
  fi
  if is_true_local "${SMARTBLOG_MUSETALK_AUTO_INSTALL:-0}" && {
    [[ ! -x "$ROOT/venv/bin/python" ]] \
      || [[ ! -s "$ROOT/models/musetalkV15/unet.pth" ]] \
      || [[ ! -s "$ROOT/models/whisper/pytorch_model.bin" ]] \
      || [[ ! -s "$ROOT/models/sd-vae/diffusion_pytorch_model.bin" ]] \
      || [[ ! -s "$ROOT/models/face-parse-bisent/79999_iter.pth" ]]
  }; then
    log "MuseTalk auto-install requested"
    SMARTBLOG_MUSETALK_ROOT="$ROOT" "$ROOT_DIR/scripts/install_musetalk_service.sh"
  fi
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    if health >/dev/null 2>&1; then
      log "MuseTalk service already ready pid=$pid"
    else
      log "MuseTalk service already running but not ready yet pid=$pid"
    fi
    return 0
  fi
  if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
  fi
  mkdir -p "$RUN_STATE_DIR" "$LOG_DIR"
  local log_file="$LOG_DIR/musetalk_service_$(date '+%Y%m%d_%H%M%S').log"
  printf '%s\n' "$log_file" >"$CURRENT_LOG_FILE"
  log "starting MuseTalk service host=$HOST port=$PORT root=$ROOT python=$PYTHON log=$log_file"
  (
    cd "$ROOT_DIR"
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${SMARTBLOG_MUSETALK_CUDA_VISIBLE_DEVICES:-0}"
    export HF_HOME="${SMARTBLOG_MUSETALK_HF_HOME:-${WORKER_ASSET_ROOT:-/workspace/smartblog-assets}/musetalk/hf}"
    export HUGGING_FACE_HUB_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}}}"
    export HF_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}}"
    exec "$PYTHON" -m avalife.musetalk_service
  ) >"$log_file" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$PID_FILE"
  log "MuseTalk service pid=$pid"
}

wait_ready() {
  if ! is_true_local "$ENABLED"; then
    return 0
  fi
  local deadline=$(( $(date +%s) + TIMEOUT_SEC ))
  while (( $(date +%s) < deadline )); do
    if health >/dev/null 2>&1; then
      log "MuseTalk service health endpoint ready"
      return 0
    fi
    local pid
    pid="$(current_pid)"
    if [[ -n "$pid" ]] && ! pid_alive "$pid"; then
      log "MuseTalk service exited before ready pid=$pid"
      tail -n 160 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
      return 1
    fi
    sleep 2
  done
  log "MuseTalk service not ready after ${TIMEOUT_SEC}s"
  tail -n 160 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
  return 1
}

stop() {
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    log "stopping MuseTalk service pid=$pid"
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
  log "enabled=$ENABLED pid=${pid:-none} host=$HOST port=$PORT root=$ROOT"
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
