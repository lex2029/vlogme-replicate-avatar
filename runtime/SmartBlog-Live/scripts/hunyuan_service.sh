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
PID_FILE="${SMARTBLOG_HUNYUAN_SERVICE_PID_FILE:-$RUN_STATE_DIR/hunyuan_service.pid}"
LOG_DIR="${SMARTBLOG_HUNYUAN_SERVICE_LOG_DIR:-$ROOT_DIR/logs/hunyuan_service}"
CURRENT_LOG_FILE="${SMARTBLOG_HUNYUAN_SERVICE_CURRENT_LOG_FILE:-$RUN_STATE_DIR/current_hunyuan_service_log.txt}"

load_worker_env "$LOCKED_CONF" "$RUNTIME_CONF" "$PROFILE_CONF" "$SECRETS_ENV"

ENABLED="${SMARTBLOG_HUNYUAN_SERVICE_ENABLED:-0}"
HOST="${SMARTBLOG_HUNYUAN_SERVICE_HOST:-127.0.0.1}"
PORT="${SMARTBLOG_HUNYUAN_SERVICE_PORT:-8798}"
PYTHON="${SMARTBLOG_HUNYUAN_PYTHON:-/opt/hunyuan/venv/bin/python}"
MODEL_ID="${SMARTBLOG_HUNYUAN_MODEL_ID:-hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled}"
TIMEOUT_SEC="${SMARTBLOG_HUNYUAN_SERVICE_READY_TIMEOUT_SEC:-1200}"

is_true() {
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
    log "Hunyuan service disabled"
    return 0
  fi
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    if health >/dev/null 2>&1; then
      log "Hunyuan service already ready pid=$pid"
    else
      log "Hunyuan service already running but not ready yet pid=$pid"
    fi
    return 0
  fi
  if [[ ! -x "$PYTHON" ]]; then
    log "Hunyuan python not executable: $PYTHON"
    return 1
  fi

  mkdir -p "$RUN_STATE_DIR" "$LOG_DIR"
  local log_file="$LOG_DIR/hunyuan_service_$(date '+%Y%m%d_%H%M%S').log"
  printf '%s\n' "$log_file" >"$CURRENT_LOG_FILE"
  log "starting Hunyuan service host=$HOST port=$PORT model=$MODEL_ID python=$PYTHON log=$log_file"
  (
    cd "$ROOT_DIR"
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${SMARTBLOG_HUNYUAN_CUDA_VISIBLE_DEVICES:-2}"
    export HF_HOME="${SMARTBLOG_HUNYUAN_HF_HOME:-/root/smartblog-assets/hunyuan/hf}"
    export HUGGING_FACE_HUB_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
    export HF_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HF_TOKEN:-}}}"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    exec "$PYTHON" -m avalife.hunyuan_service \
      --host "$HOST" \
      --port "$PORT" \
      --model-id "$MODEL_ID"
  ) >"$log_file" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$PID_FILE"
  log "Hunyuan service pid=$pid"
}

wait_ready() {
  if ! is_true "$ENABLED"; then
    return 0
  fi
  local deadline=$(( $(date +%s) + TIMEOUT_SEC ))
  while (( $(date +%s) < deadline )); do
    if health >/dev/null 2>&1; then
      log "Hunyuan service ready"
      return 0
    fi
    local pid
    pid="$(current_pid)"
    if [[ -n "$pid" ]] && ! pid_alive "$pid"; then
      log "Hunyuan service exited before ready pid=$pid"
      tail -n 120 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
      return 1
    fi
    sleep 3
  done
  log "Hunyuan service not ready after ${TIMEOUT_SEC}s"
  tail -n 160 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
  return 1
}

stop() {
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    log "stopping Hunyuan service pid=$pid"
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
  log "enabled=$ENABLED pid=${pid:-none} host=$HOST port=$PORT model=$MODEL_ID"
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
    echo "Usage: $0 start|start-wait|wait|stop|restart|restart-wait|status" >&2
    exit 2
    ;;
esac
